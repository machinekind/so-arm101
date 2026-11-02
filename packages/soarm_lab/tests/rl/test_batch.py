"""The batch environment (MuJoCo Warp, JAX arrays) and a short training run.

Skipped without the `train` extra (jax, mujoco-warp). The CPU tests run MuJoCo Warp on Warp's CPU
device (`device="cpu"`); its first run compiles the kernels (~20 s, then cached). The GPU tests skip
without a CUDA GPU in JAX (`pip install "soarm-lab[train]"` on Linux brings jax[cuda12]).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")

import jax.numpy as jnp  # noqa: E402

from soarm_lab import blocks  # noqa: E402
from soarm_lab.rl import tasks as tk  # noqa: E402
from soarm_lab.rl.batch import CONTACTS, BatchEnv, contact_budget  # noqa: E402
from soarm_lab.rl.env import ArmEnv  # noqa: E402
from soarm_lab.rl.policy import Policy  # noqa: E402
from soarm_lab.rl.randomize import Dynamics, Randomization  # noqa: E402
from soarm_lab.station import Station  # noqa: E402


def _cuda() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:                                  # no GPU backend at all
        return False


gpu = pytest.mark.skipif(not _cuda(), reason="no CUDA GPU in JAX for MuJoCo Warp")

#: The info keys of each task.
INFO = {"reach": {"distance"}, "lift": {"height"}, "pickplace": {"height", "dist_target", "carried"}}


def random_steps(env: BatchEnv, n: int, seed: int = 0):
    key = jax.random.key(seed)
    out = None
    for _ in range(n):
        key, k = jax.random.split(key)
        out = env.step(jax.random.uniform(k, (env.num_envs, env.act_dim), minval=-1.0, maxval=1.0))
    return out


# --------------------------------------------------------------------------- GPU


@gpu
@pytest.mark.parametrize("task", ["reach", "lift", "pickplace"])
def test_the_batch_steps_on_the_station(task):
    env = BatchEnv(task, num_envs=64, station=Station.load("sim"))
    obs = env.reset()
    assert obs.shape == (64, env.obs_dim)
    obs, rew, done, info = random_steps(env, 20)
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all()
    assert not info["broken"].any()


@gpu
def test_a_short_training_run_saves_a_policy(tmp_path, capsys):
    """`soarm-lab train` is the JAX trainer's (`rl/ppo.py`); here it runs on the batch env and saves."""
    from soarm_lab.cli import main

    assert main(["train", "task=reach", "ppo.num_envs=256", "ppo.iterations=2", "eval_episodes=1",
                 f"out={tmp_path}", "name=smoke"]) == 0
    pol = Policy.load(tmp_path / "smoke" / "policy.npz")
    assert pol.critic is not None                                    # for fine-tuning
    assert pol.meta.iterations == 2 and set(pol.meta.evals) >= {"gpu", "cpu_nominal"}


# --------------------------------------------------------------------------- CPU device


@pytest.mark.parametrize("task", ["reach", "lift", "pickplace"])
def test_the_batch_steps_on_the_cpu_device(task, station):
    """The contract the trainer codes against: JAX arrays on `env.device`, the shapes and dtypes."""
    env = BatchEnv(task, num_envs=4, station=station, device="cpu")
    obs = env.reset()
    assert isinstance(obs, jax.Array) and obs.shape == (4, env.obs_dim) and obs.dtype == jnp.float32
    assert obs.devices() == {env.device} and env.wp_device == "cpu"
    obs, rew, done, info = random_steps(env, 20)
    assert obs.shape == (4, env.obs_dim) and rew.shape == (4,) and rew.dtype == jnp.float32
    assert done.shape == (4,) and done.dtype == jnp.bool_
    assert set(info) == {"success", "time_outs", "broken", "finished"} | INFO[task]
    assert all(isinstance(v, jax.Array) and v.shape == (4,) for v in info.values())
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all()
    assert not info["broken"].any()


def test_a_block_with_variants_steps_on_the_cpu_device(station):
    """Cylinder worlds include a descriptor after the previous action."""
    variants = ("cube", "cuboid", "cylinder")
    task = tk.make_task("lift", spec=station.spec(), variants=variants, block="cylinder")
    env = BatchEnv(task, num_envs=4, station=station, device="cpu")
    assert env.obs_dim == tk.OBS_DIMS["lift"] + 6 == 39
    assert contact_budget(task, "so101") == CONTACTS["so101"]
    assert env.yaw_period == pytest.approx(np.pi)
    env.reset()
    _, _, _, info = env.step(jnp.zeros((4, env.act_dim)))
    # The initial cylinder centre is one radius above the table.
    np.testing.assert_allclose(np.asarray(info["height"]), 0.0, atol=2e-3)
    obs, rew, done, info = random_steps(env, 20)
    assert obs.shape == (4, 39) and obs.dtype == jnp.float32 and rew.shape == (4,)
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()
    assert set(info) == {"success", "time_outs", "broken", "finished"} | INFO["lift"]
    np.testing.assert_array_equal(np.asarray(obs[:, -6:]), np.tile(task.descriptor(), (4, 1)))


def test_every_world_collides_with_its_own_block_size(station):
    """Each world uses its own collision geometry and settles at its scaled resting height."""
    task = tk.make_task("lift", spec=station.spec(), variants=("cuboid",), block="cuboid")
    rand = replace(Randomization.none(), block_tolerance_mm=5.0)
    env = BatchEnv(task, num_envs=2, station=station, randomization=rand, device="cpu", seed=0)
    obs = env.reset()
    size = np.asarray(env.s["size"], float)
    spec = task.block_spec
    rest = spec.half_size[2]
    # Require enough height variation [mm] to expose incorrect collision geometry.
    height, nominal = 2000.0 * rest * size[:, 2], 2000.0 * rest
    assert abs(height[0] - height[1]) > 2.0 and abs(height[0] - nominal) > 2.0
    assert (np.abs(2000.0 * np.asarray(spec.half_size) * (size - 1.0)) <= 5.0 + 1e-4).all()
    b, g = env.cube_body, env.cube_geoms[0]
    geo = blocks.scaled_geometry(spec, size, spec.mass * size.prod(1))
    fields = {"geom_size": (env.m.geom_size.numpy()[:, g], geo.size),
              "geom_rbound": (env.m.geom_rbound.numpy()[:, g], geo.rbound),
              "geom_aabb": (env.m.geom_aabb.numpy()[:, g, 1], geo.aabb),
              "body_inertia": (env.m.body_inertia.numpy()[:, b], geo.inertia),
              "body_mass": (env.m.body_mass.numpy()[:, b], spec.mass * size.prod(1))}
    for name, (got, want) in fields.items():
        assert got.shape[0] == 2, name
        np.testing.assert_allclose(got, want, rtol=1e-6, err_msg=name)
    assert not env.m.geom_aabb.numpy()[:, g, 0].any()
    np.testing.assert_array_equal(np.asarray(obs[:, -3:]), np.tile(task.descriptor()[-3:], (2, 1)))
    for _ in range(40):
        obs, rew, done, info = env.step(jnp.zeros((2, env.act_dim)))
    assert not done.any() and not info["broken"].any()
    z = ((env.d.xpos.numpy()[:, b] - env.t_b) @ env.R_b)[:, 2]
    np.testing.assert_allclose(z, rest * size[:, 2], atol=5e-4)
    np.testing.assert_allclose(np.asarray(info["height"]), 0.0, atol=5e-4)


def test_without_a_size_tolerance_the_geometry_is_neither_batched_nor_written(station):
    task = tk.make_task("lift", spec=station.spec(), variants=("cuboid",), block="cuboid")
    env = BatchEnv(task, num_envs=2, station=station, randomization=Randomization.none(), device="cpu")
    env.reset()
    assert env.block is None and env.m.geom_size.shape[0] == env.m.geom_aabb.shape[0] == 1
    assert "geom_size" not in env.s["fields"]
    assert np.asarray(env.s["size"]).tolist() == [[1.0, 1.0, 1.0]] * 2
    assert env._size(env.s) is None


def test_the_3cm_cube_takes_a_size_tolerance_and_stays_square(station):
    env = BatchEnv("lift", num_envs=2, station=station, device="cpu",
                   randomization=replace(Randomization.none(), block_tolerance_mm=5.0))
    env.reset()
    size = np.asarray(env.s["size"])
    assert env.block is blocks.block("cube3cm")
    assert np.array_equal(size[:, 0], size[:, 1]) and not np.array_equal(size[:, 1], size[:, 2])
    assert (np.abs(30.0 * size - 30.0) <= 5.0 + 1e-4).all()                 # Sides [mm].
    np.testing.assert_allclose(env.m.geom_size.numpy()[:, env.cube_geoms[0]], 0.015 * size, rtol=1e-6)


def test_a_block_that_cannot_scale_refuses_a_size_tolerance(station, monkeypatch):
    one = blocks.block("cube").parts[0]
    two = blocks.BlockSpec("two", (0.025, 0.0125, 0.0125), 0.02, (one, one), np.pi)
    monkeypatch.setitem(blocks.classic_world.BLOCKS, "two", two)
    with pytest.raises(ValueError, match="one-part block"):
        BatchEnv(tk.make_task("lift", block="two"), num_envs=2, station=station, device="cpu",
                 randomization=replace(Randomization(), block_tolerance_mm=2.0))


class _Replay:
    """Return stored draws in order through the NumPy generator interface."""

    def __init__(self, *draws):
        self.draws = list(draws)

    def uniform(self, *_, **__):
        return self.draws.pop(0)


@pytest.mark.parametrize("own,name", [("cuboid", "lift"), ("cylinder", "pickplace")])
def test_both_resets_place_a_block_the_same(own, name, station):
    """NumPy and JAX resets agree in the world frame with a rotated base."""
    n = 16
    task = tk.make_task(name, spec=station.spec(), block=own, standing_fraction=0.5)
    rand = replace(Randomization.none(), block_tolerance_mm=5.0)
    env = BatchEnv(task, num_envs=n, station=station, randomization=rand, device="cpu", seed=2)
    cpu = ArmEnv(task, station=station, randomization=Randomization.none())
    # A rotated base exposes errors in rotation order.
    c, s = np.cos(0.6), np.sin(0.6)
    base = np.eye(4)
    base[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    base[:3, 3] = [0.1, -0.2, 0.75]
    q_base = np.array([np.cos(0.3), 0.0, 0.0, np.sin(0.3)])
    env.R_b, env.t_b, env.q_base = base[:3, :3].astype(np.float32), base[:3, 3].astype(np.float32), \
        q_base.astype(np.float32)
    cpu.scene.T_base2world = base
    cpu.T_world2base = np.linalg.inv(base)
    try:
        env.reset()
        a = env.cube_qadr
        world = env.d.qpos.numpy()[:, a:a + 7].astype(float)
        rot = np.asarray(env.s["hist_rot"][0], float)
        pos = np.asarray(env.s["hist_pos"][0], float)
        size = np.asarray(env.s["size"], float)
        standing = rot[:, 2, 0] > 0.5
        assert 0 < standing.sum() < n
        yaw = np.where(standing, blocks.yaw_on_end(np, rot), np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
        radius, bearing = np.hypot(pos[:, 0], pos[:, 1]), np.arctan2(pos[:, 1], pos[:, 0])
        p, q = tk.sample_cubes(task, _Replay(radius, bearing, yaw, np.where(standing, 0.0, 1.0)), n, size)
        np.testing.assert_allclose(p, pos, atol=2e-6)
        spec = task.block_spec
        z = np.where(standing, spec.half_size[0] * size[:, 0], spec.half_size[2] * size[:, 2])
        np.testing.assert_allclose(pos[:, 2], z, rtol=1e-6)
        for k in range(n):
            cpu.set_cube(p[k], q[k])
            got = cpu.scene.data.qpos[cpu.cube_qadr:cpu.cube_qadr + 7]
            np.testing.assert_allclose(got, world[k], atol=2e-6, err_msg=str(k))
            mujoco.mj_kinematics(cpu.scene.model, cpu.scene.data)
            np.testing.assert_allclose(cpu.state()["cube_rot"], rot[k], atol=2e-6, err_msg=str(k))
        np.testing.assert_allclose(tk.make_task(name, block=own).height(np, pos, rot, size), 0.0, atol=2e-7)
    finally:
        cpu.close()


@pytest.mark.usefixtures("square_end")
@pytest.mark.parametrize("own,period", [("cuboid", 180.0), ("square_end", 90.0), ("cylinder", 0.0)])
def test_blocks_stand_on_warps_cpu_device(own, period, station):
    n = 8
    task = tk.make_task("lift", spec=station.spec(), block=own, standing_fraction=1.0)
    rand = replace(Randomization.none(), fold_yaw=True)
    env = BatchEnv(task, num_envs=n, station=station, randomization=rand, device="cpu", seed=3)
    obs = np.asarray(env.reset())
    b = env.cube_body
    z = ((env.d.xpos.numpy()[:, b] - env.t_b) @ env.R_b)[:, 2]
    np.testing.assert_allclose(z, task.block_spec.half_size[0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 21:24], np.tile([0.0, 0.0, 1.0], (n, 1)), atol=1e-6)
    yaw = np.degrees(np.arctan2(-obs[:, 24], obs[:, 25]))
    assert (np.abs(yaw) <= period / 2 + 1e-3).all()
    start = (env.d.xpos.numpy()[:, b] - env.t_b) @ env.R_b
    for _ in range(int(task.control_hz)):
        obs, rew, done, info = env.step(jnp.zeros((n, env.act_dim)))
        assert not info["broken"].any() and not done.any()
    end = (env.d.xpos.numpy()[:, b] - env.t_b) @ env.R_b
    rot = env.R_b.T @ env.d.xmat.numpy()[:, b]
    assert (np.abs(np.asarray(info["height"])) < 1e-3).all()
    assert (rot[:, 2, 0] > np.cos(np.radians(1.0))).all()
    assert (np.linalg.norm(end[:, :2] - start[:, :2], axis=1) < 2e-3).all()
    np.testing.assert_allclose(np.asarray(obs[:, 21:24]), np.tile([0.0, 0.0, 1.0], (n, 1)), atol=1e-6)


def test_a_time_out_resets_a_world_on_its_end(station):
    task = tk.make_task("lift", spec=station.spec(), block="cuboid", standing_fraction=0.5, episode_steps=2)
    rand = Randomization.none()
    env = BatchEnv(task, num_envs=16, station=station, randomization=rand, device="cpu", seed=4)
    env.reset()
    first = np.asarray(env.s["hist_rot"][0, :, 2, 0]) > 0.5
    for _ in range(task.episode_steps):
        _, _, done, _ = env.step(jnp.zeros((16, env.act_dim)))
    assert done.all()
    rot = np.asarray(env.s["hist_rot"][0])
    second = rot[:, 2, 0] > 0.5
    assert 0 < second.sum() < 16 and not np.array_equal(first, second)
    np.testing.assert_allclose(rot, env.R_b.T @ env.d.xmat.numpy()[:, env.cube_body], atol=2e-6)
    half = task.block_spec.half_size
    z = ((env.d.xpos.numpy()[:, env.cube_body] - env.t_b) @ env.R_b)[:, 2]
    np.testing.assert_allclose(z, np.where(second, half[0], half[2]), atol=1e-6)


@pytest.mark.parametrize("own,fraction", [("cuboid", 0.5), ("cube", 1.0)])
def test_the_standing_draw_keeps_the_state_key_and_the_other_numbers(own, fraction, station):
    runs = {}
    for f in (0.0, fraction):
        task = tk.make_task("lift", spec=station.spec(), block=own, standing_fraction=f)
        env = BatchEnv(task, num_envs=8, station=station, device="cpu", seed=5)
        env.reset()
        runs[f] = (env.d.qpos.numpy().copy(), np.asarray(jax.random.key_data(env.s["key"])), env)
    (q0, key0, env), (q1, key1, _) = runs[0.0], runs[fraction]
    assert np.array_equal(key0, key1)
    assert np.array_equal(q0[:, env.qadr], q1[:, env.qadr])
    assert np.array_equal(q0[:, env.cube_qadr:env.cube_qadr + 2], q1[:, env.cube_qadr:env.cube_qadr + 2])
    if own == "cube":
        assert np.array_equal(q0, q1)


def test_the_contact_budget_counts_the_collision_parts(monkeypatch):
    assert CONTACTS == {"so101": (48, 288), "a1x": (64, 384)}
    assert contact_budget(tk.make_task("reach"), "so101") == (16, 64)
    assert contact_budget(tk.make_task("reach"), "a1x") == (16, 64)
    for name in ("lift", "pickplace"):
        assert contact_budget(tk.make_task(name), "so101") == (48, 288)
        assert contact_budget(tk.make_task(name), "a1x") == (64, 384)
    one = blocks.block("cube").parts[0]
    two = blocks.BlockSpec("two", (0.025, 0.0125, 0.0125), 0.02, (one, one), np.pi)
    monkeypatch.setitem(blocks.classic_world.BLOCKS, "two", two)
    assert contact_budget(tk.make_task("lift", block="two"), "so101") == (96, 576)
    assert contact_budget(tk.make_task("lift", block="two"), "a1x") == (128, 768)


def test_a_reset_observes_the_joints_it_wrote(station):
    """After `reset()`, and after a time-out inside `step`, the observed joints are the ones in `d.qpos`.

    The observation of a done world already belongs to its new episode (no observation noise here).
    """
    task = replace(tk.make_task("lift", spec=station.spec()), episode_steps=3)
    env = BatchEnv(task, num_envs=4, station=station, randomization=Randomization.none(), device="cpu")
    na = env.act_dim

    def joints_in_physics():
        return tk.normalize_q(np, env.limits, env.d.qpos.numpy()[:, env.kin.qadr].astype(float))

    obs = env.reset()
    np.testing.assert_allclose(np.asarray(obs[:, :na]), joints_in_physics(), atol=1e-6)
    start = env.d.qpos.numpy().copy()
    for _ in range(task.episode_steps):
        obs, rew, done, info = env.step(jnp.zeros((4, na)))
    assert done.all() and info["time_outs"].all()
    assert (np.asarray(env.s["t"]) == 0).all()
    assert not np.allclose(env.d.qpos.numpy(), start)                # a new start, not the old one
    np.testing.assert_allclose(np.asarray(obs[:, :na]), joints_in_physics(), atol=1e-6)
    np.testing.assert_allclose(np.asarray(obs[:, na:2 * na]), joints_in_physics(), atol=1e-6)   # q_cmd = q0


def test_the_cpu_device_perceives_the_biased_delayed_pose_and_a_slow_jaw(station):
    """Worlds apply their sampled pose bias, delay and servo speed limit."""
    rand = replace(Randomization.nominal(Dynamics(grip_close_s=1.3)), pose_bias_m=0.03, cube_min_delay=2,
                   cube_delay=2, latch_m=0.05)
    env = BatchEnv("lift", num_envs=4, station=station, randomization=rand, device="cpu")
    obs = env.reset()
    bias = np.asarray(env.s["bias"])
    assert bias.shape == (4, 2)
    assert (np.linalg.norm(bias, axis=1) > 1e-4).all() and (np.linalg.norm(bias, axis=1) <= 0.03 + 1e-6).all()
    assert (np.asarray(env.s["cube_d"]) == 2).all()
    true = (env.d.xpos.numpy()[:, env.cube_body] - env.t_b) @ env.R_b
    np.testing.assert_allclose(np.asarray(obs[:, 15:17]), true[:, :2] + bias, atol=1e-5)
    np.testing.assert_allclose(np.asarray(obs[:, 17]), true[:, 2], atol=1e-5)

    g = env.limits.grip
    span = float(env.limits.hi[g] - env.limits.lo[g])
    rate = np.asarray(env.s["grip_rate"])
    np.testing.assert_allclose(rate, span / 26, rtol=1e-5)                  # span / (1.3 s x 20 Hz)
    ctrl0, cmd0 = np.asarray(env.s["ctrl"][:, env.act[g]]), np.asarray(env.s["q_cmd"][:, g])
    close = jnp.zeros((4, env.act_dim)).at[:, g].set(-1.0)
    env.step(close)
    np.testing.assert_allclose(ctrl0 - np.asarray(env.s["ctrl"][:, env.act[g]]), rate, rtol=1e-5)
    np.testing.assert_allclose(cmd0 - np.asarray(env.s["q_cmd"][:, g]), env.task.grip_step * span, rtol=1e-5)
    obs, rew, done, info = random_steps(env, 20)
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()


@pytest.mark.parametrize("jaw", ["free", "slow"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("task", ["reach", "lift"])
def test_one_world_follows_cpu_mujoco(task, device, jaw, station):
    """Both backends stay within 0.01 deg per joint from the same initial state.

    Zero actions hold the arm and move the gripper to mid-range for 40 policy ticks.
    Both cases use fixed dynamics, with or without a gripper servo speed limit.
    """
    rand = Randomization.none() if jaw == "free" else Randomization.nominal(Dynamics(grip_close_s=1.3))
    env = BatchEnv(task, num_envs=1, station=station, randomization=rand, device=device)
    env.reset()
    qpos = env.d.qpos.numpy()[0].astype(float)
    options = {"q": qpos[env.kin.qadr]}
    if env.task.cube:
        a = env.cube_qadr
        conj, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(conj, env.q_base.astype(float))
        mujoco.mju_mulQuat(quat, conj, qpos[a + 3:a + 7])           # q_world = q_base * q_yaw
        options["cube"] = ((qpos[a:a + 3] - env.t_b) @ env.R_b, quat)
    cpu = ArmEnv(task, station=station, randomization=rand)
    try:
        cpu.reset(seed=0, options=options)
        worst = 0.0
        for _ in range(40):
            env.step(jnp.zeros((1, env.act_dim)))
            cpu.step(np.zeros(env.act_dim))
            q_gpu = env.d.qpos.numpy()[0, env.kin.qadr].astype(float)
            worst = max(worst, float(np.degrees(np.abs(q_gpu - cpu.scene.data.qpos[cpu.kin.qadr])).max()))
        moved = float(np.degrees(np.abs(cpu.scene.data.qpos[cpu.kin.qadr] - options["q"])).max())
    finally:
        cpu.close()
    assert moved > 1.0                                               # the comparison is of a motion
    assert worst < 0.01
