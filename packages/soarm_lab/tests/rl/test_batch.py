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

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.batch import BLOCK_CONTACTS, CONTACTS, BatchEnv, contact_budget  # noqa: E402
from soarm_lab.rl.env import ArmEnv  # noqa: E402
from soarm_lab.rl.objects import DESCRIPTOR_DIM, descriptor  # noqa: E402
from soarm_lab.rl.policy import Policy  # noqa: E402
from soarm_lab.rl.randomize import Randomization  # noqa: E402
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

    assert main(["train", "--task", "reach", "--envs", "256", "--iters", "2", "--eval-episodes", "1",
                 "--out", str(tmp_path), "--name", "smoke"]) == 0
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


def test_a_shape_conditioned_block_steps_on_the_cpu_device(station):
    """The cylinder (another collision function than the boxes), with its descriptor after the action."""
    task = replace(tk.make_task("lift", spec=station.spec()), object_kind="cylinder", shape_conditioned=True)
    env = BatchEnv(task, num_envs=4, station=station, device="cpu")
    assert env.obs_dim == tk.OBS_DIMS["lift"] + DESCRIPTOR_DIM == 40
    assert contact_budget(task, "so101") == BLOCK_CONTACTS["so101"]                # one part
    assert contact_budget(tk.make_task("lift"), "so101") == CONTACTS["so101"]       # the legacy cube
    assert env.yaw_period == pytest.approx(np.pi)
    env.reset()
    _, _, _, info = env.step(jnp.zeros((4, env.act_dim)))
    # The cylinder lies on its side: its lowest point on the table, its centre one radius above it.
    np.testing.assert_allclose(np.asarray(info["height"]), 0.0, atol=2e-3)
    obs, rew, done, info = random_steps(env, 20)
    assert obs.shape == (4, 40) and obs.dtype == jnp.float32 and rew.shape == (4,)
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()
    assert set(info) == {"success", "time_outs", "broken", "finished"} | INFO["lift"]
    want = np.tile(descriptor("cylinder"), (4, 1))
    np.testing.assert_array_equal(np.asarray(obs[:, -DESCRIPTOR_DIM:]), want)


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


def test_the_cpu_device_perceives_the_biased_delayed_pose(station):
    """The perception additions in the batch: each world's own offset, the least delay, the latch."""
    rand = replace(Randomization.none(), pose_bias_m=0.03, cube_min_delay=2, cube_delay=2, latch_m=0.05)
    env = BatchEnv("lift", num_envs=4, station=station, randomization=rand, device="cpu")
    obs = env.reset()
    bias = np.asarray(env.s["bias"])
    assert bias.shape == (4, 2)
    assert (np.linalg.norm(bias, axis=1) > 1e-4).all() and (np.linalg.norm(bias, axis=1) <= 0.03 + 1e-6).all()
    assert (np.asarray(env.s["cube_d"]) == 2).all()
    true = (env.d.xpos.numpy()[:, env.cube_body] - env.t_b) @ env.R_b
    np.testing.assert_allclose(np.asarray(obs[:, 15:17]), true[:, :2] + bias, atol=1e-5)
    np.testing.assert_allclose(np.asarray(obs[:, 17]), true[:, 2], atol=1e-5)
    obs, rew, done, info = random_steps(env, 20)
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("task", ["reach", "lift"])
def test_one_world_follows_cpu_mujoco(task, device, station):
    """One world stepped with zero actions in BatchEnv and in ArmEnv stays within 0.01 deg per joint.

    Both without randomization (`Randomization.none()`), from the same start: the batch env's start
    pose, and in lift its cube, go to `ArmEnv.reset(options=)`. 40 ticks (2 s) of the arm holding its
    targets against gravity while the gripper moves to mid-range. MuJoCo Warp is float32, CPU MuJoCo
    float64; on Warp's CPU device the largest difference is 0.00001 deg (measured), the torch version
    of the batch env measured 0.001 deg after 2 s on an RTX A4500.
    """
    rand = Randomization.none()
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
