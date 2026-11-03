"""The lab's RL environments: NumPy/JAX agreement, Gymnasium ids, renders from the station's cameras."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.rl import tasks as tk  # noqa: E402
from soarm_lab.rl.env import TARGET_RGBA, VIEW, ArmEnv  # noqa: E402
from soarm_lab.rl.randomize import Dynamics, Randomization  # noqa: E402
from soarm_lab.station import Station  # noqa: E402


def test_observation_and_reward_are_the_same_in_numpy_and_jax():
    """One task definition for the CPU, the GPU and the arm: the formulas must not drift apart."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    task = tk.make_task("pickplace")
    env = ArmEnv(task, randomization=Randomization.none())
    lim = env.limits
    env.close()
    rng = np.random.default_rng(0)
    n = 16
    q = rng.uniform(lim.lo, lim.hi, (n, task.act_dim))
    q_cmd = rng.uniform(lim.lo, lim.hi, (n, task.act_dim))
    tcp, goal, cube = rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3)), rng.normal(0, 0.2, (n, 3))
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    act, prev = rng.uniform(-1, 1, (n, task.act_dim)), rng.uniform(-1, 1, (n, task.act_dim))
    jaws = rng.integers(0, 2, (n, 2)).astype(float)

    task = tk.make_task("pickplace", limit_penalty=0.5, hold_still=0.5, end_on_success=2, roll_penalty=0.3)
    cube[: n // 2, 2] = 0.09                                   # half lifted: the carry latch, the lift term
    streak = rng.integers(0, 3, n)
    # Target, cube velocity, latch and open gripper, set so that some worlds succeed.
    target = np.c_[rng.normal(0, 0.2, (n, 2)), np.zeros(n)]
    vel = rng.normal(0, 0.02, (n, 3))
    carried = rng.integers(0, 2, n).astype(bool)
    opened = rng.integers(0, 2, n).astype(bool)
    cube[n // 2:, 2] = task.block_spec.half_size[2] + rng.uniform(-0.002, 0.004, n - n // 2)
    cube[n // 2::2, :2] = target[n // 2::2, :2] + rng.normal(0, 0.01, (len(cube[n // 2::2]), 2))
    ok = slice(n // 2, n, 4)                                   # every success condition at once here
    vel[ok], jaws[ok], carried[ok], opened[ok] = 0.001, 0.0, True, True

    def run(xp, conv):
        b = (lambda x: x) if xp is np else jnp.asarray
        obs = task.observe(xp, lim, conv(q), conv(q_cmd), conv(tcp), conv(prev), goal=conv(goal),
                         cube_pos=conv(cube), cube_rot=conv(rot), target=conv(target))
        r, s, f = task.reward(xp, conv(tcp), conv(act), conv(prev), goal=conv(goal), cube_pos=conv(cube),
                            cube_rot=conv(rot), jaw_contacts=conv(jaws), q_cmd=conv(q_cmd), limits=lim, target=conv(target),
                            cube_vel=conv(vel), carried=b(carried), grip_open=b(opened))
        nxt = tk.apply_action(xp, task, lim, conv(q_cmd), conv(act))
        st, done = tk.success_streak(xp, task, streak if xp is np else jnp.asarray(streak), s)
        latch = tk.carry_latch(xp, task, b(carried), conv(jaws), conv(cube), conv(rot))
        grip = tk.grip_is_open(xp, lim, conv(q_cmd))
        return [np.asarray(x) for x in (obs, r, s, f, nxt, st, done, latch, grip)]

    a = run(np, lambda x: x)
    with jax.enable_x64():
        b = run(jnp, lambda x: jnp.asarray(x, dtype=jnp.float64))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)
    assert a[0].shape == (n, task.obs_dim)
    assert 0 < a[2].sum() < n                                  # success in some worlds: both branches run


#: Expected rewards for the default cube with reset seed 3 and action seed 7.
#: These actions leave the cube untouched; test_task_backends.py covers grasp rewards.
PINNED = {
    ("lift", "none"): [-0.038297, -0.060949, -0.080177, -0.10761, -0.110454, -0.088212, -0.110115, -0.098116,
                       -0.007524, -0.133061, -0.061515, -0.035416, -0.163077, -0.082272, -0.063145, -0.083786,
                       -0.08302, -0.085151, -0.043596, -0.065157],
    ("lift", "default"): [-0.040315, -0.064228, -0.083141, -0.109993, -0.113994, -0.089289, -0.111033,
                          -0.100593, -0.009758, -0.135502, -0.064, -0.038297, -0.165972, -0.083678, -0.065125,
                          -0.086361, -0.083931, -0.086851, -0.047217, -0.068642],
}
#: Observation entries 12..26 contain TCP, cube, cube-TCP, and rotation in that order.
PINNED_OBS0 = {"none": [0.321912, 0.000313, 0.145093, 0.191204, 0.026726, 0.015, -0.130708, 0.026413, -0.130093,
                        0.076343, 0.997082, 0.0, -0.997082, 0.076343, 0.0],
               "default": [0.323696, -0.019901, 0.158223, 0.166277, 0.151155, 0.015, -0.157419, 0.171056,
                           -0.143223, 0.976999, 0.213242, 0.0, -0.213242, 0.976999, 0.0]}
PINNED_CUBE20 = {"none": [0.191204, 0.026726, 0.014987], "default": [0.168353, 0.148136, 0.014987]}
#: Initial target and target-cube values with randomization disabled.
PINNED_TARGET0 = [0.23113, -0.083134, 0.0, 0.039926, -0.10986, -0.015]


@pytest.mark.parametrize("rand", ["none", "default"])
def test_the_3cm_cube_env_keeps_its_numbers(rand):
    """The default cube retains its scene and observations while lying flat."""
    task = tk.make_task("lift")
    assert task.obs_dim == tk.OBS_DIMS["lift"] == 33
    env = ArmEnv(task, randomization=Randomization.none() if rand == "none" else Randomization())
    try:
        obs, _ = env.reset(seed=3)
        np.testing.assert_allclose(obs[12:27], PINNED_OBS0[rand], atol=2e-6)
        rng = np.random.default_rng(7)
        rewards = []
        for _ in range(20):
            obs, r, _, _, info = env.step(rng.uniform(-1, 1, task.act_dim))
            rewards.append(r)
        np.testing.assert_allclose(rewards, PINNED[("lift", rand)], atol=2e-6)
        np.testing.assert_allclose(obs[15:18], PINNED_CUBE20[rand], atol=2e-6)
        assert info["height"] == pytest.approx(-0.0000134, abs=1e-7)
        m = env.scene.model
        assert np.flatnonzero(m.geom_bodyid == m.body("cube").id).tolist() == [m.geom("cube").id]
        assert m.geom_type[m.geom("cube").id] == mujoco.mjtGeom.mjGEOM_BOX
        assert m.geom_size[m.geom("cube").id] == pytest.approx([0.015] * 3)
        if rand == "none":
            assert m.body_mass[m.body("cube").id] == pytest.approx(0.03)
        assert m.opt.ccd_tolerance == pytest.approx(1e-6)
    finally:
        env.close()

    env = ArmEnv("pickplace", randomization=Randomization.none())
    try:
        obs, _ = env.reset(seed=3)
        np.testing.assert_allclose(obs[27:33], PINNED_TARGET0, atol=2e-6)
    finally:
        env.close()


def test_a_block_under_the_pressing_gripper_stays_on_the_table_top():
    """Table penetration stays below 1 mm while the closed gripper presses at the arm's torque limit."""
    env = ArmEnv("lift", randomization=Randomization.none())
    try:
        half = env.task.block_spec.half_size[2]
        env.reset(seed=0, options={"cube": ((0.2, 0.0, half), (1.0, 0.0, 0.0, 0.0))})
        kin, scene = env.kin, env.scene
        m, d = scene.model, scene.data
        down = np.column_stack([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        joints = kin.ik(np.array([0.2, 0.0, 2 * half + 0.04]), down, seed=dict(kin.spec.home, gripper=0.0)).joints
        scene.set_joints(joints)
        lows = []
        for z in np.linspace(2 * half + 0.04, 2 * half - 0.02, 20):
            joints = dict(kin.ik_step(np.array([0.2, 0.0, z]), down, seed=joints).joints, gripper=0.0)
            scene.command(joints)
            for _ in range(10):
                mujoco.mj_step(m, d)
        for _ in range(300):
            mujoco.mj_step(m, d)
            mujoco.mj_kinematics(m, d)
            s = env.state()
            lows.append(env.task.height(np, s["cube_pos"][None], s["cube_rot"][None])[0])
        # Require contact under load at the shoulder or elbow torque limit.
        limit = m.actuator_forcerange[scene.act_ids[2], 1]
        assert np.abs(d.actuator_force[scene.act_ids[1:3]]).max() == pytest.approx(limit)
        assert s["jaws"].max() > 0.5
        assert min(lows) > -0.001
    finally:
        env.close()


#: Descriptor order for these tests.
VARIANTS = ("cube", "cuboid", "cylinder")


def _plain_observation(env: ArmEnv) -> np.ndarray:
    """Return the current observation without a descriptor."""
    s = env.state()
    pos, rot = env._cube_seen
    plain = replace(env.task, variants=())
    return plain.observe(np, env.limits, s["q"][None], env.q_cmd[None], s["tcp"][None], env.prev_action[None],
                         goal=env.goal[None], cube_pos=pos[None], cube_rot=rot[None],
                         target=env.target[None])[0].astype(np.float32)


@pytest.mark.parametrize("own", VARIANTS)
def test_a_lift_env_with_variants_for_every_block(own):
    """The descriptor follows the unchanged lift observation in every variant."""
    task = tk.make_task("lift", variants=VARIANTS, block=own)
    env = ArmEnv(task, randomization=Randomization.none())

    def height():
        s = env.state()
        return task.height(np, s["cube_pos"][None], s["cube_rot"][None])[0]

    try:
        assert env.observation_space.shape == (39,)
        obs, _ = env.reset(seed=0)
        assert abs(height()) < 1e-12
        rng = np.random.default_rng(1)
        for t in range(21):
            assert obs.shape == (39,) and np.isfinite(obs).all()
            assert np.array_equal(obs[:33], _plain_observation(env)), t
            assert np.array_equal(obs[33:], task.descriptor())
            obs, r, _, _, info = env.step(rng.uniform(-1, 1, task.act_dim))
            assert np.isfinite(r) and np.isfinite(info["height"])
            assert info["height"] == height()
        geoms = np.flatnonzero(env.scene.model.geom_bodyid == env.fields.cube_body)
        assert env.fields.cube_geoms.tolist() == geoms.tolist()
        assert env.scene.model.body_mass[env.fields.cube_body] == pytest.approx(block(own).mass)
    finally:
        env.close()


def test_the_block_randomizes_on_every_part_and_folds_its_own_symmetry():
    """Randomization scales object friction and mass and folds yaw by its symmetry."""
    task = tk.make_task("lift", block="cuboid")
    env = ArmEnv(task, randomization=Randomization(cube_friction=(1.2, 1.2), cube_mass=(1.5, 1.5)))
    try:
        m = env.scene.model
        env.reset(seed=0)
        assert m.geom_friction[env.fields.cube_geoms, 0] == pytest.approx(1.2)
        assert m.body_mass[env.fields.cube_body] == pytest.approx(1.5 * block("cuboid").mass)
        assert env.yaw_period == np.pi
        yaw = np.radians(130.0)
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        folded = tk.fold_yaw(np, R[None], env.yaw_period)[0]
        assert np.degrees(np.arctan2(folded[1, 0], folded[0, 0])) == pytest.approx(-50.0)
    finally:
        env.close()


def test_every_part_of_a_block_randomizes_its_friction(monkeypatch):
    """All parts share the friction multiplier and contribute to one body mass."""
    from soarm_lab import blocks
    from soarm_lab.station import Part

    half = (0.03, 0.0125, 0.0125)
    two = blocks.BlockSpec("two", half, 0.04, (Part("box", (0.015, 0.0125, 0.0125), pos=(-0.015, 0.0, 0.0)),
                                               Part("box", (0.015, 0.0125, 0.0125), pos=(0.015, 0.0, 0.0))),
                           np.pi)
    monkeypatch.setitem(blocks.classic_world.BLOCKS, "two", two)
    env = ArmEnv(tk.make_task("lift", block="two"), randomization=Randomization(cube_friction=(1.2, 1.2),
                                                                                 cube_mass=(1.5, 1.5)))
    try:
        env.reset(seed=0)
        m = env.scene.model
        assert len(env.fields.cube_geoms) == 2
        assert m.geom_friction[env.fields.cube_geoms, 0].tolist() == pytest.approx([1.2, 1.2])
        assert m.body_mass[env.fields.cube_body] == pytest.approx(1.5 * 0.04)
    finally:
        env.close()


@pytest.mark.parametrize("own", VARIANTS)
def test_a_block_starts_anywhere_on_the_ring_at_any_yaw(own):
    """Resets sample position and yaw while keeping body z vertical."""
    task = tk.make_task("lift", block=own)
    env = ArmEnv(task, randomization=Randomization.none())
    try:
        poses = []
        for seed in range(8):
            env.reset(seed=seed)
            s = env.state()
            poses.append((*s["cube_pos"], np.arctan2(s["cube_rot"][1, 0], s["cube_rot"][0, 0])))
            assert s["cube_rot"][2, 2] == pytest.approx(1.0)
            assert s["cube_pos"][2] == pytest.approx(block(own).half_size[2])
    finally:
        env.close()
    poses = np.array(poses)
    radius = np.hypot(poses[:, 0], poses[:, 1])
    assert radius.min() >= task.cube_radius[0] and radius.max() <= task.cube_radius[1]
    assert np.ptp(poses[:, 1]) > 0.05 and np.ptp(poses[:, 3]) > np.pi / 2


def test_each_reset_paints_the_block_without_touching_the_physics():
    """Seeds repeat colours. Every part of the block gets the same colour."""
    task = tk.make_task("lift", block="cube", variants=("cube",))
    env = ArmEnv(task, randomization=Randomization.none())
    try:
        m = env.scene.model
        geoms = env.fields.cube_geoms

        def paint(seed=None):
            obs, _ = env.reset(seed=seed)
            rgba = m.geom_rgba[geoms]
            assert (rgba == rgba[0]).all()
            return obs, tuple(rgba[0])

        first, colour = paint(4)
        assert paint(4) == (pytest.approx(first), colour)
        assert paint(5)[1] != colour
        assert len({paint()[1] for _ in range(5)}) == 5
    finally:
        env.close()


#: Cube position in the base frame [m].
CUBE = np.array([0.2, 0.0, 0.015])
#: Upright quaternion (w, x, y, z).
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def test_the_pose_bias_is_constant_within_an_episode():
    """Each episode adds one xy pose bias bounded by `pose_bias_m` [m]. The z coordinate stays true."""
    env = ArmEnv("lift", randomization=replace(Randomization.none(), pose_bias_m=0.03))
    try:
        obs, _ = env.reset(seed=1)
        s = env.state()
        off = obs[15:17] - s["cube_pos"][:2]
        assert 1e-4 < np.linalg.norm(off) <= 0.03 + 1e-6
        assert obs[17] == pytest.approx(s["cube_pos"][2], abs=1e-6)
        for _ in range(10):
            obs, _, _, _, info = env.step(np.zeros(env.task.act_dim))
        np.testing.assert_allclose(obs[15:17] - env.state()["cube_pos"][:2], off, atol=1e-6)
        assert abs(info["height"]) < 1e-3                                  # The reward uses the truth.
        obs, _ = env.reset(seed=2)
        assert np.linalg.norm(obs[15:17] - env.state()["cube_pos"][:2] - off) > 1e-4
    finally:
        env.close()


def test_the_pose_delay_is_at_least_the_minimum():
    """Equal delay bounds give a fixed pose delay [policy ticks]."""
    env = ArmEnv("lift", randomization=replace(Randomization.none(), cube_min_delay=3, cube_delay=3))
    try:
        env.reset(seed=0, options={"q": env.limits.home, "cube": (CUBE, UPRIGHT)})
        assert env._cube_delay == 3
        for k in range(1, 7):
            env.set_cube(CUBE + [0.01 * k, 0.0, 0.0], UPRIGHT)
            obs, *_ = env.step(np.zeros(env.task.act_dim))
            assert obs[15] == pytest.approx(CUBE[0] + 0.01 * max(0, k - 3), abs=1e-3), k
    finally:
        env.close()


def test_the_seen_pose_freezes_while_the_tcp_is_near_the_block():
    """Pose updates resume when the object leaves the TCP occlusion radius."""
    env = ArmEnv("lift", randomization=replace(Randomization.none(), latch_m=0.08))
    zero = np.zeros(env.task.act_dim)
    try:
        # The TCP hovers above the cube without touching it.
        hover = env.kin.to_q(env.kin.ik([0.2, 0.0, 0.06]).joints)
        obs, _ = env.reset(options={"q": hover, "cube": (CUBE, UPRIGHT)})
        s = env.state()
        assert np.linalg.norm(s["tcp"] - s["cube_pos"]) < 0.08
        before = obs[15:18].copy()
        env.set_cube(np.array([0.22, 0.0, 0.015]), UPRIGHT)
        obs, *_ = env.step(zero)
        assert np.array_equal(obs[15:18], before)
        assert env.state()["cube_pos"][0] == pytest.approx(0.22, abs=2e-3)
        env.set_cube(np.array([0.32, 0.0, 0.015]), UPRIGHT)
        obs, *_ = env.step(zero)
        assert obs[15] == pytest.approx(0.32, abs=2e-3)
        assert env.state()["cube_pos"][0] == pytest.approx(0.32, abs=2e-3)

        # From home the TCP is far enough for a refresh every tick.
        env.reset(options={"q": env.limits.home, "cube": (CUBE, UPRIGHT)})
        for k in range(1, 4):
            env.set_cube(CUBE + [0.01 * k, 0.0, 0.0], UPRIGHT)
            obs, *_ = env.step(zero)
            assert obs[15] == pytest.approx(CUBE[0] + 0.01 * k, abs=2e-3), k
    finally:
        env.close()


def test_the_jaw_closes_at_the_servos_own_speed_and_the_command_at_its_own():
    """The gripper actuator obeys the servo speed limit independently of the command rate."""
    for rand, ticks in ((Randomization.nominal(Dynamics(grip_close_s=1.3)), 30), (Randomization.none(), 9)):
        env = ArmEnv("lift", randomization=rand)
        try:
            g = env.limits.grip
            lo, span = env.limits.lo[g], env.limits.hi[g] - env.limits.lo[g]
            q0 = env.limits.home.copy()
            q0[g] = env.limits.hi[g]
            env.reset(options={"q": q0, "cube": (np.array([0.3, 0.15, 0.015]), UPRIGHT)})
            d, ag = env.scene.data, env.scene.act_ids[g]
            close = np.zeros(env.task.act_dim)
            close[g] = -1.0
            ctrl, jaw = [float(d.ctrl[ag])], []
            for t in range(1, ticks + 1):
                env.step(close)
                ctrl.append(float(d.ctrl[ag]))
                jaw.append(float(d.qpos[env.kin.qadr[g]]))
                if t >= 7:
                    assert env.q_cmd[g] == pytest.approx(lo), t
                if rand.centre.grip_close_s == 0:
                    assert ctrl[-1] == env.q_cmd[g], t
            if rand.centre.grip_close_s > 0:
                assert max(np.diff(-np.array(ctrl))) <= span / 26 + 1e-9         # span / (1.3 s x 20 Hz)
                assert ctrl[25] > lo + 0.03 and ctrl[26] == pytest.approx(lo, abs=1e-9)
                assert jaw[10 - 1] > lo + 0.1 and jaw[28 - 1] < lo + 0.02
            else:
                assert jaw[9 - 1] < lo + 0.02
        finally:
            env.close()


def test_the_target_pad_is_white():
    assert TARGET_RGBA == (1.0, 1.0, 1.0, 1.0)
    env = ArmEnv("pickplace", randomization=Randomization.none())
    try:
        m = env.scene.model
        assert m.geom_rgba[m.geom("target").id].tolist() == list(TARGET_RGBA)
    finally:
        env.close()


def test_gymnasium_ids_of_the_lab():
    """Importing `soarm_lab.rl` registers the lab's environments as `SoArmLab/*-v0`."""
    import soarm_lab.rl  # noqa: F401

    env = gym.make("SoArmLab/Reach-v0")
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (tk.OBS_DIMS["reach"],)
        assert type(env.unwrapped) is ArmEnv
    finally:
        env.close()
    assert {"SoArmLab/Lift-v0", "SoArmLab/PickPlace-v0"} <= set(gym.registry)


# --------------------------------------------------------------------------- renders


@pytest.mark.render
def test_the_environment_renders_from_the_station_camera(station):
    env = ArmEnv("lift", station=station, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()
    assert env.render_camera == "sym-lewa"
    assert frame.shape == (480, 640, 3) and frame.std() > 0


@pytest.mark.render
def test_a_station_without_cameras_gets_the_preview_camera(tmp_path, station):
    """`env.scene_config(view=True)`: the preview camera `widok` when the station has none."""
    folder = tmp_path / "blind"
    folder.mkdir()
    xml = station.scene_path.read_text(encoding="utf-8").splitlines(keepends=True)
    (folder / "scene.xml").write_text("".join(line for line in xml if '<camera name="sym-' not in line),
                                      encoding="utf-8")
    (folder / "station.json").write_text(json.dumps({**station.data, "cameras": []}), encoding="utf-8")

    env = ArmEnv("reach", station=Station.load(folder), render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()
    assert env.render_camera == VIEW
    assert frame.shape == (360, 480, 3) and frame.std() > 0
