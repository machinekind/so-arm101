"""The lab's RL environments: numpy/torch agreement, Gymnasium ids, renders from the station's cameras."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from soarm_lab.rl import objects as ob  # noqa: E402
from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.env import VIEW, ArmEnv  # noqa: E402
from soarm_lab.rl.randomize import Dynamics, Randomization  # noqa: E402
from soarm_lab.station import Station  # noqa: E402


def test_observation_and_reward_are_the_same_in_numpy_and_torch():
    """One task definition for the CPU, the GPU and the arm: the formulas must not drift apart."""
    torch = pytest.importorskip("torch")
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
    cube[n // 2:, 2] = task.cube_half + rng.uniform(-0.002, 0.004, n - n // 2)
    cube[n // 2::2, :2] = target[n // 2::2, :2] + rng.normal(0, 0.01, (len(cube[n // 2::2]), 2))
    ok = slice(n // 2, n, 4)                                   # every success condition at once here
    vel[ok], jaws[ok], carried[ok], opened[ok] = 0.001, 0.0, True, True

    def run(xp, conv):
        b = (lambda x: x) if xp is np else torch.as_tensor
        obs = tk.observe(xp, task, lim, conv(q), conv(q_cmd), conv(tcp), conv(prev), goal=conv(goal),
                         cube_pos=conv(cube), cube_rot=conv(rot), target=conv(target))
        r, s, f = tk.reward(xp, task, conv(tcp), conv(act), conv(prev), goal=conv(goal), cube_pos=conv(cube),
                            jaw_contacts=conv(jaws), q_cmd=conv(q_cmd), limits=lim, target=conv(target),
                            cube_vel=conv(vel), carried=b(carried), grip_open=b(opened))
        nxt = tk.apply_action(xp, task, lim, conv(q_cmd), conv(act))
        st, done = tk.success_streak(xp, task, streak if xp is np else torch.as_tensor(streak), s)
        latch = tk.carry_latch(xp, task, b(carried), conv(jaws), conv(cube))
        grip = tk.grip_is_open(xp, lim, conv(q_cmd))
        return [np.asarray(x) for x in (obs, r, s, f, nxt, st, done, latch, grip)]

    a = run(np, lambda x: x)
    b = run(torch, lambda x: torch.as_tensor(x, dtype=torch.float64))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-9)
    assert a[0].shape == (n, task.obs_dim)
    assert 0 < a[2].sum() < n                                  # success in some worlds: both branches run


#: The legacy cube on the code before the wooden blocks: ArmEnv(name, randomization=...), reset(seed=3),
#: 20 ticks of default_rng(7).uniform(-1, 1, 6). The default randomization covers the mass and friction
#: path of `ModelFields`. The ticks do not reach the cube; `test_task_backends.py` pins the grasp terms.
LEGACY = {
    ("lift", "none"): [-0.038308, -0.060959, -0.080185, -0.107616, -0.110462, -0.088215, -0.110117, -0.098122,
                       -0.00753, -0.133068, -0.061522, -0.035424, -0.163084, -0.082276, -0.06315, -0.083793,
                       -0.083023, -0.085155, -0.043604, -0.065165],
    ("lift", "default"): [-0.040315, -0.064228, -0.083141, -0.109993, -0.113994, -0.089289, -0.111033,
                          -0.100593, -0.009758, -0.135502, -0.064, -0.038297, -0.165972, -0.083678, -0.065125,
                          -0.086361, -0.083931, -0.086851, -0.047217, -0.068642],
}
#: Entries 12..26 of the first lift observation (TCP, cube, cube - TCP, rotation), and the cube's
#: centre in the last one, per randomization.
LEGACY_OBS0 = {"none": [0.321912, 0.000313, 0.145093, 0.191204, 0.026726, 0.015, -0.130708, 0.026413, -0.130093,
                        0.076343, 0.997082, 0.0, -0.997082, 0.076343, 0.0],
               "default": [0.323696, -0.019901, 0.158223, 0.166277, 0.151155, 0.015, -0.157419, 0.171056,
                           -0.143223, 0.976999, 0.213242, 0.0, -0.213242, 0.976999, 0.0]}
LEGACY_CUBE20 = {"none": [0.191204, 0.026726, 0.014892], "default": [0.168353, 0.148136, 0.014892]}
#: pickplace with `Randomization.none()`: the target and target - cube in the first observation.
LEGACY_TARGET0 = [0.23113, -0.083134, 0.0, 0.039926, -0.10986, -0.015]


@pytest.mark.parametrize("rand", ["none", "default"])
def test_the_legacy_cube_env_is_unchanged(rand):
    """The base policies' task gives the numbers it gave before the wooden blocks: the same scene,
    observation and reward. Only what both versions have is used, so it runs against either."""
    task = tk.make_task("lift")             # the defaults: object_kind="cube", shape_conditioned=False
    assert task.obs_dim == tk.OBS_DIMS["lift"] == 33
    env = ArmEnv(task, randomization=Randomization.none() if rand == "none" else Randomization())
    try:
        obs, _ = env.reset(seed=3)
        np.testing.assert_allclose(obs[12:27], LEGACY_OBS0[rand], atol=2e-6)
        rng = np.random.default_rng(7)
        rewards = []
        for _ in range(20):
            obs, r, _, _, info = env.step(rng.uniform(-1, 1, task.act_dim))
            rewards.append(r)
        np.testing.assert_allclose(rewards, LEGACY[("lift", rand)], atol=2e-6)
        np.testing.assert_allclose(obs[15:18], LEGACY_CUBE20[rand], atol=2e-6)
        assert info["height"] == pytest.approx(-0.0001078, abs=1e-7)
        m = env.scene.model
        assert np.flatnonzero(m.geom_bodyid == m.body("cube").id).tolist() == [m.geom("cube").id]
        assert m.geom_size[m.geom("cube").id] == pytest.approx([0.015] * 3)       # one plain box
        assert m.opt.ccd_tolerance == pytest.approx(1e-6)
    finally:
        env.close()

    env = ArmEnv("pickplace", randomization=Randomization.none())
    try:
        obs, _ = env.reset(seed=3)
        np.testing.assert_allclose(obs[27:33], LEGACY_TARGET0, atol=2e-6)
    finally:
        env.close()


def test_the_default_task_is_the_legacy_cube():
    task = tk.make_task("lift")
    assert (task.object_kind, task.shape_conditioned) == ("cube", False) and tk.legacy_cube(task)
    spec = tk.task_object(task)
    assert (spec.half_size, spec.mass, spec.yaw_period) == ((0.015,) * 3, 0.03, np.pi / 2)
    assert not tk.legacy_cube(tk.make_task("lift", shape_conditioned=True))       # the 25 mm block
    assert not tk.legacy_cube(tk.make_task("lift", object_kind="cuboid5"))
    with pytest.raises(ValueError, match="unknown object kind 'arch'"):
        tk.make_task("lift", object_kind="arch")
    with pytest.raises(ValueError, match="reach has no object"):
        tk.make_task("reach", shape_conditioned=True)


def _plain_observation(env: ArmEnv) -> np.ndarray:
    """The unconditioned observation of the env's current state, as `ArmEnv._obs` builds it."""
    s = env.state()
    pos, rot = env._cube_seen
    plain = replace(env.task, shape_conditioned=False)
    return tk.observe(np, plain, env.limits, s["q"][None], env.q_cmd[None], s["tcp"][None], env.prev_action[None],
                      goal=env.goal[None], cube_pos=pos[None], cube_rot=rot[None],
                      target=env.target[None])[0].astype(np.float32)


@pytest.mark.parametrize("kind", ob.KINDS)
def test_a_shape_conditioned_lift_env_for_every_kind(kind):
    """40 numbers for the SO-101: the unconditioned lift observation of the same state, then the
    block's descriptor. The block lies at rest at the start: its lowest point on the table."""
    task = tk.make_task("lift", object_kind=kind, shape_conditioned=True)
    env = ArmEnv(task, randomization=Randomization.none())
    try:
        assert env.observation_space.shape == (40,)
        obs, _ = env.reset(seed=0)
        assert abs(env.state()["bottom"]) < 1e-12
        rng = np.random.default_rng(1)
        for t in range(21):
            assert obs.shape == (40,) and np.isfinite(obs).all()
            assert np.array_equal(obs[:33], _plain_observation(env)), t
            assert np.array_equal(obs[33:], ob.descriptor(kind))
            obs, r, _, _, info = env.step(rng.uniform(-1, 1, task.act_dim))
            assert np.isfinite(r) and np.isfinite(info["height"])
            assert info["height"] == env.state()["bottom"]
        geoms = np.flatnonzero(env.scene.model.geom_bodyid == env.fields.cube_body)
        assert env.fields.cube_geoms.tolist() == geoms.tolist()
        assert env.scene.model.body_mass[env.fields.cube_body] == pytest.approx(ob.object_spec(kind).mass)
    finally:
        env.close()


def test_the_block_randomizes_on_every_part_and_folds_its_own_symmetry():
    """Friction multiplies every geom of the block and the mass its body; a cuboid's yaw folds to +-90 deg."""
    task = tk.make_task("lift", object_kind="cuboid85", shape_conditioned=True)
    env = ArmEnv(task, randomization=Randomization(cube_friction=(1.2, 1.2), cube_mass=(1.5, 1.5)))
    try:
        m = env.scene.model
        env.reset(seed=0)
        assert m.geom_friction[env.fields.cube_geoms, 0] == pytest.approx(1.2)
        assert m.body_mass[env.fields.cube_body] == pytest.approx(1.5 * ob.object_spec("cuboid85").mass)
        assert env._yaw_period == np.pi
        yaw = np.radians(130.0)
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        folded = tk.fold_yaw(np, R[None], env._yaw_period)[0]
        assert np.degrees(np.arctan2(folded[1, 0], folded[0, 0])) == pytest.approx(-50.0)
    finally:
        env.close()


#: A cube lying on the table in front of the arm, its quaternion (w, x, y, z): no turn.
CUBE = np.array([0.2, 0.0, 0.015])
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def test_the_pose_bias_is_constant_within_an_episode():
    """The seen xy is the truth plus one offset per episode, of up to `pose_bias_m`; z is the truth."""
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
        assert abs(info["height"]) < 1e-3                                  # the reward's height: the truth
        obs, _ = env.reset(seed=2)
        assert np.linalg.norm(obs[15:17] - env.state()["cube_pos"][:2] - off) > 1e-4    # a new offset
    finally:
        env.close()


def test_the_pose_delay_is_at_least_the_minimum():
    """With `cube_min_delay = cube_delay = 3` the policy sees where the cube was exactly 3 ticks ago."""
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
    """The occlusion latch: no refresh while the true TCP is within `latch_m` of the cube, then again."""
    env = ArmEnv("lift", randomization=replace(Randomization.none(), latch_m=0.08))
    zero = np.zeros(env.task.act_dim)
    try:
        # A hover 4.5 cm above the cube's centre; no jaw geom touches it.
        hover = env.kin.to_q(env.kin.ik([0.2, 0.0, 0.06]).joints)
        obs, _ = env.reset(options={"q": hover, "cube": (CUBE, UPRIGHT)})
        s = env.state()
        assert np.linalg.norm(s["tcp"] - s["cube_pos"]) < 0.08
        before = obs[15:18].copy()
        env.set_cube(np.array([0.22, 0.0, 0.015]), UPRIGHT)                   # along +x, away from the jaws
        obs, *_ = env.step(zero)
        assert np.array_equal(obs[15:18], before)                             # frozen
        assert env.state()["cube_pos"][0] == pytest.approx(0.22, abs=2e-3)    # the cube was not shot away
        env.set_cube(np.array([0.32, 0.0, 0.015]), UPRIGHT)                   # 12.8 cm from the TCP
        obs, *_ = env.step(zero)
        assert obs[15] == pytest.approx(0.32, abs=2e-3)                       # refreshed
        assert env.state()["cube_pos"][0] == pytest.approx(0.32, abs=2e-3)

        # From home the TCP is 16.5 cm away: the pose refreshes every tick.
        env.reset(options={"q": env.limits.home, "cube": (CUBE, UPRIGHT)})
        for k in range(1, 4):
            env.set_cube(CUBE + [0.01 * k, 0.0, 0.0], UPRIGHT)
            obs, *_ = env.step(zero)
            assert obs[15] == pytest.approx(CUBE[0] + 0.01 * k, abs=2e-3), k
    finally:
        env.close()


def test_the_jaw_closes_at_the_servos_own_speed_and_the_command_at_its_own():
    """A 1.3 s servo: the command still closes in 7 ticks, the actuator in 26, the jaw follows it.

    Without a closing time of its own (today) the actuator is the command, and the jaw closes in 9
    ticks (0.45 s).
    """
    for rand, ticks in ((Randomization.nominal(Dynamics(grip_close_s=1.3)), 30), (Randomization.none(), 9)):
        env = ArmEnv("lift", randomization=rand)
        try:
            g = env.limits.grip
            lo, span = env.limits.lo[g], env.limits.hi[g] - env.limits.lo[g]
            q0 = env.limits.home.copy()
            q0[g] = env.limits.hi[g]                                       # fully open
            env.reset(options={"q": q0, "cube": (np.array([0.3, 0.15, 0.015]), UPRIGHT)})   # away from the jaws
            d, ag = env.scene.data, env.scene.act_ids[g]
            close = np.zeros(env.task.act_dim)
            close[g] = -1.0
            ctrl, jaw = [float(d.ctrl[ag])], []
            for t in range(1, ticks + 1):
                env.step(close)
                ctrl.append(float(d.ctrl[ag]))
                jaw.append(float(d.qpos[env.kin.qadr[g]]))
                if t >= 7:
                    assert env.q_cmd[g] == pytest.approx(lo), t                # the command: its own rate
                if rand.centre.grip_close_s == 0:
                    assert ctrl[-1] == env.q_cmd[g], t                         # exactly the command
            if rand.centre.grip_close_s > 0:
                assert max(np.diff(-np.array(ctrl))) <= span / 26 + 1e-9         # S / (1.3 s x 20 Hz)
                assert ctrl[25] > lo + 0.03 and ctrl[26] == pytest.approx(lo, abs=1e-9)
                assert jaw[10 - 1] > lo + 0.1 and jaw[28 - 1] < lo + 0.02
            else:
                assert jaw[9 - 1] < lo + 0.02
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
