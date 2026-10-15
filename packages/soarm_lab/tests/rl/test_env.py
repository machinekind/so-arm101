"""The lab's RL environments: numpy/torch agreement, Gymnasium ids, renders from the station's cameras."""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.env import VIEW, ArmEnv  # noqa: E402
from soarm_lab.rl.randomize import Randomization  # noqa: E402
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
