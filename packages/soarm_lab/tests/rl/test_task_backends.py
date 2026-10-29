"""The task formulas give the same numbers in numpy (`env.ArmEnv`) and in jax.numpy (`batch.BatchEnv`).

One task definition for the CPU, the GPU and the arm: the backends must not drift apart. numpy gets
float64 inputs, as in ArmEnv; jax.numpy gets the same numbers in float32, as in BatchEnv. CPU only,
no GPU and no MuJoCo Warp needed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
jnp = pytest.importorskip("jax.numpy")

from soarm_lab import sim as sc  # noqa: E402
from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.env import scene_config  # noqa: E402

#: Worlds in a batch.
N = 32


@pytest.fixture(scope="module")
def limits(station) -> tk.Limits:
    scene = sc.build(scene_config(tk.make_task("reach", spec=station.spec()), station))
    try:
        return tk.Limits.of(scene.kin)
    finally:
        scene.close()


def inputs(task: tk.TaskConfig, lim: tk.Limits, seed: int) -> dict[str, np.ndarray]:
    """A random batch with the shapes ArmEnv passes (N worlds instead of 1), set so that both sides of
    every condition occur: success and not, grasped and not, carried and not, near a limit and not."""
    rng = np.random.default_rng(seed)
    nj = task.act_dim
    tcp = rng.normal(0, 0.2, (N, 3))
    goal = rng.normal(0, 0.2, (N, 3))
    goal[::3] = tcp[::3] + rng.normal(0, 0.005, (len(goal[::3]), 3))      # reach: some within success_dist
    cube = rng.normal(0, 0.2, (N, 3))
    cube[: N // 2, 2] = 0.09                                    # half lifted: the carry latch, the lift term
    cube[N // 2:, 2] = task.cube_half + rng.uniform(-0.002, 0.004, N - N // 2)
    cube[::5, 2] = -0.08                                        # off the table: failure
    target = np.c_[rng.normal(0, 0.2, (N, 2)), np.zeros(N)]
    cube[N // 2::2, :2] = target[N // 2::2, :2] + rng.normal(0, 0.01, (len(cube[N // 2::2]), 2))
    jaws = rng.integers(0, 2, (N, 2)).astype(float)
    jaws[1:N // 2:3] = 1.0                                      # grasped and lifted: lift's success
    vel = rng.normal(0, 0.02, (N, 3))
    carried = rng.integers(0, 2, N).astype(bool)
    opened = rng.integers(0, 2, N).astype(bool)
    ok = slice(N // 2, N, 4)                                    # every pickplace success condition at once
    vel[ok], jaws[ok], carried[ok], opened[ok] = 0.001, 0.0, True, True
    q = rng.uniform(lim.lo, lim.hi, (N, nj))
    q_cmd = rng.uniform(lim.lo, lim.hi, (N, nj))
    q_cmd[::4, : lim.n_arm] = lim.lo[: lim.n_arm] + rng.uniform(0, 0.1, (len(q_cmd[::4]), lim.n_arm))
    yaw = rng.uniform(-np.pi, np.pi, N)
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(N)])
    rot[::2] = [[[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]] for y in yaw[::2]]
    streak = rng.integers(0, 12, N)
    streak[::2] = max(task.end_on_success - 1, 0)               # one success short of the end
    return {"q": q, "q_cmd": q_cmd, "tcp": tcp, "goal": goal, "cube": cube, "rot": rot, "target": target,
            "vel": vel, "jaws": jaws, "carried": carried, "opened": opened,
            "act": rng.uniform(-1.2, 1.2, (N, nj)), "prev": rng.uniform(-1, 1, (N, nj)),
            "streak": streak}


def run(xp, task: tk.TaskConfig, lim: tk.Limits, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every `xp` function of `task` on the batch `x`: floats as float64 in numpy, float32 in jax.numpy."""
    def f(a):
        return np.asarray(a, float) if xp is np else jnp.asarray(a, jnp.float32)

    def b(a):
        return np.asarray(a) if xp is np else jnp.asarray(a)

    cube = task.cube is not None
    kw = {}
    if task.target:
        kw = dict(target=f(x["target"]), cube_vel=f(x["vel"]), carried=b(x["carried"]),
                  grip_open=b(x["opened"]))
    out = {
        "observe": tk.observe(xp, task, lim, f(x["q"]), f(x["q_cmd"]), f(x["tcp"]), f(x["prev"]),
                              goal=f(x["goal"]), cube_pos=f(x["cube"]) if cube else None,
                              cube_rot=f(x["rot"]) if cube else None, target=f(x["target"])),
        "apply_action": tk.apply_action(xp, task, lim, f(x["q_cmd"]), f(x["act"])),
        "fold_yaw": tk.fold_yaw(xp, f(x["rot"])),
        "carry_latch": tk.carry_latch(xp, task, b(x["carried"]), f(x["jaws"]), f(x["cube"])),
        "grip_is_open": tk.grip_is_open(xp, lim, f(x["q_cmd"])),
        # reach has no limit penalty of its own; `limit_cost` is checked with one anyway.
        "limit_cost": tk.limit_cost(xp, replace(task, limit_penalty=0.5), lim, f(x["q_cmd"])),
    }
    r, s, fail = tk.reward(xp, task, f(x["tcp"]), f(x["act"]), f(x["prev"]), goal=f(x["goal"]),
                           cube_pos=f(x["cube"]) if cube else None,
                           jaw_contacts=f(x["jaws"]) if cube else None, q_cmd=f(x["q_cmd"]), limits=lim, **kw)
    out.update(reward=r, success=s, failure=fail)
    out["streak"], out["streak_done"] = tk.success_streak(xp, task, b(x["streak"]), s)
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.mark.parametrize("name", tk.TASKS)
def test_numpy_and_jax_compute_the_same_task(name, station, limits):
    task = tk.make_task(name, spec=station.spec())
    x = inputs(task, limits, seed=tk.TASKS.index(name))
    a, b = run(np, task, limits, x), run(jnp, task, limits, x)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        if a[k].dtype == bool or a[k].dtype.kind in "iu":
            assert np.array_equal(a[k], b[k]), k
        else:
            assert b[k].dtype == np.float32, k
            np.testing.assert_allclose(b[k], a[k], rtol=1e-5, atol=1e-6, err_msg=k)
    assert a["observe"].shape == (N, task.obs_dim)
    # Both sides of the conditions ran, so `where` and the masks were checked, not one branch.
    assert 0 < a["success"].sum() < N
    assert 0 < a["limit_cost"].astype(bool).sum() < N
    if task.cube:
        assert 0 < a["failure"].sum() < N
    if task.end_on_success:
        assert 0 < a["streak_done"].sum() < N
