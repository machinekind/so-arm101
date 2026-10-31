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
from soarm_lab.rl import objects as ob  # noqa: E402
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
    rest = tk.rest_height(task) if task.cube else task.cube_half
    cube[N // 2:, 2] = rest + rng.uniform(-0.002, 0.004, N - N // 2)
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
    if cube:
        kw["bottom"] = tk.object_bottom(xp, task, f(x["cube"]), f(x["rot"]))
    if task.target:
        kw.update(target=f(x["target"]), cube_vel=f(x["vel"]), carried=b(x["carried"]),
                  grip_open=b(x["opened"]))
    out = {
        "observe": tk.observe(xp, task, lim, f(x["q"]), f(x["q_cmd"]), f(x["tcp"]), f(x["prev"]),
                              goal=f(x["goal"]), cube_pos=f(x["cube"]) if cube else None,
                              cube_rot=f(x["rot"]) if cube else None, target=f(x["target"])),
        "apply_action": tk.apply_action(xp, task, lim, f(x["q_cmd"]), f(x["act"])),
        "fold_yaw": tk.fold_yaw(xp, f(x["rot"]), tk.yaw_period(task)),
        "carry_latch": tk.carry_latch(xp, task, b(x["carried"]), f(x["jaws"]), f(x["cube"]),
                                      bottom=kw.get("bottom")),
        "grip_is_open": tk.grip_is_open(xp, lim, f(x["q_cmd"])),
        # reach has no limit penalty of its own; `limit_cost` is checked with one anyway.
        "limit_cost": tk.limit_cost(xp, replace(task, limit_penalty=0.5), lim, f(x["q_cmd"])),
    }
    r, s, fail = tk.reward(xp, task, f(x["tcp"]), f(x["act"]), f(x["prev"]), goal=f(x["goal"]),
                           cube_pos=f(x["cube"]) if cube else None,
                           jaw_contacts=f(x["jaws"]) if cube else None, q_cmd=f(x["q_cmd"]), limits=lim, **kw)
    out.update(reward=r, success=s, failure=fail)
    if cube:
        out["bottom"] = kw["bottom"]
    out["streak"], out["streak_done"] = tk.success_streak(xp, task, b(x["streak"]), s)
    return {k: np.asarray(v) for k, v in out.items()}


def _compare(task: tk.TaskConfig, lim: tk.Limits, x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    a, b = run(np, task, lim, x), run(jnp, task, lim, x)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        if a[k].dtype == bool or a[k].dtype.kind in "iu":
            assert np.array_equal(a[k], b[k]), k
        else:
            assert b[k].dtype == np.float32, k
            np.testing.assert_allclose(b[k], a[k], rtol=1e-5, atol=1e-6, err_msg=k)
    assert a["observe"].shape == (N, task.obs_dim)
    return a


@pytest.mark.parametrize("name", tk.TASKS)
def test_numpy_and_jax_compute_the_same_task(name, station, limits):
    task = tk.make_task(name, spec=station.spec())
    x = inputs(task, limits, seed=tk.TASKS.index(name))
    a = _compare(task, limits, x)
    # Both sides of the conditions ran, so `where` and the masks were checked, not one branch.
    assert 0 < a["success"].sum() < N
    assert 0 < a["limit_cost"].astype(bool).sum() < N
    if task.cube:
        assert 0 < a["failure"].sum() < N
    if task.end_on_success:
        assert 0 < a["streak_done"].sum() < N


@pytest.mark.parametrize("kind", ob.KINDS)
@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_numpy_and_jax_compute_the_same_shape_conditioned_task(name, kind, station, limits):
    """A conditioned task: the descriptor after the action, the block's lowest point in the reward."""
    task = tk.make_task(name, spec=station.spec(), object_kind=kind, shape_conditioned=True)
    x = inputs(task, limits, seed=10 + ob.KINDS.index(kind))
    a = _compare(task, limits, x)
    plain = tk.observe(np, replace(task, shape_conditioned=False), limits, x["q"], x["q_cmd"], x["tcp"],
                       x["prev"], cube_pos=x["cube"], cube_rot=x["rot"], target=x["target"])
    k = plain.shape[1]
    assert a["observe"].shape == (N, k + ob.DESCRIPTOR_DIM)
    assert np.array_equal(a["observe"][:, :k], plain)                       # the unconditioned layout first
    assert np.array_equal(a["observe"][:, k:], np.broadcast_to(ob.descriptor(kind), (N, ob.DESCRIPTOR_DIM)))
    np.testing.assert_array_equal(a["bottom"], ob.bottom_height(np, kind, x["cube"], x["rot"]))
    assert 0 < a["success"].sum() < N and 0 < a["failure"].sum() < N


def _legacy_batch(task: tk.TaskConfig, lim: tk.Limits) -> dict[str, np.ndarray]:
    """12 worlds around the TCP, from 7 cm below the table to 10 cm above it, some grasped."""
    rng = np.random.default_rng(5)
    n, nj = 12, task.act_dim
    tcp = rng.normal(0, 0.05, (n, 3)) + [0.2, 0.0, 0.05]
    cube = tcp + rng.normal(0, 0.01, (n, 3))
    cube[:, 2] = task.cube_half + np.linspace(-0.07, 0.10, n)
    jaws = np.ones((n, 2))
    jaws[::3] = 0.0
    target = np.c_[cube[:, :2] + rng.normal(0, 0.02, (n, 2)), np.zeros(n)]
    vel = rng.normal(0, 0.01, (n, 3))
    q_cmd = rng.uniform(lim.lo, lim.hi, (n, nj))
    act, prev = rng.uniform(-1, 1, (n, nj)), rng.uniform(-1, 1, (n, nj))
    return dict(tcp=tcp, cube=cube, jaws=jaws, target=target, vel=vel, carried=np.arange(n) % 2 == 0,
                opened=np.arange(n) % 4 < 2, q_cmd=q_cmd, act=act, prev=prev)


#: The legacy cube's rewards, successes and failures on `_legacy_batch`, from the code before the wooden
#: blocks; the carry latch is [1, 0, 1, 0, ..., 1, 1] in both tasks.
LEGACY_REWARDS = {
    "lift": ([-0.05087789, 1.010613325, 0.988474619, 0.830648662, 1.148904974, 1.832259039, 0.284883369,
              3.803095015, 4.393715791, 0.242831223, 6.846265757, 6.189205521],
             [0] * 10 + [1, 1], [1, 1] + [0] * 10),
    "pickplace": ([10.175731089, 1.010613325, 3.637924755, 0.830648662, 7.987700552, 1.589834797,
                   3.18401043, 2.530367742, 6.250467077, 0.242831223, 6.107233112, 3.005231585],
                  [1] + [0] * 11, [1, 1] + [0] * 10),
}
LEGACY_LATCH = [1, 0] * 5 + [1, 1]


@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_the_legacy_reward_terms_are_pinned(name, station, limits):
    """The legacy cube's grasp, lift, place and failure terms: the env's ticks in `test_env.py` never
    reach the cube, these do. Only what both versions have is used, so it runs against either."""
    task = tk.make_task(name, spec=station.spec())
    x = _legacy_batch(task, limits)
    kw = dict(target=x["target"], cube_vel=x["vel"], carried=x["carried"], grip_open=x["opened"]) \
        if task.target else {}
    r, s, f = tk.reward(np, task, x["tcp"], x["act"], x["prev"], cube_pos=x["cube"], jaw_contacts=x["jaws"],
                        q_cmd=x["q_cmd"], limits=limits, **kw)
    rew, succ, fail = LEGACY_REWARDS[name]
    np.testing.assert_allclose(r, rew, atol=2e-9)
    assert s.astype(int).tolist() == succ and f.astype(int).tolist() == fail
    assert tk.carry_latch(np, task, x["carried"], x["jaws"], x["cube"]).astype(int).tolist() == LEGACY_LATCH


@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_the_legacy_bottom_is_the_old_height(name, station, limits):
    """Passing the legacy cube's `object_bottom` changes nothing: it is the centre minus `cube_half`."""
    task = tk.make_task(name, spec=station.spec())
    assert tk.legacy_cube(task)
    x = _legacy_batch(task, limits)
    tilted = np.array([np.linalg.qr(np.random.default_rng(k).normal(size=(3, 3)))[0] for k in range(12)])
    bottom = tk.object_bottom(np, task, x["cube"], tilted)
    assert np.array_equal(bottom, x["cube"][:, 2] - task.cube_half)
    kw = dict(target=x["target"], cube_vel=x["vel"], carried=x["carried"], grip_open=x["opened"]) \
        if task.target else {}
    args = (np, task, x["tcp"], x["act"], x["prev"])
    a = tk.reward(*args, cube_pos=x["cube"], jaw_contacts=x["jaws"], q_cmd=x["q_cmd"], limits=limits, **kw)
    b = tk.reward(*args, cube_pos=x["cube"], jaw_contacts=x["jaws"], q_cmd=x["q_cmd"], limits=limits,
                  bottom=bottom, **kw)
    for u, v in zip(a, b, strict=True):
        assert np.array_equal(u, v)
    assert tk.carry_latch(np, task, x["carried"], x["jaws"], x["cube"], bottom=bottom).astype(int).tolist() \
        == LEGACY_LATCH



def test_numpy_and_jax_perceive_the_same_pose(limits):
    """The cameras' pose (offset, noise, fold), the occlusion latch and the gripper servo's own speed:
    one definition for both envs."""
    rng = np.random.default_rng(21)
    pos = rng.normal(0, 0.2, (N, 3))
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(N)])
    bias, noise = rng.uniform(-0.03, 0.03, (N, 2)), rng.normal(0, 0.002, (N, 2))
    tcp, cube = rng.normal(0, 0.2, (N, 3)), rng.normal(0, 0.2, (N, 3))
    tcp[::3] = cube[::3] + rng.normal(0, 0.01, (len(tcp[::3]), 3))        # the hand over the object
    task = tk.make_task("lift")
    g = limits.grip
    ctrl, target = rng.uniform(limits.lo[g], limits.hi[g], N), rng.uniform(limits.lo[g], limits.hi[g], N)
    rate = rng.uniform(0.01, 0.2, N)
    close_s = np.tile([0.0, 0.33, 1.3, -1.0], N // 4)

    def run(xp):
        def f(a):
            return np.asarray(a, float) if xp is np else jnp.asarray(a, jnp.float32)

        out = {}
        for fold in (True, False):
            p, r = tk.perceived(xp, f(pos), f(rot), f(bias), f(noise), fold, np.pi / 2)
            out[f"pos_{fold}"], out[f"rot_{fold}"] = p, r
        out["occluded"] = tk.occluded(xp, f(tcp), f(cube), 0.05)
        out["never"] = tk.occluded(xp, f(tcp), f(cube), 0.0)
        out["grip_rate"] = tk.grip_rate(xp, task, limits, f(close_s))
        out["follow"] = tk.servo_follow(xp, f(ctrl), f(target), f(rate))
        out["follow_inf"] = tk.servo_follow(xp, f(ctrl), f(target), f(np.full(N, np.inf)))
        return {k: np.asarray(v) for k, v in out.items()}

    a, b = run(np), run(jnp)
    for k in a:
        assert a[k].shape == b[k].shape, k
        if a[k].dtype == bool:
            assert np.array_equal(a[k], b[k]), k
        else:
            assert b[k].dtype == np.float32, k
            np.testing.assert_allclose(b[k], a[k], rtol=1e-5, atol=1e-6, err_msg=k)
    assert 0 < a["occluded"].sum() < N and a["occluded"][::3].all()          # both sides of the latch
    assert a["never"].shape == (N,) and not a["never"].any()
    np.testing.assert_array_equal(a["pos_True"][:, :2], pos[:, :2] + bias + noise)
    np.testing.assert_array_equal(a["pos_True"][:, 2], pos[:, 2])                  # z is the truth
    assert np.array_equal(a["rot_False"], rot)
    # Without an offset and noise the pose is the true one, bit for bit.
    zero = np.zeros((N, 2))
    assert np.array_equal(tk.perceived(np, pos, rot, zero, zero, False, np.pi / 2)[0], pos)

    # The servo: the full span in close_s at 20 Hz, none of its own for close_s <= 0.
    span = limits.hi[g] - limits.lo[g]
    np.testing.assert_allclose(a["grip_rate"][1:3], [span / (0.33 * 20), span / 26])
    assert np.isinf(a["grip_rate"][0]) and np.isinf(a["grip_rate"][3])
    step = a["follow"] - ctrl
    assert (np.abs(step) <= rate + 1e-12).all() and (np.abs(step) < np.abs(target - ctrl) - 1e-12).any()
    np.testing.assert_allclose(step, np.clip(target - ctrl, -rate, rate), atol=1e-15)
    # No speed of its own: exactly the command, in both backends.
    assert np.array_equal(a["follow_inf"], target)
    assert np.array_equal(b["follow_inf"], target.astype(np.float32))
