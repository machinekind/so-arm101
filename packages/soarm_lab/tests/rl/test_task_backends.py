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

from soarm_lab import blocks  # noqa: E402
from soarm_lab import sim as sc  # noqa: E402
from soarm_lab.rl import tasks as tk  # noqa: E402
from soarm_lab.rl.env import scene_config  # noqa: E402

#: Worlds in a batch.
N = 32
#: Descriptor order for the task variants under test.
VARIANTS = ("cube", "cuboid", "cylinder")


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
    rest = task.block_spec.half_size[2] if task.cube else 0.015
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
    if task.target:
        kw = dict(target=f(x["target"]), cube_vel=f(x["vel"]), carried=b(x["carried"]),
                  grip_open=b(x["opened"]))
    period = task.block_spec.yaw_period if cube else np.pi / 2
    out = {
        "observe": task.observe(xp, lim, f(x["q"]), f(x["q_cmd"]), f(x["tcp"]), f(x["prev"]),
                              goal=f(x["goal"]), cube_pos=f(x["cube"]) if cube else None,
                              cube_rot=f(x["rot"]) if cube else None, target=f(x["target"])),
        "apply_action": tk.apply_action(xp, task, lim, f(x["q_cmd"]), f(x["act"])),
        "fold_yaw": tk.fold_yaw(xp, f(x["rot"]), period),
        "carry_latch": tk.carry_latch(xp, task if task.cube else tk.make_task("lift"),
                                      b(x["carried"]), f(x["jaws"]), f(x["cube"]), f(x["rot"])),
        "grip_is_open": tk.grip_is_open(xp, lim, f(x["q_cmd"])),
        # reach has no limit penalty of its own; `limit_cost` is checked with one anyway.
        "limit_cost": tk.limit_cost(xp, replace(task, limit_penalty=0.5), lim, f(x["q_cmd"])),
    }
    r, s, fail = task.reward(xp, f(x["tcp"]), f(x["act"]), f(x["prev"]), goal=f(x["goal"]),
                           cube_pos=f(x["cube"]) if cube else None, cube_rot=f(x["rot"]) if cube else None,
                           jaw_contacts=f(x["jaws"]) if cube else None, q_cmd=f(x["q_cmd"]), limits=lim, **kw)
    out.update(reward=r, success=s, failure=fail)
    if cube:
        out["height"] = task.height(xp, f(x["cube"]), f(x["rot"]))
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


@pytest.mark.parametrize("own", VARIANTS)
@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_numpy_and_jax_compute_the_same_task_with_variants(name, own, station, limits):
    """NumPy and JAX agree on variant observations and lowest-point rewards."""
    task = tk.make_task(name, spec=station.spec(), variants=VARIANTS, block=own)
    x = inputs(task, limits, seed=10 + VARIANTS.index(own))
    a = _compare(task, limits, x)
    plain = replace(task, variants=()).observe(np, limits, x["q"], x["q_cmd"], x["tcp"], x["prev"],
                                               cube_pos=x["cube"], cube_rot=x["rot"], target=x["target"])
    k = plain.shape[1]
    assert a["observe"].shape == (N, k + len(VARIANTS) + 3)
    assert np.array_equal(a["observe"][:, :k], plain)
    assert np.array_equal(a["observe"][:, k:], np.broadcast_to(task.descriptor(), (N, len(VARIANTS) + 3)))
    np.testing.assert_array_equal(a["height"], blocks.bottom_height(np, task.block_spec, x["cube"], x["rot"]))
    assert 0 < a["success"].sum() < N and 0 < a["failure"].sum() < N


@pytest.mark.parametrize("own", VARIANTS)
def test_numpy_and_jax_compute_the_same_sized_block(own, station, limits):
    """NumPy and JAX agree on block scales, descriptors, lowest points and resting heights."""
    task = tk.make_task("lift", spec=station.spec(), variants=VARIANTS, block=own)
    spec = task.block_spec
    x = inputs(task, limits, seed=30 + VARIANTS.index(own))
    draws = np.random.default_rng(7).uniform(-5.0, 5.0, (3, N))         # side changes [mm]
    size = blocks.scale_from_mm(np, spec, *draws)
    assert size.shape == (N, 3)
    np.testing.assert_allclose(np.asarray(blocks.scale_from_mm(jnp, spec, *jnp.asarray(draws, jnp.float32))),
                               size, rtol=1e-6)
    nominal = 2000.0 * np.asarray(spec.half_size)                         # [mm]
    assert np.abs(nominal * size - nominal).max() <= 5.0 + 1e-9
    shared = {"cube": (0, 1), "cylinder": (1, 2)}.get(own)
    if shared:
        assert np.array_equal(size[:, shared[0]], size[:, shared[1]])
    assert len({tuple(c) for c in size.T.round(12).tolist()}) == (2 if shared else 3)

    def run(xp):
        def f(a):
            return np.asarray(a, float) if xp is np else jnp.asarray(a, jnp.float32)

        obs = task.observe(xp, limits, f(x["q"]), f(x["q_cmd"]), f(x["tcp"]), f(x["prev"]),
                           cube_pos=f(x["cube"]), cube_rot=f(x["rot"]), size=f(size))
        height = task.height(xp, f(x["cube"]), f(x["rot"]), f(size))
        return {k: np.asarray(v) for k, v in
                {"observe": obs, "height": height, "rest": task.rest_heights(f(size))}.items()}

    a, b = run(np), run(jnp)
    for k in a:
        assert b[k].dtype == np.float32 and a[k].shape == b[k].shape, k
        np.testing.assert_allclose(b[k], a[k], rtol=1e-5, atol=1e-6, err_msg=k)
    np.testing.assert_array_equal(a["height"], blocks.bottom_height(np, spec, x["cube"], x["rot"], size))
    np.testing.assert_allclose(a["rest"], spec.half_size[2] * size[:, 2], rtol=1e-15)
    one = np.ones((N, 3))
    args = (np, limits, x["q"], x["q_cmd"], x["tcp"], x["prev"])
    plain = task.observe(*args, cube_pos=x["cube"], cube_rot=x["rot"])
    assert np.array_equal(a["observe"], plain)
    assert np.array_equal(task.height(np, x["cube"], x["rot"], one), task.height(np, x["cube"], x["rot"]))


def test_the_3cm_cube_scales_like_any_block(station):
    """The cube preserves its yaw symmetry and uses scaled dimensions to compute its lowest point."""
    task = tk.make_task("lift", spec=station.spec())
    draws = np.random.default_rng(8).uniform(-5.0, 5.0, (3, 4))         # side changes [mm]
    size = blocks.scale_from_mm(np, task.block_spec, *draws)
    assert np.array_equal(size[:, 0], size[:, 1]) and not np.array_equal(size[:, 1], size[:, 2])
    pos, upright = np.tile([0.2, 0.0, 0.05], (4, 1)), np.tile(np.eye(3), (4, 1, 1))
    # The bottom moves by half the height change.
    np.testing.assert_allclose(task.height(np, pos, upright, size), 0.05 - (30.0 + draws[2]) / 2000.0,
                               rtol=1e-12)


@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_the_lift_verdicts_use_the_scaled_block(name, station, limits):
    """Success and carrying depend on the scaled block's lowest point."""
    task = tk.make_task(name, spec=station.spec(), block="cuboid")
    x = _pinned_batch(task, limits)
    pos = np.array([[0.2, 0.0, task.block_spec.half_size[2] + task.lift_height + 0.001]])
    rot, jaws, no = x["rot"][:1], np.ones((1, 2)), np.zeros(1, bool)
    # A block 3 mm taller has its bottom 1.5 mm lower, below the 1 mm margin.
    same = np.ones((1, 3))
    taller = blocks.scale_from_mm(np, task.block_spec, *np.array([[0.0], [0.0], [3.0]]))
    assert tk.carry_latch(np, task, no, jaws, pos, rot, same).tolist() == [True]
    assert tk.carry_latch(np, task, no, jaws, pos, rot, taller).tolist() == [False]
    if name == "lift":
        args = (np, x["tcp"][:1], x["act"][:1], x["prev"][:1])
        kw = dict(cube_pos=pos, cube_rot=rot, jaw_contacts=jaws, q_cmd=x["q_cmd"][:1], limits=limits)
        assert task.reward(*args, size=same, **kw)[1].tolist() == [True]
        assert task.reward(*args, size=taller, **kw)[1].tolist() == [False]


def _pinned_batch(task: tk.TaskConfig, lim: tk.Limits) -> dict[str, np.ndarray]:
    """Build fixed states spanning grasp, lift, placement, and failure conditions."""
    rng = np.random.default_rng(5)
    n, nj = 12, task.act_dim
    tcp = rng.normal(0, 0.05, (n, 3)) + [0.2, 0.0, 0.05]
    cube = tcp + rng.normal(0, 0.01, (n, 3))
    cube[:, 2] = task.block_spec.half_size[2] + np.linspace(-0.07, 0.10, n)
    jaws = np.ones((n, 2))
    jaws[::3] = 0.0
    target = np.c_[cube[:, :2] + rng.normal(0, 0.02, (n, 2)), np.zeros(n)]
    vel = rng.normal(0, 0.01, (n, 3))
    q_cmd = rng.uniform(lim.lo, lim.hi, (n, nj))
    act, prev = rng.uniform(-1, 1, (n, nj)), rng.uniform(-1, 1, (n, nj))
    return dict(tcp=tcp, cube=cube, rot=np.broadcast_to(np.eye(3), (n, 3, 3)), jaws=jaws, target=target,
                vel=vel, carried=np.arange(n) % 2 == 0, opened=np.arange(n) % 4 < 2, q_cmd=q_cmd, act=act,
                prev=prev)


#: Expected rewards, successes, and failures for the upright default cube in `_pinned_batch`.
PINNED_REWARDS = {
    "lift": ([-0.05087789, 1.010613325, 0.988474619, 0.830648662, 1.148904974, 1.832259039, 0.284883369,
              3.803095015, 4.393715791, 0.242831223, 6.846265757, 6.189205521],
             [0] * 10 + [1, 1], [1, 1] + [0] * 10),
    "pickplace": ([10.175731089, 1.010613325, 3.637924755, 0.830648662, 7.987700552, 1.589834797,
                   3.18401043, 2.530367742, 6.250467077, 0.242831223, 6.107233112, 3.005231585],
                  [1] + [0] * 11, [1, 1] + [0] * 10),
}
PINNED_LATCH = [1, 0] * 5 + [1, 1]


@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_the_reward_terms_are_pinned(name, station, limits):
    """The upright default cube preserves grasp, lift, placement, and failure rewards."""
    task = tk.make_task(name, spec=station.spec())
    x = _pinned_batch(task, limits)
    kw = dict(target=x["target"], cube_vel=x["vel"], carried=x["carried"], grip_open=x["opened"]) \
        if task.target else {}
    r, s, f = task.reward(np, x["tcp"], x["act"], x["prev"], cube_pos=x["cube"], cube_rot=x["rot"],
                          jaw_contacts=x["jaws"], q_cmd=x["q_cmd"], limits=limits, **kw)
    rew, succ, fail = PINNED_REWARDS[name]
    np.testing.assert_allclose(r, rew, atol=2e-9)
    assert s.astype(int).tolist() == succ and f.astype(int).tolist() == fail
    latch = tk.carry_latch(np, task, x["carried"], x["jaws"], x["cube"], x["rot"])
    assert latch.astype(int).tolist() == PINNED_LATCH


@pytest.mark.parametrize("name", ("lift", "pickplace"))
def test_a_tilted_cube_counts_its_lowest_point(name, station, limits):
    """A tilted cube uses its lowest corner for height, carrying, and lift success."""
    task = tk.make_task(name, spec=station.spec())
    x = _pinned_batch(task, limits)
    c = s = np.sqrt(0.5)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])            # 45 deg about x
    rot = np.broadcast_to(tilt, (12, 3, 3))
    flat = task.height(np, x["cube"], x["rot"])
    np.testing.assert_allclose(flat, x["cube"][:, 2] - 0.015, atol=1e-15)
    tilted = task.height(np, x["cube"], rot)
    np.testing.assert_allclose(tilted, x["cube"][:, 2] - 0.015 * np.sqrt(2.0), atol=1e-15)
    # Tilting lowers the corner below the lift threshold.
    z = 0.015 + task.lift_height + 0.002
    pos = np.array([[0.2, 0.0, z]])
    jaws, no = np.ones((1, 2)), np.zeros(1, bool)
    assert tk.carry_latch(np, task, no, jaws, pos, x["rot"][:1]).tolist() == [True]
    assert tk.carry_latch(np, task, no, jaws, pos, tilt[None]).tolist() == [False]
    if name == "lift":
        args = (np, x["tcp"][:1], x["act"][:1], x["prev"][:1])
        kw = dict(cube_pos=pos, jaw_contacts=jaws, q_cmd=x["q_cmd"][:1], limits=limits)
        assert task.reward(*args, cube_rot=x["rot"][:1], **kw)[1].tolist() == [True]
        assert task.reward(*args, cube_rot=tilt[None], **kw)[1].tolist() == [False]


def test_numpy_and_jax_perceive_the_same_pose(limits):
    """NumPy and JAX agree on perception, occlusion and gripper speed limits."""
    rng = np.random.default_rng(21)
    pos = rng.normal(0, 0.2, (N, 3))
    rot = np.array([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(N)])
    bias, noise = rng.uniform(-0.03, 0.03, (N, 2)), rng.normal(0, 0.002, (N, 2))
    tcp, cube = rng.normal(0, 0.2, (N, 3)), rng.normal(0, 0.2, (N, 3))
    # Put some TCP positions inside the occlusion radius.
    tcp[::3] = cube[::3] + rng.normal(0, 0.01, (len(tcp[::3]), 3))
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
    assert 0 < a["occluded"].sum() < N and a["occluded"][::3].all()
    assert a["never"].shape == (N,) and not a["never"].any()
    np.testing.assert_array_equal(a["pos_True"][:, :2], pos[:, :2] + bias + noise)
    np.testing.assert_array_equal(a["pos_True"][:, 2], pos[:, 2])
    assert np.array_equal(a["rot_False"], rot)
    zero = np.zeros((N, 2))
    assert np.array_equal(tk.perceived(np, pos, rot, zero, zero, False, np.pi / 2)[0], pos)

    # Nonpositive closing times disable the servo speed limit.
    span = limits.hi[g] - limits.lo[g]
    np.testing.assert_allclose(a["grip_rate"][1:3], [span / (0.33 * 20), span / 26])
    assert np.isinf(a["grip_rate"][0]) and np.isinf(a["grip_rate"][3])
    step = a["follow"] - ctrl
    assert (np.abs(step) <= rate + 1e-12).all() and (np.abs(step) < np.abs(target - ctrl) - 1e-12).any()
    np.testing.assert_allclose(step, np.clip(target - ctrl, -rate, rate), atol=1e-15)
    assert np.array_equal(a["follow_inf"], target)
    assert np.array_equal(b["follow_inf"], target.astype(np.float32))
