"""The policies BUNDLED with the package (`soarm_lab/policies`) on the CPU: these drive the real arm.

The scene is the package's `sim` station.

The other RL tests check the environment and the reward; these check how the policy files the
operator picks in the panel behave. A test that only asks "did it lift" passes the old `lift-v2`,
which ended with the cube 13-22 cm above the table (task: 6 cm), held arm joints at their limits for
tens of ticks and swung up to 72 deg after the grasp. The thresholds below pass `lift-v3` (measured:
cube at 8-14 cm, 6 ticks at a limit after the grasp, up to 46 deg of motion after the first success)
and stop `lift-v2`: cube at 16-18 cm on average, up to 22 cm, 24-56 ticks at a limit after the
grasp, up to 72 deg of motion after success.

An episode on the CPU takes ~0.07 s, so 50 episodes take a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab.rl.env import ArmEnv  # noqa: E402
from soarm_lab.rl.policy import Policy, bundled_dir  # noqa: E402
from soarm_lab.rl.randomize import Randomization  # noqa: E402

REACH = bundled_dir() / "reach-v3" / "policy.npz"
LIFT = bundled_dir() / "lift-v3" / "policy.npz"
#: "At a limit" = a joint command closer than 1 deg to its limit (`task.Limits`).
NEAR_LIMIT = np.radians(1.0)


@dataclass
class Episode:
    q_cmd: np.ndarray            # (T, 6) commands [rad]
    success: list[bool]
    grasped: list[bool]
    info: dict
    end: float                   # reach: TCP distance to the goal [m]; lift: cube height above the table [m]

    def near_limit(self, limits, joints=slice(0, 5)) -> np.ndarray:
        q = self.q_cmd[:, joints]
        return (np.abs(q - limits.lo[joints]) < NEAR_LIMIT) | (np.abs(q - limits.hi[joints]) < NEAR_LIMIT)

    def first(self, flags: list[bool]) -> int | None:
        return next((i for i, f in enumerate(flags) if f), None)


def run(policy: Policy, rand: Randomization, episodes: int, seed: int = 10_000) -> tuple[list[Episode], object]:
    task = policy.task
    env = ArmEnv(task, randomization=rand)
    out = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            qs, succ, grasp = [], [], []
            for _ in range(task.episode_steps):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                qs.append(env.q_cmd.copy())
                succ.append(info["success"])
                grasp.append(info.get("grasped", False))
                if term or trunc:
                    break
            end = info["distance"] if task.name == "reach" else info["height"]
            out.append(Episode(np.array(qs), succ, grasp, info, end))
        return out, env.limits
    finally:
        env.close()


def lift_report(policy: Policy, rand: Randomization, episodes: int) -> dict:
    """The numbers that tell `lift-v3` from `lift-v2` (see the module docstring)."""
    eps, limits = run(policy, rand, episodes)
    heights = np.array([e.end for e in eps])
    limit_after_grasp, swing = 0, 0.0
    for e in eps:
        g, s = e.first(e.grasped), e.first(e.success)
        if g is not None:
            limit_after_grasp += int(e.near_limit(limits)[g:].any(axis=1).sum())
        if s is not None:
            after = e.q_cmd[s:, :5]
            swing = max(swing, float(np.degrees(after.max(0) - after.min(0)).max()))
    return {"finished": sum(e.info["finished"] and e.info["success"] for e in eps), "episodes": len(eps),
            "height_min": heights.min(), "height_mean": heights.mean(), "height_max": heights.max(),
            "limit_ticks_after_grasp": limit_after_grasp, "swing_after_success_deg": swing}


def check_lift(rep: dict, lift_height: float) -> None:
    # It ends by itself: a success streak (`end_on_success`), not the time limit, with the cube up.
    assert rep["finished"] == rep["episodes"], rep
    assert rep["height_min"] > lift_height, rep
    # No waving the cube up to the ceiling: lift-v2 averaged 16-18 cm, up to 22 cm.
    assert rep["height_max"] < 0.16 and rep["height_mean"] < 0.135, rep
    # Arm joints do not sit at their limits after the grasp (lift-v2: 24 and 56 ticks).
    assert rep["limit_ticks_after_grasp"] <= 12, rep
    # After the first success the arm only holds and finishes (lift-v2: up to 72 deg of joint motion).
    # Threshold 70 deg: the XML rounding of the exported scene scatters the trajectory after contact
    # from run to run.
    assert rep["swing_after_success_deg"] < 70.0, rep


def test_both_base_policies_are_bundled():
    assert REACH.is_file() and LIFT.is_file()
    assert REACH.with_name("meta.json").is_file() and LIFT.with_name("meta.json").is_file()


def test_shipped_lift_lifts_ends_and_keeps_the_arm_calm():
    pol = Policy.load(LIFT)
    check_lift(lift_report(pol, Randomization(), 30), pol.task.lift_height)


def test_shipped_reach_ends_near_the_goal():
    pol = Policy.load(REACH)
    eps, limits = run(pol, Randomization.nominal(), 20)
    dist = np.array([e.end for e in eps])
    assert all(e.info["success"] for e in eps)
    assert dist.max() < 0.005 and np.median(dist) < 0.003, dist           # measured: median 1.8, max 2.9 mm
    # No arm joint drives to a limit, the wrist roll included. The roll does not change the TCP, so
    # the reach reward's TCP terms do not see it: reach-v2 turned it to +150 deg in every episode
    # (729 of 2000 ticks at the limit). reach-v3, trained with `roll_penalty`: median 13 deg, max 48.
    near = sum(int(e.near_limit(limits).sum()) for e in eps)
    assert near == 0, near
    roll = np.degrees(np.abs(np.concatenate([e.q_cmd[:, 4] for e in eps]) - limits.home[4]))
    assert roll.max() < 90.0 and np.median(roll) < 30.0, (np.median(roll), roll.max())
