"""Reach a point in the arm base frame."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...kinematics import RobotKinematics
from ...robot import RobotSpec
from .base import TaskBase, _cat, _norm, action_cost, normalize_q


@dataclass(frozen=True)
class ReachConfig(TaskBase):
    name: str = "reach"
    # Weight of squared wrist roll displacement from home, normalized by pi.
    roll_penalty: float = 0.5
    # Maximum TCP distance from the goal for success [m].
    success_dist: float = 0.02
    # Goal sampling radius range from the base axis [m].
    goal_radius: tuple[float, float] = (0.13, 0.32)
    # Goal sampling height range above the table [m].
    goal_height: tuple[float, float] = (0.03, 0.22)

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_joints + 3 + 3 + 3 + self.n_joints

    @property
    def cube(self) -> None:
        return None

    @property
    def target(self) -> None:
        return None

    def observe(self, xp, limits, q, q_cmd, tcp, prev_action, goal=None, **_):
        """Layout: q(n), commands(n), TCP(3), goal(3), goal-TCP(3), previous action(n)."""
        parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
        parts += [goal, goal - tcp]
        parts.append(prev_action)
        return _cat(xp, parts)

    def reward(self, xp, tcp, action, prev_action, goal=None, q_cmd=None, limits=None, **_):
        """Success is a TCP within success_dist of the goal.
        The episode ends only at its time limit or success streak.
        """
        cost = action_cost(xp, self, action, prev_action, q_cmd, limits)
        d = _norm(goal - tcp)

        r = 0.5 * (1.0 - xp.tanh(d / 0.10)) + 0.5 * (1.0 - xp.tanh(d / 0.01))
        success = d < self.success_dist
        failure = d < -1.0
        return r - cost, success, failure

    def sample_reset(self, jax, n, keys, goal_pool, **_):
        k = jax.random.randint(keys[0], (n,), 0, goal_pool.shape[0])
        return {"goal": goal_pool[k]}

    def for_spec(self, spec: RobotSpec):
        d = spec.desk
        return replace(self, n_joints=len(spec.joints), goal_radius=d.goal_radius,
                       goal_height=d.goal_height)


def sample_goals(kin: RobotKinematics, task: ReachConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    spec = kin.spec
    out = []
    ranges = {**{j: (np.degrees(lo), np.degrees(hi)) for j, lo, hi in zip(spec.joints, kin.lo, kin.hi)},
              **spec.wave_ranges}
    while len(out) < n:
        j = {name: float(rng.uniform(*ranges[name])) for name in spec.arm_joints}
        p = kin.tcp(j)[:3, 3]
        r = float(np.hypot(p[0], p[1]))
        if task.goal_radius[0] <= r <= task.goal_radius[1] and task.goal_height[0] <= p[2] <= task.goal_height[1] \
                and p[0] > 0.05:
            out.append(p)
    return np.array(out)
