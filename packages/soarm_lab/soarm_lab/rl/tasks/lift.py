"""Grasp and lift a cube."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ...robot import RobotSpec
from .base import TaskBase, _cast, _cat, _clip, _norm, action_cost, normalize_q


@dataclass(frozen=True)
class LiftConfig(TaskBase):
    name: str = "lift"
    # Maximum policy ticks per episode.
    episode_steps: int = 200
    # Consecutive successes that truncate the episode.
    end_on_success: int = 10
    # Weight of the penalty for arm commands near joint limits.
    limit_penalty: float = 0.5
    # Cube half side length [m].
    cube_half: float = 0.015
    # Cube mass [kg].
    cube_mass: float = 0.03
    # Cube sampling radius range from the base axis [m].
    cube_radius: tuple[float, float] = (0.15, 0.25)
    # Cube sampling bearing range in the base frame [rad].
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    # Cube bottom height above the table needed for lifting or carrying [m].
    lift_height: float = 0.06
    # Weight of squared arm actions during success.
    hold_still: float = 0.5

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_joints + 3 + 3 + 3 + 6 + self.n_joints

    @property
    def cube(self) -> Literal["cube"]:
        return "cube"

    @property
    def target(self) -> str | None:
        return None

    def _observe_parts(self, xp, limits, q, q_cmd, tcp, cube_pos, cube_rot):
        parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
        rot6 = _cat(xp, [cube_rot[..., :, 0], cube_rot[..., :, 1]])
        return parts + [cube_pos, cube_pos - tcp, rot6]

    def observe(self, xp, limits, q, q_cmd, tcp, prev_action, cube_pos=None, cube_rot=None, **_):
        """Layout: q(n), commands(n), TCP(3), cube(3), cube-TCP(3), rotation(6), previous action(n)."""
        parts = self._observe_parts(xp, limits, q, q_cmd, tcp, cube_pos, cube_rot)
        return _cat(xp, parts + [prev_action])

    def reward(self, xp, tcp, action, prev_action, cube_pos=None, jaw_contacts=None,
               q_cmd=None, limits=None, **_):
        """Success is a cube grasped above lift_height.
        Falls terminate episodes. Time limits and success streaks truncate them.
        """
        cost = action_cost(xp, self, action, prev_action, q_cmd, limits)
        d = _norm(cube_pos - tcp)
        grasped = (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)
        height = cube_pos[..., 2] - self.cube_half
        lifted = _clip(xp, height / self.lift_height, 0.0, 1.0)
        success = grasped & (height > self.lift_height)
        g, s = _cast(xp, grasped, tcp), _cast(xp, success, tcp)
        r = (1.0 - xp.tanh(d / 0.05)) + g + 4.0 * g * lifted + 2.0 * s
        if self.hold_still > 0:
            na = action.shape[-1] - 1 if limits is None else limits.n_arm
            cost = cost + self.hold_still * s * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
        failure = height < -0.05
        return r - cost, success, failure

    def sample_reset(self, jax, n, keys, **_):
        jnp = jax.numpy
        _, k_r, k_b, k_yaw, _ = keys
        u = jax.random.uniform
        r = u(k_r, (n,), minval=self.cube_radius[0], maxval=self.cube_radius[1])
        b = u(k_b, (n,), minval=self.cube_bearing[0], maxval=self.cube_bearing[1])
        p = jnp.stack([r * jnp.cos(b), r * jnp.sin(b), jnp.full_like(r, self.cube_half)], 1)
        yaw = u(k_yaw, (n,), minval=-np.pi, maxval=np.pi)
        return {"cube_pos": p, "cube_yaw": yaw}

    def for_spec(self, spec: RobotSpec):
        d = spec.desk
        return replace(self, n_joints=len(spec.joints), cube_radius=d.cube_radius,
                       cube_bearing=d.cube_bearing)


def sample_cubes(task: LiftConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.cube_half)], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    return pos, quat
