"""Grasp and lift a block."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ... import blocks
from ...robot import RobotSpec
from .base import TaskBase, _as, _cast, _cat, _clip, _norm, action_cost, normalize_q

#: Descriptor half extents are normalized by this length [m].
SIZE_SCALE = 0.05


@dataclass(frozen=True)
class LiftConfig(TaskBase):
    name: str = "lift"
    # Maximum policy ticks per episode.
    episode_steps: int = 200
    # Consecutive successes that truncate the episode.
    end_on_success: int = 10
    # Weight of the penalty for arm commands near joint limits.
    limit_penalty: float = 0.5
    # Catalog name of the scene object.
    block: str = "cube3cm"
    # Descriptor indices follow these block names. Empty omits the descriptor.
    variants: tuple[str, ...] = ()
    # Cube sampling radius range from the base axis [m].
    cube_radius: tuple[float, float] = (0.15, 0.25)
    # Cube sampling bearing range in the base frame [rad].
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    # Height of the block's lowest point above the table needed for lifting or carrying [m].
    lift_height: float = 0.06
    # Weight of squared arm actions during success.
    hold_still: float = 0.5

    def __post_init__(self) -> None:
        for name in (self.block, *self.variants):
            blocks.block(name)
        if len(set(self.variants)) != len(self.variants):
            raise ValueError(f"a variant repeats in {tuple(self.variants)}")
        if self.variants and self.block not in self.variants:
            raise ValueError(f"the block {self.block!r} is not one of the variants {tuple(self.variants)}")

    @property
    def obs_dim(self) -> int:
        size = len(self.variants) + 3 if self.variants else 0
        return 2 * self.n_joints + 3 + 3 + 3 + 6 + self.n_joints + size

    @property
    def cube(self) -> Literal["cube"]:
        return "cube"

    @property
    def target(self) -> str | None:
        return None

    @property
    def block_spec(self) -> blocks.BlockSpec:
        return blocks.block(self.block)

    @property
    def variant(self) -> str:
        return self.block

    def with_variant(self, name: str):
        """Return this task with block `name`, restricted to `variants` when set."""
        if self.variants and name not in self.variants:
            raise ValueError(f"{name!r} is not a variant of this task; its variants: {tuple(self.variants)}")
        return replace(self, block=name)

    def variant_tasks(self) -> list:
        """One task per variant, in the order of `variants`."""
        return [self.with_variant(name) for name in self.variants]

    def descriptor(self) -> np.ndarray:
        """Encode the block as one-hot `variants` followed by half extents divided by `SIZE_SCALE`."""
        n = len(self.variants)
        out = np.zeros(n + 3, np.float32)
        out[self.variants.index(self.block)] = 1.0
        out[n:] = np.asarray(self.block_spec.half_size) / SIZE_SCALE
        return out

    def height(self, xp, cube_pos, cube_rot):
        """Return lowest-point heights above the table [m] as an (N,) array."""
        return blocks.bottom_height(xp, self.block_spec, cube_pos, cube_rot)

    def _observe_parts(self, xp, limits, q, q_cmd, tcp, cube_pos, cube_rot):
        parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
        rot6 = _cat(xp, [cube_rot[..., :, 0], cube_rot[..., :, 1]])
        return parts + [cube_pos, cube_pos - tcp, rot6]

    def _with_descriptor(self, xp, parts, q):
        """Concatenate observation parts and append the descriptor when variants are configured."""
        if self.variants:
            parts = parts + [_as(xp, self.descriptor(), q) + xp.zeros_like(q[..., :1])]
        return _cat(xp, parts)

    def observe(self, xp, limits, q, q_cmd, tcp, prev_action, cube_pos=None, cube_rot=None, **_):
        """Pack q(n), commands(n), TCP(3), cube(3), cube-TCP(3), rotation(6), and previous action(n).

        Append the descriptor when variants are configured.
        """
        parts = self._observe_parts(xp, limits, q, q_cmd, tcp, cube_pos, cube_rot)
        return self._with_descriptor(xp, parts + [prev_action], q)

    def reward(self, xp, tcp, action, prev_action, cube_pos=None, cube_rot=None, jaw_contacts=None,
               q_cmd=None, limits=None, **_):
        """Count success when both jaws touch and the lowest point exceeds `lift_height`.

        Falls terminate episodes. Time limits and success streaks truncate them.
        """
        cost = action_cost(xp, self, action, prev_action, q_cmd, limits)
        d = _norm(cube_pos - tcp)
        grasped = (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)
        height = self.height(xp, cube_pos, cube_rot)
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
        p = jnp.stack([r * jnp.cos(b), r * jnp.sin(b), jnp.full_like(r, self.block_spec.half_size[2])], 1)
        yaw = u(k_yaw, (n,), minval=-np.pi, maxval=np.pi)
        return {"cube_pos": p, "cube_yaw": yaw}

    def for_spec(self, spec: RobotSpec):
        d = spec.desk
        return replace(self, n_joints=len(spec.joints), cube_radius=d.cube_radius,
                       cube_bearing=d.cube_bearing)


def sample_cubes(task: LiftConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.block_spec.half_size[2])], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    return pos, quat
