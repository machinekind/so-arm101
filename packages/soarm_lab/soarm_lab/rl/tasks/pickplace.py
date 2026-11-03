"""Carry a block onto a target and release it."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ...robot import RobotSpec
from .base import _cast, _clip, _norm, action_cost, grasped, grip_is_open
from .lift import LiftConfig


@dataclass(frozen=True)
class PickPlaceConfig(LiftConfig):
    name: str = "pickplace"
    # Maximum policy ticks per episode.
    episode_steps: int = 300
    # Maximum cube-to-target distance in the table plane for success [m].
    place_radius: float = 0.03
    # Maximum height of the block's lowest point above the table for placement [m].
    place_height: float = 0.01
    # Minimum initial cube-to-target separation in the table plane [m].
    target_min_sep: float = 0.10
    # Maximum cube speed for placement success [m/s].
    settle_speed: float = 0.02
    # Target sampling radius range from the base axis [m].
    target_radius: tuple[float, float] = (0.15, 0.25)
    # Target sampling bearing range in the base frame [rad].
    target_bearing: tuple[float, float] = (-0.8, 0.8)

    @property
    def obs_dim(self) -> int:
        return super().obs_dim + 3 + 3

    @property
    def target(self) -> Literal["target"]:
        return "target"

    def observe(self, xp, limits, q, q_cmd, tcp, prev_action, cube_pos=None, cube_rot=None,
                target=None, **_):
        """Pack the lift layout with target(3) and target-cube(3) before previous action(n).

        Append the descriptor when variants are configured.
        """
        parts = self._observe_parts(xp, limits, q, q_cmd, tcp, cube_pos, cube_rot)
        return self._with_descriptor(xp, parts + [target, target - cube_pos, prev_action], q)

    def reward(self, xp, tcp, action, prev_action, cube_pos=None, cube_rot=None, jaw_contacts=None, q_cmd=None,
               limits=None, target=None, cube_vel=None, carried=None, grip_open=None, **_):
        """Count success when a previously carried block rests on the target after release.
        Falls terminate episodes. Time limits and success streaks truncate them.
        """
        cost = action_cost(xp, self, action, prev_action, q_cmd, limits)
        opened = grip_open if grip_open is not None else grip_is_open(xp, limits, q_cmd)
        d = _norm(cube_pos - tcp)
        g = grasped(xp, jaw_contacts)
        height = self.height(xp, cube_pos, cube_rot)
        lifted = _clip(xp, height / self.lift_height, 0.0, 1.0)
        d_t = _norm(cube_pos[..., :2] - target[..., :2])
        placed = (d_t < self.place_radius) & (height < self.place_height)
        released = ~g & opened
        still = _norm(cube_vel) < self.settle_speed
        success = carried & placed & released & still
        gf, cf = _cast(xp, g, tcp), _cast(xp, carried, tcp)
        pf, sf = _cast(xp, placed, tcp), _cast(xp, success, tcp)
        r = ((1.0 - xp.tanh(d / 0.05)) + gf + 2.0 * gf * lifted + 4.0 * cf * (1.0 - xp.tanh(d_t / 0.10))
             + 4.0 * cf * pf + 4.0 * sf)
        if self.hold_still > 0:
            na = action.shape[-1] - 1 if limits is None else limits.n_arm
            cost = cost + self.hold_still * sf * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
        failure = height < -0.05
        return r - cost, success, failure

    def sample_reset(self, jax, n, keys, **_):
        sampled = super().sample_reset(jax, n, keys)
        return {**sampled, "target": sample_targets_jax(jax, self, keys[4], sampled["cube_pos"])}

    def for_spec(self, spec: RobotSpec):
        base = super().for_spec(spec)
        return replace(base, target_radius=spec.desk.cube_radius, target_bearing=spec.desk.cube_bearing)


def carry_latch(xp, task: LiftConfig, carried, jaw_contacts, cube_pos, cube_rot):
    """Latch carrying after a grasped block clears `lift_height`, before computing reward."""
    return carried | (grasped(xp, jaw_contacts) & (task.height(xp, cube_pos, cube_rot) > task.lift_height))


TARGET_TRIES = 64


def sample_targets(task: PickPlaceConfig, rng: np.random.Generator, cube_pos: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube_pos, float).reshape(-1, 3)
    n = len(cube)
    out = np.zeros((n, 3))
    bad = np.ones(n, bool)
    for _ in range(TARGET_TRIES):
        k = int(bad.sum())
        if k == 0:
            break
        r = rng.uniform(*task.target_radius, k)
        b = rng.uniform(*task.target_bearing, k)
        out[bad, 0], out[bad, 1] = r * np.cos(b), r * np.sin(b)
        bad = np.hypot(*(out[:, :2] - cube[:, :2]).T) <= task.target_min_sep
    if bad.any():
        cb = np.arctan2(cube[bad, 1], cube[bad, 0])
        lo, hi = task.target_bearing
        far = np.where(cb > 0.5 * (lo + hi), lo, hi)
        r = task.target_radius[1]
        out[bad, 0], out[bad, 1] = r * np.cos(far), r * np.sin(far)
    return out


def sample_targets_jax(jax, tsk: PickPlaceConfig, key, cube_p):
    jnp = jax.numpy
    n = cube_p.shape[0]

    def bad_rows(out):
        return jnp.sqrt(((out[:, :2] - cube_p[:, :2]) ** 2).sum(1)) <= tsk.target_min_sep

    def draw(carry):
        i, out, bad = carry
        kr, kb = jax.random.split(jax.random.fold_in(key, i))
        r = jax.random.uniform(kr, (n,), minval=tsk.target_radius[0], maxval=tsk.target_radius[1])
        b = jax.random.uniform(kb, (n,), minval=tsk.target_bearing[0], maxval=tsk.target_bearing[1])
        new = jnp.stack([r * jnp.cos(b), r * jnp.sin(b), jnp.zeros_like(r)], 1)
        out = jnp.where(bad[:, None], new, out)
        return i + 1, out, bad_rows(out)

    def more(carry):
        i, _, bad = carry
        return (i < TARGET_TRIES) & bad.any()

    init = (jnp.int32(0), jnp.zeros((n, 3), cube_p.dtype), jnp.ones(n, bool))
    _, out, bad = jax.lax.while_loop(more, draw, init)
    cb = jnp.arctan2(cube_p[:, 1], cube_p[:, 0])
    lo, hi = tsk.target_bearing
    far = jnp.where(cb > 0.5 * (lo + hi), lo, hi)
    r = tsk.target_radius[1]
    fallback = jnp.stack([r * jnp.cos(far), r * jnp.sin(far), jnp.zeros_like(far)], 1)
    return jnp.where(bad[:, None], fallback, out)
