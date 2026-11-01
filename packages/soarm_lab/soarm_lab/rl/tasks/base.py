"""Shared control, limits and array operations for RL tasks.

Arrays have a leading world dimension. Positions use metres in the arm base frame.
Joint angles use MJCF radians, with the gripper last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...kinematics import RobotKinematics


@dataclass(frozen=True)
class TaskBase:
    # Policy frequency [Hz].
    control_hz: float = 20.0
    # Physics integration step [s].
    timestep: float = 0.005
    # Maximum policy ticks per episode.
    episode_steps: int = 100
    # Largest arm joint target increment per tick [rad].
    arm_step: float = 0.05
    # Largest gripper move per tick, as a fraction of its range.
    grip_step: float = 0.15
    # Uniform noise around the arm home pose [deg].
    start_noise_deg: float = 8.0
    # Weight of squared action changes between ticks.
    action_rate: float = 0.02
    # Consecutive successes that truncate the episode (zero disables).
    end_on_success: int = 0
    # Weight of the penalty for arm commands near joint limits.
    limit_penalty: float = 0.0
    # Distance from a joint limit where the penalty starts [rad].
    limit_margin: float = 0.15
    # Weight of squared wrist roll displacement from home, normalized by pi.
    roll_penalty: float = 0.0
    # Number of controlled joints, with the gripper last.
    n_joints: int = 6

    @property
    def cube(self) -> str | None:
        return None

    @property
    def target(self) -> str | None:
        return None

    @property
    def variant(self) -> str | None:
        """Return the task variant name, or None for tasks without variants."""
        return None

    @property
    def act_dim(self) -> int:
        return self.n_joints

    @property
    def substeps(self) -> int:
        return int(round(1.0 / self.control_hz / self.timestep))


@dataclass(frozen=True)
class Limits:
    """Joint target ranges in radians, intersected with the safety limits."""

    # Lowest joint targets [rad].
    lo: np.ndarray
    # Highest joint targets [rad].
    hi: np.ndarray
    # Home joint targets clipped to these limits [rad].
    home: np.ndarray

    # Number of arm joints, excluding the gripper.
    n_arm: int
    # Gripper joint index.
    grip: int
    # Tool roll joint index, or None when absent.
    roll: int | None

    @staticmethod
    def of(kin: RobotKinematics) -> Limits:
        spec = kin.spec
        if spec.gripper != spec.joints[-1]:
            raise ValueError("zadania RL zakladaja chwytak jako ostatni staw")
        lo = kin.to_q({n: spec.limits[n][0] for n in spec.joints})
        hi = kin.to_q({n: spec.limits[n][1] for n in spec.joints})
        lo, hi = np.maximum(lo, kin.lo), np.minimum(hi, kin.hi)
        grip = len(spec.joints) - 1
        roll = spec.joints.index(spec.roll_joint) if spec.roll_joint else None
        return Limits(lo, hi, np.clip(kin.to_q(spec.home), lo, hi), n_arm=grip, grip=grip, roll=roll)


def _cat(xp, parts):
    return xp.concatenate(parts, axis=-1)


def _clip(xp, x, lo, hi):
    return xp.clip(x, lo, hi)


def _norm(x):
    return ((x * x).sum(-1)) ** 0.5


def _as(xp, a, like):
    return xp.asarray(a, dtype=like.dtype)


def _cast(xp, b, like):
    return b.astype(like.dtype)


def apply_action(xp, task: TaskBase, limits: Limits, q_cmd, action):
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo, q_cmd), _as(xp, limits.hi, q_cmd)
    a = _clip(xp, action, -1.0, 1.0)
    arm = _clip(xp, q_cmd[..., :na] + a[..., :na] * task.arm_step, lo[:na], hi[:na])
    span = hi[na:] - lo[na:]
    want = lo[na:] + (a[..., na:] + 1.0) * 0.5 * span
    step = task.grip_step * span
    grip = q_cmd[..., na:] + _clip(xp, want - q_cmd[..., na:], -step, step)
    return _cat(xp, [arm, grip])


def normalize_q(xp, limits: Limits, q):
    lo, hi = _as(xp, limits.lo, q), _as(xp, limits.hi, q)
    return 2.0 * (q - lo) / (hi - lo) - 1.0


def fold_yaw(xp, R, period):
    """Return yaw-only rotation matrices folded into [-period/2, period/2)."""
    yaw = xp.arctan2(R[..., 1, 0], R[..., 0, 0])
    yaw = (yaw + period / 2) % period - period / 2
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    rows = [xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)]
    return xp.stack(rows, -2)


def perceived(xp, pos, rot, bias, noise, fold: bool, period: float):
    """Apply perception errors to a delayed pose, returning position (N, 3) and rotation (N, 3, 3).

    `bias` and `noise` have shape (N, 2) and use metres. The z coordinate stays unchanged.
    `fold` folds yaw by `period` [rad].
    """
    xy = pos[..., :2] + bias + noise
    return _cat(xp, [xy, pos[..., 2:]]), (fold_yaw(xp, rot, period) if fold else rot)


def occluded(xp, tcp, cube_pos, latch_m: float):
    """Return an (N,) occlusion mask within `latch_m` [m] of the true TCP. Nonpositive values disable it."""
    if latch_m <= 0:
        return xp.zeros(tcp.shape[:-1], dtype=bool)
    return _norm(tcp - cube_pos) < latch_m


def grip_rate(xp, task: TaskBase, limits: Limits, close_s):
    """Return maximum gripper steps (N,) [rad per policy tick] for full closing times `close_s` [s].

    Nonpositive closing times give an infinite speed limit.
    """
    g = limits.grip
    span = float(limits.hi[g] - limits.lo[g])
    return xp.where(close_s > 0, span / (xp.maximum(close_s, 1e-9) * task.control_hz), xp.inf)


def servo_follow(xp, ctrl, target, rate):
    """Move `ctrl` toward `target` by at most `rate` [rad]. An infinite rate gives `target` exactly."""
    return _clip(xp, target, ctrl - rate, ctrl + rate)


def limit_cost(xp, task: TaskBase, limits: Limits, q_cmd):
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo[:na], q_cmd), _as(xp, limits.hi[:na], q_cmd)
    room = xp.minimum(q_cmd[..., :na] - lo, hi - q_cmd[..., :na])
    near = _clip(xp, 1.0 - room / task.limit_margin, 0.0, 1.0)
    return task.limit_penalty * (near * near).sum(-1)


def grasped(xp, jaw_contacts):
    return (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)


def grip_is_open(xp, limits: Limits, q_cmd):
    g = limits.grip
    lo, hi = float(limits.lo[g]), float(limits.hi[g])
    return q_cmd[..., g] > lo + 0.5 * (hi - lo)


def success_streak(xp, task: TaskBase, streak, success):
    """Success streaks truncate episodes so PPO bootstraps the state value."""
    streak = xp.where(success, streak + 1, streak * 0)
    done = streak >= task.end_on_success if task.end_on_success > 0 else success & ~success
    return streak, done


def sample_start(task: TaskBase, limits: Limits, rng: np.random.Generator, n: int) -> np.ndarray:
    na, g = limits.n_arm, limits.grip
    q = np.repeat(limits.home[None], n, axis=0)
    q[:, :na] += np.radians(rng.uniform(-task.start_noise_deg, task.start_noise_deg, (n, na)))
    if task.cube:
        span = limits.hi[g] - limits.lo[g]
        q[:, g] = limits.lo[g] + rng.uniform(0.6, 1.0, n) * span
    return np.clip(q, limits.lo, limits.hi)


def action_cost(xp, task: TaskBase, action, prev_action, q_cmd, limits):
    cost = task.action_rate * ((action - prev_action) ** 2).sum(-1)
    if task.limit_penalty > 0 and q_cmd is not None and limits is not None:
        cost = cost + limit_cost(xp, task, limits, q_cmd)
    if task.roll_penalty > 0 and q_cmd is not None and limits is not None and limits.roll is not None:
        k = limits.roll
        roll = (q_cmd[..., k] - float(limits.home[k])) / np.pi
        cost = cost + task.roll_penalty * roll * roll
    return cost


def sample_start_jax(jax, tsk: TaskBase, limits: Limits, home, lo, hi, N, k_q, k_g):
    jnp = jax.numpy
    u = jax.random.uniform
    noise = np.radians(tsk.start_noise_deg)
    na, g = limits.n_arm, limits.grip
    q0 = jnp.broadcast_to(home, (N, len(home)))
    q0 = q0.at[:, :na].add(u(k_q, (N, na), minval=-noise, maxval=noise))
    if tsk.cube:
        q0 = q0.at[:, g].set(lo[g] + u(k_g, (N,), minval=0.6, maxval=1.0) * (hi[g] - lo[g]))
    q0 = jnp.clip(q0, lo, hi)
    return q0
