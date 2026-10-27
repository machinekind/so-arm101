"""RL tasks: what the policy sees, what it does and what it is rewarded for.

One definition for the three places a policy lives in:

* the Gymnasium environment on the CPU (`env.ArmEnv`, numpy, one environment),
* the batch environment on the GPU (`batch.BatchEnv`, JAX, thousands of MuJoCo Warp worlds),
* the loop on the real arm (`strategies.policy.PolicyStrategy` in the twin's `StrategyRunner`, numpy,
  angles from the servos).

The functions below take arrays with a batch dimension `(N, ...)` and a module `xp`, numpy or
jax.numpy (torch works too), so all three compute EXACTLY the same formulas. Two copies of "almost
the same" observation are a classic silent source of sim-to-real gaps: the policy trains on one and
runs on the other.

All positions are in the arm's BASE frame (the table top at z = 0), joint angles in MJCF radians, in
the order of `RobotSpec.joints` (gripper last).

The action is `n_joints` numbers in [-1, 1] (6 for the SO-101: five arm joints and the gripper):

* arm joints: an INCREMENT of the target by at most `arm_step` rad per tick, from the previous
  target (not from the measured angles), clipped to the same limits the twin's safety supervisor
  enforces on the real arm;
* gripper: an ABSOLUTE target (-1 closed, +1 open), rate-limited the way the supervisor limits it
  (`grip_step` of the range per tick).

Tasks: `reach` (the TCP to a point), `lift` (grasp and lift the cube) and `pickplace` (carry the cube
onto a target on the table and release it). In pickplace only a CARRIED cube counts (the
`carry_latch`: grasped and lifted above `lift_height`), then lying on the target, released and
still; pushing it along the table does not count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..kinematics import RobotKinematics
from ..robot import RobotSpec

TASKS = ("reach", "lift", "pickplace")


@dataclass(frozen=True)
class TaskConfig:
    name: str = "reach"
    #: Policy frequency and physics step [s]. 20 Hz at 5 ms is 10 physics steps per action.
    control_hz: float = 20.0
    timestep: float = 0.005
    episode_steps: int = 100
    #: The largest increment of an arm joint's target per tick [rad] (0.05 at 20 Hz = 57 deg/s).
    arm_step: float = 0.05
    #: The largest move of the gripper's target per tick, as a fraction of its range (0.15 at 20 Hz =
    #: 300 units/s, as much as the safety supervisor lets through).
    grip_step: float = 0.15
    #: reach: success when the TCP is closer to the goal than this [m].
    success_dist: float = 0.02
    #: reach: where the goal is sampled: radius from the base axis and height above the table [m].
    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    #: lift and pickplace: the cube (half side [m], mass [kg]) and where it lies on the table.
    cube_half: float = 0.015
    cube_mass: float = 0.03
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    #: lift: success when the cube's bottom is higher above the table than this [m].
    #: pickplace: a cube grasped higher than this is "carried" (`carry_latch`).
    lift_height: float = 0.06
    #: The arm's start: the home pose plus/minus this many degrees on every arm joint.
    start_noise_deg: float = 8.0
    #: Penalty for jerky actions: the weight of ||a - a_previous||^2.
    action_rate: float = 0.02
    #: How many success ticks IN A ROW end the episode (0 = never). The strategy on the arm stops the
    #: same way ("task done"). Without it lift-v2 kept moving after the grasp: base -105 deg, elbow
    #: -92, wrist +95 and -150 (all at their limits), the cube 37 cm above the table instead of 6, and
    #: it held that until the end of the episode (8.5 s).
    end_on_success: int = 0
    #: Penalty for an arm joint target near its limit: weight * sum((1 - room/limit_margin)^2) over
    #: the joints closer to a limit than `limit_margin` [rad]. Without it the reward says nothing
    #: about posture, so nothing keeps the policy away from the limits.
    limit_penalty: float = 0.0
    limit_margin: float = 0.15
    #: Penalty for moving the arm once the task is done: weight * ||a_arm||^2. The `end_on_success`
    #: streak is meant to hold the cube, not to swing it around.
    hold_still: float = 0.0
    #: Penalty for rolling the wrist away from the home pose: weight * ((q_roll - home) / pi)^2. In
    #: reach the reward counts only the TCP position, which the wrist roll does not change: reach-v2
    #: turned it to the +150 deg limit in every episode (643 of 2000 ticks at the limit).
    roll_penalty: float = 0.0
    #: The number of arm joints including the gripper (gripper last): the action dimension. A
    #: `meta.json` without it is an SO-101 policy, so 6.
    n_joints: int = 6
    #: pickplace: the cube is "on the target" when its centre is closer to the target's centre (in xy)
    #: than `place_radius` [m] and its bottom is lower above the table than `place_height` [m].
    place_radius: float = 0.03
    place_height: float = 0.01
    #: pickplace: the target is at least this far [m] (in xy) from the cube at the start.
    target_min_sep: float = 0.10
    #: pickplace: the cube "lies still" when its speed is below this [m/s].
    settle_speed: float = 0.02
    #: pickplace: where the target on the table is sampled, like the cube (`make_task(spec=)` takes
    #: the cube's ring from `spec.desk`).
    target_radius: tuple[float, float] = (0.15, 0.25)
    target_bearing: tuple[float, float] = (-0.8, 0.8)

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.name, self.n_joints)

    @property
    def act_dim(self) -> int:
        return self.n_joints

    @property
    def substeps(self) -> int:
        return int(round(1.0 / self.control_hz / self.timestep))

    @property
    def cube(self) -> str | None:
        return "cube" if self.name in ("lift", "pickplace") else None

    @property
    def target(self) -> str | None:
        """The name of the place target's body (the mocap pad in the scene), or None."""
        return "target" if self.name == "pickplace" else None


def obs_dim(name: str, n: int) -> int:
    """The observation size of task `name` with `n` joints.

    q(n) + joint targets(n) + TCP(3) + [reach: goal(3) + goal-TCP(3)]
    + [lift, pickplace: cube(3) + cube-TCP(3) + rotation(6)]
    + [pickplace: place target(3) + target-cube(3)] + action(n).
    """
    return 2 * n + 3 + (6 if name == "reach" else 12) + (6 if name == "pickplace" else 0) + n


#: Observation sizes for the SO-101 (6 joints): the layout of the base policies.
OBS_DIMS = {name: obs_dim(name, 6) for name in TASKS}


def make_task(name: str = "reach", spec: RobotSpec | None = None, **overrides) -> TaskConfig:
    """The task `name`; with `spec`, scaled to that arm (number of joints, rings of goals and cubes)."""
    if name not in TASKS:
        raise ValueError(f"nieznane zadanie {name!r}; dostepne: {', '.join(TASKS)}")
    base = TaskConfig(name=name, episode_steps=100 if name == "reach" else 200)
    if name == "reach":
        base = replace(base, roll_penalty=0.5)
    if name == "lift":
        base = replace(base, end_on_success=10, limit_penalty=0.5, hold_still=0.5)
    if name == "pickplace":
        base = replace(base, episode_steps=300, end_on_success=10, limit_penalty=0.5, hold_still=0.5)
    if spec is not None:
        d = spec.desk
        base = replace(base, n_joints=len(spec.joints), goal_radius=d.goal_radius, goal_height=d.goal_height,
                       cube_radius=d.cube_radius, cube_bearing=d.cube_bearing,
                       target_radius=d.cube_radius, target_bearing=d.cube_bearing)
    return replace(base, **overrides)


# --------------------------------------------------------------- limits
@dataclass(frozen=True)
class Limits:
    """The range of joint targets [rad]: the supervisor's limits intersected with the MJCF ranges.

    The supervisor's limits are `limits` from `robot.yaml`, the ones the twin uses by default
    (`robot_config` without a config file).
    """

    lo: np.ndarray
    hi: np.ndarray
    home: np.ndarray
    #: The number of arm joints (without the gripper): these move by `arm_step` increments.
    n_arm: int
    #: The gripper's index (the last joint).
    grip: int
    #: The index of the tool roll joint (`spec.roll_joint`) for `roll_penalty`; None = no such joint.
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


# -------------------------------------------- numpy / jax.numpy / torch
# numpy and jax.numpy share their names (`concatenate`, `clip`, `asarray`, `.astype`); torch has its
# own (`cat`, `clamp`, `as_tensor`, `.to`), so the helpers ask the module what it has.
def _cat(xp, parts):
    return xp.cat(parts, dim=-1) if hasattr(xp, "cat") else xp.concatenate(parts, axis=-1)


def _clip(xp, x, lo, hi):
    return xp.clamp(x, lo, hi) if hasattr(xp, "clamp") else xp.clip(x, lo, hi)


def _norm(x):
    return ((x * x).sum(-1)) ** 0.5


def _as(xp, a, like):
    """A numpy constant as an array of the same kind, dtype and device as `like`.

    jax.numpy: the device follows `jit` (a constant inside a traced function lands where it runs).
    """
    if hasattr(xp, "as_tensor"):
        return xp.as_tensor(a, dtype=like.dtype, device=like.device)
    return xp.asarray(a, dtype=like.dtype)


def _cast(xp, b, like):
    """`b` (e.g. a boolean mask) as an array of `like`'s dtype: `.astype` in numpy and JAX, `.to` in torch."""
    return b.astype(like.dtype) if hasattr(b, "astype") else b.to(like.dtype)


# ---------------------------------------------------------------- action
def apply_action(xp, task: TaskConfig, limits: Limits, q_cmd, action):
    """The new joint target (N, n_joints) from the previous target and an action in [-1, 1]."""
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo, q_cmd), _as(xp, limits.hi, q_cmd)
    a = _clip(xp, action, -1.0, 1.0)
    arm = _clip(xp, q_cmd[..., :na] + a[..., :na] * task.arm_step, lo[:na], hi[:na])
    span = hi[na:] - lo[na:]
    want = lo[na:] + (a[..., na:] + 1.0) * 0.5 * span
    step = task.grip_step * span
    grip = q_cmd[..., na:] + _clip(xp, want - q_cmd[..., na:], -step, step)
    return _cat(xp, [arm, grip])


# ----------------------------------------------------------- observation
def normalize_q(xp, limits: Limits, q):
    lo, hi = _as(xp, limits.lo, q), _as(xp, limits.hi, q)
    return 2.0 * (q - lo) / (hi - lo) - 1.0


def observe(xp, task: TaskConfig, limits: Limits, q, q_cmd, tcp, prev_action, goal=None,
            cube_pos=None, cube_rot=None, target=None):
    """The observation vector (N, obs_dim). `cube_rot` are rotation matrices (N, 3, 3).

    `target` (N, 3), pickplace: the place target on the table top (z = 0) in the base frame.
    """
    parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
    if task.name == "reach":
        parts += [goal, goal - tcp]
    else:
        rot6 = _cat(xp, [cube_rot[..., :, 0], cube_rot[..., :, 1]])
        parts += [cube_pos, cube_pos - tcp, rot6]
        if task.name == "pickplace":
            parts += [target, target - cube_pos]
    parts.append(prev_action)
    return _cat(xp, parts)


def fold_yaw(xp, R):
    """The rotation about z folded to +-45 deg: how the detector sees the cube (a 90 deg symmetry)."""
    atan2 = xp.arctan2 if hasattr(xp, "arctan2") else xp.atan2
    yaw = atan2(R[..., 1, 0], R[..., 0, 0])
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    rows = [xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)]
    return xp.stack(rows, -2)


# ---------------------------------------------------------------- reward
def limit_cost(xp, task: TaskConfig, limits: Limits, q_cmd):
    """(N,) penalty for arm joint targets closer to a limit than `limit_margin` (0 mid-range)."""
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo[:na], q_cmd), _as(xp, limits.hi[:na], q_cmd)
    room = xp.minimum(q_cmd[..., :na] - lo, hi - q_cmd[..., :na])
    near = _clip(xp, 1.0 - room / task.limit_margin, 0.0, 1.0)
    return task.limit_penalty * (near * near).sum(-1)


def grasped(xp, jaw_contacts):
    """(N,) whether the cube touches both jaws at once (`jaw_contacts` (N, 2))."""
    return (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)


def carry_latch(xp, task: TaskConfig, carried, jaw_contacts, cube_pos):
    """(N,) the "cube was carried" latch: grasped by both jaws higher above the table than `lift_height`.

    Once set it holds until a reset (the caller clears it at the start of an episode). It is computed
    BEFORE the reward and passed to it: without it a cube pushed along the table onto the target would
    count as "placed", and with it `reward` stays a function without state.
    """
    height = cube_pos[..., 2] - task.cube_half
    return carried | (grasped(xp, jaw_contacts) & (height > task.lift_height))


def grip_is_open(xp, limits: Limits, q_cmd):
    """(N,) whether the gripper command is in the open half of its range (the cube is released)."""
    g = limits.grip
    lo, hi = float(limits.lo[g]), float(limits.hi[g])
    return q_cmd[..., g] > lo + 0.5 * (hi - lo)


def reward(xp, task: TaskConfig, tcp, action, prev_action, goal=None, cube_pos=None, jaw_contacts=None,
           q_cmd=None, limits: Limits | None = None, target=None, cube_vel=None, carried=None,
           grip_open=None):
    """(reward, success, failure) for every world.

    `jaw_contacts` (N, 2): whether the cube touches the fixed and the moving jaw.
    Failure = the cube fell off the table; the episode then ends.
    `q_cmd` (N, n_joints) and `limits` are for the limit and roll penalties (`limit_penalty`,
    `roll_penalty`); without them there are none.

    pickplace: `target` (N, 3) the target on the table, `cube_vel` (N, 3) the cube's velocity [m/s],
    `carried` (N,) the latch from `carry_latch` (computed before the reward), `grip_open` (N,) the
    gripper command is open (by default from `q_cmd` and `limits`, see `grip_is_open`).
    """
    cost = task.action_rate * ((action - prev_action) ** 2).sum(-1)
    if task.limit_penalty > 0 and q_cmd is not None and limits is not None:
        cost = cost + limit_cost(xp, task, limits, q_cmd)
    if task.roll_penalty > 0 and q_cmd is not None and limits is not None and limits.roll is not None:
        k = limits.roll
        roll = (q_cmd[..., k] - float(limits.home[k])) / np.pi
        cost = cost + task.roll_penalty * roll * roll
    if task.name == "reach":
        d = _norm(goal - tcp)
        # Two scales: the coarse one pulls from afar, the fine one rewards the last millimetres.
        r = 0.5 * (1.0 - xp.tanh(d / 0.10)) + 0.5 * (1.0 - xp.tanh(d / 0.01))
        success = d < task.success_dist
        failure = d < -1.0                                  # reach has no failure
        return r - cost, success, failure
    if task.name == "pickplace":
        return _reward_pickplace(xp, task, tcp, action, cost, cube_pos, jaw_contacts, limits, target,
                                 cube_vel, carried,
                                 grip_open if grip_open is not None else grip_is_open(xp, limits, q_cmd))
    d = _norm(cube_pos - tcp)
    grasped = (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)
    height = cube_pos[..., 2] - task.cube_half
    lifted = _clip(xp, height / task.lift_height, 0.0, 1.0)
    success = grasped & (height > task.lift_height)
    g, s = _cast(xp, grasped, tcp), _cast(xp, success, tcp)
    r = (1.0 - xp.tanh(d / 0.05)) + g + 4.0 * g * lifted + 2.0 * s
    if task.hold_still > 0:
        na = action.shape[-1] - 1 if limits is None else limits.n_arm
        cost = cost + task.hold_still * s * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
    failure = height < -0.05
    return r - cost, success, failure


def _reward_pickplace(xp, task: TaskConfig, tcp, action, cost, cube_pos, jaw_contacts, limits, target,
                      cube_vel, carried, opened):
    """pickplace: approach, grasp, lift, carry over the target, put down, release.

    Success = it was carried (`carried`) & it lies on the target (`placed`) & the jaws do not hold it
    and the gripper command is open (`released`) & it is still (`still`). The rewards for nearing
    and being on the target also count only after carrying, otherwise pushing along the table would pay.
    """
    d = _norm(cube_pos - tcp)
    g = grasped(xp, jaw_contacts)
    height = cube_pos[..., 2] - task.cube_half
    lifted = _clip(xp, height / task.lift_height, 0.0, 1.0)
    d_t = _norm(cube_pos[..., :2] - target[..., :2])
    placed = (d_t < task.place_radius) & (height < task.place_height)
    released = ~g & opened
    still = _norm(cube_vel) < task.settle_speed
    success = carried & placed & released & still
    gf, cf = _cast(xp, g, tcp), _cast(xp, carried, tcp)
    pf, sf = _cast(xp, placed, tcp), _cast(xp, success, tcp)
    r = ((1.0 - xp.tanh(d / 0.05)) + gf + 2.0 * gf * lifted + 4.0 * cf * (1.0 - xp.tanh(d_t / 0.10))
         + 4.0 * cf * pf + 4.0 * sf)
    if task.hold_still > 0:
        na = action.shape[-1] - 1 if limits is None else limits.n_arm
        cost = cost + task.hold_still * sf * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
    failure = height < -0.05
    return r - cost, success, failure


def success_streak(xp, task: TaskConfig, streak, success):
    """(the new count of successes in a row, whether it ends the episode), for env and batch.

    The strategy on the arm counts its streak by the same rule. An end after a streak is a truncation
    (like the time limit), not a terminal state: PPO bootstraps it with the state value. With zero
    instead, the policy would learn NOT to finish: a cube held just below `lift_height` earns ~6 per
    tick for the rest of the episode, finishing only 8 x 10 ticks.
    """
    streak = xp.where(success, streak + 1, streak * 0)
    done = streak >= task.end_on_success if task.end_on_success > 0 else success & ~success
    return streak, done


# ---------------------------------------------------------------- starts
def sample_goals(kin: RobotKinematics, task: TaskConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """`reach` goals (n, 3) in the base frame, all of them reachable.

    Samples in JOINT space (`spec.wave_ranges`, else the MJCF range) and takes the TCP from forward
    kinematics, like the calibration wave: every goal is reachable by construction, and the filter
    keeps it within `goal_radius` / `goal_height` and in front of the arm (x > 0.05 m).
    """
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


def sample_start(task: TaskConfig, limits: Limits, rng: np.random.Generator, n: int) -> np.ndarray:
    """Start angles (n, n_joints): the home pose with noise, the gripper open in the tasks with a cube."""
    na, g = limits.n_arm, limits.grip
    q = np.repeat(limits.home[None], n, axis=0)
    q[:, :na] += np.radians(rng.uniform(-task.start_noise_deg, task.start_noise_deg, (n, na)))
    if task.cube:
        span = limits.hi[g] - limits.lo[g]
        q[:, g] = limits.lo[g] + rng.uniform(0.6, 1.0, n) * span
    return np.clip(q, limits.lo, limits.hi)


def sample_cubes(task: TaskConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cubes on the table: positions (n, 3) in the base frame and quaternions (n, 4) of a turn about z."""
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.cube_half)], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    return pos, quat


#: How many times `sample_targets` redraws a target too close to the cube before taking the fallback.
TARGET_TRIES = 64


def sample_targets(task: TaskConfig, rng: np.random.Generator, cube_pos: np.ndarray) -> np.ndarray:
    """pickplace targets (n, 3) on the table (z = 0), at least `target_min_sep` (xy) from the cubes (n, 3).

    Polar sampling in `target_radius` / `target_bearing`, like the cubes; rows too close to their
    cube are drawn again (only those), at most `TARGET_TRIES` times. A row that still does not fit
    gets a target on the outer radius at the end of the bearing range away from its cube.
    On the GPU: `batch.BatchEnv._sample_targets`, the same rule in JAX.
    """
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
