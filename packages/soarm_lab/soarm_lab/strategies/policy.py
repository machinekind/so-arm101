"""Run a lab policy through the twin's `policy` strategy entry point.

The twin builds `PolicySettings` from its configuration or panel form. Its `StrategyRunner`
claims the arm and sends returned joint targets through the safety supervisor.
The strategy uses the plugin API and runs policy inference with NumPy.

    policy:
      policy: workspace/policies/run/policy.npz

Observations use `task.observe` and actions use `tasks.apply_action`, as in training.
The strategy reads joints, commands, goals, object poses, and time from `Observation`.
Robot details, limits, and simulation status come from `RunInfo`.

The run holds the measured pose if tracking error exceeds `max_tracking_deg`.
`episode_limit` caps the run at the training episode length.
`end_on_success` consecutive success ticks end the run.
Reach goals are projected into the training area.

Lift success requires a fresh object pose and a holding jaw. `max_pose_age_s` limits pose age.
`allow_last_seen` permits the last known pose. A dropped object never counts.
Pickplace success requires a carried object seen resting on the target, released and still.
Its target is projected onto the table and the training target ring.

`obs.goal` supplies the goal, falling back to `goal_x`, `goal_y`, and `goal_z`.
`object_name` selects an entry in `obs.objects` and, for a policy with variants, its task variant.
An empty setting uses the policy's saved object. A missing entry falls back to `obs.objects["cube"]`.
Object poses use the base frame. Their source and `age_s` determine freshness.
An empty source counts as simulation truth on a simulated arm and as stale on hardware.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo, Strategy

from ..kinematics import RobotKinematics
from ..rl import tasks as tk
from ..rl.policy import Policy, find_policy
from ..robot import load_robot

logger = logging.getLogger(__name__)

#: Lift accepts camera detections, tracked carried poses, and simulation truth.
FRESH_CUBE_SOURCES = ("kamery", "w dloni", "symulacja")
#: Last-seen poses count for lift success only when `allow_last_seen` is enabled.
LAST_SEEN = "ostatnie widziane"
#: Placement accepts camera detections and simulation truth because carried poses follow the jaws.
RESTING_CUBE_SOURCES = ("kamery", "symulacja")
#: Maximum object displacement during a placement success streak [m].
PLACE_STILL = 0.005


class NoCube(RuntimeError):
    """The required object pose is missing from the observation."""


@dataclass
class PolicySettings:
    #: A policy name from `soarm-lab policies` or a path to policy.npz.
    policy: str = "reach-v3"
    #: Maximum tracking error before stopping at the measured pose [deg].
    max_tracking_deg: float = 25.0
    #: Limit the run to the training episode length.
    episode_limit: bool = True
    #: Maximum wait for the first object pose [s].
    wait_for_cube_s: float = 3.0
    #: Maximum object pose age for lift or placement success [s].
    max_pose_age_s: float = 1.0
    #: Permit last-seen poses within `max_pose_age_s` for lift success.
    allow_last_seen: bool = False
    #: Fallback goal coordinates when the observation has no goal [m, base frame].
    goal_x: float = 0.22
    goal_y: float = 0.0
    goal_z: float = 0.12
    #: Select an `obs.objects` entry and task variant. Empty uses the policy's saved object.
    object_name: str = ""


@dataclass
class PolicyStatus:
    """Record strategy progress and stop reasons for logs and tests."""

    #: Executed policy ticks exclude time spent waiting for the first pose.
    step: int = 0
    distance: float = float("nan")
    success: bool = False
    stopped_because: str = ""
    last_action: list[float] = field(default_factory=list)
    #: Consecutive ticks satisfying the success conditions.
    success_streak: int = 0
    #: The goal or place target was projected into its training area.
    goal_clamped: bool = False
    #: Records whether the object has been carried during this run.
    carried: bool = False


def project_goal(task: tk.ReachConfig, goal: np.ndarray) -> np.ndarray:
    """Project a base-frame goal [m] into the training area to keep the arm above the table."""
    g = np.asarray(goal, float).copy()
    g[2] = np.clip(g[2], *task.goal_height)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.goal_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    # Keep a 1 mm margin from the training boundary at base-frame x = 0.05 m.
    b_max = float(np.arccos(min(1.0, 0.051 / r_new)))
    bearing = float(np.clip(bearing, -b_max, b_max))
    g[0], g[1] = r_new * np.cos(bearing), r_new * np.sin(bearing)
    return g


def project_target(task: tk.PickPlaceConfig, goal: np.ndarray) -> np.ndarray:
    """Project a base-frame goal [m] onto the table at z = 0 within the training target ring."""
    g = np.asarray(goal, float)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.target_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    bearing = float(np.clip(bearing, *task.target_bearing))
    return np.array([r_new * np.cos(bearing), r_new * np.sin(bearing), 0.0])


class PolicyStrategy(Strategy[PolicySettings]):
    """An RL policy of soarm-lab (reach, lift, pickplace) driving the arm, as in training."""

    settings = PolicySettings

    def __init__(self, settings: PolicySettings | None = None, *, policy: Policy | None = None):
        """Use a supplied `policy` or load the policy named in `settings`."""
        self.config = settings if settings is not None else PolicySettings()
        if self.config.max_tracking_deg <= 0:
            raise ValueError(f"policy: max_tracking_deg must be positive, is {self.config.max_tracking_deg}")
        if not self.config.max_pose_age_s >= 0:
            raise ValueError(f"policy: max_pose_age_s must be zero or more, is {self.config.max_pose_age_s}")
        self.policy = policy if policy is not None else Policy.load(find_policy(self.config.policy))
        self.task = self._task_for(self.config.object_name)
        self._object_name = self.config.object_name
        if self.task.cube and not self._object_name:
            self._object_name = cast(tk.LiftConfig, self.task).block
        #: The runner uses the training control frequency [Hz].
        self.hz = float(self.task.control_hz)
        #: Episode and success limits are enforced by the strategy.
        self.time_limit_s = None
        self.status = PolicyStatus()
        end = getattr(self.task, "end_on_success", None)
        #: Consecutive success ticks required to end the run; zero disables this limit.
        self.end_on_success = int(end) if end is not None else (10 if self.task.cube else 0)
        self.max_tracking = np.radians(self.config.max_tracking_deg)
        self.episode_limit = self.config.episode_limit
        self.wait_for_cube = self.config.wait_for_cube_s
        self._default_goal = np.array([self.config.goal_x, self.config.goal_y, self.config.goal_z])
        self._goal = self._default_goal.copy()
        # Kinematics and limits are initialized by `start`.
        self._kin: RobotKinematics | None = None
        self._limits: tk.Limits | None = None
        self.g, self.na = 0, 0
        self.q_cmd = np.zeros(self.task.act_dim)
        self.prev_action = np.zeros(self.task.act_dim)
        self._hw_lo = self._hw_hi = np.zeros(self.task.act_dim)
        self.simulated = False
        self._cube: np.ndarray | None = None
        self._cube_rot = np.eye(3)
        self._cube_src: str | None = None
        self._cube_age = 0.0
        self._grip_prev: float | None = None
        self._streak_grip = 0.0
        #: Carrying stays latched until the next run starts.
        self._carried = False
        #: Object position at the start of a placement success streak [m, base frame].
        self._streak_cube: np.ndarray | None = None

    def _task_for(self, name: str) -> tk.TaskConfig:
        """Select a task variant by object name when the policy has variants."""
        task = self.policy.task
        if not name:
            return task
        if not isinstance(task, tk.LiftConfig):
            raise ValueError(f"policy: object_name {name!r} needs a task with an object, not {task.name}")
        if not task.variants:
            return task
        if name not in task.variants:
            raise ValueError(f"policy: object_name {name!r} is not one of the policy's variants: "
                             f"{', '.join(task.variants)}")
        return task.with_variant(name)

    @property
    def started(self) -> bool:
        """Return whether the run has initialized kinematics and limits."""
        return self._kin is not None

    @property
    def kin(self) -> RobotKinematics:
        """Return the arm kinematics after `start`."""
        if self._kin is None:
            raise RuntimeError("policy: start the run first")
        return self._kin

    @property
    def limits(self) -> tk.Limits:
        """Return the training joint limits after `start`."""
        if self._limits is None:
            raise RuntimeError("policy: start the run first")
        return self._limits

    # -------------------------------------------------------------------- goal

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value) -> None:
        g = np.asarray(value, float)
        if not self.task.cube:
            p = project_goal(cast(tk.ReachConfig, self.task), g)
            self.status.goal_clamped = bool(np.linalg.norm(p - g) > 1e-4)
            g = p
        elif self.task.target:
            # Report placement clamping only when horizontal coordinates change.
            p = project_target(cast(tk.PickPlaceConfig, self.task), g)
            self.status.goal_clamped = bool(np.linalg.norm(p[:2] - g[:2]) > 1e-4)
            g = p
        self._goal = g

    @property
    def target(self) -> np.ndarray:
        """Return the projected place target in the base frame [m], with z = 0."""
        return project_target(cast(tk.PickPlaceConfig, self.task), self._goal)

    # ------------------------------------------------------------------- start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Validate the policy for the arm and initialize commands from its measured pose."""
        robot = getattr(self.policy.meta, "robot", "so101")
        if robot != run.robot:
            raise RuntimeError(f"polityka uczona na {robot}, stanowisko ma {run.robot}")
        spec = load_robot(run.folder)
        if self.policy.meta.act_dim != len(spec.joints):
            raise RuntimeError(f"polityka ma {self.policy.meta.act_dim} wyjsc, a ramie {spec.name} "
                               f"{len(spec.joints)} stawow")
        self._kin = RobotKinematics(spec)
        self._limits = tk.Limits.of(self._kin)
        self.g, self.na = self.limits.grip, self.limits.n_arm
        #: Holding thresholds use the gripper's MJCF joint units.
        jaw = spec.jaw
        self.grip_still, self.grip_block_margin, self.grip_open_margin = jaw.still, jaw.margin, jaw.margin
        #: Maximum jaw closure during a success streak in MJCF joint units.
        self.grip_creep = jaw.creep
        self.simulated = bool(run.simulated)

        measured = dict(obs.joints)
        self._hw_lo, self._hw_hi = self._hardware_limits(run.limits)
        self.q_cmd = np.clip(self.kin.to_q(measured), self._hw_lo, self._hw_hi)
        grip = self._squeezing_grip_cmd(measured, obs.command)
        if grip is not None:
            self.q_cmd[self.g] = grip
        self.prev_action = np.zeros(self.task.act_dim)
        self.status = PolicyStatus(goal_clamped=self.status.goal_clamped)
        self._cube, self._cube_src, self._cube_age = None, None, 0.0
        self._cube_rot = np.eye(3)
        self._grip_prev = None
        self._carried, self._streak_cube = False, None
        self._take_goal(obs)

    def _squeezing_grip_cmd(self, measured: Mapping[str, float],
                            command: Mapping[str, float]) -> float | None:
        """Return the startup squeeze command [rad], or None, to preserve an existing grip."""
        name = self.kin.spec.gripper
        if name is None or name not in measured:
            return None
        cmd = command.get(name)
        if cmd is None:
            return None
        g = self.g
        q_cmd = float(self.kin.to_q({**measured, name: float(cmd)})[g])
        if float(self.kin.to_q(measured)[g]) - q_cmd <= self.grip_block_margin:
            return None
        return float(np.clip(q_cmd, self._hw_lo[g], self._hw_hi[g]))

    def _hardware_limits(self, hw: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Intersect hardware and training limits [rad] for target clipping.

        Observation normalization keeps the training limits.
        An empty intersection falls back to the training limits.
        """
        lo, hi = self.limits.lo.copy(), self.limits.hi.copy()
        for k, name in enumerate(self.kin.spec.joints):
            if name in hw:
                a, b = self.kin.to_q({name: hw[name][0]})[k], self.kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = self.limits.lo[bad], self.limits.hi[bad]
        return lo, hi

    def _take_goal(self, obs: Observation) -> None:
        self.goal = self._default_goal if obs.goal is None else obs.goal

    # -------------------------------------------------------------- observation

    def observation(self, measured: Mapping[str, float],
                    objects: Mapping[str, ObjectPose] | None = None) -> np.ndarray:
        """Build the policy observation (obs_dim,) using the training task layout."""
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        target = None
        if self.task.cube:
            seen = objects or {}
            got = seen.get(self._object_name, seen.get(self.task.cube))
            if got is None:
                raise NoCube(f"zadanie {self.task.name} potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = np.asarray(got.position, float), np.asarray(got.rotation, float)
            self._cube_src = str(got.source or "") or None
            self._cube_age = float(got.age_s)
            self._cube, self._cube_rot = cube_pos.copy(), cube_rot.copy()
            if self.policy.meta.randomization.get("fold_yaw", False):
                # Apply the training yaw convention to every pose source.
                cube_rot = tk.fold_yaw(np, cube_rot, cast(tk.LiftConfig, self.task).block_spec.yaw_period)
        if self.task.target:
            target = self.target[None]
        obs = self.task.observe(np, self.limits, q[None], self.q_cmd[None], tcp[None],
                         self.prev_action[None], goal=self.goal[None], cube_pos=cube_pos[None],
                         cube_rot=cube_rot[None], target=target)
        return obs[0].astype(np.float32)

    # --------------------------------------------------------------------- tick

    def act(self, obs: Observation) -> Action:
        """Return joint targets for one policy tick or stop the run."""
        if self._kin is None:
            raise RuntimeError("policy: act() przed start()")
        st = self.status
        measured = dict(obs.joints)
        self._take_goal(obs)
        q = self.kin.to_q(measured)
        lag = np.abs(q[:self.na] - self.q_cmd[:self.na]).max()
        if lag > self.max_tracking:
            # Hold the measured pose to stop driving into a possible obstacle.
            return self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                              target=self._hold_pose(measured, q))
        holding = self._jaw_holding(q[self.g])
        try:
            vec = self.observation(measured, obs.objects)
        except NoCube as exc:
            # Wait only for the first detection so losing the object during motion stops the run.
            if st.step == 0 and float(obs.t) < self.wait_for_cube:
                st.stopped_because = "czekam na kostke z kamer"
                return Action(note=st.stopped_because)
            return self._halt(str(exc))
        action = self.policy.act(vec)
        q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], action[None])[0]
        self.q_cmd = np.clip(q_cmd, self._hw_lo, self._hw_hi)
        target = self.kin.from_q(self.q_cmd)
        self.prev_action = action
        st.step += 1
        st.stopped_because = ""
        st.last_action = action.tolist()
        if not self.task.cube:
            st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
            st.success = st.distance < cast(tk.ReachConfig, self.task).success_dist
        elif self.task.target:
            st.success = self._placed(holding)
            st.carried = self._carried
        else:
            # Limit cumulative jaw closure so slow slipping cannot count as a stable grasp.
            if holding and st.success_streak > 0 and q[self.g] < self._streak_grip - self.grip_creep:
                holding = False
            st.success = bool(self._cube is not None and self._cube_fresh() and holding
                              and self._height(self._cube) > cast(tk.LiftConfig, self.task).lift_height)
            if st.success and st.success_streak == 0:
                self._streak_grip = float(q[self.g])
        st.success_streak = st.success_streak + 1 if st.success else 0
        if self.end_on_success > 0 and st.success_streak >= self.end_on_success:
            return self._halt("zadanie wykonane", target=target, success=True)
        if self.episode_limit and st.step >= self.task.episode_steps:
            return self._halt("koniec epizodu", target=target, success=st.success)
        return Action(target=target, success=st.success, error=self._error(), note=self._note())

    def _placed(self, holding: bool) -> bool:
        """Check placement success and update carrying state and target distance.

        Success needs a previously carried object seen resting on the target after release.
        The object must stay within `PLACE_STILL` [m] of its pose when the success streak began.
        """
        cube, src, task = self._cube, self._cube_src, cast(tk.PickPlaceConfig, self.task)
        if cube is None:
            return False
        if src is None and self.simulated:
            src = "symulacja"                               # An unspecified simulation source is scene truth.
        height = self._height(cube)
        tgt = self.target
        d_t = float(np.hypot(*(cube[:2] - tgt[:2])))
        self.status.distance = d_t
        carried_before = self._carried
        if src in ("symulacja", "w dloni") and height > task.lift_height:
            self._carried = True
        grip_open = bool(tk.grip_is_open(np, self.limits, self.q_cmd[None])[0])
        ok = (src in RESTING_CUBE_SOURCES and carried_before and d_t < task.place_radius
              and height < task.place_height and not holding and grip_open)
        if ok and self.status.success_streak > 0 and self._streak_cube is not None \
                and np.linalg.norm(cube - self._streak_cube) >= PLACE_STILL:
            ok = False
        if ok and (self.status.success_streak == 0 or self._streak_cube is None):
            self._streak_cube = np.asarray(cube, float).copy()
        return ok

    def _height(self, cube: np.ndarray) -> float:
        """Return the object's lowest height above the table [m] using the last observed rotation."""
        return float(cast(tk.LiftConfig, self.task).height(np, cube[None], self._cube_rot[None])[0])

    def _jaw_holding(self, grip_q: float) -> bool:
        """Check for a stationary jaw wider than its previous command and its closed limit."""
        prev, self._grip_prev = self._grip_prev, float(grip_q)
        still = prev is not None and abs(grip_q - prev) < self.grip_still
        squeezing = (grip_q - self.q_cmd[self.g] > self.grip_block_margin
                     and grip_q > self.limits.lo[self.g] + self.grip_open_margin)
        return bool(still and squeezing)

    def _hold_pose(self, measured: Mapping[str, float], q: np.ndarray) -> dict[str, float]:
        """Hold measured joint angles while preserving a squeezing gripper command to maintain grip."""
        pose = {name: float(measured[name]) for name in self.kin.spec.joints if name in measured}
        grip = self.kin.spec.gripper
        if grip in pose and q[self.g] - self.q_cmd[self.g] > self.grip_block_margin:
            del pose[grip]
        return pose

    def _cube_fresh(self) -> bool:
        """Check pose age and source so stale or dropped detections cannot count as success."""
        if not self._cube_age <= self.config.max_pose_age_s:
            return False                                    # Reject stale and NaN ages.
        if self._cube_src is None:
            return self.simulated
        if self._cube_src == LAST_SEEN:
            return self.config.allow_last_seen
        return self._cube_src in FRESH_CUBE_SOURCES

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, target: Mapping[str, float] | None = None,
              success: bool = False) -> Action:
        """End the run with a stop reason and optional final joint targets."""
        self.status.stopped_because = reason
        logger.info("Polityka zatrzymana: %s", reason)
        return Action(target=dict(target or {}), done=True, success=success, error=self._error(),
                      note=reason)

    def stop(self, reason: str) -> None:
        """Record the stop reason supplied by the twin."""
        # Keep the specific stop reason when the twin acknowledges completion.
        if reason in ("zadanie wykonane", "strategia skonczyla") and self.status.stopped_because:
            return
        self.status.stopped_because = reason or "zatrzymana"

    def _error(self) -> float | None:
        d = self.status.distance
        return float(d) if np.isfinite(d) else None

    def _note(self) -> str:
        st = self.status
        if not self.task.cube:
            txt = f"do celu {st.distance * 1000:.0f} mm" + (" (w celu)" if st.success else "")
            return txt + (" - cel przyciety do obszaru z treningu" if st.goal_clamped else "")
        if self.task.target:
            txt = f"kostka do celu {st.distance * 1000:.0f} mm" + (" (niesiona)" if st.carried else "")
            return txt + (" - cel przesuniety na obszar z treningu" if st.goal_clamped else "")
        if self._cube is None:
            return ""
        height = 100 * self._height(self._cube)
        return f"kostka {height:.1f} cm nad blatem ({self._cube_src or 'bez zrodla'})"
