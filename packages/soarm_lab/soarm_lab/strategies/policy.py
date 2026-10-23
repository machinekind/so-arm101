"""A lab RL policy as a strategy for the twin: `policy` under the `machinekind_twin.strategies` entry points.

The twin finds `PolicyStrategy` through the entry point in this package's `pyproject.toml`, builds
`PolicySettings` from its config file's `policy:` section or the panel's form, creates the strategy as
`PolicyStrategy(settings)` and runs it through its `StrategyRunner`: the runner claims the arm, builds
an observation every tick and sends the strategy's joint targets through the safety supervisor, in
simulation and on hardware alike. Nothing here imports the twin. `Strategy`, `RunInfo`, `Observation`,
`ObjectPose` and `Action` come from its plugin contracts, `machinekind_twin_api.strategy`. The strategy reads
`obs.joints`, `obs.command`, `obs.goal`, `obs.objects` and `obs.t`, and `run.robot`, `run.folder`,
`run.limits` and `run.simulated`. It answers with an `Action`.

    # configs/local.yaml of the twin: prefills the panel's form
    policy:
      policy: lift-v3              # a bundled policy, one trained here (workspace/policies) or a path

The policy runs in numpy (`rl.policy.Policy`); the strategy needs no torch.

The loop does exactly what the training environment does - the observation from `task.observe`, the
action through `task.apply_action` - only the angles come from the servos, not from physics. Claiming the
arm, the thread, STOP and Home belong to the twin's `StrategyRunner`. The strategy adds the guards a real
arm needs and the training environment does not have:

* **stop on lag**: when the measured pose falls behind the target by more than `max_tracking_deg`
  (a blocked joint, a collision, a servo that cannot keep up) the run ends and the arm holds where it
  MEASURES - the twin's own tracking guard (STOP) comes on top, for every owner;
* **time limit**: a run on the arm lasts as many ticks as a training episode (`episode_limit`);
* **end on success**: `end_on_success` success ticks in a row end the run (lift: the cube above the
  table). Lift success counts only from a fresh cube pose (cameras, "in the hand", or simulation
  truth), never from "last seen", and only while the jaw holds (still, commanded tighter, not
  creeping closed during the streak);
* **reach goal in the training area**: a goal outside it (under the table, behind the base) is
  projected onto the area training goals were drawn from;
* **pickplace**: the place target is the goal projected onto the table (z = 0) and onto the training
  ring of targets. Success counts only for a cube SEEN lying (cameras, not "in the hand", not "last
  seen") on the target, released (the jaw does not hold it, the gripper command is open), still
  during the streak, and only after it was carried (the same latch as in training).

Where the inputs come from: the goal is `obs.goal` (the panel's goal); without one the settings'
`goal_x`, `goal_y`, `goal_z` hold it. The cube is `obs.objects["cube"]`: a pose in the base frame and
its `source` ("kamery", "w dloni", "ostatnie widziane", "symulacja"); an empty source counts as
simulation truth on a simulated arm and as stale on a real one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo, Strategy

from ..kinematics import RobotKinematics
from ..rl import task as tk
from ..rl.policy import Policy, find_policy
from ..robot import load_robot

logger = logging.getLogger(__name__)

#: Cube pose sources (`ObjectPose.source`, as the twin's perception publishes it) lift success may be
#: counted from: a fresh camera detection, the cube carried in the jaws, or simulation truth.
FRESH_CUBE_SOURCES = ("kamery", "w dloni", "symulacja")
#: Sources pickplace success may be counted from: the cube SEEN lying (not "w dloni": that pose follows
#: the jaws) or simulation truth.
RESTING_CUBE_SOURCES = ("kamery", "symulacja")
#: pickplace: during a success streak the cube moves no more than this [m].
PLACE_STILL = 0.005


class NoCube(RuntimeError):
    """A task with a cube, and the observation does not say where it is."""


@dataclass
class PolicySettings:
    #: A bundled policy (reach-v3, lift-v3), one trained here (`soarm-lab policies`), or a path to policy.npz.
    policy: str = "reach-v3"
    #: The run ends, holding the measured pose, when an arm joint lags its target by more than this [deg].
    max_tracking_deg: float = 25.0
    #: End the run after a training episode's number of ticks.
    episode_limit: bool = True
    #: How long the arm waits for the first cube pose [s] (lift, pickplace).
    wait_for_cube_s: float = 3.0
    #: The goal when the observation has none [m, base frame]: reach drives the TCP there, pickplace
    #: places the cube under it.
    goal_x: float = 0.22
    goal_y: float = 0.0
    goal_z: float = 0.12


@dataclass
class PolicyStatus:
    """What the strategy knows about its run, for logs and tests (the twin keeps its own runner status)."""

    #: Policy ticks done (a tick waiting for the cube does not count).
    step: int = 0
    distance: float = float("nan")
    success: bool = False
    stopped_because: str = ""
    last_action: list[float] = field(default_factory=list)
    #: How many ticks in a row the task is done.
    success_streak: int = 0
    #: reach: the goal was clipped to the training area. pickplace: the place target was moved onto the
    #: training ring of targets.
    goal_clamped: bool = False
    #: pickplace: the cube was carried in this run (latch).
    carried: bool = False


def project_goal(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """`goal` projected onto the area `sample_goals` draws reach goals from.

    The radius from the base axis within `goal_radius`, the height within `goal_height`, x > 0.05 m.
    The policy has never been outside that area: a goal under the table drives the jaws into the table.
    """
    g = np.asarray(goal, float).copy()
    g[2] = np.clip(g[2], *task.goal_height)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.goal_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    # x > 0.05 at radius r is a bearing of at most arccos(0.05 / r) from the x axis; 1 mm margin.
    b_max = float(np.arccos(min(1.0, 0.051 / r_new)))
    bearing = float(np.clip(bearing, -b_max, b_max))
    g[0], g[1] = r_new * np.cos(bearing), r_new * np.sin(bearing)
    return g


def project_target(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """The pickplace place target from `goal` (base frame): onto the table (z = 0) and the target ring.

    The radius from the base axis within `target_radius`, the bearing within `target_bearing`: the ring
    `sample_targets` draws training targets from. The policy has never placed a cube outside it.
    """
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
        """`policy`: an already loaded policy instead of `settings.policy` (tests, notebooks)."""
        self.config = settings if settings is not None else PolicySettings()
        if self.config.max_tracking_deg <= 0:
            raise ValueError(f"policy: max_tracking_deg must be positive, is {self.config.max_tracking_deg}")
        self.policy = policy if policy is not None else Policy.load(find_policy(self.config.policy))
        self.task = self.policy.task
        #: The policy's own rate: the twin's runner ticks at it.
        self.hz = float(self.task.control_hz)
        #: The run ends itself (episode limit, success); the twin needs no time limit of its own.
        self.time_limit_s = None
        self.status = PolicyStatus()
        end = getattr(self.task, "end_on_success", None)
        #: Success ticks in a row that end the run (0 = never), from the task. A policy saved without
        #: the field gets the task's default (`PolicyMeta.task_config`): lift 10, reach 0 (its goal is
        #: dragged live).
        self.end_on_success = int(end) if end is not None else (10 if self.task.cube else 0)
        self.max_tracking = np.radians(self.config.max_tracking_deg)
        self.episode_limit = self.config.episode_limit
        #: How long [s] the arm stands, waiting for the first cube pose.
        self.wait_for_cube = self.config.wait_for_cube_s
        self._default_goal = np.array([self.config.goal_x, self.config.goal_y, self.config.goal_z])
        self._goal = self._default_goal.copy()
        # Known from `start`: the arm's kinematics, the training limits and the hardware's.
        self._kin: RobotKinematics | None = None
        self._limits: tk.Limits | None = None
        self.g, self.na = 0, 0
        self.q_cmd = np.zeros(self.task.act_dim)
        self.prev_action = np.zeros(self.task.act_dim)
        self._hw_lo = self._hw_hi = np.zeros(self.task.act_dim)
        self.simulated = False
        self._cube: np.ndarray | None = None
        self._cube_src: str | None = None
        self._grip_prev: float | None = None
        self._streak_grip = 0.0
        #: pickplace: the cube was carried (as `task.carry_latch` in training), and where it was when the
        #: success streak began.
        self._carried = False
        self._streak_cube: np.ndarray | None = None

    @property
    def started(self) -> bool:
        """Whether `start` ran: the arm's kinematics and limits are known."""
        return self._kin is not None

    @property
    def kin(self) -> RobotKinematics:
        """The arm's kinematics, known from `start`."""
        if self._kin is None:
            raise RuntimeError("policy: start the run first")
        return self._kin

    @property
    def limits(self) -> tk.Limits:
        """The training limits of this arm, known from `start`."""
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
        if self.task.name == "reach":
            p = project_goal(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p - g) > 1e-4)
            g = p
        elif self.task.target:
            # The place target lies on the table: z = 0 is not clamping but the definition.
            p = project_target(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p[:2] - g[:2]) > 1e-4)
            g = p
        self._goal = g

    @property
    def target(self) -> np.ndarray:
        """pickplace: the place target (base frame, z = 0), `goal` projected by `project_target`."""
        return project_target(self.task, self._goal)

    # ------------------------------------------------------------------- start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Checks the policy against the arm and starts from where the arm is. Raises to refuse the run."""
        robot = getattr(self.policy.meta, "robot", "so101")
        if robot != run.robot:
            raise RuntimeError(f"polityka uczona na {robot}, stanowisko ma {run.robot}")
        spec = load_robot(run.folder)
        if self.policy.meta.act_dim != len(spec.joints):
            raise RuntimeError(f"polityka ma {self.policy.meta.act_dim} wyjsc, a ramie {spec.name} "
                               f"{len(spec.joints)} stawow")
        self._kin = RobotKinematics(spec)
        self._limits = tk.Limits.of(self._kin)
        #: The gripper's index and the number of arm joints (the gripper is last, `Limits.of`).
        self.g, self.na = self.limits.grip, self.limits.n_arm
        #: The jaw "holds" (lift success) as in the twin's `CubeTracker`: it stands still (moves less than
        #: `grip_still` between ticks), is wider than commanded by more than `grip_block_margin` and wider
        #: than closed by more than `grip_open_margin`. In the gripper's MJCF units (`spec.jaw`; SO-101: rad).
        jaw = spec.jaw
        self.grip_still, self.grip_block_margin, self.grip_open_margin = jaw.still, jaw.margin, jaw.margin
        #: During a success streak the jaw may close by no more than this (SO-101: 0.03 rad, about 3 units).
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
        self._cube, self._cube_src = None, None
        self._grip_prev = None
        self._carried, self._streak_cube = False, None
        self._take_goal(obs)

    def _squeezing_grip_cmd(self, measured: Mapping[str, float],
                            command: Mapping[str, float]) -> float | None:
        """The gripper command angle [rad] from `obs.command` when the gripper SQUEEZES, else None.

        It squeezes when the command is tighter than the measured angle by more than `grip_block_margin`.
        Starting from the measured jaw angle is zero force: a new run with a cube in the jaws (after
        STOP, Home or Stop) would release the grip with its first action, and `_jaw_holding` would never
        see the jaw wider than commanded, so lift success could not count. In training, too, the q_cmd
        of a held cube is tighter than the measured angle. The other joints start from the measured pose.
        """
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
        """The training limits intersected with the limits that really apply on the arm (`run.limits`).

        The observation's normalization stays on the training limits: another scale would be another
        observation for the policy. Only the target is clipped: without it q_cmd stood up to 7 deg past
        the servo limit (arm no. 2, wrist_flex at +88). An empty intersection keeps the training limits.
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
        """The policy's observation vector, as `task.observe` builds it in training."""
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        target = None
        if self.task.cube:
            got = (objects or {}).get(self.task.cube)
            if got is None:
                raise NoCube(f"zadanie {self.task.name} potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = np.asarray(got.position, float), np.asarray(got.rotation, float)
            self._cube_src = str(got.source or "") or None
            self._cube = cube_pos.copy()
            if self.policy.meta.randomization.get("fold_yaw", False):
                # In training the policy saw the yaw folded to +-45 deg (as the detector gives it); the
                # "w dloni" pose and simulation truth are not folded, so fold every pose the same way.
                cube_rot = tk.fold_yaw(np, cube_rot)
        if self.task.target:
            target = self.target[None]
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], tcp[None],
                         self.prev_action[None], goal=self.goal[None], cube_pos=cube_pos[None],
                         cube_rot=cube_rot[None], target=target)
        return obs[0].astype(np.float32)

    # --------------------------------------------------------------------- tick

    def act(self, obs: Observation) -> Action:
        """One policy tick: the next joint targets, or the end of the run."""
        if self._kin is None:
            raise RuntimeError("policy: act() przed start()")
        st = self.status
        measured = dict(obs.joints)
        self._take_goal(obs)
        q = self.kin.to_q(measured)
        lag = np.abs(q[:self.na] - self.q_cmd[:self.na]).max()
        if lag > self.max_tracking:
            # The arm stops where it IS: the last target then lies ~25 deg past the obstacle.
            return self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                              target=self._hold_pose(measured, q))
        holding = self._jaw_holding(q[self.g])
        try:
            vec = self.observation(measured, obs.objects)
        except NoCube as exc:
            # At the start perception may not have its first detection yet: the arm stands and
            # waits. A cube lost DURING the episode stops the policy.
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
        if self.task.name == "reach":
            st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
            st.success = st.distance < self.task.success_dist
        elif self.task.target:
            st.success = self._placed(holding)
            st.carried = self._carried
        else:
            # The arm has no jaw contact sensors: success is the cube, from a fresh pose
            # (`_cube_fresh`), higher above the table than `lift_height`, and only while the jaw
            # HOLDS it (still, commanded tighter than it is). Measured in the twin (1 of 18
            # successes): a 1 s streak at 10.9 cm while the cube turned in the jaws; the jaw closed
            # from 31.8 to 0 over 3.5 s, and the cube fell 4 s after "zadanie wykonane".
            # A jaw closing slowly still "stands" from tick to tick (9 units/s is 0.005 rad per
            # tick), so in addition the jaw may not close by more than `grip_creep` during a streak.
            if holding and st.success_streak > 0 and q[self.g] < self._streak_grip - self.grip_creep:
                holding = False
            st.success = bool(self._cube is not None and self._cube_fresh() and holding
                              and self._cube[2] - self.task.cube_half > self.task.lift_height)
            if st.success and st.success_streak == 0:
                self._streak_grip = float(q[self.g])
        st.success_streak = st.success_streak + 1 if st.success else 0
        if self.end_on_success > 0 and st.success_streak >= self.end_on_success:
            return self._halt("zadanie wykonane", target=target, success=True)
        if self.episode_limit and st.step >= self.task.episode_steps:
            return self._halt("koniec epizodu", target=target, success=st.success)
        return Action(target=target, success=st.success, error=self._error(), note=self._note())

    def _placed(self, holding: bool) -> bool:
        """pickplace: this tick's success (and updates the "carried" latch and `status.distance`).

        The cube is seen lying (`RESTING_CUBE_SOURCES`) on the target (xy nearer than `place_radius`, its
        bottom lower than `place_height`), the jaw does not hold it, the gripper command is open, it was
        carried before this tick, and it has not moved by `PLACE_STILL` since the streak began. The latch:
        simulation truth or "w dloni" from the tracker, higher above the table than `lift_height`.
        """
        cube, src, task = self._cube, self._cube_src, self.task
        if cube is None:
            return False
        if src is None and self.simulated:
            src = "symulacja"                               # no source in simulation = the scene's truth
        height = float(cube[2]) - task.cube_half
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
            ok = False                                      # the cube still rolls or slides
        if ok and (self.status.success_streak == 0 or self._streak_cube is None):
            self._streak_cube = np.asarray(cube, float).copy()
        return ok

    def _jaw_holding(self, grip_q: float) -> bool:
        """Whether the jaw holds: still since the last tick, wider than commanded and than a closed jaw.

        The same thresholds as the twin's `CubeTracker` ("w dloni"). The command is the PREVIOUS tick's:
        the servo is driving towards it now. A jaw that is closing (the cube slips out or turns) is not
        still.
        """
        prev, self._grip_prev = self._grip_prev, float(grip_q)
        still = prev is not None and abs(grip_q - prev) < self.grip_still
        squeezing = (grip_q - self.q_cmd[self.g] > self.grip_block_margin
                     and grip_q > self.limits.lo[self.g] + self.grip_open_margin)
        return bool(still and squeezing)

    def _hold_pose(self, measured: Mapping[str, float], q: np.ndarray) -> dict[str, float]:
        """The last target of a run stopped on lag: the arm where it is, as the twin's `hold_measured`.

        A gripper that squeezes (measured wider than commanded) is left out of the target and keeps its
        command: commanding the measured angle of a blocked jaw is zero force, and a carried cube falls.
        """
        pose = {name: float(measured[name]) for name in self.kin.spec.joints if name in measured}
        grip = self.kin.spec.gripper
        if grip in pose and q[self.g] - self.q_cmd[self.g] > self.grip_block_margin:
            del pose[grip]
        return pose

    def _cube_fresh(self) -> bool:
        """Whether the cube pose is fresh: lift success may count only from a fresh pose.

        The twin's tracker keeps the last pose for up to 6 s as "ostatnie widziane". Measured in the twin
        (camera a): the cube fell out of the jaws after 0.5 s of carrying, and success counted from the
        stale pose in the air ended the run with "zadanie wykonane" while the cube lay on the table.
        Without a source: fresh in simulation (the provider gives the scene's truth), stale on a real arm.
        """
        if self._cube_src is None:
            return self.simulated
        return self._cube_src in FRESH_CUBE_SOURCES

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, target: Mapping[str, float] | None = None,
              success: bool = False) -> Action:
        """The run is over: the last target (or the measured pose) and the reason for the twin's panel."""
        self.status.stopped_because = reason
        logger.info("Polityka zatrzymana: %s", reason)
        return Action(target=dict(target or {}), done=True, success=success, error=self._error(),
                      note=reason)

    def stop(self, reason: str) -> None:
        """The twin ended the run (Stop, STOP, Home, the arm taken, or our own `done`)."""
        # After our own `done` the twin reports "zadanie wykonane" or "strategia skonczyla"; ours says more.
        if reason in ("zadanie wykonane", "strategia skonczyla") and self.status.stopped_because:
            return
        self.status.stopped_because = reason or "zatrzymana"

    def _error(self) -> float | None:
        d = self.status.distance
        return float(d) if np.isfinite(d) else None

    def _note(self) -> str:
        st = self.status
        if self.task.name == "reach":
            txt = f"do celu {st.distance * 1000:.0f} mm" + (" (w celu)" if st.success else "")
            return txt + (" - cel przyciety do obszaru z treningu" if st.goal_clamped else "")
        if self.task.target:
            txt = f"kostka do celu {st.distance * 1000:.0f} mm" + (" (niesiona)" if st.carried else "")
            return txt + (" - cel przesuniety na obszar z treningu" if st.goal_clamped else "")
        if self._cube is None:
            return ""
        height = 100 * (float(self._cube[2]) - self.task.cube_half)
        return f"kostka {height:.1f} cm nad blatem ({self._cube_src or 'bez zrodla'})"
