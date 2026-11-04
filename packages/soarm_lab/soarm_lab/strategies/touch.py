"""Touching objects as a strategy for the twin: `touch` under the `machinekind_twin.strategies` entry points.

The twin finds `TouchStrategy` through the entry point, builds `TouchSettings` from its config file's
`touch:` section or the panel's form, and runs the strategy through its `StrategyRunner`, as it runs `ik`.
Nothing here imports the twin: the contract types come from `machinekind_twin_api.strategy`.

    # configs/local.yaml of the twin: prefills the panel's form
    touch:
      label: cube                  # the object's label in obs.objects, as the detector names it
      touch_z: 0.03                # the TCP height at the bottom of the descent [m above the table]
      hover_m: 0.06                # the hover point lies this much above the touch point [m]
      max_pose_age_s: 30.0         # an older pose is never taken [s]
      wait_for_pose_s: 30.0        # how long to wait for a usable pose [s]

The object's pose comes from the twin's perception, and the detector it is built for is the coarse one:
a language model answers seconds late, 1 to 3 cm off, with no size and no orientation, and its point lies
on the table top, not at the object's centre. So the strategy takes only x and y from the first usable
pose and puts the touch point at `touch_z` above the table, with the hover point `hover_m` above it. The
target is then latched: later poses never move it, because the hand hides the object during the descent.

The motion is `ik`'s: every tick a point (the carrot) moves toward the phase's point by at most one step,
and `RobotKinematics.ik_step`, started from the last target, turns it into joints. The carrot goes on
from the last carrot while the targets reach it, so the path stays on a straight line, and from the TCP
of the last target when they do not, as in `ik`. A step is shortened when it would move a joint faster
than the supervisor allows (`RunInfo.max_vel`, which includes the panel's speed cap). The gripper closes
at the start (`close_jaw`), so the closed tips probe the object instead of straddling it. The TCP goes to
the hover point at `speed_mps` (straight up first when it starts below the hover height), then straight
down at `descend_speed_mps`.

The arm is measured against the supervisor's command (`obs.command`), not against the strategy's own
target: the command trails the targets, more so when the operator lowers the speed cap during a run. A
point counts as reached once the last target is on it, the command has reached that target, and the
measured TCP is near the point or stands still within `SAG_M` of it, for `SETTLE_S` in a row. Where the
point lies above the arm's way (a lift, the hover), an arm that stands still `sag_m` or less below it, and
mostly straight below it, counts as reached too: a servo without an integral term settles short under load.

At the hover the strategy notes where each joint stands against its command. During the descent the arm
has met something once the measured TCP stands more than `contact_gap_m` above where the arm should be, or
more than half of it while the arm stands still at the touch point. A joint that settled short of its
command is expected to stay where it hangs until its command comes back to it, as friction holds it, and
then to follow the command; any other joint keeps its offset from the command. The arm then holds the pose
it measures, so it stops pressing. That hold goes out as an ordinary target and the verdict waits until the
command has reached it, because the twin's runner drops the target sent with `done`.

Guards:

* **speed limits**: no target step moves an arm joint farther than `SPEED_SHARE` of `RunInfo.max_vel` in
  one tick;
* **pose check**: a pose counts only from the sources in `TOUCH_SOURCES` (an empty source only in
  simulation), no older than `max_pose_age_s`, with a finite position. Without one the arm holds its
  start pose, and the run ends after `wait_for_pose_s`;
* **reach check**: when the pose is taken (not in `start`, which may come before any pose), the full
  multi-start `RobotKinematics.ik` must reach both the hover and the touch point within the limits on
  the arm, or the run ends;
* **stop on contact**: the descent ends where the arm stalls, holding the measured pose;
* **contact band**: a stall more than `contact_band_m` above the touch point is not the object;
* **the arm's own body**: near the base the hand reaches a point low over the table only through the
  shoulder, and that stall would read as a touch. The hover and touch points must be reachable without
  two of the arm's bodies meeting (`RobotKinematics.self_contacts`), and the run ends on the first
  target that would bring two of them together. On the SO-101 at the defaults that happens within
  about 17 cm of the base;
* **stop on lag**: the run ends when an arm joint falls behind the supervisor's command by more than
  `max_tracking_deg`. Unlike `ik`, `act` then raises: the twin's runner holds the arm where it measures
  after an exception, while it would drop a hold pose sent with `done` and keep the command it had already
  sent past the obstacle;
* **an arm that does not move**: standing still counts as arrival only within `SAG_M` of the point, or
  `sag_m` below a point above the arm's way, and a stall is a touch only after the measured TCP came down
  from the hover by `contact_gap_m` or more. Frozen readings, a servo without torque or a jammed joint
  never end a run as a success;
* **stop on a still arm**: a run whose arm has stood still for `stall_s` short of a point it waits for
  ends, and the note names the phase and the joints that lag;
* **the jaw**: IK never moves the gripper. It keeps the command it squeezes something with, else it
  closes (`close_jaw`), else it keeps its start angle.

Success means contact: the descent came down from the hover and stopped on a stall within the band, and
the runner reports "zadanie wykonane". A descent that reaches the touch point in free air is an honest
miss ("brak kontaktu"), unless `need_contact` is off. Whether the thing the arm met was the labelled
object is left to whoever judges the touch. The hand, not the TCP, meets the object first: position-only
IK tilts the approach axis by 20 to 45 degrees, and in the twin the TCP stalled 4 to 16 mm above a 3 cm
cube, so the stall height says little about the object's height. A touch also pushes a 30 g cube by about
5 mm. test_touch_in_twin.py and test_touch_from_cameras.py judge the runs by the simulation's own contact
sensors instead. The thresholds were tried in the twin and against simulated servo lag, sag and speed
caps; a real arm's sag and jitter may need another `contact_gap_m` or `descend_speed_mps`, with
`need_contact` off as the fallback.

Phases (`TouchStatus.phase`): wait, then lift (only when the TCP starts below the hover height), approach,
descend and touched, then done. The "touched" phase is the seam for a decision step after the descent:
`after_touch` judges it and is the single method to replace; `hold` keeps the arm where it is while a
decision is pending; `retarget` aims at a new point within the same run, lifting straight up first;
`descent_verdict` gives the contact geometry's own verdict. `touch_jev` (`soarm_lab.strategies.touch_jev`)
is such a step. It asks the Jev decision model after each descent. `_drive` accepts an optional rotation
goal for subclasses. A judge's latency counts toward `time_limit_s`. Near the hand, the twin's tracker
rejects single-camera answers and the pose keeps ageing. Repeated answers do not refresh it. This strategy
rejects "w dloni", which a closed jaw on a real arm may report.

The strategy reads `obs.objects[label]` (x, y, source and age_s), `obs.joints`, `obs.command` and
`obs.t`. It ignores `obs.goal`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo, Strategy

from ..kinematics import RobotKinematics, turn_toward
from ..robot import load_robot

logger = logging.getLogger(__name__)

#: Pose sources (`ObjectPose.source`) a target may come from. They match exactly, so annotated last-seen
#: poses ("ostatnie widziane (upuszczona)") and "w dloni" never count.
TOUCH_SOURCES = ("kamery", "ostatnie widziane", "symulacja")
#: The lowest `touch_z` [m]. The hand reaches below the TCP (in the twin the TCP stalled up to 16 mm above
#: a cube's top), so lower than this the table could stall the descent and read as a touch.
MIN_TOUCH_Z = 0.02

#: An arm that stands still counts as arrived at a point only this close to it [m]: enough for a servo's sag
#: or a hand held up a few millimetres, too little for an arm that never moved (readings that stopped, no
#: torque, a jammed joint).
SAG_M = 0.01
#: How long the arm stays arrived at the hover and at the touch point before the strategy reads it [s]. A
#: servo that lags its command still moves when the command arrives, and the offset at the hover is the
#: reference of the whole descent.
SETTLE_S = 0.25
#: The share of the supervisor's speed limits (`RunInfo.max_vel`) one target step may use: the twin's 50 Hz
#: loop gives the command 40 to 60 ms to follow each tick at 20 Hz.
SPEED_SHARE = 0.8
#: The supervisor's command is on the target when no arm joint differs by more than this [deg]: a servo
#: backend reports the command rounded to its steps (0.09 degrees on an STS3215).
COMMAND_TOL_DEG = 0.2
#: The longest wait for the command to reach the pose the arm holds after a descent, before the verdict [s].
HOLD_WAIT_S = 2.0
#: Maximum turn rate for an optional orientation goal [deg/s]. Small steps avoid wrist limits.
TURN_DEG_S = 90.0
#: A sagging arm stands below the point within this angle of straight down [deg].
SAG_CONE_DEG = 60.0
#: A stop on a still arm names the joints that lag their command by at least this [deg].
LAG_NOTE_DEG = 0.5

#: Phases of a run (`TouchStatus.phase`).
WAIT = "wait"
LIFT = "lift"
APPROACH = "approach"
DESCEND = "descend"
TOUCHED = "touched"
DONE = "done"
#: The phases' names in the panel's notes.
PHASE_NAMES = {WAIT: "czekam na polozenie", LIFT: "w gore przed nowym celem", APPROACH: "nad obiekt",
               DESCEND: "w dol", TOUCHED: "dotyk", DONE: "koniec"}


@dataclass
class TouchSettings:
    #: The object's key in `obs.objects`, compared after lower-casing and collapsing white space.
    label: str = "cube"
    #: The TCP height at the bottom of the descent, the object's top [m above the table, base frame].
    #: The pose's own height is ignored.
    touch_z: float = 0.03
    #: The hover point lies this much above the touch point [m].
    hover_m: float = 0.06
    #: How long the arm waits for a usable pose before the run ends [s].
    wait_for_pose_s: float = 30.0
    #: A pose older than this (`ObjectPose.age_s`) is never taken [s]. Checked only when the pose is taken.
    max_pose_age_s: float = 30.0
    #: Close the gripper at the start, so the closed tips probe the object; a squeezing command always wins.
    close_jaw: bool = True
    #: Success needs the descent to stop on contact; False also counts reaching the touch point.
    need_contact: bool = True
    #: Ticks per second: the twin's runner ticks at this rate.
    hz: float = 20.0
    #: Highest TCP speed on the way to the hover point and on a lift [m/s]; lower where the supervisor's
    #: speed limits (`RunInfo.max_vel`) need it.
    speed_mps: float = 0.10
    #: TCP speed going down [m/s], at most; lower where the supervisor's speed limits need it.
    descend_speed_mps: float = 0.02
    #: Contact: during the descent the measured TCP sits this much higher above the supervisor's command than
    #: at the hover [m]; half of it once the arm stands still at the touch point.
    contact_gap_m: float = 0.004
    #: A stall higher than this above the touch point is a block, not the object [m].
    contact_band_m: float = 0.03
    #: The run ends, the arm holding where it measures, when an arm joint lags the supervisor's command by
    #: more than this [deg].
    max_tracking_deg: float = 25.0
    #: How far below a lift or hover point a still arm may stand and count as there [m]; 0 turns it off.
    sag_m: float = 0.02
    #: The run ends when the arm has stood still this long short of a point it waits for [s].
    stall_s: float = 2.0
    #: A safety net for the whole run [s]; 0 means none.
    time_limit_s: float = 60.0


@dataclass
class TouchStatus:
    """What the strategy knows about its run, for logs and tests (the twin keeps its own runner status)."""

    #: One of `WAIT`, `LIFT`, `APPROACH`, `DESCEND`, `TOUCHED`, `DONE`.
    phase: str = WAIT
    #: Ticks done; the tick that stops the run on lag does not count.
    step: int = 0
    #: Descents started in this run.
    attempt: int = 0
    #: Where the current descent aims [m, base frame].
    touch_point: tuple[float, float, float] | None = None
    #: The source and age of the pose the first target came from, when it was taken.
    pose_source: str = ""
    pose_age_s: float = float("nan")
    #: The last descent stopped on a stall.
    contact: bool = False
    #: The measured TCP height where the last descent stopped [m].
    stop_z_m: float = float("nan")
    #: How far the measured TCP came down from the hover in the last descent [m].
    drop_m: float = float("nan")
    #: How much higher the measured TCP sits above the supervisor's command than it did at the hover [m].
    rise_m: float = float("nan")
    #: The measured TCP's distance from the current phase's point [m].
    distance: float = float("nan")
    #: The last target misses its carrot: the arm cannot follow.
    out_of_reach: bool = False
    success: bool = False
    stopped_because: str = ""


class TouchStrategy(Strategy[TouchSettings]):
    """Touches an object from `obs.objects`: hover above it, then straight down until the arm stalls on it."""

    settings = TouchSettings
    #: The phases' names in the panel's notes; subclasses add theirs.
    phase_names: ClassVar[dict[str, str]] = PHASE_NAMES

    def __init__(self, settings: TouchSettings | None = None):
        self.config = settings if settings is not None else TouchSettings()
        config = self.config
        self.label = normalised(config.label)
        if not self.label:
            raise ValueError("touch: label must not be empty")
        if not config.touch_z >= MIN_TOUCH_Z:
            raise ValueError(f"touch: touch_z must be at least {MIN_TOUCH_Z} m, is {config.touch_z}")
        for name in ("hover_m", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                     "max_tracking_deg"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"touch: {name} must be positive, is {value}")
        for name in ("wait_for_pose_s", "max_pose_age_s", "time_limit_s", "sag_m"):
            value = getattr(config, name)
            if not value >= 0:
                raise ValueError(f"touch: {name} must be zero or more, is {value}")
        if not config.stall_s >= SETTLE_S:
            raise ValueError(f"touch: stall_s must be at least {SETTLE_S} s, is {config.stall_s}")
        self.hz = float(config.hz)
        #: The runner reads it once, when it is built.
        self.time_limit_s = float(config.time_limit_s) if config.time_limit_s > 0 else None
        self.status = TouchStatus()
        self.max_tracking = np.radians(config.max_tracking_deg)
        #: The longest carrot steps [m]: the speeds over one tick.
        self.approach_step_m = config.speed_mps / self.hz
        self.descend_step_m = config.descend_speed_mps / self.hz
        #: How close IK puts the TCP to the carrot [m]: at most 1 mm, and a quarter of a descent step.
        self.ik_tol = min(1e-3, self.descend_step_m / 4)
        #: The measured TCP stands still when it moves less than this in one tick [m].
        self.still_m = self.descend_step_m / 2
        #: Maximum rotation of the orientation goal per tick [rad].
        self.turn_step = np.radians(TURN_DEG_S) / self.hz
        #: Ticks in a row the arm stays arrived before the strategy reads it (`SETTLE_S`).
        self.settle_ticks = max(1, int(np.ceil(SETTLE_S * self.hz)))
        #: Ticks in a row a still arm may wait short of a point before the run ends (`stall_s`).
        self.stall_ticks = max(1, int(np.ceil(config.stall_s * self.hz - 1e-9)))
        self.command_tol = np.radians(COMMAND_TOL_DEG)
        # Known from `start`: the arm's kinematics, the limits that apply on it [rad] and the farthest each
        # joint's target may move in one tick [rad].
        self._kin: RobotKinematics | None = None
        self._lo = self._hi = self._max_step = np.zeros(0)
        #: The last target [rad, the description's order], and the pairs of the arm's bodies that touch in it.
        self.q_cmd = np.zeros(0)
        self._contacts: set[tuple[str, str]] = set()
        #: The TCP of the last target and the point the last tick asked IK for [m, base frame].
        self.tcp_cmd = np.zeros(3)
        self.carrot = np.zeros(3)
        #: The object's x and y the run aims at [m, base frame]; NaN until a pose is taken.
        self.touch_xy = np.full(2, np.nan)
        self._simulated = False
        self._phase_t0 = 0.0
        self._t = 0.0
        # This tick's measurements, for the phase handlers.
        self._measured: dict[str, float] = {}
        self._q_measured = np.zeros(0)
        self._q_command = np.zeros(0)
        self._tcp_measured = np.zeros(3)
        self._still = False
        #: Ticks in a row the measured TCP stood still in this phase.
        self._still_count = 0
        self._command_on_target = True
        self._offset_z = 0.0
        self._tcp_prev: np.ndarray | None = None
        self._settle_count = 0
        #: The last target when the phase began [rad], and the point the arm has stood still short of for
        #: `stall_s` on this tick, if any.
        self._q_enter = np.zeros(0)
        self._stalled: np.ndarray | None = None
        #: The measured TCP's height above the supervisor's command at the hover, and its height there [m].
        self._bias_z = 0.0
        self._hover_z = float("nan")
        #: The measured joints and the command where the descent started [rad].
        self._hang_q = np.zeros(0)
        self._hang_command_q = np.zeros(0)
        self._lift_point = np.zeros(3)
        self._phases: dict[str, Callable[[Observation], Action | None]] = {
            WAIT: self._wait, LIFT: self._lift, APPROACH: self._approach, DESCEND: self._descend,
            TOUCHED: self._judge, DONE: lambda obs: self.hold(),
        }

    @property
    def started(self) -> bool:
        """Whether `start` ran: the arm's kinematics and limits are known."""
        return self._kin is not None

    @property
    def kin(self) -> RobotKinematics:
        """The arm's kinematics, known from `start`."""
        if self._kin is None:
            raise RuntimeError("touch: start the run first")
        return self._kin

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of the targets [rad], known from `start` (`_hardware_limits`)."""
        if self._kin is None:
            raise RuntimeError("touch: start the run first")
        return self._lo, self._hi

    @property
    def hover_point(self) -> np.ndarray:
        """Where the descent starts [m, base frame]."""
        return np.array([*self.touch_xy, self.config.touch_z + self.config.hover_m])

    @property
    def touch_point(self) -> np.ndarray:
        """Where the descent ends at the latest [m, base frame]."""
        return np.array([*self.touch_xy, self.config.touch_z])

    # ------------------------------------------------------------------- start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Starts from where the arm is, with or without a pose: the pose may come later."""
        spec = load_robot(run.folder)
        self._kin = RobotKinematics(spec)
        obs = self._observed(obs)
        self._lo, self._hi = self._hardware_limits(run.limits)
        self._max_step = self._step_limits(run.max_vel)
        self._simulated = bool(run.simulated)

        measured = dict(obs.joints)
        self.q_cmd = np.clip(self.kin.to_q(measured), self._lo, self._hi)
        if spec.gripper is not None:
            g = spec.joints.index(spec.gripper)
            grip = self._squeezing_grip_cmd(measured, obs.command)
            if grip is not None:
                self.q_cmd[g] = grip
            elif self.config.close_jaw:
                self.q_cmd[g] = self._lo[g]
        self.status = TouchStatus()
        self.touch_xy = np.full(2, np.nan)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        self._tcp_prev = None
        self._still_count = 0
        self._command_on_target = True
        self._bias_z = 0.0
        self._hover_z = float("nan")
        self._q_measured = self.kin.to_q(measured)
        self._q_command = self.q_cmd.copy()
        self._hang_q, self._hang_command_q = self._q_measured.copy(), self.q_cmd.copy()
        self._t = float(obs.t)
        self._enter(WAIT)

    def _observed(self, obs: Observation) -> Observation:
        """`obs` as the strategy reads it; subclasses may convert readings. Touch takes it as it is."""
        return obs

    def _hardware_limits(self, hw: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """The description's limits and the MJCF ranges, intersected with the ones on the arm (`run.limits`).

        The same limits `ik` clips its targets to. An empty intersection keeps the description's limits.
        """
        kin, spec = self.kin, self.kin.spec
        a = kin.to_q({name: spec.limits[name][0] for name in spec.joints})
        b = kin.to_q({name: spec.limits[name][1] for name in spec.joints})
        base_lo, base_hi = np.maximum(np.minimum(a, b), kin.lo), np.minimum(np.maximum(a, b), kin.hi)
        lo, hi = base_lo.copy(), base_hi.copy()
        for k, name in enumerate(spec.joints):
            if name in hw:
                a, b = kin.to_q({name: hw[name][0]})[k], kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = base_lo[bad], base_hi[bad]
        return lo, hi

    def _step_limits(self, max_vel: Mapping[str, float]) -> np.ndarray:
        """The farthest each arm joint's target may move in one tick [rad]: `SPEED_SHARE` of the supervisor's
        limit (`run.max_vel`, the description's `max_vel` for a joint it leaves out). The gripper has none."""
        spec = self.kin.spec
        steps = np.full(len(spec.joints), np.inf)
        for k, name in enumerate(spec.joints):
            speed = float(max_vel.get(name, spec.max_vel.get(name, np.inf)))
            if name != spec.gripper and speed > 0:
                steps[k] = np.radians(speed) * SPEED_SHARE / self.hz
        return steps

    def _within_limits(self, joints: Mapping[str, float]) -> bool:
        """Whether the arm joints of an IK solution lie within the limits that apply on the arm."""
        q, arm = self.kin.to_q(joints), self.kin.arm
        return bool(np.all(q[arm] >= self._lo[arm] - 1e-9) and np.all(q[arm] <= self._hi[arm] + 1e-9))

    def _clear(self, joints: Mapping[str, float]) -> bool:
        """Whether an IK solution lies within the limits and keeps the arm off itself."""
        return self._within_limits(joints) and not self.kin.self_contacts(joints)

    def _squeezing_grip_cmd(self, measured: Mapping[str, float],
                            command: Mapping[str, float]) -> float | None:
        """The gripper command angle [rad] from `obs.command` when the gripper SQUEEZES, else None.

        It squeezes when the command is tighter than the measured angle by more than the jaw's `margin`
        (`spec.jaw`), as `ik` decides it.
        """
        name = self.kin.spec.gripper
        if name is None or name not in measured:
            return None
        cmd = command.get(name)
        if cmd is None:
            return None
        g = self.kin.spec.joints.index(name)
        q_cmd = float(self.kin.to_q({**measured, name: float(cmd)})[g])
        if float(self.kin.to_q(measured)[g]) - q_cmd <= self.kin.spec.jaw.margin:
            return None
        return float(np.clip(q_cmd, self._lo[g], self._hi[g]))

    # --------------------------------------------------------------------- tick

    def act(self, obs: Observation) -> Action:
        """One tick: the next joint targets, or the end of the run."""
        if self._kin is None:
            raise RuntimeError("touch: act() przed start()")
        st = self.status
        kin, arm = self.kin, self.kin.arm
        self._t = float(obs.t)
        measured = dict(obs.joints)
        # The arm follows the supervisor's command, which trails the targets under the operator's speed cap.
        # A joint the command leaves out takes the last target.
        q_command = kin.to_q({**kin.from_q(self.q_cmd), **obs.command})
        lag = float(np.abs(kin.to_q(measured)[arm] - q_command[arm]).max())
        if lag > self.max_tracking:
            reason = f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada"
            self._end(reason)
            # Raised, not returned with `done`: the twin's runner then holds the arm where it measures. It
            # drops a target sent with `done`, and the command it keeps lies past the obstacle.
            raise RuntimeError(reason)

        st.step += 1
        st.stopped_because = ""
        self._measured = measured
        self._q_measured, self._q_command = kin.to_q(measured), q_command
        tcp_measured = kin.tcp(measured)[:3, 3].copy()
        moved = np.inf if self._tcp_prev is None else float(np.linalg.norm(tcp_measured - self._tcp_prev))
        self._still = moved <= self.still_m
        self._still_count = self._still_count + 1 if self._still else 0
        self._tcp_measured = self._tcp_prev = tcp_measured
        self._command_on_target = float(np.abs(q_command[arm] - self.q_cmd[arm]).max()) <= self.command_tol
        # Against the command, which the arm follows, so a command held back by the speed cap is no stall.
        self._offset_z = float(tcp_measured[2] - kin.tcp(kin.from_q(q_command))[2, 3])

        # A handler returns None when it moved to another phase, which then handles the same tick.
        self._stalled = None
        for _ in range(len(self._phases)):
            action = self._phases[st.phase](obs)
            if action is not None:
                if self._stalled is not None and not action.done:
                    return self._halt(self._stall_reason(self._stalled), target=self.kin.from_q(self.q_cmd))
                return action
        raise RuntimeError(f"touch: no answer in phase {st.phase}")

    def _wait(self, obs: Observation) -> Action | None:
        """No target yet: holds the start pose until a usable pose comes, or the wait runs out."""
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        why = self._pose_problem(pose)
        if why:
            if self._t - self._phase_t0 >= self.config.wait_for_pose_s:
                return self._halt(f"brak uzytecznego polozenia obiektu '{self.label}' po "
                                  f"{self.config.wait_for_pose_s:.0f} s ({why})",
                                  target=self.kin.from_q(self.q_cmd))
            return self.hold(f"czekam na polozenie obiektu '{self.label}' ({why})")
        assert pose is not None
        self.status.pose_source, self.status.pose_age_s = str(pose.source), float(pose.age_s)
        try:
            self._aim(np.asarray(pose.position, float)[:2])
        except RuntimeError as exc:
            return self._halt(str(exc), target=self.kin.from_q(self.q_cmd))
        self._after_aim()
        return None

    def _pose_problem(self, pose: ObjectPose | None) -> str:
        """Why the pose cannot give a target, for the panel's note; empty when it can."""
        if pose is None:
            return "brak w obserwacji"
        source = str(pose.source or "")
        if not source and not self._simulated:
            return "bez zrodla poza symulacja"
        if source and source not in TOUCH_SOURCES:
            return f"zrodlo '{source}'"
        age = float(pose.age_s)
        if not age <= self.config.max_pose_age_s:
            return f"za stare: {age:.0f} s"
        position = np.asarray(pose.position, float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return "polozenie nieskonczone"
        return ""

    def _aim(self, xy: np.ndarray) -> None:
        """Aims the run at `xy`. Raises, changing nothing, when the arm cannot take the hover or the touch
        point within its limits without hitting itself."""
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise RuntimeError("punkt poza zasiegiem ramienia: wspolrzedne nieskonczone")
        z = self.config.touch_z
        seed = self.kin.from_q(self.q_cmd)
        for point in (np.array([*xy, z + self.config.hover_m]), np.array([*xy, z])):
            sol = self.kin.ik(point, seed=seed, accept=self._clear)
            if not (sol.ok and self._clear(sol.joints)):
                where = f"punkt ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"
                reachable = self.kin.ik(point, seed=seed, accept=self._within_limits)
                if reachable.ok and self._within_limits(reachable.joints):
                    raise RuntimeError(f"{where}: ramie uderzyloby samo w siebie")
                raise RuntimeError(f"{where} poza zasiegiem ramienia")
            seed = sol.joints
        self.touch_xy = xy.copy()
        self.status.touch_point = (float(xy[0]), float(xy[1]), float(z))

    def _after_aim(self) -> None:
        """To the hover point; straight up to its height first when the last target lies below it."""
        hover_z = self.config.touch_z + self.config.hover_m
        if self.tcp_cmd[2] < hover_z - self.ik_tol:
            self._lift_point = np.array([self.tcp_cmd[0], self.tcp_cmd[1], hover_z])
            self._enter(LIFT)
        else:
            self._enter(APPROACH)

    def _lift(self, obs: Observation) -> Action | None:
        if self._arrived(self._lift_point, sag=True):
            self._enter(APPROACH)
            return None
        return self._drive(self._lift_point, self.approach_step_m)

    def _approach(self, obs: Observation) -> Action | None:
        if self._settled(self.hover_point, sag=True):
            self._take_reference()
            self.status.attempt += 1
            self._enter(DESCEND)
            return None
        return self._drive(self.hover_point, self.approach_step_m)

    def _descend(self, obs: Observation) -> Action | None:
        st = self.status
        st.rise_m = self._contact_rise()
        if st.rise_m > self.config.contact_gap_m:
            return self._stall()
        if self._settled(self.touch_point):
            # Held up just above the touch point: the command stops there, so the rise stops growing.
            if st.rise_m > self.config.contact_gap_m / 2:
                return self._stall()
            self._touched(contact=False)
            return None
        return self._drive(self.touch_point, self.descend_step_m)

    def _stall(self) -> Action:
        """The arm stalls on something: it holds the pose it measures and stops pressing. The verdict comes
        on a later tick, because the twin's runner drops the target sent with `done`."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        self._touched(contact=True)
        return self.hold()

    def _touched(self, contact: bool) -> None:
        st = self.status
        st.contact = contact
        st.stop_z_m = float(self._tcp_measured[2])
        st.drop_m = self._hover_z - st.stop_z_m
        st.distance = float(np.linalg.norm(self.touch_point - self._tcp_measured))
        self._enter(TOUCHED)

    def _judge(self, obs: Observation) -> Action:
        """The phase "touched": `after_touch`, once the supervisor's command has reached the pose the arm
        holds, or after `HOLD_WAIT_S`."""
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        return self.after_touch(obs)

    def after_touch(self, obs: Observation) -> Action:
        """Judges the descent, on every tick while the phase is "touched"; the seam for a decision step.

        It runs once the supervisor's command has reached the pose the arm holds (`HOLD_WAIT_S` at most).
        A decision model replaces this method (in a subclass, or later behind a setting). While its answer
        is pending it returns `hold`, which keeps the arm on the pose it stopped in with the lag guard on;
        `obs.t - self.phase_started` gives it a timeout of its own. To try again it calls `retarget` and
        returns `hold`. This default ends the run at once with `descent_verdict`, a success when it says
        "touched" or "reached". `touch_jev` replaces it with the Jev decision model.
        """
        kind, reason = self.descent_verdict()
        return self._halt(reason, self.kin.from_q(self.q_cmd), success=kind in ("touched", "reached"))

    def descent_verdict(self) -> tuple[str, str]:
        """What the contact geometry says about the last descent: its kind and the reason for the panel.

        "touched": a contact within `contact_band_m` of the touch point, after the arm came down from the
        hover. "blocked": a stall above the band. "stuck": a stall where the arm never came down, as an arm
        that does not move gives. "missed": the touch point in free air, or "reached" when `need_contact`
        is off.
        """
        st = self.status
        if st.contact:
            above = st.stop_z_m - self.config.touch_z
            if above > self.config.contact_band_m:
                return "blocked", (f"ramie zablokowane {above * 1000:.0f} mm nad punktem dotyku "
                                   "- za wysoko na obiekt")
            if not st.drop_m >= self.config.contact_gap_m:
                return "stuck", (f"ramie nie zjechalo z zawisu ({st.drop_m * 1000:.0f} mm) "
                                 "- zablokowane albo odczyt pozycji stoi")
            return "touched", f"obiekt dotkniety {above * 1000:.0f} mm nad punktem dotyku"
        if not self.config.need_contact:
            return "reached", "punkt dotyku osiagniety"
        return "missed", "brak kontaktu w punkcie dotyku - obiektu tam nie ma"

    def hold(self, note: str = "") -> Action:
        """A tick that keeps the last target: the arm stays where it was sent, the run goes on."""
        return Action(target=self.kin.from_q(self.q_cmd), error=self._error(), note=note or self._note())

    def retarget(self, x: float, y: float) -> None:
        """Aims the same run at a new point, from a decision step; `start` is never called again.

        Raises, changing nothing, when the run is over, or when the arm cannot take the new hover or touch
        point within its limits without hitting itself. The arm first goes straight up to the hover height
        above where it is, so the hand is never dragged across the object, then on to the new hover point
        and down again, going on from its last target. The next hover arrival counts a new attempt.
        """
        if self.status.phase == DONE:
            raise RuntimeError("touch: the run is over")
        self._aim(np.array([x, y], float))
        st = self.status
        st.contact = False
        st.rise_m = st.stop_z_m = st.drop_m = float("nan")
        self._after_aim()

    @property
    def phase_started(self) -> float:
        """When the current phase began [s, `obs.t`]."""
        return self._phase_t0

    def _enter(self, phase: str) -> None:
        self.status.phase = phase
        self._phase_t0 = self._t
        self._settle_count = self._still_count = 0
        self._q_enter = self.q_cmd.copy()
        self._stalled = None

    def _arrived(self, point: np.ndarray, *, sag: bool = False, watch: bool = True) -> bool:
        """The last target puts the TCP on `point`, the supervisor's command has reached that target, and
        the measured TCP is near the point or stands still within `SAG_M` of it.

        As in `ik`, the last target must be on the point, because the twin's runner drops the target sent
        with `done`. Standing still counts too, because an arm that sags or is held up never gets closer,
        but only nearby: an arm that never moves stays where it was. With `sag`, for a point above the
        arm's way, an arm that stood still for `settle_ticks` counts within `sag_m` below the point
        (`_sags`). With `watch`, an arm that stood still for `stall_ticks` short of the point ends the run
        on this tick (`act`).
        """
        if float(np.linalg.norm(point - self.tcp_cmd)) > self.ik_tol or not self._command_on_target:
            return False
        distance = float(np.linalg.norm(self._tcp_measured - point))
        if distance <= self.config.contact_gap_m / 2 or (self._still and distance <= SAG_M):
            return True
        if sag and self._still_count >= self.settle_ticks and self._sags(point):
            return True
        if watch and self._still_count >= self.stall_ticks:
            self._stalled = np.asarray(point, float).copy()
        return False

    def _settled(self, point: np.ndarray, *, sag: bool = False, watch: bool = True) -> bool:
        """Whether the arm has been arrived at `point` (`_arrived`) for `settle_ticks` ticks in a row."""
        self._settle_count = self._settle_count + 1 if self._arrived(point, sag=sag, watch=watch) else 0
        return self._settle_count >= self.settle_ticks

    def _sags(self, point: np.ndarray) -> bool:
        """Whether the measured TCP stands below `point` as a sagging arm does: at most `sag_m` away, and
        within `SAG_CONE_DEG` of straight down."""
        error = self._tcp_measured - point
        distance = float(np.linalg.norm(error))
        return (distance <= self.config.sag_m
                and -float(error[2]) >= distance * float(np.cos(np.radians(SAG_CONE_DEG))))

    def _take_reference(self) -> None:
        """Take the stall reference of a descent that starts where the arm stands now."""
        self._bias_z = self._offset_z
        self._hover_z = float(self._tcp_measured[2])
        self._hang_q, self._hang_command_q = self._q_measured.copy(), self._q_command.copy()

    def _contact_rise(self) -> float:
        """How far the measured TCP stands above where the arm should be in this descent [m].

        A joint that stood off its command at the reference stays where it was until its command comes
        back to it, as friction holds a servo that settled short, and then follows the command. A joint
        whose command moves away from where it stood keeps its offset from the command.
        """
        hang, command = self._hang_q, self._q_command
        offset = hang - self._hang_command_q
        expected = np.where(offset >= 0, np.maximum(command, np.minimum(hang, command + offset)),
                            np.minimum(command, np.maximum(hang, command + offset)))
        z = self.kin.tcp(self.kin.from_q(expected))[2, 3]
        return float(self._tcp_measured[2] - z)

    def _stall_reason(self, point: np.ndarray) -> str:
        """Why the run ends on an arm that stands still short of `point`, for the panel: the phase, how far
        and where the TCP stands from the point, and the joints that lag their command [deg]."""
        kin = self.kin
        error = self._tcp_measured - point
        distance = float(np.linalg.norm(error))
        cone = distance * float(np.cos(np.radians(SAG_CONE_DEG)))
        where = "pod celem" if -error[2] >= cone else "nad celem" if error[2] >= cone else "od celu"
        lag = np.degrees(self._q_measured - self._q_command)
        move = self._q_command - self._q_enter
        groups: dict[str, list[str]] = {}
        for k in sorted(np.flatnonzero(kin.arm), key=lambda k: -abs(lag[k])):
            if abs(lag[k]) < LAG_NOTE_DEG:
                continue
            # Short of the command along this phase's move, or past it.
            word = "za malo" if lag[k] * move[k] < 0 else "za daleko" if lag[k] * move[k] > 0 else "od celu"
            groups.setdefault(word, []).append(f"{kin.spec.joints[k]} {abs(lag[k]):.1f} st.")
        joints = "; ".join(f"{', '.join(names)} {word}" for word, names in groups.items())
        name = self._phase_name()
        return (f"{name}: ramie stoi {distance * 1000:.0f} mm {where}" + (f" ({joints})" if joints else "")
                + " - koniec")

    def _phase_name(self) -> str:
        """The current phase's name for the panel."""
        return self.phase_names.get(self.status.phase, self.status.phase)

    def _drive(self, goal: np.ndarray, step_m: float, *, rotation: np.ndarray | None = None) -> Action:
        """Step toward `goal` [m, base frame] with the gripper unchanged.

        End the run if the target would cause self-contact. An optional `rotation` is a 3x3 TCP
        rotation in the base frame. It advances by at most `turn_step` per tick. IK treats
        orientation as a soft goal, so the caller must check the resulting orientation.
        `touch` uses position alone."""
        st = self.status
        kin, arm = self.kin, self.kin.arm
        # Unlike `ik`, the carrot goes on from the last carrot when the last target reached it: IK's
        # residual then never adds up, and the path stays on the straight line. From the last target's TCP
        # when the arm could not follow, as in `ik`.
        reached = float(np.linalg.norm(self.tcp_cmd - self.carrot)) <= self.ik_tol
        base = self.carrot if reached else self.tcp_cmd
        to_goal = goal - base
        d = float(np.linalg.norm(to_goal))
        step, seed, tries = min(step_m, d), kin.from_q(self.q_cmd), 1
        if rotation is not None:
            rotation = turn_toward(kin.tcp(seed)[:3, :3], rotation, self.turn_step)
        # The supervisor's speed limits: a step that would move an arm joint farther than `_max_step` in one
        # tick is shortened, and after three tries the joints' move is scaled down.
        while True:
            carrot = goal.copy() if step >= d else base + to_goal * (step / d)
            q_new = kin.to_q(kin.ik_step(carrot, rotation, seed=seed, tol=self.ik_tol).joints)
            q_new[~arm] = self.q_cmd[~arm]                 # the gripper keeps its target
            q_new = np.clip(q_new, self._lo, self._hi)
            over = float(np.max(np.abs(q_new - self.q_cmd)[arm] / self._max_step[arm]))
            if over <= 1.0:
                break
            if tries == 3:
                q_new[arm] = self.q_cmd[arm] + (q_new - self.q_cmd)[arm] / over
                carrot = kin.tcp(kin.from_q(q_new))[:3, 3].copy()
                break
            step, tries = step / over, tries + 1
        self.carrot = carrot
        target = kin.from_q(q_new)
        contacts = kin.self_contacts(target)
        # Only new pairs count: a run may start from a folded pose and move out of it.
        new = contacts - self._contacts
        if new:
            pairs = ", ".join(f"{a} - {b}" for a, b in sorted(new))
            return self._halt(f"ramie uderzyloby samo w siebie ({pairs})", target=kin.from_q(self.q_cmd))
        self.q_cmd, self._contacts = q_new, contacts
        self.tcp_cmd = kin.tcp(target)[:3, 3].copy()
        st.out_of_reach = bool(np.linalg.norm(self.tcp_cmd - self.carrot) > self.ik_tol)
        st.distance = float(np.linalg.norm(goal - self._tcp_measured))
        return Action(target=target, error=self._error(), note=self._note())

    def _hold_pose(self, measured: Mapping[str, float]) -> dict[str, float]:
        """The arm joints where they are, the gripper as it was: the pose the arm holds after a stall."""
        spec = self.kin.spec
        pose = {name: float(measured[name]) for name in spec.arm_joints if name in measured}
        if spec.gripper is not None:
            pose[spec.gripper] = self.kin.from_q(self.q_cmd)[spec.gripper]
        return pose

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, target: Mapping[str, float], success: bool = False) -> Action:
        """The run is over: the last target and the reason for the twin's panel."""
        self._end(reason, success)
        return Action(target=dict(target), done=True, success=success, error=self._error(), note=reason)

    def _end(self, reason: str, success: bool = False) -> None:
        self.status.phase = DONE
        self.status.success = success
        self.status.stopped_because = reason
        logger.info("Dotyk zatrzymany: %s", reason)

    def stop(self, reason: str) -> None:
        """The twin ended the run (Stop, STOP, Home, the arm taken, the time limit, or our own end)."""
        st = self.status
        st.phase = DONE
        # After our own end the twin reports "zadanie wykonane", "strategia skonczyla", or the lag guard's
        # exception as "RuntimeError: <our reason>"; ours says as much or more.
        if st.stopped_because and (reason in ("zadanie wykonane", "strategia skonczyla")
                                   or reason.endswith(st.stopped_because)):
            return
        st.stopped_because = reason or "zatrzymana"

    def _error(self) -> float | None:
        d = self.status.distance
        return float(d) if np.isfinite(d) else None

    def _note(self) -> str:
        st = self.status
        mm = st.distance * 1000
        if st.phase == LIFT:
            text = f"w gore przed nowym celem: {mm:.0f} mm"
        elif st.phase == APPROACH:
            text = f"nad obiekt: {mm:.0f} mm"
        elif st.phase == DESCEND:
            text = f"w dol: {mm:.0f} mm do punktu dotyku"
        elif st.phase == TOUCHED:
            text = "dotyk: oparty o przeszkode" if st.contact else "dotyk: w punkcie dotyku, bez kontaktu"
        else:
            text = st.stopped_because or st.phase
        return text + (" - cel poza zasiegiem ramienia" if st.out_of_reach else "")


def normalised(label: str) -> str:
    """A label as the twin keys `obs.objects`: lower case, white space collapsed."""
    return " ".join(label.split()).lower()
