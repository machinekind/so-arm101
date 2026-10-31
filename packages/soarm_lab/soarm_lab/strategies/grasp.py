"""Grasping a block as a strategy for the twin: `grasp` under the `machinekind_twin.strategies` entry points.

`GraspStrategy` is `touch` with a pick-up after its probe. The twin builds `GraspSettings` from its config
file's `grasp:` section or the panel's form and runs the strategy through its `StrategyRunner`. Nothing here
imports the twin.

    # configs/local.yaml of the twin: prefills the panel's form
    grasp:
      object_kind: cube            # cube, cuboid5, cuboid85 or cylinder: the block's size [rl.objects]
      label: ""                    # the block's label in obs.objects; empty = object_kind
      touch_z: 0.0                 # the TCP height at the probe's bottom [m]; 0 = from the block
      hover_m: 0.04                # the hover lies this much above the probe's bottom [m]
      open_margin_m: 0.008         # each open tip clears the block's short side by this much [m]
      squeeze_m: 0.008             # the close is commanded this much tighter than the short side [m]
      grasp_depth: 0.5             # how far the tips go below the block's top, a share of its height
      lift_m: 0.04                 # how far the block is lifted [m]
      hold_s: 1.0                  # how long the jaw must hold at the top [s]
      max_attempts: 3              # probe descents in one run
      time_limit_s: 120.0          # a safety net for the whole run [s]

The block's dimensions come from `rl.objects.object_spec(object_kind)`, never from the image: the short
side (25 mm on every block of the set) sets the openings, the height the grasp depth. The jaw opening in
metres, the fingertips' midpoint and the jaw's sole come from the MJCF through `RobotKinematics.jaw_gap`,
`gripper_for_gap`, `jaw_centre` and `jaw_reach`. The strategy plans for the JAW POINT, the fingertips'
midpoint, and turns it into the TCP point for the gripper command a move is made with.

Method. The pose is latched as in `touch`, and from then on every move asks for a top-down hand: the
approach straight down, the jaws closing across the block's short side (`grasp_headings`: the pose's yaw
folded by the block's symmetry, the heading nearest the current one first; with no rotation in the pose
the jaws keep their current heading). The probe is `touch`'s: the closed tips over the block centre, down
from the hover until the arm stalls; the TCP at the stall minus `jaw_reach` is the block's measured top.
`touch_z` 0 puts the probe's bottom 10 mm under the block's top plus `jaw_reach`, so the hand stalls on
the top before the TCP gets there. Then:

1. rise straight up to the hover with the jaw still closed;
2. move sideways so the open jaw will be centred on the block, opening it to the short side plus twice
   `open_margin_m` on the way (41 mm on the 25 mm blocks);
3. go straight down to the grasp height, the tips' centres `grasp_depth` of the block's height below its
   measured top;
4. close to the short side minus `squeeze_m` and wait until the jaw has left its opening and stands;
5. the jaw rule (`jaw.JawMonitor`) says held: lift `lift_m` straight up, and hold there for `hold_s`.

The gripper command ramps at `SPEED_SHARE` of the supervisor's gripper speed limit, 240 units/s on the
SO-101, about 10 mm of gap a tick: the open takes 4 ticks, the close 3, and the servo follows later.

Guards (touch's, and):

* **the jaw rule**: the close counts as held only after the measured jaw dropped by `jaw.margin` below its
  opening (a jaw that never moved stands still at its opening and squeezes against a tighter command by
  the rule alone) and then stood still for `SETTLE_S` with the command on it; held means it stands wider
  than commanded and than closed by more than the margin. On the way up and at the top the jaw must keep
  squeezing and must not creep closed by more than `jaw.creep` since the close: either ends the run;
* **openings the jaw can make**: `start` refuses a run whose opening or squeeze lies outside the jaw's
  travel, a squeeze within the jaw margin (a held block would not read held) or an opening within it (a
  jaw closing on the block could not be told from one that never moved);
* **clear before close**: after a jaw tip landed on the block, the jaw stays open while the arm rises and
  closes only 10 mm higher, so the moving tip does not sweep across the block's top;
* **reach with orientation**: when the pose is taken, the full multi-start IK must reach the probe hover,
  the probe's bottom, the straddle hover, the grasp point and the lift point with the hand within 5
  degrees of top-down, within the limits and off the arm's own body, for one of the candidate headings.
  On the SO-101 that holds for a cube's centre in a band: x 0.14-0.27 m straight ahead with the 4 cm hover
  (0.18-0.24 with touch's 6 cm), 0.12-0.25 m at 10 cm to the side. Beyond it wrist_flex stops at its 95
  degree limit, and the run ends with "chwytak nie siega tam pionowo". The hand turns toward top-down at
  touch's `TURN_DEG_S`, and the target is checked to be top-down within 8 degrees before the probe and
  before the straddle;
* **the attempt budget**: `max_attempts` probe descents in one run.

A jaw tip that stalls on the block during the straddle holds the measured pose until the supervisor's
command reaches it, rises (closing the jaw once clear) and probes again half the short side along the
closing heading (`SHIFTS`): toward the moving jaw first. When that probe misses, the block lies on the
other side, and the next probe is half the short side the other way from the original point. A close that
finds nothing ends the run ("chwyt pusty"): after a straddle that met nothing the block was within the
opening, and the jaw alone cannot say whether the probe pushed it away or it sits at the jaws' edge.

Success: the jaw holds for `hold_s` at the lift height, "blok w szczekach, podniesiony o N mm". The block
stays held after the run: the last target squeezes, the twin's runner sends the done tick's target before
it disengages, and disengaging keeps the supervisor's command; a later STOP or Home keeps the squeeze
through the twin's `hold_measured`. The twin's tracker says "w dloni" from the close on, with the same jaw
thresholds (`ObjectTracker.for_spec`).

Phases (`TouchStatus.phase`): touch's wait, lift, approach, descend and touched, then rise, open, straddle
(straddled after a tip stall), close, lift_off and check, then done. Seams for later steps: `after_miss`
(a probe that met nothing), `before_descent` (every tick of a descent until it returns None once: wrist
centring), `before_straddle` and `_looking` (at the straddle hover), `after_grasp_failed` (a tip on the
block or an empty close), `carrying` (every tick from the hold verdict on: recovery), `grasp_headings` (the
candidate headings: the grasp library) and `grasp_verdict` (for logs and a decision step). `after_touch`
stays the seam of `touch`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from ..kinematics import IKSolution, RobotKinematics
from ..rl.objects import KINDS, ObjectSpec, object_spec
from .jaw import JawMonitor, JawReading
from .touch import (
    DESCEND,
    HOLD_WAIT_S,
    MIN_TOUCH_Z,
    SPEED_SHARE,
    WAIT,
    TouchSettings,
    TouchStrategy,
    normalised,
)

logger = logging.getLogger(__name__)

#: The probe's bottom lies this far under the block's top plus the jaw's reach, so the hand stalls on the
#: top before the TCP gets there [m]. The stall is read at 4 mm of rise (`contact_gap_m`), so this bounds
#: nothing the arm presses; it must exceed the rise plus what gives way. At 5 mm a probe on which only the
#: moving jaw's tip landed (it reaches 2 mm less deep than the fixed jaw's sole) went to the bottom in the
#: twin without a stall, the block giving 2 mm under it.
PROBE_BELOW_M = 0.010
#: The lowest TCP height of the grasp point [m]: the jaw's sole stays above the table.
MIN_GRASP_Z = 0.012
#: How long the jaw may take to open, to start closing and to stand after the close [s].
JAW_WAIT_S = 3.0
#: The supervisor's gripper command is on the target within this many app units.
JAW_COMMAND_TOL = 0.5
#: The reach check's largest orientation error of a top-down IK solution [deg].
MAX_ROT_ERR_DEG = 5.0
#: The hand counts as top-down when its approach is this close to vertical and its closing heading this
#: close to the chosen one [deg].
MAX_TILT_DEG = 8.0
#: After a jaw tip landed on the block, the jaw closes only once the TCP rose this much [m].
CLEAR_BEFORE_CLOSE_M = 0.010
#: The SO-101's `jaw_reach(0)` [m]: the probe height before `start` knows the model, replaced there.
JAW_REACH_M = 0.0066
#: The sideways retargets after a jaw tip landed on the block, in half short sides along the closing
#: heading from the first aim (+ toward the moving jaw).
SHIFTS = (1, -1, 2, -2)

#: Phases of a run after touch's (`TouchStatus.phase`).
RISE = "rise"
OPEN = "open"
STRADDLE = "straddle"
STRADDLED = "straddled"
CLOSE = "close"
LIFT_OFF = "lift_off"
CHECK = "check"
#: Where the rise leads: to the open, or to a retarget.
RETARGET = "retarget"

#: What `grasp_verdict` says, per verdict.
VERDICTS = {
    "none": "brak werdyktu chwytu",
    "tip": "szczeka oparla sie o blok przy zjezdzie",
    "empty": "chwyt pusty",
    "held": "szczeka trzyma blok",
    "slipped": "blok wypadl ze szczeki",
    "crept": "blok zsuwa sie w szczekach",
    "lifted": "blok podniesiony",
}


@dataclass
class GraspSettings(TouchSettings):
    #: The block's key in `obs.objects`, compared after lower-casing and collapsing white space; empty = the
    #: same as `object_kind`.
    label: str = ""
    #: The TCP height at the bottom of the probe [m above the table, base frame]; 0 = from the block: 10 mm
    #: under its top plus the jaw's reach below the TCP. A value of its own must be at least 0.02 m.
    touch_z: float = 0.0
    #: The hover lies this much above the probe's bottom [m]. Lower than touch's: on the SO-101 a top-down
    #: hand reaches a block straight ahead at x 0.14-0.27 m with this hover, only at 0.18-0.24 with 6 cm.
    hover_m: float = 0.04
    #: A safety net for the whole run [s]; 0 means none.
    time_limit_s: float = 120.0
    #: Which wooden block (`rl.objects.KINDS`): its size sets the openings and the grasp height.
    object_kind: str = field(default="cube", metadata={"choices": KINDS})
    #: Each open tip clears the block's short side by this much when the jaw straddles it [m].
    open_margin_m: float = 0.008
    #: The close is commanded this much tighter than the block's short side [m]; it must exceed the jaw
    #: margin's gap (about 5 mm on the SO-101) for a held block to read held.
    squeeze_m: float = 0.008
    #: How far below the block's measured top the tips' centres go before the close, as a share of the
    #: block's height [0..1].
    grasp_depth: float = 0.5
    #: How far straight up the block is lifted before the final check [m].
    lift_m: float = 0.04
    #: How long the jaw must hold at the top before the run succeeds [s].
    hold_s: float = 1.0
    #: Probe descents in one run: the first and the retries after a jaw tip landed on the block [count].
    max_attempts: int = 3


@dataclass
class GraspStatus:
    """What the strategy knows about the grasp, beside touch's `TouchStatus`, for logs and tests."""

    #: The block centre the run aims at [m, base frame]; None until the pose is taken.
    block_xy: tuple[float, float] | None = None
    #: The closing heading of the top-down hand [deg, base frame].
    heading_deg: float = float("nan")
    #: The block's top measured by the probe [m].
    top_m: float = float("nan")
    #: The TCP point of the close [m, base frame].
    grasp_point: tuple[float, float, float] | None = None
    #: The gap of the commanded and of the measured jaw, between the tip centres [m].
    gap_cmd_m: float = float("nan")
    jaw_gap_m: float = float("nan")
    #: The jaw rule on the last tick.
    holding: bool = False
    #: Ticks in a row the jaw held at the lift height.
    hold_ticks: int = 0
    straddle_stalls: int = 0
    empty_closes: int = 0
    #: How far the measured TCP rose above the grasp point [m].
    lifted_m: float = float("nan")
    #: '', 'tip', 'empty', 'held', 'slipped', 'crept' or 'lifted'.
    verdict: str = ""


class GraspStrategy(TouchStrategy):
    """Grasps and lifts a block after `touch`'s probe; the jaw rule checks the hold on the way up."""

    settings = GraspSettings

    def __init__(self, settings: GraspSettings | None = None):
        config = settings if settings is not None else GraspSettings()
        kind = config.object_kind
        if kind not in KINDS:
            raise ValueError(f"grasp: object_kind must be one of {', '.join(KINDS)}, is {kind!r}")
        # Before touch's checks, so the operator reads grasp's texts.
        if not (config.touch_z == 0 or config.touch_z >= MIN_TOUCH_Z):
            raise ValueError(f"grasp: touch_z must be zero (from the block) or at least {MIN_TOUCH_Z} m, "
                             f"is {config.touch_z}")
        for name in ("open_margin_m", "squeeze_m", "lift_m", "hold_s"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"grasp: {name} must be positive, is {value}")
        if not 0 < config.grasp_depth <= 1:
            raise ValueError(f"grasp: grasp_depth must lie within (0, 1], is {config.grasp_depth}")
        if not config.max_attempts >= 1:
            raise ValueError(f"grasp: max_attempts must be at least 1, is {config.max_attempts}")
        if not config.close_jaw:
            raise ValueError("grasp: close_jaw must be on: the probe needs closed tips")
        block = object_spec(kind)
        #: The block as the simulation builds it: half sizes (x long, y short, z height), yaw symmetry.
        self.block: ObjectSpec = block
        self.short_m = 2 * block.half_size[1]
        self.height_m = 2 * block.half_size[2]
        #: The block's top when it lies on the table [m]; the probe measures the real one.
        self.top_nominal = self.height_m
        #: The jaw's reach below the TCP [m], from the model once `start` has it.
        self._reach_m = JAW_REACH_M
        #: The operator's settings; `config` holds the probe height and label in use.
        self.grasp_settings = config
        self._auto_touch_z = config.touch_z == 0
        super().__init__(replace(config, touch_z=self._probe_z(), label=normalised(config.label) or kind))

        self.grasp = GraspStatus()
        #: The rotation of the block's pose in the base frame, identity until a pose is taken.
        self.block_rotation = np.eye(3)
        #: The top-down hand of the run and its closing heading [rad]; None / NaN before the pose.
        self._rot: np.ndarray | None = None
        self._psi = float("nan")
        #: The candidate headings `_aim` tries, in order [rad].
        self._headings: list[float] = []
        #: The first block centre aimed at, and how many sideways retargets were taken from it.
        self._block_xy0: np.ndarray | None = None
        self._shift_k = 0
        self._retarget_xy: np.ndarray | None = None
        # The jaw: the monitor, this tick's reading, the gripper index and its ramp [rad].
        self._jaw: JawMonitor | None = None
        self._reading: JawReading | None = None
        self._g = -1
        self._jaw_goal_q = float("nan")
        self._grip_step_q = float("inf")
        self._jaw_tol_q = 0.0
        self._jaw_on_command = True
        #: The openings [m] and their gripper commands [app units], the commands known from `start`.
        self.gap_open_m = self.short_m + 2 * config.open_margin_m
        self.gap_closed_m = self.short_m - config.squeeze_m
        self.cmd_open = self.cmd_closed = self.cmd_block = self.cmd_probe = float("nan")
        #: The fingertips' midpoint in the TCP frame for the probe's and the open jaw [m].
        self._centre_probe = np.zeros(3)
        self._centre_open = np.zeros(3)
        # The rise: where it leads, where it started and goes to.
        self._rise_then = OPEN
        self._rise_z0 = float("nan")
        self._rise_point = np.zeros(3)
        # The grasp: the TCP x and y of the straddle, the TCP height of the close and of the lift [m].
        self._straddle_xy = np.full(2, np.nan)
        self._grasp_z = float("nan")
        self._lift_off_point = np.zeros(3)
        # The close: the jaw angle when it began [rad], whether the jaw left it, ticks it stood since.
        self._q_open_entry = float("nan")
        self._jaw_moved = False
        self._still_ticks = 0
        #: When the hover was reached before the hand turned top-down [s, `obs.t`].
        self._turn_t0 = float("nan")
        # The seams of the hovers: asked until they return None once per descent and per straddle.
        self._before_descent_pending = False
        self._before_straddle_pending = False
        self._descent_waited = False
        self._phases.update({RISE: self._rise, OPEN: self._open, STRADDLE: self._straddle,
                             STRADDLED: self._straddled, CLOSE: self._close, LIFT_OFF: self._lift_off,
                             CHECK: self._check})

    def _probe_z(self) -> float:
        """The TCP height at the probe's bottom [m]: the operator's, or `PROBE_BELOW_M` under the block's top
        plus the jaw's reach, so the hand stalls on the top before the TCP gets there."""
        if not self._auto_touch_z:
            return float(self.grasp_settings.touch_z)
        return max(MIN_TOUCH_Z, self.top_nominal - PROBE_BELOW_M + self._reach_m)

    # ------------------------------------------------------------------- start
    def start(self, run: RunInfo, obs: Observation) -> None:
        """touch's start, then the jaw: the openings as gripper commands, refused before the arm moves when
        the jaw cannot make them or the jaw rule could not read them."""
        super().start(run, obs)
        kin, spec, cfg = self.kin, self.kin.spec, self.grasp_settings
        if spec.gripper is None or spec.gripper_range_q is None:
            raise RuntimeError("grasp: the arm has no gripper")
        g = self._g = spec.joints.index(spec.gripper)
        closed, opened = spec.gripper_range_q
        try:
            self.cmd_probe = kin.from_q(self._lo)[spec.gripper]
            self._reach_m = kin.jaw_reach(self.cmd_probe)
            self.cmd_open = kin.gripper_for_gap(self.gap_open_m)
            self.cmd_closed = kin.gripper_for_gap(self.gap_closed_m)
            self.cmd_block = kin.gripper_for_gap(self.short_m)
        except ValueError as exc:
            raise RuntimeError(f"grasp: {exc}") from exc
        if self._auto_touch_z and abs(self._reach_m - JAW_REACH_M) > 1e-4:
            self.config = replace(self.config, touch_z=self._probe_z())
        self._centre_probe = kin.jaw_centre(self.cmd_probe)
        self._centre_open = kin.jaw_centre(self.cmd_open)

        jaw = spec.jaw
        margin_units = jaw.margin * 100.0 / (opened - closed)
        if self._q(self.cmd_block) - self._q(self.cmd_closed) <= jaw.margin + jaw.still:
            margin_gap = kin.jaw_gap(self.cmd_block) - kin.jaw_gap(max(0.0, self.cmd_block - margin_units))
            raise RuntimeError(f"grasp: squeeze_m {cfg.squeeze_m} m is within the jaw margin "
                               f"({margin_gap:.4f} m): a held block would not count as held")
        if self._q(self.cmd_open) - self._q(self.cmd_block) <= jaw.margin + jaw.still:
            raise RuntimeError(f"grasp: open_margin_m {cfg.open_margin_m} m is within the jaw margin: a jaw "
                               "closing on the block could not be told from one that never moved")

        # The hardware's closed limit, as the twin's hand has it.
        self._jaw = JawMonitor(jaw, closed_q=float(self._lo[g]))
        self._reading = None
        # The gripper takes the same share of the supervisor's limit as the arm joints: the twin's 50 Hz
        # loop needs that slack to follow each tick.
        vel = float(run.max_vel.get(spec.gripper, spec.max_vel.get(spec.gripper, np.inf)))
        self._grip_step_q = (vel * SPEED_SHARE / self.hz * (opened - closed) / 100.0 if vel > 0
                             else float("inf"))
        self._jaw_tol_q = JAW_COMMAND_TOL * (opened - closed) / 100.0
        self._jaw_goal_q = float(self.q_cmd[g])
        self._jaw_on_command = True

        self.grasp = GraspStatus()
        self.block_rotation = np.eye(3)
        self._rot, self._psi, self._headings = None, float("nan"), []
        self._block_xy0, self._shift_k, self._retarget_xy = None, 0, None
        self._straddle_xy, self._grasp_z = np.full(2, np.nan), float("nan")
        self._turn_t0 = float("nan")
        self._before_descent_pending = self._before_straddle_pending = self._descent_waited = False

    def _q(self, gripper: float) -> float:
        """A gripper command [app units] as the gripper joint's angle [rad]."""
        spec = self.kin.spec
        assert spec.gripper is not None
        return float(self.kin.to_q({spec.gripper: gripper})[self._g])

    # --------------------------------------------------------------------- tick
    def act(self, obs: Observation) -> Action:
        """One tick: the jaw rule on this tick's jaw, the headings when the pose is taken, touch's tick."""
        if self._kin is None or self._jaw is None:
            raise RuntimeError("grasp: act() przed start()")
        kin, g = self.kin, self._g
        q_measured = kin.to_q(obs.joints)
        # The supervisor's command, the last target for joints it leaves out, as touch's lag guard has it.
        q_command = kin.to_q({**kin.from_q(self.q_cmd), **obs.command})
        self._reading = self._jaw.update(float(q_measured[g]), float(q_command[g]))
        self._jaw_on_command = abs(float(q_command[g] - self.q_cmd[g])) <= self._jaw_tol_q
        self.grasp.holding = self._reading.holding
        fixed, moving = kin.fingertips(obs.joints)
        self.grasp.jaw_gap_m = float(np.linalg.norm(moving - fixed))
        if self.status.phase == WAIT:
            self._take_heading(obs)
        return super().act(obs)

    @property
    def reading(self) -> JawReading:
        """This tick's jaw rule."""
        if self._reading is None:
            raise RuntimeError("grasp: no jaw reading before the first tick")
        return self._reading

    def _take_heading(self, obs: Observation) -> None:
        """The candidate headings from the pose `_wait` is about to take, on the same tick."""
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        if pose is None or self._pose_problem(pose):
            return
        rotation = np.asarray(pose.rotation, float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            rotation = np.eye(3)
        self._heading_from(rotation)

    def _heading_from(self, rotation: np.ndarray) -> None:
        """Takes `rotation` as the block's and lists the headings `_aim` tries, the current one's first."""
        self.block_rotation = np.asarray(rotation, float).copy()
        self._headings = grasp_headings(self.block_rotation, self.block.yaw_period, self._heading_now())

    def _heading_now(self) -> float:
        """The closing heading of the last target [rad, base frame]."""
        return heading(self.kin, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------- aiming
    def _aim(self, xy: np.ndarray) -> None:
        """Aims the run at the block centre `xy`: the first candidate heading whose five points the arm
        reaches top-down. Raises, changing nothing, when none does."""
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise RuntimeError("punkt poza zasiegiem ramienia: wspolrzedne nieskonczone")
        headings = self._headings or grasp_headings(self.block_rotation, self.block.yaw_period,
                                                    self._heading_now())
        top = self.grasp.top_m if np.isfinite(self.grasp.top_m) else self.top_nominal
        grasp_z = self._grasp_height(top)
        first_failure: tuple[np.ndarray, dict[str, float], IKSolution] | None = None
        for psi in headings:
            plan = self._plan(xy, psi, grasp_z)
            failure = self._reach(plan, psi)
            if failure is None:
                self._take_plan(xy, psi, plan, grasp_z)
                return
            first_failure = first_failure or failure
        assert first_failure is not None
        raise RuntimeError(self._unreachable(*first_failure))

    def _plan(self, xy: np.ndarray, psi: float, grasp_z: float) -> dict[str, np.ndarray]:
        """The TCP points of one heading: the jaw point over the block centre, as the TCP sees it."""
        u = np.array([math.cos(psi), math.sin(psi)])
        hover_z = self.config.touch_z + self.config.hover_m
        # The jaw point J = TCP + R @ centre, with R top-down: horizontally centre[2] along the heading,
        # vertically centre[0] down.
        probe_xy = xy - self._centre_probe[2] * u
        straddle_xy = xy - self._centre_open[2] * u
        return {"probe_xy": probe_xy, "straddle_xy": straddle_xy,
                "probe_hover": np.array([*probe_xy, hover_z]),
                "probe_bottom": np.array([*probe_xy, self.config.touch_z]),
                "straddle_hover": np.array([*straddle_xy, hover_z]),
                "grasp": np.array([*straddle_xy, grasp_z]),
                "lift": np.array([*straddle_xy, grasp_z + self.grasp_settings.lift_m])}

    def _reach(self, plan: dict[str, np.ndarray], psi: float
               ) -> tuple[np.ndarray, dict[str, float], IKSolution] | None:
        """None when the full IK reaches every point of `plan` top-down; else the first point that fails,
        the seed it was tried from and the solution."""
        kin, R = self.kin, top_down(psi)
        spec = kin.spec
        assert spec.gripper is not None
        seed = kin.from_q(self.q_cmd)
        for name in ("probe_hover", "probe_bottom", "straddle_hover", "grasp", "lift"):
            if name == "straddle_hover":
                seed = dict(seed, **{spec.gripper: self.cmd_open})
            point = plan[name]
            sol = kin.ik(point, R, seed=seed, accept=self._clear)
            if not (sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)):
                return point, seed, sol
            seed = sol.joints
        return None

    def _unreachable(self, point: np.ndarray, seed: dict[str, float], sol: IKSolution) -> str:
        """Why the arm cannot take `point` top-down, for the panel."""
        where = f"punkt ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"
        if sol.ok and self._clear(sol.joints):
            return f"{where}: chwytak nie ustawi sie pionowo (blad {math.degrees(sol.rot_err):.0f} st.)"
        reachable = self.kin.ik(point, seed=seed, accept=self._within_limits)
        if not (reachable.ok and self._within_limits(reachable.joints)):
            return f"{where} poza zasiegiem ramienia"
        if not self.kin.self_contacts(reachable.joints):
            return f"{where}: chwytak nie siega tam pionowo"
        return f"{where}: ramie uderzyloby samo w siebie"

    def _take_plan(self, xy: np.ndarray, psi: float, plan: dict[str, np.ndarray], grasp_z: float) -> None:
        self._psi, self._rot = float(psi), top_down(psi)
        self._headings = [float(psi)]
        self.grasp.heading_deg = math.degrees(psi)
        self.touch_xy = plan["probe_xy"].copy()
        x, y = (float(v) for v in self.touch_xy)
        self.status.touch_point = (x, y, float(self.config.touch_z))
        self.grasp.block_xy = (float(xy[0]), float(xy[1]))
        if self._block_xy0 is None:
            self._block_xy0 = xy.copy()
        self._straddle_xy = plan["straddle_xy"].copy()
        self._grasp_z = float(grasp_z)
        self.grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)

    def _grasp_height(self, top_m: float) -> float:
        """The TCP height of the close [m]: the tips' centres `grasp_depth` of the block's height below its
        top, the jaw's sole above `MIN_GRASP_Z`."""
        depth = self.grasp_settings.grasp_depth * self.height_m
        return max(MIN_GRASP_Z, top_m - depth + float(self._centre_open[0]))

    @property
    def straddle_hover(self) -> np.ndarray:
        """Where the open jaw waits above the block before the straddle [m, base frame]."""
        return np.array([*self._straddle_xy, self.config.touch_z + self.config.hover_m])

    @property
    def grasp_target(self) -> np.ndarray:
        """Where the TCP closes the jaw [m, base frame]."""
        return np.array([*self._straddle_xy, self._grasp_z])

    def _drive(self, goal: np.ndarray, step_m: float, *, rotation: np.ndarray | None = None) -> Action:
        """touch's step, asking for the top-down hand once the pose is taken."""
        return super()._drive(goal, step_m, rotation=self._rot if rotation is None else rotation)

    def _vertical(self) -> bool:
        """Whether the last target's hand is top-down on the chosen heading within `MAX_TILT_DEG`."""
        joints = self.kin.from_q(self.q_cmd)
        limit = math.radians(MAX_TILT_DEG)
        return tilt(self.kin, joints) <= limit and abs(wrap(heading(self.kin, joints) - self._psi)) <= limit

    def _enter(self, phase: str) -> None:
        super()._enter(phase)
        if phase == DESCEND:
            self._before_descent_pending, self._descent_waited = True, False
        if phase == OPEN:
            self._before_straddle_pending = True

    # ------------------------------------------------------------------ the probe
    def _approach(self, obs: Observation) -> Action | None:
        """touch's approach, but the descent starts only with the hand top-down."""
        if self._rot is None or self._vertical():
            self._turn_t0 = float("nan")
            return super()._approach(obs)
        # The hover reached before the hand has turned: it turns on the spot, for `JAW_WAIT_S` at most.
        if self._arrived(self.hover_point):
            if not np.isfinite(self._turn_t0):
                self._turn_t0 = self._t
            elif self._t - self._turn_t0 >= JAW_WAIT_S:
                return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
        self._settle_count = 0
        return self._drive(self.hover_point, self.approach_step_m)

    def _not_vertical(self) -> str:
        return (f"chwytak nie ustawil sie pionowo nad blokiem "
                f"({math.degrees(tilt(self.kin, self.kin.from_q(self.q_cmd))):.0f} st.)")

    def _descend(self, obs: Observation) -> Action | None:
        if self._before_descent_pending:
            action = self.before_descent(obs)
            if action is not None:
                self._descent_waited = True
                return action
            self._before_descent_pending = False
            if self._descent_waited:
                # The seam held or moved the arm at the hover: the descent's reference is taken again.
                self._bias_z, self._hover_z = self._offset_z, float(self._tcp_measured[2])
                self._settle_count = 0
        return super()._descend(obs)

    def before_descent(self, obs: Observation) -> Action | None:
        """Called on every tick of a descent until it returns None once (then not again for that descent).

        It may hold or drive the TCP at the hover height and re-aim through `_aim`; it returns None only
        with the arm settled at the current hover point. The default returns None. Wrist-camera centring
        looks here.
        """
        return None

    def after_touch(self, obs: Observation) -> Action:
        """The probe's verdict: a touch measures the block's top and leads to the grasp; anything else goes
        to `after_miss`. A decision step may override it and call this or `after_miss` after its answer."""
        kind, reason = self.descent_verdict()
        if kind not in ("touched", "reached"):
            return self.after_miss(kind, reason, obs)
        # The jaw's sole rests on the block: the TCP at the stall minus the sole's reach.
        top = self.status.stop_z_m - self._reach_m if kind == "touched" else self.top_nominal
        self.grasp.top_m = float(top)
        self._grasp_z = self._grasp_height(top)
        self.grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)
        self._start_rise(OPEN)
        return self.hold("w gore po dotyku")

    def after_miss(self, kind: str, reason: str, obs: Observation) -> Action:
        """A probe that did not touch: "missed", "blocked" or "stuck" with touch's reason.

        After a sideways retarget a miss means the shift went the wrong way (the first probe found the
        block), so the next one goes to the other side while attempts remain. Otherwise the run ends with
        touch's reason.
        """
        attempt = self.status.attempt
        if kind == "missed" and self._shift_k >= 1:
            if attempt < self.grasp_settings.max_attempts and self._next_shift() is not None:
                self._start_rise(RETARGET)
                return self.hold("obok pusto - probuje z drugiej strony")
            return self._halt(f"wyczerpane proby chwytu ({attempt}): {reason}", self.kin.from_q(self.q_cmd))
        return self._halt(reason, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------ the grasp
    def _start_rise(self, then: str) -> None:
        """Straight up to the hover height from the held pose; then `then` (OPEN or RETARGET)."""
        self._rise_then = then
        self._rise_z0 = float(self._tcp_measured[2])
        hover_z = self.config.touch_z + self.config.hover_m
        self._rise_point = np.array([self.tcp_cmd[0], self.tcp_cmd[1], max(hover_z, float(self.tcp_cmd[2]))])
        self._enter(RISE)

    def _rise(self, obs: Observation) -> Action | None:
        # After a stall the command must reach the held pose before the arm moves up, as for touch's verdict.
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        arrived = self._arrived(self._rise_point)
        # The jaw closes on the way up only once the tips are clear of the block: after a straddle stall one
        # open tip rests on the top and the moving tip would sweep across it.
        if arrived or self._tcp_measured[2] >= self._rise_z0 + CLEAR_BEFORE_CLOSE_M:
            self._jaw_goal_q = float(self._lo[self._g])
            self._ramp_jaw()
        if arrived and self._jaw_ready() and self._jaw_goal_q == float(self._lo[self._g]):
            if self._rise_then == OPEN:
                self._enter(OPEN)
                return None
            assert self._retarget_xy is not None
            try:
                self.retarget(float(self._retarget_xy[0]), float(self._retarget_xy[1]))
            except RuntimeError as exc:
                return self._halt(str(exc), self.kin.from_q(self.q_cmd))
            return self.hold("ponowna proba obok")
        return self._drive(self._rise_point, self.approach_step_m)

    def _open(self, obs: Observation) -> Action | None:
        self._set_jaw_goal(self.cmd_open)
        self._ramp_jaw()
        reading = self.reading
        settled = self._settled(self.straddle_hover)
        jaw_open = abs(reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
        vertical = self._vertical()
        ready = settled and self._jaw_ready() and jaw_open and vertical
        if self._before_straddle_pending and (ready or self._looking):
            action = self.before_straddle(obs, settled=ready)
            if action is not None:
                return action
            self._before_straddle_pending = False
        if ready:
            # The straddle's reference, taken again: the TCP moved and the jaw opened since the probe.
            self._bias_z, self._hover_z = self._offset_z, float(self._tcp_measured[2])
            self._enter(STRADDLE)
            return None
        travel = float(np.linalg.norm(self.straddle_hover - self._rise_point)) / self.config.speed_mps
        if not self._looking and self._t - self._phase_t0 >= JAW_WAIT_S + travel:
            if not jaw_open:
                gaps = f"{self.grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm"
                return self._halt(f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s ({gaps})",
                                  self.kin.from_q(self.q_cmd))
            if not vertical:
                return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
            distance = float(np.linalg.norm(self.straddle_hover - self._tcp_measured)) * 1000
            return self._halt(f"ramie nie stanelo nad blokiem ({distance:.0f} mm od punktu)",
                              self.kin.from_q(self.q_cmd))
        return self._drive(self.straddle_hover, self.approach_step_m)

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        """Called at the straddle hover with the jaw open, from the first tick the arm is settled there until
        it returns None once (then not again for that straddle), and on every tick while `_looking` holds:
        the same contract as `before_descent`. The default returns None."""
        return None

    @property
    def _looking(self) -> bool:
        """Whether a look at the straddle hover is under way (`before_straddle`); never in grasp itself."""
        return False

    def _straddle(self, obs: Observation) -> Action | None:
        st = self.status
        st.rise_m = self._offset_z - self._bias_z
        if st.rise_m > self.config.contact_gap_m:
            return self._straddle_stall()
        if self._settled(self.grasp_target):
            if st.rise_m > self.config.contact_gap_m / 2:
                return self._straddle_stall()
            self._enter(CLOSE)
            self._q_open_entry = self.reading.q
            self._jaw_moved, self._still_ticks = False, 0
            return None
        return self._drive(self.grasp_target, self.descend_step_m)

    def _straddle_stall(self) -> Action:
        """A jaw tip on the block: the arm holds the pose it measures and stops pressing, as touch's stall."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        self.grasp.straddle_stalls += 1
        self.grasp.verdict = "tip"
        self._enter(STRADDLED)
        return self.hold("szczeka na bloku")

    def _straddled(self, obs: Observation) -> Action:
        """The verdict waits for the supervisor's command to reach the held pose, as touch's does."""
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        return self.after_grasp_failed("tip", obs)

    def _close(self, obs: Observation) -> Action | None:
        self._set_jaw_goal(self.cmd_closed)
        self._ramp_jaw()
        reading, jaw = self.reading, self.kin.spec.jaw
        # A jaw standing at its opening against a tighter command reads holding by the rule alone, so the
        # verdict waits until the jaw has visibly left the opening.
        self._jaw_moved |= reading.q <= self._q_open_entry - jaw.margin
        self._still_ticks = self._still_ticks + 1 if self._jaw_moved and reading.still else 0
        if self._jaw_ready() and self._still_ticks >= self.settle_ticks:
            if reading.holding:
                assert self._jaw is not None
                self._jaw.mark()
                self.grasp.verdict = "held"
                self._lift_off_point = self.grasp_target + np.array([0.0, 0.0, self.grasp_settings.lift_m])
                self._enter(LIFT_OFF)
                return None
            self.grasp.empty_closes += 1
            self.grasp.verdict = "empty"
            return self.after_grasp_failed("empty", obs)
        if self._t - self._phase_t0 >= JAW_WAIT_S:
            if not self._jaw_moved:
                return self._halt(f"szczeka nie ruszyla po rozkazie zamkniecia "
                                  f"({self.grasp.jaw_gap_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
            return self._halt(f"szczeka nie staje {JAW_WAIT_S:.0f} s po zamknieciu - serwo albo odczyt",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def _lift_off(self, obs: Observation) -> Action | None:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - self._grasp_z
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        if self._arrived(self._lift_off_point):
            self.grasp.hold_ticks = 0
            self._enter(CHECK)
            return None
        return self._drive(self._lift_off_point, self.approach_step_m)

    def _check(self, obs: Observation) -> Action:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - self._grasp_z
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp = self.grasp
        grasp.hold_ticks = grasp.hold_ticks + 1 if self.reading.holding else 0
        if grasp.hold_ticks >= math.ceil(self.grasp_settings.hold_s * self.hz):
            grasp.verdict = "lifted"
            return self._halt(f"blok w szczekach, podniesiony o {grasp.lifted_m * 1000:.0f} mm "
                              f"(szczeka {grasp.jaw_gap_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd),
                              success=True)
        if self._t - self._phase_t0 >= self.grasp_settings.hold_s + JAW_WAIT_S:
            return self._halt("szczeka nie stoi po podniesieniu - blok sie rusza w szczekach",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Called first on every LIFT_OFF and CHECK tick with the tick's jaw rule; None = carry on.

        The default ends the run when the jaw stops squeezing (the block fell out) or creeps closed since the
        close (it slides out). A vibrating jaw is no slip: squeezing, not stillness, guards the lift. The
        seam for the recovery loop, which reconciles the jaw with the twin's "w dloni".
        """
        if not reading.squeezing:
            self.grasp.verdict = "slipped"
            return self._halt("blok wypadl ze szczeki podczas podnoszenia", self.kin.from_q(self.q_cmd))
        if reading.crept:
            self.grasp.verdict = "crept"
            return self._halt("blok zsuwa sie w szczekach", self.kin.from_q(self.q_cmd))
        return None

    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        """A straddle that failed, the arm holding: "tip" (a jaw tip landed on the block) or "empty" (the
        jaw closed on nothing).

        After a tip the arm rises and probes again half the short side along the closing heading, while
        attempts remain. An empty close ends the run: after a straddle that met nothing the block was within
        the opening along the heading, and the jaw alone cannot say where it went. A decision step that
        waits for an answer enters a phase of its own.
        """
        attempt = self.status.attempt
        if why == "empty":
            return self._halt("chwyt pusty - blok przesuniety albo obok szczek", self.kin.from_q(self.q_cmd))
        if attempt >= self.grasp_settings.max_attempts or self._next_shift() is None:
            return self._halt(f"wyczerpane proby chwytu ({attempt}): szczeka oparla sie o blok przy zjezdzie",
                              self.kin.from_q(self.q_cmd))
        self._start_rise(RETARGET)
        return self.hold("szczeka na bloku - w gore i obok")

    def _next_shift(self) -> np.ndarray | None:
        """The next block centre of `SHIFTS` from the first aim, kept in `_retarget_xy`; None when spent."""
        self._shift_k += 1
        if self._shift_k > len(SHIFTS) or self._block_xy0 is None:
            return None
        shift = SHIFTS[self._shift_k - 1] * self.short_m / 2
        self._retarget_xy = self._block_xy0 + shift * np.array([math.cos(self._psi), math.sin(self._psi)])
        return self._retarget_xy

    def grasp_verdict(self) -> tuple[str, str]:
        """The grasp's verdict and its text, for logs and a decision step."""
        verdict = self.grasp.verdict or "none"
        return verdict, VERDICTS.get(verdict, verdict)

    # ------------------------------------------------------------------ the jaw
    def _set_jaw_goal(self, gripper: float) -> None:
        self._jaw_goal_q = float(np.clip(self._q(gripper), self._lo[self._g], self._hi[self._g]))

    def _ramp_jaw(self) -> None:
        """The gripper target one step toward `_jaw_goal_q`: `SPEED_SHARE` of the supervisor's gripper limit
        per tick, the share the arm joints take, so the twin's 50 Hz loop follows each tick."""
        g = self._g
        delta = float(np.clip(self._jaw_goal_q - self.q_cmd[g], -self._grip_step_q, self._grip_step_q))
        before = float(self.q_cmd[g])
        self.q_cmd[g] = float(np.clip(before + delta, self._lo[g], self._hi[g]))
        if self.q_cmd[g] != before:
            # A new gripper target: the supervisor's command has not had it yet.
            self._jaw_on_command = False
        spec = self.kin.spec
        assert spec.gripper is not None
        self.grasp.gap_cmd_m = self.kin.jaw_gap(self.kin.from_q(self.q_cmd)[spec.gripper])

    def _jaw_ready(self) -> bool:
        """The gripper target reached its goal and the supervisor's command reached the target."""
        return abs(float(self.q_cmd[self._g]) - self._jaw_goal_q) <= 1e-9 and self._jaw_on_command

    # ------------------------------------------------------------------ the panel
    def _note(self) -> str:
        st, grasp = self.status, self.grasp
        mm = st.distance * 1000
        if st.phase == RISE:
            text = ("w gore po dotyku" if self._rise_then == OPEN else "w gore znad bloku") + f": {mm:.0f} mm"
        elif st.phase == OPEN:
            text = f"nad blok, otwieram szczeke: {grasp.gap_cmd_m * 1000:.0f} mm"
        elif st.phase == STRADDLE:
            text = f"w dol na blok: {mm:.0f} mm"
        elif st.phase == STRADDLED:
            text = "szczeka na bloku - czekam na rozkaz"
        elif st.phase == CLOSE:
            text = "sprawdzam chwyt" if self._jaw_ready() else "zamykam szczeke na bloku"
        elif st.phase == LIFT_OFF:
            text = f"podnosze blok: {mm:.0f} mm"
        elif st.phase == CHECK:
            text = f"trzymam blok: {grasp.hold_ticks / self.hz:.1f} s"
        else:
            return super()._note()
        return text + (" - cel poza zasiegiem ramienia" if st.out_of_reach else "")


# ------------------------------------------------------------------ geometry
def top_down(psi: float) -> np.ndarray:
    """The TCP rotation in the base frame with the approach straight down and the jaws closing along the
    heading `psi` [rad]: columns approach, y, closing (the TCP frame's x, y, z)."""
    c, s = math.cos(psi), math.sin(psi)
    return np.column_stack([(0.0, 0.0, -1.0), (-s, c, 0.0), (c, s, 0.0)])


def wrap(angle: float) -> float:
    """`angle` folded into (-pi, pi] [rad]."""
    folded = (angle + math.pi) % (2 * math.pi) - math.pi
    return math.pi if folded == -math.pi else folded


def tilt(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """How far the approach of `joints` leans from straight down [rad]."""
    approach, _ = kin.tool_axes(joints)
    return math.acos(float(np.clip(-approach[2], -1.0, 1.0)))


def heading(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """The heading of the closing axis of `joints`, its horizontal projection [rad, base frame]."""
    _, closing = kin.tool_axes(joints)
    return math.atan2(float(closing[1]), float(closing[0]))


def grasp_headings(rotation: np.ndarray, yaw_period: float, current: float) -> list[float]:
    """The closing headings worth trying on a block with pose `rotation`, the one nearest `current` first.

    With an identity rotation (a detector that gives no orientation) the jaws keep their heading `current`
    and its mirror. Otherwise they close across the block's short side: the rotation's x column is the
    block's long side (`rl.objects`), folded by the block's yaw symmetry `yaw_period`, each with its mirror
    (the same grasp, the wrist turned half way): 4 headings for a cube, 2 for the cuboids and the cylinder.
    The seam for a grasp library, which would list its cached grasps instead.
    """
    R = np.asarray(rotation, float)
    if np.allclose(R, np.eye(3), atol=1e-6):
        return [wrap(current), wrap(current + math.pi)]
    base = math.atan2(float(R[1, 0]), float(R[0, 0])) + math.pi / 2
    out: list[float] = []
    for k in range(max(1, round(2 * math.pi / yaw_period))):
        for mirror in (0.0, math.pi):
            psi = wrap(base + k * yaw_period + mirror)
            if all(abs(wrap(psi - other)) > 1e-6 for other in out):
                out.append(psi)
    return sorted(out, key=lambda psi: abs(wrap(psi - current)))

