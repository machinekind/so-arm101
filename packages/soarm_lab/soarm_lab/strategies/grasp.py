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
      max_attempts: 3              # probe descents in one run, regrasps after a drop included
      carry_s: 10.0                # how long the block is carried under the watch after the lift [s]
      release: false               # open the jaw after carry_s; false = the operator takes the block
      wait_s: 20.0                 # after a drop, how long to wait for the cameras to see the block [s]
      recover_m: 0.15              # a block seen farther than this from where it fell was taken [m]
      time_limit_s: 120.0          # a safety net for the whole run: a grasp, a drop, a regrasp [s]

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
5. the jaw rule (`jaw.JawMonitor`) says held: lift `lift_m` straight up, and hold there for `hold_s`;
6. carry the block there for `carry_s` under the watch (see Recovery), then end with the block held or,
   with `release`, open the jaw.

The gripper command ramps at `SPEED_SHARE` of the supervisor's gripper speed limit, 240 units/s on the
SO-101, about 10 mm of gap a tick: the open takes 4 ticks, the close 3, and the servo follows later.

Guards (touch's, and):

* **the jaw rule**: the close counts as held only after the measured jaw dropped by `jaw.margin` below its
  opening (a jaw that never moved stands still at its opening and squeezes against a tighter command by
  the rule alone) and then stood still for `SETTLE_S` with the command on it; held means it stands wider
  than commanded and than closed by more than the margin. From then on the watch (see Recovery) reads
  the jaw every tick: an empty jaw is a drop, and a jaw that creeps closed by more than `jaw.creep` while
  it still squeezes only restarts the check;
* **openings the jaw can make**: `start` refuses a run whose opening or squeeze lies outside the jaw's
  travel, a squeeze within the jaw margin (a held block would not read held) or an opening within it (a
  jaw closing on the block could not be told from one that never moved);
* **clear before close**: after a jaw tip landed on the block, the jaw stays open while the arm rises and
  closes only 10 mm higher, so the moving tip does not sweep across the block's top;
* **reach with orientation**: when the pose is taken, the full multi-start IK must reach the probe hover,
  the probe's bottom, the straddle hover, the grasp point and the lift point with the hand within 5
  degrees of top-down, within the limits and off the arm's own body, for one of the candidate headings.
  On the SO-101, from the rest pose and for a pose without rotation, that holds for a cube's centre in a
  band: x 0.14-0.27 m straight ahead with the 4 cm hover (0.18-0.24 with touch's 6 cm); 10 cm to the
  side x 0.12-0.25 m on the right (y -0.10) and 0.03-0.25 m on the left (y +0.10). Beyond it wrist_flex
  stops at its 95 degree limit, and the run ends with "chwytak nie siega tam pionowo". The hand turns
  toward top-down at touch's `TURN_DEG_S`, and the target is checked to be top-down within 8 degrees
  before the probe and before the straddle;
* **the lift's time limit**: the measured TCP must reach the lift point `JAW_WAIT_S` after the lift's
  travel (`lift_m` at `speed_mps`, or the ticks the target and the supervisor's command took under the
  speed limits), counted from the phase clock; otherwise the run ends with "ramie nie podnioslo bloku",
  the squeeze kept;
* **the attempt budget**: `max_attempts` probe descents in one run, the sideways retries and the regrasps
  after a drop together.

A jaw tip that stalls on the block during the straddle holds the measured pose until the supervisor's
command reaches it, rises (closing the jaw once clear) and probes again half the short side along the
closing heading (`SHIFTS`): toward the moving jaw first. When that probe misses, the block lies on the
other side, and the next probe is half the short side the other way from the original point. A close that
finds nothing ends the run ("chwyt pusty"): after a straddle that met nothing the block was within the
opening, and the jaw alone cannot say whether the probe pushed it away or it sits at the jaws' edge.

Success: the jaw holds for `hold_s` at the lift height and keeps holding while the arm carries the block
there for `carry_s`, "blok w szczekach, podniesiony o N mm, niesiony T s". The block stays held after the
run: the last target squeezes, the twin's runner sends the done tick's target before it disengages, and
disengaging keeps the supervisor's command; a later STOP or Home keeps the squeeze through the twin's
`hold_measured`. The twin's tracker says "w dloni" from the close on, with the same jaw thresholds
(`ObjectTracker.for_spec`). With `release` on, the jaw opens to the straddle opening after `carry_s`
instead, and the run ends "blok oddany" once the measured jaw stands there: the strategy's responsibility
ends with that command, and a block nobody takes falls to the table.

Recovery. From the close verdict "held" until `release()` or the end of the run, every lift_off, check and
carry tick first calls `carrying`, whose default is the watch: the jaw rule against the block's pose in
`obs.objects`, as the plan's table has it.

    jaw rule            the block's pose (`Sighting`)               action
    squeezing           any, "w dloni" as a rule                    carry on
    squeezing           seen on the table away from the fingertips  carry on, note it (`elsewhere`)
    empty, twice        any                                         a drop: rise to the safe height
    (jaw open)          seen on the table since the drop, near      grasp it again through `retarget`
    (jaw open)          seen farther than `recover_m`               stop for a person: taken
    (jaw open)          not seen since the drop within `wait_s`     stop for a person

* A squeezing jaw carries on whatever the twin says: the jaw rule is the contact signal, the cameras only
  confirm. A plain "kamery" pose lying on the table more than `IN_HAND_M` from the fingertips' midpoint is
  noted for a decision step (a second block or a tracker error). A creep is no drop: it re-marks the jaw
  monitor and restarts the check's hold streak.
* A jaw that reads empty `EMPTY_TICKS` ticks in a row has dropped the block. The arm rises straight up to
  the safe height (the probe hover, or higher when it already is) while the jaw OPENS past the twin
  tracker's `open` gate, to `jaw.open` plus two margins (0.50 rad on the SO-101, 55 mm between the tips)
  or the straddle opening when that is wider. The tracker accepts a single camera's detection next to a
  standing hand only with the jaw that open, and a block still hanging in the tips falls where the
  cameras see it.
* A sighting is a plain "kamery" or "ostatnie widziane" pose no older than `max_pose_age_s`, lying at rest
  height (the block's largest half size plus 1 cm), captured `DROP_SETTLE_S` or more after the drop was
  declared; in simulation the scene's own pose at rest height counts too. Annotated last-seen poses
  ("ostatnie widziane (upuszczona)", the tracker's guess under the hand, or "(1 kamera widzi ja gdzie
  indziej)"), "w dloni" and anything captured before the drop never do: the strategy never descends to a
  stale pose.
* The arm waits at the safe height, the jaw open, at most `wait_s` from the drop. A sighting within
  `recover_m` of where the block fell is grasped again: the jaw closes at the hover before the probe, and
  the regrasp's descent counts a new attempt against the same `max_attempts`. A sighting farther away is a
  block taken away, and a snatch is not a miss: it is never chased. Farther, out of reach, the attempts
  spent, or no sighting within `wait_s`: the run ends without success, the arm holding the safe height
  with the jaw open for a person.

With the LLM detector alone a single camera cannot settle a sighting next to the hand (the tracker wants
three detections within 1.5 s, the detector answers every 10 s), so one-camera recovery needs the seeded
detector or a second camera; with the LLM detector's answers arriving seconds old, its sighting is the
plain "ostatnie widziane". `release()` ends the watch.

Phases (`TouchStatus.phase`): touch's wait, lift, approach, descend and touched, then rise, open, straddle
(straddled after a tip stall), close, lift_off, check and carry, recover after a drop and release, then
done. Seams for later steps: `after_miss` (a probe that met nothing), `before_descent` (every tick of a
descent until it returns None once: wrist centring), `before_straddle` and `_looking` (at the straddle
hover), `after_grasp_failed` (a tip on the block or an empty close), `carrying` (every tick from the hold
verdict until the release: the watch above, which a decision step may extend), `release` (ends the watch
and opens the jaw: a place task or the Jev step calls it), `grasp_headings` (the candidate headings: the
grasp library) and `grasp_verdict` (for logs and a decision step). `after_touch` stays the seam of `touch`.
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
    APPROACH,
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
#: The verified block carried at the lift height under the watch, for `carry_s`.
CARRY = "carry"
#: After a drop: the arm waits at the safe height, the jaw open, for a sighting.
RECOVER = "recover"
#: `release()` opens the jaw; the watch is off.
RELEASE = "release"
#: Where the rise leads: to the open, to a retarget, or after a drop to RECOVER.
RETARGET = "retarget"

#: Consecutive non-squeezing jaw readings that make a drop: 0.1 s at 20 Hz, so one bad gripper read is none.
EMPTY_TICKS = 2
#: A sighting counts only when captured this long after the drop was declared [s]: a 6 cm fall takes 0.11 s,
#: a bounce or a roll a little more.
DROP_SETTLE_S = 0.3
#: A pose nearer than this to the fingertips' midpoint is the block in the jaws [m]: the twin tracker's own
#: `grab_radius`.
IN_HAND_M = 0.06
#: Pose sources a sighting after a drop may come from, matched exactly as `TOUCH_SOURCES`: annotated last-seen
#: poses and "w dloni" never count. The scene's own pose ("symulacja", or no source in simulation) is judged
#: by its height and the jaw instead.
SIGHTING_SOURCES = ("kamery", "ostatnie widziane")
#: What a `Sighting` says about the block.
IN_HAND = "in_hand"
SEEN = "seen"
STALE = "stale"

#: What `grasp_verdict` says, per verdict.
VERDICTS = {
    "none": "brak werdyktu chwytu",
    "tip": "szczeka oparla sie o blok przy zjezdzie",
    "empty": "chwyt pusty",
    "held": "szczeka trzyma blok",
    "lifted": "blok podniesiony",
    "dropped": "blok wypadl ze szczeki",
    "recovering": "ponowna proba po upadku",
    "taken": "blok zabrany - czekam na czlowieka",
    "released": "blok oddany",
}


@dataclass(frozen=True)
class Sighting:
    """The block's pose in one observation, as the watch reads it.

    `kind` is `IN_HAND` (the twin says "w dloni", or in simulation the scene's pose lies at the fingertips
    while the jaw squeezes), `SEEN` (it lies on the table where a regrasp may aim: a plain "kamery" or
    "ostatnie widziane" pose, fresh enough, at rest height) or `STALE` (anything else: an annotated
    last-seen pose, a pose too old or above rest height, an unknown source). `seen_t = obs.t - age_s` is
    when the pose was captured, on the run's clock [s].
    """

    kind: str
    #: The pose's x and y [m, base frame], its height [m] and rotation (identity without one).
    xy: tuple[float, float]
    z: float
    rotation: np.ndarray
    source: str
    age_s: float
    seen_t: float


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
    #: A safety net for the whole run [s]; 0 means none. A grasp takes about 10 s; with the defaults a grasp,
    #: `carry_s`, a drop with its `wait_s`, a regrasp and a second carry take about 65 s.
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
    #: Probe descents in one run: the first, the retries after a jaw tip landed on the block and the regrasps
    #: after a drop [count].
    max_attempts: int = 3
    #: How long the verified block is carried at the lift height under the watch before the run ends as a
    #: success, or the release is commanded [s]; 0 = end at the verified lift.
    carry_s: float = 10.0
    #: After `carry_s` the jaw opens to the straddle opening and the run succeeds once it is open; off = the
    #: run ends with the block held and the operator releases it.
    release: bool = False
    #: After a drop, how long the arm waits at the safe height with the jaw open for the cameras to see the
    #: block on the table, before the run stops for a person [s].
    wait_s: float = 20.0
    #: A block seen after a drop farther than this from where it fell was taken away, not dropped: the run
    #: stops for a person instead of chasing it [m].
    recover_m: float = 0.15


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
    #: '', 'tip', 'empty', 'held', 'lifted', 'dropped' (the jaw lost the block while carrying it),
    #: 'recovering' (a regrasp after a drop is under way), 'taken' (the run stopped for a person after a
    #: drop) or 'released'.
    verdict: str = ""
    #: The source of the block's pose on the last watched or recovery tick; '' without a pose.
    block_source: str = ""
    #: Drops while carrying, regrasps after one, and creeps of a squeezing jaw [count].
    drops: int = 0
    recoveries: int = 0
    creeps: int = 0
    #: Where the cameras saw the block on the table while the jaw squeezed it [m, base frame]; None = never.
    elsewhere: tuple[float, float] | None = None
    #: How long the verified block has been carried [s].
    carried_s: float = 0.0
    #: When the last drop was declared [s, `obs.t`], and how long ago on the last recovery tick [s].
    dropped_t: float = float("nan")
    lost_s: float = float("nan")
    #: `release()` was commanded.
    released: bool = False


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
        for name in ("open_margin_m", "squeeze_m", "lift_m", "hold_s", "wait_s", "recover_m"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"grasp: {name} must be positive, is {value}")
        if not config.carry_s >= 0:
            raise ValueError(f"grasp: carry_s must be zero or more, is {config.carry_s}")
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
        #: A pose of the block no higher than this lies on the table [m]: its centre on any face, plus 1 cm.
        self.rest_z_max = max(block.half_size) + 0.01
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
        #: The lift's ticks on which the target or the supervisor's command was still on its way [count].
        self._lift_ticks = 0
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
        # The watch: the tick it last ran on (`TouchStatus.step`), non-squeezing readings in a row; the last
        # drop's time [s, `obs.t`] and the fingertips' midpoint there [m, base frame]; the jaw angle the
        # recovery opens to [rad].
        self._watched_step = -1
        self._empty_ticks = 0
        self._drop_t = float("nan")
        self._drop_xy: np.ndarray | None = None
        self._recover_open_q = float("nan")
        self._phases.update({RISE: self._rise, OPEN: self._open, STRADDLE: self._straddle,
                             STRADDLED: self._straddled, CLOSE: self._close, LIFT_OFF: self._lift_off,
                             CHECK: self._check, CARRY: self._carry, RECOVER: self._recover,
                             RELEASE: self._release})

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
        # After a drop the jaw opens past the twin tracker's `open` gate by two margins (the gate reads the
        # measured jaw, the margins cover a servo's error), or to the straddle opening when that is wider.
        self._recover_open_q = min(float(self._hi[g]), max(self._q(self.cmd_open), jaw.open + 2 * jaw.margin))
        self._watched_step, self._empty_ticks = -1, 0
        self._drop_t, self._drop_xy = float("nan"), None

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
        """touch's approach, but the descent starts only with the hand top-down and the jaw on its goal.

        The jaw ramps toward its goal on the way: after a regrasp it closes from the recovery's opening, as
        the probe needs closed tips. Settled at the hover with the ramp unfinished or the supervisor's
        command not on it, the arm holds until `JAW_WAIT_S` after the phase began. The first grasp of a run
        never waits here: its jaw is closed from `start`.
        """
        self._ramp_jaw()
        if self._rot is not None and not self._vertical():
            # The hover reached before the hand has turned: it turns on the spot, for `JAW_WAIT_S` at most.
            if self._arrived(self.hover_point):
                if not np.isfinite(self._turn_t0):
                    self._turn_t0 = self._t
                elif self._t - self._turn_t0 >= JAW_WAIT_S:
                    return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
            self._settle_count = 0
            return self._drive(self.hover_point, self.approach_step_m)
        self._turn_t0 = float("nan")
        # touch's body, repeated with the jaw gate: `_settled` counts once per tick, so it cannot be wrapped.
        if self._settled(self.hover_point):
            if not self._jaw_ready():
                if self._t - self._phase_t0 >= JAW_WAIT_S:
                    return self._halt(f"szczeka nie zamknela sie przed zejsciem w {JAW_WAIT_S:.0f} s",
                                      self.kin.from_q(self.q_cmd))
                return self.hold()
            # The reference of the descent: how the measured TCP stands against the command at the hover.
            self._bias_z = self._offset_z
            self._hover_z = float(self._tcp_measured[2])
            self.status.attempt += 1
            self._enter(DESCEND)
            return None
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
        """Straight up from the last target to the hover height, the safe height (or the target's own height
        when that is higher); then `then`: OPEN or RETARGET with the jaw closed, RECOVER with it open
        (`_recover_open_q`)."""
        self._rise_then = then
        self._rise_z0 = float(self._tcp_measured[2])
        hover_z = self.config.touch_z + self.config.hover_m
        self._rise_point = np.array([self.tcp_cmd[0], self.tcp_cmd[1], max(hover_z, float(self.tcp_cmd[2]))])
        self._empty_ticks = 0
        self._jaw_goal_q = self._recover_open_q if then == RECOVER else float(self._lo[self._g])
        self._enter(RISE)

    def _rise(self, obs: Observation) -> Action | None:
        # After a stall the command must reach the held pose before the arm moves up, as for touch's verdict.
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        arrived = self._arrived(self._rise_point)
        # The jaw closes on the way up only once the tips are clear of the block: after a straddle stall one
        # open tip rests on the top and the moving tip would sweep across it. After a drop it opens at once.
        clear = self._tcp_measured[2] >= self._rise_z0 + CLEAR_BEFORE_CLOSE_M
        if self._rise_then == RECOVER or arrived or clear:
            self._ramp_jaw()
        if arrived and self._jaw_ready():
            if self._rise_then == OPEN:
                self._enter(OPEN)
                return None
            if self._rise_then == RECOVER:
                self._enter(RECOVER)
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
                self._lift_ticks = 0
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
        # The lift travels while the target moves toward the lift point (a target the IK cannot take further
        # does not) or the supervisor's command has not reached the target, so a speed cap lowered during the
        # run lengthens the limit too.
        on_point = float(np.linalg.norm(self._lift_off_point - self.tcp_cmd)) <= self.ik_tol
        if (not on_point and not self.status.out_of_reach) or not self._command_on_target:
            self._lift_ticks += 1
        # Counted from the phase clock: the travel at `speed_mps`, or the ticks it took when that was longer,
        # then `JAW_WAIT_S` for the arm to follow. The squeeze stays in the last target.
        lift_m = self.grasp_settings.lift_m
        travel = max(lift_m / self.config.speed_mps, self._lift_ticks / self.hz)
        if self._t - self._phase_t0 >= travel + JAW_WAIT_S:
            return self._halt(f"ramie nie podnioslo bloku ({self.grasp.lifted_m * 1000:.0f} mm z "
                              f"{lift_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
        return self._drive(self._lift_off_point, self.approach_step_m)

    def _check(self, obs: Observation) -> Action | None:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - self._grasp_z
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp = self.grasp
        # A creep zeroes the streak; `carrying` re-marked the monitor, so it restarts on the next tick.
        reading = self.reading
        grasp.hold_ticks = grasp.hold_ticks + 1 if reading.holding and not reading.crept else 0
        if grasp.hold_ticks >= math.ceil(self.grasp_settings.hold_s * self.hz):
            grasp.verdict = "lifted"
            self._enter(CARRY)
            return None
        if self._t - self._phase_t0 >= self.grasp_settings.hold_s + JAW_WAIT_S:
            return self._halt("szczeka nie stoi po podniesieniu - blok sie rusza w szczekach",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def _carry(self, obs: Observation) -> Action:
        """The verified block held at the lift height under the watch for `carry_s`; then the run ends with
        the block held, or with `release` the release is commanded."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp, cfg = self.grasp, self.grasp_settings
        grasp.carried_s = self._t - self._phase_t0
        if grasp.carried_s >= cfg.carry_s:
            if cfg.release:
                return self.release()
            return self._halt(f"blok w szczekach, podniesiony o {grasp.lifted_m * 1000:.0f} mm, niesiony "
                              f"{grasp.carried_s:.1f} s (szczeka {grasp.jaw_gap_m * 1000:.0f} mm)",
                              self.kin.from_q(self.q_cmd), success=True)
        return self.hold()

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Called first on every LIFT_OFF, CHECK and CARRY tick with the tick's jaw rule; None = carry on, an
        Action = the recovery or a release took over (the phase's handler returns it as it is).

        The default is the watch: the jaw rule against the block's pose (`_sighting`). A squeezing jaw
        carries on whatever the twin says; a creep re-marks the monitor and is counted, and a fresh pose on
        the table away from the fingertips is noted in `GraspStatus.elsewhere`. A jaw that reads empty
        `EMPTY_TICKS` ticks in a row has dropped the block: the arm rises to the safe height with the jaw
        opening (`_start_rise(RECOVER)`). A tick whose phase changes within it (lift_off to check, check to
        carry) calls it twice; the watch reads the tick once. A decision step may override it and call
        `super().carrying`; when it calls `release()` it returns what `release()` returns.
        """
        if self._watched_step == self.status.step:
            # Watched on this tick already, and it said carry on: one reading counts once.
            return None
        self._watched_step = self.status.step
        grasp = self.grasp
        seen = self._sighting(obs, reading.squeezing)
        grasp.block_source = seen.source if seen is not None else ""
        jaw_xy = self._jaw_point(obs.joints)[:2]
        if reading.squeezing:
            self._empty_ticks = 0
            if reading.crept:
                # The block turns or slides but is still squeezed: no drop; the check restarts its streak.
                grasp.creeps += 1
                assert self._jaw is not None
                self._jaw.mark()
                logger.info("Blok przesuwa sie w szczekach (%d)", grasp.creeps)
            if (seen is not None and seen.kind == SEEN and grasp.elsewhere is None
                    and float(np.linalg.norm(np.asarray(seen.xy) - jaw_xy)) > IN_HAND_M):
                grasp.elsewhere = seen.xy
                logger.info("Kamery widza blok gdzie indziej: (%.3f, %.3f) m", *seen.xy)
            return None
        # One empty reading is no drop: the gripper read may be off for a tick.
        self._empty_ticks += 1
        if self._empty_ticks < EMPTY_TICKS:
            return None
        grasp.drops += 1
        grasp.verdict = "dropped"
        self._drop_t = grasp.dropped_t = self._t
        self._drop_xy = jaw_xy.copy()
        grasp.elsewhere = None
        grasp.lost_s = 0.0
        logger.info("Blok wypadl ze szczeki (zrodlo: %s)", grasp.block_source or "brak")
        self._start_rise(RECOVER)
        return self.hold("blok wypadl ze szczeki - w gore")

    def release(self) -> Action:
        """Ends the watch and opens the jaw to the straddle opening; the run succeeds once it is open.

        Only from a carrying phase (LIFT_OFF, CHECK, CARRY), else it raises. It returns the tick's answer, so
        a subclass may call it from `carrying` and return the result. The arm does not move: a block nobody
        holds falls to the table, which is the handover's responsibility, not the strategy's.
        """
        if self.status.phase not in (LIFT_OFF, CHECK, CARRY):
            raise RuntimeError("grasp: release() outside a carrying phase")
        self.grasp.released = True
        self.grasp.verdict = "released"
        self._set_jaw_goal(self.cmd_open)
        self._enter(RELEASE)
        return self.hold("oddaje blok: otwieram szczeke")

    def _release(self, obs: Observation) -> Action:
        self._ramp_jaw()
        grasp = self.grasp
        if self._jaw_ready() and abs(self.reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin:
            return self._halt(f"blok oddany: szczeka otwarta ({grasp.jaw_gap_m * 1000:.0f} mm)",
                              self.kin.from_q(self.q_cmd), success=True)
        if self._t - self._phase_t0 >= JAW_WAIT_S:
            return self._halt(f"szczeka nie otworzyla sie po oddaniu bloku w {JAW_WAIT_S:.0f} s "
                              f"({grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm)",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    # ------------------------------------------------------------------ recovery
    def _recover(self, obs: Observation) -> Action:
        """After a drop: the arm holds the safe point with the jaw open until a sighting since the drop,
        then grasps the block again; a block taken away, or none seen within `wait_s`, stops the run."""
        grasp, cfg = self.grasp, self.grasp_settings
        seen = self._sighting(obs, self.reading.squeezing)
        grasp.block_source = seen.source if seen is not None else ""
        lost_s = grasp.lost_s = self._t - self._drop_t
        target = self.kin.from_q(self.q_cmd)
        # Captured after the block had landed: a plain pose from before the drop shows it in the jaws or where
        # it was picked, and never counts.
        if seen is not None and seen.kind == SEEN and seen.seen_t >= self._drop_t + DROP_SETTLE_S:
            assert self._drop_xy is not None
            far = float(np.linalg.norm(np.asarray(seen.xy) - self._drop_xy))
            if far > cfg.recover_m:
                # A snatch is not a miss: never chased.
                grasp.verdict = "taken"
                return self._halt(f"blok zabrany: lezy {far * 100:.0f} cm od miejsca upadku - czekam na "
                                  "czlowieka", target)
            if self.status.attempt >= cfg.max_attempts:
                grasp.verdict = "taken"
                return self._halt(f"wyczerpane proby chwytu ({self.status.attempt}): blok wypadl ze szczeki",
                                  target)
            return self._regrasp(seen)
        if lost_s >= cfg.wait_s:
            grasp.verdict = "taken"
            if seen is None:
                why = "kamery go nie widza"
            elif seen.kind == SEEN:
                # Seen, but captured before the drop: not to be read as a sighting since.
                why = f"zrodlo '{seen.source}' sprzed upadku"
            else:
                high = f", {seen.z * 100:.0f} cm nad blatem" if seen.z > self.rest_z_max else ""
                why = f"zrodlo '{seen.source}'{high}"
            return self._halt(f"blok zabrany ze szczeki: {why} od {lost_s:.0f} s - czekam na czlowieka",
                              target)
        return self.hold()

    def _regrasp(self, seen: Sighting) -> Action:
        """Aims the run at the sighting through `retarget`: the probe and the grasp again, from the hover.

        The jaw closes on the way to the hover (`_approach`), and the regrasp's descent counts a new attempt.
        A sighting out of reach changes nothing and stops the run for a person.
        """
        grasp = self.grasp
        saved = self._block_xy0, self._shift_k, self.block_rotation, self._headings
        # The sideways shifts after a tip stall start from the new block point.
        self._block_xy0, self._shift_k = None, 0
        self._heading_from(seen.rotation)
        try:
            self.retarget(seen.xy[0], seen.xy[1])
        except RuntimeError as exc:
            self._block_xy0, self._shift_k, self.block_rotation, self._headings = saved
            grasp.verdict = "taken"
            return self._halt(f"blok upadl poza zasiegiem: {exc} - czekam na czlowieka",
                              self.kin.from_q(self.q_cmd))
        self._jaw_goal_q = float(self._lo[self._g])
        grasp.elsewhere = None
        grasp.recoveries += 1
        grasp.verdict = "recovering"
        self._empty_ticks = 0
        logger.info("Blok lezy na blacie w (%.3f, %.3f) m - ponowna proba", *seen.xy)
        return self.hold("blok lezy na blacie - ponowna proba")

    def _sighting(self, obs: Observation, squeezing: bool) -> Sighting | None:
        """The block's pose in `obs` as the watch reads it; None without one or with no finite position.

        Pure: it reads only `obs`, `squeezing` (the tick's jaw rule) and the settings, so any observation may
        be judged.
        """
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        if pose is None:
            return None
        position = np.asarray(pose.position, float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None
        source = str(pose.source or "")
        age = float(pose.age_s)
        if not math.isfinite(age):
            age = math.inf
        rotation = np.asarray(pose.rotation, float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            rotation = np.eye(3)
        z = float(position[2])
        lying = z <= self.rest_z_max
        if source == "w dloni":
            kind = IN_HAND
        elif source in SIGHTING_SOURCES and age <= self.config.max_pose_age_s and lying:
            # The LLM detector's answers arrive seconds old and the twin publishes them as "ostatnie
            # widziane", never "kamery": the plain last-seen source is the sighting there.
            kind = SEEN
        elif source == "symulacja" or (not source and self._simulated):
            # The scene's truth: the jaw decides whether it is in the hand, not the distance alone (a
            # cuboid85 landed on its end under the hand has its centre 4 cm below the tips).
            near = float(np.linalg.norm(position - self._jaw_point(obs.joints))) < IN_HAND_M
            kind = IN_HAND if near and squeezing else SEEN if lying else STALE
        else:
            kind = STALE
        t = float(obs.t)
        return Sighting(kind, (float(position[0]), float(position[1])), z, rotation.copy(), source, age,
                        t - age)

    def _jaw_point(self, joints: Mapping[str, float]) -> np.ndarray:
        """The fingertips' midpoint of `joints` [m, base frame]."""
        return np.mean(self.kin.fingertips(joints), axis=0)

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
            rise = {OPEN: "w gore po dotyku", RECOVER: "blok wypadl ze szczeki - w gore"}
            text = rise.get(self._rise_then, "w gore znad bloku") + f": {mm:.0f} mm"
        elif st.phase == APPROACH and not self._jaw_ready():
            text = "nad obiekt: zamykam szczeke"
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
        elif st.phase == CARRY:
            text = f"niose blok: {grasp.carried_s:.1f} s"
            if grasp.elsewhere is not None:
                text += " - kamery widza blok gdzie indziej"
        elif st.phase == RECOVER:
            text = f"blok wypadl, czekam na kamery: {grasp.lost_s:.1f} s"
        elif st.phase == RELEASE:
            text = "oddaje blok: otwieram szczeke"
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

