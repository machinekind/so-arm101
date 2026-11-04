"""Grip and lift a block from above, then monitor the hold and recover after a drop.

The twin loads `grasp` through `machinekind_twin.strategies` and builds `GraspSettings`
from its configuration or panel. See the package README for settings.

`blocks.block` supplies dimensions and yaw symmetry. The strategy latches each grasp's pose
and uses the model's jaw geometry to plan TCP targets. By default, it opens at the hover,
straddles below the nominal top, closes and verifies the hold through the lift.
With `probe: true`, a closed-jaw descent first measures the top.
All planned points must be reachable with the hand down. The straddle stalls once the arm stands more
than `straddle_push_m` above where it should be. A stall with both jaw tips at least `close_on_stall_m`
below the block's top, or with the TCP within `CLOSE_NEAR_M` of the grasp target, closes the jaw where
the arm stands, and the close and lift checks judge the grasp. Any other stall is a tip on the block: it
retries half the chosen grasp width away within `max_attempts`. Without the probe, an empty close after
such a retry tries the other side. Any other empty close ends the run.

After closing, `carrying` checks the jaw and pose on each lift, check and carry tick.
A squeezing jaw keeps the block held. Creep restarts the check's hold streak.
`EMPTY_TICKS` consecutive empty readings declare a drop. The arm rises to at least the
hover and opens the jaw so the tracker can accept nearby detections.

Recovery requires a plain camera or last-seen pose captured at least `DROP_SETTLE_S` after
the drop. The pose must be fresh enough and at rest height. Simulation poses at rest height
also count. Annotated last-seen poses cannot guide a regrasp. A sighting within `recover_m`
triggers another grasp through `retarget`, sharing the attempt budget. The regrasp skips the
probe by default. A farther sighting, an unreachable pose, exhausted attempts or no sighting
within `wait_s` stops the run for a person with the jaw open.

The verified block stays at lift height for `carry_s`. Success keeps the squeeze unless
`release` is enabled. `release()` ends monitoring and opens the jaw without moving the arm.

Subclasses can refine the aim through `before_straddle`. With the probe on, `before_descent`
also refines the aim, and `after_touch` and `after_miss` judge the probe. `before_close` runs at the
bottom of the straddle with the open jaw around the block, before the jaw closes. Without the probe and
with the hand straight down, `lead_in` may give the last horizontal leg into the open-jaw hover. Every
straddle then starts from a hover the hand reached along that leg: the approach goes to the leg's start
first, and after the aim moves (a look, a correction or a retry) the hand goes back to the start of the
leg for the new hover and runs it again, so it always arrives from the same side. A leg's start the arm
does not get to in time is given up, and the hand goes straight on.
`after_grasp_failed` handles grasp failures. A subclass that can look again (`_looks_again`) also treats
a jaw closed well inside the block's width as a failed close (`EDGE_GRIP_M`), and retries an empty or
edge close from the hover within `max_attempts`, with the jaw open on the way up.

`jaw_gap_offset_m`, or the running mean that `jaw_file` learns from verified lifts, says how much wider the
model's jaw gap is than the real one at the same gripper reading. The strategy turns the gripper's
reading and command into the model's on the way in and back on the way out, so every opening, close,
release, jaw check and fingertip position uses the real jaw.
Subclasses can extend `carrying` and return `release()` for a handover.
`candidates` supplies geometry or library grasps. `grasp_verdict` supplies the verdict for logs
and judges.

`grasp_source: library` composes cached grasps with the block rotation and symmetries.
A pose without orientation uses geometry. Candidates are filtered and ranked before the
jaw rules and reach checks. The chosen width sets openings and sideways retry distance.
Depth and offset place the jaw midpoint below the nominal top by default.
With `probe: true`, depth counts from the measured top and the probe targets the block centre.
`MIN_BITE_M` keeps the shallower tip below the top despite its swing.

A block whose pose puts its bottom more than `ON_BLOCK_M` above the table rests on another block.
The hover, probe and grasp heights then count from its bottom, the pose height minus its resting
half height. The open jaw's lowest point stays `STACK_CLEAR_M` above the block under it, which
limits the grasp depth, and a grasp whose shallower tip then bites less than `MIN_BITE_M` is skipped.
While the jaw closes, the hand keeps the fingertips' midpoint over the block, and the jaw slows until the
hand keeps up, so the moving jaw does not slide the block across the one under it. On the table the hand
stays put, but the jaw closes as slowly until it reaches the block's width, so the moving jaw pushes the
block instead of striking it into the table.
Where the arm cannot reach the hover or the lift with the hand down, both come down in
`HEADROOM_STEP_M` steps to `MIN_HOVER_CLEAR_M` over the top and `MIN_LIFT_M`.
A regrasp after a drop counts from the bottom of the block as seen after the drop, so a block that
fell back onto a block keeps the same limits.

With `max_tilt_deg` above zero and the probe off, a grasp out of reach with the hand straight down is
tried with the hand tilted in `TILT_STEP_DEG` steps. The hand tilts about its closing axis and leans
away from the base, so the jaws still close level across the block. The arm can tilt the hand only in
its own vertical plane, so a tilted grasp needs a closing heading square to the arm, and so the
block's yaw: a block whose pose has none is tried tilted only when a look before the straddle measures
it (`_measures_yaw`). While tilted, the hand keeps its heading against the arm as the arm turns, and the
jaw clearances count along straight down for the tilted hand.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from ..blocks import BlockSpec, block
from ..blocks.classic_world import BLOCKS
from ..grasp_library import GraspLibrary, compose, load_library, rejections_text, roll_axis, select, wrap
from ..kinematics import IK_RESTARTS, IKSolution, RobotKinematics
from .jaw import JawMonitor, JawReading
from .touch import (
    APPROACH,
    DESCEND,
    HOLD_WAIT_S,
    MIN_TOUCH_Z,
    PHASE_NAMES,
    SPEED_SHARE,
    WAIT,
    TouchSettings,
    TouchStrategy,
    normalised,
)

logger = logging.getLogger(__name__)

#: Probe depth below the nominal top before adding jaw reach [m].
#: It must exceed the stall threshold plus contact deflection.
PROBE_BELOW_M = 0.010
#: The lowest TCP height of the grasp point [m]: the jaw's sole stays above the table.
MIN_GRASP_Z = 0.012
#: A block whose bottom lies higher than this above the table rests on another block [m].
ON_BLOCK_M = 0.007
#: The open jaw's lowest point stays this far above the block under the picked one [m].
STACK_CLEAR_M = 0.006
#: Over another block, a hover and a lift out of reach come down in steps of this [m].
HEADROOM_STEP_M = 0.005
#: The lowest hover over another block leaves the open jaw this far above the picked block's top [m].
MIN_HOVER_CLEAR_M = 0.015
#: The lowest lift off another block [m].
MIN_LIFT_M = 0.02
#: How long the jaw may take to open, to start closing and to stand after the close [s].
JAW_WAIT_S = 3.0
#: The supervisor's gripper command is on the target within this many app units.
JAW_COMMAND_TOL = 0.5
#: The reach check's largest orientation error of a top-down IK solution [deg].
MAX_ROT_ERR_DEG = 5.0
#: Allowed errors in tilt from vertical and in the chosen closing heading [deg].
MAX_TILT_DEG = 8.0
#: After a jaw tip landed on the block, the jaw closes only once the TCP rose this much [m].
CLEAR_BEFORE_CLOSE_M = 0.010
#: Default jaw reach below the TCP [m], replaced by the model's value in `start`.
JAW_REACH_M = 0.0066
#: A straddle hover this close to where the last leg from `lead_in` ended counts as reached along it [m].
LEG_SAME_M = 0.0005
#: Retry offsets from the first aim in half chosen-grasp widths along the closing heading.
#: Positive offsets point toward the moving jaw.
SHIFTS = (1, -1, 2, -2)
#: Where the grasp comes from (`GraspSettings.grasp_source`): the block's geometry or its grasp library.
GEOMETRY = "geometry"
LIBRARY = "library"
GRASP_SOURCES = (GEOMETRY, LIBRARY)
#: Minimum bite of the shallower fingertip below the top [m], after accounting for tip swing.
MIN_BITE_M = 0.003
#: Jaw-travel, squeeze and bite rejection labels checked before reach.
SKIPS = ("szczeka", "scisk", "zbyt plytko")
#: Hand tilts from straight down are tried in steps of this [deg].
TILT_STEP_DEG = 5.0
#: The largest `max_tilt_deg` [deg].
MAX_HAND_TILT_DEG = 45.0
#: Random IK starts for a tilted hand: fewer than straight down, so the search over tilts stays short.
TILT_RESTARTS = 2
#: Why no tilted grasp was tried: the block's yaw is unknown.
NEEDS_YAW = "dlon pochylona potrzebuje obrotu bloku"
#: A held jaw whose gap is narrower than the chosen grasp width by more than this grips an edge or a
#: corner [m]. A wider gap is a full grip.
EDGE_GRIP_M = 0.006
#: `jaw_file` keeps a running mean of the jaw calibration over at most this many verified lifts.
JAW_RUNS = 10
#: A verified lift whose held gap differs from the block's width by more than this does not teach the jaw
#: calibration [m].
JAW_OFFSET_MAX_M = 0.02
#: A straddle that stalls with the measured TCP this close to the grasp target closes there [m].
CLOSE_NEAR_M = 0.005

#: Phases of a run after touch's (`TouchStatus.phase`).
RISE = "rise"
OPEN = "open"
STRADDLE = "straddle"
STRADDLED = "straddled"
CLOSE = "close"
LIFT_OFF = "lift_off"
CHECK = "check"
#: Monitor the verified hold at lift height for `carry_s`.
CARRY = "carry"
#: Wait for a sighting at the safe height with the jaw open.
RECOVER = "recover"
#: Open the jaw after ending hold monitoring.
RELEASE = "release"
RETARGET = "retarget"
#: Open the jaw on an edge grip before rising for another attempt.
REOPEN = "reopen"
#: A rise target: straddle again at a corrected aim with the jaw kept open (`_start_rise`).
RESTRADDLE = "restraddle"
#: The phases' names in the panel's notes.
GRASP_PHASE_NAMES = {**PHASE_NAMES, RISE: "w gore znad bloku", OPEN: "nad blok", STRADDLE: "w dol na blok",
                     STRADDLED: "szczeka na bloku", CLOSE: "zamykam szczeke", LIFT_OFF: "podnosze blok",
                     CHECK: "trzymam blok", CARRY: "niose blok", RECOVER: "blok wypadl",
                     RELEASE: "oddaje blok", REOPEN: "otwieram szczeke"}

#: Consecutive empty readings required to declare a drop. Ignore an isolated bad reading.
EMPTY_TICKS = 2
#: Minimum delay from drop declaration to a usable capture [s], allowing the block to land.
DROP_SETTLE_S = 0.3
#: Distance from the fingertip midpoint for a simulated pose to count as near the jaw [m].
IN_HAND_M = 0.06
#: Exact sources accepted for a regrasp. Simulation poses use height and jaw state.
SIGHTING_SOURCES = ("kamery", "ostatnie widziane")
#: Pose classifications used by hold monitoring.
IN_HAND = "in_hand"
SEEN = "seen"
STALE = "stale"

#: Display text for each grasp verdict.
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
    "edge": "szczeka chwycila blok za krawedz",
    "off": "kamera widzi blok poza srodkiem szczek",
}


@dataclass(frozen=True)
class Sighting:
    """A block pose classified for hold monitoring and recovery.

    `IN_HAND` means the tracker reports a hold, or the simulated pose is near squeezing jaws.
    `SEEN` means a plain camera or last-seen pose is fresh enough and at rest height.
    `STALE` covers other poses. `seen_t = obs.t - age_s` is capture time on the run clock [s].
    """

    kind: str
    #: Position [m, base frame] and rotation (3x3, identity if absent or invalid).
    xy: tuple[float, float]
    z: float
    rotation: np.ndarray
    source: str
    age_s: float
    seen_t: float


@dataclass
class GraspSettings(TouchSettings):
    #: Key in `obs.objects`, normalized to lower case and single spaces. Empty uses `block`.
    label: str = ""
    #: Bottom TCP height of the probe [m above the table, base frame]. Ignored without `probe`.
    #: Zero uses the block dimensions and jaw reach. Explicit heights must be at least `MIN_TOUCH_Z`.
    touch_z: float = 0.0
    #: Hover offset above the probe's bottom TCP height [m].
    #: Without `probe`, the block dimensions set this reference height.
    #: A lower hover keeps more downward poses reachable.
    hover_m: float = 0.04
    #: Time limit for the whole run, including recovery [s]. Zero disables it.
    time_limit_s: float = 120.0
    #: Block name for `blocks.block`, which supplies dimensions and yaw symmetry.
    block: str = field(default="cube", metadata={"choices": tuple(BLOCKS)})
    #: Measure the top with a closed-jaw descent before the straddle. False uses the catalog top height.
    #: Without the probe, `close_jaw`, `touch_z` and `need_contact` have no effect.
    probe: bool = False
    #: Each open tip clears the block's short side by this much when the jaw straddles it [m].
    open_margin_m: float = 0.008
    #: Squeeze below the block's short side [m]. It must exceed the jaw rule's holding margin.
    squeeze_m: float = 0.008
    #: Depth of the tip centres below the top as a fraction of block height (0, 1].
    grasp_depth: float = 0.5
    #: How far the straddle may push past a light touch before it counts as a stall [m].
    straddle_push_m: float = 0.004
    #: A straddle that stalls with both jaw tips at least this far below the block's top closes the jaw
    #: there [m]; 0 turns it off. A stall with a tip higher is a tip on the block: rise and retry beside.
    close_on_stall_m: float = 0.003
    #: How far straight up the block is lifted before the final check [m].
    lift_m: float = 0.04
    #: How long the jaw must hold at the top before the run succeeds [s].
    hold_s: float = 1.0
    #: Maximum attempts from the hover, including tip retries and regrasps after drops [count].
    max_attempts: int = 3
    #: Carry duration after the verified lift [s]. Zero finishes at the lift check.
    carry_s: float = 10.0
    #: Open the jaw after `carry_s`. False finishes with the block held for the operator.
    release: bool = False
    #: Time from a drop to a usable sighting before stopping for a person [s].
    wait_s: float = 20.0
    #: Maximum sighting distance from the drop point for a regrasp [m].
    recover_m: float = 0.15
    #: Candidate source [geometry|library]. A pose without orientation uses geometry.
    grasp_source: str = field(default=GEOMETRY, metadata={"choices": GRASP_SOURCES})
    #: Library folder [path]. Empty uses bundled files. Missing or invalid libraries refuse the run.
    library_dir: str = ""
    #: Library grasps whose generator score lies below this are skipped [0..1].
    min_score: float = 0.0
    #: Largest tilt of the hand from straight down [deg], tried when straight down is out of reach. The
    #: tilted hand needs the block square to the arm. Zero, or `probe: true`, keeps the hand straight down.
    max_tilt_deg: float = 0.0
    #: How much wider the model's jaw gap is than the real gap between the pads at the same gripper reading
    #: [m]: this stand's gripper calibration, 0 in simulation. Every opening, close and release uses it.
    jaw_gap_offset_m: float = 0.0
    #: JSON file with this stand's jaw calibration learned from verified lifts; once it holds one, it
    #: replaces `jaw_gap_offset_m`. Empty keeps none [path].
    jaw_file: str = ""


@dataclass
class GraspStatus:
    """Grasp state alongside `TouchStatus` for logs and tests."""

    #: The block centre the run aims at [m, base frame]; None until the pose is taken.
    block_xy: tuple[float, float] | None = None
    #: The closing heading of the top-down hand [deg, base frame].
    heading_deg: float = float("nan")
    #: The block's top measured by the probe [m above the table]. NaN without `probe`.
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
    #: Closes that held the block with a gap far from its width (`EDGE_GRIP_M`).
    edge_closes: int = 0
    #: Straddles that stalled with the jaw tips low beside the block and closed there.
    low_closes: int = 0
    #: The jaw calibration in use [m] and how many lifts taught it; 0 runs for the setting.
    jaw_offset_m: float = 0.0
    jaw_runs: int = 0
    #: How far the measured TCP rose above the grasp point [m].
    lifted_m: float = float("nan")
    #: Grasp verdict key in `VERDICTS`, or '' before a verdict.
    verdict: str = ""
    #: Latest monitored pose source, or '' without a valid pose.
    block_source: str = ""
    #: Drop, regrasp and jaw-creep counts.
    drops: int = 0
    recoveries: int = 0
    creeps: int = 0
    #: Camera sighting away from squeezing jaws [m, base frame]. None until seen.
    elsewhere: tuple[float, float] | None = None
    #: How long the verified block has been carried [s].
    carried_s: float = 0.0
    #: Last drop time [s, `obs.t`] and elapsed time on the latest recovery tick [s].
    dropped_t: float = float("nan")
    lost_s: float = float("nan")
    #: `release()` was commanded.
    released: bool = False
    #: Last candidate source, including geometry fallback. Empty before the first aim.
    grasp_source: str = ""
    #: Candidate count and filter rejections for the last aim.
    candidates: int = 0
    rejected: str = ""
    #: Jaw-rule rejection counts before reach checking.
    skipped: str = ""
    #: Chosen library index and score. Geometry uses -1 and NaN.
    library_index: int = -1
    library_score: float = float("nan")
    #: How far below the block's top the fingertips' midpoint goes [m], and its offset from the block centre
    #: [m, base frame].
    depth_m: float = float("nan")
    offset_xy: tuple[float, float] = (0.0, 0.0)
    #: How far the hand leans from straight down for this grasp [deg].
    tilt_deg: float = 0.0
    #: The last approach's horizontal leg into the open-jaw hover [m]; 0 for the direct way.
    lead_m: float = 0.0


@dataclass(frozen=True)
class GraspCandidate:
    """A candidate heading, jaw midpoint, width and score for `_aim`."""

    #: The closing heading of the top-down hand [rad, base frame].
    psi: float
    #: The fingertips' midpoint minus the block centre, horizontally [m, base frame].
    offset_xy: tuple[float, float]
    #: How far below the block's top the fingertips' midpoint goes [m].
    depth_m: float
    #: The block's width between the jaws [m]: the openings follow from it.
    width_m: float
    #: The library's score; 1 for a geometry grasp.
    score: float
    #: "geometry", or "library:<index>" for the library's grasp of that index.
    origin: str


@dataclass(frozen=True)
class JawPlan:
    """Jaw openings [m], commands [app units], open-tip midpoint [m, TCP frame] and tip swing [m]."""

    gap_open_m: float
    gap_closed_m: float
    cmd_open: float
    cmd_closed: float
    cmd_block: float
    centre_open: np.ndarray
    swing_m: float


class GraspStrategy(TouchStrategy):
    """Grasps and lifts a block from above; the jaw rule checks the hold on the way up."""

    settings = GraspSettings
    phase_names = GRASP_PHASE_NAMES

    def __init__(self, settings: GraspSettings | None = None):
        config = settings if settings is not None else GraspSettings()
        if config.block not in BLOCKS:
            raise ValueError(f"grasp: block must be one of {', '.join(BLOCKS)}, is {config.block!r}")
        # Validate here so errors use grasp's messages.
        if config.probe and not (config.touch_z == 0 or config.touch_z >= MIN_TOUCH_Z):
            raise ValueError(f"grasp: touch_z must be zero (from the block) or at least {MIN_TOUCH_Z} m, "
                             f"is {config.touch_z}")
        for name in ("open_margin_m", "squeeze_m", "lift_m", "hold_s", "wait_s", "recover_m",
                     "straddle_push_m"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"grasp: {name} must be positive, is {value}")
        if not config.close_on_stall_m >= 0:
            raise ValueError(f"grasp: close_on_stall_m must be zero or more, is {config.close_on_stall_m}")
        if not config.carry_s >= 0:
            raise ValueError(f"grasp: carry_s must be zero or more, is {config.carry_s}")
        if not 0 < config.grasp_depth <= 1:
            raise ValueError(f"grasp: grasp_depth must lie within (0, 1], is {config.grasp_depth}")
        if not config.max_attempts >= 1:
            raise ValueError(f"grasp: max_attempts must be at least 1, is {config.max_attempts}")
        if config.probe and not config.close_jaw:
            raise ValueError("grasp: close_jaw must be on: the probe needs closed tips")
        if config.grasp_source not in GRASP_SOURCES:
            raise ValueError(f"grasp: grasp_source must be one of {', '.join(GRASP_SOURCES)}, is "
                             f"{config.grasp_source!r}")
        if not 0 <= config.min_score <= 1:
            raise ValueError(f"grasp: min_score must lie within [0, 1], is {config.min_score}")
        if not 0 <= config.max_tilt_deg <= MAX_HAND_TILT_DEG:
            raise ValueError(f"grasp: max_tilt_deg must lie within [0, {MAX_HAND_TILT_DEG:.0f}], is "
                             f"{config.max_tilt_deg}")
        spec = block(config.block)
        #: Block geometry in its body frame (x long, y short, z height) and yaw symmetry.
        self.block: BlockSpec = spec
        #: Validate the block library before a run. None in geometry mode.
        self.library: GraspLibrary | None = None
        if config.grasp_source == LIBRARY:
            folder = Path(config.library_dir).expanduser() if config.library_dir else None
            try:
                self.library = load_library(config.block, folder)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"grasp: {exc}") from exc
        #: The axis a cylinder's library grasps are rolled about to point down; None for a box.
        self._axis = roll_axis(spec)
        self.short_m = 2 * spec.half_size[1]
        self.height_m = 2 * spec.half_size[2]
        #: The picked block's bottom above the table [m]: 0 on the table, the top of the block under it
        #: on a stack.
        self.base_z = 0.0
        #: Maximum centre height accepted as resting on the table [m, base frame].
        self.rest_z_max = max(spec.half_size) + 0.01
        #: The jaw's reach below the TCP [m], from the model once `start` has it.
        self._reach_m = JAW_REACH_M
        #: Original settings. `config` holds the resolved probe height and label.
        self.grasp_settings = config
        # Without the probe, block dimensions set the hover reference.
        self._auto_touch_z = config.touch_z == 0 or not config.probe
        label = normalised(config.label) or config.block
        super().__init__(replace(config, touch_z=self._probe_z(), label=label))

        self.grasp = GraspStatus()
        #: The rotation of the block's pose in the base frame, identity until a pose is taken.
        self.block_rotation = np.eye(3)
        #: TCP rotation and closing heading [rad, base frame]. None / NaN before latching a pose.
        self._rot: np.ndarray | None = None
        self._psi = float("nan")
        #: The hand's tilt about its closing axis [rad], signed as in `hand_down`, and the closing heading
        #: against the direction of the TCP from the base's turning axis [rad].
        self._tilt = 0.0
        self._psi_rel = float("nan")
        #: `_psi_rel` of the grasp: a held block stays level while the hand keeps it.
        self._grip_rel = float("nan")
        #: Where the base's turning axis crosses the table [m, base frame], from the model in `start`.
        self._pan_xy = np.zeros(2)
        #: The last aim's grasp passed `plan_problem`, and why it tried no tilted grasp, or ''.
        self._fitted = False
        self._tilt_why = ""
        #: The candidate headings `_aim` tries, in order [rad].
        self._headings: list[float] = []
        #: Original block centre [m, base frame] and retry index.
        self._block_xy0: np.ndarray | None = None
        self._shift_k = 0
        self._retarget_xy: np.ndarray | None = None
        self._jaw: JawMonitor | None = None
        self._reading: JawReading | None = None
        self._g = -1
        self._jaw_goal_q = float("nan")
        self._grip_step_q = float("inf")
        self._jaw_tol_q = 0.0
        self._jaw_on_command = True
        #: Openings [m] and commands [app units], initially sized for the short side.
        self.gap_open_m = self.short_m + 2 * config.open_margin_m
        self.gap_closed_m = self.short_m - config.squeeze_m
        self.cmd_open = self.cmd_closed = self.cmd_block = self.cmd_probe = float("nan")
        #: Chosen candidate and its width, midpoint depth and offset [m]. None before aiming.
        self._cand: GraspCandidate | None = None
        self._width_m = self.short_m
        self._depth_m = config.grasp_depth * self.height_m
        self._offset_xy = (0.0, 0.0)
        #: The fingertips' midpoint in the TCP frame for the probe's and the open jaw [m].
        self._centre_probe = np.zeros(3)
        self._centre_open = np.zeros(3)
        self._rise_then = OPEN
        self._rise_z0 = float("nan")
        self._rise_point = np.zeros(3)
        #: A sideways move at the current height before the rise [m, base frame], or None.
        self._rise_ease: np.ndarray | None = None
        # Straddle and lift TCP targets [m, base frame].
        self._straddle_xy = np.full(2, np.nan)
        self._grasp_z = float("nan")
        #: How far the lift raises the block [m]: `lift_m`, or less off another block when out of reach.
        self._lift_m = config.lift_m
        self._lift_off_point = np.zeros(3)
        #: The jaw calibration in use [m] (`jaw_gap_offset_m`, `jaw_file`), and whether this run taught it.
        self._jaw_offset = 0.0
        self._jaw_learned = False
        #: Where the TCP closes the jaw [m, base frame]: the grasp target, or where a low stall held it.
        self._close_at = np.full(3, np.nan)
        #: When the straddle stalled low beside the block, to close where it stands [s, `obs.t`]; NaN if not.
        self._low_stall_t = float("nan")
        #: Lift ticks spent moving the target or waiting for the supervisor's command [count].
        self._lift_ticks = 0
        # Initial closing angle [rad] and progress toward a settled jaw.
        self._q_open_entry = float("nan")
        self._jaw_moved = False
        self._still_ticks = 0
        #: When the hover was reached before the hand turned top-down [s, `obs.t`].
        self._turn_t0 = float("nan")
        #: Where the last horizontal leg into the open-jaw hover starts [m, base frame], or None; whether the
        #: hand still has to go there; and the straddle hover the last leg ended at.
        self._lead_point: np.ndarray | None = None
        self._lead_pending = False
        self._leg_end: np.ndarray | None = None
        #: Where the TCP was sent when the approach began [m, base frame].
        self._approach_from = np.zeros(3)
        # Hover hooks remain pending until they return None.
        self._before_descent_pending = False
        self._before_straddle_pending = False
        self._before_close_pending = False
        self._descent_waited = False
        # Last monitored step, empty streak, drop time [s, obs.t], drop position [m, base frame]
        # and recovery opening [rad].
        self._watched_step = -1
        self._empty_ticks = 0
        self._drop_t = float("nan")
        self._drop_xy: np.ndarray | None = None
        self._recover_open_q = float("nan")
        self._phases.update({RISE: self._rise, OPEN: self._open, STRADDLE: self._straddle,
                             STRADDLED: self._straddled, CLOSE: self._close, LIFT_OFF: self._lift_off,
                             CHECK: self._check, CARRY: self._carry, RECOVER: self._recover,
                             RELEASE: self._release, REOPEN: self._reopen})

    def _probe_z(self) -> float:
        """Return the probe's bottom TCP height [m above the table, base frame].

        With the probe on, an explicit `touch_z` overrides the nominal top minus `PROBE_BELOW_M`
        plus the jaw's reach."""
        if not self._auto_touch_z:
            return float(self.grasp_settings.touch_z)
        return max(MIN_TOUCH_Z, self.top_nominal - PROBE_BELOW_M + self._reach_m)

    @property
    def top_nominal(self) -> float:
        """Nominal top above the table [m], used until a probe measures it."""
        return self.base_z + self.height_m

    def _rest_at(self, base: float) -> None:
        """Set the picked block's bottom [m above the table]: the table up to `ON_BLOCK_M`."""
        base = float(base) if base > ON_BLOCK_M else 0.0
        if base != self.base_z:
            self.base_z = base
            if self._auto_touch_z:
                self.config = replace(self.config, touch_z=self._probe_z())

    def _grasp_floor(self, cmd_open: float, tilt: float = 0.0) -> float:
        """The lowest TCP height of the close [m]: above the table, or with the jaw opened to `cmd_open`
        [app units] and the hand tilted by `tilt` [rad] `STACK_CLEAR_M` above the block under the picked
        one."""
        if self.base_z <= 0:
            return MIN_GRASP_Z
        return max(MIN_GRASP_Z, self.base_z + self.kin.jaw_reach(cmd_open, tilt=tilt) + STACK_CLEAR_M)

    # ------------------------------------------------------------------- start
    def start(self, run: RunInfo, obs: Observation) -> None:
        """Initialize the probe and reject openings that the jaw cannot reach or distinguish as a hold."""
        self.grasp = GraspStatus()
        self._load_jaw()
        self._jaw_learned = False
        super().start(run, obs)
        kin, spec, cfg = self.kin, self.kin.spec, self.grasp_settings
        if spec.gripper is None or spec.gripper_range_q is None:
            raise RuntimeError("grasp: the arm has no gripper")
        # Reset to short-side openings so a previous library choice cannot carry into this run.
        self.gap_open_m = self.short_m + 2 * cfg.open_margin_m
        self.gap_closed_m = self.short_m - cfg.squeeze_m
        self._cand, self._width_m, self._offset_xy = None, self.short_m, (0.0, 0.0)
        self._depth_m = cfg.grasp_depth * self.height_m
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
        stacked, self.base_z = self.base_z != 0.0, 0.0
        if self.config.hover_m != cfg.hover_m:
            self.config = replace(self.config, hover_m=cfg.hover_m)
        if self._auto_touch_z and (stacked or abs(self._reach_m - JAW_REACH_M) > 1e-4):
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

        self._jaw = JawMonitor(jaw, closed_q=float(self._lo[g]))
        self._reading = None
        # Leave speed margin so the supervisor can follow each target.
        vel = float(run.max_vel.get(spec.gripper, spec.max_vel.get(spec.gripper, np.inf)))
        self._grip_step_q = (vel * SPEED_SHARE / self.hz * (opened - closed) / 100.0 if vel > 0
                             else float("inf"))
        self._jaw_tol_q = JAW_COMMAND_TOL * (opened - closed) / 100.0
        self._jaw_goal_q = float(self.q_cmd[g])
        self._jaw_on_command = True
        # Open far enough for the tracker to accept nearby detections despite servo error.
        self._recover_open_q = min(float(self._hi[g]), max(self._q(self.cmd_open), jaw.open + 2 * jaw.margin))
        self._watched_step, self._empty_ticks = -1, 0
        self._drop_t, self._drop_xy = float("nan"), None

        self.grasp = GraspStatus(jaw_offset_m=self.grasp.jaw_offset_m, jaw_runs=self.grasp.jaw_runs)
        self.block_rotation = np.eye(3)
        self._rot, self._psi, self._headings = None, float("nan"), []
        self._tilt, self._psi_rel, self._grip_rel = 0.0, float("nan"), float("nan")
        self._pan_xy = kin.joint_anchor(spec.arm_joints[0], spec.home)[:2].copy()
        self._block_xy0, self._shift_k, self._retarget_xy = None, 0, None
        self._straddle_xy, self._grasp_z = np.full(2, np.nan), float("nan")
        self._close_at, self._low_stall_t = np.full(3, np.nan), float("nan")
        self._lift_m = cfg.lift_m
        self._turn_t0 = float("nan")
        self._lead_point, self._lead_pending, self._leg_end = None, False, None
        self._before_descent_pending = self._before_straddle_pending = self._descent_waited = False
        self._before_close_pending = False

    def _q(self, gripper: float) -> float:
        """A gripper command [app units] as the gripper joint's angle [rad]."""
        spec = self.kin.spec
        assert spec.gripper is not None
        return float(self.kin.to_q({spec.gripper: gripper})[self._g])

    # --------------------------------------------------------------------- tick
    def act(self, obs: Observation) -> Action:
        """Update the jaw reading and candidate headings before running the touch tick."""
        if self._kin is None or self._jaw is None:
            raise RuntimeError("grasp: act() przed start()")
        obs = self._observed(obs)
        kin, g = self.kin, self._g
        q_measured = kin.to_q(obs.joints)
        # Missing command joints fall back to the last target.
        q_command = kin.to_q({**kin.from_q(self.q_cmd), **obs.command})
        self._reading = self._jaw.update(float(q_measured[g]), float(q_command[g]))
        self._jaw_on_command = abs(float(q_command[g] - self.q_cmd[g])) <= self._jaw_tol_q
        self.grasp.holding = self._reading.holding
        fixed, moving = kin.fingertips(obs.joints)
        self.grasp.jaw_gap_m = float(np.linalg.norm(moving - fixed))
        if self.status.phase == WAIT:
            self._take_heading(obs)
        return self._commanded(super().act(obs))

    # ------------------------------------------------------------------ the jaw's calibration
    def _observed(self, obs: Observation) -> Observation:
        """`obs` with the gripper's reading and command turned into the model's (`jaw_gap_offset_m`)."""
        name = self.kin.spec.gripper
        if not self._jaw_offset or name is None:
            return obs
        joints, command = dict(obs.joints), dict(obs.command)
        for values in (joints, command):
            if name in values:
                values[name] = self._regrip(float(values[name]), -self._jaw_offset)
        return replace(obs, joints=joints, command=command)

    def _commanded(self, action: Action) -> Action:
        """`action` with the gripper's target turned from the model's into the real gripper's."""
        name = self.kin.spec.gripper
        if not self._jaw_offset or name is None or name not in action.target:
            return action
        target = dict(action.target)
        target[name] = self._regrip(float(target[name]), self._jaw_offset)
        return replace(action, target=target)

    def _regrip(self, gripper: float, wider_m: float) -> float:
        """The gripper value [app units] whose model jaw gap is `wider_m` [m] wider than at `gripper`'s."""
        kin = self.kin
        lo, hi = kin.jaw_gap(0.0), kin.jaw_gap(100.0)
        return kin.gripper_for_gap(min(max(kin.jaw_gap(gripper) + wider_m, lo), hi))

    def _load_jaw(self) -> None:
        """Take the jaw calibration from `jaw_file` when it holds one, else from `jaw_gap_offset_m`."""
        cfg = self.grasp_settings
        self._jaw_offset, runs = float(cfg.jaw_gap_offset_m), 0
        if cfg.jaw_file:
            try:
                data = json.loads(Path(cfg.jaw_file).expanduser().read_text())
                offset, runs = float(data["jaw_gap_offset_m"]), int(data["runs"])
                if math.isfinite(offset) and runs > 0:
                    self._jaw_offset = offset
                else:
                    runs = 0
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Nie odczytano kalibracji szczeki z %s: %s", cfg.jaw_file, exc)
                runs = 0
        self.grasp.jaw_offset_m, self.grasp.jaw_runs = self._jaw_offset, runs

    def _learn_jaw(self) -> None:
        """Add this lift's held gap minus the block's width to `jaw_file`'s running mean."""
        cfg = self.grasp_settings
        sample = self.grasp.jaw_gap_m + self._jaw_offset - self._width_m
        if not cfg.jaw_file or self._jaw_learned or not abs(sample) <= JAW_OFFSET_MAX_M:
            return
        self._jaw_learned = True
        runs = min(self.grasp.jaw_runs + 1, JAW_RUNS)
        mean = self._jaw_offset + (sample - self._jaw_offset) / runs if self.grasp.jaw_runs else sample
        try:
            Path(cfg.jaw_file).expanduser().write_text(json.dumps({"jaw_gap_offset_m": mean, "runs": runs}))
        except OSError as exc:
            logger.warning("Nie zapisano kalibracji szczeki do %s: %s", cfg.jaw_file, exc)
            return
        logger.info("Kalibracja szczeki: %.1f mm (srednia z %d)", mean * 1000, runs)

    @property
    def reading(self) -> JawReading:
        """Return this tick's jaw reading."""
        if self._reading is None:
            raise RuntimeError("grasp: no jaw reading before the first tick")
        return self._reading

    def _take_heading(self, obs: Observation) -> None:
        """Take candidate headings and the block's bottom from the pose that `_wait` will latch this tick."""
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        if pose is None or self._pose_problem(pose):
            return
        self._rest_at(float(np.asarray(pose.position, float)[2]) - self.block.half_size[2])
        rotation = np.asarray(pose.rotation, float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            rotation = np.eye(3)
        self._heading_from(rotation)

    def _heading_from(self, rotation: np.ndarray) -> None:
        """Store the block rotation and list candidate headings nearest the current heading first."""
        self.block_rotation = np.asarray(rotation, float).copy()
        self._headings = grasp_headings(self.block_rotation, self.block.yaw_period, self._heading_now())

    def _heading_now(self) -> float:
        """The closing heading of the last target [rad, base frame]."""
        return heading(self.kin, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------- aiming
    def _aim(self, xy: np.ndarray, *, strict: bool = False) -> None:
        """Choose the first candidate that passes jaw rules and reaches every planned point.

        Straight down first, then each tilt from `_tilts`. With more than one tilt, a grasp must also pass
        `plan_problem`; when none does, the first reachable straight-down grasp is taken, or with `strict`
        the aim raises. A block without yaw is tried tilted only when `_measures_yaw`. If none works, raise
        while retaining the aim and updating rejection status.
        """
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise RuntimeError("punkt poza zasiegiem ramienia: wspolrzedne nieskonczone")
        candidates = self.candidates(self.block_rotation)
        top = self.grasp.top_m if np.isfinite(self.grasp.top_m) else self.top_nominal
        skips: dict[str, int] = dict.fromkeys(SKIPS, 0)
        plans: dict[float, JawPlan | str] = {}
        first_failure: tuple[np.ndarray, dict[str, float], IKSolution] | None = None
        fallback: tuple[GraspCandidate, JawPlan, dict[str, np.ndarray], float, float] | None = None
        misfit = ""
        tilts = self._tilts()
        self._fitted, self._tilt_why = False, ""
        # Without the block's yaw any heading grips it straight down, and a tilted hand needs one square to
        # the arm: that is a guess until a look measures the yaw.
        free = np.allclose(self.block_rotation, np.eye(3), atol=1e-6) and candidates[0].origin == GEOMETRY
        blind = free and not self._measures_yaw
        bearing = self._bearing(xy)
        square = [replace(candidates[0], psi=wrap(bearing + side)) for side in (-math.pi / 2, math.pi / 2)]
        for k, size in enumerate(tilts):
            if size and blind:
                break
            for cand in square if size and free else candidates:
                jaw = self._candidate_jaw(cand, plans)
                if isinstance(jaw, str):
                    skips[jaw] += k == 0
                    continue
                if cand.origin != GEOMETRY and cand.depth_m - jaw.swing_m / 2 < MIN_BITE_M:
                    # The shallower tip lies half the swing above the planned midpoint.
                    skips["zbyt plytko"] += k == 0
                    continue
                tilt = self._signed_tilt(cand.psi, xy, size)
                if tilt is None:
                    continue
                grasp_z = self._grasp_height(top, cand.depth_m, jaw.centre_open, jaw.cmd_open, tilt)
                if self.base_z > 0:
                    # Over another block the floor may cut the depth: the shallower tip must still bite.
                    swing = jaw.swing_m if math.isfinite(jaw.swing_m) else self.kin.tip_swing(jaw.cmd_open)
                    mid_z = grasp_z - float(jaw.centre_open[0]) * math.cos(tilt)
                    if top - mid_z - swing / 2 < MIN_BITE_M:
                        skips["zbyt plytko"] += k == 0
                        continue
                if tilt and self.plan_problem(xy, cand, jaw, tilt, grasp_z,
                                              self._plan(xy, cand, jaw, grasp_z, tilt=tilt)):
                    # A tilt is tried for the rest of the run, so that check comes first: it fails sooner.
                    continue
                plan, failure = self._reachable(xy, cand, jaw, grasp_z, top, tilt)
                if failure is not None:
                    first_failure = first_failure or failure
                    continue
                problem = ""
                if len(tilts) > 1 and not tilt:
                    problem = self.plan_problem(xy, cand, jaw, tilt, grasp_z, plan)
                if not problem:
                    self._fitted = len(tilts) > 1
                    self.grasp.skipped = rejections_text(skips, SKIPS)
                    self._take_plan(xy, cand, jaw, plan, grasp_z, tilt)
                    return
                misfit = misfit or problem
                fallback = fallback or (cand, jaw, plan, grasp_z, tilt)
        self.grasp.skipped = rejections_text(skips, SKIPS)
        if len(tilts) > 1 and blind:
            self._tilt_why = NEEDS_YAW
        if fallback is not None:
            if not strict:
                self._take_plan(xy, *fallback)
                return
            reason = misfit
        elif first_failure is not None:
            reason = self._unreachable(*first_failure)
        else:
            raise RuntimeError(f"biblioteka chwytow '{self.block.name}': zaden chwyt nie pasuje do szczeki "
                               f"({self.grasp.skipped})")
        raise RuntimeError("; ".join(filter(None, (reason, self._tilt_why))))

    def plan_problem(self, xy: np.ndarray, cand: GraspCandidate, jaw: JawPlan, tilt: float, grasp_z: float,
                     plan: dict[str, np.ndarray]) -> str:
        """Why the rest of the run does not fit a reachable grasp at hand tilt `tilt` [rad], or ''.

        `_aim` asks only when it may try more than one tilt. The default accepts every grasp."""
        return ""

    @property
    def _measures_yaw(self) -> bool:
        """Whether a look before each straddle measures the block's yaw. Plain grasp has none."""
        return False

    def _refuse(self, reason: str) -> Action:
        """End the run before the straddle: the grasp cannot be aimed safely."""
        return self._halt(reason, self.kin.from_q(self.q_cmd))

    def _tilts(self) -> list[float]:
        """Hand tilts to try [rad]: straight down, then `TILT_STEP_DEG` steps up to `max_tilt_deg`."""
        cfg = self.grasp_settings
        most = 0.0 if cfg.probe else cfg.max_tilt_deg
        steps = [k * TILT_STEP_DEG for k in range(1, math.ceil(most / TILT_STEP_DEG - 1e-9))]
        return [0.0] + [math.radians(t) for t in [*steps, most] if most > 0]

    def _bearing(self, xy: np.ndarray) -> float:
        """The direction of `xy` from the base's turning axis [rad, base frame]."""
        d = np.asarray(xy, float)[:2] - self._pan_xy
        return math.atan2(float(d[1]), float(d[0]))

    def _signed_tilt(self, psi: float, xy: np.ndarray, size: float) -> float | None:
        """Return tilt `size` [rad] signed so the hand at heading `psi` over `xy` leans away from the base.

        None when the heading lies too far from square to the arm for the arm to tilt the hand that much.
        The arm's reach check decides within that."""
        if not size:
            return 0.0
        off = self._bearing(xy) - psi
        if math.sin(size) * abs(math.cos(off)) > math.sin(math.radians(2 * MAX_ROT_ERR_DEG)):
            return None
        return math.copysign(size, math.sin(off))

    def candidates(self, rotation: np.ndarray) -> list[GraspCandidate]:
        """Return geometry or library candidates in attempt order.

        Geometry uses the short side, configured depth and heading choices. Library mode composes
        with `rotation` and selects candidates for the jaw. Identity rotation falls back to geometry.
        Raise with rejection reasons if library selection is empty.
        """
        R = np.asarray(rotation, float)
        cfg, grasp, kin = self.grasp_settings, self.grasp, self.kin
        psi_now = self._heading_now()
        if self.library is None or np.allclose(R, np.eye(3), atol=1e-6):
            grasp.grasp_source = GEOMETRY if self.library is None else f"{GEOMETRY} (brak obrotu)"
            headings = self._headings or grasp_headings(R, self.block.yaw_period, psi_now)
            depth = cfg.grasp_depth * self.height_m
            grasp.candidates, grasp.rejected = len(headings), ""
            return [GraspCandidate(float(psi), (0.0, 0.0), depth, self.short_m, 1.0, GEOMETRY)
                    for psi in headings]
        composed = compose(self.library, R, self.block.yaw_period, axis=self._axis)
        kept, counts = select(composed, psi_now=psi_now, max_tilt_rad=math.radians(MAX_TILT_DEG),
                              min_score=cfg.min_score, gap_min_m=kin.jaw_gap(0.0),
                              gap_max_m=kin.jaw_gap(100.0), open_margin_m=cfg.open_margin_m,
                              squeeze_m=cfg.squeeze_m, half_size=self.block.half_size)
        grasp.grasp_source, grasp.candidates, grasp.rejected = LIBRARY, len(kept), rejections_text(counts)
        if not kept:
            raise RuntimeError(f"biblioteka chwytow '{self.block.name}': zaden z {len(composed)} chwytow nie "
                               f"pasuje ({grasp.rejected})")
        return [GraspCandidate(c.psi, c.offset_xy, c.depth_m, c.width_m, c.score, f"{LIBRARY}:{c.index}")
                for c in kept]

    def _candidate_jaw(self, cand: GraspCandidate, plans: dict[float, JawPlan | str]) -> JawPlan | str:
        """Return a jaw plan or a `SKIPS` reason for `cand`.

        Reuse current openings for geometry at the current width. Cache other plans by width.
        """
        if cand.origin == GEOMETRY and abs(cand.width_m - self._width_m) < 1e-12:
            return JawPlan(self.gap_open_m, self.gap_closed_m, self.cmd_open, self.cmd_closed, self.cmd_block,
                           self._centre_open, float("nan"))
        if cand.width_m not in plans:
            plans[cand.width_m] = self._jaw_plan(cand.width_m)
        return plans[cand.width_m]

    def _jaw_plan(self, width_m: float) -> JawPlan | str:
        """Return a jaw plan across `width_m` [m], or a travel or squeeze rejection.

        Add opening margins and retain any extra widening from a regrasp. Use the unwidened
        midpoint to preserve the fixed-tip target. Close to width minus `squeeze_m`.
        """
        kin, cfg, jaw = self.kin, self.grasp_settings, self.kin.spec.jaw
        plain = width_m + 2 * cfg.open_margin_m
        widened = self.gap_open_m - (self._width_m + 2 * cfg.open_margin_m)
        widened = widened if widened > 1e-9 else 0.0
        try:
            cmd_plain = kin.gripper_for_gap(plain)
            cmd_open = kin.gripper_for_gap(plain + widened) if widened else cmd_plain
            cmd_closed = kin.gripper_for_gap(width_m - cfg.squeeze_m)
            cmd_block = kin.gripper_for_gap(width_m)
        except ValueError:
            return "szczeka"
        if self._q(cmd_block) - self._q(cmd_closed) <= jaw.margin + jaw.still:
            return "scisk"
        return JawPlan(plain + widened, width_m - cfg.squeeze_m, cmd_open, cmd_closed, cmd_block,
                       kin.jaw_centre(cmd_plain), kin.tip_swing(cmd_open))

    def _reachable(self, xy: np.ndarray, cand: GraspCandidate, jaw: JawPlan, grasp_z: float, top: float,
                   tilt: float = 0.0
                   ) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, dict[str, float], IKSolution] | None]:
        """Return the plan with the set hover and lift and its first unreachable point, or None.

        Over another block, lower the hover and the lift step by step to their least while one of them is
        out of reach.
        """
        hover = self.config.touch_z + self.grasp_settings.hover_m
        lift = grasp_z + self.grasp_settings.lift_m
        while True:
            plan = self._plan(xy, cand, jaw, grasp_z, hover, lift, tilt)
            failure = self._reach(plan, cand.psi, jaw.cmd_open, tilt)
            heights = ("probe_hover", "straddle_hover", "lift")
            if (failure is None or self.base_z <= 0
                    or not any(np.array_equal(failure[0], plan[name]) for name in heights)):
                return plan, failure
            low_hover = min(hover, top + self.kin.jaw_reach(jaw.cmd_open, tilt=tilt) + MIN_HOVER_CLEAR_M)
            low_lift = min(lift, grasp_z + MIN_LIFT_M)
            if hover <= low_hover and lift <= low_lift:
                return plan, failure
            hover, lift = max(hover - HEADROOM_STEP_M, low_hover), max(lift - HEADROOM_STEP_M, low_lift)

    def _plan(self, xy: np.ndarray, cand: GraspCandidate, jaw: JawPlan, grasp_z: float,
              hover_z: float | None = None, lift_z: float | None = None, tilt: float = 0.0
              ) -> dict[str, np.ndarray]:
        """Return candidate TCP targets in the base frame [m].

        Probe over the block centre. Place the open jaw at that centre plus the grasp offset. The hover
        and lift heights default to the set ones.
        """
        u = np.array([math.cos(cand.psi), math.sin(cand.psi)])
        n = np.array([-math.sin(cand.psi), math.cos(cand.psi)]) * math.sin(tilt)
        hover_z = self.config.touch_z + self.config.hover_m if hover_z is None else hover_z
        lift_z = grasp_z + self.grasp_settings.lift_m if lift_z is None else lift_z
        # Jaw midpoint = TCP + R @ centre: centre[2] lies along the heading, and a tilt moves centre[0]
        # along n.
        probe_xy = xy - self._centre_probe[2] * u - self._centre_probe[0] * n
        straddle_xy = xy + np.asarray(cand.offset_xy, float) - jaw.centre_open[2] * u - jaw.centre_open[0] * n
        return {"probe_xy": probe_xy, "straddle_xy": straddle_xy,
                "probe_hover": np.array([*probe_xy, hover_z]),
                "probe_bottom": np.array([*probe_xy, self.config.touch_z]),
                "straddle_hover": np.array([*straddle_xy, hover_z]),
                "grasp": np.array([*straddle_xy, grasp_z]),
                "lift": np.array([*straddle_xy, lift_z])}

    def _reach(self, plan: dict[str, np.ndarray], psi: float, cmd_open: float, tilt: float = 0.0
               ) -> tuple[np.ndarray, dict[str, float], IKSolution] | None:
        """Check full IK at each planned point with the hand down, tilted by `tilt` [rad].

        Use `cmd_open` from straddle hover onward. Return None on success, or the first failed
        point, its seed and its solution.
        """
        kin, R = self.kin, hand_down(psi, tilt)
        spec = kin.spec
        assert spec.gripper is not None
        seed = kin.from_q(self.q_cmd)
        names = ("probe_hover", "probe_bottom") if self.grasp_settings.probe else ("probe_hover",)
        for name in (*names, "straddle_hover", "grasp", "lift"):
            if name == "straddle_hover":
                seed = dict(seed, **{spec.gripper: cmd_open})
            point = plan[name]
            sol = kin.ik(point, R, seed=seed, accept=self._clear, restarts=ik_restarts(tilt))
            if not (sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)):
                return point, seed, sol
            seed = sol.joints
        return None

    def _unreachable(self, point: np.ndarray, seed: dict[str, float], sol: IKSolution) -> str:
        """Describe why the arm cannot reach `point` with the hand down."""
        where = f"punkt ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) m"
        if sol.ok and self._clear(sol.joints):
            return f"{where}: chwytak nie ustawi sie pionowo (blad {math.degrees(sol.rot_err):.0f} st.)"
        reachable = self.kin.ik(point, seed=seed, accept=self._within_limits)
        if not (reachable.ok and self._within_limits(reachable.joints)):
            return f"{where} poza zasiegiem ramienia"
        if not self.kin.self_contacts(reachable.joints):
            return f"{where}: chwytak nie siega tam pionowo"
        return f"{where}: ramie uderzyloby samo w siebie"

    def _take_plan(self, xy: np.ndarray, cand: GraspCandidate, jaw: JawPlan, plan: dict[str, np.ndarray],
                   grasp_z: float, tilt: float = 0.0) -> None:
        psi = cand.psi
        self._tilt = float(tilt)
        self._set_heading(psi, plan["straddle_xy"])
        self._grip_rel = self._psi_rel
        self.grasp.tilt_deg = math.degrees(abs(self._tilt))
        if self._tilt:
            logger.info("Dlon pochylona o %.0f st. od pionu", self.grasp.tilt_deg)
        self._headings = [float(psi)]
        self._cand = cand
        self._width_m, self._depth_m, self._offset_xy = cand.width_m, cand.depth_m, cand.offset_xy
        self.gap_open_m, self.gap_closed_m = jaw.gap_open_m, jaw.gap_closed_m
        self.cmd_open, self.cmd_closed, self.cmd_block = jaw.cmd_open, jaw.cmd_closed, jaw.cmd_block
        self._centre_open = np.asarray(jaw.centre_open, float).copy()
        g, jaw_rule = self._g, self.kin.spec.jaw
        # Keep recovery opening beyond the tracker's open threshold.
        self._recover_open_q = min(float(self._hi[g]),
                                   max(self._q(self.cmd_open), jaw_rule.open + 2 * jaw_rule.margin))
        grasp = self.grasp
        grasp.heading_deg = math.degrees(psi)
        grasp.depth_m = float(cand.depth_m)
        grasp.offset_xy = (float(cand.offset_xy[0]), float(cand.offset_xy[1]))
        library = cand.origin.startswith(f"{LIBRARY}:")
        grasp.library_index = int(cand.origin.split(":")[1]) if library else -1
        grasp.library_score = float(cand.score) if library else float("nan")
        self.touch_xy = plan["probe_xy"].copy()
        x, y = (float(v) for v in self.touch_xy)
        self.status.touch_point = (x, y, float(self.config.touch_z))
        grasp.block_xy = (float(xy[0]), float(xy[1]))
        if self._block_xy0 is None:
            self._block_xy0 = xy.copy()
        self._straddle_xy = plan["straddle_xy"].copy()
        self._grasp_z = float(grasp_z)
        cfg = self.grasp_settings
        hover_m = float(plan["straddle_hover"][2]) - self.config.touch_z
        hover_m = cfg.hover_m if abs(hover_m - cfg.hover_m) < 1e-9 else hover_m
        if hover_m != self.config.hover_m:
            self.config = replace(self.config, hover_m=hover_m)
        lift_m = float(plan["lift"][2]) - self._grasp_z
        self._lift_m = cfg.lift_m if abs(lift_m - cfg.lift_m) < 1e-9 else lift_m
        grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)

    def _grasp_height(self, top_m: float, depth_m: float | None = None, centre: np.ndarray | None = None,
                      cmd_open: float | None = None, tilt: float | None = None) -> float:
        """Return closing TCP height [m above the table, base frame], bounded by `_grasp_floor`.

        Place the midpoint `depth_m` below `top_m`, using open-tip `centre` in the TCP frame [m], opening
        `cmd_open` [app units] and hand tilt `tilt` [rad]. Omitted values use the chosen grasp.
        """
        depth = self._depth_m if depth_m is None else depth_m
        centre = self._centre_open if centre is None else centre
        tilt = self._tilt if tilt is None else tilt
        floor = self._grasp_floor(self.cmd_open if cmd_open is None else cmd_open, tilt)
        return max(floor, top_m - depth + float(centre[0]) * math.cos(tilt))

    @property
    def straddle_hover(self) -> np.ndarray:
        """Where the open jaw waits above the block before the straddle [m, base frame]."""
        return np.array([*self._straddle_xy, self.config.touch_z + self.config.hover_m])

    @property
    def grasp_target(self) -> np.ndarray:
        """Where the TCP closes the jaw [m, base frame]."""
        return np.array([*self._straddle_xy, self._grasp_z])

    def _drive(self, goal: np.ndarray, step_m: float, *, rotation: np.ndarray | None = None) -> Action:
        """Run touch's step with the chosen downward orientation after latching the pose.

        A tilted hand keeps its heading against the arm as the arm turns, so a held block stays level."""
        if rotation is None and self._tilt:
            rotation = hand_down(self._bearing(self.tcp_cmd) + self._psi_rel, self._tilt)
        return super()._drive(goal, step_m, rotation=self._rot if rotation is None else rotation)

    def _set_heading(self, psi: float, xy: np.ndarray) -> None:
        """Set the closing heading `psi` [rad] the hand has with its TCP over `xy` [m, base frame]."""
        self._psi, self._rot = float(psi), hand_down(psi, self._tilt)
        self._psi_rel = wrap(float(psi) - self._bearing(xy))

    def _vertical(self) -> bool:
        """Whether the last target's hand has the chosen tilt and heading within `MAX_TILT_DEG`."""
        joints = self.kin.from_q(self.q_cmd)
        limit = math.radians(MAX_TILT_DEG)
        return (abs(tilt(self.kin, joints) - abs(self._tilt)) <= limit
                and abs(wrap(heading(self.kin, joints) - self._psi)) <= limit)

    def _enter(self, phase: str) -> None:
        super()._enter(phase)
        if phase == CARRY and self.grasp.verdict == "lifted":
            self._learn_jaw()
        if phase == APPROACH:
            self._lead_point = self._lead_start()
            self._approach_from = self.tcp_cmd.copy()
        if phase == DESCEND:
            self._before_descent_pending, self._descent_waited = True, False
        if phase == OPEN:
            self._before_straddle_pending = True
        if phase == STRADDLE:
            self._before_close_pending, self._low_stall_t = True, float("nan")

    # ------------------------------------------------------------------ the hover and the probe
    def lead_in(self, xy: np.ndarray) -> np.ndarray | None:
        """The last horizontal leg into the open-jaw hover over block centre `xy` [m, base frame x and y], or
        None for the direct way. The default has none."""
        return None

    def _lead_start(self) -> np.ndarray | None:
        """Where the approach to the current aim starts its last leg into the open-jaw hover, or None: no leg,
        the probe on, the hand tilted, or the start out of reach with the hand down."""
        self.grasp.lead_m = 0.0
        if self.grasp_settings.probe or self._tilt or self.grasp.block_xy is None:
            return None
        leg = self.lead_in(np.asarray(self.grasp.block_xy, float))
        if leg is None or not float(np.linalg.norm(leg)) > self.ik_tol:
            return None
        start = self.straddle_hover - np.array([*np.asarray(leg, float)[:2], 0.0])
        sol = self.kin.ik(start, hand_down(self._psi), seed=self.kin.from_q(self.q_cmd), accept=self._clear)
        if not (sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)):
            return None
        self.grasp.lead_m = float(np.linalg.norm(leg))
        return start

    def _lead_time(self, start: np.ndarray) -> float:
        """How long the hand may take from `start` to the leg's start [s]: twice the travel, and
        `JAW_WAIT_S`."""
        assert self._lead_point is not None
        return 2 * float(np.linalg.norm(self._lead_point - start)) / self.config.speed_mps + JAW_WAIT_S

    def _give_up_leg(self) -> None:
        """The arm did not get to the leg's start in time: go straight on, without the leg."""
        logger.warning("Ramie nie doszlo do poczatku podejscia - podchodze wprost")
        self._lead_point, self.grasp.lead_m = None, 0.0
        # Not again into this hover.
        self._leg_end = self.straddle_hover.copy()

    def _leg_needed(self) -> bool:
        """Whether a leg applies and the hand did not reach the current straddle hover along it. Sets the
        leg's start."""
        end = self._leg_end
        if end is not None and float(np.linalg.norm(end - self.straddle_hover)) <= LEG_SAME_M:
            return False
        self._lead_point = self._lead_start()
        return self._lead_point is not None

    def _approach(self, obs: Observation) -> Action | None:
        """Reach the hover with the hand down, then descend for the probe or open the jaw.

        After recovery, ramp the jaw closed on the way to hover. With the probe on, wait at most
        `JAW_WAIT_S` from phase entry for the target and supervisor command to agree before descending.
        With a leg from `lead_in` the hover is the leg's start, and OPEN runs the leg.
        """
        self._ramp_jaw()
        if self._lead_point is not None and self._t - self._phase_t0 >= self._lead_time(self._approach_from):
            self._give_up_leg()
        hover = self.hover_point if self._lead_point is None else self._lead_point
        # A still arm short of a leg's start gives the leg up in time instead of stopping the run.
        watch = self._lead_point is None
        if self._rot is not None and not self._vertical():
            # Finish turning at the hover within `JAW_WAIT_S`.
            if self._arrived(hover, sag=True, watch=watch):
                if not np.isfinite(self._turn_t0):
                    self._turn_t0 = self._t
                elif self._t - self._turn_t0 >= JAW_WAIT_S:
                    return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
            self._settle_count = 0
            return self._drive(hover, self.approach_step_m)
        self._turn_t0 = float("nan")
        # Keep this check here so `_settled` counts only once per tick.
        if self._settled(hover, sag=True, watch=watch):
            if not self.grasp_settings.probe:
                self.status.attempt += 1
                # OPEN times its travel to the straddle hover from here, along the leg if there is one.
                self._rise_point = hover
                self._lead_pending = False
                if self._lead_point is not None:
                    self._leg_end = self.straddle_hover.copy()
                self._enter(OPEN)
                return None
            if not self._jaw_ready():
                if self._t - self._phase_t0 >= JAW_WAIT_S:
                    return self._halt(f"szczeka nie zamknela sie przed zejsciem w {JAW_WAIT_S:.0f} s",
                                      self.kin.from_q(self.q_cmd))
                return self.hold()
            self._take_reference()
            self.status.attempt += 1
            self._enter(DESCEND)
            return None
        return self._drive(hover, self.approach_step_m)

    def _not_vertical(self) -> str:
        now = math.degrees(tilt(self.kin, self.kin.from_q(self.q_cmd)))
        if self._tilt:
            return f"chwytak nie pochylil sie o {self.grasp.tilt_deg:.0f} st. nad blokiem ({now:.0f} st.)"
        return f"chwytak nie ustawil sie pionowo nad blokiem ({now:.0f} st.)"

    def _descend(self, obs: Observation) -> Action | None:
        if self._before_descent_pending:
            action = self.before_descent(obs)
            if action is not None:
                self._descent_waited = True
                return action
            self._before_descent_pending = False
            if self._descent_waited:
                # Refresh the descent reference after the hook held or moved the arm.
                self._take_reference()
                self._settle_count = 0
        return super()._descend(obs)

    def before_descent(self, obs: Observation) -> Action | None:
        """Run before each probe descent tick until this hook returns None. Never runs without `probe`.

        Overrides may hold or move at hover height and re-aim through `_aim`. Return None only
        when settled at the current hover point. The default returns None immediately."""
        return None

    def after_touch(self, obs: Observation) -> Action:
        """Use a touch to measure the block's top and start the grasp.

        A probe that reaches its bottom uses the nominal top. Other verdicts go to `after_miss`."""
        kind, reason = self.descent_verdict()
        if kind not in ("touched", "reached"):
            return self.after_miss(kind, reason, obs)
        top = self.status.stop_z_m - self._reach_m if kind == "touched" else self.top_nominal
        self.grasp.top_m = float(top)
        self._grasp_z = self._grasp_height(top)
        self.grasp.grasp_point = (float(self._straddle_xy[0]), float(self._straddle_xy[1]), self._grasp_z)
        self._start_rise(OPEN)
        return self.hold("w gore po dotyku")

    def after_miss(self, kind: str, reason: str, obs: Observation) -> Action:
        """Handle a probe verdict of "missed", "blocked" or "stuck".

        A miss after a sideways shift tries the other side while attempts remain.
        Other failures end the run with touch's reason."""
        attempt = self.status.attempt
        if kind == "missed" and self._shift_k >= 1:
            if attempt < self.grasp_settings.max_attempts and self._next_shift() is not None:
                self._start_rise(RETARGET)
                return self.hold("obok pusto - probuje z drugiej strony")
            return self._halt(f"wyczerpane proby chwytu ({attempt}): {reason}", self.kin.from_q(self.q_cmd))
        return self._halt(reason, self.kin.from_q(self.q_cmd))

    # ------------------------------------------------------------------ the grasp
    def _start_rise(self, then: str, *, jaw_open: bool = False, ease: np.ndarray | None = None) -> None:
        """Rise vertically to at least the hover, then enter `then`.

        RECOVER opens to `_recover_open_q`. OPEN and RETARGET close the jaw after clearing the block, unless
        `jaw_open` keeps it at the grasp opening, as for a rise from the open jaw around the block.
        RESTRADDLE then goes to OPEN at the current aim. `ease` [m, base frame x and y] first moves the hand
        sideways at its height, so a jaw beside the block leaves its face before going up.
        """
        self._rise_then = then
        self._rise_z0 = float(self._tcp_measured[2])
        hover_z = self.config.touch_z + self.config.hover_m
        start = self.tcp_cmd.copy()
        self._rise_ease = None
        if ease is not None and float(np.linalg.norm(ease)) > self.ik_tol:
            start[:2] += np.asarray(ease, float)
            self._rise_ease = start.copy()
        self._rise_point = np.array([start[0], start[1], max(hover_z, float(start[2]))])
        self._empty_ticks = 0
        if then == RECOVER:
            self._jaw_goal_q = self._recover_open_q
        elif jaw_open:
            self._set_jaw_goal(self.cmd_open)
        else:
            self._jaw_goal_q = float(self._lo[self._g])
        self._enter(RISE)

    def _rise(self, obs: Observation) -> Action | None:
        # Let the supervisor reach the held pose before rising.
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        if self._rise_ease is not None:
            if not self._arrived(self._rise_ease):
                return self._drive(self._rise_ease, self.descend_step_m)
            self._rise_ease = None
        arrived = self._arrived(self._rise_point, sag=True)
        # Clear the block before closing so the moving tip cannot sweep across its top.
        # Recovery opens the jaw immediately.
        clear = self._tcp_measured[2] >= self._rise_z0 + CLEAR_BEFORE_CLOSE_M
        if self._rise_then == RECOVER or arrived or clear:
            self._ramp_jaw()
        if arrived and self._jaw_ready():
            if self._rise_then in (OPEN, RESTRADDLE):
                self._lead_pending = self._leg_needed()
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
        if self._lead_pending and self._lead_point is not None:
            # Back to the start of the leg into this hover first.
            lead = self._lead_point
            late = self._t - self._phase_t0 >= self._lead_time(self._rise_point)
            # A still arm short of the leg's start gives the leg up in time instead of stopping the run.
            if not self._arrived(lead, sag=True, watch=False) and not late:
                return self._drive(lead, self.approach_step_m)
            if late:
                self._give_up_leg()
            self._lead_pending = False
            self._leg_end = self.straddle_hover.copy()
            # The rest of the phase is timed from here.
            self._rise_point, self._phase_t0, self._settle_count = self.tcp_cmd.copy(), self._t, 0
        reading = self.reading
        settled = self._settled(self.straddle_hover, sag=True)
        jaw_open = abs(reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
        vertical = self._vertical()
        ready = settled and self._jaw_ready() and jaw_open and vertical
        if self._before_straddle_pending and (ready or self._looking):
            action = self.before_straddle(obs, settled=ready)
            if action is not None:
                return action
            self._before_straddle_pending = False
        if ready:
            if self._leg_needed():
                # The look moved the aim: the hand comes into the new hover along the leg again.
                self._lead_pending = True
                self._rise_point, self._phase_t0, self._settle_count = self.tcp_cmd.copy(), self._t, 0
                return self.hold()
            # Refresh the stall reference after moving the TCP and opening the jaw.
            self._take_reference()
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
        """Run at the open-jaw hover once settled, or on every tick while `_looking` is true.

        Return None to finish this hook for the straddle. Descent waits for the arm and jaw
        to settle at the current hover point. The default returns None immediately."""
        return None

    @property
    def _looking(self) -> bool:
        """Whether a subclass is inspecting the block at the straddle hover."""
        return False

    def _straddle(self, obs: Observation) -> Action | None:
        st = self.status
        if math.isfinite(self._low_stall_t):
            # Close where the arm stands, once the supervisor's command has reached the held pose.
            if not self._command_on_target and self._t - self._low_stall_t < HOLD_WAIT_S:
                return self.hold()
            return self._before_close_then(self.tcp_cmd, obs)
        push = self.grasp_settings.straddle_push_m
        st.rise_m = self._contact_rise()
        if st.rise_m > push:
            return self._straddle_stall()
        if self._settled(self.grasp_target):
            if st.rise_m > push / 2:
                return self._straddle_stall()
            return self._before_close_then(self.grasp_target, obs)
        return self._drive(self.grasp_target, self.descend_step_m)

    def _before_close_then(self, at: np.ndarray, obs: Observation) -> Action | None:
        """Run `before_close` until it returns None, then close the jaw with the TCP at `at` [m]."""
        if self._before_close_pending:
            action = self.before_close(obs)
            if action is not None:
                return action
            self._before_close_pending = False
        self._close_at = np.asarray(at, float).copy()
        self.grasp.grasp_point = (float(at[0]), float(at[1]), float(at[2]))
        self._enter(CLOSE)
        self._q_open_entry = self.reading.q
        self._jaw_moved, self._still_ticks = False, 0
        return None

    def before_close(self, obs: Observation) -> Action | None:
        """Run at the bottom of the straddle, settled with the open jaw around the block, until it returns
        None.

        Overrides may hold, or end the straddle through `_start_rise` and return its action. Return None to
        close the jaw. The default returns None immediately."""
        return None

    def _straddle_stall(self) -> Action:
        """Hold the measured pose when the straddle stalls. With the jaw tips low beside the block
        (`close_on_stall_m`), or the TCP within `CLOSE_NEAR_M` of the grasp target, the jaw closes there;
        otherwise a tip stands on the block."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()
        near = float(np.linalg.norm(self._tcp_measured - self.grasp_target)) <= CLOSE_NEAR_M
        if near or self._tips_low():
            self._low_stall_t = self._t
            self.grasp.low_closes += 1
            return self.hold("szczeka nisko przy bloku - zamykam")
        self.grasp.straddle_stalls += 1
        self.grasp.verdict = "tip"
        self._enter(STRADDLED)
        return self.hold("szczeka na bloku")

    def _straddled(self, obs: Observation) -> Action:
        """Wait for the supervisor's command to reach the held pose before handling the failure."""
        if not self._command_on_target and self._t - self._phase_t0 < HOLD_WAIT_S:
            return self.hold()
        return self.after_grasp_failed("tip", obs)

    def _close(self, obs: Observation) -> Action | None:
        self._set_jaw_goal(self.cmd_closed)
        gripper = self.kin.spec.gripper
        assert gripper is not None
        # Slow until the jaw reaches the block's width; only a stacked block also has the hand follow.
        slow = self.kin.from_q(self.q_cmd)[gripper] > self.cmd_block
        self._ramp_jaw(self._follow_step_q() if slow else math.inf)
        reading, jaw = self.reading, self.kin.spec.jaw
        # A jaw that has not started closing can satisfy the holding rule.
        self._jaw_moved |= reading.q <= self._q_open_entry - jaw.margin
        self._still_ticks = self._still_ticks + 1 if self._jaw_moved and reading.still else 0
        if self._jaw_ready() and self._still_ticks >= self.settle_ticks:
            if reading.holding and self._edge_grip():
                self.grasp.edge_closes += 1
                self.grasp.verdict = "edge"
                return self.after_grasp_failed("edge", obs)
            if reading.holding:
                assert self._jaw is not None
                self._jaw.mark()
                self.grasp.verdict = "held"
                closed_at = self._close_at
                if self.base_z > 0:
                    closed_at = self._close_point()
                    self.grasp.grasp_point = (float(closed_at[0]), float(closed_at[1]), float(closed_at[2]))
                self._lift_off_point = closed_at + np.array([0.0, 0.0, self._lift_m])
                self._lift_ticks = 0
                self._enter(LIFT_OFF)
                return None
            self.grasp.empty_closes += 1
            self.grasp.verdict = "empty"
            return self.after_grasp_failed("empty", obs)
        # Allow for float rounding of `obs.t`, so a whole number of ticks ends the wait on time.
        if self._t - self._phase_t0 >= JAW_WAIT_S - 1e-9:
            if not self._jaw_moved:
                return self._halt(f"szczeka nie ruszyla po rozkazie zamkniecia "
                                  f"({self.grasp.jaw_gap_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
            return self._halt(f"szczeka nie staje {JAW_WAIT_S:.0f} s po zamknieciu - serwo albo odczyt",
                              self.kin.from_q(self.q_cmd))
        if self.base_z > 0:
            # The tips meet the block together, so it does not slide across the block under it.
            return self._drive(self._close_point(), self.approach_step_m)
        return self.hold()

    def _edge_grip(self) -> bool:
        """Whether the held jaw's gap is more than `EDGE_GRIP_M` narrower than the chosen grasp width.

        Checked only where a failed close is retried after another look (`_looks_again`) and attempts
        remain: the last attempt lifts what it holds."""
        if not self._looks_again or self.status.attempt >= self.grasp_settings.max_attempts:
            return False
        return self.grasp.jaw_gap_m < self._width_m - EDGE_GRIP_M

    @property
    def _looks_again(self) -> bool:
        """Whether a retry from the hover looks at the block again. Plain grasp has no camera."""
        return False

    def _follow_step_q(self) -> float:
        """The jaw step [rad] that moves the fingertips' midpoint `descend_step_m` in one tick."""
        span_m = abs(float(self._centre_open[2] - self.kin.jaw_centre(self.cmd_block)[2]))
        span_q = abs(self._q(self.cmd_open) - self._q(self.cmd_block))
        return span_q * self.descend_step_m / span_m if span_m > 1e-9 else math.inf

    def _close_point(self) -> np.ndarray:
        """Where the TCP keeps the fingertips' midpoint over the block while the jaw closes onto it [m]."""
        gripper = self.kin.spec.gripper
        assert gripper is not None
        cmd = max(self.kin.from_q(self.q_cmd)[gripper], self.cmd_block)
        shift = float(self._centre_open[2] - self.kin.jaw_centre(cmd)[2])
        at = self._close_at
        return np.array([*(at[:2] + shift * np.array([math.cos(self._psi), math.sin(self._psi)])), at[2]])

    def _tips_low(self) -> bool:
        """Whether both measured jaw tips stand `close_on_stall_m` or more below the block's top."""
        below = self.grasp_settings.close_on_stall_m
        if not below > 0:
            return False
        top = self.grasp.top_m if np.isfinite(self.grasp.top_m) else self.top_nominal
        return max(float(tip[2]) for tip in self.kin.fingertips(self._measured)) <= top - below

    def _lift_off(self, obs: Observation) -> Action | None:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - float(self._close_at[2])
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        if self._arrived(self._lift_off_point, sag=True):
            self.grasp.hold_ticks = 0
            self._enter(CHECK)
            return None
        # Extend travel time while targets advance or the supervisor follows under a speed cap.
        on_point = float(np.linalg.norm(self._lift_off_point - self.tcp_cmd)) <= self.ik_tol
        if (not on_point and not self.status.out_of_reach) or not self._command_on_target:
            self._lift_ticks += 1
        # Allow `JAW_WAIT_S` after travel for the measured arm to arrive.
        lift_m = self._lift_m
        travel = max(lift_m / self.config.speed_mps, self._lift_ticks / self.hz)
        if self._t - self._phase_t0 >= travel + JAW_WAIT_S:
            return self._halt(f"ramie nie podnioslo bloku ({self.grasp.lifted_m * 1000:.0f} mm z "
                              f"{lift_m * 1000:.0f} mm)", self.kin.from_q(self.q_cmd))
        return self._drive(self._lift_off_point, self.approach_step_m)

    def _check(self, obs: Observation) -> Action | None:
        self.grasp.lifted_m = float(self._tcp_measured[2]) - float(self._close_at[2])
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp = self.grasp
        # Creep resets the hold streak for a fresh check on the next tick.
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
        """Monitor the block at lift height for `carry_s`, then finish holding or command release."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        grasp, cfg = self.grasp, self.grasp_settings
        grasp.carried_s = self._t - self._phase_t0
        # Allow for float rounding of `obs.t`, so a whole number of ticks ends the carry on time.
        if grasp.carried_s >= cfg.carry_s - 1e-9:
            if cfg.release:
                return self.release()
            return self._halt(f"blok w szczekach, podniesiony o {grasp.lifted_m * 1000:.0f} mm, niesiony "
                              f"{grasp.carried_s:.1f} s (szczeka {grasp.jaw_gap_m * 1000:.0f} mm)"
                              + self._close_summary(), self.kin.from_q(self.q_cmd), success=True)
        return self.hold()

    def _close_summary(self) -> str:
        """Text added to the success note about how the close was aimed. Plain grasp adds none."""
        return ""

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Monitor the hold once per tick during LIFT_OFF, CHECK and CARRY.

        Return None to continue or an action to interrupt the phase. A squeezing jaw remains held.
        Creep resets the hold reference. A sighting away from squeezing jaws is recorded in
        `elsewhere`. Consecutive empty readings start a recovery rise with the jaw opening.
        Overrides that call `release()` must return its action.
        """
        if self._watched_step == self.status.step:
            # A phase transition can call this twice. Count each reading once.
            return None
        self._watched_step = self.status.step
        grasp = self.grasp
        seen = self._sighting(obs, reading.squeezing)
        grasp.block_source = seen.source if seen is not None else ""
        jaw_xy = self._jaw_point(obs.joints)[:2]
        if reading.squeezing:
            self._empty_ticks = 0
            if reading.crept:
                # Still squeezed, but shifted. Restart the hold reference.
                grasp.creeps += 1
                assert self._jaw is not None
                self._jaw.mark()
                logger.info("Blok przesuwa sie w szczekach (%d)", grasp.creeps)
            if (seen is not None and seen.kind == SEEN and grasp.elsewhere is None
                    and float(np.linalg.norm(np.asarray(seen.xy) - jaw_xy)) > IN_HAND_M):
                grasp.elsewhere = seen.xy
                logger.info("Kamery widza blok gdzie indziej: (%.3f, %.3f) m", *seen.xy)
            return None
        # Ignore isolated empty readings.
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
        """End monitoring and open the jaw without moving the arm.

        Call only during LIFT_OFF, CHECK or CARRY. Return this action from a `carrying` override.
        The run succeeds once the measured jaw reaches the opening. The receiver must support
        the block during release.
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
        """Wait at the safe height with the jaw open for a sighting captured after the drop.

        Regrasp nearby sightings. Stop for a person if the block is too far away or `wait_s` expires.
        """
        grasp, cfg = self.grasp, self.grasp_settings
        seen = self._sighting(obs, self.reading.squeezing)
        grasp.block_source = seen.source if seen is not None else ""
        lost_s = grasp.lost_s = self._t - self._drop_t
        target = self.kin.from_q(self.q_cmd)
        # Require a capture after the block had time to land.
        if seen is not None and seen.kind == SEEN and seen.seen_t >= self._drop_t + DROP_SETTLE_S:
            assert self._drop_xy is not None
            far = float(np.linalg.norm(np.asarray(seen.xy) - self._drop_xy))
            if far > cfg.recover_m:
                # A distant block requires a person's help.
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
                # This capture predates the drop.
                why = f"zrodlo '{seen.source}' sprzed upadku"
            else:
                high = f", {seen.z * 100:.0f} cm nad blatem" if seen.z > self.rest_z_max else ""
                why = f"zrodlo '{seen.source}'{high}"
            return self._halt(f"blok zabrany ze szczeki: {why} od {lost_s:.0f} s - czekam na czlowieka",
                              target)
        return self.hold()

    def _regrasp(self, seen: Sighting) -> Action:
        """Re-aim through `retarget` and count the next grasp against the attempt budget.

        Close the jaw on the way to hover. By default, open there and straddle below the catalog top.
        With `probe: true`, measure the top before straddling. The grasp counts from the block's bottom
        as seen, so a block on another block gets the stack's limits.
        An unreachable sighting stops the run and keeps the old aim.
        """
        grasp = self.grasp
        saved = self._block_xy0, self._shift_k, self.block_rotation, self._headings
        base = self.base_z
        # The sideways shifts after a tip stall start from the new block point.
        self._block_xy0, self._shift_k = None, 0
        self._heading_from(seen.rotation)
        self._rest_at(seen.z - self.block.half_size[2])
        try:
            self.retarget(seen.xy[0], seen.xy[1])
        except RuntimeError as exc:
            self._block_xy0, self._shift_k, self.block_rotation, self._headings = saved
            self._rest_at(base)
            grasp.verdict = "taken"
            return self._halt(f"blok upadl poza zasiegiem: {exc} - czekam na czlowieka",
                              self.kin.from_q(self.q_cmd))
        self._jaw_goal_q = float(self._lo[self._g])
        grasp.elsewhere = None
        grasp.recoveries += 1
        grasp.verdict = "recovering"
        self._empty_ticks = 0
        where = "na blacie" if self.base_z <= 0 else "na innym bloku"
        logger.info("Blok lezy %s w (%.3f, %.3f) m - ponowna proba", where, *seen.xy)
        return self.hold(f"blok lezy {where} - ponowna proba")

    def _sighting(self, obs: Observation, squeezing: bool) -> Sighting | None:
        """Classify the observed block pose for hold monitoring and recovery.

        Return None if the pose is absent or its position is invalid. Classification uses the
        observation, `squeezing` and configuration without changing strategy state.
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
            # Delayed detections can remain usable as plain last-seen poses.
            kind = SEEN
        elif source == "symulacja" or (not source and self._simulated):
            # A resting block can be near the fingertips. Require squeeze to classify it as held.
            near = float(np.linalg.norm(position - self._jaw_point(obs.joints))) < IN_HAND_M
            kind = IN_HAND if near and squeezing else SEEN if lying else STALE
        else:
            kind = STALE
        t = float(obs.t)
        return Sighting(kind, (float(position[0]), float(position[1])), z, rotation.copy(), source, age,
                        t - age)

    def _jaw_point(self, joints: Mapping[str, float]) -> np.ndarray:
        """Return the fingertips' midpoint at `joints` [m, base frame]."""
        return np.mean(self.kin.fingertips(joints), axis=0)

    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        """Handle "tip", "empty", "edge" or "off" while holding the arm's pose.

        Where a retry looks again (`_looks_again`), an empty or edge close or an aim the camera saw off the
        block retries from the hover at the current aim while attempts remain (`retry`). Otherwise a tip
        stall retries half the chosen grasp width away while attempts remain, and, without the probe, an
        empty close after a sideways retry tries the other side within the same attempt budget. Any other
        empty close ends the run. Overrides that await a judge must enter their own phase."""
        attempt = self.status.attempt
        if why in ("edge", "off") or (why == "empty" and self._looks_again):
            return self.retry(why)
        if why == "empty":
            if not self.grasp_settings.probe and self._shift_k >= 1:
                if attempt < self.grasp_settings.max_attempts and self._next_shift() is not None:
                    self._start_rise(RETARGET)
                    return self.hold("chwyt pusty obok - probuje z drugiej strony")
                return self._halt(f"wyczerpane proby chwytu ({attempt}): chwyt pusty",
                                  self.kin.from_q(self.q_cmd))
            return self._halt("chwyt pusty - blok przesuniety albo obok szczek", self.kin.from_q(self.q_cmd))
        if attempt >= self.grasp_settings.max_attempts or self._next_shift() is None:
            return self._halt(f"wyczerpane proby chwytu ({attempt}): szczeka oparla sie o blok przy zjezdzie",
                              self.kin.from_q(self.q_cmd))
        self._start_rise(RETARGET)
        return self.hold("szczeka na bloku - w gore i obok")

    def retry(self, why: str, ease: np.ndarray | None = None) -> Action:
        """Start another attempt at the current aim after a failed close `why`, or end the run.

        An edge grip opens the jaw first (REOPEN). The arm rises with the jaw open, after the sideways `ease`
        [m] if given (`_start_rise`), re-aims at the current block point and approaches the hover again, so a
        look before the straddle sees the block anew. With `probe` the jaw closes once clear of the block, so
        the next probe descends with the closed jaw."""
        attempt, cfg = self.status.attempt, self.grasp_settings
        text = VERDICTS.get(why, why)
        if attempt >= cfg.max_attempts or self.grasp.block_xy is None:
            return self._halt(f"wyczerpane proby chwytu ({attempt}): {text}", self.kin.from_q(self.q_cmd))
        self._retarget_xy = np.array(self.grasp.block_xy, float)
        # Sideways retries after a tip stall start again from what the next look sees.
        self._block_xy0, self._shift_k = None, 0
        logger.info("Chwyt nieudany (%s) - proba %d z %d", text, attempt + 1, cfg.max_attempts)
        if why == "edge":
            self._set_jaw_goal(self.cmd_open)
            self._enter(REOPEN)
            return self.hold(f"{text} - otwieram szczeke, proba {attempt + 1}")
        self._start_rise(RETARGET, jaw_open=not cfg.probe, ease=ease)
        return self.hold(f"{text} - w gore, proba {attempt + 1}")

    def _reopen(self, obs: Observation) -> Action:
        """Open the jaw on an edge grip without moving the arm, then rise for another attempt."""
        self._ramp_jaw()
        if self._jaw_ready() and abs(self.reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin:
            self._start_rise(RETARGET, jaw_open=not self.grasp_settings.probe)
            return self.hold(f"{VERDICTS['edge']} - w gore, proba {self.status.attempt + 1}")
        if self._t - self._phase_t0 >= JAW_WAIT_S:
            gaps = f"{self.grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm"
            return self._halt(f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s ({gaps})",
                              self.kin.from_q(self.q_cmd))
        return self.hold()

    def _next_shift(self) -> np.ndarray | None:
        """Store and return the next retry centre, or None when shifts are exhausted.

        Offsets follow `SHIFTS` in half chosen-grasp widths along the closing heading.
        """
        self._shift_k += 1
        if self._shift_k > len(SHIFTS) or self._block_xy0 is None:
            return None
        shift = SHIFTS[self._shift_k - 1] * self._width_m / 2
        self._retarget_xy = self._block_xy0 + shift * np.array([math.cos(self._psi), math.sin(self._psi)])
        return self._retarget_xy

    def grasp_verdict(self) -> tuple[str, str]:
        """Return the grasp verdict and its text for logs and judges."""
        verdict = self.grasp.verdict or "none"
        return verdict, VERDICTS.get(verdict, verdict)

    # ------------------------------------------------------------------ the jaw
    def _set_jaw_goal(self, gripper: float) -> None:
        self._jaw_goal_q = float(np.clip(self._q(gripper), self._lo[self._g], self._hi[self._g]))

    def _ramp_jaw(self, step_q: float = math.inf) -> None:
        """Advance the gripper target toward its goal at `SPEED_SHARE` of the supervisor's speed limit, and at
        most `step_q` [rad] per tick."""
        g = self._g
        step = min(self._grip_step_q, step_q)
        delta = float(np.clip(self._jaw_goal_q - self.q_cmd[g], -step, step))
        before = float(self.q_cmd[g])
        self.q_cmd[g] = float(np.clip(before + delta, self._lo[g], self._hi[g]))
        if self.q_cmd[g] != before:
            # The supervisor has not yet received the new target.
            self._jaw_on_command = False
        spec = self.kin.spec
        assert spec.gripper is not None
        self.grasp.gap_cmd_m = self.kin.jaw_gap(self.kin.from_q(self.q_cmd)[spec.gripper])

    def _jaw_ready(self) -> bool:
        """Whether the gripper target reached its goal and the supervisor's command reached that target."""
        return abs(float(self.q_cmd[self._g]) - self._jaw_goal_q) <= 1e-9 and self._jaw_on_command

    # ------------------------------------------------------------------ the panel
    def _phase_name(self) -> str:
        if self.status.phase == RISE:
            rise = {OPEN: "w gore po dotyku", RECOVER: "blok wypadl ze szczeki - w gore",
                    RESTRADDLE: "w gore, poprawiam chwyt"}
            return rise.get(self._rise_then, "w gore znad bloku")
        return super()._phase_name()

    def _note(self) -> str:
        st, grasp = self.status, self.grasp
        mm = st.distance * 1000
        if st.phase == RISE:
            text = self._phase_name() + f": {mm:.0f} mm"
        elif st.phase == REOPEN:
            text = "chwyt za krawedz - otwieram szczeke"
        elif st.phase == APPROACH:
            text = ("nad obiekt: zamykam szczeke" if not self._jaw_ready() else f"nad obiekt: {mm:.0f} mm")
            text += self._source_note() + self._tilt_note()
        elif st.phase == OPEN:
            text = f"nad blok, otwieram szczeke: {grasp.gap_cmd_m * 1000:.0f} mm" + self._tilt_note()
        elif st.phase == STRADDLE:
            low = math.isfinite(self._low_stall_t)
            text = "szczeka nisko przy bloku - zamykam" if low else f"w dol na blok: {mm:.0f} mm"
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


    def _tilt_note(self) -> str:
        return f" (dlon pochylona {self.grasp.tilt_deg:.0f} st.)" if self._tilt else ""

    def _source_note(self) -> str:
        """Return the chosen library index and score, or the geometry fallback, for the panel."""
        if self.library is None:
            return ""
        if self._cand is not None and self._cand.origin != GEOMETRY:
            return f" (biblioteka #{self.grasp.library_index}, wynik {self.grasp.library_score:.2f})"
        return " (geometria: brak obrotu)"


# ------------------------------------------------------------------ geometry
def top_down(psi: float) -> np.ndarray:
    """Return the TCP rotation with approach down and closing along heading `psi` [rad, base frame].

    The columns are approach, y and closing, corresponding to TCP x, y and z."""
    c, s = math.cos(psi), math.sin(psi)
    return np.column_stack([(0.0, 0.0, -1.0), (-s, c, 0.0), (c, s, 0.0)])


def ik_restarts(tilt: float) -> int:
    """Random IK starts for a hand tilted by `tilt` [rad]."""
    return TILT_RESTARTS if tilt else IK_RESTARTS


def hand_down(psi: float, tilt: float = 0.0) -> np.ndarray:
    """Return `top_down(psi)` turned by `tilt` [rad] about its closing axis.

    A positive tilt leans the approach toward the TCP y axis, (-sin psi, cos psi, 0)."""
    R = top_down(psi)
    if not tilt:
        return R
    c, s = math.cos(tilt), math.sin(tilt)
    return np.column_stack([c * R[:, 0] + s * R[:, 1], c * R[:, 1] - s * R[:, 0], R[:, 2]])


def tilt(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """How far the approach of `joints` leans from straight down [rad]."""
    approach, _ = kin.tool_axes(joints)
    return math.acos(float(np.clip(-approach[2], -1.0, 1.0)))


def heading(kin: RobotKinematics, joints: Mapping[str, float]) -> float:
    """Return the horizontal heading of the closing axis [rad, base frame]."""
    _, closing = kin.tool_axes(joints)
    return math.atan2(float(closing[1]), float(closing[0]))


def grasp_headings(rotation: np.ndarray, yaw_period: float, current: float) -> list[float]:
    """Return candidate closing headings [rad], nearest `current` first.

    An identity rotation keeps `current` and its half-turn mirror. Otherwise the jaws close
    across the block's short side. The body's x axis defines its long side. `yaw_period`
    sets the equivalent headings, each with a half-turn mirror.

    `candidates` uses these headings for geometry grasps.
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

