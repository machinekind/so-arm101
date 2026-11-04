"""Pick a block, place it on another block or on a table spot, then verify the result with the cameras.

`pick_place` runs one planner step. It picks block `label` (catalog `block`) as `grasp` does, with
the wrist-camera look of `grasp_wrist` (skipped when `wrist_camera` is empty) and, with `judge: jev`,
the close and lift checks of `grasp_jev`. A drop goes through grasp's recovery and the run then
continues to the place. See the package README for settings.

The target is block `place_on` (catalog `place_on_block`) or, with `place_on` empty, a pose: the held
block's centre at `place_x_m`, `place_y_m`, its bottom `place_z_m` above the table and its heading
`place_yaw_deg`, resting on the blocks named in `supports` or on the table. A block target's top comes
from its pose and the catalog height; a pose at table height counts as a block lying on the table. A
target rotation without yaw, such as identity, counts as yaw 0. Before the pick the target block or
each support must have a usable pose and a place plan must fit, with the held block's place in the hand
estimated.

After the verified lift the run plans the place. The held block's centre and bottom in the hand
come from the fingertips and the grasp height. With `align_yaw`, the hand turns so the held block
takes the target's yaw, modulo the smaller yaw period of the two blocks. The carry height keeps the
held block `carry_clearance_m` above the target and the catalog's tallest block, lowered toward
`MIN_CARRY_CLEAR_M` when the arm cannot reach it. The release height leaves `place_gap_m` between
the held block and the target's top, and `GRIP_SLIP_M` more for a block that slid down in the grip
on the way, which nothing the arm measures shows. Every planned point must be reachable with the hand
down.

After the release the open hand must leave without touching the placed block. The jaw opens where the
hand stands, and the hand eases at most `release_ease_m` toward the fixed jaw, so the fixed pad leaves
the block's face. It then rises straight up at `retreat_speed_mps` until the open jaw's lowest point,
from the measured joints, stands `retreat_clear_m` above the block's top; an arm that settles short
rises further, by `sag_m` at most. Only then does the hand move toward the start position, never
lower, at `retreat_speed_mps` and with its orientation held while the jaw is near the block. Where the
arm cannot rise that high, a jaw `JAW_CLEAR_M` clear still leaves; below that the jaw opens fully, so
that its moving side clears the block, and the way to the start position must lead away from the
fixed jaw. Headings with the first exit win.

Over a block target the run latches the newest fresh camera pose of the target seen since the carry
began, if it was captured at most `fresh_pose_s` before the hand started over the target. The hand
may hide the target at the hover, so the pose may come from the way there. A target that moved
re-plans the hover, at most `MAX_REPLANS` times. The descent settles `SLOW_ABOVE_M` over the release
height, or at the carry height when that is lower, then runs at `place_speed_mps` and stops on a
stall or at the release height. The jaw opens `release_margin_m` past the held block on each side,
and the hand leaves as above. Fresh poses captured after the release then verify the result.

A target not seen at the hover, an unreachable place or a target that keeps moving puts the block
back where it was picked, without verification. The carry back stays above the target the hand
hovered over. `PlaceStatus.outcome` names the result with one of `OUTCOMES`.

A block that fell is picked and placed again, `place_retries` times at most in a run: one the check
finds off the target, unless it still sits on a target or supports that moved, and one that dropped
out of the jaw on the way, through grasp's recovery. A retry after the check needs fresh poses of the
block, lying flat, and of the target or each support. The hand first moves over the block at its
height, and each new pick gets the whole `max_attempts`. `PlaceStatus.retries` counts the retries.

With `approach_file`, the approach table that `calibrate_approach` measured on this stand moves where the
hand is sent: the TCP goes to the place the cameras see plus the table's correction there, at the height and
from the direction of the approach, so the held block lands where the target really is. The final leg into
the open-jaw hover at the pick and into the carry hover at the place then always runs `approach_lead_m`
along one direction (`approach_direction`, else the table's preferred one), as the table was measured. The
pick's first aim takes the table's correction too. Where the table gives none (outside its corners, at
another height, with the hand tilted or the leg out of reach), the place takes the pick's own correction
instead: the aim the wrist camera centred the jaws on minus the detector's (`pick_shift`). A table that
disagrees with that by more than `table_check_m` is not trusted for the run. A put-back takes neither.
The corrections move only what the arm is sent to; the latch, the jaw clearances and the check after the
release still compare camera poses with camera poses.

At a pose, only headings whose jaw stays `JAW_SIDE_CLEAR_M` clear of every other object at the release
count, from the objects' poses and catalog sizes; the held block and its supports do not count. Over
supports the run latches each one's newest fresh pose as it does a target block's and moves the pose
with them: across by their mean shift, up by the highest rise. Supports that moved apart by more than
`place_tol_m` put the block back as `target_moved`. The check after the release also needs the block
level within `LEVEL_TOL_DEG`, and each support seen within `place_tol_m` of its latched pose.

A block may be picked off another block: the grasp counts from its observed bottom, and every carry
clears what it was picked from. The blocks under it, found from their poses before the pick, are
checked with fresh camera poses once the hand is away from them; a shift over `place_tol_m` is
reported in `PlaceStatus.support_moved` and the stop reason. A put-back returns the block onto its
stack, moved with the block under it when that one was seen to move.

Where the place is out of reach with the hand straight down, the pick takes the least tilt up to
`max_tilt_deg` at which both the pick and the place fit, and the hand keeps that tilt to the end: the
held block then stays level. The tilted hand keeps its heading against the arm, so the place heading
follows from the pick's, and a turn to line up with the target must stay within `MAX_ROT_ERR_DEG` of
block tilt. The jaw clearances count along straight down for the tilted hand. A tilted pick needs the
block's yaw from its pose or the wrist look; when the look finds the block does not fit the tilted
pick, or gives no yaw for a block whose pose has none, the run ends as `unreachable` before the jaw
goes down.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

from ..approach_table import DIRECTIONS, ApproachTable, Hit
from ..blocks import BlockSpec, block
from ..blocks.classic_world import BLOCKS
from ..planner.world import WorldObject, overlap, world_state
from .approach import carry_heights, direction_vector, place_heights
from .grasp import (
    CARRY,
    DROP_SETTLE_S,
    IN_HAND_M,
    JAW_WAIT_S,
    MAX_ROT_ERR_DEG,
    ON_BLOCK_M,
    SEEN,
    GraspCandidate,
    JawPlan,
    Sighting,
    hand_down,
    heading,
    ik_restarts,
    wrap,
)
from .grasp_jev import JEV_PHASE_NAMES, LET_GO, GraspJevSettings, GraspJevStrategy
from .jaw import JawReading
from .judges import Judge, JudgeError, Questions, Reply
from .touch import APPROACH, DONE, SETTLE_S, TURN_DEG_S, WAIT, TouchStrategy, normalised
from .wrist import WristGraspSettings, WristGraspStrategy, fold

logger = logging.getLogger(__name__)

#: Judges for `PickPlaceSettings.judge`.
JEV = "jev"
NO_JUDGE = "none"
JUDGES = (JEV, NO_JUDGE)

#: Outcomes in `PlaceStatus.outcome`, for a planner.
PLACED = "placed"
NOT_FOUND = "not_found"
TARGET_NOT_SEEN = "target_not_seen"
PICK_FAILED = "pick_failed"
DROPPED = "dropped"
FELL_OFF = "fell_off"
TARGET_MOVED = "target_moved"
UNREACHABLE = "unreachable"
STOPPED = "stopped"
OUTCOMES = (PLACED, NOT_FOUND, TARGET_NOT_SEEN, PICK_FAILED, DROPPED, FELL_OFF, TARGET_MOVED, UNREACHABLE,
            STOPPED)

#: Phases after the verified lift.
TARGET = "target"
CARRY_UP = "carry_up"
CARRY_OVER = "carry_over"
LATCH = "latch"
PLACE = "place"
PARK = "park"
VERIFY = "verify"
#: After a fall: wait for fresh poses before picking the block again, then move over it.
RETRY = "retry"
TRANSIT = "transit"
#: The phases' names in the panel's notes.
PLACE_PHASE_NAMES = {**JEV_PHASE_NAMES, TARGET: "planuje odlozenie", CARRY_UP: "unosze blok do przeniesienia",
                     CARRY_OVER: "niose blok nad cel", LATCH: "sprawdzam cel w kamerach",
                     PLACE: "opuszczam blok", PARK: "odsuwam ramie od bloku", VERIFY: "sprawdzam odlozenie",
                     RETRY: "ponawiam odlozenie", TRANSIT: "przenosze reke nad blok"}

#: When the pick's own correction moves the place (`PickPlaceSettings.pick_shift`): where a table that is
#: set gives none, also without a table, or never.
AUTO = "auto"
ALWAYS = "always"
OFF = "off"
PICK_SHIFTS = (AUTO, ALWAYS, OFF)
#: Where the correction of a place came from, in `PlaceStatus.correction_from`.
FROM_TABLE = "table"
FROM_PICK = "pick"

#: Hover re-plans after the latched target moved, before the block goes back.
MAX_REPLANS = 2
#: The least clearance of the held block's bottom above the target while carrying [m].
MIN_CARRY_CLEAR_M = 0.01
#: The top of the catalog's tallest block lying on the table [m].
TABLE_TOP_M = max(2 * b.half_size[2] for b in BLOCKS.values())
#: A block the picked one rests on has its centre at most this far from the picked one's across [m].
SUPPORT_XY_M = max(b.half_size[0] for b in BLOCKS.values()) + 0.005
#: The slow part of the descent starts this far above the release height [m].
SLOW_ABOVE_M = 0.015
#: A held block may slide this far down in the grip on the way. The release starts this much higher, so
#: such a block is not pressed onto the target [m].
GRIP_SLIP_M = 0.003
#: A placed block's centre may lie this far above or below the expected height [m].
PLACED_Z_TOL_M = 0.01
#: A fresh target pose this close to the planned one is latched without a new hover [m, deg].
LATCH_SAME_M = 0.004
LATCH_SAME_DEG = 3.0
#: With a final leg, a latch that moves the hover more than this runs the leg again [m].
RELEG_M = 0.001
#: The least clearance of the open jaw's lowest point over the placed block's top before the hand leaves,
#: where the arm cannot rise `retreat_clear_m` [m].
JAW_CLEAR_M = 0.004
#: The jaw counts as `retreat_clear_m` clear this much short of it [m].
CLEAR_TOL_M = 0.001
#: The largest `release_ease_m` [m].
MAX_RELEASE_EASE_M = 0.003
#: The fully open jaw [app units].
FULL_OPEN = 100.0
#: A hand that leaves with the fixed jaw beside the placed block moves at least this much away from
#: it: the cosine of its direction against the closing axis is at most minus this.
AWAY_COS = 0.25
#: At a pose, the open jaw keeps this far from other objects at the release [m].
JAW_SIDE_CLEAR_M = 0.005
#: A block placed at a pose leans at most this much from level [deg].
LEVEL_TOL_DEG = 5.0


@dataclass
class PickPlaceSettings(GraspJevSettings, WristGraspSettings):
    #: Whole-run time limit including model waits [s]. Zero disables it.
    time_limit_s: float = 240.0
    #: Judge of each close and lift [jev|none]. `none` needs no key.
    judge: str = field(default=JEV, metadata={"choices": JUDGES})
    #: Detector label of the block to place on. Empty places on the table spot.
    place_on: str = ""
    #: Catalog name of the block to place on.
    place_on_block: str = field(default="cube", metadata={"choices": tuple(BLOCKS)})
    #: The pose when `place_on` is empty: the held block's centre [m, base frame], the height its bottom
    #: lands at [m above the table] and the heading of its long side [deg, counter-clockwise from +x].
    place_x_m: float = 0.2
    place_y_m: float = 0.08
    place_z_m: float = 0.0
    place_yaw_deg: float = 0.0
    #: Labels of the blocks the held block will rest on at the pose, comma-separated. Empty is the table.
    supports: str = ""
    #: Turn the held block to the target block's yaw, or the pose's.
    align_yaw: bool = True
    #: Clearance of the held block's bottom over the target and the table's blocks while carrying [m].
    carry_clearance_m: float = 0.03
    #: Gap between the held block's bottom, where it was gripped, and the target's top at the release [m].
    place_gap_m: float = 0.002
    #: TCP speed of the last part of the descent onto the target [m/s].
    place_speed_mps: float = 0.02
    #: The jaw opens this much past the held block on each side to release it [m].
    release_margin_m: float = 0.004
    #: Then the hand moves this far toward the fixed jaw, so its pad leaves the placed block [m]; at most
    #: `MAX_RELEASE_EASE_M` and `release_margin_m`. Zero rises from where the jaw opened.
    release_ease_m: float = 0.002
    #: How far the open hand rises after the release, if the arm reaches [m].
    retreat_m: float = 0.05
    #: TCP speed of the open hand's rise off the placed block and of its move away while near it [m/s].
    retreat_speed_mps: float = 0.02
    #: The open jaw's lowest point rises this far above the placed block's top before the hand moves away
    #: [m]; less where the arm cannot reach that, but `JAW_CLEAR_M` at least.
    retreat_clear_m: float = 0.02
    #: The latch takes a target pose captured at most this long before the hand started over the target,
    #: the verification one at most this old [s].
    fresh_pose_s: float = 1.0
    #: How long to wait for a target pose after the lift and at the hover [s].
    target_wait_s: float = 5.0
    #: How long to wait for fresh poses after the release [s].
    verify_wait_s: float = 3.0
    #: A placed block's centre may lie this far from the target's centre [m].
    place_tol_m: float = 0.008
    #: How many times a run picks and places the block again after it fell off the target or out of the
    #: jaw on the way.
    place_retries: int = 2
    #: As in `grasp`, but on by default: the hand tilts where the pick or the place needs it [deg].
    max_tilt_deg: float = 30.0
    #: The approach table `calibrate_approach` wrote for this stand [path]. Empty sends the hand where the
    #: cameras see the place.
    approach_file: str = ""
    #: The side the final leg into the pick's and the place's hover comes from [out|in|ccw|cw]. Empty takes
    #: the table's preferred one, and without a table the direct way.
    approach_direction: str = field(default="", metadata={"choices": ("", *DIRECTIONS)})
    #: Length of that final horizontal leg [m].
    approach_lead_m: float = 0.03
    #: Where the table gives no correction, the place takes the pick's own: the aim the wrist camera centred
    #: the jaws on minus the detector's [auto|always|off]. `auto` only with a table set, `always` without
    #: one too.
    pick_shift: str = field(default=ALWAYS, metadata={"choices": PICK_SHIFTS})
    #: The table's correction at the pick may differ this much from the pick's own before the run stops
    #: trusting the table [m].
    table_check_m: float = 0.004
    #: A table older than this many days is not used. Zero takes any age.
    max_table_age_days: float = 0.0


@dataclass
class PlaceStatus:
    """Place state alongside `GraspStatus` for logs, tests and a planner."""

    #: One of `OUTCOMES` once the run has ended, '' before.
    outcome: str = ""
    #: The target's label, or '' for a pose.
    target: str = ""
    #: The blocks the held block rests on at a pose; () for the table or a target block.
    rests_on: tuple[str, ...] = ()
    #: Where the held block's centre goes [m, base frame], the surface it lands on [m above the table]
    #: and the target's yaw [deg]. NaN yaw when unknown.
    target_xy: tuple[float, float] | None = None
    target_top_m: float = float("nan")
    target_yaw_deg: float = float("nan")
    #: The closing heading at the release [deg] and the turn from the pick's heading [deg].
    heading_deg: float = float("nan")
    turn_deg: float = float("nan")
    #: TCP heights of the carry, the release and the retreat [m above the table].
    carry_z: float = float("nan")
    release_z: float = float("nan")
    retreat_z: float = float("nan")
    #: How far the TCP sits above the held block's bottom [m].
    hang_m: float = float("nan")
    #: Hover re-plans after the target moved.
    replans: int = 0
    #: The descent stopped on a stall.
    contact: bool = False
    #: The jaw opened fully to leave the placed block, because the arm could not lift it clear.
    opened_wide: bool = False
    #: How far the open jaw's lowest point stood above the placed block's top as the hand left [m].
    jaw_clear_m: float = float("nan")
    #: Picks of the block after it fell (`place_retries`), and the outcome each one followed.
    retries: int = 0
    retried: tuple[str, ...] = ()
    #: The block went back to where it was picked.
    put_back: bool = False
    #: The verified block's centre [m, base frame], its distance from the target centre [m], its height
    #: against the expected one [m] and how far it leans from level [deg] (NaN without a rotation).
    placed_xyz: tuple[float, float, float] | None = None
    error_m: float = float("nan")
    height_error_m: float = float("nan")
    placed_tilt_deg: float = float("nan")
    #: How far the target, or the farthest support of a pose, moved by the verification [m]. NaN when
    #: not seen.
    target_shift_m: float = float("nan")
    #: The blocks the picked one rested on, the largest shift of one seen after the pick [m] (NaN when
    #: none was seen) and whether it exceeds `place_tol_m`.
    supports: tuple[str, ...] = ()
    support_shift_m: float = float("nan")
    support_moved: bool = False
    #: Why `approach_file` is not used, '' when it is or none is set, and the final leg's direction, '' for
    #: the direct way.
    table_problem: str = ""
    direction: str = ""
    #: About the table in use: settings it was measured with that differ from these, or a pick it disagreed
    #: with.
    table_note: str = ""
    #: The correction added to where the hand is sent at the release [m, base frame x and y], where it came
    #: from (`FROM_TABLE`, `FROM_PICK` or '') and why the table gave none there ('' when it did or none is
    #: set).
    correction_m: tuple[float, float] = (0.0, 0.0)
    correction_from: str = ""
    table_miss: str = ""
    #: How far the table's correction at the pick lay from the pick's own [m]; NaN when not compared. Over
    #: `table_check_m` the table is not used for the rest of the run.
    table_check_m: float = float("nan")
    table_trusted: bool = True
    #: The place's final leg into the carry hover [m]; 0 for the direct way.
    lead_m: float = 0.0


@dataclass(frozen=True)
class PlacePlan:
    """A planned place: TCP targets and the target they came from."""

    #: Where the held block's centre goes [m, base frame], the surface's top [m] and the target yaw [rad].
    target_xy: np.ndarray
    top_m: float
    yaw: float | None
    #: Closing heading at the release [rad, base frame], its turn from the carry's heading [rad] and the
    #: TCP's x and y there as the cameras see the place [m].
    psi: float
    turn: float
    tcp_xy: np.ndarray
    #: The start of the carry: the TCP's x and y when planned [m].
    from_xy: np.ndarray
    #: TCP heights [m above the table].
    carry_z: float
    release_z: float
    retreat_z: float
    #: At the retreat the jaw still flanks the placed block, so it opens fully and the hand leaves away
    #: from the fixed jaw.
    wide: bool = False
    #: Added to `tcp_xy` where the hand is sent [m], and where it came from (`FROM_TABLE`, `FROM_PICK` or '').
    correction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    source: str = ""
    #: The final horizontal leg into the hover at the carry height [m, base frame x and y], or None.
    leg: np.ndarray | None = None
    #: Why the table gave no correction here, '' when it did or none is set.
    miss: str = ""

    @property
    def cmd_xy(self) -> np.ndarray:
        """Where the TCP is sent over the place [m, base frame x and y]."""
        return self.tcp_xy + self.correction

    @property
    def hover(self) -> np.ndarray:
        return np.array([*self.cmd_xy, self.carry_z])

    @property
    def release(self) -> np.ndarray:
        return np.array([*self.cmd_xy, self.release_z])

    @property
    def lead_point(self) -> np.ndarray | None:
        """Where the final leg starts, at the carry height; None without one."""
        return None if self.leg is None else np.array([*(self.cmd_xy - self.leg), self.carry_z])


@dataclass(frozen=True)
class Correction:
    """One way to correct a place: the correction [m], where it came from and the final leg it needs (None:
    none), or a leg it takes if the arm reaches it."""

    delta: np.ndarray
    source: str
    leg: np.ndarray | None
    leg_required: bool = False
    #: Why the table gave no correction here, '' when it did or none is set.
    miss: str = ""


#: Why the table gave no correction at the place, for the panel: the table's own reasons and this run's.
MISS_NOTES = {"table": "tablica nieuzyteczna", "height": "wysokosc poza tablica", "direction":
              "tablica bez tego kierunku", "outside": "cel poza tablica", "stale": "tablica za stara",
              "tilt": "dlon pochylona", "check": "tablica nie zgadza sie z chwytem",
              "lead": "podejscie z kierunku tablicy poza zasiegiem"}


class NoJudge:
    """The judge of `judge: none`: never asked and needs no key."""

    name = NO_JUDGE

    def check(self) -> None:
        pass

    def ask(self, state: Mapping[str, Any], questions: Questions, *, images: Mapping[str, np.ndarray],
            timeout_s: float, cancel: threading.Event) -> Reply:
        raise JudgeError("brak sedziego", kind="error")


def rot2(angle: float) -> np.ndarray:
    """Return the 2D rotation by `angle` [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def yaw_of(rotation: Any) -> float | None:
    """Return the yaw of a pose rotation [rad], or None when it is invalid."""
    R = np.asarray(rotation, float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return None
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def known_yaw(rotation: Any) -> float | None:
    """Return the yaw of a pose rotation [rad], or None when it is identity or invalid.

    The grasp family treats identity as a pose without orientation."""
    yaw = yaw_of(rotation)
    if yaw is None or np.allclose(np.asarray(rotation, float), np.eye(3), atol=1e-6):
        return None
    return yaw


class PickPlaceStrategy(GraspJevStrategy, WristGraspStrategy):
    """Picks a block and places it on another block or at a pose, then checks the result in the cameras."""

    settings = PickPlaceSettings
    phase_names = PLACE_PHASE_NAMES
    camera_required = False

    def __init__(self, settings: PickPlaceSettings | None = None, *, judge: Judge | None = None,
                 threaded: bool = True, clock: Callable[[], float] = time.monotonic,
                 blob: Callable[[np.ndarray], float | None] | None = None):
        config = settings if settings is not None else PickPlaceSettings()
        if config.judge not in JUDGES:
            raise ValueError(f"pick_place: judge must be one of {', '.join(JUDGES)}, is {config.judge!r}")
        if config.place_on_block not in BLOCKS:
            raise ValueError(f"pick_place: place_on_block must be one of {', '.join(BLOCKS)}, "
                             f"is {config.place_on_block!r}")
        for name in ("carry_clearance_m", "place_speed_mps", "release_margin_m", "retreat_m",
                     "retreat_speed_mps", "retreat_clear_m", "fresh_pose_s", "place_tol_m"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"pick_place: {name} must be positive, is {value}")
        for name in ("place_gap_m", "target_wait_s", "verify_wait_s", "place_retries"):
            value = getattr(config, name)
            if not value >= 0:
                raise ValueError(f"pick_place: {name} must be zero or more, is {value}")
        if not 0 <= config.release_ease_m <= MAX_RELEASE_EASE_M:
            raise ValueError(f"pick_place: release_ease_m must lie within [0, {MAX_RELEASE_EASE_M}], is "
                             f"{config.release_ease_m}")
        for name in ("place_x_m", "place_y_m", "place_yaw_deg"):
            value = getattr(config, name)
            if not math.isfinite(value):
                raise ValueError(f"pick_place: {name} must be finite, is {value}")
        if not config.place_z_m >= 0 or not math.isfinite(config.place_z_m):
            raise ValueError(f"pick_place: place_z_m must be finite and zero or more, is {config.place_z_m}")
        rests = tuple(normalised(part) for part in config.supports.split(",") if part.strip())
        if rests:
            if normalised(config.place_on):
                raise ValueError("pick_place: supports needs place_on empty: they carry a pose")
            if len(set(rests)) < len(rests):
                raise ValueError(f"pick_place: supports names a block twice: {config.supports!r}")
            if config.place_z_m <= ON_BLOCK_M:
                raise ValueError(f"pick_place: supports need place_z_m at their top, is {config.place_z_m}")
        elif not normalised(config.place_on) and config.place_z_m > ON_BLOCK_M:
            raise ValueError(f"pick_place: place_z_m {config.place_z_m} above the table needs supports")
        if config.send_frames and not config.wrist_camera.strip():
            raise ValueError("pick_place: send_frames needs wrist_camera")
        if config.approach_direction not in ("", *DIRECTIONS):
            raise ValueError(f"pick_place: approach_direction must be empty or one of "
                             f"{', '.join(DIRECTIONS)}, is {config.approach_direction!r}")
        if config.pick_shift not in PICK_SHIFTS:
            raise ValueError(f"pick_place: pick_shift must be one of {', '.join(PICK_SHIFTS)}, is "
                             f"{config.pick_shift!r}")
        for name in ("approach_lead_m", "table_check_m"):
            value = getattr(config, name)
            if not value > 0 or not math.isfinite(value):
                raise ValueError(f"pick_place: {name} must be positive, is {value}")
        if not config.max_table_age_days >= 0:
            raise ValueError(f"pick_place: max_table_age_days must be zero or more, is "
                             f"{config.max_table_age_days}")
        #: Jev judges the close and lift. Without it the grasp follows plain `grasp`.
        self.judged = config.judge == JEV
        super().__init__(config, judge=judge if self.judged else NoJudge(), threaded=threaded, clock=clock,
                         blob=blob)
        self.place_settings = config
        #: The target's key in `obs.objects`; '' for a pose.
        self.target_label = normalised(config.place_on)
        if self.target_label and self.target_label == self.label:
            raise ValueError(f"pick_place: place_on must name another block than label, both are "
                             f"{self.label!r}")
        if self.label in rests:
            raise ValueError(f"pick_place: supports must name other blocks than label {self.label!r}")
        #: The blocks the held block rests on at a pose; () for the table or a target block.
        self.rest_on = rests
        self.target_block = block(config.place_on_block)
        #: How far the hand eases toward the fixed jaw after the release [m].
        self._ease_m = min(config.release_ease_m, config.release_margin_m)
        self.place = PlaceStatus(target=self.target_label, rests_on=self.rest_on)
        self._reset_place()
        #: The jaw bodies' points in the TCP frame by gripper command.
        self._jaw_clouds: dict[float, tuple[np.ndarray, ...]] = {}
        #: The latest observation, for a pick again that grasp's recovery starts.
        self._obs: Observation | None = None
        #: The approach table in use, or None, and the final leg's direction ('' for the direct way).
        self._table: ApproachTable | None = None
        self._direction = ""
        self._phases.update({CARRY: self._lifted, TARGET: self._target, CARRY_UP: self._carry_up,
                             CARRY_OVER: self._carry_over, LATCH: self._latch, PLACE: self._place,
                             PARK: self._park, VERIFY: self._verify, RETRY: self._retry,
                             TRANSIT: self._transit})

    def _reset_place(self) -> None:
        #: Outcome for the next end of the run, '' for the default.
        self._pending = ""
        #: The target had a usable pose before the pick.
        self._target_ready = False
        #: The place sequence runs: the block is held after the verified lift.
        self._placing = False
        self._putting_back = False
        self._put_back_outcome = ""
        self._put_back_why = ""
        self._place_plan: PlacePlan | None = None
        self._replans = 0
        #: Held block centre in the hand frame [m], TCP height above its bottom [m] and its yaw against
        #: the closing heading [rad].
        self._hand_xy = np.zeros(2)
        self._hang = float("nan")
        self._held_rel = -math.pi / 2
        #: Where the held block was picked: its centre in the hand and before the close [m, base frame].
        self._pick_xy = np.zeros(2)
        self._pick_aim = np.zeros(2)
        self._start_tcp = np.zeros(3)
        self._park_point = np.zeros(3)
        self._park_s = 0.0
        self._move_s = 0.0
        #: When the hand last started over the target [s, `obs.t`].
        self._over_t0 = float("nan")
        #: The newest fresh camera pose of the target while carrying, and when it was captured [s, `obs.t`].
        self._target_seen: tuple[ObjectPose, float] | None = None
        #: The highest target top the hand hovered over while carrying [m above the table].
        self._over_top = 0.0
        #: The descent settled above the target and goes on slowly.
        self._slow = False
        #: The TCP height where the jaw opened [m above the table].
        self._release_tcp_z = float("nan")
        #: The jaw is open enough to leave the placed block.
        self._opened = True
        #: How much higher than planned the open hand rose, for an arm that settled short [m].
        self._rise_extra = 0.0
        #: While the TCP is this close to the placed block across, the leaving hand moves slowly and keeps its
        #: orientation while the arm can [m].
        self._near_m = 0.0
        self._park_held = True
        #: When the open hand finished rising [s, `obs.t`].
        self._released_t = float("nan")
        #: The outcome and stop reason of a fall the next pick of the block retries, or None; and the outcome
        #: the pick under way retries, or ''.
        self._fallen: tuple[str, str] | None = None
        self._retrying = ""
        #: Where the hand moves over the block before picking it again [m, base frame].
        self._transit_point = np.zeros(3)
        #: Positions of the blocks the picked one rested on before the pick and since the lift [m].
        self._supports: dict[str, np.ndarray] = {}
        self._support_now: dict[str, np.ndarray] = {}
        #: When the first verified lift ended [s, `obs.t`].
        self._lifted_t = float("nan")
        #: The place the pick plans for: the target's centre [m], top [m] and yaw [rad] or None.
        self._goal: tuple[np.ndarray, float, float | None] | None = None
        #: Positions of a pose's supports when the pose was taken, the newest fresh camera pose of each
        #: while carrying with its capture time [s, `obs.t`], and the positions the place latched [m].
        self._rest_ref: dict[str, np.ndarray] = {}
        self._rest_seen: dict[str, tuple[ObjectPose, float]] = {}
        self._rest_pos: dict[str, np.ndarray] = {}
        #: The latest objects, and the other objects a jaw at a pose must clear worked out from them.
        self._scene: Mapping[str, ObjectPose] = {}
        self._others: list[WorldObject] | None = None
        #: The table's correction at the pick [m], or None; whether the table is still trusted; whether the
        #: carry has run the final leg, and whether it runs it again after the latch.
        self._pick_hit: np.ndarray | None = None
        self._table_trusted = True
        self._leg_done = True
        self._releg = False

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Start the grasp family's checks, clear the place state and read the approach table."""
        # A retry's pick widened the attempt budget.
        self.grasp_settings = self.place_settings
        super().start(run, obs)
        self.place = PlaceStatus(target=self.target_label, rests_on=self.rest_on)
        self._reset_place()
        self._start_tcp = self.tcp_cmd.copy()
        self._load_table()

    def _load_table(self) -> None:
        """Read `approach_file` and choose the final leg's direction. A table that cannot be used leaves a
        note, and the run goes on without it."""
        cfg, place = self.place_settings, self.place
        path = cfg.approach_file.strip()
        self._table, problem = None, ""
        if path:
            try:
                table = ApproachTable.load(path)
            except (OSError, ValueError) as exc:
                problem = f"tablica podejscia nieczytelna: {exc}"
            else:
                why = table.problem(self.kin.spec.name)
                if why:
                    problem = f"tablica podejscia nieuzyta: {why}"
                else:
                    self._table = table
        self._direction = cfg.approach_direction or (self._table.preferred_direction if self._table else "")
        if self._table is not None and not self._direction:
            self._table, problem = None, "tablica podejscia bez kierunku podejscia"
        place.table_problem, place.direction = problem, self._direction
        if problem:
            logger.warning("%s (%s)", problem, path)
        if self._table is not None:
            differ = self._protocol_differences(self._table)
            if differ:
                logger.warning("Tablica podejscia zmierzona z innymi ustawieniami: %s", differ)
                place.table_note = f"tablica zmierzona z innymi ustawieniami: {differ}"

    def _protocol_differences(self, table: ApproachTable) -> str:
        """The settings that differ from those the table was measured with, as `name table/here`."""
        protocol = table.data.get("protocol") or {}
        cfg = self.place_settings
        pairs = (("lead_m", cfg.approach_lead_m), ("carry_clearance_m", cfg.carry_clearance_m),
                 ("place_gap_m", cfg.place_gap_m), ("place_speed_mps", cfg.place_speed_mps))
        out = []
        for key, here in pairs:
            there = protocol.get(key)
            if isinstance(there, (int, float)) and abs(float(there) - float(here)) > 1e-9:
                out.append(f"{key} {float(there):g}/{float(here):g}")
        return ", ".join(out)

    def act(self, obs: Observation) -> Action:
        self._scene, self._others, self._obs = obs.objects, None, obs
        self._note_supports(obs)
        return super().act(obs)

    # ------------------------------------------------------------------ judge on or off
    def after_touch(self, obs: Observation) -> Action:
        if self.judged:
            return super().after_touch(obs)
        return super(GraspJevStrategy, self).after_touch(obs)

    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        if self.judged:
            return super().after_grasp_failed(why, obs)
        return super(GraspJevStrategy, self).after_grasp_failed(why, obs)

    def _edge_grip(self) -> bool:
        """As in `grasp`, unless Jev judges the close."""
        return not self.judged and super()._edge_grip()

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Monitor the hold as the grasp family does. A drop ends the place sequence until a regrasp, which
        is a retry when the block fell on the way to the target."""
        if self.judged:
            action = super().carrying(obs, reading)
        else:
            action = super(GraspJevStrategy, self).carrying(obs, reading)
        if action is not None and self.grasp.verdict == "dropped":
            if self._placing and not self._putting_back:
                self._fall(DROPPED, "blok wypadl ze szczeki w drodze")
            self._placing, self._place_plan = False, None
        return action

    # ------------------------------------------------------------------ before the pick
    def _wait(self, obs: Observation) -> Action | None:
        """Require a usable target pose and a reachable place before grasp takes the pick pose."""
        if self.target_label and not self._target_ready:
            why = self._pose_problem(self._pose_of(obs, self.target_label))
            if why:
                if self._t - self._phase_t0 >= self.config.wait_for_pose_s:
                    wait_s = self.config.wait_for_pose_s
                    return self._finish(TARGET_NOT_SEEN, f"brak uzytecznego polozenia celu "
                                                         f"'{self.target_label}' po {wait_s:.0f} s ({why})")
                return self.hold(f"czekam na polozenie celu '{self.target_label}' ({why})")
            self._target_ready = True
        if self.rest_on and not self._target_ready:
            poses = {label: self._pose_of(obs, label) for label in self.rest_on}
            missing = ", ".join(f"'{label}' ({why})" for label, pose in poses.items()
                                if (why := self._pose_problem(pose)))
            if missing:
                if self._t - self._phase_t0 >= self.config.wait_for_pose_s:
                    wait_s = self.config.wait_for_pose_s
                    return self._finish(TARGET_NOT_SEEN, f"brak uzytecznego polozenia podpory {missing} "
                                                         f"po {wait_s:.0f} s")
                return self.hold(f"czekam na polozenie podpory {missing}")
            self._rest_ref = {label: np.asarray(pose.position, float).copy()
                              for label, pose in poses.items() if pose is not None}
            self._target_ready = True
        # Grasp ends here only without a usable pose or with an unreachable one.
        self._pending = NOT_FOUND if self._pose_problem(self._pose_of(obs, self.label)) else UNREACHABLE
        self._goal = self._goal_from(obs)
        action = super()._wait(obs)
        if self.status.phase == DONE:
            return action
        self._pending = ""
        if self.status.phase == WAIT:
            return action
        problem = "" if self._fitted else self._place_problem(obs)
        if problem:
            return self._finish(UNREACHABLE, "; ".join(filter(None, (problem, self._tilt_why))))
        self._find_supports(obs)
        return action

    def _goal_from(self, obs: Observation) -> tuple[np.ndarray, float, float | None] | None:
        """Return the target's centre, top and yaw from `obs`, the pose's, or None without a target pose."""
        if not self.target_label:
            return self._pose()
        pose = self._pose_of(obs, self.target_label)
        return None if self._pose_problem(pose) else self._target_from(pose)

    def plan_problem(self, xy: np.ndarray, cand: GraspCandidate, jaw: JawPlan, tilt: float, grasp_z: float,
                     plan: dict[str, np.ndarray]) -> str:
        """Why no place plan fits a pick with hand tilt `tilt` [rad], the held block estimated centred
        between the closed jaws, or ''."""
        if self._putting_back:
            spot, top = self._put_back_spot()
            goal: tuple[np.ndarray, float, float | None] | None = (spot, top, None)
        else:
            goal = self._goal
        if goal is None:
            return ""
        straddle = plan["straddle_xy"]
        n = np.array([-math.sin(cand.psi), math.cos(cand.psi)])
        hand = np.array([float(self.kin.jaw_centre(jaw.cmd_block)[2]), float((xy - straddle) @ n)])
        saved = self._psi, self._rot, self._tilt, self._psi_rel, self._grip_rel, self._width_m
        self._tilt, self._width_m = tilt, cand.width_m
        self._set_heading(cand.psi, straddle)
        self._grip_rel = self._psi_rel
        try:
            fit = self._plan_place(*goal, from_xy=straddle, hang=grasp_z - self._bottom(), hand_xy=hand)
        finally:
            self._psi, self._rot, self._tilt, self._psi_rel, self._grip_rel, self._width_m = saved
        return fit if isinstance(fit, str) else ""

    def _refuse(self, reason: str) -> Action:
        return self._finish(UNREACHABLE, reason)

    def _place_problem(self, obs: Observation) -> str:
        """Return why no place plan fits, with the held block estimated centred at the grasp height, or ''."""
        if self.target_label:
            pose = self._pose_of(obs, self.target_label)
            if self._pose_problem(pose):
                return ""
            xy, top, yaw = self._target_from(pose)
        else:
            xy, top, yaw = self._pose()
        pick_xy = self._straddle_xy if np.all(np.isfinite(self._straddle_xy)) else self.tcp_cmd[:2]
        plan = self._plan_place(xy, top, yaw, from_xy=pick_xy, hang=self._grasp_z - self._bottom(),
                                hand_xy=np.zeros(2))
        return plan if isinstance(plan, str) else ""

    def _find_supports(self, obs: Observation) -> None:
        """Note the blocks the picked one rests on: centres below its bottom by at most the tallest block's
        height and within `SUPPORT_XY_M` of it across."""
        self._supports = {}
        if self.base_z > 0 and self.grasp.block_xy is not None:
            xy = np.asarray(self.grasp.block_xy, float)
            for key, pose in obs.objects.items():
                label = normalised(str(key))
                if label == self.label or self._pose_problem(pose):
                    continue
                p = np.asarray(pose.position, float)
                if (self.base_z - TABLE_TOP_M <= p[2] < self.base_z
                        and float(np.linalg.norm(p[:2] - xy)) <= SUPPORT_XY_M):
                    self._supports[label] = p.copy()
        self.place.supports = tuple(sorted(self._supports))

    def _note_supports(self, obs: Observation) -> None:
        """After the lift, keep the newest fresh pose of each block the picked one rested on, taken with the
        hand away from it, and how far the farthest moved."""
        if not self._supports or not math.isfinite(self._lifted_t):
            return
        for label, before in self._supports.items():
            pose = self._fresh(obs, label, self._lifted_t)
            if pose is not None and float(np.linalg.norm(self.tcp_cmd[:2] - before[:2])) > IN_HAND_M:
                self._support_now[label] = np.asarray(pose.position, float).copy()
        shifts = [float(np.linalg.norm(now - self._supports[k])) for k, now in self._support_now.items()]
        if not shifts:
            return
        place = self.place
        place.support_shift_m = max(shifts)
        moved = place.support_shift_m > self.place_settings.place_tol_m
        if moved and not place.support_moved:
            logger.warning("Blok pod podniesionym przesuniety o %.0f mm", place.support_shift_m * 1000)
        place.support_moved = moved

    def _put_back_spot(self) -> tuple[np.ndarray, float]:
        """Return where the held block goes back [m, base frame] and the top it lands on [m above the table].

        On the table that is where it was picked. On a stack it is where it stood before the pick, moved with
        the one block it was picked from when that one was seen to move. After a regrasp the aim comes from a
        sighting after the drop, which already shows where the blocks are."""
        if self.base_z <= 0:
            return self._pick_xy, 0.0
        xy, top = self._pick_aim.copy(), self._bottom()
        if len(self._supports) == 1 and not self.grasp.recoveries:
            (label, before), = self._supports.items()
            now = self._support_now.get(label)
            if now is not None and float(np.linalg.norm(now - before)) > self.place_settings.place_tol_m:
                xy, top = xy + (now - before)[:2], max(top + float(now[2] - before[2]), 0.0)
        return xy, top

    # ------------------------------------------------------------------ the approach table
    def lead_in(self, xy: np.ndarray) -> np.ndarray | None:
        """The final leg into the hover over `xy`: `approach_lead_m` along the approach direction, or None
        without one."""
        if not self._direction:
            return None
        try:
            d = direction_vector(self._direction, xy, self._pan_xy)
        except ValueError:
            return None
        return self.place_settings.approach_lead_m * d

    def _lookup(self, xy: np.ndarray, z: float) -> Hit | str:
        """The table's correction for the TCP sent over `xy` at height `z` along the approach direction, or
        why there is none ('' without a table)."""
        if self._table is None:
            return ""
        if not self._table_trusted:
            return "check"
        if self._tilt:
            return "tilt"
        found = self._table.lookup(xy, z, self._direction, robot=self.kin.spec.name,
                                   max_age_days=self.place_settings.max_table_age_days)
        return found if isinstance(found, Hit) else found.reason

    def _first_shift(self, xy: np.ndarray) -> np.ndarray:
        """The pick's first aim takes the table's correction at the grasp height; the residual without one."""
        self._pick_hit = None
        found = self._lookup(np.asarray(xy, float), self._grasp_height(self.top_nominal))
        if isinstance(found, Hit):
            self._pick_hit = np.asarray(found.delta_xy, float).copy()
            self.wrist.residual_m, self.wrist.residual_runs = (0.0, 0.0), 0
            return self._pick_hit
        return super()._first_shift(xy)

    @property
    def _table_aimed(self) -> bool:
        """Whether the pick's first aim took the table's correction."""
        return self._pick_hit is not None and bool(np.any(self._first_shifted))

    @property
    def _saves_residual(self) -> bool:
        return not self._table_aimed

    def _pick_correction(self) -> np.ndarray | None:
        """The pick's own correction for the place, when `pick_shift` allows it and the pick has one."""
        cfg = self.place_settings
        if cfg.pick_shift == OFF or (cfg.pick_shift == AUTO and not cfg.approach_file.strip()):
            return None
        return self.pick_shift()

    def _check_table(self) -> None:
        """Compare the table's correction at the pick with the pick's own. Over `table_check_m` the table is
        not used for the rest of the run, when the pick's correction can stand in for it."""
        if not self._table_aimed or self._pick_hit is None:
            return
        shift = self.pick_shift()
        if shift is None:
            return
        place = self.place
        off = place.table_check_m = float(np.linalg.norm(shift - self._pick_hit))
        if off <= self.place_settings.table_check_m or self._pick_correction() is None:
            return
        self._table_trusted = place.table_trusted = False
        place.table_note = f"tablica nie zgadza sie z chwytem o {off * 1000:.0f} mm - poprawka z chwytu"
        logger.warning("%s", place.table_note)

    def _corrections(self, kind: str, tcp_xy: np.ndarray, release_z: float) -> Correction | None:
        """The correction of `kind` for a release with the TCP over `tcp_xy` as the cameras see it, at
        `release_z`, or None when it does not apply.

        `FROM_TABLE` is the table's, with the final leg it was measured with. The fallback is the pick's own,
        else the table's without the leg, else none; its leg is taken where the arm reaches it."""
        if self._putting_back:
            return None if kind == FROM_TABLE else Correction(np.zeros(2), "", None)
        leg = None if self._tilt else self.lead_in(tcp_xy)
        found = self._lookup(tcp_xy, release_z)
        miss = "" if isinstance(found, Hit) else found
        if kind == FROM_TABLE:
            if not isinstance(found, Hit) or leg is None:
                return None
            return Correction(np.asarray(found.delta_xy, float), FROM_TABLE, leg, leg_required=True)
        if isinstance(found, Hit) and leg is not None:
            # The table applied, but no plan with its leg fitted.
            miss = "lead"
        shift = self._pick_correction()
        if shift is not None:
            return Correction(shift, FROM_PICK, leg, miss=miss)
        if isinstance(found, Hit):
            return Correction(np.asarray(found.delta_xy, float), FROM_TABLE, None, miss=miss)
        return Correction(np.zeros(2), "", leg, miss=miss)

    # ------------------------------------------------------------------ planning
    def _pose(self, shift: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Return the pose's centre [m], the height the held block's bottom lands at [m] and its yaw [rad],
        moved by `shift` (x, y, z) [m]."""
        cfg = self.place_settings
        d = np.zeros(3) if shift is None else np.asarray(shift, float)
        return (np.array([cfg.place_x_m + d[0], cfg.place_y_m + d[1]]), cfg.place_z_m + float(d[2]),
                math.radians(cfg.place_yaw_deg))

    @property
    def _yaw_period(self) -> float:
        """The yaw period the held block lines up with its target by [rad]."""
        if self.target_label:
            return min(self.block.yaw_period, self.target_block.yaw_period)
        return self.block.yaw_period

    def _where(self) -> str:
        """The target block, a pose's supports or the table, for notes."""
        if self.target_label:
            return f"'{self.target_label}'"
        return _names(self.rest_on) if self.rest_on else "stol"

    def _bottom(self) -> float:
        """The picked block's bottom height at the close [m above the table]."""
        top = self.grasp.top_m if math.isfinite(self.grasp.top_m) else self.top_nominal
        return top - self.height_m

    def _pose_of(self, obs: Observation, label: str) -> ObjectPose | None:
        return {normalised(str(key)): value for key, value in obs.objects.items()}.get(label)

    def _fresh(self, obs: Observation, label: str, since: float = -math.inf,
               max_age: float | None = None) -> ObjectPose | None:
        """Return a usable pose of `label` captured at `since` or later and no older than `max_age`
        (`fresh_pose_s` by default)."""
        pose = self._pose_of(obs, label)
        if pose is None or self._pose_problem(pose):
            return None
        age = float(pose.age_s)
        limit = self.place_settings.fresh_pose_s if max_age is None else max_age
        if not age <= limit or float(obs.t) - age < since - 1e-9:
            return None
        return pose

    def _target_from(self, pose: ObjectPose | None) -> tuple[np.ndarray, float, float | None]:
        """Return a target block's centre [m], top [m above the table] and yaw [rad] or None."""
        assert pose is not None
        position = np.asarray(pose.position, float)
        half = self.target_block.half_size[2]
        # A pose at table height still has a block below it.
        return position[:2].copy(), max(float(position[2]) + half, 2 * half), yaw_of(pose.rotation)

    def _reaches(self, point: np.ndarray, psi: float, seed: Mapping[str, float]) -> dict[str, float] | None:
        """Return joints that put the TCP on `point` with the hand down, at heading `psi` and the pick's tilt,
        or None."""
        sol = self.kin.ik(point, hand_down(psi, self._tilt), seed=seed, accept=self._clear,
                          restarts=ik_restarts(self._tilt))
        if sol.ok and self._clear(sol.joints) and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG):
            return sol.joints
        return None

    def _turns(self, yaw: float | None, xy: np.ndarray | None = None, hand_xy: np.ndarray | None = None
               ) -> list[float]:
        """Return heading turns [rad] that line the held block up with `yaw`, smallest first.

        A tilted hand keeps the grasp's heading against the arm over `xy`, the held block's centre, with the
        block at `hand_xy` in the hand: the turn that does so, or the turns that line up within
        `MAX_ROT_ERR_DEG` of block tilt from it."""
        free = yaw is None or not self.place_settings.align_yaw or self._putting_back
        turns = [0.0]
        if not free:
            assert yaw is not None
            period = self._yaw_period
            d = fold(yaw - self._held_rel - self._psi, period)
            near = {round(d + k * period, 9) for k in range(-4, 5) if abs(d + k * period) <= math.pi + 1e-9}
            turns = sorted(near, key=abs)
        if not self._tilt or xy is None or hand_xy is None:
            return turns
        psi = self._bearing(xy) + self._grip_rel
        for _ in range(2):
            psi = self._bearing(xy - rot2(psi) @ hand_xy) + self._grip_rel
        keep = wrap(psi - self._psi)
        if free:
            return [keep]
        limit = math.asin(min(1.0, math.sin(math.radians(MAX_ROT_ERR_DEG)) / math.sin(abs(self._tilt))))
        return [t for t in turns if abs(wrap(t - keep)) <= limit]

    def _plan_place(self, xy: np.ndarray, top: float, yaw: float | None, *, from_xy: np.ndarray | None = None,
                    hang: float | None = None, hand_xy: np.ndarray | None = None,
                    over_m: float = 0.0) -> PlacePlan | str:
        """Plan the carry, release and retreat over `xy` at surface `top`, or say why none is reachable.

        The carry starts at `from_xy` (the TCP now) and also clears `over_m` and what the block was picked
        from. `hang` and `hand_xy` place the held block in the hand; the measured ones by default. The
        table's correction with its final leg comes first, at any heading; the fallback (`_corrections`)
        after it.
        """
        cfg = self.place_settings
        hang = self._hang if hang is None else hang
        hand_xy = self._hand_xy if hand_xy is None else hand_xy
        from_xy = self.tcp_cmd[:2].copy() if from_xy is None else np.asarray(from_xy, float).copy()
        release_z, want, low = place_heights(top, hang, cfg.place_gap_m, GRIP_SLIP_M, TABLE_TOP_M,
                                             cfg.carry_clearance_m, MIN_CARRY_CLEAR_M,
                                             over=max(over_m, self._bottom()))
        if over_m > 0:
            # Never lower the held block over what the hand hovers above.
            want = max(want, float(self.tcp_cmd[2]))
        turns = self._turns(yaw, np.asarray(xy, float), hand_xy)
        if not turns:
            return "dlon pochylona nie ustawi bloku zgodnie z celem"
        # TCP heights where the jaw at the release opening clears the placed block by `retreat_clear_m` and
        # by `JAW_CLEAR_M`, and where the fully open moving jaw does.
        placed_top = top + cfg.place_gap_m + self.height_m
        reach = self.kin.jaw_reach(self._place_open_cmd, tilt=self._tilt)
        safe_z = placed_top + cfg.retreat_clear_m + reach
        clear_z = placed_top + JAW_CLEAR_M + reach
        wide_z = placed_top + JAW_CLEAR_M + self.kin.jaw_reach(FULL_OPEN, moving=True, tilt=self._tilt)
        heights = (release_z, want, low, safe_z, clear_z, wide_z)
        problem = ""
        for kind in (FROM_TABLE, ""):
            fallback, problem = None, ""
            for turn in turns:
                psi = wrap(self._psi + turn)
                tcp_xy = np.asarray(xy, float) - rot2(psi) @ hand_xy
                corr = self._corrections(kind, tcp_xy, release_z)
                if corr is None:
                    continue
                plan = self._plan_turn(xy, top, yaw, turn, tcp_xy, corr, heights, from_xy, fallback is None)
                if isinstance(plan, str):
                    problem = problem or plan
                    continue
                if not plan.wide:
                    return plan
                # Once a heading that opens fully is known, only one that rises clear is worth more.
                fallback = fallback or plan
            if fallback is not None:
                return fallback
        return problem

    def _plan_turn(self, xy: np.ndarray, top: float, yaw: float | None, turn: float, tcp_xy: np.ndarray,
                   corr: Correction, heights: tuple[float, float, float, float, float, float], from_xy: np.ndarray,
                   may_open_wide: bool) -> PlacePlan | str:
        """Plan the place at heading turn `turn` with the TCP over `tcp_xy` as the cameras see it, sent there
        plus `corr`, or say why it does not fit. The jaw's clearance counts where the cameras see the
        place; the reach where the hand is sent."""
        release_z, want, low, safe_z, clear_z, wide_z = heights
        psi = wrap(self._psi + turn)
        u = np.array([math.cos(psi), math.sin(psi)])
        cmd_xy = tcp_xy + corr.delta
        eased, eased_seen = cmd_xy - self._ease_m * u, tcp_xy - self._ease_m * u
        away = self._start_tcp[:2] - eased
        leaves = float(away @ u) <= -AWAY_COS * float(np.linalg.norm(away)) - 1e-9
        where = f"({cmd_xy[0]:.3f}, {cmd_xy[1]:.3f}, {release_z:.3f}) m"
        seed0 = self.kin.from_q(self.q_cmd)
        hit = self._jaw_hit(psi, (np.array([*tcp_xy, release_z]), np.array([*eased_seen, release_z])),
                            self._place_open_cmd)
        if hit:
            return f"szczeka zahaczylaby o '{hit}' przy odkladaniu"
        at_release = self._reaches(np.array([*cmd_xy, release_z]), psi, seed0)
        if at_release is None:
            return f"punkt odlozenia {where} poza zasiegiem z chwytakiem w dol"
        wide_ok = wide_z if leaves and may_open_wide else math.inf
        retreat = self._retreat(eased, psi, release_z, (safe_z, clear_z), wide_ok, at_release)
        if retreat is None:
            return f"po odlozeniu szczeki nie mina bloku: wysokosc {clear_z:.3f} m poza zasiegiem"
        retreat_z, wide = retreat
        hit = self._jaw_hit(psi, (np.array([*eased_seen, retreat_z]),), FULL_OPEN) if wide else ""
        if hit:
            return f"otwarta szczeka zahaczylaby o '{hit}' przy odjezdzie"
        carry_z, leg = None, corr.leg
        for z in carry_heights(want, low):
            up = self._reaches(np.array([*from_xy, z]), self._psi, seed0)
            over = None if up is None else self._reaches(np.array([*cmd_xy, z]), psi, up)
            if over is None:
                continue
            if leg is not None and self._reaches(np.array([*(cmd_xy - leg), z]), psi, over) is None:
                if corr.leg_required:
                    continue
                leg = None
            carry_z = z
            break
        if carry_z is None:
            if corr.leg_required:
                return f"podejscie do {where} z kierunku {self._direction} poza zasiegiem"
            return f"przeniesienie nad cel na wysokosci {low:.3f} m poza zasiegiem"
        return PlacePlan(np.asarray(xy, float).copy(), float(top), yaw, psi, turn, np.asarray(tcp_xy, float),
                         from_xy, float(carry_z), float(release_z), float(retreat_z), wide,
                         np.asarray(corr.delta, float).copy(), corr.source,
                         None if leg is None else np.asarray(leg, float).copy(), corr.miss)

    def _retreat(self, eased: np.ndarray, psi: float, release_z: float, clear: tuple[float, float],
                 wide_z: float, seed: Mapping[str, float]) -> tuple[float, bool] | None:
        """Return the open hand's retreat height [m] and whether the jaw must then open fully, or None.

        `clear` holds the heights where the jaw clears the block by `retreat_clear_m` and by `JAW_CLEAR_M`.
        The hand rises `retreat_m` and to the first, or less when out of reach, but at least to the second.
        Below that, a retreat at or above `wide_z` lets the fully open jaw clear the block; infinite rules
        it out.
        """
        safe, low = (max(z, release_z) for z in clear)
        for z in dict.fromkeys((max(release_z + self.place_settings.retreat_m, safe), safe, low)):
            if self._reaches(np.array([*eased, z]), psi, seed) is not None:
                return z, False
        for z in carry_heights(low, max(wide_z, release_z))[1:]:
            if self._reaches(np.array([*eased, z]), psi, seed) is not None:
                return z, True
        return None

    def _jaw_hit(self, psi: float, tcps: tuple[np.ndarray, ...], cmd: float) -> str:
        """At a pose, the first other object that the jaw at `cmd` [app units], with heading `psi` [rad] and
        its TCP at any of `tcps` [m], comes within `JAW_SIDE_CLEAR_M` of; '' when none does.

        Each jaw body counts as its outline along and across the closing axis, from its points below the
        object's top."""
        if self.target_label or self._putting_back:
            return ""
        others = self._other_objects()
        if not others:
            return ""
        R = hand_down(psi, self._tilt)
        u, v = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
        clouds = [np.vstack([points @ R.T + tcp for tcp in tcps]) for points in self._jaw_cloud(cmd)]
        for obj in others:
            for cloud in clouds:
                low = cloud[cloud[:, 2] < obj.top_m + JAW_SIDE_CLEAR_M]
                if not len(low):
                    continue
                a, b = low[:, :2] @ u, low[:, :2] @ v
                corners = ((a.min(), b.min()), (a.max(), b.min()), (a.max(), b.max()), (a.min(), b.max()))
                box = tuple((float(p[0]), float(p[1])) for p in (u * s + v * t for s, t in corners))
                if overlap(box, obj.footprint) > -JAW_SIDE_CLEAR_M:
                    return obj.label
        return ""

    def _jaw_cloud(self, cmd: float) -> tuple[np.ndarray, ...]:
        """The jaw bodies' points in the TCP frame at `cmd` [app units], cached."""
        key = round(float(cmd), 3)
        if key not in self._jaw_clouds:
            self._jaw_clouds[key] = self.kin.jaw_points(key)
        return self._jaw_clouds[key]

    def _other_objects(self) -> list[WorldObject]:
        """The latest observation's objects besides the held block, a pose's supports and anything in the
        gripper, with their sizes from the catalog."""
        if self._others is None:
            skip = {self.label, *self.rest_on}
            world = world_state(self._scene, place_check=_any_place)
            self._others = [o for o in world.objects.values() if o.label not in skip and not o.in_gripper]
        return self._others

    def _take_plan_place(self, plan: PlacePlan) -> None:
        self._place_plan = plan
        place = self.place
        place.target_xy = (float(plan.target_xy[0]), float(plan.target_xy[1]))
        place.target_top_m = plan.top_m
        place.target_yaw_deg = math.degrees(plan.yaw) if plan.yaw is not None else float("nan")
        place.heading_deg = math.degrees(plan.psi)
        place.turn_deg = math.degrees(plan.turn)
        place.carry_z, place.release_z, place.retreat_z = plan.carry_z, plan.release_z, plan.retreat_z
        place.correction_m = (float(plan.correction[0]), float(plan.correction[1]))
        place.correction_from, place.table_miss = plan.source, plan.miss
        place.lead_m = 0.0 if plan.leg is None else float(np.linalg.norm(plan.leg))
        self._over_top = max(self._over_top, plan.top_m)
        if not self._putting_back:
            self._goal = (plan.target_xy.copy(), plan.top_m, plan.yaw)

    # ------------------------------------------------------------------ after the lift
    def _lifted(self, obs: Observation) -> Action | None:
        """Start the place once the lift is verified, instead of grasp's carry."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        self._start_place()
        return None

    def _start_place(self) -> None:
        """Measure the held block in the hand, then look for the target."""
        kin, measured = self.kin, self._measured
        fixed, moving = kin.fingertips(measured)
        u = (moving - fixed)[:2]
        norm = float(np.linalg.norm(u))
        u = u / norm if norm > 1e-9 else np.array([math.cos(self._psi), math.sin(self._psi)])
        mid = ((fixed + moving) / 2)[:2]
        assert self.grasp.block_xy is not None
        aim = np.asarray(self.grasp.block_xy, float)
        # The jaws centre the block along the closing axis. Across it the block stays at the aim.
        centre = aim + float((mid - aim) @ u) * u
        T = kin.tcp(measured)
        self._hand_xy = rot2(-heading(kin, measured)) @ (centre - T[:2, 3])
        self._hang = float(self._close_at[2]) - self._bottom()
        self._pick_xy, self._pick_aim = centre, aim.copy()
        yaw = known_yaw(self.block_rotation)
        # Squeezed faces lie flat on the pads, so the block sits a whole quarter turn from the heading.
        rel = -math.pi / 2 if yaw is None else wrap(yaw - self._psi)
        self._held_rel = round(rel / (math.pi / 2)) * (math.pi / 2)
        self.place.hang_m = self._hang
        if not math.isfinite(self._lifted_t):
            self._lifted_t = self._t
        self._placing, self._place_plan, self._replans, self._over_top = True, None, 0, 0.0
        self._retrying = ""
        self._target_seen, self._rest_seen = None, {}
        self._check_table()
        self._enter(TARGET)

    def _target(self, obs: Observation) -> Action | None:
        """Plan the place from the target's pose, or the pick spot when putting the block back."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        over = 0.0
        if self._putting_back:
            (xy, top), yaw, over = self._put_back_spot(), None, self._over_top
        elif self.target_label:
            pose = self._pose_of(obs, self.target_label)
            why = self._pose_problem(pose)
            if why:
                if self._t - self._phase_t0 >= self.place_settings.target_wait_s:
                    return self._put_back(TARGET_NOT_SEEN,
                                          f"brak polozenia celu '{self.target_label}' ({why})")
                return self.hold(f"czekam na polozenie celu '{self.target_label}' ({why})")
            xy, top, yaw = self._target_from(pose)
        else:
            xy, top, yaw = self._pose()
        plan = self._plan_place(xy, top, yaw, over_m=over)
        if isinstance(plan, str):
            if self._putting_back:
                return self._finish(self._put_back_outcome, f"{self._put_back_why}; odlozenie z powrotem "
                                                            f"niemozliwe: {plan} - blok w szczekach")
            return self._put_back(UNREACHABLE, plan)
        self._take_plan_place(plan)
        self._move(CARRY_UP, np.array([*plan.from_xy, plan.carry_z]))
        return None

    def _note_target(self, obs: Observation) -> None:
        """Keep the newest fresh camera pose of the target, or of each support of a pose, while carrying
        toward it."""
        if self._putting_back:
            return
        if self.target_label:
            pose = self._fresh(obs, self.target_label)
            if pose is not None:
                self._target_seen = (pose, float(obs.t) - float(pose.age_s))
        for label in self.rest_on:
            pose = self._fresh(obs, label)
            if pose is not None:
                self._rest_seen[label] = (pose, float(obs.t) - float(pose.age_s))

    def _put_back(self, outcome: str, why: str) -> Action:
        """Return the held block to where it was picked, then end with `outcome`."""
        self._putting_back, self._put_back_outcome, self._put_back_why = True, outcome, why
        self.place.put_back = True
        logger.info("Odkladam blok z powrotem: %s", why)
        self._enter(TARGET)
        return self.hold(f"{why} - odkladam blok z powrotem")

    def _carry_up(self, obs: Observation) -> Action | None:
        """Move the held block straight to the carry height."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        self._note_target(obs)
        plan = self._place_plan
        assert plan is not None
        point = np.array([*plan.from_xy, plan.carry_z])
        if self._arrived(point, sag=True):
            self._set_heading(plan.psi, plan.cmd_xy)
            self._carry_to(plan)
            return None
        if self._timed_out():
            mm = self.status.distance * 1000
            return self._halt(f"ramie nie unioslo bloku na wysokosc przeniesienia ({mm:.0f} mm od celu)",
                              self.kin.from_q(self.q_cmd))
        return self._drive(point, self.approach_step_m)

    def _carry_to(self, plan: PlacePlan) -> None:
        """Carry the block over the target, through the start of the plan's final leg when it has one."""
        self._leg_done = plan.leg is None
        self._move(CARRY_OVER, plan.hover, plan.turn, via=plan.lead_point)

    def _carry_over(self, obs: Observation) -> Action | None:
        """Carry the block over the target and turn it to the planned heading, the final leg last."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        self._note_target(obs)
        plan = self._place_plan
        assert plan is not None
        lead = plan.lead_point
        if not self._leg_done and lead is not None:
            if self._arrived(lead, sag=True):
                self._leg_done = True
            elif self._timed_out():
                mm = self.status.distance * 1000
                return self._halt(f"ramie nie doszlo do poczatku podejscia ({mm:.0f} mm od niego)",
                                  self.kin.from_q(self.q_cmd))
            else:
                return self._drive(lead, self.approach_step_m)
        if self._settled(plan.hover, sag=True) and self._vertical():
            # The descent's stall reference.
            self._take_reference()
            self._slow = False
            latch = (self.target_label or self.rest_on) and not self._putting_back and not self._releg
            self._releg = False
            self._enter(LATCH if latch else PLACE)
            return None
        if self._timed_out():
            if not self._vertical():
                return self._halt(self._not_vertical(), self.kin.from_q(self.q_cmd))
            return self._halt(f"ramie nie stanelo nad celem ({self.status.distance * 1000:.0f} mm od zawisu)",
                              self.kin.from_q(self.q_cmd))
        return self._drive(plan.hover, self.approach_step_m)

    def _latch(self, obs: Observation) -> Action | None:
        """Latch the target pose seen on the way or at the hover, or re-plan the hover if the target moved."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        if not self.target_label:
            return self._latch_supports(obs)
        plan, cfg, label = self._place_plan, self.place_settings, self.target_label
        assert plan is not None
        self._note_target(obs)
        seen = self._target_seen
        pose = seen[0] if seen is not None and seen[1] >= self._over_t0 - cfg.fresh_pose_s - 1e-9 else None
        if pose is None:
            if self._t - self._phase_t0 >= cfg.target_wait_s:
                return self._put_back(TARGET_NOT_SEEN, f"cel '{label}' bez swiezego polozenia od "
                                                       f"{cfg.target_wait_s:.0f} s")
            return self.hold()
        xy, top, yaw = self._target_from(pose)
        self._over_top = max(self._over_top, top)
        moved = float(np.linalg.norm(xy - plan.target_xy))
        turned = 0.0
        if cfg.align_yaw and yaw is not None and plan.yaw is not None:
            period = min(self.block.yaw_period, self.target_block.yaw_period)
            turned = abs(fold(yaw - plan.yaw, period))
        small = (moved <= LATCH_SAME_M and abs(top - plan.top_m) <= LATCH_SAME_M
                 and turned <= math.radians(LATCH_SAME_DEG))
        return self._follow(xy, top, yaw, moved, small, f"cel '{label}' nie stoi w miejscu")

    def _latch_supports(self, obs: Observation) -> Action | None:
        """Latch each support's pose seen on the way or at the hover and move the pose with them, or put the
        block back when they moved apart."""
        plan, cfg = self._place_plan, self.place_settings
        assert plan is not None
        self._note_target(obs)
        since = self._over_t0 - cfg.fresh_pose_s - 1e-9
        missing = tuple(label for label in self.rest_on
                        if label not in self._rest_seen or self._rest_seen[label][1] < since)
        if missing:
            if self._t - self._phase_t0 >= cfg.target_wait_s:
                return self._put_back(TARGET_NOT_SEEN, f"podpora {_names(missing)} bez swiezego polozenia "
                                                       f"od {cfg.target_wait_s:.0f} s")
            return self.hold()
        at = {label: np.asarray(self._rest_seen[label][0].position, float).copy() for label in self.rest_on}
        shifts = np.array([at[label] - self._rest_ref[label] for label in self.rest_on])
        apart = max(float(np.linalg.norm(a[:2] - b[:2])) for a in shifts for b in shifts)
        if apart > cfg.place_tol_m:
            return self._put_back(TARGET_MOVED, f"podpory {_names(self.rest_on)} przesuniete wzgledem siebie "
                                                f"o {apart * 1000:.0f} mm")
        xy, top, yaw = self._pose(np.array([*shifts[:, :2].mean(axis=0), shifts[:, 2].max()]))
        self._rest_pos = at
        self._over_top = max(self._over_top, top)
        moved = float(np.linalg.norm(xy - plan.target_xy))
        small = moved <= LATCH_SAME_M and abs(top - plan.top_m) <= LATCH_SAME_M
        return self._follow(xy, top, yaw, moved, small, f"podpory {_names(self.rest_on)} nie stoja w miejscu")

    def _follow(self, xy: np.ndarray, top: float, yaw: float | None, moved: float, small: bool,
                why: str) -> Action | None:
        """Place at the latched target: a small change rides along the descent, a larger one re-plans the
        hover, and a target that keeps moving sends the block back with `why`."""
        plan = self._place_plan
        assert plan is not None
        if not small and self._replans >= MAX_REPLANS:
            return self._put_back(TARGET_MOVED, f"{why} ({moved * 1000:.0f} mm)")
        new = self._plan_place(xy, top, yaw)
        if isinstance(new, str):
            return self._put_back(UNREACHABLE, new)
        if small and abs(wrap(new.psi - plan.psi)) <= math.radians(LATCH_SAME_DEG):
            if new.leg is not None and float(np.linalg.norm(new.cmd_xy - plan.cmd_xy)) > RELEG_M:
                # The hand must still arrive along the final leg: back to the leg's start and along it again,
                # then down without another latch.
                self._take_plan_place(replace(new, from_xy=plan.from_xy, turn=plan.turn))
                self._set_heading(new.psi, new.cmd_xy)
                self._releg = True
                self._carry_to(new)
                return self.hold(f"cel przesuniety o {moved * 1000:.0f} mm - podchodze jeszcze raz")
            # Small corrections ride along the descent, which starts no higher than the hand is.
            self._take_plan_place(replace(new, from_xy=plan.from_xy, turn=plan.turn,
                                          carry_z=min(new.carry_z, plan.carry_z)))
            self._set_heading(new.psi, new.cmd_xy)
            self._enter(PLACE)
            return None
        if self._replans >= MAX_REPLANS:
            return self._put_back(TARGET_MOVED, f"{why} ({moved * 1000:.0f} mm)")
        self._replans = self.place.replans = self._replans + 1
        # Keep the carry's start: the arm is already at carry height.
        self._take_plan_place(replace(new, from_xy=plan.from_xy))
        self._set_heading(new.psi, new.cmd_xy)
        logger.info("Cel przesuniety o %.0f mm - nowy zawis", moved * 1000)
        self._carry_to(new)
        return self.hold(f"cel przesuniety o {moved * 1000:.0f} mm - nowy zawis")

    def _place(self, obs: Observation) -> Action | None:
        """Lower the held block until it stalls on the target or reaches the release height.

        The arm goes down at `speed_mps` to `SLOW_ABOVE_M` over the release height, or stays at the
        carry height when that is lower, and settles there. The rest runs at `place_speed_mps` against
        a fresh stall reference.
        """
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        plan = self._place_plan
        assert plan is not None
        slow = self.place_settings.place_speed_mps
        above_z = min(plan.release_z + SLOW_ABOVE_M, plan.carry_z)
        down_s = (2 * (plan.carry_z - above_z) / self.config.speed_mps + (above_z - plan.release_z) / slow
                  + SETTLE_S)
        if self._t - self._phase_t0 >= down_s + JAW_WAIT_S:
            self._hold_measured()
            return self._halt("ramie nie opuscilo bloku na cel w czasie", self.kin.from_q(self.q_cmd))
        if not self._slow:
            above = np.array([*plan.cmd_xy, above_z])
            if not self._settled(above):
                return self._drive(above, self.approach_step_m)
            self._slow = True
            self._take_reference()
        st = self.status
        st.rise_m = self._contact_rise()
        if st.rise_m > self.config.contact_gap_m:
            # Stop pressing once the block meets the target.
            self._hold_measured()
            self.place.contact = True
            return self._let_go_now()
        if self._arrived(plan.release):
            return self._let_go_now()
        return self._drive(plan.release, slow / self.hz)

    def _let_go_now(self) -> Action | None:
        self._released, self._rise_extra = False, 0.0
        self._release_tcp_z = float(self.kin.tcp(self.kin.from_q(self.q_cmd))[2, 3])
        self._enter(LET_GO)
        return None

    # ------------------------------------------------------------------ the release
    @property
    def _place_open_cmd(self) -> float:
        """The jaw command that opens `release_margin_m` past the held block on each side, and past the held
        jaw once a release began [app units]."""
        held = self._held_gap_m if math.isfinite(self._held_gap_m) else self._width_m
        gap = max(self._width_m, held) + 2 * self.place_settings.release_margin_m
        return self.kin.gripper_for_gap(min(gap, self.kin.jaw_gap(FULL_OPEN)))

    @property
    def release_cmd(self) -> float:
        """While placing, open just past the held block. Otherwise open as `grasp_jev` does."""
        return self._place_open_cmd if self._placing else super().release_cmd

    @property
    def ease_off_m(self) -> float:
        """While placing, a few millimetres at most: the jaw opened in place around the block."""
        return self._ease_m if self._placing else super().ease_off_m

    @property
    def rise_step_m(self) -> float:
        """While placing, the open hand rises at `retreat_speed_mps`."""
        return self.place_settings.retreat_speed_mps / self.hz if self._placing else super().rise_step_m

    def _let_go_rise_z(self) -> float:
        """While placing, rise to the retreat, higher by as much as a stall left the block higher if the arm
        reaches."""
        if not self._placing or self._place_plan is None:
            return super()._let_go_rise_z()
        plan, z = self._place_plan, float(self.tcp_cmd[2])
        rise_z = plan.retreat_z + max(self._release_tcp_z - plan.release_z, 0.0)
        if rise_z > plan.retreat_z + 1e-9:
            eased = self.tcp_cmd[:2] - self.ease_off_m * np.array([math.cos(self._psi), math.sin(self._psi)])
            if self._reaches(np.array([*eased, rise_z]), self._psi, self.kin.from_q(self.q_cmd)) is None:
                rise_z = plan.retreat_z
        return max(rise_z, z)

    def _after_let_go(self) -> Action | None:
        """While placing, leave the block for the cameras once the open jaw clears it.

        The open jaw's lowest point, from the measured joints, must stand `retreat_clear_m` above the placed
        block's top. An arm that settled short rises by what is missing, `sag_m` at most in all, where it
        reaches. Where it cannot, a jaw `JAW_CLEAR_M` clear still leaves. Below that the jaw opens fully
        first, and the hand must leave away from the fixed jaw with the moving jaw over the block.
        """
        if not self._placing:
            return super()._after_let_go()
        plan, cfg = self._place_plan, self.place_settings
        assert plan is not None
        top = self._placed_top()
        clear = self._jaw_low_z() - top
        short = cfg.retreat_clear_m - clear
        if short > CLEAR_TOL_M and self._rise_extra + short <= self.config.sag_m + 1e-9:
            point = self._release_point + np.array([0.0, 0.0, short])
            if self._reaches(point, self._psi, self.kin.from_q(self.q_cmd)) is not None:
                self._rise_extra += short
                self._release_point = point
                return self._drive(point, self.rise_step_m)
        self._released_t = self._t
        self.place.jaw_clear_m = clear
        park_z = max(float(self._start_tcp[2]), float(self.tcp_cmd[2]))
        self._park_point = np.array([*self._start_tcp[:2], park_z])
        self._opened = clear >= JAW_CLEAR_M - CLEAR_TOL_M
        if not self._opened:
            u = np.array([math.cos(self._psi), math.sin(self._psi)])
            away = self._park_point[:2] - self.tcp_cmd[:2]
            leaves = float(away @ u) <= -AWAY_COS * float(np.linalg.norm(away)) + 1e-9
            wide = float(self._tcp_measured[2]) - self.kin.jaw_reach(FULL_OPEN, moving=True, tilt=self._tilt)
            if not leaves or wide - top < JAW_CLEAR_M - CLEAR_TOL_M:
                return self._finish(STOPPED, "otwarta reka nie odjedzie od odlozonego bloku bez "
                                             "potracenia go")
            self.place.opened_wide = True
        elif clear < cfg.retreat_clear_m - CLEAR_TOL_M:
            logger.info("Szczeka tylko %.0f mm nad odlozonym blokiem", clear * 1000)
        distance = float(np.linalg.norm(self._park_point - self.tcp_cmd))
        self._near_m = (self._jaw_radius(self._place_open_cmd if self._opened else FULL_OPEN)
                        + math.hypot(*self.block.half_size[:2]) + cfg.retreat_clear_m)
        self._park_s = (min(distance, self._near_m) / cfg.retreat_speed_mps + distance / self.config.speed_mps
                        + JAW_WAIT_S * (1 if self._opened else 2))
        self._park_held = True
        self._enter(PARK)
        return None

    def _placed_top(self) -> float:
        """The placed block's top [m above the table], higher when the descent stalled above the release
        height."""
        plan = self._place_plan
        assert plan is not None
        return (plan.top_m + self.place_settings.place_gap_m + self.height_m
                + max(self._release_tcp_z - plan.release_z, 0.0))

    def _jaw_low_z(self) -> float:
        """The jaw's lowest point at the measured joints [m above the table]."""
        joints, name = self._measured, self.kin.spec.gripper
        assert name is not None
        grip = float(joints.get(name, self.kin.from_q(self.q_cmd)[name]))
        T = self.kin.tcp(joints)
        points = np.vstack(self._jaw_cloud(round(grip, 1))) @ T[:3, :3].T + T[:3, 3]
        return float(points[:, 2].min())

    def _jaw_radius(self, cmd: float) -> float:
        """How far the jaw at `cmd` [app units] reaches across from the TCP of the last target [m]."""
        R = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, :3]
        points = np.vstack(self._jaw_cloud(cmd)) @ R.T
        return float(np.hypot(points[:, 0], points[:, 1]).max())

    def _park(self, obs: Observation) -> Action | None:
        """Open the jaw fully if leaving needs it, then move the open hand toward the start position.

        The way never goes lower. While the TCP is within `_near_m` of the placed block across, the hand
        keeps its orientation and moves at `retreat_speed_mps`. A point out of reach ends the move early.
        """
        if not self._opened:
            self._set_jaw_goal(FULL_OPEN)
            self._ramp_jaw()
            jaw_open = abs(self.reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
            if self._jaw_ready() and jaw_open:
                self._opened = True
            elif self._t - self._phase_t0 >= JAW_WAIT_S:
                return self._finish(STOPPED, f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s przed odjazdem")
            return self.hold()
        # The move ends on its own time limit, so a still arm short of the point is no stall here.
        if self._arrived(self._park_point, sag=True, watch=False) or self._t - self._phase_t0 >= self._park_s:
            if self._putting_back:
                return self._finish(self._put_back_outcome,
                                    f"{self._put_back_why} - blok odlozony z powrotem")
            self._enter(VERIFY)
            return None
        plan = self._place_plan
        assert plan is not None
        if float(np.linalg.norm(self.tcp_cmd[:2] - plan.target_xy)) >= self._near_m:
            return TouchStrategy._drive(self, self._park_point, self.approach_step_m)
        if not self._park_held:
            return TouchStrategy._drive(self, self._park_point, self.rise_step_m)
        action = self._drive(self._park_point, self.rise_step_m)
        # Where the arm cannot keep the hand's orientation, the way goes on without it.
        self._park_held = not self.status.out_of_reach
        return action

    # ------------------------------------------------------------------ the check
    def _verify(self, obs: Observation) -> Action:
        """Check fresh poses captured after the release against the plan."""
        plan, cfg, place = self._place_plan, self.place_settings, self.place
        assert plan is not None
        since = self._released_t + DROP_SETTLE_S
        mine = self._fresh(obs, self.label, since)
        late = self._t - self._phase_t0 >= cfg.verify_wait_s - 1e-9
        if mine is None:
            if late:
                return self._finish(NOT_FOUND, f"odlozony blok '{self.label}' niewidoczny w kamerach od "
                                               f"{cfg.verify_wait_s:.0f} s")
            return self.hold()
        rests = {label: self._fresh(obs, label, since) for label in self.rest_on}
        if not late and (any(label not in self._support_now for label in self._supports)
                         or any(pose is None for pose in rests.values())):
            # The blocks the picked one rested on, and a pose's supports, get the same wait.
            return self.hold()
        # What the block rests on may have moved: where it should lie now [m], and why.
        moved, shift = "", np.zeros(3)
        if self.target_label:
            target = self._fresh(obs, self.target_label, since)
            if target is not None:
                xy, top, _ = self._target_from(target)
                shift = np.array([*(xy - plan.target_xy), top - plan.top_m])
                place.target_shift_m = float(np.linalg.norm(shift[:2]))
                if place.target_shift_m > cfg.place_tol_m:
                    moved = (f"cel '{self.target_label}' przesuniety o {place.target_shift_m * 1000:.0f} mm "
                             "przy odkladaniu")
        offs = {label: np.asarray(pose.position, float) - self._rest_pos[label]
                for label, pose in rests.items() if pose is not None}
        if offs:
            worst = max(offs, key=lambda label: float(np.linalg.norm(offs[label])))
            place.target_shift_m = float(np.linalg.norm(offs[worst]))
            d = np.array(list(offs.values()))
            shift = np.array([*d[:, :2].mean(axis=0), d[:, 2].max()])
            if place.target_shift_m > cfg.place_tol_m:
                moved = (f"podpora '{worst}' przesunieta o {place.target_shift_m * 1000:.0f} mm przy "
                         "odkladaniu")
        p = np.asarray(mine.position, float)
        error = float(np.linalg.norm(p[:2] - plan.target_xy))
        dz = float(p[2]) - (plan.top_m + self.height_m / 2)
        place.placed_xyz = (float(p[0]), float(p[1]), float(p[2]))
        place.error_m, place.height_error_m = error, dz
        place.placed_tilt_deg = lean = _lean_deg(mine.rotation, self.block)
        # A target block's place is not checked for level.
        level = bool(self.target_label) or not lean > LEVEL_TOL_DEG
        if moved:
            # A block still on what moved needs no new pick.
            off = float(np.linalg.norm(p[:2] - plan.target_xy - shift[:2]))
            if off <= cfg.place_tol_m and abs(dz - shift[2]) <= PLACED_Z_TOL_M and level:
                return self._finish(TARGET_MOVED, moved)
            return self._fell_off(TARGET_MOVED, moved)
        where = self._where()
        offsets = f"{error * 1000:.0f} mm od srodka, {dz * 1000:+.0f} mm w pionie"
        if not level:
            offsets += f", pochylony o {lean:.0f} st."
        if error <= cfg.place_tol_m and abs(dz) <= PLACED_Z_TOL_M and level:
            tilted = f" (dlon pochylona {self.grasp.tilt_deg:.0f} st.)" if self._tilt else ""
            return self._finish(PLACED, f"blok '{self.label}' odlozony na {where}: {offsets}{tilted}",
                                success=True)
        return self._fell_off(FELL_OFF, f"blok '{self.label}' nie lezy na {where}: {offsets}")

    # ------------------------------------------------------------------ after a fall
    def _fall(self, outcome: str, why: str) -> bool:
        """Note that the block fell with `outcome` and `why`. Return whether a retry is left; its pick then
        gets the whole attempt budget."""
        cfg = self.place_settings
        self._fallen = (outcome, why)
        if self.place.retries >= cfg.place_retries:
            return False
        self.grasp_settings = replace(cfg, max_attempts=self.status.attempt + cfg.max_attempts)
        return True

    def _fell_off(self, outcome: str, why: str) -> Action:
        """After the check found the block off its place: look for it to pick it again, or end."""
        if not self._fall(outcome, why):
            return self._finish(outcome, why)
        self._enter(RETRY)
        return self.hold(f"{why} - szukam bloku do ponowienia")

    def _retry(self, obs: Observation) -> Action:
        """Wait for fresh poses of the fallen block and of the target or each support, then pick it again.

        A block that leans, or poses that stay missing for `target_wait_s`, end the run without a retry."""
        assert self._fallen is not None
        outcome, why = self._fallen
        since = self._released_t + DROP_SETTLE_S
        mine = self._fresh(obs, self.label, since)
        if mine is not None and _lean_deg(mine.rotation, self.block) > LEVEL_TOL_DEG:
            return self._finish(outcome, f"{why} - bez ponowienia: blok pochylony")
        under = (self.target_label,) if self.target_label else self.rest_on
        missing = tuple(label for label in under if self._fresh(obs, label, since) is None)
        if mine is None or missing:
            if self._t - self._phase_t0 < self.place_settings.target_wait_s:
                return self.hold()
            what = (f"blok '{self.label}'" if mine is None
                    else f"{'cel' if self.target_label else 'podpora'} {_names(missing)}")
            return self._finish(outcome, f"{why} - bez ponowienia: {what} niewidoczny w kamerach")
        p = np.asarray(mine.position, float)
        rotation = np.asarray(mine.rotation, float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            rotation = np.eye(3)
        age = float(mine.age_s)
        return self._regrasp(Sighting(SEEN, (float(p[0]), float(p[1])), float(p[2]), rotation,
                                      str(mine.source), age, float(obs.t) - age))

    def _recover(self, obs: Observation) -> Action:
        """As grasp does, but a block that fell on the way with no retry left ends the run at once."""
        if self._fallen is not None and self.place.retries >= self.place_settings.place_retries:
            return self._finish(*self._fallen)
        return super()._recover(obs)

    def _regrasp(self, seen: Sighting) -> Action:
        """Pick the block again where `seen` puts it, as grasp's recovery does.

        After a fall from the place sequence this is a retry: the place starts over, the place must still
        fit, and a hand above the new hover first moves over it at its height, clear of the target."""
        if self._fallen is None:
            return super()._regrasp(seen)
        outcome, why = self._fallen
        obs, cfg, place = self._obs, self.place_settings, self.place
        assert obs is not None
        self._fallen = None
        self._placing = self._putting_back = False
        self._place_plan = None
        self._lifted_t, self._support_now = float("nan"), {}
        place.support_shift_m, place.support_moved = float("nan"), False
        self._held_gap_m = float("nan")
        self._goal = self._goal_from(obs)
        # An aim that fails ends the run with the fall's outcome.
        self._pending = outcome
        z = float(self.tcp_cmd[2])
        action = super()._regrasp(seen)
        if self.status.phase == DONE:
            return action
        problem = "" if self._fitted else self._place_problem(obs)
        if problem:
            return self._finish(outcome, f"{why} - ponowienie niemozliwe: {problem}")
        self._pending = ""
        self._find_supports(obs)
        place.retries += 1
        place.retried = (*place.retried, outcome)
        logger.info("%s - ponawiam (%d z %d)", why, place.retries, cfg.place_retries)
        self._retrying = outcome
        over = self._over_z(z) if self.status.phase == APPROACH else None
        if over is not None:
            self._transit_point = np.array([*self.hover_point[:2], over])
            self._move(TRANSIT, self._transit_point)
        return self.hold(f"{why} - ponawiam ({place.retries} z {cfg.place_retries})")

    def _over_z(self, z: float) -> float | None:
        """The TCP height over the new hover the hand moves to first from height `z` [m], or None when the
        way straight to the hover is clear.

        The jaw clears every block the cameras report by `retreat_clear_m` there. The highest of `z`, the
        middle and that clearance the arm reaches with the hand down wins; with none, None."""
        hover = self.hover_point
        world = world_state(self._scene, place_check=_any_place)
        tops = [o.top_m for o in world.objects.values() if o.label != self.label and not o.in_gripper]
        name = self.kin.spec.gripper
        assert name is not None
        grip = self.kin.from_q(self.q_cmd)[name]
        reach = max(self.kin.jaw_reach(grip), self.kin.jaw_reach(self.cmd_open))
        safe = max(tops, default=0.0) + self.place_settings.retreat_clear_m + reach
        if float(hover[2]) >= safe or z <= safe:
            return None
        seed = self.kin.from_q(self.q_cmd)
        for h in (z, (z + safe) / 2, safe):
            if self._reaches(np.array([*hover[:2], h]), self._psi, seed) is not None:
                return h
        return None

    def _transit(self, obs: Observation) -> Action | None:
        """Move the hand over the block at its height, then down to the hover as grasp's approach does."""
        self._ramp_jaw()
        point = self._transit_point
        if self._arrived(point, sag=True):
            self._enter(APPROACH)
            return None
        if self._timed_out():
            return self._halt(f"ramie nie przenioslo reki nad blok ({self.status.distance * 1000:.0f} mm od "
                              "celu)", self.kin.from_q(self.q_cmd))
        return self._drive(point, self.approach_step_m)

    # ------------------------------------------------------------------ the end
    def _finish(self, outcome: str, reason: str, success: bool = False) -> Action:
        self._pending = outcome
        if self.place.support_moved:
            reason += f"; blok pod podniesionym przesuniety o {self.place.support_shift_m * 1000:.0f} mm"
        return self._halt(reason, self.kin.from_q(self.q_cmd), success=success)

    def _halt(self, reason: str, target: Mapping[str, float], success: bool = False) -> Action:
        if self.place.retries:
            reason += f"; ponowienia: {self.place.retries} z {self.place_settings.place_retries}"
        return super()._halt(reason, target, success)

    def _end(self, reason: str, success: bool = False) -> None:
        super()._end(reason, success)
        if not self.place.outcome:
            self.place.outcome = self._pending or self._default_outcome()
            logger.info("Wynik odkladania: %s", self.place.outcome)
        self._pending = ""

    def _default_outcome(self) -> str:
        if self.grasp.verdict == "taken":
            return DROPPED
        if self._retrying:
            return self._retrying
        return STOPPED if self._placing else PICK_FAILED

    def stop(self, reason: str) -> None:
        super().stop(reason)
        if not self.place.outcome:
            self.place.outcome = STOPPED

    def _move(self, phase: str, goal: np.ndarray, turn: float = 0.0, via: np.ndarray | None = None) -> None:
        """Enter a carrying phase with a time limit of twice its travel, through `via` if given, its turn and
        `JAW_WAIT_S`."""
        path = (float(np.linalg.norm(goal - self.tcp_cmd)) if via is None else
                float(np.linalg.norm(via - self.tcp_cmd)) + float(np.linalg.norm(goal - via)))
        travel = path / self.config.speed_mps
        self._move_s = 2 * travel + abs(turn) / math.radians(TURN_DEG_S) + JAW_WAIT_S
        if phase == CARRY_OVER:
            self._over_t0 = self._t
        self._enter(phase)

    def _timed_out(self) -> bool:
        return self._t - self._phase_t0 >= self._move_s

    # ------------------------------------------------------------------ the panel
    def _note(self) -> str:
        text = self._phase_note()
        if self.place.retries and self.status.phase != DONE:
            text += f" | ponowienie {self.place.retries} z {self.place_settings.place_retries}"
        return text

    def _phase_note(self) -> str:
        st = self.status
        mm = st.distance * 1000
        if self._putting_back:
            where = "miejsce chwytu" if self.base_z > 0 else "stol"
        else:
            where = self._where()
        if st.phase == TARGET:
            if self.target_label:
                return f"czekam na polozenie celu '{self.target_label}'"
            return f"planuje odlozenie na {where}"
        if st.phase == CARRY_UP:
            return f"unosze blok do przeniesienia: {mm:.0f} mm"
        if st.phase == CARRY_OVER:
            return f"niose blok nad {where}: {mm:.0f} mm" + self._correction_note()
        if st.phase == LATCH:
            what = f"cel '{self.target_label}'" if self.target_label else f"podpory {_names(self.rest_on)}"
            return f"sprawdzam {what} w kamerach"
        if st.phase == PLACE:
            return f"opuszczam blok na {where}: {mm:.0f} mm" + self._correction_note()
        if st.phase == PARK:
            return f"odsuwam ramie od bloku: {mm:.0f} mm"
        if st.phase == VERIFY:
            return f"sprawdzam odlozenie w kamerach: {self._t - self._phase_t0:.1f} s"
        if st.phase == RETRY:
            return f"szukam bloku do ponowienia: {self._t - self._phase_t0:.1f} s"
        if st.phase == TRANSIT:
            return f"przenosze reke nad blok: {mm:.0f} mm"
        return super()._note()

    def _correction_note(self) -> str:
        """The place's correction for the panel, with why the table gave none; '' without a correction."""
        place = self.place
        mm = math.hypot(*place.correction_m) * 1000
        if place.correction_from == FROM_TABLE:
            return f"; poprawka z tablicy {mm:.0f} mm ({place.direction})"
        if place.correction_from == FROM_PICK:
            why = MISS_NOTES.get(place.table_miss, place.table_miss)
            return f"; poprawka z chwytu {mm:.0f} mm" + (f" - {why}" if why else "")
        return ""


def _any_place(x: float, y: float, z: float, spec: BlockSpec) -> bool:
    """A place check that allows every place: the jaw check reads only the objects' sizes."""
    return True


def _names(labels: tuple[str, ...]) -> str:
    return " i ".join(f"'{label}'" for label in labels)


def _lean_deg(rotation: Any, spec: BlockSpec) -> float:
    """How far a block at `rotation` leans from lying flat [deg]; NaN when the rotation is invalid.

    A cylinder lies flat with its long body x axis level. A box does with its body z axis, or an axis as
    long as that one, vertical."""
    R = np.asarray(rotation, float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return float("nan")
    if any(part.kind == "cylinder" for part in spec.parts):
        return math.degrees(math.asin(min(1.0, abs(float(R[2, 0])))))
    up = max(abs(float(R[2, k])) for k in range(3) if abs(spec.half_size[k] - spec.half_size[2]) < 1e-6)
    return math.degrees(math.acos(min(1.0, up)))

