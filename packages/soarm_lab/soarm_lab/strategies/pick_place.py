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

After the release the open hand must leave without touching the placed block. Where the arm reaches,
it rises until the jaw clears the block's top. Otherwise the jaw opens fully, so that its moving side
clears the block, and the way to the start position must lead away from the fixed jaw. Headings with
the first exit win.

Over a block target the run latches the newest fresh camera pose of the target seen since the carry
began, if it was captured at most `fresh_pose_s` before the hand started over the target. The hand
may hide the target at the hover, so the pose may come from the way there. A target that moved
re-plans the hover, at most `MAX_REPLANS` times. The descent settles `SLOW_ABOVE_M` over the release
height, or at the carry height when that is lower, then runs at `place_speed_mps` and stops on a
stall or at the release height. The jaw opens `release_margin_m` past the held block on each side,
eases off toward the fixed jaw as `grasp_jev` does, rises and moves to the start position. Fresh
poses captured after the release then verify the result.

A target not seen at the hover, an unreachable place or a target that keeps moving puts the block
back where it was picked, without verification. The carry back stays above the target the hand
hovered over. `PlaceStatus.outcome` names the result with one of `OUTCOMES`.

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

from ..blocks import BlockSpec, block
from ..blocks.classic_world import BLOCKS
from ..planner.world import WorldObject, overlap, world_state
from .grasp import (
    CARRY,
    DROP_SETTLE_S,
    IN_HAND_M,
    JAW_WAIT_S,
    MAX_ROT_ERR_DEG,
    ON_BLOCK_M,
    GraspCandidate,
    JawPlan,
    hand_down,
    heading,
    ik_restarts,
    wrap,
)
from .grasp_jev import JEV_PHASE_NAMES, LET_GO, GraspJevSettings, GraspJevStrategy
from .jaw import JawReading
from .judges import Judge, JudgeError, Questions, Reply
from .touch import DONE, SETTLE_S, TURN_DEG_S, WAIT, TouchStrategy, normalised
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
#: The phases' names in the panel's notes.
PLACE_PHASE_NAMES = {**JEV_PHASE_NAMES, TARGET: "planuje odlozenie", CARRY_UP: "unosze blok do przeniesienia",
                     CARRY_OVER: "niose blok nad cel", LATCH: "sprawdzam cel w kamerach",
                     PLACE: "opuszczam blok", PARK: "odsuwam ramie od bloku", VERIFY: "sprawdzam odlozenie"}

#: Hover re-plans after the latched target moved, before the block goes back.
MAX_REPLANS = 2
#: The least clearance of the held block's bottom above the target while carrying [m].
MIN_CARRY_CLEAR_M = 0.01
#: Height step for lower carry and retreat heights when the wanted one is out of reach [m].
HEIGHT_STEP_M = 0.005
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
#: The open jaw's lowest point clears the placed block's top by this much before the hand leaves [m].
JAW_CLEAR_M = 0.004
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
    #: How far the open hand rises after the release, if the arm reaches [m].
    retreat_m: float = 0.05
    #: The latch takes a target pose captured at most this long before the hand started over the target,
    #: the verification one at most this old [s].
    fresh_pose_s: float = 1.0
    #: How long to wait for a target pose after the lift and at the hover [s].
    target_wait_s: float = 5.0
    #: How long to wait for fresh poses after the release [s].
    verify_wait_s: float = 3.0
    #: A placed block's centre may lie this far from the target's centre [m].
    place_tol_m: float = 0.008
    #: As in `grasp`, but on by default: the hand tilts where the pick or the place needs it [deg].
    max_tilt_deg: float = 30.0


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


@dataclass(frozen=True)
class PlacePlan:
    """A planned place: TCP targets and the target they came from."""

    #: Where the held block's centre goes [m, base frame], the surface's top [m] and the target yaw [rad].
    target_xy: np.ndarray
    top_m: float
    yaw: float | None
    #: Closing heading at the release [rad, base frame], its turn from the carry's heading [rad] and the
    #: TCP's x and y there [m].
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

    @property
    def hover(self) -> np.ndarray:
        return np.array([*self.tcp_xy, self.carry_z])

    @property
    def release(self) -> np.ndarray:
        return np.array([*self.tcp_xy, self.release_z])


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
        for name in ("carry_clearance_m", "place_speed_mps", "release_margin_m", "retreat_m", "fresh_pose_s",
                     "place_tol_m"):
            value = getattr(config, name)
            if not value > 0:
                raise ValueError(f"pick_place: {name} must be positive, is {value}")
        for name in ("place_gap_m", "target_wait_s", "verify_wait_s"):
            value = getattr(config, name)
            if not value >= 0:
                raise ValueError(f"pick_place: {name} must be zero or more, is {value}")
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
        self.place = PlaceStatus(target=self.target_label, rests_on=self.rest_on)
        self._reset_place()
        #: The jaw bodies' points in the TCP frame by gripper command.
        self._jaw_clouds: dict[float, tuple[np.ndarray, ...]] = {}
        self._phases.update({CARRY: self._lifted, TARGET: self._target, CARRY_UP: self._carry_up,
                             CARRY_OVER: self._carry_over, LATCH: self._latch, PLACE: self._place,
                             PARK: self._park, VERIFY: self._verify})

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
        #: When the open hand finished rising [s, `obs.t`].
        self._released_t = float("nan")
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

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Start the grasp family's checks, then clear the place state."""
        super().start(run, obs)
        self.place = PlaceStatus(target=self.target_label, rests_on=self.rest_on)
        self._reset_place()
        self._start_tcp = self.tcp_cmd.copy()

    def act(self, obs: Observation) -> Action:
        self._scene, self._others = obs.objects, None
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
        """Monitor the hold as the grasp family does. A drop ends the place sequence until a regrasp."""
        if self.judged:
            action = super().carrying(obs, reading)
        else:
            action = super(GraspJevStrategy, self).carrying(obs, reading)
        if action is not None and self.grasp.verdict == "dropped":
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
        from. `hang` and `hand_xy` place the held block in the hand; the measured ones by default.
        """
        cfg = self.place_settings
        hang = self._hang if hang is None else hang
        hand_xy = self._hand_xy if hand_xy is None else hand_xy
        from_xy = self.tcp_cmd[:2].copy() if from_xy is None else np.asarray(from_xy, float).copy()
        floor = max(top, TABLE_TOP_M, over_m, self._bottom())
        release_z = top + cfg.place_gap_m + GRIP_SLIP_M + hang
        low = floor + MIN_CARRY_CLEAR_M + hang
        want = max(floor + cfg.carry_clearance_m + hang, low)
        if over_m > 0:
            # Never lower the held block over what the hand hovers above.
            want = max(want, float(self.tcp_cmd[2]))
        # TCP heights where the jaw at the release opening, or the fully open moving jaw, clears the
        # placed block.
        placed_top = top + cfg.place_gap_m + self.height_m
        clear_z = placed_top + JAW_CLEAR_M + self.kin.jaw_reach(self._place_open_cmd, tilt=self._tilt)
        wide_z = placed_top + JAW_CLEAR_M + self.kin.jaw_reach(FULL_OPEN, moving=True, tilt=self._tilt)
        seed0 = self.kin.from_q(self.q_cmd)
        problem, fallback = "", None
        turns = self._turns(yaw, np.asarray(xy, float), hand_xy)
        if not turns:
            return "dlon pochylona nie ustawi bloku zgodnie z celem"
        for turn in turns:
            psi = wrap(self._psi + turn)
            u = np.array([math.cos(psi), math.sin(psi)])
            tcp_xy = xy - rot2(psi) @ hand_xy
            eased = tcp_xy - cfg.release_margin_m * u
            away = self._start_tcp[:2] - eased
            leaves = float(away @ u) <= -AWAY_COS * float(np.linalg.norm(away)) - 1e-9
            where = f"({tcp_xy[0]:.3f}, {tcp_xy[1]:.3f}, {release_z:.3f}) m"
            hit = self._jaw_hit(psi, (np.array([*tcp_xy, release_z]), np.array([*eased, release_z])),
                                self._place_open_cmd)
            if hit:
                problem = problem or f"szczeka zahaczylaby o '{hit}' przy odkladaniu"
                continue
            at_release = self._reaches(np.array([*tcp_xy, release_z]), psi, seed0)
            if at_release is None:
                problem = problem or f"punkt odlozenia {where} poza zasiegiem z chwytakiem w dol"
                continue
            # Once a heading that opens fully is known, only one that rises clear is worth more.
            retreat = self._retreat(eased, psi, release_z, clear_z, wide_z if leaves and fallback is None
                                    else math.inf, at_release)
            if retreat is None:
                problem = problem or (f"po odlozeniu szczeki nie mina bloku: wysokosc {clear_z:.3f} m poza "
                                      "zasiegiem")
                continue
            retreat_z, wide = retreat
            hit = self._jaw_hit(psi, (np.array([*eased, retreat_z]),), FULL_OPEN) if wide else ""
            if hit:
                problem = problem or f"otwarta szczeka zahaczylaby o '{hit}' przy odjezdzie"
                continue
            carry_z = None
            for z in _heights(want, low):
                up = self._reaches(np.array([*from_xy, z]), self._psi, seed0)
                if up is not None and self._reaches(np.array([*tcp_xy, z]), psi, up) is not None:
                    carry_z = z
                    break
            if carry_z is None:
                problem = problem or f"przeniesienie nad cel na wysokosci {low:.3f} m poza zasiegiem"
                continue
            plan = PlacePlan(np.asarray(xy, float).copy(), float(top), yaw, psi, turn, tcp_xy, from_xy,
                             float(carry_z), float(release_z), float(retreat_z), wide)
            if not wide:
                return plan
            fallback = plan
        return fallback if fallback is not None else problem

    def _retreat(self, eased: np.ndarray, psi: float, release_z: float, clear_z: float, wide_z: float,
                 seed: Mapping[str, float]) -> tuple[float, bool] | None:
        """Return the open hand's retreat height [m] and whether the jaw must then open fully, or None.

        The hand rises `retreat_m`, or less when out of reach, but at least to `clear_z`. Below that,
        a retreat at or above `wide_z` lets the fully open jaw clear the block; infinite rules it out.
        """
        low = max(clear_z, release_z)
        for z in (max(release_z + self.place_settings.retreat_m, low), low):
            if self._reaches(np.array([*eased, z]), psi, seed) is not None:
                return z, False
        for z in _heights(low, max(wide_z, release_z))[1:]:
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
        self._target_seen, self._rest_seen = None, {}
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
            self._set_heading(plan.psi, plan.tcp_xy)
            self._move(CARRY_OVER, plan.hover, plan.turn)
            return None
        if self._timed_out():
            mm = self.status.distance * 1000
            return self._halt(f"ramie nie unioslo bloku na wysokosc przeniesienia ({mm:.0f} mm od celu)",
                              self.kin.from_q(self.q_cmd))
        return self._drive(point, self.approach_step_m)

    def _carry_over(self, obs: Observation) -> Action | None:
        """Carry the block over the target and turn it to the planned heading."""
        action = self.carrying(obs, self.reading)
        if action is not None:
            return action
        self._note_target(obs)
        plan = self._place_plan
        assert plan is not None
        if self._settled(plan.hover, sag=True) and self._vertical():
            # The descent's stall reference.
            self._take_reference()
            self._slow = False
            latch = (self.target_label or self.rest_on) and not self._putting_back
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
            # Small corrections ride along the descent, which starts no higher than the hand is.
            self._take_plan_place(replace(new, from_xy=plan.from_xy, turn=plan.turn,
                                          carry_z=min(new.carry_z, plan.carry_z)))
            self._set_heading(new.psi, new.tcp_xy)
            self._enter(PLACE)
            return None
        if self._replans >= MAX_REPLANS:
            return self._put_back(TARGET_MOVED, f"{why} ({moved * 1000:.0f} mm)")
        self._replans = self.place.replans = self._replans + 1
        # Keep the carry's start: the arm is already at carry height.
        self._take_plan_place(replace(new, from_xy=plan.from_xy))
        self._set_heading(new.psi, new.tcp_xy)
        logger.info("Cel przesuniety o %.0f mm - nowy zawis", moved * 1000)
        self._move(CARRY_OVER, new.hover, new.turn)
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
            above = np.array([*plan.tcp_xy, above_z])
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
        self._released = False
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
        return self.place_settings.release_margin_m if self._placing else super().ease_off_m

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
        """While placing, leave the block for the cameras.

        The hand leaves straight when the jaw clears the block's top. Otherwise the jaw opens fully
        first, and the hand must leave away from the fixed jaw with the moving jaw over the block.
        """
        if not self._placing:
            return super()._after_let_go()
        plan = self._place_plan
        assert plan is not None
        self._released_t = self._t
        z = float(self._release_point[2])
        park_z = max(float(self._start_tcp[2]), float(self.tcp_cmd[2]))
        self._park_point = np.array([*self._start_tcp[:2], park_z])
        distance = float(np.linalg.norm(self._park_point - self.tcp_cmd))
        self._park_s = distance / self.config.speed_mps + JAW_WAIT_S
        # The placed block's top, higher when the descent stalled above the release height.
        top = (plan.top_m + self.place_settings.place_gap_m + self.height_m
               + max(self._release_tcp_z - plan.release_z, 0.0) + JAW_CLEAR_M)
        self._opened = z - self.kin.jaw_reach(self._place_open_cmd, tilt=self._tilt) >= top - 1e-6
        if not self._opened:
            u = np.array([math.cos(self._psi), math.sin(self._psi)])
            away = self._park_point[:2] - self.tcp_cmd[:2]
            leaves = float(away @ u) <= -AWAY_COS * float(np.linalg.norm(away)) + 1e-9
            if not leaves or z - self.kin.jaw_reach(FULL_OPEN, moving=True, tilt=self._tilt) < top - 1e-6:
                return self._finish(STOPPED, "otwarta reka nie odjedzie od odlozonego bloku bez "
                                             "potracenia go")
            self.place.opened_wide = True
            self._park_s += JAW_WAIT_S
        self._enter(PARK)
        return None

    def _park(self, obs: Observation) -> Action | None:
        """Open the jaw fully if leaving needs it, then move the open hand toward the start position.

        A point out of reach ends the move early.
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
        return TouchStrategy._drive(self, self._park_point, self.approach_step_m)

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
        if self.target_label:
            target = self._fresh(obs, self.target_label, since)
            if target is not None:
                shift = float(np.linalg.norm(np.asarray(target.position, float)[:2] - plan.target_xy))
                place.target_shift_m = shift
                if shift > cfg.place_tol_m:
                    return self._finish(TARGET_MOVED, f"cel '{self.target_label}' przesuniety o "
                                                      f"{shift * 1000:.0f} mm przy odkladaniu")
        shifts = {label: float(np.linalg.norm(np.asarray(pose.position, float) - self._rest_pos[label]))
                  for label, pose in rests.items() if pose is not None}
        if shifts:
            moved = max(shifts, key=shifts.__getitem__)
            place.target_shift_m = shift = shifts[moved]
            if shift > cfg.place_tol_m:
                return self._finish(TARGET_MOVED, f"podpora '{moved}' przesunieta o {shift * 1000:.0f} mm "
                                                  "przy odkladaniu")
        p = np.asarray(mine.position, float)
        error = float(np.linalg.norm(p[:2] - plan.target_xy))
        dz = float(p[2]) - (plan.top_m + self.height_m / 2)
        place.placed_xyz = (float(p[0]), float(p[1]), float(p[2]))
        place.error_m, place.height_error_m = error, dz
        place.placed_tilt_deg = lean = _lean_deg(mine.rotation, self.block)
        # A target block's place is not checked for level.
        level = bool(self.target_label) or not lean > LEVEL_TOL_DEG
        where = self._where()
        offsets = f"{error * 1000:.0f} mm od srodka, {dz * 1000:+.0f} mm w pionie"
        if not level:
            offsets += f", pochylony o {lean:.0f} st."
        if error <= cfg.place_tol_m and abs(dz) <= PLACED_Z_TOL_M and level:
            tilted = f" (dlon pochylona {self.grasp.tilt_deg:.0f} st.)" if self._tilt else ""
            return self._finish(PLACED, f"blok '{self.label}' odlozony na {where}: {offsets}{tilted}",
                                success=True)
        return self._finish(FELL_OFF, f"blok '{self.label}' nie lezy na {where}: {offsets}")

    # ------------------------------------------------------------------ the end
    def _finish(self, outcome: str, reason: str, success: bool = False) -> Action:
        self._pending = outcome
        if self.place.support_moved:
            reason += f"; blok pod podniesionym przesuniety o {self.place.support_shift_m * 1000:.0f} mm"
        return self._halt(reason, self.kin.from_q(self.q_cmd), success=success)

    def _end(self, reason: str, success: bool = False) -> None:
        super()._end(reason, success)
        if not self.place.outcome:
            self.place.outcome = self._pending or self._default_outcome()
            logger.info("Wynik odkladania: %s", self.place.outcome)
        self._pending = ""

    def _default_outcome(self) -> str:
        if self.grasp.verdict == "taken":
            return DROPPED
        return STOPPED if self._placing else PICK_FAILED

    def stop(self, reason: str) -> None:
        super().stop(reason)
        if not self.place.outcome:
            self.place.outcome = STOPPED

    def _move(self, phase: str, goal: np.ndarray, turn: float = 0.0) -> None:
        """Enter a carrying phase with a time limit of twice its travel, its turn and `JAW_WAIT_S`."""
        travel = float(np.linalg.norm(goal - self.tcp_cmd)) / self.config.speed_mps
        self._move_s = 2 * travel + abs(turn) / math.radians(TURN_DEG_S) + JAW_WAIT_S
        if phase == CARRY_OVER:
            self._over_t0 = self._t
        self._enter(phase)

    def _timed_out(self) -> bool:
        return self._t - self._phase_t0 >= self._move_s

    # ------------------------------------------------------------------ the panel
    def _note(self) -> str:
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
            return f"niose blok nad {where}: {mm:.0f} mm"
        if st.phase == LATCH:
            what = f"cel '{self.target_label}'" if self.target_label else f"podpory {_names(self.rest_on)}"
            return f"sprawdzam {what} w kamerach"
        if st.phase == PLACE:
            return f"opuszczam blok na {where}: {mm:.0f} mm"
        if st.phase == PARK:
            return f"odsuwam ramie od bloku: {mm:.0f} mm"
        if st.phase == VERIFY:
            return f"sprawdzam odlozenie w kamerach: {self._t - self._phase_t0:.1f} s"
        return super()._note()


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


def _heights(high: float, low: float) -> list[float]:
    """Return heights from `high` down to `low` in `HEIGHT_STEP_M` steps [m]."""
    out = [high]
    while out[-1] - HEIGHT_STEP_M > low + 1e-9:
        out.append(out[-1] - HEIGHT_STEP_M)
    if low < high - 1e-9:
        out.append(low)
    return out
