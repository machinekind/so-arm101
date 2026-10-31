"""Jev in the grasp loop: `grasp_jev` under the `machinekind_twin.strategies` entry points.

`GraspJevStrategy` is `grasp` with a decision step at three points of every attempt, as `touch_jev` is
`touch` with one after each descent. It asks Jev, TypeSafe's decision model (`jev.JevJudge`), through
`judges.Consultation`; grasp's motion, guards, recovery and stop reasons stay as they are, and `grasp`
itself never asks. Nothing here imports the twin.

    # configs/local.yaml of the twin: prefills the panel's form
    grasp_jev:
      object_kind: cube              # cube, cuboid5, cuboid85 or cylinder [rl.objects]; grasp's fields first
      model: jev-1.14.0              # pinned: the thresholds belong to one version
      min_confidence: 0.8            # a less certain verdict stops the arm for a person [0..1]
      min_correction_confidence: 0.6 # a less certain correction or next step does too [0..1]
      correction_m: 0.02             # one step of a correction after a missed probe [m]
      answer_timeout_s: 5.0          # the longest wait for one answer, also the HTTP timeout [s]
      send_frames: false             # also send the wrist camera's frame as a PNG (not production-ready)
      wrist_camera: wrist            # the wrist camera's name in obs.frames (robot.yaml `cameras:`)
      time_limit_s: 180.0            # the whole run, model waits included [s]

The key comes from `TYPESAFE_API_KEY` in the twin's environment (`uv run --env-file .env twin ui`); `start`
refuses the run without it, before the arm moves.

The points. Jev is asked once per point and attempt, with a state of facts computed in code and named
buckets beside the millimetres, because Jev is weak with numbers:

1. the probe, on the "touched" phase as in `touch_jev`: its `touch_questions` (`touched`, `correction`) on
   its `descent_state`, whose touch point is the BLOCK centre (`grasp` aims the jaw point, so its
   `touch_xy` is the probe's TCP point, 15.6 mm toward the moving jaw);
2. the close: a held close on the first lift_off tick (through `carrying`), an empty close or a jaw tip on
   the block in the straddle (through `after_grasp_failed`, the arm holding in a phase of its own, judge);
3. the lift, on the first check tick (through `carrying`).

At the close and the lift the questions are `grasp_questions`: `held` (nothing_in_jaws, block_held,
held_by_edge) and `next` (lift, regrasp_wider, regrasp_rotated, give_up), on `grasp_state`: how the close
ended, the measured jaw gap against the block's short side (closed, narrower_than_block, block_wide,
open), whether the jaw stands and whether it moved since the close, how far the arm lifted (not_lifted,
partly, lifted), whether the wrist camera shows the block between the jaws (`blob`, "unknown" without
one), the twin's latest sighting of the block against the jaws with its source (annotated sources named
apart: last_seen_dropped, last_seen, none), whether the jaw heading came from the block's rotation or was
a guess, the earlier attempts and `available_next`: the steps the geometry allows, computed in code.

The rules, in order. At the probe they are touch_jev's, except that a confident "touched" goes on into
`grasp`'s rise, open and straddle instead of ending the run, and a "touched" on a probe that met nothing,
or that stalled more than `contact_band_m` above the probe height, ends for review: a straddle needs the
stall's top. At the close and the lift:

* a `held` answer less confident than `min_confidence` ends the run for review, the arm holding;
* a `next` answer less confident than `min_correction_confidence` does too;
* a `next` the geometry forbids (not in `available_next`) ends for review, or without review when it is a
  regrasp on the last of `max_attempts`;
* lift goes on: the lift's time limit and the check's `hold_s` count from the answer, not from the ask;
* give_up ends the run with the block where it is: a held block stays held;
* regrasp_wider opens the jaw one `open_margin_m` wider on each side; regrasp_rotated turns the jaws by 90
  degrees at the same block centre (only for a guessed heading on a block whose grasp a turn changes).
  After an empty close or a tip on the block the arm rises (`grasp`'s sideways retry, half the short
  side, for regrasp_wider); with the block held on the table the jaw first opens there (phase let_go);
  with the block in the air the arm first lowers it to the grasp point (phase lower: a stall on the way
  down counts as arrival, so nothing presses into the table). The jaw opens before the arm rises and
  closes only at the hover, then the run aims at the block centre between the fingertips and probes again.

The held answer never moves the arm by itself: it is a record, and a disagreement with the jaw rule is
counted and said in a review or a stop reason ("szczeka: trzyma", "szczeka: pusta").

A block that drops out of the jaws goes to `grasp`'s recovery (its `carrying` watch, also on the way down
in lower): a close or lift question in flight or in its retry pause is cancelled and recorded as such,
the arm rises, waits for a sighting and grasps the block again, whose probe Jev judges as any other. The
plan's ambiguous row, a held block that the cameras see elsewhere, reaches Jev through `block_sighting`.

The call. `judges.Consultation` asks on a thread of its own and the tick polls it, holding the arm with
touch's lag guard on; the 20 Hz tick never waits. A call that does not answer within `answer_timeout_s`
(counted on `obs.t` from the ask) is given up; a timeout, an overload or a dropped connection is asked
again once after 0.5 s with the same state; then, and after any other failure, the run ends for review.
Model waits count toward `time_limit_s`, 180 s by default.

The records. Every call gives one `judges.Decision` per question with the exact state and questions sent,
the frames' references, the confidence and the response time; failed and cancelled calls too. The state
carries `point` (probe, close, lift) and the questions' version (`touch_jev/1` at the probe, `grasp_jev/1`
otherwise). They collect in `judging.decisions` and `drain_decisions`, one "Ocena chwytu" log line per
call, and a summary in the notes.

Frames. With `send_frames` on, only the wrist camera's frame of the asking tick goes out, by its name
`wrist_camera`: halved (or more) by integer striding when a side exceeds Jev's 1568 px (the MJCF's camera
renders 1920x1080), left out with its reason below 280 px, when missing, or when older than
`FRAME_MAX_AGE_S` (a stalled stream keeps serving its last image with a growing age, and never stands for
the current view). The state then always carries `camera_views`, `frames_left_out`, `frames_scaled` and
`frames_age_s`.

The blob seam. `blob=` takes a callable that gives the share of the wrist frame between the jaws filled
by the block's colour (0..1), on the tick thread; the state says "unknown" until one is wired in.

What this module uses of `grasp` (its seams and private names): `after_touch`, `after_grasp_failed`,
`carrying` (the watch), `_aim` with `_headings`, `_psi`, `_rot`, `_block_xy0`, `_shift_k` and
`_next_shift`, `_start_rise` with `_retarget_xy` and RETARGET, `_regrasp`, `_sighting`, `_jaw_point`,
`_set_jaw_goal`, `_ramp_jaw`, `_jaw_ready`, `_centre_open`, `_recover_open_q`, `_q`, `_g`, `reading`,
`gap_open_m`, `cmd_open`, `cmd_closed`, `block`, `block_rotation`, the `grasp` status, and touch's
`_phase_t0` as the only phase clock of lift_off and check.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Frame, Observation, RunInfo

from .grasp import (
    CARRY,
    CHECK,
    JAW_WAIT_S,
    LIFT_OFF,
    OPEN,
    RETARGET,
    RISE,
    STRADDLE,
    GraspSettings,
    GraspStrategy,
    wrap,
)
from .grasp import Sighting as WatchSighting
from .jaw import JawReading
from .jev import DEFAULT_MODEL, IMAGE_SIDE_PX, JevJudge, frame_problem
from .judges import Consultation, Decision, Judge, JudgeError, JudgeStatus, Reply
from .touch import APPROACH, DESCEND, DONE, LIFT, normalised
from .touch_jev import (
    CORRECTION,
    CORRECTIONS_PL,
    FIRST_SIGHTING,
    GEOMETRY_AS_OPTION,
    GEOMETRY_PL,
    SAME_PLACE_M,
    SOURCES,
    TOUCHED,
    VERDICTS_PL,
    DescentFacts,
    Sighting,
    correction_target,
    descent_state,
    millimetres,
    seconds_ago,
    stop_height,
    touch_questions,
)
from .touch_jev import QUESTIONS_VERSION as PROBE_QUESTIONS_VERSION

logger = logging.getLogger(__name__)

#: The version of the grasp state and questions, recorded with every decision at the close and the lift.
QUESTIONS_VERSION = "grasp_jev/1"
#: The share of the wrist frame between the jaws the block's colour must fill to count as seen there [0..1].
BLOB_FILL_MIN = 0.15
#: A jaw gap within this of the block's short side stands on the block [m].
EDGE_M = 0.003
#: A sighting this close to the aimed block centre lies near the jaws [m]; at `SAME_PLACE_M` at them.
NEAR_M = 0.03
#: A wrist frame older than this is left out [s], as the twin's detectors drop frames (`MAX_FRAME_AGE_S`).
FRAME_MAX_AGE_S = 1.0

#: Phases of a run after grasp's (`TouchStatus.phase`): waiting for the answer after an empty close or a tip
#: on the block; lowering a held block to the table before a regrasp; opening the jaw there and rising.
JUDGE = "judge"
LOWER = "lower"
LET_GO = "let_go"

#: Where in the run a question is asked (`state.point` of the records).
POINT_PROBE = "probe"
POINT_CLOSE = "close"
POINT_LIFT = "lift"

HELD = "held"
NEXT = "next"
HELD_OPTIONS = ("nothing_in_jaws", "block_held", "held_by_edge")
NEXT_OPTIONS = ("lift", "regrasp_wider", "regrasp_rotated", "give_up")
#: How the close ended, as the state names it, by grasp's verdict.
ENDED = {"held": "jaw_stopped_on_something", "empty": "jaw_closed_on_nothing",
         "tip": "jaw_tip_landed_on_block_before_closing"}
#: For notes and stop reasons, in Polish without diacritics.
HELD_PL = {"nothing_in_jaws": "szczeka pusta", "block_held": "blok trzymany",
           "held_by_edge": "trzymany za krawedz"}
NEXT_PL = {"lift": "podniesc", "regrasp_wider": "chwyt szerzej", "regrasp_rotated": "chwyt obrocony",
           "give_up": "rezygnacja"}
POINT_PL = {POINT_CLOSE: "po zamknieciu", POINT_LIFT: "po podniesieniu"}

#: The phases whose notes end with the last answer's summary.
SUMMARY_PHASES = (LIFT, APPROACH, DESCEND, RISE, OPEN, STRADDLE, LIFT_OFF, CHECK, CARRY)
#: The frames of one question: the cameras sent (name -> reference), left out (name -> why), scaled (name ->
#: "WxH -> wxh") and their ages [s]; and the images by reference.
Frames = tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, float], dict[str, np.ndarray]]


@dataclass
class GraspJevSettings(GraspSettings):
    #: A safety net for the whole run, the model's waits and retries included [s]; 0 means none.
    time_limit_s: float = 180.0
    #: The Jev model, a version rather than an alias: the thresholds belong to one version.
    model: str = DEFAULT_MODEL
    #: The confidence a verdict (touched, held) needs to be acted on without a person [0..1].
    min_confidence: float = 0.8
    #: The confidence a correction after a missed probe or a next step needs [0..1].
    min_correction_confidence: float = 0.6
    #: One step of a correction that moves the block centre aimed at after a missed probe [m]; more than
    #: `SAME_PLACE_M`, at most 0.1.
    correction_m: float = 0.02
    #: The longest wait for one answer, counted from the question; also the HTTP timeout [s].
    answer_timeout_s: float = 5.0
    #: Also send the wrist camera's frame of the asking tick as an image (Jev's images are not
    #: production-ready).
    send_frames: bool = False
    #: The wrist camera's name in `obs.frames` (robot.yaml `cameras:`), matched exactly after stripping
    #: white space.
    wrist_camera: str = "wrist"


@dataclass(frozen=True)
class BlockSighting:
    """The block's pose in one observation, any source: what the state says about the twin's view."""

    source: str
    #: When its frames were taken [s, `obs.t`].
    seen_t: float
    #: Its x and y and its height [m, base frame].
    xy: tuple[float, float]
    z: float


@dataclass(frozen=True)
class GraspFacts:
    """What the state of one close or lift question is built from, read on the tick thread."""

    label: str
    kind: str
    #: The block's sides [m].
    short_m: float
    long_m: float
    height_m: float
    attempt: int
    max_attempts: int
    point: str
    #: How the close ended (`ENDED`).
    ended: str
    #: The measured and the commanded gap between the fingertips, and the jaw rule's margin as a gap [m].
    jaw_gap_m: float
    commanded_gap_m: float
    margin_gap_m: float
    jaw_still: bool
    jaw_moved_after_close: bool
    #: How far the TCP rose since the close (NaN before the lift) and how far the lift goes [m].
    lifted_m: float
    lift_m: float
    #: The straddle's opening [m].
    gap_open_m: float
    #: The share of the wrist frame the block's colour fills between the jaws, None without a blob.
    blob_share: float | None
    sighting: BlockSighting | None
    #: The block centre the run aims at [m, base frame] and the jaws' closing heading [rad, base frame].
    aim_xy: tuple[float, float]
    heading_rad: float
    #: "block_rotation" when the heading came from the pose's rotation, "guess" when it had none.
    heading_from: str
    yaw_period_rad: float
    #: A jaw one more margin wider on each side fits within the jaw's travel.
    wider_possible: bool
    now_t: float
    earlier: tuple[Mapping[str, Any], ...] = ()
    with_frames: bool = False
    camera_views: Mapping[str, str] = field(default_factory=dict)
    frames_left_out: Mapping[str, str] = field(default_factory=dict)
    frames_scaled: Mapping[str, str] = field(default_factory=dict)
    frames_age_s: Mapping[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ pure state
def fit_frame(image: np.ndarray) -> tuple[np.ndarray, int]:
    """`image` strided by the smallest integer `k` that brings its longer side within Jev's 1568 px, and `k`:
    numpy only, no resampling (1920x1080 gives 960x540 with k = 2)."""
    height, width = image.shape[:2]
    k = max(1, math.ceil(max(height, width) / IMAGE_SIDE_PX[1]))
    return np.ascontiguousarray(image[::k, ::k]), k


def source_name(source: str) -> str:
    """A twin pose source as the state names it; the annotated last-seen poses by their prefix."""
    if source in SOURCES:
        return SOURCES[source]
    if source.startswith("ostatnie widziane (upuszczona)"):
        return "last_seen_dropped"
    if source.startswith("ostatnie widziane ("):
        return "last_seen"
    if source.startswith("brak ("):
        return "none"
    return "other"


def jaw_gap_bucket(gap_m: float, short_m: float, commanded_m: float, margin_gap_m: float) -> str:
    """The measured jaw gap against the block's short side, in named buckets."""
    if gap_m <= commanded_m + margin_gap_m:
        return "closed"
    if gap_m < short_m - EDGE_M:
        return "narrower_than_block"
    if gap_m <= short_m + EDGE_M:
        return "block_wide"
    return "open"


def lifted_bucket(lifted_m: float, lift_m: float) -> str:
    if not lifted_m >= 0.005:
        return "not_lifted"
    return "partly" if lifted_m < lift_m / 2 else "lifted"


def blob_bucket(share: float | None) -> str:
    if share is None:
        return "unknown"
    return "yes" if share >= BLOB_FILL_MIN else "no"


def sighting_state(sighting: BlockSighting | None, aim_xy: tuple[float, float], heading_rad: float,
                   now_t: float) -> dict[str, Any] | None:
    """Where the twin's latest pose places the block against the aimed centre, along and across the jaws."""
    if sighting is None:
        return None
    offset = np.subtract(sighting.xy, aim_xy)
    distance = float(np.linalg.norm(offset))
    where = ("at_the_jaws" if distance <= SAME_PLACE_M else "near_the_jaws" if distance <= NEAR_M
             else "elsewhere")
    along = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    across = np.array([-along[1], along[0]])
    return {"source": source_name(sighting.source), "seen_s_ago": seconds_ago(sighting.seen_t, now_t),
            "where": where, "offset_mm": {"along_jaws": millimetres(float(offset @ along)),
                                          "across_jaws": millimetres(float(offset @ across))},
            "height_mm": millimetres(sighting.z)}


def next_problem(option: str, facts: GraspFacts) -> str:
    """Why the geometry forbids a next step (Polish, for the panel); empty when it allows it."""
    if option == "lift":
        return "" if facts.ended == ENDED["held"] else "blok nie jest trzymany"
    if option in ("regrasp_wider", "regrasp_rotated") and facts.attempt >= facts.max_attempts:
        return "wyczerpane proby chwytu"
    if option == "regrasp_wider" and not facts.wider_possible:
        return "szczeka nie otworzy sie szerzej"
    if option == "regrasp_rotated":
        if facts.heading_from == "block_rotation":
            return "kierunek szczek znany z polozenia bloku"
        if facts.yaw_period_rad < math.pi - 1e-6:
            return "obrot nic nie zmienia dla tego bloku"
    return ""


def available_next(facts: GraspFacts) -> list[str]:
    """The next steps the geometry allows, computed in code: Jev is weak with numbers."""
    return [option for option in NEXT_OPTIONS if not next_problem(option, facts)]


def grasp_state(facts: GraspFacts) -> dict[str, Any]:
    """The state Jev judges a close or a lift from: JSON values, finite numbers or null, the same keys every
    time (the four frame keys only with frames on)."""
    sizes = f"{millimetres(facts.long_m)} x {millimetres(facts.short_m)} x {millimetres(facts.height_m)} mm"
    task = (f"A robot arm probed the top of the block '{facts.label}' ({facts.kind}, {sizes}) with its "
            f"closed jaw tips, rose, opened the jaw to {millimetres(facts.gap_open_m)} mm")
    if facts.ended == ENDED["tip"]:
        task += " and lowered it around the block; a jaw tip landed on the block before the close."
    else:
        task += f", lowered it around the block and closed it to {millimetres(facts.commanded_gap_m)} mm."
    lift = None
    if facts.point == POINT_LIFT:
        task += f" It then lifted the jaw {millimetres(facts.lifted_m)} mm."
        lift = {"lifted": lifted_bucket(facts.lifted_m, facts.lift_m),
                "lifted_mm": millimetres(facts.lifted_m)}
    state: dict[str, Any] = {
        "task": task + " Judge the grasp.",
        "attempt": facts.attempt,
        "attempts_left": max(0, facts.max_attempts - facts.attempt),
        "point": facts.point,
        "close": {
            "ended": facts.ended,
            "jaw_gap": jaw_gap_bucket(facts.jaw_gap_m, facts.short_m, facts.commanded_gap_m,
                                      facts.margin_gap_m),
            "jaw_gap_mm": millimetres(facts.jaw_gap_m),
            "block_short_side_mm": millimetres(facts.short_m),
            "commanded_gap_mm": millimetres(facts.commanded_gap_m),
            "jaw_still": bool(facts.jaw_still),
            "jaw_moved_after_close": bool(facts.jaw_moved_after_close),
        },
        "lift": lift,
        "block_seen_between_jaws": blob_bucket(facts.blob_share),
        "block_sighting": sighting_state(facts.sighting, facts.aim_xy, facts.heading_rad, facts.now_t),
        "heading": {"from": facts.heading_from,
                    "yaw_symmetry_deg": round(math.degrees(facts.yaw_period_rad))},
        "available_next": available_next(facts),
        "earlier_attempts": [dict(entry) for entry in facts.earlier],
    }
    if facts.with_frames:
        state["camera_views"] = dict(facts.camera_views)
        state["frames_left_out"] = dict(facts.frames_left_out)
        state["frames_scaled"] = dict(facts.frames_scaled)
        state["frames_age_s"] = dict(facts.frames_age_s)
    return state


def grasp_questions(with_frames: bool = False) -> dict[str, dict[str, Any]]:
    """The two choice questions of every close and lift call, in TypeSafe's JSON shape. Question names are
    not sent to the model; option names and descriptions are, so they say what they mean."""
    frames = (["The image named in `camera_views` is the wrist camera's view between the jaws."]
              if with_frames else [])
    return {
        HELD: {
            "type": "choice",
            "instructions": {
                "question": "What is in the jaws now?",
                "focus": "`close` and `lift`",
                "known": [
                    "The jaw is commanded tighter than the block's short side, so a jaw that stops "
                    "`block_wide` is held apart by the block.",
                    "A gap `narrower_than_block` means the tips hold a corner or an edge, or something "
                    "thinner than the block.",
                    "A jaw at `closed` reached its command and met nothing.",
                    "A jaw that stopped `open`, wider than the block, holds it askew or across a corner, or "
                    "holds more than one block.",
                    "After a lift, `jaw_moved_after_close` true means the block shifted in the jaws.",
                    *frames,
                ],
            },
            "criteria": {
                "nothing_in_jaws": {
                    "what": "The jaw closed to its command, or the straddle ended before it closed.",
                    "not_for": "A jaw held apart by something.",
                },
                "block_held": {
                    "what": "The jaw stands `block_wide` on the block.",
                    "not_for": "A gap narrower or wider than the block.",
                },
                "held_by_edge": {
                    "what": ("The jaw holds a corner or an edge: `narrower_than_block` or `open`, or a jaw "
                             "that moved after the close."),
                    "not_for": "A jaw at `closed` or a steady block-wide jaw.",
                },
            },
        },
        NEXT: {
            "type": "choice",
            "instructions": {
                "question": "What should the arm do next?",
                "focus": "`available_next`, `close`, `lift` and `earlier_attempts`",
                "known": [
                    "Only an option in `available_next` can be done.",
                    "regrasp_wider opens the jaw wider before the next straddle: for a tip that landed on "
                    "the block.",
                    "regrasp_rotated turns the jaws by 90 degrees: for a heading that was a guess.",
                    "give_up leaves the block where it is and ends the run for a person.",
                    *frames,
                ],
            },
            "criteria": {
                "lift": {
                    "what": "Lift the block, or go on holding it.",
                    "not_for": "An empty jaw.",
                },
                "regrasp_wider": {
                    "what": "Let the block go, then grasp it again with the jaw opened one margin wider.",
                    "not_for": "A block held `block_wide`, or an option missing from `available_next`.",
                },
                "regrasp_rotated": {
                    "what": "Let the block go, then grasp it again with the jaws turned by 90 degrees.",
                    "not_for": ("A heading from the block's rotation, or an option missing from "
                                "`available_next`."),
                },
                "give_up": {
                    "what": "Stop and leave the block where it is for a person.",
                    "not_for": "A grasp another option can still mend.",
                },
            },
        },
    }


# ------------------------------------------------------------------ the strategy
class GraspJevStrategy(GraspStrategy):
    """Grasps a block as `grasp` does; Jev judges each probe, close and lift and picks the next step."""

    settings = GraspJevSettings

    def __init__(self, settings: GraspJevSettings | None = None, *, judge: Judge | None = None,
                 threaded: bool = True, clock: Callable[[], float] = time.monotonic,
                 blob: Callable[[np.ndarray], float | None] | None = None):
        config = settings if settings is not None else GraspJevSettings()
        super().__init__(config)
        self.jev_settings = config
        if not config.model.strip():
            raise ValueError("grasp_jev: model must not be empty")
        for name in ("min_confidence", "min_correction_confidence"):
            value = getattr(config, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"grasp_jev: {name} must lie within 0..1, is {value}")
        if not SAME_PLACE_M < config.correction_m <= 0.1:
            raise ValueError(f"grasp_jev: correction_m must lie within ({SAME_PLACE_M:g}, 0.1] m, "
                             f"is {config.correction_m}")
        if not 0.0 < config.answer_timeout_s <= 60.0:
            raise ValueError(f"grasp_jev: answer_timeout_s must lie within (0, 60] s, "
                             f"is {config.answer_timeout_s}")
        #: The camera's key in `obs.frames`: exact, so never lower-cased.
        self.wrist_camera = config.wrist_camera.strip()
        if not self.wrist_camera:
            raise ValueError("grasp_jev: wrist_camera must not be empty")
        #: Reads no key and opens no connection until `start`.
        self.judge: Judge = judge if judge is not None else JevJudge(config.model.strip())
        #: The share of a wrist frame the block fills between the jaws, 0..1; None = unknown (the seam).
        self.blob = blob
        self.touch_questions = touch_questions(config.correction_m, config.send_frames)
        self.grasp_questions = grasp_questions(config.send_frames)
        self.consult = Consultation(self.judge, threaded=threaded, clock=clock, log_label="Ocena chwytu")
        #: The jaw rule's margin as a gap at the close [m], from `start`.
        self.margin_gap_m = math.nan
        self._phases.update({JUDGE: self._judge_grasp, LOWER: self._lower, LET_GO: self._let_go})
        self._reset_judging()

    def _reset_judging(self) -> None:
        self.consult.reset()
        # The question asked last, (point, attempt), and the lift_off and check points already decided.
        self._asked: tuple[str, int] | None = None
        self._decided: set[tuple[str, int]] = set()
        # The probe's geometry (`descent_verdict`), and what the open questions were built from.
        self._geometry = ""
        self._facts: DescentFacts | None = None
        self._gfacts: GraspFacts | None = None
        self._failed_why = ""
        #: The jaw angle on the first lift_off tick [rad].
        self._q_close = math.nan
        # The run's aim so far, as touch_jev keeps it for the probe question.
        self._first_sighting: Sighting | None = None
        self._aim_sighting: Sighting | None = None
        self._newest_used_seen_t = -math.inf
        self._chosen_by = FIRST_SIGHTING
        self._earlier_descents: list[dict[str, Any]] = []
        self._earlier_grasps: list[dict[str, Any]] = []
        self._tried: list[tuple[float, float]] = []
        self._summary = ""
        #: A turn of the closing heading for the next aim [rad] (regrasp_rotated).
        self._turn_rad = 0.0
        # Letting the block go before a regrasp: whether the jaw is open, where it rises to, where to aim.
        self._released = False
        self._release_point = np.zeros(3)
        self._regrasp_xy: np.ndarray | None = None

    @property
    def judging(self) -> JudgeStatus:
        """What the decision step did in this run (`judges.JudgeStatus`)."""
        return self.consult.judging

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Refuses the run when the judge cannot be asked (no key), before the arm moves; then grasp's start
        with the operator's opening, which a regrasp_wider of an earlier run may have widened."""
        self.judge.check()
        self.gap_open_m = self.short_m + 2 * self.grasp_settings.open_margin_m
        super().start(run, obs)
        self._reset_judging()
        spec = self.kin.spec
        assert spec.gripper_range_q is not None
        closed, opened = spec.gripper_range_q
        units = spec.jaw.margin / (opened - closed) * 100.0
        self.margin_gap_m = (self.kin.jaw_gap(min(100.0, self.cmd_closed + units))
                             - self.kin.jaw_gap(self.cmd_closed))

    def act(self, obs: Observation) -> Action:
        action = super().act(obs)
        if self._first_sighting is None and self.grasp.block_xy is not None:
            # `_wait` took the pose on this tick, so its frames were taken `pose_age_s` before it.
            seen_t = float(obs.t) - self.status.pose_age_s
            self._first_sighting = self._aim_sighting = Sighting(self.status.pose_source, seen_t,
                                                                 self.grasp.block_xy)
            self._newest_used_seen_t = seen_t
        return self._with_decisions(action)

    def _with_decisions(self, action: Action) -> Action:
        """The tick's answer, to carry the decisions since the previous tick: as touch_jev's, the pinned api
        (0.1.2) has no `Action.decisions`, so the records stay in `drain_decisions`."""
        return action

    def wait_for_answer(self, timeout_s: float | None = None) -> None:
        """Waits for the call in flight, if any (tests); the next tick collects its answer."""
        self.consult.wait(timeout_s)

    def drain_decisions(self) -> tuple[Decision, ...]:
        """The records since the last drain, each once; cancelled calls at `stop` are only in
        `judging.decisions`."""
        return self.consult.drain()

    def _begin(self, state: Mapping[str, Any], questions: Mapping[str, Any], images: Mapping[str, np.ndarray],
               obs: Observation, point: str, geometry: str) -> None:
        version = PROBE_QUESTIONS_VERSION if point == POINT_PROBE else QUESTIONS_VERSION
        attempt = self.status.attempt
        meta = {"questions": version, "model_asked": self.jev_settings.model, "attempt": attempt,
                "point": point, "geometry": geometry}
        self.consult.begin(state, questions, images, now_t=float(obs.t),
                           timeout_s=self.jev_settings.answer_timeout_s, meta=meta,
                           context=f"proba {attempt}")

    # ------------------------------------------------------------------ point 1: the probe
    def after_touch(self, obs: Observation) -> Action:
        """Asks the judge once per probe, holds while the answer is pending, then acts on it."""
        st = self.status
        key = (POINT_PROBE, st.attempt)
        if self._asked != key:
            if st.contact and not st.drop_m >= self.config.contact_gap_m:
                # A stall where the arm never came down from the hover: grasp's default ends the run.
                return super().after_touch(obs)
            kind, _ = self.descent_verdict()
            self._asked, self._geometry = key, kind
            views, left_out, scaled, ages, images = self._frames(obs)
            self._facts = self._probe_facts(obs, views, left_out)
            state = descent_state(self._facts)
            if self.jev_settings.send_frames:
                state["frames_scaled"] = scaled
                state["frames_age_s"] = ages
            self._begin(state, self.touch_questions, images, obs, POINT_PROBE, kind)
            return self.hold(f"{self.judge.name}: pytam o ocene dotyku (proba {st.attempt})")
        result = self.consult.step(float(obs.t))
        if result is None:
            return self.hold(self.consult.note)
        if isinstance(result, JudgeError):
            return self._review(result.reason)
        return self._decide_probe(result, obs)

    def _probe_facts(self, obs: Observation, views: dict[str, str], left_out: dict[str, str]) -> DescentFacts:
        st, config = self.status, self.jev_settings
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        latest = None
        if pose is not None and not self._pose_problem(pose):
            position = np.asarray(pose.position, float)
            latest = Sighting(str(pose.source or ""), float(obs.t) - float(pose.age_s),
                              (float(position[0]), float(position[1])))
        block = self.grasp.block_xy
        assert block is not None
        return DescentFacts(
            label=self.label, attempt=st.attempt, max_attempts=self.grasp_settings.max_attempts,
            contact=st.contact, drop_m=st.drop_m, stop_above_m=st.stop_z_m - self.config.touch_z,
            rise_m=st.rise_m, contact_band_m=self.config.contact_band_m, touch_xy=block,
            correction_m=config.correction_m, now_t=float(obs.t), chosen_by=self._chosen_by,
            aim_sighting=self._aim_sighting, latest=latest, newest_used_seen_t=self._newest_used_seen_t,
            earlier=tuple(self._earlier_descents), tried_xy=tuple(self._tried),
            with_frames=config.send_frames, camera_views=views, frames_left_out=left_out)

    def _decide_probe(self, reply: Reply, obs: Observation) -> Action:
        """touch_jev's rules, except that a confident touch goes on into grasp's straddle, and a touch the
        geometry says met nothing or stalled above the band ends for review."""
        config, st, name = self.jev_settings, self.status, self.judge.name
        facts = self._facts
        assert facts is not None
        verdict, correction = reply.answers[TOUCHED], reply.answers[CORRECTION]
        sure, option, option_sure = verdict.confidence, correction.choice, correction.confidence
        kind = self._geometry
        target = self.kin.from_q(self.q_cmd)
        suffix = ""
        if verdict.choice != GEOMETRY_AS_OPTION.get(kind):
            self.judging.disagreements += 1
            suffix = f"; geometria: {GEOMETRY_PL.get(kind, kind)}"
        if sure < config.min_confidence:
            return self._review(f"{name} niepewny: {VERDICTS_PL[verdict.choice]} z pewnoscia {sure:.2f} "
                                f"< {config.min_confidence:.2f}{suffix}")
        if verdict.choice == "touched_object":
            if kind in ("touched", "reached"):
                self._summary = f"{name} #{st.attempt}: dotkniety {sure:.2f}"
                return super().after_touch(obs)
            if kind == "missed":
                return self._review(f"{name}: dotkniety ({sure:.2f}), ale sonda nie napotkala nic{suffix}")
            return self._review(f"{name}: dotkniety ({sure:.2f}), ale ramie stanelo "
                                f"{1000 * facts.stop_above_m:.0f} mm nad punktem dotyku{suffix}")
        if verdict.choice == "blocked_above_object":
            return self._halt(f"{name}: ramie zablokowane nad obiektem (pewnosc {sure:.2f}){suffix}", target)
        max_attempts = self.grasp_settings.max_attempts
        if st.attempt >= max_attempts:
            return self._halt(f"{name}: pudlo w ostatniej probie ({st.attempt} z {max_attempts}, "
                              f"pewnosc {sure:.2f}){suffix}", target)
        if option_sure < config.min_correction_confidence:
            return self._review(f"{name}: pudlo ({sure:.2f}), ale poprawka niepewna: "
                                f"{CORRECTIONS_PL[option]} {option_sure:.2f} "
                                f"< {config.min_correction_confidence:.2f}{suffix}")
        point = correction_target(option, facts)
        if isinstance(point, str):
            return self._review(f"{name}: poprawka '{CORRECTIONS_PL[option]}' niewykonalna: {point}{suffix}")
        descent = {"attempt": st.attempt, "chosen_by": self._chosen_by,
                   "ended": "stalled_on_something" if facts.contact else "reached_touch_point_in_free_air",
                   "stop_height": stop_height(facts), "verdict": verdict.choice, "correction": option}
        # The sideways retries after a tip start from the corrected block centre, as after a recovery.
        saved = self._block_xy0, self._shift_k
        self._block_xy0, self._shift_k = None, 0
        try:
            self.retarget(*point)
        except RuntimeError as exc:
            self._block_xy0, self._shift_k = saved
            return self._review(f"{name}: poprawka '{CORRECTIONS_PL[option]}' niewykonalna: {exc}{suffix}")
        self._earlier_descents.append(descent)
        self._tried.append(facts.touch_xy)
        self._chosen_by = f"correction:{option}"
        if option == "aim_at_latest_sighting" and facts.latest is not None:
            self._aim_sighting = facts.latest
            self._newest_used_seen_t = max(self._newest_used_seen_t, facts.latest.seen_t)
        else:
            self._aim_sighting = None
        self._summary = (f"{name} #{st.attempt}: {VERDICTS_PL[verdict.choice]} {sure:.2f}, "
                         f"{CORRECTIONS_PL[option]} {option_sure:.2f}")
        return self.hold(f"{name}: pudlo ({sure:.2f}), poprawka {CORRECTIONS_PL[option]} ({option_sure:.2f}) "
                         f"- proba {st.attempt + 1}")

    # ------------------------------------------------------------------ point 2: a failed close or a tip
    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        """An empty close or a jaw tip on the block: asks the judge, the arm holding, in the phase judge."""
        self._failed_why = why
        attempt = self.status.attempt
        self._asked = (POINT_CLOSE, attempt)
        views, left_out, scaled, ages, images = self._frames(obs)
        self._gfacts = self._grasp_facts(obs, POINT_CLOSE, ENDED[why], self.reading,
                                         (views, left_out, scaled, ages))
        self._begin(grasp_state(self._gfacts), self.grasp_questions, images, obs, POINT_CLOSE, why)
        self._enter(JUDGE)
        name = self.judge.name
        if why == "empty":
            return self.hold(f"{name}: pytam o ocene chwytu po zamknieciu (proba {attempt})")
        return self.hold(f"{name}: pytam o ocene chwytu (szczeka na bloku, proba {attempt})")

    def _judge_grasp(self, obs: Observation) -> Action:
        result = self.consult.step(float(obs.t))
        if result is None:
            return self.hold(self.consult.note)
        if isinstance(result, JudgeError):
            return self._review(result.reason)
        action = self._decide_grasp(result, obs, POINT_CLOSE, from_judge=True)
        assert action is not None, "lift is never available after an empty close or a tip"
        return action

    # ------------------------------------------------------------------ points 2 and 3: held, lifted
    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        """Grasp's watch first: a drop goes to its recovery, cancelling a pending question. Then, on the first
        lift_off tick after a held close and on the first check tick after the lift, the grasp question;
        the arm holds until the answer, and lift goes on from it."""
        watch = super().carrying(obs, reading)
        if watch is not None:
            if self.consult.pending:
                self.consult.cancel(f"{self.judge.name}: pytanie odwolane (blok wypadl ze szczeki)")
            if self.grasp.verdict == "dropped":
                self._earlier_grasps.append({"attempt": self.status.attempt, "probe": self._geometry,
                                             "close": "dropped_while_carrying", "verdict": None,
                                             "next": "recovery", "chosen_by": self._chosen_by})
            return watch
        phase = self.status.phase
        if phase not in (LIFT_OFF, CHECK):
            return None
        point = POINT_CLOSE if phase == LIFT_OFF else POINT_LIFT
        attempt = self.status.attempt
        key = (point, attempt)
        if key in self._decided:
            return None
        if self._asked != key:
            self._asked = key
            if point == POINT_CLOSE:
                self._q_close = reading.q
            views, left_out, scaled, ages, images = self._frames(obs)
            self._gfacts = self._grasp_facts(obs, point, ENDED["held"], reading,
                                             (views, left_out, scaled, ages))
            self._begin(grasp_state(self._gfacts), self.grasp_questions, images, obs, point, "held")
            return self.hold(f"{self.judge.name}: pytam o ocene chwytu {POINT_PL[point]} (proba {attempt})")
        result = self.consult.step(float(obs.t))
        if result is None:
            return self.hold(self.consult.note)
        self._decided.add(key)
        if isinstance(result, JudgeError):
            return self._review(result.reason)
        return self._decide_grasp(result, obs, point, from_judge=False)

    def _decide_grasp(self, reply: Reply, obs: Observation, point: str, from_judge: bool) -> Action | None:
        """Acts on a close or lift answer: None = lift goes on (from `carrying` only), else the tick's
        Action."""
        config, st, name = self.jev_settings, self.status, self.judge.name
        facts = self._gfacts
        assert facts is not None
        held, nxt = reply.answers[HELD], reply.answers[NEXT]
        choice, sure, step, step_sure = held.choice, held.confidence, nxt.choice, nxt.confidence
        attempt = st.attempt
        target = self.kin.from_q(self.q_cmd)
        is_held = facts.ended == ENDED["held"]
        suffix = ""
        if choice != ("block_held" if is_held else "nothing_in_jaws"):
            self.judging.disagreements += 1
            suffix = f"; szczeka: {'trzyma' if is_held else 'pusta'}"
        if sure < config.min_confidence:
            return self._review(f"{name} niepewny: {HELD_PL[choice]} z pewnoscia {sure:.2f} "
                                f"< {config.min_confidence:.2f}{suffix}")
        if step_sure < config.min_correction_confidence:
            return self._review(f"{name}: {HELD_PL[choice]} ({sure:.2f}), ale krok niepewny: {NEXT_PL[step]} "
                                f"{step_sure:.2f} < {config.min_correction_confidence:.2f}{suffix}")
        problem = next_problem(step, facts)
        if problem:
            if step in ("regrasp_wider", "regrasp_rotated") and attempt >= facts.max_attempts:
                return self._halt(f"{name}: {HELD_PL[choice]} w ostatniej probie ({attempt} z "
                                  f"{facts.max_attempts}, pewnosc {sure:.2f}){suffix}", target)
            return self._review(f"{name}: krok '{NEXT_PL[step]}' niewykonalny: {problem}{suffix}")
        self._earlier_grasps.append({"attempt": attempt, "probe": self._geometry, "close": self.grasp.verdict,
                                     "verdict": choice, "next": step, "chosen_by": self._chosen_by})
        self._summary = (f"{name} #{attempt} {POINT_PL[point]}: {HELD_PL[choice]} {sure:.2f}, "
                         f"{NEXT_PL[step]} {step_sure:.2f}")
        if step == "lift":
            # The answer restarts the phase clock: the lift's time limit and the check's hold count from it.
            self._phase_t0 = self._t
            return None
        if step == "give_up":
            # The jaw keeps its command: a held block stays held after the run, as grasp's handover.
            where = "w szczekach" if is_held else "na stole"
            return self._halt(f"{name}: rezygnacja ({HELD_PL[choice]} {sure:.2f}){suffix} - blok {where}",
                              target)
        note = (f"{name}: {HELD_PL[choice]} ({sure:.2f}), {NEXT_PL[step]} ({step_sure:.2f}) "
                f"- proba {attempt + 1}")
        if step == "regrasp_wider":
            self._widen()
        else:
            self._turn_rad = math.pi / 2
        self._chosen_by, self._aim_sighting = f"next:{step}", None
        if from_judge:
            if step == "regrasp_wider":
                # grasp's sideways retry: half the short side along the jaws, toward the moving jaw first.
                if self._next_shift() is None:
                    return self._halt(f"{name}: wyczerpane przesuniecia chwytu ({attempt})", target)
            else:
                # A wrong heading, not a wrong place: the same block centre.
                assert self.grasp.block_xy is not None
                self._retarget_xy = np.array(self.grasp.block_xy, float)
            self._start_rise(RETARGET)
            return self.hold(note)
        # The block lies between the fingertips: their midpoint is where it will be let go.
        self._regrasp_xy = self._jaw_point(obs.joints)[:2].copy()
        self._released = False
        self._enter(LET_GO if point == POINT_CLOSE else LOWER)
        return self.hold(note)

    def _widen(self) -> None:
        """The straddle's opening one `open_margin_m` wider on each side, for every later straddle."""
        spec = self.kin.spec
        self.gap_open_m += 2 * self.grasp_settings.open_margin_m
        self.cmd_open = self.kin.gripper_for_gap(self.gap_open_m)
        self._centre_open = self.kin.jaw_centre(self.cmd_open)
        self._recover_open_q = min(float(self._hi[self._g]),
                                   max(self._q(self.cmd_open), spec.jaw.open + 2 * spec.jaw.margin))

    # ------------------------------------------------------------------ setting the block down
    def _lower(self, obs: Observation) -> Action | None:
        """The held block back down to the grasp point; a stall on the way counts as arrival."""
        watch = self.carrying(obs, self.reading)
        if watch is not None:
            return watch
        assert self.grasp.grasp_point is not None
        point = np.array(self.grasp.grasp_point, float)
        if self._offset_z - self._bias_z > self.config.contact_gap_m:
            # The block meets the table before the TCP reaches the grasp point and holds the arm up: the arm
            # holds what it measures and nothing presses into the table.
            self._hold_measured()
            self._released = False
            self._enter(LET_GO)
            return None
        if self._arrived(point):
            self._released = False
            self._enter(LET_GO)
            return None
        down_s = self.grasp_settings.lift_m / self.config.descend_speed_mps
        if self._t - self._phase_t0 >= down_s + JAW_WAIT_S:
            self._hold_measured()
            return self._halt("ramie nie opuscilo bloku na stol w czasie", self.kin.from_q(self.q_cmd))
        return self._drive(point, self.descend_step_m)

    def _let_go(self, obs: Observation) -> Action | None:
        """Opens the jaw on the block on the table, rises with it open, and at the hover rises on to the
        regrasp: grasp's rise closes the jaw there and aims at the block centre again."""
        if not self._released:
            self._set_jaw_goal(self.cmd_open)
            self._ramp_jaw()
            jaw_open = abs(self.reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
            if self._jaw_ready() and jaw_open:
                self._released = True
                hover_z = self.config.touch_z + self.config.hover_m
                self._release_point = np.array([self.tcp_cmd[0], self.tcp_cmd[1],
                                                max(hover_z, float(self.tcp_cmd[2]))])
            elif self._t - self._phase_t0 >= JAW_WAIT_S:
                gaps = f"{self.grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm"
                return self._halt(f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s ({gaps})",
                                  self.kin.from_q(self.q_cmd))
            else:
                return self.hold()
        if self._arrived(self._release_point):
            assert self._regrasp_xy is not None
            self._retarget_xy = self._regrasp_xy.copy()
            self._start_rise(RETARGET)
            return None
        return self._drive(self._release_point, self.approach_step_m)

    def _hold_measured(self) -> None:
        """The arm joints where they are, the jaw as commanded: the arm stops pressing."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()

    # ------------------------------------------------------------------ aiming
    def _aim(self, xy: np.ndarray) -> None:
        """grasp's aim, with a pending turn of the heading (regrasp_rotated) tried first, then its mirror."""
        if not self._turn_rad:
            super()._aim(xy)
            return
        saved = self._headings
        psi = wrap(self._psi + self._turn_rad)
        self._headings = [psi, wrap(psi + math.pi)]
        try:
            super()._aim(xy)
        except RuntimeError:
            self._headings = saved
            raise
        self._turn_rad = 0.0

    def _regrasp(self, seen: WatchSighting) -> Action:
        """grasp's regrasp after a drop; the probe question then knows its aim came from the recovery."""
        before = self.grasp.block_xy
        action = super()._regrasp(seen)
        if self.status.phase != DONE and self.grasp.block_xy != before:
            self._chosen_by = "recovery"
            self._aim_sighting = Sighting(seen.source, seen.seen_t, seen.xy)
            self._newest_used_seen_t = max(self._newest_used_seen_t, seen.seen_t)
            # The block lies somewhere new: the points tried before say nothing about it.
            self._tried = []
        return action

    # ------------------------------------------------------------------ facts
    def _wrist_frame(self, obs: Observation) -> tuple[Frame | None, str]:
        """The wrist camera's frame of this tick, or why there is none (Polish, for the state)."""
        try:
            frame = obs.frames[self.wrist_camera]
        except KeyError:
            return None, "brak klatki"
        age = float(frame.age_s)
        if not age <= FRAME_MAX_AGE_S:
            return None, f"klatka sprzed {age:.1f} s"
        return frame, ""

    def _frames(self, obs: Observation) -> Frames:
        """With `send_frames`: the wrist camera's frame fitted to Jev's sides, or why it is left out."""
        views: dict[str, str] = {}
        left_out: dict[str, str] = {}
        scaled: dict[str, str] = {}
        ages: dict[str, float] = {}
        images: dict[str, np.ndarray] = {}
        if not self.jev_settings.send_frames:
            return views, left_out, scaled, ages, images
        name = self.wrist_camera
        frame, problem = self._wrist_frame(obs)
        if frame is None:
            left_out[name] = problem
            return views, left_out, scaled, ages, images
        image = np.asarray(frame.image)
        if image.ndim != 3:
            left_out[name] = frame_problem(image.shape)
            return views, left_out, scaled, ages, images
        fitted, k = fit_frame(image)
        problem = frame_problem(fitted.shape)
        if problem:
            left_out[name] = problem
            return views, left_out, scaled, ages, images
        reference = uuid.uuid4().hex
        views[name] = reference
        images[reference] = np.array(fitted, copy=True)
        ages[name] = round(float(frame.age_s), 1)
        if k > 1:
            scaled[name] = f"{image.shape[1]}x{image.shape[0]} -> {fitted.shape[1]}x{fitted.shape[0]}"
        return views, left_out, scaled, ages, images

    def _blob_share(self, obs: Observation) -> float | None:
        """The blob seam on the wrist frame of this tick; None without a blob or a fresh frame."""
        if self.blob is None:
            return None
        frame, _ = self._wrist_frame(obs)
        if frame is None:
            return None
        share = self.blob(np.asarray(frame.image))
        if share is None or not math.isfinite(float(share)):
            return None
        return float(min(max(float(share), 0.0), 1.0))

    def _block_sighting(self, obs: Observation, reading: JawReading) -> BlockSighting | None:
        """The block's pose of this tick from any source, annotated ones included."""
        seen = self._sighting(obs, reading.squeezing)
        if seen is None:
            return None
        return BlockSighting(seen.source, seen.seen_t, seen.xy, seen.z)

    def _grasp_facts(self, obs: Observation, point: str, ended: str, reading: JawReading,
                     frames: tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, float]]
                     ) -> GraspFacts:
        spec, kin = self.kin.spec, self.kin
        assert spec.gripper is not None and self.grasp.block_xy is not None
        views, left_out, scaled, ages = frames
        half = self.block.half_size
        moved = (point == POINT_LIFT and math.isfinite(self._q_close)
                 and abs(reading.q - self._q_close) > spec.jaw.still)
        guessed = bool(np.allclose(self.block_rotation, np.eye(3), atol=1e-6))
        margin = self.grasp_settings.open_margin_m
        return GraspFacts(
            label=self.label, kind=self.block.kind, short_m=2 * half[1], long_m=2 * half[0],
            height_m=2 * half[2], attempt=self.status.attempt, max_attempts=self.grasp_settings.max_attempts,
            point=point, ended=ended, jaw_gap_m=self.grasp.jaw_gap_m,
            commanded_gap_m=kin.jaw_gap(kin.from_q(self.q_cmd)[spec.gripper]), margin_gap_m=self.margin_gap_m,
            jaw_still=reading.still, jaw_moved_after_close=moved,
            lifted_m=self.grasp.lifted_m if point == POINT_LIFT else math.nan,
            lift_m=self.grasp_settings.lift_m, gap_open_m=self.gap_open_m, blob_share=self._blob_share(obs),
            sighting=self._block_sighting(obs, reading), aim_xy=self.grasp.block_xy, heading_rad=self._psi,
            heading_from="guess" if guessed else "block_rotation", yaw_period_rad=self.block.yaw_period,
            wider_possible=self.gap_open_m + 2 * margin <= kin.jaw_gap(100.0) + 1e-9, now_t=float(obs.t),
            earlier=tuple(self._earlier_grasps), with_frames=self.jev_settings.send_frames,
            camera_views=views, frames_left_out=left_out, frames_scaled=scaled, frames_age_s=ages)

    # ------------------------------------------------------------------ the end
    def _review(self, reason: str) -> Action:
        """Ends the run with the arm holding where it is and the jaw as commanded, for a person to judge."""
        self.judging.needs_review = True
        return self._halt(f"{reason} - ramie stoi, do oceny czlowieka", self.kin.from_q(self.q_cmd))

    def stop(self, reason: str) -> None:
        """Also gives up on a pending question; its records go only to `judging.decisions`, as the run's log
        is closed by now. Never waits and never raises."""
        self.consult.cancel(f"{self.judge.name}: pytanie odwolane ({reason})", to_outbox=False)
        super().stop(reason)

    def _note(self) -> str:
        st = self.status
        if st.phase == JUDGE:
            return self.consult.note or "czekam na ocene chwytu"
        if st.phase == LET_GO:
            if not self._released:
                return f"puszczam blok: {self.grasp.gap_cmd_m * 1000:.0f} mm"
            return f"w gore z otwarta szczeka: {st.distance * 1000:.0f} mm"
        if st.phase == LOWER:
            return f"opuszczam blok na stol: {st.distance * 1000:.0f} mm"
        text = super()._note()
        if self._summary and st.phase in SUMMARY_PHASES:
            return f"{text} | {self._summary}"
        return text
