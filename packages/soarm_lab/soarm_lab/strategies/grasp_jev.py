"""Ask Jev after each close and lift, then apply the chosen grasp step.

`grasp_jev` extends `grasp` through `after_grasp_failed` and `carrying`.
With `probe: true`, it also uses `after_touch`. `Consultation` asks off the tick thread while
the arm holds. `start` validates the judge before motion.
See the package README for settings and `TYPESAFE_API_KEY` setup.

The task sent to Jev says whether a probe preceded the straddle.
With `probe: true`, Jev also judges each probe. Probe questions use touch's state with the block
centre as the touch point. A confident touch continues the grasp only when the probe geometry
can supply a usable top. Misses may correct the aim within the attempt budget.

Close and lift questions receive jaw, lift, pose-source and optional wrist-image facts.
`held` records the verdict. `next` selects lift, wider regrasp, rotated regrasp or give-up.
Confidence thresholds and geometry constrain the action. A wider regrasp keeps the fixed
tip in place. A held block is lowered and released before another grasp.
The open hand moves toward the fixed jaw along the closing axis before rising.
This clears the fixed tip from the block's face.

Grasp's drop monitoring remains active while awaiting answers and lowering. A drop cancels
the pending consultation and starts recovery. Recovered attempts are judged again.

Each call records its state, questions and outcome, including failures and cancellations.
Each tick's `Action.decisions` carries the records new since the previous tick, for the run's log.
Model waits count toward `time_limit_s`. Optional frames are limited to the named wrist
camera and checked for size and age. `blob=` supplies colour occupancy between the jaws;
without it the state reports unknown.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Frame, Observation, RunInfo

from .grasp import (
    CARRY,
    CHECK,
    GRASP_PHASE_NAMES,
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

#: State and question version recorded for close and lift decisions.
QUESTIONS_VERSION = "grasp_jev/1"
#: Minimum block-colour share between the jaws for a positive sighting [0..1].
BLOB_FILL_MIN = 0.15
#: Gap tolerance around the block's short side [m].
EDGE_M = 0.003
#: Near-jaw sighting radius around the aim [m]. `SAME_PLACE_M` defines the at-jaw radius.
NEAR_M = 0.03
#: Maximum wrist-frame age for sending [s].
FRAME_MAX_AGE_S = 1.0

#: Added phases for judging a failed grasp, lowering a held block and releasing before a regrasp.
JUDGE = "judge"
LOWER = "lower"
LET_GO = "let_go"
#: The phases' names in the panel's notes.
JEV_PHASE_NAMES = {**GRASP_PHASE_NAMES, JUDGE: "ocena chwytu", LOWER: "odkladam blok na stol",
                   LET_GO: "puszczam blok"}

#: Decision points stored in each record's `state.point`.
POINT_PROBE = "probe"
POINT_CLOSE = "close"
POINT_LIFT = "lift"

HELD = "held"
NEXT = "next"
HELD_OPTIONS = ("nothing_in_jaws", "block_held", "held_by_edge")
NEXT_OPTIONS = ("lift", "regrasp_wider", "regrasp_rotated", "give_up")
#: State labels for each closing verdict.
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
#: Frame metadata maps names to references, omission reasons, scaled sizes and ages [s].
#: Images are keyed by reference.
Frames = tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, float], dict[str, np.ndarray]]


@dataclass
class GraspJevSettings(GraspSettings):
    #: Whole-run time limit including model waits and retries [s]. Zero disables it.
    time_limit_s: float = 180.0
    #: Pinned Jev model version used with these confidence thresholds.
    model: str = DEFAULT_MODEL
    #: The confidence a verdict (touched, held) needs to be acted on without a person [0..1].
    min_confidence: float = 0.8
    #: The confidence a correction after a missed probe or a next step needs [0..1].
    min_correction_confidence: float = 0.6
    #: Probe correction step [m], greater than `SAME_PLACE_M` and at most 0.1.
    correction_m: float = 0.02
    #: The longest wait for one answer, counted from the question; also the HTTP timeout [s].
    answer_timeout_s: float = 5.0
    #: Send the wrist frame captured for the question.
    send_frames: bool = False
    #: Wrist camera key in `obs.frames`, matched exactly after trimming whitespace.
    wrist_camera: str = "wrist"


@dataclass(frozen=True)
class BlockSighting:
    """An observed block pose and capture time, including annotated tracker sources."""

    source: str
    #: When its frames were taken [s, `obs.t`].
    seen_t: float
    #: Its x and y and its height [m, base frame].
    xy: tuple[float, float]
    z: float


@dataclass(frozen=True)
class GraspFacts:
    """Facts captured on the tick thread for a close or lift question."""

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
    #: Whether a closed-jaw probe preceded the straddle. Controls the task description sent to Jev.
    probed: bool = False
    with_frames: bool = False
    camera_views: Mapping[str, str] = field(default_factory=dict)
    frames_left_out: Mapping[str, str] = field(default_factory=dict)
    frames_scaled: Mapping[str, str] = field(default_factory=dict)
    frames_age_s: Mapping[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ pure state
def fit_frame(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Return `image` with integer stride `k` and the stride used.

    Choose the smallest stride that fits the longer side within Jev's image limit.
    The result is contiguous and uses no resampling.
    """
    height, width = image.shape[:2]
    k = max(1, math.ceil(max(height, width) / IMAGE_SIDE_PX[1]))
    return np.ascontiguousarray(image[::k, ::k]), k


def source_name(source: str) -> str:
    """Map a pose source to a state label, recognizing annotated last-seen prefixes."""
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
    """Classify measured gap relative to block width and the commanded opening.

    Only a jaw commanded tighter than the block can count as closed at its command.
    """
    if commanded_m < short_m - EDGE_M and gap_m <= commanded_m + margin_gap_m:
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
    """Return pose source, age and offsets along and across the closing heading."""
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
    """Return a Polish reason a step is unavailable, or an empty string when allowed."""
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
    """Return steps allowed by the grasp geometry and attempt budget."""
    return [option for option in NEXT_OPTIONS if not next_problem(option, facts)]


def grasp_state(facts: GraspFacts) -> dict[str, Any]:
    """Build the close or lift state with JSON values and finite numbers or null.

    The schema is fixed. Frame metadata is included only when frame sending is enabled.
    """
    sizes = f"{millimetres(facts.long_m)} x {millimetres(facts.short_m)} x {millimetres(facts.height_m)} mm"
    block = f"the block '{facts.label}' ({facts.kind}, {sizes})"
    if facts.probed:
        task = (f"A robot arm probed the top of {block} with its closed jaw tips, rose, opened the jaw to "
                f"{millimetres(facts.gap_open_m)} mm")
    else:
        task = f"A robot arm opened its jaw to {millimetres(facts.gap_open_m)} mm above {block}"
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
    """Return the close and lift choice questions in TypeSafe's JSON format.

    The model receives option names and descriptions, but not question names.
    """
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
    """Grasps a block as `grasp` does; Jev judges each close and lift and picks the next step."""

    settings = GraspJevSettings
    phase_names = JEV_PHASE_NAMES
    #: Refuse an empty `wrist_camera`. Subclasses with an optional camera turn this off.
    camera_required = True

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
        #: Case-sensitive camera key in `obs.frames`.
        self.wrist_camera = config.wrist_camera.strip()
        if not self.wrist_camera and self.camera_required:
            raise ValueError("grasp_jev: wrist_camera must not be empty")
        #: Judge setup is deferred until `start`.
        self.judge: Judge = judge if judge is not None else JevJudge(config.model.strip())
        #: Colour occupancy callback [0..1]. None leaves the sighting unknown.
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
        # Track the last question and decision points already handled for this attempt.
        self._asked: tuple[str, int] | None = None
        self._decided: set[tuple[str, int]] = set()
        # Retain geometry and facts used for the pending question.
        self._geometry = ""
        self._facts: DescentFacts | None = None
        self._gfacts: GraspFacts | None = None
        self._failed_why = ""
        #: The jaw angle on the first lift_off tick [rad].
        self._q_close = math.nan
        # Pose history for judging probe corrections.
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
        # Release progress, ease-off and rise targets, and next aim before regrasping.
        self._released = False
        #: The held jaw's gap when the release began [m]; NaN before a release.
        self._held_gap_m = float("nan")
        self._eased = False
        self._ease_point = np.zeros(3)
        self._release_point = np.zeros(3)
        self._regrasp_xy: np.ndarray | None = None

    @property
    def judging(self) -> JudgeStatus:
        """Return this run's judge status and decision records."""
        return self.consult.judging

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Validate the judge before motion and start grasp with the configured opening.

        Discard any wider opening left by the previous run.
        """
        self.judge.check()
        self.gap_open_m = self.short_m + 2 * self.grasp_settings.open_margin_m
        self._held_gap_m = float("nan")
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
            # Recover capture time from the age of the pose just latched.
            seen_t = float(obs.t) - self.status.pose_age_s
            self._first_sighting = self._aim_sighting = Sighting(self.status.pose_source, seen_t,
                                                                 self.grasp.block_xy)
            self._newest_used_seen_t = seen_t
        return self._with_decisions(action)

    def _with_decisions(self, action: Action) -> Action:
        """Return the action carrying the records drained since the previous tick, for the run's log."""
        drained = self.drain_decisions()
        return replace(action, decisions=(*action.decisions, *drained)) if drained else action

    def wait_for_answer(self, timeout_s: float | None = None) -> None:
        """Wait for an in-flight call in tests. The next tick collects its result."""
        self.consult.wait(timeout_s)

    def drain_decisions(self) -> tuple[Decision, ...]:
        """Return records since the last drain, once each.

        Records cancelled at stop remain only in `judging.decisions`.
        """
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
        """Ask once per probe and hold until the answer selects the next step."""
        st = self.status
        key = (POINT_PROBE, st.attempt)
        if self._asked != key:
            if st.contact and not st.drop_m >= self.config.contact_gap_m:
                # Let grasp handle a stall without any descent.
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
        """Apply touch's verdict rules, continuing grasp after a usable touch.

        A claimed touch without contact or above the contact band requires review.
        """
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
        # Tip retries start from the corrected block centre.
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
        """Hold after an empty close or tip stall and ask the judge in phase JUDGE."""
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
        """Run grasp's drop monitor before consulting the judge.

        A drop cancels the pending question and starts recovery. Otherwise ask once after a held
        close and once after the lift. Hold while waiting and continue from the answer.
        """
        watch = super().carrying(obs, reading)
        if watch is not None:
            if self.consult.pending:
                self.consult.cancel(f"{self.judge.name}: pytanie odwolane (blok wypadl ze szczeki)")
            if self.grasp.verdict == "dropped":
                self._earlier_grasps.append({"attempt": self.status.attempt, "probe": self._geometry or None,
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
        """Apply a close or lift answer. Return None to continue carrying, or an action to interrupt it."""
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
        self._earlier_grasps.append({"attempt": attempt, "probe": self._geometry or None,
                                     "close": self.grasp.verdict, "verdict": choice, "next": step,
                                     "chosen_by": self._chosen_by})
        self._summary = (f"{name} #{attempt} {POINT_PL[point]}: {HELD_PL[choice]} {sure:.2f}, "
                         f"{NEXT_PL[step]} {step_sure:.2f}")
        if step == "lift":
            # Exclude model wait time from lift travel and hold verification.
            self._phase_t0 = self._t
            return None
        if step == "give_up":
            # Keep a held block squeezed when giving up.
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
            if step == "regrasp_wider" and self._failed_why == "tip":
                # Use grasp's sideways retry, toward the moving jaw first.
                if self._next_shift() is None:
                    return self._halt(f"{name}: wyczerpane przesuniecia chwytu ({attempt})", target)
            else:
                # Keep the block centre after an empty close or heading correction.
                assert self.grasp.block_xy is not None
                self._retarget_xy = np.array(self.grasp.block_xy, float)
            self._start_rise(RETARGET)
            return self.hold(note)
        # Release at the measured fingertip midpoint.
        self._regrasp_xy = self._jaw_point(obs.joints)[:2].copy()
        self._released = False
        self._enter(LET_GO if point == POINT_CLOSE else LOWER)
        return self.hold(note)

    def _widen(self) -> None:
        """Increase later straddle openings by twice `open_margin_m` on the moving jaw's side.

        Keep the original `_centre_open` so the fixed tip and grasp height stay in place.
        """
        spec = self.kin.spec
        self.gap_open_m += 2 * self.grasp_settings.open_margin_m
        self.cmd_open = self.kin.gripper_for_gap(self.gap_open_m)
        self._recover_open_q = min(float(self._hi[self._g]),
                                   max(self._q(self.cmd_open), spec.jaw.open + 2 * spec.jaw.margin))

    # ------------------------------------------------------------------ setting the block down
    def _lower(self, obs: Observation) -> Action | None:
        """Lower to the grasp point. A stall counts as arrival and stops further pressure."""
        watch = self.carrying(obs, self.reading)
        if watch is not None:
            return watch
        assert self.grasp.grasp_point is not None
        point = np.array(self.grasp.grasp_point, float)
        if self._offset_z - self._bias_z > self.config.contact_gap_m:
            # Stop pressing when the lowered block meets the table.
            self._hold_measured()
            self._released = False
            self._enter(LET_GO)
            return None
        if self._arrived(point):
            self._released = False
            self._enter(LET_GO)
            return None
        down_s = self._lift_m / self.config.descend_speed_mps
        if self._t - self._phase_t0 >= down_s + JAW_WAIT_S:
            self._hold_measured()
            return self._halt("ramie nie opuscilo bloku na stol w czasie", self.kin.from_q(self.q_cmd))
        return self._drive(point, self.descend_step_m)

    def _let_go(self, obs: Observation) -> Action | None:
        """Release at the table and clear the fixed tip before rising.

        Move the open hand toward the fixed jaw along the closing axis at release height.
        Then rise with the jaw open before re-aiming.
        """
        if not self._released:
            self._set_jaw_goal(self.release_cmd)
            self._ramp_jaw()
            jaw_open = abs(self.reading.q - float(self.q_cmd[self._g])) <= self.kin.spec.jaw.margin
            if self._jaw_ready() and jaw_open:
                self._released = True
                # Move toward the fixed jaw (-u) to clear its tip before rising.
                u = np.array([math.cos(self._psi), math.sin(self._psi)])
                eased = self.tcp_cmd[:2] - self.ease_off_m * u
                self._eased = False
                self._ease_point = np.array([*eased, float(self.tcp_cmd[2])])
                self._release_point = np.array([*eased, self._let_go_rise_z()])
            elif self._t - self._phase_t0 >= JAW_WAIT_S:
                gaps = f"{self.grasp.jaw_gap_m * 1000:.0f} mm zamiast {self.gap_open_m * 1000:.0f} mm"
                return self._halt(f"szczeka nie otworzyla sie w {JAW_WAIT_S:.0f} s ({gaps})",
                                  self.kin.from_q(self.q_cmd))
            else:
                return self.hold()
        if not self._eased:
            if not self._arrived(self._ease_point):
                return self._drive(self._ease_point, self.descend_step_m)
            self._eased = True
        if self._arrived(self._release_point, sag=True):
            return self._after_let_go()
        return self._drive(self._release_point, self.rise_step_m)

    @property
    def release_cmd(self) -> float:
        """Return the gripper command that releases a held block [app units]: the open jaw, and at least
        `open_margin_m` past the held jaw on each side."""
        return max(self.cmd_open, self._past_held(self.grasp_settings.open_margin_m))

    def _past_held(self, margin_m: float) -> float:
        """The gripper command `margin_m` [m] past the held jaw on each side [app units]; 0 before a hold."""
        if not math.isfinite(self._held_gap_m):
            return 0.0
        kin = self.kin
        return kin.gripper_for_gap(min(self._held_gap_m + 2 * margin_m, kin.jaw_gap(100.0)))

    def _enter(self, phase: str) -> None:
        if phase == LET_GO:
            # The release opens relative to the jaw as it holds now.
            self._held_gap_m = self.grasp.jaw_gap_m
        super()._enter(phase)

    def _let_go_rise_z(self) -> float:
        """Return the TCP height the open hand rises to after easing off [m, base frame]."""
        hover_z = self.config.touch_z + self.config.hover_m
        return max(hover_z, float(self.tcp_cmd[2]))

    def _after_let_go(self) -> Action | None:
        """Re-aim at the released block once the open hand has risen."""
        assert self._regrasp_xy is not None
        self._retarget_xy = self._regrasp_xy.copy()
        self._start_rise(RETARGET)
        return None

    @property
    def rise_step_m(self) -> float:
        """Return the open hand's rise per tick after the release [m]: at the approach speed."""
        return self.approach_step_m

    @property
    def ease_off_m(self) -> float:
        """Return the clearance toward the fixed jaw along the closing axis before rising [m].

        Use `open_margin_m` so the fixed tip leaves the released block's face.
        """
        return self.grasp_settings.open_margin_m

    def _hold_measured(self) -> None:
        """Hold measured arm joints while retaining the jaw command."""
        self.q_cmd = np.clip(self.kin.to_q(self._hold_pose(self._measured)), self._lo, self._hi)
        self._contacts = self.kin.self_contacts(self.kin.from_q(self.q_cmd))
        self.tcp_cmd = self.kin.tcp(self.kin.from_q(self.q_cmd))[:3, 3].copy()
        self.carrot = self.tcp_cmd.copy()

    # ------------------------------------------------------------------ aiming
    def _aim(self, xy: np.ndarray, *, strict: bool = False) -> None:
        """Try the pending rotated heading and its mirror before accepting a new aim."""
        if not self._turn_rad:
            super()._aim(xy, strict=strict)
            return
        saved = self._headings
        psi = wrap(self._psi + self._turn_rad)
        self._headings = [psi, wrap(psi + math.pi)]
        try:
            super()._aim(xy, strict=strict)
        except RuntimeError:
            self._headings = saved
            raise
        self._turn_rad = 0.0

    def _regrasp(self, seen: WatchSighting) -> Action:
        """Run grasp recovery and mark the new probe aim as chosen by recovery."""
        before = self.grasp.block_xy
        action = super()._regrasp(seen)
        if self.status.phase != DONE and self.grasp.block_xy != before:
            self._chosen_by = "recovery"
            self._aim_sighting = Sighting(seen.source, seen.seen_t, seen.xy)
            self._newest_used_seen_t = max(self._newest_used_seen_t, seen.seen_t)
            # Discard old probe points after the block moves.
            self._tried = []
        return action

    # ------------------------------------------------------------------ facts
    def _wrist_frame(self, obs: Observation) -> tuple[Frame | None, str]:
        """Return this tick's usable wrist frame or the reason it is unavailable."""
        try:
            frame = obs.frames[self.wrist_camera]
        except KeyError:
            return None, "brak klatki"
        age = float(frame.age_s)
        if not age <= FRAME_MAX_AGE_S:
            return None, f"klatka sprzed {age:.1f} s"
        return frame, ""

    def _frames(self, obs: Observation) -> Frames:
        """Prepare the optional wrist image and metadata about sent, scaled or omitted frames."""
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
        """Return wrist colour occupancy, or None without a blob function and fresh frame."""
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
        """Return this tick's block pose from any source, including annotated ones."""
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
            label=self.label, kind=self.block.name, short_m=2 * half[1], long_m=2 * half[0],
            height_m=2 * half[2], attempt=self.status.attempt, max_attempts=self.grasp_settings.max_attempts,
            point=point, ended=ended, jaw_gap_m=self.grasp.jaw_gap_m,
            commanded_gap_m=kin.jaw_gap(kin.from_q(self.q_cmd)[spec.gripper]), margin_gap_m=self.margin_gap_m,
            jaw_still=reading.still, jaw_moved_after_close=moved,
            lifted_m=self.grasp.lifted_m if point == POINT_LIFT else math.nan,
            lift_m=self._lift_m, gap_open_m=self.gap_open_m, blob_share=self._blob_share(obs),
            sighting=self._block_sighting(obs, reading), aim_xy=self.grasp.block_xy, heading_rad=self._psi,
            heading_from="guess" if guessed else "block_rotation", yaw_period_rad=self.block.yaw_period,
            wider_possible=self.gap_open_m + 2 * margin <= kin.jaw_gap(100.0) + 1e-9, now_t=float(obs.t),
            earlier=tuple(self._earlier_grasps), probed=self.grasp_settings.probe,
            with_frames=self.jev_settings.send_frames,
            camera_views=views, frames_left_out=left_out, frames_scaled=scaled, frames_age_s=ages)

    # ------------------------------------------------------------------ the end
    def _review(self, reason: str) -> Action:
        """Stop for review with the current arm target and jaw command retained."""
        self.judging.needs_review = True
        return self._halt(f"{reason} - ramie stoi, do oceny czlowieka", self.kin.from_q(self.q_cmd))

    def stop(self, reason: str) -> None:
        """Cancel pending questions without waiting and keep their records in `judging.decisions`.

        The run log is closed, so cancellation records are not added to the outbox.
        """
        self.consult.cancel(f"{self.judge.name}: pytanie odwolane ({reason})", to_outbox=False)
        super().stop(reason)

    def _note(self) -> str:
        st = self.status
        if st.phase == JUDGE:
            return self.consult.note or "czekam na ocene chwytu"
        if st.phase == LET_GO:
            if not self._released:
                return f"puszczam blok: {self.grasp.gap_cmd_m * 1000:.0f} mm"
            if not self._eased:
                return f"odsuwam szczeke od bloku: {st.distance * 1000:.0f} mm"
            return f"w gore z otwarta szczeka: {st.distance * 1000:.0f} mm"
        if st.phase == LOWER:
            return f"opuszczam blok na stol: {st.distance * 1000:.0f} mm"
        text = super()._note()
        if self._summary and st.phase in SUMMARY_PHASES:
            return f"{text} | {self._summary}"
        return text
