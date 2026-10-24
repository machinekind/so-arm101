"""Jev in the touch loop: `touch_jev` under the `machinekind_twin.strategies` entry points.

`JevTouchStrategy` is `touch` with a decision step after each descent. It asks Jev, TypeSafe's decision model
(`jev.JevJudge`), whether the gripper touched the object, and where to aim when it missed; touch's motion,
guards and stop reasons stay as they are. Nothing here imports the twin.

    # configs/local.yaml of the twin: prefills the panel's form
    touch_jev:
      label: cube
      model: jev-1.14.0              # pinned: the thresholds belong to one version
      min_confidence: 0.8            # a less certain verdict stops the arm for a person
      min_correction_confidence: 0.6
      max_attempts: 3                # descents per run, the first included
      answer_timeout_s: 5.0

The key comes from `TYPESAFE_API_KEY` in the twin's environment (`uv run --env-file .env twin ui`); `start`
refuses the run without it, before the arm moves.

The question. On the first tick of the "touched" phase of a descent the strategy builds the run's state
(`descent_state`) and asks two choice questions in one call (`touch_questions`): `touched` (missed_object,
touched_object, blocked_above_object) and `correction` (aim_at_latest_sighting, or one `correction_m` step
farther from the base, closer to it, to the robot's left or to its right). The state holds facts computed in
code, with named buckets beside the millimetres, because Jev is weak with numbers: how the descent ended,
how high it stopped against the object's top, how far it came down, where the latest sighting places the
object against the touch point and whether it is a new look, the earlier descents of the run, and which
corrections would repeat a point already tried. It never holds the geometry's own verdict, joint values,
`TouchStatus.distance` or the key. Missed_object is the first option because Jev leans toward the first
one, and a wrong miss costs one more descent while a wrong touch is a false success. The correction is
always asked with the verdict, so one round trip serves a miss, and read only after a miss.

The rules, in order:

* a stall where the arm never came down from the hover is not asked about, at any height: touch's
  default ends the run;
* a verdict less confident than `min_confidence` ends the run, the arm holding where it stopped, for a
  person to judge (`JudgeStatus.needs_review`);
* touched_object is a success, and blocked_above_object ends the run;
* missed_object on the last attempt (`max_attempts`) ends the run;
* missed_object with a correction less confident than `min_correction_confidence` ends the run for review;
* otherwise code turns the option into a point (`correction_target`; the model never gives coordinates)
  and calls `retarget`: the arm rises straight up, goes to the new hover point and down again, and the
  next descent is asked about. An option that gives no new point (no new look, a point already tried) or
  that the arm cannot reach ends the run for review.

Confidence is how concentrated the answer's probabilities are, not the chance that it is right, and the
thresholds are TypeSafe's starting points (0.8 to act alone, 0.6 as a floor), not tuned for this task yet.
When the model's verdict differs from the geometry's (`descent_verdict`), the reason says so.

The call. `JudgeRunner` asks on a thread of its own and `after_touch` polls it every tick, returning `hold`
meanwhile, so the arm keeps the pose it stopped in with touch's lag guard on and the 20 Hz tick never
waits. The worker gets the state as ready JSON values and copies of the frames, never the kinematics. A
call that does not answer within `answer_timeout_s` (counted on `obs.t` from the ask) is given up. A
timeout, an overload or a dropped connection is asked again once, after `RETRY_WAIT_S`, with the same
state; after that, or after any other failure (a refused key, a refused request, an unusable answer), the
run ends for review. A failed call is never read as an answer. Model waits count toward `time_limit_s`.

The records. Every call gives one `judges.Decision` per question, with the answer as received (or the
error), the exact state sent, the confidence and the response time; failed, timed-out and cancelled calls
too. They collect in `judging.decisions` and in `drain_decisions`, and a summary goes into the notes, the
stop reason and one log line per call. The pinned api (0.1.2) has no `Action.decisions`: the twin's runner
would drop an extra attribute, so `act` returns plain `Action`s. Once the pin reaches api 0.1.3,
`_with_decisions` hands the drained records to `Action.decisions` and the twin's episode log keeps them.
`state.sent` and `state.questions_sent`, with `QUESTIONS_VERSION`, let Claude, Jev or a person be asked the
identical input later.

With `send_frames` on, each camera frame of the asking tick within 280..1568 px also goes out as a PNG
image named in the state (`camera_views`), for an A/B against the text-only state. Jev's images are not
production-guaranteed by TypeSafe.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from .jev import DEFAULT_MODEL, JevJudge, frame_problem
from .judges import Decision, Judge, JudgeError, JudgeRunner, Outcome, Reply
from .touch import APPROACH, DESCEND, LIFT, TouchSettings, TouchStrategy, normalised

logger = logging.getLogger(__name__)

#: The version of the state and the questions, recorded with every decision.
QUESTIONS_VERSION = "touch_jev/1"
#: Calls per descent: the first and one retry after a timeout, an overload or a dropped connection.
ASKS_PER_DESCENT = 2
#: The pause before that retry [s].
RETRY_WAIT_S = 0.5
#: Two touch points this close are the same place [m].
SAME_PLACE_M = 0.01
#: A sighting is a new look when its frames are this much newer than those of every sighting the run aimed
#: at [s]. A repeated "ostatnie widziane" answer keeps its capture time, so it never is.
NEW_LOOK_S = 0.5

TOUCHED = "touched"
CORRECTION = "correction"
FIRST_SIGHTING = "first_sighting"
VERDICTS = ("missed_object", "touched_object", "blocked_above_object")
CORRECTIONS = ("aim_at_latest_sighting", "farther_from_base", "closer_to_base", "to_robot_left",
               "to_robot_right")
#: The twin's pose sources (`ObjectPose.source`) as the state names them.
SOURCES = {"kamery": "cameras", "ostatnie widziane": "last_seen", "symulacja": "simulation",
           "w dloni": "in_hand", "": "unnamed"}
#: For notes and stop reasons, in Polish without diacritics.
VERDICTS_PL = {"missed_object": "pudlo", "touched_object": "dotkniety",
               "blocked_above_object": "zablokowany nad obiektem"}
CORRECTIONS_PL = {"aim_at_latest_sighting": "do ostatniego polozenia",
                  "farther_from_base": "dalej od podstawy", "closer_to_base": "blizej podstawy",
                  "to_robot_left": "w lewo", "to_robot_right": "w prawo"}
GEOMETRY_PL = {"touched": "kontakt przy obiekcie", "reached": "punkt dotyku bez kontaktu",
               "missed": "brak kontaktu", "blocked": "zablokowane nad obiektem"}
#: The verdict option the geometry's own verdict (`descent_verdict`) agrees with.
GEOMETRY_AS_OPTION = {"touched": "touched_object", "reached": "missed_object", "missed": "missed_object",
                      "blocked": "blocked_above_object"}
DIRECTIONS = {"farther_from_base": "farther from the arm's base",
              "closer_to_base": "closer to the arm's base",
              "to_robot_left": "to the robot's left", "to_robot_right": "to the robot's right"}


@dataclass
class JevTouchSettings(TouchSettings):
    #: A safety net for the whole run, retries and model waits included [s]; 0 means none.
    time_limit_s: float = 90.0
    #: The Jev model, a version rather than an alias: the thresholds belong to one version.
    model: str = DEFAULT_MODEL
    #: The confidence of the verdict needed to act without a person, 0..1.
    min_confidence: float = 0.8
    #: The confidence of the correction needed to aim again after a miss, 0..1.
    min_correction_confidence: float = 0.6
    #: Descents per run, the first included.
    max_attempts: int = 3
    #: One step of a correction that moves the touch point [m]; more than `SAME_PLACE_M`, or every step
    #: would count as the point already tried.
    correction_m: float = 0.02
    #: The longest wait for one answer, counted from the question; also the HTTP timeout [s].
    answer_timeout_s: float = 5.0
    #: Also send the camera frames of the asking tick as images (Jev's images are not production-ready).
    send_frames: bool = False


@dataclass
class JudgeStatus:
    """What the decision step did in this run, for the panel's logs and tests."""

    judge: str = ""
    #: The model that answered last.
    model: str = ""
    asks: int = 0
    answers: int = 0
    failures: int = 0
    last_response_s: float = float("nan")
    last_verdict: str = ""
    last_correction: str = ""
    #: The run ended for a person to judge: an unsure answer, a failed call, an impossible correction.
    needs_review: bool = False
    #: Answers whose verdict differs from the geometry's.
    disagreements: int = 0
    #: Every record of the run, cancelled calls included.
    decisions: list[Decision] = field(default_factory=list)


@dataclass(frozen=True)
class Sighting:
    """Where a pose placed the object, and when its frames were taken [s, `obs.t`]."""

    source: str
    seen_t: float
    xy: tuple[float, float]


@dataclass(frozen=True)
class DescentFacts:
    """What the state of one question is built from, read on the tick thread."""

    label: str
    attempt: int
    max_attempts: int
    contact: bool
    drop_m: float
    #: The measured TCP's height above the touch point where the descent stopped [m].
    stop_above_m: float
    rise_m: float
    contact_band_m: float
    touch_xy: tuple[float, float]
    correction_m: float
    now_t: float
    #: How the current touch point was chosen: "first_sighting" or "correction:<option>".
    chosen_by: str = FIRST_SIGHTING
    #: The sighting the current touch point came from; None after a step in a direction.
    aim_sighting: Sighting | None = None
    #: The object's usable pose on the asking tick.
    latest: Sighting | None = None
    #: When the frames of the newest sighting the run aimed at were taken [s, `obs.t`].
    newest_used_seen_t: float = -math.inf
    earlier: tuple[Mapping[str, Any], ...] = ()
    #: The touch points of the earlier descents [m, base frame].
    tried_xy: tuple[tuple[float, float], ...] = ()
    with_frames: bool = False
    camera_views: Mapping[str, str] = field(default_factory=dict)
    frames_left_out: Mapping[str, str] = field(default_factory=dict)


def axes(touch_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors at the touch point: away from the base, and to the robot's left (base +y side)."""
    xy = np.asarray(touch_xy, float)
    length = float(np.linalg.norm(xy))
    out = xy / length if length > 1e-9 else np.array([1.0, 0.0])
    return out, np.array([-out[1], out[0]])


def direction(offset: np.ndarray, out: np.ndarray, left: np.ndarray) -> str:
    """Which way `offset` points from the touch point, by its larger component, or "same_place"."""
    if float(np.linalg.norm(offset)) <= SAME_PLACE_M:
        return "same_place"
    along, across = float(offset @ out), float(offset @ left)
    if abs(along) >= abs(across):
        return "farther_from_base" if along > 0 else "closer_to_base"
    return "to_robot_left" if across > 0 else "to_robot_right"


def new_look(facts: DescentFacts) -> bool:
    """Whether the latest sighting was seen after every sighting the run aimed at."""
    latest = facts.latest
    return latest is not None and latest.seen_t > facts.newest_used_seen_t + NEW_LOOK_S


def millimetres(value: float) -> int | None:
    return round(1000 * value) if math.isfinite(value) else None


def seconds_ago(seen_t: float, now_t: float) -> int | None:
    value = now_t - seen_t
    return round(value) if math.isfinite(value) else None


def sighting_state(sighting: Sighting | None, now_t: float) -> dict[str, Any] | None:
    if sighting is None:
        return None
    return {"source": SOURCES.get(sighting.source, "other"),
            "seen_s_ago": seconds_ago(sighting.seen_t, now_t)}


def latest_state(facts: DescentFacts) -> dict[str, Any] | None:
    latest = facts.latest
    if latest is None:
        return None
    out, left = axes(facts.touch_xy)
    offset = np.subtract(latest.xy, facts.touch_xy)
    return {**(sighting_state(latest, facts.now_t) or {}), "new_look": new_look(facts),
            "direction": direction(offset, out, left),
            "offset_mm": {"farther": millimetres(float(offset @ out)),
                          "left": millimetres(float(offset @ left))}}


def stop_height(facts: DescentFacts) -> str:
    if not facts.contact:
        return "at_touch_point"
    return "near_object_top" if facts.stop_above_m <= facts.contact_band_m else "far_above_object_top"


def descent_state(facts: DescentFacts) -> dict[str, Any]:
    """The state Jev judges a descent from: JSON values, finite numbers or null, the same keys every time
    (the frame keys only with frames on)."""
    state: dict[str, Any] = {
        "task": (f"A robot arm lowered its gripper straight down toward the top of the object "
                 f"'{facts.label}' and stopped. Judge that descent."),
        "attempt": facts.attempt,
        "attempts_left": max(0, facts.max_attempts - facts.attempt),
        "descent": {
            "ended": "stalled_on_something" if facts.contact else "reached_touch_point_in_free_air",
            "drop_from_hover_mm": millimetres(facts.drop_m),
            "stop_height": stop_height(facts),
            "stop_above_touch_point_mm": millimetres(facts.stop_above_m),
            "pushed_back_mm": millimetres(facts.rise_m),
        },
        "aim": {"chosen_by": facts.chosen_by, "sighting": sighting_state(facts.aim_sighting, facts.now_t)},
        "latest_sighting": latest_state(facts),
        "earlier_descents": [dict(entry) for entry in facts.earlier],
        "would_repeat_earlier_point": repeated_options(facts),
    }
    if facts.with_frames:
        state["camera_views"] = dict(facts.camera_views)
        state["frames_left_out"] = dict(facts.frames_left_out)
    return state


def correction_point(option: str, facts: DescentFacts) -> np.ndarray | None:
    """Where a correction option puts the touch point [m, base frame]; None when aim_at_latest_sighting has
    no new look to aim at."""
    touch = np.asarray(facts.touch_xy, float)
    if option == "aim_at_latest_sighting":
        latest = facts.latest
        return np.asarray(latest.xy, float) if latest is not None and new_look(facts) else None
    out, left = axes(facts.touch_xy)
    unit = {"farther_from_base": out, "closer_to_base": -out, "to_robot_left": left, "to_robot_right": -left}
    return touch + facts.correction_m * unit[option]


def repeats_a_tried_point(point: np.ndarray, facts: DescentFacts) -> bool:
    tried = (*facts.tried_xy, facts.touch_xy)
    return any(float(np.linalg.norm(point - np.asarray(xy, float))) <= SAME_PLACE_M for xy in tried)


def repeated_options(facts: DescentFacts) -> list[str]:
    """The correction options that would aim at a touch point this run already tried, the current one
    included. Computed in code: Jev is weak with numbers."""
    points = {option: correction_point(option, facts) for option in CORRECTIONS}
    return [option for option, point in points.items()
            if point is not None and repeats_a_tried_point(point, facts)]


def correction_target(option: str, facts: DescentFacts) -> tuple[float, float] | str:
    """The new touch point for a correction option, or why there is none (Polish, for the panel)."""
    point = correction_point(option, facts)
    if point is None:
        return "brak nowego polozenia z kamer"
    if repeats_a_tried_point(point, facts):
        return "punkt juz probowany"
    return float(point[0]), float(point[1])


def touch_questions(correction_m: float, with_frames: bool = False) -> dict[str, dict[str, Any]]:
    """The two choice questions of every call, in TypeSafe's JSON shape. Question names are not sent to
    the model; option names and descriptions are, so they say what they mean."""
    frames = ["The images named in `camera_views` show the scene after the descent."] if with_frames else []
    step = f"{correction_m * 100:g} cm"
    directions = {
        option: {"what": f"Move the touch point one step, {step}, {text}.",
                 "not_for": "An option listed in `would_repeat_earlier_point`."}
        for option, text in DIRECTIONS.items()
    }
    return {
        TOUCHED: {
            "type": "choice",
            "instructions": {
                "question": ("Did the descent in `descent` end with the gripper resting on top of the object "
                             "named in `task`?"),
                "focus": "`descent`",
                "known": [
                    "The hand reaches lower than the tool point, so a hand that lands on the object's top "
                    "stops with the tool point a little above the touch point.",
                    "A descent that reached the touch point in free air met nothing on the way down.",
                    *frames,
                ],
            },
            "criteria": {
                "missed_object": {
                    "what": ("The gripper went down to the touch point without meeting anything: "
                             "`descent.ended` is reached_touch_point_in_free_air."),
                    "not_for": "A descent that stalled on something.",
                },
                "touched_object": {
                    "what": ("The gripper stalled on something near the object's top: `descent.ended` is "
                             "stalled_on_something and `descent.stop_height` is near_object_top."),
                    "not_for": "A stall far above the object's top, or a descent that met nothing.",
                },
                "blocked_above_object": {
                    "what": ("The gripper stalled on something far above the object's top: "
                             "`descent.stop_height` is far_above_object_top."),
                    "not_for": "A stall near the object's top.",
                },
            },
        },
        CORRECTION: {
            "type": "choice",
            "instructions": {
                "question": ("If the gripper missed the object, which change of the touch point is most "
                             "likely to put it on the object?"),
                "focus": "`latest_sighting`, `earlier_descents` and `would_repeat_earlier_point`",
                "known": [
                    "The cameras place the object up to 3 cm off.",
                    "Only a sighting with `new_look` true tells something the earlier descents did not "
                    "already try.",
                    "Directions are as seen from the arm's base looking out over the touch point.",
                    *frames,
                ],
            },
            "criteria": {
                "aim_at_latest_sighting": {
                    "what": "Aim where `latest_sighting` places the object.",
                    "not_for": "A missing sighting, `new_look` false, or `direction` same_place.",
                },
                **directions,
            },
        },
    }


class JevTouchStrategy(TouchStrategy):
    """Touches an object as `touch` does; Jev judges each descent and picks where to try again."""

    settings = JevTouchSettings

    def __init__(self, settings: JevTouchSettings | None = None, *, judge: Judge | None = None,
                 threaded: bool = True, clock: Callable[[], float] = time.monotonic):
        config = settings if settings is not None else JevTouchSettings()
        super().__init__(config)
        self.jev_settings = config
        if not config.model.strip():
            raise ValueError("touch_jev: model must not be empty")
        for name in ("min_confidence", "min_correction_confidence"):
            value = getattr(config, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"touch_jev: {name} must lie within 0..1, is {value}")
        if not config.max_attempts >= 1:
            raise ValueError(f"touch_jev: max_attempts must be at least 1, is {config.max_attempts}")
        if not SAME_PLACE_M < config.correction_m <= 0.1:
            raise ValueError(f"touch_jev: correction_m must lie within ({SAME_PLACE_M:g}, 0.1] m, "
                             f"is {config.correction_m}")
        if not 0.0 < config.answer_timeout_s <= 60.0:
            raise ValueError(f"touch_jev: answer_timeout_s must lie within (0, 60] s, "
                             f"is {config.answer_timeout_s}")
        #: Reads no key and opens no connection until `start`.
        self.judge: Judge = judge if judge is not None else JevJudge(config.model.strip())
        self.questions = touch_questions(config.correction_m, config.send_frames)
        self._threaded = threaded
        self._clock = clock
        self._runner = JudgeRunner(self.judge, threaded=threaded, clock=clock)
        self._reset_judging()

    def _reset_judging(self) -> None:
        self.judging = JudgeStatus(judge=self.judge.name)
        self._outbox: list[Decision] = []
        # The current descent's question: its attempt, the calls made, when the last one went out, when to
        # ask again after a failure, and what was sent.
        self._asked_attempt = 0
        self._asks = 0
        self._asked_t = 0.0
        self._retry_at: float | None = None
        self._facts: DescentFacts | None = None
        self._sent: dict[str, Any] = {}
        self._images: dict[str, np.ndarray] = {}
        self._geometry = ""
        # The run's aim so far: the first sighting, how the current point was chosen and from which
        # sighting, the earlier descents and their touch points.
        self._first_sighting: Sighting | None = None
        self._aim_sighting: Sighting | None = None
        self._newest_used_seen_t = -math.inf
        self._chosen_by = FIRST_SIGHTING
        self._earlier: list[dict[str, Any]] = []
        self._tried: list[tuple[float, float]] = []
        self._summary = ""

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Refuses the run when the judge cannot be asked (no key), before the arm moves."""
        self.judge.check()
        super().start(run, obs)
        self._runner.cancel()
        self._runner = JudgeRunner(self.judge, threaded=self._threaded, clock=self._clock)
        self._reset_judging()

    def act(self, obs: Observation) -> Action:
        action = super().act(obs)
        if self._first_sighting is None and self.status.touch_point is not None:
            # `_wait` took the pose on this tick, so its frames were taken `pose_age_s` before it.
            seen_t = float(obs.t) - self.status.pose_age_s
            self._first_sighting = self._aim_sighting = Sighting(
                self.status.pose_source, seen_t, (float(self.touch_xy[0]), float(self.touch_xy[1])))
            self._newest_used_seen_t = seen_t
        return self._with_decisions(action)

    def _with_decisions(self, action: Action) -> Action:
        """The tick's answer, to carry the decisions that came back since the previous tick.

        The pinned api (0.1.2) has no `Action.decisions`, so the records stay in `drain_decisions` and
        `judging.decisions` and the notes carry the summary. Once the pin reaches api 0.1.3 this returns
        `dataclasses.replace(action, decisions=self.drain_decisions())`.
        """
        return action

    def wait_for_answer(self, timeout_s: float | None = None) -> None:
        """Waits for the call in flight, if any (tests); the next tick collects its answer."""
        self._runner.wait(timeout_s)

    def drain_decisions(self) -> tuple[Decision, ...]:
        """The records since the last drain, each once; cancelled calls are only in `judging.decisions`."""
        drained, self._outbox = tuple(self._outbox), []
        return drained

    # ------------------------------------------------------------------ the decision step

    def after_touch(self, obs: Observation) -> Action:
        """Asks the judge once per descent, holds while the answer is pending, then acts on it."""
        st = self.status
        if self._asked_attempt != st.attempt:
            if st.contact and not st.drop_m >= self.config.contact_gap_m:
                # A stall where the arm never came down from the hover, frozen or jammed: touch calls it
                # stuck within the contact band and a block above it, and either way its default ends it.
                return super().after_touch(obs)
            kind, _ = self.descent_verdict()
            self._asked_attempt, self._asks, self._geometry = st.attempt, 0, kind
            views, left_out, self._images = self._frames(obs)
            self._facts = self._descent_facts(obs, views, left_out)
            self._sent = descent_state(self._facts)
            self._ask(obs)
            return self.hold(f"{self.judge.name}: pytam o ocene dotyku (proba {st.attempt})")
        outcome = self._runner.poll()
        if outcome is None:
            return self._waiting(obs)
        self._record(outcome)
        if outcome.error is not None:
            return self._failed(outcome.error, obs)
        assert outcome.reply is not None
        return self._decide(outcome.reply)

    def _frames(self, obs: Observation) -> tuple[dict[str, str], dict[str, str], dict[str, np.ndarray]]:
        """With `send_frames`: the cameras sent (camera -> reference), the ones left out and why, and copies
        of the images by reference."""
        views: dict[str, str] = {}
        left_out: dict[str, str] = {}
        images: dict[str, np.ndarray] = {}
        if not self.jev_settings.send_frames:
            return views, left_out, images
        for camera in obs.frames:
            try:
                image = np.asarray(obs.frames[camera].image)
            except KeyError:
                left_out[camera] = "brak klatki"
                continue
            problem = frame_problem(image.shape)
            if problem:
                left_out[camera] = problem
                continue
            reference = uuid.uuid4().hex
            views[camera] = reference
            images[reference] = np.array(image, copy=True)
        return views, left_out, images

    def _descent_facts(self, obs: Observation, views: dict[str, str],
                       left_out: dict[str, str]) -> DescentFacts:
        st, config = self.status, self.jev_settings
        pose = {normalised(str(key)): value for key, value in obs.objects.items()}.get(self.label)
        latest = None
        if pose is not None and not self._pose_problem(pose):
            position = np.asarray(pose.position, float)
            latest = Sighting(str(pose.source or ""), float(obs.t) - float(pose.age_s),
                              (float(position[0]), float(position[1])))
        return DescentFacts(
            label=self.label, attempt=st.attempt, max_attempts=config.max_attempts, contact=st.contact,
            drop_m=st.drop_m, stop_above_m=st.stop_z_m - config.touch_z, rise_m=st.rise_m,
            contact_band_m=config.contact_band_m, touch_xy=(float(self.touch_xy[0]), float(self.touch_xy[1])),
            correction_m=config.correction_m, now_t=float(obs.t), chosen_by=self._chosen_by,
            aim_sighting=self._aim_sighting, latest=latest, newest_used_seen_t=self._newest_used_seen_t,
            earlier=tuple(self._earlier), tried_xy=tuple(self._tried), with_frames=config.send_frames,
            camera_views=views, frames_left_out=left_out)

    def _ask(self, obs: Observation) -> None:
        self._asks += 1
        self._asked_t = float(obs.t)
        self._retry_at = None
        self.judging.asks += 1
        self._runner.ask(self._sent, self.questions, self._images, self.jev_settings.answer_timeout_s)

    def _waiting(self, obs: Observation) -> Action:
        """No outcome yet: ask again once the retry pause is over, give up after the timeout, else hold."""
        name, attempt = self.judge.name, self.status.attempt
        if self._retry_at is not None:
            if obs.t >= self._retry_at:
                self._ask(obs)
                return self.hold(f"{name}: pytam ponownie (proba {attempt})")
            return self.hold(f"{name}: pytam ponownie za chwile (proba {attempt})")
        waited = float(obs.t) - self._asked_t
        timeout = self.jev_settings.answer_timeout_s
        if waited < timeout:
            return self.hold(f"{name}: czekam na odpowiedz {waited:.1f} s (proba {attempt})")
        self._runner.cancel()
        error = JudgeError(f"{name} nie odpowiedzial w {timeout:g} s", kind="timeout", retryable=True)
        self._record(Outcome(reply=None, error=error, response_s=None))
        return self._failed(error, obs)

    def _failed(self, error: JudgeError, obs: Observation) -> Action:
        if error.retryable and self._asks < ASKS_PER_DESCENT:
            self._retry_at = float(obs.t) + RETRY_WAIT_S
            return self.hold(f"{error.reason}, pytam ponownie za {RETRY_WAIT_S:g} s")
        return self._review(error.reason)

    def _decide(self, reply: Reply) -> Action:
        config, st, name = self.jev_settings, self.status, self.judge.name
        facts = self._facts
        assert facts is not None
        verdict, correction = reply.answers[TOUCHED], reply.answers[CORRECTION]
        sure, option, option_sure = verdict.confidence, correction.choice, correction.confidence
        target = self.kin.from_q(self.q_cmd)
        suffix = ""
        if verdict.choice != GEOMETRY_AS_OPTION.get(self._geometry):
            self.judging.disagreements += 1
            suffix = f"; geometria: {GEOMETRY_PL.get(self._geometry, self._geometry)}"
        if sure < config.min_confidence:
            return self._review(f"{name} niepewny: {VERDICTS_PL[verdict.choice]} z pewnoscia {sure:.2f} "
                                f"< {config.min_confidence:.2f}{suffix}")
        if verdict.choice == "touched_object":
            return self._halt(f"{name}: obiekt dotkniety (pewnosc {sure:.2f}){suffix}", target, success=True)
        if verdict.choice == "blocked_above_object":
            return self._halt(f"{name}: ramie zablokowane nad obiektem (pewnosc {sure:.2f}){suffix}", target)
        if st.attempt >= config.max_attempts:
            return self._halt(f"{name}: pudlo w ostatniej probie ({st.attempt} z {config.max_attempts}, "
                              f"pewnosc {sure:.2f}){suffix}", target)
        if option_sure < config.min_correction_confidence:
            return self._review(f"{name}: pudlo ({sure:.2f}), ale poprawka niepewna: "
                                f"{CORRECTIONS_PL[option]} {option_sure:.2f} "
                                f"< {config.min_correction_confidence:.2f}{suffix}")
        point = correction_target(option, facts)
        if isinstance(point, str):
            return self._review(f"{name}: poprawka '{CORRECTIONS_PL[option]}' niewykonalna: {point}{suffix}")
        sent = self._sent["descent"]
        descent = {"attempt": st.attempt, "chosen_by": self._chosen_by, "ended": sent["ended"],
                   "stop_height": sent["stop_height"], "verdict": verdict.choice, "correction": option}
        try:
            self.retarget(*point)
        except RuntimeError as exc:
            return self._review(f"{name}: poprawka '{CORRECTIONS_PL[option]}' niewykonalna: {exc}{suffix}")
        self._earlier.append(descent)
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

    def _review(self, reason: str) -> Action:
        """Ends the run with the arm holding where it stopped, for a person to judge."""
        self.judging.needs_review = True
        return self._halt(f"{reason} - ramie stoi, do oceny czlowieka", self.kin.from_q(self.q_cmd))

    def _record(self, outcome: Outcome, to_outbox: bool = True) -> None:
        """One `Decision` per question, the answer as received or the error; and a log line."""
        judging, reply, error = self.judging, outcome.reply, outcome.error
        state = {"sent": self._sent, "questions": QUESTIONS_VERSION, "questions_sent": self.questions,
                 "judge": self.judge.name, "model_asked": self.jev_settings.model,
                 "attempt": self._asked_attempt, "ask": self._asks, "geometry": self._geometry,
                 "frames": sorted(self._sent.get("camera_views", {}))}
        entry: dict[str, Any] = {"attempt": self._asked_attempt, "ask": self._asks, "judge": self.judge.name,
                                 "geometry": self._geometry, "response_s": outcome.response_s}
        if reply is not None:
            judging.answers += 1
            judging.model = reply.model
            judging.last_response_s = float("nan") if outcome.response_s is None else outcome.response_s
            judging.last_verdict = reply.answers[TOUCHED].choice
            judging.last_correction = reply.answers[CORRECTION].choice
            records = [Decision(question=question, answer=json.dumps({"model": reply.model, **answer.raw},
                                                                     sort_keys=True),
                                state=state, confidence=answer.confidence, response_s=outcome.response_s)
                       for question, answer in reply.answers.items()]
            answers = {question: {"choice": answer.choice, "confidence": round(answer.confidence, 3)}
                       for question, answer in reply.answers.items()}
            entry.update(model=reply.model, usage=dict(reply.usage), answers=answers)
        else:
            assert error is not None
            judging.failures += 1
            text = json.dumps({"error": error.kind, "status": error.status, "reason": error.reason,
                               "detail": error.detail}, sort_keys=True)
            records = [Decision(question=question, answer=text, state=state, response_s=outcome.response_s)
                       for question in self.questions]
            entry.update(error={"kind": error.kind, "status": error.status, "reason": error.reason})
        judging.decisions.extend(records)
        if to_outbox:
            self._outbox.extend(records)
        logger.info("Ocena dotyku: %s", json.dumps(entry, sort_keys=True))

    def _note(self) -> str:
        text = super()._note()
        if self._summary and self.status.phase in (LIFT, APPROACH, DESCEND):
            return f"{text} | {self._summary}"
        return text

    def stop(self, reason: str) -> None:
        """Also gives up on a pending call; its records go only to `judging.decisions`, as the run's log is
        closed by now. Never waits and never raises."""
        outcome = self._runner.poll()
        if outcome is None and self._runner.pending:
            outcome = Outcome(reply=None, error=JudgeError(f"{self.judge.name}: pytanie odwolane ({reason})",
                                                           kind="cancelled"), response_s=None)
        self._runner.cancel()
        if outcome is not None:
            self._record(outcome, to_outbox=False)
        super().stop(reason)
