"""touch_jev without the twin and without Jev: the state it sends, the decision rules, the wait and the
records.

The judges are fakes at the model's boundary: `Scripted` answers from a list (replies built through the
same `checked_reply` Jev's are), `Gated` blocks until the test lets it go, and two tests run the real Jev
client on a fake HTTP transport. As in test_touch.py, `follow` plays the twin's supervisor and the arm: the
command goes to each target at once, the arm follows it, and where the TCP would enter an obstacle the arm
stays where it was, which the strategy reads as a stall. test_touch_in_twin.py runs the same strategy
through the twin's runner and physics. The live test runs only with TYPESAFE_API_KEY set.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.jev import KEY_ENV, JevJudge  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.touch import TouchSettings  # noqa: E402
from soarm_lab.strategies.touch_jev import (  # noqa: E402
    CORRECTIONS,
    QUESTIONS_VERSION,
    VERDICTS,
    DescentFacts,
    JevTouchSettings,
    JevTouchStrategy,
    Sighting,
    axes,
    correction_target,
    descent_state,
    touch_questions,
)

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Where the cube lies [m, base frame], and a sighting 3 cm to its left.
XY = (0.22, 0.05)
OFF = (0.22, 0.08)
KEY = "sk-test-SECRET-9d2e"
NEW_FIELDS = ["model", "min_confidence", "min_correction_confidence", "max_attempts", "correction_m",
              "answer_timeout_s", "send_frames"]

Objects = Callable[[float, JevTouchStrategy], dict[str, ObjectPose]]


def run_info() -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits=dict(SPEC.limits),
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)


def obs(joints: Mapping[str, float], t: float = 0.0, objects: Mapping[str, ObjectPose] | None = None,
        command: Mapping[str, float] | None = None, frames: Mapping[str, Frame] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=dict(joints), command=dict(command or {}),
                       objects=dict(objects or {}), frames=dict(frames or {}))


def seen(xy: tuple[float, float] = XY, source: str = "ostatnie widziane", age_s: float = 8.0) -> dict:
    return {"cube": ObjectPose(position=np.array([*xy, 0.0]), rotation=np.eye(3), source=source, age_s=age_s)}


def ageing(xy: tuple[float, float], age_s: float = 8.0) -> Objects:
    """Seen once, `age_s` before the run started: the twin's "ostatnie widziane", ageing from then on."""
    return lambda t, s: seen(xy, age_s=age_s + t)


def truth(xy: tuple[float, float] = XY) -> Objects:
    return lambda t, s: seen(xy, source="symulacja", age_s=0.0)


def then(before: Objects, after: Objects) -> Objects:
    """`before` until the first descent starts, `after` from then on."""
    return lambda t, s: after(t, s) if s.status.attempt >= 1 else before(t, s)


def cube_at(xy: tuple[float, float], top: float, half: float = 0.015) -> Callable[[np.ndarray], bool]:
    return lambda p: bool(np.all(np.abs(p[:2] - xy) < half) and p[2] < top)


def pose_at(point: tuple[float, float, float]) -> dict[str, float]:
    """The joints that put the TCP on `point` [m, base frame], the jaw closed."""
    solution = KIN.ik(np.array(point, float), seed=SPEC.home)
    assert solution.ok
    return dict(solution.joints, gripper=0.0)


def tcp(s: JevTouchStrategy, joints: Mapping[str, float]) -> np.ndarray:
    return s.kin.tcp(dict(joints))[:3, 3]


def follow(s: JevTouchStrategy, objects: Objects | Mapping[str, ObjectPose], ticks: int = 1500,
           joints: Mapping[str, float] | None = None, t0: float = 0.0,
           obstacle: Callable[[np.ndarray], bool] | None = None, frames: Mapping[str, Frame] | None = None,
           until: Callable[[JevTouchStrategy], bool] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor and the arm, ending on `done` or once `until` holds.

    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    joints = dict(joints or SPEC.home)
    command, t = dict(joints), t0
    for _ in range(ticks):
        now = objects(t, s) if callable(objects) else objects
        answers.append(a := s.act(obs(joints, t=t, objects=now, command=command, frames=frames)))
        phases.append(s.status.phase)
        t += 1 / s.hz
        command = {**command, **{name: float(value) for name, value in a.target.items()}}
        if obstacle is None or not obstacle(tcp(s, command)):
            joints = dict(command)
        if a.done or (until is not None and until(s)):
            break
    return answers, phases, joints, t


def runs(phases: list[str]) -> list[str]:
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def raw_answer(choice: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    """A choice answer as Jev sends it, with probabilities that give `confidence`."""
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    probabilities = {option: top if option == choice else (1 - top) / (n - 1) for option in options}
    return {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}


def answered(verdict: str = "touched_object", sure: float = 0.95, correction: str = "aim_at_latest_sighting",
             correction_sure: float = 0.9) -> dict[str, Any]:
    """The JSON body of Jev's answer to both questions."""
    return {"model": "jev-1.14.0", "usage": {"input_tokens": 612, "output_tokens": 4},
            "answers": {"touched": raw_answer(verdict, sure, VERDICTS),
                        "correction": raw_answer(correction, correction_sure, CORRECTIONS)}}


def reply(*args: Any, **kwargs: Any) -> Reply:
    body = answered(*args, **kwargs)
    return checked_reply("jev", body["answers"], body["model"], body["usage"], touch_questions(0.02))


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Scripted:
    """Answers from `outcomes` (a `Reply` or an exception to raise), advancing `clock` by `latency_s`."""

    name = "jev"

    def __init__(self, *outcomes: Reply | Exception, clock: Clock | None = None, latency_s: float = 0.0):
        self.outcomes = list(outcomes)
        self.clock, self.latency_s = clock, latency_s
        self.states: list[Mapping[str, Any]] = []
        self.questions: list[Mapping[str, Any]] = []
        self.images: list[Mapping[str, np.ndarray]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.states.append(state)
        self.questions.append(questions)
        self.images.append(images)
        if self.clock is not None:
            self.clock.t += self.latency_s
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Gated:
    """Answers `outcome` once the test opens the gate; keeps every cancel event it was given."""

    name = "jev"

    def __init__(self, outcome: Reply | None = None):
        self.outcome = outcome or reply()
        self.gate = threading.Event()
        self.cancels: list[threading.Event] = []
        self.states: list[Mapping[str, Any]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.cancels.append(cancel)
        self.states.append(state)
        self.gate.wait(10.0)
        return self.outcome


def eventually(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    """Waits until a worker thread has made `condition` true."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the worker thread never got there"
        time.sleep(0.001)


class FakeTransport:
    """The HTTP boundary: answers each request from `responses`, (status, JSON body) pairs."""

    def __init__(self, *responses: tuple[int, Any]):
        self.responses = list(responses)
        self.bodies: list[bytes] = []

    def __call__(self, url: str, headers: Mapping[str, str], body: bytes,
                 timeout_s: float) -> tuple[int, bytes]:
        self.bodies.append(body)
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode()


def started(judge: Any, threaded: bool = False, clock: Callable[[], float] = time.monotonic,
            joints: Mapping[str, float] | None = None, **settings: Any) -> JevTouchStrategy:
    s = JevTouchStrategy(JevTouchSettings(**settings), judge=judge, threaded=threaded, clock=clock)
    s.start(run_info(), obs(joints or SPEC.home))
    return s


def on_cube(s: JevTouchStrategy, above: float = 0.015) -> Callable[[np.ndarray], bool]:
    return cube_at(XY, s.config.touch_z + above)


def answer_of(decision: Any) -> dict[str, Any]:
    return json.loads(decision.answer)


# ------------------------------------------------------------ the contract
def test_it_follows_the_strategy_protocol_with_a_fake_judge():
    """test_entry_points.py's protocol check, which skips touch_jev for want of a key."""
    assert {f.type for f in dataclasses.fields(JevTouchSettings)} <= {"str", "int", "float", "bool"}
    strategy = JevTouchStrategy(JevTouchSettings(), judge=Scripted())
    assert isinstance(strategy.hz, float) and strategy.hz > 0 and strategy.time_limit_s == 90.0
    strategy.start(run_info(), obs(SPEC.home))
    action = strategy.act(Observation(t=1 / strategy.hz, step=1, joints=dict(SPEC.home), command={}))

    assert isinstance(action, Action)
    assert set(action.target) == set(SPEC.joints) and not action.done
    strategy.stop("zatrzymana")


def test_it_builds_without_a_key_and_refuses_to_start_without_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    for strategy in (JevTouchStrategy(), JevTouchStrategy(judge=JevJudge(environ={}))):
        with pytest.raises(RuntimeError, match="brak klucza jev: ustaw TYPESAFE_API_KEY"):
            strategy.start(run_info(), obs(SPEC.home))
        assert not strategy.started

    strategy = JevTouchStrategy(judge=JevJudge(environ={KEY_ENV: KEY}))
    strategy.start(run_info(), obs(SPEC.home))
    assert strategy.started


def test_its_settings_are_touchs_and_the_judges():
    names = [f.name for f in dataclasses.fields(JevTouchSettings)]

    assert names == [f.name for f in dataclasses.fields(TouchSettings)] + NEW_FIELDS
    defaults = JevTouchSettings()
    assert defaults.time_limit_s == 90.0 and defaults.model == "jev-1.14.0" and defaults.max_attempts == 3
    assert not defaults.send_frames


@pytest.mark.parametrize("settings, message", [
    ({"model": "  "}, "model must not be empty"),
    ({"min_confidence": 1.5}, "min_confidence must lie within 0..1"),
    ({"min_correction_confidence": math.nan}, "min_correction_confidence must lie within 0..1"),
    ({"max_attempts": 0}, "max_attempts must be at least 1"),
    ({"correction_m": 0.0}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"correction_m": 0.01}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"correction_m": 0.2}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"answer_timeout_s": 0.0}, r"answer_timeout_s must lie within \(0, 60\] s"),
    ({"answer_timeout_s": 120.0}, r"answer_timeout_s must lie within \(0, 60\] s"),
])
def test_bad_settings_are_refused(settings: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=f"touch_jev: {message}"):
        JevTouchStrategy(JevTouchSettings(**settings), judge=Scripted())


# ------------------------------------------------------------ a touch, a miss and a correction
def test_a_confident_touch_on_the_cube_is_a_success_recorded_with_both_answers():
    clock = Clock()
    judge = Scripted(reply("touched_object", 0.95), clock=clock, latency_s=0.12)
    s = started(judge, clock=clock, correction_m=0.03)
    answers, phases, *_ = follow(s, seen(), obstacle=on_cube(s))

    last, asking = answers[-1], answers[-2]
    assert runs(phases)[-3:] == ["descend", "touched", "done"]
    assert last.done and last.success and last.note == "jev: obiekt dotkniety (pewnosc 0.95)"
    assert s.status.stopped_because == last.note
    assert asking.note == "jev: pytam o ocene dotyku (proba 1)" and not asking.done
    assert last.target == pytest.approx(asking.target)                     # the arm holds where it stopped
    assert len(judge.states) == 1 and s.judging.asks == 1 and s.judging.answers == 1
    assert s.judging.disagreements == 0 and not s.judging.needs_review

    decisions = s.judging.decisions
    assert [d.question for d in decisions] == ["touched", "correction"]
    assert all(d.response_s == pytest.approx(0.12) for d in decisions)
    assert decisions[0].confidence == pytest.approx(0.95) and decisions[1].confidence == pytest.approx(0.9)
    assert answer_of(decisions[0])["choice"] == "touched_object"
    assert answer_of(decisions[0])["model"] == "jev-1.14.0"
    state = decisions[0].state
    assert state["sent"] is judge.states[0] and state["questions"] == QUESTIONS_VERSION
    assert state["geometry"] == "touched" and state["judge"] == "jev" and state["model_asked"] == "jev-1.14.0"
    assert state["attempt"] == 1 and state["ask"] == 1 and state["frames"] == []
    assert state["questions_sent"] == judge.questions[0] == touch_questions(0.03)
    assert "one step, 3 cm," in json.dumps(state["questions_sent"])
    sent = judge.states[0]
    assert sent["descent"]["ended"] == "stalled_on_something"
    assert sent["descent"]["stop_height"] == "near_object_top"
    asked_at = (len(answers) - 2) / s.hz                     # the pose, taken at t = 0, was then 8 s old
    assert sent["aim"] == {"chosen_by": "first_sighting",
                           "sighting": {"source": "last_seen", "seen_s_ago": round(asked_at + 8)}}
    assert s.drain_decisions() == tuple(decisions) and s.drain_decisions() == ()


def test_after_a_miss_it_aims_at_the_new_sighting_rises_first_and_touches_the_cube():
    judge = Scripted(reply("missed_object", 0.95, "aim_at_latest_sighting", 0.9),
                     reply("touched_object", 0.95))
    s = started(judge)
    start = tcp(s, s.kin.from_q(s.q_cmd))
    answers, phases, *_ = follow(s, then(ageing(OFF), truth(XY)), obstacle=on_cube(s))

    assert runs(phases) == ["approach", "descend", "touched", "lift", "approach", "descend", "touched",
                            "done"]
    assert answers[-1].success and s.status.attempt == 2 and s.status.contact
    assert s.status.touch_point == pytest.approx((*XY, s.config.touch_z))
    retry = phases.index("lift")
    assert answers[retry].note == "jev: pudlo (0.95), poprawka do ostatniego polozenia (0.90) - proba 2"
    assert answers[retry].target == answers[retry - 1].target
    assert answers[retry + 1].note.endswith(" | jev #1: pudlo 0.95, do ostatniego polozenia 0.90")
    points = [start] + [tcp(s, a.target) for a in answers]
    up = [p for p, phase in zip(points[1:], phases, strict=True) if phase == "lift"]
    assert all(np.linalg.norm(p[:2] - OFF) <= 2 * s.ik_tol for p in up)      # straight up, off the cube
    assert len(s.judging.decisions) == 4

    first, second = judge.states
    assert first["descent"]["ended"] == "reached_touch_point_in_free_air"
    assert first["descent"]["stop_height"] == "at_touch_point"
    assert first["latest_sighting"]["source"] == "simulation" and first["latest_sighting"]["new_look"]
    assert first["latest_sighting"]["direction"] == "to_robot_right"
    assert first["latest_sighting"]["offset_mm"]["left"] == pytest.approx(-28, abs=1)
    assert first["earlier_descents"] == [] and first["attempts_left"] == 2
    assert second["aim"]["chosen_by"] == "correction:aim_at_latest_sighting"
    assert second["aim"]["sighting"]["source"] == "simulation"
    assert second["earlier_descents"] == [{"attempt": 1, "chosen_by": "first_sighting",
                                           "ended": "reached_touch_point_in_free_air",
                                           "stop_height": "at_touch_point", "verdict": "missed_object",
                                           "correction": "aim_at_latest_sighting"}]
    assert "aim_at_latest_sighting" in second["would_repeat_earlier_point"]   # the sighting is where it is


@pytest.mark.parametrize("option", ["farther_from_base", "closer_to_base", "to_robot_left", "to_robot_right"])
def test_a_direction_moves_the_touch_point_one_step(option: str):
    judge = Scripted(reply("missed_object", 0.95, option, 0.9), reply("missed_object", 0.95))
    s = started(judge, max_attempts=2)
    answers, *_ = follow(s, ageing(XY))

    out, left = axes(XY)
    step = {"farther_from_base": out, "closer_to_base": -out, "to_robot_left": left, "to_robot_right": -left}
    assert s.status.touch_point == pytest.approx((*(np.array(XY) + 0.02 * step[option]), s.config.touch_z))
    assert answers[-1].note == "jev: pudlo w ostatniej probie (2 z 2, pewnosc 0.95)"
    assert judge.states[1]["aim"] == {"chosen_by": f"correction:{option}", "sighting": None}
    back = {"farther_from_base": "closer_to_base", "closer_to_base": "farther_from_base",
            "to_robot_left": "to_robot_right", "to_robot_right": "to_robot_left"}[option]
    assert judge.states[1]["would_repeat_earlier_point"] == [back]


def test_a_sighting_aimed_at_is_no_new_look_on_the_next_descent():
    """The cameras place the cube 3 cm off during the first descent, then lose it, and the twin keeps that
    same sighting as "ostatnie widziane": the second descent already aimed at it."""
    captured: list[float] = []

    def objects(t: float, s: JevTouchStrategy) -> dict[str, ObjectPose]:
        if s.status.attempt == 0:
            return ageing(XY)(t, s)
        if not captured:
            captured.append(t)
        source = "kamery" if s.status.attempt == 1 else "ostatnie widziane"
        return seen(OFF, source=source, age_s=t - captured[0])

    judge = Scripted(reply("missed_object", 0.95, "aim_at_latest_sighting", 0.9),
                     reply("missed_object", 0.95))
    s = started(judge, max_attempts=2)
    follow(s, objects)

    first, second = judge.states
    assert first["latest_sighting"]["source"] == "cameras" and first["latest_sighting"]["new_look"]
    assert s.status.touch_point == pytest.approx((*OFF, s.config.touch_z))
    assert second["aim"]["sighting"]["source"] == "cameras"
    latest = second["latest_sighting"]
    assert latest["source"] == "last_seen" and not latest["new_look"] and latest["direction"] == "same_place"


def test_a_repeated_stale_sighting_is_no_new_look_and_aiming_at_it_stops_for_review():
    judge = Scripted(reply("missed_object", 0.95, "aim_at_latest_sighting", 0.9))
    s = started(judge)
    answers, *_ = follow(s, ageing(OFF))

    latest = judge.states[0]["latest_sighting"]
    assert latest["source"] == "last_seen" and not latest["new_look"] and latest["direction"] == "same_place"
    assert answers[-1].done and not answers[-1].success and s.judging.needs_review
    assert answers[-1].note == ("jev: poprawka 'do ostatniego polozenia' niewykonalna: brak nowego polozenia "
                                "z kamer - ramie stoi, do oceny czlowieka")


# ------------------------------------------------------------ when the run ends
@pytest.mark.parametrize("verdict, above, said", [
    ("touched_object", 0.015, "dotkniety"),
    ("missed_object", None, "pudlo"),
    ("blocked_above_object", 0.05, "zablokowany nad obiektem"),
])
def test_an_unsure_verdict_stops_the_arm_where_it_is_for_a_person(verdict: str, above: float | None,
                                                                   said: str):
    """Whatever the verdict, even a miss with a confident correction: the geometry agrees each time."""
    judge = Scripted(reply(verdict, 0.5, "to_robot_left", 0.9))
    s = started(judge)
    answers, *_ = follow(s, seen(), obstacle=None if above is None else on_cube(s, above=above))

    last = answers[-1]
    assert last.done and not last.success and s.judging.needs_review
    assert last.note == f"jev niepewny: {said} z pewnoscia 0.50 < 0.80 - ramie stoi, do oceny czlowieka"
    assert last.target == pytest.approx(answers[-2].target)
    assert len(judge.states) == 1 and s.status.attempt == 1
    assert s.status.touch_point == pytest.approx((*XY, s.config.touch_z))


def test_answers_exactly_at_the_thresholds_are_acted_on():
    judge = Scripted(reply("missed_object", 0.8, "to_robot_left", 0.6), reply("missed_object", 0.8))
    s = started(judge, max_attempts=2)
    answers, *_ = follow(s, ageing(XY))

    _, left = axes(XY)
    assert not s.judging.needs_review and s.status.attempt == 2 and len(judge.states) == 2
    assert s.status.touch_point == pytest.approx((*(np.array(XY) + 0.02 * left), s.config.touch_z))
    assert answers[-1].note == "jev: pudlo w ostatniej probie (2 z 2, pewnosc 0.80)"


def test_an_unsure_correction_stops_for_review():
    s = started(Scripted(reply("missed_object", 0.95, "to_robot_left", 0.4)))
    answers, *_ = follow(s, ageing(XY))

    assert s.judging.needs_review and s.status.attempt == 1
    assert answers[-1].note == ("jev: pudlo (0.95), ale poprawka niepewna: w lewo 0.40 < 0.60 - ramie stoi, "
                                "do oceny czlowieka")


def test_a_miss_on_the_last_attempt_ends_the_run_without_review():
    s = started(Scripted(reply("missed_object", 0.95)), max_attempts=1)
    answers, *_ = follow(s, ageing(XY))

    assert answers[-1].done and not answers[-1].success and not s.judging.needs_review
    assert answers[-1].note == "jev: pudlo w ostatniej probie (1 z 1, pewnosc 0.95)"


def test_a_block_above_the_object_ends_the_run_without_a_retry():
    judge = Scripted(reply("blocked_above_object", 0.9))
    s = started(judge)
    answers, *_ = follow(s, seen(), obstacle=on_cube(s, above=0.05))

    assert answers[-1].note == "jev: ramie zablokowane nad obiektem (pewnosc 0.90)"
    assert len(judge.states) == 1 and judge.states[0]["descent"]["stop_height"] == "far_above_object_top"


def test_a_correction_the_arm_cannot_take_stops_for_review_and_changes_nothing():
    """At 46.5 cm the arm still reaches the touch point; 2 cm farther it does not."""
    s = started(Scripted(reply("missed_object", 0.95, "farther_from_base", 0.9)))
    answers, *_ = follow(s, ageing((0.465, 0.0)))

    assert s.judging.needs_review and s.status.touch_point == pytest.approx((0.465, 0.0, s.config.touch_z))
    assert answers[-1].note.startswith("jev: poprawka 'dalej od podstawy' niewykonalna: punkt (0.485, 0.000")
    assert answers[-1].note.endswith("poza zasiegiem ramienia - ramie stoi, do oceny czlowieka")


def test_the_model_decides_alone_and_a_disagreement_with_the_geometry_is_said():
    s = started(Scripted(reply("touched_object", 0.9)))
    answers, *_ = follow(s, ageing(XY))

    assert answers[-1].success and s.judging.disagreements == 1
    assert answers[-1].note == "jev: obiekt dotkniety (pewnosc 0.90); geometria: brak kontaktu"


@pytest.mark.parametrize("hover_m, kind", [(0.06, "blocked"), (0.025, "stuck")])
def test_an_arm_that_never_came_down_is_not_asked_about(hover_m: float, kind: str):
    """Frozen readings on the hover point: the stall is no touch, and touch's default ends the run. With the
    default hover the stall lies above the contact band, which touch calls a block."""
    frozen = pose_at((*XY, JevTouchSettings().touch_z + hover_m))
    s = started(Scripted(), joints=frozen, hover_m=hover_m)
    answers = []
    command = dict(frozen)
    for k in range(400):
        answers.append(s.act(obs(frozen, t=k / s.hz, objects=seen(), command=command)))
        command = {**command, **answers[-1].target}
        if answers[-1].done:
            break

    assert answers[-1].done and not answers[-1].success and s.status.drop_m < s.config.contact_gap_m
    assert s.descent_verdict() == (kind, answers[-1].note)
    assert s.judging.asks == 0 and s.judging.decisions == []


# ------------------------------------------------------------ waiting for the answer
def test_while_the_answer_is_pending_the_arm_holds_and_the_next_tick_after_it_decides():
    judge = Gated()
    s = started(judge, threaded=True)
    answers, _, joints, t = follow(s, seen(), obstacle=on_cube(s), until=lambda s: s.judging.asks == 1)
    asking = answers[-1]
    command = dict(joints)

    pending = []
    for k in range(10):
        pending.append(s.act(obs(joints, t=t + k / s.hz, objects=seen(), command=command)))
    assert all(not a.done and a.target == asking.target for a in pending)
    assert all(a.note.startswith("jev: czekam na odpowiedz") for a in pending)
    eventually(lambda: len(judge.cancels) == 1)

    judge.gate.set()
    s.wait_for_answer(5.0)
    last = s.act(obs(joints, t=t + 0.5, objects=seen(), command=command))
    assert last.done and last.success and len(judge.cancels) == 1


def test_a_judge_that_never_answers_is_asked_once_more_then_the_run_stops_for_review():
    judge = Gated()
    s = started(judge, threaded=True, answer_timeout_s=0.5)
    try:
        answers, *_ = follow(s, seen(), obstacle=on_cube(s))
    finally:
        judge.gate.set()
    s.wait_for_answer(5.0)
    eventually(lambda: len(judge.cancels) == 2)

    asks = ("jev: pytam o ocene", "jev: pytam ponownie (")
    asked = [k for k, a in enumerate(answers) if a.note.startswith(asks)]
    assert len(asked) == 2 and len(judge.cancels) == 2 and all(c.is_set() for c in judge.cancels)
    assert (asked[1] - asked[0]) / s.hz >= 1.0 - 1e-9                      # 0.5 s timeout, then 0.5 s pause
    assert (len(answers) - 1 - asked[1]) / s.hz >= 0.5 - 1e-9              # counted from the second ask
    assert len(judge.states) == 2 and judge.states[1] == judge.states[0]
    assert answers[-1].note == "jev nie odpowiedzial w 0.5 s - ramie stoi, do oceny czlowieka"
    decisions = s.judging.decisions
    assert len(decisions) == 4 and all(d.response_s is None and d.confidence is None for d in decisions)
    assert {answer_of(d)["error"] for d in decisions} == {"timeout"}
    assert s.judging.decisions == decisions and len(s.drain_decisions()) == 4      # late answers add nothing


def test_the_lag_guard_still_ends_the_run_while_the_answer_is_pending():
    judge = Gated()
    s = started(judge, threaded=True)
    try:
        _, _, joints, t = follow(s, seen(), obstacle=on_cube(s), until=lambda s: s.judging.asks == 1)
        pushed = {**joints, "shoulder_pan": joints["shoulder_pan"] + 30.0}
        with pytest.raises(RuntimeError, match="ramie nie nadaza za celem"):
            s.act(obs(pushed, t=t, objects=seen(), command=joints))
        s.stop(f"RuntimeError: {s.status.stopped_because}")
    finally:
        judge.gate.set()

    assert s.status.stopped_because.startswith("ramie nie nadaza za celem")
    assert [answer_of(d)["error"] for d in s.judging.decisions] == ["cancelled", "cancelled"]


def test_a_stop_while_pending_cancels_the_call_and_a_new_start_begins_afresh():
    judge = Gated()
    s = started(judge, threaded=True)
    follow(s, seen(), obstacle=on_cube(s), until=lambda s: s.judging.asks == 1)
    eventually(lambda: len(judge.cancels) == 1)
    began = time.monotonic()
    s.stop("zatrzymana z panelu")

    assert time.monotonic() - began < 0.5 and judge.cancels[0].is_set()
    assert s.status.stopped_because == "zatrzymana z panelu"
    assert [answer_of(d)["error"] for d in s.judging.decisions] == ["cancelled", "cancelled"]
    assert s.drain_decisions() == ()                                   # the run's log is closed by now
    judge.gate.set()
    s.wait_for_answer(5.0)
    assert len(s.judging.decisions) == 2

    s.start(run_info(), obs(SPEC.home))
    assert s.judging.decisions == [] and s.judging.asks == 0 and s.status.phase == "wait"


# ------------------------------------------------------------ the Jev client in the loop
def test_an_overloaded_jev_is_asked_again_with_the_same_state():
    transport = FakeTransport((529, {"detail": "Overloaded"}), (200, answered("touched_object", 0.95)))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, *_ = follow(s, seen(), obstacle=on_cube(s))

    assert answers[-1].success and s.judging.asks == 2 and s.judging.failures == 1
    assert any(a.note == "jev chwilowo niedostepny (529), pytam ponownie za 0.5 s" for a in answers)
    assert transport.bodies[0] == transport.bodies[1]
    sent = json.loads(transport.bodies[0])
    assert sent["model"] == "jev-1.14.0" and sent["questions"] == s.questions
    decisions = s.judging.decisions
    assert [answer_of(d).get("error") for d in decisions] == ["overloaded", "overloaded", None, None]
    assert answer_of(decisions[0])["status"] == 529 and decisions[2].confidence == pytest.approx(0.95)


def test_a_refused_request_ends_the_run_at_once_for_review():
    refused = {"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}
    transport = FakeTransport((422, refused))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, *_ = follow(s, seen(), obstacle=on_cube(s))

    assert len(transport.bodies) == 1 and s.judging.needs_review
    assert answers[-1].note == ("jev odrzucil pytanie (422: body.state: Field required) - ramie stoi, "
                                "do oceny czlowieka")


def test_an_error_inside_the_judge_ends_the_run_honestly():
    s = started(Scripted(KeyError("touched")))
    answers, *_ = follow(s, seen(), obstacle=on_cube(s))

    assert answers[-1].done and not answers[-1].success and s.judging.needs_review
    assert answers[-1].note == "blad sedziego: KeyError: 'touched' - ramie stoi, do oceny czlowieka"


def test_the_key_appears_in_no_record_note_or_log_line(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    transport = FakeTransport((503, {"detail": f"echo {KEY}"}), (200, answered("missed_object", 0.95)),
                              (401, {"detail": f"Invalid API key {KEY}"}))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, *_ = follow(s, then(ageing(OFF), truth(XY)))

    assert answers[-1].note.startswith(f"jev odrzucil klucz {KEY_ENV} (401)")
    seen_text = json.dumps([[d.question, d.answer, d.state] for d in s.judging.decisions])
    seen_text += " ".join(a.note for a in answers) + s.status.stopped_because + caplog.text + repr(s.judge)
    assert KEY not in seen_text and "Bearer" not in seen_text
    assert all(f"Bearer {KEY}".encode() not in body for body in transport.bodies)


# ------------------------------------------------------------ frames
def test_with_frames_on_each_usable_frame_goes_with_the_question():
    image = np.full((300, 400, 3), 120, np.uint8)
    frames = {"front": Frame(image=image, age_s=0.05),
              "wrist": Frame(image=np.zeros((120, 160, 3), np.uint8), age_s=0.05)}
    judge = Scripted(reply("touched_object", 0.95))
    s = started(judge, send_frames=True)
    follow(s, seen(), obstacle=on_cube(s), frames=frames)

    sent, images = judge.states[0], judge.images[0]
    reference = sent["camera_views"]["front"]
    assert list(images) == [reference] and images[reference].tolist() == image.tolist()
    assert images[reference] is not image
    assert sent["frames_left_out"] == {"wrist": "160x120 px, poza 280..1568"}
    assert s.judging.decisions[0].state["frames"] == ["front"]
    assert "camera_views" in json.dumps(s.questions["touched"])


# ------------------------------------------------------------ the request, without the arm
def test_the_questions_are_typesafe_choices_with_their_options_in_order():
    """Literal names, as api.typesafe.ai reads them: a wrong key or type is a 422 on every descent, and the
    order sets which way Jev's lean toward the first option errs."""
    questions = touch_questions(0.02)

    assert list(questions) == ["touched", "correction"]
    for question in questions.values():
        assert set(question) == {"type", "instructions", "criteria"} and question["type"] == "choice"
        assert 1 <= len(question["criteria"]) <= 255
    assert list(questions["touched"]["criteria"]) == ["missed_object", "touched_object",
                                                      "blocked_above_object"]
    assert list(questions["correction"]["criteria"]) == ["aim_at_latest_sighting", "farther_from_base",
                                                         "closer_to_base", "to_robot_left", "to_robot_right"]


# ------------------------------------------------------------ the state, without the arm
def facts(**changes: Any) -> DescentFacts:
    base = DescentFacts(label="cube", attempt=1, max_attempts=3, contact=False, drop_m=0.06,
                        stop_above_m=0.0004, rise_m=0.0002, contact_band_m=0.03, touch_xy=(0.2, 0.0),
                        correction_m=0.02, now_t=20.0, newest_used_seen_t=5.0,
                        aim_sighting=Sighting("ostatnie widziane", 5.0, (0.2, 0.0)))
    return dataclasses.replace(base, **changes)


def test_the_state_is_json_with_finite_numbers_and_the_same_keys_every_time():
    plain = descent_state(facts(drop_m=math.nan, rise_m=math.inf))
    with_sighting = descent_state(facts(latest=Sighting("kamery", 19.5, (0.2, -0.03))))

    for state in (plain, with_sighting):
        json.dumps(state, allow_nan=False)
        assert list(state) == ["task", "attempt", "attempts_left", "descent", "aim", "latest_sighting",
                               "earlier_descents", "would_repeat_earlier_point"]
    assert plain["descent"]["drop_from_hover_mm"] is None and plain["descent"]["pushed_back_mm"] is None
    assert plain["latest_sighting"] is None
    assert with_sighting["latest_sighting"] == {"source": "cameras", "seen_s_ago": 0, "new_look": True,
                                                "direction": "to_robot_right",
                                                "offset_mm": {"farther": 0, "left": -30}}
    assert with_sighting["aim"] == {"chosen_by": "first_sighting",
                                    "sighting": {"source": "last_seen", "seen_s_ago": 15}}
    framed = descent_state(facts(with_frames=True, camera_views={"a": "ref"}))
    assert framed["camera_views"] == {"a": "ref"} and framed["frames_left_out"] == {}


@pytest.mark.parametrize("contact, above, height", [
    (False, 0.0, "at_touch_point"),
    (True, 0.012, "near_object_top"),
    (True, 0.03, "near_object_top"),
    (True, 0.031, "far_above_object_top"),
])
def test_the_stop_height_is_a_named_bucket(contact: bool, above: float, height: str):
    state = descent_state(facts(contact=contact, stop_above_m=above))

    assert state["descent"]["stop_height"] == height
    ended = "stalled_on_something" if contact else "reached_touch_point_in_free_air"
    assert state["descent"]["ended"] == ended


@pytest.mark.parametrize("offset, expected", [
    ((0.02, 0.0), "farther_from_base"),
    ((-0.02, 0.0), "closer_to_base"),
    ((0.0, 0.02), "to_robot_left"),
    ((0.0, -0.02), "to_robot_right"),
    ((0.006, 0.006), "same_place"),
])
def test_the_direction_is_seen_from_the_base(offset: tuple[float, float], expected: str):
    latest = Sighting("kamery", 19.0, (0.2 + offset[0], offset[1]))

    assert descent_state(facts(latest=latest))["latest_sighting"]["direction"] == expected


def test_corrections_never_aim_at_a_point_already_tried():
    tried = facts(tried_xy=((0.2, 0.02),), latest=Sighting("kamery", 19.0, (0.205, 0.0)))

    assert descent_state(tried)["would_repeat_earlier_point"] == ["aim_at_latest_sighting", "to_robot_left"]
    assert correction_target("to_robot_left", tried) == "punkt juz probowany"
    assert correction_target("to_robot_right", tried) == pytest.approx((0.2, -0.02))
    assert correction_target("aim_at_latest_sighting", facts()) == "brak nowego polozenia z kamer"


def test_a_sighting_seen_no_later_than_the_aim_is_no_new_look():
    old = facts(latest=Sighting("ostatnie widziane", 5.3, (0.2, 0.03)))

    assert not descent_state(old)["latest_sighting"]["new_look"]
    assert correction_target("aim_at_latest_sighting", old) == "brak nowego polozenia z kamer"


# ------------------------------------------------------------ live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"),
                    reason="a live call to Jev needs TYPESAFE_API_KEY")
def test_jev_answers_the_touch_questions_live():
    judge = JevJudge()
    judge.check()
    questions = touch_questions(0.02)
    answer = judge.ask(descent_state(facts()), questions, images={}, timeout_s=10.0, cancel=threading.Event())

    assert answer.model.startswith("jev-")
    for name, question in questions.items():
        assert answer.answers[name].choice in question["criteria"]
        assert 0.0 <= answer.answers[name].confidence <= 1.0
