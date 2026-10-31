"""The judge seam without a model: answer checks, the decision record, the thread that asks and the
consultation that waits, retries and records one question at a time.

The judges here are fakes at the model's boundary: one answers at once on a clock the test advances, one
blocks until the test lets it go, one raises, and one fails once before it answers.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from soarm_lab.strategies.judges import (
    ASKS_PER_QUESTION,
    RETRY_WAIT_S,
    Consultation,
    Decision,
    JudgeError,
    JudgeRunner,
    JudgeStatus,
    Reply,
    checked_reply,
    choice_confidence,
)

QUESTIONS = {
    "touched": {"type": "choice", "instructions": "Did it touch?",
                "criteria": {"missed_object": "no", "touched_object": "yes", "blocked_above_object": "high"}},
}


def answers(choice: str = "touched_object", confidence: Any = 0.9, **extra: Any) -> dict[str, Any]:
    return {"touched": {"type": "choice", "choice": choice, "confidence": confidence,
                        "probabilities": {"missed_object": 0.05, "touched_object": 0.93,
                                          "blocked_above_object": 0.02}, **extra}}


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class AtOnce:
    """Answers at once, after advancing the clock by `latency_s`."""

    name = "fake"

    def __init__(self, clock: Clock, latency_s: float = 0.0, error: Exception | None = None):
        self.clock, self.latency_s, self.error = clock, latency_s, error
        self.asked: list[Mapping[str, Any]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.asked.append(state)
        self.clock.t += self.latency_s
        if self.error is not None:
            raise self.error
        return checked_reply(self.name, answers(), "fake-1", {"input_tokens": 10}, questions)


class Gated:
    """Answers once the test opens the gate; keeps the cancel event it was given."""

    name = "gated"

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.cancels: list[threading.Event] = []
        self.asked: list[Mapping[str, Any]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.asked.append(state)
        self.cancels.append(cancel)
        # Not asserted: an abandoned call answers late, and the runner drops that answer.
        self.gate.wait(10.0)
        return checked_reply(self.name, answers(), "gated-1", {}, questions)


class Failing:
    """Raises `error` on the first call and answers at once on every later one."""

    name = "jev"

    def __init__(self, clock: Clock, error: JudgeError):
        self.clock, self.error = clock, error
        self.asked: list[Mapping[str, Any]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.asked.append(state)
        if len(self.asked) == 1:
            raise self.error
        return checked_reply(self.name, answers(), "jev-1.14.0", {}, questions)


def eventually(condition: Any, timeout_s: float = 5.0) -> None:
    """Waits until a worker thread has made `condition()` true."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the worker thread never got there"
        time.sleep(0.001)


def errors(decisions: list[Decision]) -> list[str]:
    return [json.loads(d.answer)["error"] for d in decisions]


OVERLOADED = JudgeError("jev chwilowo niedostepny (529)", kind="overloaded", retryable=True, status=529)


# ------------------------------------------------------------ confidence
@pytest.mark.parametrize("probabilities, expected", [
    ((0.6, 0.3, 0.1), 0.4),
    ((0.6, 0.2, 0.2), 0.4),                        # only the top probability counts
    ((1.0, 0.0, 0.0), 1.0),
    ((0.5, 0.5), 0.0),
    ((1.0,), 1.0),
])
def test_choice_confidence_is_typesafes_formula(probabilities: tuple[float, ...], expected: float):
    assert choice_confidence(probabilities) == pytest.approx(expected)


# ------------------------------------------------------------ answer checks
def test_a_valid_answer_keeps_what_was_received():
    usage = {"input_tokens": 428, "output_tokens": 4}
    reply = checked_reply("jev", answers(), "jev-1.14.0", usage, QUESTIONS)

    answer = reply.answers["touched"]
    assert reply.model == "jev-1.14.0" and reply.usage == {"input_tokens": 428, "output_tokens": 4}
    assert answer.choice == "touched_object" and answer.confidence == 0.9
    assert answer.probabilities["touched_object"] == 0.93 and answer.raw["type"] == "choice"


@pytest.mark.parametrize("given, why", [
    (answers(choice="none_of_these"), "spoza opcji"),
    (answers(confidence=True), "pewnosc True"),
    (answers(confidence=math.nan), "pewnosc nan"),
    (answers(confidence="0.9"), "pewnosc '0.9'"),
    (answers(probabilities={"touched_object": math.inf}), "prawdopodobienstwo"),
    (answers(confidence=None, probabilities={}), "brak pewnosci i prawdopodobienstw"),
    ({}, "brak odpowiedzi na 'touched'"),
    (None, "brak odpowiedzi"),
])
def test_an_unusable_answer_is_an_error_never_an_answer(given: Any, why: str):
    with pytest.raises(JudgeError) as caught:
        checked_reply("jev", given, "jev-1.14.0", {}, QUESTIONS)

    assert caught.value.kind == "unusable" and not caught.value.retryable
    assert caught.value.reason.startswith("jev: odpowiedz nie do uzycia") and why in caught.value.reason


def test_confidence_is_clamped_and_filled_from_the_probabilities():
    clamped = checked_reply("jev", answers(confidence=1.2), "", {}, QUESTIONS).answers["touched"]
    filled = checked_reply("jev", answers(confidence=None), "", {}, QUESTIONS).answers["touched"]
    assert clamped.confidence == 1.0
    assert filled.confidence == pytest.approx((0.93 - 1 / 3) / (1 - 1 / 3))


# ------------------------------------------------------------ the record
def test_the_decision_record_has_the_twin_apis_fields():
    """machinekind_twin_api 0.1.3's `Decision`: these names in this order, frozen, compared by identity."""
    names = [f.name for f in dataclasses.fields(Decision)]
    assert names == ["question", "answer", "state", "confidence", "response_s"]
    first, second = Decision("touched", "{}"), Decision("touched", "{}")

    assert first != second and first.state == {} and first.confidence is None and first.response_s is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.answer = "changed"  # type: ignore[misc]


# ------------------------------------------------------------ the runner
def test_without_a_thread_the_outcome_waits_for_the_next_poll_and_times_the_call():
    clock = Clock()
    judge = AtOnce(clock, latency_s=0.25)
    runner = JudgeRunner(judge, threaded=False, clock=clock)
    runner.ask({"attempt": 1}, QUESTIONS, {}, 5.0)

    assert runner.pending and judge.asked == [{"attempt": 1}]
    outcome = runner.poll()
    assert outcome is not None and outcome.error is None and outcome.reply is not None
    assert outcome.response_s == pytest.approx(0.25)
    assert runner.poll() is None and not runner.pending


def test_on_a_thread_poll_returns_nothing_until_the_answer_is_in():
    judge = Gated()
    runner = JudgeRunner(judge)
    runner.ask({}, QUESTIONS, {}, 5.0)

    assert runner.poll() is None and runner.pending
    with pytest.raises(RuntimeError, match="already pending"):
        runner.ask({}, QUESTIONS, {}, 5.0)
    judge.gate.set()
    runner.wait(5.0)
    outcome = runner.poll()
    assert outcome is not None and outcome.reply is not None and outcome.reply.model == "gated-1"


def test_a_cancelled_call_is_told_so_and_its_late_answer_is_dropped():
    judge = Gated()
    runner = JudgeRunner(judge)
    runner.ask({}, QUESTIONS, {}, 5.0)
    deadline = time.monotonic() + 5.0
    while not judge.cancels:
        assert time.monotonic() < deadline, "the worker thread never asked"
        time.sleep(0.001)
    runner.cancel()

    assert judge.cancels[0].is_set() and not runner.pending
    judge.gate.set()
    runner.wait(5.0)
    assert runner.poll() is None


def test_a_judges_own_error_is_kept_and_any_other_becomes_kind_error():
    clock = Clock()
    refused = JudgeError("jev odrzucil pytanie (422: body.state: Field required)", kind="rejected",
                         status=422)
    runner = JudgeRunner(AtOnce(clock, error=refused), threaded=False, clock=clock)
    runner.ask({}, QUESTIONS, {}, 5.0)
    kept = runner.poll()
    assert kept is not None and kept.error is refused

    runner = JudgeRunner(AtOnce(clock, error=ValueError("bad state")), threaded=False, clock=clock)
    runner.ask({}, QUESTIONS, {}, 5.0)
    outcome = runner.poll()
    assert outcome is not None and outcome.reply is None and outcome.error is not None
    assert outcome.error.kind == "error" and not outcome.error.retryable
    assert outcome.error.reason == "blad sedziego: ValueError: bad state"


def test_an_unknown_error_kind_is_a_bug():
    with pytest.raises(ValueError, match="unknown error kind"):
        JudgeError("x", kind="mystery")


# ------------------------------------------------------------ one question at a time
def test_a_consultation_hands_over_the_reply_once_and_records_one_decision_per_question():
    clock = Clock()
    consult = Consultation(AtOnce(clock, latency_s=0.25), threaded=False, clock=clock)
    consult.begin({"attempt": 1}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={"point": "x", "attempt": 1},
                  context="proba 1")

    assert consult.pending and consult.asks == 1
    reply = consult.step(0.05)
    assert isinstance(reply, Reply) and reply.answers["touched"].choice == "touched_object"
    assert not consult.pending and consult.step(0.1) is None and consult.note == ""
    judging = consult.judging
    assert judging.asks == judging.answers == 1 and judging.failures == 0
    assert judging.last_verdict == "touched_object" and judging.last_correction == ""
    assert judging.model == "fake-1" and judging.last_response_s == pytest.approx(0.25)
    [decision] = judging.decisions
    assert set(decision.state) == {"sent", "questions_sent", "judge", "ask", "frames", "point", "attempt"}
    assert decision.state["sent"] == {"attempt": 1} and decision.state["questions_sent"] == QUESTIONS
    assert decision.response_s == pytest.approx(0.25) and decision.confidence == pytest.approx(0.9)
    assert consult.drain() == (decision,) and consult.drain() == ()


def test_a_consultation_times_out_on_the_strategys_clock_asks_once_more_then_fails():
    judge = Gated()
    consult = Consultation(judge)
    try:
        consult.begin({"attempt": 1}, QUESTIONS, {}, now_t=0.0, timeout_s=0.5, meta={}, context="proba 1")
        eventually(lambda: len(judge.cancels) == 1)
        assert consult.step(0.3) is None and consult.note == "gated: czekam na odpowiedz 0.3 s (proba 1)"
        assert consult.step(0.5) is None and judge.cancels[0].is_set()
        assert consult.note == f"gated nie odpowiedzial w 0.5 s, pytam ponownie za {RETRY_WAIT_S:g} s"
        assert consult.step(0.7) is None and consult.note == "gated: pytam ponownie za chwile (proba 1)"
        assert consult.step(1.0) is None and consult.note == "gated: pytam ponownie (proba 1)"
        eventually(lambda: len(judge.cancels) == 2)
        assert consult.step(1.3) is None and consult.note == "gated: czekam na odpowiedz 0.3 s (proba 1)"
        error = consult.step(1.5)
    finally:
        judge.gate.set()

    assert isinstance(error, JudgeError) and error.kind == "timeout"
    assert error.reason == "gated nie odpowiedzial w 0.5 s"
    assert consult.asks == 0 and not consult.pending and ASKS_PER_QUESTION == 2
    assert all(cancel.is_set() for cancel in judge.cancels) and judge.asked[1] == judge.asked[0]
    assert errors(consult.judging.decisions) == ["timeout", "timeout"]
    assert consult.judging.asks == 2 and consult.judging.failures == 2


def test_a_retryable_error_is_asked_again_after_the_pause_and_a_final_one_at_once():
    clock = Clock()
    consult = Consultation(Failing(clock, OVERLOADED), threaded=False, clock=clock)
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")

    assert consult.step(0.05) is None and consult.pending
    assert consult.note == "jev chwilowo niedostepny (529), pytam ponownie za 0.5 s"
    assert consult.step(0.3) is None and consult.note == "jev: pytam ponownie za chwile (proba 1)"
    assert consult.step(0.55) is None and consult.note == "jev: pytam ponownie (proba 1)"
    assert isinstance(consult.step(0.6), Reply) and not consult.pending
    assert consult.judging.failures == 1 and consult.judging.answers == 1 and consult.judging.asks == 2

    refused = JudgeError("jev odrzucil pytanie (422: body.state: Field required)", kind="rejected",
                         status=422)
    consult = Consultation(Failing(clock, refused), threaded=False, clock=clock)
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")
    assert consult.step(0.05) is refused and not consult.pending and consult.judging.asks == 1


def test_a_cancelled_consultation_records_the_cancel_and_drains_nothing():
    judge = Gated()
    consult = Consultation(judge)
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={"point": "close"}, context="proba 1")
    eventually(lambda: len(judge.cancels) == 1)
    consult.cancel("odwolane", to_outbox=False)

    assert judge.cancels[0].is_set() and not consult.pending
    [decision] = consult.judging.decisions
    assert errors([decision]) == ["cancelled"] and json.loads(decision.answer)["reason"] == "odwolane"
    assert decision.state["point"] == "close" and consult.drain() == ()
    judge.gate.set()
    consult.wait(5.0)
    assert consult.step(1.0) is None and len(consult.judging.decisions) == 1
    consult.cancel("again")                                       # nothing pending: no record
    assert len(consult.judging.decisions) == 1


def test_a_cancel_during_the_retry_pause_records_it_and_frees_the_consultation():
    clock = Clock()
    judge = Failing(clock, OVERLOADED)
    consult = Consultation(judge, threaded=False, clock=clock)
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")
    assert consult.step(0.05) is None and consult.pending              # the retry pause: no call in flight
    consult.cancel("odwolane")

    assert not consult.pending and consult.note == ""
    assert errors(consult.judging.decisions) == ["overloaded", "cancelled"]
    assert errors(list(consult.drain())) == ["overloaded", "cancelled"]
    consult.begin({"next": True}, QUESTIONS, {}, now_t=0.1, timeout_s=5.0, meta={}, context="proba 2")
    assert isinstance(consult.step(0.15), Reply) and judge.asked[-1] == {"next": True}


def test_a_consultation_refuses_a_second_question_while_one_is_pending():
    clock = Clock()
    consult = Consultation(AtOnce(clock), threaded=False, clock=clock)
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")

    with pytest.raises(RuntimeError, match="judges: a question is already pending"):
        consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")
    consult.reset()
    assert not consult.pending and consult.judging.decisions == [] and consult.judging.asks == 0
    assert isinstance(consult.judging, JudgeStatus) and consult.judging.judge == "fake"
    consult.begin({}, QUESTIONS, {}, now_t=0.0, timeout_s=5.0, meta={}, context="proba 1")
    assert isinstance(consult.step(0.05), Reply)


def test_the_records_carry_the_meta_and_the_frames_sent(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    clock = Clock()
    consult = Consultation(AtOnce(clock), threaded=False, clock=clock, log_label="Ocena chwytu")
    meta = {"questions": "grasp_jev/1", "model_asked": "jev-1.14.0", "attempt": 2, "point": "lift",
            "geometry": "held"}
    consult.begin({"camera_views": {"wrist": "ref"}}, QUESTIONS, {"ref": np.zeros((300, 300, 3), np.uint8)},
                  now_t=0.0, timeout_s=5.0, meta=meta, context="proba 2")
    consult.step(0.05)

    [decision] = consult.judging.decisions
    assert decision.state["frames"] == ["wrist"] and decision.state["ask"] == 1
    assert {key: decision.state[key] for key in meta} == meta
    [line] = [r.getMessage() for r in caplog.records if r.name == "soarm_lab.strategies.judges"]
    assert line.startswith("Ocena chwytu: ")
    entry = json.loads(line.removeprefix("Ocena chwytu: "))
    assert entry["point"] == "lift" and entry["answers"]["touched"]["choice"] == "touched_object"
