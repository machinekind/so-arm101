"""The judge seam without a model: answer checks, the decision record and the thread that asks.

The judges here are fakes at the model's boundary: one answers at once on a clock the test advances, one
blocks until the test lets it go, one raises.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from soarm_lab.strategies.judges import (
    Decision,
    JudgeError,
    JudgeRunner,
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

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.cancels.append(cancel)
        assert self.gate.wait(5.0)
        return checked_reply(self.name, answers(), "gated-1", {}, questions)


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
