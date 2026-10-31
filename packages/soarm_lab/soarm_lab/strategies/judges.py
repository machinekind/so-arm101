"""Judges: models a strategy asks about its run, and the thread that keeps the tick from waiting on them.

A judge takes the run's state as JSON values and named questions, and answers each question with one of its
options. The questions use TypeSafe's JSON shape, so the Jev client sends them as they are and another judge
can read them too: each is `{"type": "choice", "instructions": ..., "criteria": {option: description}}`,
with the options in the order they are sent. Only choice questions are used so far. `checked_reply` holds the
checks every judge's answer goes through, so a failed or malformed call is never read as an answer.

`JudgeRunner` asks on a thread of its own, one question at a time, as the twin's `DetectorRunner` and the
`dream` strategy do: the strategy polls it every tick and never waits. The worker gets only ready JSON
values, the questions and copies of the frames; it never touches the strategy's kinematics, whose MuJoCo
data is shared. A cancelled or superseded question's late answer is dropped.

`Decision` records one answer for the run's log. It has exactly the fields of
`machinekind_twin_api.strategy.Decision` (api 0.1.3). The pinned api (0.1.2) has no such class, so this one
stands in for it until the pin moves.

`Consultation` is one strategy's question at a time on top of the runner: the call, the wait counted on
the strategy's clock, one retry after a timeout, an overload or a dropped connection, and one `Decision` per
question of every call, cancelled and failed ones too. It is the flow `touch_jev` wrote for its descents;
`grasp_jev` asks through it, and `touch_jev` keeps its private copy for now. `JudgeStatus` is the summary
both keep for the panel's logs and tests.

Nothing here knows Jev: `jev.JevJudge` is one judge. A judge that runs the `claude` CLI headless would be
another, behind the same `Judge` protocol.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

#: Questions by name, in TypeSafe's JSON shape.
Questions = Mapping[str, Mapping[str, Any]]

#: Why a call gave no answer (`JudgeError.kind`).
ERROR_KINDS = ("timeout", "unreachable", "key_rejected", "rejected", "overloaded", "unusable", "error",
               "cancelled")
#: The longest technical detail an error keeps.
DETAIL_LIMIT = 300
#: Calls per question (`Consultation`): the first and one retry after a timeout, an overload or a dropped
#: connection.
ASKS_PER_QUESTION = 2
#: The pause before that retry [s].
RETRY_WAIT_S = 0.5


@dataclass(frozen=True)
class ChoiceAnswer:
    """One choice question's answer."""

    #: The option the judge picked, one of the question's criteria.
    choice: str
    #: How concentrated the answer's probabilities are, 0..1; not the chance that it is right.
    confidence: float
    #: Each option's probability, as the judge gave them.
    probabilities: Mapping[str, float]
    #: The answer as received.
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class Reply:
    """A judge's answers to one call."""

    #: The model that answered, which may differ from the one asked for (an alias).
    model: str
    answers: Mapping[str, ChoiceAnswer]
    #: Token counts or the like, as the judge reports them.
    usage: Mapping[str, int] = field(default_factory=dict)


class JudgeError(RuntimeError):
    """A call that gave no answer. `reason` is for the panel (Polish, without diacritics); `detail` is
    technical. Neither ever holds a key."""

    def __init__(self, reason: str, *, kind: str, retryable: bool = False, status: int | None = None,
                 detail: str = ""):
        super().__init__(reason)
        if kind not in ERROR_KINDS:
            raise ValueError(f"judges: unknown error kind {kind!r}")
        self.reason = reason
        self.kind = kind
        #: Worth asking again after a pause: a timeout, an unreachable or overloaded service.
        self.retryable = retryable
        #: The HTTP status, when there was one.
        self.status = status
        self.detail = detail[:DETAIL_LIMIT]


class Judge(Protocol):
    """A model that answers questions about a run's state."""

    #: A short name for notes and records, e.g. "jev".
    name: str

    def check(self) -> None:
        """Raises RuntimeError, with a reason for the panel, when the judge cannot be asked (no key, no
        program). Uses no network: `start` calls it before the arm moves."""

    def ask(self, state: Mapping[str, Any], questions: Questions, *, images: Mapping[str, np.ndarray],
            timeout_s: float, cancel: threading.Event) -> Reply:
        """Answers `questions` about `state`, blocking; runs on the worker thread.

        `images` are RGB frames by the reference `state` names them with. Raises `JudgeError` when there is
        no usable answer; any other exception is a bug the runner records as kind "error".
        """
        ...


def choice_confidence(probabilities: Sequence[float]) -> float:
    """TypeSafe's confidence of a choice: (p_max - 1/n) / (1 - 1/n), 1 when all probability is on one
    option and 0 when it is spread evenly. For a judge that gives probabilities only."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    value = (max(probabilities) - 1.0 / n) / (1.0 - 1.0 / n)
    return min(max(value, 0.0), 1.0)


def checked_reply(judge: str, answers: Any, model: Any, usage: Any, questions: Questions) -> Reply:
    """A judge's raw answers (JSON values) as a `Reply`, or `JudgeError` of kind "unusable".

    Every question needs an answer whose choice is one of its options. Probabilities and confidence must be
    finite numbers, never bools; the confidence is clamped to 0..1, and computed from the probabilities
    (`choice_confidence`) when the answer has none.
    """
    if not isinstance(answers, Mapping):
        raise unusable(judge, "brak odpowiedzi")
    checked = {}
    for name, question in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, Mapping):
            raise unusable(judge, f"brak odpowiedzi na '{name}'")
        options = [str(option) for option in question.get("criteria", {})]
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in options:
            raise unusable(judge, f"'{name}': wybor {choice!r} spoza opcji")
        given = answer.get("probabilities", {})
        if not isinstance(given, Mapping):
            raise unusable(judge, f"'{name}': prawdopodobienstwa to nie slownik")
        probabilities = {}
        for option, value in given.items():
            number = finite_number(value)
            if number is None:
                raise unusable(judge, f"'{name}': prawdopodobienstwo {option!r} = {value!r}")
            probabilities[str(option)] = number
        if answer.get("confidence") is None:
            if not probabilities:
                raise unusable(judge, f"'{name}': brak pewnosci i prawdopodobienstw")
            confidence = choice_confidence([probabilities.get(option, 0.0) for option in options])
        else:
            confidence = finite_number(answer["confidence"])
            if confidence is None:
                raise unusable(judge, f"'{name}': pewnosc {answer['confidence']!r}")
        checked[name] = ChoiceAnswer(choice=choice, confidence=min(max(confidence, 0.0), 1.0),
                                     probabilities=probabilities, raw=dict(answer))
    counts = {str(key): value for key, value in usage.items()} if isinstance(usage, Mapping) else {}
    return Reply(model=str(model or ""), answers=checked,
                 usage={key: int(value) for key, value in counts.items() if finite_number(value) is not None})


def finite_number(value: Any) -> float | None:
    """`value` as a float when it is a finite int or float (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def unusable(judge: str, detail: str) -> JudgeError:
    return JudgeError(f"{judge}: odpowiedz nie do uzycia ({detail})", kind="unusable", detail=detail)


@dataclass(frozen=True, eq=False)
class Decision:
    """One question a strategy put to a model, with its answer, for the run's log. It never reaches the arm.

    A stand-in for `machinekind_twin_api.strategy.Decision` (api 0.1.3), with its fields in its order; it
    compares by identity, as that class does.
    """

    #: What the strategy asked: the prompt, or its name when the prompt is long.
    question: str
    #: The model's answer as received: text, or JSON as text.
    answer: str
    #: What was sent with the question besides the frames, as JSON values.
    state: Mapping[str, Any] = field(default_factory=dict)
    #: The model's confidence, 0..1, when it gives one.
    confidence: float | None = None
    #: Seconds from asking to the answer.
    response_s: float | None = None


@dataclass(frozen=True)
class Outcome:
    """How one call ended: a reply or an error, and how long it took on the runner's clock [s]."""

    reply: Reply | None
    error: JudgeError | None
    #: None when the strategy gave up waiting (a timeout) or the call was cancelled.
    response_s: float | None


@dataclass(eq=False)
class _Ticket:
    """One call: only the current ticket's outcome is kept."""

    cancel: threading.Event = field(default_factory=threading.Event)


class JudgeRunner:
    """Asks a judge on a thread of its own, one call at a time; the tick collects the outcome with `poll`.

    `threaded=False` runs the call inline in `ask`, for tests on a simulated clock. Either way the outcome
    waits for the next `poll`.
    """

    def __init__(self, judge: Judge, *, threaded: bool = True, clock: Callable[[], float] = time.monotonic):
        self.judge = judge
        self.threaded = threaded
        self._clock = clock
        self._lock = threading.Lock()
        self._ticket: _Ticket | None = None
        self._outcome: Outcome | None = None
        self._thread: threading.Thread | None = None

    @property
    def pending(self) -> bool:
        """Whether a call was made and its outcome not yet collected."""
        with self._lock:
            return self._ticket is not None

    def ask(self, state: Mapping[str, Any], questions: Questions, images: Mapping[str, np.ndarray],
            timeout_s: float) -> None:
        """Starts a call. Raises RuntimeError while another one is pending."""
        with self._lock:
            if self._ticket is not None:
                raise RuntimeError("judges: a call is already pending")
            ticket = self._ticket = _Ticket()
            self._outcome = None
        if not self.threaded:
            self._run(ticket, state, questions, images, timeout_s)
            return
        self._thread = threading.Thread(target=self._run, args=(ticket, state, questions, images, timeout_s),
                                        name=f"{self.judge.name}-ask", daemon=True)
        self._thread.start()

    def _run(self, ticket: _Ticket, state: Mapping[str, Any], questions: Questions,
             images: Mapping[str, np.ndarray], timeout_s: float) -> None:
        t0 = self._clock()
        reply, error = None, None
        try:
            reply = self.judge.ask(state, questions, images=images, timeout_s=timeout_s, cancel=ticket.cancel)
        except JudgeError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 - sorted on the tick thread
            reason = f"blad sedziego: {type(exc).__name__}: {exc}"[:DETAIL_LIMIT]
            error = JudgeError(reason, kind="error", detail=reason)
        outcome = Outcome(reply=reply, error=error, response_s=self._clock() - t0)
        with self._lock:
            if ticket is self._ticket:
                self._outcome = outcome

    def poll(self) -> Outcome | None:
        """The finished call's outcome, once; None while it runs or when nothing was asked."""
        with self._lock:
            outcome, self._outcome = self._outcome, None
            if outcome is not None:
                self._ticket = None
        return outcome

    def cancel(self) -> None:
        """Gives up on the pending call: its cancel event is set and its outcome will be dropped. Never
        waits and never raises."""
        with self._lock:
            ticket, self._ticket = self._ticket, None
            self._outcome = None
        if ticket is not None:
            ticket.cancel.set()

    def wait(self, timeout_s: float | None = None) -> None:
        """Waits for the last call's thread, if any (tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)


@dataclass
class JudgeStatus:
    """What the decision step did in this run, for the panel's logs and tests.

    `touch_jev` keeps one for its descents and `Consultation` one for its questions.
    """

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


class Consultation:
    """One question to a judge at a time, for one strategy: the call, the wait, one retry and the records.

    The flow is `touch_jev`'s (its `after_touch`, `_waiting`, `_failed` and `_record`), with the same notes.
    `begin` asks on `JudgeRunner`'s thread (inline with `threaded=False`) and the strategy calls `step` on
    every tick with its own clock, holding the arm meanwhile. `step` returns None while the answer is
    pending, with the tick's note in `note`; then the `Reply`, or the final `JudgeError` after the retry,
    once. The wait is counted on the strategy's clock from the ask: a call that does not answer within
    `timeout_s` is given up. A timeout, an overload or a dropped connection is asked again once, after
    `RETRY_WAIT_S`, with the same state; any other error is final. Every call gives one `Decision` per
    question in `judging.decisions` and in `drain`, and one log line headed `log_label`.

    `cancel` gives up on a question in flight, recorded as cancelled, and also drops a retry pause, which
    touch_jev's own `stop` does not; after it `begin` is always accepted.
    """

    def __init__(self, judge: Judge, *, threaded: bool = True, clock: Callable[[], float] = time.monotonic,
                 log_label: str = "Ocena"):
        self.judge = judge
        self.log_label = log_label
        self._threaded = threaded
        self._clock = clock
        self._runner = JudgeRunner(judge, threaded=threaded, clock=clock)
        self.judging = JudgeStatus(judge=judge.name)
        #: The tick's note while a question is pending; empty otherwise.
        self.note = ""
        self._outbox: list[Decision] = []
        self._clear()

    def reset(self) -> None:
        """A new run: gives up on a pending call without a record and starts the records afresh."""
        self._runner.cancel()
        self._runner = JudgeRunner(self.judge, threaded=self._threaded, clock=self._clock)
        self.judging = JudgeStatus(judge=self.judge.name)
        self._outbox = []
        self._clear()

    def _clear(self) -> None:
        self._state: Mapping[str, Any] = {}
        self._questions: Questions = {}
        self._images: Mapping[str, np.ndarray] = {}
        self._meta: dict[str, Any] = {}
        self._context = ""
        self._asks = 0
        self._asked_t = math.nan
        self._retry_at: float | None = None
        self._timeout_s = 0.0
        self._open = False
        self.note = ""

    @property
    def pending(self) -> bool:
        """Whether a question is open: a call in flight or a retry pause, from `begin` until `step` hands
        over its final outcome or `cancel` or `reset` runs."""
        return self._open

    @property
    def asks(self) -> int:
        """Calls made for the current question."""
        return self._asks

    def begin(self, state: Mapping[str, Any], questions: Questions, images: Mapping[str, np.ndarray], *,
              now_t: float, timeout_s: float, meta: Mapping[str, Any], context: str) -> None:
        """Asks `questions` about `state`, with `images` by the references `state` names them with.

        `now_t` is the strategy's clock [s], `timeout_s` the longest wait for one answer [s]. `meta` goes
        into every record's state and log line as it is (JSON values); `context` ends the notes, e.g.
        "proba 2". Raises RuntimeError while a question is pending.
        """
        if self._open:
            raise RuntimeError("judges: a question is already pending")
        self._state, self._questions, self._images = state, questions, images
        self._meta, self._context, self._timeout_s = dict(meta), context, float(timeout_s)
        self._open, self._asks, self.note = True, 0, ""
        self._ask(now_t)

    def _ask(self, now_t: float) -> None:
        self._asks += 1
        self._asked_t = float(now_t)
        self._retry_at = None
        self.judging.asks += 1
        self._runner.ask(self._state, self._questions, self._images, self._timeout_s)

    def step(self, now_t: float) -> Reply | JudgeError | None:
        """The pending question on this tick: None while it waits (the note says why), else its reply or
        its final error, handed over once."""
        if not self._open:
            self.note = ""
            return None
        name, context = self.judge.name, self._context
        if self._retry_at is not None:
            if now_t >= self._retry_at:
                self._ask(now_t)
                self.note = f"{name}: pytam ponownie ({context})"
            else:
                self.note = f"{name}: pytam ponownie za chwile ({context})"
            return None
        outcome = self._runner.poll()
        if outcome is None:
            waited = float(now_t) - self._asked_t
            if waited < self._timeout_s:
                self.note = f"{name}: czekam na odpowiedz {waited:.1f} s ({context})"
                return None
            self._runner.cancel()
            error = JudgeError(f"{name} nie odpowiedzial w {self._timeout_s:g} s", kind="timeout",
                               retryable=True)
            outcome = Outcome(reply=None, error=error, response_s=None)
        self._record(outcome)
        if outcome.error is not None:
            error = outcome.error
            if error.retryable and self._asks < ASKS_PER_QUESTION:
                self._retry_at = float(now_t) + RETRY_WAIT_S
                self.note = f"{error.reason}, pytam ponownie za {RETRY_WAIT_S:g} s"
                return None
            self._clear()
            return error
        reply = outcome.reply
        assert reply is not None
        self._clear()
        return reply

    def cancel(self, reason: str, *, to_outbox: bool = True) -> None:
        """Gives up on the pending question, in flight or in its retry pause, recorded as cancelled with
        `reason`; an answer that came in just now is recorded as it is and never acted on. Without
        `to_outbox` the records go only to `judging.decisions` (a run's log closed by now). A no-op with
        nothing pending; never waits and never raises."""
        if not self._open:
            self._runner.cancel()
            return
        outcome = self._runner.poll()
        if outcome is None:
            outcome = Outcome(reply=None, error=JudgeError(reason, kind="cancelled"), response_s=None)
        self._runner.cancel()
        self._record(outcome, to_outbox)
        self._clear()

    def drain(self) -> tuple[Decision, ...]:
        """The records since the last drain, each once; cancelled calls with `to_outbox` off only in
        `judging.decisions`."""
        drained, self._outbox = tuple(self._outbox), []
        return drained

    def wait(self, timeout_s: float | None = None) -> None:
        """Waits for the call in flight, if any (tests); the next `step` collects its answer."""
        self._runner.wait(timeout_s)

    def _record(self, outcome: Outcome, to_outbox: bool = True) -> None:
        """One `Decision` per question, the answer as received or the error; and a log line."""
        judging, reply, error = self.judging, outcome.reply, outcome.error
        state = {"sent": self._state, "questions_sent": self._questions, "judge": self.judge.name,
                 "ask": self._asks, "frames": sorted(self._state.get("camera_views", {})), **self._meta}
        entry: dict[str, Any] = {"ask": self._asks, "judge": self.judge.name,
                                 "response_s": outcome.response_s, **self._meta}
        if reply is not None:
            judging.answers += 1
            judging.model = reply.model
            judging.last_response_s = math.nan if outcome.response_s is None else outcome.response_s
            given = list(reply.answers.values())
            judging.last_verdict = given[0].choice if given else ""
            judging.last_correction = given[1].choice if len(given) > 1 else ""
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
                       for question in self._questions]
            entry.update(error={"kind": error.kind, "status": error.status, "reason": error.reason})
        judging.decisions.extend(records)
        if to_outbox:
            self._outbox.extend(records)
        logger.info("%s: %s", self.log_label, json.dumps(entry, sort_keys=True))
