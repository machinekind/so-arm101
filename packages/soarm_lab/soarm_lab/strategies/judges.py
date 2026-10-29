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

Nothing here knows Jev: `jev.JevJudge` is one judge. A judge that runs the `claude` CLI headless would be
another, behind the same `Judge` protocol.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

#: Questions by name, in TypeSafe's JSON shape.
Questions = Mapping[str, Mapping[str, Any]]

#: Why a call gave no answer (`JudgeError.kind`).
ERROR_KINDS = ("timeout", "unreachable", "key_rejected", "rejected", "overloaded", "unusable", "error",
               "cancelled")
#: The longest technical detail an error keeps.
DETAIL_LIMIT = 300


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
