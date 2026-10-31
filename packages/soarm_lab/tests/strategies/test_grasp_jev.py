"""grasp_jev without the twin and without Jev: the states it sends at the probe, the close and the lift, the
decision rules, the regrasps, the wait, the wrist frame and the records.

The judges are fakes at the model's boundary, as in test_touch_jev.py: `Scripted` answers from a list
(replies built through the same `checked_reply` Jev's are), `Gated` blocks the calls of one point until the
test lets it go, and some tests run the real Jev client on a fake HTTP transport. As in test_grasp.py,
`follow` plays the twin's supervisor, the arm and the jaw on a block of the set (`Block`): the arm stalls
where the jaw's sole would go into the block, the jaw closes onto the block when both tips straddle it, and
the block then rides with the fingertips. Hooks let a block be missed by the jaw on an attempt (`grip`),
slip out above a height (`slip_z`), be taken away (`taken_at`) or meet the table under the jaw (`floor`).
test_grasp_jev_in_twin.py runs the strategy through the twin's runner and physics. The live test runs only
with TYPESAFE_API_KEY set.
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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, heading, wrap  # noqa: E402
from soarm_lab.strategies.grasp_jev import (  # noqa: E402
    ENDED,
    HELD_OPTIONS,
    NEXT_OPTIONS,
    QUESTIONS_VERSION,
    BlockSighting,
    GraspFacts,
    GraspJevSettings,
    GraspJevStrategy,
    available_next,
    blob_bucket,
    fit_frame,
    grasp_questions,
    grasp_state,
    jaw_gap_bucket,
    lifted_bucket,
    next_problem,
    sighting_state,
    source_name,
)
from soarm_lab.strategies.jev import KEY_ENV, JevJudge  # noqa: E402
from soarm_lab.strategies.judges import JudgeError, Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.touch_jev import CORRECTIONS, VERDICTS, touch_questions  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Where the block lies in most tests [m, base frame]: inside the top-down band, as in test_grasp.py.
XY = (0.21, 0.02)
NAN = float("nan")
CUBE = object_spec("cube").half_size
CUBOID5 = object_spec("cuboid5").half_size
KEY = "sk-test-SECRET-9d2e"
NEW_FIELDS = ["model", "min_confidence", "min_correction_confidence", "correction_m", "answer_timeout_s",
              "send_frames", "wrist_camera"]
#: The objects of each observation: fixed, or a function of its time and the strategy.
Objects = Mapping[str, ObjectPose] | Callable[[float, GraspJevStrategy], Mapping[str, ObjectPose]]
#: The frames of each observation, likewise.
Frames = Mapping[str, Frame] | Callable[[float, GraspJevStrategy], Mapping[str, Frame]]


def run_info() -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits=dict(SPEC.limits),
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)


def obs(joints: Mapping[str, float], t: float = 0.0, objects: Mapping[str, ObjectPose] | None = None,
        command: Mapping[str, float] | None = None, frames: Mapping[str, Frame] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=dict(joints), command=dict(command or {}),
                       objects=dict(objects or {}), frames=dict(frames or {}))


def seen(xy: tuple[float, float] = XY, source: str = "ostatnie widziane", age_s: float = 8.0, z: float = 0.0,
         label: str = "cube", rotation: np.ndarray | None = None) -> dict[str, ObjectPose]:
    """A pose as the LLM detector gives one: no orientation unless `rotation` is given."""
    rotation = np.eye(3) if rotation is None else rotation
    return {label: ObjectPose(position=np.array([*xy, z], float), rotation=rotation, source=source,
                              age_s=age_s)}


def ageing(xy: tuple[float, float], age_s: float = 8.0, label: str = "cube") -> Objects:
    """Seen once, `age_s` before the run started: the twin's "ostatnie widziane", ageing from then on."""
    return lambda t, s: seen(xy, age_s=age_s + t, label=label)


def tcp(joints: Mapping[str, float]) -> np.ndarray:
    return KIN.tcp(dict(joints))[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """The phases in order, repeats dropped."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def contains(seq: list[str], part: list[str]) -> bool:
    """Whether `part` appears in `seq` in order, not necessarily next to each other."""
    it = iter(seq)
    return all(p in it for p in part)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def first_heading() -> float:
    """The closing heading of the rest pose [rad]: the one an identity rotation keeps."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


@dataclass
class Block:
    """A block of the set lying on the table, axis-aligned after its yaw, as the fake arm meets it."""

    xy: np.ndarray
    half: tuple[float, float, float]
    yaw: float = 0.0
    #: The jaw holds it: it rides with the fingertips and no longer stalls the arm.
    gripped: bool = False
    #: The block is there at all; False = taken away.
    present: bool = True
    #: When the jaw last lost the block: the time of the first observation after it [s]; NaN = never.
    dropped_t: float = NAN
    path: list[np.ndarray] = field(default_factory=list)

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def local(self, point: np.ndarray) -> np.ndarray:
        """`point`'s x and y in the block's frame [m]."""
        return rot_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    def stalls(self, joints: Mapping[str, float]) -> bool:
        """The arm cannot take `joints`: the jaw's sole below the top with a fingertip over the block."""
        if not self.present or self.gripped:
            return False
        sole = KIN.tcp(joints)[2, 3] - KIN.jaw_reach(joints[SPEC.gripper])
        if sole >= self.height:
            return False
        return any(bool(np.all(np.abs(self.local(tip)) < self.half[:2])) for tip in KIN.fingertips(joints))

    def between(self, joints: Mapping[str, float]) -> bool:
        """Both tips below the top, on opposite sides of the block's centre along one of its axes, and
        within its extent along the other."""
        if not self.present:
            return False
        if self.gripped:
            return True
        tips = KIN.fingertips(joints)
        if any(tip[2] >= self.height for tip in tips):
            return False
        a, b = (self.local(tip) for tip in tips)
        return any(a[k] * b[k] < 0 and abs(a[1 - k]) < self.half[1 - k] and abs(b[1 - k]) < self.half[1 - k]
                   for k in (0, 1))

    def across(self, joints: Mapping[str, float]) -> float:
        """The block's half size along its axis the tips lie farther apart on [m]."""
        a, b = (self.local(tip) for tip in KIN.fingertips(joints))
        return self.half[0 if abs(a[0] - b[0]) >= abs(a[1] - b[1]) else 1]

    def units(self, joints: Mapping[str, float]) -> float:
        """The gripper command whose tip gap spans the block across the jaws [app units]."""
        return KIN.gripper_for_gap(min(2 * self.across(joints), KIN.jaw_gap(100.0)))

    def push(self, joints: Mapping[str, float]) -> None:
        """The moving tip of `joints` pushes the block along the line between the tips toward the fixed tip,
        as far as its near face lets it."""
        fixed, moving = (tip[:2] for tip in KIN.fingertips(joints))
        gap = float(np.linalg.norm(moving - fixed))
        if gap < 1e-9:
            return
        unit = (moving - fixed) / gap
        half = self.across(joints)
        centre = float((self.xy - fixed) @ unit)
        if gap - centre < half:
            self.xy = self.xy + (gap - half - centre) * unit


def block_at(xy: tuple[float, float] = XY, half: tuple[float, float, float] = CUBE,
             yaw: float = 0.0) -> Block:
    return Block(np.array(xy, float), half, yaw)


def follow(s: GraspJevStrategy, objects: Objects | None, ticks: int = 2500,
           joints: Mapping[str, float] | None = None, t0: float = 0.0, block: Block | None = None,
           slip_z: float | None = None,
           frames: Frames | None = None,
           until: Callable[[GraspJevStrategy], bool] | None = None, grip: Callable[[int], bool] | None = None,
           taken_at: Callable[[float, GraspJevStrategy], bool] | None = None,
           floor: Callable[[GraspJevStrategy, np.ndarray], bool] | None = None,
           commands: list[dict[str, float]] | None = None, measured: list[dict[str, float]] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor, the arm and the jaw, ending on `done` or once `until` holds.

    The supervisor's command goes to each target at once and the arm follows it, staying where it was when
    `block` says the new pose would put the jaw's sole into it, or when `floor(s, command's TCP)` holds.
    The measured gripper is the command, or the block's width while both tips straddle it (`grip(attempt)`
    False: the block is not between the tips on that attempt). The closing tip pushes a straddled block
    toward the fixed tip once it reaches its face (test_grasp.py's model moves it only once gripped, and a
    jaw opened wider than its 41 mm would sweep through the block). Once the jaw stands at the block's
    width with the command tighter, the block is gripped and rides with the fingertips, until the command
    opens wider, the TCP rises above `slip_z` (once) or `taken_at(t, s)` takes it away for good; the first
    observation after that stamps `block.dropped_t`. Returns the answers, the phase after each tick, the
    last measured joints and the time to go on from.
    """
    answers, phases = [], []
    joints = dict(joints or SPEC.home)
    command, t = dict(joints), t0
    g = SPEC.gripper
    assert g is not None
    for _ in range(ticks):
        if commands is not None:
            commands.append(command)
        if measured is not None:
            measured.append(joints)
        gripped = block is not None and block.gripped
        now = objects(t, s) if callable(objects) else objects
        shown = frames(t, s) if callable(frames) else frames
        answers.append(a := s.act(obs(joints, t=t, objects=now, command=command, frames=shown)))
        phases.append(s.status.phase)
        t += 1 / s.hz
        command = {**command, **{name: float(value) for name, value in a.target.items()}}
        arm = dict(command)
        arm[g] = joints[g]
        if block is not None and block.present and taken_at is not None and taken_at(t, s):
            block.present = block.gripped = False
        between = (block is not None and block.between(arm)
                   and (grip is None or block.gripped or grip(s.status.attempt)))
        arm[g] = max(command[g], block.units(arm)) if between and block is not None else command[g]
        if block is not None and between and not block.gripped:
            block.push(arm)
        if block is not None:
            units = block.units(arm) if between else NAN
            slips = slip_z is not None and math.isnan(block.dropped_t) and tcp(arm)[2] > slip_z
            if block.gripped and (command[g] > units + 0.5 or slips):
                block.gripped = False
                arm[g] = command[g]
            elif between and not block.gripped and abs(arm[g] - units) < 0.5 and command[g] < units - 0.5:
                block.gripped = True
            if gripped and not block.gripped:
                block.dropped_t = t
            if block.stalls(arm):
                arm = dict(joints, **{g: arm[g]})
            if block.gripped:
                block.xy = np.mean(KIN.fingertips(arm), axis=0)[:2]
            block.path.append(block.xy.copy())
        if floor is not None and floor(s, tcp(command)):
            arm = dict(joints, **{g: arm[g]})
        joints = arm
        if a.done or (until is not None and until(s)):
            break
    return answers, phases, joints, t


def raw_answer(choice: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    """A choice answer as Jev sends it, with probabilities that give `confidence`."""
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    probabilities = {option: top if option == choice else (1 - top) / (n - 1) for option in options}
    return {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}


def answered_touch(verdict: str = "touched_object", sure: float = 0.95,
                   correction: str = "aim_at_latest_sighting",
                   correction_sure: float = 0.9) -> dict[str, Any]:
    """The JSON body of Jev's answer to the probe's questions."""
    return {"model": "jev-1.14.0", "usage": {"input_tokens": 612, "output_tokens": 4},
            "answers": {"touched": raw_answer(verdict, sure, VERDICTS),
                        "correction": raw_answer(correction, correction_sure, CORRECTIONS)}}


def answered_grasp(held: str = "block_held", sure: float = 0.9, nxt: str = "lift",
                   nsure: float = 0.9) -> dict[str, Any]:
    """The JSON body of Jev's answer to the close or lift questions."""
    return {"model": "jev-1.14.0", "usage": {"input_tokens": 700, "output_tokens": 4},
            "answers": {"held": raw_answer(held, sure, HELD_OPTIONS),
                        "next": raw_answer(nxt, nsure, NEXT_OPTIONS)}}


def reply_touch(*args: Any, **kwargs: Any) -> Reply:
    body = answered_touch(*args, **kwargs)
    return checked_reply("jev", body["answers"], body["model"], body["usage"], touch_questions(0.02))


def reply_grasp(*args: Any, **kwargs: Any) -> Reply:
    body = answered_grasp(*args, **kwargs)
    return checked_reply("jev", body["answers"], body["model"], body["usage"], grasp_questions())


TOUCHED = reply_touch("touched_object", 0.95)
HELD = reply_grasp("block_held", 0.9, "lift", 0.9)
EDGE_WIDER = reply_grasp("held_by_edge", 0.9, "regrasp_wider", 0.8)


class Scripted:
    """Answers from `outcomes` (a `Reply` or an exception to raise), in order."""

    name = "jev"

    def __init__(self, *outcomes: Reply | Exception):
        self.outcomes = list(outcomes)
        self.states: list[Mapping[str, Any]] = []
        self.questions: list[Mapping[str, Any]] = []
        self.images: list[Mapping[str, np.ndarray]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.states.append(state)
        self.questions.append(questions)
        self.images.append(images)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Gated:
    """Answers the probe's questions with `probe` at once, and the grasp questions with `grasp`, but those of
    the point `point` only once the test opens the gate; keeps every cancel event it was given."""

    name = "jev"

    def __init__(self, point: str = "close", grasp: Reply = HELD, probe: Reply = TOUCHED):
        self.point, self.grasp, self.probe = point, grasp, probe
        self.gate = threading.Event()
        self.cancels: list[threading.Event] = []
        self.states: list[Mapping[str, Any]] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.states.append(state)
        if "held" not in questions:
            return self.probe
        if state.get("point") == self.point:
            self.cancels.append(cancel)
            self.gate.wait(10.0)
        return self.grasp


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
            joints: Mapping[str, float] | None = None,
            blob: Callable[[np.ndarray], float | None] | None = None, **settings: Any) -> GraspJevStrategy:
    settings.setdefault("carry_s", 0.5)
    s = GraspJevStrategy(GraspJevSettings(**settings), judge=judge, threaded=threaded, clock=clock, blob=blob)
    s.start(run_info(), obs(joints or SPEC.home))
    return s


def answer_of(decision: Any) -> dict[str, Any]:
    return json.loads(decision.answer)


def points(s: GraspJevStrategy) -> list[str]:
    return [d.state["point"] for d in s.judging.decisions]


def gradient(size: tuple[int, int]) -> np.ndarray:
    rows, cols = size
    image = np.zeros((rows, cols, 3), np.uint8)
    image[..., 0] = (np.arange(cols) % 256)[None, :]
    image[..., 1] = (np.arange(rows) % 256)[:, None]
    image[..., 2] = 90
    return image


def wrist_frames(age_s: float = 0.05, size: tuple[int, int] = (1080, 1920),
                 camera: str = "wrist") -> dict[str, Frame]:
    return {"front": Frame(image=np.full((300, 400, 3), 120, np.uint8), age_s=0.05),
            camera: Frame(image=gradient(size), age_s=age_s)}


def asked_at(answers: list[Action], start: str) -> int:
    return next(k for k, a in enumerate(answers) if a.note.startswith(start))


# ------------------------------------------------------------ the contract
def test_it_follows_the_strategy_protocol_with_a_fake_judge():
    """test_entry_points.py's protocol check, which skips grasp_jev for want of a key."""
    assert {f.type for f in dataclasses.fields(GraspJevSettings)} <= {"str", "int", "float", "bool"}
    strategy = GraspJevStrategy(GraspJevSettings(), judge=Scripted())
    assert isinstance(strategy.hz, float) and strategy.hz > 0 and strategy.time_limit_s == 180.0
    strategy.start(dataclasses.replace(run_info(), limits={}), obs(SPEC.home))
    action = strategy.act(Observation(t=1 / strategy.hz, step=1, joints=dict(SPEC.home), command={}))

    assert isinstance(action, Action)
    assert set(action.target) == set(SPEC.joints) and not action.done
    strategy.stop("zatrzymana")


def test_it_builds_without_a_key_and_refuses_to_start_without_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    for strategy in (GraspJevStrategy(), GraspJevStrategy(judge=JevJudge(environ={}))):
        with pytest.raises(RuntimeError, match="brak klucza jev: ustaw TYPESAFE_API_KEY"):
            strategy.start(run_info(), obs(SPEC.home))
        assert not strategy.started

    strategy = GraspJevStrategy(judge=JevJudge(environ={KEY_ENV: KEY}))
    strategy.start(run_info(), obs(SPEC.home))
    assert strategy.started


def test_its_settings_are_grasps_and_the_judges():
    names = [f.name for f in dataclasses.fields(GraspJevSettings)]

    assert names == [f.name for f in dataclasses.fields(GraspSettings)] + NEW_FIELDS
    defaults = GraspJevSettings()
    assert defaults.model == "jev-1.14.0" and defaults.wrist_camera == "wrist" and not defaults.send_frames
    assert defaults.time_limit_s == 180.0 and defaults.max_attempts == 3 and defaults.carry_s == 10.0
    for given, kept in ((" wrist ", "wrist"), ("Wrist", "Wrist")):
        assert GraspJevStrategy(GraspJevSettings(wrist_camera=given), judge=Scripted()).wrist_camera == kept


@pytest.mark.parametrize("settings, message", [
    ({"model": "  "}, "model must not be empty"),
    ({"min_confidence": 1.5}, "min_confidence must lie within 0..1"),
    ({"min_correction_confidence": NAN}, "min_correction_confidence must lie within 0..1"),
    ({"correction_m": 0.0}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"correction_m": 0.01}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"correction_m": 0.2}, r"correction_m must lie within \(0.01, 0.1\] m"),
    ({"answer_timeout_s": 0.0}, r"answer_timeout_s must lie within \(0, 60\] s"),
    ({"answer_timeout_s": 120.0}, r"answer_timeout_s must lie within \(0, 60\] s"),
    ({"wrist_camera": "  "}, "wrist_camera must not be empty"),
])
def test_bad_settings_are_refused(settings: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=f"grasp_jev: {message}"):
        GraspJevStrategy(GraspJevSettings(**settings), judge=Scripted())


# ------------------------------------------------------------ a whole grasp
def test_a_confident_probe_close_and_lift_lift_the_cube_with_three_consultations():
    judge = Scripted(TOUCHED, HELD, HELD)
    s = started(judge)
    answers, phases, *_ = follow(s, seen(), block=block_at())

    last = answers[-1]
    assert runs(phases)[-10:] == ["descend", "touched", "rise", "open", "straddle", "close", "lift_off",
                                  "check", "carry", "done"]
    assert last.done and last.success and last.note.startswith("blok w szczekach, podniesiony o"), last.note
    for start in ("jev: pytam o ocene dotyku (proba 1)", "jev: pytam o ocene chwytu po zamknieciu (proba 1)",
                  "jev: pytam o ocene chwytu po podniesieniu (proba 1)"):
        k = asked_at(answers, start)
        assert answers[k].target == pytest.approx(answers[k - 1].target) and not answers[k].done
    close, lift = judge.states[1], judge.states[2]
    assert close["point"] == "close" and close["lift"] is None
    assert close["close"]["ended"] == "jaw_stopped_on_something" and close["close"]["jaw_gap"] == "block_wide"
    assert close["available_next"] == ["lift", "regrasp_wider", "give_up"]
    assert close["heading"] == {"from": "guess", "yaw_symmetry_deg": 90}
    assert close["block_seen_between_jaws"] == "unknown" and close["earlier_attempts"] == []
    assert lift["lift"]["lifted"] == "lifted" and lift["lift"]["lifted_mm"] == pytest.approx(40, abs=3)
    assert lift["close"]["jaw_moved_after_close"] is False and lift["point"] == "lift"
    assert lift["earlier_attempts"] == [{"attempt": 1, "probe": "touched", "close": "held",
                                         "verdict": "block_held", "next": "lift",
                                         "chosen_by": "first_sighting"}]
    assert points(s) == ["probe", "probe", "close", "close", "lift", "lift"]
    versions = [d.state["questions"] for d in s.judging.decisions]
    assert versions == ["touch_jev/1"] * 2 + [QUESTIONS_VERSION] * 4
    assert s.judging.disagreements == 0 and not s.judging.needs_review and s.judging.asks == 3
    after_close = phases.index("lift_off") + 1
    assert answers[after_close].note.endswith(" | jev #1 po zamknieciu: blok trzymany 0.90, podniesc 0.90")
    assert s.drain_decisions() == tuple(s.judging.decisions) and s.drain_decisions() == ()


def test_the_probe_correction_moves_the_block_aim_not_the_tcp():
    left = np.array([-math.sin(math.atan2(XY[1], XY[0])), math.cos(math.atan2(XY[1], XY[0]))])
    real = tuple(np.asarray(XY) + 0.02 * left)
    judge = Scripted(reply_touch("missed_object", 0.95, "to_robot_left", 0.9), TOUCHED, HELD, HELD)
    s = started(judge)
    answers, *_ = follow(s, ageing(XY), block=block_at(real))

    assert answers[-1].success, answers[-1].note
    assert s.status.attempt == 2 and s.grasp.block_xy == pytest.approx(real, abs=1e-3)
    u = np.array([math.cos(math.radians(s.grasp.heading_deg)), math.sin(math.radians(s.grasp.heading_deg))])
    assert s.status.touch_point is not None
    # The probe's TCP lies 15.6 mm toward the moving jaw from the block centre the run aims at.
    assert np.asarray(s.status.touch_point[:2]) == pytest.approx(np.asarray(real) + 0.0156 * u, abs=1e-3)
    first = judge.states[0]
    assert first["latest_sighting"]["offset_mm"] == {"farther": 0, "left": 0}       # against the block centre
    assert first["descent"]["ended"] == "reached_touch_point_in_free_air"
    assert judge.states[1]["aim"] == {"chosen_by": "correction:to_robot_left", "sighting": None}


def test_jev_touched_on_a_probe_that_met_nothing_stops_for_review():
    s = started(Scripted(reply_touch("touched_object", 0.9)))
    answers, phases, *_ = follow(s, seen())

    assert answers[-1].note == ("jev: dotkniety (0.90), ale sonda nie napotkala nic; geometria: brak "
                                "kontaktu - ramie stoi, do oceny czlowieka")
    assert s.judging.needs_review and s.judging.disagreements == 1 and "rise" not in phases


def test_jev_touched_on_a_stall_far_above_the_block_stops_for_review():
    """Something 2 cm taller than the cube under the probe: with a 1 cm contact band the stall, 3 cm above
    the probe's bottom, is a block. A straddle from that top would miss the cube."""
    s = started(Scripted(reply_touch("touched_object", 0.9)), contact_band_m=0.01)
    tall = block_at(half=(CUBE[0], CUBE[1], CUBE[2] + 0.01))
    answers, phases, *_ = follow(s, seen(), block=tall)

    note = answers[-1].note
    assert s.descent_verdict()[0] == "blocked"
    assert note.startswith("jev: dotkniety (0.90), ale ramie stanelo "), note
    assert note.endswith(" mm nad punktem dotyku; geometria: zablokowane nad obiektem - ramie stoi, do oceny "
                         "czlowieka")
    assert 25 <= int(note.split("stanelo ")[1].split(" mm")[0]) <= 35
    assert s.judging.needs_review and s.judging.disagreements == 1 and "rise" not in phases
    assert answers[-1].target == pytest.approx(answers[-2].target)


def test_an_arm_that_never_came_down_is_not_asked_about():
    """Readings that freeze on the hover once the descent starts: the stall is no touch, and grasp's default
    ends the run with touch's reason (a block: the hover lies 4 cm above the probe's bottom)."""
    s = started(Scripted())
    answers, *_ = follow(s, seen(), floor=lambda s, p: s.status.phase == "descend")

    assert answers[-1].done and not answers[-1].success and s.status.drop_m < s.config.contact_gap_m
    assert s.descent_verdict() == ("blocked", answers[-1].note)
    assert s.judging.asks == 0 and s.judging.decisions == []


# ------------------------------------------------------------ the close and the regrasps
def test_an_empty_close_asks_and_regrasp_wider_opens_one_margin_more_and_shifts_sideways():
    judge = Scripted(TOUCHED, reply_grasp("nothing_in_jaws", 0.9, "regrasp_wider", 0.8), TOUCHED, HELD, HELD)
    s = started(judge)
    aims: list[tuple[float, float]] = []
    commands: list[dict[str, float]] = []
    answers, phases, *_ = follow(s, seen(), block=block_at(), grip=lambda attempt: attempt >= 2,
                                 commands=commands,
                                 until=lambda s: bool(aims.append(s.grasp.block_xy) or False))

    assert answers[-1].success, answers[-1].note
    close = judge.states[1]
    assert close["close"]["ended"] == "jaw_closed_on_nothing" and close["close"]["jaw_gap"] == "closed"
    assert close["available_next"] == ["regrasp_wider", "give_up"]
    assert s.gap_open_m == pytest.approx(0.041 + 0.016, abs=1e-6)
    assert s.cmd_open == pytest.approx(KIN.gripper_for_gap(0.057), abs=0.1)
    u = np.array([math.cos(math.radians(s.grasp.heading_deg)), math.sin(math.radians(s.grasp.heading_deg))])
    assert s.grasp.block_xy == pytest.approx(np.asarray(XY) + 0.0125 * u, abs=1e-6)
    assert contains(runs(phases), ["close", "judge", "rise", "approach", "descend", "touched", "rise", "open",
                                   "straddle", "close", "lift_off", "check"])
    second = [k for k, p in enumerate(phases) if p == "close"][-1]
    fixed, moving = KIN.fingertips(dict(answers[second].target, gripper=s.cmd_open))
    aim = np.asarray(s.grasp.block_xy)
    along = sorted([(fixed[:2] - aim) @ u, (moving[:2] - aim) @ u])
    assert along == pytest.approx([-0.0285, 0.0285], abs=1e-3)
    assert judge.states[3]["earlier_attempts"] == [{"attempt": 1, "probe": "touched", "close": "empty",
                                                    "verdict": "nothing_in_jaws", "next": "regrasp_wider",
                                                    "chosen_by": "first_sighting"}]
    assert judge.states[2]["aim"] == {"chosen_by": "next:regrasp_wider", "sighting": None}
    decide = phases.index("rise", phases.index("judge"))
    assert answers[decide].note == "jev: szczeka pusta (0.90), chwyt szerzej (0.80) - proba 2"
    assert s.status.attempt == 2 and s.judging.disagreements == 0


def test_a_tip_stall_on_a_cuboid_with_a_guessed_heading_regrasps_rotated():
    psi0 = first_heading()
    turn = reply_grasp("nothing_in_jaws", 0.9, "regrasp_rotated", 0.8)
    judge = Scripted(TOUCHED, turn, TOUCHED, HELD, HELD)
    s = started(judge, object_kind="cuboid5")
    # The long side along the jaws' closing heading: the 41 mm opening lands a tip on its top.
    block = block_at(half=CUBOID5, yaw=psi0)
    answers, phases, *_ = follow(s, seen(label="cuboid5"), block=block, ticks=3000)

    assert answers[-1].success, answers[-1].note
    tip = judge.states[1]
    assert tip["close"]["ended"] == "jaw_tip_landed_on_block_before_closing"
    assert "regrasp_rotated" in tip["available_next"] and tip["heading"]["from"] == "guess"
    assert tip["heading"]["yaw_symmetry_deg"] == 180
    assert abs(wrap(math.radians(s.grasp.heading_deg) - (psi0 + math.pi / 2))) < 0.02 or \
        abs(wrap(math.radians(s.grasp.heading_deg) - (psi0 - math.pi / 2))) < 0.02
    assert s.grasp.block_xy == pytest.approx(XY, abs=1e-6)                        # the same place, turned
    assert s.grasp.straddle_stalls == 1 and s.status.attempt == 2
    assert contains(runs(phases), ["straddle", "straddled", "judge", "rise", "approach", "descend"])


def gfacts(**changes: Any) -> GraspFacts:
    base = GraspFacts(label="cube", kind="cube", short_m=0.025, long_m=0.025, height_m=0.025, attempt=1,
                      max_attempts=3, point="close", ended=ENDED["held"], jaw_gap_m=0.025,
                      commanded_gap_m=0.017, margin_gap_m=0.004, jaw_still=True, jaw_moved_after_close=False,
                      lifted_m=NAN, lift_m=0.04, gap_open_m=0.041, blob_share=None, sighting=None,
                      aim_xy=(0.2, 0.0), heading_rad=math.pi / 2, heading_from="guess",
                      yaw_period_rad=math.pi / 2, wider_possible=True, now_t=20.0)
    return dataclasses.replace(base, **changes)


def test_regrasp_rotated_is_not_available_for_a_cube_or_a_measured_rotation():
    cuboid = {"kind": "cuboid5", "long_m": 0.05, "yaw_period_rad": math.pi}

    assert next_problem("regrasp_rotated", gfacts()) == "obrot nic nie zmienia dla tego bloku"
    assert next_problem("regrasp_rotated", gfacts(**cuboid, heading_from="block_rotation")) == \
        "kierunek szczek znany z polozenia bloku"
    assert next_problem("regrasp_rotated", gfacts(**cuboid)) == ""
    for option in ("regrasp_wider", "regrasp_rotated"):
        assert next_problem(option, gfacts(**cuboid, attempt=3)) == "wyczerpane proby chwytu"
    assert next_problem("regrasp_wider", gfacts(wider_possible=False)) == "szczeka nie otworzy sie szerzej"
    assert next_problem("lift", gfacts(ended=ENDED["empty"])) == "blok nie jest trzymany"
    assert next_problem("give_up", gfacts(attempt=3, ended=ENDED["tip"])) == ""
    assert available_next(gfacts(**cuboid)) == list(NEXT_OPTIONS)


def watched(block: Block, log: list[tuple[str, bool]]) -> Callable[[GraspJevStrategy], bool]:
    """An `until` that never ends the run and logs each tick's phase and whether the jaw grips the block."""

    def record(s: GraspJevStrategy) -> bool:
        log.append((s.status.phase, block.gripped))
        return False

    return record


def test_held_by_edge_after_the_close_releases_the_block_and_tries_again():
    judge = Scripted(TOUCHED, reply_grasp("held_by_edge", 0.9, "regrasp_wider", 0.8), TOUCHED, HELD, HELD)
    s = started(judge)
    start = s.kin.from_q(s.q_cmd)["gripper"]
    block = block_at()
    log: list[tuple[str, bool]] = []
    answers, phases, *_ = follow(s, seen(), block=block, ticks=3000, until=watched(block, log))

    assert answers[-1].success and s.status.attempt == 2, answers[-1].note
    assert contains(runs(phases), ["close", "lift_off", "let_go", "rise", "approach", "descend", "touched",
                                   "rise", "open", "straddle", "close", "lift_off", "check", "carry", "done"])
    first_close = s.judging.decisions[2].state["sent"]
    assert first_close["point"] == "close" and first_close["close"]["jaw_gap"] == "block_wide"
    assert s.judging.disagreements == 1
    decide = phases.index("let_go")
    assert answers[decide].note == "jev: trzymany za krawedz (0.90), chwyt szerzej (0.80) - proba 2"
    let_go = [k for k, p in enumerate(phases) if p == "let_go"]
    grasp_z = KIN.tcp(dict(answers[decide].target))[2, 3]
    opened = next(k for k in let_go if answers[k].target["gripper"] >= s.cmd_open - 1e-6)
    assert all(tcp(answers[k].target)[2] <= grasp_z + 0.002 for k in let_go if k <= opened)
    assert [gripped for phase, gripped in log if phase == "let_go"][-1] is False
    grips = [start] + [a.target["gripper"] for a in answers]
    assert np.max(np.abs(np.diff(grips))) <= 12.0 + 1e-6


def test_after_the_lift_a_regrasp_lowers_the_block_first():
    judge = Scripted(TOUCHED, HELD, EDGE_WIDER, TOUCHED, HELD, HELD)
    s = started(judge)
    block = block_at()
    answers, phases, *_ = follow(s, seen(), block=block, ticks=3000)

    assert answers[-1].success and s.status.attempt == 2, answers[-1].note
    assert contains(runs(phases), ["check", "lower", "let_go", "rise", "approach", "descend", "touched"])
    lower = [k for k, p in enumerate(phases) if p == "lower"]
    let_go = [k for k, p in enumerate(phases) if p == "let_go"]
    assert s.judging.decisions[4].state["point"] == "lift"
    grasp_z = answers[phases.index("close")].target
    closed = answers[lower[0]].target["gripper"]
    assert closed == pytest.approx(s.cmd_closed)
    assert tcp(answers[lower[-1]].target)[2] == pytest.approx(tcp(grasp_z)[2], abs=1e-3)
    assert all(answers[k].target["gripper"] == pytest.approx(closed) for k in lower)
    assert answers[let_go[-1]].target["gripper"] == pytest.approx(s.cmd_open)
    # The drives carry the grasp's rotation: the hand stays top-down on its heading, within a degree.
    psi = heading(KIN, grasp_z)
    for k in lower + let_go:
        approach, _ = KIN.tool_axes(answers[k].target)
        assert approach[2] < -math.cos(math.radians(1.0))
        assert abs(wrap(heading(KIN, answers[k].target) - psi)) < math.radians(1.0)


def test_a_block_that_meets_the_table_early_is_released_where_it_stalled():
    judge = Scripted(TOUCHED, HELD, EDGE_WIDER, TOUCHED, HELD, HELD)
    s = started(judge)
    measured: list[dict[str, float]] = []
    grasp_z: list[float] = []

    def table(s: GraspJevStrategy, point: np.ndarray) -> bool:
        if s.grasp.grasp_point is not None and not grasp_z:
            grasp_z.append(s.grasp.grasp_point[2])
        # The block hangs 8 mm lower in the jaw than at the close: it meets the table that much early.
        return s.status.phase == "lower" and point[2] < grasp_z[0] + 0.008

    answers, phases, *_ = follow(s, seen(), block=block_at(), ticks=3000, floor=table, measured=measured)

    assert answers[-1].success and s.status.attempt == 2, answers[-1].note
    assert contains(runs(phases), ["lower", "let_go", "rise"])
    first = phases.index("let_go")
    held = tcp(answers[first].target)[2]
    assert held == pytest.approx(tcp(measured[first])[2], abs=1e-3) and held >= grasp_z[0] + 0.005
    assert answers[first].target["gripper"] > answers[first - 1].target["gripper"]    # the jaw opens there


def test_a_slip_while_lowering_goes_to_the_recovery_without_opening_the_jaw():
    judge = Scripted(TOUCHED, HELD, reply_grasp("held_by_edge", 0.9, "regrasp_wider", 0.8))
    s = started(judge, wait_s=1.0)
    lowering: list[float] = []

    def taken(t: float, s: GraspJevStrategy) -> bool:
        if s.status.phase == "lower":
            lowering.append(t)
        return len(lowering) >= 2

    answers, phases, *_ = follow(s, seen(), block=block_at(), ticks=3000, taken_at=taken)

    assert answers[-1].done and not answers[-1].success
    assert contains(runs(phases), ["check", "lower", "rise", "recover", "done"]) and "let_go" not in phases
    assert s.grasp.drops == 1 and s.grasp.verdict == "taken"
    assert answers[-1].note.startswith("blok zabrany ze szczeki: ")
    assert s.judging.asks == 3 and points(s)[-1] == "lift"


def test_give_up_after_a_held_close_ends_the_run_with_the_block_held():
    s = started(Scripted(TOUCHED, reply_grasp("block_held", 0.9, "give_up", 0.9)))
    block = block_at()
    answers, *_ = follow(s, seen(), block=block)

    last = answers[-1]
    assert last.done and not last.success and not s.judging.needs_review
    assert last.note == "jev: rezygnacja (blok trzymany 0.90) - blok w szczekach"
    assert last.target["gripper"] == pytest.approx(s.cmd_closed, abs=0.1) and block.gripped


def test_an_unsure_held_verdict_stops_for_review_with_the_arm_on_the_block():
    s = started(Scripted(TOUCHED, reply_grasp("block_held", 0.5, "lift", 0.9)))
    answers, *_ = follow(s, seen(), block=block_at())

    assert answers[-1].note == ("jev niepewny: blok trzymany z pewnoscia 0.50 < 0.80 - ramie stoi, do oceny "
                                "czlowieka")
    assert answers[-1].target == pytest.approx(answers[-2].target) and s.judging.needs_review

    s = started(Scripted(TOUCHED, reply_grasp("held_by_edge", 0.5, "lift", 0.9)))
    answers, *_ = follow(s, seen(), block=block_at())
    assert answers[-1].note == ("jev niepewny: trzymany za krawedz z pewnoscia 0.50 < 0.80; szczeka: trzyma "
                                "- ramie stoi, do oceny czlowieka")
    assert s.judging.disagreements == 1

    s = started(Scripted(TOUCHED, reply_grasp("block_held", 0.9, "regrasp_wider", 0.5)))
    answers, *_ = follow(s, seen(), block=block_at())
    assert answers[-1].note == ("jev: blok trzymany (0.90), ale krok niepewny: chwyt szerzej 0.50 < 0.60 - "
                                "ramie stoi, do oceny czlowieka")


@pytest.mark.parametrize("case", ["lift after an empty close", "rotated on the cube",
                                  "wider beyond the travel"])
def test_a_next_step_jev_cannot_have_stops_for_review(case: str):
    settings: dict[str, Any] = {}
    grip: Callable[[int], bool] | None = None
    if case == "lift after an empty close":
        judged, grip = reply_grasp("block_held", 0.9, "lift", 0.9), (lambda attempt: False)
        said = "'podniesc' niewykonalny: blok nie jest trzymany; szczeka: pusta"
    elif case == "rotated on the cube":
        judged = reply_grasp("block_held", 0.9, "regrasp_rotated", 0.9)
        said = "'chwyt obrocony' niewykonalny: obrot nic nie zmienia dla tego bloku"
    else:
        # 25 mm and twice 2 cm: 65 mm open, and 105 mm more than the jaw's 88 mm.
        judged, settings = reply_grasp("block_held", 0.9, "regrasp_wider", 0.9), {"open_margin_m": 0.02}
        said = "'chwyt szerzej' niewykonalny: szczeka nie otworzy sie szerzej"
    s = started(Scripted(TOUCHED, judged), **settings)
    answers, *_ = follow(s, seen(), block=block_at(), grip=grip)

    assert answers[-1].note == f"jev: krok {said} - ramie stoi, do oceny czlowieka"
    assert s.judging.needs_review and answers[-1].target == pytest.approx(answers[-2].target)


def test_a_regrasp_on_the_last_attempt_ends_the_run_without_review():
    s = started(Scripted(TOUCHED, reply_grasp("nothing_in_jaws", 0.9, "regrasp_wider", 0.9)), max_attempts=1)
    answers, *_ = follow(s, seen(), block=block_at(), grip=lambda attempt: False)

    assert answers[-1].note == "jev: szczeka pusta w ostatniej probie (1 z 1, pewnosc 0.90)"
    assert answers[-1].done and not s.judging.needs_review


@pytest.mark.parametrize("point", ["close", "lift"])
def test_the_phase_clock_starts_after_the_answer(point: str):
    """80 ticks (4 s) of waiting would spend the lift's 3.4 s and the check's 1 s plus 3 s: both count from
    the answer, so the run still lifts and holds for `hold_s` after it."""
    judge = Gated(point=point)
    s = started(judge, threaded=True, answer_timeout_s=10.0, hold_s=1.0, carry_s=0.0)
    block = block_at()
    try:
        answers, phases, joints, t = follow(s, seen(), block=block, until=lambda s: len(judge.cancels) == 1)
        eventually(lambda: len(judge.cancels) == 1)
        pending, pending_phases, joints, t = follow(s, seen(), ticks=80, joints=joints, t0=t, block=block)
    finally:
        judge.gate.set()
    s.wait_for_answer(5.0)
    rest, rest_phases, *_ = follow(s, seen(), joints=joints, t0=t, block=block)

    assert all(not a.done for a in pending)
    assert all(a.note.startswith("jev: czekam na odpowiedz") for a in pending[1:])
    assert set(pending_phases) == {"lift_off" if point == "close" else "check"}
    assert rest[-1].success, rest[-1].note
    if point == "lift":
        assert len(rest) >= s.grasp_settings.hold_s * s.hz
    assert s.judging.asks == 3 and not s.judging.needs_review


# ------------------------------------------------------------ a block that drops out: grasp's recovery
def landed(block: Block, delay_s: float = 0.6) -> Objects:
    """The twin's pose around a drop: the LLM detector's pose at `XY` until the jaw loses the block, nothing
    for `delay_s`, then the cameras see it lying where it fell."""

    def objects(t: float, s: GraspJevStrategy) -> Mapping[str, ObjectPose]:
        since = t - block.dropped_t
        if math.isnan(since):
            return seen()
        if since < delay_s:
            return {}
        return seen(tuple(block.xy), source="kamery", age_s=0.1)

    return objects


def test_a_block_that_slips_on_the_way_up_is_grasped_again_and_jev_judges_the_regrasp():
    judge = Scripted(TOUCHED, HELD, TOUCHED, HELD, HELD)
    s = started(judge)
    block = block_at()
    slip_z = max(0.012, 0.025 - 0.0125 + 0.003) + 0.02
    answers, phases, *_ = follow(s, landed(block), block=block, ticks=4000, slip_z=slip_z)

    assert answers[-1].success, answers[-1].note
    assert contains(runs(phases), ["lift_off", "rise", "recover", "approach", "descend", "touched", "rise",
                                   "open", "straddle", "close", "lift_off", "check", "carry", "done"])
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1 and s.status.attempt == 2
    assert points(s) == ["probe"] * 2 + ["close"] * 2 + ["probe"] * 2 + ["close"] * 2 + ["lift"] * 2
    regrasp = judge.states[2]
    assert regrasp["aim"]["chosen_by"] == "recovery" and regrasp["aim"]["sighting"]["source"] == "cameras"
    assert judge.states[3]["earlier_attempts"][-1] == {"attempt": 1, "probe": "touched",
                                                       "close": "dropped_while_carrying", "verdict": None,
                                                       "next": "recovery", "chosen_by": "first_sighting"}


def test_a_slip_while_the_close_answer_is_pending_cancels_it_and_the_recovery_takes_over():
    judge = Gated(point="close")
    s = started(judge, threaded=True, wait_s=1.0)
    block = block_at()
    try:
        answers, phases, joints, t = follow(s, seen(), block=block, until=lambda s: len(judge.cancels) == 1)
        eventually(lambda: len(judge.cancels) == 1)
        block.present = block.gripped = False
        rest, rest_phases, *_ = follow(s, seen(), joints=joints, t0=t, block=block)
    finally:
        judge.gate.set()

    assert judge.cancels[0].is_set()
    assert contains(runs(rest_phases), ["lift_off", "rise", "recover", "done"])
    assert s.grasp.drops == 1 and not rest[-1].success
    close = [answer_of(d).get("error") for d in s.judging.decisions if d.state["point"] == "close"]
    assert close == ["cancelled", "cancelled"]
    drained = s.drain_decisions()
    assert [answer_of(d).get("error") for d in drained if d.state["point"] == "close"] == close
    assert any(a.note == "blok wypadl ze szczeki - w gore" for a in rest)


def test_a_slip_during_the_retry_pause_of_the_close_question_goes_to_the_recovery_without_raising():
    overloaded = JudgeError("jev chwilowo niedostepny (529)", kind="overloaded", retryable=True, status=529)
    judge = Scripted(TOUCHED, overloaded, TOUCHED, HELD, HELD)
    s = started(judge)
    block = block_at()
    asked: list[float] = []

    def taken(t: float, s: GraspJevStrategy) -> bool:
        if s.judging.asks >= 2:
            asked.append(t)
        # Two ticks after the close question failed: inside its 0.5 s retry pause.
        return len(asked) >= 3 and s.status.attempt == 1

    answers, phases, *_ = follow(s, landed(block), block=block, ticks=4000, taken_at=taken)
    notes = [a.note for a in answers]
    failed = notes.index("jev chwilowo niedostepny (529), pytam ponownie za 0.5 s")
    drop = notes.index("blok wypadl ze szczeki - w gore")
    assert 0 < drop - failed < 0.5 * s.hz                          # inside the retry pause, which never asks
    assert not any(note.startswith("jev: pytam ponownie (") for note in notes[failed:drop])
    close = [answer_of(d).get("error") for d in s.judging.decisions if d.state["point"] == "close"]
    assert close[:4] == ["overloaded", "overloaded", "cancelled", "cancelled"]
    assert s.grasp.drops == 1 and s.judging.asks >= 3
    assert judge.states[2].get("point") is None                     # the regrasp's probe question is next


# ------------------------------------------------------------ waiting for the answer
def test_while_the_answer_is_pending_the_arm_holds_and_the_next_tick_after_it_decides():
    judge = Gated(point="close")
    s = started(judge, threaded=True)
    block = block_at()
    try:
        answers, _, joints, t = follow(s, seen(), block=block, until=lambda s: len(judge.cancels) == 1)
        asking = answers[-1]
        eventually(lambda: len(judge.cancels) == 1)
        pending, phases, joints, t = follow(s, seen(), ticks=10, joints=joints, t0=t, block=block)
    finally:
        judge.gate.set()

    assert all(not a.done and a.target == pytest.approx(asking.target) for a in pending)
    assert all(a.note.startswith("jev: czekam na odpowiedz") for a in pending)
    s.wait_for_answer(5.0)
    after, after_phases, *_ = follow(s, seen(), ticks=1, joints=joints, t0=t, block=block)
    assert after_phases == ["lift_off"] and after[0].target != pytest.approx(asking.target)


def test_a_judge_that_never_answers_at_the_close_is_asked_once_more_then_the_run_stops_for_review():
    judge = Gated(point="close")
    s = started(judge, threaded=True, answer_timeout_s=0.5)
    try:
        answers, *_ = follow(s, seen(), block=block_at())
    finally:
        judge.gate.set()
    s.wait_for_answer(5.0)

    close = [state for state in judge.states if state.get("point") == "close"]
    assert len(close) == 2 and close[1] == close[0] and all(c.is_set() for c in judge.cancels)
    assert answers[-1].note == "jev nie odpowiedzial w 0.5 s - ramie stoi, do oceny czlowieka"
    assert answers[-1].target["gripper"] == pytest.approx(s.cmd_closed, abs=0.1)        # still squeezing
    errors = [answer_of(d)["error"] for d in s.judging.decisions if d.state["point"] == "close"]
    assert errors == ["timeout"] * 4


def test_the_lag_guard_still_ends_the_run_while_the_answer_is_pending():
    judge = Gated(point="close")
    s = started(judge, threaded=True)
    try:
        _, _, joints, t = follow(s, seen(), block=block_at(), until=lambda s: len(judge.cancels) == 1)
        pushed = {**joints, "shoulder_pan": joints["shoulder_pan"] + 30.0}
        with pytest.raises(RuntimeError, match="ramie nie nadaza za celem"):
            s.act(obs(pushed, t=t, objects=seen(), command=joints))
        s.stop(f"RuntimeError: {s.status.stopped_because}")
    finally:
        judge.gate.set()

    assert s.status.stopped_because.startswith("ramie nie nadaza za celem")
    assert [answer_of(d).get("error") for d in s.judging.decisions if d.state["point"] == "close"] == \
        ["cancelled", "cancelled"]


def test_a_stop_while_pending_cancels_the_call_and_a_new_start_begins_afresh():
    judge = Gated(point="close")
    s = started(judge, threaded=True)
    follow(s, seen(), block=block_at(), until=lambda s: len(judge.cancels) == 1)
    eventually(lambda: len(judge.cancels) == 1)
    began = time.monotonic()
    s.stop("zatrzymana z panelu")

    assert time.monotonic() - began < 0.5 and judge.cancels[0].is_set()
    assert s.status.stopped_because == "zatrzymana z panelu"
    assert [answer_of(d).get("error") for d in s.judging.decisions][-2:] == ["cancelled", "cancelled"]
    assert all(d.state["point"] == "probe" for d in s.drain_decisions())      # the run's log is closed by now
    judge.gate.set()
    s.wait_for_answer(5.0)
    assert len(s.judging.decisions) == 4

    s.start(run_info(), obs(SPEC.home))
    assert s.judging.decisions == [] and s.judging.asks == 0 and s.status.phase == "wait"


# ------------------------------------------------------------ the Jev client in the loop
def test_an_overloaded_jev_at_the_lift_is_asked_again_with_the_same_state():
    transport = FakeTransport((200, answered_touch()), (200, answered_grasp()),
                              (529, {"detail": "Overloaded"}), (200, answered_grasp()))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, *_ = follow(s, seen(), block=block_at())

    assert answers[-1].success, answers[-1].note
    assert s.judging.asks == 4 and s.judging.failures == 1
    assert transport.bodies[2] == transport.bodies[3]
    sent = json.loads(transport.bodies[1])
    assert sent["questions"] == s.grasp_questions and sent["state"]["point"] == "close"
    assert sent["model"] == "jev-1.14.0" and json.loads(transport.bodies[2])["state"]["point"] == "lift"
    assert any(a.note == "jev chwilowo niedostepny (529), pytam ponownie za 0.5 s" for a in answers)


def test_a_refused_request_at_the_close_ends_the_run_at_once_for_review():
    refused = {"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}
    transport = FakeTransport((200, answered_touch()), (422, refused))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, phases, *_ = follow(s, seen(), block=block_at())

    assert len(transport.bodies) == 2 and s.judging.needs_review and phases[-2] == "lift_off"
    assert answers[-1].note == ("jev odrzucil pytanie (422: body.state: Field required) - ramie stoi, "
                                "do oceny czlowieka")
    assert answers[-1].target == pytest.approx(answers[-2].target)


def test_an_error_inside_the_judge_ends_the_run_honestly():
    s = started(Scripted(TOUCHED, KeyError("held")))
    answers, *_ = follow(s, seen(), block=block_at())

    assert answers[-1].done and not answers[-1].success and s.judging.needs_review
    assert answers[-1].note == "blad sedziego: KeyError: 'held' - ramie stoi, do oceny czlowieka"


def test_the_key_appears_in_no_record_note_or_log_line(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    transport = FakeTransport((503, {"detail": f"echo {KEY}"}), (200, answered_touch()),
                              (200, answered_grasp()), (401, {"detail": f"Invalid API key {KEY}"}))
    s = started(JevJudge(transport=transport, environ={KEY_ENV: KEY}))
    answers, *_ = follow(s, seen(), block=block_at())

    assert answers[-1].note.startswith(f"jev odrzucil klucz {KEY_ENV} (401)")
    assert "Ocena chwytu: " in caplog.text
    seen_text = json.dumps([[d.question, d.answer, d.state] for d in s.judging.decisions])
    seen_text += " ".join(a.note for a in answers) + s.status.stopped_because + caplog.text + repr(s.judge)
    assert KEY not in seen_text and "Bearer" not in seen_text


# ------------------------------------------------------------ the wrist frame and the blob
def test_the_wrist_frame_alone_goes_with_every_question_when_frames_are_on():
    frames = wrist_frames()
    judge = Scripted(TOUCHED, HELD, HELD)
    s = started(judge, send_frames=True)
    answers, *_ = follow(s, seen(), block=block_at(), frames=frames)

    assert answers[-1].success, answers[-1].note
    wrist = frames["wrist"].image
    for state, images in zip(judge.states, judge.images, strict=True):
        [(reference, image)] = images.items()
        assert image.shape == (540, 960, 3) and np.array_equal(image, wrist[::2, ::2])
        assert state["camera_views"] == {"wrist": reference} and state["frames_left_out"] == {}
        assert state["frames_scaled"] == {"wrist": "1920x1080 -> 960x540"}
        assert state["frames_age_s"] == {"wrist": 0.1}
        assert list(state)[-4:] == ["camera_views", "frames_left_out", "frames_scaled", "frames_age_s"]
    assert all(d.state["frames"] == ["wrist"] for d in s.judging.decisions)
    assert "camera_views" in json.dumps(s.touch_questions["touched"])
    assert "camera_views" in json.dumps(s.grasp_questions["held"])


@pytest.mark.parametrize("case", ["missing", "small", "stale", "renamed"])
def test_a_missing_too_small_or_stale_wrist_frame_is_left_out_with_its_reason(case: str):
    calls: list[tuple[int, ...]] = []

    def blob(image: np.ndarray) -> float:
        calls.append(image.shape)
        return 0.4

    frames = {"missing": {"front": wrist_frames()["front"]}, "small": wrist_frames(size=(120, 160)),
              "stale": wrist_frames(age_s=3.0), "renamed": wrist_frames(camera="cam2")}[case]
    settings = {"wrist_camera": "cam2"} if case == "renamed" else {}
    judge = Scripted(TOUCHED, HELD)
    s = started(judge, send_frames=True, blob=blob, **settings)
    follow(s, seen(), block=block_at(), frames=frames, until=lambda s: s.judging.asks == 2)

    state, images = judge.states[1], judge.images[1]
    if case == "renamed":
        assert list(state["camera_views"]) == ["cam2"] and state["frames_left_out"] == {}
        assert state["block_seen_between_jaws"] == "yes"
        return
    why = {"missing": "brak klatki", "small": "160x120 px, poza 280..1568",
           "stale": "klatka sprzed 3.0 s"}[case]
    assert state["frames_left_out"] == {"wrist": why} and images == {} and state["camera_views"] == {}
    assert state["frames_scaled"] == {} and state["frames_age_s"] == {}
    if case == "stale":
        assert state["block_seen_between_jaws"] == "unknown" and calls == []


def test_the_blob_seam_fills_block_seen_between_jaws():
    frames = wrist_frames()
    for share, said in ((0.4, "yes"), (0.05, "no"), (None, "unknown")):
        calls: list[tuple[int, ...]] = []

        def blob(image: np.ndarray, share: float | None = share,
                 calls: list[tuple[int, ...]] = calls) -> float | None:
            calls.append(image.shape)
            return share

        judge = Scripted(TOUCHED, HELD)
        s = started(judge, blob=None if share is None else blob)
        follow(s, seen(), block=block_at(), frames=frames, until=lambda s: s.judging.asks == 2)

        assert judge.states[1]["block_seen_between_jaws"] == said
        assert "camera_views" not in judge.states[1]
        assert calls == ([] if share is None else [(1080, 1920, 3)])


# ------------------------------------------------------------ the state, without the arm
@pytest.mark.parametrize("gap, bucket", [(0.017, "closed"), (0.0205, "closed"),
                                         (0.0215, "narrower_than_block"), (0.025, "block_wide"),
                                         (0.0275, "block_wide"), (0.03, "open")])
def test_the_jaw_gap_bucket_is_computed_in_code(gap: float, bucket: str):
    assert jaw_gap_bucket(gap, 0.025, 0.017, 0.004) == bucket


def test_the_lift_and_blob_buckets_are_computed_in_code():
    assert [lifted_bucket(m, 0.04) for m in (NAN, 0.004, 0.019, 0.02, 0.06)] == \
        ["not_lifted", "not_lifted", "partly", "lifted", "lifted"]
    assert [blob_bucket(share) for share in (None, 0.14, 0.15)] == ["unknown", "no", "yes"]


def test_fit_frame_strides_a_frame_into_jevs_sides():
    image = gradient((1080, 1920))
    fitted, k = fit_frame(image)
    assert k == 2 and fitted.shape == (540, 960, 3) and np.array_equal(fitted, image[::2, ::2])
    assert fitted.flags["C_CONTIGUOUS"]
    assert fit_frame(gradient((360, 640)))[1] == 1 and fit_frame(gradient((3000, 4000)))[1] == 3


def test_the_grasp_state_is_json_with_finite_numbers_and_the_same_keys_every_time():
    keys = ["task", "attempt", "attempts_left", "point", "close", "lift", "block_seen_between_jaws",
            "block_sighting", "heading", "available_next", "earlier_attempts"]
    sighting = BlockSighting("kamery", 19.0, (0.2, 0.0), 0.0125)
    close = grasp_state(gfacts(jaw_gap_m=NAN, heading_rad=NAN, sighting=sighting))
    lift = grasp_state(gfacts(point="lift", lifted_m=NAN))
    framed = grasp_state(gfacts(with_frames=True, camera_views={"wrist": "ref"}))

    for state in (close, lift, framed):
        json.dumps(state, allow_nan=False)
    assert list(close) == keys and list(lift) == keys
    assert list(framed) == keys + ["camera_views", "frames_left_out", "frames_scaled", "frames_age_s"]
    assert close["lift"] is None and lift["lift"] == {"lifted": "not_lifted", "lifted_mm": None}
    assert close["close"]["jaw_gap_mm"] is None and close["block_sighting"]["offset_mm"] == \
        {"along_jaws": None, "across_jaws": None}
    assert close["task"].endswith("closed it to 17 mm. Judge the grasp.")
    assert "a jaw tip landed on the block" in grasp_state(gfacts(ended=ENDED["tip"]))["task"]
    assert "It then lifted the jaw 40 mm." in grasp_state(gfacts(point="lift", lifted_m=0.04))["task"]


@pytest.mark.parametrize("distance, where", [(0.005, "at_the_jaws"), (0.02, "near_the_jaws"),
                                             (0.05, "elsewhere")])
def test_a_sighting_is_placed_against_the_jaws(distance: float, where: str):
    # The jaws close along +y (heading 90 deg): a sighting toward +x lies across them, to their right.
    state = sighting_state(BlockSighting("kamery", 18.6, (0.2 + distance, 0.0), 0.0125), (0.2, 0.0),
                           math.pi / 2, 20.0)

    assert state is not None and state["where"] == where and state["source"] == "cameras"
    assert state["offset_mm"] == {"along_jaws": 0, "across_jaws": -round(1000 * distance)}
    assert state["seen_s_ago"] == 1 and state["height_mm"] == 12
    assert sighting_state(None, (0.2, 0.0), 0.0, 20.0) is None


def test_the_twins_sources_are_named_and_the_annotated_ones_kept_apart():
    names = {"kamery": "cameras", "ostatnie widziane": "last_seen", "symulacja": "simulation",
             "w dloni": "in_hand", "": "unnamed", "ostatnie widziane (upuszczona)": "last_seen_dropped",
             "ostatnie widziane (kamera b widzi gdzie indziej)": "last_seen", "brak (zaslonieta)": "none",
             "xyz": "other"}
    assert {source: source_name(source) for source in names} == names


def test_the_questions_are_typesafe_choices_with_their_options_in_order():
    questions = grasp_questions()

    assert list(questions) == ["held", "next"]
    for question in questions.values():
        assert set(question) == {"type", "instructions", "criteria"} and question["type"] == "choice"
    assert list(questions["held"]["criteria"]) == list(HELD_OPTIONS) == ["nothing_in_jaws", "block_held",
                                                                         "held_by_edge"]
    assert list(questions["next"]["criteria"]) == list(NEXT_OPTIONS) == ["lift", "regrasp_wider",
                                                                         "regrasp_rotated", "give_up"]
    known = " ".join(questions["held"]["instructions"]["known"])
    assert all(f"`{bucket}`" in known for bucket in ("closed", "narrower_than_block", "block_wide", "open"))
    assert "camera_views" not in json.dumps(questions) and "camera_views" in json.dumps(grasp_questions(True))


# ------------------------------------------------------------ notes and the end
def test_act_before_start_and_the_notes_per_phase():
    with pytest.raises(RuntimeError, match=r"grasp: act\(\) przed start\(\)"):
        GraspJevStrategy(judge=Scripted()).act(obs(SPEC.home))

    judge = Scripted(TOUCHED, HELD, EDGE_WIDER, TOUCHED, HELD, HELD)
    s = started(judge)
    answers, phases, *_ = follow(s, seen(), block=block_at(), ticks=3000)
    notes = {phase: [a.note for a, p in zip(answers, phases, strict=True) if p == phase] for phase in phases}

    assert all(note.startswith("opuszczam blok na stol: ") for note in notes["lower"][1:])
    assert notes["let_go"][1].startswith("puszczam blok: ")
    assert notes["let_go"][-1].startswith("w gore z otwarta szczeka: ")

    s = started(Scripted(TOUCHED, reply_grasp("nothing_in_jaws", 0.9, "give_up", 0.9)))
    answers, phases, *_ = follow(s, seen(), block=block_at(), grip=lambda attempt: False)
    judging = [a.note for a, p in zip(answers, phases, strict=True) if p == "judge"]
    assert judging == ["jev: pytam o ocene chwytu po zamknieciu (proba 1)"]
    assert answers[-1].note == "jev: rezygnacja (szczeka pusta 0.90) - blok na stole"


# ------------------------------------------------------------ live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"),
                    reason="a live call to Jev needs TYPESAFE_API_KEY")
def test_jev_answers_the_grasp_questions_live():
    judge = JevJudge()
    judge.check()
    questions = grasp_questions()
    answer = judge.ask(grasp_state(gfacts()), questions, images={}, timeout_s=10.0, cancel=threading.Event())

    assert answer.model.startswith("jev-")
    for name, question in questions.items():
        assert answer.answers[name].choice in question["criteria"]
        assert 0.0 <= answer.answers[name].confidence <= 1.0
