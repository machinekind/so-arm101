"""The twin finds the lab's touch strategy, builds it from its settings and touches a simulated cube.

Everything goes through the twin's own code: discovery and strict settings (`machinekind_twin.strategies`),
the run (`StrategyRunner`), the safety supervisor and the twin's physics (`Twin`, simulated, its loop
without a thread on a virtual clock). The strategy package itself imports nothing from the twin.

The poses come from the simulation, as the panel's "symulacja" gives them, or are made coarse the way the
LLM detector's are: off by a centimetre or more, at the table top, "ostatnie widziane", seconds old. The
runs are judged by the simulation, not by the strategy's own claim: the contact sensors the twin puts
between the cube and the jaws (`grasp_sensors`), and how far the touch pushed and tilted the cube. Other
runs put the panel's speed cap on the arm, at the start or during the run, or a fixed wall in its way,
after which the arm must hold where it measures.

`touch_jev` runs here with a judge that answers from the same contact sensors instead of Jev, or with the
real Jev client on a fake HTTP transport; no test needs TYPESAFE_API_KEY.

The twin's robot discovery is pinned to `soarm`'s folder (fixture `twin_robots`): the twin registers an
SO-101 of its own under the same name.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.twin
pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as twin_scene  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.detectors import detectors  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import ObjectPose, Observation, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.strategies.jev import KEY_ENV, JevJudge  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.touch import TouchSettings, TouchStrategy  # noqa: E402
from soarm_lab.strategies.touch_jev import CORRECTIONS, VERDICTS, JevTouchSettings, JevTouchStrategy  # noqa: E402

NAME = "touch"
TOUCH_FIELDS = ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                "max_tracking_deg", "time_limit_s"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: The cube: half its side [m], its place on the table [m, base frame] and its mass [kg].
HALF = 0.015
CUBE_XY = (0.2, 0.0)
CUBE_MASS = 0.03

Objects = Callable[[], dict[str, ObjectPose]]


@pytest.fixture
def twin(twin_robots):
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


@dataclass
class Clock:
    """The test's simulated time [s]: the runner's ticks and the made-up detector both read it."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], float] | None = None) -> float:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock.

    Stops early once the run is over or `until()` holds. Returns the highest value `watch()` gave after
    any physics step.
    """
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    most = 0.0
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
            if watch is not None:
                most = max(most, watch())
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
        if not runner.status.running or (until is not None and until()):
            break
    return most


def add_cube(twin: Twin, xy: tuple[float, float] = CUBE_XY) -> None:
    """A 3 cm cube on the table, with contact sensors against both jaws. Before `connect`."""
    twin.configure(objects=[twin_scene.Box("cube", (HALF, HALF, HALF), xy, rgba=(0.85, 0.25, 0.2, 1.0),
                                           mass=CUBE_MASS)], grasp_sensors=["cube"])


def truth(twin: Twin) -> tuple[np.ndarray, np.ndarray]:
    """The cube's position and rotation in the base frame, as the simulation has them."""
    with twin.lock:
        s = twin.scene
        b = s.model.body("cube").id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def fixed_pose(xy: tuple[float, float]) -> Objects:
    """A pose at the table top that never changes, as "symulacja" gives a cube that does not move."""
    pose = {"cube": ObjectPose(np.array([*xy, 0.0]), np.eye(3), "symulacja")}
    return lambda: pose


def sim_cube(twin: Twin) -> dict[str, ObjectPose]:
    """The cube as the twin's simulation knows it (what the panel's "symulacja" gives)."""
    position, rotation = truth(twin)
    return {"cube": ObjectPose(position, rotation, "symulacja")}


def coarse(twin: Twin, clock: Clock, offset: tuple[float, float], age_s: float = 8.0,
           appear_s: float = 0.0) -> Objects:
    """The cube as the LLM detector gives it: seen once at `appear_s`, `offset` off in x and y, at the
    table top, with no orientation, as "ostatnie widziane" `age_s` old and ageing from then on."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if clock.t < appear_s:
            return {}
        if not seen:
            position, _ = truth(twin)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        age = age_s + clock.t - appear_s
        return {"cube": ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age)}

    return objects


def contact(twin: Twin) -> float:
    """The most contacts between the cube and either jaw right now (each sensor counts contacts)."""
    with twin.lock:
        s = twin.scene
        return max(float(s.data.sensordata[s.model.sensor(f"cube_jaw{k}").adr[0]]) for k in range(2))


@dataclass
class SceneTouch:
    """What the simulation says about a run: the cube's contact with the hand, and how the cube moved."""

    contact_at_end: bool
    contact_settled: bool
    most_contacts: float
    push_m: float
    tilt_deg: float

    @property
    def ok(self) -> bool:
        """A touch: the hand rests on the cube when the run ends and half a second later, and the cube
        moved less than 1 cm and tilted less than 20 degrees."""
        return self.contact_at_end and self.contact_settled and self.push_m < 0.01 and self.tilt_deg < 20.0


def run_and_judge(twin: Twin, runner: StrategyRunner, clock: Clock, seconds: float = 20.0,
                  until: Callable[[], bool] | None = None) -> SceneTouch:
    """Drives the run to its end, then 0.5 s more physics, and judges it from the simulation."""
    start_position, start_rotation = truth(twin)
    most = drive(twin, runner, seconds, clock, until=until, watch=lambda: contact(twin))
    assert not runner.status.running, runner.status
    at_end = contact(twin) > 0.5
    for _ in range(round(0.5 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    settled = contact(twin) > 0.5
    position, rotation = truth(twin)
    tilt = np.degrees(np.arccos(np.clip(start_rotation[:, 2] @ rotation[:, 2], -1.0, 1.0)))
    push = float(np.linalg.norm((position - start_position)[:2]))
    return SceneTouch(at_end, settled, max(most, contact(twin)), push, float(tilt))


class TryAgain(TouchStrategy):
    """A stand-in for a decision step: after a descent with no contact it aims at the cube's true place."""

    def __init__(self, settings: TouchSettings, cube: Callable[[], tuple[float, float]]):
        super().__init__(settings)
        self.cube = cube

    def after_touch(self, obs: Observation):
        if not self.status.contact and self.status.attempt == 1:
            self.retarget(*self.cube())
            return self.hold()
        return super().after_touch(obs)


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_the_strategy_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.touch:TouchStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is TouchStrategy
    assert plugin.settings is TouchSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.touch:TouchStrategy)"
    assert plugin.summary.startswith("Touches an object")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == TOUCH_FIELDS
    assert fields["label"].kind is str and fields["label"].default == "cube"
    assert fields["close_jaw"].kind is bool and fields["close_jaw"].default is True
    assert fields["need_contact"].kind is bool and fields["need_contact"].default is True
    floats = [name for name in fields if name not in ("label", "close_jaw", "need_contact")]
    assert all(fields[name].kind is float for name in floats)
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji touch: \['lable'\]"):
        plugin.create({"lable": "red pen"})
    with pytest.raises(ValueError, match="touch.hover_m: potrzeba float"):
        plugin.create({"hover_m": "0.06"})

    strategy = plugin.create({"label": "Red Pen", "hover_m": 1, "hz": 25})
    assert isinstance(strategy, TouchStrategy)
    assert strategy.label == "red pen"
    assert strategy.config.hover_m == 1.0 and isinstance(strategy.config.hover_m, float)
    assert strategy.hz == 25.0


def test_the_twin_group_installs_the_llm_detector():
    """The detector `touch` is built for, from the pinned twin revision."""
    assert detectors()["llm"].value == "machinekind_twin_llm:LlmDetector"
    assert importlib.metadata.version("machinekind-twin-llm") == "0.1.0"


# --------------------------------------------------------------------------- touches in the simulated twin


def test_touch_finds_the_cube_from_its_true_pose(twin):
    add_cube(twin)
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    assert twin.owner == "strategia"
    scene = run_and_judge(twin, runner, Clock())

    assert runner.status.done and runner.status.success
    assert runner.status.stopped_because == "zadanie wykonane"
    assert strategy.status.stopped_because.startswith("obiekt dotkniety")
    assert strategy.status.contact and strategy.status.attempt == 1
    assert scene.ok, (scene, strategy.status)
    assert twin.owner is None


def test_touch_waits_for_a_coarse_late_pose_and_finds_the_cube(twin):
    """The LLM detector's first answer comes after 4 s, 1 cm off in x and y and 8 s old."""
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, clock, (0.01, 0.01), appear_s=4.0))

    runner.start(threaded=False)
    drive(twin, runner, 3.0, clock)
    assert runner.status.running and strategy.status.phase == "wait"
    assert runner.status.note.startswith("czekam na polozenie obiektu 'cube'")
    scene = run_and_judge(twin, runner, clock)

    assert runner.status.stopped_because == "zadanie wykonane"
    assert strategy.status.pose_source == "ostatnie widziane" and strategy.status.pose_age_s >= 8.0
    assert scene.ok, (scene, strategy.status)


def test_a_pose_3_cm_off_is_an_honest_miss(twin):
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, clock, (0.0, -0.03)))

    runner.start(threaded=False)
    scene = run_and_judge(twin, runner, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because.startswith("brak kontaktu")
    assert scene.most_contacts <= 0.5 and scene.push_m < 0.001


def test_a_decision_step_aims_again_after_a_miss_and_touches_the_cube(twin):
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    strategy = TryAgain(TouchSettings(), cube=lambda: tuple(truth(twin)[0][:2]))
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, clock, (0.0, 0.03)))

    runner.start(threaded=False)
    scene = run_and_judge(twin, runner, clock, seconds=30.0)

    assert runner.status.stopped_because == "zadanie wykonane"
    assert strategy.status.attempt == 2 and strategy.status.contact
    assert scene.ok, (scene, strategy.status)


@pytest.mark.parametrize("xy", [(0.2, 0.0), (0.22, 0.05)])
def test_under_the_panels_speed_cap_a_descent_in_free_air_is_a_miss(twin, xy):
    """The twin app caps the arm joints at 20 deg/s from its start (`Twin.set_speed`), and `RunInfo.max_vel`
    says so: the targets keep to it, so the run neither ends on lag nor reads a lagging command as a touch."""
    twin.set_speed(20.0)
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=fixed_pose(xy))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, Clock())

    assert runner.status.stopped_because == "strategia skonczyla", strategy.status
    assert strategy.status.stopped_because.startswith("brak kontaktu"), strategy.status


def test_under_the_panels_speed_cap_the_cube_is_touched(twin):
    twin.set_speed(20.0)
    add_cube(twin)
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    scene = run_and_judge(twin, runner, Clock(), seconds=60.0)

    assert runner.status.stopped_because == "zadanie wykonane", strategy.status
    assert scene.ok, (scene, strategy.status)


def test_a_speed_cap_lowered_during_the_run_never_reads_as_a_touch(twin):
    """The operator lowers the cap to 4 deg/s after the start, so `RunInfo.max_vel` does not know it. The
    supervisor's command then trails the targets, and the arm is measured against that command."""
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=fixed_pose((0.22, 0.05)))

    runner.start(threaded=False)
    twin.set_speed(4.0)
    drive(twin, runner, 60.0, Clock())

    assert runner.status.stopped_because == "strategia skonczyla", strategy.status
    assert strategy.status.stopped_because.startswith("brak kontaktu"), strategy.status


def test_on_lag_the_twin_holds_the_arm_where_it_measures(twin):
    """A fixed wall in the way from home: the arm stalls on it and the run ends on lag. The arm then holds
    where it measures, not at the command the supervisor had already sent past the wall."""
    twin.configure(objects=[twin_scene.Box("wall", (0.006, 0.06, 0.065), (0.215, 0.0), free=False)])
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=fixed_pose((0.2, 0.0)))

    runner.start(threaded=False)
    drive(twin, runner, 10.0, Clock())

    assert not runner.status.running and not runner.status.success
    assert strategy.status.stopped_because.startswith("ramie nie nadaza za celem"), strategy.status
    assert runner.status.stopped_because.endswith(strategy.status.stopped_because)
    for _ in range(round(2.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    arm = strategy.kin.spec.arm_joints
    status = twin.status
    assert status.state != "ESTOP" and twin.owner is None
    assert {n: status.command[n] for n in arm} == pytest.approx({n: status.measured[n] for n in arm}, abs=1.0)


def test_without_a_pose_the_run_ends_after_the_wait(twin):
    """The Strategies tab without a detector passes no objects: touch waits, then gives up."""
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({"wait_for_pose_s": 1.0})
    runner = StrategyRunner(twin, strategy)
    start = dict(twin.status.measured)

    runner.start(threaded=False)
    drive(twin, runner, 3.0, Clock())

    assert not runner.status.running
    assert runner.status.stopped_because == "strategia skonczyla"
    assert strategy.status.stopped_because.startswith("brak uzytecznego polozenia obiektu 'cube'")
    arm = strategy.kin.spec.arm_joints
    assert {n: twin.status.measured[n] for n in arm} == pytest.approx({n: start[n] for n in arm}, abs=0.5)


def test_the_time_limit_ends_a_run_that_still_waits(twin):
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({"time_limit_s": 1.0})
    runner = StrategyRunner(twin, strategy)

    runner.start(threaded=False)
    drive(twin, runner, 3.0, Clock())

    assert runner.status.stopped_because == "limit czasu" and not runner.status.done
    assert strategy.status.stopped_because == "limit czasu" and strategy.status.phase == "done"


def test_a_plain_stop_during_the_descent(twin):
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    drive(twin, runner, 10.0, clock, until=lambda: strategy.status.phase == "descend")
    assert runner.status.running and strategy.status.phase == "descend"
    runner.stop()

    assert runner.status.stopped_because == "zatrzymana"
    assert strategy.status.stopped_because == "zatrzymana" and strategy.status.phase == "done"
    assert twin.owner is None


# --------------------------------------------------------------------------- touch_jev


class ContactJudge:
    """Answers touch_jev's questions from the simulation's contact sensors instead of Jev: touched_object
    while a jaw touches the cube, else missed_object, with the confidences it is given."""

    name = "jev"

    def __init__(self, twin: Twin, sure: float = 0.95, correction: str = "aim_at_latest_sighting",
                 correction_sure: float = 0.9):
        self.twin, self.sure, self.correction, self.correction_sure = twin, sure, correction, correction_sure
        self.states: list[dict[str, Any]] = []
        self.verdicts: list[str] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.states.append(state)
        verdict = "touched_object" if contact(self.twin) > 0.5 else "missed_object"
        self.verdicts.append(verdict)
        answers = {"touched": choice(verdict, self.sure, VERDICTS),
                   "correction": choice(self.correction, self.correction_sure, CORRECTIONS)}
        return checked_reply(self.name, answers, "contact-sensors", {}, questions)


def choice(option: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    return {"type": "choice", "choice": option, "confidence": confidence,
            "probabilities": {o: top if o == option else (1 - top) / (n - 1) for o in options}}


def phases_of(strategy: TouchStrategy, phases: list[str]) -> Callable[[], bool]:
    """An `until` that never ends the run and keeps the strategy's phases in `phases`, repeats dropped."""

    def record() -> bool:
        if not phases or phases[-1] != strategy.status.phase:
            phases.append(strategy.status.phase)
        return False

    return record


def test_the_twin_finds_touch_jev_and_builds_it_without_a_key(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    plugin = load_strategy("touch_jev")
    fields = {field.name: field for field in plugin.fields}

    assert plugin.cls is JevTouchStrategy and plugin.settings is JevTouchSettings
    assert plugin.summary.startswith("Touches an object as `touch` does, and asks Jev after each descent")
    assert list(fields) == TOUCH_FIELDS + ["model", "min_confidence", "min_correction_confidence",
                                           "max_attempts", "correction_m", "answer_timeout_s", "send_frames"]
    assert fields["model"].kind is str and fields["model"].default == "jev-1.14.0"
    assert fields["max_attempts"].kind is int and fields["send_frames"].kind is bool
    assert fields["time_limit_s"].default == 90.0
    assert all(fields[name].kind is float for name in ("min_confidence", "min_correction_confidence",
                                                       "correction_m", "answer_timeout_s"))
    strategy = plugin.create({"max_attempts": 2, "model": "jev-latest"})
    assert isinstance(strategy, JevTouchStrategy) and strategy.jev_settings.max_attempts == 2


def test_without_the_key_touch_jev_refuses_the_run_before_the_arm_moves(twin, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, load_strategy("touch_jev").create({}))

    with pytest.raises(RuntimeError, match="brak klucza jev"):
        runner.start(threaded=False)
    assert runner.status.stopped_because.startswith(f"strategia odmowila: brak klucza jev: ustaw {KEY_ENV}")
    assert twin.owner is None


def test_jev_in_the_loop_aims_again_after_a_miss_and_touches_the_cube(twin):
    """The first pose is 3 cm off; once the first descent has started the simulation's truth comes in, as
    a camera seeing the cube again would. The judge says missed, the strategy rises, aims at the new
    sighting and touches."""
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    judge = ContactJudge(twin)
    strategy = JevTouchStrategy(JevTouchSettings(), judge=judge, threaded=False)
    first = coarse(twin, clock, (0.0, 0.03))
    runner = StrategyRunner(twin, strategy,
                            objects=lambda: sim_cube(twin) if strategy.status.attempt >= 1 else first())
    phases: list[str] = []

    runner.start(threaded=False)
    scene = run_and_judge(twin, runner, clock, seconds=40.0, until=phases_of(strategy, phases))

    assert runner.status.stopped_because == "zadanie wykonane", strategy.status
    assert strategy.status.stopped_because == "jev: obiekt dotkniety (pewnosc 0.95)"
    assert judge.verdicts == ["missed_object", "touched_object"] and strategy.status.attempt == 2
    # The recorder sees every tick but the last, which ends the run.
    assert phases[phases.index("touched"):] == ["touched", "lift", "approach", "descend", "touched"]
    assert strategy.status.phase == "done"
    assert judge.states[0]["latest_sighting"]["direction"] == "to_robot_right"      # base -y: the right
    assert judge.states[1]["aim"]["chosen_by"] == "correction:aim_at_latest_sighting"
    assert len(strategy.judging.decisions) == 4 and not strategy.judging.needs_review
    assert scene.ok, (scene, strategy.status)


def test_an_unsure_answer_stops_the_run_with_the_arm_holding_on_the_cube(twin):
    add_cube(twin)
    twin.connect("sim", threaded=False)
    clock = Clock()
    strategy = JevTouchStrategy(JevTouchSettings(), judge=ContactJudge(twin, sure=0.4), threaded=False)
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    scene = run_and_judge(twin, runner, clock)
    held = dict(twin.status.measured)
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because == ("jev niepewny: dotkniety z pewnoscia 0.40 < 0.80 - ramie "
                                               "stoi, do oceny czlowieka")
    assert strategy.judging.needs_review and strategy.status.attempt == 1
    arm = strategy.kin.spec.arm_joints
    assert {n: twin.status.measured[n] for n in arm} == pytest.approx({n: held[n] for n in arm}, abs=0.5)
    assert {n: twin.status.command[n] for n in arm} == pytest.approx({n: held[n] for n in arm}, abs=1.0)
    assert scene.contact_at_end and scene.contact_settled and twin.owner is None


def test_a_refused_key_ends_the_run_honestly_through_the_real_client(twin):
    """The Jev client on a fake transport that answers 401: no answer, no retry, the arm holds."""
    add_cube(twin)
    twin.connect("sim", threaded=False)
    sent: list[bytes] = []

    def refuse(url: str, headers: Any, body: bytes, timeout_s: float) -> tuple[int, bytes]:
        sent.append(body)
        return 401, b'{"detail": "Invalid API key"}'

    judge = JevJudge(transport=refuse, environ={KEY_ENV: "test-key"})
    strategy = JevTouchStrategy(JevTouchSettings(), judge=judge, threaded=False)
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    run_and_judge(twin, runner, Clock())

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because == (f"jev odrzucil klucz {KEY_ENV} (401) - ramie stoi, "
                                               "do oceny czlowieka")
    assert len(sent) == 1 and strategy.judging.needs_review and strategy.judging.answers == 0
    assert [d.confidence for d in strategy.judging.decisions] == [None, None]
