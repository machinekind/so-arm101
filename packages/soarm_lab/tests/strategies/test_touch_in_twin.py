"""The twin finds the lab's touch strategy, builds it from its settings and touches a simulated cube.

Everything goes through the twin's own code: discovery and strict settings (`machinekind_twin.strategies`),
the run (`StrategyRunner`), the safety supervisor and the twin's physics (`Twin`, simulated, its loop
without a thread on a virtual clock). The strategy package itself imports nothing from the twin.

The poses come from the simulation, as the panel's "symulacja" gives them, or are made coarse the way the
LLM detector's are: off by a centimetre or more, at the table top, "ostatnie widziane", seconds old. The
runs are judged by the simulation, not by the strategy's own claim: the contact sensors the twin puts
between the cube and the jaws (`grasp_sensors`), and how far the touch pushed and tilted the cube.

The twin's robot discovery is pinned to `soarm`'s folder (fixture `twin_robots`): the twin registers an
SO-101 of its own under the same name.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass

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

from soarm_lab.strategies.touch import TouchSettings, TouchStrategy  # noqa: E402

NAME = "touch"
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


def run_and_judge(twin: Twin, runner: StrategyRunner, clock: Clock, seconds: float = 20.0) -> SceneTouch:
    """Drives the run to its end, then 0.5 s more physics, and judges it from the simulation."""
    start_position, start_rotation = truth(twin)
    most = drive(twin, runner, seconds, clock, watch=lambda: contact(twin))
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

    assert list(fields) == ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                            "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m",
                            "contact_band_m", "max_tracking_deg", "time_limit_s"]
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
