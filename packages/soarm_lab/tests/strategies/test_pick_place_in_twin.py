"""Test pick-and-place through the twin's discovery, runner, supervisor and physics.

Each block has its own tracker fed with simulation poses while the hand is not near it. A fake
judge answers from the jaw contact sensors. Simulation poses check where the blocks end up.
The wrist-camera case needs rendering. No test here needs a model key.
`twin_robots` pins discovery to soarm's folder.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.twin
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as twin_scene  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.perception import CubeDetection, ObjectTracker  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import ObjectPose, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp_jev import HELD_OPTIONS, NEXT_OPTIONS  # noqa: E402
from soarm_lab.strategies.jev import KEY_ENV  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.pick_place import PickPlaceSettings, PickPlaceStrategy  # noqa: E402

NAME = "pick_place"
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the picked block and the target lie [m, base frame].
PICK_XY = (0.2, -0.04)
TARGET_XY = (0.2, 0.06)
#: The hand hides a block from the cameras nearer than this [m].
HIDDEN_M = 0.05
#: Stiff contact uses a time constant twice the physics step and high impedance.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
RED = (0.85, 0.25, 0.2, 1.0)
BLUE = (0.2, 0.4, 0.9, 1.0)
SPEC = load_robot()
KIN = RobotKinematics(SPEC)

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
    """Simulated time shared by the runner and trackers [s]."""

    t: float = 0.0


@dataclass(frozen=True)
class Placed:
    """A block for the scene: its name (also its label), catalog kind, position, yaw and paint."""

    name: str
    kind: str
    xy: tuple[float, float]
    yaw_deg: float = 0.0
    rgba: tuple[float, float, float, float] = RED


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """Run twin and strategy ticks on a shared simulated clock.

    Call `watch` after each physics step. Stop when the run ends or `until()` is true.
    """
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
            if watch is not None:
                watch()
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
        if not runner.status.running or (until is not None and until()):
            break


def add_blocks(twin: Twin, *blocks: Placed) -> None:
    """Add yawed blocks with stiff contacts and jaw sensors, then connect the simulated arm."""
    twin.configure(objects=[twin_scene.Box(b.name, block(b.kind).half_size, b.xy, rgba=b.rgba,
                                           mass=block(b.kind).mass) for b in blocks],
                   grasp_sensors=[b.name for b in blocks])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        for b in blocks:
            g = model.geom(b.name).id
            model.geom_solref[g] = STIFF_SOLREF
            model.geom_solimp[g][:2] = STIFF_SOLIMP
            model.geom_priority[g] = 1
            if b.yaw_deg:
                yaw = math.radians(b.yaw_deg)
                adr = model.jnt_qposadr[model.body_jntadr[model.body(b.name).id]]
                data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)


def truth(twin: Twin, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the simulated block position [m, base frame] and rotation."""
    with twin.lock:
        s = twin.scene
        b = s.model.body(name).id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def tilt_deg(rotation: np.ndarray) -> float:
    """How far a block's z axis leans from vertical [deg]."""
    return math.degrees(math.acos(min(1.0, abs(float(rotation[2, 2])))))


def yaw_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


def move(twin: Twin, name: str, xy: tuple[float, float], z: float) -> None:
    """Put a block at `xy` and centre height `z` at rest, as a person would."""
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        body = model.body(name).id
        adr = model.jnt_qposadr[model.body_jntadr[body]]
        T = twin.scene.T_base2world
        world = T[:3, :3] @ np.array([*xy, z]) + T[:3, 3]
        data.qpos[adr:adr + 3] = world
        data.qvel[model.body_dofadr[body]:][:6] = 0.0
        mujoco.mj_forward(model, data)


def contact(twin: Twin, name: str) -> list[float]:
    """Return the contacts with the fixed and moving jaws."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


@dataclass
class Cameras:
    """Tracker poses of every block from visible simulation detections.

    A block within `HIDDEN_M` of the TCP, or named in `hidden`, is not detected.
    """

    twin: Twin
    clock: Clock
    names: tuple[str, ...]
    hidden: set[str] = field(default_factory=set)
    sources: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = self.twin.workspace.spec()
        self.trackers = {n: ObjectTracker.for_spec(spec, hold_s=30.0) for n in self.names}
        spec = KIN.spec
        assert spec.gripper is not None
        self.g = spec.joints.index(spec.gripper)
        self.closed = float(KIN.to_q({spec.gripper: self.twin.joint_limits()[spec.gripper][0]})[self.g])

    def __call__(self) -> dict[str, ObjectPose]:
        status = self.twin.status
        measured, command = dict(status.measured), dict(status.command) or dict(status.measured)
        T_tcp = KIN.tcp(measured)
        out = {}
        for name, tracker in self.trackers.items():
            position, rotation = truth(self.twin, name)
            seen = None
            if name not in self.hidden and np.linalg.norm(position - T_tcp[:3, 3]) > HIDDEN_M:
                seen = CubeDetection(position, rotation, area_px=0.0, confidence=1.0, n_cameras=2,
                                     t=self.clock.t)
            grip_q, grip_cmd = float(KIN.to_q(measured)[self.g]), float(KIN.to_q(command)[self.g])
            got = tracker.update(seen, T_tcp, grip_q, grip_cmd, self.closed, self.clock.t)
            self.sources.setdefault(name, []).append((self.clock.t, tracker.source))
            if got is None:
                continue
            age = 0.0 if tracker.source == "w dloni" else max(0.0, self.clock.t - tracker.t_seen)
            out[name] = ObjectPose(np.asarray(got[0], float), np.asarray(got[1], float), tracker.source, age)
        return out


def choice(option: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    return {"type": "choice", "choice": option, "confidence": confidence,
            "probabilities": {o: top if o == option else (1 - top) / (n - 1) for o in options}}


class ContactJudge:
    """Answer close and lift questions from the picked block's jaw contacts."""

    name = "jev"

    def __init__(self, twin: Twin, picked: str):
        self.twin, self.picked = twin, picked
        self.points: list[str] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.points.append(state["point"])
        held = min(contact(self.twin, self.picked)) > 0.5
        answers = {"held": choice("block_held" if held else "nothing_in_jaws", 0.95, HELD_OPTIONS),
                   "next": choice("lift" if held else "regrasp_wider", 0.9, NEXT_OPTIONS)}
        return checked_reply(self.name, answers, "contact-sensors", {}, questions)


def built(judge: Any = None, **settings: Any) -> PickPlaceStrategy:
    settings.setdefault("judge", "none" if judge is None else "jev")
    settings.setdefault("wrist_camera", "")
    return PickPlaceStrategy(PickPlaceSettings(**settings), judge=judge, threaded=False)


def phases_of(strategy: PickPlaceStrategy, phases: list[str]) -> Callable[[], bool]:
    """Return an always-false stop predicate that records phase changes."""

    def record() -> bool:
        if strategy.started and (not phases or phases[-1] != strategy.status.phase):
            phases.append(strategy.status.phase)
        return False

    return record


def settle(twin: Twin, seconds: float = 1.0) -> None:
    for _ in range(round(seconds / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_pick_place_and_builds_it_without_a_key(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    assert strategies()[NAME].value == "soarm_lab.strategies.pick_place:PickPlaceStrategy"
    plugin = load_strategy(NAME)
    fields = {f.name: f for f in plugin.fields}

    assert plugin.cls is PickPlaceStrategy and plugin.settings is PickPlaceSettings
    assert plugin.summary == ("Picks a block and places it on another block or a table spot, then checks the "
                              "result in the cameras.")
    assert fields["judge"].default == "jev" and fields["place_on"].default == ""
    assert not any(f.required for f in fields.values())
    strategy = plugin.create({"block": "cuboid", "place_on": "blue", "judge": "none"})
    assert isinstance(strategy, PickPlaceStrategy) and strategy.target_label == "blue"


def test_with_judge_none_it_runs_without_a_key(twin, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, load_strategy(NAME).create({"judge": "none", "wrist_camera": ""}))
    runner.start(threaded=False)
    assert runner.status.running
    runner.stop()


# --------------------------------------------------------------------------- places in the simulated twin


def test_a_cube_goes_on_another_cube_and_stays_there(twin):
    """The placed cube sits flat and aligned on the lower one, which neither sinks nor moves."""
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("blue", "cube", TARGET_XY, 20.0, BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    cameras = Cameras(twin, clock, ("red", "blue"))
    runner = StrategyRunner(twin, strategy, objects=cameras)
    blue0, _ = truth(twin, "blue")
    phases: list[str] = []
    lowest: list[float] = []

    def watch() -> None:
        lowest.append(float(truth(twin, "blue")[0][2]))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, until=phases_of(strategy, phases), watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place, phases)
    assert strategy.place.outcome == "placed"
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, blue_rot = truth(twin, "blue")
    half = block("cube").half_size[2]
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004
    assert red[2] == pytest.approx(blue[2] + 2 * half, abs=0.001)
    assert blue[2] == pytest.approx(half, abs=0.0005) and min(lowest) > half - 0.0007
    assert np.linalg.norm(blue[:2] - blue0[:2]) < 0.002
    assert tilt_deg(red_rot) < 2.0 and tilt_deg(blue_rot) < 2.0
    assert abs(((yaw_deg(red_rot) - yaw_deg(blue_rot) + 45) % 90) - 45) < 4.0
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_third_cube_goes_on_two_and_the_open_jaw_leaves_the_stack_untouched(twin):
    """Near the arm's reach the jaw opens fully to leave; no block of the stack moves or tilts.

    The hand hides the target at the hover and its tracker takes it for the held block, so the latch
    uses the pose seen on the way there.
    """
    half = block("cube").half_size[2]
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("base", "cube", TARGET_XY, rgba=BLUE),
               Placed("blue", "cube", (0.2, 0.12), rgba=BLUE))
    move(twin, "blue", TARGET_XY, 3 * half)
    settle(twin)
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "base", "blue")))
    stack0 = {name: truth(twin, name)[0] for name in ("base", "blue")}
    shift: list[float] = []

    def watch() -> None:
        shift.append(max(float(np.linalg.norm(truth(twin, name)[0] - p0)) for name, p0 in stack0.items()))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed" and strategy.place.opened_wide
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, blue_rot = truth(twin, "blue")
    base, base_rot = truth(twin, "base")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004
    assert red[2] == pytest.approx(5 * half, abs=0.001) and blue[2] == pytest.approx(3 * half, abs=0.001)
    assert max(shift) < 0.002 and base[2] == pytest.approx(half, abs=0.0005)
    assert max(tilt_deg(r) for r in (red_rot, blue_rot, base_rot)) < 2.0
    assert abs(((yaw_deg(red_rot) - yaw_deg(blue_rot) + 45) % 90) - 45) < 4.0
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_cube_goes_to_a_table_spot(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED))
    clock = Clock()
    spot = (0.22, 0.07)
    strategy = built(label="red", place_x_m=spot[0], place_y_m=spot[1])
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red",)))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed"
    assert strategy.status.stopped_because.startswith("blok 'red' odlozony na stol")
    settle(twin)
    red, rotation = truth(twin, "red")
    assert np.linalg.norm(red[:2] - np.array(spot)) < 0.004
    assert red[2] == pytest.approx(block("cube").half_size[2], abs=0.0005) and tilt_deg(rotation) < 2.0


def test_a_cube_goes_on_a_yawed_cuboid_lined_up_with_it(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED),
               Placed("long", "cuboid", TARGET_XY, -30.0, BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="long", place_on_block="cuboid")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "long")))
    long0, _ = truth(twin, "long")

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    settle(twin)
    red, red_rot = truth(twin, "red")
    long, long_rot = truth(twin, "long")
    assert np.linalg.norm(red[:2] - long[:2]) < 0.004 and np.linalg.norm(long[:2] - long0[:2]) < 0.002
    assert red[2] == pytest.approx(2 * block("cuboid").half_size[2] + block("cube").half_size[2], abs=0.001)
    assert long[2] == pytest.approx(block("cuboid").half_size[2], abs=0.0005)
    assert abs(((yaw_deg(red_rot) - yaw_deg(long_rot) + 45) % 90) - 45) < 4.0 and tilt_deg(red_rot) < 2.0


def test_a_target_that_disappears_sends_the_cube_back_where_it_was(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("blue", "cube", TARGET_XY, rgba=BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    cameras = Cameras(twin, clock, ("red", "blue"))
    runner = StrategyRunner(twin, strategy, objects=cameras)

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=lambda: strategy.started and strategy.status.phase == "check")
    picked, _ = truth(twin, "red")
    cameras.hidden.add("blue")
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.place.outcome == "target_not_seen" and strategy.place.put_back
    assert strategy.status.stopped_because.endswith("blok odlozony z powrotem")
    settle(twin)
    red, rotation = truth(twin, "red")
    assert np.linalg.norm(red[:2] - picked[:2]) < 0.004
    assert red[2] == pytest.approx(block("cube").half_size[2], abs=0.0005) and tilt_deg(rotation) < 2.0
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_cube_that_falls_off_is_reported(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("blue", "cube", TARGET_XY, rgba=BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "blue")))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=lambda: strategy.started and strategy.status.phase == "park")
    move(twin, "red", (0.2, 0.12), block("cube").half_size[2])
    drive(twin, runner, 30.0, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.place.outcome == "fell_off", (strategy.status, strategy.place)
    assert strategy.status.stopped_because.startswith("blok 'red' nie lezy na 'blue'")
    assert strategy.place.height_error_m < -0.02


def test_a_cube_dropped_on_the_way_is_picked_again_and_placed(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("blue", "cube", TARGET_XY, rgba=BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue", wait_s=10.0)
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "blue")))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=lambda: strategy.started and strategy.status.phase == "carry_over")
    move(twin, "red", (0.18, 0.0), block("cube").half_size[2])
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.place, strategy.grasp)
    assert strategy.grasp.drops == 1 and strategy.grasp.recoveries == 1 and strategy.place.outcome == "placed"
    settle(twin)
    red, _ = truth(twin, "red")
    blue, _ = truth(twin, "blue")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004
    assert red[2] == pytest.approx(blue[2] + 2 * block("cube").half_size[2], abs=0.001)


def test_jev_judges_the_pick_and_the_cube_still_goes_on_top(twin):
    add_blocks(twin, Placed("red", "cube", PICK_XY, rgba=RED), Placed("blue", "cube", TARGET_XY, 15.0, BLUE))
    clock = Clock()
    judge = ContactJudge(twin, "red")
    strategy = built(judge, label="red", place_on="blue")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "blue")))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert judge.points == ["close", "lift"] and strategy.place.outcome == "placed"
    assert not strategy.judging.needs_review


@pytest.mark.render
def test_the_wrist_look_and_jev_pick_a_cube_and_place_it_on_another(twin):
    add_blocks(twin, Placed("red", "cube", (0.205, -0.05), 25.0, RED),
               Placed("blue", "cube", TARGET_XY, 10.0, BLUE))
    clock = Clock()
    judge = ContactJudge(twin, "red")
    strategy = built(judge, label="red", place_on="blue", wrist_camera="wrist")
    cameras = Cameras(twin, clock, ("red", "blue"))
    runner = StrategyRunner(twin, strategy, objects=cameras, cameras=("wrist",))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.place, strategy.wrist)
    assert strategy.wrist.verdicts[:1] == ["centred"] and judge.points == ["close", "lift"]
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, blue_rot = truth(twin, "blue")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004 and tilt_deg(red_rot) < 2.0
    assert red[2] == pytest.approx(blue[2] + 2 * block("cube").half_size[2], abs=0.001)
    assert abs(((yaw_deg(red_rot) - yaw_deg(blue_rot) + 45) % 90) - 45) < 4.0
