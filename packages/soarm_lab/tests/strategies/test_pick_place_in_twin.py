"""Test pick-and-place through the twin's discovery, runner, supervisor and physics.

Each block has its own tracker fed with simulation poses while the hand is not near it. A fake
judge answers from the jaw contact sensors. Simulation poses check where the blocks end up.
The wrist-camera case needs rendering. No test here needs a model key.
`twin_robots` pins discovery to soarm's folder.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
from soarm_lab.planner import Place, Step, TargetPose, holds, resolve, world_state  # noqa: E402
from soarm_lab.planner.schema import SIDE_VECTORS  # noqa: E402
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
#: Where the base's turning axis crosses the table [m, base frame].
PAN_XY = KIN.joint_anchor(SPEC.arm_joints[0], SPEC.home)[:2]

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

    A block within `HIDDEN_M` of the TCP, or named in `hidden`, is not detected. With `nearest_grab`, only
    the block nearest the TCP can be taken by a closing jaw, as the twin's stacking tracker does.
    """

    twin: Twin
    clock: Clock
    names: tuple[str, ...]
    hidden: set[str] = field(default_factory=set)
    sources: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    nearest_grab: bool = False

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
        poses = {name: truth(self.twin, name) for name in self.trackers}
        nearest = min(poses, key=lambda n: float(np.linalg.norm(poses[n][0] - T_tcp[:3, 3])))
        out = {}
        for name, tracker in self.trackers.items():
            position, rotation = poses[name]
            seen = None
            if name not in self.hidden and np.linalg.norm(position - T_tcp[:3, 3]) > HIDDEN_M:
                seen = CubeDetection(position, rotation, area_px=0.0, confidence=1.0, n_cameras=2,
                                     t=self.clock.t)
            grip_q, grip_cmd = float(KIN.to_q(measured)[self.g]), float(KIN.to_q(command)[self.g])
            if self.nearest_grab and name != nearest:
                grip_cmd = grip_q
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
    assert plugin.summary == ("Picks a block and places it on another block or at a pose, then checks the "
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


def square(xy: tuple[float, float]) -> float:
    """The yaw [deg] that sets a block at `xy` square to the arm."""
    return math.degrees(math.atan2(xy[1] - PAN_XY[1], xy[0] - PAN_XY[0]))


def tower_of(twin: Twin, xy: tuple[float, float], under: int, red_yaw: float) -> list[str]:
    """Stack `under` blue cubes square to the arm at `xy`, with the red cube at `PICK_XY` and `red_yaw` [deg].
    Return the stack's names, bottom first."""
    half = block("cube").half_size[2]
    names = [f"base{k}" for k in range(under - 1)] + ["blue"]
    aside = [(0.12, 0.2), (0.12, -0.2), (0.3, 0.2)]
    add_blocks(twin, Placed("red", "cube", PICK_XY, red_yaw, RED),
               *(Placed(name, "cube", aside[k], square(xy), BLUE) for k, name in enumerate(names)))
    for k, name in enumerate(names):
        move(twin, name, xy, (2 * k + 1) * half)
        settle(twin, 0.5)
    settle(twin)
    return names


def without_yaw(objects: Objects, label: str) -> Objects:
    """`objects` with `label`'s rotation dropped, as for a block located without its outline."""

    def poses() -> dict[str, ObjectPose]:
        out = objects()
        if label in out:
            out[label] = replace(out[label], rotation=np.eye(3))
        return out

    return poses


@pytest.mark.parametrize("xy, under", [((0.24, 0.0), 2), ((0.21, 0.08), 3)], ids=["third far out", "fourth"])
def test_a_cube_out_of_straight_down_reach_goes_on_its_stack_with_the_hand_tilted(twin, xy, under):
    """The hand tilts for the pick and keeps the tilt, so the cube stays level and lands square; the stack
    neither moves nor sinks."""
    half = block("cube").half_size[2]
    names = tower_of(twin, xy, under, square(PICK_XY))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", *names)))
    stack0 = {name: truth(twin, name)[0] for name in names}
    shift: list[float] = []
    lowest: list[float] = []
    carried: list[float] = []

    def watch() -> None:
        shift.append(max(float(np.linalg.norm(truth(twin, name)[0] - p0)) for name, p0 in stack0.items()))
        lowest.append(float(truth(twin, names[0])[0][2]))
        if strategy.started and strategy.status.phase in ("carry_up", "carry_over", "latch", "place"):
            carried.append(tilt_deg(truth(twin, "red")[1]))

    runner.start(threaded=False)
    drive(twin, runner, 150.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed" and 0.0 < strategy.grasp.tilt_deg <= 30.0
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, blue_rot = truth(twin, "blue")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004
    assert red[2] == pytest.approx((2 * under + 1) * half, abs=0.001)
    assert blue[2] == pytest.approx((2 * under - 1) * half, abs=0.001)
    assert max(shift) < 0.002 and min(lowest) > half - 0.0004
    assert carried and max(carried) < 5.0
    assert max(tilt_deg(truth(twin, name)[1]) for name in ("red", *names)) < 2.0
    assert abs(((yaw_deg(red_rot) - yaw_deg(blue_rot) + 45) % 90) - 45) < 4.0
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_tilted_pick_closes_on_the_cube_without_pressing_it_into_the_table(twin):
    """The hand tilts for a place on a two-cube stack. The jaw closes on the picked cube slowly enough that
    the cube never sinks below its rest height, from the straddle through the lift."""
    half = block("cube").half_size[2]
    add_blocks(twin, Placed("red", "cube", (0.178, 0.041), 63.0, RED),
               Placed("base", "cube", (0.196, -0.033), 16.0, BLUE),
               Placed("blue", "cube", (0.2, 0.11), 9.0, BLUE))
    move(twin, "blue", (0.196, -0.033), 3 * half)
    settle(twin)
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, ("red", "base", "blue"),
                                                            nearest_grab=True))
    lowest: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.status.phase in ("straddle", "close", "lift_off"):
            lowest.append(float(truth(twin, "red")[0][2]))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed" and strategy.grasp.tilt_deg > 0.0
    assert lowest and min(lowest) > half - 0.0007, min(lowest)


@pytest.mark.render
@pytest.mark.parametrize("off", [0.0, 40.0], ids=["square", "40 deg off square"])
def test_a_tilted_pick_of_a_cube_without_yaw_in_its_pose_follows_the_wrist_look(twin, off):
    """The pose has no yaw, so the tilted pick guesses the cube square to the arm and the wrist camera checks
    it. A square cube goes on the stack; one 40 degrees off is left where it lies, untouched."""
    half = block("cube").half_size[2]
    names = tower_of(twin, (0.24, 0.0), 2, square(PICK_XY) + off)
    clock = Clock()
    strategy = built(label="red", place_on="blue", wrist_camera="wrist")
    cameras = without_yaw(Cameras(twin, clock, ("red", *names)), "red")
    runner = StrategyRunner(twin, strategy, objects=cameras, cameras=("wrist",))
    red0, red_rot0 = truth(twin, "red")
    stack0 = {name: truth(twin, name)[0] for name in names}
    shift: list[float] = []
    touched: list[float] = []

    def watch() -> None:
        shift.append(max(float(np.linalg.norm(truth(twin, name)[0] - p0)) for name, p0 in stack0.items()))
        touched.append(max(contact(twin, "red")))

    runner.start(threaded=False)
    drive(twin, runner, 150.0, clock, watch=watch)
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, _ = truth(twin, "blue")

    assert max(shift) < 0.002 and all(c < 0.5 for c in contact(twin, "red"))
    if not off:
        assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
        assert strategy.place.outcome == "placed" and strategy.grasp.tilt_deg > 0.0
        assert strategy.wrist.verdicts[:1] == ["centred"], strategy.wrist
        assert np.linalg.norm(red[:2] - blue[:2]) < 0.004 and red[2] == pytest.approx(5 * half, abs=0.001)
        assert tilt_deg(red_rot) < 2.0
        return
    assert strategy.place.outcome == "unreachable", (strategy.status, strategy.place, strategy.wrist)
    assert strategy.status.stopped_because.startswith("blok w kamerze obrocony o "), strategy.status
    assert max(touched) < 0.5 and np.linalg.norm(red - red0) < 0.001
    assert abs(yaw_deg(red_rot) - yaw_deg(red_rot0)) < 1.0


def stacked(twin: Twin, lower: Placed, *others: Placed) -> None:
    """Put a red cube on `lower` at the pick spot, with `others` around, and let the stack settle."""
    add_blocks(twin, lower, Placed("red", "cube", (0.2, 0.12), rgba=RED), *others)
    move(twin, "red", lower.xy, 2 * block(lower.kind).half_size[2] + block("cube").half_size[2])
    settle(twin)


def watched(twin: Twin, name: str) -> tuple[np.ndarray, list[float], list[float], Callable[[], None]]:
    """Return a block's position now, lists of its shift and height as the run goes, and the watch."""
    p0 = truth(twin, name)[0]
    shift: list[float] = []
    heights: list[float] = []

    def watch() -> None:
        p = truth(twin, name)[0]
        shift.append(float(np.linalg.norm(p - p0)))
        heights.append(float(p[2]))

    return p0, shift, heights, watch


def test_the_top_cube_of_a_two_cube_stack_goes_on_another_cube_and_the_lower_one_stays(twin):
    half = block("cube").half_size[2]
    stacked(twin, Placed("base", "cube", PICK_XY, rgba=BLUE), Placed("blue", "cube", TARGET_XY, 20.0, BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    cameras = Cameras(twin, clock, ("red", "base", "blue"), nearest_grab=True)
    runner = StrategyRunner(twin, strategy, objects=cameras)
    _, shift, heights, watch = watched(twin, "base")

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed" and strategy.base_z == pytest.approx(2 * half, abs=0.001)
    assert strategy.place.supports == ("base",) and not strategy.place.support_moved
    assert strategy.place.support_shift_m < 0.002
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, blue_rot = truth(twin, "blue")
    base, base_rot = truth(twin, "base")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004
    assert red[2] == pytest.approx(blue[2] + 2 * half, abs=0.001)
    assert max(shift) < 0.002 and min(heights) > half - 0.0004 and base[2] == pytest.approx(half, abs=0.0005)
    assert max(tilt_deg(r) for r in (red_rot, blue_rot, base_rot)) < 2.0
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_cube_comes_off_a_cuboid_to_a_table_spot_and_the_cuboid_stays(twin):
    stacked(twin, Placed("long", "cuboid", PICK_XY, -30.0, BLUE))
    clock = Clock()
    spot = (0.22, 0.07)
    strategy = built(label="red", place_x_m=spot[0], place_y_m=spot[1])
    cameras = Cameras(twin, clock, ("red", "long"), nearest_grab=True)
    runner = StrategyRunner(twin, strategy, objects=cameras)
    _, shift, heights, watch = watched(twin, "long")
    half = block("cuboid").half_size[2]

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.place)
    assert strategy.place.outcome == "placed" and strategy.place.supports == ("long",)
    assert not strategy.place.support_moved
    settle(twin)
    red, red_rot = truth(twin, "red")
    long, long_rot = truth(twin, "long")
    assert np.linalg.norm(red[:2] - np.array(spot)) < 0.004
    assert red[2] == pytest.approx(block("cube").half_size[2], abs=0.0005) and tilt_deg(red_rot) < 2.0
    assert max(shift) < 0.002 and min(heights) > half - 0.0004 and long[2] == pytest.approx(half, abs=0.0005)
    assert tilt_deg(long_rot) < 2.0


def test_a_target_that_disappears_sends_the_cube_back_onto_its_stack(twin):
    half = block("cube").half_size[2]
    stacked(twin, Placed("base", "cube", PICK_XY, rgba=BLUE), Placed("blue", "cube", TARGET_XY, rgba=BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue")
    cameras = Cameras(twin, clock, ("red", "base", "blue"), nearest_grab=True)
    runner = StrategyRunner(twin, strategy, objects=cameras)
    red0, _ = truth(twin, "red")
    _, shift, heights, watch = watched(twin, "base")

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=lambda: strategy.started and strategy.status.phase == "check",
          watch=watch)
    cameras.hidden.add("blue")
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.place.outcome == "target_not_seen" and strategy.place.put_back
    settle(twin)
    red, red_rot = truth(twin, "red")
    base, base_rot = truth(twin, "base")
    assert np.linalg.norm(red[:2] - red0[:2]) < 0.004 and red[2] == pytest.approx(3 * half, abs=0.001)
    assert max(shift) < 0.002 and min(heights) > half - 0.0004 and base[2] == pytest.approx(half, abs=0.0005)
    assert tilt_deg(red_rot) < 2.0 and tilt_deg(base_rot) < 2.0
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


@pytest.mark.render
def test_a_cube_pushed_after_the_wrist_look_is_centred_between_the_jaw_tips_and_still_placed(twin):
    add_blocks(twin, Placed("red", "cube", (0.205, -0.05), 25.0, RED),
               Placed("blue", "cube", TARGET_XY, 10.0, BLUE))
    clock = Clock()
    strategy = built(label="red", place_on="blue", wrist_camera="wrist")
    cameras = Cameras(twin, clock, ("red", "blue"))
    runner = StrategyRunner(twin, strategy, objects=cameras, cameras=("wrist",))
    pushed: list[np.ndarray] = []

    def watch() -> None:
        if strategy.status.phase == "straddle" and strategy.wrist.verdicts and not pushed:
            # Toward the fixed jaw, after the hover look and before the jaw goes down.
            psi = math.radians(strategy.grasp.heading_deg)
            push = -0.005 * np.array([math.cos(psi), math.sin(psi)])
            move(twin, "red", tuple(truth(twin, "red")[0][:2] + push), truth(twin, "red")[0][2])
            pushed.append(push)

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.place, strategy.wrist)
    looks = [(c.verdict, round(c.along_m * 1000)) for c in strategy.wrist.close_looks]
    assert looks[0] == ("moved", -5) and looks[-1][0] == "centred" and strategy.status.attempt == 1
    settle(twin)
    red, red_rot = truth(twin, "red")
    blue, _ = truth(twin, "blue")
    assert np.linalg.norm(red[:2] - blue[:2]) < 0.004 and tilt_deg(red_rot) < 2.0


# --------------------------------------------------------------------------- places at a pose
GREEN = (0.2, 0.75, 0.3, 1.0)


@contextmanager
def fresh_twin() -> Iterator[Twin]:
    """A new simulated twin, for tests that run several scenes."""
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def labelled(objects: Objects, labels: Mapping[str, str]) -> Objects:
    """`objects` keyed by `labels` (simulation name -> label), named by colour and shape as the twin does."""

    def poses() -> dict[str, ObjectPose]:
        return {labels.get(name, name): pose for name, pose in objects().items()}

    return poses


def truth_world(twin: Twin, labels: Mapping[str, str]):
    """The planner's world state from the simulation's poses, keyed by `labels`."""
    poses = {label: ObjectPose(*truth(twin, name), "kamery", 0.0) for name, label in labels.items()}
    return world_state(poses)


@dataclass(frozen=True)
class PoseScene:
    """Blocks for one place at a pose, the planner step and what came of it."""

    blocks: tuple[Placed, ...]
    labels: dict[str, str]
    step: Step


@dataclass
class PoseRun:
    """How a place at a pose ended: the outcome, the placed block's offset from the pose [m], its lean [deg],
    the largest shift of another block [m] and the planner's check of the relation."""

    outcome: str
    error_m: float
    tilt_deg: float
    others_shift_m: float
    holds: tuple[bool, str]
    pose: TargetPose

    @property
    def ok(self) -> bool:
        return self.error_m <= 0.008 and self.tilt_deg <= 3.0

    def line(self) -> str:
        return (f"{self.outcome:<14} error {self.error_m * 1000:4.1f} mm  tilt {self.tilt_deg:4.1f} deg  "
                f"others moved {self.others_shift_m * 1000:4.1f} mm  holds {self.holds[0]} ({self.holds[1]})")


def bridge_scene(seed: int) -> PoseScene:
    """A cuboid to lay across two cubes that stand a gap apart, lined up with the line between them."""
    rng = np.random.default_rng(seed)
    centre = np.array([0.2, 0.06]) + rng.uniform(-0.012, 0.012, 2)
    direction = rng.uniform(0.0, math.pi)
    gap = rng.uniform(0.015, 0.03)
    u = np.array([math.cos(direction), math.sin(direction)])
    d = (gap + 2 * block("cube").half_size[0]) / 2
    a, b = centre - d * u, centre + d * u
    cube_yaws = math.degrees(direction) + rng.uniform(-5.0, 5.0, 2)
    pick = (0.2 + rng.uniform(-0.015, 0.015), -0.06 + rng.uniform(-0.015, 0.015))
    blocks = (Placed("red", "cuboid", pick, float(rng.uniform(-90.0, 90.0)), RED),
              Placed("a", "cube", (float(a[0]), float(a[1])), float(cube_yaws[0]), BLUE),
              Placed("b", "cube", (float(b[0]), float(b[1])), float(cube_yaws[1]), GREEN))
    labels = {"red": "red cuboid", "a": "blue cube", "b": "green cube"}
    return PoseScene(blocks, labels, Step("red cuboid", Place("across", ("blue cube", "green cube"))))


def next_to_scene(seed: int) -> PoseScene:
    """A cube to set beside another cube, on a random side and with a random gap."""
    rng = np.random.default_rng(seed)
    side = str(rng.choice(list(SIDE_VECTORS)))
    gap = float(rng.uniform(0.005, 0.02))
    # The place lies near the middle of the reach, the target on its far side.
    place = np.array([0.205, 0.045]) + rng.uniform(-0.012, 0.012, 2)
    target = place - np.array(SIDE_VECTORS[side]) * (2 * block("cube").half_size[0] + gap)
    pick = (0.2 + rng.uniform(-0.015, 0.015), -0.07 + rng.uniform(-0.01, 0.01))
    yaws = rng.uniform(-45.0, 45.0, 2)
    blocks = (Placed("red", "cube", pick, float(yaws[0]), RED),
              Placed("blue", "cube", (float(target[0]), float(target[1])), float(yaws[1]), BLUE))
    labels = {"red": "red cube", "blue": "blue cube"}
    return PoseScene(blocks, labels, Step("red cube", Place("next_to", ("blue cube",), side, gap)))


def place_at_pose(twin: Twin, scene: PoseScene, on_carry: Callable[[Twin], None] | None = None,
                  **settings: Any) -> tuple[PoseRun, PickPlaceStrategy]:
    """Lay out `scene`, resolve its step with the planner from the simulation's poses, run pick_place to that
    pose with truth-fed trackers and measure the result. `on_carry` runs once as the carry starts."""
    add_blocks(twin, *scene.blocks)
    settle(twin)
    held, *others = (b.name for b in scene.blocks)
    pose = resolve(scene.step, truth_world(twin, scene.labels))
    supports = scene.step.place.targets if scene.step.place.relation in ("on", "across") else ()
    kind = next(b.kind for b in scene.blocks if b.name == held)
    clock = Clock()
    strategy = built(label=scene.labels[held], block=kind, place_x_m=pose.x_m, place_y_m=pose.y_m,
                     place_z_m=pose.z_m, place_yaw_deg=math.degrees(pose.yaw_rad),
                     supports=", ".join(supports), **settings)
    cameras = Cameras(twin, clock, tuple(scene.labels))
    runner = StrategyRunner(twin, strategy, objects=labelled(cameras, scene.labels))
    start = {name: truth(twin, name)[0] for name in others}
    shift: list[float] = [0.0]
    carried: list[bool] = []

    def watch() -> None:
        if on_carry is not None and not carried and strategy.started and strategy.status.phase == "carry_up":
            carried.append(True)
            on_carry(twin)
            start.update({name: truth(twin, name)[0] for name in others})
        shift.append(max(float(np.linalg.norm(truth(twin, n)[0] - p0)) for n, p0 in start.items()))

    runner.start(threaded=False)
    drive(twin, runner, 150.0, clock, watch=watch)
    settle(twin)
    position, rotation = truth(twin, held)
    error = float(np.linalg.norm(position[:2] - np.array([pose.x_m, pose.y_m])))
    after = truth_world(twin, scene.labels)
    return PoseRun(strategy.place.outcome, error, tilt_deg(rotation), max(shift),
                   holds(scene.step, after), pose), strategy


def test_a_cuboid_goes_across_two_cubes_as_a_bridge(twin):
    """The planner's pose for a cuboid across two cubes: the cuboid lands on both, level and lined up, and
    neither cube moves or sinks."""
    run, strategy = place_at_pose(twin, bridge_scene(0))

    assert run.outcome == "placed", (strategy.status, strategy.place)
    assert run.error_m < 0.004 and run.tilt_deg < 2.0, run.line()
    assert run.others_shift_m < 0.002 and run.holds[0], run.line()
    assert strategy.place.rests_on == ("blue cube", "green cube") and strategy.place.error_m < 0.004
    red, red_rot = truth(twin, "red")
    assert red[2] == pytest.approx(run.pose.z_m + block("cuboid").half_size[2], abs=0.001)
    assert abs(((yaw_deg(red_rot) - math.degrees(run.pose.yaw_rad) + 90) % 180) - 90) < 4.0
    for name in ("a", "b"):
        assert truth(twin, name)[0][2] == pytest.approx(block("cube").half_size[2], abs=0.0005)
    assert all(c < 0.5 for c in contact(twin, "red"))


def test_a_cube_goes_next_to_a_cube_with_the_jaw_clear_of_it(twin):
    """The jaw closes across the face that does not look at the other cube, so it neither touches nor moves
    it."""
    scene = next_to_scene(0)
    run, strategy = place_at_pose(twin, scene)

    assert run.outcome == "placed", (strategy.status, strategy.place)
    assert run.error_m < 0.004 and run.tilt_deg < 2.0 and run.holds[0], run.line()
    assert run.others_shift_m < 0.001, run.line()
    assert all(c < 0.5 for c in contact(twin, "blue"))
    red, _ = truth(twin, "red")
    assert red[2] == pytest.approx(block("cube").half_size[2], abs=0.0005)


def test_cubes_nudged_as_the_cuboid_is_carried_take_the_bridge_with_them(twin):
    """Both cubes move 6 mm as the carry starts: the latch sees it, the hover follows, and the cuboid lands
    across them where they stand now."""
    nudge = np.array([0.006, -0.002])

    def push(twin: Twin) -> None:
        for name in ("a", "b"):
            p = truth(twin, name)[0]
            move(twin, name, tuple(p[:2] + nudge), float(p[2]))

    run, strategy = place_at_pose(twin, bridge_scene(1), on_carry=push)

    assert run.outcome == "placed" and strategy.place.replans == 1, (strategy.status, strategy.place)
    red, red_rot = truth(twin, "red")
    assert np.linalg.norm(red[:2] - (np.array([run.pose.x_m, run.pose.y_m]) + nudge)) < 0.004
    assert tilt_deg(red_rot) < 2.0 and run.holds[0] and run.others_shift_m < 0.002, run.line()


@pytest.mark.parametrize("scene_of, needed", [(bridge_scene, 8), (next_to_scene, 9)],
                         ids=["across", "next_to"])
def test_seeded_places_at_a_pose_land_within_8_mm_and_level(twin_robots, scene_of, needed):
    """Ten seeded scenes per relation, each in a fresh twin: the placed block ends within 8 mm of the
    planner's pose and level in at least `needed` of them, and no other block moves."""
    runs = []
    for seed in range(10):
        with fresh_twin() as twin:
            runs.append(place_at_pose(twin, scene_of(seed))[0])
    report = "\n".join(f"seed {k}: {run.line()}" for k, run in enumerate(runs))
    print(report)
    assert sum(run.ok for run in runs) >= needed, report
    assert max(run.others_shift_m for run in runs) < 0.002, report
