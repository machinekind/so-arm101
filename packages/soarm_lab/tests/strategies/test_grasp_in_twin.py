"""Test grasp discovery, settings, motion and tracking through the twin's runner and physics.

The tracker receives simulation poses while the block is visible and no detections near the
hand. Other tests use a coarse detector pose. Simulation height and jaw contacts verify the
hold. The `twin_robots` fixture pins robot discovery to soarm's folder."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

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

from soarm_lab.blocks import block, random_rgba  # noqa: E402
from soarm_lab.blocks.classic_world import BLOCKS  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy, wrap  # noqa: E402

NAME = "grasp"
TOUCH_FIELDS = ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                "max_tracking_deg", "sag_m", "stall_s", "time_limit_s"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the block lies [m, base frame].
BLOCK_XY = (0.2, 0.0)
#: The hand hides the block from the cameras nearer than this [m].
HIDDEN_M = 0.05
#: Stiff contact uses a time constant twice the physics step and high impedance.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
KIN = RobotKinematics(load_robot())

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
    """Simulated time shared by the runner and tracker [s]."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """Run the twin and strategy at their own rates on a shared simulated clock.

    Call `watch` after each physics step. Stop when the run ends or `until()` is true."""
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


def add_block(twin: Twin, kind: str, xy: tuple[float, float] = BLOCK_XY, yaw_deg: float = 0.0,
              stiff: bool = True) -> None:
    """Add a block at `yaw_deg` with contact sensors for both jaws, then connect the simulated arm.

    `stiff` sets contact stiffness and priority for the block's table and jaw contacts."""
    spec = block(kind)
    rgba = random_rgba(np.random.default_rng(len(kind)))
    twin.configure(objects=[twin_scene.Box(kind, spec.half_size, xy, rgba=rgba, mass=spec.mass)],
                   grasp_sensors=[kind])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        if stiff:
            g = model.geom(kind).id
            model.geom_solref[g] = STIFF_SOLREF
            model.geom_solimp[g][:2] = STIFF_SOLIMP
            model.geom_priority[g] = 1
        if yaw_deg:
            yaw = math.radians(yaw_deg)
            adr = model.jnt_qposadr[model.body_jntadr[model.body(kind).id]]
            # The free joint is unnamed. World and base yaw coincide in this scene.
            data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            mujoco.mj_forward(model, data)


def truth(twin: Twin, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The block's position and rotation in the base frame, as the simulation has them."""
    with twin.lock:
        s = twin.scene
        b = s.model.body(name).id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def contact(twin: Twin, name: str) -> list[float]:
    """Return the current contacts with the fixed and moving jaws."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def tracked(twin: Twin, name: str, clock: Clock, sources: list[tuple[float, str, float]],
            label: str | None = None) -> Objects:
    """Return tracker poses from visible simulation detections.

    Hide detections within `HIDDEN_M`. Record (t, source, age) on each call."""
    tracker = ObjectTracker.for_spec(twin.workspace.spec(), hold_s=30.0)
    spec = KIN.spec
    assert spec.gripper is not None
    g = spec.joints.index(spec.gripper)
    closed = float(KIN.to_q({spec.gripper: twin.joint_limits()[spec.gripper][0]})[g])

    def objects() -> dict[str, ObjectPose]:
        status = twin.status
        measured, command = dict(status.measured), dict(status.command) or dict(status.measured)
        T_tcp = KIN.tcp(measured)
        position, rotation = truth(twin, name)
        seen = None
        if np.linalg.norm(position - T_tcp[:3, 3]) > HIDDEN_M:
            seen = CubeDetection(position, rotation, area_px=0.0, confidence=1.0, n_cameras=2, t=clock.t)
        got = tracker.update(seen, T_tcp, float(KIN.to_q(measured)[g]), float(KIN.to_q(command)[g]), closed,
                             clock.t)
        if got is None:
            sources.append((clock.t, tracker.source, float("nan")))
            return {}
        # A pose held in the hand stays current.
        age = 0.0 if tracker.source == "w dloni" else max(0.0, clock.t - tracker.t_seen)
        sources.append((clock.t, tracker.source, age))
        pose = ObjectPose(np.asarray(got[0], float), np.asarray(got[1], float), tracker.source, age)
        return {label or name: pose}

    return objects


def coarse(twin: Twin, name: str, clock: Clock, offset: np.ndarray, age_s: float = 8.0) -> Objects:
    """Return a single detector pose at table height, shifted by `offset` in x and y [m].

    The pose has no orientation and starts at age `age_s` [s]. Its age grows with the clock."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position, _ = truth(twin, name)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {name: ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def phases_of(strategy: GraspStrategy, phases: list[str]) -> Callable[[], bool]:
    """Return an always-false stop predicate that records phase changes in `phases`."""

    def record() -> bool:
        if not phases or phases[-1] != strategy.status.phase:
            phases.append(strategy.status.phase)
        return False

    return record


def closing_heading() -> float:
    """Return the rest pose's closing heading [rad], used for poses without orientation."""
    _, closing = KIN.tool_axes(dict(KIN.spec.home, gripper=0.0))
    return math.atan2(float(closing[1]), float(closing[0]))


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_grasp_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.grasp:GraspStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is GraspStrategy and plugin.settings is GraspSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.grasp:GraspStrategy)"
    assert plugin.summary == ("Grasps and lifts a block from above; the jaw rule checks the hold on the way "
                              "up.")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == TOUCH_FIELDS + ["block", "probe", "open_margin_m", "squeeze_m", "grasp_depth",
                                           "straddle_push_m", "close_on_stall_m",
                                           "lift_m", "hold_s", "max_attempts", "carry_s", "release", "wait_s",
                                           "recover_m", "grasp_source", "library_dir", "min_score",
                                           "max_tilt_deg", "held_gap_offset_m"]
    assert fields["block"].kind is str and fields["block"].default == "cube"
    assert fields["block"].choices == tuple(BLOCKS)
    assert fields["probe"].kind is bool and fields["probe"].default is False
    assert fields["grasp_source"].kind is str and fields["grasp_source"].default == "geometry"
    assert fields["grasp_source"].choices == ("geometry", "library")
    assert fields["library_dir"].kind is str and fields["library_dir"].default == ""
    assert fields["min_score"].default == 0.0 and fields["max_tilt_deg"].default == 0.0
    assert fields["label"].default == "" and fields["touch_z"].default == 0.0
    assert fields["hover_m"].default == 0.04 and fields["time_limit_s"].default == 120.0
    assert fields["max_attempts"].kind is int
    assert fields["close_jaw"].kind is bool and fields["need_contact"].kind is bool
    assert fields["release"].kind is bool and fields["release"].default is False
    assert fields["carry_s"].default == 10.0 and fields["wait_s"].default == 20.0
    assert fields["recover_m"].default == 0.15
    floats = [name for name in fields if name not in ("label", "block", "probe", "close_jaw", "need_contact",
                                                      "max_attempts", "release", "grasp_source",
                                                      "library_dir")]
    assert all(fields[name].kind is float for name in floats)
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji grasp: \['kind'\]"):
        plugin.create({"kind": "cube"})
    with pytest.raises(ValueError, match="grasp: block"):
        plugin.create({"block": "ball"})
    # A missing block library refuses the run before startup.
    with pytest.raises(ValueError, match="grasp: no grasp library for 'cuboid'"):
        plugin.create({"block": "cuboid", "grasp_source": "library"})
    strategy = plugin.create({"block": "cuboid"})
    assert isinstance(strategy, GraspStrategy) and strategy.block.name == "cuboid"
    assert strategy.label == "cuboid"


# --------------------------------------------------------------------------- grasps in the simulated twin


def test_grasp_lifts_the_cube_and_the_twin_says_w_dloni(twin):
    add_block(twin, "cube")
    clock = Clock()
    sources: list[tuple[float, str, float]] = []
    strategy = load_strategy(NAME).create({"carry_s": 1.0})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, sources))
    phases: list[str] = []
    holding: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.grasp.holding and not holding:
            holding.append(clock.t)

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=phases_of(strategy, phases), watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert strategy.grasp.verdict == "lifted" and strategy.status.success
    # The default path skips the probe phases.
    assert math.isnan(strategy.grasp.top_m) and not {"descend", "touched", "rise"} & set(phases)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))
    # Disengaging keeps the squeeze sent on the final tick.
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))

    before = [s for t, s, _ in sources if t < holding[0] - 1.0]
    assert "kamery" in before
    in_hand = [t for t, s, _ in sources if s == "w dloni"]
    assert in_hand and abs(in_hand[0] - holding[0]) <= 0.5
    assert {s for t, s, _ in sources if t >= in_hand[0]} == {"w dloni"}
    assert all(age == 0.0 for _, s, age in sources if s == "w dloni")
    assert twin.owner is None


@pytest.mark.parametrize("need_contact", [True, False], ids=["need_contact", "need_contact off"])
def test_at_the_twins_default_contact_the_probe_stops_on_the_cube_and_the_grasp_lifts_it(twin,
                                                                                        need_contact: bool):
    """The twin's own block contacts are stiff: the probe stalls on the top, the cube stays on the table."""
    add_block(twin, "cube", stiff=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({"need_contact": need_contact, "probe": True})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))
    phases: list[str] = []
    lowest: list[float] = [1.0]

    def watch() -> None:
        lowest[0] = min(lowest[0], float(truth(twin, "cube")[0][2]))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=phases_of(strategy, phases), watch=watch)

    half = block("cube").half_size[2]
    assert lowest[0] > half - 0.001, lowest[0]                        # pressed into the table by under 1 mm
    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp, phases)
    assert strategy.grasp.top_m == pytest.approx(2 * half, abs=0.002)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))


def test_at_the_twins_default_contact_the_default_grasp_lifts_the_cube(twin):
    """The default grasp lifts the cube with the twin's default contacts."""
    add_block(twin, "cube", stiff=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))


@pytest.mark.parametrize("sign", [-1, 1], ids=["moving tip lands", "fixed tip lands"])
def test_a_coarse_pose_off_along_the_jaws_is_grasped_after_a_sideways_retarget(twin, sign: int):
    """A coarse pose puts an open tip on the block. Sideways straddles recover the grasp.

    When the fixed tip lands, the first shift lands it on the block again and the other side holds."""
    add_block(twin, "cube")
    clock = Clock()
    psi = closing_heading()
    off = sign * 0.012 * np.array([math.cos(psi), math.sin(psi)])
    strategy = load_strategy(NAME).create({})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, off))
    start, _ = truth(twin, "cube")
    pushed: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.status.phase in ("wait", "lift", "approach", "descend", "touched",
                                                          "rise", "open", "straddle", "straddled", "close"):
            pushed.append(float(np.linalg.norm(truth(twin, "cube")[0][:2] - start[:2])))

    runner.start(threaded=False)
    drive(twin, runner, 90.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert strategy.grasp.straddle_stalls == (1 if sign < 0 else 2)
    assert strategy.status.attempt == (2 if sign < 0 else 3)
    assert max(pushed) < 0.015
    assert truth(twin, "cube")[0][2] > 0.04


#: A lower block standing against the cube along the jaws: its half sizes [m] and a mass the jaw cannot push.
STEP_HALF = (0.03, 0.0145, 0.010)
STEP_MASS = 2.0


def add_cube_by_a_step(twin: Twin, side: int) -> None:
    """The cube at `BLOCK_XY` and, 3 mm from it on the moving jaw's side (`side` 1) or the fixed jaw's
    (`side` -1), a heavy block lower than the cube, across the jaws."""
    cube = block("cube")
    psi = closing_heading()
    u = np.array([math.cos(psi), math.sin(psi)])
    xy = np.asarray(BLOCK_XY) + side * u * (cube.half_size[0] + 0.003 + STEP_HALF[1])
    twin.configure(objects=[twin_scene.Box("cube", cube.half_size, BLOCK_XY, rgba=(0.85, 0.25, 0.2, 1.0),
                                           mass=cube.mass),
                            twin_scene.Box("step", STEP_HALF, (float(xy[0]), float(xy[1])),
                                           rgba=(0.2, 0.4, 0.85, 1.0), mass=STEP_MASS)],
                   grasp_sensors=["cube"])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        for name in ("cube", "step"):
            g = model.geom(name).id
            model.geom_solref[g] = STIFF_SOLREF
            model.geom_solimp[g][:2] = STIFF_SOLIMP
            model.geom_priority[g] = 1
        yaw = psi + math.pi / 2
        adr = model.jnt_qposadr[model.body_jntadr[model.body("step").id]]
        data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)


@pytest.mark.parametrize("side", [1, -1], ids=["under the moving tip", "under the fixed tip"])
@pytest.mark.parametrize("close_on_stall_m", [0.003, 0.0], ids=["default", "off"])
def test_a_jaw_tip_stopped_low_beside_the_cube_closes_there(twin, side: int, close_on_stall_m: float):
    """One open tip comes down on the edge of a lower block standing against the cube, below the cube's top.
    The jaw closes where it stopped and lifts the cube. With `close_on_stall_m` off it reads as a tip on the
    cube, rises, tries beside and runs out of attempts."""
    add_cube_by_a_step(twin, side)
    clock = Clock()
    strategy = load_strategy(NAME).create({"carry_s": 1.0, "close_on_stall_m": close_on_stall_m})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.zeros(2)))
    step, _ = truth(twin, "step")

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    if close_on_stall_m:
        assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
        assert strategy.grasp.low_closes == 1 and strategy.grasp.straddle_stalls == 0
        assert strategy.status.attempt == 1
        assert truth(twin, "cube")[0][2] > 0.04 and all(c > 0.5 for c in contact(twin, "cube"))
    else:
        assert not runner.status.success and strategy.status.stopped_because.startswith("wyczerpane proby")
        assert strategy.grasp.low_closes == 0 and strategy.grasp.straddle_stalls == 3
    assert np.linalg.norm(truth(twin, "step")[0] - step) < 0.002


@pytest.mark.parametrize("probe", [False, True], ids=["default", "probe"])
def test_a_pose_3_cm_off_is_an_honest_miss(twin, probe: bool):
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({"probe": probe})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.0, -0.03])))
    phases: list[str] = []
    start, _ = truth(twin, "cube")

    runner.start(threaded=False)
    drive(twin, runner, 40.0, clock, until=phases_of(strategy, phases))

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    if probe:
        assert strategy.status.stopped_because.startswith("brak kontaktu")
    else:
        assert strategy.status.stopped_because == "chwyt pusty - blok przesuniety albo obok szczek"
    assert "rise" not in phases
    assert np.linalg.norm(truth(twin, "cube")[0][:2] - start[:2]) < 0.002


def test_a_cuboid_is_gripped_across_its_short_side(twin):
    add_block(twin, "cuboid", yaw_deg=35.0)
    clock = Clock()
    strategy = load_strategy(NAME).create({"block": "cuboid"})
    assert isinstance(strategy, GraspStrategy)

    def objects() -> dict[str, ObjectPose]:
        position, rotation = truth(twin, "cuboid")
        return {"cuboid": ObjectPose(position, rotation, "symulacja")}

    runner = StrategyRunner(twin, strategy, objects=objects)
    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    across = math.degrees(wrap(math.radians(strategy.grasp.heading_deg - (35.0 + 90.0))))
    assert min(abs(across), 180.0 - abs(across)) < 5.0
    assert strategy.grasp.jaw_gap_m == pytest.approx(2 * block("cuboid").half_size[1], abs=0.003)
    assert truth(twin, "cuboid")[0][2] > 0.04


def test_a_library_grasp_lifts_a_yawed_cube_and_the_twin_says_w_dloni(twin):
    """A bundled library grasp lifts a yawed cube and the tracker reports the hold."""
    add_block(twin, "cube", BLOCK_XY, yaw_deg=20.0)
    clock = Clock()
    sources: list[tuple[float, str, float]] = []
    strategy = load_strategy(NAME).create({"grasp_source": "library", "carry_s": 1.0})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, sources))
    phases: list[str] = []
    holding: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.grasp.holding and not holding:
            holding.append(clock.t)

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=phases_of(strategy, phases), watch=watch)

    grasp = strategy.grasp
    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, grasp)
    assert grasp.grasp_source == "library" and grasp.library_index == 0 and grasp.library_score == 0.92
    # The selected heading closes across a cube face.
    across = (grasp.heading_deg - 20.0) % 90.0
    assert min(across, 90.0 - across) < 5.0
    assert strategy.gap_open_m == pytest.approx(2 * block("cube").half_size[1] + 0.016)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))
    in_hand = [t for t, s, _ in sources if s == "w dloni"]
    assert holding and in_hand and abs(in_hand[0] - holding[0]) <= 0.5
    assert {s for t, s, _ in sources if t >= in_hand[0]} == {"w dloni"}


@pytest.mark.parametrize("probe", [False, True], ids=["before the straddle", "after the probe"])
def test_a_block_taken_away_ends_without_success(twin, probe: bool):
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({"probe": probe})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=lambda: {
        "cube": ObjectPose(truth(twin, "cube")[0], np.eye(3), "symulacja")} if strategy.status.phase == "wait"
        else {})
    taken_in = "rise" if probe else "open"

    runner.start(threaded=False)
    drive(twin, runner, 30.0, clock, until=lambda: strategy.status.phase == taken_in)
    assert strategy.status.phase == taken_in
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        adr = model.jnt_qposadr[model.body_jntadr[model.body("cube").id]]
        data.qpos[adr:adr + 2] += (0.0, 0.10)
        mujoco.mj_forward(model, data)
    drive(twin, runner, 30.0, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because == "chwyt pusty - blok przesuniety albo obok szczek"
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    arm = KIN.spec.arm_joints
    status = twin.status
    assert {n: status.measured[n] for n in arm} == pytest.approx({n: status.command[n] for n in arm}, abs=1.0)


def test_under_the_panels_speed_cap_the_cube_is_lifted(twin):
    twin.set_speed(20.0)
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    drive(twin, runner, 90.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", strategy
    assert clock.t < 90.0


def test_a_plain_stop_during_the_straddle(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    drive(twin, runner, 30.0, clock, until=lambda: strategy.status.phase == "straddle")
    assert runner.status.running and strategy.status.phase == "straddle"
    runner.stop()

    assert runner.status.stopped_because == "zatrzymana"
    assert strategy.status.stopped_because == "zatrzymana" and strategy.status.phase == "done"
    assert twin.owner is None
