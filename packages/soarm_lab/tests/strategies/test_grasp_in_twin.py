"""The twin finds the lab's grasp strategy, builds it from its settings and lifts a simulated block.

Everything goes through the twin's own code, as in test_touch_in_twin.py: discovery and strict settings,
the run (`StrategyRunner`), the safety supervisor and the twin's physics on a virtual clock. The block is
one of the Classic World set (`rl.objects`), with contact sensors against both jaws.

The poses come from the twin's own `ObjectTracker` (`tracked`), fed with the simulation's truth while the
hand is more than 5 cm from the block, as the cameras would see it, and with nothing nearer, where the hand
hides it. The tracker says "kamery", then "ostatnie widziane", and "w dloni" once the jaw holds the block,
with the jaw thresholds of the robot description: the strategy's jaw rule is the same. Other runs give a
coarse pose 1 or 3 cm off, as the LLM detector does. The runs are judged by the simulation: where the block
is, how high, and whether both jaws touch it.

The twin's robot discovery is pinned to `soarm`'s folder (fixture `twin_robots`).
"""

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

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import COLOURS, KINDS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy, wrap  # noqa: E402

NAME = "grasp"
TOUCH_FIELDS = ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                "max_tracking_deg", "time_limit_s"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the block lies [m, base frame].
BLOCK_XY = (0.2, 0.0)
#: The hand hides the block from the cameras nearer than this [m].
HIDDEN_M = 0.05
#: The block's contact: the time constant twice the 5 ms step (MuJoCo's stiffest) and a high impedance.
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
    """The test's simulated time [s]: the runner's ticks and the tracker both read it."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock; `watch` after
    every physics step. Stops early once the run is over or `until()` holds."""
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
    """The block `kind` on the table with contact sensors against both jaws, then the simulated arm; the
    block turned by `yaw_deg` about the vertical.

    With `stiff` the block's contacts are made as stiff as the 5 ms step allows (`STIFF_SOLREF`,
    `STIFF_SOLIMP`, the gripper's priority, so they govern the block against the table and the jaws). Wood
    on a table does not give, but at the twin's default contact, which is what the panel's simulated block
    gets (`scene.Box` has no contact settings), the arm's push sinks the 11 g cube 8 mm into the table and
    the probe reaches its bottom without ever stalling. Stiffened, the cube sinks and gives under the sole
    by about 2.5 mm together, and under the jaws by about 1 mm each.
    """
    spec = object_spec(kind)
    twin.configure(objects=[twin_scene.Box(kind, spec.half_size, xy, rgba=COLOURS[kind], mass=spec.mass)],
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
            # The free joint is unnamed; the simulated table's base_yaw is 0: the world's yaw is the base's.
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
    """The contacts between the block and each jaw right now (fixed, moving)."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def tracked(twin: Twin, name: str, clock: Clock, sources: list[tuple[float, str, float]],
            label: str | None = None) -> Objects:
    """The block as the twin's tracker gives it: the simulation's truth while the hand is farther than
    `HIDDEN_M`, nothing nearer; "w dloni" while the jaw holds it. Each call records (t, source, age)."""
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
        # As the twin's tracks give it: a pose in the hand is current.
        age = 0.0 if tracker.source == "w dloni" else max(0.0, clock.t - tracker.t_seen)
        sources.append((clock.t, tracker.source, age))
        pose = ObjectPose(np.asarray(got[0], float), np.asarray(got[1], float), tracker.source, age)
        return {label or name: pose}

    return objects


def coarse(twin: Twin, name: str, clock: Clock, offset: np.ndarray, age_s: float = 8.0) -> Objects:
    """The block as the LLM detector gives it: seen once, `offset` off in x and y, at the table top, with no
    orientation, "ostatnie widziane" `age_s` old and ageing."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position, _ = truth(twin, name)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {name: ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def phases_of(strategy: GraspStrategy, phases: list[str]) -> Callable[[], bool]:
    """An `until` that never ends the run and keeps the strategy's phases in `phases`, repeats dropped."""

    def record() -> bool:
        if not phases or phases[-1] != strategy.status.phase:
            phases.append(strategy.status.phase)
        return False

    return record


def closing_heading() -> float:
    """The closing heading of the rest pose [rad]: the one a pose without rotation keeps."""
    _, closing = KIN.tool_axes(dict(KIN.spec.home, gripper=0.0))
    return math.atan2(float(closing[1]), float(closing[0]))


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_grasp_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.grasp:GraspStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is GraspStrategy and plugin.settings is GraspSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.grasp:GraspStrategy)"
    assert plugin.summary == ("Grasps and lifts a block after `touch`'s probe; the jaw rule checks the hold "
                              "on the way up.")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == TOUCH_FIELDS + ["object_kind", "open_margin_m", "squeeze_m", "grasp_depth",
                                           "lift_m", "hold_s", "max_attempts", "carry_s", "release", "wait_s",
                                           "recover_m"]
    assert fields["object_kind"].kind is str and fields["object_kind"].default == "cube"
    assert fields["object_kind"].choices == KINDS
    assert fields["label"].default == "" and fields["touch_z"].default == 0.0
    assert fields["hover_m"].default == 0.04 and fields["time_limit_s"].default == 120.0
    assert fields["max_attempts"].kind is int
    assert fields["close_jaw"].kind is bool and fields["need_contact"].kind is bool
    assert fields["release"].kind is bool and fields["release"].default is False
    assert fields["carry_s"].default == 10.0 and fields["wait_s"].default == 20.0
    assert fields["recover_m"].default == 0.15
    floats = [name for name in fields if name not in ("label", "object_kind", "close_jaw", "need_contact",
                                                      "max_attempts", "release")]
    assert all(fields[name].kind is float for name in floats)
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji grasp: \['kind'\]"):
        plugin.create({"kind": "cube"})
    with pytest.raises(ValueError, match="grasp: object_kind"):
        plugin.create({"object_kind": "ball"})
    strategy = plugin.create({"object_kind": "cuboid5"})
    assert isinstance(strategy, GraspStrategy) and strategy.block.kind == "cuboid5"
    assert strategy.label == "cuboid5"


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
    assert strategy.grasp.top_m == pytest.approx(0.025, abs=0.003)
    assert truth(twin, "cube")[0][2] > 0.04
    assert all(c > 0.5 for c in contact(twin, "cube"))
    # The squeeze outlives the run: the done tick's target was sent and disengaging keeps the command.
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
def test_at_the_twins_default_contact_the_probe_misses_and_need_contact_off_lifts_the_cube(twin,
                                                                                          need_contact: bool):
    """The block as the panel's simulation builds it today. The arm's push sinks the cube into the table
    instead of stalling, so the run ends as a miss; with `need_contact` off the probe's bottom counts, the
    grasp height comes from the block's nominal top, and the cube is lifted. When the twin's blocks get a
    contact of their own, the miss turns into a lift and the README's note on the panel goes."""
    add_block(twin, "cube", stiff=False)
    clock = Clock()
    strategy = load_strategy(NAME).create({"need_contact": need_contact})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))
    phases: list[str] = []
    lowest: list[float] = [1.0]

    def watch() -> None:
        lowest[0] = min(lowest[0], float(truth(twin, "cube")[0][2]))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=phases_of(strategy, phases), watch=watch)

    # The cube's centre lies 12.5 mm above the table at rest; the probe pressed it 8 mm lower.
    assert lowest[0] < 0.0125 - 0.006
    if need_contact:
        assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
        assert strategy.status.stopped_because == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"
        assert "rise" not in phases
    else:
        assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
        assert strategy.grasp.top_m == pytest.approx(0.025, abs=1e-9)
        assert truth(twin, "cube")[0][2] > 0.04
        assert all(c > 0.5 for c in contact(twin, "cube"))


@pytest.mark.parametrize("sign", [-1, 1], ids=["moving tip lands", "fixed tip lands"])
def test_a_coarse_pose_off_along_the_jaws_is_grasped_after_a_sideways_retarget(twin, sign: int):
    """1.2 cm off: an open tip overlaps the block by 4 mm at the straddle and lands on it. At 1 cm the
    fixed tip overlapped it by 2 mm, and the probe had pushed the block that far already."""
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
    assert strategy.grasp.straddle_stalls == 1
    assert strategy.status.attempt == (2 if sign < 0 else 3)
    assert max(pushed) < 0.015
    assert truth(twin, "cube")[0][2] > 0.04


def test_a_pose_3_cm_off_is_an_honest_miss(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.0, -0.03])))
    phases: list[str] = []

    runner.start(threaded=False)
    drive(twin, runner, 40.0, clock, until=phases_of(strategy, phases))

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because.startswith("brak kontaktu")
    assert "rise" not in phases


def test_a_cuboid_is_gripped_across_its_short_side(twin):
    add_block(twin, "cuboid5", yaw_deg=35.0)
    clock = Clock()
    strategy = load_strategy(NAME).create({"object_kind": "cuboid5"})
    assert isinstance(strategy, GraspStrategy)

    def objects() -> dict[str, ObjectPose]:
        position, rotation = truth(twin, "cuboid5")
        return {"cuboid5": ObjectPose(position, rotation, "symulacja")}

    runner = StrategyRunner(twin, strategy, objects=objects)
    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    across = math.degrees(wrap(math.radians(strategy.grasp.heading_deg - (35.0 + 90.0))))
    assert min(abs(across), 180.0 - abs(across)) < 5.0
    assert strategy.grasp.jaw_gap_m == pytest.approx(0.025, abs=0.003)
    assert truth(twin, "cuboid5")[0][2] > 0.04


def test_a_block_taken_away_after_the_probe_ends_without_success(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = load_strategy(NAME).create({})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, strategy, objects=lambda: {
        "cube": ObjectPose(truth(twin, "cube")[0], np.eye(3), "symulacja")} if strategy.status.phase == "wait"
        else {})

    runner.start(threaded=False)
    drive(twin, runner, 30.0, clock, until=lambda: strategy.status.phase == "rise")
    assert strategy.status.phase == "rise"
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
