"""Test wrist-camera grasp discovery, settings and motion through the twin.

Rendered frames come from the robot description's wrist camera. Coarse detector poses
have no orientation. Simulation poses and measured joint motion verify the grasp.
Only render-marked cases need OpenGL. `twin_robots` pins discovery to soarm's folder.
"""

from __future__ import annotations

import math
import time
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
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import Action, ObjectPose, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.blocks.classic_world import BLOCKS  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies import wrist  # noqa: E402
from soarm_lab.strategies.grasp import top_down  # noqa: E402
from soarm_lab.strategies.wrist import WristGraspSettings, WristGraspStrategy, fold  # noqa: E402

NAME = "grasp_wrist"
TOUCH_FIELDS = ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                "max_tracking_deg", "sag_m", "stall_s", "time_limit_s"]
GRASP_FIELDS = TOUCH_FIELDS + ["block", "probe", "open_margin_m", "squeeze_m", "grip_strength", "grasp_depth",
                               "straddle_push_m", "close_on_stall_m", "lift_m",
                               "hold_s", "max_attempts", "carry_s", "release", "wait_s", "recover_m",
                               "grasp_source", "library_dir", "min_score", "max_tilt_deg",
                               "jaw_gap_offset_m", "jaw_file"]
GRASP_WRIST_FIELDS = GRASP_FIELDS + ["wrist_camera", "max_frame_age_s", "frame_wait_s", "hue_deg",
                                     "label_hues", "hue_window_deg", "min_saturation", "min_value",
                                     "focal_px", "tip_pixel_x", "tip_pixel_y", "rotate_180", "centre_tol_m",
                                     "max_shift_m", "rounds", "min_fit_iou", "look_before_straddle",
                                     "close_rounds", "close_tol_m", "correction_gain", "residual_file"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the block lies [m, base frame].
BLOCK_XY = (0.2, 0.0)
#: Stiff contact uses a time constant twice the physics step and high impedance.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
SPEC = load_robot()
KIN = RobotKinematics(SPEC)
CAMERA = SPEC.cameras["wrist"]
#: Synthetic paints for the twin's test blocks [0..1].
PALETTE = ((0.85, 0.25, 0.2, 1.0), (0.95, 0.75, 0.15, 1.0), (0.2, 0.65, 0.35, 1.0), (0.2, 0.4, 0.9, 1.0))
PAINT = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(BLOCKS)}

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
    """Simulated time shared by the runner and pose source [s]."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """Run twin and strategy ticks on a shared simulated clock.

    Call `watch` after each strategy tick. Stop when the run ends or `until()` is true.
    """
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
            if watch is not None:
                watch()
        if not runner.status.running or (until is not None and until()):
            break


def add_block(twin: Twin, kind: str, xy: tuple[float, float] = BLOCK_XY, yaw_deg: float = 0.0) -> None:
    """Add a yawed block with stiff contacts on the table, then connect the simulated arm."""
    spec = block(kind)
    twin.configure(objects=[twin_scene.Box(kind, spec.half_size, xy, rgba=PAINT[kind], mass=spec.mass)],
                   grasp_sensors=[kind])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        g = model.geom(kind).id
        model.geom_solref[g] = STIFF_SOLREF
        model.geom_solimp[g][:2] = STIFF_SOLIMP
        model.geom_priority[g] = 1
        if yaw_deg:
            yaw = math.radians(yaw_deg)
            adr = model.jnt_qposadr[model.body_jntadr[model.body(kind).id]]
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


def coarse(twin: Twin, name: str, clock: Clock, offset: np.ndarray, age_s: float = 8.0) -> Objects:
    """Return one detector pose at table height, shifted by `offset` [m, base frame x and y].

    It has no orientation and starts at age `age_s` [s], which grows with the clock.
    """
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position, _ = truth(twin, name)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {name: ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def created(**settings) -> WristGraspStrategy:
    """Build the strategy through the twin with a shorter carry duration."""
    strategy = load_strategy(NAME).create({"carry_s": 1.0, **settings})
    assert isinstance(strategy, WristGraspStrategy)
    return strategy


def closing_heading(joints: dict[str, float]) -> float:
    """Return the measured closing heading [rad, base frame]."""
    _, closing = KIN.tool_axes(joints)
    return math.atan2(float(closing[1]), float(closing[0]))


def probe_hover(xy: tuple[float, float] = BLOCK_XY) -> dict[str, float]:
    """Return cube probe-hover joints with the rest pose's heading.

    The closed tips' midpoint is centred over `xy`.
    """
    home = dict(SPEC.home, gripper=0.0)
    psi = closing_heading(home)
    u = np.array([math.cos(psi), math.sin(psi)])
    tcp_xy = np.asarray(xy) - KIN.jaw_centre(0.0)[2] * u
    sol = KIN.ik(np.array([*tcp_xy, 0.0616]), top_down(psi), seed=home)
    assert sol.ok
    return dict(sol.joints, gripper=0.0)


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_grasp_wrist_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.wrist:WristGraspStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is WristGraspStrategy and plugin.settings is WristGraspSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.wrist:WristGraspStrategy)"
    assert plugin.summary == ("Grasps a block as `grasp` does, after the wrist camera centres the jaws on it "
                              "and sets the yaw.")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == GRASP_WRIST_FIELDS
    assert fields["wrist_camera"].kind is str and fields["wrist_camera"].default == "wrist"
    assert fields["rounds"].kind is int and fields["rounds"].default == 2
    assert fields["rotate_180"].kind is bool and fields["rotate_180"].default is False
    assert fields["look_before_straddle"].kind is bool and fields["look_before_straddle"].default is True
    assert fields["hue_deg"].default == -1.0 and fields["centre_tol_m"].default == 0.003
    floats = GRASP_WRIST_FIELDS[GRASP_WRIST_FIELDS.index("max_frame_age_s"):]
    floats = [name for name in floats if name not in ("rounds", "rotate_180", "look_before_straddle",
                                                      "close_rounds", "residual_file", "label_hues")]
    assert fields["label_hues"].kind is str and fields["label_hues"].default == ""
    assert fields["close_rounds"].kind is int and fields["close_rounds"].default == 3
    assert fields["residual_file"].kind is str and fields["residual_file"].default == ""
    assert all(fields[name].kind is float for name in floats)
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji grasp_wrist: \['camera'\]"):
        plugin.create({"camera": "wrist"})
    with pytest.raises(ValueError, match="grasp_wrist: rounds"):
        plugin.create({"rounds": 0})
    strategy = plugin.create({"block": "cuboid", "wrist_camera": ""})
    assert isinstance(strategy, WristGraspStrategy) and strategy.block.name == "cuboid"


# --------------------------------------------------------------------------- the camera in the twin


class Looker:
    """A strategy stand-in that holds the arm and keeps what it sees of the wrist camera."""

    hz = 10.0

    def __init__(self) -> None:
        self.seen: list[tuple[tuple[int, ...], np.dtype, float]] = []
        self.joints: dict[str, float] = {}

    def start(self, run, obs) -> None:
        self.joints = dict(obs.joints)

    def stop(self, reason: str) -> None:
        pass

    def act(self, obs) -> Action:
        frame = obs.frames["wrist"]
        self.seen.append((frame.image.shape, frame.image.dtype, float(frame.age_s)))
        return Action(target=self.joints, done=len(self.seen) >= 2, success=True)


@pytest.mark.render
def test_the_twin_renders_the_wrist_camera_of_the_so101(twin):
    twin.connect("sim", threaded=False)
    image, captured = twin.cameras.arm_frame_t("wrist")

    assert image is not None and image.shape == (CAMERA.height, CAMERA.width, 3) == (1080, 1920, 3)
    assert image.dtype == np.uint8 and 0.0 <= time.monotonic() - captured < 1.0
    looker = Looker()
    runner = StrategyRunner(twin, looker, cameras=("wrist",))
    runner.start(threaded=False)
    drive(twin, runner, 2.0, Clock())

    assert runner.status.stopped_because == "zadanie wykonane"
    assert [shape for shape, _, _ in looker.seen] == [(1080, 1920, 3)] * 2
    assert all(dtype == np.uint8 and 0.0 <= age < 1.0 for _, dtype, age in looker.seen)


@pytest.mark.render
def test_the_rendered_frame_matches_the_labs_camera_model(twin):
    """The lab's pinhole and camera pose place cube corners over the twin's rendered cube."""
    cv2 = pytest.importorskip("cv2")
    add_block(twin, "cube")
    joints = probe_hover()
    with twin.lock:
        twin.scene.set_joints(joints)
    image = twin.render("wrist")
    position, rotation = truth(twin, "cube")
    half = np.asarray(block("cube").half_size)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    pixels, _ = wrist.project(wrist.pinhole(CAMERA.fovy_deg, image.shape), KIN.camera("wrist", joints),
                              corners @ rotation.T + position)
    hue = __import__("colorsys").rgb_to_hsv(*PAINT["cube"][:3])[0] * 360.0
    red = wrist.segment(image, wrist.HsvWindow(hue, 20.0, 0.35, 0.25), cv2)
    rows, cols = np.nonzero(red)

    # Projected and rendered cube bounds agree at both resolutions.
    rendered = np.array([cols.min(), rows.min(), cols.max(), rows.max()], float)
    model = np.array([*pixels.min(axis=0), *pixels.max(axis=0)])
    assert rendered == pytest.approx(model, abs=9.0)


# --------------------------------------------------------------------------- grasps in the simulated twin


@pytest.mark.render
def test_a_cube_1_5_cm_off_is_centred_and_lifted(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = created()
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.012, 0.009])),
                            cameras=("wrist",))
    aimed: list[tuple[np.ndarray, np.ndarray]] = []

    def watch() -> None:
        # The aim and the cube when the look ended, before the straddle went down.
        if strategy.wrist.verdicts and not aimed and strategy.grasp.block_xy is not None:
            aimed.append((np.asarray(strategy.grasp.block_xy), truth(twin, "cube")[0][:2]))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.wrist)
    assert strategy.wrist.verdicts == ["centred"] and strategy.wrist.looks == 1
    assert strategy.wrist.where == "straddle"
    assert strategy.grasp.straddle_stalls == 0 and strategy.status.attempt == 1
    aim, cube = aimed[0]
    assert np.linalg.norm(aim - cube) < 0.0025
    assert truth(twin, "cube")[0][2] > 0.04
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    assert truth(twin, "cube")[0][2] > 0.04
    assert twin.owner is None


@pytest.mark.render
def test_a_yawed_cuboid_seen_without_a_rotation_is_gripped_across_its_short_side(twin):
    add_block(twin, "cuboid", yaw_deg=35.0)
    clock = Clock()
    strategy = created(block="cuboid")
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cuboid", clock, np.array([0.006, -0.008])),
                            cameras=("wrist",))
    straddle: list[float] = []

    def watch() -> None:
        if strategy.status.phase == "straddle" and not straddle:
            straddle.append(closing_heading(dict(twin.status.measured)))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.wrist)
    across = math.radians(35.0 + 90.0)
    assert abs(math.degrees(fold(math.radians(strategy.grasp.heading_deg) - across, math.pi))) < 4.0
    # Check the measured heading at straddle.
    assert abs(math.degrees(fold(straddle[0] - across, math.pi))) < 4.0
    assert strategy.grasp.jaw_gap_m == pytest.approx(2 * block("cuboid").half_size[1], abs=0.003)
    assert truth(twin, "cuboid")[0][2] > 0.04


@pytest.mark.render
def test_the_straddle_look_sees_the_push_of_the_probe(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = created(probe=True)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.zeros(2)),
                            cameras=("wrist",))
    looked: list[tuple[np.ndarray, np.ndarray]] = []

    def watch() -> None:
        if len(strategy.wrist.verdicts) == 2 and not looked and strategy.grasp.block_xy is not None:
            looked.append((np.asarray(strategy.grasp.block_xy), truth(twin, "cube")[0][:2]))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.wrist)
    assert strategy.wrist.where == "straddle" and np.all(np.isfinite(strategy.wrist.offset_m))
    aim, cube = looked[0]
    assert np.linalg.norm(aim - cube) < 0.002


def test_without_the_camera_in_the_run_the_looks_are_skipped_and_the_cube_is_still_lifted(twin):
    pytest.importorskip("cv2")
    add_block(twin, "cube")
    clock = Clock()
    strategy = created()
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.005, 0.0])))
    notes: list[str] = []

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, watch=lambda: notes.append(runner.status.note))

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert "centrowanie pominiete: brak klatki z kamery wrist" in notes
    assert strategy.wrist.verdicts == ["skipped"] and strategy.wrist.measurements == 0


@pytest.mark.render
def test_under_the_panels_speed_cap_the_cube_is_lifted(twin):
    twin.set_speed(20.0)
    add_block(twin, "cube")
    clock = Clock()
    strategy = created()
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.01, 0.0])),
                            cameras=("wrist",))

    runner.start(threaded=False)
    drive(twin, runner, 120.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.wrist)
    assert clock.t < 120.0 and strategy.wrist.verdicts == ["centred"]


@pytest.mark.render
def test_a_plain_stop_during_a_look(twin):
    add_block(twin, "cube")
    clock = Clock()
    strategy = created()
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, np.array([0.01, 0.0])),
                            cameras=("wrist",))

    runner.start(threaded=False)
    drive(twin, runner, 30.0, clock,
          until=lambda: strategy.wrist.looks == 1 and runner.status.note.startswith("centruje po kamerze"))
    assert runner.status.running and strategy.status.phase == "open"
    runner.stop()

    assert runner.status.stopped_because == "zatrzymana"
    assert strategy.status.stopped_because == "zatrzymana" and strategy.status.phase == "done"
    assert twin.owner is None
