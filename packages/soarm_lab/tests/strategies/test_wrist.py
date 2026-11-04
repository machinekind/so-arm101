"""Test wrist-camera geometry, segmentation, fitting and hover looks.

Synthetic frames contain block silhouettes and jaw hulls drawn from measured joints.
A fake supervisor, arm and jaw run the grasp. Render tests compare measurements with
MuJoCo images. `test_wrist_in_twin.py` covers the twin's runner and wrist camera.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, ObjectPose, Observation, RunInfo

import soarm

mujoco = pytest.importorskip("mujoco")

from soarm_lab.blocks import block, box_block  # noqa: E402
from soarm_lab.blocks.classic_world import BLOCKS  # noqa: E402
from soarm_lab.kinematics import IKSolution, RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies import wrist  # noqa: E402
from soarm_lab.strategies.grasp import (  # noqa: E402
    CLEAR_BEFORE_CLOSE_M,
    GraspSettings,
    GraspStrategy,
    heading,
    top_down,
)
from soarm_lab.strategies.wrist import (  # noqa: E402
    HsvWindow,
    WristGraspSettings,
    WristGraspStrategy,
    box_corners,
    fold,
    pinhole,
    rotation_z,
)

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
CAMERA = SPEC.cameras["wrist"]
#: Block position inside the downward reach band [m, base frame].
XY = (0.21, 0.02)
NAN = float("nan")
CUBE = block("cube").half_size
#: The synthetic frames' colours: the twin's table and its gold jaws [0..1].
TABLE = (0.55, 0.50, 0.44)
GOLD = (0.80, 0.65, 0.10)
#: Synthetic block paints [0..1] and hues [deg]. Test shapes take them in turn.
PALETTE = (((0.85, 0.25, 0.2), 4.6), ((0.95, 0.75, 0.15), 45.0), ((0.2, 0.65, 0.35), 140.0),
           ((0.2, 0.4, 0.9), 222.9))
PAINT = {name: PALETTE[i % len(PALETTE)][0] for i, name in enumerate(BLOCKS)}
#: Bare wood, which the saturation threshold leaves out [0..1].
WOOD = (0.84, 0.74, 0.65)
#: Synthetic frame size (rows, columns) [px], matching the working size.
SIZE = (360, 640)
#: Jaw model inputs are time [s], gripper command [app units] and straddle state.
Jaw = Callable[[float, float, bool], float]
#: Wrist frames from the measured joints of a tick.
Frames = Callable[[dict[str, float]], Mapping[str, Frame]]


def hue_of(colour: tuple[float, ...]) -> float:
    return __import__("colorsys").rgb_to_hsv(*colour[:3])[0] * 360.0


def window_of(colour: tuple[float, ...]) -> HsvWindow:
    return HsvWindow(hue_of(colour), 20.0, 0.35, 0.25)


def cv2_or_skip():
    return pytest.importorskip("cv2", reason="the wrist camera's tests need OpenCV")


# ------------------------------------------------------------ the fake arm (as in test_grasp.py)
def run_info(limits: dict[str, tuple[float, float]] | None = None, simulated: bool = True,
             max_vel: dict[str, float] | None = None) -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper,
                   limits=dict(SPEC.limits) if limits is None else limits,
                   max_vel=dict(SPEC.max_vel) if max_vel is None else max_vel,
                   folder=soarm.ROBOT_DIR, simulated=simulated)


def obs(joints: dict[str, float], t: float = 0.0, objects: Mapping[str, ObjectPose] | None = None,
        command: dict[str, float] | None = None, frames: Mapping[str, Frame] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=joints, command=command or {}, objects=objects or {},
                       frames=frames or {})


def seen(xy: tuple[float, float] = XY, label: str = "cube", source: str = "ostatnie widziane",
         age_s: float = 8.0, yaw_deg: float | None = None) -> dict[str, ObjectPose]:
    """Return a detector pose without orientation, or a box-fit pose with yaw."""
    rotation = np.eye(3) if yaw_deg is None else rotation_z(math.radians(yaw_deg))
    return {label: ObjectPose(position=np.array([*xy, 0.0], float), rotation=rotation, source=source,
                              age_s=age_s)}


def started(objects: dict[str, ObjectPose] | None = None, joints: dict[str, float] | None = None,
            strategy: type[GraspStrategy] = WristGraspStrategy, max_vel: dict[str, float] | None = None,
            **settings) -> WristGraspStrategy:
    settings.setdefault("carry_s", 0.5)
    if strategy is WristGraspStrategy:
        # Looks start with the block's paint unless a test samples the hue.
        settings.setdefault("hue_deg", hue_of(PAINT[settings.get("block", "cube")]))
    config = (WristGraspSettings if strategy is WristGraspStrategy else GraspSettings)(**settings)
    s = strategy(config)
    s.start(run_info(max_vel=max_vel), obs(dict(joints or SPEC.home), objects=objects))
    return s  # type: ignore[return-value]


def capped(cap_deg_s: float) -> dict[str, float]:
    """Return joint speed limits [deg/s] under the panel cap, with an optional gripper override."""
    return {name: value if name == SPEC.gripper else min(value, cap_deg_s)
            for name, value in SPEC.max_vel.items()}


def tcp(joints: Mapping[str, float]) -> np.ndarray:
    return KIN.tcp(joints)[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """Return phase changes in order."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def first_heading() -> float:
    """Return the rest pose closing heading [rad], used for poses without orientation."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


@dataclass
class Block:
    """A block on the table with the given dimensions and yaw."""

    xy: np.ndarray
    half: tuple[float, float, float]
    yaw: float = 0.0
    kind: str = "cube"
    #: A gripped block follows the fingertips and no longer stalls the arm.
    gripped: bool = False
    path: list[np.ndarray] = field(default_factory=list)

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def local(self, point: np.ndarray) -> np.ndarray:
        return rotation_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    def stalls(self, joints: Mapping[str, float]) -> bool:
        """Whether a fingertip would enter the block at `joints`."""
        if self.gripped:
            return False
        sole = KIN.tcp(joints)[2, 3] - KIN.jaw_reach(joints[SPEC.gripper])
        if sole >= self.height:
            return False
        return any(bool(np.all(np.abs(self.local(tip)) < self.half[:2])) for tip in KIN.fingertips(joints))

    def between(self, joints: Mapping[str, float]) -> bool:
        """Whether both tips lie below the top and straddle a block axis."""
        if self.gripped:
            return True
        tips = KIN.fingertips(joints)
        if any(tip[2] >= self.height for tip in tips):
            return False
        a, b = (self.local(tip) for tip in tips)
        return any(a[k] * b[k] < 0 and abs(a[1 - k]) < self.half[1 - k] and abs(b[1 - k]) < self.half[1 - k]
                   for k in (0, 1))

    def units(self, joints: Mapping[str, float]) -> float:
        """Return the gripper command spanning the block along the straddled axis [app units]."""
        a, b = (self.local(tip) for tip in KIN.fingertips(joints))
        k = int(np.argmax(np.abs(a - b)))
        return KIN.gripper_for_gap(min(2 * self.half[k], KIN.jaw_gap(100.0)))


def block_at(xy: tuple[float, float] = XY, kind: str = "cube", yaw_deg: float = 0.0) -> Block:
    return Block(np.array(xy, float), block(kind).half_size, math.radians(yaw_deg), kind)


def supervised(command: dict[str, float], target: Mapping[str, float],
               step_deg: float | None) -> dict[str, float]:
    """Advance each arm joint toward its target by at most `step_deg` [deg]."""
    out = dict(command)
    for name, value in target.items():
        if step_deg is None or name == SPEC.gripper:
            out[name] = float(value)
        else:
            out[name] = command[name] + float(np.clip(value - command[name], -step_deg, step_deg))
    return out


def follow(s: GraspStrategy, joints: dict[str, float], objects: Mapping[str, ObjectPose] | None, ticks: int,
           t0: float = 0.0, block: Block | None = None, cap_deg_s: float | None = None,
           frames: Frames | None = None, measured: list[dict[str, float]] | None = None,
           on_tick: Callable[[float], None] | None = None, until: Callable[[], bool] | None = None,
           held_wider_m: float = 0.0,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Run up to `ticks` ticks, stopping on `done`.

    The arm follows the supervisor command at once, stalling at `block`. The jaw closes to
    block width when the tips straddle it, then the block follows the fingertips. `frames`
    supplies wrist frames from measured joints. `on_tick` runs after each tick, and `until`
    can end the loop. A jaw squeezing the block reads `held_wider_m` wider than it is, as a
    gripper's calibration may. Return actions, phases, final joints and final time.
    """
    answers, phases = [], []
    command = dict(joints)
    t = t0
    g = SPEC.gripper
    assert g is not None
    shown: float | None = None
    for _ in range(ticks):
        if measured is not None:
            measured.append(joints)
        seen_now = frames(joints) if frames is not None else None
        read = joints if shown is None else dict(joints, **{g: shown})
        answers.append(a := s.act(obs(read, t=t, objects=objects, command=command, frames=seen_now)))
        phases.append(s.status.phase)
        if on_tick is not None:
            on_tick(t)
        t += 1 / s.hz
        command = supervised(command, a.target, None if cap_deg_s is None else cap_deg_s / s.hz)
        arm = dict(command)
        arm[g] = joints[g]
        between = block is not None and block.between(arm)
        grip = max(command[g], block.units(arm)) if block is not None and between else command[g]
        arm[g] = grip
        if block is not None:
            units = block.units(arm) if between else NAN
            if block.gripped and command[g] > units + 0.5:
                block.gripped = False
                arm[g] = command[g]
            elif between and not block.gripped and abs(grip - units) < 0.5 and command[g] < units - 0.5:
                block.gripped = True
            if block.stalls(arm):
                arm = dict(joints, **{g: arm[g]})
            if block.gripped:
                block.xy = np.mean(KIN.fingertips(arm), axis=0)[:2]
            block.path.append(block.xy.copy())
            squeezing = between and command[g] < units - 0.5
            wider = KIN.jaw_gap(units) + held_wider_m
            shown = KIN.gripper_for_gap(wider) if held_wider_m and squeezing else None
        joints = arm
        if a.done or (until is not None and until()):
            break
    return answers, phases, joints, t


# ------------------------------------------------------------ the synthetic wrist camera
def rgb8(colour: tuple[float, ...]) -> np.ndarray:
    return np.round(np.asarray(colour[:3], float) * 255).astype(np.uint8)


def synthetic_frame(joints: Mapping[str, float], kind: str, block_xy: np.ndarray | tuple[float, float],
                    yaw: float, *, focal_px: float = 0.0, age_s: float = 0.0,
                    extra_blocks: tuple[tuple[str, tuple[float, float], float], ...] = (), jaws: bool = True,
                    size: tuple[int, int] = SIZE, colour: tuple[float, ...] | None = None) -> Frame:
    """Draw the table, block silhouettes and gold jaw hulls from `joints`.

    `colour` overrides the first block's `PAINT` colour. `yaw` is in radians.
    Zero `focal_px` uses camera intrinsics.
    """
    cv2 = cv2_or_skip()
    image = np.empty((*size, 3), np.uint8)
    image[:] = rgb8(TABLE)
    T = KIN.camera("wrist", joints)
    K = pinhole(CAMERA.fovy_deg, size, focal_px)
    for i, (k, xy, angle) in enumerate(((kind, tuple(block_xy), yaw), *extra_blocks)):
        mask = wrist.silhouette(K, T, wrist.block_points(block(k), xy, angle), size, cv2)
        image[mask] = rgb8(colour if i == 0 and colour is not None else PAINT[k])
    if jaws:
        image[wrist.jaw_mask(KIN, joints, K, T, size, cv2, 0)] = rgb8(GOLD)
    return Frame(image=image, age_s=age_s)


class LazyFrames(Mapping[str, Frame]):
    """Draw observation frames only when read."""

    def __init__(self, draw: Callable[[], Frame | None], name: str = "wrist"):
        self._draw, self._name = draw, name
        self.reads = 0

    def __getitem__(self, key: str) -> Frame:
        if key != self._name:
            raise KeyError(key)
        self.reads += 1
        frame = self._draw()
        if frame is None:
            raise KeyError(f"{key}: brak klatki z kamery")
        return frame

    def __iter__(self) -> Iterator[str]:
        return iter([self._name])

    def __len__(self) -> int:
        return 1


def camera_of(block: Block, **kwargs) -> Frames:
    """Return block frames from each tick's measured joints."""
    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        return LazyFrames(lambda: synthetic_frame(joints, block.kind, block.xy, block.yaw, **kwargs))

    return frames


def hover_of(kind: str = "cube", xy: tuple[float, float] = XY, straddle: bool = False,
             yaw_deg: float | None = None, hue_deg: float | None = None, label: str = "", **settings
             ) -> tuple[WristGraspStrategy, dict[str, float]]:
    """Return a started strategy and measured joints at the probe or open-jaw straddle hover.

    The hover is reached without looking. `hue_deg` defaults to the hue of the block's `PAINT`. `label`
    defaults to the block's kind.
    """
    hue = hue_of(PAINT[kind]) if hue_deg is None else hue_deg
    s = started(seen(xy, label or kind, yaw_deg=yaw_deg), block=kind, probe=True, look_before_straddle=False,
                need_contact=False, hue_deg=hue, label=label, **settings)
    measured: list[dict[str, float]] = []
    phase = "straddle" if straddle else "descend"
    # Skip the probe look without frames and reach straddle with contact detection off.
    s.wrist_settings.frame_wait_s = 0.0
    _, phases, *_ = follow(s, dict(SPEC.home), seen(xy, label or kind, yaw_deg=yaw_deg), 2000,
                           measured=measured, until=lambda: s.status.phase == phase)
    assert phases[-1] == phase
    s.wrist_settings.frame_wait_s = 1.0
    return s, measured[-1]


def probe_hover(kind: str = "cube", xy: tuple[float, float] = XY) -> dict[str, float]:
    """Return IK joints at the probe hover for the selected block and `xy`.

    Centre closed tips over the block with the rest pose's heading and the hand down.
    The TCP is `hover_m` above the probe bottom.
    """
    config = GraspStrategy(GraspSettings(block=kind)).config
    psi = first_heading()
    tcp_xy = np.asarray(xy) - KIN.jaw_centre(0.0)[2] * np.array([math.cos(psi), math.sin(psi)])
    sol = KIN.ik(np.array([*tcp_xy, config.touch_z + config.hover_m]), top_down(psi),
                 seed=dict(SPEC.home, gripper=0.0))
    assert sol.ok
    return dict(sol.joints, gripper=0.0)


def frame_K_T(joints: Mapping[str, float], size: tuple[int, int] = SIZE, focal_px: float = 0.0
              ) -> tuple[np.ndarray, np.ndarray]:
    return pinhole(CAMERA.fovy_deg, size, focal_px), KIN.camera("wrist", joints)


def along_closing(joints: Mapping[str, float], metres: float) -> np.ndarray:
    """Return `metres` along the closing axis [m, base frame x and y], which points image-right."""
    psi = heading(KIN, joints)
    return metres * np.array([math.cos(psi), math.sin(psi)])


# ------------------------------------------------------------ the camera's geometry
def test_the_pinhole_follows_the_fovy_and_the_frame_height():
    assert pinhole(48.4555, (1080, 1920))[0, 0] == pytest.approx(1200.0, abs=0.01)
    assert pinhole(48.4555, (360, 640))[0, 0] == pytest.approx(400.0, abs=0.01)
    assert pinhole(48.4555, (480, 640))[1, 1] == pytest.approx(533.33, abs=0.01)
    K = pinhole(48.4555, (360, 640), focal_px=500.0)
    assert K[0, 0] == K[1, 1] == 500.0
    assert (K[0, 2], K[1, 2]) == ((640 - 1) / 2, (360 - 1) / 2)


def test_projection_and_back_projection_round_trip():
    joints = probe_hover()
    K, T = frame_K_T(joints)
    rng = np.random.default_rng(3)
    points = np.column_stack([XY[0] + rng.uniform(-0.04, 0.04, 50), XY[1] + rng.uniform(-0.04, 0.04, 50),
                              np.full(50, 0.025)])
    pixels, depth = wrist.project(K, T, points)

    assert np.all(depth > 0.05)
    assert wrist.back_project(K, T, pixels, 0.025) == pytest.approx(points, abs=1e-9)
    # Rays far below the frame point above the horizon.
    up = wrist.back_project(K, T, np.array([[320.0, 2000.0]]), 0.025)
    assert np.all(np.isnan(up))


def test_box_corners_stand_on_the_table_and_turn_with_the_yaw():
    half = block("cuboid").half_size
    corners = box_corners(half, (0.2, 0.05), math.radians(35.0))

    assert corners[:, 2].min() == 0.0 and corners[:, 2].max() == pytest.approx(2 * half[2])
    assert corners[:, :2].mean(axis=0) == pytest.approx([0.2, 0.05])
    long_side = corners[4, :2] - corners[0, :2]
    assert long_side / np.linalg.norm(long_side) == pytest.approx([math.cos(math.radians(35.0)),
                                                                   math.sin(math.radians(35.0))])
    assert np.linalg.norm(long_side) == pytest.approx(2 * half[0])


def test_a_cylinder_is_drawn_by_its_end_circles_and_a_box_by_its_corners():
    spec = block("cylinder")
    yaw = math.radians(-60.0)
    points = wrist.block_points(spec, (0.2, 0.05), yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    relative = points - np.array([0.2, 0.05, spec.half_size[2]])
    along = relative @ axis

    # Cylinder points sample both end rims and lie above the table.
    assert len(points) == 2 * wrist.RIM_POINTS
    assert np.abs(along) == pytest.approx(np.full(len(points), spec.half_size[0]))
    assert np.linalg.norm(relative - np.outer(along, axis), axis=1) == pytest.approx(spec.half_size[1])
    assert points[:, 2].min() == pytest.approx(0.0, abs=1e-12)
    assert points[:, 2].max() == pytest.approx(2 * spec.half_size[2])
    cuboid = block("cuboid")
    corners = box_corners(cuboid.half_size, (0.2, 0.05), yaw)
    assert sorted(map(tuple, np.round(wrist.block_points(cuboid, (0.2, 0.05), yaw), 12))) == \
        sorted(map(tuple, np.round(corners, 12)))


def test_fold_wraps_to_the_half_open_period():
    deg = math.radians
    assert fold(deg(100.0), deg(90.0)) == pytest.approx(deg(10.0))
    assert fold(deg(-50.0), deg(90.0)) == pytest.approx(deg(40.0))
    assert fold(deg(125.0), deg(180.0)) == pytest.approx(deg(-55.0))
    assert fold(deg(90.0), deg(180.0)) == pytest.approx(deg(90.0))
    assert fold(deg(-90.0), deg(180.0)) == pytest.approx(deg(90.0))


# ------------------------------------------------------------ the colour mask and the jaws
@pytest.mark.parametrize("colour, hue", PALETTE)
def test_a_hsv_window_holds_its_colour_and_not_the_table(colour: tuple[float, ...], hue: float):
    cv2 = cv2_or_skip()
    window = window_of(colour)
    image = np.empty((24, 24, 3), np.uint8)
    image[:] = rgb8(TABLE)
    image[4:16, 6:18] = rgb8(colour)

    mask = wrist.segment(image, window, cv2)
    assert mask.sum() == 144 and mask[4:16, 6:18].all()
    assert window.hue_deg == pytest.approx(hue, abs=0.1)
    if hue < 10.0:
        # Hue wrapping includes red pixels on both sides of zero.
        _, sat, val = __import__("colorsys").rgb_to_hsv(*colour[:3])
        other = np.array(__import__("colorsys").hsv_to_rgb(356 / 360, sat, val))
        image[4:16, 6:18] = np.round(other * 255).astype(np.uint8)
        assert wrist.segment(image, window, cv2).sum() == 144


def test_the_jaws_are_masked_out_of_the_colour_mask():
    cv2 = cv2_or_skip()
    joints = probe_hover("cuboid")
    image = synthetic_frame(joints, "cuboid", XY, math.radians(30.0)).image
    K, T = frame_K_T(joints)
    window = window_of(PAINT["cuboid"])
    gold = np.all(image == rgb8(GOLD), axis=2)
    colour = wrist.segment(image, window, cv2)
    jaws = wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)

    # Model hulls mask gold jaws that match the yellow block colour.
    assert gold.sum() > 1000 and (colour & gold).sum() > 0.9 * gold.sum()
    labels, found = wrist.blobs(colour & ~jaws, cv2)
    assert len(found) == 1 and not (colour & ~jaws & gold).any()


# ------------------------------------------------------------ the box fit
FIT_CASES = [("cube", 0.0, (0.0, 0.0)), ("cube", 20.0, (0.008, -0.006)), ("cuboid", 35.0, (0.0, 0.012)),
             ("cuboid", 125.0, (-0.01, 0.0)), ("cuboid", -70.0, (0.0, -0.01)),
             ("cylinder", -60.0, (0.015, 0.0))]


@pytest.mark.parametrize("kind, yaw_deg, offset", FIT_CASES, ids=[f"{k} {y:.0f}" for k, y, _ in FIT_CASES])
def test_the_fit_recovers_a_box_drawn_by_the_same_projection(kind: str, yaw_deg: float,
                                                            offset: tuple[float, float]):
    cv2 = cv2_or_skip()
    joints = probe_hover(kind)
    truth = np.asarray(XY) + offset
    image = synthetic_frame(joints, kind, truth, math.radians(yaw_deg)).image
    K, T = frame_K_T(joints)
    spec = block(kind)
    visible = ~wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)
    labels, found = wrist.blobs(wrist.segment(image, window_of(PAINT[kind]), cv2)
                                & visible, cv2)
    xy0, yaw0 = wrist.blob_pose(labels, found[0], K, T, 2 * spec.half_size[2], spec.yaw_period, cv2)
    fit = wrist.fit_block(labels == found[0].label, K, T, spec, xy0, yaw0, cv2, visible=visible)

    assert fit.xy == pytest.approx(truth, abs=5e-4)
    assert abs(math.degrees(fold(fit.yaw - math.radians(yaw_deg), spec.yaw_period))) < 1.0
    assert fit.iou > 0.95 and fit.evals < wrist.FIT_MAX_EVALS and fit.free == (False, False)


def test_the_fit_removes_the_side_face_bias():
    cv2 = cv2_or_skip()
    joints = probe_hover("cube")
    image = synthetic_frame(joints, "cube", XY, 0.0).image
    K, T = frame_K_T(joints)
    labels, found = wrist.blobs(wrist.segment(image, window_of(PAINT["cube"]),
                                              cv2), cv2)
    centroid = wrist.back_project(K, T, found[0].centroid_px, 2 * CUBE[2])[0, :2]
    xy0, yaw0 = wrist.blob_pose(labels, found[0], K, T, 2 * CUBE[2], math.pi / 2, cv2)
    fit = wrist.fit_block(labels == found[0].label, K, T, block("cube"), xy0, yaw0, cv2)

    # Silhouette fitting corrects centroid bias from the visible side face.
    assert np.linalg.norm(centroid - XY) > 0.003
    assert np.linalg.norm(fit.xy - XY) < 5e-4


def test_a_side_the_image_cannot_place_keeps_the_aims_position_along_it():
    """A long side hidden by the frame and jaws keeps the aim's position along that axis."""
    cv2 = cv2_or_skip()
    s, joints = hover_of("cuboid")
    # A cuboid longer than the view, with the catalog cuboid's cross-section.
    half = block("cuboid").half_size
    s.block = box_block("long cuboid", (85.0, 2000 * half[1], 2000 * half[2]), 1.0, yaw_period=math.pi)
    view = np.array([-math.sin(heading(KIN, joints)), math.cos(heading(KIN, joints))])   # image up and down
    yaw = math.atan2(view[1], view[0])
    truth = np.asarray(XY) + 0.006 * view + along_closing(joints, 0.004)
    K, T = frame_K_T(joints)
    image = np.empty((*SIZE, 3), np.uint8)
    image[:] = rgb8(TABLE)
    points = wrist.block_points(s.block, tuple(truth), yaw)
    image[wrist.silhouette(K, T, points, SIZE, cv2)] = rgb8(PAINT["cuboid"])
    image[wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, 0)] = rgb8(GOLD)
    m = s.measure(image, joints)

    assert m.problem == ""
    # Resolve the visible axis and keep the aim on the hidden axis.
    assert (m.xy - XY) @ view == pytest.approx(0.0, abs=2e-4)
    assert m.xy == pytest.approx(np.asarray(XY) + along_closing(joints, 0.004), abs=5e-4)


# ------------------------------------------------------------ which blob is the block
@pytest.mark.parametrize("case", ["one block 8 mm off", "a second block 4.5 cm off", "one block 7 cm off",
                                  "a distractor 7 cm off", "a quarter of the box"])
def test_measure_picks_the_blob_nearest_the_aim_and_reports_two_or_none(case: str):
    cv2 = cv2_or_skip()
    s, joints = hover_of("cube")
    side = along_closing(joints, 1.0)
    near = np.asarray(XY) + 0.008 * side
    if case == "one block 8 mm off":
        image = synthetic_frame(joints, "cube", near, 0.0).image
    elif case == "a second block 4.5 cm off":
        second = (("cube", tuple(np.asarray(XY) + 0.045 * side), 0.0),)
        image = synthetic_frame(joints, "cube", XY, 0.0, extra_blocks=second).image
    elif case == "one block 7 cm off":
        image = synthetic_frame(joints, "cube", np.asarray(XY) + 0.07 * side, 0.0).image
    elif case == "a distractor 7 cm off":
        distractor = (("cube", tuple(np.asarray(XY) - 0.07 * side), 0.0),)
        image = synthetic_frame(joints, "cube", near, 0.0, extra_blocks=distractor).image
    else:
        image = synthetic_frame(joints, "cube", XY, 0.0).image
        K, T = frame_K_T(joints)
        full = wrist.silhouette(K, T, box_corners(CUBE, XY, 0.0), SIZE, cv2)
        rows, cols = np.nonzero(full)
        keep = np.zeros_like(full)
        # Keep only the top-left quadrant of the silhouette.
        middle = ((rows.min() + rows.max()) // 2 + 1, (cols.min() + cols.max()) // 2 + 1)
        keep[rows.min():middle[0], cols.min():middle[1]] = True
        image[full & ~keep] = rgb8(TABLE)
    m = s.measure(image, joints)

    if case in ("one block 8 mm off", "a distractor 7 cm off"):
        assert m.problem == "" and m.xy == pytest.approx(near, abs=5e-4)
    elif case == "a second block 4.5 cm off":
        assert m.problem == "dwa bloki w kamerze"
    elif case == "one block 7 cm off":
        assert m.problem == "brak bloku w kamerze"
    else:
        assert m.problem.startswith("plama w kamerze nie pasuje rozmiarem (x0.")
        assert float(m.problem.split("(x")[1].rstrip(")")) == pytest.approx(0.25, abs=0.08)


def test_a_cube_on_another_block_is_measured_at_its_own_height():
    cv2 = cv2_or_skip()
    base, yaw = 2 * CUBE[2], math.radians(10.0)
    pose = {"cube": ObjectPose(np.array([*XY, base + CUBE[2]]), rotation_z(yaw), "kamery", 0.1)}
    s = started(pose)
    s.act(obs(dict(SPEC.home), t=0.05, objects=pose))
    assert s.base_z == pytest.approx(base)
    sol = KIN.ik(s.straddle_hover, top_down(s._psi), seed=dict(SPEC.home, gripper=s.cmd_open))
    assert sol.ok
    joints = dict(sol.joints, gripper=s.cmd_open)
    # A green cube on the table and the red one 6 mm off the aim on top of it.
    truth = np.asarray(XY) + along_closing(joints, 0.006)
    image = np.empty((*SIZE, 3), np.uint8)
    image[:] = rgb8(TABLE)
    K, T = frame_K_T(joints)
    lower = wrist.block_points(block("cube"), XY, yaw)
    upper = wrist.block_points(block("cube"), truth, yaw) + np.array([0.0, 0.0, base])
    image[wrist.silhouette(K, T, lower, SIZE, cv2)] = rgb8(PALETTE[2][0])
    image[wrist.silhouette(K, T, upper, SIZE, cv2)] = rgb8(PAINT["cube"])
    image[wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, 0)] = rgb8(GOLD)

    m = s.measure(image, joints)
    assert m.problem == "" and m.xy == pytest.approx(truth, abs=5e-4)
    # Measured as if it lay on the table, the same frame misplaces it or does not fit.
    s.base_z = 0.0
    flat = s.measure(image, joints)
    assert flat.problem or float(np.linalg.norm(flat.xy - truth)) > 0.002


@pytest.mark.parametrize("where", ["side by side", "one behind the other"])
def test_two_touching_blocks_of_one_colour_are_one_blob_too_large_for_the_box(where: str):
    cv2_or_skip()
    s, joints = hover_of("cube")
    side = along_closing(joints, 1.0)
    # Test adjacent blocks across the image and along the viewing direction.
    step = 2 * CUBE[0] * (side if where == "side by side" else np.array([-side[1], side[0]]))
    # Keep both blocks inside the frame.
    first = np.asarray(XY) - 0.2 * step
    second = (("cube", tuple(first + step), 0.0),)
    image = synthetic_frame(joints, "cube", first, 0.0, extra_blocks=second).image
    m = s.measure(image, joints)

    assert m.problem.startswith("plama w kamerze nie pasuje rozmiarem (x")
    assert float(m.problem.split("(x")[1].rstrip(")")) >= (1.6 if where == "side by side" else 1.4)
    assert np.all(np.isnan(m.xy))


def test_a_speck_at_the_jaws_edge_is_not_a_second_block():
    """A sliver beside the jaw mask fits too little visible silhouette to count as another block."""
    cv2 = cv2_or_skip()
    s, joints = hover_of("cube")
    image = synthetic_frame(joints, "cube", XY, 0.0).image
    K, T = frame_K_T(joints)
    visible = ~wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)
    speck = np.zeros(SIZE, bool)
    speck[234:264, 232:240] = True
    speck &= visible
    image[speck] = rgb8(PAINT["cube"])
    # Only the visible-share check rejects this sliver.
    rows, cols = np.nonzero(speck)
    centre = np.array([[cols.mean(), rows.mean()]])
    assert np.linalg.norm(wrist.back_project(K, T, centre, 2 * CUBE[2])[0, :2] - XY) < 0.04
    spec = block("cube")
    blob = wrist.Blob(int(speck.sum()), centre[0], 1)
    xy0, yaw0 = wrist.blob_pose(speck.astype(np.int32), blob, K, T, 2 * CUBE[2], spec.yaw_period, cv2)
    fit = wrist.fit_block(speck, K, T, spec, xy0, yaw0, cv2, visible=visible)
    points = wrist.block_points(spec, fit.xy, fit.yaw)
    model = int(np.count_nonzero(wrist.silhouette(K, T, points, SIZE, cv2) & visible))
    assert wrist.AREA_WINDOW[0] <= speck.sum() / model <= wrist.AREA_WINDOW[1]
    assert model < wrist.MIN_VISIBLE_SHARE * wrist.hull_area(K, T, points, cv2)

    m = s.measure(image, joints)
    assert m.problem == "" and m.xy == pytest.approx(XY, abs=5e-4)


# ------------------------------------------------------------ the block's hue
def test_the_dominant_hue_is_the_blocks_inside_its_silhouette_and_none_on_the_table():
    """The expected silhouette samples the block's hue and rejects the unsaturated table."""
    cv2 = cv2_or_skip()
    joints = probe_hover("cube")
    K, T = frame_K_T(joints)
    visible = ~wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)
    blue, hue = PALETTE[3]
    image = synthetic_frame(joints, "cube", XY, 0.0, colour=blue).image
    inside = wrist.silhouette(K, T, box_corners(CUBE, XY, 0.0), SIZE, cv2) & visible
    found, share = wrist.dominant_hue(image, inside, 20.0, 0.35, 0.25, cv2)
    assert found == pytest.approx(hue, abs=2.0) and share > 0.95

    beside_xy = np.asarray(XY) + along_closing(joints, 0.045)
    beside = wrist.silhouette(K, T, box_corners(CUBE, beside_xy, 0.0), SIZE, cv2) & visible
    assert beside.any()
    found, share = wrist.dominant_hue(image, beside, 20.0, 0.35, 0.25, cv2)
    assert math.isnan(found) and share == 0.0


def test_the_dominant_hue_wraps_at_zero():
    cv2 = cv2_or_skip()
    image = np.zeros((10, 20, 3), np.uint8)
    colorsys = __import__("colorsys")
    image[:, :10] = rgb8(colorsys.hsv_to_rgb(356 / 360, 0.8, 0.8))
    image[:, 10:] = rgb8(colorsys.hsv_to_rgb(6 / 360, 0.8, 0.8))
    found, share = wrist.dominant_hue(image, np.ones((10, 20), bool), 20.0, 0.35, 0.25, cv2)
    assert abs((found + 180.0) % 360.0 - 180.0) < 2.0 and share == 1.0


@pytest.mark.parametrize("colour", [c for c, _ in PALETTE], ids=["red", "yellow", "green", "blue"])
def test_the_first_frame_gives_the_hue_of_a_block_of_any_colour(colour: tuple[float, ...]):
    """The first usable frame sets the run's hue from the block's pixels."""
    cv2_or_skip()
    s, joints = hover_of("cube", hue_deg=-1.0)
    assert math.isnan(s.wrist.hue_deg)
    near = np.asarray(XY) + along_closing(joints, 0.006)
    m = s.measure(synthetic_frame(joints, "cube", near, 0.0, colour=colour).image, joints)

    assert m.problem == "" and m.xy == pytest.approx(near, abs=5e-4)
    assert abs((s.wrist.hue_deg - hue_of(colour) + 180.0) % 360.0 - 180.0) < 2.0
    assert s.wrist.hue_share > 0.5
    # The hue stays with the run: a block of another colour is not this one.
    other = next(c for c, _ in PALETTE if c != colour)
    m = s.measure(synthetic_frame(joints, "cube", near, 0.0, colour=other).image, joints)
    assert m.problem == "brak bloku w kamerze"


def test_bare_wood_gives_no_hue_and_the_look_is_skipped():
    """Bare wood gives an unusable measurement, leaving hue sampling for a later frame."""
    cv2_or_skip()
    s, joints = hover_of("cube", hue_deg=-1.0)
    m = s.measure(synthetic_frame(joints, "cube", XY, 0.0, colour=WOOD).image, joints)

    assert m.problem == "brak jednego koloru w miejscu bloku (0%)"
    assert math.isnan(s.wrist.hue_deg) and s.wrist.hue_share == 0.0
    # A later frame can still give the hue.
    m = s.measure(synthetic_frame(joints, "cube", XY, 0.0).image, joints)
    assert m.problem == "" and s.wrist.hue_deg == pytest.approx(hue_of(PAINT["cube"]), abs=2.0)


#: Paints at the hues `LABEL_HUES` gives "light blue" and "green" [0..1].
LIGHT_BLUE = tuple(__import__("colorsys").hsv_to_rgb(190 / 360, 0.99, 0.8))
GREEN = tuple(__import__("colorsys").hsv_to_rgb(143 / 360, 0.7, 0.6))


@pytest.mark.parametrize("label, colour", [("light blue cube", LIGHT_BLUE), ("red cube 2", PALETTE[0][0]),
                                           ("green cube", GREEN)], ids=["light blue", "red", "green"])
@pytest.mark.parametrize("hue_deg", [-1.0, 146.0, 50.0], ids=["no hue set", "green set", "yellow set"])
def test_a_colour_word_in_the_label_finds_the_block_of_that_colour(label: str, colour, hue_deg: float):
    """The label's colour sets the hue; a set hue of another colour does not replace it."""
    cv2_or_skip()
    s, joints = hover_of("cube", hue_deg=hue_deg, label=label)
    m = s.measure(synthetic_frame(joints, "cube", XY, 0.0, colour=colour).image, joints)

    assert m.problem == "" and m.xy == pytest.approx(XY, abs=1e-3)
    assert s.wrist.hue_source == ("setting" if label == "green cube" and hue_deg == 146.0 else "label")
    other = LIGHT_BLUE if colour is not LIGHT_BLUE else PALETTE[0][0]
    m = s.measure(synthetic_frame(joints, "cube", XY, 0.0, colour=other).image, joints)
    assert m.problem == "brak bloku w kamerze"


def test_a_set_hue_near_the_labels_colour_fine_tunes_it():
    cv2_or_skip()
    s, _ = hover_of("cube", hue_deg=196.0, label="light blue cube")
    assert s.wrist.hue_source == "setting" and s.wrist.hue_deg == 196.0


def test_a_label_without_a_colour_word_takes_the_set_hue_or_samples_it():
    cv2_or_skip()
    s, _ = hover_of("cube", hue_deg=-1.0, label="cube 2")
    assert s.wrist.hue_source == "sampled" and math.isnan(s.wrist.hue_deg)
    s, _ = hover_of("cube", hue_deg=146.0, label="wooden cube")
    assert s.wrist.hue_source == "setting" and s.wrist.hue_deg == 146.0


def test_label_hues_add_and_replace_colour_words():
    cv2_or_skip()
    hues = wrist.colour_hues(" Light  Blue: 200, orange:20 ")
    assert hues["light blue"] == 200.0 and hues["orange"] == 20.0 and hues["red"] == wrist.LABEL_HUES["red"]
    assert next(iter(hues)) == "light blue"
    s, _ = hover_of("cube", hue_deg=-1.0, label="orange cube", label_hues="orange: 20")
    assert s.wrist.hue_source == "label" and s.wrist.hue_deg == 20.0
    for text in ("blue", "blue: 400", ": 20", "blue: x"):
        with pytest.raises(ValueError, match="grasp_wrist: label_hues"):
            WristGraspStrategy(WristGraspSettings(label_hues=text))


def test_a_set_hue_is_used_as_given():
    cv2_or_skip()
    s, joints = hover_of("cube", hue_deg=hue_of(PALETTE[3][0]))
    assert s.wrist.hue_deg == pytest.approx(hue_of(PALETTE[3][0]))
    m = s.measure(synthetic_frame(joints, "cube", XY, 0.0, colour=PALETTE[0][0]).image, joints)
    assert m.problem == "brak bloku w kamerze"


@pytest.mark.parametrize("kind", tuple(BLOCKS))
@pytest.mark.parametrize("where", ["toward", "away", "across"])
def test_the_area_window_holds_one_block_anywhere_in_range(kind: str, where: str):
    cv2_or_skip()
    s, joints = hover_of(kind)
    side = along_closing(joints, 1.0)
    view = np.array([-side[1], side[0]])
    shift = {"toward": -0.04 * view, "away": 0.04 * view, "across": 0.04 * side}[where]
    truth = np.asarray(XY) + shift
    m = s.measure(synthetic_frame(joints, kind, truth, 0.3).image, joints)

    assert m.problem == "", m.problem
    assert 0.8 < m.ratio < 1.3


def test_a_tip_pixel_shifts_the_principal_point():
    cv2_or_skip()
    s, joints = hover_of("cube")
    image = synthetic_frame(joints, "cube", XY, 0.0).image
    plain = s.measure(image, joints)
    K, T = frame_K_T(joints)
    tip = (KIN.tcp(joints) @ np.array([*KIN.jaw_centre(0.0), 1.0]))[:3]
    predicted = wrist.project(K, T, tip)[0][0]
    assert predicted == pytest.approx([300.5, 266.1], abs=1.0)

    s.wrist_settings.tip_pixel_x, s.wrist_settings.tip_pixel_y = predicted[0] + 10.0, predicted[1]
    moved = s.measure(image, joints)
    # A tip calibration shift moves the measured block toward the fixed jaw (-closing).
    ends = wrist.back_project(K, T, np.array([[320.0, 120.0], [330.0, 120.0]]), 2 * CUBE[2])
    mm_per_px = float(np.linalg.norm(ends[1, :2] - ends[0, :2])) / 10
    assert moved.xy - plain.xy == pytest.approx(-10 * mm_per_px * along_closing(joints, 1.0), abs=3e-4)
    assert 0.0002 < mm_per_px < 0.0004

    s.wrist_settings.tip_pixel_x = predicted[0] + 200.0
    assert s.measure(image, joints).problem.startswith("tip_pixel daleko od modelu (200 px)")


def test_rotate_180_undoes_an_upside_down_camera():
    cv2_or_skip()
    s, joints = hover_of("cuboid")
    image = synthetic_frame(joints, "cuboid", np.asarray(XY) + (0.004, -0.006), 0.5).image
    plain = s.measure(image, joints)
    s.wrist_settings.rotate_180 = True
    turned = s.measure(np.ascontiguousarray(image[::-1, ::-1]), joints)

    assert plain.problem == turned.problem == ""
    assert turned.xy == pytest.approx(plain.xy, abs=1e-9) and turned.yaw == pytest.approx(plain.yaw)
    assert turned.pixel == pytest.approx([639 - plain.pixel[0], 359 - plain.pixel[1]])
    # Tip calibration uses received-frame coordinates even when the image is rotated.
    tip = (KIN.tcp(joints) @ np.array([*KIN.jaw_centre(0.0), 1.0]))[:3]
    predicted = wrist.project(*frame_K_T(joints), tip)[0][0]
    s.wrist_settings.rotate_180 = False
    s.wrist_settings.tip_pixel_x, s.wrist_settings.tip_pixel_y = predicted[0] + 6.0, predicted[1] - 4.0
    shifted = s.measure(image, joints)
    s.wrist_settings.rotate_180 = True
    s.wrist_settings.tip_pixel_x = 639 - predicted[0] - 6.0
    s.wrist_settings.tip_pixel_y = 359 - predicted[1] + 4.0
    turned = s.measure(np.ascontiguousarray(image[::-1, ::-1]), joints)
    assert np.linalg.norm(shifted.xy - plain.xy) > 0.001
    assert turned.xy == pytest.approx(shifted.xy, abs=1e-9)


RENDER_CASES = [("cube", 20.0), ("cuboid", 35.0), ("cuboid", -70.0), ("cylinder", -60.0),
                ("cylinder", 0.0)]


@pytest.mark.render
@pytest.mark.parametrize("kind, yaw_deg", RENDER_CASES, ids=[f"{k} {y:.0f}" for k, y in RENDER_CASES])
def test_measure_on_the_sims_own_render_finds_the_block(kind: str, yaw_deg: float):
    """Fit blocks in MuJoCo wrist frames with shading, jaw meshes and visible side faces."""
    cv2_or_skip()
    from soarm_lab import sim
    from soarm_lab.blocks import scene_object
    from soarm_lab.station import Station

    aim = (0.2, 0.0)
    truth = np.asarray(aim) + (0.012, 0.0)
    s, joints = hover_of(kind, aim)
    objects = [scene_object(block(kind), kind, tuple(truth), rgba=(*PAINT[kind], 1.0))]
    scene = sim.build(sim.SceneConfig(Station.load("sim"), objects=objects))
    scene.set_joints(joints)
    model, data = scene.model, scene.data
    adr = model.jnt_qposadr[model.body_jntadr[model.body(kind).id]]
    yaw = math.radians(yaw_deg)
    data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=CAMERA.height, width=CAMERA.width) as renderer:
        renderer.update_scene(data, camera="robot/wrist")
        image = renderer.render().copy()
    m = s.measure(image, joints)

    assert m.problem == "", m.problem
    error_mm = float(np.linalg.norm(m.xy - truth)) * 1000
    error_deg = abs(math.degrees(fold(m.yaw - yaw, block(kind).yaw_period)))
    # Fit the cylinder silhouette from its end circles.
    assert error_mm < 1.0 and error_deg < 2.0, (error_mm, error_deg)


# ------------------------------------------------------------ the looks
@pytest.mark.parametrize("again", [True, False], ids=["look_before_straddle", "look_before_straddle off"])
def test_the_look_centres_the_jaws_on_the_block_before_the_straddle(again: bool):
    """Without the probe, the open-jaw hover look runs for either `look_before_straddle` value."""
    cv2_or_skip()
    truth = (0.205, 0.03)
    coarse = (truth[0] + 0.009, truth[1] - 0.012)
    block = block_at(truth, "cuboid", 25.0)
    s = started(seen(coarse, "cuboid"), block="cuboid", look_before_straddle=again)
    measured: list[dict[str, float]] = []
    aims: list[tuple[str, tuple[float, float] | None]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(coarse, "cuboid"), 2500, block=block,
                                 frames=camera_of(block), measured=measured,
                                 on_tick=lambda t: aims.append((s.status.phase, s.grasp.block_xy)))

    assert answers[-1].done and answers[-1].success, answers[-1].note
    assert s.wrist.verdicts == ["centred"] and s.wrist.looks == 1 and s.wrist.where == "straddle"
    assert s.status.attempt == 1
    assert runs(phases)[:3] == ["approach", "open", "straddle"]
    opening = [k for k, p in enumerate(phases) if p == "open"]
    notes = [answers[k].note for k in opening]
    assert any(n.startswith("centruje po kamerze: przesuwam o 15 mm, obrot 59 st.") for n in notes)
    assert any(n.startswith("wysrodkowano po kamerze: 0 mm") for n in notes)
    # The look sees the block between the open jaws.
    first_look = next(k for k in opening if answers[k].note.startswith("centruje po kamerze"))
    assert answers[first_look].target["gripper"] == pytest.approx(s.cmd_open)
    # The straddle centres the jaws and selects the nearest heading across the short side.
    straddle_aim = [xy for p, xy in aims if p == "straddle"][0]
    assert straddle_aim is not None and np.linalg.norm(np.asarray(straddle_aim) - truth) < 0.002
    assert s.grasp.heading_deg == pytest.approx(-65.0, abs=2.0)
    # Measured roll reaches the chosen heading before the straddle.
    _, closing = KIN.tool_axes(measured[phases.index("straddle")])
    rolled = math.degrees(fold(math.atan2(closing[1], closing[0]) - math.radians(-65.0), math.pi))
    assert abs(rolled) < 4.0


def test_the_look_waits_for_the_wrist_roll_before_it_measures():
    cv2_or_skip()
    truth = (0.205, 0.03)
    block = block_at(truth, "cuboid", 60.0)
    s = started(seen((truth[0] + 0.002, truth[1]), "cuboid"), block="cuboid")
    reads: list[int] = []
    frames_now: list[LazyFrames] = []

    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        lazy = LazyFrames(lambda: synthetic_frame(joints, "cuboid", block.xy, block.yaw))
        frames_now.append(lazy)
        return lazy

    measured: list[dict[str, float]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen((truth[0] + 0.002, truth[1]), "cuboid"), 400,
                                 block=block, frames=frames, measured=measured,
                                 until=lambda: s.status.phase == "touched")
    # The hover look's reads; the look between the jaw tips reads one more frame at the bottom.
    reads = [k for k, lazy in enumerate(frames_now) if lazy.reads and phases[k] == "open"]
    assert len(reads) == 2 and s.wrist.verdicts == ["centred"]
    assert [c.verdict for c in s.wrist.close_looks] == ["centred"]
    first, second = reads
    assert answers[first].note.startswith("centruje po kamerze: przesuwam o 2 mm, obrot")
    roll = SPEC.joints.index("wrist_roll")
    targets = [KIN.to_q(a.target) for a in answers[first - 1:second]]
    cap = math.radians(SPEC.max_vel["wrist_roll"] * 0.8 / s.hz)
    assert max(abs(b[roll] - a[roll]) for a, b in zip(targets, targets[1:], strict=False)) <= cap + 1e-9
    assert math.degrees(abs(targets[-1][roll] - targets[0][roll])) > 20.0
    # Measure only after every arm joint settles for the full window.
    arm = KIN.arm
    still = [float(np.degrees(np.abs(KIN.to_q(measured[k]) - KIN.to_q(measured[k - 1]))[arm].max())) <= 0.5
             for k in range(second - 4, second + 1)]
    assert all(still)
    rolling = [float(np.degrees(np.abs(KIN.to_q(measured[k]) - KIN.to_q(measured[k - 1]))[arm].max())) <= 0.5
               for k in range(first + 1, first + 6)]
    assert not all(rolling)


@pytest.mark.parametrize("look", [True, False], ids=["looks again", "look_before_straddle off"])
def test_the_look_at_the_straddle_hover_corrects_a_push(look: bool):
    cv2_or_skip()
    block = block_at(XY, "cube")
    # Without the look between the jaw tips, which would catch the push at the bottom.
    s = started(seen(XY), probe=True, look_before_straddle=look, close_rounds=0)
    pushed: list[np.ndarray] = []
    straddle: list[tuple[np.ndarray, np.ndarray]] = []

    def on_tick(t: float) -> None:
        if s.status.phase == "rise" and not pushed:
            # Move the block along the closing heading to simulate a probe push.
            block.xy = block.xy + along_closing(dict(SPEC.home, gripper=0.0), 0.0) + 0.006 * np.array(
                [math.cos(math.radians(s.grasp.heading_deg)), math.sin(math.radians(s.grasp.heading_deg))])
            pushed.append(block.xy.copy())
        if s.status.phase == "close" and not straddle:
            straddle.append(KIN.fingertips(dict(KIN.from_q(s.q_cmd), gripper=s.cmd_open)))

    measured: list[dict[str, float]] = []
    answers, *_ = follow(s, dict(SPEC.home), seen(XY), 2500, block=block, frames=camera_of(block),
                         measured=measured, on_tick=on_tick)

    assert answers[-1].success, answers[-1].note
    centre = pushed[0] if look else np.asarray(XY)
    assert s.grasp.block_xy == pytest.approx(centre, abs=1e-3)
    u = np.array([math.cos(math.radians(s.grasp.heading_deg)), math.sin(math.radians(s.grasp.heading_deg))])
    fixed, moving = straddle[0]
    assert (fixed[:2] - centre) @ u == pytest.approx(-(CUBE[1] + 0.008), abs=1e-3)
    assert (moving[:2] - centre) @ u == pytest.approx(CUBE[1] + 0.008, abs=1e-3)
    if look:
        assert s.wrist.verdicts == ["centred", "centred"]
        assert s.wrist.shift_m == pytest.approx(0.006, abs=1e-3)
    else:
        assert s.wrist.verdicts == ["centred"] and s.wrist.looks == 1


def test_without_a_camera_name_the_strategy_is_grasp():
    plain = started(seen(), strategy=GraspStrategy)
    s = started(seen(), wrist_camera="")
    a_plain, p_plain, *_ = follow(plain, dict(SPEC.home), seen(), 1500, block=block_at())
    a_wrist, p_wrist, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())

    assert a_wrist[-1].success and s.wrist.looks == 0 and s.wrist.measurements == 0
    assert p_wrist == p_plain
    assert [a.target for a in a_wrist] == [a.target for a in a_plain]
    assert [a.note for a in a_wrist] == [a.note for a in a_plain]


@pytest.mark.parametrize("case", ["no frame", "a stale frame"])
def test_a_missing_or_stale_frame_skips_the_look_and_the_probe_goes_on(case: str):
    coarse = (XY[0] + 0.005, XY[1])
    block = block_at()
    s = started(seen(coarse), probe=True)
    if case == "no frame":
        frames: Frames = lambda joints: {}                        # noqa: E731
        why = "brak klatki z kamery wrist"
    else:
        cv2_or_skip()
        frames = camera_of(block, age_s=2.0)
        why = "klatka z kamery wrist za stara (2.0 s)"
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(coarse), 2500, block=block, frames=frames)

    descent = [a.note for a, p in zip(answers, phases, strict=True) if p == "descend"]
    waiting = [k for k, n in enumerate(descent) if n == "centruje po kamerze: czekam na klatke"]
    assert len(waiting) == round(s.wrist_settings.frame_wait_s * s.hz)
    assert waiting == list(range(waiting[0], waiting[-1] + 1))
    assert descent[waiting[-1] + 1] == f"centrowanie pominiete: {why}"
    assert s.wrist.verdicts == ["skipped", "skipped"] and s.wrist.skipped == why
    assert "touched" in phases and answers[-1].success, answers[-1].note
    assert s.grasp.block_xy == pytest.approx(coarse)


def test_without_the_probe_a_missing_frame_skips_the_one_look_and_the_straddle_goes_on():
    coarse = (XY[0] + 0.005, XY[1])
    s = started(seen(coarse))
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(coarse), 2500, block=block_at(),
                                 frames=lambda joints: {})

    opening = [a.note for a, p in zip(answers, phases, strict=True) if p == "open"]
    waiting = [k for k, n in enumerate(opening) if n == "centruje po kamerze: czekam na klatke"]
    assert len(waiting) == round(s.wrist_settings.frame_wait_s * s.hz)
    assert opening[waiting[-1] + 1] == "centrowanie pominiete: brak klatki z kamery wrist"
    assert s.wrist.verdicts == ["skipped"] and not {"descend", "touched"} & set(phases)
    assert answers[-1].success, answers[-1].note
    assert s.grasp.block_xy == pytest.approx(coarse)


@pytest.mark.parametrize("case", ["no block", "two blocks", "a far block"])
def test_no_blob_two_blobs_and_a_far_block_skip_the_look(case: str):
    cv2_or_skip()
    s = started(seen(), probe=True, look_before_straddle=False)
    side = along_closing(dict(SPEC.home, gripper=0.0), 1.0)
    if case == "no block":
        draw: Frames = lambda joints: {"wrist": synthetic_frame(joints, "cube", (0.6, 0.4), 0.0)}  # noqa: E731
        why = "brak bloku w kamerze"
    elif case == "two blocks":
        second = (("cube", tuple(np.asarray(XY) + 0.045 * side), 0.0),)
        draw = lambda joints: {"wrist": synthetic_frame(joints, "cube", XY, 0.0, extra_blocks=second)}  # noqa: E731
        why = "dwa bloki w kamerze"
    else:
        far = np.asarray(XY) + 0.046 * side
        draw = lambda joints: {"wrist": synthetic_frame(joints, "cube", far, 0.0)}  # noqa: E731
        why = "blok w kamerze 46 mm od celu - nie ufam"
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 600, block=block_at(), frames=draw,
                                 until=lambda: s.status.phase == "touched")

    assert s.wrist.verdict == "skipped" and s.wrist.skipped == why, s.wrist.skipped
    assert any(a.note == f"centrowanie pominiete: {why}" for a in answers)
    assert s.grasp.block_xy == pytest.approx(XY) and phases[-1] == "touched"


def lying(s: WristGraspStrategy, block: Block, gain: float) -> Frames:
    """Supply frames with the true block size and an offset multiplied by `gain`."""

    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        def draw() -> Frame:
            assert s.grasp.block_xy is not None
            aim = np.asarray(s.grasp.block_xy)
            return synthetic_frame(joints, block.kind, aim + gain * (block.xy - aim), block.yaw)

        return LazyFrames(draw)

    return frames


def test_a_move_that_does_not_match_the_image_is_caught_and_the_aim_reverted():
    cv2_or_skip()
    coarse = XY
    side = along_closing(dict(SPEC.home, gripper=0.0), 1.0)
    block = block_at(tuple(np.asarray(XY) + 0.012 * side))
    s = started(seen(coarse), probe=True, look_before_straddle=False)
    log: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    def record(t: float) -> None:
        log.append((s.status.phase, s.wrist.verdict, s.tcp_cmd.copy(), s.hover_point.copy()))

    answers, phases, *_ = follow(s, dict(SPEC.home), seen(coarse), 600, block=block,
                                 frames=lying(s, block, 2.0), on_tick=record,
                                 until=lambda: s.status.phase == "touched")
    hover0 = next(h for p, _, _, h in log if p == "descend")

    assert s.wrist.verdict == "reverted" and s.wrist.measurements == 2
    note = next(a.note for a in answers if a.note.startswith("centrowanie cofniete"))
    numbers = [int(word) for word in note.replace("(", " ").split() if word.isdigit()]
    assert note.startswith("centrowanie cofniete: przesuniecie nie zgadza sie z obrazem (")
    assert numbers == pytest.approx([24, 24], abs=1)
    assert s.grasp.block_xy == pytest.approx(coarse) and np.allclose(s.block_rotation, np.eye(3))
    # Return to the old hover before resuming descent.
    leaving = next(k for k, (p, v, _, _) in enumerate(log) if p == "descend" and v == "reverted"
                   and answers[k].note.startswith("w dol"))
    assert np.linalg.norm(log[leaving - 1][2] - hover0) < s.ik_tol + 1e-9
    assert s.status.attempt == 1


def test_a_revert_the_arm_cannot_take_keeps_the_moved_aim(monkeypatch: pytest.MonkeyPatch):
    cv2_or_skip()
    side = along_closing(dict(SPEC.home, gripper=0.0), 1.0)
    block = block_at(tuple(np.asarray(XY) + 0.012 * side))
    s = started(seen(), probe=True, look_before_straddle=False)
    lie = lying(s, block, 2.0)
    reads: list[int] = []
    real_ik = s.kin.ik

    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        inner = lie(joints)
        return LazyFrames(lambda: (reads.append(1), inner["wrist"])[1])

    def ik(*args, **kwargs) -> IKSolution:
        if len(reads) >= 2:                      # Reject the return aim after the second frame.
            seed = kwargs.get("seed") or dict(SPEC.home)
            return IKSolution(dict(seed), 1.0, 0.0, False)
        return real_ik(*args, **kwargs)

    monkeypatch.setattr(s.kin, "ik", ik)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 600, block=block, frames=frames,
                                 until=lambda: s.status.phase == "touched")

    assert s.wrist.verdict == "skipped"
    assert s.wrist.skipped.startswith("przesuniecie nie zgadza sie z obrazem, cofniecie niemozliwe: punkt (")
    assert any(a.note.startswith("centrowanie pominiete: przesuniecie nie zgadza sie z obrazem, cofniecie "
                                 "niemozliwe") for a in answers)
    assert s.grasp.block_xy == pytest.approx(np.asarray(XY) + 0.024 * side, abs=1e-3)
    assert phases[-1] == "touched" and not answers[-1].done


def test_the_look_never_counts_as_an_attempt_and_keeps_the_speed_limits():
    cv2_or_skip()
    block = block_at((XY[0] + 0.006, XY[1] - 0.012))
    # Focal error leaves a proportional residual after each move.
    s = started(seen(), max_vel=capped(20.0), rounds=2, centre_tol_m=0.0005, probe=True,
                look_before_straddle=False)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 3000, block=block, cap_deg_s=20.0,
                                 frames=camera_of(block, focal_px=440.0),
                                 until=lambda: s.status.phase == "touched")

    assert s.status.attempt == 1 and s.wrist.measurements == 3 and s.wrist.verdict == "centred"
    assert any(a.note.endswith("po 2 ruchach") for a in answers)
    targets = [KIN.to_q(a.target) for a, p in zip(answers, phases, strict=True) if p == "descend"]
    steps = np.degrees(np.abs(np.diff(np.array(targets), axis=0)))
    assert steps[:, KIN.arm].max() <= 20.0 * 0.8 / s.hz + 1e-6
    grips = [a.target["gripper"] for a in answers]
    assert np.max(np.abs(np.diff(grips))) <= 12.0 + 1e-6


def test_an_unreachable_re_aim_keeps_the_old_aim(monkeypatch: pytest.MonkeyPatch):
    cv2_or_skip()
    block = block_at((XY[0] + 0.01, XY[1]))
    s = started(seen(), probe=True, look_before_straddle=False)
    real_ik = s.kin.ik
    frames = camera_of(block)
    looked: list[bool] = []

    def ik(*args, **kwargs) -> IKSolution:
        sol = real_ik(*args, **kwargs)
        return IKSolution(sol.joints, sol.pos_err, 0.3, sol.ok) if looked else sol

    monkeypatch.setattr(s.kin, "ik", ik)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 600, block=block, frames=frames,
                                 on_tick=lambda t: looked.append(True) if s.wrist.looks else None,
                                 until=lambda: s.status.phase == "touched")

    assert s.wrist.verdict == "skipped" and "chwytak nie ustawi sie pionowo (blad 17 st.)" in s.wrist.skipped
    assert s.grasp.block_xy == pytest.approx(XY) and phases[-1] == "touched"


def test_a_look_whose_arm_never_stands_still_is_skipped():
    """A jittering joint prevents settling and skips the look after `LOOK_STALL_S`."""
    cv2_or_skip()
    block = block_at()
    s = started(seen(), probe=True, look_before_straddle=False)
    frames = camera_of(block)
    jitter = [0.0]

    def shaky(joints: dict[str, float]) -> Mapping[str, Frame]:
        return frames(joints)

    answers: list[Action] = []
    joints, t = dict(SPEC.home), 0.0
    command = dict(joints)
    for k in range(400):
        if s.status.phase == "descend":
            jitter[0] = 1.0 if k % 2 else -1.0
        noisy = dict(joints, wrist_flex=joints["wrist_flex"] + jitter[0])
        a = s.act(obs(noisy, t=t, objects=seen(), command=command, frames=shaky(noisy)))
        answers.append(a)
        command = dict(a.target)
        joints = dict(a.target)
        t += 1 / s.hz
        if s.status.phase == "touched" or a.done:
            break

    assert s.wrist.verdict == "skipped" and s.wrist.skipped.startswith("ramie nie stoi nad blokiem")
    looking = [a for a in answers if a.note.startswith("centruje po kamerze")]
    assert len(looking) / s.hz == pytest.approx(wrist.LOOK_STALL_S, abs=0.3)


# ------------------------------------------------------------ the look between the jaw tips
def closing_u(s: WristGraspStrategy) -> np.ndarray:
    psi = math.radians(s.grasp.heading_deg)
    return np.array([math.cos(psi), math.sin(psi)])


def push_after_hover_look(s: WristGraspStrategy, block: Block, along_m: float, across_m: float
                          ) -> Callable[[float], None]:
    """Move the block once the hover look is done and the straddle starts down."""
    pushed: list[bool] = []

    def on_tick(t: float) -> None:
        if s.status.phase == "straddle" and s.wrist.verdicts and not pushed:
            u = closing_u(s)
            block.xy = block.xy + along_m * u + across_m * np.array([-u[1], u[0]])
            pushed.append(True)
    return on_tick


def offset_at_close(s: WristGraspStrategy, phases: list[str], measured: list[dict[str, float]],
                    where: list[np.ndarray]) -> tuple[float, float]:
    """The block minus the open tips' midpoint when the jaw last started closing, along the closing line
    and across it [m]."""
    k = max(i for i, p in enumerate(phases) if p == "close" and phases[i - 1] != "close")
    d = where[k] - np.mean(KIN.fingertips(measured[k]), axis=0)[:2]
    u = closing_u(s)
    return float(d @ u), float(d @ np.array([-u[1], u[0]]))


def test_the_look_between_the_jaw_tips_catches_a_push_after_the_hover_look():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY))
    measured: list[dict[str, float]] = []
    where: list[np.ndarray] = []
    push = push_after_hover_look(s, block, -0.004, 0.003)

    def on_tick(t: float) -> None:
        where.append(block.xy.copy())
        push(t)

    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 2500, block=block, frames=camera_of(block),
                                 measured=measured, on_tick=on_tick)

    assert answers[-1].success, answers[-1].note
    looks = s.wrist.close_looks
    assert [(c.attempt, c.round, c.verdict) for c in looks] == [(1, 0, "moved"), (1, 1, "centred")]
    assert looks[0].along_m == pytest.approx(-0.004, abs=1e-3)
    assert looks[0].across_m == pytest.approx(0.003, abs=1e-3)
    # One hover look: the arm came back from the correction without another.
    assert s.wrist.looks == 1 and s.status.attempt == 1
    assert runs(phases)[2:7] == ["straddle", "rise", "open", "straddle", "close"]
    rising = [answers[k] for k, p in enumerate(phases) if p == "rise"]
    assert all(a.target["gripper"] == pytest.approx(s.cmd_open) for a in rising)
    assert any(a.note.startswith("kamera w szczekach: blok 5 mm (wzdluz -4, w poprzek +3 mm) od srodka - "
                                 "poprawka 1 z 3") for a in answers)
    # Only the offset across the closing line is corrected: along it the block sits between the pads.
    along, across = offset_at_close(s, phases, measured, where)
    assert abs(across) < 0.001 and along == pytest.approx(-0.004, abs=1e-3)
    assert s.wrist.final_offset_m == pytest.approx((-0.004, 0.0), abs=1e-3)
    assert re.search(r"; proby 1, kamera w szczekach [45] mm$", answers[-1].note)


def test_without_close_rounds_the_jaw_closes_where_the_hover_look_aimed():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY), close_rounds=0)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 2500, block=block, frames=camera_of(block),
                                 on_tick=push_after_hover_look(s, block, -0.004, 0.0))

    assert s.wrist.close_looks == [] and "rise" not in phases
    assert s.grasp.block_xy == pytest.approx(XY, abs=1e-3)


def biased_measure(s: WristGraspStrategy, *offsets: tuple[float, float]) -> None:
    """The looks between the tips see the block at `offsets` from the jaws' midpoint in turn, along the
    closing line and across it [m]."""
    plain = s.measure
    calls: list[int] = []

    def measure(rgb, joints, ref_xy=None):
        if ref_xy is None:
            return plain(rgb, joints)
        m = plain(rgb, joints, ref_xy)
        along, across = offsets[len(calls) % len(offsets)]
        calls.append(1)
        u = closing_u(s)
        xy = np.asarray(ref_xy, float) + along * u + across * np.array([-u[1], u[0]])
        return wrist.Measurement(xy, m.yaw, m.iou, m.area_px, m.ratio, m.pixel)

    s.measure = measure  # type: ignore[method-assign]


def test_a_block_still_off_after_the_last_round_is_another_attempt_and_the_last_attempt_closes():
    """Off across the closing line by more than the face's middle third, and not the same after each
    correction: another attempt, and the last attempt closes."""
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY), close_rounds=1, max_attempts=2)
    biased_measure(s, (0.0, 0.006), (0.0, -0.006))
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block))

    assert answers[-1].success, answers[-1].note
    assert [(c.attempt, c.round, c.verdict) for c in s.wrist.close_looks] == [
        (1, 0, "moved"), (1, 1, "off"), (2, 0, "moved"), (2, 1, "off")]
    assert s.status.attempt == 2 and s.wrist.looks == 2
    notes = [a.note for a in answers]
    assert any(n.startswith("kamera widzi blok poza srodkiem szczek - w gore, proba 2") for n in notes)
    assert any(n.startswith("kamera w szczekach: blok 6 mm (wzdluz +0, w poprzek -6 mm) od srodka po 1 "
                            "poprawkach - zamykam w ostatniej probie") for n in notes)


def test_a_block_off_along_the_closing_line_but_between_the_pads_closes_at_once():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY))
    biased_measure(s, (-0.006, 0.001))
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block))

    assert answers[-1].success, answers[-1].note
    assert [c.verdict for c in s.wrist.close_looks] == ["centred"] and "rise" not in phases


def test_an_offset_along_the_closing_line_beyond_the_pads_is_corrected_only_as_far_as_the_pads_allow():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY))
    biased_measure(s, (0.012, 0.0), (0.0, 0.0))
    room = (s.gap_open_m - 2 * CUBE[1]) / 2 - KIN.fingertip_radius()
    aims: list[np.ndarray] = []

    def on_tick(t: float) -> None:
        if s.grasp.block_xy is not None and (not aims or not np.allclose(aims[-1], s.grasp.block_xy)):
            aims.append(np.asarray(s.grasp.block_xy, float).copy())

    answers, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block),
                         on_tick=on_tick)

    assert answers[-1].success, answers[-1].note
    assert [c.verdict for c in s.wrist.close_looks] == ["moved", "centred"]
    assert float((aims[-1] - aims[-2]) @ closing_u(s)) == pytest.approx(room, abs=1e-6)


def test_an_offset_a_correction_leaves_unchanged_is_the_cameras_error_and_the_jaw_closes():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY))
    biased_measure(s, (0.0, 0.004), (0.0005, 0.0045))
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block))

    assert answers[-1].success, answers[-1].note
    assert [(c.round, c.verdict) for c in s.wrist.close_looks] == [(0, "moved"), (1, "bias")]
    assert s.status.attempt == 1
    assert any(a.note.endswith("jak przed poprawka - blad kamery, zamykam") for a in answers)


def test_after_the_last_correction_a_block_off_across_within_the_faces_middle_third_closes():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY), close_rounds=1)
    biased_measure(s, (0.0, 0.0035), (0.0, -0.0035))
    answers, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block))

    assert answers[-1].success, answers[-1].note
    assert [(c.round, c.verdict) for c in s.wrist.close_looks] == [(0, "moved"), (1, "off")]
    assert s.status.attempt == 1
    assert any(a.note.endswith("od srodka po 1 poprawkach - zamykam") for a in answers)


def test_an_empty_close_looks_again_and_lifts_on_the_next_attempt():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY))
    moved: list[bool] = []

    def on_tick(t: float) -> None:
        if s.status.phase == "close" and not moved:
            # Taken from between the jaws as they close.
            block.xy = block.xy + 0.025 * closing_u(s)[::-1] * np.array([-1.0, 1.0])
            moved.append(True)

    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block),
                                 on_tick=on_tick)

    assert answers[-1].success, answers[-1].note
    assert s.grasp.empty_closes == 1 and s.status.attempt == 2 and s.wrist.looks == 2
    assert any(a.note.startswith("chwyt pusty - w gore, proba 2") for a in answers)
    # The hover look of the second attempt found the block where it was taken to.
    assert [c.verdict for c in s.wrist.close_looks] == ["centred", "centred"]


@pytest.mark.parametrize("why", ["empty", "edge"])
def test_with_the_probe_a_retry_closes_the_jaw_before_it_probes_again(why: str):
    cv2_or_skip()
    if why == "edge":
        # The jaw stops 8 mm narrower than the cube it was sized for.
        block = Block(np.array(XY, float), (CUBE[0] - 0.004, CUBE[1] - 0.004, CUBE[2]), 0.0, "cube")
    else:
        block = block_at(XY, "cube")
    # The edge case squeezes deeper, so the jaw still squeezes the narrower block.
    s = started(seen(XY), probe=True, max_attempts=2, squeeze_m=0.012 if why == "edge" else 0.008)
    moved: list[bool] = []

    def on_tick(t: float) -> None:
        if why == "empty" and s.status.phase == "close" and not moved:
            # Taken from between the jaws as they close.
            block.xy = block.xy + 0.025 * closing_u(s)[::-1] * np.array([-1.0, 1.0])
            moved.append(True)

    measured: list[dict[str, float]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 6000, block=block, frames=camera_of(block),
                                 measured=measured, on_tick=on_tick)

    assert answers[-1].success, answers[-1].note
    assert s.status.attempt == 2
    assert (s.grasp.empty_closes if why == "empty" else s.grasp.edge_closes) == 1
    starts = [k for k, p in enumerate(phases) if p == "descend" and phases[k - 1] != "descend"]
    assert len(starts) == 2
    for k in starts:
        assert answers[k].target["gripper"] == pytest.approx(s.cmd_probe)
    # The retry's rise keeps the jaw as it was until the tips are clear of the block.
    rise = [k for k, p in enumerate(phases) if p == "rise" and phases.index("close") < k < starts[1]]
    z0 = tcp(measured[rise[0]])[2]
    low = [k for k in rise if tcp(measured[k])[2] < z0 + CLEAR_BEFORE_CLOSE_M - 1e-4]
    assert low and all(answers[k].target["gripper"] == answers[rise[0]].target["gripper"] for k in low)
    assert answers[rise[-1]].target["gripper"] == pytest.approx(s.cmd_probe)


def test_an_empty_close_on_the_last_attempt_ends_the_run():
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY), max_attempts=1)

    def on_tick(t: float) -> None:
        if s.status.phase == "close":
            block.xy = np.array([0.3, 0.2])

    answers, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block),
                         on_tick=on_tick)

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note == "wyczerpane proby chwytu (1): chwyt pusty"


def test_an_edge_grip_opens_the_jaw_and_tries_again_and_the_last_attempt_lifts():
    cv2_or_skip()
    # The jaw stops 8 mm narrower than the cube it was sized for.
    block = Block(np.array(XY, float), (CUBE[0] - 0.004, CUBE[1] - 0.004, CUBE[2]), 0.0, "cube")
    s = started(seen(XY), max_attempts=2, squeeze_m=0.012)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block))

    assert answers[-1].success, answers[-1].note
    assert s.grasp.edge_closes == 1 and s.status.attempt == 2
    assert "reopen" in phases and phases.index("reopen") < phases.index("lift_off")
    notes = [a.note for a in answers]
    assert any(n.startswith("szczeka chwycila blok za krawedz - otwieram szczeke, proba 2") for n in notes)
    reopen = [answers[k] for k, p in enumerate(phases) if p == "reopen"]
    assert reopen[-1].target["gripper"] == pytest.approx(s.cmd_open)


def test_a_jaw_that_reads_wider_than_the_block_it_holds_is_a_full_grip():
    """A gripper whose reading puts the held jaw 10 mm wider than the block: no edge grip."""
    cv2_or_skip()
    block = block_at(XY, "cube")
    s = started(seen(XY), max_attempts=2)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block),
                                 held_wider_m=0.010)

    assert answers[-1].success, answers[-1].note
    assert s.grasp.edge_closes == 0 and s.status.attempt == 1 and "reopen" not in phases
    assert s.grasp.jaw_gap_m == pytest.approx(2 * CUBE[1] + 0.010, abs=0.001)


def test_with_the_stands_held_gap_offset_a_narrow_grip_is_still_an_edge():
    """The same gripper with `held_gap_offset_m` set: a jaw 8 mm narrower than the cube is an edge grip."""
    cv2_or_skip()
    block = Block(np.array(XY, float), (CUBE[0] - 0.004, CUBE[1] - 0.004, CUBE[2]), 0.0, "cube")
    s = started(seen(XY), max_attempts=2, held_gap_offset_m=0.010, squeeze_m=0.012)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, frames=camera_of(block),
                                 held_wider_m=0.010)

    assert answers[-1].success, answers[-1].note
    assert s.grasp.edge_closes == 1 and s.status.attempt == 2


def test_plain_grasp_still_ends_the_run_on_an_empty_close():
    block = block_at(XY, "cube")
    s = started(seen(XY), strategy=GraspStrategy)

    def on_tick(t: float) -> None:
        if s.status.phase == "close":
            block.xy = np.array([0.3, 0.2])

    answers, *_ = follow(s, dict(SPEC.home), seen(XY), 4000, block=block, on_tick=on_tick)
    assert answers[-1].note == "chwyt pusty - blok przesuniety albo obok szczek"


# ------------------------------------------------------------ the residual
def test_the_residual_is_kept_after_a_lift_and_moves_the_next_runs_first_aim(tmp_path):
    cv2_or_skip()
    file = tmp_path / "stand" / "residual.json"
    coarse = (XY[0] + 0.012, XY[1] - 0.006)
    first_aims: list[tuple[float, float]] = []

    def one_run() -> WristGraspStrategy:
        block = block_at(XY, "cube")
        s = started(seen(coarse), residual_file=str(file))

        def on_tick(t: float) -> None:
            if s.grasp.block_xy is not None and len(first_aims) < runs_done[0] + 1:
                first_aims.append(s.grasp.block_xy)

        answers, *_ = follow(s, dict(SPEC.home), seen(coarse), 2500, block=block, frames=camera_of(block),
                             on_tick=on_tick)
        assert answers[-1].success, answers[-1].note
        runs_done[0] += 1
        return s

    runs_done = [0]
    first = one_run()
    assert first.wrist.residual_m == (0.0, 0.0) and first.wrist.residual_runs == 0
    kept = json.loads(file.read_text())
    assert kept["runs"] == 1 and kept["mean_m"] == pytest.approx([-0.012, 0.006], abs=1e-3)
    assert first_aims[0] == pytest.approx(coarse, abs=1e-9)
    second = one_run()
    assert second.wrist.residual_m == pytest.approx((-0.012, 0.006), abs=1e-3)
    assert first_aims[1] == pytest.approx(XY, abs=1e-3)
    assert json.loads(file.read_text())["runs"] == 2


def test_an_unreadable_residual_is_ignored_and_a_failed_run_keeps_none(tmp_path):
    cv2_or_skip()
    file = tmp_path / "residual.json"
    file.write_text("{not json")
    s = started(seen(XY), residual_file=str(file))
    assert s.wrist.residual_m == (0.0, 0.0)
    # No lift, nothing written.
    follow(s, dict(SPEC.home), seen(XY), 50)
    assert file.read_text() == "{not json"
    assert wrist.load_residual(str(tmp_path / "missing.json"))[1] == 0
    wrist.save_residual(str(file), np.array([0.001, -0.002]), 3)
    mean, n = wrist.load_residual(str(file))
    assert n == 3 and mean == pytest.approx([0.001, -0.002])


# ------------------------------------------------------------ settings and the protocol
@pytest.mark.parametrize("settings, match", [
    ({"max_frame_age_s": 0.0}, "grasp_wrist: max_frame_age_s must be positive"),
    ({"hue_deg": 400.0}, r"grasp_wrist: hue_deg must lie within \[0, 360\)"),
    ({"hue_window_deg": 0.0}, r"grasp_wrist: hue_window_deg must lie within \(0, 180\)"),
    ({"min_saturation": 1.5}, r"grasp_wrist: min_saturation must lie within \[0, 1\]"),
    ({"focal_px": -1.0}, "grasp_wrist: focal_px must be zero"),
    ({"tip_pixel_x": 10.0}, "grasp_wrist: tip_pixel_x and tip_pixel_y come together"),
    ({"centre_tol_m": 0.0}, "grasp_wrist: centre_tol_m must be positive"),
    ({"max_shift_m": 0.002}, "grasp_wrist: max_shift_m must exceed centre_tol_m"),
    ({"rounds": 0}, "grasp_wrist: rounds must be at least 1"),
    ({"min_fit_iou": 1.0}, r"grasp_wrist: min_fit_iou must lie within \(0, 1\)"),
    ({"frame_wait_s": -1.0}, "grasp_wrist: frame_wait_s must be zero or more"),
    ({"close_rounds": -1}, "grasp_wrist: close_rounds must be zero or more"),
    ({"close_tol_m": 0.0}, "grasp_wrist: close_tol_m must be positive"),
    ({"correction_gain": 0.0}, r"grasp_wrist: correction_gain must lie within \(0, 2\]"),
    ({"correction_gain": 2.5}, r"grasp_wrist: correction_gain must lie within \(0, 2\]"),
])
def test_settings_are_validated(settings: dict, match: str):
    with pytest.raises(ValueError, match=match):
        WristGraspStrategy(WristGraspSettings(**settings))


def test_start_refuses_without_opencv_or_the_camera(monkeypatch: pytest.MonkeyPatch):
    s = WristGraspStrategy()
    with monkeypatch.context() as patched:
        patched.setattr(wrist, "_optional", lambda name: None)
        with pytest.raises(RuntimeError, match="opencv-python-headless"):
            s.start(run_info(), obs(dict(SPEC.home)))
    cv2_or_skip()
    s = WristGraspStrategy(WristGraspSettings(wrist_camera="nope"))
    with pytest.raises(RuntimeError, match=r"grasp_wrist: camera 'nope' is not in robot.yaml `cameras`"):
        s.start(run_info(), obs(dict(SPEC.home)))
    assert s.wrist.looks == 0


@pytest.mark.parametrize("camera", ["", "wrist"], ids=["no camera", "the defaults"])
def test_the_strategy_follows_the_protocol(camera: str):
    if camera:
        cv2_or_skip()
    cls = WristGraspStrategy
    s = cls(cls.settings(wrist_camera=camera))
    assert isinstance(s.hz, float) and s.hz > 0
    run = RunInfo(robot=SPEC.name, joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits={},
                  max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)
    s.start(run, Observation(t=0.0, step=0, joints=dict(SPEC.home), command={}))
    action = s.act(Observation(t=1 / s.hz, step=1, joints=dict(SPEC.home), command={}))
    assert isinstance(action, Action) and set(action.target) == set(SPEC.joints) and not action.done
    s.stop("zatrzymana")


def test_act_before_start_and_the_notes():
    s = WristGraspStrategy()
    with pytest.raises(RuntimeError, match=r"grasp: act\(\) przed start\(\)"):
        s.act(obs(dict(SPEC.home)))
    cv2_or_skip()
    block = block_at((XY[0] + 0.008, XY[1]))
    s = started(seen())
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 600, block=block, frames=camera_of(block),
                                 until=lambda: s.status.phase == "touched")
    looking = [a.note for a in answers if a.note.startswith(("centruje", "wysrodkowano", "centrowanie"))]
    assert looking[0].startswith("centruje po kamerze: ") and looking[0].endswith(" mm do zawisu")
    assert looking[-1].startswith("wysrodkowano po kamerze: ")
