"""The wrist-camera grasp without the twin: the camera geometry, the colour mask, the box fit and the looks.

The geometry tests check the pinhole, the projection and the box fit against frames drawn by the same
projection. The look tests play the twin's supervisor and the arm as test_grasp.py does (`follow`, the
block's jaw model), and give the strategy wrist frames from a synthetic renderer (`synthetic_frame`): the
table's colour, the block's silhouette in its colour and the jaws in gold, drawn from the arm's measured
joints on every tick. The renders of the lab's own simulation check the measurement on MuJoCo's image;
test_wrist_in_twin.py runs the strategy through the twin's runner on its rendered wrist camera.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, ObjectPose, Observation, RunInfo

import soarm

mujoco = pytest.importorskip("mujoco")

from soarm_lab.kinematics import IKSolution, RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import COLOURS, KINDS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies import wrist  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy, heading, top_down  # noqa: E402
from soarm_lab.strategies.wrist import (  # noqa: E402
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
#: Where the block lies in most tests [m, base frame]: inside the top-down band.
XY = (0.21, 0.02)
NAN = float("nan")
CUBE = object_spec("cube").half_size
#: The synthetic frames' colours: the twin's table and its gold jaws [0..1].
TABLE = (0.55, 0.50, 0.44)
GOLD = (0.80, 0.65, 0.10)
#: The synthetic frames' size (rows, columns) [px]: the working size, so no scaling.
SIZE = (360, 640)
#: The supervisor and the arm per tick: (time [s], the supervisor's gripper command, both tips straddle).
Jaw = Callable[[float, float, bool], float]
#: Wrist frames from the measured joints of a tick.
Frames = Callable[[dict[str, float]], Mapping[str, Frame]]


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
    """A pose as the LLM detector gives one (no orientation), or with a yaw as the box fit gives it."""
    rotation = np.eye(3) if yaw_deg is None else rotation_z(math.radians(yaw_deg))
    return {label: ObjectPose(position=np.array([*xy, 0.0], float), rotation=rotation, source=source,
                              age_s=age_s)}


def started(objects: dict[str, ObjectPose] | None = None, joints: dict[str, float] | None = None,
            strategy: type[GraspStrategy] = WristGraspStrategy, max_vel: dict[str, float] | None = None,
            **settings) -> WristGraspStrategy:
    settings.setdefault("carry_s", 0.5)
    config = (WristGraspSettings if strategy is WristGraspStrategy else GraspSettings)(**settings)
    s = strategy(config)
    s.start(run_info(max_vel=max_vel), obs(dict(joints or SPEC.home), objects=objects))
    return s  # type: ignore[return-value]


def capped(cap_deg_s: float) -> dict[str, float]:
    """The supervisor's speed limits under the panel's speed cap: the arm joints at most `cap_deg_s`."""
    return {name: value if name == SPEC.gripper else min(value, cap_deg_s)
            for name, value in SPEC.max_vel.items()}


def tcp(joints: Mapping[str, float]) -> np.ndarray:
    return KIN.tcp(joints)[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """The phases in order, repeats dropped."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def first_heading() -> float:
    """The closing heading of the rest pose [rad]: the one an identity rotation keeps."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


@dataclass
class Block:
    """A block of the set lying on the table, axis-aligned after its yaw, as the fake arm meets it."""

    xy: np.ndarray
    half: tuple[float, float, float]
    yaw: float = 0.0
    kind: str = "cube"
    #: The jaw holds it: it rides with the fingertips and no longer stalls the arm.
    gripped: bool = False
    path: list[np.ndarray] = field(default_factory=list)

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def local(self, point: np.ndarray) -> np.ndarray:
        return rotation_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    def stalls(self, joints: Mapping[str, float]) -> bool:
        """The arm cannot take `joints`: the jaw's sole below the top with a fingertip over the block."""
        if self.gripped:
            return False
        sole = KIN.tcp(joints)[2, 3] - KIN.jaw_reach(joints[SPEC.gripper])
        if sole >= self.height:
            return False
        return any(bool(np.all(np.abs(self.local(tip)) < self.half[:2])) for tip in KIN.fingertips(joints))

    def between(self, joints: Mapping[str, float]) -> bool:
        """Both tips below the top, on opposite sides of the block's centre along one of its axes."""
        if self.gripped:
            return True
        tips = KIN.fingertips(joints)
        if any(tip[2] >= self.height for tip in tips):
            return False
        a, b = (self.local(tip) for tip in tips)
        return any(a[k] * b[k] < 0 and abs(a[1 - k]) < self.half[1 - k] and abs(b[1 - k]) < self.half[1 - k]
                   for k in (0, 1))

    def units(self, joints: Mapping[str, float]) -> float:
        """The gripper command whose tip gap spans the block across the jaws [app units]: along the block's
        axis the tips lie farther apart on."""
        a, b = (self.local(tip) for tip in KIN.fingertips(joints))
        k = int(np.argmax(np.abs(a - b)))
        return KIN.gripper_for_gap(min(2 * self.half[k], KIN.jaw_gap(100.0)))


def block_at(xy: tuple[float, float] = XY, kind: str = "cube", yaw_deg: float = 0.0) -> Block:
    return Block(np.array(xy, float), object_spec(kind).half_size, math.radians(yaw_deg), kind)


def supervised(command: dict[str, float], target: Mapping[str, float],
               step_deg: float | None) -> dict[str, float]:
    """The supervisor's next command: toward the target, every arm joint by at most `step_deg`."""
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
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor, the arm and the jaw, ending on `done`, as test_grasp.py's.

    The arm follows the supervisor's command at once and stays where it was when `block` says the new pose
    would put the jaw's sole into it; the jaw closes onto the block when both tips straddle it, which then
    rides with the fingertips. `frames` gives each observation's wrist frames from the measured joints;
    `on_tick` runs after each tick with its time; `until` ends the loop after a tick on which it holds.
    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    command = dict(joints)
    t = t0
    g = SPEC.gripper
    assert g is not None
    for _ in range(ticks):
        if measured is not None:
            measured.append(joints)
        seen_now = frames(joints) if frames is not None else None
        answers.append(a := s.act(obs(joints, t=t, objects=objects, command=command, frames=seen_now)))
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
                    size: tuple[int, int] = SIZE) -> Frame:
    """A wrist frame as the camera would see the scene from `joints`: the table's colour, each block's
    silhouette (`wrist.block_points`, `wrist.silhouette`) filled with its kind's colour, and one gold hull
    per jaw body (`wrist.jaw_mask`, undilated) over them. `yaw` [rad]; `focal_px` 0 = the camera's."""
    cv2 = cv2_or_skip()
    image = np.empty((*size, 3), np.uint8)
    image[:] = rgb8(TABLE)
    T = KIN.camera("wrist", joints)
    K = pinhole(CAMERA.fovy_deg, size, focal_px)
    for k, xy, angle in ((kind, tuple(block_xy), yaw), *extra_blocks):
        mask = wrist.silhouette(K, T, wrist.block_points(object_spec(k), xy, angle), size, cv2)
        image[mask] = rgb8(COLOURS[k])
    if jaws:
        image[wrist.jaw_mask(KIN, joints, K, T, size, cv2, 0)] = rgb8(GOLD)
    return Frame(image=image, age_s=age_s)


class LazyFrames(Mapping[str, Frame]):
    """`obs.frames` as the twin's runner gives it: the frame is drawn only when read."""

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
    """Frames of `block` where it lies now, from each tick's measured joints."""
    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        return LazyFrames(lambda: synthetic_frame(joints, block.kind, block.xy, block.yaw, **kwargs))

    return frames


def hover_of(kind: str = "cube", xy: tuple[float, float] = XY, straddle: bool = False,
             yaw_deg: float | None = None) -> tuple[WristGraspStrategy, dict[str, float]]:
    """A started `grasp_wrist` aimed at `xy` and the measured joints at its probe hover (or, `straddle`, at
    the straddle hover with the jaw open), reached without looking."""
    s = started(seen(xy, kind, yaw_deg=yaw_deg), object_kind=kind, look_before_straddle=False,
                need_contact=False)
    measured: list[dict[str, float]] = []
    phase = "straddle" if straddle else "descend"
    # The probe's look skips at once without frames; the straddle is reached with need_contact off.
    s.wrist_settings.frame_wait_s = 0.0
    _, phases, *_ = follow(s, dict(SPEC.home), seen(xy, kind, yaw_deg=yaw_deg), 2000, measured=measured,
                           until=lambda: s.status.phase == phase)
    assert phases[-1] == phase
    s.wrist_settings.frame_wait_s = 1.0
    return s, measured[-1]


def probe_hover(kind: str = "cube", xy: tuple[float, float] = XY) -> dict[str, float]:
    """The joints of `grasp`'s probe hover over the block `kind` at `xy`, from the IK alone: the closed tips'
    midpoint over the block centre at the rest pose's heading, the hand top-down, the TCP at the probe's
    bottom plus `hover_m`. The pose `hover_of` reaches by running the strategy."""
    config = GraspStrategy(GraspSettings(object_kind=kind)).config
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
    """`metres` along the closing axis of `joints` [m, base frame x and y]: image right."""
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
    # Image down leans up, against the approach: rows far below the frame see above the horizon.
    up = wrist.back_project(K, T, np.array([[320.0, 2000.0]]), 0.025)
    assert np.all(np.isnan(up))


def test_box_corners_stand_on_the_table_and_turn_with_the_yaw():
    half = object_spec("cuboid5").half_size
    corners = box_corners(half, (0.2, 0.05), math.radians(35.0))

    assert corners[:, 2].min() == 0.0 and corners[:, 2].max() == pytest.approx(2 * half[2])
    assert corners[:, :2].mean(axis=0) == pytest.approx([0.2, 0.05])
    long_side = corners[4, :2] - corners[0, :2]            # +x against -x, the same y and z
    assert long_side / np.linalg.norm(long_side) == pytest.approx([math.cos(math.radians(35.0)),
                                                                   math.sin(math.radians(35.0))])
    assert np.linalg.norm(long_side) == pytest.approx(2 * half[0])


def test_a_cylinder_is_drawn_by_its_end_circles_and_a_box_by_its_corners():
    spec = object_spec("cylinder")
    yaw = math.radians(-60.0)
    points = wrist.block_points(spec, (0.2, 0.05), yaw)
    axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    relative = points - np.array([0.2, 0.05, spec.half_size[2]])
    along = relative @ axis

    # Two rims of `RIM_POINTS`, at the ends of the 50 mm axis, 12.5 mm from it, lying on the table.
    assert len(points) == 2 * wrist.RIM_POINTS
    assert np.abs(along) == pytest.approx(np.full(len(points), spec.half_size[0]))
    assert np.linalg.norm(relative - np.outer(along, axis), axis=1) == pytest.approx(spec.half_size[1])
    assert points[:, 2].min() == pytest.approx(0.0, abs=1e-12)
    assert points[:, 2].max() == pytest.approx(2 * spec.half_size[2])
    cuboid = object_spec("cuboid5")
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
@pytest.mark.parametrize("kind", KINDS)
def test_the_hsv_window_of_each_kind_holds_its_colour_and_not_the_table(kind: str):
    cv2 = cv2_or_skip()
    window = wrist.hsv_window(COLOURS[kind], 20.0, 0.35, 0.25)
    image = np.empty((24, 24, 3), np.uint8)
    image[:] = rgb8(TABLE)
    image[4:16, 6:18] = rgb8(COLOURS[kind])

    mask = wrist.segment(image, window, cv2)
    assert mask.sum() == 144 and mask[4:16, 6:18].all()
    assert window.hue_deg == pytest.approx({"cube": 4.6, "cuboid5": 45.0, "cuboid85": 140.0,
                                            "cylinder": 285.0}[kind], abs=0.1)
    if kind == "cube":
        # Red wraps around 0: a hue of 356 at the same saturation and value is the cube too.
        _, sat, val = __import__("colorsys").rgb_to_hsv(*COLOURS[kind][:3])
        other = np.array(__import__("colorsys").hsv_to_rgb(356 / 360, sat, val))
        image[4:16, 6:18] = np.round(other * 255).astype(np.uint8)
        assert wrist.segment(image, window, cv2).sum() == 144


def test_the_jaws_are_masked_out_of_the_colour_mask():
    cv2 = cv2_or_skip()
    joints = probe_hover("cuboid5")
    image = synthetic_frame(joints, "cuboid5", XY, math.radians(30.0)).image
    K, T = frame_K_T(joints)
    window = wrist.hsv_window(COLOURS["cuboid5"], 20.0, 0.35, 0.25)
    gold = np.all(image == rgb8(GOLD), axis=2)
    colour = wrist.segment(image, window, cv2)
    jaws = wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)

    # The gold jaws read as the yellow cuboid by colour; the model's hulls take them out.
    assert gold.sum() > 1000 and (colour & gold).sum() > 0.9 * gold.sum()
    labels, found = wrist.blobs(colour & ~jaws, cv2)
    assert len(found) == 1 and not (colour & ~jaws & gold).any()


# ------------------------------------------------------------ the box fit
FIT_CASES = [("cube", 0.0, (0.0, 0.0)), ("cube", 20.0, (0.008, -0.006)), ("cuboid5", 35.0, (0.0, 0.012)),
             ("cuboid5", 125.0, (-0.01, 0.0)), ("cuboid85", -70.0, (0.0, -0.01)),
             ("cylinder", -60.0, (0.015, 0.0))]


@pytest.mark.parametrize("kind, yaw_deg, offset", FIT_CASES, ids=[f"{k} {y:.0f}" for k, y, _ in FIT_CASES])
def test_the_fit_recovers_a_box_drawn_by_the_same_projection(kind: str, yaw_deg: float,
                                                            offset: tuple[float, float]):
    cv2 = cv2_or_skip()
    joints = probe_hover(kind)
    truth = np.asarray(XY) + offset
    image = synthetic_frame(joints, kind, truth, math.radians(yaw_deg)).image
    K, T = frame_K_T(joints)
    spec = object_spec(kind)
    visible = ~wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)
    labels, found = wrist.blobs(wrist.segment(image, wrist.hsv_window(COLOURS[kind], 20.0, 0.35, 0.25), cv2)
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
    labels, found = wrist.blobs(wrist.segment(image, wrist.hsv_window(COLOURS["cube"], 20.0, 0.35, 0.25),
                                              cv2), cv2)
    centroid = wrist.back_project(K, T, found[0].centroid_px, 0.025)[0, :2]
    xy0, yaw0 = wrist.blob_pose(labels, found[0], K, T, 0.025, math.pi / 2, cv2)
    fit = wrist.fit_block(labels == found[0].label, K, T, object_spec("cube"), xy0, yaw0, cv2)

    # The visible side face pulls the plain centroid toward the camera; the box fit does not.
    assert np.linalg.norm(centroid - XY) > 0.003
    assert np.linalg.norm(fit.xy - XY) < 5e-4


def test_a_side_the_image_cannot_place_keeps_the_aims_position_along_it():
    """A cuboid85 lying along the view leaves the frame at the top and hides under the jaws at the bottom:
    its position along its long side is not in the image, so the aim's stands there."""
    cv2_or_skip()
    s, joints = hover_of("cuboid85")
    view = np.array([-math.sin(heading(KIN, joints)), math.cos(heading(KIN, joints))])   # image up and down
    yaw = math.atan2(view[1], view[0])
    truth = np.asarray(XY) + 0.006 * view + along_closing(joints, 0.004)
    m = s.measure(synthetic_frame(joints, "cuboid85", truth, yaw).image, joints)

    assert m.problem == ""
    # Across the block the image places it; along it the aim stands.
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
        # A quarter of the box's pixels in one corner: the top-left quadrant of its rows and columns.
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


@pytest.mark.parametrize("where", ["side by side", "one behind the other"])
def test_two_touching_blocks_of_one_colour_are_one_blob_too_large_for_the_box(where: str):
    cv2_or_skip()
    s, joints = hover_of("cube")
    side = along_closing(joints, 1.0)
    # Across the image, or along the view with the second block farther away, where it looks smaller.
    step = 0.025 * (side if where == "side by side" else np.array([-side[1], side[0]]))
    first = np.asarray(XY) + 0.2 * step
    second = (("cube", tuple(first + step), 0.0),)
    image = synthetic_frame(joints, "cube", first, 0.0, extra_blocks=second).image
    m = s.measure(image, joints)

    assert m.problem.startswith("plama w kamerze nie pasuje rozmiarem (x")
    assert float(m.problem.split("(x")[1].rstrip(")")) >= (1.6 if where == "side by side" else 1.4)
    assert np.all(np.isnan(m.xy))


def test_a_speck_at_the_jaws_edge_is_not_a_second_block():
    """A sliver of the block's colour along the jaw mask, 3 cm from the aim: the only block it fits lies
    behind the jaws, and shows 2 % of itself, so it is no candidate beside the real block."""
    cv2 = cv2_or_skip()
    s, joints = hover_of("cube")
    image = synthetic_frame(joints, "cube", XY, 0.0).image
    K, T = frame_K_T(joints)
    visible = ~wrist.jaw_mask(KIN, joints, K, T, SIZE, cv2, wrist.JAW_MASK_DILATE_PX)
    speck = np.zeros(SIZE, bool)
    speck[234:264, 232:240] = True
    speck &= visible
    image[speck] = rgb8(COLOURS["cube"])
    # The sliver lies in range and its fitted block's area would pass the window: only the share drops it.
    rows, cols = np.nonzero(speck)
    centre = np.array([[cols.mean(), rows.mean()]])
    assert np.linalg.norm(wrist.back_project(K, T, centre, 0.025)[0, :2] - XY) < 0.04
    spec = object_spec("cube")
    blob = wrist.Blob(int(speck.sum()), centre[0], 1)
    xy0, yaw0 = wrist.blob_pose(speck.astype(np.int32), blob, K, T, 0.025, spec.yaw_period, cv2)
    fit = wrist.fit_block(speck, K, T, spec, xy0, yaw0, cv2, visible=visible)
    points = wrist.block_points(spec, fit.xy, fit.yaw)
    model = int(np.count_nonzero(wrist.silhouette(K, T, points, SIZE, cv2) & visible))
    assert wrist.AREA_WINDOW[0] <= speck.sum() / model <= wrist.AREA_WINDOW[1]
    assert model < wrist.MIN_VISIBLE_SHARE * wrist.hull_area(K, T, points, cv2)

    m = s.measure(image, joints)
    assert m.problem == "" and m.xy == pytest.approx(XY, abs=5e-4)


@pytest.mark.parametrize("kind", KINDS)
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
    # The camera sees the tips 10 px to the right of the model: everything the model draws lies 10 px too
    # far left, so the block measures 10 px further left, toward the fixed jaw (-closing).
    ends = wrist.back_project(K, T, np.array([[320.0, 120.0], [330.0, 120.0]]), 0.025)
    mm_per_px = float(np.linalg.norm(ends[1, :2] - ends[0, :2])) / 10
    assert moved.xy - plain.xy == pytest.approx(-10 * mm_per_px * along_closing(joints, 1.0), abs=3e-4)
    assert 0.0002 < mm_per_px < 0.0004

    s.wrist_settings.tip_pixel_x = predicted[0] + 200.0
    assert s.measure(image, joints).problem.startswith("tip_pixel daleko od modelu (200 px)")


def test_rotate_180_undoes_an_upside_down_camera():
    cv2_or_skip()
    s, joints = hover_of("cuboid5")
    image = synthetic_frame(joints, "cuboid5", np.asarray(XY) + (0.004, -0.006), 0.5).image
    plain = s.measure(image, joints)
    s.wrist_settings.rotate_180 = True
    turned = s.measure(np.ascontiguousarray(image[::-1, ::-1]), joints)

    assert plain.problem == turned.problem == ""
    assert turned.xy == pytest.approx(plain.xy, abs=1e-9) and turned.yaw == pytest.approx(plain.yaw)
    assert turned.pixel == pytest.approx([639 - plain.pixel[0], 359 - plain.pixel[1]])
    # The tip pixel is read off the frame as received: on the turned frame the same calibration measures
    # alike.
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


RENDER_CASES = [("cube", 20.0), ("cuboid5", 35.0), ("cuboid85", -70.0), ("cylinder", -60.0),
                ("cylinder", 0.0)]


@pytest.mark.render
@pytest.mark.parametrize("kind, yaw_deg", RENDER_CASES, ids=[f"{k} {y:.0f}" for k, y in RENDER_CASES])
def test_measure_on_the_sims_own_render_finds_the_block(kind: str, yaw_deg: float):
    """MuJoCo's image of the lab's simulation, from the scene's `robot/wrist` camera at 1920x1080: shading,
    the gold jaws' meshes and the side faces as rendered."""
    cv2_or_skip()
    from soarm_lab import sim
    from soarm_lab.rl.objects import scene_object
    from soarm_lab.station import Station

    aim = (0.2, 0.0)
    truth = np.asarray(aim) + (0.012, 0.0)
    s, joints = hover_of(kind, aim)
    scene = sim.build(sim.SceneConfig(Station.load("sim"), objects=[scene_object(kind, kind, tuple(truth))]))
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
    error_deg = abs(math.degrees(fold(m.yaw - yaw, object_spec(kind).yaw_period)))
    # The cylinder too: its silhouette is drawn from its end circles, not its bounding box.
    assert error_mm < 1.0 and error_deg < 2.0, (error_mm, error_deg)


# ------------------------------------------------------------ the looks
def test_the_look_centres_the_probe_on_the_block_before_the_descent():
    cv2_or_skip()
    truth = (0.205, 0.03)
    coarse = (truth[0] + 0.009, truth[1] - 0.012)                    # 15 mm off
    block = block_at(truth, "cuboid5", 25.0)
    s = started(seen(coarse, "cuboid5"), object_kind="cuboid5")
    measured: list[dict[str, float]] = []
    aims: list[tuple[str, tuple[float, float] | None]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(coarse, "cuboid5"), 2500, block=block,
                                 frames=camera_of(block), measured=measured,
                                 on_tick=lambda t: aims.append((s.status.phase, s.grasp.block_xy)))

    assert answers[-1].done and answers[-1].success, answers[-1].note
    assert s.wrist.verdicts == ["centred", "centred"] and s.wrist.looks == 2
    assert s.status.attempt == 1
    assert runs(phases)[:3] == ["approach", "descend", "touched"]
    descent = [k for k, p in enumerate(phases) if p == "descend"]
    notes = [answers[k].note for k in descent]
    assert notes[0].startswith("centruje po kamerze")
    assert any(n.startswith("centruje po kamerze: przesuwam o 15 mm, obrot 59 st.") for n in notes)
    assert any(n.startswith("wysrodkowano po kamerze: 0 mm") for n in notes)
    # The probe aimed within 2 mm of the block's centre, the jaws across its short side: 25 + 90 degrees
    # folded by the cuboid's 180, the one nearest the start heading.
    probe_aim = [xy for p, xy in aims if p == "touched"][0]
    assert probe_aim is not None and np.linalg.norm(np.asarray(probe_aim) - truth) < 0.002
    assert s.grasp.heading_deg == pytest.approx(-65.0, abs=2.0)
    # The wrist really rolled before the probe left the hover.
    first_down = next(k for k in descent if not answers[k].note.startswith(("centruje", "wysrodkowano")))
    _, closing = KIN.tool_axes(measured[first_down])
    rolled = math.degrees(fold(math.atan2(closing[1], closing[0]) - math.radians(-65.0), math.pi))
    assert abs(rolled) < 4.0


def test_the_look_waits_for_the_wrist_roll_before_it_measures():
    cv2_or_skip()
    truth = (0.205, 0.03)
    block = block_at(truth, "cuboid5", 60.0)
    s = started(seen((truth[0] + 0.002, truth[1]), "cuboid5"), object_kind="cuboid5")
    reads: list[int] = []
    frames_now: list[LazyFrames] = []

    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        lazy = LazyFrames(lambda: synthetic_frame(joints, "cuboid5", block.xy, block.yaw))
        frames_now.append(lazy)
        return lazy

    measured: list[dict[str, float]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen((truth[0] + 0.002, truth[1]), "cuboid5"), 400,
                                 block=block, frames=frames, measured=measured,
                                 until=lambda: s.status.phase == "touched")
    reads = [k for k, lazy in enumerate(frames_now) if lazy.reads]
    assert len(reads) == 2 and s.wrist.verdicts == ["centred"]
    first, second = reads
    assert answers[first].note.startswith("centruje po kamerze: przesuwam o 2 mm, obrot")
    roll = SPEC.joints.index("wrist_roll")
    targets = [KIN.to_q(a.target) for a in answers[first - 1:second]]
    cap = math.radians(SPEC.max_vel["wrist_roll"] * 0.8 / s.hz)
    assert max(abs(b[roll] - a[roll]) for a, b in zip(targets, targets[1:], strict=False)) <= cap + 1e-9
    assert math.degrees(abs(targets[-1][roll] - targets[0][roll])) > 20.0
    # Measured only after the arm stood still for the settle window: 5 ticks with every arm joint within
    # 0.5 degrees of the tick before.
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
    s = started(seen(XY), look_before_straddle=look)
    pushed: list[np.ndarray] = []
    straddle: list[tuple[np.ndarray, np.ndarray]] = []

    def on_tick(t: float) -> None:
        if s.status.phase == "rise" and not pushed:
            # The probe pushed the block 6 mm along the closing heading.
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
    assert (fixed[:2] - centre) @ u == pytest.approx(-(0.0125 + 0.008), abs=1e-3)
    assert (moving[:2] - centre) @ u == pytest.approx(0.0125 + 0.008, abs=1e-3)
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
    s = started(seen(coarse))
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


@pytest.mark.parametrize("case", ["no block", "two blocks", "a far block"])
def test_no_blob_two_blobs_and_a_far_block_skip_the_look(case: str):
    cv2_or_skip()
    s = started(seen(), look_before_straddle=False)
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
    """Frames that claim the block lies `gain` times as far from the aim as it does, at its true size."""

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
    s = started(seen(coarse), look_before_straddle=False)
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
    # Back at the old hover before the descent goes on.
    leaving = next(k for k, (p, v, _, _) in enumerate(log) if p == "descend" and v == "reverted"
                   and answers[k].note.startswith("w dol"))
    assert np.linalg.norm(log[leaving - 1][2] - hover0) < s.ik_tol + 1e-9
    assert s.status.attempt == 1


def test_a_revert_the_arm_cannot_take_keeps_the_moved_aim(monkeypatch: pytest.MonkeyPatch):
    cv2_or_skip()
    side = along_closing(dict(SPEC.home, gripper=0.0), 1.0)
    block = block_at(tuple(np.asarray(XY) + 0.012 * side))
    s = started(seen(), look_before_straddle=False)
    lie = lying(s, block, 2.0)
    reads: list[int] = []
    real_ik = s.kin.ik

    def frames(joints: dict[str, float]) -> Mapping[str, Frame]:
        inner = lie(joints)
        return LazyFrames(lambda: (reads.append(1), inner["wrist"])[1])

    def ik(*args, **kwargs) -> IKSolution:
        if len(reads) >= 2:                      # the revert's reach check, after the second frame
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
    # A focal length 10 % longer than the strategy assumes: each move leaves a tenth of the offset.
    s = started(seen(), max_vel=capped(20.0), rounds=2, centre_tol_m=0.0005, look_before_straddle=False)
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
    s = started(seen(), look_before_straddle=False)
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
    """A servo whose reading jitters by a degree is never still for the look: skipped after LOOK_STALL_S."""
    cv2_or_skip()
    block = block_at()
    s = started(seen(), look_before_straddle=False)
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
