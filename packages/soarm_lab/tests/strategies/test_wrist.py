"""The wrist camera's geometry without the twin: the pinhole, the projection, the colour mask and the fit.

The tests check the pinhole, the projection and the block fit against frames drawn by the same projection
(`synthetic_frame`: the table's colour, the block's silhouette in its colour and the jaws in gold), from the
probe hover `grasp` takes over a block.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from machinekind_twin_api.strategy import Frame

mujoco = pytest.importorskip("mujoco")

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import COLOURS, KINDS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies import wrist  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy, heading, top_down  # noqa: E402
from soarm_lab.strategies.wrist import box_corners, fold, pinhole  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
CAMERA = SPEC.cameras["wrist"]
#: Where the block lies in most tests [m, base frame]: inside the top-down band.
XY = (0.21, 0.02)
#: The synthetic frames' colours: the twin's table and its gold jaws [0..1].
TABLE = (0.55, 0.50, 0.44)
GOLD = (0.80, 0.65, 0.10)
#: The synthetic frames' size (rows, columns) [px]: the working size, so no scaling.
SIZE = (360, 640)


def cv2_or_skip():
    return pytest.importorskip("cv2", reason="the wrist camera's tests need OpenCV")


def first_heading() -> float:
    """The closing heading of the rest pose [rad]: the one an identity rotation keeps."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


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


# ------------------------------------------------------------ the fit
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
