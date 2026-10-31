"""Wrist-camera centring as a strategy for the twin: `grasp_wrist` under `machinekind_twin.strategies`.

`WristGraspStrategy` is `grasp` with a look through the wrist camera at each hover: before the probe and
again at the straddle hover with the jaw open. The twin builds `WristGraspSettings` from its config file's
`grasp_wrist:` section or the panel's form and runs the strategy through its `StrategyRunner`, handing it
the camera's frames in `obs.frames`. Nothing here imports the twin.

    # configs/local.yaml of the twin: prefills the panel's form
    grasp_wrist:
      object_kind: cuboid5         # cube, cuboid5, cuboid85 or cylinder: the block's size [rl.objects]
      wrist_camera: wrist          # the camera in obs.frames (robot.yaml `cameras:`); "" = plain grasp
      max_frame_age_s: 0.2         # an older frame is not measured [s]
      frame_wait_s: 1.0            # how long a look waits for a usable frame [s]
      hue_deg: -1.0                # the block's hue [deg, 0..360); -1 = the kind's colour [rl.objects]
      hue_window_deg: 20.0         # half-width of the hue window [deg]
      min_saturation: 0.35         # less saturated pixels are not the block [0..1]
      min_value: 0.25              # darker pixels are not the block [0..1]
      focal_px: 0.0                # the camera's focal length [px]; 0 = from robot.yaml's fovy
      tip_pixel_x: -1.0            # the closed tips' midpoint in the frame [px]; -1 = the model's
      tip_pixel_y: -1.0            # its row [px]; -1 = the model's
      rotate_180: false            # the frame is upside down (the camera mounted turned)
      centre_tol_m: 0.003          # a look ends with the block this close to the aim [m]
      max_shift_m: 0.04            # a block measured farther from the aim is not trusted [m]
      rounds: 2                    # the most moves of the aim per look [count]
      min_fit_iou: 0.5             # a poorer fit is not the block [0..1]
      look_before_straddle: true   # look again at the straddle hover with the jaw open

`grasp`'s settings come first in the form (see `grasp.py`).

Method. The wrist camera rides on the gripper (`robot.yaml` `cameras: wrist`, the MJCF's `wrist_cam`):
53 mm behind the TCP along the approach, 55 mm to its side and 12 mm across, looking 33 degrees off the
approach toward the jaws, image right along the closing axis. At the probe hover (the TCP 37 mm above a
25 mm block's top) the block's top lies 105 mm along the optical axis, 0.26 mm per pixel at 640x360; frames
wider than `WORK_WIDTH_PX` (the MJCF renders 1920x1080) are scaled down to 640 columns first. A
measurement, on the tick thread, takes 9-18 ms:

1. the camera's pose in the base frame from the arm's measured joints (`RobotKinematics.camera`) and an
   ideal pinhole from the frame's height and the field of view (`pinhole`, or `focal_px`), its principal
   point moved by the stand's tip-pixel calibration when one is set;
2. the block's colour in an HSV window (`segment`); the arm's own jaws (gold in the twin, the hue of
   cuboid5) are masked out from the model, one convex hull per jaw body of its collision primitives
   (`jaw_mask`), and only the pixels outside that mask count from here on;
3. the colour blobs (`blobs`) whose centroid, back-projected onto the block's top plane, lies within
   `max_shift_m` + `SEARCH_SLACK_M` of the aim; the gold jaws' remnants lie 73 mm (probe hover) and 101 mm
   (straddle hover) away, outside it;
4. the block's KNOWN shape (`rl.objects.object_spec`: a box's corners, a cylinder's end circles,
   `block_points`) fitted to each blob by the IoU of its silhouette (`fit_block`): the block's centre and
   yaw in the base frame directly. The visible side face moves a plain centroid by 6 mm toward the camera;
   the fit lands within 0.5 mm and 1.5 degrees of the truth on the simulation's render, every kind of the
   set at both hovers (IoU 0.96-0.99). A side the image cannot place (both of its ends out of the frame or
   behind the jaws, a cuboid85 lying along the view) keeps the aim's position along it (`FREE_SHIFT_M`);
5. the blob is the block when the block fitted to it shows at least `MIN_VISIBLE_SHARE` of itself and its
   area matches that block's visible silhouette (`AREA_WINDOW`): one block reads 0.94-1.01, two touching
   blocks of one colour 1.52-1.83.

The look then re-aims through `grasp`'s own `_aim` with the measured yaw as the block's rotation, so the
jaws turn across the short side, and drives to the new hover with the top-down hand (`grasp._drive` carries
the heading), so the wrist rolls to the new heading under the roll's speed limit. It waits until every arm
joint stood still (`JOINT_STILL_DEG`) for touch's settle window, the roll included, which `_settled` alone
does not see, and measures again: that measurement verifies the move.

Guards (grasp's, and):

* **the frame**: a frame older than `max_frame_age_s` (`Frame.age_s`) is not measured; without a usable
  frame for `frame_wait_s` the look is skipped;
* **the jaw mask**: the jaws come out of the image from the model, never by colour;
* **the blob**: no blob in range, a blob whose area does not fit one block (two touching blocks; in the
  twin also a yellow block more than 3 cm toward the camera, merged with the gold jaws), or two
  candidates skip the look; the probe decides;
* **the fit**: an IoU under `min_fit_iou` skips the look;
* **the shift cap**: a block measured more than `max_shift_m` from the aim is not trusted;
* **the move guard**: after a move the block must lie within `MOVE_GUARD` of the move plus `centre_tol_m`
  of the new aim. A turned or mirrored frame, or a wrong mount, sends the arm the wrong way and shows up
  here: the aim reverts to where the look found it and the arm goes back there. A focal-length error does
  not: the area window rejects it beyond x1.2 or below x0.6, and inside that it converges;
* **the reach**: a re-aim the arm cannot take top-down (`_aim` raises) keeps the old aim;
* **the arm**: a look whose arm does not stand still at the hover within `LOOK_STALL_S` of its target
  stopping is skipped;
* **OpenCV**: with a camera named, `start` refuses a run without OpenCV, naming the package; a camera
  not in robot.yaml is refused too. `wrist_camera: ""` is plain `grasp`.

A look ends "centred" when the block lies within `centre_tol_m` of the aim and its yaw within
`YAW_TOL_DEG` of the heading's, or after `rounds` moves with the last aim standing; "skipped" on every
guard above, with a note that says why; "reverted" after the move guard. A look never counts an attempt,
never retargets and never ends a run: after any verdict the probe, or the straddle, goes on as `grasp`'s.

Phases: the looks ride on `grasp`'s seams, `before_descent` (every DESCEND tick until it returns None once,
before the probe leaves the hover) and `before_straddle` with `_looking` (in OPEN, at the straddle hover
with the jaw open and the hand top-down). `WristStatus` keeps what the looks saw.

Stand calibration: sample the block in the panel and set `hue_deg` and the thresholds per block when the
lighting differs from the sim's colours; `focal_px` from the camera's intrinsics when its field of view is
not the MJCF's 48.46 degrees; `tip_pixel_x/y`, the pixel where the closed tips' midpoint appears in the
frame (the model puts it at (902.6, 799.2) of 1920x1080, (300.5, 266.1) of 640x360), which corrects a
turned mount to first order; `rotate_180` for a camera mounted upside down.
"""

from __future__ import annotations

import colorsys
import importlib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from ..kinematics import RobotKinematics
from ..rl.objects import COLOURS, ObjectSpec
from ..robot import RobotCamera
from .grasp import GraspSettings, GraspStrategy, wrap

logger = logging.getLogger(__name__)

#: The frame is measured at most this wide [px]: a wider one is scaled down first (INTER_AREA), so the
#: MJCF camera's 1920x1080 is measured at 640x360.
WORK_WIDTH_PX = 640
#: A blob counts when its back-projected centroid lies within `max_shift_m` plus this of the aim [m]: the
#: side face moves a centroid by up to 1 cm.
SEARCH_SLACK_M = 0.015
#: A blob's area over the silhouette of the known block fitted to it, both outside the jaw mask. Measured on
#: the simulation's render at both hovers, every kind of the set: one block reads 0.94-1.01 anywhere within
#: 4 cm of the aim, cut by the frame's edge or partly behind the jaws; two touching blocks of one colour read
#: 1.83 side by side and 1.52 one behind the other along the view (the far one smaller); a yellow block more
#: than 3 cm toward the camera merges with the twin's gold jaws and reads 1.34-1.36; a quarter of a block
#: 0.25.
AREA_WINDOW = (0.55, 1.25)
#: Smaller connected components are noise, not looked at [px].
MIN_BLOB_PX = 64
#: A blob counts only when the block fitted to it shows at least this share of its silhouette inside the
#: frame and outside the jaw mask. On the simulation's render a block within 4 cm of the aim shows 0.28 or
#: more of itself (a cube 4 cm toward the camera at the probe hover, the rest below the frame and behind the
#: jaws); a speck of the block's colour at the open jaws' edge fits a block that shows 0.00-0.02, a gold
#: jaw remnant one that shows 0.05.
MIN_VISIBLE_SHARE = 0.2
#: Every arm joint moved less than this between ticks [deg]: the look's own stillness, which sees the wrist
#: roll `_settled` does not (the TCP stays put while the wrist rolls).
JOINT_STILL_DEG = 0.5
#: The dilation of the jaw mask at a 640 px wide frame [px], scaled by the width. The collision primitives
#: cover the rendered jaws but for a few remnants away from the block.
JAW_MASK_DILATE_PX = 12
#: Jaw points nearer the camera than this are not projected [m].
MIN_JAW_DEPTH_M = 0.02
#: The fit's pattern search: per level the step in x and y [m] and in yaw [deg].
FIT_STEPS = ((0.008, 8.0), (0.002, 2.0), (0.0005, 0.5))
#: The most silhouettes one fit draws.
FIT_MAX_EVALS = 300
#: Points on each end circle of a cylinder part when its silhouette is drawn: the polygon lies within
#: 0.1 mm of the 12.5 mm circle.
RIM_POINTS = 24
#: A side the image cannot place: moving the fitted block this far along it either way [m] costs less
#: than `FREE_IOU_DROP` of IoU, as when both of its ends leave the frame or hide behind the jaws (a cuboid85
#: lying along the view). The aim's own position along that side stands.
FREE_SHIFT_M = 0.003
FREE_IOU_DROP = 0.01
#: A look ends when the block's yaw lies this close to the one the heading grips across [deg].
YAW_TOL_DEG = 3.0
#: After a move the residual the image shows may be at most this share of the move plus `centre_tol_m`. A
#: direction error of theta leaves 2 sin(theta / 2) of the move, so the guard fires beyond about 40
#: degrees, and a mirrored or turned frame at once; a focal-length error k leaves |1 - k| of the move.
MOVE_GUARD = 0.5
#: The square kernel of the colour mask's opening [px]: a patch smaller than it does not survive.
OPEN_KERNEL_PX = 3
#: The share of the frame's width a calibrated tip pixel may lie from the model's.
MAX_TIP_SHIFT = 0.2
#: A look whose arm has not stood still at the hover this long after its target stopped moving is skipped
#: [s]: a servo that jitters past `JOINT_STILL_DEG`, or a roll the arm cannot finish.
LOOK_STALL_S = 3.0
#: The target moved on a tick when an arm joint's target moved more than this [deg].
TARGET_MOVED_DEG = 0.05
#: Sub-pixel precision of the drawn silhouettes: fixed-point bits of `cv2.fillConvexPoly`.
_SHIFT = 4

#: The phases of a look.
MEASURE = "measure"
RETURN = "return"
DONE = "done"


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class HsvWindow:
    """The colours that count as the block: a hue window [deg] and the least saturation and value [0..1]."""

    hue_deg: float
    window_deg: float
    min_s: float
    min_v: float


@dataclass(frozen=True)
class Blob:
    """One connected component of the block's mask: its area [px], centroid (u, v) [px] and label."""

    area: int
    centroid_px: np.ndarray
    label: int


@dataclass(frozen=True)
class Fit:
    """The known block fitted to a blob: its centre x and y [m, base frame], yaw [rad], the IoU, the draws,
    and which of its sides (x, y) the image cannot place (`FREE_SHIFT_M`)."""

    xy: np.ndarray
    yaw: float
    iou: float
    evals: int
    free: tuple[bool, bool] = (False, False)


def pinhole(fovy_deg: float, shape: tuple[int, ...], focal_px: float = 0.0) -> np.ndarray:
    """The 3x3 intrinsics of an ideal pinhole for a frame of `shape` (height, width, ...): the focal length
    `focal_px`, or (H / 2) / tan(fovy / 2) from the frame's height; the principal point at the centre
    ((W - 1) / 2, (H - 1) / 2), the twin's convention. 1200 px at 1080 rows, 400 at 360 for the MJCF's."""
    height, width = int(shape[0]), int(shape[1])
    f = focal_px if focal_px > 0 else (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def project(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points (N, 3) as pixels (N, 2) (u = column, v = row) and their depths along the optical
    axis (N,) [m], through the camera pose `T_base_cam` (OpenCV axes) and the intrinsics `K`."""
    points = np.atleast_2d(np.asarray(points, float))
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p = (points - t) @ R
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def back_project(K: np.ndarray, T_base_cam: np.ndarray, pixels: np.ndarray, z: float) -> np.ndarray:
    """Pixels (N, 2) as the points (N, 3) where their rays meet the plane at height `z` [m, base frame]; NaN
    rows for rays that do not go down."""
    pixels = np.atleast_2d(np.asarray(pixels, float))
    rays = np.column_stack([(pixels[:, 0] - K[0, 2]) / K[0, 0], (pixels[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(pixels))]) @ T_base_cam[:3, :3].T
    origin = T_base_cam[:3, 3]
    down = rays[:, 2] < -1e-9
    out = np.full((len(pixels), 3), np.nan)
    out[down] = origin + rays[down] * ((z - origin[2]) / rays[down, 2])[:, None]
    return out


def rotation_z(yaw: float) -> np.ndarray:
    """The rotation by `yaw` [rad] about the vertical."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_corners(half: tuple[float, float, float] | np.ndarray, xy: np.ndarray | tuple[float, float],
                yaw: float) -> np.ndarray:
    """The 8 corners (8, 3) of a box of half sizes `half` standing on the table (its bottom at z = 0) with
    its centre at `xy` and its x side turned by `yaw` [rad] about the vertical [m, base frame]."""
    hx, hy, hz = (float(v) for v in half)
    local = np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 2 * hz)])
    out = local @ rotation_z(yaw).T
    out[:, 0] += float(xy[0])
    out[:, 1] += float(xy[1])
    return out


def _quat_mat(q: tuple[float, float, float, float]) -> np.ndarray:
    """The rotation of a unit quaternion (w, x, y, z), MuJoCo's order."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def block_points(spec: ObjectSpec, xy: np.ndarray | tuple[float, float], yaw: float) -> np.ndarray:
    """Points (N, 3) whose convex hull is the block `spec` lying on the table with its centre at `xy`, its x
    side turned by `yaw` [rad] about the vertical [m, base frame]: the 8 corners of a box part, the two end
    circles of a cylinder part (`RIM_POINTS` each). A cylinder is the convex hull of its end discs, so the
    hull of the projected rims is its silhouette; its bounding box is not (on the simulation's render a lying
    cylinder fitted as its box landed up to 5.8 mm off across its axis, at IoU 0.72-0.87). The body's origin
    lies `half_size[2]` above the table, as `rl.objects` builds it."""
    local = []
    for part in spec.parts:
        if part.kind == "box":
            hx, hy, hz = part.size
            points = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1)
                               for sz in (-1, 1)])
        elif part.kind == "cylinder":
            radius, half_length = part.size[0], part.size[1]
            angle = np.linspace(0.0, 2 * math.pi, RIM_POINTS, endpoint=False)
            ring = radius * np.column_stack([np.cos(angle), np.sin(angle)])
            points = np.vstack([np.column_stack([ring, np.full(RIM_POINTS, z)])
                                for z in (-half_length, half_length)])
        else:
            raise ValueError(f"{spec.kind}: a part of kind {part.kind!r} cannot be drawn")
        local.append(points @ _quat_mat(part.quat).T + np.asarray(part.pos, float))
    body = np.vstack(local)
    body[:, 2] += spec.half_size[2]
    out = body @ rotation_z(yaw).T
    out[:, 0] += float(xy[0])
    out[:, 1] += float(xy[1])
    return out


def fold(angle: float, period: float) -> float:
    """`angle` folded into (-period/2, period/2] [rad]."""
    folded = (angle + period / 2) % period - period / 2
    return folded + period if folded <= -period / 2 + 1e-12 else folded


def hsv_window(rgba: tuple[float, ...], window_deg: float, min_s: float, min_v: float) -> HsvWindow:
    """The window around the hue of the colour `rgba` (0..1 channels): cube 4.6, cuboid5 45, cuboid85 140,
    cylinder 285 degrees for the set's colours."""
    hue, _, _ = colorsys.rgb_to_hsv(float(rgba[0]), float(rgba[1]), float(rgba[2]))
    return HsvWindow(hue * 360.0, window_deg, min_s, min_v)


def segment(rgb: np.ndarray, window: HsvWindow, cv2: Any) -> np.ndarray:
    """The pixels of `rgb` (H, W, 3, uint8) within the colour window, as a bool mask, opened by an
    `OPEN_KERNEL_PX` square so patches smaller than it are dropped. The hue window wraps at 360."""
    image = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0                 # OpenCV's 8-bit hue runs 0..179
    dh = np.abs((hue - window.hue_deg + 180.0) % 360.0 - 180.0)
    mask = ((dh <= window.window_deg) & (hsv[..., 1] >= window.min_s * 255.0)
            & (hsv[..., 2] >= window.min_v * 255.0)).astype(np.uint8)
    kernel = np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)


def _fill(mask: np.ndarray, pixels: np.ndarray, cv2: Any) -> None:
    """Fills the convex hull of `pixels` (N, 2) into the uint8 `mask` at sub-pixel precision; clipped at the
    frame's edges."""
    if len(pixels) < 3:
        return
    fixed = np.round(np.clip(pixels, -1e5, 1e5) * (1 << _SHIFT)).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(fixed), 1, cv2.LINE_8, _SHIFT)


def silhouette(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray, shape: tuple[int, ...],
               cv2: Any) -> np.ndarray:
    """The convex hull of `points` (base frame) as the camera sees it, filled, as a bool mask of `shape`.
    Points behind the camera are dropped."""
    pixels, depth = project(K, T_base_cam, points)
    mask = np.zeros(shape[:2], np.uint8)
    _fill(mask, pixels[depth > 1e-6], cv2)
    return mask.astype(bool)


def hull_area(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray, cv2: Any) -> float:
    """The area of the convex hull of `points` (base frame) as the camera sees it, unclipped by the frame's
    edges [px]; points behind the camera are dropped."""
    pixels, depth = project(K, T_base_cam, points)
    front = pixels[depth > 1e-6]
    if len(front) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(front.astype(np.float32))))


def jaw_mask(kin: RobotKinematics, joints: Mapping[str, float], K: np.ndarray, T_base_cam: np.ndarray,
             shape: tuple[int, ...], cv2: Any, dilate_px: int) -> np.ndarray:
    """Where the arm's own jaws lie in the image, from the model: per body of `spec.jaw_bodies` one convex
    hull of its collision primitives' extreme points (`collision_points`), those nearer the camera than
    `MIN_JAW_DEPTH_M` dropped, dilated by a (2 `dilate_px` + 1) square. One hull per body, never one over
    both: that would bridge the gap between the open jaws and cover the block."""
    mask = np.zeros(shape[:2], np.uint8)
    for body in kin.spec.jaw_bodies:
        if not body:
            continue
        pixels, depth = project(K, T_base_cam, kin.collision_points(body, joints))
        _fill(mask, pixels[depth > MIN_JAW_DEPTH_M], cv2)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return mask.astype(bool)


def blobs(mask: np.ndarray, cv2: Any) -> tuple[np.ndarray, list[Blob]]:
    """The connected components (8-connected) of `mask` of `MIN_BLOB_PX` or more: the label image and the
    blobs, largest first."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = [Blob(int(stats[k, cv2.CC_STAT_AREA]), np.asarray(centroids[k], float), k) for k in range(1, n)
             if stats[k, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    return labels, sorted(found, key=lambda blob: -blob.area)


def blob_pose(labels: np.ndarray, blob: Blob, K: np.ndarray, T_base_cam: np.ndarray, z: float,
              yaw_period: float, cv2: Any) -> tuple[np.ndarray, float]:
    """Where a blob says the block is, before the fit: its centroid back-projected onto the plane at height
    `z` [m, base frame] (NaN when its ray does not go down), and the angle of the longer side of the
    smallest rectangle around its outline back-projected there [rad], folded by `yaw_period`."""
    xy = back_project(K, T_base_cam, blob.centroid_px, z)[0, :2]
    own = (labels == blob.label).astype(np.uint8)
    contours, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=len).reshape(-1, 2).astype(float)
    ground = back_project(K, T_base_cam, outline, z)[:, :2]
    ground = ground[np.all(np.isfinite(ground), axis=1)]
    if len(ground) < 3:
        return xy, 0.0
    (_, _), (w, h), angle = cv2.minAreaRect((ground * 1000.0).astype(np.float32))
    longer = angle if w >= h else angle + 90.0
    return xy, fold(math.radians(longer), yaw_period)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """The intersection over union of two bool masks; 0 when both are empty."""
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def fit_block(blob_mask: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, spec: ObjectSpec,
              xy0: np.ndarray, yaw0: float, cv2: Any, max_evals: int = FIT_MAX_EVALS,
              visible: np.ndarray | None = None) -> Fit:
    """The known block `spec` lying on the table whose silhouette (`block_points`) best covers `blob_mask`
    (IoU), by a pattern search from (`xy0`, `yaw0`) over the levels of `FIT_STEPS`: per level, while a move
    improves and draws remain, the best of the six moves +- the step along the block's two sides and in yaw
    is taken. With `visible` only its pixels count (the jaws hide the rest). Four more draws then test which
    sides the image places (`Fit.free`). The yaw comes out folded by `spec.yaw_period`."""
    shape = blob_mask.shape
    blob = blob_mask if visible is None else blob_mask & visible
    rows, cols = np.nonzero(blob)
    blob_area = len(rows)

    def score(x: float, y: float, yaw: float) -> float:
        model = silhouette(K, T_base_cam, block_points(spec, (x, y), yaw), shape, cv2)
        if visible is not None:
            model &= visible
        inter = int(np.count_nonzero(model[rows, cols]))
        union = blob_area + int(np.count_nonzero(model)) - inter
        return inter / union if union else 0.0

    def search(start: tuple[float, float, float], value: float, steps: tuple[tuple[float, float], ...],
               evals: int) -> tuple[tuple[float, float, float], float, int]:
        best, best_score = start, value
        for step_m, step_deg in steps:
            step_yaw = math.radians(step_deg)
            improved = True
            while improved and evals < max_evals:
                improved = False
                x, y, yaw = best
                # Along the block's own sides: a block cut by the frame's edge scores on a narrow ridge along
                # its long side, which steps in the base's x and y cannot follow.
                ux, uy = step_m * math.cos(yaw), step_m * math.sin(yaw)
                moves = ((x + ux, y + uy, yaw), (x - ux, y - uy, yaw), (x - uy, y + ux, yaw),
                         (x + uy, y - ux, yaw), (x, y, yaw + step_yaw), (x, y, yaw - step_yaw))
                for move in moves:
                    if evals >= max_evals:
                        break
                    value = score(*move)
                    evals += 1
                    if value > best_score + 1e-12:
                        best, best_score, improved = move, value, True
        return best, best_score, evals

    best, best_score, evals = search((float(xy0[0]), float(xy0[1]), float(yaw0)),
                                     score(float(xy0[0]), float(xy0[1]), float(yaw0)), FIT_STEPS, 1)
    x, y, yaw = best
    R = rotation_z(yaw)
    free = [max(best_score - score(x + d * R[0, k], y + d * R[1, k], yaw)
                for d in (FREE_SHIFT_M, -FREE_SHIFT_M)) < FREE_IOU_DROP for k in (0, 1)]
    return Fit(np.array([x, y]), fold(yaw, spec.yaw_period), float(best_score), evals + 4, (free[0], free[1]))


def working(image: np.ndarray, K: np.ndarray, cv2: Any, width: int = WORK_WIDTH_PX
            ) -> tuple[np.ndarray, np.ndarray]:
    """`image` scaled down to `width` columns when wider (INTER_AREA), and `K` for the scaled image."""
    height_in, width_in = image.shape[:2]
    if width_in <= width:
        return image, K
    height = max(1, round(height_in * width / width_in))
    sx, sy = width / width_in, height / height_in
    small = cv2.resize(np.ascontiguousarray(image), (width, height), interpolation=cv2.INTER_AREA)
    out = K.copy()
    out[0, 0], out[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    # INTER_AREA maps pixel centres: (u + 0.5) s - 0.5.
    out[0, 2], out[1, 2] = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    return small, out


def _optional(name: str) -> Any:
    """The module `name`, or None when it cannot be imported (as `dream._optional`)."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ------------------------------------------------------------------ the strategy
@dataclass
class WristGraspSettings(GraspSettings):
    #: The camera in `obs.frames` to look through: the robot.yaml `cameras:` key, and the name the twin
    #: gives the physical one; "" = no looks, the strategy is plain `grasp`.
    wrist_camera: str = "wrist"
    #: A frame older than this (`Frame.age_s`) is not measured [s]: under touch's 0.25 s settle window, so a
    #: frame read once the arm has stood still for it was captured with the arm still.
    max_frame_age_s: float = 0.2
    #: How long a look waits at the hover for a usable frame before it is skipped [s].
    frame_wait_s: float = 1.0
    #: The block's hue [deg, 0..360); -1 = the colour of `object_kind` (cube 4.6, cuboid5 45, cuboid85 140,
    #: cylinder 285).
    hue_deg: float = -1.0
    #: Half-width of the hue window around `hue_deg` [deg]; it wraps at 360.
    hue_window_deg: float = 20.0
    #: Pixels less saturated than this are not the block [0..1]; the twin's table is 0.2.
    min_saturation: float = 0.35
    #: Pixels darker than this are not the block [0..1].
    min_value: float = 0.25
    #: The camera's focal length in the frame as received [px]; 0 = from robot.yaml's `fovy_deg` and the
    #: frame's height (1200 at 1080 rows, 400 at 360).
    focal_px: float = 0.0
    #: The column where the closed jaw tips' midpoint appears in the frame on the stand [px]; -1 = the
    #: model's (902.6 of 1920).
    tip_pixel_x: float = -1.0
    #: Its row [px]; -1 = the model's (799.2 of 1080). Set both or neither.
    tip_pixel_y: float = -1.0
    #: Turn the frame by 180 degrees first: a camera mounted upside down.
    rotate_180: bool = False
    #: A look ends when the block lies this close to the aim and its yaw within `YAW_TOL_DEG` [m].
    centre_tol_m: float = 0.003
    #: A block measured farther than this from the aim is not trusted [m]; the LLM's pose is 1-3 cm off.
    #: Plus `SEARCH_SLACK_M` it is the radius blobs are looked for in.
    max_shift_m: float = 0.04
    #: The most moves of the aim per look [count]; one more measurement verifies the last move, so a look
    #: measures at most rounds + 1 times.
    rounds: int = 2
    #: A fit poorer than this is not the block [0..1]: every block of the set fits at 0.96-0.99 on the
    #: simulation's render.
    min_fit_iou: float = 0.5
    #: Look again at the straddle hover with the jaw open: the probe pushes a light block a few millimetres.
    look_before_straddle: bool = True


@dataclass(frozen=True)
class Measurement:
    """What one wrist frame says about the block; unusable when `problem` is set (the panel's text)."""

    #: The block's centre [m, base frame] and yaw [rad], folded by its symmetry.
    xy: np.ndarray
    yaw: float
    #: The fit's IoU, the blob's area [px, at the working size] and its area over the fitted block's.
    iou: float
    area_px: int
    ratio: float
    #: The block top's centre in the frame as received [px].
    pixel: np.ndarray
    problem: str = ""


@dataclass
class Look:
    """One look at one hover: its state from the first call of the seam until it returns None."""

    #: 'probe' or 'straddle'.
    where: str
    t0: float
    #: The aim, the block's rotation, the candidate headings and the shifts' origin when the look began.
    xy0: np.ndarray
    rot0: np.ndarray
    headings0: list[float]
    block_xy0: np.ndarray | None
    #: MEASURE, RETURN (after a revert) or DONE.
    phase: str = MEASURE
    #: Moves of the aim so far, and the last one's length [m].
    round: int = 0
    last_shift_m: float = float("nan")
    #: When the wait for a usable frame began [s]; NaN when not waiting.
    t_wait0: float = float("nan")
    #: When the arm's target last moved [s].
    t_target: float = float("nan")
    #: Ticks in a row every arm joint stood still, and the last tick's measured joints [rad].
    still_ticks: int = 0
    q_prev: np.ndarray | None = None
    #: Why the aim was reverted; the panel's note for this tick, '' for the default.
    reverted: str = ""
    text: str = ""


@dataclass
class WristStatus:
    """What the wrist camera saw, beside grasp's `GraspStatus`, for logs and tests."""

    #: Looks begun and frames measured in this run [count].
    looks: int = 0
    measurements: int = 0
    #: The hover of the last look: 'probe' or 'straddle'.
    where: str = ""
    #: The last measurement: the block's offset from the aim [m, base frame], its yaw [deg], the fit's IoU,
    #: the blob's area [px] and its area over the fitted block's.
    offset_m: tuple[float, float] = (float("nan"), float("nan"))
    yaw_deg: float = float("nan")
    fit_iou: float = float("nan")
    area_px: int = 0
    ratio: float = float("nan")
    #: The last move of the aim [m].
    shift_m: float = float("nan")
    #: The last look's verdict: '', 'centred', 'skipped' or 'reverted', and why it was not centred.
    verdict: str = ""
    skipped: str = ""
    #: The verdict of every finished look, in order.
    verdicts: list[str] = field(default_factory=list)


class WristGraspStrategy(GraspStrategy):
    """Grasps a block as `grasp` does, after the wrist camera centres the jaws on it and sets the yaw."""

    settings = WristGraspSettings

    def __init__(self, settings: WristGraspSettings | None = None):
        config = settings if settings is not None else WristGraspSettings()
        if not config.max_frame_age_s > 0:
            raise ValueError(f"grasp_wrist: max_frame_age_s must be positive, is {config.max_frame_age_s}")
        if not config.frame_wait_s >= 0:
            raise ValueError(f"grasp_wrist: frame_wait_s must be zero or more, is {config.frame_wait_s}")
        if not (config.hue_deg == -1 or 0 <= config.hue_deg < 360):
            raise ValueError(f"grasp_wrist: hue_deg must lie within [0, 360) or be -1 (the kind's), "
                             f"is {config.hue_deg}")
        if not 0 < config.hue_window_deg < 180:
            raise ValueError(f"grasp_wrist: hue_window_deg must lie within (0, 180), "
                             f"is {config.hue_window_deg}")
        for name in ("min_saturation", "min_value"):
            value = getattr(config, name)
            if not 0 <= value <= 1:
                raise ValueError(f"grasp_wrist: {name} must lie within [0, 1], is {value}")
        if not config.focal_px >= 0:
            raise ValueError(f"grasp_wrist: focal_px must be zero (from the field of view) or more, "
                             f"is {config.focal_px}")
        tips = (config.tip_pixel_x, config.tip_pixel_y)
        if not (tips == (-1, -1) or (tips[0] >= 0 and tips[1] >= 0)):
            raise ValueError(f"grasp_wrist: tip_pixel_x and tip_pixel_y come together, both -1 or both a "
                             f"pixel, are {tips}")
        if not config.centre_tol_m > 0:
            raise ValueError(f"grasp_wrist: centre_tol_m must be positive, is {config.centre_tol_m}")
        if not config.max_shift_m > config.centre_tol_m:
            raise ValueError(f"grasp_wrist: max_shift_m must exceed centre_tol_m, is {config.max_shift_m}")
        if not config.rounds >= 1:
            raise ValueError(f"grasp_wrist: rounds must be at least 1, is {config.rounds}")
        if not 0 < config.min_fit_iou < 1:
            raise ValueError(f"grasp_wrist: min_fit_iou must lie within (0, 1), is {config.min_fit_iou}")
        super().__init__(config)
        self.wrist_settings = config
        self.wrist = WristStatus()
        self._look: Look | None = None
        self._cv2: Any = None
        self._window: HsvWindow | None = None
        self._camera: RobotCamera | None = None
        #: The closed tips' midpoint in the TCP frame [m], from the model once `start` has it.
        self._tip_tcp = np.zeros(3)

    def start(self, run: RunInfo, obs: Observation) -> None:
        """grasp's start, then the camera: refused before the arm moves without OpenCV or with a camera that
        robot.yaml does not declare. A missing frame is no refusal: every look is then skipped."""
        super().start(run, obs)
        cfg = self.wrist_settings
        self.wrist = WristStatus()
        self._look = None
        self._cv2, self._window, self._camera = None, None, None
        if not cfg.wrist_camera:
            return
        cv2 = _optional("cv2")
        if cv2 is None:
            raise RuntimeError("grasp_wrist: centring needs OpenCV: pip install opencv-python-headless (the "
                               "twin brings it), or set wrist_camera to \"\" to run without the camera")
        cameras = self.kin.spec.cameras
        camera = cameras.get(cfg.wrist_camera)
        if camera is None:
            raise RuntimeError(f"grasp_wrist: camera {cfg.wrist_camera!r} is not in robot.yaml `cameras` "
                               f"({', '.join(cameras) or 'none'})")
        self._cv2, self._camera = cv2, camera
        hue = cfg.hue_deg if cfg.hue_deg >= 0 else hsv_window(COLOURS[self.block.kind], 0.0, 0.0, 0.0).hue_deg
        self._window = HsvWindow(hue, cfg.hue_window_deg, cfg.min_saturation, cfg.min_value)
        self._tip_tcp = self.kin.jaw_centre(0.0)

    # ------------------------------------------------------------------ measuring
    def measure(self, rgb: np.ndarray, joints: Mapping[str, float]) -> Measurement:
        """What the frame `rgb` (RGB, as received) says about the block, with the arm at `joints` (measured,
        the frame's capture pose): its centre and yaw from the known block fitted to its blob, or a
        problem."""
        cv2, camera, window = self._cv2, self._camera, self._window
        if cv2 is None or camera is None or window is None:
            raise RuntimeError("grasp_wrist: measure() without a camera: start the run with wrist_camera set")
        cfg, kin = self.wrist_settings, self.kin
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or min(image.shape[:2]) < 16:
            return _unusable(f"klatka z kamery {cfg.wrist_camera} w zlym ksztalcie {tuple(image.shape)}")
        if cfg.rotate_180:
            image = image[::-1, ::-1]
        height, width = image.shape[:2]
        K = pinhole(camera.fovy_deg, image.shape, cfg.focal_px)
        T = kin.camera(cfg.wrist_camera, joints)
        if cfg.tip_pixel_x >= 0:
            tip = kin.tcp(joints) @ np.array([*self._tip_tcp, 1.0])
            predicted = project(K, T, tip[:3])[0][0]
            # The operator reads the pixel off the frame as received; the model draws the turned one.
            seen = np.array([cfg.tip_pixel_x, cfg.tip_pixel_y])
            if cfg.rotate_180:
                seen = np.array([width - 1, height - 1]) - seen
            shift = seen - predicted
            off = float(np.linalg.norm(shift))
            if not off <= MAX_TIP_SHIFT * width:
                return _unusable(f"tip_pixel daleko od modelu ({off:.0f} px)")
            # The model draws everything `shift` short of where the camera sees it.
            K[0, 2] += shift[0]
            K[1, 2] += shift[1]
        small, Kw = working(image, K, cv2)
        shape = small.shape[:2]
        dilate = round(JAW_MASK_DILATE_PX * shape[1] / WORK_WIDTH_PX)
        visible = ~jaw_mask(kin, joints, Kw, T, shape, cv2, dilate)
        labels, found = blobs(segment(small, window, cv2) & visible, cv2)

        assert self.grasp.block_xy is not None
        aim = np.asarray(self.grasp.block_xy, float)
        period, top = self.block.yaw_period, self.height_m
        radius = cfg.max_shift_m + SEARCH_SLACK_M
        near: list[tuple[Blob, Fit, float]] = []
        for blob in found:
            xy_c = back_project(Kw, T, blob.centroid_px, top)[0, :2]
            if not (np.all(np.isfinite(xy_c)) and float(np.linalg.norm(xy_c - aim)) <= radius):
                continue
            xy_b, yaw_b = blob_pose(labels, blob, Kw, T, top, period, cv2)
            fit = fit_block(labels == blob.label, Kw, T, self.block, xy_b, yaw_b, cv2, visible=visible)
            # The known block where the fit puts it, as much of it as the frame and the jaws leave visible.
            points = block_points(self.block, fit.xy, fit.yaw)
            model = int(np.count_nonzero(silhouette(Kw, T, points, shape, cv2) & visible))
            if not model >= MIN_VISIBLE_SHARE * hull_area(Kw, T, points, cv2):
                # A speck at the jaws' edge: the only block it fits lies behind the jaws or out of the frame.
                continue
            near.append((blob, fit, blob.area / model))
        if not near:
            return _unusable("brak bloku w kamerze")
        candidates = [c for c in near if AREA_WINDOW[0] <= c[2] <= AREA_WINDOW[1]]
        if not candidates:
            largest = max(near, key=lambda c: c[0].area)
            return _unusable(f"plama w kamerze nie pasuje rozmiarem (x{largest[2]:.2f})", largest[0].area,
                             largest[2])
        if len(candidates) > 1:
            return _unusable("dwa bloki w kamerze")
        blob, fit, ratio = candidates[0]
        w = self.wrist
        w.fit_iou, w.area_px, w.ratio, w.yaw_deg = fit.iou, blob.area, ratio, math.degrees(fit.yaw)
        if fit.iou < cfg.min_fit_iou:
            return _unusable(f"ksztalt nie pasuje (IoU {fit.iou:.2f})", blob.area, ratio)
        xy = fit.xy.copy()
        R = rotation_z(fit.yaw)
        for k in (0, 1):
            if fit.free[k]:
                # The image cannot place the block along this side: the aim's position along it stands.
                u = R[:2, k]
                xy -= float((xy - aim) @ u) * u
        pixel = project(K, T, np.array([*xy, top]))[0][0]
        if cfg.rotate_180:
            pixel = np.array([width - 1 - pixel[0], height - 1 - pixel[1]])
        return Measurement(xy, fit.yaw, fit.iou, blob.area, ratio, pixel)

    # ------------------------------------------------------------------ the seams
    def before_descent(self, obs: Observation) -> Action | None:
        """The look before the probe leaves the hover; None at once without a camera."""
        if not self.wrist_settings.wrist_camera:
            return None
        if self._look is None:
            self._look = self._new_look("probe")
        # The look owns the settle counter in DESCEND: grasp's descent runs only after this returns None.
        settled = self._settled(self.hover_point)
        return self._look_step(obs, lambda: self.hover_point, settled)

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        """The look at the straddle hover with the jaw open (`look_before_straddle`); `settled` is OPEN's own
        readiness, which counts the settle window itself."""
        cfg = self.wrist_settings
        if not cfg.wrist_camera or not cfg.look_before_straddle:
            return None
        if self._look is None:
            self._look = self._new_look("straddle")
        return self._look_step(obs, lambda: self.straddle_hover, settled)

    @property
    def _looking(self) -> bool:
        return self._look is not None

    def _new_look(self, where: str) -> Look:
        assert self.grasp.block_xy is not None
        self.wrist.looks += 1
        self.wrist.where = where
        xy0 = None if self._block_xy0 is None else self._block_xy0.copy()
        return Look(where, self._t, np.asarray(self.grasp.block_xy, float), self.block_rotation.copy(),
                    list(self._headings), xy0, t_target=self._t)

    def _look_step(self, obs: Observation, hover: Callable[[], np.ndarray], settled: bool) -> Action | None:
        """One tick of the look: wait for the arm to stand still at the hover, read a frame, measure, then end
        the look or move the aim; None once the look is over (the tick after its verdict)."""
        look, cfg, kin = self._look, self.wrist_settings, self.kin
        assert look is not None
        if look.phase == DONE:
            self._look = None
            return None
        look.text = ""
        q = kin.to_q(self._measured)
        moved = math.inf if look.q_prev is None else float(np.degrees(np.abs(q - look.q_prev)[kin.arm]).max())
        look.still_ticks = look.still_ticks + 1 if moved <= JOINT_STILL_DEG else 0
        look.q_prev = q
        if not (settled and look.still_ticks >= self.settle_ticks):
            return self._wait_for_arm(look, hover)
        if look.phase == RETURN:
            return self._finish_look("reverted", look.reverted)

        frame = obs.frames.get(cfg.wrist_camera)
        age = float(frame.age_s) if frame is not None else math.nan
        if frame is None or not age <= cfg.max_frame_age_s:
            if math.isnan(look.t_wait0):
                look.t_wait0 = self._t
            if self._t - look.t_wait0 >= cfg.frame_wait_s - 1e-9:
                why = (f"brak klatki z kamery {cfg.wrist_camera}" if frame is None
                       else f"klatka z kamery {cfg.wrist_camera} za stara ({age:.1f} s)")
                return self._finish_look("skipped", why)
            return self.hold("centruje po kamerze: czekam na klatke")
        look.t_wait0 = math.nan

        m = self.measure(frame.image, self._measured)
        self.wrist.measurements += 1
        if m.problem:
            return self._finish_look("skipped", m.problem)
        assert self.grasp.block_xy is not None
        offset = m.xy - np.asarray(self.grasp.block_xy, float)
        dist = float(np.linalg.norm(offset))
        self.wrist.offset_m = (float(offset[0]), float(offset[1]))
        # The block's x side lies a quarter turn from the closing heading that grips it across.
        yaw_change = abs(fold(m.yaw - (self._psi - math.pi / 2), self.block.yaw_period))
        if dist > cfg.max_shift_m:
            return self._finish_look("skipped", f"blok w kamerze {dist * 1000:.0f} mm od celu - nie ufam")
        if look.round >= 1 and dist > MOVE_GUARD * look.last_shift_m + cfg.centre_tol_m:
            return self._revert(look, hover, dist)
        if dist <= cfg.centre_tol_m and yaw_change <= math.radians(YAW_TOL_DEG):
            return self._finish_look("centred", f"wysrodkowano po kamerze: {dist * 1000:.0f} mm")
        if look.round >= cfg.rounds:
            return self._finish_look("centred", f"wysrodkowano po kamerze: {dist * 1000:.0f} mm po "
                                                f"{look.round} ruchach")
        return self._reaim(look, hover, m, dist)

    def _wait_for_arm(self, look: Look, hover: Callable[[], np.ndarray]) -> Action:
        """To the hover with the top-down hand, until the arm stands still there; skipped after
        `LOOK_STALL_S` with the target standing."""
        before = self.q_cmd.copy()
        action = self._drive(hover(), self.approach_step_m)
        if action.done:
            return action
        arm = self.kin.arm
        if float(np.degrees(np.abs(self.q_cmd - before)[arm]).max()) > TARGET_MOVED_DEG:
            look.t_target = self._t
        elif self._t - look.t_target >= LOOK_STALL_S:
            mm = self.status.distance * 1000
            return self._finish_look("skipped", f"ramie nie stoi nad blokiem ({mm:.0f} mm od zawisu)")
        return action

    def _block_rotation(self, yaw: float) -> np.ndarray:
        """The block's rotation for a measured `yaw`: about the vertical, a symmetry turn added when it would
        be the identity, which `grasp_headings` reads as no orientation."""
        R = rotation_z(yaw)
        return rotation_z(yaw + self.block.yaw_period) if np.allclose(R, np.eye(3), atol=1e-6) else R

    def _reaim(self, look: Look, hover: Callable[[], np.ndarray], m: Measurement, dist: float) -> Action:
        """Aims at the measured block through grasp's `_aim`, the jaws across its short side, and drives to
        the new hover; the old aim stays when the arm cannot take the new one."""
        saved = self.block_rotation.copy(), list(self._headings)
        psi = self._psi
        self._heading_from(self._block_rotation(m.yaw))
        try:
            self._aim(m.xy)
        except RuntimeError as exc:
            self.block_rotation, self._headings = saved
            return self._finish_look("skipped", str(exc))
        # A tip stall's sideways retries start from the block the camera saw.
        self._block_xy0 = np.asarray(m.xy, float).copy()
        look.round += 1
        look.last_shift_m = self.wrist.shift_m = dist
        look.still_ticks = 0
        look.t_target = self._phase_t0 = self._t
        turn = math.degrees(abs(wrap(self._psi - psi)))
        look.text = f"centruje po kamerze: przesuwam o {dist * 1000:.0f} mm, obrot {turn:.0f} st."
        logger.info("Kamera na nadgarstku: blok %.1f mm od celu, obrot %.0f st.", dist * 1000, turn)
        return self._drive(hover(), self.approach_step_m)

    def _revert(self, look: Look, hover: Callable[[], np.ndarray], dist: float) -> Action:
        """The move guard fired: back to the aim, rotation and heading the look began with; when the arm
        cannot take them again, the moved aim stands and the look is skipped."""
        moved = self.block_rotation.copy(), list(self._headings)
        self.block_rotation, self._headings = look.rot0.copy(), list(look.headings0)
        why = (f"przesuniecie nie zgadza sie z obrazem ({dist * 1000:.0f} mm po korekcie "
               f"{look.last_shift_m * 1000:.0f} mm)")
        try:
            self._aim(look.xy0)
        except RuntimeError as exc:
            self.block_rotation, self._headings = moved
            why = f"przesuniecie nie zgadza sie z obrazem, cofniecie niemozliwe: {exc}"
            return self._finish_look("skipped", why)
        self._block_xy0 = look.block_xy0
        look.phase, look.reverted = RETURN, why
        self.wrist.verdict, self.wrist.skipped = "reverted", why
        look.still_ticks = 0
        look.t_target = self._phase_t0 = self._t
        logger.warning("Kamera na nadgarstku: %s - cofam cel", why)
        return self._drive(hover(), self.approach_step_m)

    def _finish_look(self, verdict: str, why: str) -> Action:
        """The look's verdict: one tick that holds the arm and shows it; the seam returns None on the next."""
        look, w = self._look, self.wrist
        assert look is not None
        w.verdict, w.skipped = verdict, "" if verdict == "centred" else why
        w.verdicts.append(verdict)
        look.phase = DONE
        if verdict == "centred":
            look.text = why
            logger.info("Kamera na nadgarstku (%s): %s", look.where, why)
        else:
            look.text = f"centrowanie {'cofniete' if verdict == 'reverted' else 'pominiete'}: {why}"
            logger.warning("Kamera na nadgarstku (%s): %s", look.where, look.text)
        return self.hold(look.text)

    # ------------------------------------------------------------------ the panel
    def _note(self) -> str:
        look = self._look
        if look is None:
            return super()._note()
        if look.text:
            return look.text
        what = "cofam centrowanie" if look.phase == RETURN else "centruje po kamerze"
        return f"{what}: {self.status.distance * 1000:.0f} mm do zawisu"


def _unusable(problem: str, area_px: int = 0, ratio: float = math.nan) -> Measurement:
    """A measurement that says only why the frame is no use."""
    nan = np.full(2, np.nan)
    return Measurement(nan, math.nan, math.nan, area_px, ratio, nan.copy(), problem)
