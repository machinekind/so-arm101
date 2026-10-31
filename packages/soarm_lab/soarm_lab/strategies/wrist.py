"""The wrist camera's geometry for `grasp_wrist`: where a block lies, from a frame of the camera on the arm.

The camera's pose comes from the robot description (`robot.yaml` `cameras:`) through `RobotKinematics.camera`,
in OpenCV axes (z forward, y down, x right); its intrinsics are an ideal pinhole from the field of view and
the frame's height (`pinhole`, the twin's convention). A pixel goes to the base frame where its ray meets a
horizontal plane (`back_project`), the block's top. The block is found by its colour (`segment`, an HSV
window), with the arm's own jaws masked out from the model (`jaw_mask`: one convex hull per jaw body of its
collision primitives), and placed by fitting its known shape to its blob (`fit_block`: the IoU of the
silhouette of `block_points`, a box's corners or a cylinder's end circles). At the probe hover the visible
side face pulls a plain centroid 6 mm toward the camera; the fit lands within 0.5 mm.

OpenCV comes in as the argument `cv2` and is never imported here: the strategy resolves it when it starts.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kinematics import RobotKinematics
from ..rl.objects import ObjectSpec

#: The frame is measured at most this wide [px]: a wider one is scaled down first (INTER_AREA), so the
#: MJCF camera's 1920x1080 is measured at 640x360.
WORK_WIDTH_PX = 640
#: Smaller connected components are noise, not looked at [px].
MIN_BLOB_PX = 64
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
#: The square kernel of the colour mask's opening [px]: a patch smaller than it does not survive.
OPEN_KERNEL_PX = 3
#: Sub-pixel precision of the drawn silhouettes: fixed-point bits of `cv2.fillConvexPoly`.
_SHIFT = 4


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
