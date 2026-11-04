"""Centre the jaws and set grasp yaw from a wrist camera at the hover.

`grasp_wrist` extends `grasp` through `before_straddle`.
With `probe: true`, it also uses `before_descent`.
Without the probe, the look runs at the open-jaw hover before each straddle. With it, a look
runs before the probe and, with `look_before_straddle`, again before the straddle. A look reads
`obs.frames`, fits the block's known silhouette to a colour blob and updates the aim.
With the default `hue_deg: -1`, the first usable frame supplies the hue.
Sampling finds the hue window with the most pixels above the saturation and value thresholds
inside the expected visible silhouette at the aim. That window must cover at least `MIN_HUE_SHARE`
of the region. If no window meets this share, the look is skipped and the grasp keeps its aim.
See the package README for settings and camera calibration.

Camera pose comes from measured joints and `RobotSpec.cameras`. Intrinsics come from
field of view or `focal_px`, adjusted by the calibrated tip pixel when set. Frames are
scaled to `WORK_WIDTH_PX` before measuring. A model-based mask removes the jaws.

Candidates must match the fitted silhouette's area and visible share. The fit keeps the
current aim along any block side the image cannot locate. Each move is measured again
after all arm joints settle. A residual inconsistent with that move restores the original
aim if it remains reachable.

A block on another block is measured at its own height, from the bottom the grasp took from its pose.

Looks finish after convergence or `rounds` moves. Missing or stale frames, ambiguous blobs,
poor fits, excessive shifts and an unsettled arm skip the look. The straddle (or probe) then
continues from the settled aim. Looks consume no grasp attempts. An empty `wrist_camera`
runs plain `grasp`; a named camera requires OpenCV and a matching robot camera entry.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo

from ..blocks import BlockSpec
from ..kinematics import RobotKinematics
from ..robot import RobotCamera
from .grasp import GraspSettings, GraspStrategy, wrap

logger = logging.getLogger(__name__)

#: Maximum working frame width [px]. Wider frames are scaled before measuring.
WORK_WIDTH_PX = 640
#: Extra search radius around the aim [m] to allow for centroid bias from visible side faces.
SEARCH_SLACK_M = 0.015
#: Allowed ratio of blob area to fitted silhouette area outside the jaw mask.
AREA_WINDOW = (0.55, 1.25)
#: Minimum connected-component area [px].
MIN_BLOB_PX = 64
#: Minimum share of the expected visible silhouette covered by the winning hue window.
MIN_HUE_SHARE = 0.3
#: Minimum fitted silhouette share inside the frame and outside the jaw mask.
MIN_VISIBLE_SHARE = 0.2
#: Maximum joint movement per tick for stillness [deg], including roll that leaves the TCP in place.
JOINT_STILL_DEG = 0.5
#: Jaw-mask dilation [px] at `WORK_WIDTH_PX`, scaled with frame width.
JAW_MASK_DILATE_PX = 12
#: Jaw points nearer the camera than this are not projected [m].
MIN_JAW_DEPTH_M = 0.02
#: Search step levels in position [m] and yaw [deg].
FIT_STEPS = ((0.008, 8.0), (0.002, 2.0), (0.0005, 0.5))
#: Maximum silhouette evaluations in the fit search.
FIT_MAX_EVALS = 300
#: Samples per cylinder end circle for silhouette projection.
RIM_POINTS = 24
#: Test displacement along each block axis [m]. An IoU loss below `FREE_IOU_DROP`
#: marks that position as unresolved, so it keeps the current aim.
FREE_SHIFT_M = 0.003
FREE_IOU_DROP = 0.01
#: Yaw tolerance relative to the closing heading across the short side [deg].
YAW_TOL_DEG = 3.0
#: Maximum residual as a fraction of the last move, plus `centre_tol_m`.
MOVE_GUARD = 0.5
#: Square opening-kernel size for removing small colour patches [px].
OPEN_KERNEL_PX = 3
#: Maximum tip calibration offset as a fraction of frame width.
MAX_TIP_SHIFT = 0.2
#: Settle timeout after the target stops moving [s].
LOOK_STALL_S = 3.0
#: Minimum target change that restarts the settle timeout [deg].
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
    """Hue window [deg] and minimum saturation and value [0..1] for block segmentation."""

    hue_deg: float
    window_deg: float
    min_s: float
    min_v: float


@dataclass(frozen=True)
class Blob:
    """A mask component with area [px], centroid (column, row) [px] and label."""

    area: int
    centroid_px: np.ndarray
    label: int


@dataclass(frozen=True)
class Fit:
    """A silhouette fit with centre [m, base frame], yaw [rad], IoU and evaluation count.

    `free` marks block axes (x, y) whose position the image cannot determine.
    """

    xy: np.ndarray
    yaw: float
    iou: float
    evals: int
    free: tuple[bool, bool] = (False, False)


def pinhole(fovy_deg: float, shape: tuple[int, ...], focal_px: float = 0.0) -> np.ndarray:
    """Return 3x3 pinhole intrinsics for `shape` (height, width, ...).

    Use positive `focal_px`, otherwise derive focal length from vertical `fovy_deg` [deg].
    Square pixels share a focal length. The principal point is ((W - 1) / 2, (H - 1) / 2).
    """
    height, width = int(shape[0]), int(shape[1])
    f = focal_px if focal_px > 0 else (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def project(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points (N, 3) through camera pose `T_base_cam` and intrinsics `K`.

    Return pixels (N, 2), ordered as column and row, and optical depths (N,) [m].
    The camera pose uses OpenCV axes.
    """
    points = np.atleast_2d(np.asarray(points, float))
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p = (points - t) @ R
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def back_project(K: np.ndarray, T_base_cam: np.ndarray, pixels: np.ndarray, z: float) -> np.ndarray:
    """Intersect rays from pixels (N, 2) with the plane at height `z` [m, base frame].

    Return points (N, 3) with NaN rows for rays that do not point down.
    """
    pixels = np.atleast_2d(np.asarray(pixels, float))
    rays = np.column_stack([(pixels[:, 0] - K[0, 2]) / K[0, 0], (pixels[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(pixels))]) @ T_base_cam[:3, :3].T
    origin = T_base_cam[:3, 3]
    down = rays[:, 2] < -1e-9
    out = np.full((len(pixels), 3), np.nan)
    out[down] = origin + rays[down] * ((z - origin[2]) / rays[down, 2])[:, None]
    return out


def rotation_z(yaw: float) -> np.ndarray:
    """Return the rotation about vertical by `yaw` [rad]."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_corners(half: tuple[float, float, float] | np.ndarray, xy: np.ndarray | tuple[float, float],
                yaw: float) -> np.ndarray:
    """Return box corners (8, 3) [m, base frame] with centre `xy` and bottom at z = 0.

    `half` gives half sizes. `yaw` rotates the body x axis about vertical [rad].
    """
    hx, hy, hz = (float(v) for v in half)
    local = np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 2 * hz)])
    out = local @ rotation_z(yaw).T
    out[:, 0] += float(xy[0])
    out[:, 1] += float(xy[1])
    return out


def _quat_mat(q: tuple[float, float, float, float]) -> np.ndarray:
    """Return the rotation matrix for quaternion (w, x, y, z), normalized before use."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def block_points(spec: BlockSpec, xy: np.ndarray | tuple[float, float], yaw: float) -> np.ndarray:
    """Return silhouette points (N, 3) [m, base frame] for `spec` at `xy` and `yaw` [rad].

    Use corners for box parts and sampled end circles for cylinders. The convex hull of
    projected cylinder rims gives its silhouette. The body origin lies `half_size[2]` above
    the table, with yaw about vertical.
    """
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
            raise ValueError(f"{spec.name}: a part of kind {part.kind!r} cannot be drawn")
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


def segment(rgb: np.ndarray, window: HsvWindow, cv2: Any) -> np.ndarray:
    """Return a bool colour mask for `rgb` (H, W, 3, uint8).

    Hue wraps at 360 degrees. Opening with `OPEN_KERNEL_PX` removes small patches.
    """
    image = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0                 # OpenCV's 8-bit hue runs 0..179
    dh = np.abs((hue - window.hue_deg + 180.0) % 360.0 - 180.0)
    mask = ((dh <= window.window_deg) & (hsv[..., 1] >= window.min_s * 255.0)
            & (hsv[..., 2] >= window.min_v * 255.0)).astype(np.uint8)
    kernel = np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)


def dominant_hue(rgb: np.ndarray, region: np.ndarray, window_deg: float, min_s: float, min_v: float,
                 cv2: Any) -> tuple[float, float]:
    """Return the mean hue [deg] and pixel share [0..1] of the most populated hue window.

    `rgb` is (H, W, 3, uint8) and `region` is an (H, W) bool mask.
    The window has half-width `window_deg` [deg] and wraps at 360 degrees.
    Only pixels with at least `min_s` saturation and `min_v` value vote.
    The share divides the winning window's count by all pixels in `region`.
    The hue is NaN and the share is 0 when no pixels vote.
    """
    total = int(np.count_nonzero(region))
    if total == 0:
        return math.nan, 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), cv2.COLOR_RGB2HSV)
    voting = region & (hsv[..., 1] >= min_s * 255.0) & (hsv[..., 2] >= min_v * 255.0)
    hist = np.bincount(hsv[..., 0][voting].ravel(), minlength=180).astype(float)  # 2-degree bins
    half = int(window_deg // 2)
    votes = np.zeros_like(hist)
    for k in range(-half, half + 1):
        votes += np.roll(hist, k)
    best = int(np.argmax(votes))
    if votes[best] == 0:
        return math.nan, 0.0
    bins = (best + np.arange(-half, half + 1)) % 180
    angle, weight = np.radians(bins * 2.0), hist[bins]
    hue = math.degrees(math.atan2(float(weight @ np.sin(angle)), float(weight @ np.cos(angle)))) % 360.0
    return hue, float(votes[best]) / total


def _fill(mask: np.ndarray, pixels: np.ndarray, cv2: Any) -> None:
    """Fill the convex hull of pixels (N, 2) into a uint8 mask, clipped to the frame."""
    if len(pixels) < 3:
        return
    fixed = np.round(np.clip(pixels, -1e5, 1e5) * (1 << _SHIFT)).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(fixed), 1, cv2.LINE_8, _SHIFT)


def silhouette(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray, shape: tuple[int, ...],
               cv2: Any) -> np.ndarray:
    """Return the projected convex hull of base-frame points as a bool mask of `shape`.

    Ignore points behind the camera.
    """
    pixels, depth = project(K, T_base_cam, points)
    mask = np.zeros(shape[:2], np.uint8)
    _fill(mask, pixels[depth > 1e-6], cv2)
    return mask.astype(bool)


def hull_area(K: np.ndarray, T_base_cam: np.ndarray, points: np.ndarray, cv2: Any) -> float:
    """Return projected hull area [px] before frame clipping, ignoring points behind the camera."""
    pixels, depth = project(K, T_base_cam, points)
    front = pixels[depth > 1e-6]
    if len(front) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(front.astype(np.float32))))


def jaw_mask(kin: RobotKinematics, joints: Mapping[str, float], K: np.ndarray, T_base_cam: np.ndarray,
             shape: tuple[int, ...], cv2: Any, dilate_px: int) -> np.ndarray:
    """Return a bool jaw mask from each jaw body's collision primitives.

    Separate hulls preserve the gap between open jaws. Drop points within `MIN_JAW_DEPTH_M`
    of the camera and dilate with a square of side 2 * `dilate_px` + 1 [px].
    """
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
    """Return the label image and 8-connected components of at least `MIN_BLOB_PX`, largest first."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    found = [Blob(int(stats[k, cv2.CC_STAT_AREA]), np.asarray(centroids[k], float), k) for k in range(1, n)
             if stats[k, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    return labels, sorted(found, key=lambda blob: -blob.area)


def blob_pose(labels: np.ndarray, blob: Blob, K: np.ndarray, T_base_cam: np.ndarray, z: float,
              yaw_period: float, cv2: Any) -> tuple[np.ndarray, float]:
    """Estimate block centre [m, base frame] and yaw [rad] from the blob on plane `z`.

    Back-project the centroid and outline. Use the outline rectangle's long side for yaw,
    folded by `yaw_period`. The centre is NaN if its ray does not point down.
    """
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
    """Return intersection over union of bool masks, or zero when both are empty."""
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def fit_block(blob_mask: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, spec: BlockSpec,
              xy0: np.ndarray, yaw0: float, cv2: Any, max_evals: int = FIT_MAX_EVALS,
              visible: np.ndarray | None = None) -> Fit:
    """Fit the block silhouette to `blob_mask` from initial centre `xy0` and yaw `yaw0`.

    Search along the block axes and in yaw with `FIT_STEPS`. `visible` limits the scored
    pixels. Return centre, symmetry-folded yaw, IoU, evaluation count and unresolved axes.
    """
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
                # Search in block axes to follow the score ridge of a partly visible long side.
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
    """Scale frames wider than `width` and return the image with matching intrinsics."""
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
    """Import `name`, returning None if import fails."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ------------------------------------------------------------------ the strategy
@dataclass
class WristGraspSettings(GraspSettings):
    #: Camera key in `obs.frames` and robot.yaml `cameras`. Empty runs plain grasp.
    wrist_camera: str = "wrist"
    #: Maximum frame age [s]. Keep below the settle window to use frames captured while still.
    max_frame_age_s: float = 0.2
    #: How long a look waits at the hover for a usable frame before it is skipped [s].
    frame_wait_s: float = 1.0
    #: Block hue [deg, 0..360). -1 samples the dominant hue at the aim in the first usable frame.
    hue_deg: float = -1.0
    #: Half-width of the hue window around `hue_deg` [deg]; it wraps at 360.
    hue_window_deg: float = 20.0
    #: Minimum pixel saturation [0..1].
    min_saturation: float = 0.35
    #: Minimum pixel value [0..1].
    min_value: float = 0.25
    #: Focal length in the received frame [px]. Zero derives it from field of view and frame height.
    focal_px: float = 0.0
    #: Closed-tip midpoint column in the received frame [px]. -1 uses the model. Set both coordinates.
    tip_pixel_x: float = -1.0
    #: Closed-tip midpoint row in the received frame [px]. -1 uses the model.
    tip_pixel_y: float = -1.0
    #: Turn an upside-down camera frame by 180 degrees before measuring.
    rotate_180: bool = False
    #: A look ends when the block lies this close to the aim and its yaw within `YAW_TOL_DEG` [m].
    centre_tol_m: float = 0.003
    #: Maximum accepted offset from the aim [m]. Search adds `SEARCH_SLACK_M` for centroid bias.
    max_shift_m: float = 0.04
    #: Maximum aim moves per look [count]. One further measurement checks the final move.
    rounds: int = 2
    #: Minimum silhouette IoU [0..1].
    min_fit_iou: float = 0.5
    #: With `probe`, look again before the straddle to correct movement caused by the probe.
    #: Without the probe, the open-jaw hover look always runs when a camera is configured.
    look_before_straddle: bool = True


@dataclass(frozen=True)
class Measurement:
    """A frame measurement with centre and yaw, or an unusable result with `problem` set."""

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
    """State for one hover look until its hook returns None."""

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
    #: Revert reason and current panel note. Empty text uses the default note.
    reverted: str = ""
    text: str = ""


@dataclass
class WristStatus:
    """Wrist measurements and look verdicts alongside `GraspStatus` for logs and tests."""

    #: Looks begun and frames measured in this run [count].
    looks: int = 0
    measurements: int = 0
    #: The hover of the last look: 'probe' or 'straddle'.
    where: str = ""
    #: Last offset [m, base frame], yaw [deg], IoU, area [px] and blob-to-silhouette area ratio.
    offset_m: tuple[float, float] = (float("nan"), float("nan"))
    yaw_deg: float = float("nan")
    fit_iou: float = float("nan")
    area_px: int = 0
    ratio: float = float("nan")
    #: The last move of the aim [m].
    shift_m: float = float("nan")
    #: Set or sampled block hue [deg]. NaN until known.
    hue_deg: float = float("nan")
    #: Winning hue window's share of the expected visible silhouette on the latest sampling attempt [0..1].
    hue_share: float = float("nan")
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
            raise ValueError(f"grasp_wrist: hue_deg must lie within [0, 360) or be -1 (sampled), "
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
        """Initialize grasp and validate the named camera and OpenCV before motion.

        With no camera name, run plain grasp. Missing frames skip individual looks.
        """
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
        if cfg.hue_deg >= 0:
            self._window = HsvWindow(cfg.hue_deg, cfg.hue_window_deg, cfg.min_saturation, cfg.min_value)
            self.wrist.hue_deg = cfg.hue_deg
        self._tip_tcp = self.kin.jaw_centre(0.0)

    # ------------------------------------------------------------------ measuring
    def measure(self, rgb: np.ndarray, joints: Mapping[str, float]) -> Measurement:
        """Fit the block to RGB frame `rgb` using measured capture-pose `joints`.

        Return centre and yaw or a measurement with `problem` set.
        """
        cv2, camera = self._cv2, self._camera
        if cv2 is None or camera is None:
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
            # Convert calibration from the received frame to the rotated frame.
            seen = np.array([cfg.tip_pixel_x, cfg.tip_pixel_y])
            if cfg.rotate_180:
                seen = np.array([width - 1, height - 1]) - seen
            shift = seen - predicted
            off = float(np.linalg.norm(shift))
            if not off <= MAX_TIP_SHIFT * width:
                return _unusable(f"tip_pixel daleko od modelu ({off:.0f} px)")
            # Align the model projection with the observed tip pixel.
            K[0, 2] += shift[0]
            K[1, 2] += shift[1]
        small, Kw = working(image, K, cv2)
        shape = small.shape[:2]
        dilate = round(JAW_MASK_DILATE_PX * shape[1] / WORK_WIDTH_PX)
        visible = ~jaw_mask(kin, joints, Kw, T, shape, cv2, dilate)
        # The block models rest on the table: a camera lowered by the block's bottom sees them as it
        # sees the block on a stack.
        T = T.copy()
        T[2, 3] -= self.base_z
        assert self.grasp.block_xy is not None
        aim = np.asarray(self.grasp.block_xy, float)
        window = self._window or self._sample_hue(small, Kw, T, visible, aim)
        if window is None:
            return _unusable(f"brak jednego koloru w miejscu bloku ({self.wrist.hue_share:.0%})")
        labels, found = blobs(segment(small, window, cv2) & visible, cv2)

        period, top = self.block.yaw_period, self.height_m
        radius = cfg.max_shift_m + SEARCH_SLACK_M
        near: list[tuple[Blob, Fit, float]] = []
        for blob in found:
            xy_c = back_project(Kw, T, blob.centroid_px, top)[0, :2]
            if not (np.all(np.isfinite(xy_c)) and float(np.linalg.norm(xy_c - aim)) <= radius):
                continue
            xy_b, yaw_b = blob_pose(labels, blob, Kw, T, top, period, cv2)
            fit = fit_block(labels == blob.label, Kw, T, self.block, xy_b, yaw_b, cv2, visible=visible)
            # Compare areas over the visible part of the fitted silhouette.
            points = block_points(self.block, fit.xy, fit.yaw)
            model = int(np.count_nonzero(silhouette(Kw, T, points, shape, cv2) & visible))
            if not model >= MIN_VISIBLE_SHARE * hull_area(Kw, T, points, cv2):
                # Too little of the fitted block is visible to identify it.
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
                # Keep the aim along an axis the image cannot resolve.
                u = R[:2, k]
                xy -= float((xy - aim) @ u) * u
        pixel = project(K, T, np.array([*xy, top]))[0][0]
        if cfg.rotate_180:
            pixel = np.array([width - 1 - pixel[0], height - 1 - pixel[1]])
        return Measurement(xy, fit.yaw, fit.iou, blob.area, ratio, pixel)

    def _sample_hue(self, small: np.ndarray, Kw: np.ndarray, T: np.ndarray, visible: np.ndarray,
                    aim: np.ndarray) -> HsvWindow | None:
        """Sample the hue inside the expected visible silhouette at `aim`.

        Set the hue window when its share reaches `MIN_HUE_SHARE`. Otherwise return None.
        """
        cfg = self.wrist_settings
        yaw = math.atan2(self.block_rotation[1, 0], self.block_rotation[0, 0])
        region = silhouette(Kw, T, block_points(self.block, aim, yaw), small.shape[:2], self._cv2) & visible
        hue, share = dominant_hue(small, region, cfg.hue_window_deg, cfg.min_saturation, cfg.min_value,
                                  self._cv2)
        self.wrist.hue_share = share
        if not share >= MIN_HUE_SHARE:
            return None
        self.wrist.hue_deg = hue
        self._window = HsvWindow(hue, cfg.hue_window_deg, cfg.min_saturation, cfg.min_value)
        return self._window

    # ------------------------------------------------------------------ the seams
    def before_descent(self, obs: Observation) -> Action | None:
        """Look before the probe leaves hover. Return None immediately without a camera."""
        if not self.wrist_settings.wrist_camera:
            return None
        if self._look is None:
            self._look = self._new_look("probe")
        # Grasp resumes descent only after this hook returns None.
        settled = self._settled(self.hover_point)
        return self._look_step(obs, lambda: self.hover_point, settled)

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        """Look at the open-jaw hover before straddling.

        With the probe on, `look_before_straddle` controls this second look.
        Without the probe, the look always runs when a camera is configured.
        `settled` includes the OPEN phase's settle window.
        """
        cfg = self.wrist_settings
        if not cfg.wrist_camera or (cfg.probe and not cfg.look_before_straddle):
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
        """Advance a look by waiting, measuring or moving the aim.

        Return None on the tick after the final verdict to resume grasp.
        """
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
        """Drive to hover with the hand down and wait for all arm joints to settle.

        Skip after `LOOK_STALL_S` with a stationary target.
        """
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
        """Return block rotation for measured `yaw` [rad].

        Replace identity with an equivalent symmetry turn so grasp recognizes the orientation.
        """
        R = rotation_z(yaw)
        return rotation_z(yaw + self.block.yaw_period) if np.allclose(R, np.eye(3), atol=1e-6) else R

    def _reaim(self, look: Look, hover: Callable[[], np.ndarray], m: Measurement, dist: float) -> Action:
        """Aim at the measured centre and yaw, then drive to hover.

        Keep the old aim if the arm cannot reach the new one with the hand down.
        """
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
        """Restore the look's original aim, rotation and heading after a failed move check.

        If the original aim is unreachable, keep the current aim and skip the look.
        """
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
        """Hold and show the verdict for one tick. The hook returns None on the next tick."""
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
    """Return a measurement containing only the reason the frame is unusable."""
    nan = np.full(2, np.nan)
    return Measurement(nan, math.nan, math.nan, area_px, ratio, nan.copy(), problem)
