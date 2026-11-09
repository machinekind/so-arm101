"""The approach calibration's ChArUco board: its geometry, its detection and the wrist camera's measurement.

The board lies flat on the table. Its frame is OpenCV's: the origin at the board's top-left corner as
printed, x along the `squares_x` side, y along the `squares_y` side, z into the table. Inner corner `k`
lies at ((k % (squares_x - 1) + 1) * square, (k // (squares_x - 1) + 1) * square, 0).

A wrist measurement needs no joint reading. The camera is rigid to the jaws, so the TCP sits at a fixed
point `p` in the camera's axes. solvePnP on the board gives the board's pose in the camera, so corner `k`
lies at `c = R_cb @ b_k + t_cb`, and `c - p` is the vector from the TCP to the corner. The board's yaw in
the base frame, from the fixed cameras, turns that vector into base axes.

A constant error `d` of `p` in the camera's axes (a mount or focal length error) adds `rot2(psi) @ b` to
the horizontal result of a hand pointing straight down with closing heading `psi`, where `b` is constant in
the hand's frame. Two looks at one point with headings half a turn apart give `b` (`flip_bias`).

OpenCV is passed in as `cv2`, so the module imports without it. Pixels follow OpenCV: (0, 0) is the centre
of the top-left pixel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: A wrist look needs this many board points for solvePnP.
MIN_POINTS = 6
#: Marker corners join the ChArUco corners below this many points.
MARKERS_BELOW = 6
#: A wrist look's reprojection error may be at most this [px, RMS]. A point off its fit by more than
#: `OUTLIER_PX` is dropped, at most `MAX_OUTLIERS` of them.
MAX_REPROJ_PX = 1.5
OUTLIER_PX = 3.0
MAX_OUTLIERS = 2
#: Points closer to one line than this are collinear [m, smallest spread across the line].
MIN_SPREAD_M = 0.002
#: The intrinsics are trusted from this many views, this residual [px] and this spread of board tilts [deg],
#: with a focal length within this factor of the nominal one.
MIN_VIEWS = 10
MAX_RMS_PX = 1.0
MIN_TILT_SPREAD_DEG = 35.0
FOCAL_FACTOR = 2.0
#: Below this many views the third radial distortion coefficient stays zero.
FIX_K3_BELOW = 15
#: A view whose residual exceeds this factor of the median, and the largest residual allowed, is dropped,
#: at most `MAX_BAD_VIEWS` of them.
BAD_VIEW_FACTOR = 3.0
MAX_BAD_VIEWS = 3


def rot2(angle: float) -> np.ndarray:
    """The 2D rotation by `angle` [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------------ the board
@dataclass(frozen=True)
class BoardSpec:
    """A printed ChArUco board as the operator measured it."""

    squares_x: int = 3
    squares_y: int = 5
    #: The side of a square and of a marker, measured on the print [m].
    square_m: float = 0.045
    marker_m: float = 0.035
    #: An OpenCV predefined dictionary, by its name in `cv2.aruco`.
    dictionary: str = "DICT_6X6_250"
    #: The board's top above the table [m].
    thickness_m: float = 0.0

    def problem(self) -> str:
        """Why the spec cannot describe a board; '' when it can."""
        if not (isinstance(self.squares_x, int) and isinstance(self.squares_y, int)
                and self.squares_x >= 2 and self.squares_y >= 2):
            return f"squares_x and squares_y must be whole numbers of at least 2, are {self.squares_x}, " \
                   f"{self.squares_y}"
        if self.squares_x * self.squares_y < 6:
            return "the board needs at least 6 squares"
        if not (self.square_m > 0 and math.isfinite(self.square_m)):
            return f"square_m must be positive, is {self.square_m}"
        if not 0 < self.marker_m < self.square_m:
            return f"marker_m must lie between 0 and square_m ({self.square_m}), is {self.marker_m}"
        if not (isinstance(self.dictionary, str) and self.dictionary.startswith("DICT_")):
            return f"dictionary must name an OpenCV dictionary such as DICT_6X6_250, is {self.dictionary!r}"
        if not (0 <= self.thickness_m < 0.02):
            return f"thickness_m must lie within [0, 0.02), is {self.thickness_m}"
        return ""

    def check(self) -> None:
        why = self.problem()
        if why:
            raise ValueError(f"charuco board: {why}")

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    def corner_xy(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Inner corners `ids` in the board frame (n, 2) [m]."""
        ids = np.asarray(ids, int).ravel()
        cols = self.squares_x - 1
        return np.column_stack([(ids % cols + 1) * self.square_m,
                                (ids // cols + 1) * self.square_m]).astype(float)

    def corner_obj(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Inner corners `ids` in the board frame (n, 3) [m], z = 0."""
        xy = self.corner_xy(ids)
        return np.column_stack([xy, np.zeros(len(xy))])

    def as_dict(self) -> dict[str, Any]:
        return {"squares": [self.squares_x, self.squares_y], "square_m": self.square_m,
                "marker_m": self.marker_m, "dictionary": self.dictionary, "thickness_m": self.thickness_m}

    def cv_dictionary(self, cv2: Any) -> Any:
        code = getattr(cv2.aruco, self.dictionary, None)
        if not isinstance(code, int):
            raise ValueError(f"charuco board: OpenCV has no dictionary {self.dictionary!r}")
        return cv2.aruco.getPredefinedDictionary(code)

    def cv_board(self, cv2: Any) -> Any:
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m,
                                      self.cv_dictionary(cv2))

    def image(self, cv2: Any, px_per_m: float, margin_m: float = 0.0) -> np.ndarray:
        """The board as a grey image at `px_per_m`, with a white margin of `margin_m` around it."""
        w, h = (int(round(s * px_per_m)) for s in self.size_m)
        board = self.cv_board(cv2).generateImage((w, h), marginSize=0, borderBits=1)
        m = int(round(margin_m * px_per_m))
        if m <= 0:
            return board
        out = np.full((h + 2 * m, w + 2 * m), 255, np.uint8)
        out[m:m + h, m:m + w] = board
        return out


def corner_label(prefix: str, corner: int) -> str:
    """The object label of inner corner `corner`, e.g. "charuco c3": never a bare trailing number."""
    return f"{' '.join(prefix.split()).lower()} c{int(corner)}"


def corner_of(prefix: str, key: str) -> int | None:
    """The corner number of an `Observation.objects` key such as "charuco c3"; None for any other key."""
    head = f"{' '.join(prefix.split()).lower()} c"
    key = " ".join(str(key).split()).lower()
    if not key.startswith(head):
        return None
    tail = key[len(head):]
    return int(tail) if tail.isdigit() else None


# ------------------------------------------------------------------ the board in the base frame
@dataclass(frozen=True)
class BoardFit:
    """The board's pose on the table from its corners in the base frame: a 2D similarity fit."""

    #: The board's yaw [rad]: its x axis turned from the base's x axis.
    yaw: float
    #: Where the board's origin lies [m, base frame x and y].
    origin: np.ndarray
    #: The fitted size over the set one, and the corners' residual [m, RMS].
    scale: float
    rms_m: float

    @property
    def R_base_board(self) -> np.ndarray:
        """The board frame's axes in the base frame: z points down, into the table."""
        return board_rotation(self.yaw)

    def to_base(self, board_xy: np.ndarray) -> np.ndarray:
        """Board-frame points (n, 2) in the base frame (n, 2), without the fitted scale."""
        return np.asarray(board_xy, float) @ self.R_base_board[:2, :2].T + self.origin


def board_rotation(yaw: float) -> np.ndarray:
    """The axes of a board lying face up with yaw `yaw` [rad]: z points into the table."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])


def fit_board(board_xy: np.ndarray, base_xy: np.ndarray) -> BoardFit:
    """The yaw, origin and scale that map board-frame corners (n, 2) onto their base-frame points (n, 2).

    A board lying face up mirrors its y axis in the base's view, so the fit maps (x, -y) by a rotation."""
    src = np.asarray(board_xy, float) * np.array([1.0, -1.0])
    dst = np.asarray(base_xy, float)
    if len(src) < 2:
        raise ValueError("fitting the board needs at least two corners")
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    # Umeyama in 2D: the rotation from the cross-covariance, the scale from the variances.
    cov = b.T @ a / len(src)
    yaw = math.atan2(cov[1, 0] - cov[0, 1], cov[0, 0] + cov[1, 1])
    R = rot2(yaw)
    var = float((a ** 2).sum() / len(src))
    scale = float(np.trace(R.T @ cov) / var) if var > 0 else 1.0
    t = md - scale * R @ ms
    res = dst - (scale * src @ R.T + t)
    rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    origin = md - R @ ms
    return BoardFit(yaw, origin, scale, rms)


# ------------------------------------------------------------------ detection
def charuco_detector(spec: BoardSpec, cv2: Any, *, min_markers: int = 2) -> tuple[Any, Any]:
    """(the OpenCV board, a ChArUco detector refining marker corners to sub-pixels).

    `min_markers` 1 also interpolates corners next to a single marker: a wrist camera close to the board
    often sees only one."""
    board = spec.cv_board(cv2)
    params = cv2.aruco.CharucoParameters()
    params.minMarkers = int(min_markers)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return board, cv2.aruco.CharucoDetector(board, params, detector_params)


def grey(image: np.ndarray, cv2: Any) -> np.ndarray:
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def detect_corners(image: np.ndarray, detector: Any, cv2: Any) -> tuple[np.ndarray, np.ndarray]:
    """The ChArUco corners in `image`: ids (n,) and pixels (n, 2); empty when the board is not seen."""
    corners, ids, _, _ = detector.detectBoard(grey(image, cv2))
    if ids is None or corners is None or len(ids) == 0:
        return np.zeros(0, int), np.zeros((0, 2))
    return np.asarray(ids, int).ravel(), np.asarray(corners, float).reshape(-1, 2)


@dataclass(frozen=True)
class BoardPoints:
    """Board points found in one frame: their board-frame positions and pixels."""

    obj: np.ndarray
    img: np.ndarray
    #: ChArUco corners and marker corners among them.
    n_corners: int
    n_markers: int


def board_points(image: np.ndarray, board: Any, detector: Any, cv2: Any, *, mask: np.ndarray | None = None,
                 markers_below: int = MARKERS_BELOW, all_markers: bool = False) -> BoardPoints:
    """The ChArUco corners outside `mask` (True = hidden), and marker corners outside it when fewer than
    `markers_below` corners remain, or always with `all_markers`."""
    corners, ids, marker_corners, marker_ids = detector.detectBoard(grey(image, cv2))
    obj: list[np.ndarray] = []
    img: list[np.ndarray] = []

    def visible(px: np.ndarray) -> np.ndarray:
        if mask is None:
            return np.ones(len(px), bool)
        h, w = mask.shape[:2]
        u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        return ~mask[v, u]

    n_corners = 0
    if ids is not None and corners is not None and len(ids):
        px = np.asarray(corners, float).reshape(-1, 2)
        cid = np.asarray(ids, int).ravel()
        keep = visible(px)
        all_obj = np.asarray(board.getChessboardCorners(), float).reshape(-1, 3)
        obj.append(all_obj[cid[keep]])
        img.append(px[keep])
        n_corners = int(keep.sum())
    n_markers = 0
    if (all_markers or n_corners < markers_below) and marker_ids is not None and len(marker_ids):
        board_ids = [int(i) for i in np.asarray(board.getIds()).ravel()]
        marker_obj = np.asarray(board.getObjPoints(), float).reshape(-1, 4, 3)
        for quad, mid in zip(marker_corners, np.asarray(marker_ids, int).ravel(), strict=True):
            if int(mid) not in board_ids:
                continue
            px = np.asarray(quad, float).reshape(4, 2)
            if not visible(px).all():
                continue
            obj.append(marker_obj[board_ids.index(int(mid))])
            img.append(px)
            n_markers += 4
    if not obj:
        return BoardPoints(np.zeros((0, 3)), np.zeros((0, 2)), 0, 0)
    return BoardPoints(np.vstack(obj), np.vstack(img), n_corners, n_markers)


def spread_m(obj: np.ndarray) -> float:
    """How far board points (n, 3) spread across their best line [m]: 0 for collinear points."""
    xy = np.asarray(obj, float)[:, :2]
    if len(xy) < 3:
        return 0.0
    s = np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)
    return float(s[-1] / math.sqrt(len(xy)))


@dataclass(frozen=True)
class BoardView:
    """The board's pose in one wrist frame: corner k lies at `R_cb @ b_k + t_cb` in the camera's axes."""

    R_cb: np.ndarray
    t_cb: np.ndarray
    reproj_px: float
    points: int
    n_corners: int
    n_markers: int
    problem: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem


def _unusable(problem: str, points: int = 0, n_corners: int = 0, n_markers: int = 0,
              reproj: float = math.nan) -> BoardView:
    return BoardView(np.eye(3), np.zeros(3), reproj, points, n_corners, n_markers, problem)


def board_pose(points: BoardPoints, K: np.ndarray, dist: np.ndarray | None, cv2: Any, *,
               min_points: int = MIN_POINTS, max_reproj_px: float = MAX_REPROJ_PX) -> BoardView:
    """solvePnP on board points: IPPE for the planar board, then a Levenberg-Marquardt refinement.

    A point far off the others' fit (a corner interpolated next to a marker at the frame's edge) is dropped,
    up to `MAX_OUTLIERS` of them, while at least `min_points` remain."""
    obj = np.ascontiguousarray(points.obj, np.float64)
    img = np.ascontiguousarray(points.img, np.float64)
    n0 = len(obj)
    if n0 < min_points:
        return _unusable(f"za malo punktow tablicy ({n0})", n0, points.n_corners, points.n_markers)
    if spread_m(obj) < MIN_SPREAD_M:
        return _unusable("punkty tablicy na jednej linii", n0, points.n_corners, points.n_markers)
    Km = np.asarray(K, np.float64)
    d = None if dist is None else np.asarray(dist, np.float64)
    rvec, tvec, reproj = np.zeros(3), np.zeros(3), math.inf
    for dropped in range(MAX_OUTLIERS + 1):
        ok, rvec, tvec = cv2.solvePnP(obj, img, Km, d, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return _unusable("solvePnP bez wyniku", len(obj), points.n_corners, points.n_markers)
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, Km, d, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, Km, d)
        errors = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        reproj = float(np.sqrt((errors ** 2).mean()))
        worst = int(np.argmax(errors))
        if (reproj <= max_reproj_px or dropped == MAX_OUTLIERS or len(obj) <= min_points
                or errors[worst] <= OUTLIER_PX):
            break
        keep = np.arange(len(obj)) != worst
        if spread_m(obj[keep]) < MIN_SPREAD_M:
            break
        obj, img = obj[keep], img[keep]
    R = cv2.Rodrigues(rvec)[0]
    t = np.asarray(tvec, float).ravel()
    n = len(obj)
    if not reproj <= max_reproj_px:
        return _unusable(f"blad reprojekcji {reproj:.1f} px", n, points.n_corners, points.n_markers, reproj)
    if not t[2] > 0:
        return _unusable("tablica za kamera", n, points.n_corners, points.n_markers, reproj)
    return BoardView(np.asarray(R, float), t, reproj, n, points.n_corners, points.n_markers)


# ------------------------------------------------------------------ the TCP against a corner
def tcp_to_corner(view: BoardView, corner_obj: np.ndarray, p_cam: np.ndarray, R_base_board: np.ndarray
                  ) -> np.ndarray:
    """The vector from the TCP to a board corner in base axes (3,) [m].

    `corner_obj` is the corner in the board frame, `p_cam` the TCP in the camera's axes and `R_base_board`
    the board's axes in the base frame."""
    c = view.R_cb @ np.asarray(corner_obj, float) + view.t_cb
    return R_base_board @ (view.R_cb.T @ (c - np.asarray(p_cam, float)))


def remove_bias(raw_xy: np.ndarray, psi: float, bias: np.ndarray | Sequence[float]) -> np.ndarray:
    """A horizontal measurement without the hand-frame bias `bias` seen at closing heading `psi` [rad]."""
    return np.asarray(raw_xy, float)[:2] - rot2(psi) @ np.asarray(bias, float)


def flip_bias(raw_a: np.ndarray, raw_b: np.ndarray, psi_a: float) -> tuple[np.ndarray, np.ndarray]:
    """From one point measured at heading `psi_a` and half a turn from it: the hand-frame bias, and the
    measurement without it (their mean)."""
    a, b = np.asarray(raw_a, float)[:2], np.asarray(raw_b, float)[:2]
    return rot2(-psi_a) @ (a - b) / 2, (a + b) / 2


# ------------------------------------------------------------------ intrinsics
@dataclass
class Intrinsics:
    """A camera calibration from board views, and whether it can be trusted."""

    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    views: int
    size: tuple[int, int]
    tilt_spread_deg: float
    trusted: bool
    reason: str = ""
    per_view_px: list[float] = field(default_factory=list)


def tilt_spread_deg(normals: Sequence[np.ndarray]) -> float:
    """The combined spread of board tilts in the camera [deg]: about the image's x and y axes."""
    n = np.array([np.asarray(v, float) for v in normals])
    if len(n) == 0:
        return 0.0
    n = n * np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]
    about_x = np.degrees(np.arctan2(n[:, 1], n[:, 2]))
    about_y = np.degrees(np.arctan2(n[:, 0], n[:, 2]))
    return float(np.hypot(np.ptp(about_x), np.ptp(about_y)))


def calibrate(views: Sequence[BoardPoints], size: tuple[int, int], nominal_f: float, cv2: Any, *,
              min_views: int = MIN_VIEWS, max_rms_px: float = MAX_RMS_PX,
              min_tilt_spread_deg: float = MIN_TILT_SPREAD_DEG) -> Intrinsics:
    """K and distortion from board views of one camera at `size` (width, height), and the verdict.

    The verdict follows the twin's ChArUco rules: enough views, a small residual, a spread of board tilts
    (a board always seen at one tilt cannot tell focal length from distance) and a focal length within
    `FOCAL_FACTOR` of `nominal_f`."""
    usable = [v for v in views if len(v.obj) >= MIN_POINTS and spread_m(v.obj) >= MIN_SPREAD_M]
    W, H = int(size[0]), int(size[1])
    K0 = np.array([[nominal_f, 0.0, (W - 1) / 2], [0.0, nominal_f, (H - 1) / 2], [0.0, 0.0, 1.0]])
    if len(usable) < 4:
        return Intrinsics(K0, np.zeros(5), math.nan, len(usable), (W, H), 0.0, False,
                          f"za malo ujec tablicy ({len(usable)})")
    rms, K, dist, rvecs, per_view = math.inf, K0, np.zeros(5), [], []
    # A view far off the others' fit (a marker corner next to the jaws) is dropped, a few at most.
    for _ in range(MAX_BAD_VIEWS + 1):
        obj = [np.ascontiguousarray(v.obj, np.float32) for v in usable]
        img = [np.ascontiguousarray(v.img, np.float32) for v in usable]
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_FIX_K3 if len(usable) < FIX_K3_BELOW else 0)
        out = cv2.calibrateCameraExtended(obj, img, (W, H), K0.copy(), np.zeros(5), flags=flags)
        rms, rvecs = float(out[0]), out[3]
        K, dist = np.asarray(out[1], float), np.asarray(out[2], float).ravel()
        per_view = [float(e) for e in np.asarray(out[7]).ravel()]
        worst = int(np.argmax(per_view))
        limit = max(BAD_VIEW_FACTOR * float(np.median(per_view)), max_rms_px)
        if per_view[worst] <= limit or len(usable) <= 4:
            break
        usable = usable[:worst] + usable[worst + 1:]
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rvecs]
    spread = tilt_spread_deg(normals)
    reasons = []
    if len(usable) < min_views:
        reasons.append(f"tylko {len(usable)} ujec, potrzeba {min_views}")
    if not rms <= max_rms_px:
        reasons.append(f"residuum {rms:.2f} px > {max_rms_px} px")
    if spread < min_tilt_spread_deg:
        reasons.append(f"rozrzut pochylen {spread:.0f} st. < {min_tilt_spread_deg:.0f} st.")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if not (nominal_f / FOCAL_FACTOR <= fx <= nominal_f * FOCAL_FACTOR
            and nominal_f / FOCAL_FACTOR <= fy <= nominal_f * FOCAL_FACTOR):
        reasons.append(f"nieprawdopodobna ogniskowa fx {fx:.0f}, fy {fy:.0f} px (oczekiwane okolo "
                       f"{nominal_f:.0f} px)")
    return Intrinsics(K, dist, rms, len(usable), (W, H), spread, not reasons, "; ".join(reasons), per_view)
