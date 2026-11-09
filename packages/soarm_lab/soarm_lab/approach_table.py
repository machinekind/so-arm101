"""The approach table: how far the hand lands from where the stand's cameras see a point, and its lookup.

`calibrate_approach` measures it and `pick_place` reads it. A sample is the horizontal correction at one
board corner, one height and one final-leg direction (`strategies.approach`): the true corner minus where
the TCP really ended up, with the arm commanded to the corner as the cameras see it. Commanding the TCP
to the perceived point plus the correction puts it on the true point.

The file is JSON, written whole through a temporary file. Its fields:

- `format`, `version`, `created`, `updated`, `robot`, `stand` and `complete` (False for a partial run);
- `board`: squares, square_m, marker_m, dictionary, thickness_m;
- `protocol`: directions, lead_m, carry_clearance_m, place_gap_m, speeds, gripper, hang_m;
- `wrist`: camera, frame size, K, distortion, residual, views, the TCP in the camera's axes, the mount
  bias per level, its spread, and a suggested `focal_px` and tip pixel for `grasp_wrist`;
- `levels`: per height level the surface the held block lands on and the TCP's height there;
- `spots`: each board placement measured, with its corners in the base frame;
- `samples`: spot, corner, perceived xy, level, direction, repeat, heading, the correction `delta_m`
  (x, y, z), the raw measurement, the points used, the reprojection error and the time;
- `mount_pairs`, `skipped`, `spread_m` (the median spread of repeats per direction), `spread_mean_m`,
  `spread_by_level`, `preferred_direction`, `verify` and `time_s`.

`lookup` takes the nearest level within `LEVEL_TOL_M`, the samples of one direction averaged over repeats
per corner, and a moving least squares fit of a value and a gradient around the asked point. It misses
outside the measured corners' hull by more than `EXTRAPOLATE_M`.

numpy only.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "soarm-approach-table"
VERSION = 1
#: Final-leg directions a table may hold, in report order.
DIRECTIONS = ("out", "in", "ccw", "cw")
#: A height further than this from every level gets no answer [m].
LEVEL_TOL_M = 0.012
#: A point further than this outside the measured corners' hull gets no answer [m].
EXTRAPOLATE_M = 0.03
#: Width of the lookup's weights [m] and the ridge on the gradient, relative to the weights.
SIGMA_M = 0.05
RIDGE = 0.01
#: Corners a direction needs at a level.
MIN_POINTS = 3
#: Median repeat spreads closer than this count as equal when the preferred direction is chosen [m].
SPREAD_STEP_M = 0.0001
#: Why a lookup gives no answer.
MISS_TABLE = "table"
MISS_HEIGHT = "height"
MISS_DIRECTION = "direction"
MISS_OUTSIDE = "outside"
MISS_STALE = "stale"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class Hit:
    """A correction to add to the commanded TCP [m, base frame x and y]."""

    delta_xy: np.ndarray
    #: The level it comes from, the corners it fits and the fit's weighted residual [m, RMS].
    level: int
    n: int
    rms_m: float


@dataclass(frozen=True)
class Miss:
    """No answer: `reason` is one of the MISS_ names, `detail` says more for a note."""

    reason: str
    detail: str = ""


def now_text(t: float | None = None) -> str:
    return time.strftime(TIME_FORMAT, time.localtime(time.time() if t is None else t))


def _parse_time(text: str) -> float | None:
    try:
        return time.mktime(time.strptime(str(text), TIME_FORMAT))
    except (TypeError, ValueError):
        return None


def write_atomic(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Write `data` as JSON to `path` through a temporary file in the same folder, made if missing."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_plain(data), f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _plain(value: Any) -> Any:
    """`value` as JSON data: arrays and tuples as lists, numpy scalars as numbers, NaN as None."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# ------------------------------------------------------------------ geometry
def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull (m, 2) of points (n, 2), counter-clockwise; fewer points for degenerate sets."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, float)})
    if len(pts) <= 2:
        return np.array(pts, float).reshape(-1, 2)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-15:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-15:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], float)


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    s = 0.0 if denom <= 0 else min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return float(np.linalg.norm(p - (a + s * ab)))


def outside_m(xy: np.ndarray, points: np.ndarray) -> float:
    """How far `xy` lies outside the convex hull of `points` (n, 2) [m]; 0 inside it."""
    p = np.asarray(xy, float)[:2]
    hull = _hull(points)
    if len(hull) == 0:
        return math.inf
    if len(hull) == 1:
        return float(np.linalg.norm(p - hull[0]))
    if len(hull) == 2:
        return _segment_distance(p, hull[0], hull[1])
    inside = True
    for k in range(len(hull)):
        a, b = hull[k], hull[(k + 1) % len(hull)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
            break
    if inside:
        return 0.0
    return min(_segment_distance(p, hull[k], hull[(k + 1) % len(hull)]) for k in range(len(hull)))


def mls(xy: np.ndarray, points: np.ndarray, values: np.ndarray, sigma: float = SIGMA_M,
        ridge: float = RIDGE) -> tuple[np.ndarray, float]:
    """Moving least squares at `xy`: a value plus a gradient fitted to `values` (n, k) at `points` (n, 2),
    weighted by a Gaussian of width `sigma`, the gradient held back by `ridge`. Returns the value at `xy`
    and the weighted residual [RMS]."""
    p = np.asarray(points, float) - np.asarray(xy, float)[:2]
    v = np.asarray(values, float)
    w = np.exp(-(p ** 2).sum(axis=1) / (2 * sigma ** 2))
    total = float(w.sum())
    if not total > 0:
        w = np.ones(len(p))
        total = float(len(p))
    A = np.column_stack([np.ones(len(p)), p])
    M = (A * w[:, None]).T @ A
    M[1:, 1:] += np.eye(2) * ridge * total * sigma ** 2
    coef = np.linalg.solve(M, (A * w[:, None]).T @ v)
    res = v - A @ coef
    rms = float(np.sqrt((w[:, None] * res ** 2).sum() / total))
    return coef[0], rms


# ------------------------------------------------------------------ the table
class ApproachTable:
    """An approach table held as its JSON data."""

    def __init__(self, data: Mapping[str, Any]):
        self.data: dict[str, Any] = dict(data)

    # ---------------------------------------------------------- files
    @classmethod
    def new(cls, *, robot: str, stand: str, board: Mapping[str, Any], protocol: Mapping[str, Any],
            wrist: Mapping[str, Any], levels: Sequence[Mapping[str, float]]) -> ApproachTable:
        stamp = now_text()
        return cls({"format": FORMAT, "version": VERSION, "created": stamp, "updated": stamp, "robot": robot,
                    "stand": stand, "complete": False, "board": dict(board), "protocol": dict(protocol),
                    "wrist": dict(wrist), "levels": [dict(level) for level in levels], "spots": [],
                    "samples": [], "mount_pairs": [], "skipped": [], "spread_m": {}, "spread_by_level": {},
                    "preferred_direction": "", "verify": [], "time_s": 0.0})

    @classmethod
    def load(cls, path: str | Path) -> ApproachTable:
        """The table in `path`. Raises OSError for a missing file, ValueError for one that is no table."""
        path = Path(path).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not JSON ({exc})") from exc
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(f"{path}: not an approach table")
        if data.get("version") != VERSION:
            raise ValueError(f"{path}: approach table version {data.get('version')!r}, this code reads "
                             f"{VERSION}")
        for key in ("levels", "samples", "spots"):
            if not isinstance(data.get(key), list):
                raise ValueError(f"{path}: approach table without {key!r}")
        return cls(data)

    def save(self, path: str | Path) -> Path:
        self.data["updated"] = now_text()
        return write_atomic(path, self.data)

    # ---------------------------------------------------------- contents
    @property
    def levels(self) -> list[dict[str, Any]]:
        return list(self.data.get("levels", []))

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self.data.get("samples", []))

    @property
    def preferred_direction(self) -> str:
        return str(self.data.get("preferred_direction") or "")

    def problem(self, robot: str | None = None) -> str:
        """Why the table cannot answer at all; '' when it can."""
        if self.data.get("format") != FORMAT or self.data.get("version") != VERSION:
            return "not a version 1 approach table"
        if robot is not None and self.data.get("robot") != robot:
            return f"made on robot {self.data.get('robot')!r}, not {robot!r}"
        if not self.data.get("complete"):
            return "the calibration did not finish"
        return ""

    def age_days(self, now: float | None = None) -> float:
        """Days since the table was last written; infinite when it carries no readable time."""
        t = _parse_time(self.data.get("updated") or self.data.get("created") or "")
        if t is None:
            return math.inf
        return max(0.0, ((time.time() if now is None else now) - t) / 86400.0)

    def level_of(self, z: float) -> tuple[int, float]:
        """The level whose TCP height is nearest `z`, and how far it is [m]; (-1, inf) without levels."""
        best, gap = -1, math.inf
        for k, level in enumerate(self.levels):
            d = abs(float(level["tcp_z_m"]) - float(z))
            if d < gap:
                best, gap = k, d
        return best, gap

    def points(self, level: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
        """Perceived corners (n, 2) and their corrections (n, 2) at `level` from `direction`, each the
        mean over its repeats."""
        groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for s in self.samples:
            if int(s.get("level", -1)) == level and s.get("direction") == direction:
                groups.setdefault((s.get("spot"), s.get("corner")), []).append(s)
        if not groups:
            return np.zeros((0, 2)), np.zeros((0, 2))

        def mean(key: str) -> np.ndarray:
            return np.array([np.mean([np.asarray(s[key], float)[:2] for s in g], axis=0)
                             for g in groups.values()])

        return mean("xy_m"), mean("delta_m")

    def lookup(self, xy: np.ndarray | Sequence[float], z: float, direction: str, *, robot: str | None = None,
               max_age_days: float = 0.0, now: float | None = None, level_tol_m: float = LEVEL_TOL_M,
               extrapolate_m: float = EXTRAPOLATE_M) -> Hit | Miss:
        """The correction for a TCP commanded over perceived `xy` at height `z` after a final leg from
        `direction`; a `Miss` when the table cannot tell."""
        why = self.problem(robot)
        if why:
            return Miss(MISS_TABLE, why)
        if max_age_days > 0:
            age = self.age_days(now)
            if age > max_age_days:
                return Miss(MISS_STALE, f"{age:.0f} days old")
        level, gap = self.level_of(z)
        if level < 0 or gap > level_tol_m:
            return Miss(MISS_HEIGHT, f"{gap * 1000:.0f} mm from the nearest level")
        points, deltas = self.points(level, direction)
        if len(points) < MIN_POINTS:
            return Miss(MISS_DIRECTION, f"{len(points)} corners from {direction!r} at level {level + 1}")
        p = np.asarray(xy, float)[:2]
        out = outside_m(p, points)
        if out > extrapolate_m:
            return Miss(MISS_OUTSIDE, f"{out * 1000:.0f} mm outside the measured corners")
        value, rms = mls(p, points, deltas)
        return Hit(np.asarray(value, float), level, len(points), rms)

    # ---------------------------------------------------------- spreads
    def repeat_spreads(self) -> dict[tuple[int, str], list[float]]:
        """Per (level, direction): the spread of each corner's repeats [m], the largest distance between
        two of them. Only the ordinary repeats (numbered from 1) count."""
        groups: dict[tuple[Any, Any, int, str], list[np.ndarray]] = {}
        for s in self.samples:
            if int(s.get("repeat", 0)) < 1:
                continue
            key = (s.get("spot"), s.get("corner"), int(s.get("level", -1)), str(s.get("direction")))
            groups.setdefault(key, []).append(np.asarray(s["delta_m"], float)[:2])
        out: dict[tuple[int, str], list[float]] = {}
        for (_, _, level, direction), values in groups.items():
            if len(values) < 2:
                continue
            v = np.array(values)
            spread = float(np.linalg.norm(v[:, None] - v[None], axis=2).max())
            out.setdefault((level, direction), []).append(spread)
        return out

    def summarize(self) -> None:
        """Refresh `spread_m`, `spread_mean_m`, `spread_by_level` and `preferred_direction` from the samples.

        The preferred direction has the smallest median spread, to `SPREAD_STEP_M`; the mean spread breaks a
        tie. A direction whose repeats differ only now and then, after some ways in, has a median of zero
        and a mean that tells."""
        spreads = self.repeat_spreads()
        by_direction: dict[str, list[float]] = {}
        by_level: dict[str, dict[str, float]] = {}
        for (level, direction), values in spreads.items():
            by_direction.setdefault(direction, []).extend(values)
            by_level.setdefault(str(level), {})[direction] = float(np.median(values))
        spread = {d: float(np.median(v)) for d, v in by_direction.items()}
        mean = {d: float(np.mean(v)) for d, v in by_direction.items()}
        order = [d for d in DIRECTIONS if d in spread] + sorted(d for d in spread if d not in DIRECTIONS)
        self.data["spread_m"] = {d: spread[d] for d in order}
        self.data["spread_mean_m"] = {d: mean[d] for d in order}
        self.data["spread_by_level"] = {k: by_level[k] for k in sorted(by_level, key=int)}
        def rank(d: str) -> tuple[int, float]:
            return round(spread[d] / SPREAD_STEP_M), mean[d]

        self.data["preferred_direction"] = min(order, key=rank) if order else ""

    # ---------------------------------------------------------- merging runs
    def merge_problem(self, *, robot: str, board: Mapping[str, Any], size: Sequence[int] | None,
                      levels: Iterable[Mapping[str, float]]) -> str:
        """Why a new run cannot add its spot to this table; '' when it can."""
        if self.data.get("robot") != robot:
            return f"robot {self.data.get('robot')!r} w pliku, teraz {robot!r}"
        if dict(self.data.get("board") or {}) != dict(_plain(dict(board))):
            return "inna tablica niz w pliku"
        have = (self.data.get("wrist") or {}).get("size")
        if size is not None and have is not None and list(have) != [int(s) for s in size]:
            return f"klatka kamery nadgarstka {have[0]}x{have[1]} w pliku, teraz {size[0]}x{size[1]}"
        mine = [round(float(level["tcp_z_m"]), 4) for level in self.levels]
        theirs = [round(float(level["tcp_z_m"]), 4) for level in levels]
        if mine != theirs:
            return "inne poziomy wysokosci niz w pliku"
        return ""

    def next_spot(self) -> int:
        return 1 + max((int(s.get("id", 0)) for s in self.data.get("spots", [])), default=0)


def summary(table: ApproachTable) -> str:
    """A plain-text account of a table for the command line."""
    d = table.data
    wrist = d.get("wrist") or {}
    lines = [f"approach table: robot {d.get('robot')}, stand {d.get('stand') or '-'}, "
             f"{'complete' if d.get('complete') else 'PARTIAL'}",
             f"created {d.get('created')}, updated {d.get('updated')}",
             f"board: {d.get('board')}",
             f"spots: {len(d.get('spots', []))}, samples: {len(d.get('samples', []))}, "
             f"skipped: {len(d.get('skipped', []))}, time {float(d.get('time_s') or 0.0) / 60:.1f} min"]
    for k, level in enumerate(d.get("levels", [])):
        lines.append(f"level {k + 1}: surface {level['surface_m'] * 1000:.0f} mm, "
                     f"TCP at {level['tcp_z_m'] * 1000:.1f} mm")
    spread = d.get("spread_m") or {}
    if spread:
        lines.append("repeat spread [mm]: " + ", ".join(f"{k} {v * 1000:.2f}" for k, v in spread.items()))
    lines.append(f"preferred direction: {d.get('preferred_direction') or '-'}")
    verify = [v.get("error_m") for v in d.get("verify", []) if v.get("error_m") is not None]
    if verify:
        lines.append(f"self-check: median {np.median(verify) * 1000:.2f} mm, "
                     f"worst {max(verify) * 1000:.2f} mm over {len(verify)} approaches")
    if wrist.get("K") is not None:
        K = np.asarray(wrist["K"], float)
        lines.append(f"wrist camera {wrist.get('camera')}: {wrist.get('size')}, "
                     f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, "
                     f"residual {wrist.get('rms_px')} px, {wrist.get('views')} views")
    bias = wrist.get("mount_bias_m") or {}
    if bias:
        parts = [f"level {int(k) + 1} ({v[0] * 1000:.2f}, {v[1] * 1000:.2f})"
                 for k, v in sorted(bias.items(), key=lambda kv: int(kv[0]))]
        lines.append("mount bias [mm]: " + ", ".join(parts))
    suggested = wrist.get("suggested") or {}
    if suggested:
        tip = suggested.get("tip_pixel") or [None, None]
        lines.append(f"grasp_wrist: focal_px {suggested.get('focal_px')}, tip_pixel_x {tip[0]}, "
                     f"tip_pixel_y {tip[1]}")
    return "\n".join(lines)
