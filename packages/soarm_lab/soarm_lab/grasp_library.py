"""Load, convert and select offline grasp libraries for blocks.

Each block has a JSON file under `LIBRARY_DIR` or a supplied folder. A grasp stores a pad
point [m] and unit approach and closing vectors in the block frame. The body origin is
the block centre, x follows its long side and z is up in the resting pose. The strategy
places the fingertip midpoint at the pad point, defined by the gripper's open sweep box.

`convert_isaac_grasps` maps GraspGenX poses into this frame. `compose` applies the observed
rotation and block symmetries. `select` filters, deduplicates and ranks the executable
results. The loader checks dimensions against `blocks.block` so changed dimensions require
regeneration. GraspGenX runs in a separate environment. See tools/graspgenx/README.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .blocks import BlockSpec, block
from .blocks.classic_world import BLOCKS
from .kinematics import RobotKinematics, inverse, pose

#: The `format` and `version` of a library file.
FORMAT = "soarm-lab-grasps"
VERSION = 1
#: The libraries bundled with the lab.
LIBRARY_DIR = Path(__file__).resolve().parent / "grasps"
#: Stored coordinate-frame description.
FRAME = ("block frame of soarm_lab/blocks (centre, x the long side, z up); position = the pad point, the "
         "centre of the gripper description's open sweep box, where the strategy puts its fingertip "
         "midpoint; approach and closing are unit vectors in the block frame (gripperframe x and z)")
#: GraspGenX's approach and closing axes in its aligned gripper frame.
GRASPGENX_APPROACH = (0.0, 0.0, 1.0)
GRASPGENX_CLOSING = (1.0, 0.0, 0.0)
#: Score-bin width for ranking [0..1]. Wrist turn breaks ties within each bin.
SCORE_TIE = 0.05
#: Axis-length and orthogonality tolerance, allowing for JSON rounding.
UNIT_TOL = 1e-5
#: A pad point may lie this far outside the block's bounding box [m].
INSIDE_TOL_M = 0.001
#: Decimal precision for duplicate heading, offset, depth and width keys.
KEY_DECIMALS = 5
#: Rim vertices of a cylinder's OBJ at each end.
CYLINDER_SEGMENTS = 64
#: Rejection labels in filter order.
REJECTIONS = ("pochylenie", "wynik", "szczeka", "glebokosc", "poza blokiem")


@dataclass(frozen=True)
class LibraryGrasp:
    """One grasp of a library, in the block frame."""

    #: The pad point [m].
    position: tuple[float, float, float]
    #: The gripperframe's approach (x) and closing (z) directions, unit vectors.
    approach: tuple[float, float, float]
    closing: tuple[float, float, float]
    #: Block width along `closing` [m].
    width_m: float
    #: The generator's score, higher is better.
    score: float

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return approach and closing as an orthonormal pair, correcting file-rounding error."""
        a = np.asarray(self.approach, float)
        a = a / np.linalg.norm(a)
        c = np.asarray(self.closing, float)
        c = c - float(c @ a) * a
        return a, c / np.linalg.norm(c)

    def rotation(self) -> np.ndarray:
        """Return the block-frame TCP rotation with columns approach, closing × approach, closing."""
        a, c = self.axes()
        return np.column_stack([a, np.cross(c, a), c])


@dataclass(frozen=True)
class GraspLibrary:
    """Grasps for one block, sorted by descending score."""

    kind: str
    #: Block half sizes at generation [m], checked against the current catalog.
    half_size: tuple[float, float, float]
    #: Generator identity, version, gripper, checkpoint, date, notes and dropped count.
    generator: dict[str, str]
    grasps: tuple[LibraryGrasp, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the file-format mapping at full precision and in schema order."""
        return {"format": FORMAT, "version": VERSION, "kind": self.kind, "half_size": list(self.half_size),
                "frame": FRAME, "generator": dict(self.generator),
                "grasps": [{"position": list(g.position), "approach": list(g.approach),
                            "closing": list(g.closing), "width_m": g.width_m, "score": g.score}
                           for g in self.grasps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraspLibrary:
        """Validate a library mapping, raising ValueError with the invalid field or grasp."""
        if not isinstance(data, Mapping):
            raise ValueError("a grasp library is a JSON object")
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError(f"not a grasp library of version {VERSION}: format {data.get('format')!r}, "
                             f"version {data.get('version')!r}")
        kind = data.get("kind")
        if kind not in BLOCKS:
            raise ValueError(f"grasp library for an unknown kind {kind!r}; known: {', '.join(BLOCKS)}")
        spec = block(kind)
        half = _vector(data.get("half_size"), f"{kind} library: half_size")
        if not np.allclose(half, spec.half_size, rtol=0.0, atol=1e-9):
            made, table = (" x ".join(f"{2000 * h:.1f}" for h in sizes) for sizes in (half, spec.half_size))
            raise ValueError(f"{kind} library was made for a {made} mm block, the catalog says {table} mm: "
                             "regenerate it (tools/graspgenx/README.md)")
        generator = data.get("generator", {})
        if not isinstance(generator, Mapping):
            raise ValueError(f"{kind} library: generator is not an object")
        entries = data.get("grasps")
        if not isinstance(entries, Sequence) or isinstance(entries, str):
            raise ValueError(f"{kind} library: grasps is not a list")
        grasps = tuple(_grasp(entry, spec, f"{kind} library, grasp {i}") for i, entry in enumerate(entries))
        return cls(kind, spec.half_size, {str(k): str(v) for k, v in generator.items()}, grasps)


def _vector(value: Any, what: str) -> np.ndarray:
    """Return three finite numbers or raise ValueError naming `what`."""
    try:
        v = np.asarray(value, float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not three numbers: {value!r}") from None
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{what} is not three finite numbers: {value!r}")
    return v


def _grasp(entry: Any, spec: BlockSpec, where: str) -> LibraryGrasp:
    """Validate one stored grasp against the block geometry."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: not an object")
    p, a, c = (_vector(entry.get(name), f"{where}: {name}") for name in ("position", "approach", "closing"))
    for name, v in (("approach", a), ("closing", c)):
        if abs(float(np.linalg.norm(v)) - 1.0) > UNIT_TOL:
            raise ValueError(f"{where}: {name} is not a unit vector (length {np.linalg.norm(v):.6f})")
    if abs(float(a @ c)) >= UNIT_TOL:
        raise ValueError(f"{where}: approach and closing are not perpendicular (dot {float(a @ c):.6f})")
    if np.any(np.abs(p) > np.asarray(spec.half_size) + INSIDE_TOL_M):
        raise ValueError(f"{where}: the pad point {tuple(round(float(x), 4) for x in p)} m lies outside the "
                         "block")
    try:
        width, score = float(entry["width_m"]), float(entry["score"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: width_m and score must be numbers") from None
    if not (math.isfinite(width) and width > 0):
        raise ValueError(f"{where}: width_m must be positive, is {width}")
    if not math.isfinite(score):
        raise ValueError(f"{where}: score must be finite, is {score}")
    return LibraryGrasp(_triple(p), _triple(a), _triple(c), width, score)


def _triple(v: np.ndarray) -> tuple[float, float, float]:
    return float(v[0]), float(v[1]), float(v[2])


@dataclass(frozen=True)
class GripperConfig:
    """Fields read from a GraspGenX gripper config.json."""

    #: The turn from the URDF root to the aligned frame (4x4).
    base_rotation: np.ndarray
    #: The centre of the open sweep box in the aligned frame (`sweep_volume.offset`) [m]: the pad point.
    pad_point: np.ndarray
    fingertip: np.ndarray
    #: The joint values of the open and the closed jaw [rad].
    open: dict[str, float]
    close: dict[str, float]
    type: str
    symmetric: bool


@dataclass(frozen=True)
class ComposedGrasp:
    """A library grasp transformed into an executable straddle candidate."""

    #: The grasp's index in the library.
    index: int
    #: The closing heading [rad, base frame].
    psi: float
    #: The pad point minus the block centre, horizontally [m, base frame].
    offset_xy: tuple[float, float]
    #: How far the pad point lies below the block's top [m].
    depth_m: float
    width_m: float
    score: float
    #: How far the library's approach leans from straight down [rad].
    tilt_rad: float
    #: True for the half-turn heading with the jaws swapped.
    flipped: bool
    #: Which turn of the block's yaw symmetry.
    fold: int
    #: How far a cylinder's grasp was rolled about its axis to point down [rad].
    roll_rad: float = 0.0


# ---------------------------------------------------------------- rotations
def wrap(angle: float) -> float:
    """`angle` folded into (-pi, pi] [rad]."""
    folded = (angle + math.pi) % (2 * math.pi) - math.pi
    return math.pi if folded == -math.pi else folded


def rot_z(angle: float) -> np.ndarray:
    """Return a rotation about z by `angle` [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    """Return a rotation about `axis` by `angle` [rad]."""
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def quat_wxyz_to_mat(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the rotation of quaternion (w, x, y, z), normalized first."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- files
def library_path(kind: str, folder: str | Path | None = None) -> Path:
    """Return the selected block's JSON path in `folder`, defaulting to the bundled directory."""
    return Path(LIBRARY_DIR if folder is None else folder) / f"{kind}.json"


def load_library(kind: str, folder: str | Path | None = None) -> GraspLibrary:
    """Load and validate a block library from `folder`, defaulting to the bundled directory.

    Raise FileNotFoundError if absent or ValueError for invalid content or a mismatched block.
    """
    path = library_path(kind, folder)
    if not path.is_file():
        raise FileNotFoundError(f"no grasp library for {kind!r}: {path} (see tools/graspgenx/README.md)")
    library = GraspLibrary.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if library.kind != kind:
        raise ValueError(f"{path} holds the grasp library of {library.kind!r}, not of {kind!r}")
    return library


def save_library(library: GraspLibrary, path: str | Path) -> Path:
    """Write JSON with indent 1 and six-decimal floats, then return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rounded(library.to_dict()), indent=1) + "\n", encoding="utf-8")
    return path


def _rounded(value: Any) -> Any:
    """Round floats recursively to six decimals and normalize negative zero."""
    if isinstance(value, float):
        return round(value, 6) + 0.0
    if isinstance(value, Mapping):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


# ---------------------------------------------------------------- the block
def block_width(spec: BlockSpec, closing: Sequence[float] | np.ndarray) -> float:
    """Return block width along unit `closing` [m, block frame].

    Use the outermost support points across all collision parts.
    """
    c = np.asarray(closing, float)
    c = c / np.linalg.norm(c)
    lo, hi = math.inf, -math.inf
    for part in spec.parts:
        R = quat_wxyz_to_mat(part.quat)
        local = R.T @ c
        if part.kind == "box":
            support = float(np.abs(local) @ np.asarray(part.size, float))
        elif part.kind == "cylinder":
            r, half_length = part.size[0], part.size[1]
            along = abs(float(local[2]))
            support = half_length * along + r * math.sqrt(max(0.0, 1.0 - along * along))
        else:
            raise ValueError(f"{spec.name}: no support function for a {part.kind!r} part")
        centre = float(c @ np.asarray(part.pos, float))
        lo, hi = min(lo, centre - support), max(hi, centre + support)
    return hi - lo


def roll_axis(spec: BlockSpec) -> np.ndarray | None:
    """Return the body-frame symmetry axis for a single cylinder part, or None otherwise."""
    if len(spec.parts) != 1 or spec.parts[0].kind != "cylinder":
        return None
    return quat_wxyz_to_mat(spec.parts[0].quat) @ np.array([0.0, 0.0, 1.0])


def write_block_obj(kind: str, path: str | Path) -> Path:
    """Write an ASCII OBJ mesh in the selected block's body frame [m].

    Faces wind counter-clockwise from outside. The file contains vertices and triangles
    without normals or materials.
    """
    spec = block(kind)
    if len(spec.parts) != 1:
        raise ValueError(f"{kind}: the mesh writer knows blocks of one part, this one has {len(spec.parts)}")
    part = spec.parts[0]
    R = quat_wxyz_to_mat(part.quat)
    if part.kind == "box":
        h = np.asarray(part.size, float)
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
        # Corner index = 4 * (x > 0) + 2 * (y > 0) + (z > 0); two triangles per face.
        faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                 (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    elif part.kind == "cylinder":
        r, half_length = part.size[0], part.size[1]
        n = CYLINDER_SEGMENTS
        ring = [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
        # The part's frame: its axis is z. Rims at -L (0..n-1) and +L (n..2n-1), then the two end centres.
        local = np.array([(x, y, z) for z in (-half_length, half_length) for x, y in ring]
                         + [(0.0, 0.0, -half_length), (0.0, 0.0, half_length)], float)
        bottom, top = 2 * n, 2 * n + 1
        faces = []
        for k in range(n):
            k1 = (k + 1) % n
            faces += [(k, k1, n + k1), (k, n + k1, n + k), (bottom, k1, k), (top, n + k, n + k1)]
    else:
        raise ValueError(f"{kind}: no mesh for a {part.kind!r} part")
    vertices = local @ R.T + np.asarray(part.pos, float)
    centre = vertices.mean(axis=0)
    oriented = []
    for i, j, k in faces:
        normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
        outward = float(normal @ ((vertices[i] + vertices[j] + vertices[k]) / 3 - centre)) > 0
        oriented.append((i, j, k) if outward else (i, k, j))
    size = " x ".join(f"{2000 * h:.1f}" for h in spec.half_size)
    lines = [f"# soarm-lab block {kind}: {size} mm"]
    lines += [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
    lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in oriented]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# ---------------------------------------------------------------- GraspGenX
def read_gripper_config(path: str | Path) -> GripperConfig:
    """Read the gripper configuration fields used for conversion.

    Raise FileNotFoundError if absent or ValueError if `base_rotation` is not a proper 4x4 rotation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gripper config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    B = np.asarray(data["base_rotation"], float)
    if B.shape != (4, 4) or abs(float(np.linalg.det(B[:3, :3])) - 1.0) > 1e-6 or not np.allclose(
            B[:3, :3] @ B[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError(f"{path}: base_rotation is not a proper rotation")
    return GripperConfig(base_rotation=B, pad_point=np.asarray(data["sweep_volume"]["offset"], float),
                         fingertip=np.asarray(data["fingertip"], float),
                         open={str(k): float(v) for k, v in data["open"].items()},
                         close={str(k): float(v) for k, v in data["close"].items()},
                         type=str(data["type"]), symmetric=bool(data["symmetric"]))


def graspgenx_frame(kin: RobotKinematics, base_rotation: np.ndarray) -> np.ndarray:
    """Return T_aligned_from_tcp (4x4) for GraspGenX conversion.

    `base_rotation` aligns the URDF root at `hand_body`. This transform is independent of jaw opening.
    """
    spec = kin.spec
    hand = kin.body(spec.hand_body, spec.home)
    return np.asarray(base_rotation, float) @ inverse(hand) @ kin.tcp(spec.home)


def convert_isaac_grasps(data: Mapping[str, Any], kind: str, *, frame: np.ndarray, pad_point: np.ndarray,
                         generator: Mapping[str, str]) -> GraspLibrary:
    """Convert parsed `isaac_grasp` YAML to a block library.

    Transform the pad point by each grasp pose. The TCP pose is T_grasp @ `frame`, whose x
    and z columns define approach and closing. Derive width from block support and use
    confidence as score. Drop pad points beyond `INSIDE_TOL_M` and record their count in
    `generator["dropped"]`. Sort by score, preserving file order for ties.
    """
    fmt = data.get("format") if isinstance(data, Mapping) else None
    if fmt != "isaac_grasp":
        raise ValueError(f"not an isaac_grasp file: format {fmt!r}")
    entries = data.get("grasps") or {}
    if not isinstance(entries, Mapping):
        raise ValueError("isaac_grasp: grasps is not a mapping")
    spec = block(kind)
    half = np.asarray(spec.half_size, float)
    frame, pad = np.asarray(frame, float), np.asarray(pad_point, float)
    kept: list[LibraryGrasp] = []
    for key, entry in entries.items():
        try:
            orientation = entry["orientation"]
            q = np.array([float(orientation["w"]), *(float(v) for v in orientation["xyz"])])
            t = np.asarray(entry["position"], float)
            score = float(entry["confidence"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"isaac_grasp {key}: needs confidence, position and orientation w, "
                             "xyz") from None
        norm = float(np.linalg.norm(q))
        if q.shape != (4,) or not np.all(np.isfinite(q)) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"isaac_grasp {key}: the orientation is not a unit quaternion (norm {norm:.4f})")
        if t.shape != (3,) or not np.all(np.isfinite(t)) or not math.isfinite(score):
            raise ValueError(f"isaac_grasp {key}: position or confidence is not finite")
        R = quat_wxyz_to_mat(q)
        point = R @ pad + t
        if np.any(np.abs(point) > half + INSIDE_TOL_M):
            continue
        tcp = pose(R, t) @ frame
        approach, closing = tcp[:3, 0], tcp[:3, 2]
        kept.append(LibraryGrasp(_triple(point), _triple(approach), _triple(closing),
                                 block_width(spec, closing), score))
    kept.sort(key=lambda g: -g.score)
    made = {str(k): str(v) for k, v in generator.items()}
    made["dropped"] = f"{len(entries) - len(kept)} of {len(entries)}"
    return GraspLibrary(kind, spec.half_size, made, tuple(kept))


# ---------------------------------------------------------------- at run time
def compose(library: GraspLibrary, rotation: np.ndarray, yaw_period: float,
            axis: Sequence[float] | np.ndarray | None = None) -> list[ComposedGrasp]:
    """Compose a library with `rotation` (base ← block), yaw symmetries and swapped jaws.

    For a cylinder `axis`, roll approaches toward down when they have a transverse component.
    End grasps along the axis keep their roll. Expand every grasp over `yaw_period` and both
    closing headings in library, fold and flip order. Filtering and deduplication belong to `select`.
    """
    R = np.asarray(rotation, float)
    folds = max(1, round(2 * math.pi / yaw_period))
    top = library.half_size[2]
    down = np.array([0.0, 0.0, -1.0])
    u = None if axis is None else np.asarray(axis, float) / np.linalg.norm(axis)
    out: list[ComposedGrasp] = []
    for i, grasp in enumerate(library.grasps):
        p = np.asarray(grasp.position, float)
        a, c = grasp.axes()
        roll = 0.0
        if u is not None:
            across = a - float(a @ u) * u
            d = down - float(down @ u) * u
            if np.linalg.norm(across) >= 1e-9 and np.linalg.norm(d) >= 1e-9:
                across, d = across / np.linalg.norm(across), d / np.linalg.norm(d)
                roll = math.atan2(float(u @ np.cross(across, d)), float(across @ d))
                turn = rot_axis(u, roll)
                p, a, c = turn @ p, turn @ a, turn @ c
        for k in range(folds):
            Rk = R @ rot_z(k * yaw_period)
            pk, ak, ck = Rk @ p, Rk @ a, Rk @ c
            tilt = math.acos(float(np.clip(-ak[2], -1.0, 1.0)))
            psi = math.atan2(float(ck[1]), float(ck[0]))
            offset = (float(pk[0]), float(pk[1]))
            depth = top - float(pk[2])
            for flipped in (False, True):
                out.append(ComposedGrasp(i, wrap(psi + math.pi if flipped else psi), offset, depth,
                                         grasp.width_m, grasp.score, tilt, flipped, k, roll))
    return out


def select(composed: Sequence[ComposedGrasp], *, psi_now: float, max_tilt_rad: float, min_score: float,
           gap_min_m: float, gap_max_m: float, open_margin_m: float, squeeze_m: float,
           half_size: Sequence[float]) -> tuple[list[ComposedGrasp], dict[str, int]]:
    """Return executable candidates and rejection counts in filter order.

    Check tilt, score, jaw travel, depth and horizontal extent. Count only the first failure
    per candidate. Keep the first instance of each executed heading, offset, depth and width.
    Duplicates add no rejection count. Rank survivors by `SCORE_TIE` bins, wrist turn from
    `psi_now`, library index and unflipped before flipped.
    """
    half = [float(h) for h in half_size]
    reach = math.hypot(half[0], half[1]) + INSIDE_TOL_M
    counts = dict.fromkeys(REJECTIONS, 0)
    seen: set[tuple[float, ...]] = set()
    kept: list[ComposedGrasp] = []
    for g in composed:
        if g.tilt_rad > max_tilt_rad:
            counts["pochylenie"] += 1
        elif g.score < min_score:
            counts["wynik"] += 1
        elif g.width_m + 2 * open_margin_m > gap_max_m or g.width_m - squeeze_m < gap_min_m:
            counts["szczeka"] += 1
        elif not 0.0 <= g.depth_m <= 2 * half[2]:
            counts["glebokosc"] += 1
        elif math.hypot(*g.offset_xy) > reach:
            counts["poza blokiem"] += 1
        else:
            # The heading as its cosine and sine: -pi and pi are one heading.
            key = tuple(round(v, KEY_DECIMALS) + 0.0 for v in (math.cos(g.psi), math.sin(g.psi), *g.offset_xy,
                                                               g.depth_m, g.width_m))
            if key not in seen:
                seen.add(key)
                kept.append(g)
    kept.sort(key=lambda g: (-math.floor(g.score / SCORE_TIE + 1e-9), abs(wrap(g.psi - psi_now)), g.index,
                             g.flipped))
    return kept, {name: n for name, n in counts.items() if n}


def rejections_text(counts: Mapping[str, int], order: Sequence[str] = REJECTIONS) -> str:
    """Return nonzero rejection counts in `order` for display."""
    return ", ".join(f"{name} {counts[name]}" for name in order if counts.get(name))
