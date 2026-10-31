"""Grasp libraries: the grasps GraspGenX computed offline for each wooden block, in the block's own frame.

    write_block_obj("cube", "workspace/grasps/meshes/cube.obj")     # the block as GraspGenX's input
    lib = convert_isaac_grasps(yaml.safe_load(text), "cube", frame=graspgenx_frame(kin, cfg.base_rotation),
                               pad_point=cfg.pad_point, generator={"name": "graspgenx"})
    save_library(lib, library_path("cube", folder))
    composed = compose(load_library("cube"), pose.rotation, spec.yaw_period)   # with the twin's pose
    kept, counts = select(composed, psi_now=..., max_tilt_rad=..., ...)      # what the jaw can do, ranked

A library is one JSON file per block kind, `<kind>.json`; the ones bundled with the lab are under
`soarm_lab/grasps/` (`LIBRARY_DIR`). Each grasp is three vectors in the BLOCK FRAME of `rl.objects` (the
block's centre, x its long side, z up as it lies on the table): `position`, the PAD POINT, and the unit
`approach` and `closing` directions of the gripperframe site (its x and z). The pad point is the centre of
the open sweep box of the gripper description (`sweep_volume.offset` of tools/graspgenx/so101/config.json),
the slab GraspGenX puts between the pads, 27.4 mm behind the fixed fingertip. The `grasp` strategy puts its
fingertips' midpoint there.

GraspGenX's grasp is the pose of its aligned gripper frame (+Z approach, +X closing, origin at the URDF
root) in the mesh frame, which is the block frame. One proper rotation maps that frame to the TCP's:
`graspgenx_frame` gives T_aligned_from_tcp, on the SO-101 [[0, 0, 1, 0.0120], [0, -1, 0, 0.0002],
[1, 0, 0, 0.0981]], and a grasp's TCP pose in the block frame is T_grasp @ T_aligned_from_tcp.

GraspGenX never runs here (it needs CUDA, torch<2.7 and numpy 1.26): the lab reads only the files it wrote,
a data boundary like `twin export`. tools/graspgenx/README.md has the run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .kinematics import RobotKinematics, inverse, pose
from .rl.objects import KINDS, MEASURED, ObjectSpec, object_spec

#: The `format` and `version` of a library file.
FORMAT = "soarm-lab-grasps"
VERSION = 1
#: The libraries bundled with the lab.
LIBRARY_DIR = Path(__file__).resolve().parent / "grasps"
#: What the library's `frame` says.
FRAME = ("block frame of rl/objects.py (centre, x the long side, z up); position = the pad point, the centre "
         "of the gripper description's open sweep box, where the strategy puts its fingertip midpoint; "
         "approach and closing are unit vectors in the block frame (gripperframe x and z)")
#: GraspGenX's approach and closing axes in its aligned gripper frame.
GRASPGENX_APPROACH = (0.0, 0.0, 1.0)
GRASPGENX_CLOSING = (1.0, 0.0, 0.0)
#: Scores within one step of this rank as equal; the smaller wrist turn decides between them [0..1].
SCORE_TIE = 0.05
#: How far a stored unit vector may be from length 1 and two stored axes from perpendicular: what six-decimal
#: JSON can carry (rounding moves |a . c| by up to 1.3e-6).
UNIT_TOL = 1e-5
#: A pad point may lie this far outside the block's bounding box [m].
INSIDE_TOL_M = 0.001
#: Two composed grasps are one when their heading (its cosine and sine), offset, depth and width [m] agree
#: to this many decimals.
KEY_DECIMALS = 5
#: Rim vertices of a cylinder's OBJ at each end.
CYLINDER_SEGMENTS = 64
#: Why `select` turned a composed grasp down, in the order it checks: the approach leans too far from
#: vertical, the score is too low, the jaw cannot open or close to the width, the pad point is above the
#: block's top or below its bottom, or outside the block.
REJECTIONS = ("pochylenie", "wynik", "szczeka", "glebokosc", "poza blokiem")


@dataclass(frozen=True)
class LibraryGrasp:
    """One grasp of a library, in the block frame."""

    #: The pad point [m].
    position: tuple[float, float, float]
    #: The gripperframe's approach (x) and closing (z) directions, unit vectors.
    approach: tuple[float, float, float]
    closing: tuple[float, float, float]
    #: The block's width along `closing` [m]: what the jaw closes on.
    width_m: float
    #: The generator's score, higher is better.
    score: float

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """The approach and the closing direction as an exact orthonormal pair: the approach normalised, the
        closing with its approach component removed, normalised. The file's rounding never reaches them."""
        a = np.asarray(self.approach, float)
        a = a / np.linalg.norm(a)
        c = np.asarray(self.closing, float)
        c = c - float(c @ a) * a
        return a, c / np.linalg.norm(c)

    def rotation(self) -> np.ndarray:
        """The gripperframe's rotation in the block frame: columns approach, closing x approach, closing."""
        a, c = self.axes()
        return np.column_stack([a, np.cross(c, a), c])


@dataclass(frozen=True)
class GraspLibrary:
    """The grasps of one block kind, sorted by score, best first."""

    kind: str
    #: The block's half sizes the library was made for [m]: `object_spec(kind).half_size` then.
    half_size: tuple[float, float, float]
    #: Who made it: name ("graspgenx", or "hand" for a hand-written file), version, commit, gripper,
    #: checkpoint, date, note, and how many grasps the converter dropped ("3 of 50").
    generator: dict[str, str]
    grasps: tuple[LibraryGrasp, ...]

    def to_dict(self) -> dict[str, Any]:
        """The library as its JSON file holds it, keys in the file's order, at full precision."""
        return {"format": FORMAT, "version": VERSION, "kind": self.kind, "half_size": list(self.half_size),
                "frame": FRAME, "generator": dict(self.generator),
                "grasps": [{"position": list(g.position), "approach": list(g.approach),
                            "closing": list(g.closing), "width_m": g.width_m, "score": g.score}
                           for g in self.grasps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraspLibrary:
        """A library from its JSON content, checked; ValueError names what is wrong (and the grasp)."""
        if not isinstance(data, Mapping):
            raise ValueError("a grasp library is a JSON object")
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError(f"not a grasp library of version {VERSION}: format {data.get('format')!r}, "
                             f"version {data.get('version')!r}")
        kind = data.get("kind")
        if kind not in KINDS:
            raise ValueError(f"grasp library for an unknown kind {kind!r}; known: {', '.join(KINDS)}")
        spec = object_spec(kind)
        half = _vector(data.get("half_size"), f"{kind} library: half_size")
        if not np.allclose(half, spec.half_size, rtol=0.0, atol=1e-9):
            made, table = (" x ".join(f"{2000 * h:.1f}" for h in sizes) for sizes in (half, spec.half_size))
            raise ValueError(f"{kind} library was made for a {made} mm block, the table says {table} mm: "
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
    """Three finite numbers, else ValueError naming `what`."""
    try:
        v = np.asarray(value, float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not three numbers: {value!r}") from None
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{what} is not three finite numbers: {value!r}")
    return v


def _grasp(entry: Any, spec: ObjectSpec, where: str) -> LibraryGrasp:
    """One grasp of a library file, checked against the block `spec`."""
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
    """What the lab reads of a GraspGenX gripper description's config.json."""

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
    """A library grasp composed with the block's pose, as the top-down strategy would execute it."""

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
    #: The jaws swapped (the heading turned half way): the same straddle, the other jaw pushing.
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
    """The rotation by `angle` [rad] about z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    """The rotation by `angle` [rad] about `axis` (Rodrigues)."""
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def quat_wxyz_to_mat(q: Sequence[float] | np.ndarray) -> np.ndarray:
    """The rotation of the quaternion `q` = (w, x, y, z), normalised first (the formula of `rl.objects`)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- files
def library_path(kind: str, folder: str | Path | None = None) -> Path:
    """Where the library of `kind` lies: `<folder>/<kind>.json`, the bundled folder by default."""
    return Path(LIBRARY_DIR if folder is None else folder) / f"{kind}.json"


def load_library(kind: str, folder: str | Path | None = None) -> GraspLibrary:
    """The library of `kind` from `folder` (the bundled one by default), checked.

    FileNotFoundError without one, ValueError when it is not a valid library of `kind` (`from_dict`).
    """
    path = library_path(kind, folder)
    if not path.is_file():
        raise FileNotFoundError(f"no grasp library for {kind!r}: {path} (see tools/graspgenx/README.md)")
    library = GraspLibrary.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if library.kind != kind:
        raise ValueError(f"{path} holds the grasp library of {library.kind!r}, not of {kind!r}")
    return library


def save_library(library: GraspLibrary, path: str | Path) -> Path:
    """Writes `library` as JSON (indent 1, floats rounded to 6 decimals) and returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rounded(library.to_dict()), indent=1) + "\n", encoding="utf-8")
    return path


def _rounded(value: Any) -> Any:
    """`value` with every float rounded to 6 decimals (and -0.0 written as 0.0)."""
    if isinstance(value, float):
        return round(value, 6) + 0.0
    if isinstance(value, Mapping):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


# ---------------------------------------------------------------- the block
def block_width(spec: ObjectSpec, closing: Sequence[float] | np.ndarray) -> float:
    """The block's width along the unit direction `closing` (block frame) [m]: its support both ways.

    A box part spans 2 sum_i |c_i| h_i; a cylinder part (radius r, half length L, axis u) 2 (L |c . u| + r
    sqrt(1 - (c . u)^2)). Several parts span from the lowest to the highest support of any of them.
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
            raise ValueError(f"{spec.kind}: no support function for a {part.kind!r} part")
        centre = float(c @ np.asarray(part.pos, float))
        lo, hi = min(lo, centre - support), max(hi, centre + support)
    return hi - lo


def roll_axis(spec: ObjectSpec) -> np.ndarray | None:
    """The axis of the block's continuous symmetry in its frame: a cylinder's axis, (1, 0, 0) for the set's
    cylinder lying along x; None for a box or a block of several parts."""
    if len(spec.parts) != 1 or spec.parts[0].kind != "cylinder":
        return None
    return quat_wxyz_to_mat(spec.parts[0].quat) @ np.array([0.0, 0.0, 1.0])


def write_block_obj(kind: str, path: str | Path) -> Path:
    """The block `kind` as an ASCII OBJ in its block frame, metres: GraspGenX's input mesh.

    A box is its 8 corners and 12 triangles; a cylinder `CYLINDER_SEGMENTS` rim vertices at each end (the
    extreme ones at exactly its radius), the two end centres, 128 side and 2 x 64 cap triangles. Every
    triangle runs counter-clockwise seen from outside. No normals, no materials.
    """
    spec = object_spec(kind)
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
    measured = "measured" if MEASURED else "placeholder: MEASURED = False"
    lines = [f"# soarm-lab block {kind}: {size} mm ({measured})"]
    lines += [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
    lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in oriented]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# ---------------------------------------------------------------- GraspGenX
def read_gripper_config(path: str | Path) -> GripperConfig:
    """The parts of a GraspGenX gripper description's config.json the lab uses.

    FileNotFoundError without the file; ValueError when `base_rotation` is not a proper 4x4 rotation.
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
    """T_aligned_from_tcp (4x4): the TCP site in GraspGenX's aligned gripper frame, the URDF root (the
    robot's `hand_body`) turned by `base_rotation`. Neither depends on the jaw: both ride on the hand."""
    spec = kin.spec
    hand = kin.body(spec.hand_body, spec.home)
    return np.asarray(base_rotation, float) @ inverse(hand) @ kin.tcp(spec.home)


def convert_isaac_grasps(data: Mapping[str, Any], kind: str, *, frame: np.ndarray, pad_point: np.ndarray,
                         generator: Mapping[str, str]) -> GraspLibrary:
    """GraspGenX's `isaac_grasp` output (parsed YAML) for the block `kind` as a library.

    Per grasp: T_grasp from its position and its (w, xyz) quaternion; the pad point T_grasp @ `pad_point`;
    the TCP pose T_grasp @ `frame` (`graspgenx_frame`), whose x and z columns are the approach and the
    closing direction; the block's width along the closing direction; the confidence as the score. A grasp
    whose pad point lies outside the block by more than `INSIDE_TOL_M` is dropped and counted in
    `generator["dropped"]` ("3 of 50"). The rest are sorted by score, best first, ties in file order.
    """
    fmt = data.get("format") if isinstance(data, Mapping) else None
    if fmt != "isaac_grasp":
        raise ValueError(f"not an isaac_grasp file: format {fmt!r}")
    entries = data.get("grasps") or {}
    if not isinstance(entries, Mapping):
        raise ValueError("isaac_grasp: grasps is not a mapping")
    spec = object_spec(kind)
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
    """Every grasp of `library` on a block with pose `rotation` (base <- block), in every symmetric yaw and
    with the jaws both ways round: len(grasps) x folds x 2 entries, in library, fold, flip order.

    1. With `axis` (a cylinder's, `roll_axis`), a grasp whose approach has a part across the axis is first
       rolled about it until that part points straight down in the block frame: a turn about the cylinder's
       own axis is a symmetry of the block, so it is the same grasp from above. An end grasp along the axis
       is not rolled.
    2. The detector cannot tell the block's symmetric yaws apart (`yaw_period`), and the generator's samples
       need not be symmetric, so every grasp is folded over them: the composed set is the same whichever
       symmetric yaw the pose reports.
    3. Each comes twice, the heading and its half turn (`flipped`): the jaws swapped.

    Nothing is de-duplicated or filtered here (`select` does), so a rejected entry never shadows a kept one.
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
    """The composed grasps a top-down straddle can execute, without duplicates, best first; and how many
    each filter turned down.

    The filters, in `REJECTIONS` order, the first failing one counted: the approach leans more than
    `max_tilt_rad` from straight down; the score lies below `min_score`; the jaw cannot open to the width
    plus two `open_margin_m` (`gap_max_m`) or close to it minus `squeeze_m` (`gap_min_m`); the pad point lies
    above the block's top or below its bottom; it lies outside the block's horizontal extent. Then the
    survivors with the same executed grasp (heading, offset, depth and width to `KEY_DECIMALS`) are one, the
    first kept: compose keeps library order and libraries are score-sorted, so it is the best-scored.
    Duplicates are not counted. The kept grasps rank by score in steps of `SCORE_TIE`, then by the smaller
    turn from the heading `psi_now`, then by library order and unflipped first.
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
    """The non-zero counts in `order`, for the panel: "pochylenie 2, szczeka 1"."""
    return ", ".join(f"{name} {counts[name]}" for name in order if counts.get(name))
