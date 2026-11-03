"""World state for the block planner, built from `Observation.objects`.

Positions are in the robot's base frame [m]: x away from the robot, y to its left, z up from the table.
Yaw is the heading of a block's body x axis (its long side), counter-clockwise from +x seen from above,
reduced by the block's yaw symmetry. An identity rotation means the perception knows no orientation.
Labels are the twin's keys; identical blocks are numbered ("red cube", "red cube 2"). A block's kind is
its label's shape word, looked up in the Classic World catalog.
The build area lists, per stack height, a box where a `PlaceCheck` lets the arm set every block down,
even when the blocks below stand `HEIGHT_MARGIN_M` higher than their nominal size.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from ..blocks import BlockSpec, bottom_height
from ..blocks.classic_world import BLOCKS
from ..robot import Desk
from .placing import PlaceCheck, arm_reach

#: Pose source (`ObjectPose.source`) of an object the gripper holds.
IN_GRIPPER_SOURCE = "w dloni"
#: Supports named in `WorldObject.resting_on` besides block labels.
TABLE = "table"
GRIPPER = "gripper"
#: A block rests on a surface within this height of its bottom [m].
SUPPORT_TOL_M = 0.01
#: Footprints must overlap by more than this for one block to carry another [m].
SUPPORT_OVERLAP_M = 0.002
#: Half side of the square footprint and half height assumed for an object of unknown kind [m].
UNKNOWN_HALF_M = 0.015
#: Shape words that name a catalog kind other than by its name.
KIND_WORDS = {"brick": "cuboid"}
#: Stack heights the build area covers by default.
MAX_LEVELS = 3
#: The build area counts levels in this block, the tallest kind that carries others.
LEVEL_BLOCK = BLOCKS["cube"]
#: The build area still lets the arm set a block down this much above a level's nominal height, for the
#: blocks' size tolerance and the cameras' height error [m].
HEIGHT_MARGIN_M = 0.003
#: Grid step of the build area and of `best_spot` [m].
AREA_STEP_M = 0.01

Point = tuple[float, float]
Polygon = tuple[Point, ...]

_DESK = Desk()


@dataclass(frozen=True)
class Reach:
    """The table area the arm places blocks in: a band of distance and bearing around the base axis."""

    #: Distance from the base axis [m], (min, max).
    radius_m: tuple[float, float] = _DESK.cube_radius
    #: Bearing from +x toward +y [rad], (min, max).
    bearing_rad: tuple[float, float] = _DESK.cube_bearing

    @classmethod
    def from_desk(cls, desk: Desk) -> Reach:
        """The band where the robot description places cubes (`RobotSpec.desk`)."""
        (r0, r1), (b0, b1) = desk.cube_radius, desk.cube_bearing
        return cls((float(r0), float(r1)), (float(b0), float(b1)))

    def contains(self, x: float, y: float) -> bool:
        r, bearing = math.hypot(x, y), math.atan2(y, x)
        return (self.radius_m[0] <= r <= self.radius_m[1]
                and self.bearing_rad[0] <= bearing <= self.bearing_rad[1])

    def to_json(self) -> dict[str, Any]:
        return {"radius_m": [_m(v) for v in self.radius_m],
                "bearing_deg": [round(math.degrees(v)) for v in self.bearing_rad]}


@dataclass(frozen=True)
class WorldObject:
    """One object of the world state, with its geometry worked out from the catalog."""

    label: str
    #: Catalog name (`classic_world.BLOCKS`), or None for an unknown kind.
    kind: str | None
    #: Centre [m, base frame].
    x_m: float
    y_m: float
    z_m: float
    #: Heading of the body x axis [rad], reduced by the yaw symmetry; None when unknown.
    yaw_rad: float | None
    source: str
    age_s: float
    #: Heights of the lowest and highest point above the table [m].
    bottom_m: float
    top_m: float
    #: Outline seen from above, counter-clockwise [m, base frame].
    footprint: Polygon
    #: What carries it: (TABLE,), (GRIPPER,), block labels, or () when unknown.
    resting_on: tuple[str, ...] = ()

    @property
    def spec(self) -> BlockSpec | None:
        return BLOCKS.get(self.kind) if self.kind else None

    @property
    def in_gripper(self) -> bool:
        return self.source == IN_GRIPPER_SOURCE

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m])

    def to_json(self) -> dict[str, Any]:
        spec = self.spec
        if not self.resting_on:
            resting: str | list[str] | None = None
        elif len(self.resting_on) == 1:
            resting = self.resting_on[0]
        else:
            resting = list(self.resting_on)
        return {
            "kind": self.kind,
            "size_mm": None if spec is None else size_mm(spec),
            "x_m": _m(self.x_m), "y_m": _m(self.y_m), "z_m": _m(self.z_m),
            "yaw_deg": None if self.yaw_rad is None else round(math.degrees(self.yaw_rad)),
            "resting_on": resting,
            "source": self.source,
            "age_s": round(self.age_s, 1) + 0.0 if math.isfinite(self.age_s) else None,
        }


@dataclass(frozen=True)
class Box:
    """Where structures of one height can be built: an x/y box on the table [m, base frame]."""

    levels: int
    x_m: tuple[float, float]
    y_m: tuple[float, float]

    def contains(self, x: float, y: float) -> bool:
        return self.x_m[0] <= x <= self.x_m[1] and self.y_m[0] <= y <= self.y_m[1]

    def span(self) -> str:
        return f"x {self.x_m[0]:.2f}..{self.x_m[1]:.2f} m, y {self.y_m[0]:.2f}..{self.y_m[1]:.2f} m"

    def text(self) -> str:
        return f"a structure {self.levels} high fits in {self.span()}"

    def to_json(self) -> dict[str, Any]:
        return {"blocks_high": self.levels, "x_m": [_m(v) for v in self.x_m],
                "y_m": [_m(v) for v in self.y_m]}


@dataclass(frozen=True)
class World:
    """Objects by label, the reachable table area and where the arm can set blocks down."""

    objects: Mapping[str, WorldObject] = field(default_factory=dict)
    reach: Reach = field(default_factory=Reach)
    #: Whether the arm can set a block down at a place and height; the arm model by default.
    place_check: PlaceCheck = field(default_factory=arm_reach, compare=False, repr=False)
    #: The build area covers structures 1 to this many blocks high.
    levels: int = MAX_LEVELS

    def get(self, label: str) -> WorldObject | None:
        return self.objects.get(label_key(label))

    def above(self, label: str) -> list[str]:
        """Labels of the objects resting on `label`."""
        key = label_key(label)
        return [o.label for o in self.objects.values() if key in o.resting_on]

    def level(self, label: str) -> int:
        """1 for a block on the table, 2 for one on such a block, and so on; 1 when the support is unknown."""
        return self._level(label_key(label), ())

    def _level(self, key: str, seen: tuple[str, ...]) -> int:
        obj = self.objects.get(key)
        if obj is None or key in seen:
            return 1
        below = [s for s in obj.resting_on if s in self.objects]
        return 1 + max((self._level(s, (*seen, key)) for s in below), default=0)

    @cached_property
    def build_area(self) -> tuple[Box | None, ...]:
        """For 1 to `levels` blocks high, the largest box where the arm sets blocks down at every level
        up to that one; None where there is none."""
        return find_build_area(self.place_check, self.reach, self.levels)

    def box(self, levels: int) -> Box | None:
        """The build area's box for a structure `levels` blocks high."""
        return self.build_area[levels - 1] if 1 <= levels <= len(self.build_area) else None

    def to_json(self) -> dict[str, Any]:
        """The state for the model and the logs: numbers rounded to mm and whole degrees.

        Objects are keyed by label, numbered labels in number order.
        """
        area = [box.to_json() if box else {"blocks_high": n, "x_m": None, "y_m": None}
                for n, box in enumerate(self.build_area, start=1)]
        return {"reach": self.reach.to_json(), "build_area": area, "catalog": catalog_json(),
                "objects": {k: self.objects[k].to_json() for k in sorted(self.objects, key=_natural)}}


def world_state(objects: Mapping[str, ObjectPose], reach: Reach | None = None,
                place_check: PlaceCheck | None = None, levels: int = MAX_LEVELS) -> World:
    """Build the world state from `Observation.objects`.

    Labels are lower-cased with white space collapsed. Objects without a finite position are left out.
    `place_check` defaults to the arm model's (`arm_reach`).
    """
    shaped: dict[str, WorldObject] = {}
    for key, pose in objects.items():
        label = label_key(str(key))
        position = np.asarray(pose.position, float)
        if not label or position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        age = float(pose.age_s)
        shaped[label] = _shape(label, position, _rotation(pose.rotation), str(pose.source or ""),
                               age if math.isfinite(age) else math.inf)
    placed = {k: _with_supports(o, shaped.values()) for k, o in shaped.items()}
    return World(placed, reach or Reach(), place_check or arm_reach(), levels)


def label_key(label: str) -> str:
    """A label as the twin keys `obs.objects`: lower case, white space collapsed."""
    return " ".join(label.split()).lower()


def kind_of(label: str) -> str | None:
    """The catalog kind a label names by its shape word, the last word naming one: "red cube 2" -> "cube"."""
    for word in reversed(label_key(label).split()):
        kind = word if word in BLOCKS else KIND_WORDS.get(word)
        if kind:
            return kind
    return None


def size_mm(spec: BlockSpec) -> list[int]:
    return [round(2000.0 * h) for h in spec.half_size]


def catalog_json() -> dict[str, Any]:
    return {name: {"size_mm": size_mm(spec), "yaw_period_deg": round(math.degrees(spec.yaw_period))}
            for name, spec in BLOCKS.items()}


def reduce_yaw(yaw: float, period: float) -> float:
    """`yaw` [rad] moved by whole periods into (-period / 2, period / 2]."""
    return period / 2 - (period / 2 - yaw) % period


def rectangle(x: float, y: float, yaw: float, half: Sequence[float]) -> Polygon:
    """The footprint of a block lying flat: half sides `half[0]` along the heading `yaw`, `half[1]` across."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = ((half[0], half[1]), (-half[0], half[1]), (-half[0], -half[1]), (half[0], -half[1]))
    return tuple((x + c * u - s * v, y + s * u + c * v) for u, v in corners)


def overlap(p: Polygon, q: Polygon) -> float:
    """How far two convex outlines overlap [m]; negative for the gap between them along a separating axis."""
    P, Q = np.asarray(p, float), np.asarray(q, float)
    depth = math.inf
    for poly in (P, Q):
        edges = np.roll(poly, -1, axis=0) - poly
        for ex, ey in edges:
            length = math.hypot(ex, ey)
            if length < 1e-9:
                continue
            axis = np.array([-ey, ex]) / length
            a, b = P @ axis, Q @ axis
            depth = min(depth, float(min(a.max(), b.max()) - max(a.min(), b.min())))
    return depth


def extent(poly: Polygon, centre: np.ndarray, direction: np.ndarray) -> float:
    """How far the outline reaches from `centre` along the unit `direction` [m]."""
    return float(np.max((np.asarray(poly, float) - centre) @ direction))


def level_bottom(levels: int, margin_m: float = 0.0) -> float:
    """Height of the bottom of a structure's top block, `levels` cubes high, plus `margin_m` when that
    block does not stand on the table [m]."""
    return (levels - 1) * 2 * LEVEL_BLOCK.half_size[2] + (margin_m if levels > 1 else 0.0)


def find_build_area(check: PlaceCheck, reach: Reach, levels: int,
                    margin_m: float = HEIGHT_MARGIN_M) -> tuple[Box | None, ...]:
    """For 1 to `levels` blocks high, the largest x/y box inside `reach` where `check` lets the arm set a
    block down at every level up to that one, `margin_m` above the level's nominal height, on a grid of
    `AREA_STEP_M`; None where there is none.

    Each box holds the box of the next height, so a lower structure may stand wherever a taller one may.
    """
    xs, ys = _area_grid(reach)
    ok = np.array([[reach.contains(x, y) for y in ys] for x in xs])
    allowed = []
    for n in range(1, levels + 1):
        z = level_bottom(n, margin_m)
        ok = ok & np.array([[bool(c) and check(x, y, z, LEVEL_BLOCK) for y, c in zip(ys, row, strict=True)]
                            for x, row in zip(xs, ok, strict=True)])
        allowed.append(ok)
    boxes: list[Box | None] = []
    inner = None
    for n in range(levels, 0, -1):
        inner = _largest_box(allowed[n - 1], inner) or inner
        boxes.append(None if inner is None else
                     Box(n, (xs[inner[0]], xs[inner[1]]), (ys[inner[2]], ys[inner[3]])))
    return tuple(reversed(boxes))


def reach_box(check: PlaceCheck, within: Box, z: float, block: BlockSpec) -> Box | None:
    """The largest part of `within` where `check` lets the arm set `block` down with its bottom `z` [m]
    above the table, on the grid of `AREA_STEP_M`; None where there is none."""
    xs, ys = _lines(*within.x_m), _lines(*within.y_m)
    found = _largest_box(np.array([[check(x, y, z, block) for y in ys] for x in xs]))
    if found is None:
        return None
    return Box(within.levels, (xs[found[0]], xs[found[1]]), (ys[found[2]], ys[found[3]]))


def best_spot(world: World, levels: int, footprint: tuple[float, float], yaw_rad: float = 0.0,
              clearance_m: float = 0.01) -> tuple[float, float] | None:
    """A free table spot for the centre of a structure `levels` blocks high, or None.

    `footprint` is the structure's size seen from above [m], along `yaw_rad` and across it. The arm must
    set blocks down at every level up to `levels` over the footprint, with the build area's height
    margin, and the footprint must stay inside the reach and `clearance_m` from every object. Spots far
    from other objects and near the middle of the build area win.
    """
    cube = LEVEL_BLOCK
    half = (footprint[0] / 2, footprint[1] / 2)
    # Where blocks of the structure may sit: its middle and its corners, a cube's half side inwards.
    inset = [(u * max(half[0] - cube.half_size[0], 0.0), v * max(half[1] - cube.half_size[1], 0.0))
             for u, v in ((0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1))]
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    others = [o for o in world.objects.values() if not o.in_gripper]
    box = world.box(levels)
    middle = (np.mean(box.x_m), np.mean(box.y_m)) if box else (np.mean(world.reach.radius_m), 0.0)
    best: tuple[float, float, float, float] | None = None
    xs, ys = _area_grid(world.reach)
    for x in xs:
        for y in ys:
            outline = rectangle(x, y, yaw_rad, half)
            if not all(world.reach.contains(px, py) for px, py in outline):
                continue
            gap = min((-overlap(outline, o.footprint) for o in others), default=math.inf)
            if gap < clearance_m:
                continue
            points = [(x + c * u - s * v, y + s * u + c * v) for u, v in inset]
            if not all(world.place_check(px, py, level_bottom(n, HEIGHT_MARGIN_M), cube)
                       for n in range(1, levels + 1) for px, py in points):
                continue
            score = (-min(gap, 0.05), math.hypot(x - middle[0], y - middle[1]))
            if best is None or score < best[:2]:
                best = (*score, x, y)
    return None if best is None else (_m(best[2]), _m(best[3]))


def _area_grid(reach: Reach) -> tuple[list[float], list[float]]:
    """Grid lines every `AREA_STEP_M` over the box around `reach` [m]."""
    bearings = np.linspace(*reach.bearing_rad, 91)
    edge = np.concatenate([np.stack([r * np.cos(bearings), r * np.sin(bearings)], axis=1)
                           for r in reach.radius_m])
    lo, hi = np.floor(edge.min(axis=0) / AREA_STEP_M), np.ceil(edge.max(axis=0) / AREA_STEP_M)
    return (_lines(lo[0] * AREA_STEP_M, hi[0] * AREA_STEP_M),
            _lines(lo[1] * AREA_STEP_M, hi[1] * AREA_STEP_M))


def _lines(lo: float, hi: float) -> list[float]:
    """Grid lines every `AREA_STEP_M` from `lo` to `hi`, both on the grid [m]."""
    return [round(k * AREA_STEP_M, 3) for k in range(round(lo / AREA_STEP_M), round(hi / AREA_STEP_M) + 1)]


def _largest_box(ok: np.ndarray, inside: tuple[int, int, int, int] | None = None
                 ) -> tuple[int, int, int, int] | None:
    """Rows i0..i1 and columns j0..j1 of the largest all-true block of `ok` that holds `inside`, or None."""
    best, found = 0, None
    for i0 in range(ok.shape[0] if inside is None else inside[0] + 1):
        cols = np.ones(ok.shape[1], bool)
        for i1 in range(i0, ok.shape[0]):
            cols &= ok[i1]
            if not cols.any():
                break
            if inside is not None and i1 < inside[1]:
                continue
            for j0, j1 in _runs(cols):
                if inside is not None and not j0 <= inside[2] <= inside[3] <= j1:
                    continue
                if (i1 - i0 + 1) * (j1 - j0 + 1) > best:
                    best, found = (i1 - i0 + 1) * (j1 - j0 + 1), (i0, i1, j0, j1)
    return found


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """First and last index of each run of true values."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _natural(label: str) -> list[Any]:
    """Sort key that orders numbers by value: "red cube 2" before "red cube 10"."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label)]


def _m(value: float) -> float:
    """Metres rounded to mm, without a negative zero."""
    return round(float(value), 3) + 0.0


def _rotation(rotation: object) -> np.ndarray | None:
    """The rotation matrix, or None when it is missing, invalid or the identity (no orientation known)."""
    R = np.asarray(rotation, float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)) or np.allclose(R, np.eye(3), atol=1e-6):
        return None
    return R


def _shape(label: str, position: np.ndarray, R: np.ndarray | None, source: str, age: float) -> WorldObject:
    x, y, z = (float(v) for v in position)
    kind = kind_of(label)
    spec = BLOCKS.get(kind) if kind else None
    yaw = None if R is None else math.atan2(float(R[1, 0]), float(R[0, 0]))
    if spec is None:
        footprint = rectangle(x, y, 0.0, (UNKNOWN_HALF_M, UNKNOWN_HALF_M))
        if yaw is not None:
            yaw = reduce_yaw(yaw, 2 * math.pi)
        return WorldObject(label, kind, x, y, z, yaw, source, age, z - UNKNOWN_HALF_M, z + UNKNOWN_HALF_M,
                           footprint)
    rot = np.eye(3) if R is None else R
    bottom = float(bottom_height(np, spec, position[None], rot[None])[0])
    if yaw is not None:
        yaw = reduce_yaw(yaw, spec.yaw_period)
    return WorldObject(label, kind, x, y, z, yaw, source, age, bottom, 2 * z - bottom,
                       _footprint(spec, x, y, rot))


def _footprint(spec: BlockSpec, x: float, y: float, R: np.ndarray) -> Polygon:
    """The convex outline of the block's bounding box seen from above."""
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * np.asarray(spec.half_size)
    points = sorted({(round(float(px), 9), round(float(py), 9)) for px, py in (corners @ R.T)[:, :2]})
    hull: list[Point] = []
    for chain in (points, points[::-1]):
        part: list[Point] = []
        for pt in chain:
            while len(part) >= 2 and _cross(part[-2], part[-1], pt) <= 0:
                part.pop()
            part.append(pt)
        hull.extend(part[:-1])
    return tuple((x + px, y + py) for px, py in hull)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _with_supports(obj: WorldObject, others: Iterable[WorldObject]) -> WorldObject:
    """`obj` with `resting_on`: the gripper, the blocks whose tops meet its bottom, or the table."""
    if obj.in_gripper:
        return replace(obj, resting_on=(GRIPPER,))
    if obj.spec is None:
        return obj
    blocks = [(abs(obj.bottom_m - o.top_m), o.label) for o in others
              if o.label != obj.label and o.spec is not None and not o.in_gripper
              and abs(obj.bottom_m - o.top_m) <= SUPPORT_TOL_M
              and overlap(obj.footprint, o.footprint) > SUPPORT_OVERLAP_M]
    table = max(obj.bottom_m, 0.0)
    if blocks and min(err for err, _ in blocks) < table:
        return replace(obj, resting_on=tuple(sorted(label for _, label in blocks)))
    if table <= SUPPORT_TOL_M:
        return replace(obj, resting_on=(TABLE,))
    return obj
