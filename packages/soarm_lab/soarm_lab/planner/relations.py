"""Turn a plan step's relation into the pose where the held block goes, and check it held afterwards.

`resolve` works from the world state at the moment the step runs, so earlier steps' shifts count.
The pose names the held block's centre in xy, the height where its bottom lands and its heading.
Blocks lie as the catalog describes them; a held block keeps its body z vertical. The world's
`place_check` decides whether the arm can set the block down there at that height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..blocks import BlockSpec
from ..blocks.classic_world import BLOCKS
from .schema import ACROSS, AT, NEXT_TO, ON, SIDE_VECTORS, Place, Step, place_problem
from .world import (
    MAX_LEVELS,
    TABLE,
    World,
    WorldObject,
    extent,
    label_key,
    level_bottom,
    overlap,
    reach_box,
    rectangle,
    reduce_yaw,
)

#: Kinds whose top is round: nothing rests on them.
ROUND_TOP = ("cylinder",)
#: An object counts as in the way only when its top rises this far above the landing height [m].
ABOVE_M = 0.005


@dataclass(frozen=True)
class TargetPose:
    """Where the held block goes, in the base frame."""

    #: Centre of the held block [m].
    x_m: float
    y_m: float
    #: Height above the table where the held block's bottom lands [m].
    z_m: float
    #: Heading of the held block's body x axis [rad], reduced by its yaw symmetry to (-period/2, period/2].
    yaw_rad: float


@dataclass(frozen=True)
class Limits:
    """Tolerances and limits of the relations."""

    #: Most blocks in one stack, the table level counting as 1.
    max_stack: int = MAX_LEVELS
    #: next_to's gap when the step gives none [m].
    gap_m: float = 0.01
    #: across: the two tops must match within this [m].
    top_match_m: float = 0.005
    #: across: the held block must lie on each support at least this far [m].
    min_overlap_m: float = 0.01
    #: Free space kept around the placed block to other objects [m].
    clearance_m: float = 0.005
    #: holds: the block's centre within this of its place [m].
    hold_xy_m: float = 0.008
    #: holds: the block's bottom within this of the expected height [m].
    hold_z_m: float = 0.008


DEFAULT_LIMITS = Limits()


class RelationError(ValueError):
    """A step cannot run in the current world; the message says why."""


def resolve(step: Step, world: World, held_kind: str | None = None,
            limits: Limits = DEFAULT_LIMITS) -> TargetPose:
    """The pose for `step`'s block, after checking the step can run now; raise `RelationError` if not.

    `held_kind` is the catalog kind of the picked block; None takes it from the world state. The picked
    block may still lie where it is or already be in the gripper.
    """
    place = step.place
    problem = place_problem(place)
    if problem:
        raise RelationError(problem)
    pick = world.get(step.pick)
    if pick is None:
        raise RelationError(f"no object labelled '{label_key(step.pick)}' in the world state")
    targets = []
    for label in place.targets:
        target = world.get(label)
        if label_key(label) == pick.label:
            raise RelationError(f"'{pick.label}' cannot be placed relative to itself")
        if target is None:
            raise RelationError(f"no object labelled '{label_key(label)}' in the world state")
        targets.append(target)
    kind = held_kind or pick.kind
    held = BLOCKS.get(kind) if kind else None
    if held is None:
        raise RelationError(f"'{pick.label}' is not a known block kind ({', '.join(BLOCKS)})")
    if not pick.in_gripper and world.above(pick.label):
        raise RelationError(f"'{pick.label}' has {_names(world.above(pick.label))} on top; move that first")
    for target in targets:
        _check_target(target, place.relation, pick.label, world)

    pose = _pose(place, targets, held, limits, strict=True)
    stacked = place.relation in (ON, ACROSS)
    level = 1 + max((world.level(t.label) for t in targets), default=0) if stacked else 1
    if level > limits.max_stack:
        raise RelationError(f"the stack would be {level} blocks high; the limit is {limits.max_stack}")
    if not world.reach.contains(pose.x_m, pose.y_m):
        r, bearing = math.hypot(pose.x_m, pose.y_m), math.degrees(math.atan2(pose.y_m, pose.x_m))
        raise RelationError(f"the place ({pose.x_m:.3f}, {pose.y_m:.3f}) m is outside the reachable area "
                            f"(distance {r:.3f} m, bearing {bearing:.0f} deg)")
    if not world.place_check(pose.x_m, pose.y_m, pose.z_m, held):
        raise RelationError(f"the arm cannot set '{pick.label}' down at ({pose.x_m:.3f}, {pose.y_m:.3f}) m "
                            f"with its bottom {pose.z_m * 1000:.0f} mm above the table; "
                            f"{_where_instead(world, pose, held, level)}")
    in_way = _in_the_way(world, pose, held, {pick.label, *(t.label for t in targets)}, limits.clearance_m)
    if in_way is not None:
        raise RelationError(f"'{in_way}' is in the way at the place")
    return pose


def holds(step: Step, world_after: World, limits: Limits = DEFAULT_LIMITS) -> tuple[bool, str]:
    """Whether the step's relation holds in the world seen after it ran, and a short reason.

    Checks the block's centre and bottom height against the relation worked out from the targets' poses
    now, and what the block rests on. Yaw is not checked.
    """
    place = step.place
    problem = place_problem(place)
    if problem:
        return False, problem
    block = world_after.get(step.pick)
    if block is None:
        return False, f"'{label_key(step.pick)}' is not seen"
    if block.in_gripper:
        return False, f"'{block.label}' is still in the gripper"
    held = block.spec
    if held is None:
        return False, f"'{block.label}' is not a known block kind"
    targets = []
    for label in place.targets:
        target = world_after.get(label)
        if target is None:
            return False, f"'{label_key(label)}' is not seen"
        targets.append(target)
    try:
        expected = _pose(place, targets, held, limits, strict=False)
    except RelationError as exc:
        return False, str(exc)
    off = math.hypot(block.x_m - expected.x_m, block.y_m - expected.y_m)
    if off > limits.hold_xy_m:
        return False, f"'{block.label}' is {off * 1000:.0f} mm from its place"
    dz = block.bottom_m - expected.z_m
    if abs(dz) > limits.hold_z_m:
        return False, f"'{block.label}' sits {dz * 1000:+.0f} mm from the expected height"
    needed = [t.label for t in targets] if place.relation in (ON, ACROSS) else [TABLE]
    missing = [n for n in needed if n not in block.resting_on]
    if missing:
        return False, f"'{block.label}' does not rest on {_names(missing)}"
    return True, f"'{block.label}' rests on {_names(needed)} where planned"


def _where_instead(world: World, pose: TargetPose, held: BlockSpec, level: int) -> str:
    """Where a structure as high as this one can stand instead, for a refused place."""
    box = world.box(level)
    if box is None:
        return f"the build area has no place for a structure {level} high"
    if not box.contains(pose.x_m, pose.y_m):
        return box.text()
    # Inside the box: the blocks below stand higher than the box allows for.
    over = (pose.z_m - level_bottom(level)) * 1000
    below = f"{level - 1} cube{'s' if level > 2 else ''}"
    why = (f"the blocks below stand {over:.0f} mm higher than {below}" if over >= 0.5
           else "the arm falls short here")
    room = reach_box(world.place_check, box, pose.z_m, held)
    if room is None:
        return (f"{why}, and the arm reaches that high nowhere in the build area; "
                "use fewer or thinner blocks below")
    return f"{why}; at that height the arm reaches only {room.span()}"


def _check_target(target: WorldObject, relation: str, pick: str, world: World) -> None:
    if target.in_gripper:
        raise RelationError(f"'{target.label}' is in the gripper")
    if target.spec is None:
        raise RelationError(f"'{target.label}' is not a known block kind, so its size is unknown")
    if relation in (ON, ACROSS):
        if target.kind in ROUND_TOP:
            raise RelationError(f"nothing stays on the round top of '{target.label}'")
        above = [a for a in world.above(target.label) if a != pick]
        if above:
            raise RelationError(f"'{target.label}' has {_names(above)} on top")


def _pose(place: Place, targets: list[WorldObject], held: BlockSpec, limits: Limits, *,
          strict: bool) -> TargetPose:
    """The relation's pose from the targets' poses; `strict` also checks an across span."""
    if place.relation == AT:
        yaw = math.radians(place.yaw_deg or 0.0)
        return TargetPose(float(place.x_m or 0.0), float(place.y_m or 0.0), 0.0,
                          reduce_yaw(yaw, held.yaw_period))
    if place.relation == ON:
        (a,) = targets
        return TargetPose(a.x_m, a.y_m, a.top_m, reduce_yaw(a.yaw_rad or 0.0, held.yaw_period))
    if place.relation == NEXT_TO:
        (a,) = targets
        gap = limits.gap_m if place.gap_m is None else place.gap_m
        return _next_to(a, np.array(SIDE_VECTORS[str(place.side)]), gap, held)
    if place.relation == ACROSS:
        a, b = targets
        return _across(a, b, held, limits, strict)
    raise RelationError(f"unknown relation {place.relation!r}")


def _next_to(a: WorldObject, side: np.ndarray, gap: float, held: BlockSpec) -> TargetPose:
    """Beside `a` across its face that looks most toward `side`, with the held block turned like `a`."""
    psi = a.yaw_rad or 0.0
    c, s = math.cos(psi), math.sin(psi)
    faces = [np.array(v) for v in ((c, s), (-s, c), (-c, -s), (s, -c))]
    u = max(faces, key=lambda v: float(v @ side))
    yaw = reduce_yaw(psi, held.yaw_period)
    centre = a.xy + u * (extent(a.footprint, a.xy, u) + gap + _half_along(held, yaw, u))
    return TargetPose(float(centre[0]), float(centre[1]), 0.0, yaw)


def _across(a: WorldObject, b: WorldObject, held: BlockSpec, limits: Limits, strict: bool) -> TargetPose:
    """A bridge on `a` and `b`: centred between them, its long side along the line from `a` to `b`."""
    d_vec = b.xy - a.xy
    d = float(np.linalg.norm(d_vec))
    if d < 1e-3:
        raise RelationError(f"'{a.label}' and '{b.label}' are at the same place")
    u = d_vec / d
    if strict:
        if abs(a.top_m - b.top_m) > limits.top_match_m:
            raise RelationError(f"the tops of '{a.label}' and '{b.label}' differ by "
                                f"{abs(a.top_m - b.top_m) * 1000:.0f} mm; a bridge needs them level")
        half = held.half_size[0]
        for support, toward in ((a, u), (b, -u)):
            resting = half - (d / 2 - extent(support.footprint, support.xy, toward))
            if resting < limits.min_overlap_m:
                raise RelationError(f"a {held.name} ({2000 * half:.0f} mm long) would rest only "
                                    f"{resting * 1000:.0f} mm on '{support.label}' (needs "
                                    f"{limits.min_overlap_m * 1000:.0f} mm); the supports are too far apart")
    mid = (a.xy + b.xy) / 2
    yaw = reduce_yaw(math.atan2(float(u[1]), float(u[0])), held.yaw_period)
    return TargetPose(float(mid[0]), float(mid[1]), max(a.top_m, b.top_m), yaw)


def _half_along(spec: BlockSpec, yaw: float, u: np.ndarray) -> float:
    """How far a block lying flat at heading `yaw` reaches from its centre along the unit `u` [m]."""
    angle = math.atan2(float(u[1]), float(u[0])) - yaw
    return abs(math.cos(angle)) * spec.half_size[0] + abs(math.sin(angle)) * spec.half_size[1]


def _in_the_way(world: World, pose: TargetPose, held: BlockSpec, skip: set[str],
                clearance: float) -> str | None:
    """The first object closer than `clearance` to the placed block and rising above its landing height."""
    foot = rectangle(pose.x_m, pose.y_m, pose.yaw_rad, held.half_size)
    for obj in world.objects.values():
        if obj.label in skip or obj.in_gripper or obj.top_m <= pose.z_m + ABOVE_M:
            continue
        if overlap(foot, obj.footprint) > -clearance:
            return obj.label
    return None


def _names(labels: list[str]) -> str:
    return " and ".join(f"'{label}'" for label in labels)
