"""Walk plan steps through the world they will meet: check a plan, and work out what is left of a fixed one.

Each step is resolved against the world as the steps before it leave it: its block goes to the step's
`TargetPose`, lying as the catalog describes it. `check_steps` says why a plan cannot run, for the model's
second try. `remaining` replans a fixed plan from the world seen now: every step again from the first one
that does not hold, with the blocks lying on a block it moves or builds on set aside on free table spots
first.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from .relations import DEFAULT_LIMITS, Limits, RelationError, TargetPose, holds, resolve
from .schema import ACROSS, AT, NEXT_TO, ON, Place, Step
from .world import Reach, World, best_spot, label_key

#: Pose source of a block where an earlier step of the plan puts it.
PLANNED_SOURCE = "plan"
#: Free space kept around a block set aside [m].
ASIDE_CLEARANCE_M = 0.03
#: A block is set aside at least this far from the base [m]: closer in, the arm folds onto itself.
ASIDE_MIN_RADIUS_M = 0.2

#: Builds the world state from object poses.
WorldOf = Callable[[Mapping[str, ObjectPose]], World]


def describe(step: Step) -> str:
    """A short English text of a step: "red cube on blue cube"."""
    place = step.place
    if place.relation == ON:
        return f"{step.pick} on {place.targets[0]}"
    if place.relation == ACROSS:
        return f"{step.pick} across {place.targets[0]} and {place.targets[1]}"
    if place.relation == NEXT_TO:
        return f"{step.pick} next to {place.targets[0]} ({place.side})"
    return f"{step.pick} at ({place.x_m or 0.0:.3f}, {place.y_m or 0.0:.3f}) m"


def after_step(poses: Mapping[str, ObjectPose], step: Step, pose: TargetPose,
               half_height_m: float) -> dict[str, ObjectPose]:
    """The poses once `step` has set its block, `half_height_m` half high as it lies, down at `pose`."""
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = {label_key(str(k)): v for k, v in poses.items()}
    out[label_key(step.pick)] = ObjectPose(np.array([pose.x_m, pose.y_m, pose.z_m + half_height_m]), rotation,
                                           PLANNED_SOURCE, 0.0)
    return out


def _moved(poses: Mapping[str, ObjectPose], world: World, step: Step,
           pose: TargetPose) -> dict[str, ObjectPose]:
    obj = world.get(step.pick)
    spec = obj.spec if obj is not None else None
    if spec is None:
        raise RelationError(f"'{label_key(step.pick)}' is not a known block kind")
    return after_step(poses, step, pose, spec.half_size[2])


def check_steps(steps: Sequence[Step], poses: Mapping[str, ObjectPose], world_of: WorldOf, *,
                limits: Limits = DEFAULT_LIMITS, max_steps: int = 8) -> list[str]:
    """Why the steps cannot run one after another from `poses`; empty when they can.

    The walk stops at the first step that cannot run, as the ones after it depend on it.
    """
    problems = []
    if len(steps) > max_steps:
        problems.append(f"the plan has {len(steps)} steps; at most {max_steps} are allowed")
    world = world_of(poses)
    for i, step in enumerate(steps, start=1):
        try:
            pose = resolve(step, world, limits=limits)
            poses = _moved(poses, world, step, pose)
        except RelationError as exc:
            problems.append(f"step {i} ({describe(step)}): {exc}")
            break
        world = world_of(poses)
    return problems


def first_unmet(goal: Sequence[Step], world: World, limits: Limits = DEFAULT_LIMITS) -> int | None:
    """The index of the first step of `goal` whose relation does not hold in `world`, or None."""
    return next((i for i, step in enumerate(goal) if not holds(step, world, limits)[0]), None)


def remaining(goal: Sequence[Step], poses: Mapping[str, ObjectPose], world_of: WorldOf, *,
              limits: Limits = DEFAULT_LIMITS) -> list[Step]:
    """The steps that finish the fixed plan `goal` from `poses`; empty when every step holds.

    From the first step that does not hold, every step runs again. Before one, each block lying on its
    block or its targets goes to a free table spot, top first. Raises `RelationError` when a step cannot
    run, with the step named.
    """
    world = world_of(poses)
    first = first_unmet(goal, world, limits)
    if first is None:
        return []
    out: list[Step] = []
    for step in goal[first:]:
        for label in (step.pick, *step.place.targets):
            for top in _on_top(world, label):
                try:
                    aside = _aside(world, top, limits)
                    pose = resolve(aside, world, limits=limits)
                    poses = _moved(poses, world, aside, pose)
                except RelationError as exc:
                    raise RelationError(f"cannot set '{top}' aside from '{label}': {exc}") from exc
                world = world_of(poses)
                out.append(aside)
        try:
            pose = resolve(step, world, limits=limits)
            poses = _moved(poses, world, step, pose)
        except RelationError as exc:
            raise RelationError(f"{describe(step)}: {exc}") from exc
        world = world_of(poses)
        out.append(step)
    return out


def _on_top(world: World, label: str) -> list[str]:
    """The blocks lying on `label`, directly or higher up, the topmost first."""
    out: list[str] = []

    def visit(key: str, seen: tuple[str, ...]) -> None:
        for above in world.above(key):
            if above not in seen and above not in out:
                visit(above, (*seen, above))
                out.append(above)

    visit(label_key(label), (label_key(label),))
    return out


def _aside(world: World, label: str, limits: Limits) -> Step:
    """A step that sets `label` down on a free table spot."""
    obj = world.get(label)
    spec = obj.spec if obj is not None else None
    if obj is None or spec is None:
        raise RelationError(f"'{label}' is not a known block kind")
    size = (2 * spec.half_size[0], 2 * spec.half_size[1])
    lo, hi = world.reach.radius_m
    outer = dataclasses.replace(world, reach=Reach((min(max(lo, ASIDE_MIN_RADIUS_M), hi), hi),
                                                   world.reach.bearing_rad))
    spot = best_spot(outer, 1, size, clearance_m=max(ASIDE_CLEARANCE_M, limits.clearance_m))
    if spot is None:
        raise RelationError(f"no free table spot for '{label}'")
    return Step(obj.label, Place(AT, (), None, None, spot[0], spot[1], None), "set aside")
