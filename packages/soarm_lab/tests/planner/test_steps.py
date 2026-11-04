"""Walking plan steps through the world they meet: the check of a plan and the replan of a fixed one."""

from __future__ import annotations

import math

import numpy as np
import pytest
from machinekind_twin_api.strategy import ObjectPose

from soarm_lab.blocks.classic_world import BLOCKS
from soarm_lab.planner.relations import Limits, RelationError, TargetPose, holds, resolve
from soarm_lab.planner.schema import ACROSS, AT, NEXT_TO, ON, Place, Step
from soarm_lab.planner.steps import after_step, check_steps, describe, first_unmet, remaining
from soarm_lab.planner.world import world_state

HALF = BLOCKS["cube"].half_size[2]


def cube(x: float, y: float, level: int = 1) -> ObjectPose:
    return ObjectPose(np.array([x, y, (2 * level - 1) * HALF]), np.eye(3), "kamery", 0.1)


def on(pick: str, target: str) -> Step:
    return Step(pick, Place(ON, (target,)))


TABLE = {"blue cube": cube(0.2, 0.0), "green cube": cube(0.18, 0.1), "red cube": cube(0.18, -0.1)}
TOWER = [on("green cube", "blue cube"), on("red cube", "green cube")]


def world_of(poses):
    return world_state(poses)


def test_after_a_step_its_block_lies_at_the_pose_turned_like_it() -> None:
    poses = after_step(TABLE, on("red cube", "blue cube"), TargetPose(0.2, 0.0, 2 * HALF, math.pi / 6), HALF)
    red = poses["red cube"]
    assert red.position == pytest.approx([0.2, 0.0, 3 * HALF]) and red.source == "plan"
    assert math.atan2(red.rotation[1, 0], red.rotation[0, 0]) == pytest.approx(math.pi / 6)
    assert world_of(poses).get("red cube").resting_on == ("blue cube",)
    assert poses["green cube"] is TABLE["green cube"]


def test_a_tower_plan_walks_through() -> None:
    assert check_steps(TOWER, TABLE, world_of) == []
    assert describe(TOWER[0]) == "green cube on blue cube"
    assert describe(Step("red cube", Place(AT, (), None, None, 0.2, 0.05))) == "red cube at (0.200, 0.050) m"


def test_the_walk_names_the_first_step_that_cannot_run() -> None:
    problems = check_steps([on("purple cube", "blue cube"), *TOWER], TABLE, world_of)
    assert problems == ["step 1 (purple cube on blue cube): no object labelled 'purple cube' in the world "
                        "state"]
    # The first step puts green on blue, so blue can no longer be picked.
    problems = check_steps([TOWER[0], on("blue cube", "red cube")], TABLE, world_of)
    assert len(problems) == 1 and problems[0].startswith("step 2 (blue cube on red cube): 'blue cube' has")
    problems = check_steps(TOWER, TABLE, world_of, limits=Limits(max_stack=2))
    assert problems == ["step 2 (red cube on green cube): the stack would be 3 blocks high; the limit is 2"]


def test_the_walk_refuses_too_many_steps() -> None:
    problems = check_steps(TOWER, TABLE, world_of, max_steps=1)
    assert problems == ["the plan has 2 steps; at most 1 are allowed"]


def test_a_bridge_plan_walks_through_next_to_and_across() -> None:
    """Green goes beside blue, then the cuboid across both: each step meets the world the one before left."""
    lying = BLOCKS["cuboid"].half_size[2]
    poses = {"blue cube": cube(0.2, -0.03), "green cube": cube(0.18, 0.1),
             "yellow cuboid": ObjectPose(np.array([0.17, -0.1, lying]), np.eye(3), "kamery", 0.1)}
    beside = Step("green cube", Place(NEXT_TO, ("blue cube",), "left", 0.015))
    bridge = Step("yellow cuboid", Place(ACROSS, ("blue cube", "green cube")))
    assert check_steps([beside, bridge], poses, world_of) == []
    assert describe(beside) == "green cube next to blue cube (left)"
    assert describe(bridge) == "yellow cuboid across blue cube and green cube"
    pose = resolve(beside, world_of(poses))
    assert (pose.x_m, pose.y_m, pose.z_m) == pytest.approx((0.2, -0.03 + 2 * HALF + 0.015, 0.0))
    poses = after_step(poses, beside, pose, HALF)
    pose = resolve(bridge, world_of(poses))
    assert (pose.x_m, pose.z_m) == pytest.approx((0.2, 2 * HALF))
    assert pose.yaw_rad == pytest.approx(math.pi / 2)
    poses = after_step(poses, bridge, pose, lying)
    assert holds(beside, world_of(poses))[0] and holds(bridge, world_of(poses))[0]
    assert world_of(poses).get("yellow cuboid").resting_on == ("blue cube", "green cube")


def test_a_fixed_plan_that_holds_needs_nothing_more() -> None:
    built = {**TABLE, "green cube": cube(0.2, 0.0, 2), "red cube": cube(0.2, 0.0, 3)}
    assert first_unmet(TOWER, world_of(built)) is None
    assert remaining(TOWER, built, world_of) == []


def test_a_fixed_plan_goes_on_from_its_first_step_that_does_not_hold() -> None:
    half_done = {**TABLE, "green cube": cube(0.2, 0.0, 2)}
    assert first_unmet(TOWER, world_of(half_done)) == 1
    assert remaining(TOWER, half_done, world_of) == [TOWER[1]]
    assert remaining(TOWER, TABLE, world_of) == TOWER


def test_a_knocked_tower_sets_the_top_block_aside_and_builds_again() -> None:
    """Green fell off blue with red still on it: red goes to a free spot, then the whole tower again."""
    knocked = {**TABLE, "green cube": cube(0.17, -0.06), "red cube": cube(0.17, -0.06, 2)}
    steps = remaining(TOWER, knocked, world_of)
    assert [s.pick for s in steps] == ["red cube", "green cube", "red cube"] and steps[1:] == TOWER
    aside = steps[0].place
    assert aside.relation == AT and steps[0].why == "set aside"
    spot = np.array([aside.x_m, aside.y_m])
    for label, pose in knocked.items():
        if label != "red cube":
            assert np.linalg.norm(spot - pose.position[:2]) >= 0.029 + 0.03 - 1e-9, label
    # The walk of the replan itself passes.
    assert check_steps(steps, knocked, world_of) == []


def test_a_fixed_plan_that_cannot_go_on_says_which_step() -> None:
    gone = {k: v for k, v in TABLE.items() if k != "green cube"}
    with pytest.raises(RelationError, match="^green cube on blue cube: no object labelled 'green cube'"):
        remaining(TOWER, gone, world_of)
    knocked = {**TABLE, "green cube": cube(0.17, -0.06), "red cube": cube(0.17, -0.06, 2)}
    with pytest.raises(RelationError, match="^cannot set 'red cube' aside from 'green cube': no free table"):
        remaining(TOWER, knocked, world_of, limits=Limits(clearance_m=1.0))


def test_holds_matches_the_walk() -> None:
    poses = TABLE
    for step in TOWER:
        assert check_steps([step], poses, world_of) == []
        target = world_of(poses).get(step.place.targets[0])
        poses = after_step(poses, step, TargetPose(target.x_m, target.y_m, target.top_m, 0.0), HALF)
        assert holds(step, world_of(poses))[0]
