"""Relations to target poses: the bridge example, each relation, numbered blocks, every refusal, the arm's
place check, and the check after a step."""

from __future__ import annotations

import math

import numpy as np
import pytest
from machinekind_twin_api.strategy import ObjectPose

from soarm_lab.blocks import BlockSpec
from soarm_lab.planner.relations import Limits, RelationError, TargetPose, holds, resolve
from soarm_lab.planner.schema import Place, Step
from soarm_lab.planner.world import HEIGHT_MARGIN_M, World, world_state

CUBE, CUBOID_H = 0.029, 0.014
HALF_PI = math.pi / 2


def pose(x: float, y: float, z: float, yaw_deg: float | None = None, source: str = "kamery") -> ObjectPose:
    R = np.eye(3)
    if yaw_deg is not None:
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return ObjectPose(np.array([x, y, z]), R, source, 0.1)


def step(pick: str, relation: str, *targets: str, **values) -> Step:
    return Step(pick, Place(relation, targets, **values))


#: The table before the bridge: two cubes as pillars, the deck, a cube for the top and a spare cube.
TABLE = {
    "red cube": pose(0.2, 0.03, CUBE / 2),
    "green cube": pose(0.17, 0.12, CUBE / 2),
    "yellow cuboid": pose(0.22, -0.12, CUBOID_H / 2),
    "blue cube": pose(0.16, -0.05, CUBE / 2),
}
#: Where each block goes in the bridge.
GREEN_AT = (0.2, 0.03 - CUBE - 0.03)
DECK_AT = (0.2, (0.03 + GREEN_AT[1]) / 2)


def world(**changes: ObjectPose) -> World:
    return world_state({**TABLE, **changes})


PILLARS = {"green cube": pose(*GREEN_AT, CUBE / 2)}
BRIDGE = {**PILLARS, "yellow cuboid": pose(*DECK_AT, CUBE + CUBOID_H / 2, yaw_deg=90.0)}
TOPPED = {**BRIDGE, "blue cube": pose(*DECK_AT, CUBE + CUBOID_H + CUBE / 2)}

STEP1 = step("green cube", "next_to", "red cube", side="right", gap_m=0.03)
STEP2 = step("yellow cuboid", "across", "red cube", "green cube")
STEP3 = step("blue cube", "on", "yellow cuboid")


def same_yaw(a: float, b: float, period: float) -> bool:
    return abs(math.remainder(a - b, period)) < 1e-9


def test_the_bridge_example() -> None:
    first = resolve(STEP1, world())
    assert (first.x_m, first.y_m, first.z_m) == pytest.approx((*GREEN_AT, 0.0))
    assert first.yaw_rad == pytest.approx(0.0)

    deck = resolve(STEP2, world(**PILLARS))
    assert (deck.x_m, deck.y_m, deck.z_m) == pytest.approx((*DECK_AT, CUBE))
    assert same_yaw(deck.yaw_rad, HALF_PI, math.pi)       # long side from red to green
    assert -HALF_PI < deck.yaw_rad <= HALF_PI

    top = resolve(STEP3, world(**BRIDGE))
    assert (top.x_m, top.y_m, top.z_m) == pytest.approx((*DECK_AT, CUBE + CUBOID_H))
    assert same_yaw(top.yaw_rad, 0.0, HALF_PI)


def test_the_bridge_holds_after_each_step() -> None:
    assert holds(STEP1, world(**PILLARS)) == (True, "'green cube' rests on 'table' where planned")
    assert holds(STEP2, world(**BRIDGE))[0]
    assert holds(STEP3, world(**TOPPED))[0]


def test_on_turns_the_block_like_its_support() -> None:
    w = world(**{"red cube": pose(0.2, 0.03, CUBE / 2, yaw_deg=30.0)})
    cuboid = resolve(step("yellow cuboid", "on", "red cube"), w)
    assert cuboid.yaw_rad == pytest.approx(math.radians(30.0)) and cuboid.z_m == pytest.approx(CUBE)
    w = world(**{"yellow cuboid": pose(0.2, -0.05, CUBOID_H / 2, yaw_deg=100.0),
                 "blue cube": pose(0.2, 0.1, CUBE / 2)})
    cube = resolve(step("blue cube", "on", "yellow cuboid"), w)
    assert cube.yaw_rad == pytest.approx(math.radians(10.0)) and cube.z_m == pytest.approx(CUBOID_H)


def test_next_to_uses_the_face_toward_the_side() -> None:
    w = world(**{"red cube": pose(0.18, 0.0, CUBE / 2, yaw_deg=30.0)})
    target = resolve(step("blue cube", "next_to", "red cube", side="front"), w)
    reach = CUBE + 0.01                                   # half + default gap + half
    assert (target.x_m, target.y_m) == pytest.approx((0.18 + reach * math.cos(math.radians(30)),
                                                      reach * math.sin(math.radians(30))))
    assert target.yaw_rad == pytest.approx(math.radians(30.0)) and target.z_m == 0.0


def test_next_to_a_cuboid_adds_both_half_sizes() -> None:
    w = world(**{"yellow cuboid": pose(0.2, 0.0, CUBOID_H / 2)})
    beside = resolve(step("green cube", "next_to", "yellow cuboid", side="back", gap_m=0.0), w)
    assert (beside.x_m, beside.y_m) == pytest.approx((0.2 - 0.03 - CUBE / 2, 0.0))


def test_at_places_on_the_table_with_the_yaw_reduced() -> None:
    target = resolve(step("yellow cuboid", "at", x_m=0.22, y_m=0.09, yaw_deg=120.0), world())
    assert target == TargetPose(0.22, 0.09, 0.0, pytest.approx(math.radians(-60.0)))
    assert resolve(step("blue cube", "at", x_m=0.21, y_m=0.08), world()).yaw_rad == 0.0


def test_the_held_kind_and_a_block_already_in_the_gripper() -> None:
    w = world(**{"block": pose(0.2, -0.1, CUBE / 2)})
    assert resolve(step("block", "on", "red cube"), w, held_kind="cube").z_m == pytest.approx(CUBE)
    held = world(**{"blue cube": pose(0.2, 0.03, 0.1, source="w dloni")})
    assert resolve(step("blue cube", "on", "red cube"), held).z_m == pytest.approx(CUBE)


@pytest.mark.parametrize(("changes", "the_step", "message"), [
    ({}, step("purple cube", "on", "red cube"), "no object labelled 'purple cube'"),
    ({}, step("blue cube", "on", "pink cube"), "no object labelled 'pink cube'"),
    ({}, step("blue cube", "on", "Blue Cube"), "relative to itself"),
    ({}, step("blue cube", "on"), "on takes 1 target"),
    ({"blue pen": pose(0.2, -0.1, 0.005)}, step("blue pen", "on", "red cube"), "not a known block kind"),
    ({"blue pen": pose(0.2, -0.1, 0.005)}, step("blue cube", "next_to", "blue pen", side="left"),
     "size is unknown"),
    (TOPPED, step("red cube", "at", x_m=0.2, y_m=0.15), "'red cube' has 'yellow cuboid' on top"),
    (BRIDGE, step("blue cube", "on", "red cube"), "'red cube' has 'yellow cuboid' on top"),
    ({"white cylinder": pose(0.2, -0.1, CUBE / 2)}, step("blue cube", "on", "white cylinder"), "round top"),
    ({"blue cube": pose(0.2, 0.0, 0.1, source="w dloni")}, step("green cube", "on", "blue cube"),
     "'blue cube' is in the gripper"),
    ({}, step("blue cube", "next_to", "red cube", side="front", gap_m=0.1), "outside the reachable area"),
    ({}, step("green cube", "at", x_m=0.16, y_m=-0.05), "'blue cube' is in the way"),
    ({"blue cube": pose(0.2, -0.06, CUBE / 2)}, STEP1, "'blue cube' is in the way"),
    ({}, step("blue cube", "across", "red cube", "yellow cuboid"), "differ by 15 mm"),
    ({"green cube": pose(0.2, 0.03 - CUBE - 0.045, CUBE / 2)}, STEP2, "too far apart"),
])
def test_a_step_that_cannot_run_is_refused(changes, the_step: Step, message: str) -> None:
    with pytest.raises(RelationError, match=message):
        resolve(the_step, world(**changes))


def test_numbered_blocks_are_told_apart() -> None:
    w = world_state({"red cube": pose(0.2, 0.03, CUBE / 2), "red cube 2": pose(0.2, -0.05, CUBE / 2),
                     "red cube 3": pose(0.17, 0.12, CUBE / 2)})
    on = resolve(step("red cube 3", "on", "red cube 2"), w)
    assert (on.x_m, on.y_m, on.z_m) == pytest.approx((0.2, -0.05, CUBE))
    bridge = world_state({**TABLE, "red cube 2": pose(*GREEN_AT, CUBE / 2)})
    deck = resolve(step("yellow cuboid", "across", "red cube", "Red Cube 2"), bridge)
    assert (deck.x_m, deck.y_m) == pytest.approx(DECK_AT)
    with pytest.raises(RelationError, match="no object labelled 'red cube 4'"):
        resolve(step("red cube 4", "on", "red cube"), w)


def test_a_place_the_arm_cannot_reach_at_that_height_is_refused() -> None:
    stack = {"red cube": pose(0.155, 0.0, CUBE / 2), "green cube": pose(0.155, 0.0, CUBE * 1.5)}
    with pytest.raises(RelationError) as refused:
        resolve(step("blue cube", "on", "green cube"), world(**stack))
    message = str(refused.value)
    assert message.startswith("the arm cannot set 'blue cube' down at (0.155, 0.000) m with its bottom 58 mm")
    assert message.endswith("; a structure 3 high fits in x 0.19..0.21 m, y -0.08..0.08 m")
    assert resolve(step("blue cube", "at", x_m=0.155, y_m=0.0), world()).z_m == 0.0
    near = {k: pose(0.2, 0.0, v.position[2]) for k, v in stack.items()}
    assert resolve(step("blue cube", "on", "green cube"), world(**near)).z_m == pytest.approx(2 * CUBE)


def test_a_stack_a_little_high_still_takes_its_top_inside_the_build_area() -> None:
    """Blocks a few mm taller than nominal do not stop the top block anywhere in the 3-high box. Where the
    blocks below stand too high, the refusal says so and names where the arm reaches that high."""
    box = world().box(3)
    assert box is not None and box.span() == "x 0.19..0.21 m, y -0.08..0.08 m"
    places = [(box.x_m[i], box.y_m[j]) for i in (0, 1) for j in (0, 1)] + [(0.2, 0.0), (0.195, 0.035)]
    for x, y in places:
        for extra in (0.001, 0.002, HEIGHT_MARGIN_M):
            stack = {"red cube": pose(x, y, CUBE / 2), "green cube": pose(x, y, CUBE * 1.5 + extra)}
            on = resolve(step("blue cube", "on", "green cube"), world(**stack))
            assert on.z_m == pytest.approx(2 * CUBE + extra), (x, y, extra)
    tall = {"red cube": pose(0.19, 0.0, CUBE / 2), "green cube": pose(0.19, 0.0, CUBE * 1.5 + 0.004)}
    with pytest.raises(RelationError) as refused:
        resolve(step("blue cube", "on", "green cube"), world(**tall))
    assert str(refused.value).endswith("bottom 62 mm above the table; the blocks below stand 4 mm higher "
                                       "than 2 cubes; at that height the arm reaches only x 0.20..0.21 m, "
                                       "y -0.05..0.05 m")
    too_tall = {"red cube": pose(0.2, 0.0, CUBE / 2), "green cube": pose(0.2, 0.0, CUBE * 1.5 + 0.007)}
    with pytest.raises(RelationError, match="stand 7 mm higher than 2 cubes, and the arm reaches that high "
                                            "nowhere in the build area; use fewer or thinner blocks below"):
        resolve(step("blue cube", "on", "green cube"), world(**too_tall))


def test_the_place_check_is_pluggable() -> None:
    asked: list[tuple[float, float, float, str]] = []

    def low_only(x: float, y: float, z: float, block: BlockSpec) -> bool:
        asked.append((x, y, z, block.name))
        return z < 0.02

    w = world_state(TABLE, place_check=low_only)
    asked.clear()
    assert resolve(STEP1, w).z_m == 0.0 and len(asked) == 1
    assert asked[0][:3] == pytest.approx((*GREEN_AT, 0.0)) and asked[0][3] == "cube"
    w = world_state({**TABLE, **PILLARS}, place_check=low_only)
    with pytest.raises(RelationError, match="the build area has no place for a structure 2 high"):
        resolve(STEP2, w)


def test_the_stack_limit() -> None:
    assert resolve(STEP3, world(**BRIDGE), limits=Limits(max_stack=3)).z_m == pytest.approx(CUBE + CUBOID_H)
    with pytest.raises(RelationError, match="3 blocks high; the limit is 2"):
        resolve(STEP3, world(**BRIDGE), limits=Limits(max_stack=2))


def test_a_bridge_just_short_enough_and_one_with_touching_pillars() -> None:
    gap = 0.039                                           # rests 10.5 mm on each pillar
    w = world(**{"green cube": pose(0.2, 0.03 - CUBE - gap, CUBE / 2)})
    assert resolve(STEP2, w).z_m == pytest.approx(CUBE)
    w = world(**{"green cube": pose(0.2, 0.03 - CUBE, CUBE / 2)})
    assert resolve(STEP2, w).z_m == pytest.approx(CUBE)


@pytest.mark.parametrize(("the_step", "changes", "message"), [
    (STEP1, {"green cube": pose(GREEN_AT[0] + 0.02, GREEN_AT[1], CUBE / 2)}, "20 mm from its place"),
    (STEP1, {"green cube": pose(*GREEN_AT, 0.1, source="w dloni")}, "still in the gripper"),
    (STEP3, {**BRIDGE, "blue cube": pose(*DECK_AT, CUBE + CUBOID_H + CUBE / 2 + 0.015)}, "+15 mm"),
    (STEP3, {**BRIDGE, "blue cube": pose(0.17, -0.12, CUBE / 2)}, "from its place"),
    (STEP2, {**BRIDGE, "green cube": pose(*GREEN_AT, CUBE / 2 - 0.012)}, "does not rest on 'green cube'"),
    (STEP2, {**BRIDGE, "green cube": pose(0.2, 0.03, CUBE / 2)}, "at the same place"),
])
def test_a_step_that_did_not_hold(the_step: Step, changes, message: str) -> None:
    ok, reason = holds(the_step, world(**changes))
    assert not ok and message in reason


def test_holds_needs_the_blocks_seen() -> None:
    w = world_state({k: v for k, v in TABLE.items() if k != "green cube"})
    assert holds(STEP1, w) == (False, "'green cube' is not seen")
    alone = world_state({"blue cube": TABLE["blue cube"]})
    assert holds(STEP3, alone) == (False, "'yellow cuboid' is not seen")
    assert holds(step("blue cube", "on"), world())[0] is False
