"""World state from fake observations: kinds, numbered labels, sizes, yaw, what each object rests on, the
build area, free spots and the JSON."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from machinekind_twin_api.strategy import ObjectPose

from soarm_lab.blocks import BlockSpec
from soarm_lab.blocks.classic_world import BLOCKS
from soarm_lab.planner.world import (
    GRIPPER,
    HEIGHT_MARGIN_M,
    TABLE,
    Box,
    Reach,
    best_spot,
    find_build_area,
    kind_of,
    level_bottom,
    overlap,
    reach_box,
    rectangle,
    reduce_yaw,
    world_state,
)
from soarm_lab.robot import Desk

CUBE, CUBOID_H = 0.029, 0.014
CUBE_SPEC = BLOCKS["cube"]


def rot_z(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(x: float, y: float, z: float, yaw_deg: float | None = None, source: str = "kamery",
         age: float = 0.2, rotation: np.ndarray | None = None) -> ObjectPose:
    R = rotation if rotation is not None else np.eye(3) if yaw_deg is None else rot_z(yaw_deg)
    return ObjectPose(np.array([x, y, z]), R, source, age)


def test_a_block_on_the_table() -> None:
    world = world_state({"  Red   Cube ": pose(0.2, 0.03, CUBE / 2)})
    red = world.get("red cube")
    assert red is not None and red.label == "red cube" and red.kind == "cube"
    assert red.resting_on == (TABLE,)
    assert red.bottom_m == pytest.approx(0.0) and red.top_m == pytest.approx(CUBE)
    assert red.yaw_rad is None                        # identity: no orientation known
    assert red.to_json() == {"kind": "cube", "size_mm": [29, 29, 29], "x_m": 0.2, "y_m": 0.03, "z_m": 0.015,
                             "yaw_deg": None, "resting_on": "table", "source": "kamery", "age_s": 0.2}


def test_a_block_on_a_block_and_one_in_the_gripper() -> None:
    world = world_state({
        "red cube": pose(0.2, 0.0, CUBE / 2),
        "blue cube": pose(0.203, 0.002, CUBE * 1.5 + 0.003),
        "green cube": pose(0.18, 0.1, 0.08, source="w dloni"),
    })
    assert world.objects["blue cube"].resting_on == ("red cube",)
    assert world.objects["green cube"].resting_on == (GRIPPER,)
    assert world.above("red cube") == ["blue cube"] and world.above("blue cube") == []
    assert world.level("red cube") == 1 and world.level("blue cube") == 2


def test_a_bridge_rests_on_both_supports() -> None:
    world = world_state({
        "red cube": pose(0.2, 0.03, CUBE / 2),
        "green cube": pose(0.2, -0.03, CUBE / 2),
        "yellow cuboid": pose(0.2, 0.0, CUBE + CUBOID_H / 2, yaw_deg=90.0),
        "blue cube": pose(0.2, 0.0, CUBE + CUBOID_H + CUBE / 2),
    })
    cuboid = world.objects["yellow cuboid"]
    assert cuboid.resting_on == ("green cube", "red cube")
    assert cuboid.to_json()["resting_on"] == ["green cube", "red cube"]
    assert cuboid.to_json()["yaw_deg"] == 90
    assert world.objects["blue cube"].resting_on == ("yellow cuboid",)
    assert world.level("blue cube") == 3


def test_a_thin_block_under_a_cube_carries_it() -> None:
    world = world_state({"yellow cuboid": pose(0.2, 0.0, CUBOID_H / 2),
                         "red cube": pose(0.2, 0.0, CUBOID_H + CUBE / 2)})
    assert world.objects["red cube"].resting_on == ("yellow cuboid",)
    assert world.objects["yellow cuboid"].resting_on == (TABLE,)


def test_side_by_side_blocks_do_not_carry_each_other() -> None:
    world = world_state({"red cube": pose(0.2, 0.0, CUBE / 2),
                         "yellow cuboid": pose(0.2, 0.03, CUBOID_H / 2)})
    assert world.objects["red cube"].resting_on == (TABLE,)


def test_a_block_in_the_air_rests_on_nothing_known() -> None:
    world = world_state({"red cube": pose(0.2, 0.0, 0.1, source="ostatnie widziane", age=4.0)})
    red = world.objects["red cube"]
    assert red.resting_on == ()
    assert red.to_json()["resting_on"] is None and red.to_json()["age_s"] == 4.0


def test_an_unknown_kind_has_no_size() -> None:
    world = world_state({"blue pen": pose(0.2, 0.0, 0.005)})
    pen = world.objects["blue pen"]
    assert pen.kind is None and pen.spec is None
    assert pen.to_json()["size_mm"] is None and pen.to_json()["resting_on"] is None


def test_a_standing_cuboid_uses_its_rotation_for_height() -> None:
    upright = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # body x points up
    world = world_state({"yellow cuboid": pose(0.2, 0.0, 0.03, rotation=upright)})
    cuboid = world.objects["yellow cuboid"]
    assert cuboid.bottom_m == pytest.approx(0.0) and cuboid.top_m == pytest.approx(0.06)
    assert cuboid.resting_on == (TABLE,)
    xs = [p[0] for p in cuboid.footprint]
    assert max(xs) - min(xs) == pytest.approx(CUBOID_H)


@pytest.mark.parametrize(("label", "yaw_deg", "expected"), [
    ("yellow cuboid", 100.0, -80), ("red cube", 100.0, 10), ("red cube", -45.0, 45),
    ("green cylinder", 270.0, 90),
])
def test_yaw_is_reduced_by_the_block_symmetry(label: str, yaw_deg: float, expected: int) -> None:
    world = world_state({label: pose(0.2, 0.0, 0.0145, yaw_deg=yaw_deg)})
    assert world.objects[label].to_json()["yaw_deg"] == expected


def test_bad_poses_are_left_out() -> None:
    world = world_state({"red cube": pose(float("nan"), 0.0, 0.0145), "": pose(0.2, 0.0, 0.0145),
                         "green cube": ObjectPose(np.zeros(2), np.eye(3), "kamery", 0.0)})
    assert world.objects == {}


def test_the_json_holds_reach_build_area_catalog_and_objects_by_label() -> None:
    world = world_state({"red cube": pose(0.2, 0.0, CUBE / 2), "blue cube": pose(0.17, 0.05, CUBE / 2)})
    state = json.loads(json.dumps(world.to_json()))
    assert list(state) == ["reach", "build_area", "catalog", "objects"]
    assert state["reach"] == {"radius_m": [0.15, 0.25], "bearing_deg": [-46, 46]}
    assert [a["blocks_high"] for a in state["build_area"]] == [1, 2, 3]
    assert state["catalog"]["cuboid"] == {"size_mm": [60, 29, 14], "yaw_period_deg": 180}
    assert set(state["catalog"]) == {"cube", "cuboid", "cylinder"}
    assert list(state["objects"]) == ["blue cube", "red cube"]
    assert state["objects"]["red cube"]["x_m"] == 0.2


def test_reach_comes_from_the_desk() -> None:
    desk = Desk()
    reach = Reach.from_desk(desk)
    assert reach == Reach() and reach.radius_m == desk.cube_radius and reach.bearing_rad == desk.cube_bearing
    assert reach.contains(0.2, 0.0) and not reach.contains(0.3, 0.0) and not reach.contains(0.0, 0.2)


def test_kinds_come_from_the_shape_word() -> None:
    assert kind_of("Red Cube") == "cube" and kind_of("wooden cuboid") == "cuboid"
    assert kind_of("red cube 2") == "cube" and kind_of("light blue cylinder 12") == "cylinder"
    assert kind_of("cylinder") == "cylinder" and kind_of("cube red") == "cube"
    assert kind_of("yellow brick 3") == "cuboid"
    assert kind_of("red block 2") is None and kind_of("") is None


def test_numbered_blocks_keep_their_exact_keys_in_number_order() -> None:
    world = world_state({"red cube": pose(0.2, 0.0, CUBE / 2), "Red Cube 10": pose(0.17, 0.1, CUBE / 2),
                         "red cube 2": pose(0.2, 0.0, CUBE * 1.5),
                         "red  brick 3": pose(0.2, -0.1, CUBOID_H / 2)})
    assert world.get("red cube 2") is not None and world.get("red cube 2") is not world.get("red cube")
    assert world.objects["red cube 2"].kind == "cube" and world.objects["red brick 3"].kind == "cuboid"
    assert world.objects["red cube 2"].resting_on == ("red cube",)
    assert world.above("red cube") == ["red cube 2"] and world.level("red cube 2") == 2
    state = world.to_json()
    assert list(state["objects"]) == ["red brick 3", "red cube", "red cube 2", "red cube 10"]
    assert state["objects"]["red cube 2"]["resting_on"] == "red cube"


def lower_further_out(x: float, y: float, z: float, block: BlockSpec) -> bool:
    """A fake arm: sets blocks down up to 0.25 m out on the table, and 1 cm less out per cm of height."""
    return x <= 0.25 - z


def test_the_build_area_shrinks_with_height() -> None:
    area = find_build_area(lower_further_out, Reach(), 3)
    assert [box.levels for box in area if box] == [1, 2, 3]
    first, second, third = (box for box in area if box)
    assert third.x_m == (0.15, 0.18) and third.y_m == (-0.15, 0.15)   # the largest box inside the band
    assert first.x_m == second.x_m == (0.15, 0.2)                      # the largest that holds the next
    for low, high in ((first, second), (second, third)):
        assert low.x_m[0] <= high.x_m[0] <= high.x_m[1] <= low.x_m[1]
        assert low.y_m[0] <= high.y_m[0] <= high.y_m[1] <= low.y_m[1]
    assert first.to_json() == {"blocks_high": 1, "x_m": [0.15, 0.2], "y_m": [-0.15, 0.15]}
    assert third.text() == "a structure 3 high fits in x 0.15..0.18 m, y -0.15..0.15 m"
    assert find_build_area(lambda x, y, z, block: True, Reach(), 1)[0] == Box(1, (0.15, 0.21), (-0.13, 0.13))


def test_the_build_area_leaves_room_for_blocks_that_stand_a_little_high() -> None:
    """Levels above the table are checked a margin higher, so a slightly tall stack still fits its box."""
    asked: set[float] = set()

    def record(x: float, y: float, z: float, block: BlockSpec) -> bool:
        asked.add(round(z, 4))
        return lower_further_out(x, y, z, block)

    find_build_area(record, Reach(), 3)
    assert asked == {0.0, round(CUBE + HEIGHT_MARGIN_M, 4), round(2 * CUBE + HEIGHT_MARGIN_M, 4)}
    assert level_bottom(1, HEIGHT_MARGIN_M) == 0.0 and level_bottom(3) == pytest.approx(2 * CUBE)
    exact = find_build_area(lower_further_out, Reach(), 3, margin_m=0.0)[2]
    assert exact is not None and exact.x_m == (0.15, 0.19)          # without the margin: 1 cm further out
    third = find_build_area(lower_further_out, Reach(), 3)[2]
    assert third is not None and third.span() == "x 0.15..0.18 m, y -0.15..0.15 m"
    higher = reach_box(lower_further_out, third, 0.075, CUBE_SPEC)
    assert higher == Box(3, (0.15, 0.17), (-0.15, 0.15))
    assert reach_box(lower_further_out, third, 0.11, CUBE_SPEC) is None


def test_a_height_out_of_reach_has_no_box() -> None:
    world = world_state({}, place_check=lambda x, y, z, block: z < 0.04)
    assert [box is None for box in world.build_area] == [False, False, True]
    assert world.box(3) is None and world.box(4) is None and isinstance(world.box(2), Box)
    assert world.to_json()["build_area"][2] == {"blocks_high": 3, "x_m": None, "y_m": None}
    assert len(world_state({}, place_check=lower_further_out, levels=4).build_area) == 4


def test_the_arm_model_gives_a_smaller_box_three_high() -> None:
    first, second, third = world_state({}).build_area
    assert first is not None and second is not None and third is not None
    assert first.x_m[0] < third.x_m[0] and third.x_m[1] <= first.x_m[1]


def test_a_free_spot_avoids_blocks_and_respects_the_height() -> None:
    blocks = {"red cube": pose(0.17, 0.0, CUBE / 2), "green cube": pose(0.2, 0.06, CUBE / 2)}
    world = world_state(blocks, place_check=lower_further_out)
    spot = best_spot(world, 3, (0.03, 0.12))
    assert spot is not None
    x, y = spot
    outline = rectangle(x, y, 0.0, (0.015, 0.06))
    assert all(-overlap(outline, o.footprint) >= 0.01 for o in world.objects.values())
    assert x <= 0.18                                   # the third level sets down 58 mm up, plus the margin
    assert best_spot(world, 3, (0.12, 0.03), yaw_rad=math.pi / 2) == spot
    assert best_spot(world, 3, (0.03, 0.12), yaw_rad=math.pi / 2) is None   # 12 cm deep: no room
    alone = best_spot(world_state({}, place_check=lower_further_out), 1, (0.03, 0.03))
    assert alone is not None and alone[1] == 0.0 and 0.17 <= alone[0] <= 0.18   # the middle of the box
    assert best_spot(world, 2, (0.3, 0.3)) is None


def test_geometry_helpers() -> None:
    assert reduce_yaw(math.pi / 2, math.pi) == pytest.approx(math.pi / 2)
    assert reduce_yaw(-math.pi / 2, math.pi) == pytest.approx(math.pi / 2)
    assert reduce_yaw(0.3 + 2 * math.pi, math.pi / 2) == pytest.approx(0.3)
    a = rectangle(0.0, 0.0, 0.0, (0.01, 0.01))
    assert overlap(a, rectangle(0.015, 0.0, 0.0, (0.01, 0.01))) == pytest.approx(0.005)
    assert overlap(a, rectangle(0.03, 0.0, 0.0, (0.01, 0.01))) == pytest.approx(-0.01)
