"""The hand-down place check on the SO-101 model: height, distance, the first joint's limits and turns,
and a grid that never claims more reach than the arm has."""

from __future__ import annotations

import math

import numpy as np
import pytest

from soarm_lab.blocks.classic_world import BLOCKS
from soarm_lab.planner.placing import CARRY_CLEAR_M, HEADINGS, PLACE_GAP_M, arm_reach
from soarm_lab.strategies.grasp import MAX_ROT_ERR_DEG, top_down

CUBE = BLOCKS["cube"]
LEVEL = 0.029


@pytest.mark.parametrize(("x", "y", "levels_up", "expected"), [
    (0.20, 0.0, 0, True), (0.20, 0.0, 2, True),
    (0.155, 0.0, 0, True), (0.155, 0.0, 2, False),          # close to the base only low
    (0.17, 0.0, 1, True), (0.17, 0.0, 2, False),
    (0.30, 0.0, 0, True), (0.30, 0.0, 2, False),            # far out only low
    (0.20, 0.0, 3, False),                                  # four high nowhere with the hand down
    (0.12, 0.2, 0, True), (-0.05, 0.2, 0, False),           # the first joint turns 68 deg, not 114
])
def test_where_the_arm_sets_a_cube_down(x: float, y: float, levels_up: int, expected: bool) -> None:
    assert arm_reach()(x, y, levels_up * LEVEL, CUBE) is expected


def test_a_flat_cuboid_hangs_lower_in_the_hand() -> None:
    reach = arm_reach()
    assert reach._grip(BLOCKS["cuboid"])[1] < reach._grip(CUBE)[1]
    assert reach(0.20, 0.0, LEVEL, BLOCKS["cuboid"])


def test_reach_turns_with_the_first_joint() -> None:
    """Off the x axis, full IK at the place itself agrees with the grid worked out on the axis."""
    reach = arm_reach()
    kin = reach.kin
    cmd, hang = reach._grip(CUBE)
    seed = {**kin.spec.home, kin.spec.gripper or "": cmd}
    for x, y, z in ((0.12, 0.2, 0.0), (0.15, -0.17, 0.0), (0.19, 0.12, 2 * LEVEL), (0.10, 0.25, 2 * LEVEL),
                    (0.24, 0.1, 2 * LEVEL)):
        bearing = math.atan2(y - reach.axis_xy[1], x - reach.axis_xy[0])
        direct = False
        for heading in HEADINGS:
            hits = []
            for tcp_z in (z + PLACE_GAP_M + hang, z + CARRY_CLEAR_M + hang):
                sol = kin.ik(np.array([x, y, tcp_z]), top_down(bearing + heading), seed=seed)
                hits.append(sol.ok and sol.rot_err <= math.radians(MAX_ROT_ERR_DEG)
                            and not kin.self_contacts(sol.joints))
            direct = direct or all(hits)
        assert reach(x, y, z, CUBE) is direct, (x, y, z)


def test_the_grid_never_claims_more_reach_than_the_arm_has() -> None:
    """Near the third level's limit, every place the grid allows also passes IK at its exact distance and
    heights, and the grid allows most of them."""
    reach = arm_reach()
    cmd, hang = reach._grip(CUBE)
    allowed = 0
    for x in (0.165, 0.175, 0.185, 0.205, 0.225, 0.245):
        distance = x - float(reach.axis_xy[0])
        for z in np.arange(0.054, 0.066, 0.0015):
            if not reach(x, 0.0, float(z), CUBE):
                continue
            allowed += 1
            heights = (z + PLACE_GAP_M + hang, z + CARRY_CLEAR_M + hang)
            solved = [all(reach._solve(h, distance, t, cmd) is not None for t in heights) for h in HEADINGS]
            assert any(solved), (x, z)
    assert allowed >= 20


def test_answers_come_from_the_cache() -> None:
    reach = arm_reach()
    reach(0.21, 0.05, LEVEL, CUBE)
    cells = len(reach._cells)
    assert reach(0.21, 0.05, LEVEL, CUBE) and len(reach._cells) == cells
