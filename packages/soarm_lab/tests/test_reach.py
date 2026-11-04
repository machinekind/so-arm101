"""Test where the place reach helper lets the SO-101 put a cube, straight down or tilted."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.reach import Reach, can_place, reach_map  # noqa: E402

#: A cube's height [m].
CUBE_H = 2 * block("cube").half_size[2]


def test_a_second_cube_fits_straight_down_and_higher_ones_need_the_hand_tilted():
    second = can_place("cube", 0.22, 0.0, CUBE_H)
    third = can_place("cube", 0.24, 0.0, 2 * CUBE_H)
    fourth = can_place("cube", 0.21, 0.08, 3 * CUBE_H)

    assert second.ok and second.tilt_deg == 0.0
    assert third.ok and 0.0 < third.tilt_deg <= 30.0
    assert not can_place("cube", 0.24, 0.0, 2 * CUBE_H, max_tilt_deg=0.0).ok
    assert fourth.ok and third.tilt_deg < fourth.tilt_deg <= 30.0
    # A tilted hand closes square to the arm.
    pan = Reach().pan_xy
    bearing = math.atan2(0.08 - pan[1], 0.21 - pan[0])
    assert abs(math.cos(math.radians(fourth.heading_deg) - bearing)) < 1e-9
    sixth = can_place("cube", 0.2, 0.0, 5 * CUBE_H)
    assert not sixth.ok and math.isnan(sixth.tilt_deg) and math.isnan(sixth.heading_deg)


def test_a_tilted_place_lines_up_with_a_yaw_only_near_square_to_the_arm():
    square = can_place("cube", 0.24, 0.0, 2 * CUBE_H, yaw_deg=0.0)
    assert square.ok and square.tilt_deg == can_place("cube", 0.24, 0.0, 2 * CUBE_H).tilt_deg
    assert not can_place("cube", 0.24, 0.0, 2 * CUBE_H, yaw_deg=40.0).ok


def test_the_map_spreads_each_distance_from_the_base_around_it():
    reach = Reach()
    m = reach.map("cube", 2 * CUBE_H, x_m=(0.2, 0.28), y_m=(-0.04, 0.04), step_m=0.04)

    assert m.x_m == pytest.approx([0.2, 0.24, 0.28]) and m.y_m == pytest.approx([-0.04, 0.0, 0.04])
    assert m.tilt_deg.shape == (3, 3)
    # Spots as far from the base's turning axis share a value, which `place` gives for that distance.
    assert np.array_equal(m.tilt_deg[0], m.tilt_deg[2], equal_nan=True)
    r = round(float(np.linalg.norm(np.array([0.24, 0.0]) - reach.pan_xy)) / 0.04) * 0.04
    assert m.tilt_deg[1, 1] == reach.place("cube", *(reach.pan_xy + np.array([r, 0.0])), 2 * CUBE_H).tilt_deg
    assert 0.0 < m.tilt_deg[1, 1] <= 30.0
    behind = reach_map("cube", 0.0, x_m=(-0.12, -0.12), y_m=(0.0, 0.0))
    assert behind.tilt_deg.shape == (1, 1) and np.isnan(behind.tilt_deg[0, 0])
