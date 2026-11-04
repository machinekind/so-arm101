"""The approach table: its file, merging runs, the lookup and its misses, the repeat spreads and the preferred
direction. numpy only."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from soarm_lab.approach_table import (
    FORMAT,
    MISS_DIRECTION,
    MISS_HEIGHT,
    MISS_OUTSIDE,
    MISS_STALE,
    MISS_TABLE,
    VERSION,
    ApproachTable,
    Hit,
    Miss,
    mls,
    outside_m,
    summary,
    write_atomic,
)

BOARD = {"squares": [3, 5], "square_m": 0.045, "marker_m": 0.035, "dictionary": "DICT_6X6_250",
         "thickness_m": 0.0}
LEVELS = [{"surface_m": 0.0, "tcp_z_m": 0.022}, {"surface_m": 0.029, "tcp_z_m": 0.051}]
#: Corners of a board lying near (0.2, 0), long side across the arm [m].
CORNERS = [(0.1775 + 0.045 * (k % 2), -0.0675 + 0.045 * (k // 2)) for k in range(8)]


def field(xy) -> np.ndarray:
    """An affine correction field [m]."""
    x, y = xy
    return np.array([0.004 + 0.05 * (x - 0.2) - 0.02 * y, -0.006 + 0.01 * (x - 0.2) + 0.04 * y])


def table(samples=None, *, complete=True, robot="so101") -> ApproachTable:
    t = ApproachTable.new(robot=robot, stand="desk", board=BOARD, protocol={"lead_m": 0.03},
                          wrist={"size": [1920, 1080]}, levels=LEVELS)
    t.data["samples"] = list(samples or [])
    t.data["complete"] = complete
    return t


def sample(corner, level, direction, repeat, delta, spot=1) -> dict:
    return {"spot": spot, "corner": corner, "xy_m": list(CORNERS[corner]), "level": level,
            "direction": direction, "repeat": repeat, "delta_m": [*delta, 0.0]}


def affine_samples(level=0, direction="ccw", repeats=2) -> list[dict]:
    return [sample(k, level, direction, r, field(CORNERS[k]))
            for k in range(8) for r in range(1, repeats + 1)]


# --------------------------------------------------------------------------- the file
def test_a_table_goes_to_a_new_folder_and_comes_back(tmp_path):
    t = table(affine_samples())
    path = t.save(tmp_path / "stands" / "desk" / "approach.json")
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    back = ApproachTable.load(path)
    assert back.data["format"] == FORMAT and back.data["version"] == VERSION
    assert back.samples == json.loads(json.dumps(t.samples))
    a, b = back.lookup(CORNERS[3], 0.022, "ccw"), t.lookup(CORNERS[3], 0.022, "ccw")
    assert isinstance(a, Hit) and isinstance(b, Hit) and a.delta_xy == pytest.approx(b.delta_xy)


def test_a_write_that_fails_leaves_the_old_file(tmp_path):
    path = write_atomic(tmp_path / "t.json", {"a": 1})
    with pytest.raises(TypeError):
        write_atomic(path, {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1} and not list(tmp_path.glob("*.tmp"))


def test_numpy_values_and_nan_become_plain_json(tmp_path):
    path = write_atomic(tmp_path / "t.json", {"v": np.array([1.0, math.nan]), "f": np.float32(0.5)})
    assert json.loads(path.read_text()) == {"v": [1.0, None], "f": 0.5}


@pytest.mark.parametrize("text, words", [
    ("{", "not JSON"),
    ('{"format": "other"}', "not an approach table"),
    (json.dumps({"format": FORMAT, "version": 99}), "version 99"),
    (json.dumps({"format": FORMAT, "version": VERSION, "levels": []}), "without 'samples'"),
])
def test_a_file_that_is_no_table_is_refused(tmp_path, text, words):
    (tmp_path / "t.json").write_text(text)
    with pytest.raises(ValueError, match=words):
        ApproachTable.load(tmp_path / "t.json")


def test_a_new_spot_merges_only_into_a_table_of_the_same_robot_board_frame_and_levels():
    t = table(affine_samples())
    t.data["spots"] = [{"id": 1}, {"id": 4}]
    assert t.next_spot() == 5
    ok = dict(robot="so101", board=BOARD, size=(1920, 1080), levels=LEVELS)
    assert t.merge_problem(**ok) == ""
    assert "robot" in t.merge_problem(**{**ok, "robot": "a1x"})
    assert "tablica" in t.merge_problem(**{**ok, "board": {**BOARD, "square_m": 0.04}})
    assert "1920x1080" in t.merge_problem(**{**ok, "size": (1280, 720)})
    assert "poziomy" in t.merge_problem(**{**ok, "levels": LEVELS[:1]})


# --------------------------------------------------------------------------- the lookup
def test_an_affine_field_comes_back_inside_the_corners():
    t = table(affine_samples())
    for xy in [(0.19, -0.03), (0.21, 0.05), CORNERS[5]]:
        hit = t.lookup(xy, 0.022, "ccw")
        assert isinstance(hit, Hit) and hit.level == 0 and hit.n == 8
        assert hit.delta_xy == pytest.approx(field(xy), abs=5e-5)
        assert hit.rms_m < 1e-4


def test_repeats_are_averaged_per_corner():
    samples = [sample(k, 0, "cw", 1, (0.001, 0.0)) for k in range(8)]
    samples += [sample(k, 0, "cw", 2, (0.003, 0.0)) for k in range(8)]
    hit = table(samples).lookup((0.2, 0.0), 0.022, "cw")
    assert isinstance(hit, Hit) and hit.n == 8 and hit.delta_xy == pytest.approx([0.002, 0.0], abs=1e-9)


def test_the_nearest_level_answers_within_its_tolerance():
    t = table(affine_samples(level=0) + [sample(k, 1, "ccw", 1, (0.0, 0.0)) for k in range(8)])
    assert t.lookup((0.2, 0.0), 0.030, "ccw").level == 0
    assert t.lookup((0.2, 0.0), 0.045, "ccw").level == 1
    miss = t.lookup((0.2, 0.0), 0.08, "ccw")
    assert isinstance(miss, Miss) and miss.reason == MISS_HEIGHT


def test_the_misses_say_why():
    t = table(affine_samples() + [sample(k, 0, "out", 1, (0.0, 0.0)) for k in (0, 1)])
    assert t.lookup((0.2, 0.0), 0.022, "out").reason == MISS_DIRECTION
    assert t.lookup((0.2, 0.0), 0.022, "in").reason == MISS_DIRECTION
    assert isinstance(t.lookup((0.2, 0.09), 0.022, "ccw"), Hit)
    assert t.lookup((0.2, 0.11), 0.022, "ccw").reason == MISS_OUTSIDE
    assert table(affine_samples(), complete=False).lookup((0.2, 0.0), 0.022, "ccw").reason == MISS_TABLE
    assert t.lookup((0.2, 0.0), 0.022, "ccw", robot="a1x").reason == MISS_TABLE
    t.data["version"] = 2
    assert t.lookup((0.2, 0.0), 0.022, "ccw").reason == MISS_TABLE


def test_an_old_table_is_stale_only_with_a_limit():
    t = table(affine_samples())
    t.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 10 * 86400))
    assert t.age_days() == pytest.approx(10.0, abs=0.01)
    assert isinstance(t.lookup((0.2, 0.0), 0.022, "ccw"), Hit)
    assert isinstance(t.lookup((0.2, 0.0), 0.022, "ccw", max_age_days=30), Hit)
    assert t.lookup((0.2, 0.0), 0.022, "ccw", max_age_days=7).reason == MISS_STALE


def test_outside_the_hull_is_measured_to_its_nearest_edge():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    assert outside_m((0.5, 0.2), square) == 0.0
    assert outside_m((1.5, 0.5), square) == pytest.approx(0.5)
    assert outside_m((2.0, 2.0), square) == pytest.approx(math.sqrt(2))
    assert outside_m((0.5, 1.0), np.array([[0.0, 0.0], [1.0, 0.0]])) == pytest.approx(1.0)


def test_points_on_one_line_still_give_a_value_without_a_wild_gradient():
    points = np.array([[0.2, y] for y in (-0.05, 0.0, 0.05)])
    values = np.array([[0.001, 0.0], [0.002, 0.0], [0.003, 0.0]])
    value, rms = mls((0.21, 0.0), points, values)
    assert value == pytest.approx([0.002, 0.0], abs=1e-5) and rms < 5e-5


# --------------------------------------------------------------------------- spreads
def test_the_preferred_direction_has_the_smallest_repeat_spread():
    samples = affine_samples(direction="ccw")
    for k in range(8):
        samples += [sample(k, 0, "out", 1, (0.0, 0.0)), sample(k, 0, "out", 2, (0.002 * (k % 2), 0.0))]
        samples += [sample(k, 0, "cw", 1, (0.0, 0.0)), sample(k, 0, "cw", 2, (0.0, 0.0005))]
    samples.append(sample(0, 0, "in", 0, (0.0, 0.0)))
    t = table(samples)
    t.summarize()
    assert t.data["spread_m"] == pytest.approx({"out": 0.001, "ccw": 0.0, "cw": 0.0005})
    assert t.preferred_direction == "ccw"
    assert t.data["spread_by_level"] == {"0": pytest.approx({"out": 0.001, "ccw": 0.0, "cw": 0.0005})}


def test_a_tie_of_median_spreads_goes_to_the_smaller_mean():
    samples = []
    for k in range(8):
        # "out" differs at a few corners only: its median spread is zero, like that of "cw".
        off = 0.003 if k < 3 else 0.0
        samples += [sample(k, 0, "out", 1, (0.0, 0.0)), sample(k, 0, "out", 2, (off, 0.0))]
        samples += [sample(k, 0, "cw", 1, (0.0, 0.0)), sample(k, 0, "cw", 2, (0.0, 0.0))]
    t = table(samples)
    t.summarize()
    assert t.data["spread_m"]["out"] == 0.0 and t.preferred_direction == "cw"


def test_the_summary_reads_like_a_report():
    t = table(affine_samples())
    t.summarize()
    t.data["wrist"] = {"camera": "wrist", "size": [1920, 1080], "K": np.eye(3).tolist(), "rms_px": 0.3,
                       "views": 16, "mount_bias_m": {"0": [0.001, -0.002]},
                       "suggested": {"focal_px": 1200.0, "tip_pixel": [1150.0, 830.0]}}
    text = summary(t)
    assert "preferred direction: ccw" in text and "mount bias [mm]: level 1 (1.00, -2.00)" in text
    assert "focal_px 1200.0" in text and "level 2: surface 29 mm" in text
