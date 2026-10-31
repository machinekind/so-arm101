"""Grasp libraries without the twin and without IK: the block meshes, the frame mapping between GraspGenX and
the TCP, the converter of GraspGenX's output, the loader's checks, and the composition with a pose and the
selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab.grasp_library import (  # noqa: E402
    GRASPGENX_APPROACH,
    GRASPGENX_CLOSING,
    ComposedGrasp,
    GraspLibrary,
    block_width,
    compose,
    convert_isaac_grasps,
    graspgenx_frame,
    load_library,
    quat_wxyz_to_mat,
    read_gripper_config,
    rejections_text,
    roll_axis,
    rot_z,
    save_library,
    select,
    wrap,
    write_block_obj,
)
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.paths import REPO_ROOT, source_checkout  # noqa: E402
from soarm_lab.rl.objects import KINDS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402

KIN = RobotKinematics(load_robot())
#: Rx(180 deg): the URDF root (the MJCF `gripper` body) to GraspGenX's aligned frame.
BASE_ROTATION = np.diag([1.0, -1.0, -1.0, 1.0])
#: The centre of the open sweep box in the aligned frame [m]; a test pins it against config.json.
PAD_POINT = np.array([0.0344, 0.0, 0.0736])
CONFIG = REPO_ROOT / "tools/graspgenx/so101/config.json"
HAND = {"name": "hand", "version": "", "commit": "", "gripper": "so101", "checkpoint": "",
        "date": "2026-10-31", "note": ""}
CUBE = object_spec("cube").half_size
#: The selection of the spec's example: the 8 deg tilt, the SO-101's jaw travel, the default margins.
SELECT = {"max_tilt_rad": math.radians(8.0), "gap_min_m": 0.009, "gap_max_m": 0.088, "open_margin_m": 0.008,
          "squeeze_m": 0.008, "half_size": CUBE}


def isaac(entries: list[tuple[float, list[float], float, list[float]]]) -> dict:
    """An isaac_grasp file's content from (confidence, position, w, xyz) tuples."""
    return {"format": "isaac_grasp", "format_version": 1.0,
            "grasps": {f"grasp_{i}": {"confidence": c, "position": p, "orientation": {"w": w, "xyz": xyz}}
                       for i, (c, p, w, xyz) in enumerate(entries)}}


def gg_pose(approach, closing, pad) -> tuple[np.ndarray, np.ndarray]:
    """GraspGenX's pose for a grasp given in the TCP's terms: position P - R @ PAD_POINT with R = [closing,
    approach x closing, approach] (columns), and its rotation."""
    a, c = np.asarray(approach, float), np.asarray(closing, float)
    R = np.column_stack([c, np.cross(a, c), a])
    return np.asarray(pad, float) - R @ PAD_POINT, R


def example_yaml() -> Path:
    needs_checkout()
    return REPO_ROOT / "tools/graspgenx/examples/cube.yml"


def needs_checkout() -> None:
    """tools/graspgenx lies beside the package only in a checkout of the repository."""
    if not source_checkout():
        pytest.skip("needs a source checkout (tools/graspgenx)")


def entry(pad, approach, closing, width=0.025, score=0.9) -> dict:
    return {"position": list(pad), "approach": list(approach), "closing": list(closing), "width_m": width,
            "score": score}


def library(kind: str, grasps: list[dict]) -> GraspLibrary:
    return GraspLibrary.from_dict({"format": "soarm-lab-grasps", "version": 1, "kind": kind,
                                   "half_size": list(object_spec(kind).half_size), "generator": HAND,
                                   "grasps": grasps})


def degrees(entries: list[ComposedGrasp]) -> list[float]:
    return [round(math.degrees(e.psi), 6) % 360 for e in entries]


# ------------------------------------------------------------ the block meshes
@pytest.mark.parametrize("kind", KINDS)
def test_the_mesh_writer_puts_each_block_in_its_body_frame(kind: str, tmp_path):
    path = write_block_obj(kind, tmp_path / f"{kind}.obj")
    lines = path.read_text().splitlines()
    vertices = np.array([[float(x) for x in line.split()[1:]] for line in lines if line.startswith("v ")])
    faces = np.array([[int(x) for x in line.split()[1:]] for line in lines if line.startswith("f ")])
    half = np.asarray(object_spec(kind).half_size)

    assert lines[0].startswith(f"# soarm-lab block {kind}: ")
    assert vertices.min(axis=0) == pytest.approx(-half, abs=1e-9)
    assert vertices.max(axis=0) == pytest.approx(half, abs=1e-9)
    assert len(faces) == (256 if kind == "cylinder" else 12)
    assert faces.min() == 1 and faces.max() == len(vertices)
    # Counter-clockwise seen from outside: the triangles enclose a positive volume, the block's.
    v = vertices[faces - 1]
    volume = float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6
    exact = 8 * half.prod() if kind != "cylinder" else math.pi * half[1] ** 2 * 2 * half[0]
    assert volume == pytest.approx(exact, rel=0.01)
    if kind == "cylinder":
        assert lines[0] == "# soarm-lab block cylinder: 50.0 x 25.0 x 25.0 mm (placeholder: MEASURED = False)"


def test_block_width_is_the_support_width():
    diagonal = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    cube, cuboid, cylinder = (object_spec(k) for k in ("cube", "cuboid5", "cylinder"))

    assert block_width(cube, (1.0, 0.0, 0.0)) == pytest.approx(0.025, abs=1e-9)
    assert block_width(cube, diagonal) == pytest.approx(0.035355, abs=1e-6)
    assert block_width(cuboid, (0.0, 1.0, 0.0)) == pytest.approx(0.025, abs=1e-9)
    assert block_width(cuboid, (1.0, 0.0, 0.0)) == pytest.approx(0.05, abs=1e-9)
    assert block_width(cylinder, (0.0, 1.0, 0.0)) == pytest.approx(0.025, abs=1e-9)
    assert block_width(cylinder, (0.0, 0.0, 1.0)) == pytest.approx(0.025, abs=1e-9)
    assert block_width(cylinder, (1.0, 0.0, 0.0)) == pytest.approx(0.05, abs=1e-9)
    assert block_width(cylinder, np.array([0.0, 1.0, 1.0]) / math.sqrt(2)) == pytest.approx(0.025, abs=1e-9)


def test_roll_axis_is_the_cylinders_and_none_for_a_box():
    axis = roll_axis(object_spec("cylinder"))

    assert axis is not None and axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert all(roll_axis(object_spec(kind)) is None for kind in ("cube", "cuboid5", "cuboid85"))


# ------------------------------------------------------------ the frame mapping
def test_the_graspgenx_frame_of_the_so101_is_a_proper_rotation():
    T = graspgenx_frame(KIN, BASE_ROTATION)
    R = T[:3, :3]

    assert T == pytest.approx(np.array([[0, 0, 1, 0.0120], [0, -1, 0, 0.0002], [1, 0, 0, 0.0981],
                                        [0, 0, 0, 1]], float), abs=1e-4)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    assert R @ np.array([1.0, 0.0, 0.0]) == pytest.approx(GRASPGENX_APPROACH, abs=1e-9)
    assert R @ np.array([0.0, 0.0, 1.0]) == pytest.approx(GRASPGENX_CLOSING, abs=1e-9)


def test_the_so101_config_names_the_pad_point_the_tests_use():
    needs_checkout()
    cfg = read_gripper_config(CONFIG)
    raw = json.loads(CONFIG.read_text())

    assert cfg.pad_point == pytest.approx(PAD_POINT, abs=1e-9)
    assert cfg.base_rotation == pytest.approx(BASE_ROTATION)
    assert cfg.type == "revolute_2f" and cfg.symmetric is False
    # As GraspGenX's wizard writes it: the fingertip is the open box's centre, the standoff half its depth.
    assert cfg.fingertip == pytest.approx(PAD_POINT, abs=1e-9)
    assert raw["standoff"] == pytest.approx([0.0, raw["sweep_volume"]["extents"][2] / 2])
    assert set(raw) == {"open", "close", "fingertip", "sweep_volume", "links", "standoff", "symmetric",
                        "type", "bbox", "base_rotation"}


def test_a_gripper_config_needs_a_proper_base_rotation(tmp_path):
    needs_checkout()
    raw = json.loads(CONFIG.read_text())
    raw["base_rotation"] = np.diag([1.0, -1.0, 1.0, 1.0]).tolist()
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="base_rotation is not a proper rotation"):
        read_gripper_config(bad)
    with pytest.raises(FileNotFoundError, match="gripper config not found"):
        read_gripper_config(tmp_path / "nope.json")


# ------------------------------------------------------------ the converter
def test_the_converter_maps_a_top_down_grasp_to_the_pad_point():
    frame = graspgenx_frame(KIN, BASE_ROTATION)
    lib = convert_isaac_grasps(isaac([(0.92, [0.0, -0.0344, 0.0736], 0.0, [0.70710678, 0.70710678, 0.0]),
                                      (0.80, [0.0736, -0.0344, 0.0], 0.5, [-0.5, -0.5, 0.5])]),
                               "cube", frame=frame, pad_point=PAD_POINT, generator=HAND)
    top, side = lib.grasps

    assert top.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert top.approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert top.closing == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert top.width_m == pytest.approx(0.025) and top.score == 0.92
    assert side.position == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert side.approach == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert side.closing == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert lib.generator["dropped"] == "0 of 2"

    # Every grasp of the hand-written example is GraspGenX's pose of its approach, closing and pad point.
    import yaml

    data = yaml.safe_load(example_yaml().read_text())
    converted = convert_isaac_grasps(data, "cube", frame=frame, pad_point=PAD_POINT, generator=HAND)
    for grasp in converted.grasps:
        position, R = gg_pose(grasp.approach, grasp.closing, grasp.position)
        listed = [g for g in data["grasps"].values() if g["confidence"] == grasp.score]
        assert len(listed) == 1
        assert position == pytest.approx(listed[0]["position"], abs=1e-6)
        w, xyz = listed[0]["orientation"]["w"], listed[0]["orientation"]["xyz"]
        assert quat_wxyz_to_mat([w, *xyz]) == pytest.approx(R, abs=1e-6)


def test_the_converter_sorts_by_score_drops_grasps_outside_the_block_and_refuses_other_formats():
    frame = graspgenx_frame(KIN, BASE_ROTATION)
    down_y = (0.0, [0.70710678, 0.70710678, 0.0])
    deep, _ = gg_pose((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (0.0, 0.0, -0.02))
    data = isaac([(0.5, [0.0, -0.0344, 0.0736], *down_y), (0.9, [0.0, -0.0344, 0.0786], *down_y),
                  (0.95, list(deep), *down_y)])
    lib = convert_isaac_grasps(data, "cube", frame=frame, pad_point=PAD_POINT, generator=HAND)

    assert [g.score for g in lib.grasps] == [0.9, 0.5]
    assert lib.grasps[0].position == pytest.approx((0.0, 0.0, 0.005), abs=1e-6)
    assert lib.generator["dropped"] == "1 of 3" and lib.generator["name"] == "hand"
    with pytest.raises(ValueError, match="isaac_grasp"):
        convert_isaac_grasps({"format": "other", "grasps": {}}, "cube", frame=frame, pad_point=PAD_POINT,
                             generator=HAND)
    with pytest.raises(ValueError, match="grasp_0"):
        convert_isaac_grasps(isaac([(0.9, [0.0, 0.0, 0.0], 0.0, [0.5, 0.0, 0.0])]), "cube", frame=frame,
                             pad_point=PAD_POINT, generator=HAND)


def test_the_bundled_cube_library_is_the_conversion_of_the_example():
    import yaml

    needs_checkout()
    cfg = read_gripper_config(CONFIG)
    converted = convert_isaac_grasps(yaml.safe_load(example_yaml().read_text()), "cube",
                                     frame=graspgenx_frame(KIN, cfg.base_rotation), pad_point=cfg.pad_point,
                                     generator=HAND)
    bundled = load_library("cube")

    assert bundled.generator["name"] == "hand" and bundled.generator["dropped"] == "0 of 8"
    assert len(converted.grasps) == len(bundled.grasps) == 8
    for mine, theirs in zip(converted.grasps, bundled.grasps, strict=True):
        for name in ("position", "approach", "closing"):
            assert getattr(mine, name) == pytest.approx(getattr(theirs, name), abs=1e-6)
        assert mine.width_m == pytest.approx(theirs.width_m, abs=1e-6) and mine.score == theirs.score
    assert [g.score for g in bundled.grasps] == [0.92, 0.91, 0.9, 0.89, 0.85, 0.8, 0.7, 0.55]


# ------------------------------------------------------------ the loader
GOOD = entry((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


@pytest.mark.parametrize("change, match", [
    ({"kind": "ball"}, "unknown kind 'ball'"),
    ({"version": 2}, "not a grasp library of version 1"),
    ({"half_size": [0.015, 0.0125, 0.0125]}, r"made for a 30.0 x 25.0 x 25.0 mm block.*regenerate"),
    ({"grasps": [GOOD, dict(GOOD, approach=[0.0, 0.0, -1.001])]}, "grasp 1: approach is not a unit vector"),
    ({"grasps": [dict(GOOD, closing=[0.0, 0.99995, 0.01])]}, "grasp 0: approach and closing are not perp"),
    ({"grasps": [dict(GOOD, position=[0.0, 0.0, 0.0145])]}, "grasp 0: the pad point .* lies outside"),
    ({"grasps": [dict(GOOD, width_m=0.0)]}, "grasp 0: width_m must be positive"),
    ({"grasps": [dict(GOOD, score=float("nan"))]}, "grasp 0: score must be finite"),
    ({"grasps": [dict(GOOD, position=[0.0, 0.0])]}, "grasp 0: position is not three finite numbers"),
])
def test_a_library_is_validated(change: dict, match: str):
    data = {"format": "soarm-lab-grasps", "version": 1, "kind": "cube", "half_size": list(CUBE),
            "generator": HAND, "grasps": [GOOD]}
    assert len(GraspLibrary.from_dict(data).grasps) == 1

    with pytest.raises(ValueError, match=match):
        GraspLibrary.from_dict({**data, **change})


def test_a_library_round_trips_an_oblique_grasp_through_the_file(tmp_path):
    R = quat_wxyz_to_mat((0.8, 0.3, -0.4, 0.33))
    grasp = entry((0.003, -0.004, 0.002), R[:, 0], R[:, 2], width=0.0301, score=0.42)
    path = save_library(library("cube", [grasp]), tmp_path / "cube.json")
    loaded = load_library("cube", tmp_path).grasps[0]

    assert path == tmp_path / "cube.json"
    assert loaded.approach == pytest.approx(R[:, 0], abs=2e-6)
    assert loaded.closing == pytest.approx(R[:, 2], abs=2e-6)
    assert loaded.position == pytest.approx((0.003, -0.004, 0.002), abs=1e-9)
    a, c = loaded.axes()
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-12)
    assert abs(float(a @ c)) < 1e-12
    assert np.linalg.det(loaded.rotation()) == pytest.approx(1.0, abs=1e-12)
    assert loaded.rotation()[:, 0] == pytest.approx(a) and loaded.rotation()[:, 2] == pytest.approx(c)


def test_load_library_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"cuboid5\.json.*tools/graspgenx"):
        load_library("cuboid5")
    with pytest.raises(FileNotFoundError, match=r"cube\.json.*tools/graspgenx"):
        load_library("cube", tmp_path)
    save_library(library("cube", [GOOD]), tmp_path / "cylinder.json")
    with pytest.raises(ValueError, match="holds the grasp library of 'cube', not of 'cylinder'"):
        load_library("cylinder", tmp_path)


# ------------------------------------------------------------ composition and selection
def test_compose_folds_the_cube_over_its_yaws_and_flips_the_jaws():
    composed = compose(load_library("cube"), rot_z(math.radians(20.0)), math.pi / 2)
    by_index = {i: [c for c in composed if c.index == i] for i in range(8)}

    assert len(composed) == 64
    order = [(c.index, c.fold, c.flipped) for c in composed[:4]]
    assert order == [(0, 0, False), (0, 0, True), (0, 1, False), (0, 1, True)]
    first = by_index[0]
    assert sorted(set(degrees(first))) == pytest.approx([20.0, 110.0, 200.0, 290.0], abs=1e-6)
    assert all(degrees(first).count(d) == 2 for d in degrees(first))
    for k in range(4):
        unflipped = next(c for c in first if c.fold == k and not c.flipped)
        mirror = next(c for c in first if c.fold == (k + 2) % 4 and c.flipped)
        assert wrap(unflipped.psi - mirror.psi) == pytest.approx(0.0, abs=1e-9)
    assert all(c.offset_xy == pytest.approx((0.0, 0.0), abs=1e-12) for c in first)
    assert all(c.depth_m == pytest.approx(0.0125) and c.tilt_rad == pytest.approx(0.0) for c in first)
    assert all(c.tilt_rad == pytest.approx(math.pi / 2) for c in by_index[5])
    assert all(c.depth_m == pytest.approx(0.023) for c in by_index[6])
    for c in by_index[4]:
        assert math.hypot(*c.offset_xy) == pytest.approx(0.005)
        assert math.degrees(math.atan2(c.offset_xy[1], c.offset_xy[0])) % 360 == pytest.approx(
            (20.0 + 90.0 * c.fold) % 360)
        if not c.flipped:
            assert math.degrees(c.psi) % 360 == pytest.approx((110.0 + 90.0 * c.fold) % 360)
    assert all(c.roll_rad == 0.0 for c in composed)


def test_select_filters_by_tilt_score_jaw_and_depth_and_ranks_by_score_then_heading():
    composed = compose(load_library("cube"), rot_z(math.radians(20.0)), math.pi / 2)
    psi_now = math.radians(100.0)

    kept, counts = select(composed, psi_now=psi_now, min_score=0.6, **SELECT)
    assert counts == {"pochylenie": 8, "wynik": 8}
    assert len(kept) == 16
    assert kept[0].index == 0 and kept[0].score == 0.92
    assert math.degrees(kept[0].psi) == pytest.approx(110.0, abs=1e-6)
    assert not {c.index for c in kept} & {1, 2, 3}
    assert sorted(degrees([c for c in kept if c.index == 0])) == pytest.approx([20.0, 110.0, 200.0, 290.0])
    assert [c.index for c in kept[:4]] == [0, 0, 0, 0] and kept[4].index == 4
    assert rejections_text(counts) == "pochylenie 8, wynik 8"

    narrow = dict(SELECT, gap_max_m=0.030)
    kept, counts = select(composed, psi_now=psi_now, min_score=0.0, **narrow)
    assert counts == {"pochylenie": 8, "szczeka": 56} and kept == []
    # The diagonal (0.55) counts as a low score, not as too wide: the filters' order is pinned.
    kept, counts = select(composed, psi_now=psi_now, min_score=0.6, **narrow)
    assert counts == {"pochylenie": 8, "wynik": 8, "szczeka": 48}

    def alone(depth: float, offset: tuple[float, float] = (0.0, 0.0)) -> dict[str, int]:
        one = [ComposedGrasp(0, 0.0, offset, depth, 0.025, 0.9, 0.0, False, 0)]
        return select(one, psi_now=0.0, min_score=0.0, **SELECT)[1]

    assert alone(-0.001) == {"glebokosc": 1} and alone(0.026) == {"glebokosc": 1}
    assert alone(0.0125, (0.02, 0.0)) == {"poza blokiem": 1}
    assert alone(0.0125) == {}


def test_a_heading_of_a_half_turn_is_one_heading_whatever_its_sign():
    """pi and -pi (a closing direction (-1, +-1e-17, 0)) make the same executed grasp."""
    a = ComposedGrasp(0, math.pi, (0.0, 0.0), 0.0125, 0.025, 0.9, 0.0, False, 0)
    b = ComposedGrasp(1, -math.pi + 1e-12, (0.0, 0.0), 0.0125, 0.025, 0.9, 0.0, False, 1)

    kept, _ = select([a, b], psi_now=0.0, min_score=0.0, **SELECT)
    assert kept == [a]


def test_a_cylinder_library_rolls_a_side_grasp_upright_and_folds_twice():
    side = entry((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    end = entry((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    axis = roll_axis(object_spec("cylinder"))
    composed = compose(library("cylinder", [side]), rot_z(math.radians(-60.0)), math.pi, axis=axis)

    assert len(composed) == 4
    assert all(c.tilt_rad == pytest.approx(0.0, abs=1e-9) for c in composed)
    assert all(c.roll_rad == pytest.approx(math.pi / 2, abs=1e-9) for c in composed)
    assert sorted(set(degrees(composed))) == pytest.approx([30.0, 210.0], abs=1e-6)
    assert all(c.depth_m == pytest.approx(0.0125) and c.width_m == 0.025 for c in composed)

    ends = compose(library("cylinder", [end]), rot_z(math.radians(-60.0)), math.pi, axis=axis)
    assert all(c.tilt_rad == pytest.approx(math.pi / 2) and c.roll_rad == 0.0 for c in ends)
    boxed = compose(library("cube", [side]), rot_z(math.radians(-60.0)), math.pi / 2)
    assert all(c.tilt_rad == pytest.approx(math.pi / 2) for c in boxed)
