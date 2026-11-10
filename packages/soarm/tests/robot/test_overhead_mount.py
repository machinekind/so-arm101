"""The overhead camera mount fixture: its meshes, colliders, stack, camera site and placements.

test_twin_overhead_mount.py places it beside the SO-101 in the twin.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import yaml

import soarm

FOLDER = soarm.ROBOT_DIR / "fixtures" / "overhead_cam_32"
TWO, THREE = ["bottom", "top"], ["bottom", "middle", "top"]
#: Height of the camera head's top above the foot [m]: two parts and the full design.
HEIGHT = {"two": 0.4054, "three": 0.5604}


@pytest.fixture(scope="module")
def fixture() -> dict:
    return yaml.safe_load((FOLDER / "fixture.yaml").read_text(encoding="utf-8"))


def vertices(name: str) -> np.ndarray:
    data = (FOLDER / name).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    triangle = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(data[84:84 + 50 * n], dtype=triangle)["v"].reshape(-1, 3).astype(float)


def inside(points: np.ndarray, box: dict, pad: float = 0.0) -> np.ndarray:
    return np.all(np.abs(points - np.array(box["pos"])) <= np.array(box["size"]) + pad, axis=1)


def offsets(fixture: dict, parts: list[str]) -> list[float]:
    """Height of each part's frame in the stack."""
    z, out = 0.0, []
    for p in parts:
        out.append(z)
        z += fixture["parts"][p].get("top", [0.0, 0.0, 0.0])[2]
    return out


def test_every_mesh_lies_in_its_part_s_colliders(fixture: dict) -> None:
    for name, part in fixture["parts"].items():
        v = vertices(part["mesh"])
        boxes = list(part["colliders"].values())
        covered = np.any([inside(v, b, 1e-4) for b in boxes], axis=0)
        assert covered.all(), name
        # Fitted, not padded: every box but the camera head's touches the mesh on its sides and top (a box
        # on another one reaches down to it).
        for box_name, b in part["colliders"].items():
            if box_name == "head":
                continue
            near = v[inside(v, b, 1e-4)]
            lo, hi = np.array(b["pos"]) - b["size"], np.array(b["pos"]) + b["size"]
            assert np.allclose(near.min(axis=0)[:2], lo[:2], atol=1e-4), (name, box_name)
            assert np.allclose(near.max(axis=0), hi, atol=1e-4), (name, box_name)


def test_two_and_three_parts_reach_their_heights(fixture: dict) -> None:
    top = vertices(fixture["parts"]["top"]["mesh"])[:, 2].max()
    assert offsets(fixture, TWO)[-1] + top == pytest.approx(HEIGHT["two"], abs=5e-4)
    assert offsets(fixture, THREE)[-1] + top == pytest.approx(HEIGHT["three"], abs=5e-4)
    assert fixture["stack"] == THREE


def test_the_camera_looks_down_and_forward_from_the_head(fixture: dict) -> None:
    top = fixture["parts"]["top"]
    T = np.array(top["cameras"]["camera"])
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6) and np.linalg.det(T[:3, :3]) > 0
    z = T[:3, 2]
    assert z[2] < -0.85 and z[0] > 0.4 and abs(z[1]) < 1e-6         # down, 25 deg forward
    assert inside(T[None, :3, 3], top["colliders"]["head"])[0]


def test_the_placements_mirror_left_and_right(fixture: dict) -> None:
    left, right = (np.array(fixture["placements"][s]["xyz"]) for s in ("left", "right"))
    assert left[1] > 0.1 and right[1] == pytest.approx(-left[1])
    assert left[[0, 2]] == pytest.approx(right[[0, 2]])
