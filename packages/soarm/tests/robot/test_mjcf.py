from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

import soarm


def names(root: ET.Element, tag: str) -> list[str]:
    return [el.get("name") for el in root.iter(tag) if el.get("name")]


@pytest.fixture
def mjcf(robot_yaml: dict) -> ET.Element:
    return ET.parse(soarm.ROBOT_DIR / robot_yaml["mjcf"]).getroot()


def test_joint_names_match_mjcf(robot_yaml: dict, mjcf: ET.Element) -> None:
    assert names(mjcf, "joint") == robot_yaml["joints"]


def test_named_parts_exist_in_mjcf(robot_yaml: dict, mjcf: ET.Element) -> None:
    bodies, sites = set(names(mjcf, "body")), set(names(mjcf, "site"))
    geoms, cameras = set(names(mjcf, "geom")), set(names(mjcf, "camera"))
    assert robot_yaml["tcp_site"] in sites
    assert {robot_yaml["hand_body"], robot_yaml["base_body"], *robot_yaml["jaw_bodies"]} <= bodies
    assert set(robot_yaml["fingertips"]) <= geoms
    assert set(robot_yaml["onboard_cameras"]) <= cameras


def test_limits_inside_mjcf_ranges(robot_yaml: dict, mjcf: ET.Element) -> None:
    assert mjcf.find("compiler").get("angle") == "radian"
    arm_joints = [j for j in robot_yaml["joints"] if j != robot_yaml["gripper"]]
    for joint in mjcf.iter("joint"):
        name = joint.get("name")
        if name not in arm_joints:
            continue
        lo, hi = (math.degrees(float(v)) for v in joint.get("range").split())
        app_lo, app_hi = robot_yaml["limits"][name]
        assert lo - 1e-3 <= app_lo < app_hi <= hi + 1e-3, (name, lo, hi)


def test_meshes_exist(robot_yaml: dict, mjcf: ET.Element) -> None:
    meshdir = soarm.ROBOT_DIR / mjcf.find("compiler").get("meshdir", ".")
    files = [m.get("file") for m in mjcf.iter("mesh") if m.get("file")]
    assert files
    assert [f for f in files if not (meshdir / f).is_file()] == []
