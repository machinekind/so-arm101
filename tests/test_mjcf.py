from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from soarm.spec import RobotSpec


def names(root: ET.Element, tag: str) -> list[str]:
    return [el.get("name") for el in root.iter(tag) if el.get("name")]


@pytest.fixture
def mjcf(ready_spec: RobotSpec) -> ET.Element:
    return ET.parse(ready_spec.mjcf_path).getroot()


def test_joint_names_match_mjcf(ready_spec: RobotSpec, mjcf: ET.Element) -> None:
    assert names(mjcf, "joint") == list(ready_spec.joints)


def test_named_parts_exist_in_mjcf(ready_spec: RobotSpec, mjcf: ET.Element) -> None:
    bodies, sites = set(names(mjcf, "body")), set(names(mjcf, "site"))
    geoms, cameras = set(names(mjcf, "geom")), set(names(mjcf, "camera"))
    assert ready_spec.tcp_site in sites
    assert {ready_spec.hand_body, ready_spec.base_body, *ready_spec.jaw_bodies} <= bodies
    assert set(ready_spec.fingertips) <= geoms
    assert set(ready_spec.onboard_cameras) <= cameras


def test_limits_inside_mjcf_ranges(ready_spec: RobotSpec, mjcf: ET.Element) -> None:
    assert mjcf.find("compiler").get("angle") == "radian"
    for joint in mjcf.iter("joint"):
        name = joint.get("name")
        if name not in ready_spec.arm_joints:
            continue
        lo, hi = (math.degrees(float(v)) for v in joint.get("range").split())
        app_lo, app_hi = ready_spec.limits[name]
        assert lo - 1e-3 <= app_lo < app_hi <= hi + 1e-3, (name, lo, hi)


def test_meshes_exist(ready_spec: RobotSpec, mjcf: ET.Element) -> None:
    meshdir = ready_spec.mjcf_path.parent / mjcf.find("compiler").get("meshdir", ".")
    files = [m.get("file") for m in mjcf.iter("mesh") if m.get("file")]
    assert files
    assert [f for f in files if not (meshdir / f).is_file()] == []
