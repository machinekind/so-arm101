from __future__ import annotations

import json
import math

import pytest
import yaml

import soarm
from conftest import SOARM_ROOT

EMBODIMENTS = sorted(soarm.data_dir("embodiment").glob("*.json"))
MOUNTS = sorted(soarm.data_dir("mounts").glob("*/transform.yaml"))


@pytest.mark.parametrize("path", EMBODIMENTS, ids=lambda p: p.name)
def test_embodiment_matches_its_robot(path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = soarm.load(soarm.robot_dir(data["robot"]))
    assert data["state"]["joints"] == list(spec.joints)
    assert data["action"]["joints"] == list(spec.joints)


@pytest.mark.parametrize("path", MOUNTS, ids=lambda p: p.parent.name)
def test_mount_transform_is_well_formed(path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(data) == {"parent", "child", "pos", "quat", "cad"}
    assert len(data["pos"]) == 3
    assert math.isclose(math.hypot(*data["quat"]), 1.0, abs_tol=1e-5)


def test_example_arm_names_a_known_robot_and_mount() -> None:
    data = yaml.safe_load((SOARM_ROOT / "arms" / "example" / "arm.yaml").read_text(encoding="utf-8"))
    spec = soarm.load(soarm.robot_dir(data["robot"]))
    assert set(data["eeprom_limits"]) == set(spec.joints)
    assert (soarm.data_dir("mounts") / data["mount"] / "transform.yaml").is_file()
