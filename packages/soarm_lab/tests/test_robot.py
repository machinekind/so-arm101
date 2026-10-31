"""The lab reads the robot description from `soarm`'s robot folder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

import soarm
from soarm_lab.robot import load_robot


def test_the_robot_description_comes_from_soarm():
    robot_yaml = yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
    spec = load_robot()
    assert spec.root == soarm.ROBOT_DIR.resolve()
    assert spec.joints == tuple(robot_yaml["joints"])
    assert spec.limits["wrist_roll"] == tuple(robot_yaml["limits"]["wrist_roll"])
    assert spec.desk.cube_radius == tuple(robot_yaml["desk"]["cube_radius"])
    assert spec.jaw_bodies == tuple(robot_yaml["jaw_bodies"])


def test_the_wrist_camera_is_read_from_robot_yaml():
    robot_yaml = yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
    spec = load_robot()

    assert set(spec.cameras) == {"wrist"}
    camera = spec.cameras["wrist"]
    assert camera.body == "camera_mount" and camera.fovy_deg == 48.4555
    assert (camera.width, camera.height) == (1920, 1080)
    assert camera.pose == tuple(tuple(row) for row in robot_yaml["cameras"]["wrist"]["pose"])
    R = np.asarray(camera.pose, float)[:3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9


def _robot_folder(tmp_path: Path, change: Callable[[dict], None]) -> Path:
    """A copy of soarm's robot.yaml in `tmp_path` with its MJCF path absolute, changed by `change`."""
    data = yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
    data["mjcf"] = str(soarm.ROBOT_DIR / data["mjcf"])
    change(data["cameras"]["wrist"])
    (tmp_path / "robot.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def _scaled_row(camera: dict) -> None:
    camera["pose"][1] = [2 * v for v in camera["pose"][1]]


@pytest.mark.parametrize("change, match", [
    (_scaled_row, "`cameras.wrist`: `pose` musi byc przeksztalceniem sztywnym"),
    (lambda camera: camera.update(width=True), "`cameras.wrist`: `width` i `height` musza byc dodatnimi "
                                               "liczbami calkowitymi"),
    (lambda camera: camera.update(fovy_deg=200.0), "`cameras.wrist`: `fovy_deg` musi lezec miedzy 0 a 180"),
    (lambda camera: camera.update(lens="wide"), r"Nieznane klucze w sekcji .*cameras.wrist: \['lens'\]"),
], ids=["not rigid", "a bool width", "fovy past 180", "an unknown key"])
def test_a_bad_camera_entry_is_refused(tmp_path: Path, change: Callable[[dict], None], match: str):
    with pytest.raises(ValueError, match=match):
        load_robot(_robot_folder(tmp_path, change))
