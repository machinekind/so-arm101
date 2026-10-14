"""The lab reads the robot description from `soarm`'s robot folder."""

from __future__ import annotations

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
