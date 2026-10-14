"""The arm's kinematics on the station's robot description: units, the TCP and IK."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab import sim  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402


def test_degrees_and_gripper_units_round_trip(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    joints = {"shoulder_pan": 12.5, "shoulder_lift": -40.0, "elbow_flex": 70.0, "wrist_flex": -15.0,
              "wrist_roll": 100.0, "gripper": 42.0}
    q = kin.to_q(joints)
    assert q[:5] == pytest.approx(np.radians([12.5, -40.0, 70.0, -15.0, 100.0]))
    assert kin.from_q(q) == pytest.approx(joints)
    closed, opened = spec.gripper_range_q
    assert kin.to_q({"gripper": 0.0})[5] == pytest.approx(closed)
    assert kin.to_q({"gripper": 100.0})[5] == pytest.approx(opened)
    # Missing joints take the rest pose.
    assert kin.from_q(kin.to_q({})) == pytest.approx(spec.home)


def test_tcp_at_home_is_in_front_of_the_base_above_the_table(station):
    spec = station.spec()
    tcp = RobotKinematics(spec).tcp(spec.home)[:3, 3]
    assert 0.1 < tcp[0] < 0.35
    assert abs(tcp[1]) < 0.01
    assert 0.0 < tcp[2] < 0.3

    # The arm inside the station's scene (prefix robot/) has the same kinematics.
    scene = sim.build(sim.SceneConfig(station))
    try:
        assert scene.kin.tcp(spec.home)[:3, 3] == pytest.approx(tcp, abs=1e-5)
    finally:
        scene.close()


def test_ik_finds_the_home_tcp_again(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    target = kin.tcp(spec.home)[:3, 3]
    seed = dict(spec.home, shoulder_pan=20.0, elbow_flex=30.0)
    sol = kin.ik(target, seed=seed)
    assert sol.ok and sol.pos_err < 1e-3
    assert kin.tcp(sol.joints)[:3, 3] == pytest.approx(target, abs=1e-3)
