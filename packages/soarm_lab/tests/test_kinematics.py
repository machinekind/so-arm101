"""The arm's kinematics on the station's robot description: units, the TCP, IK, self-contacts and the jaw."""

from __future__ import annotations

import dataclasses

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


def test_ik_keeps_the_seed_gripper_whichever_start_wins(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    # The home TCP from a seed far from home, with one iteration and no random starts: only the start
    # from the rest pose hits, so it wins.
    target = kin.tcp(spec.home)[:3, 3]
    seed = dict(spec.home, shoulder_pan=60.0, elbow_flex=-40.0, wrist_flex=50.0, gripper=5.0)
    sol = kin.ik(target, seed=seed, iters=1, restarts=0)
    assert sol.ok
    assert sol.joints["shoulder_pan"] == pytest.approx(spec.home["shoulder_pan"], abs=1e-3)
    assert sol.joints["gripper"] == pytest.approx(5.0)


def test_self_contacts_find_the_hand_in_the_shoulder_near_the_base(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    near_base = kin.ik(np.array([0.15, 0.0, 0.03]), seed=spec.home)
    below_table = kin.ik(np.array([0.25, 0.0, -0.01]), seed=spec.home)
    assert near_base.ok and below_table.ok

    assert kin.self_contacts(spec.home) == set()
    assert ("gripper", "shoulder") in kin.self_contacts(near_base.joints)

    # Inside the station's scene (prefix robot/) the same pairs, without the prefix; the table does not count.
    scene = sim.build(sim.SceneConfig(station))
    try:
        assert scene.kin.self_contacts(near_base.joints) == kin.self_contacts(near_base.joints)
        assert scene.kin.self_contacts(below_table.joints) == set()
        assert scene.kin.data.ncon > 0                                    # the hand is in the table
    finally:
        scene.close()


def test_the_jaw_gap_grows_with_the_gripper_command(station):
    kin = RobotKinematics(station.spec())

    assert kin.jaw_gap(0.0) == pytest.approx(0.0090, abs=5e-4)
    assert kin.jaw_gap(39.0) == pytest.approx(0.041, abs=1e-3)
    assert kin.jaw_gap(100.0) == pytest.approx(0.088, abs=1e-3)
    assert np.all(np.diff([kin.jaw_gap(float(g)) for g in range(101)]) > 0)


def test_gripper_for_gap_inverts_the_jaw_gap(station):
    kin = RobotKinematics(station.spec())

    for gap in (0.017, 0.025, 0.041):
        assert kin.jaw_gap(kin.gripper_for_gap(gap)) == pytest.approx(gap, abs=1e-4)
    assert kin.gripper_for_gap(0.025) == pytest.approx(19.6, abs=0.1)
    for gap in (0.005, 0.1):
        with pytest.raises(ValueError, match="outside the jaw travel"):
            kin.gripper_for_gap(gap)


def test_the_fingertips_span_the_closing_axis_within_the_straddle_tolerances(station):
    """The tip line is no parallel jaw: the moving tip rides an arc, 7.4 degrees off the closing axis at
    command 0. What a straddle relies on is the midpoint and the gap, and the spread along the approach."""
    spec = station.spec()
    kin = RobotKinematics(spec)
    for g in (0.0, 10.0, 25.0, 39.0):
        joints = dict(spec.home, gripper=g)
        fixed, moving = kin.fingertips(joints)
        approach, closing = kin.tool_axes(joints)
        diff = moving - fixed
        across = np.cross(approach, closing)
        assert abs(diff @ approach) < 0.002
        assert abs(diff @ across) < 0.001
        assert abs(kin.jaw_gap(g) - float(np.linalg.norm(diff))) < 1e-9
    for g, most in ((0.0, 8.0), (39.0, 8.0), (60.0, 8.0)):
        joints = dict(spec.home, gripper=g)
        fixed, moving = kin.fingertips(joints)
        diff = (moving - fixed) / np.linalg.norm(moving - fixed)
        assert np.degrees(np.arccos(np.clip(diff @ kin.tool_axes(joints)[1], -1.0, 1.0))) < most
    assert kin.jaw_centre(0.0) == pytest.approx([0.0023, 0.0, -0.0156], abs=5e-4)
    assert abs(kin.jaw_centre(39.0)[2]) < 0.001 and abs(kin.jaw_centre(39.0)[1]) < 0.001


def test_the_jaw_reaches_below_the_tcp_by_the_fixed_jaws_sole(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    fixed, _ = kin.fingertips(dict(spec.home, gripper=0.0))
    T = kin.tcp(dict(spec.home, gripper=0.0))
    tip = T[:3, :3].T @ (fixed - T[:3, 3])
    sphere = tip[0] + kin.model.geom_size[kin.model.geom(spec.fingertips[0]).id][0]

    assert kin.jaw_reach(0.0) == pytest.approx(0.0066, abs=5e-4)
    assert kin.jaw_reach(39.0) == pytest.approx(0.0069, abs=5e-4)
    assert sphere == pytest.approx(0.0036, abs=3e-4)
    assert kin.jaw_reach(0.0) > sphere + 0.002 and kin.jaw_reach(39.0) > sphere + 0.002
    with pytest.raises(ValueError, match="jaw_bodies"):
        RobotKinematics(dataclasses.replace(spec, jaw_bodies=("", ""))).jaw_reach(0.0)
