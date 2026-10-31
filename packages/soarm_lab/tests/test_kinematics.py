"""The arm's kinematics on the station's robot description: units, the TCP, IK, self-contacts, the jaw and the
wrist camera."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from soarm_lab import sim  # noqa: E402
from soarm_lab.kinematics import RobotKinematics, pose, turn_toward  # noqa: E402


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


def test_tip_swing_grows_as_the_hinged_jaw_opens(station):
    """The moving tip rides an arc about the hinge: wide open it trails the fixed tip along the approach."""
    kin = RobotKinematics(station.spec())
    swing = {g: kin.tip_swing(g) for g in (0.0, 42.0, 75.0, 100.0)}

    assert all(value >= 0.0 for value in swing.values())
    assert swing[42.0] < 0.001
    assert swing[75.0] == pytest.approx(0.0099, abs=0.001)
    assert swing[100.0] == pytest.approx(0.0227, abs=0.001)
    assert swing[0.0] == pytest.approx(0.0011, abs=3e-4)
    # The fixed tip is the deeper one wide open.
    fixed, moving = kin.fingertips(dict(station.spec().home, gripper=100.0))
    approach, _ = kin.tool_axes(dict(station.spec().home, gripper=100.0))
    assert float(approach @ (fixed - moving)) == pytest.approx(swing[100.0], abs=1e-9)


def test_turn_toward_caps_the_angle_on_the_shortest_way():
    def about_z(deg: float) -> np.ndarray:
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    assert turn_toward(about_z(10.0), about_z(12.0), np.radians(5.0)) == pytest.approx(about_z(12.0))
    assert turn_toward(about_z(10.0), about_z(50.0), np.radians(5.0)) == pytest.approx(about_z(15.0))
    assert turn_toward(about_z(170.0), about_z(-170.0), np.radians(5.0)) == pytest.approx(about_z(175.0))


def test_the_camera_pose_from_robot_yaml_matches_the_mjcf_camera(station):
    """robot.yaml's `cameras: wrist` is the MJCF's `wrist_cam` in OpenCV axes: equal wherever the arm is."""
    spec = station.spec()
    kin = RobotKinematics(spec)
    turned = {"shoulder_pan": 30.0, "shoulder_lift": -10.0, "elbow_flex": 60.0, "wrist_flex": 40.0,
              "wrist_roll": 60.0, "gripper": 20.0}
    cam = kin.model.camera("wrist_cam").id
    for joints in (spec.home, turned):
        T = kin.camera("wrist", joints)
        kin._apply(kin.to_q(joints))
        mujoco.mj_camlight(kin.model, kin.data)
        mjcf = kin._base_inv() @ pose(kin.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]),
                                      kin.data.cam_xpos[cam])
        assert np.abs(T - mjcf).max() < 1e-6

    assert kin.camera("wrist", spec.home)[:3, 3] == pytest.approx([0.2728, 0.0548, 0.1381], abs=1e-4)
    with pytest.raises(ValueError, match="no camera 'nope'"):
        kin.camera("nope", spec.home)


def test_the_collision_points_of_the_jaws_come_from_their_primitives(station):
    spec = station.spec()
    kin = RobotKinematics(spec)
    tcp = kin.tcp(spec.home)[:3, 3]
    fixed_body, moving_body = spec.jaw_bodies

    for body in spec.jaw_bodies:
        points = kin.collision_points(body, spec.home)
        assert points.ndim == 2 and points.shape[1] == 3 and points.shape[0] >= 24
        assert np.linalg.norm(points - tcp, axis=1).max() < 0.12
    closed, opened = (dict(spec.home, gripper=g) for g in (0.0, 100.0))
    moved = kin.collision_points(moving_body, closed) - kin.collision_points(moving_body, opened)
    assert np.abs(moved).max() > 0.01
    assert kin.collision_points(fixed_body, closed) == pytest.approx(kin.collision_points(fixed_body, opened))
    with pytest.raises(KeyError):
        kin.collision_points("nope", spec.home)
