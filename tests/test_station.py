"""The exported station read with plain MuJoCo: the bundled `sim` station, task additions, kinematics."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import soarm

mujoco = pytest.importorskip("mujoco")

from soarm_lab import sim  # noqa: E402
from soarm_lab.kinematics import RobotKinematics, inverse  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.station import PREFIX, STATIONS_DIR, Box, CameraView, Station, stations  # noqa: E402

#: Bodies of the SO-101 in its MJCF, world excluded.
ARM_BODIES = 8
CUBE = Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0), mass=0.03)


@pytest.fixture(scope="module")
def station() -> Station:
    return Station.load("sim")


def test_the_sim_station_is_bundled_without_meshes(station):
    assert "sim" in stations()
    assert station.folder == STATIONS_DIR / "sim"
    assert station.robot == "so101"
    assert not (station.folder / "assets").exists()          # the meshes come from soarm's robot folder


def test_plain_mujoco_compiles_the_scene_xml(station):
    spec = mujoco.MjSpec.from_file(str(station.scene_path))
    spec.meshdir = str(soarm.ROBOT_DIR / "assets")
    model = spec.compile()

    assert model.nbody == 1 + ARM_BODIES
    assert model.njnt == model.nu == 6
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert names == {"sym-lewa", "sym-prawa", PREFIX + "wrist_cam"}
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.geom("table").size[2] * 2 == pytest.approx(station.table.height)


def test_the_station_record(station):
    assert [c.name for c in station.cameras] == ["sym-lewa", "sym-prawa"]
    cam = station.cameras[0]
    assert (cam.width, cam.height) == (640, 480)
    assert cam.K[0, 0] == pytest.approx(560.0)
    assert station.dynamics == {}
    assert station.home is None
    assert station.T_base2world[:3, 3] == pytest.approx([-0.35, 0.0, 0.75])


def test_the_scene_puts_the_base_and_the_cameras_where_the_record_says(station):
    scene = sim.build(sim.SceneConfig(station))
    try:
        assert scene.T_base2world == pytest.approx(station.T_base2world, abs=1e-9)
        for view in station.cameras:
            assert scene.camera_pose(view.name) == pytest.approx(view.T_cam2base, abs=1e-5)
            assert scene.model.cam_resolution[scene.model.camera(view.name).id].tolist() == [640, 480]
            assert scene.camera(view.name).K == pytest.approx(view.K)
    finally:
        scene.close()


def test_task_objects_and_jaw_sensors_are_added(station):
    target = Box("target", (0.035, 0.035, 0.002), (0.3, 0.1), free=False, mocap=True)
    model = station.build(objects=[CUBE, target], grasp_sensors=["cube"])

    assert model.nbody == 1 + ARM_BODIES + 2
    cube = model.body("cube")
    assert model.jnt_type[model.body_jntadr[cube.id]] == mujoco.mjtJoint.mjJNT_FREE
    geom = model.geom("cube")
    assert geom.condim[0] == 4
    assert geom.friction == pytest.approx([1.0, 0.02, 0.002])
    assert model.body_mass[cube.id] == pytest.approx(0.03)
    pad = model.geom("target")
    assert model.body_mocapid[model.body("target").id] == 0
    assert (pad.contype[0], pad.conaffinity[0]) == (0, 0)
    sensors = [model.sensor(i).name for i in range(model.nsensor)]
    assert sensors == ["cube_jaw0", "cube_jaw1"]
    assert [model.body(model.sensor_objid[i]).name for i in range(2)] == [PREFIX + "gripper",
                                                                         PREFIX + "moving_jaw_so101_v1"]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lies = inverse(station.T_base2world) @ np.r_[data.xpos[cube.id], 1.0]
    assert lies[:3] == pytest.approx([0.2, 0.0, 0.015], abs=1e-9)


def test_an_extra_camera_is_added_and_a_duplicate_refused(station):
    view = CameraView.from_fov("widok", 480, 360, 50.0, np.eye(4))
    model = station.build(cameras=[view])
    assert model.camera("widok").id >= 0
    assert model.cam_resolution[model.camera("widok").id].tolist() == [480, 360]

    with pytest.raises(ValueError, match="already has a camera 'sym-lewa'"):
        station.build(cameras=[CameraView.from_fov("sym-lewa", 64, 48, 50.0, np.eye(4))])


def test_an_export_with_its_own_assets(tmp_path, station):
    """What `lerobot-twin export` writes: scene.xml, station.json and assets/ next to them."""
    folder = tmp_path / "desk"
    shutil.copytree(station.folder, folder)
    shutil.copytree(soarm.ROBOT_DIR / "assets", folder / "assets")

    exported = Station.load(folder)
    assert exported.folder == folder
    assert mujoco.MjModel.from_xml_path(str(folder / "scene.xml")).nbody == 1 + ARM_BODIES
    assert exported.build(objects=[CUBE]).nbody == 1 + ARM_BODIES + 1


def test_unknown_stations_and_robots_are_refused(tmp_path, station):
    with pytest.raises(FileNotFoundError, match="bundled station \\(sim\\)"):
        Station.load("desk-nobody-exported")

    folder = tmp_path / "a1x-desk"
    shutil.copytree(station.folder, folder)
    data = json.loads((folder / "station.json").read_text(encoding="utf-8"))
    (folder / "station.json").write_text(json.dumps({**data, "robot": "a1x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="robot is 'a1x'"):
        Station.load(folder).spec()


# --------------------------------------------------------------------------- kinematics


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


def test_the_robot_description_comes_from_soarm(robot_yaml):
    spec = load_robot()
    assert spec.root == soarm.ROBOT_DIR.resolve()
    assert spec.joints == tuple(robot_yaml["joints"])
    assert spec.limits["wrist_roll"] == tuple(robot_yaml["limits"]["wrist_roll"])
    assert spec.desk.cube_radius == tuple(robot_yaml["desk"]["cube_radius"])
    assert spec.jaw_bodies == tuple(robot_yaml["jaw_bodies"])


# --------------------------------------------------------------------------- the twin's own scene


def test_the_scene_matches_the_one_the_twin_builds(station, twin_robots):
    """The lab's scene from the export is the twin's scene from the same station, up to XML rounding."""
    from lerobot_mp.twin import scene as twin_scene
    from lerobot_mp.twin.workspace import Workspace

    ws = Workspace.load(station.folder / "station.json")
    cube = twin_scene.Box("cube", CUBE.size, CUBE.xy, mass=CUBE.mass)
    twin_model = twin_scene.build(ws.scene_config(objects=[cube], grasp_sensors=["cube"])).model
    lab_model = station.build(objects=[CUBE], grasp_sensors=["cube"])

    for size in ("nq", "nv", "nu", "nbody", "ngeom", "nsensor", "ncam", "nmesh", "nlight", "nmeshvert"):
        assert getattr(lab_model, size) == getattr(twin_model, size), size
    for name in ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_size", "geom_pos", "geom_quat",
                 "geom_friction", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss",
                 "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "cam_pos", "cam_quat",
                 "cam_intrinsic", "qpos0"):
        assert getattr(lab_model, name) == pytest.approx(getattr(twin_model, name), abs=1e-5), name
    for option in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations"):
        assert getattr(lab_model.opt, option) == getattr(twin_model.opt, option), option


# --------------------------------------------------------------------------- renders


@pytest.mark.render
def test_the_environment_renders_from_the_station_camera(station):
    from soarm_lab.rl.env import TwinEnv

    env = TwinEnv("lift", station=station, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()
    assert env.render_camera == "sym-lewa"
    assert frame.shape == (480, 640, 3) and frame.std() > 0


@pytest.mark.render
def test_a_station_without_cameras_gets_the_preview_camera(tmp_path, station):
    """`env.scene_config(view=True)`: the twin's `widok` camera when the station has none."""
    from soarm_lab.rl.env import VIEW, TwinEnv

    folder = tmp_path / "blind"
    folder.mkdir()
    xml = station.scene_path.read_text(encoding="utf-8").splitlines(keepends=True)
    (folder / "scene.xml").write_text("".join(line for line in xml if '<camera name="sym-' not in line),
                                      encoding="utf-8")
    (folder / "station.json").write_text(json.dumps({**station.data, "cameras": []}), encoding="utf-8")

    env = TwinEnv("reach", station=Station.load(folder), render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()
    assert env.render_camera == VIEW
    assert frame.shape == (360, 480, 3) and frame.std() > 0
