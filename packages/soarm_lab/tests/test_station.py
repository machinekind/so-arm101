"""The exported station read with plain MuJoCo: the bundled `sim` station and what a task adds to it."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import soarm

mujoco = pytest.importorskip("mujoco")

from soarm_lab.kinematics import inverse  # noqa: E402
from soarm_lab.station import PREFIX, STATIONS_DIR, Box, CameraView, Station, stations  # noqa: E402

#: Bodies of the SO-101 in its MJCF, world excluded.
ARM_BODIES = 8
CUBE = Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0), mass=0.03)


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

