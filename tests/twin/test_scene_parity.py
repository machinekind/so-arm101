"""The lab's scene from an exported station is the twin's scene from the same station, up to XML rounding."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from lerobot_mp.twin import scene as twin_scene  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

from soarm_lab.station import Box, Station  # noqa: E402

CUBE = Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0), mass=0.03)


def test_the_scene_matches_the_one_the_twin_builds(twin_robots):
    station = Station.load("sim")
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
