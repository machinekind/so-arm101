"""The compiled station scene (`soarm_lab.sim`) puts the arm and the cameras where the station record says."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from soarm_lab import sim  # noqa: E402


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
