from __future__ import annotations

import pytest

import soarm

mujoco = pytest.importorskip("mujoco")


def test_mjcf_compiles(robot_yaml: dict) -> None:
    model = mujoco.MjModel.from_xml_path(str(soarm.ROBOT_DIR / robot_yaml["mjcf"]))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert joints == robot_yaml["joints"]
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot_yaml["tcp_site"]) >= 0
