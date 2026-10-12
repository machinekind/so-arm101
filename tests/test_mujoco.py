from __future__ import annotations

import pytest

from soarm.spec import RobotSpec

mujoco = pytest.importorskip("mujoco")


def test_mjcf_compiles(ready_spec: RobotSpec) -> None:
    model = mujoco.MjModel.from_xml_path(str(ready_spec.mjcf_path))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert joints == list(ready_spec.joints)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ready_spec.tcp_site) >= 0
