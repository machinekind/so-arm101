"""The twin's own code accepts this package: its strict robot loader and its backend plugin contract.

Discovery (`robot_folders()`, `backends()`, `create_backend`) is not tested here. The twin's `main` still
registers its own copy of the SO-101 (robot `so101`, backends `feetech` and `lerobot`), so with both
installed the twin's discovery raises on the duplicate names. These tests call the twin's loader and
context builder directly instead. Add a discovery test once the twin drops its copy (the twin side of #145).
"""

from __future__ import annotations

import pytest
from lerobot_mp.robot import BackendPlugin, backend_context
from lerobot_mp.robot.factory import PluginBackend
from lerobot_mp.twin.robots import load_robot, robot_config
from test_feetech import FakeBusLink

import soarm
from soarm import feetech, lerobot_backend
from soarm.feetech import ADDR_GOAL_POSITION


@pytest.fixture(scope="module")
def spec():
    return load_robot(soarm.ROBOT_DIR)


def context(spec, backend: str, **sections):
    cfg = robot_config(spec, overrides={"robot": {"backend": backend, "port": "COM_TEST"}, **sections})
    return backend_context(cfg, spec, backend)


def test_the_twin_loads_the_robot_folder(spec) -> None:
    assert spec.name == "so101"
    assert spec.root == soarm.ROBOT_DIR.resolve()
    assert spec.mjcf_path.is_file()
    assert spec.joints == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
    )
    # The panel offers these connections; this package provides both backends.
    assert {c.backend for c in spec.connections} == {"feetech", "lerobot"}


def test_feetech_plugin_runs_through_the_twin_wrapper(spec) -> None:
    arm = feetech.create(context(spec, "feetech"))
    assert isinstance(arm, BackendPlugin)
    link = FakeBusLink()
    arm.bus.open(link=link)
    backend = PluginBackend(arm, "feetech")
    backend.connect()
    try:
        assert backend.is_connected
        assert backend.info.name == "SO-101 (feetech)" and not backend.info.simulated
        positions = backend.read_joints()
        assert set(positions) == set(spec.joints)
        assert all(positions[j] == 0.0 for j in spec.arm_joints)
        assert backend.send_joints({"shoulder_pan": 10.0}) == {"shoulder_pan": pytest.approx(10.0)}
        assert link.registers[1][ADDR_GOAL_POSITION] == arm._to_ticks("shoulder_pan", 10.0)
        assert not backend.link_silent
        assert backend.faults() == []
        assert backend.joint_limits() == {}
        assert backend.gripper_range_q() == pytest.approx(spec.gripper_range_q, abs=1e-6)
        assert backend.calibration_warnings() == []
    finally:
        backend.disconnect()
    assert not backend.is_connected and link.closed


def test_the_backend_section_of_the_twin_config_reaches_the_plugin(spec) -> None:
    arm = feetech.create(context(spec, "feetech", feetech={"baudrate": 500_000}))
    assert arm.settings.baudrate == 500_000
    with pytest.raises(ValueError, match="baud"):
        feetech.create(context(spec, "feetech", feetech={"baud": 500_000}))


def test_lerobot_plugin_builds_through_the_twin_wrapper(spec) -> None:
    arm = lerobot_backend.create(context(spec, "lerobot", lerobot={"robot_id": "arm1"}))
    assert isinstance(arm, BackendPlugin)
    assert arm.settings.robot_id == "arm1"
    backend = PluginBackend(arm, "lerobot")
    assert backend.info.name == "SO101 (LeRobot)" and not backend.info.simulated
    assert not backend.is_connected and not backend.link_silent
    assert backend.faults() == []
