"""The twin's own code accepts `soarm`: its discovery, its strict robot loader, its backend context and
its backend factory. The twin holds no robot of its own, so `so101`, `feetech` and `lerobot` resolve to
this package through the entry points.
"""

from __future__ import annotations

from importlib import metadata

import pytest

pytestmark = pytest.mark.twin
pytest.importorskip("machinekind_twin")

from machinekind_twin.backends import BackendPlugin, RobotBackend, backend_context, create_backend, factory  # noqa: E402
from machinekind_twin.robots import load_robot, robot_config  # noqa: E402
from machinekind_twin_api.entry_points import BACKENDS  # noqa: E402

import soarm  # noqa: E402
from fake_bus import FakeBusLink  # noqa: E402
from soarm.feetech import ADDR_GOAL_POSITION, FeetechArm  # noqa: E402
from soarm.lerobot_backend import LeRobotArm  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return load_robot(soarm.ROBOT_DIR)


@pytest.fixture
def soarm_backends() -> None:
    """The twin's backend discovery finds `feetech` and `lerobot` in `soarm` and nowhere else."""
    ours = {ep.name: ep.value for ep in metadata.distribution("soarm").entry_points.select(group=BACKENDS)}
    found = factory.backends()
    assert {name: found[name].value for name in ours} == ours


def config(spec, backend: str, **sections):
    return robot_config(spec, overrides={"robot": {"backend": backend, "port": "COM_TEST"}, **sections})


def test_the_twin_loads_the_robot_folder(spec) -> None:
    assert spec.name == "so101"
    assert spec.root == soarm.ROBOT_DIR.resolve()
    assert spec.mjcf_path.is_file()
    assert spec.joints == (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
    )
    # The panel offers these connections; this package provides both backends.
    assert {c.backend for c in spec.connections} == {"feetech", "lerobot"}


def test_the_twin_builds_the_context_the_backend_tests_use(spec, so101_context) -> None:
    """The backend tests build their `BackendContext` from robot.yaml (conftest.py), without the twin."""
    assert backend_context(config(spec, "feetech"), spec) == so101_context()
    assert (backend_context(config(spec, "lerobot", lerobot={"robot_id": "arm1"}), spec)
            == so101_context(robot_id="arm1"))


def test_the_twin_runs_the_feetech_backend_as_it_is(spec, soarm_backends) -> None:
    """`create_backend` wraps a plugin into `PluginBackend` unless it is a `RobotBackend`, as ours are."""
    arm = create_backend(config(spec, "feetech"), spec)
    assert type(arm) is FeetechArm
    assert isinstance(arm, RobotBackend) and isinstance(arm, BackendPlugin)
    link = FakeBusLink()
    arm.bus.open(link=link)
    arm.connect()
    try:
        assert arm.is_connected
        assert arm.info.name == "SO-101 (feetech)" and not arm.info.simulated
        positions = arm.read_joints()
        assert set(positions) == set(spec.joints)
        assert all(positions[j] == 0.0 for j in spec.arm_joints)
        assert arm.send_joints({"shoulder_pan": 10.0}) == {"shoulder_pan": pytest.approx(10.0)}
        assert link.registers[1][ADDR_GOAL_POSITION] == arm._to_ticks("shoulder_pan", 10.0)
        assert not arm.link_silent
        assert arm.faults() == []
        assert arm.joint_limits() == {}
        assert arm.gripper_range_q() == pytest.approx(spec.gripper_range_q, abs=1e-6)
        assert arm.calibration_warnings() == []
    finally:
        arm.disconnect()
    assert not arm.is_connected and link.closed


def test_the_backend_section_of_the_twin_config_reaches_the_backend(spec, soarm_backends) -> None:
    arm = create_backend(config(spec, "feetech", feetech={"baudrate": 500_000}), spec)
    assert arm.settings.baudrate == 500_000
    with pytest.raises(ValueError, match="baud"):
        create_backend(config(spec, "feetech", feetech={"baud": 500_000}), spec)


def test_the_twin_builds_the_lerobot_backend_as_it_is(spec, soarm_backends) -> None:
    arm = create_backend(config(spec, "lerobot", lerobot={"robot_id": "arm1"}), spec)
    assert type(arm) is LeRobotArm
    assert isinstance(arm, RobotBackend) and isinstance(arm, BackendPlugin)
    assert arm.settings.robot_id == "arm1"
    assert arm.info.name == "SO101 (LeRobot)" and not arm.info.simulated
    assert not arm.is_connected and not arm.link_silent
    assert arm.faults() == []
