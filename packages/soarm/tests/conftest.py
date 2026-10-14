from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import yaml
from lerobot_mp_api.backend import BackendContext

import soarm


@pytest.fixture(scope="session")
def robot_yaml() -> dict:
    """`robot.yaml` of the SO-101 as plain data."""
    return yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def so101_context(robot_yaml: dict) -> Callable[..., BackendContext]:
    """What the twin hands a backend factory for the SO-101 on port COM_TEST, given the backend's config
    section: robot.yaml's joints, limits and speeds. test_twin_contract.py checks it against the twin's."""

    def build(**settings: Any) -> BackendContext:
        return BackendContext(
            robot=robot_yaml["name"],
            joints=tuple(robot_yaml["joints"]),
            gripper=robot_yaml["gripper"],
            limits={name: tuple(limits) for name, limits in robot_yaml["limits"].items()},
            max_vel=dict(robot_yaml["max_vel"]),
            port="COM_TEST",
            settings=settings,
        )

    return build
