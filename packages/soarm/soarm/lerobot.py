"""LeRobot-compatible backend. Stub.

The code comes from the twin's `src/lerobot_mp/robot/lerobot_backend.py` (ticket #145).
It wraps LeRobot's SO-follower classes and needs the `lerobot` extra (Python 3.12 or newer).
"""

from __future__ import annotations

from typing import Any

SOURCE = "twin src/lerobot_mp/robot/lerobot_backend.py"


class LeRobotArm:
    """Adapter over LeRobot `SO101Follower` / `SO100Follower`. Implements `soarm.backend.Backend`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"LeRobotArm moves here from {SOURCE} (#145)")

    def connect(self) -> None:
        """From twin robot/lerobot_backend.py LeRobotArm.connect."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """From twin robot/lerobot_backend.py LeRobotArm.disconnect."""
        raise NotImplementedError

    def read_joints(self) -> dict[str, float]:
        """From twin robot/lerobot_backend.py LeRobotArm.read_joints."""
        raise NotImplementedError

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """From twin robot/lerobot_backend.py LeRobotArm.send_joints."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """From twin robot/lerobot_backend.py LeRobotArm.is_connected."""
        raise NotImplementedError

    @property
    def link_silent(self) -> bool:
        """From twin robot/base.py RobotBackend.link_silent."""
        raise NotImplementedError

    def faults(self) -> list[str]:
        """From twin robot/lerobot_backend.py LeRobotArm.faults."""
        raise NotImplementedError

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """From twin robot/lerobot_backend.py LeRobotArm.joint_limits."""
        raise NotImplementedError
