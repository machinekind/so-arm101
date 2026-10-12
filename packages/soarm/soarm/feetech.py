"""Feetech STS3215 servo bus driver and the `feetech` backend. Stub.

The code comes from the twin's `src/lerobot_mp/robot/feetech.py` (ticket #145).
It talks to the servos directly over pyserial, without LeRobot and without torch.
"""

from __future__ import annotations

from typing import Any

SOURCE = "twin src/lerobot_mp/robot/feetech.py"


class FeetechBus:
    """Feetech protocol on one serial port or `socket://host:port` link."""

    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02) -> None:
        raise NotImplementedError(f"FeetechBus moves here from {SOURCE} (#145)")

    def open(self, link: Any = None) -> None:
        """From twin FeetechBus.open."""
        raise NotImplementedError

    def close(self) -> None:
        """From twin FeetechBus.close."""
        raise NotImplementedError

    def ping(self, dev_id: int) -> bool:
        """From twin FeetechBus.ping."""
        raise NotImplementedError

    def read(self, dev_id: int, addr: int, length: int) -> int | None:
        """From twin FeetechBus.read."""
        raise NotImplementedError

    def write(self, dev_id: int, addr: int, value: int, length: int) -> None:
        """From twin FeetechBus.write."""
        raise NotImplementedError

    def sync_read(self, addr: int, length: int, ids: list[int]) -> dict[int, int]:
        """From twin FeetechBus.sync_read."""
        raise NotImplementedError

    def sync_write(self, addr: int, length: int, values: dict[int, int]) -> None:
        """From twin FeetechBus.sync_write."""
        raise NotImplementedError


class FeetechArm:
    """SO-101 driven straight over the serial port. Implements `soarm.backend.Backend`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"FeetechArm moves here from {SOURCE} (#145)")

    def connect(self) -> None:
        """From twin robot/feetech.py FeetechArm.connect."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """From twin robot/feetech.py FeetechArm.disconnect."""
        raise NotImplementedError

    def read_joints(self) -> dict[str, float]:
        """From twin robot/feetech.py FeetechArm.read_joints."""
        raise NotImplementedError

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """From twin robot/feetech.py FeetechArm.send_joints."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """From twin robot/feetech.py FeetechArm.is_connected."""
        raise NotImplementedError

    @property
    def link_silent(self) -> bool:
        """From twin robot/base.py RobotBackend.link_silent."""
        raise NotImplementedError

    def faults(self) -> list[str]:
        """From twin robot/feetech.py FeetechArm.faults."""
        raise NotImplementedError

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """From twin robot/feetech.py FeetechArm.joint_limits."""
        raise NotImplementedError
