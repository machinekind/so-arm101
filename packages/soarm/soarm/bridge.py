"""An arm over the network. Stub.

Two halves come from the twin's `src/lerobot_mp/robot/serial_bridge.py` (ticket #145).
`serve` runs on the machine the arm is plugged into and exposes its serial port over TCP.
`BridgeArm` runs on the remote machine and drives the arm through `socket://host:port`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SOURCE = "twin src/lerobot_mp/robot/serial_bridge.py"


def serve(open_link: Callable[[], Any], host: str, port: int, allow: tuple[str, ...] = ()) -> None:
    """From twin serial_bridge.serve: one TCP client at a time, port opened per connection."""
    raise NotImplementedError(f"serve moves here from {SOURCE} (#145)")


class BridgeArm:
    """SO-101 behind a remote `serve`. Implements `soarm.backend.Backend`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"BridgeArm moves here from {SOURCE} (#145)")

    def connect(self) -> None:
        """From twin robot/feetech.py FeetechArm.connect over a socket:// link (serial_bridge.py)."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """From twin robot/feetech.py FeetechArm.disconnect over a socket:// link (serial_bridge.py)."""
        raise NotImplementedError

    def read_joints(self) -> dict[str, float]:
        """From twin robot/feetech.py FeetechArm.read_joints over a socket:// link (serial_bridge.py)."""
        raise NotImplementedError

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]:
        """From twin robot/feetech.py FeetechArm.send_joints over a socket:// link (serial_bridge.py)."""
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
