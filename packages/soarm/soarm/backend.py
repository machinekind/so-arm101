"""The interface every arm backend in this package implements.

It has the same shape as the twin's `RobotBackend` ABC in `lerobot_mp/robot/base.py`.
`soarm` never imports the twin, so the shape is restated here as a `Protocol`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joints(self) -> dict[str, float]: ...

    def send_joints(self, targets: dict[str, float]) -> dict[str, float]: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def link_silent(self) -> bool: ...

    def faults(self) -> list[str]: ...

    def joint_limits(self) -> dict[str, tuple[float, float]]: ...
