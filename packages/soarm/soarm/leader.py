"""Read-only leader arm. Stub.

The code comes from galaxeo-manipulators `so101_feetech.py` (`SO101Raw`). Ticket #145.
A leader is an SO-101 with torque off that an operator moves by hand.
This module never enables torque and never writes a register.
galaxeo returns joints in radians. Here they come back in degrees and the gripper in 0..100,
like every backend.
"""

from __future__ import annotations


class LeaderArm:
    """Positions of a hand-guided SO-101, read over its Feetech bus."""

    def __init__(self, port: str, calibration: str | None = None) -> None:
        raise NotImplementedError("LeaderArm moves here from galaxeo so101_feetech.py SO101Raw (#145)")

    def read(self) -> dict[str, float]:
        """From galaxeo SO101Raw.read: one sync read of all present servos."""
        raise NotImplementedError

    def close(self) -> None:
        """From galaxeo SO101Raw.close."""
        raise NotImplementedError
