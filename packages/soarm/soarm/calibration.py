"""Conversion between a LeRobot calibration file and servo ticks. Stub.

LeRobot stores one entry per motor: id, drive_mode, homing_offset, range_min, range_max.
The twin's `feetech` backend instead assumes joint zero at `center_ticks` and maps the gripper
through `gripper_closed_ticks..gripper_open_ticks`. This module will convert one into the other,
using the logic in the twin's `robot/lerobot_backend.py` (`gripper_ticks`, `joint_zero_offsets`). Ticket #145.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotorCalibration:
    """One motor entry of a LeRobot calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def read_lerobot(path: str | Path) -> dict[str, MotorCalibration]:
    """Parse a LeRobot calibration JSON file."""
    raise NotImplementedError("read_lerobot is new; it follows twin robot/lerobot_backend.py (#145)")


def to_ticks(calibration: dict[str, MotorCalibration], gripper: str = "gripper") -> dict[str, int]:
    """Centre tick of each joint and the closed/open ticks of the gripper."""
    raise NotImplementedError("to_ticks follows twin LeRobotArm.gripper_ticks and joint_zero_offsets (#145)")
