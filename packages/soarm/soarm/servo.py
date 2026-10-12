"""Servo tools: scan the bus, inspect one servo, read EEPROM limits, turn torque off. Stub.

The code comes from galaxeo-manipulators `so101_feetech.py` (bus reads) and the twin's
`robot/feetech.py` (register addresses, error bits, EEPROM limits). Ticket #145.
Every function here reads only, except `torque_off`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoState:
    """One servo as `inspect` sees it. Positions and limits in raw ticks (0..4095)."""

    servo_id: int
    position: int
    voltage: float
    temperature: int
    min_limit: int
    max_limit: int
    error_bits: int


def scan(port: str, baudrate: int = 1_000_000, ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> list[int]:
    """Ping each ID and return the ones that answer."""
    raise NotImplementedError("scan moves here from galaxeo so101_feetech.py FeetechBus.ping (#145)")


def inspect(port: str, servo_id: int, baudrate: int = 1_000_000) -> ServoState:
    """Read position, voltage, temperature, EEPROM limits and the error byte of one servo."""
    raise NotImplementedError("inspect moves here from twin robot/feetech.py (#145)")


def read_limits(port: str, ids: list[int], baudrate: int = 1_000_000) -> dict[int, tuple[int, int]]:
    """Min/Max angle limits from each servo's EEPROM, in ticks."""
    raise NotImplementedError("read_limits moves here from twin FeetechArm._read_servo_limits (#145)")


def torque_off(port: str, ids: list[int], baudrate: int = 1_000_000) -> None:
    """Release torque on every listed servo. The arm drops under its own weight."""
    raise NotImplementedError("torque_off moves here from twin robot/feetech.py (#145)")
