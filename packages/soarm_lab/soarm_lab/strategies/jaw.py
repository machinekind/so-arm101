"""Infer jaw contact from the measured angle and the servo command.

`still` means motion below `jaw.still` since the previous reading. `squeezing` means the
jaw is wider than both its command and its closed limit by more than `jaw.margin`.
`holding` requires both. `open` means wider than `jaw.open` without squeezing.
`crept` means the jaw closed by more than `jaw.creep` since `mark`.

Before accepting a hold, the caller must see the jaw leave its opening. A jaw that has
not moved can satisfy the holding rule against a tighter command. Call `mark` when a
hold starts and check `crept` each tick to detect slow closing.

Angles and thresholds use the gripper joint's MJCF units. The twin's tracker uses the
same thresholds from `RobotSpec.jaw`."""

from __future__ import annotations

from dataclasses import dataclass

from ..robot import JawThresholds


@dataclass(frozen=True)
class JawReading:
    """One tick's jaw reading in the gripper joint's MJCF units."""

    #: The measured jaw angle and the command the servo drives toward now.
    q: float
    cmd: float
    still: bool
    squeezing: bool
    holding: bool
    open: bool
    crept: bool


class JawMonitor:
    """Track jaw contact from each tick's measured angle and servo command."""

    def __init__(self, jaw: JawThresholds, closed_q: float):
        self.jaw = jaw
        #: Closed angle from the gripper's hardware limit [MJCF units].
        self.closed_q = float(closed_q)
        self.last: JawReading | None = None
        self._prev: float | None = None
        self._reference: float | None = None

    def reset(self) -> None:
        """Forgets the previous reading and the creep reference."""
        self.last = None
        self._prev = None
        self._reference = None

    def update(self, q: float, cmd: float) -> JawReading:
        """Return this tick's jaw reading.

        `cmd` is the current supervisor command. Use the previous target if the observation
        has no command."""
        jaw, q, cmd = self.jaw, float(q), float(cmd)
        still = self._prev is not None and abs(q - self._prev) < jaw.still
        squeezing = q - cmd > jaw.margin and q > self.closed_q + jaw.margin
        crept = self._reference is not None and q < self._reference - jaw.creep
        reading = JawReading(q=q, cmd=cmd, still=still, squeezing=squeezing, holding=still and squeezing,
                             open=q > jaw.open and not squeezing, crept=crept)
        self._prev = q
        self.last = reading
        return reading

    def mark(self) -> None:
        """Set the creep reference to the last measured angle."""
        self._reference = None if self.last is None else self.last.q

    @property
    def holding(self) -> bool:
        """Whether the last reading indicates a hold. False before the first reading."""
        return self.last is not None and self.last.holding
