"""The jaw rule: whether the gripper holds something, from its measured angle and its command alone.

The Feetech backend reads positions, not load, so the jaw is the only contact signal the arm has. A jaw
holds something when it stands still wider than it was commanded and wider than closed: something keeps it
apart. With `JawThresholds` of the robot description (`RobotSpec.jaw`, the gripper joint's MJCF units),
one tick's reading is

* `still`: the jaw moved less than `jaw.still` since the previous reading;
* `squeezing`: it stands wider than its command by more than `jaw.margin`, and wider than closed by more;
* `holding`: still and squeezing;
* `open`: wider than `jaw.open` and not squeezing, so surely empty;
* `crept`: it closed by more than `jaw.creep` since `mark` (a block slipping out closes the jaw slowly).

Two caveats a caller must add. A jaw that has not started closing yet stands still at its opening and is
`squeezing` against a tighter command, so a one-shot verdict must first see the jaw leave its opening, as
`grasp` does before it counts a hold. And `still` is per tick: a jaw closing slowly stands from tick to
tick, so a hold streak needs `mark` at its start and `crept` on every tick.

The twin's `ObjectTracker` applies the same thresholds (`ObjectTracker.for_spec`), so it reports a block
"w dloni" on the tick this monitor says holding (up to the tracker's own update period). `policy.py`
keeps a private copy of the rule for now; moving it here is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..robot import JawThresholds


@dataclass(frozen=True)
class JawReading:
    """One tick of the jaw rule. Angles in the gripper joint's MJCF units (rad on the SO-101)."""

    #: The measured jaw angle and the command the servo drives toward now.
    q: float
    cmd: float
    still: bool
    squeezing: bool
    holding: bool
    open: bool
    crept: bool


class JawMonitor:
    """The jaw rule over the ticks of a run: feed it every tick's measured angle and command."""

    def __init__(self, jaw: JawThresholds, closed_q: float):
        self.jaw = jaw
        #: The jaw's closed angle (the gripper's hardware limit), as the twin's hand has it.
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
        """The reading of this tick.

        `cmd` is the command the servo drives toward now: the supervisor's command (`obs.command`), or the
        previous tick's target when the observation has none.
        """
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
        """The start of a hold streak: `crept` counts from the last measured angle on."""
        self._reference = None if self.last is None else self.last.q

    @property
    def holding(self) -> bool:
        """Whether the last reading holds; False before any."""
        return self.last is not None and self.last.holding
