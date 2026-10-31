"""The jaw rule on plain sequences of (measured, commanded) angles: no model, no arm, no twin."""

from __future__ import annotations

import pytest

from soarm_lab.robot import JawThresholds
from soarm_lab.strategies.jaw import JawMonitor, JawReading

#: The SO-101's closed jaw [rad], the gripper's hardware limit (`gripper_range_q[0]`).
CLOSED_Q = -0.0951


def readings(pairs: list[tuple[float, float]], monitor: JawMonitor | None = None) -> list[JawReading]:
    monitor = monitor or JawMonitor(JawThresholds(), closed_q=CLOSED_Q)
    return [monitor.update(q, cmd) for q, cmd in pairs]


def test_a_jaw_closing_toward_its_command_is_not_holding():
    out = readings([(0.5, 0.1), (0.4, 0.1), (0.3, 0.1)])

    assert all(r.squeezing for r in out)
    assert not any(r.still for r in out) and not any(r.holding for r in out)


def test_a_jaw_stalled_wider_than_its_command_holds_once_it_stands():
    """The twin's tracker uses the same thresholds (`ObjectTracker.for_spec`: still 0.02, margin 0.05)."""
    first, second = readings([(0.11, 0.009), (0.11, 0.009)])

    assert JawThresholds().still == 0.02 and JawThresholds().margin == 0.05
    assert first.squeezing and not first.still and not first.holding
    assert second.still and second.squeezing and second.holding


def test_a_jaw_standing_at_its_opening_against_a_tighter_command_reads_holding():
    """The caveat the module documents: the rule alone cannot tell a jaw that never moved from one stalled
    on a block, so a caller must see the jaw leave its opening first, as grasp's close does."""
    _, second = readings([(0.314, 0.009), (0.314, 0.009)])

    assert second.holding


def test_a_jaw_at_its_command_or_closed_is_empty():
    _, at_command = readings([(0.1, 0.1), (0.1, 0.1)])
    _, closed = readings([(-0.09, -0.095), (-0.09, -0.095)])

    assert at_command.still and not at_command.squeezing and not at_command.holding
    assert closed.q - closed.cmd > 0 and not closed.squeezing and not closed.holding


def test_creep_counts_from_mark():
    monitor = JawMonitor(JawThresholds(), closed_q=CLOSED_Q)
    monitor.update(0.11, 0.009)
    monitor.mark()

    assert not monitor.update(0.09, 0.009).crept                        # 0.02 < creep 0.03
    assert monitor.update(0.075, 0.009).crept
    monitor.reset()
    assert monitor.last is None and not monitor.holding
    assert not monitor.update(0.05, 0.009).crept


def test_the_open_flag_needs_a_wide_jaw_that_is_not_squeezing():
    (wide,) = readings([(0.5, 0.5)])
    _, squeezing = readings([(0.5, 0.3), (0.5, 0.3)])

    assert wide.open and not wide.squeezing
    assert squeezing.still and squeezing.squeezing and not squeezing.open


def test_holding_is_false_before_any_reading_and_follows_the_last_one():
    monitor = JawMonitor(JawThresholds(), closed_q=CLOSED_Q)
    assert not monitor.holding and monitor.last is None

    readings([(0.11, 0.009), (0.11, 0.009)], monitor)
    assert monitor.holding and monitor.last is not None and monitor.last.q == pytest.approx(0.11)
    monitor.update(0.2, 0.2)
    assert not monitor.holding
