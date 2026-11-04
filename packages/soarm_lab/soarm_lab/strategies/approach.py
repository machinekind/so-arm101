"""The approach protocol of `calibrate_approach` and `pick_place`: final-leg directions and place heights.

The hand reaches a place in the same way when the approach table is measured and when it is used: up to the
carry height, across to the lead point at that height, a straight horizontal leg of `lead` metres along
the direction to the point, then straight down. The direction is relative to the arm: `out` and `in` run
along the line from the base's turning axis to the point, `ccw` and `cw` across it, turning the arm
counter-clockwise or clockwise seen from above.

The heights follow `pick_place`'s place rule: the release leaves a gap and a slip margin under the held
block, and the carry clears the target and the catalog's tallest block. The caller passes the constants in,
so this module imports no other strategy.
"""

from __future__ import annotations

import math

import numpy as np

#: Final-leg directions, in the order a calibration visits them.
DIRECTIONS = ("out", "in", "ccw", "cw")
#: Height step for lower carry heights when the wanted one is out of reach [m].
HEIGHT_STEP_M = 0.005


def direction_vector(name: str, xy: np.ndarray | tuple[float, float], pan_xy: np.ndarray | tuple[float, float]
                     ) -> np.ndarray:
    """The unit direction (2,) of a final leg named `name` that ends over `xy` [m, base frame].

    `pan_xy` is where the base's turning axis crosses the table. Radial `r` points from it to `xy`:
    out = r, in = -r, ccw = r turned by +90 degrees, cw = -ccw.
    """
    r = np.asarray(xy, float)[:2] - np.asarray(pan_xy, float)[:2]
    norm = float(np.linalg.norm(r))
    if not norm > 1e-6:
        raise ValueError("a direction needs a point away from the base's turning axis")
    r = r / norm
    t = np.array([-r[1], r[0]])
    vectors = {"out": r, "in": -r, "ccw": t, "cw": -t}
    if name not in vectors:
        raise ValueError(f"unknown direction {name!r}; known: {', '.join(DIRECTIONS)}")
    return vectors[name]


def lead_point(xy: np.ndarray | tuple[float, float], d: np.ndarray, lead: float) -> np.ndarray:
    """Where the final leg along unit `d` starts, `lead` metres before `xy` (2,)."""
    return np.asarray(xy, float)[:2] - float(lead) * np.asarray(d, float)


def place_heights(top: float, hang: float, gap: float, slip: float, table_top: float, clearance: float,
                  min_clear: float, over: float = 0.0) -> tuple[float, float, float]:
    """TCP heights of a place onto surface `top` [m above the table]: (release, wanted carry, lowest carry).

    The held block hangs `hang` below the TCP. The release leaves `gap` plus `slip` under it. The carry
    clears the highest of `top`, `table_top` (the tallest block on the table) and `over` by `clearance`,
    and by at least `min_clear` when the wanted height is out of reach.
    """
    floor = max(top, table_top, over)
    release = top + gap + slip + hang
    low = floor + min_clear + hang
    want = max(floor + clearance + hang, low)
    return release, want, low


def carry_heights(want: float, low: float, step: float = HEIGHT_STEP_M) -> list[float]:
    """Carry heights from `want` down to `low` in `step` steps [m], to try in order."""
    out = [want]
    while out[-1] - step > low + 1e-9:
        out.append(out[-1] - step)
    if low < want - 1e-9:
        out.append(low)
    return out


def bearing(xy: np.ndarray | tuple[float, float], pan_xy: np.ndarray | tuple[float, float]) -> float:
    """The direction of `xy` seen from the base's turning axis [rad]."""
    r = np.asarray(xy, float)[:2] - np.asarray(pan_xy, float)[:2]
    return math.atan2(float(r[1]), float(r[0]))
