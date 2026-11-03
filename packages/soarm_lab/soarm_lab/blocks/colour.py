"""Random block colours."""

from __future__ import annotations

import colorsys

import numpy as np

Rgba = tuple[float, float, float, float]

#: Scenes start with this bare wood colour.
WOOD: Rgba = (0.84, 0.74, 0.65, 1.0)
#: Share of draws that give bare wood instead of paint.
WOOD_SHARE = 0.2
#: Hue [deg], saturation and value ranges of bare wood.
WOOD_HSV = ((25.0, 40.0), (0.15, 0.35), (0.7, 0.9))
#: Hue [deg], saturation and value ranges of paint.
PAINT_HSV = ((0.0, 360.0), (0.45, 0.85), (0.5, 1.0))


def random_rgba(rng: np.random.Generator) -> Rgba:
    """Draw paint of any hue, or bare wood with probability `WOOD_SHARE`."""
    (h0, h1), (s0, s1), (v0, v1) = WOOD_HSV if rng.uniform() < WOOD_SHARE else PAINT_HSV
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(h0, h1) / 360.0 % 1.0, rng.uniform(s0, s1), rng.uniform(v0, v1))
    return (r, g, b, 1.0)
