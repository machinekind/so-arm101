"""Random block colours."""

from __future__ import annotations

import colorsys

import numpy as np
import pytest

from soarm_lab.blocks import colour


def hsv(rgba):
    assert len(rgba) == 4 and rgba[3] == 1.0
    assert all(0.0 <= c <= 1.0 for c in rgba)
    h, s, v = colorsys.rgb_to_hsv(*rgba[:3])
    return h * 360.0, s, v


def test_draws_cover_every_hue_and_some_bare_wood():
    rng = np.random.default_rng(0)
    draws = [hsv(colour.random_rgba(rng)) for _ in range(2000)]
    wood = [d for d in draws if d[1] <= colour.WOOD_HSV[1][1] + 1e-9]
    paint = [d for d in draws if d[1] >= colour.PAINT_HSV[1][0] - 1e-9]
    assert len(wood) + len(paint) == len(draws)
    assert len(wood) / len(draws) == pytest.approx(colour.WOOD_SHARE, abs=0.03)
    assert all(colour.WOOD_HSV[0][0] - 1e-6 <= h <= colour.WOOD_HSV[0][1] + 1e-6 for h, _, _ in wood)
    hues = np.histogram([h for h, _, _ in paint], bins=12, range=(0.0, 360.0))[0]
    assert hues.min() > 0.5 * hues.mean()


def test_the_same_generator_state_gives_the_same_colour():
    assert colour.random_rgba(np.random.default_rng(3)) == colour.random_rgba(np.random.default_rng(3))
    assert colour.random_rgba(np.random.default_rng(3)) != colour.random_rgba(np.random.default_rng(4))


def test_scenes_start_with_bare_wood():
    h, s, v = hsv(colour.WOOD)
    assert colour.WOOD_HSV[0][0] <= h <= colour.WOOD_HSV[0][1]
    assert colour.WOOD_HSV[1][0] <= s <= colour.WOOD_HSV[1][1]
    assert colour.WOOD_HSV[2][0] <= v <= colour.WOOD_HSV[2][1]
