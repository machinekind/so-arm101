"""Block lookup spans catalogs and includes the default cube."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab import blocks  # noqa: E402


def test_the_3cm_cube():
    spec = blocks.block("cube3cm")
    assert spec.name == "cube3cm"
    assert spec.half_size == (0.015, 0.015, 0.015)
    assert spec.mass == 0.03
    assert spec.yaw_period == np.pi / 2
    assert [p.kind for p in spec.parts] == ["box"] and spec.parts[0].size == spec.half_size


def test_a_name_is_found_in_any_catalog():
    assert blocks.block("cylinder") is blocks.classic_world.BLOCKS["cylinder"]
    assert blocks.block("cube3cm").name == "cube3cm"


def test_an_unknown_name_says_which_are_known():
    with pytest.raises(ValueError, match=r"unknown block 'arch'; known: .*cube3cm.*cylinder"):
        blocks.block("arch")


def test_no_name_is_in_two_catalogs():
    names = [name for catalog in blocks._CATALOGS for name in catalog]
    assert len(names) == len(set(names))
