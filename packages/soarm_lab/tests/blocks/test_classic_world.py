"""Catalog geometry and measurements."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from soarm_lab.blocks import classic_world  # noqa: E402


def test_the_catalog_is_the_measured_set():
    """The catalog has three shapes with valid geometry and positive masses."""
    assert tuple(classic_world.BLOCKS) == ("cube", "cuboid", "cylinder")
    for name, spec in classic_world.BLOCKS.items():
        assert spec.name == name
        assert spec.mass > 0 and all(h > 0 for h in spec.half_size)
        assert spec.parts and all(p.kind in ("box", "cylinder") for p in spec.parts)
        assert spec.yaw_period in (np.pi / 2, np.pi)
    assert classic_world.BLOCKS["cube"].yaw_period == np.pi / 2


def test_the_cylinder_lies_on_its_side():
    spec = classic_world.BLOCKS["cylinder"]
    assert spec.half_size[1] == spec.half_size[2] < spec.half_size[0]
    assert spec.parts[0].kind == "cylinder"



def test_the_measured_sizes_and_masses():
    """Half extents [m] match measured sizes, and estimated masses [kg] use the catalog density."""
    want = {"cube": ((0.0145, 0.0145, 0.0145), 0.01707), "cuboid": ((0.03, 0.0145, 0.007), 0.01705),
            "cylinder": ((0.03, 0.0145, 0.0145), 0.02774)}
    for name, (half, mass) in want.items():
        spec = classic_world.BLOCKS[name]
        assert spec.half_size == pytest.approx(half, abs=1e-9)
        assert spec.mass == pytest.approx(mass, abs=1e-5)
