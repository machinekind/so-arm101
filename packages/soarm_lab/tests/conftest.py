from __future__ import annotations

import pytest

import soarm


@pytest.fixture
def twin_robots():
    """The twin's robot discovery, checked to resolve `so101` to this repository's robot folder.

    Tests that build a twin (`Workspace`, `Twin`) take it, so a twin that found another `so101` fails here
    and not deep inside a run.
    """
    from machinekind_twin import robots

    assert robots.robot_folders()["so101"] == soarm.ROBOT_DIR.resolve()
    robots.get_spec.cache_clear()
    yield robots
    robots.get_spec.cache_clear()


@pytest.fixture
def square_end(monkeypatch):
    """Register a block with a square end and its long axis along body x. Return its name."""
    import math

    from soarm_lab import blocks

    size = (60.0, 25.0, 25.0)
    mass = blocks.classic_world.DENSITY * math.prod(size)
    spec = blocks.box_block("square_end", size, mass, yaw_period=math.pi)
    monkeypatch.setitem(blocks.classic_world.BLOCKS, spec.name, spec)
    return spec.name


@pytest.fixture(scope="module")
def station():
    """The bundled simulated station `sim`."""
    from soarm_lab.station import Station

    return Station.load("sim")
