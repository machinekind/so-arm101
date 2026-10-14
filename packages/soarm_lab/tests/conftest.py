from __future__ import annotations

import pytest

import soarm


@pytest.fixture
def twin_robots(monkeypatch: pytest.MonkeyPatch):
    """The twin's robot discovery with `so101` taken from `soarm` only.

    The twin registers an SO-101 of its own under the same name, so with both installed its discovery
    raises on the duplicate (see soarm's tests/test_twin_contract.py). Tests that build a twin (`Workspace`,
    `Twin`) pin discovery to this repository's robot folder instead.
    """
    from lerobot_mp.twin import robots

    monkeypatch.setattr(robots, "robot_folders", lambda: {"so101": soarm.ROBOT_DIR.resolve()})
    robots.get_spec.cache_clear()
    yield robots
    robots.get_spec.cache_clear()


@pytest.fixture(scope="module")
def station():
    """The bundled simulated station `sim`."""
    from soarm_lab.station import Station

    return Station.load("sim")
