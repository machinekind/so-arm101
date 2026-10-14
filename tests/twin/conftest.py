"""Tests that run `soarm` and `soarm-lab` through the twin's own code (`lerobot_mp`).

Only the tests in this folder import the twin. Without it, the whole folder skips.
"""

from __future__ import annotations

import pytest

import soarm

pytest.importorskip("lerobot_mp")


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests count on the twin's defaults; a file named by LEROBOT_MP_CONFIG would change them."""
    monkeypatch.delenv("LEROBOT_MP_CONFIG", raising=False)


@pytest.fixture
def twin_robots(monkeypatch: pytest.MonkeyPatch):
    """The twin's robot discovery with `so101` taken from `soarm` only.

    The twin registers an SO-101 of its own under the same name, so with both installed its discovery
    raises on the duplicate (see test_robot_and_backends.py). Tests that build a twin (`Workspace`,
    `Twin`) pin discovery to this repository's robot folder instead.
    """
    from lerobot_mp.twin import robots

    monkeypatch.setattr(robots, "robot_folders", lambda: {"so101": soarm.ROBOT_DIR.resolve()})
    robots.get_spec.cache_clear()
    yield robots
    robots.get_spec.cache_clear()
