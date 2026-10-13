from __future__ import annotations

import pytest
import yaml

import soarm


@pytest.fixture(autouse=True)
def _no_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests count on defaults: a file named by LEROBOT_MP_CONFIG in the shell would change them."""
    monkeypatch.delenv("LEROBOT_MP_CONFIG", raising=False)


@pytest.fixture(scope="session")
def robot_yaml() -> dict:
    """`robot.yaml` of the SO-101 as plain data."""
    return yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
