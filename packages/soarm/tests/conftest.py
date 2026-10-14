from __future__ import annotations

import pytest
import yaml

import soarm


@pytest.fixture(scope="session")
def robot_yaml() -> dict:
    """`robot.yaml` of the SO-101 as plain data."""
    return yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))
