from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import soarm
from soarm.spec import RobotSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
SOARM_ROOT = REPO_ROOT / "packages" / "soarm"
SOARM_LAB_ROOT = REPO_ROOT / "packages" / "soarm_lab"

ROBOT_DIRS = sorted(p for p in soarm.data_dir("robot").iterdir() if (p / "robot.yaml").is_file())


def strict(spec: RobotSpec) -> RobotSpec:
    if spec.is_draft:
        pytest.skip(f"{spec.name} is a draft")
    return spec


@pytest.fixture(params=ROBOT_DIRS, ids=lambda p: p.name)
def any_spec(request: pytest.FixtureRequest) -> RobotSpec:
    return soarm.load(request.param)


@pytest.fixture(params=ROBOT_DIRS, ids=lambda p: p.name)
def ready_spec(request: pytest.FixtureRequest) -> RobotSpec:
    return strict(soarm.load(request.param))


@pytest.fixture
def so101_data() -> dict:
    return yaml.safe_load((soarm.robot_dir("so101") / "robot.yaml").read_text(encoding="utf-8"))
