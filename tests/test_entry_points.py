from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest

EXPECTED = {
    ("lerobot_mp.robots", "so101"): "soarm:ROBOT_DIR",
    ("lerobot_mp.backends", "feetech"): "soarm.feetech:create",
    ("lerobot_mp.backends", "lerobot"): "soarm.lerobot_backend:create",
    ("console_scripts", "soarm-bridge"): "soarm.serial_bridge:main",
}


@pytest.fixture(scope="module")
def entry_points() -> dict[tuple[str, str], metadata.EntryPoint]:
    """This distribution's own entry points (the twin registers the same names; see test_twin_contract)."""
    try:
        dist = metadata.distribution("soarm")
    except metadata.PackageNotFoundError:
        pytest.skip("soarm is not installed; run `uv sync`")
    return {(ep.group, ep.name): ep for ep in dist.entry_points}


def test_exactly_the_expected_entry_points(entry_points: dict) -> None:
    assert {key: ep.value for key, ep in entry_points.items()} == EXPECTED


def test_robot_entry_point_is_a_folder_with_robot_yaml(entry_points: dict) -> None:
    target = entry_points[("lerobot_mp.robots", "so101")].load()
    assert isinstance(target, (str, os.PathLike))
    assert (Path(target) / "robot.yaml").is_file()


@pytest.mark.parametrize("key", [k for k in EXPECTED if k[0] != "lerobot_mp.robots"], ids=lambda k: k[1])
def test_callable_entry_points(entry_points: dict, key: tuple[str, str]) -> None:
    assert callable(entry_points[key].load())
