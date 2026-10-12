from __future__ import annotations

from importlib import metadata

import pytest

from soarm.spec import RobotSpec

EXPECTED = {
    ("lerobot_mp.robots", "so101"): "soarm.spec:so101",
    ("lerobot_mp.backends", "feetech"): "soarm.feetech:FeetechArm",
    ("lerobot_mp.backends", "bridge"): "soarm.bridge:BridgeArm",
    ("lerobot_mp.backends", "lerobot"): "soarm.lerobot:LeRobotArm",
    ("console_scripts", "soarm"): "soarm.cli:main",
}


@pytest.fixture(scope="module")
def entry_points() -> dict[tuple[str, str], metadata.EntryPoint]:
    try:
        dist = metadata.distribution("soarm")
    except metadata.PackageNotFoundError:
        pytest.skip("soarm is not installed; run `uv sync`")
    return {(ep.group, ep.name): ep for ep in dist.entry_points}


@pytest.mark.parametrize(("key", "value"), EXPECTED.items(), ids=lambda x: str(x))
def test_entry_point_resolves(entry_points: dict, key: tuple[str, str], value: str) -> None:
    ep = entry_points[key]
    assert ep.value == value
    assert callable(ep.load())


def test_robot_entry_point_returns_a_spec(entry_points: dict) -> None:
    spec = entry_points[("lerobot_mp.robots", "so101")].load()()
    assert isinstance(spec, RobotSpec)
    assert spec.name == "so101"
