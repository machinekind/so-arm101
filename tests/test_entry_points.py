from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest

#: Each distribution's entry points: (group, name) -> value.
EXPECTED = {
    "soarm": {
        ("lerobot_mp.robots", "so101"): "soarm:ROBOT_DIR",
        ("lerobot_mp.backends", "feetech"): "soarm.feetech:create",
        ("lerobot_mp.backends", "lerobot"): "soarm.lerobot_backend:create",
        ("console_scripts", "soarm-bridge"): "soarm.serial_bridge:main",
    },
    "soarm-lab": {
        ("lerobot_mp.strategies", "policy"): "soarm_lab.strategies.policy:PolicyStrategy",
        ("console_scripts", "soarm-lab"): "soarm_lab.cli:main",
    },
}
CALLABLE = [(dist, key) for dist, eps in EXPECTED.items() for key in eps if key[0] != "lerobot_mp.robots"]


def own_entry_points(dist: str) -> dict[tuple[str, str], metadata.EntryPoint]:
    """The distribution's own entry points (the twin registers the robot's names too)."""
    try:
        found = metadata.distribution(dist)
    except metadata.PackageNotFoundError:
        pytest.skip(f"{dist} is not installed; run `uv sync`")
    return {(ep.group, ep.name): ep for ep in found.entry_points}


@pytest.mark.parametrize("dist", list(EXPECTED))
def test_exactly_the_expected_entry_points(dist: str) -> None:
    assert {key: ep.value for key, ep in own_entry_points(dist).items()} == EXPECTED[dist]


def test_robot_entry_point_is_a_folder_with_robot_yaml() -> None:
    target = own_entry_points("soarm")[("lerobot_mp.robots", "so101")].load()
    assert isinstance(target, (str, os.PathLike))
    assert (Path(target) / "robot.yaml").is_file()


@pytest.mark.parametrize("dist, key", CALLABLE, ids=lambda k: k if isinstance(k, str) else k[1])
def test_callable_entry_points(dist: str, key: tuple[str, str]) -> None:
    assert callable(own_entry_points(dist)[key].load())


def test_the_strategy_entry_point_is_a_class_with_settings() -> None:
    """What the twin's loader checks (`lerobot_mp.twin.strategies.load_strategy`), without the twin."""
    import dataclasses

    cls = own_entry_points("soarm-lab")[("lerobot_mp.strategies", "policy")].load()
    assert isinstance(cls, type) and callable(getattr(cls, "act", None))
    assert dataclasses.is_dataclass(cls.settings)
    kinds = {f.name: f.type for f in dataclasses.fields(cls.settings)}
    assert set(kinds.values()) <= {"str", "int", "float", "bool"}
