from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest
from machinekind_twin_api.backend import BackendPlugin, RobotBackend, RobotInfo
from machinekind_twin_api.entry_points import BACKENDS, ROBOTS

#: The distribution's entry points: (group, name) -> value.
EXPECTED = {
    "soarm": {
        (ROBOTS, "so101"): "soarm:ROBOT_DIR",
        (BACKENDS, "feetech"): "soarm.feetech:create",
        (BACKENDS, "lerobot"): "soarm.lerobot_backend:create",
        ("console_scripts", "soarm-bridge"): "soarm.serial_bridge:main",
        ("console_scripts", "soarm-panel"): "soarm.panel:main",
        ("console_scripts", "soarm-zero"): "soarm.zero:main",
    },
}
CALLABLE = [(dist, key) for dist, eps in EXPECTED.items() for key in eps if key[0] != ROBOTS]


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
    target = own_entry_points("soarm")[(ROBOTS, "so101")].load()
    assert isinstance(target, (str, os.PathLike))
    assert (Path(target) / "robot.yaml").is_file()


@pytest.mark.parametrize("dist, key", CALLABLE, ids=lambda k: k if isinstance(k, str) else k[1])
def test_callable_entry_points(dist: str, key: tuple[str, str]) -> None:
    assert callable(own_entry_points(dist)[key].load())


@pytest.mark.parametrize("name", ["feetech", "lerobot"])
def test_a_backend_factory_builds_a_backend_plugin(name: str, so101_context) -> None:
    """What the twin's factory does with the entry point, without the twin: `create(ctx)`, nothing opened."""
    backend = own_entry_points("soarm")[(BACKENDS, name)].load()(so101_context())
    assert isinstance(backend, BackendPlugin)
    # A `RobotBackend` already, so the twin takes it as it is (test_twin_contract.py).
    assert isinstance(backend, RobotBackend) and isinstance(backend.info, RobotInfo)
    assert not backend.is_connected
