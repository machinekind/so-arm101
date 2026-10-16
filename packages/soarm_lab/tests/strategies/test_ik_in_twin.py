"""The twin finds the lab's IK strategy, builds it from its settings and runs it in the simulated twin.

Everything goes through the twin's own code: discovery and strict settings (`lerobot_mp.twin.strategies`),
the run (`StrategyRunner`), the safety supervisor and the twin's physics (`Twin`, simulated, its loop
without a thread on a virtual clock). The strategy package itself imports nothing from the twin.

The twin's robot discovery is pinned to `soarm`'s folder (fixture `twin_robots`): the twin registers an
SO-101 of its own under the same name.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.twin
pytest.importorskip("mujoco")
pytest.importorskip("lerobot_mp")

from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.strategies import load_strategy, strategies  # noqa: E402
from lerobot_mp.twin.strategy import StrategyRunner  # noqa: E402
from lerobot_mp.twin.workspace import Workspace  # noqa: E402

from soarm_lab.strategies.ik import IKSettings, IKStrategy  # noqa: E402

NAME = "ik"
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02


@pytest.fixture
def twin(twin_robots):
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def drive(twin: Twin, runner: StrategyRunner, seconds: float) -> None:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock."""
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    for k in range(1, round(seconds / CLOCK_STEP) + 1):
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(k * CLOCK_STEP)


def measured_tcp(twin: Twin, strategy: IKStrategy) -> np.ndarray:
    return strategy.kin.tcp(twin.status.measured)[:3, 3]


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_the_strategy_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.ik:IKStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is IKStrategy
    assert plugin.settings is IKSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.ik:IKStrategy)"
    assert plugin.summary.startswith("Inverse kinematics")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == ["goal_x", "goal_y", "goal_z", "hz", "speed_mps", "tol_m", "end_on_reach",
                            "max_tracking_deg"]
    assert fields["end_on_reach"].kind is bool and fields["end_on_reach"].default is True
    assert all(fields[name].kind is float for name in fields if name != "end_on_reach")
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji ik: \['sped_mps'\]"):
        plugin.create({"sped_mps": 0.05})
    with pytest.raises(ValueError, match="ik.speed_mps: potrzeba float"):
        plugin.create({"speed_mps": "0.05"})

    strategy = plugin.create({"speed_mps": 0.05, "hz": 25, "end_on_reach": False})
    assert isinstance(strategy, IKStrategy)
    assert strategy.hz == 25.0
    assert strategy.step_m == pytest.approx(0.002)
    assert strategy.config.end_on_reach is False


# --------------------------------------------------------------------------- runs in the simulated twin


def test_ik_reaches_the_panel_goal_in_the_simulated_twin(twin):
    twin.connect("sim", threaded=False)
    goal = np.array([0.25, 0.08, 0.10])
    strategy = load_strategy(NAME).create({})
    runner = StrategyRunner(twin, strategy, goal=lambda: goal)

    runner.start(threaded=False)
    assert twin.owner == "strategia"
    drive(twin, runner, 4.0)

    assert not runner.status.running
    assert runner.status.done and runner.status.success
    assert runner.status.stopped_because == "zadanie wykonane"
    assert strategy.status.stopped_because == "cel osiagniety"
    assert runner.status.error is not None and runner.status.error <= strategy.config.tol_m
    assert twin.owner is None
    assert twin.status.measured["gripper"] == pytest.approx(strategy.kin.spec.home["gripper"], abs=1.0)
    assert np.linalg.norm(measured_tcp(twin, strategy) - goal) <= strategy.config.tol_m


def test_ik_without_a_goal_drives_to_the_settings_goal(twin):
    """The panel's Strategies tab passes no goal: the strategy's settings hold it."""
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({"goal_x": 0.2, "goal_y": -0.1, "goal_z": 0.08})
    runner = StrategyRunner(twin, strategy)

    runner.start(threaded=False)
    drive(twin, runner, 4.0)

    assert runner.status.success
    assert strategy.goal == pytest.approx([0.2, -0.1, 0.08])
    assert np.linalg.norm(measured_tcp(twin, strategy) - [0.2, -0.1, 0.08]) <= strategy.config.tol_m


def test_without_end_on_reach_it_follows_a_dragged_goal_until_stopped(twin):
    twin.connect("sim", threaded=False)
    goal = np.array([0.25, 0.08, 0.10])
    strategy = load_strategy(NAME).create({"end_on_reach": False})
    runner = StrategyRunner(twin, strategy, goal=lambda: goal)

    runner.start(threaded=False)
    drive(twin, runner, 3.0)
    assert runner.status.running and runner.status.success
    goal = np.array([0.2, -0.1, 0.15])
    drive(twin, runner, 3.0)
    assert runner.status.running and runner.status.success
    assert np.linalg.norm(measured_tcp(twin, strategy) - goal) <= strategy.config.tol_m

    runner.stop()
    assert strategy.status.stopped_because == "zatrzymana"
    assert twin.owner is None


def test_a_goal_out_of_reach_refuses_the_run(twin):
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, load_strategy(NAME).create({}), goal=lambda: np.array([0.6, 0.0, 0.1]))
    start = dict(twin.status.measured)

    with pytest.raises(RuntimeError, match="poza zasiegiem ramienia"):
        runner.start(threaded=False)

    assert twin.owner is None
    assert not runner.status.running
    assert runner.status.stopped_because.startswith("strategia odmowila")
    assert twin.status.measured == pytest.approx(start)
