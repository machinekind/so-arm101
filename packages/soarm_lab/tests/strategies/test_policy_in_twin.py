"""The twin finds the lab's policy strategy, builds it from its settings and runs it in the simulated twin.

Everything goes through the twin's own code: discovery and strict settings (`machinekind_twin.strategies`),
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
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as twin_scene  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import ObjectPose, Observation, RunInfo, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

import soarm  # noqa: E402
from soarm_lab.strategies.policy import PolicySettings, PolicyStrategy  # noqa: E402

NAME = "policy"
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


def add_cube(twin: Twin, task) -> None:
    h = task.cube_half
    twin.configure(objects=[twin_scene.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                           mass=task.cube_mass)], grasp_sensors=["cube"])


def sim_cube(twin: Twin) -> dict[str, ObjectPose]:
    """The cube as the twin's simulation knows it, in the base frame (what the panel's "symulacja" gives)."""
    with twin.lock:
        s = twin.scene
        b = s.model.body("cube").id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return {"cube": ObjectPose(position.copy(), rotation.copy(), "symulacja")}


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_the_strategy_through_its_entry_point():
    assert strategies()[NAME].value == "soarm_lab.strategies.policy:PolicyStrategy"

    plugin = load_strategy(NAME)

    assert plugin.cls is PolicyStrategy
    assert plugin.settings is PolicySettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.policy:PolicyStrategy)"
    assert plugin.summary.startswith("An RL policy of soarm-lab")


def test_its_settings_become_the_panel_form():
    fields = {field.name: field for field in load_strategy(NAME).fields}

    assert list(fields) == ["policy", "max_tracking_deg", "episode_limit", "wait_for_cube_s",
                            "goal_x", "goal_y", "goal_z"]
    assert fields["policy"].kind is str and fields["policy"].default == "reach-v3"
    assert fields["episode_limit"].kind is bool and fields["episode_limit"].default is True
    assert all(fields[name].kind is float for name in ("max_tracking_deg", "wait_for_cube_s", "goal_x"))
    assert not any(field.required for field in fields.values())


def test_the_twin_builds_it_strictly():
    plugin = load_strategy(NAME)

    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji policy: \['polcy'\]"):
        plugin.create({"polcy": "lift-v3"})
    with pytest.raises(ValueError, match="policy.max_tracking_deg: potrzeba float"):
        plugin.create({"max_tracking_deg": "25"})

    strategy = plugin.create({"policy": "lift-v3", "wait_for_cube_s": 2})
    assert isinstance(strategy, PolicyStrategy)
    assert strategy.task.name == "lift"
    assert strategy.hz == 20.0
    assert strategy.config.wait_for_cube_s == 2.0


def test_start_and_act_take_the_twins_own_types(twin):
    """`RunInfo` and `Observation` as the twin builds them, without its runner."""
    twin.connect("sim", threaded=False)
    spec = twin.workspace.spec()
    run = RunInfo(robot=spec.name, joints=spec.joints, gripper=spec.gripper, limits=twin.joint_limits(),
                  max_vel=dict(spec.max_vel), folder=spec.root, simulated=True)
    joints = dict(twin.status.measured)
    strategy = load_strategy(NAME).create({})

    strategy.start(run, Observation(t=0.0, step=0, joints=joints, command=dict(twin.status.command)))
    goal = np.array([0.2, 0.0, 0.1])
    action = strategy.act(Observation(t=0.05, step=0, joints=joints, command={}, goal=goal))

    assert spec.root == soarm.ROBOT_DIR.resolve()
    assert set(action.target) == set(spec.joints)
    assert not action.done
    assert action.error == pytest.approx(strategy.status.distance)


# --------------------------------------------------------------------------- runs in the simulated twin


def test_reach_v3_reaches_the_panel_goal_in_the_simulated_twin(twin):
    twin.connect("sim", threaded=False)
    goal = np.array([0.25, 0.08, 0.10])
    strategy = load_strategy(NAME).create({"policy": "reach-v3"})
    runner = StrategyRunner(twin, strategy, goal=lambda: goal)

    runner.start(threaded=False)
    assert twin.owner == "strategia"
    drive(twin, runner, 6.0)

    assert not runner.status.running
    assert runner.status.done and runner.status.success
    assert runner.status.stopped_because == "zadanie wykonane"
    assert strategy.status.stopped_because == "koniec epizodu"
    assert strategy.status.step == strategy.task.episode_steps
    assert runner.status.error < 0.005                       # measured: 0.8 mm
    assert twin.owner is None
    tcp = strategy.kin.tcp(twin.status.measured)[:3, 3]
    assert np.linalg.norm(tcp - goal) < 0.005


def test_reach_without_a_goal_drives_to_the_settings_goal(twin):
    """The panel's Strategies tab passes no goal: the strategy's settings hold it."""
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({"goal_x": 0.2, "goal_y": -0.1, "goal_z": 0.08})
    runner = StrategyRunner(twin, strategy)

    runner.start(threaded=False)
    drive(twin, runner, 6.0)

    assert runner.status.success
    assert strategy.goal == pytest.approx([0.2, -0.1, 0.08])


def test_lift_v3_lifts_the_simulated_cube(twin):
    strategy = load_strategy(NAME).create({"policy": "lift-v3"})
    add_cube(twin, strategy.task)
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, strategy, objects=lambda: sim_cube(twin))

    runner.start(threaded=False)
    drive(twin, runner, 15.0)

    cube = sim_cube(twin)["cube"].position
    assert runner.status.stopped_because == "zadanie wykonane", (runner.status, strategy.status)
    assert cube[2] - strategy.task.cube_half > strategy.task.lift_height
    assert twin.owner is None


def test_lift_without_a_cube_waits_then_stops(twin):
    """The Strategies tab passes no objects either: lift waits for the cube, then gives up."""
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({"policy": "lift-v3", "wait_for_cube_s": 1.0})
    runner = StrategyRunner(twin, strategy)
    start = dict(twin.status.measured)

    runner.start(threaded=False)
    drive(twin, runner, 0.5)
    assert runner.status.running and runner.status.note == "czekam na kostke z kamer"
    drive(twin, runner, 2.0)

    assert not runner.status.running
    assert runner.status.stopped_because == "strategia skonczyla"
    assert "kamery jej nie widza" in strategy.status.stopped_because
    assert twin.status.measured == pytest.approx(start, abs=0.5)


def test_a_policy_for_another_robot_refuses_the_run(twin, monkeypatch):
    twin.connect("sim", threaded=False)
    strategy = load_strategy(NAME).create({})
    monkeypatch.setattr(strategy.policy.meta, "robot", "a1x")
    runner = StrategyRunner(twin, strategy)

    with pytest.raises(RuntimeError, match="uczona na a1x"):
        runner.start(threaded=False)

    assert twin.owner is None
    assert not runner.status.running
    assert runner.status.stopped_because.startswith("strategia odmowila")


def test_start_does_not_switch_off_the_clutch_of_the_next_owner(twin, monkeypatch):
    """STOP between the runner's `claim` and its clutch, then the panel takes the arm and engages it.

    The run must not disengage the panel's clutch.
    """
    twin.connect("sim", threaded=False)
    real_claim = twin.claim

    def claim_then_race(owner, preempt=None, **conditions):
        real_claim(owner, preempt, **conditions)
        twin.estop("STOP z panelu")
        twin.clear_estop()
        real_claim("panel")
        twin.set_engaged(True)

    monkeypatch.setattr(twin, "claim", claim_then_race)
    runner = StrategyRunner(twin, load_strategy(NAME).create({}))

    with pytest.raises(RuntimeError, match="odebrane"):
        runner.start(threaded=False)

    assert twin.owner == "panel"
    twin.step(TWIN_PERIOD)
    assert twin.status.engaged
