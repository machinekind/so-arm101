"""The policy strategy's own guards, the ones training does not have.

Claiming the arm, STOP and Home belong to the twin's `StrategyRunner`. The strategy's guards are tested
here without the twin: `RunInfo` and `Observation` are plain objects with the attributes the twin gives
them. `twin/test_strategy.py` runs the same strategy through the twin's runner, including the race
between a claim and the clutch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

import soarm

pytest.importorskip("mujoco")

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.policy import Policy, PolicyMeta  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.policy import PolicySettings, PolicyStrategy, project_goal, project_target  # noqa: E402

SPEC = load_robot()


def constant_policy(task: tk.TaskConfig, action: dict[int, float] | None = None) -> Policy:
    """Polityka, ktora zawsze daje te sama akcje (zera + `action`)."""
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=task.act_dim))
    for k, v in (action or {}).items():
        pol.actor[-1][1][k] = v                                  # bias of the output layer
    return pol


def run_info(robot: str = "so101", limits: dict | None = None, simulated: bool = False) -> SimpleNamespace:
    """The twin's `RunInfo` by its attributes."""
    return SimpleNamespace(robot=robot, joints=SPEC.joints, gripper=SPEC.gripper, limits=limits or {},
                           max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=simulated)


@dataclass
class Obs:
    """The twin's `Observation` by its attributes."""

    joints: dict[str, float]
    t: float = 0.0
    step: int = 0
    command: dict[str, float] = field(default_factory=dict)
    goal: np.ndarray | None = None
    objects: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)


def cube(position, source: str = "kamery") -> dict:
    return {"cube": SimpleNamespace(position=np.asarray(position, float), rotation=np.eye(3), source=source)}


def strategy(task: str = "reach", action: dict[int, float] | None = None, **settings) -> PolicyStrategy:
    return PolicyStrategy(PolicySettings(**settings), policy=constant_policy(tk.make_task(task), action))


# ------------------------------------------------------------ rozjazd = kolizja
def test_tracking_stop_holds_the_measured_pose_instead_of_pressing_on():
    """Staw zablokowany (kolizja): polityka staje, a ramie trzyma ZMIERZONA poze.

    W blizniaku `_halt` robil tylko `set_engaged(False)`, a nadzor trzymal ostatni rozkaz -
    ~25 st. za przeszkoda. Strategia konczy przebieg celem rownym pomiarowi.
    """
    s = strategy("reach", {0: 1.0}, episode_limit=False)
    blocked = dict(SPEC.home, shoulder_pan=10.0)
    s.start(run_info(), Obs(blocked))
    answers = []
    for k in range(40):
        answers.append(s.act(Obs(blocked, t=0.05 * k)))
        if answers[-1].done:
            break
    last = answers[-1]
    assert last.done and not last.success
    assert "nie nadaza" in last.note and "nie nadaza" in s.status.stopped_because
    assert last.target == pytest.approx(blocked)
    # Before that the policy kept pushing shoulder_pan further (0.05 rad per tick).
    pans = [a.target["shoulder_pan"] for a in answers[:-1]]
    assert pans == sorted(pans) and pans[-1] - 10.0 == pytest.approx(25.0, abs=np.degrees(0.05) + 1e-6)

    # A gripper squeezing a cube keeps its command (left out of the target), as in the twin's `hold_measured`.
    s = strategy("reach", {0: 1.0, 5: -1.0}, episode_limit=False)          # the gripper keeps closing
    squeezing = dict(blocked, gripper=30.0)
    s.start(run_info(), Obs(squeezing, command={"gripper": 5.0}))
    while not (last := s.act(Obs(squeezing))).done:
        pass
    assert "nie nadaza" in last.note
    assert last.target == pytest.approx({k: v for k, v in squeezing.items() if k != "gripper"})


# ------------------------------------------------------------ polityka z innego ramienia
def test_strategy_refuses_a_policy_trained_for_another_robot():
    """Polityka z innego ramienia (inne stawy, inne jednostki chwytaka) nie rusza ramienia."""
    task = tk.make_task("reach")
    pol = constant_policy(task)
    pol.meta.robot = "a1x"
    with pytest.raises(RuntimeError, match="uczona na a1x"):
        PolicyStrategy(policy=pol).start(run_info(), Obs(dict(SPEC.home)))
    pol = constant_policy(task)
    pol.meta.act_dim = 7
    with pytest.raises(RuntimeError, match="7 wyjsc"):
        PolicyStrategy(policy=pol).start(run_info(), Obs(dict(SPEC.home)))
    with pytest.raises(RuntimeError, match="stanowisko ma a1x"):
        PolicyStrategy(policy=constant_policy(task)).start(run_info(robot="a1x"), Obs(dict(SPEC.home)))
    PolicyStrategy(policy=constant_policy(task)).start(run_info(), Obs(dict(SPEC.home)))   # SO-101 na SO-101


# ------------------------------------------------------------ start, cel, limity
def test_start_keeps_a_squeezing_gripper_command():
    """Chwytak sciska kostke (rozkaz ciasniej niz pomiar): start zaczyna od rozkazu, nie od pomiaru."""
    measured = dict(SPEC.home, gripper=30.0)
    s = strategy("lift")
    s.start(run_info(), Obs(measured, command={"gripper": 5.0}))
    assert s.q_cmd[s.g] == pytest.approx(s.kin.to_q({"gripper": 5.0})[s.g])
    s.start(run_info(), Obs(measured, command={"gripper": 29.0}))        # nie sciska - od pomiaru
    assert s.q_cmd[s.g] == pytest.approx(s.kin.to_q(measured)[s.g])


def test_targets_stay_inside_the_hardware_limits():
    s = strategy("reach", {0: 1.0})
    s.start(run_info(limits={"shoulder_pan": (-20.0, 20.0)}), Obs(dict(SPEC.home)))
    pans = []
    for k in range(20):
        a = s.act(Obs(s.kin.from_q(s.q_cmd), t=0.05 * k))
        pans.append(a.target["shoulder_pan"])
    assert max(pans) == pytest.approx(20.0)


def test_a_goal_outside_the_training_area_is_projected():
    s = strategy("reach")
    s.start(run_info(), Obs(dict(SPEC.home), goal=np.array([0.25, 0.0, -0.05])))
    assert s.goal[2] == pytest.approx(s.task.goal_height[0])
    assert s.status.goal_clamped
    a = s.act(Obs(dict(SPEC.home), goal=np.array([0.2, 0.05, 0.1])))
    assert not s.status.goal_clamped
    assert a.error == pytest.approx(np.linalg.norm(np.array([0.2, 0.05, 0.1]) - s.kin.tcp(SPEC.home)[:3, 3]))
    # Without a goal in the observation: the settings' goal.
    s = strategy("reach", goal_x=0.3, goal_y=-0.1, goal_z=0.15)
    s.start(run_info(), Obs(dict(SPEC.home)))
    assert s.goal == pytest.approx([0.3, -0.1, 0.15])


def test_project_goal_and_target_stay_in_the_training_rings():
    task = tk.make_task("pickplace")
    g = project_goal(tk.make_task("reach"), np.array([-0.1, 0.0, 0.5]))
    assert g[0] > 0.05 and g[2] == pytest.approx(0.22)
    t = project_target(task, np.array([0.1, 0.5, 0.3]))          # behind the ring's edge, too far
    assert t[2] == 0.0
    assert np.hypot(t[0], t[1]) == pytest.approx(task.target_radius[1])
    assert np.arctan2(t[1], t[0]) == pytest.approx(task.target_bearing[1])


# ------------------------------------------------------------ kostka
def test_the_arm_waits_for_the_first_cube_then_gives_up():
    s = strategy("lift", wait_for_cube_s=1.0)
    s.start(run_info(), Obs(dict(SPEC.home)))
    a = s.act(Obs(dict(SPEC.home), t=0.5))
    assert not a.done and a.target == {} and a.note == "czekam na kostke z kamer"
    a = s.act(Obs(dict(SPEC.home), t=1.5))
    assert a.done and not a.success and "kamery jej nie widza" in a.note
    assert s.status.step == 0


def test_lift_success_needs_a_fresh_cube_and_a_holding_jaw():
    """Kostka nad blatem liczy sie tylko ze swiezej pozy i gdy szczeka trzyma (stoi, ciasniej rozkazana)."""
    s = strategy("lift", {5: -1.0})                     # chwytak zamyka sie, ramie stoi
    held = dict(SPEC.home, gripper=25.0)                  # szczeka zablokowana na kostce
    s.start(run_info(), Obs(held))
    lifted = [0.2, 0.0, s.task.cube_half + s.task.lift_height + 0.02]
    for k in range(10):
        s.act(Obs(held, t=0.05 * k, objects=cube(lifted, "ostatnie widziane")))
    assert not s.status.success                          # poza nieswieza
    streak = []
    for k in range(10, 30):
        a = s.act(Obs(held, t=0.05 * k, objects=cube(lifted, "w dloni")))
        streak.append(s.status.success_streak)
        if a.done:
            break
    assert a.done and a.success and a.note == "zadanie wykonane"
    assert streak[-1] == s.end_on_success == 10
    # On a real arm a pose without a source is stale; in simulation it is the scene's truth.
    s.start(run_info(simulated=False), Obs(held))
    for k in range(5):
        s.act(Obs(held, t=0.05 * k, objects=cube(lifted, "")))
    assert not s.status.success
    s.start(run_info(simulated=True), Obs(held))
    for k in range(5):
        s.act(Obs(held, t=0.05 * k, objects=cube(lifted, "")))
    assert s.status.success


def test_the_run_ends_after_a_training_episode():
    s = strategy("reach")
    s.start(run_info(), Obs(dict(SPEC.home)))
    answers = [s.act(Obs(dict(SPEC.home), t=0.05 * k)) for k in range(s.task.episode_steps)]
    assert [a.done for a in answers].index(True) == s.task.episode_steps - 1
    assert answers[-1].note == "koniec epizodu"
    s.stop("strategia skonczyla")                        # the twin's reason after our `done`
    assert s.status.stopped_because == "koniec epizodu"
    s.stop("przerwana: STOP z panelu")
    assert s.status.stopped_because == "przerwana: STOP z panelu"
