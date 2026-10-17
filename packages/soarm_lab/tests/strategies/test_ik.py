"""The IK strategy without the twin: the path, its speed, the reach check, the lag guard, the gripper.

The tests build the contracts' own `RunInfo` and `Observation` (`machinekind_twin_api`) and play the arm
themselves: in most of them the measured joints are the previous tick's target (perfect tracking).
test_ik_in_twin.py runs the same strategy through the twin's runner and physics.
"""

from __future__ import annotations

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.ik import IKSettings, IKStrategy  # noqa: E402

SPEC = load_robot()
GOAL = np.array([0.25, 0.08, 0.10])
#: How far a target's TCP may lie off the carrot: `RobotKinematics.ik_step`'s tolerance [m].
IK_TOL = 1e-3


def run_info(limits: dict[str, tuple[float, float]] | None = None) -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper,
                   limits=dict(SPEC.limits) if limits is None else limits, max_vel=dict(SPEC.max_vel),
                   folder=soarm.ROBOT_DIR, simulated=True)


def obs(joints: dict[str, float], t: float = 0.0, goal: np.ndarray | None = None,
        command: dict[str, float] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=joints, command=command or {}, goal=goal)


def started(goal: np.ndarray | None = GOAL, joints: dict[str, float] | None = None,
            limits: dict[str, tuple[float, float]] | None = None, **settings) -> IKStrategy:
    s = IKStrategy(IKSettings(**settings))
    s.start(run_info(limits), obs(dict(joints or SPEC.home), goal=goal))
    return s


def follow(s: IKStrategy, joints: dict[str, float], goal: np.ndarray | None, ticks: int,
           ) -> tuple[list[Action], dict[str, float]]:
    """Up to `ticks` ticks with perfect tracking: the measured joints are the previous tick's target."""
    answers = []
    for k in range(ticks):
        answers.append(a := s.act(obs(joints, t=k / s.hz, goal=goal)))
        joints = dict(a.target)
        if a.done:
            break
    return answers, joints


def tcp(s: IKStrategy, joints: dict[str, float]) -> np.ndarray:
    return s.kin.tcp(joints)[:3, 3]


# ------------------------------------------------------------ the path
def test_the_tcp_reaches_the_goal_along_a_straight_line():
    """The run ends once the target puts the TCP on the goal and the measured TCP is within `tol_m`."""
    s = started()
    start = tcp(s, SPEC.home)
    answers, joints = follow(s, dict(SPEC.home), GOAL, 200)

    last = answers[-1]
    assert last.done and last.success and last.note == "cel osiagniety"
    assert s.status.stopped_because == "cel osiagniety" and s.status.step == len(answers)
    assert last.error == pytest.approx(s.status.distance) and last.error <= s.config.tol_m
    assert np.linalg.norm(tcp(s, joints) - GOAL) <= s.ik_tol           # the last target: on the goal
    assert not any(a.done for a in answers[:-1])
    # 0.16 m at 0.10 m/s and 20 Hz: about 32 ticks, a few more to settle.
    assert len(answers) <= np.linalg.norm(GOAL - start) / s.step_m + 3
    u = (GOAL - start) / np.linalg.norm(GOAL - start)
    for a in answers:
        assert set(a.target) == set(SPEC.joints)
        assert a.target["gripper"] == pytest.approx(SPEC.home["gripper"])
        assert all(SPEC.limits[n][0] - 1e-6 <= v <= SPEC.limits[n][1] + 1e-6 for n, v in a.target.items())
        off = tcp(s, a.target) - start
        assert np.linalg.norm(off - (off @ u) * u) < 2 * IK_TOL          # on the line to the goal


@pytest.mark.parametrize("speed_mps, hz", [(0.10, 20.0), (0.05, 10.0), (0.20, 25.0), (0.005, 20.0)])
def test_the_tcp_never_moves_faster_than_speed_mps(speed_mps: float, hz: float):
    """Between consecutive targets the TCP moves at most `speed_mps / hz`, also a step under IK's 1 mm."""
    goal = np.array([0.15, 0.15, 0.05])
    s = started(goal, speed_mps=speed_mps, hz=hz)
    assert s.hz == hz and s.step_m == pytest.approx(speed_mps / hz)
    start = tcp(s, SPEC.home)
    answers, _ = follow(s, dict(SPEC.home), goal, int(np.linalg.norm(goal - start) / s.step_m) + 20)
    assert answers[-1].success
    points = [start] + [tcp(s, a.target) for a in answers]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.max() <= s.step_m + s.ik_tol
    assert np.median(steps) >= 0.9 * s.step_m                           # and it does go at that speed


def test_without_a_goal_in_the_observation_the_settings_hold_it():
    s = started(None, goal_x=0.2, goal_y=-0.1, goal_z=0.08)
    assert s.goal == pytest.approx([0.2, -0.1, 0.08])
    answers, joints = follow(s, dict(SPEC.home), None, 200)
    assert answers[-1].success
    assert np.linalg.norm(tcp(s, joints) - [0.2, -0.1, 0.08]) <= s.config.tol_m


# ------------------------------------------------------------ reach
def test_a_goal_out_of_reach_refuses_the_run():
    with pytest.raises(RuntimeError, match=r"cel \(0\.600, 0\.000, 0\.100\) m poza zasiegiem ramienia"):
        started(np.array([0.6, 0.0, 0.1]))
    # Within the MJCF ranges, but not within the limits on this arm: shoulder_pan would need about -53 deg.
    narrow = dict(SPEC.limits, shoulder_pan=(-20.0, 20.0))
    with pytest.raises(RuntimeError, match="poza zasiegiem"):
        started(np.array([0.15, 0.15, 0.05]), limits=narrow)
    started(np.array([0.15, 0.15, 0.05]))


def test_a_goal_dragged_past_the_limits_stops_the_arm_at_the_limit():
    """`end_on_reach` off, the panel drags the goal where shoulder_pan cannot follow: the targets stop at
    the limit, and the note says the arm cannot reach the goal."""
    narrow = dict(SPEC.limits, shoulder_pan=(-20.0, 20.0))
    s = started(GOAL, limits=narrow, end_on_reach=False)
    _, joints = follow(s, dict(SPEC.home), GOAL, 60)
    answers, joints = follow(s, joints, np.array([0.15, 0.15, 0.05]), 100)
    assert not any(a.done for a in answers)
    assert joints["shoulder_pan"] == pytest.approx(-20.0)
    assert all(a.target["shoulder_pan"] >= -20.0 - 1e-6 for a in answers)
    assert s.status.out_of_reach and answers[-1].note.endswith("cel poza zasiegiem ramienia")
    # The goal comes back within reach: the arm follows it again.
    answers, joints = follow(s, joints, GOAL, 60)
    assert answers[-1].success and not s.status.out_of_reach
    assert np.linalg.norm(tcp(s, joints) - GOAL) <= s.config.tol_m


# ------------------------------------------------------------ a goal the panel drags
def test_without_end_on_reach_the_run_goes_on_and_follows_a_moved_goal():
    s = started(end_on_reach=False)
    answers, joints = follow(s, dict(SPEC.home), GOAL, 60)
    assert not any(a.done for a in answers)
    assert answers[-1].success and answers[-1].note.endswith("(w celu)")
    moved = np.array([0.2, -0.1, 0.15])
    answers, joints = follow(s, joints, moved, 80)
    assert not any(a.done for a in answers)
    assert s.goal == pytest.approx(moved)
    assert answers[-1].success and np.linalg.norm(tcp(s, joints) - moved) <= s.config.tol_m
    assert all(a.target["gripper"] == pytest.approx(SPEC.home["gripper"]) for a in answers)


# ------------------------------------------------------------ lag = collision
def test_the_lag_guard_ends_the_run_with_the_measured_pose():
    """The measured pose stops following (a blocked arm): the run ends, no success, on the measured pose."""
    blocked = dict(SPEC.home, gripper=40.0)
    s = started(np.array([0.15, 0.15, 0.05]), joints=blocked)
    answers = []
    for k in range(200):
        answers.append(a := s.act(obs(blocked, t=k / s.hz, goal=np.array([0.15, 0.15, 0.05]))))
        if a.done:
            break
    last = answers[-1]
    assert last.done and not last.success
    assert "nie nadaza" in last.note and s.status.stopped_because == last.note
    assert last.target == pytest.approx(blocked)
    assert s.status.step == len(answers) - 1                             # the stopping tick does not count

    def lag(a: Action) -> float:
        return max(abs(a.target[n] - blocked[n]) for n in SPEC.arm_joints)

    # Before that the targets kept moving away from the measured pose, until one joint lagged 25 deg.
    assert lag(answers[-3]) <= s.config.max_tracking_deg < lag(answers[-2])


# ------------------------------------------------------------ the gripper
def test_the_gripper_keeps_its_measured_angle_or_the_command_it_squeezes_with():
    measured = dict(SPEC.home, gripper=30.0)
    s = IKStrategy()
    s.start(run_info(), obs(measured, goal=GOAL, command={"gripper": 29.0}))      # not squeezing
    answers, _ = follow(s, measured, GOAL, 200)
    assert answers[-1].success
    assert all(a.target["gripper"] == pytest.approx(30.0) for a in answers)
    s.start(run_info(), obs(measured, goal=GOAL, command={"gripper": 5.0}))       # squeezing a cube
    answers, _ = follow(s, measured, GOAL, 200)
    assert answers[-1].success
    assert all(a.target["gripper"] == pytest.approx(5.0) for a in answers)


# ------------------------------------------------------------ settings and stop
@pytest.mark.parametrize("name", ["hz", "speed_mps", "tol_m", "max_tracking_deg"])
def test_settings_must_be_positive(name: str):
    with pytest.raises(ValueError, match=f"ik: {name} must be positive"):
        IKStrategy(IKSettings(**{name: 0.0}))


def test_act_before_start_and_the_stop_reason():
    s = IKStrategy()
    assert not s.started
    with pytest.raises(RuntimeError, match="przed start"):
        s.act(obs(dict(SPEC.home)))
    s = started()
    follow(s, dict(SPEC.home), GOAL, 200)
    s.stop("zadanie wykonane")                           # the twin's reason after our `done`
    assert s.status.stopped_because == "cel osiagniety"
    s.stop("przerwana: STOP z panelu")
    assert s.status.stopped_because == "przerwana: STOP z panelu"
