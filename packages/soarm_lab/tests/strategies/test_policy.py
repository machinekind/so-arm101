"""The policy strategy's own guards, the ones training does not have.

Claiming the arm, STOP and Home belong to the twin's `StrategyRunner`. The strategy's guards are tested
here without the twin, on the contracts' own `RunInfo`, `Observation` and `ObjectPose`
(`machinekind_twin_api`). test_policy_in_twin.py runs the same strategy through the twin's runner, including
the race between a claim and the clutch.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
from machinekind_twin_api.strategy import ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.objects import DESCRIPTOR_DIM, descriptor  # noqa: E402
from soarm_lab.rl.policy import Policy, PolicyMeta  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.policy import PolicySettings, PolicyStrategy, project_goal, project_target  # noqa: E402

SPEC = load_robot()


def constant_policy(task: tk.TaskConfig, action: dict[int, float] | None = None) -> Policy:
    """A policy that always gives the same action (zeros + `action`)."""
    pol = Policy(PolicyMeta(task=asdict(task), hidden=[8], obs_dim=task.obs_dim, act_dim=task.act_dim))
    for k, v in (action or {}).items():
        pol.actor[-1][1][k] = v                                  # bias of the output layer
    return pol


def run_info(robot: str = "so101", limits: dict | None = None, simulated: bool = False) -> RunInfo:
    return RunInfo(robot=robot, joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits=limits or {},
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=simulated)


def obs(joints: dict[str, float], t: float = 0.0, command: dict[str, float] | None = None,
        goal: np.ndarray | None = None, objects: dict[str, ObjectPose] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=joints, command=command or {}, goal=goal, objects=objects or {})


def cube(position, source: str = "kamery", age_s: float = 0.0) -> dict[str, ObjectPose]:
    return {"cube": ObjectPose(position=np.asarray(position, float), rotation=np.eye(3), source=source,
                               age_s=age_s)}


def strategy(task: str = "reach", action: dict[int, float] | None = None, **settings) -> PolicyStrategy:
    return PolicyStrategy(PolicySettings(**settings), policy=constant_policy(tk.make_task(task), action))


# ------------------------------------------------------------ lag = collision
def test_tracking_stop_holds_the_measured_pose_instead_of_pressing_on():
    """A blocked joint (a collision): the policy stops, and the arm holds the MEASURED pose.

    Holding the last command would keep pressing ~25 deg past the obstacle, so the strategy ends the run
    with a target equal to the measured pose.
    """
    s = strategy("reach", {0: 1.0}, episode_limit=False)
    blocked = dict(SPEC.home, shoulder_pan=10.0)
    s.start(run_info(), obs(blocked))
    answers = []
    for k in range(40):
        answers.append(s.act(obs(blocked, t=0.05 * k)))
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
    s.start(run_info(), obs(squeezing, command={"gripper": 5.0}))
    while not (last := s.act(obs(squeezing))).done:
        pass
    assert "nie nadaza" in last.note
    assert last.target == pytest.approx({k: v for k, v in squeezing.items() if k != "gripper"})


# ------------------------------------------------------------ a policy from another arm
def test_strategy_refuses_a_policy_trained_for_another_robot():
    """A policy from another arm (other joints, other gripper units) does not move the arm."""
    task = tk.make_task("reach")
    pol = constant_policy(task)
    pol.meta.robot = "a1x"
    with pytest.raises(RuntimeError, match="uczona na a1x"):
        PolicyStrategy(policy=pol).start(run_info(), obs(dict(SPEC.home)))
    pol = constant_policy(task)
    pol.meta.act_dim = 7
    with pytest.raises(RuntimeError, match="7 wyjsc"):
        PolicyStrategy(policy=pol).start(run_info(), obs(dict(SPEC.home)))
    with pytest.raises(RuntimeError, match="stanowisko ma a1x"):
        PolicyStrategy(policy=constant_policy(task)).start(run_info(robot="a1x"), obs(dict(SPEC.home)))
    PolicyStrategy(policy=constant_policy(task)).start(run_info(), obs(dict(SPEC.home)))   # SO-101 on SO-101


# ------------------------------------------------------------ start, goal, limits
def test_start_keeps_a_squeezing_gripper_command():
    """A gripper squeezing a cube (commanded tighter than measured): start begins from the command."""
    measured = dict(SPEC.home, gripper=30.0)
    s = strategy("lift")
    s.start(run_info(), obs(measured, command={"gripper": 5.0}))
    assert s.q_cmd[s.g] == pytest.approx(s.kin.to_q({"gripper": 5.0})[s.g])
    s.start(run_info(), obs(measured, command={"gripper": 29.0}))        # not squeezing: the measured angle
    assert s.q_cmd[s.g] == pytest.approx(s.kin.to_q(measured)[s.g])


def test_targets_stay_inside_the_hardware_limits():
    s = strategy("reach", {0: 1.0})
    s.start(run_info(limits={"shoulder_pan": (-20.0, 20.0)}), obs(dict(SPEC.home)))
    pans = []
    for k in range(20):
        a = s.act(obs(s.kin.from_q(s.q_cmd), t=0.05 * k))
        pans.append(a.target["shoulder_pan"])
    assert max(pans) == pytest.approx(20.0)


def test_a_goal_outside_the_training_area_is_projected():
    s = strategy("reach")
    s.start(run_info(), obs(dict(SPEC.home), goal=np.array([0.25, 0.0, -0.05])))
    assert s.goal[2] == pytest.approx(s.task.goal_height[0])
    assert s.status.goal_clamped
    a = s.act(obs(dict(SPEC.home), goal=np.array([0.2, 0.05, 0.1])))
    assert not s.status.goal_clamped
    assert a.error == pytest.approx(np.linalg.norm(np.array([0.2, 0.05, 0.1]) - s.kin.tcp(SPEC.home)[:3, 3]))
    # Without a goal in the observation: the settings' goal.
    s = strategy("reach", goal_x=0.3, goal_y=-0.1, goal_z=0.15)
    s.start(run_info(), obs(dict(SPEC.home)))
    assert s.goal == pytest.approx([0.3, -0.1, 0.15])


def test_project_goal_and_target_stay_in_the_training_rings():
    task = tk.make_task("pickplace")
    g = project_goal(tk.make_task("reach"), np.array([-0.1, 0.0, 0.5]))
    assert g[0] > 0.05 and g[2] == pytest.approx(0.22)
    t = project_target(task, np.array([0.1, 0.5, 0.3]))          # behind the ring's edge, too far
    assert t[2] == 0.0
    assert np.hypot(t[0], t[1]) == pytest.approx(task.target_radius[1])
    assert np.arctan2(t[1], t[0]) == pytest.approx(task.target_bearing[1])


# ------------------------------------------------------------ the cube
def test_the_arm_waits_for_the_first_cube_then_gives_up():
    s = strategy("lift", wait_for_cube_s=1.0)
    s.start(run_info(), obs(dict(SPEC.home)))
    a = s.act(obs(dict(SPEC.home), t=0.5))
    assert not a.done and a.target == {} and a.note == "czekam na kostke z kamer"
    a = s.act(obs(dict(SPEC.home), t=1.5))
    assert a.done and not a.success and "kamery jej nie widza" in a.note
    assert s.status.step == 0


def test_lift_success_needs_a_fresh_cube_and_a_holding_jaw():
    """A lifted cube counts only from a fresh pose and while the jaw holds (still, commanded tighter)."""
    s = strategy("lift", {5: -1.0})                     # the gripper closes, the arm stays
    held = dict(SPEC.home, gripper=25.0)                  # the jaw blocked on the cube
    s.start(run_info(), obs(held))
    lifted = [0.2, 0.0, s.task.cube_half + s.task.lift_height + 0.02]
    for k in range(10):
        s.act(obs(held, t=0.05 * k, objects=cube(lifted, "ostatnie widziane")))
    # A stale pose: judged by the jaw and the TCP's height above its top, 1.3 cm here, under lift_height.
    assert s._pose_live and not s._cube_fresh()
    assert not s.status.success
    streak = []
    for k in range(10, 30):
        a = s.act(obs(held, t=0.05 * k, objects=cube(lifted, "w dloni")))
        streak.append(s.status.success_streak)
        if a.done:
            break
    assert a.done and a.success and a.note == "zadanie wykonane"
    assert streak[-1] == s.end_on_success == 10
    # On a real arm a pose without a source is stale; in simulation it is the scene's truth.
    s.start(run_info(simulated=False), obs(held))
    for k in range(5):
        s.act(obs(held, t=0.05 * k, objects=cube(lifted, "")))
    assert not s.status.success
    s.start(run_info(simulated=True), obs(held))
    for k in range(5):
        s.act(obs(held, t=0.05 * k, objects=cube(lifted, "")))
    assert s.status.success


def lift_success(s: PolicyStrategy, objects: dict[str, ObjectPose], ticks: int = 5) -> bool:
    """Whether a lifted cube in a holding jaw counts as success after `ticks` ticks from a fresh start."""
    held = dict(SPEC.home, gripper=25.0)                  # the jaw blocked on the cube
    s.start(run_info(), obs(held))
    for k in range(ticks):
        s.act(obs(held, t=0.05 * k, objects=objects))
    return s.status.success


def test_lift_success_needs_a_young_cube_pose():
    """A pose older than `max_pose_age_s` never counts, whatever its source says."""
    s = strategy("lift", {5: -1.0})
    # The rejected old poses go to the verdict by forward kinematics, which is 1.3 cm at `lifted` from home.
    lifted = [0.2, 0.0, s.task.cube_half + s.task.lift_height + 0.02]
    assert s.config.max_pose_age_s == 1.0
    # The default keeps today's camera path: the twin says "kamery" only up to 1 s after capture.
    assert lift_success(s, cube(lifted, "kamery", age_s=0.66))
    assert lift_success(s, cube(lifted, "kamery", age_s=1.0))
    assert not lift_success(s, cube(lifted, "kamery", age_s=1.5))
    assert not lift_success(s, cube(lifted, "kamery", age_s=float("nan")))
    assert lift_success(s, cube(lifted, "w dloni", age_s=0.0))
    s = strategy("lift", {5: -1.0}, max_pose_age_s=0.2)
    assert not lift_success(s, cube(lifted, "kamery", age_s=0.3))
    assert not lift_success(s, cube(lifted, "w dloni", age_s=0.3))


def test_lift_success_from_last_seen_only_when_the_operator_allows_it():
    """A slow detector (a language model) publishes "ostatnie widziane"; it counts only when allowed."""
    s = strategy("lift", {5: -1.0})
    lifted = [0.2, 0.0, s.task.cube_half + s.task.lift_height + 0.02]
    assert not s.config.allow_last_seen
    assert not lift_success(s, cube(lifted, "ostatnie widziane", age_s=0.1))
    s = strategy("lift", {5: -1.0}, allow_last_seen=True, max_pose_age_s=15.0)
    assert lift_success(s, cube(lifted, "ostatnie widziane", age_s=12.0))
    assert not lift_success(s, cube(lifted, "ostatnie widziane", age_s=20.0))
    assert not lift_success(s, cube(lifted, "ostatnie widziane (upuszczona)", age_s=1.0))


def test_max_pose_age_must_not_be_negative():
    with pytest.raises(ValueError, match="max_pose_age_s must be zero or more"):
        strategy("lift", max_pose_age_s=-1.0)
    with pytest.raises(ValueError, match="max_pose_age_s must be zero or more"):
        strategy("lift", max_pose_age_s=float("nan"))


def test_the_run_ends_after_a_training_episode():
    s = strategy("reach")
    s.start(run_info(), obs(dict(SPEC.home)))
    answers = [s.act(obs(dict(SPEC.home), t=0.05 * k)) for k in range(s.task.episode_steps)]
    assert [a.done for a in answers].index(True) == s.task.episode_steps - 1
    assert answers[-1].note == "koniec epizodu"
    s.stop("strategia skonczyla")                        # the twin's reason after our `done`
    assert s.status.stopped_because == "koniec epizodu"
    s.stop("przerwana: STOP z panelu")
    assert s.status.stopped_because == "przerwana: STOP z panelu"


# ------------------------------------------------------------ wooden blocks
def block_policy(kind: str = "cuboid5", action: dict[int, float] | None = None) -> Policy:
    """A constant shape-conditioned lift policy whose own block is `kind`, trained with the yaw folded."""
    pol = constant_policy(replace(tk.make_task("lift"), object_kind=kind, shape_conditioned=True), action)
    pol.meta.randomization = {"fold_yaw": True}
    return pol


def block(label: str, position, yaw_deg: float = 0.0, rotation=None, source: str = "kamery"):
    """`obs.objects` with one object under `label`, turned by `yaw_deg` about z (or by `rotation`)."""
    c, s_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) if rotation is None else rotation
    return {label: ObjectPose(position=np.asarray(position, float), rotation=R, source=source)}


def test_a_shape_conditioned_policy_sees_its_block_appended():
    """The observation is the lift observation plus the descriptor of the policy's own block."""
    s = PolicyStrategy(policy=block_policy("cuboid5"))
    s.start(run_info(), obs(dict(SPEC.home)))
    vec = s.observation(dict(SPEC.home), block("cube", [0.2, 0.05, 0.0125]))
    assert vec.shape == (tk.OBS_DIMS["lift"] + DESCRIPTOR_DIM,) == (40,) and vec.dtype == np.float32
    np.testing.assert_array_equal(vec[33:], descriptor("cuboid5"))
    legacy = PolicyStrategy(policy=constant_policy(tk.make_task("lift")))
    legacy.start(run_info(), obs(dict(SPEC.home)))
    want = legacy.observation(dict(SPEC.home), block("cube", [0.2, 0.05, 0.0125]))
    np.testing.assert_allclose(vec[:33], want)                       # the first 33: the lift layout


def test_the_settings_object_kind_overrides_the_policys_own():
    """`object_kind` picks the block: its descriptor, its name in `obs.objects` and its yaw symmetry."""
    s = PolicyStrategy(PolicySettings(object_kind="cylinder"), policy=block_policy("cuboid5"))
    assert s.task.object_kind == "cylinder" and s.policy.task.object_kind == "cuboid5"
    s.start(run_info(), obs(dict(SPEC.home)))
    seen = {**block("cube", [0.25, 0.0, 0.0125]), **block("cylinder", [0.2, 0.05, 0.0125], yaw_deg=100.0)}
    vec = s.observation(dict(SPEC.home), seen)
    np.testing.assert_array_equal(vec[33:], descriptor("cylinder"))
    np.testing.assert_allclose(vec[15:18], [0.2, 0.05, 0.0125])                 # the object named "cylinder"
    # A cylinder looks the same every 180 deg: 100 deg folds to -80 (a cube would fold it to +10).
    np.testing.assert_allclose(vec[21:23], [np.cos(np.radians(-80.0)), np.sin(np.radians(-80.0))], atol=1e-6)
    # Without an object of the kind's name, the one named "cube" (the cube detector, the twin's simulation).
    vec = s.observation(dict(SPEC.home), block("cube", [0.25, 0.0, 0.0125]))
    np.testing.assert_allclose(vec[15:18], [0.25, 0.0, 0.0125])
    with pytest.raises(ValueError, match="unknown object_kind 'pyramid'"):
        PolicyStrategy(PolicySettings(object_kind="pyramid"), policy=block_policy())
    with pytest.raises(ValueError, match="needs a task with an object"):
        PolicyStrategy(PolicySettings(object_kind="cube"), policy=constant_policy(tk.make_task("reach")))


def test_lift_success_counts_the_blocks_lowest_point():
    """A cuboid85 stood on its end is not lifted although its centre is high: its lowest point counts."""
    s = PolicyStrategy(PolicySettings(object_kind="cuboid85"), policy=block_policy("cube", {5: -1.0}))
    centre = [0.2, 0.0, 0.08]                    # lying flat its bottom is 6.75 cm up, stood up 3.75 cm
    standing = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # body x (the long side) up
    assert lift_success(s, block("cuboid85", centre))
    assert not lift_success(s, block("cuboid85", centre, rotation=standing))
    assert "3.8 cm nad blatem" in s._note()


# ------------------------------------------------------------ the latch
def latched_policy(action: dict[int, float] | None = None) -> Policy:
    """A constant lift policy trained with the occlusion latch (`latch_m` 5 cm in its randomization)."""
    pol = constant_policy(tk.make_task("lift"), action)
    pol.meta.randomization = {"fold_yaw": True, "latch_m": 0.05}
    return pol


def test_a_lost_pose_is_kept_for_latch_s_then_the_run_ends():
    s = strategy("lift", latch_s=1.0)
    s.start(run_info(), obs(dict(SPEC.home)))
    a = s.act(obs(dict(SPEC.home), t=0.0, objects=cube([0.2, 0.0, 0.015])))
    assert not a.done and s._pose_live
    a = s.act(obs(dict(SPEC.home), t=0.5))                              # gone: kept
    assert not a.done and not s._pose_live
    assert s._cube is not None and s._cube.tolist() == [0.2, 0.0, 0.015]
    assert "zatrzask" in a.note
    a = s.act(obs(dict(SPEC.home), t=1.6))                              # 1.6 s after the last sighting
    assert a.done and not a.success and "kamery jej nie widza" in a.note
    # The default (0): the first tick without the object ends the run, as before.
    s = strategy("lift")
    s.start(run_info(), obs(dict(SPEC.home)))
    assert not s.act(obs(dict(SPEC.home), t=0.0, objects=cube([0.2, 0.0, 0.015]))).done
    assert s.act(obs(dict(SPEC.home), t=0.05)).done


def test_the_latch_clock_runs_from_the_last_young_sighting():
    s = strategy("lift", latch_s=1.0, max_pose_age_s=1.0)
    s.start(run_info(), obs(dict(SPEC.home)))
    s.act(obs(dict(SPEC.home), t=2.0, objects=cube([0.2, 0.0, 0.015], "ostatnie widziane", age_s=0.5)))
    s.act(obs(dict(SPEC.home), t=2.5, objects=cube([0.2, 0.0, 0.015], "ostatnie widziane", age_s=3.0)))
    assert s._pose_live                                                 # taken, but too old to count
    assert not s.act(obs(dict(SPEC.home), t=2.9)).done                  # 0.9 s after the young one
    assert s.act(obs(dict(SPEC.home), t=3.1)).done                      # 1.1 s
    # A dropped object is never seen well: nothing to keep.
    s.start(run_info(), obs(dict(SPEC.home)))
    s.act(obs(dict(SPEC.home), t=0.0, objects=cube([0.2, 0.0, 0.015], "ostatnie widziane (upuszczona)")))
    assert s.act(obs(dict(SPEC.home), t=0.1)).done


def test_lift_success_from_a_stale_pose_needs_the_jaw_and_the_tcp_height():
    """Not live and fresh: the jaw holds and the TCP is more than `lift_height` above the block's top."""
    s = strategy("lift", {5: -1.0})
    stale = "ostatnie widziane"
    # The legacy cube on the table: its top 3 cm up, the TCP at home 12.3 cm, 9.3 cm above it.
    assert lift_success(s, cube([0.2, 0.0, 0.015], stale, age_s=3.0))
    assert "nad klockiem" in s._note() and "nieswieza" in s._note()
    # The pose says 8 cm: its top 9.5 cm up, the TCP only 2.8 cm above it.
    assert not lift_success(s, cube([0.2, 0.0, 0.08], stale, age_s=3.0))
    # The jaw commanded open, wider than measured: not holding.
    s = strategy("lift", {5: 1.0})
    s.start(run_info(), obs(dict(SPEC.home)))
    for k in range(5):
        s.act(obs(dict(SPEC.home), t=0.05 * k, objects=cube([0.2, 0.0, 0.015], stale, age_s=3.0)))
    assert not s.status.success

    # The block's top from its rotation: a cuboid85 stood on its end is 8.5 cm tall.
    s = PolicyStrategy(PolicySettings(object_kind="cuboid85"), policy=block_policy("cube", {5: -1.0}))
    standing = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # body x (the long side) up
    assert not lift_success(s, block("cuboid85", [0.2, 0.0, 0.0425], rotation=standing, source=stale))
    assert lift_success(s, block("cuboid85", [0.2, 0.0, 0.0125], source=stale))   # lying: top 2.5 cm


def test_a_latched_policy_keeps_the_pose_under_the_hand():
    """Trained with `latch_m`: the pose freezes near the TCP, while the jaw holds and for "w dloni"."""
    a_pos, b_pos = [0.2, 0.0, 0.015], [0.25, 0.0, 0.015]
    near = dict(RobotKinematics(SPEC).ik([0.2, 0.0, 0.05]).joints, gripper=25.0)   # the TCP 3.5 cm above A
    held = dict(SPEC.home, gripper=25.0)                                     # lifted: the TCP 16.5 cm from A
    # The lag guard stays out of the way: the test jumps between poses 53 deg apart.
    settings = PolicySettings(max_tracking_deg=90.0)
    s = PolicyStrategy(settings, policy=latched_policy({5: -1.0}))           # the jaw closes on the block
    s.start(run_info(), obs(near))
    s.act(obs(near, t=0.0, objects=cube(a_pos)))
    assert s._cube is not None and s._cube.tolist() == a_pos and s._pose_live   # the first sighting
    s.act(obs(near, t=0.05, objects=cube(b_pos)))
    assert s._cube.tolist() == a_pos and not s._pose_live                     # near: frozen
    s.act(obs(near, t=0.1, objects=cube([0.3, 0.1, 0.2], "w dloni")))
    assert s._cube.tolist() == a_pos
    # The carry: far from A, a camera sighting at B; the jaw is still and squeezes, so it holds.
    for k in range(3):
        s.act(obs(held, t=0.15 + 0.05 * k, objects=cube(b_pos)))
        assert s._cube.tolist() == a_pos and not s._pose_live, k

    # The jaw commanded open, wider than measured, so it does not hold: far from A, B is taken.
    s2 = PolicyStrategy(settings, policy=latched_policy({5: 1.0}))
    s2.start(run_info(), obs(dict(near, gripper=35.0)))
    s2.act(obs(dict(near, gripper=35.0), t=0.0, objects=cube(a_pos)))
    s2.act(obs(dict(SPEC.home), t=0.05, objects=cube(b_pos)))
    assert s2._cube is not None and s2._cube.tolist() == b_pos and s2._pose_live

    # Without `latch_m` (lift-v3 and every policy before) B is taken at once, even from near.
    s3 = PolicyStrategy(settings, policy=constant_policy(tk.make_task("lift")))
    s3.start(run_info(), obs(near))
    s3.act(obs(near, t=0.0, objects=cube(a_pos)))
    s3.act(obs(near, t=0.05, objects=cube(b_pos)))
    assert s3._cube is not None and s3._cube.tolist() == b_pos and s3._pose_live


def test_latch_s_must_not_be_negative():
    with pytest.raises(ValueError, match="latch_s must be zero or more"):
        strategy("lift", latch_s=-1.0)
    with pytest.raises(ValueError, match="latch_s must be zero or more"):
        strategy("lift", latch_s=float("nan"))
