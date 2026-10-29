"""The touch strategy without the twin: the wait, the pose check, the path, contact, the jaw and the seam.

The tests build the contracts' own `RunInfo`, `Observation` and `ObjectPose` (`machinekind_twin_api`) and
play the twin's supervisor and the arm themselves (`follow`): the supervisor's command goes to each target,
at once or under a speed cap, and the measured joints follow the command, at once or with a servo's lag,
and with a constant height offset at the TCP when the arm sags or is held up. Where the arm would put the
TCP into an obstacle it stays where it was. That stall is what the strategy reads as contact.
test_touch_in_twin.py runs the same strategy through the twin's runner and physics, with a cube.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.touch import TouchSettings, TouchStrategy  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Where the object lies in most tests [m, base frame].
XY = (0.22, 0.05)
NAN = float("nan")


def run_info(limits: dict[str, tuple[float, float]] | None = None, simulated: bool = True,
             max_vel: dict[str, float] | None = None) -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper,
                   limits=dict(SPEC.limits) if limits is None else limits,
                   max_vel=dict(SPEC.max_vel) if max_vel is None else max_vel,
                   folder=soarm.ROBOT_DIR, simulated=simulated)


def obs(joints: dict[str, float], t: float = 0.0, objects: dict[str, ObjectPose] | None = None,
        command: dict[str, float] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=joints, command=command or {}, objects=objects or {})


def seen(xy: tuple[float, float] = XY, source: str = "ostatnie widziane", age_s: float = 8.0, z: float = 0.0,
         label: str = "cube") -> dict[str, ObjectPose]:
    """A pose as the LLM detector gives one: at the table top, seconds old, no orientation."""
    return {label: ObjectPose(position=np.array([*xy, z], float), rotation=np.eye(3), source=source,
                              age_s=age_s)}


def started(objects: dict[str, ObjectPose] | None = None, joints: dict[str, float] | None = None,
            simulated: bool = True, command: dict[str, float] | None = None,
            strategy: type[TouchStrategy] = TouchStrategy, max_vel: dict[str, float] | None = None,
            **settings) -> TouchStrategy:
    s = strategy(TouchSettings(**settings))
    s.start(run_info(simulated=simulated, max_vel=max_vel),
            obs(dict(joints or SPEC.home), objects=objects, command=command))
    return s


def pose_at(point: tuple[float, float, float], gripper: float = 0.0) -> dict[str, float]:
    """The joints that put the TCP on `point` [m, base frame]."""
    solution = KIN.ik(np.array(point, float), seed=SPEC.home)
    assert solution.ok
    return dict(solution.joints, gripper=gripper)


def capped(cap_deg_s: float) -> dict[str, float]:
    """The supervisor's speed limits under the panel's speed cap: the arm joints at most `cap_deg_s`."""
    return {name: value if name == SPEC.gripper else min(value, cap_deg_s)
            for name, value in SPEC.max_vel.items()}


def cube_at(xy: tuple[float, float], top: float, half: float = 0.015) -> Callable[[np.ndarray], bool]:
    """An obstacle: a TCP inside this box cannot be reached."""
    return lambda p: bool(np.all(np.abs(p[:2] - xy) < half) and p[2] < top)


def follow(s: TouchStrategy, joints: dict[str, float], objects: dict[str, ObjectPose] | None, ticks: int,
           t0: float = 0.0, obstacle: Callable[[np.ndarray], bool] | None = None, offset_z: float = 0.0,
           cap_deg_s: float | None = None, lag_s: float = 0.0, commands: list[dict[str, float]] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor and the arm, ending on `done`.

    The supervisor's command goes to each target, every arm joint at most `cap_deg_s` per second (None: at
    once). The arm follows the command, as a first-order servo with time constant `lag_s` (0: at once), with
    its TCP `offset_z` higher. A pose whose TCP lies in `obstacle` is not reached: the arm stays where it was.
    `commands` collects the command each tick observed.

    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    command = dict(joints)
    share = 1.0 if lag_s <= 0 else 1.0 - np.exp(-1.0 / (s.hz * lag_s))
    t = t0
    for _ in range(ticks):
        if commands is not None:
            commands.append(command)
        answers.append(a := s.act(obs(joints, t=t, objects=objects, command=command)))
        phases.append(s.status.phase)
        t += 1 / s.hz
        command = supervised(command, a.target, None if cap_deg_s is None else cap_deg_s / s.hz)
        arm = servo(joints, held(s, command, offset_z), share)
        if obstacle is None or not obstacle(tcp(s, arm)):
            joints = arm
        if a.done:
            break
    return answers, phases, joints, t


def supervised(command: dict[str, float], target: Mapping[str, float],
               step_deg: float | None) -> dict[str, float]:
    """The supervisor's next command: toward the target, every arm joint by at most `step_deg`."""
    out = dict(command)
    for name, value in target.items():
        if step_deg is None or name == SPEC.gripper:
            out[name] = float(value)
        else:
            out[name] = command[name] + float(np.clip(value - command[name], -step_deg, step_deg))
    return out


def held(s: TouchStrategy, command: dict[str, float], offset_z: float) -> dict[str, float]:
    """Where the arm settles under `command`: its TCP `offset_z` higher (sag below zero)."""
    if offset_z == 0.0:
        return dict(command)
    point = tcp(s, command) + np.array([0.0, 0.0, offset_z])
    return dict(s.kin.ik_step(point, seed=command, tol=1e-5).joints, gripper=command[SPEC.gripper])


def servo(measured: dict[str, float], settled: dict[str, float], share: float) -> dict[str, float]:
    """The arm after one tick: every joint goes `share` of the way to where it settles."""
    return {name: measured[name] + share * (value - measured[name]) for name, value in settled.items()}


def run_frozen(s: TouchStrategy, frozen: dict[str, float], objects: dict[str, ObjectPose], ticks: int,
               with_command: bool = True) -> tuple[list[Action], list[str], str]:
    """Ticks with an arm that never moves while the supervisor follows the targets.

    Without `with_command` the observations carry no supervisor's command. Returns the answers, the phases
    and the lag guard's exception text, empty when it did not stop the run.
    """
    answers, phases = [], []
    command = dict(frozen)
    try:
        for k in range(ticks):
            answers.append(a := s.act(obs(frozen, t=k / s.hz, objects=objects,
                                          command=command if with_command else None)))
            phases.append(s.status.phase)
            command = supervised(command, a.target, None)
            if a.done:
                break
    except RuntimeError as exc:
        return answers, phases, str(exc)
    return answers, phases, ""


def tcp(s: TouchStrategy, joints) -> np.ndarray:
    return s.kin.tcp(joints)[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """The phases in order, repeats dropped."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


# ------------------------------------------------------------ waiting for a pose
def test_without_a_pose_the_arm_holds_its_start_pose_with_the_jaw_closed():
    s = started()
    a = s.act(obs(dict(SPEC.home), t=0.05))

    assert set(a.target) == set(SPEC.joints) and not a.done
    assert s.status.phase == "wait"
    assert a.note.startswith("czekam na polozenie obiektu 'cube'") and "brak w obserwacji" in a.note
    arm = SPEC.arm_joints
    assert {n: a.target[n] for n in arm} == pytest.approx({n: SPEC.home[n] for n in arm})
    assert a.target["gripper"] == pytest.approx(SPEC.limits["gripper"][0])
    assert s.status.stopped_because == ""


def test_without_a_pose_the_run_ends_after_the_wait():
    s = started(wait_for_pose_s=1.0)

    assert not s.act(obs(dict(SPEC.home), t=0.5)).done
    a = s.act(obs(dict(SPEC.home), t=1.5))

    assert a.done and not a.success
    assert "brak uzytecznego polozenia obiektu 'cube' po 1 s (brak w obserwacji)" in a.note
    assert s.status.stopped_because == a.note and s.status.phase == "done" and s.status.attempt == 0


def test_a_pose_that_stays_too_old_ends_the_run_after_the_wait():
    s = started(wait_for_pose_s=1.0)

    assert not s.act(obs(dict(SPEC.home), t=0.5, objects=seen(age_s=40.0))).done
    a = s.act(obs(dict(SPEC.home), t=1.5, objects=seen(age_s=40.0)))

    assert a.done and not a.success
    assert "brak uzytecznego polozenia obiektu 'cube' po 1 s (za stare: 40 s)" in a.note
    assert s.status.touch_point is None and s.status.attempt == 0


# ------------------------------------------------------------ which poses count
# `max_pose_age_s` differs from `wait_for_pose_s` (30 s), so a check against the wrong setting fails.
@pytest.mark.parametrize("source, age_s, simulated", [
    ("kamery", 0.2, False),
    ("ostatnie widziane", 4.0, False),
    ("ostatnie widziane", 5.0, False),                     # the limit itself
    ("symulacja", 0.0, True),
    ("", 0.0, True),                                       # no source: the simulation's truth
])
def test_a_usable_pose_gives_the_target(source: str, age_s: float, simulated: bool):
    s = started(simulated=simulated, max_pose_age_s=5.0)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen(source=source, age_s=age_s)))

    assert not a.done and s.status.phase == "approach"
    assert s.status.touch_point == pytest.approx((*XY, s.config.touch_z))
    assert s.status.pose_source == source and s.status.pose_age_s == age_s


@pytest.mark.parametrize("source, age_s, simulated, why", [
    ("ostatnie widziane", 6.0, False, "za stare: 6 s"),
    ("ostatnie widziane", NAN, False, "za stare"),
    ("ostatnie widziane (upuszczona)", 1.0, False, "zrodlo 'ostatnie widziane (upuszczona)'"),
    ("ostatnie widziane (1 kamera widzi ja gdzie indziej)", 1.0, False, "zrodlo"),
    ("w dloni", 0.0, False, "zrodlo 'w dloni'"),
    ("", 0.0, False, "bez zrodla poza symulacja"),
])
def test_an_unusable_pose_keeps_the_arm_waiting(source: str, age_s: float, simulated: bool, why: str):
    s = started(simulated=simulated, max_pose_age_s=5.0)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen(source=source, age_s=age_s)))

    assert not a.done and s.status.phase == "wait" and why in a.note
    assert s.status.touch_point is None and s.status.attempt == 0


def test_a_pose_with_no_finite_position_keeps_the_arm_waiting():
    s = started()
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen((NAN, 0.05))))

    assert s.status.phase == "wait" and "polozenie nieskonczone" in a.note


def test_the_label_is_matched_as_the_twin_keys_objects():
    s = started(label="  Red   Pen ")
    s.act(obs(dict(SPEC.home), t=0.05, objects=seen(label="red pen")))

    assert s.label == "red pen" and s.status.phase == "approach"


def test_the_height_of_the_pose_is_ignored():
    """A language model's pose lies on the table, a cube detector's at the cube's centre: same target."""
    points = []
    for z in (0.0, 0.015):
        s = started()
        s.act(obs(dict(SPEC.home), t=0.05, objects=seen(z=z)))
        points.append(s.status.touch_point)

    assert points[0] == pytest.approx(points[1])
    assert points[0] == pytest.approx((*XY, 0.03))


# ------------------------------------------------------------ the path, in free air
def test_in_free_air_the_descent_reaches_the_touch_point_and_reports_a_miss():
    s = started(seen())
    start = tcp(s, s.kin.from_q(s.q_cmd))
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 400)

    last = answers[-1]
    assert runs(phases) == ["approach", "descend", "done"]
    assert last.done and not last.success
    assert last.note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma" == s.status.stopped_because
    assert np.linalg.norm(tcp(s, joints) - s.touch_point) <= s.ik_tol            # the last target: there
    assert not s.status.contact and s.status.attempt == 1

    points = [start] + [tcp(s, a.target) for a in answers]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    for phase, step in zip(phases, steps, strict=True):
        limit = s.descend_step_m if phase == "descend" else s.approach_step_m
        assert step <= limit + s.ik_tol
    down = [p for p, phase in zip(points[1:], phases, strict=True) if phase == "descend"]
    assert all(np.linalg.norm(p[:2] - XY) <= 2 * s.ik_tol for p in down)          # straight down
    assert all(np.diff([p[2] for p in down]) <= 1e-6)
    assert all(set(a.target) == set(SPEC.joints) for a in answers)


def test_without_need_contact_reaching_the_touch_point_is_a_success():
    s = started(seen(), need_contact=False)
    answers, *_ = follow(s, dict(SPEC.home), seen(), 400)

    assert answers[-1].done and answers[-1].success and answers[-1].note == "punkt dotyku osiagniety"


# ------------------------------------------------------------ contact
def test_a_stall_on_the_object_ends_the_run_as_a_success_holding_the_measured_pose():
    s = started(seen())
    top = s.config.touch_z + 0.015
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 400, obstacle=cube_at(XY, top))

    last, contact_tick = answers[-1], answers[-2]
    assert last.done and last.success and last.note.startswith("obiekt dotkniety")
    assert s.status.contact and s.status.attempt == 1
    assert top <= s.status.stop_z_m <= top + s.config.contact_gap_m + s.descend_step_m
    # The tick that finds the contact sends the hold pose, so the arm stops pressing before the run ends.
    assert not contact_tick.done and phases[-2] == "touched"
    assert contact_tick.target == pytest.approx(last.target)
    assert last.target == pytest.approx(dict(joints, gripper=SPEC.limits["gripper"][0]))
    assert not any("nie nadaza" in a.note for a in answers)


def test_a_stall_just_above_the_touch_point_is_a_contact():
    """The hand meets the object 2.5 mm above the touch point: the rise stays under `contact_gap_m`, but an
    arm held up there is a contact, and it holds the measured pose instead of pressing on."""
    s = started(seen())
    top = s.config.touch_z + 0.0025
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 400, obstacle=cube_at(XY, top))

    last = answers[-1]
    assert last.done and last.success and last.note.startswith("obiekt dotkniety") and s.status.contact
    assert top <= s.status.stop_z_m <= top + s.descend_step_m
    assert last.target == pytest.approx(dict(joints, gripper=SPEC.limits["gripper"][0]))


@pytest.mark.parametrize("offset_z", [-0.003, 0.005])
def test_an_arm_that_sags_or_is_held_up_still_tells_a_miss_from_a_touch(offset_z: float):
    """A constant offset between the commanded and the measured TCP: the arm stands still off the hover point,
    and the offset measured there is what the descent compares with."""
    s = started(seen())
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 600, offset_z=offset_z)

    assert runs(phases) == ["approach", "descend", "done"]
    assert answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"

    s = started(seen())
    top = s.config.touch_z + 0.015
    answers, *_ = follow(s, dict(SPEC.home), seen(), 600, obstacle=cube_at(XY, top), offset_z=offset_z)

    assert answers[-1].success and answers[-1].note.startswith("obiekt dotkniety")
    assert top <= s.status.stop_z_m <= top + s.config.contact_gap_m + s.descend_step_m


def test_a_lagging_servo_coming_up_to_the_hover_is_not_a_block():
    """A first-order servo (0.15 s) starting below the hover at the object's x and y, as after a touch: the
    arm comes up to the hover from below, and the offset there is taken only once it has settled."""
    xy, below = (0.2, 0.0), pose_at((0.2, 0.0, 0.04))
    s = started(seen(xy), joints=below)
    answers, phases, *_ = follow(s, below, seen(xy), 600, lag_s=0.15)

    assert runs(phases)[0] == "lift" and runs(phases)[-2:] == ["descend", "done"]
    assert answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"

    s = started(seen(xy), joints=below)
    top = s.config.touch_z + 0.015
    answers, *_ = follow(s, below, seen(xy), 600, obstacle=cube_at(xy, top), lag_s=0.15)

    assert answers[-1].success and answers[-1].note.startswith("obiekt dotkniety")


def test_a_stall_above_the_contact_band_is_a_block():
    s = started(seen())
    top = s.config.touch_z + 0.05
    answers, *_ = follow(s, dict(SPEC.home), seen(), 400, obstacle=cube_at(XY, top))

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note.startswith("ramie zablokowane") and s.status.contact


@pytest.mark.parametrize("frozen_at, xy, settings, descends", [
    ((*XY, 0.05), XY, {}, False),                           # below the hover, at the object
    ((*XY, 0.055), (XY[0], XY[1] + 0.018), {}, False),      # 1.8 cm from the object
    ((*XY, 0.055), XY, {"hover_m": 0.025}, True),           # on the hover point
])
def test_an_arm_that_never_moves_is_never_a_touch(frozen_at, xy, settings, descends: bool):
    """Readings that stop updating, torque off or a jammed joint: the targets move, the measured pose not.

    Standing still counts as arrival only within `SAG_M`, so a frozen arm away from the hover never starts a
    descent. One frozen on the hover point does, and its stall is no touch, because it never came down.
    """
    frozen = pose_at(frozen_at)
    s = started(seen(xy), joints=frozen, **settings)
    answers, phases, _ = run_frozen(s, frozen, seen(xy), 400)

    assert not any(a.success for a in answers) and not s.status.success
    assert not any(a.note.startswith("obiekt dotkniety") for a in answers)
    assert ("descend" in phases) == descends
    if descends:
        assert answers[-1].done and answers[-1].note.startswith("ramie nie zjechalo z zawisu")


# ------------------------------------------------------------ the supervisor's speed cap
def test_targets_respect_the_supervisors_speed_limits():
    """The panel's speed cap at 20 deg/s, given in `RunInfo.max_vel`: no arm joint's target moves faster."""
    s = started(seen(), max_vel=capped(20.0))
    start = s.kin.from_q(s.q_cmd)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 3000, cap_deg_s=20.0)

    assert answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"
    targets = [start] + [a.target for a in answers]
    for before, after in zip(targets, targets[1:], strict=False):
        assert all(abs(after[n] - before[n]) <= 20.0 / s.hz + 1e-6 for n in SPEC.arm_joints)


@pytest.mark.parametrize("told", [True, False], ids=["cap at the start", "cap lowered during the run"])
def test_under_a_speed_cap_free_air_is_a_miss_and_a_stall_is_a_touch(told: bool):
    """The supervisor's command trails the targets at 4 deg/s: that is neither a contact nor a collision."""
    max_vel = capped(4.0) if told else None
    s = started(seen(), max_vel=max_vel)
    answers, *_ = follow(s, dict(SPEC.home), seen(), 4000, cap_deg_s=4.0)

    assert answers[-1].done and answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"

    s = started(seen(), max_vel=max_vel)
    top = s.config.touch_z + 0.015
    commands: list[dict[str, float]] = []
    answers, *_ = follow(s, dict(SPEC.home), seen(), 4000, obstacle=cube_at(XY, top), cap_deg_s=4.0,
                         commands=commands)

    assert answers[-1].success and answers[-1].note.startswith("obiekt dotkniety")
    assert top <= s.status.stop_z_m <= top + s.config.contact_gap_m + s.descend_step_m
    # The verdict waits for the command to reach the hold pose: the twin drops the target sent with `done`.
    arm, held = SPEC.arm_joints, answers[-1].target
    assert {n: commands[-1][n] for n in arm} == pytest.approx({n: held[n] for n in arm}, abs=0.2)


# ------------------------------------------------------------ reach
def test_a_pose_out_of_reach_ends_the_run_and_start_does_not_refuse_it():
    s = started(seen((0.6, 0.4)))
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen((0.6, 0.4))))

    assert a.done and not a.success and "poza zasiegiem ramienia" in a.note
    assert s.status.touch_point is None


def test_near_the_base_the_run_ends_before_the_hand_hits_the_arm():
    """At 15 cm from the base the hand reaches the table only through the shoulder: a stall there would
    read as a touch, so the run ends on the first target that would bring two of the arm's bodies together.
    """
    s = started(seen((0.15, 0.0)))
    answers, *_ = follow(s, dict(SPEC.home), seen((0.15, 0.0)), 400)

    assert answers[-1].done and not answers[-1].success
    assert "samo w siebie" in answers[-1].note and "gripper - shoulder" in answers[-1].note
    assert all(s.kin.self_contacts(a.target) == set() for a in answers)


def test_from_below_the_hover_height_the_arm_goes_straight_up_first():
    low = TouchStrategy()
    low.start(run_info(), obs(dict(SPEC.home)))
    joints = low.kin.ik(np.array([0.25, 0.0, 0.04]), seed=SPEC.home).joints
    s = started(seen(), joints=joints)
    start = tcp(s, s.kin.from_q(s.q_cmd))
    answers, phases, *_ = follow(s, joints, seen(), 400)

    assert runs(phases)[:2] == ["lift", "approach"]
    up = [tcp(s, a.target) for a, phase in zip(answers, phases, strict=True) if phase == "lift"]
    assert all(np.linalg.norm(p[:2] - start[:2]) <= s.ik_tol for p in up)
    assert up[-1][2] == pytest.approx(s.config.touch_z + s.config.hover_m, abs=s.ik_tol)


# ------------------------------------------------------------ the target is latched
@pytest.mark.parametrize("later", [seen((0.24, 0.07)), {}, seen(age_s=300.0)])
def test_later_poses_never_move_the_target(later: dict[str, ObjectPose]):
    """The hand hides the object during the descent: the pose may move, vanish or age."""
    s = started(seen())
    _, _, joints, t = follow(s, dict(SPEC.home), seen(), 3)
    taken = s.status.touch_point
    answers, *_ = follow(s, joints, later, 400, t0=t)

    assert answers[-1].done and s.status.touch_point == taken
    assert np.linalg.norm(tcp(s, answers[-1].target)[:2] - XY) <= s.ik_tol


# ------------------------------------------------------------ the gripper
def test_the_jaw_closes_unless_told_not_to():
    measured = dict(SPEC.home, gripper=30.0)
    closed, *_ = follow(started(seen(), joints=measured), measured, seen(), 400)
    kept, *_ = follow(started(seen(), joints=measured, close_jaw=False), measured, seen(), 400)

    assert all(a.target["gripper"] == pytest.approx(SPEC.limits["gripper"][0]) for a in closed)
    assert all(a.target["gripper"] == pytest.approx(30.0) for a in kept)


@pytest.mark.parametrize("close_jaw", [True, False])
def test_a_squeezing_command_keeps_squeezing(close_jaw: bool):
    measured = dict(SPEC.home, gripper=30.0)
    s = started(seen(), joints=measured, command={"gripper": 5.0}, close_jaw=close_jaw)
    answers, *_ = follow(s, measured, seen(), 400)

    assert all(a.target["gripper"] == pytest.approx(5.0) for a in answers)


# ------------------------------------------------------------ lag = collision
@pytest.mark.parametrize("with_command", [True, False])
def test_the_lag_guard_ends_the_run_by_raising(with_command: bool):
    """The measured pose stops following (a blocked arm) while the base turns toward the object.

    The guard raises: the twin's runner then holds the arm where it measures, while a target sent with
    `done` would be dropped. It compares the measured pose with the supervisor's command, or with the last
    target when the observation has no command.
    """
    blocked = dict(SPEC.home, gripper=40.0)
    where = seen((0.05, 0.22))
    s = started(where, joints=blocked)
    answers, _, lagged = run_frozen(s, blocked, where, 400, with_command=with_command)

    assert lagged.startswith("ramie nie nadaza za celem (")
    assert lagged.endswith(" st.) - kolizja albo blokada")
    assert not any(a.done for a in answers)
    assert s.status.phase == "done" and not s.status.success and s.status.stopped_because == lagged
    assert s.status.step == len(answers)                                 # the stopping tick does not count
    s.stop(f"RuntimeError: {lagged}")                                    # the runner's reason after the raise
    assert s.status.stopped_because == lagged



# ------------------------------------------------------------ the seam for a decision step
class TryAgain(TouchStrategy):
    """A stand-in for a decision step: after a descent with no contact it aims at `corrected`."""

    corrected = (0.0, 0.0)
    starts = 0

    def start(self, run, obs):
        self.starts += 1
        super().start(run, obs)

    def after_touch(self, obs: Observation) -> Action:
        if not self.status.contact and self.status.attempt == 1:
            self.retarget(*self.corrected)
            return self.hold("ponowna proba")
        return super().after_touch(obs)


def test_after_a_miss_a_decision_step_can_aim_again_within_the_run():
    off = (XY[0], XY[1] + 0.03)
    s = started(seen(off), strategy=TryAgain)
    s.corrected = XY
    start = tcp(s, s.kin.from_q(s.q_cmd))
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(off), 800,
                                 obstacle=cube_at(XY, s.config.touch_z + 0.01))

    assert runs(phases) == ["approach", "descend", "lift", "approach", "descend", "touched", "done"]
    assert answers[-1].done and answers[-1].success and s.status.contact
    assert s.status.attempt == 2 and s.starts == 1
    assert s.status.touch_point == pytest.approx((*XY, s.config.touch_z))
    retry = phases.index("lift")
    assert answers[retry].note == "ponowna proba" and answers[retry].target == answers[retry - 1].target

    points = [start] + [tcp(s, a.target) for a in answers]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    for phase, step in zip(phases, steps, strict=True):
        assert step <= (s.descend_step_m if phase == "descend" else s.approach_step_m) + s.ik_tol
    up = [p for p, phase in zip(points[1:], phases, strict=True) if phase == "lift"]
    assert all(np.linalg.norm(p[:2] - off) <= 2 * s.ik_tol for p in up)      # straight up, off the object


class Pending(TouchStrategy):
    """A decision step that never answers: the arm holds where it stopped."""

    def after_touch(self, obs: Observation) -> Action:
        return self.hold("czekam na ocene")


def test_while_a_decision_is_pending_the_arm_holds_and_a_bad_retarget_changes_nothing():
    s = started(seen(), strategy=Pending)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 300)
    assert phases[-1] == "touched" and not answers[-1].done
    assert answers[-1].target == pytest.approx(answers[-2].target) and answers[-1].note == "czekam na ocene"
    taken = s.status.touch_point

    with pytest.raises(RuntimeError, match="poza zasiegiem ramienia"):
        s.retarget(0.6, 0.4)

    assert s.status.phase == "touched" and s.status.touch_point == taken
    s.stop("zatrzymana z panelu")
    with pytest.raises(RuntimeError, match="the run is over"):
        s.retarget(*XY)


def test_retarget_before_start_raises():
    with pytest.raises(RuntimeError, match="start the run first"):
        TouchStrategy().retarget(*XY)


# ------------------------------------------------------------ settings and stop
@pytest.mark.parametrize("name", ["hover_m", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m",
                                  "contact_band_m", "max_tracking_deg"])
def test_settings_must_be_positive(name: str):
    with pytest.raises(ValueError, match=f"touch: {name} must be positive"):
        TouchStrategy(TouchSettings(**{name: 0.0}))


@pytest.mark.parametrize("name", ["wait_for_pose_s", "max_pose_age_s", "time_limit_s"])
@pytest.mark.parametrize("value", [-1.0, NAN])
def test_waits_ages_and_the_time_limit_must_be_zero_or_more(name: str, value: float):
    with pytest.raises(ValueError, match=f"touch: {name} must be zero or more"):
        TouchStrategy(TouchSettings(**{name: value}))


@pytest.mark.parametrize("settings", [{"touch_z": 0.01}, {"touch_z": NAN}])
def test_the_touch_point_stays_above_the_table(settings: dict[str, float]):
    with pytest.raises(ValueError, match="touch: touch_z must be at least 0.02 m"):
        TouchStrategy(TouchSettings(**settings))


def test_the_label_must_not_be_blank():
    with pytest.raises(ValueError, match="touch: label must not be empty"):
        TouchStrategy(TouchSettings(label="   "))


def test_the_time_limit_and_the_rate():
    assert TouchStrategy().time_limit_s == 60.0
    assert TouchStrategy(TouchSettings(time_limit_s=0.0)).time_limit_s is None
    s = TouchStrategy(TouchSettings(hz=25))
    assert isinstance(s.hz, float) and s.hz == 25.0


def test_act_before_start_and_the_stop_reason():
    s = TouchStrategy()
    assert not s.started
    with pytest.raises(RuntimeError, match="przed start"):
        s.act(obs(dict(SPEC.home)))
    s = started(seen())
    answers, *_ = follow(s, dict(SPEC.home), seen(), 400)
    reason = answers[-1].note
    s.stop("strategia skonczyla")                        # the twin's reason after our `done`
    assert s.status.stopped_because == reason
    s.stop("przerwana: STOP z panelu")
    assert s.status.stopped_because == "przerwana: STOP z panelu" and s.status.phase == "done"
