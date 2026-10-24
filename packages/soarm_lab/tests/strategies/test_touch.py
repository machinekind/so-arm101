"""The touch strategy without the twin: the wait, the pose check, the path, contact, the jaw and the seam.

The tests build the contracts' own `RunInfo`, `Observation` and `ObjectPose` (`machinekind_twin_api`) and
play the arm themselves: the measured joints are the previous tick's target (perfect tracking), except
where a target would put the TCP into an obstacle; there the arm stays where it was. That stall is what
the strategy reads as contact. test_touch_in_twin.py runs the same strategy through the twin's runner and
physics, with a cube.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.touch import TouchSettings, TouchStrategy  # noqa: E402

SPEC = load_robot()
#: Where the object lies in most tests [m, base frame].
XY = (0.22, 0.05)
NAN = float("nan")


def run_info(limits: dict[str, tuple[float, float]] | None = None, simulated: bool = True) -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper,
                   limits=dict(SPEC.limits) if limits is None else limits, max_vel=dict(SPEC.max_vel),
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
            strategy: type[TouchStrategy] = TouchStrategy, **settings) -> TouchStrategy:
    s = strategy(TouchSettings(**settings))
    s.start(run_info(simulated=simulated), obs(dict(joints or SPEC.home), objects=objects, command=command))
    return s


def cube_at(xy: tuple[float, float], top: float, half: float = 0.015) -> Callable[[np.ndarray], bool]:
    """An obstacle: a TCP inside this box cannot be reached."""
    return lambda p: bool(np.all(np.abs(p[:2] - xy) < half) and p[2] < top)


def follow(s: TouchStrategy, joints: dict[str, float], objects: dict[str, ObjectPose] | None, ticks: int,
           t0: float = 0.0, obstacle: Callable[[np.ndarray], bool] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of perfect tracking, ending on `done`; a target into `obstacle` is not followed.

    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    t = t0
    for _ in range(ticks):
        answers.append(a := s.act(obs(joints, t=t, objects=objects)))
        phases.append(s.status.phase)
        t += 1 / s.hz
        if obstacle is None or not obstacle(tcp(s, a.target)):
            joints = dict(a.target)
        if a.done:
            break
    return answers, phases, joints, t


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


# ------------------------------------------------------------ which poses count
@pytest.mark.parametrize("source, age_s, simulated", [
    ("kamery", 0.2, False),
    ("ostatnie widziane", 12.0, False),
    ("ostatnie widziane", 30.0, False),                    # the limit itself
    ("symulacja", 0.0, True),
    ("", 0.0, True),                                       # no source: the simulation's truth
])
def test_a_usable_pose_gives_the_target(source: str, age_s: float, simulated: bool):
    s = started(simulated=simulated)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen(source=source, age_s=age_s)))

    assert not a.done and s.status.phase == "approach"
    assert s.status.touch_point == pytest.approx((*XY, s.config.touch_z))
    assert s.status.pose_source == source and s.status.pose_age_s == age_s


@pytest.mark.parametrize("source, age_s, simulated, why", [
    ("ostatnie widziane", 31.0, False, "za stare: 31 s"),
    ("ostatnie widziane", NAN, False, "za stare"),
    ("ostatnie widziane (upuszczona)", 1.0, False, "zrodlo 'ostatnie widziane (upuszczona)'"),
    ("ostatnie widziane (1 kamera widzi ja gdzie indziej)", 1.0, False, "zrodlo"),
    ("w dloni", 0.0, False, "zrodlo 'w dloni'"),
    ("", 0.0, False, "bez zrodla poza symulacja"),
])
def test_an_unusable_pose_keeps_the_arm_waiting(source: str, age_s: float, simulated: bool, why: str):
    s = started(simulated=simulated)
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


def test_a_stall_above_the_contact_band_is_a_block():
    s = started(seen())
    top = s.config.touch_z + 0.05
    answers, *_ = follow(s, dict(SPEC.home), seen(), 400, obstacle=cube_at(XY, top))

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note.startswith("ramie zablokowane") and s.status.contact


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
def test_the_lag_guard_ends_the_run_with_the_measured_pose():
    """The measured pose stops following (a blocked arm) while the base turns toward the object."""
    blocked = dict(SPEC.home, gripper=40.0)
    where = seen((0.05, 0.22))
    s = started(where, joints=blocked)
    answers = []
    for k in range(400):
        answers.append(a := s.act(obs(blocked, t=k / s.hz, objects=where)))
        if a.done:
            break

    last = answers[-1]
    assert last.done and not last.success and "nie nadaza" in last.note
    assert last.target == pytest.approx(dict(blocked, gripper=SPEC.limits["gripper"][0]))
    assert s.status.step == len(answers) - 1                             # the stopping tick does not count


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
