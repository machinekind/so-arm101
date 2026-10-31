"""The grasp strategy without the twin: the probe, the straddle, the close, the lift and every way they end.

The tests play the twin's supervisor and the arm as test_touch.py does (`follow`), with a block of the set
instead of an obstacle point (`Block`): the arm stalls where the jaw's sole would go into the block with a
fingertip over it, and the gripper follows a jaw model of its own. The default model closes onto the block
when both tips straddle it and holds it there: the block then rides with the fingertips until the jaw opens
wider or the arm rises past `slip_z`. Other models are a jaw that starts late, never closes, does not
open, creeps closed or never stands. test_grasp_in_twin.py runs the same strategy through the twin's runner
and physics on a simulated block, with the twin's own tracker.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.kinematics import IKSolution, RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import (  # noqa: E402
    JAW_REACH_M,
    PROBE_BELOW_M,
    GraspSettings,
    GraspStrategy,
    grasp_headings,
    heading,
    top_down,
    wrap,
)
from soarm_lab.strategies.jaw import JawReading  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Where the block lies in most tests [m, base frame]: inside the top-down band.
XY = (0.21, 0.02)
NAN = float("nan")
#: The cube's half sizes [m] and the gripper command whose tip gap is its side [app units].
CUBE = object_spec("cube").half_size
CUBE_UNITS = KIN.gripper_for_gap(2 * CUBE[1])
assert SPEC.gripper_range_q is not None
#: The gripper joint's angle per app unit [rad].
PER_UNIT = (SPEC.gripper_range_q[1] - SPEC.gripper_range_q[0]) / 100
#: A jaw model: (time [s], the supervisor's gripper command, whether both tips straddle the block)
#: -> the measured gripper [app units].
Jaw = Callable[[float, float, bool], float]


def run_info(limits: dict[str, tuple[float, float]] | None = None, simulated: bool = True,
             max_vel: dict[str, float] | None = None) -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=SPEC.gripper,
                   limits=dict(SPEC.limits) if limits is None else limits,
                   max_vel=dict(SPEC.max_vel) if max_vel is None else max_vel,
                   folder=soarm.ROBOT_DIR, simulated=simulated)


def obs(joints: dict[str, float], t: float = 0.0, objects: Mapping[str, ObjectPose] | None = None,
        command: dict[str, float] | None = None) -> Observation:
    return Observation(t=t, step=0, joints=joints, command=command or {}, objects=objects or {})


def seen(xy: tuple[float, float] = XY, source: str = "ostatnie widziane", age_s: float = 8.0, z: float = 0.0,
         label: str = "cube", yaw_deg: float | None = None) -> dict[str, ObjectPose]:
    """A pose as the LLM detector gives one (no orientation), or with a yaw as the box fit gives it."""
    rotation = np.eye(3) if yaw_deg is None else rot_z(math.radians(yaw_deg))
    return {label: ObjectPose(position=np.array([*xy, z], float), rotation=rotation, source=source,
                              age_s=age_s)}


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def started(objects: dict[str, ObjectPose] | None = None, joints: dict[str, float] | None = None,
            simulated: bool = True, command: dict[str, float] | None = None,
            strategy: type[GraspStrategy] = GraspStrategy, max_vel: dict[str, float] | None = None,
            **settings) -> GraspStrategy:
    s = strategy(GraspSettings(**settings))
    s.start(run_info(simulated=simulated, max_vel=max_vel),
            obs(dict(joints or SPEC.home), objects=objects, command=command))
    return s


def pose_at(point: tuple[float, float, float], gripper: float = 0.0) -> dict[str, float]:
    """The joints that put the TCP on `point` [m, base frame]."""
    solution = KIN.ik(np.array(point, float), seed=SPEC.home)
    assert solution.ok
    return dict(solution.joints, gripper=gripper)


def capped(cap_deg_s: float, gripper: float | None = None) -> dict[str, float]:
    """The supervisor's speed limits under the panel's speed cap: the arm joints at most `cap_deg_s`."""
    return {name: (value if gripper is None else gripper) if name == SPEC.gripper else min(value, cap_deg_s)
            for name, value in SPEC.max_vel.items()}


def tcp(joints: Mapping[str, float]) -> np.ndarray:
    return KIN.tcp(joints)[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """The phases in order, repeats dropped."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def contains(seq: list[str], part: list[str]) -> bool:
    """Whether `part` appears in `seq` in order, not necessarily next to each other."""
    it = iter(seq)
    return all(p in it for p in part)


@dataclass
class Block:
    """A block of the set lying on the table, axis-aligned after its yaw, as the fake arm meets it."""

    xy: np.ndarray
    half: tuple[float, float, float]
    yaw: float = 0.0
    #: The jaw holds it: it rides with the fingertips and no longer stalls the arm.
    gripped: bool = False
    #: The block was there at all; False = taken away.
    present: bool = True
    path: list[np.ndarray] = field(default_factory=list)

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def local(self, point: np.ndarray) -> np.ndarray:
        """`point`'s x and y in the block's frame [m]."""
        return rot_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    def stalls(self, joints: Mapping[str, float]) -> bool:
        """The arm cannot take `joints`: the jaw's sole below the top with a fingertip over the block."""
        if not self.present or self.gripped:
            return False
        sole = KIN.tcp(joints)[2, 3] - KIN.jaw_reach(joints[SPEC.gripper])
        if sole >= self.height:
            return False
        return any(bool(np.all(np.abs(self.local(tip)) < self.half[:2])) for tip in KIN.fingertips(joints))

    def between(self, joints: Mapping[str, float]) -> bool:
        """Both tips below the top, on opposite sides of the block's centre along one of its axes, and
        within its extent along the other."""
        if not self.present:
            return False
        if self.gripped:
            return True
        tips = KIN.fingertips(joints)
        if any(tip[2] >= self.height for tip in tips):
            return False
        a, b = (self.local(tip) for tip in tips)
        return any(a[k] * b[k] < 0 and abs(a[1 - k]) < self.half[1 - k] and abs(b[1 - k]) < self.half[1 - k]
                   for k in (0, 1))

    def units(self, joints: Mapping[str, float]) -> float:
        """The gripper command whose tip gap spans the block across the jaws [app units]."""
        a, b = (self.local(tip) for tip in KIN.fingertips(joints))
        k = 0 if a[0] * b[0] < 0 and abs(a[1]) < self.half[1] else 1
        return KIN.gripper_for_gap(min(2 * self.half[k], KIN.jaw_gap(100.0)))


def block_at(xy: tuple[float, float] = XY, half: tuple[float, float, float] = CUBE,
             yaw: float = 0.0) -> Block:
    return Block(np.array(xy, float), half, yaw)


def follow(s: GraspStrategy, joints: dict[str, float], objects: Mapping[str, ObjectPose] | None, ticks: int,
           t0: float = 0.0, block: Block | None = None, offset_z: float = 0.0, cap_deg_s: float | None = None,
           lag_s: float = 0.0, jaw: Jaw | None = None, slip_z: float | None = None,
           commands: list[dict[str, float]] | None = None, measured: list[dict[str, float]] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor, the arm and the jaw, ending on `done`.

    The supervisor's command goes to each target, every arm joint at most `cap_deg_s` per second (None: at
    once), the gripper at once (the strategy ramps it). The arm follows the command as test_touch.py's does,
    and stays where it was when `block` says the new pose would put the jaw's sole into it. The measured
    gripper comes from `jaw` (default: the command, or the block's width when both tips straddle it). Once
    the jaw stands at the block's width with the command tighter, the block is gripped and rides with the
    fingertips' midpoint, until the command opens wider or the TCP rises above `slip_z`. `commands` and
    `measured` collect the command and the measured joints each tick observed.

    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    command = dict(joints)
    share = 1.0 if lag_s <= 0 else 1.0 - np.exp(-1.0 / (s.hz * lag_s))
    t = t0
    g = SPEC.gripper
    assert g is not None
    for _ in range(ticks):
        if commands is not None:
            commands.append(command)
        if measured is not None:
            measured.append(joints)
        answers.append(a := s.act(obs(joints, t=t, objects=objects, command=command)))
        phases.append(s.status.phase)
        t += 1 / s.hz
        command = supervised(command, a.target, None if cap_deg_s is None else cap_deg_s / s.hz)
        arm = servo(joints, held(s, command, offset_z), share)
        arm[g] = joints[g]                                   # the jaw cannot close through the block
        between = block is not None and block.between(arm)
        if jaw is not None:
            grip = jaw(t, command[g], between)
        elif between:
            assert block is not None
            grip = max(command[g], block.units(arm))
        else:
            grip = command[g]
        arm[g] = grip
        if block is not None:
            units = block.units(arm) if between else NAN
            if block.gripped and (command[g] > units + 0.5 or (slip_z is not None and tcp(arm)[2] > slip_z)):
                block.gripped = False
                if jaw is None:
                    arm[g] = command[g]
            elif between and not block.gripped and abs(grip - units) < 0.5 and command[g] < units - 0.5:
                block.gripped = True
            if block.stalls(arm):
                arm = dict(joints, **{g: arm[g]})
            if block.gripped:
                block.xy = np.mean(KIN.fingertips(arm), axis=0)[:2]
            block.path.append(block.xy.copy())
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


def held(s: GraspStrategy, command: dict[str, float], offset_z: float) -> dict[str, float]:
    """Where the arm settles under `command`: its TCP `offset_z` higher (sag below zero)."""
    if offset_z == 0.0:
        return dict(command)
    point = tcp(command) + np.array([0.0, 0.0, offset_z])
    return dict(s.kin.ik_step(point, seed=command, tol=1e-5).joints, gripper=command[SPEC.gripper])


def servo(measured: dict[str, float], settled: dict[str, float], share: float) -> dict[str, float]:
    """The arm after one tick: every joint goes `share` of the way to where it settles."""
    return {name: measured[name] + share * (value - measured[name]) for name, value in settled.items()}


def along(s: GraspStrategy, metres: float) -> np.ndarray:
    """`metres` along the run's closing heading [m, base frame x and y]."""
    return metres * np.array([math.cos(math.radians(s.grasp.heading_deg)),
                              math.sin(math.radians(s.grasp.heading_deg))])


def first_heading() -> float:
    """The closing heading of the rest pose [rad]: the one an identity rotation keeps."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


def offset(xy: tuple[float, float], metres: float) -> tuple[float, float]:
    """`xy` moved `metres` along the rest pose's closing heading."""
    psi = first_heading()
    return (xy[0] + metres * math.cos(psi), xy[1] + metres * math.sin(psi))


def grasped(**settings) -> tuple[GraspStrategy, list[Action], list[str], dict[str, float], Block]:
    """A whole grasp of the cube at `XY` from the rest pose."""
    s = started(seen(), **settings)
    block = block_at()
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 1500, block=block)
    return s, answers, phases, joints, block


# ------------------------------------------------------------ waiting, settings
def test_without_a_pose_the_arm_holds_its_start_pose_with_the_jaw_closed():
    s = started()
    a = s.act(obs(dict(SPEC.home), t=0.05))

    assert set(a.target) == set(SPEC.joints) and not a.done
    assert s.status.phase == "wait" and s.label == "cube"
    assert a.note.startswith("czekam na polozenie obiektu 'cube'")
    arm = SPEC.arm_joints
    assert {n: a.target[n] for n in arm} == pytest.approx({n: SPEC.home[n] for n in arm})
    assert a.target["gripper"] == pytest.approx(SPEC.limits["gripper"][0])


def test_the_probe_height_comes_from_the_block_and_touch_z_zero_means_automatic():
    s = GraspStrategy()

    assert s.config.touch_z == pytest.approx(0.025 - PROBE_BELOW_M + JAW_REACH_M, abs=1e-4)
    assert s.config.touch_z == pytest.approx(0.0216, abs=1e-4)
    assert s.config.hover_m == 0.04 and s.time_limit_s == 120.0 and s.label == "cube"
    s.start(run_info(), obs(dict(SPEC.home)))
    assert s.config.touch_z == pytest.approx(0.0216, abs=1e-4)
    cuboid = GraspStrategy(GraspSettings(object_kind="cuboid85"))
    assert cuboid.config.touch_z == pytest.approx(0.0216, abs=1e-4) and cuboid.label == "cuboid85"
    assert s.grasp_settings.touch_z == 0.0
    assert GraspStrategy(GraspSettings(touch_z=0.03)).config.touch_z == 0.03
    assert GraspStrategy(GraspSettings(label="Red  Cube")).label == "red cube"
    with pytest.raises(ValueError, match="grasp: object_kind must be one of cube, cuboid5"):
        GraspStrategy(GraspSettings(object_kind="ball"))


@pytest.mark.parametrize("settings, match", [
    ({"open_margin_m": 0.0}, "grasp: open_margin_m must be positive"),
    ({"squeeze_m": 0.0}, "grasp: squeeze_m must be positive"),
    ({"lift_m": 0.0}, "grasp: lift_m must be positive"),
    ({"hold_s": NAN}, "grasp: hold_s must be positive"),
    ({"grasp_depth": 0.0}, r"grasp: grasp_depth must lie within \(0, 1\]"),
    ({"grasp_depth": 1.5}, r"grasp: grasp_depth must lie within \(0, 1\]"),
    ({"max_attempts": 0}, "grasp: max_attempts must be at least 1"),
    ({"touch_z": 0.01}, r"grasp: touch_z must be zero \(from the block\) or at least 0.02 m"),
    ({"close_jaw": False}, "grasp: close_jaw must be on"),
])
def test_settings_are_validated(settings: dict, match: str):
    with pytest.raises(ValueError, match=match):
        GraspStrategy(GraspSettings(**settings))


@pytest.mark.parametrize("settings, match", [
    ({"open_margin_m": 0.04}, "grasp: jaw gap 0.1050 m is outside the jaw travel"),
    ({"squeeze_m": 0.02}, "grasp: jaw gap 0.0050 m is outside the jaw travel"),
    ({"squeeze_m": 0.003}, "grasp: squeeze_m 0.003 m is within the jaw margin"),
    ({"open_margin_m": 0.001}, "grasp: open_margin_m 0.001 m is within the jaw margin"),
])
def test_start_refuses_an_opening_or_squeeze_the_jaw_cannot_make(settings: dict, match: str):
    s = GraspStrategy(GraspSettings(**settings))
    with pytest.raises(RuntimeError, match=match):
        s.start(run_info(), obs(dict(SPEC.home)))


def test_act_before_start_and_the_stop_reason():
    s = GraspStrategy()
    with pytest.raises(RuntimeError, match=r"grasp: act\(\) przed start\(\)"):
        s.act(obs(dict(SPEC.home)))
    s, answers, *_ = grasped()
    s.stop("zadanie wykonane")
    assert s.status.stopped_because == answers[-1].note
    s.stop("przerwana: STOP z panelu")
    assert s.status.stopped_because == "przerwana: STOP z panelu" and s.status.phase == "done"


# ------------------------------------------------------------ aiming
def test_the_probe_aims_the_closed_tips_at_the_block_centre():
    s = started(seen())
    answers, phases, joints, t = follow(s, dict(SPEC.home), seen(), 400, block=block_at())

    assert s.grasp.block_xy == pytest.approx(XY)
    point = s.status.touch_point
    assert point is not None
    assert np.asarray(point[:2]) == pytest.approx(np.asarray(XY) + along(s, 0.0156), abs=1e-3)
    settled = [a for a, p, q in zip(answers, phases, phases[1:], strict=False)
               if p == "approach" and q == "descend"]
    descent = [a for a, p in zip(answers, phases, strict=True) if p == "descend"]
    for a in settled + descent:
        approach, _ = KIN.tool_axes(a.target)
        assert approach[2] < -math.cos(math.radians(8.0))
        # The closed tips' midpoint over the block centre, the TCP 15.6 mm toward the moving jaw.
        assert np.mean(KIN.fingertips(a.target), axis=0)[:2] == pytest.approx(np.asarray(XY), abs=1.5e-3)


def test_the_whole_grasp_on_a_cube_in_free_space():
    start = KIN.from_q(started().q_cmd)
    s, answers, phases, joints, block = grasped()

    assert runs(phases)[0] in ("approach", "lift")
    assert runs(phases)[-9:] == ["descend", "touched", "rise", "open", "straddle", "close", "lift_off",
                                 "check", "done"]
    last = answers[-1]
    assert last.done and last.success and s.status.success
    assert last.note.startswith("blok w szczekach, podniesiony o")
    assert int(last.note.split("podniesiony o ")[1].split(" mm")[0]) == pytest.approx(40, abs=5)
    assert s.grasp.verdict == "lifted" and s.status.attempt == 1
    assert s.grasp.top_m == pytest.approx(0.025, abs=0.002)
    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(max(0.012, s.grasp.top_m - 0.0125 + 0.003), abs=1e-3)
    grips = [start["gripper"]] + [a.target["gripper"] for a in answers]
    assert np.max(np.abs(np.diff(grips))) <= 12.0 + 1e-6
    assert answers[-1].target["gripper"] == pytest.approx(KIN.gripper_for_gap(0.017), abs=0.1)
    assert joints["gripper"] == pytest.approx(KIN.gripper_for_gap(0.025), abs=0.1)
    assert all(KIN.self_contacts(a.target) == set() for a in answers)
    # Only the moving jaw closes: it pushes the block onto the fixed jaw's tip, `open_margin_m` along -u.
    assert block.gripped
    assert block.xy - XY == pytest.approx(-along(s, 0.008), abs=1.5e-3)


def test_the_open_jaw_clears_the_block_by_the_margin():
    s, answers, phases, *_ = grasped()
    bottom = [a for a, p in zip(answers, phases, strict=True) if p == "close"][0]
    fixed, moving = KIN.fingertips(dict(bottom.target, gripper=s.cmd_open))
    u = along(s, 1.0)
    v = np.array([-u[1], u[0]])

    for tip, sign in ((fixed, -1), (moving, 1)):
        assert (tip[:2] - XY) @ u == pytest.approx(sign * (0.0125 + 0.008), abs=1e-3)
        assert abs((tip[:2] - XY) @ v) < 1e-3


# ------------------------------------------------------------ the close
def late_jaw(open_units: float, delay_s: float) -> Jaw:
    """A servo that stays at its opening for `delay_s` after the close is commanded, then closes."""
    opened: list[bool] = []
    when: list[float] = []

    def model(t: float, command: float, between: bool) -> float:
        if command >= open_units - 0.5:
            opened.append(True)
        elif opened and command < open_units - 1.0:
            when.append(t)
            if t - when[0] < delay_s:
                return open_units
        return max(command, CUBE_UNITS) if between else command

    return model


def stuck_open_jaw(open_units: float) -> Jaw:
    """A servo that opens and never closes again."""
    opened: list[bool] = []

    def model(t: float, command: float, between: bool) -> float:
        if command >= open_units - 0.5:
            opened.append(True)
        return open_units if opened else command

    return model


@pytest.mark.parametrize("stuck", [False, True], ids=["late", "never"])
def test_the_close_verdict_waits_for_the_jaw_to_leave_its_opening(stuck: bool):
    s = started(seen())
    jaw = stuck_open_jaw(s.cmd_open) if stuck else late_jaw(s.cmd_open, 0.5)
    verdicts: list[str] = []

    class Recorder(GraspStrategy):
        def act(self, obs: Observation) -> Action:
            a = super().act(obs)
            verdicts.append(self.grasp.verdict)
            return a

    s = started(seen(), strategy=Recorder)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=jaw)

    close = [k for k, p in enumerate(phases) if p == "close"]
    if stuck:
        assert answers[-1].done and not answers[-1].success and phases[-1] == "done"
        assert answers[-1].note == "szczeka nie ruszyla po rozkazie zamkniecia (41 mm)"
        assert (len(close) + 1) / s.hz == pytest.approx(3.0, abs=0.1)
        assert s.grasp.verdict == "" and "held" not in verdicts
    else:
        assert answers[-1].success
        held_at = verdicts.index("held")
        assert (held_at - close[0]) / s.hz >= 0.5


def test_after_a_straddle_stall_the_jaw_closes_only_once_the_tips_are_clear():
    s = started(seen(offset(XY, -0.010)))
    measured: list[dict[str, float]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(offset(XY, -0.010)), 1500, block=block_at(),
                                 measured=measured)

    stall = phases.index("straddled")
    stall_z = tcp(measured[stall])[2]
    rise = [k for k in range(stall, len(phases)) if phases[k] == "rise"]
    assert rise
    for k in rise:
        if tcp(measured[k])[2] < stall_z + 0.010 - 1e-4:
            assert answers[k].target["gripper"] == pytest.approx(answers[k - 1].target["gripper"])
    second = phases.index("approach", stall)
    assert answers[second - 1].target["gripper"] == pytest.approx(SPEC.limits["gripper"][0])
    assert answers[stall].target["gripper"] == pytest.approx(s.cmd_open)


@pytest.mark.parametrize("sign", [-1, 1], ids=["moving tip lands", "fixed tip lands"])
def test_a_jaw_tip_on_the_block_retargets_sideways_by_half_the_short_side(sign: int):
    """The pose 1 cm off along the jaws: an open tip lands on the block. The first shift goes toward the
    moving jaw; when the fixed jaw's tip landed, its probe misses and the next goes the other way."""
    pose = seen(offset(XY, sign * 0.010))
    aims: list[tuple[float, float]] = []

    class Aims(GraspStrategy):
        def _aim(self, xy: np.ndarray) -> None:
            super()._aim(xy)
            aims.append((float(xy[0]), float(xy[1])))

    s = started(pose, strategy=Aims)
    block = block_at()
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block)

    assert answers[-1].success and s.grasp.verdict == "lifted", answers[-1].note
    assert s.grasp.straddle_stalls == 1
    first = np.asarray(offset(XY, sign * 0.010))
    if sign < 0:
        assert s.status.attempt == 2
        assert contains(runs(phases), ["straddle", "straddled", "rise", "approach", "descend", "touched",
                                       "rise", "open", "straddle", "close", "lift_off", "check", "done"])
        assert np.asarray(aims[1]) == pytest.approx(first + along(s, 0.0125), abs=1e-6)
    else:
        assert s.status.attempt == 3
        # The miss goes on to the rise on the tick it is found, so only the descents count it.
        assert runs(phases).count("descend") == 3 and runs(phases).count("touched") == 2
        assert np.asarray(aims[1]) == pytest.approx(first + along(s, 0.0125), abs=1e-6)
        assert np.asarray(aims[2]) == pytest.approx(first - along(s, 0.0125), abs=1e-6)
    assert np.linalg.norm(block.path[0] - XY) < 1e-9


def test_an_empty_close_ends_the_run_honestly():
    s = started(seen())
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(),
                                 jaw=lambda t, command, between: command)

    assert runs(phases)[-2:] == ["close", "done"]
    assert answers[-1].note == "chwyt pusty - blok przesuniety albo obok szczek" and not answers[-1].success
    assert s.grasp.verdict == "empty" and s.grasp.empty_closes == 1 and s.status.attempt == 1
    assert answers[-1].target["gripper"] == pytest.approx(s.cmd_closed)


def test_the_sideways_retarget_respects_max_attempts_1():
    pose = seen(offset(XY, -0.010))
    s = started(pose, max_attempts=1)
    measured: list[dict[str, float]] = []
    commands: list[dict[str, float]] = []
    answers, phases, joints, _ = follow(s, dict(SPEC.home), pose, 1500, block=block_at(), measured=measured,
                                        commands=commands)

    stall = phases.index("straddled")
    arm = SPEC.arm_joints
    assert {n: answers[stall].target[n] for n in arm} == pytest.approx({n: measured[stall][n] for n in arm})
    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note == "wyczerpane proby chwytu (1): szczeka oparla sie o blok przy zjezdzie"
    assert answers[-1].target == pytest.approx(answers[stall].target)
    hold = {n: answers[stall].target[n] for n in arm}
    assert {n: commands[-1][n] for n in arm} == pytest.approx(hold, abs=0.2)
    assert len(answers) - 1 - stall <= 2


# ------------------------------------------------------------ carrying
def test_a_block_that_slips_on_the_way_up_ends_the_run():
    s = started(seen())
    # 2 cm above the grasp point the strategy plans with the cube's nominal top.
    slip_z = max(0.012, 0.025 - 0.0125 + 0.003) + 0.02
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), slip_z=slip_z)

    assert phases[-2] == "lift_off" and answers[-1].done and not answers[-1].success
    assert answers[-1].note == "blok wypadl ze szczeki podczas podnoszenia"
    assert s.grasp.verdict == "slipped"


def test_creep_during_the_check_is_a_slip():
    s = started(seen())
    creep: list[float] = []

    def jaw(t: float, command: float, between: bool) -> float:
        if not between:
            return command
        if s.status.phase == "check":
            creep.append(0.004)
        return max(command, CUBE_UNITS - sum(creep) / PER_UNIT)

    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=jaw)

    assert phases[-2] == "check" and answers[-1].done and not answers[-1].success
    assert answers[-1].note == "blok zsuwa sie w szczekach" and s.grasp.verdict == "crept"


def test_a_jaw_that_never_stands_after_the_close_ends_the_run():
    ticks: list[int] = []

    def jaw(t: float, command: float, between: bool) -> float:
        if not between:
            return command
        ticks.append(1)
        return max(command, CUBE_UNITS + (0.03 / PER_UNIT) * (len(ticks) % 2))

    s = started(seen())
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=jaw)

    assert phases[-2] == "close" and not answers[-1].success
    assert answers[-1].note == "szczeka nie staje 3 s po zamknieciu - serwo albo odczyt"


def test_a_jaw_that_does_not_open_ends_the_run():
    s = started(seen())
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(),
                                 jaw=lambda t, command, between: min(command, 15.0))

    assert phases[-2] == "open" and not answers[-1].success
    assert answers[-1].note == "szczeka nie otworzyla sie w 3 s (21 mm zamiast 41 mm)"


# ------------------------------------------------------------ headings
@pytest.mark.parametrize("kind, yaw_deg", [("cube", 20.0), ("cuboid5", 35.0), ("cuboid5", 125.0),
                                           ("cylinder", -60.0)])
def test_a_pose_rotation_puts_the_jaws_across_the_short_side(kind: str, yaw_deg: float):
    pose = seen(label=kind, yaw_deg=yaw_deg, source="kamery", age_s=0.2)
    s = started(pose, object_kind=kind)
    s.act(obs(dict(SPEC.home), t=0.05, objects=pose))
    period = math.degrees(object_spec(kind).yaw_period)
    now = math.degrees(first_heading())
    candidates = [math.degrees(wrap(math.radians(yaw_deg + 90 + k * period)))
                  for k in range(int(360 / period))]
    nearest = min(candidates, key=lambda c: abs(math.degrees(wrap(math.radians(c - now)))))

    assert s.status.phase == "approach", s.status.stopped_because
    assert s.grasp.heading_deg == pytest.approx(nearest, abs=0.5)
    assert s.gap_open_m == pytest.approx(0.041)
    if kind == "cuboid5" and yaw_deg == 125.0:
        assert s.grasp.heading_deg == pytest.approx(35.0, abs=0.5)
    assert len(grasp_headings(rot_z(math.radians(yaw_deg)), object_spec(kind).yaw_period, 0.0)) == (
        4 if kind == "cube" else 2)


def test_an_identity_rotation_keeps_the_current_heading():
    s = started(seen())
    s.act(obs(dict(SPEC.home), t=0.05, objects=seen()))

    assert math.radians(s.grasp.heading_deg) == pytest.approx(first_heading(), abs=math.radians(0.5))
    psi = 0.3
    assert grasp_headings(np.eye(3), math.pi / 2, psi) == pytest.approx([psi, wrap(psi + math.pi)])
    R = top_down(psi)
    assert R @ R.T == pytest.approx(np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert wrap(math.pi) == pytest.approx(math.pi) and wrap(-math.pi) == pytest.approx(math.pi)


# ------------------------------------------------------------ the ends of touch's probe
def test_the_missed_probe_ends_as_touch_does():
    misses: list[tuple[str, str]] = []

    class Misses(GraspStrategy):
        def after_miss(self, kind: str, reason: str, obs: Observation) -> Action:
            misses.append((kind, reason))
            return super().after_miss(kind, reason, obs)

    s = started(seen(), strategy=Misses)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500)

    assert answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"
    assert not answers[-1].success
    assert "rise" not in phases
    assert misses == [("missed", "brak kontaktu w punkcie dotyku - obiektu tam nie ma")]


@pytest.mark.parametrize("xy, why", [((0.6, 0.4), "poza zasiegiem ramienia"),
                                     ((0.28, -0.05), "chwytak nie siega tam pionowo"),
                                     ("tilted", "chwytak nie ustawi sie pionowo (blad 17 st.)")])
def test_a_pose_out_of_reach_or_unlevel_ends_the_run_and_start_does_not_refuse_it(xy, why: str,
                                                                                    monkeypatch):
    where = XY if xy == "tilted" else xy
    s = started(seen(where))
    if xy == "tilted":
        real = s.kin.ik

        def tilted(position, rotation=None, seed=None, **kwargs) -> IKSolution:
            sol = real(position, rotation, seed, **kwargs)
            return IKSolution(sol.joints, sol.pos_err, 0.3 if rotation is not None else sol.rot_err, sol.ok)

        monkeypatch.setattr(s.kin, "ik", tilted)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=seen(where)))

    assert a.done and not a.success and why in a.note, a.note
    assert s.status.touch_point is None and s.grasp.block_xy is None


# ------------------------------------------------------------ the supervisor's speed limits
def test_targets_respect_the_supervisors_speed_limits_including_the_gripper():
    max_vel = capped(20.0, gripper=100.0)
    s = started(seen(), max_vel=max_vel)
    start = KIN.from_q(s.q_cmd)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 6000, block=block_at(), cap_deg_s=20.0)

    assert answers[-1].success, answers[-1].note
    targets = [start] + [a.target for a in answers]
    # The probe's stall holds the measured pose, which trails the command under the cap: that hold may jump
    # back, as touch's does; the supervisor limits the command anyway.
    stall = phases.index("touched")
    for k, (before, after) in enumerate(zip(targets, targets[1:], strict=False)):
        if k != stall:
            assert all(abs(after[n] - before[n]) <= 20.0 / s.hz + 1e-6 for n in SPEC.arm_joints)
        assert abs(after["gripper"] - before["gripper"]) <= 100.0 * 0.8 / s.hz + 1e-6


# ------------------------------------------------------------ seams and notes
class Seams(GraspStrategy):
    """Records every seam call and leaves the default behaviour."""

    def __init__(self, settings: GraspSettings | None = None):
        super().__init__(settings)
        self.calls: list[tuple] = []

    def before_descent(self, obs: Observation) -> Action | None:
        self.calls.append(("before_descent", self.status.attempt, self._t == self._phase_t0))
        return super().before_descent(obs)

    def after_grasp_failed(self, why: str, obs: Observation) -> Action:
        self.calls.append(("after_grasp_failed", why))
        return super().after_grasp_failed(why, obs)

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        self.calls.append(("carrying", self.status.phase, isinstance(reading, JawReading)))
        return super().carrying(obs, reading)

    def after_miss(self, kind: str, reason: str, obs: Observation) -> Action:
        self.calls.append(("after_miss", kind, reason))
        return super().after_miss(kind, reason, obs)


def test_the_seams_are_called():
    s = started(seen(), strategy=Seams)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())
    assert isinstance(s, Seams)

    descents = [c for c in s.calls if c[0] == "before_descent"]
    assert descents == [("before_descent", 1, True)]
    carrying = [c for c in s.calls if c[0] == "carrying"]
    # Every lift_off and check tick, the tick that moves on to check and the one that ends the run.
    assert len(carrying) == phases.count("lift_off") + phases.count("check") + 2
    assert all(c[2] for c in carrying) and {c[1] for c in carrying} == {"lift_off", "check"}

    pose = seen(offset(XY, -0.010))
    s = started(pose, strategy=Seams)
    follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    assert isinstance(s, Seams)
    assert ("after_grasp_failed", "tip") in s.calls
    assert [c[1] for c in s.calls if c[0] == "before_descent"] == [1, 2]

    pose = seen(offset(XY, 0.010))
    s = started(pose, strategy=Seams)
    follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    assert isinstance(s, Seams)
    assert ("after_miss", "missed", "brak kontaktu w punkcie dotyku - obiektu tam nie ma") in s.calls

    s = started(seen(), strategy=Seams)
    follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=lambda t, command, between: command)
    assert isinstance(s, Seams)
    assert ("after_grasp_failed", "empty") in s.calls


class LooksBeforeDescent(GraspStrategy):
    """A wrist look that holds the arm for 0.5 s at the hover, then lets the descent go on."""

    def __init__(self, settings: GraspSettings | None = None):
        super().__init__(settings)
        self.asked: list[float] = []

    def before_descent(self, obs: Observation) -> Action | None:
        self.asked.append(self._t)
        if len(self.asked) <= 10:
            return self.hold("patrze")
        return None


def test_before_descent_is_asked_every_tick_until_it_returns_none_once():
    s = started(seen(), strategy=LooksBeforeDescent)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())
    assert isinstance(s, LooksBeforeDescent)

    assert len(s.asked) == 11 and answers[-1].success
    looking = [a for a in answers if a.note == "patrze"]
    assert len(looking) == 10 and all(a.target == looking[0].target for a in looking)


def test_the_notes_per_phase():
    s, answers, phases, *_ = grasped()
    notes = {p: [a.note for a, q in zip(answers, phases, strict=True) if q == p] for p in set(phases)}

    assert any(n.startswith("w gore po dotyku") for n in notes["rise"] + notes["touched"])
    assert any(n.startswith("nad blok, otwieram szczeke") for n in notes["open"])
    assert any(n.startswith("w dol na blok") for n in notes["straddle"])
    assert any(n.startswith("zamykam szczeke na bloku") for n in notes["close"])
    assert any(n.startswith("sprawdzam chwyt") for n in notes["close"])
    assert any(n.startswith("podnosze blok") for n in notes["lift_off"])
    assert any(n.startswith("trzymam blok") for n in notes["check"])

    pose = seen(offset(XY, -0.010))
    s = started(pose)
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    stalled = [a.note for a, p in zip(answers, phases, strict=True) if p == "straddled"]
    assert stalled and stalled[0] == "szczeka na bloku"
    assert any(a.note.startswith("w gore znad bloku") for a, p in zip(answers, phases, strict=True)
               if p == "rise")
    assert s.grasp_verdict() == ("lifted", "blok podniesiony")
