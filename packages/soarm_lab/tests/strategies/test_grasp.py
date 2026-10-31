"""The grasp strategy without the twin: the probe, the straddle, the close, the lift and every way they end.

The tests play the twin's supervisor and the arm as test_touch.py does (`follow`), with a block of the set
instead of an obstacle point (`Block`): the arm stalls where the jaw's sole would go into the block with a
fingertip over it, and the gripper follows a jaw model of its own. The default model closes onto the block
when both tips straddle it and holds it there: the block then rides with the fingertips until the jaw opens
wider or the arm rises past `slip_z`. Other models are a jaw that starts late, never closes, does not
open, creeps closed or never stands. A hook per phase (`on_phase`) can knock the block out of the jaws, and a
`Script` plays the twin's pose of the block around such a drop, stage by stage, as its tracker would publish
it. test_grasp_in_twin.py runs the same strategy through the twin's runner and physics on a simulated block,
with the twin's own tracker.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.grasp_library import GraspLibrary, save_library  # noqa: E402
from soarm_lab.kinematics import IKSolution, RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import (  # noqa: E402
    IN_HAND,
    JAW_REACH_M,
    MIN_BITE_M,
    MIN_GRASP_Z,
    PROBE_BELOW_M,
    SEEN,
    STALE,
    GraspSettings,
    GraspStatus,
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
#: The objects of each observation: fixed, or a function of the observation's time [s].
Objects = Mapping[str, ObjectPose] | Callable[[float], Mapping[str, ObjectPose]]


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
    #: When the jaw last lost the block: the time of the first observation after it [s]; NaN = never.
    dropped_t: float = NAN

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


@dataclass
class GluedBlock(Block):
    """A block the jaw grips but the arm cannot lift more than `rise_m` from where it gripped it: glued to
    the table, or too heavy. The arm then stays where it was, as on a stall."""

    rise_m: float = 0.0
    #: The TCP height when the jaw gripped the block [m].
    gripped_z: float | None = None

    def stalls(self, joints: Mapping[str, float]) -> bool:
        if not self.gripped:
            return super().stalls(joints)
        z = float(KIN.tcp(joints)[2, 3])
        if self.gripped_z is None:
            self.gripped_z = z
        return z > self.gripped_z + self.rise_m


#: One stage of the twin's pose of a dropped block: (delay after the drop [s], source or None for no pose,
#: x and y or None for where the block lies [m], age at the stage's start [s], height [m]).
Stage = tuple[float, str | None, tuple[float, float] | None, float, float]


def stage(delay_s: float, source: str | None, xy: tuple[float, float] | None = None, age_s0: float = 0.0,
          z: float = 0.0) -> Stage:
    return (delay_s, source, xy, age_s0, z)


@dataclass
class Script:
    """The twin's pose of `block` around a drop, as `follow`'s objects: `before` until the jaw loses the block
    (`Block.dropped_t`), then the latest stage whose delay since the drop has passed, ageing from its start;
    `before` again while no stage has begun."""

    block: Block
    before: Objects
    stages: list[Stage]
    label: str = "cube"

    def __call__(self, t: float) -> Mapping[str, ObjectPose]:
        since = t - self.block.dropped_t
        begun = [st for st in self.stages if since >= st[0] - 1e-9]
        if math.isnan(since) or not begun:
            return self.before(t) if callable(self.before) else self.before
        delay, source, xy, age0, z = begun[-1]
        if source is None:
            return {}
        where = self.block.xy if xy is None else np.asarray(xy, float)
        return {self.label: ObjectPose(np.array([*where, z], float), np.eye(3), source, age0 + since - delay)}


def drop_it(block: Block, where: tuple[float, float], times: int = 1) -> Callable[[float], None]:
    """A hook that knocks the block out of the jaws onto `where` on each of its first `times` calls."""
    calls: list[float] = []

    def hook(t: float) -> None:
        calls.append(t)
        if len(calls) <= times:
            block.gripped = False
            block.xy = np.asarray(where, float)

    return hook


def follow(s: GraspStrategy, joints: dict[str, float], objects: Objects | None, ticks: int,
           t0: float = 0.0, block: Block | None = None, offset_z: float = 0.0, cap_deg_s: float | None = None,
           lag_s: float = 0.0, jaw: Jaw | None = None, slip_z: float | None = None,
           commands: list[dict[str, float]] | None = None, measured: list[dict[str, float]] | None = None,
           command: Mapping[str, float] | None = None,
           on_phase: dict[str, Callable[[float], None]] | None = None,
           trace: list[tuple[float, GraspStatus]] | None = None,
           ) -> tuple[list[Action], list[str], dict[str, float], float]:
    """Up to `ticks` ticks of the supervisor, the arm and the jaw, ending on `done`.

    The supervisor's command goes to each target, every arm joint at most `cap_deg_s` per second (None: at
    once), the gripper at once (the strategy ramps it). The arm follows the command as test_touch.py's does,
    and stays where it was when `block` says the new pose would put the jaw's sole into it. The measured
    gripper comes from `jaw` (default: the command, or the block's width when both tips straddle it). Once
    the jaw stands at the block's width with the command tighter, the block is gripped and rides with the
    fingertips' midpoint, until the command opens wider, a hook lets it go or the TCP rises above `slip_z`
    (once: before any drop); the first observation after that stamps `block.dropped_t`. `objects` may be a
    function of each observation's time. A hook of `on_phase` runs on every entry into its phase, right
    after the first tick that ends in it, with that tick's time. `commands`, `measured` and `trace` collect
    the command, the measured joints, and the time with a copy of `s.grasp`, each tick observed. `command`
    is the supervisor's command to go on from (default: the measured joints).

    Returns the answers, the phase after each tick, the last measured joints and the time to go on from.
    """
    answers, phases = [], []
    command = dict(joints if command is None else command)
    share = 1.0 if lag_s <= 0 else 1.0 - np.exp(-1.0 / (s.hz * lag_s))
    t = t0
    g = SPEC.gripper
    assert g is not None
    for _ in range(ticks):
        if commands is not None:
            commands.append(command)
        if measured is not None:
            measured.append(joints)
        gripped = block is not None and block.gripped
        seen_now = objects(t) if callable(objects) else objects
        answers.append(a := s.act(obs(joints, t=t, objects=seen_now, command=command)))
        phases.append(s.status.phase)
        if trace is not None:
            trace.append((t, replace(s.grasp)))
        if on_phase and phases[-1] in on_phase and (len(phases) == 1 or phases[-2] != phases[-1]):
            on_phase[phases[-1]](t)
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
            slips = slip_z is not None and math.isnan(block.dropped_t) and tcp(arm)[2] > slip_z
            if block.gripped and (command[g] > units + 0.5 or slips):
                block.gripped = False
                if jaw is None:
                    arm[g] = command[g]
            elif between and not block.gripped and abs(grip - units) < 0.5 and command[g] < units - 0.5:
                block.gripped = True
            if gripped and not block.gripped:
                block.dropped_t = t
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
    """A whole grasp of the cube at `XY` from the rest pose, carried for 0.5 s unless told otherwise."""
    settings.setdefault("carry_s", 0.5)
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

    assert runs(phases) == ["approach", "descend", "touched", "rise", "open", "straddle", "close", "lift_off",
                            "check", "carry", "done"]
    last = answers[-1]
    assert last.done and last.success and s.status.success
    assert last.note.startswith("blok w szczekach, podniesiony o")
    assert ", niesiony 0.5 s (szczeka" in last.note
    assert int(last.note.split("podniesiony o ")[1].split(" mm")[0]) == pytest.approx(40, abs=5)
    assert s.grasp.verdict == "lifted" and s.status.attempt == 1
    assert phases.count("carry") == round(0.5 * s.hz) and s.grasp.carried_s == pytest.approx(0.5)
    assert s.grasp.drops == 0 and s.grasp.creeps == 0 and s.grasp.elsewhere is None
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
def test_a_block_the_arm_cannot_lift_ends_the_run_with_the_squeeze_kept():
    s = started(seen())
    block = GluedBlock(np.array(XY, float), CUBE, rise_m=0.012)
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 1500, block=block)

    assert phases[-2] == "lift_off" and answers[-1].done and not answers[-1].success
    note = answers[-1].note
    assert note.startswith("ramie nie podnioslo bloku (") and note.endswith(" mm z 40 mm)"), note
    assert s.grasp.lifted_m == pytest.approx(0.010, abs=0.002)
    # From the phase clock: 4 cm at 0.1 m/s, then 3 s for the arm to follow.
    assert phases.count("lift_off") / s.hz == pytest.approx(0.04 / 0.10 + 3.0, abs=0.06)
    assert answers[-1].target["gripper"] == pytest.approx(s.cmd_closed)
    assert s.grasp.verdict == "held" and block.gripped
    assert joints["gripper"] == pytest.approx(KIN.gripper_for_gap(0.025), abs=0.1)


def test_a_speed_cap_lowered_during_the_lift_lengthens_its_time_limit():
    """The strategy started without a cap, so its targets reach the lift point in 0.4 s; the supervisor's
    command then takes far longer than 3.4 s to get there, and the lift still ends in the check."""
    reference, _, phases, *_ = grasped()
    first = phases.index("lift_off")
    s = started(seen())
    block = block_at()
    sent, before, joints, t = follow(s, dict(SPEC.home), seen(), first + 1, block=block)
    assert before[-1] == "lift_off" and block.gripped
    answers, phases, *_ = follow(s, joints, seen(), 1500, t0=t, block=block, cap_deg_s=1.0,
                                 command=sent[-1].target)

    assert answers[-1].success, answers[-1].note
    assert (phases.count("lift_off") + 1) / s.hz > 0.04 / 0.10 + 3.0
    assert reference.grasp.verdict == "lifted"


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


# ------------------------------------------------------------ carrying and recovery
UPUSZCZONA = "ostatnie widziane (upuszczona)"
#: Where a knocked block lands in most recovery tests: 3 cm from where it was picked [m, base frame].
LANDS = (XY[0] + 0.03, XY[1])


def carried(s: GraspStrategy) -> Callable[[float], Mapping[str, ObjectPose]]:
    """The twin's pose before a drop: the LLM detector's pose at `XY` until the close, then "w dloni"."""
    closed: list[bool] = []

    def objects(t: float) -> Mapping[str, ObjectPose]:
        if closed or s.status.phase in ("close", "lift_off", "check", "carry"):
            closed.append(True)
            return seen(source="w dloni", age_s=0.0, z=0.05)
        return seen()

    return objects


@dataclass
class Knocked:
    """A run whose block is knocked out of the jaws: what the strategy answered and saw on each tick, and the
    last target's TCP, the TCP's touch point and `GraspStatus` at each knock."""

    s: GraspStrategy
    block: Block
    answers: list[Action]
    phases: list[str]
    trace: list[tuple[float, GraspStatus]]
    measured: list[dict[str, float]]
    at_knock: list[tuple[np.ndarray, tuple[float, float, float] | None]]

    @property
    def drop(self) -> int:
        """The tick that declared the first drop."""
        return next(k for k, (_, grasp) in enumerate(self.trace) if grasp.drops >= 1)

    def first(self, phase: str, after: int = 0) -> int:
        return self.phases.index(phase, after)

    @property
    def safe_point(self) -> np.ndarray:
        """Where the arm waits after the first knock: straight up to the probe hover, or higher."""
        z0 = self.at_knock[0][0]
        hover = self.s.config.touch_z + self.s.config.hover_m
        return np.array([z0[0], z0[1], max(hover, z0[2])])


def knocked(stages: list[Stage], phase: str = "carry", where: tuple[float, float] = LANDS, times: int = 1,
            before: Objects | None = None, block: Block | None = None, jaw: Jaw | None = None,
            strategy: type[GraspStrategy] = GraspStrategy, ticks: int = 4000, **settings) -> Knocked:
    """A grasp of the cube at `XY` whose block is knocked onto `where` on the first `times` entries into
    `phase`, the twin's pose played by `stages` (`before` until the drop: `carried`)."""
    settings.setdefault("carry_s", 0.5)
    s = started(seen(), strategy=strategy, **settings)
    block = block or block_at()
    at_knock: list[tuple[np.ndarray, tuple[float, float, float] | None]] = []
    knock = drop_it(block, where, times)

    def hook(t: float) -> None:
        if len(at_knock) < times:
            at_knock.append((s.tcp_cmd.copy(), s.status.touch_point))
        knock(t)

    trace: list[tuple[float, GraspStatus]] = []
    measured: list[dict[str, float]] = []
    script = Script(block, carried(s) if before is None else before, stages)
    answers, phases, *_ = follow(s, dict(SPEC.home), script, ticks, block=block, jaw=jaw,
                                 on_phase={phase: hook}, trace=trace, measured=measured)
    return Knocked(s, block, answers, phases, trace, measured, at_knock)


def seen_again(xy: tuple[float, float] = LANDS, delay_s: float = 0.6) -> list[Stage]:
    """The tracker's guess under the hand at once, then the cameras see the block lying at `xy`."""
    return [stage(0.0, UPUSZCZONA, XY), stage(delay_s, "kamery", xy, 0.1)]


def test_carry_s_zero_ends_the_run_at_the_verified_lift():
    s, answers, phases, *_ = grasped(carry_s=0.0)

    assert answers[-1].success and phases[-2] == "check" and "carry" not in phases
    note = answers[-1].note
    assert note.startswith("blok w szczekach, podniesiony o") and "niesiony 0.0 s" in note
    assert s.grasp.carried_s == 0.0 and s.grasp.verdict == "lifted"


@pytest.mark.parametrize("lift_m", [0.04, 0.07], ids=["lift below the hover", "lift above the hover"])
@pytest.mark.parametrize("phase", ["check", "carry"])
def test_a_block_dropped_mid_carry_and_seen_on_the_table_is_grasped_again(phase: str, lift_m: float):
    run = knocked(seen_again(), phase=phase, lift_m=lift_m)
    s, answers, phases, trace = run.s, run.answers, run.phases, run.trace

    assert answers[-1].success, answers[-1].note
    drop = run.drop
    assert runs(phases[drop:]) == ["rise", "recover", "approach", "descend", "touched", "rise", "open",
                                   "straddle", "close", "lift_off", "check", "carry", "done"]
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1 and s.status.attempt == 2
    # The regrasp aims at the sighting, never at the tracker's guess under the hand.
    assert s.grasp.block_xy == pytest.approx(LANDS, abs=1e-9)
    # Straight up from where the drop found the arm, to the probe hover or no higher than it already was.
    safe = tcp(answers[run.first("recover", drop)].target)
    assert safe == pytest.approx(run.safe_point, abs=s.ik_tol)
    if lift_m > 0.05:
        assert run.safe_point[2] > s.config.touch_z + s.config.hover_m + 0.01
    knock = phases.index(phase)
    assert trace[knock][1].block_source == "w dloni"
    assert trace[drop][1].block_source == UPUSZCZONA
    assert trace[drop][1].dropped_t == trace[drop][0] == s.grasp.dropped_t
    assert trace[drop][1].verdict == "dropped" and answers[drop].note == "blok wypadl ze szczeki - w gore"


def test_the_recovery_opens_the_jaw_and_the_regrasp_closes_it_before_the_descent():
    run = knocked(seen_again())
    s, answers = run.s, run.answers
    g = SPEC.joints.index("gripper")

    drop, recover = run.drop, run.first("recover", run.drop)
    grips = [a.target["gripper"] for a in answers[drop - 1:recover + 1]]
    assert grips[0] == pytest.approx(s.cmd_closed)
    assert all(-1e-9 <= b - a <= 12.0 + 1e-6 for a, b in zip(grips, grips[1:], strict=False))
    assert KIN.to_q(answers[recover].target)[g] == pytest.approx(s._recover_open_q)
    assert s._recover_open_q == pytest.approx(SPEC.jaw.open + 2 * SPEC.jaw.margin)
    # The measured jaw follows: nothing is left between the tips.
    assert run.measured[recover]["gripper"] == pytest.approx(grips[-1])
    assert KIN.to_q(run.measured[recover])[g] > SPEC.jaw.open

    regrasp = run.first("approach", recover)
    descend = run.first("descend", regrasp)
    closing = [a.target["gripper"] for a in answers[regrasp:descend + 1]]
    assert all(-12.0 - 1e-6 <= b - a <= 1e-9 for a, b in zip(closing, closing[1:], strict=False))
    assert answers[descend].target["gripper"] == pytest.approx(SPEC.limits["gripper"][0])
    assert any(a.note == "nad obiekt: zamykam szczeke" for a in answers[regrasp:descend])
    assert answers[regrasp].note == "blok lezy na blacie - ponowna proba"


def test_a_block_that_slips_on_the_way_up_is_grasped_again():
    s = started(seen(), carry_s=0.5)
    block = block_at()
    # 2 cm above the grasp point the strategy plans with the cube's nominal top; the block falls where the
    # jaws held it, and the cameras see it there.
    slip_z = max(0.012, 0.025 - 0.0125 + 0.003) + 0.02
    script = Script(block, carried(s), [stage(0.0, UPUSZCZONA, XY), stage(0.5, "kamery", None, 0.1)])
    answers, phases, *_ = follow(s, dict(SPEC.home), script, 4000, block=block, slip_z=slip_z)

    assert answers[-1].success, answers[-1].note
    assert contains(runs(phases), ["lift_off", "rise", "recover", "approach", "descend", "lift_off", "check",
                                   "carry", "done"])
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1
    assert not any("podczas podnoszenia" in a.note for a in answers)


def test_a_block_that_reappears_as_a_late_last_seen_pose_is_grasped_again():
    """The LLM detector's path: its answer arrives 2 s old as a plain "ostatnie widziane", 3 s after the
    drop, and shows the block where it lies."""
    where = (XY[0] + 0.02, XY[1])
    run = knocked([stage(0.0, UPUSZCZONA, XY), stage(3.0, "ostatnie widziane", where, 2.0)], where=where)
    s, phases, trace = run.s, run.phases, run.trace

    assert run.answers[-1].success, run.answers[-1].note
    assert s.grasp.recoveries == 1 and s.grasp.block_xy == pytest.approx(where, abs=1e-9)
    regrasp = run.first("approach", run.drop)
    assert set(phases[run.drop:regrasp]) == {"rise", "recover"}
    assert trace[regrasp][0] == pytest.approx(trace[run.drop][0] + 3.0, abs=1 / s.hz + 1e-6)


@pytest.mark.parametrize("case, why", [
    ("gone", "kamery go nie widza"),
    ("dropped", f"zrodlo '{UPUSZCZONA}'"),
    ("in a hand", "zrodlo 'kamery', 7 cm nad blatem"),
    ("before the drop", "zrodlo 'ostatnie widziane' sprzed upadku"),
])
def test_a_block_taken_away_stops_the_run_for_a_person(case: str, why: str):
    stages = {"gone": [stage(0.0, UPUSZCZONA, XY), stage(6.0, None)],
              "dropped": [stage(0.0, UPUSZCZONA, XY)],
              "in a hand": [stage(0.5, "kamery", XY, 0.1, z=0.07)],
              "before the drop": []}[case]
    # A capture from before the run, republished with a growing age.
    before = (lambda t: seen(age_s=8.0 + t)) if case == "before the drop" else None
    run = knocked(stages, where=XY, before=before, wait_s=8.0, max_pose_age_s=60.0)
    s, answers, trace = run.s, run.answers, run.trace
    g = SPEC.joints.index("gripper")

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note == f"blok zabrany ze szczeki: {why} od 8 s - czekam na czlowieka"
    assert s.grasp.verdict == "taken" and s.grasp.recoveries == 0
    assert 0.0 <= trace[-1][0] - (s.grasp.dropped_t + 8.0) < 1 / s.hz
    assert "approach" not in run.phases[run.drop:]
    # The arm holds the safe point for a person, the jaw open.
    assert tcp(answers[-1].target) == pytest.approx(run.safe_point, abs=s.ik_tol)
    assert KIN.to_q(answers[-1].target)[g] == pytest.approx(s._recover_open_q)


@pytest.mark.parametrize("where, settings", [((0.6, 0.4), {"recover_m": 1.0}), ((XY[0] + 0.2, XY[1]), {})],
                         ids=["out of reach", "far away"])
def test_a_block_dropped_out_of_reach_or_far_away_stops_the_run(where: tuple[float, float], settings: dict):
    run = knocked(seen_again(where), where=where, **settings)
    s, answers = run.s, run.answers
    g = SPEC.joints.index("gripper")

    assert answers[-1].done and not answers[-1].success and s.grasp.verdict == "taken"
    note = answers[-1].note
    if "recover_m" in settings:
        assert note.startswith("blok upadl poza zasiegiem: punkt (")
        assert note.endswith("poza zasiegiem ramienia - czekam na czlowieka"), note
    else:
        assert s._drop_xy is not None
        far = float(np.linalg.norm(np.asarray(where) - s._drop_xy))
        # The fingertips' midpoint at the drop: where the closed jaw held the block, about 1 cm from `XY`.
        assert far == pytest.approx(0.2, abs=0.015) and far > s.grasp_settings.recover_m
        assert note == f"blok zabrany: lezy {far * 100:.0f} cm od miejsca upadku - czekam na czlowieka"
    # The failed re-aim changed nothing: the arm on the safe point, the jaw open, the old aim kept.
    assert tcp(answers[-1].target) == pytest.approx(run.safe_point, abs=s.ik_tol)
    assert KIN.to_q(answers[-1].target)[g] == pytest.approx(s._recover_open_q)
    assert s.status.touch_point == run.at_knock[0][1]
    assert s.grasp.recoveries == 0 and "approach" not in run.phases[run.drop:]


def test_recoveries_share_the_attempt_budget():
    run = knocked(seen_again(), times=2, max_attempts=2)
    s, answers = run.s, run.answers

    assert answers[-1].note == "wyczerpane proby chwytu (2): blok wypadl ze szczeki"
    assert not answers[-1].success and s.grasp.verdict == "taken"
    assert s.grasp.drops == 2 and s.grasp.recoveries == 1 and s.status.attempt == 2


def test_a_fresh_pose_elsewhere_while_holding_is_only_noted():
    s = started(seen(), carry_s=0.5)
    other = (XY[0] + 0.1, XY[1])

    def objects(t: float) -> Mapping[str, ObjectPose]:
        if s.status.phase in ("lift_off", "check", "carry"):
            return seen(other, source="kamery", age_s=0.1)
        return seen()

    answers, phases, *_ = follow(s, dict(SPEC.home), objects, 1500, block=block_at())

    assert answers[-1].success, answers[-1].note
    assert s.grasp.elsewhere == pytest.approx(other) and s.grasp.drops == 0
    assert any(a.note.endswith(" - kamery widza blok gdzie indziej")
               for a, p in zip(answers, phases, strict=True) if p == "carry")


def creeping(s: GraspStrategy, phase: str) -> Jaw:
    """A jaw that closes on the block by 0.0175 rad on the 4th and on the 5th tick of `phase`: each step
    below `jaw.still` (0.02), the two together above `jaw.creep` (0.03)."""
    calls: list[float] = []

    def model(t: float, command: float, between: bool) -> float:
        if not between:
            return command
        if s.status.phase == phase:
            calls.append(t)
        # The model gives the next tick's measurement: its 3rd call in the phase measures the 4th tick.
        creep = 0.0175 * min(2, max(0, len(calls) - 2))
        return max(command, CUBE_UNITS - creep / PER_UNIT)

    return model


@pytest.mark.parametrize("phase", ["check", "carry"])
def test_creep_restarts_the_check_and_is_noted_while_carrying(phase: str):
    _, _, plain, *_ = grasped()
    s = started(seen(), carry_s=0.5)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=creeping(s, phase))

    assert answers[-1].success, answers[-1].note
    assert s.grasp.creeps == 1 and s.grasp.drops == 0
    # The jaw squeezes all the while: the measured jaw stays wider than the command by more than the margin.
    assert KIN.to_q({"gripper": CUBE_UNITS})[-1] - 0.035 - s._q(s.cmd_closed) > SPEC.jaw.margin
    if phase == "check":
        # The streak is zeroed on the 5th tick and starts again on the 6th.
        assert phases.count("check") - plain.count("check") == pytest.approx(5, abs=1)
        assert phases.count("check") < (s.grasp_settings.hold_s + 3.0) * s.hz
    else:
        assert phases.count("check") == plain.count("check")


class ReleasesWhileCarrying(GraspStrategy):
    """A place step's shape: it watches through `super().carrying` and commands the release 0.5 s into the
    carry."""

    def carrying(self, obs: Observation, reading: JawReading) -> Action | None:
        action = super().carrying(obs, reading)
        if action is None and self.status.phase == "carry" and self._t - self.phase_started >= 0.5:
            return self.release()
        return action


@pytest.mark.parametrize("how", ["setting", "subclass"])
def test_the_release_opens_the_jaw_and_ends_the_watch(how: str):
    if how == "setting":
        s = started(seen(), release=True, carry_s=0.5)
    else:
        s = started(seen(), strategy=ReleasesWhileCarrying)
    block = block_at()
    answers, phases, joints, _ = follow(s, dict(SPEC.home), seen(), 1500, block=block)

    assert runs(phases)[-3:] == ["carry", "release", "done"]
    assert answers[-1].success, answers[-1].note
    assert answers[-1].note.startswith("blok oddany: szczeka otwarta (")
    assert s.grasp.released and s.grasp.verdict == "released" and s.grasp.drops == 0
    release = phases.index("release")
    assert answers[release].note == "oddaje blok: otwieram szczeke"
    grips = [a.target["gripper"] for a in answers[release - 1:]]
    assert grips[0] == pytest.approx(s.cmd_closed)
    assert all(-1e-9 <= b - a <= 12.0 + 1e-6 for a, b in zip(grips, grips[1:], strict=False))
    assert grips[-1] == pytest.approx(KIN.gripper_for_gap(s.gap_open_m), abs=0.1)
    # The measured jaw follows: the block is no longer held.
    assert joints["gripper"] == pytest.approx(grips[-1]) and not block.gripped
    assert not {"rise", "recover"} & set(phases[release:])


def test_a_jaw_that_does_not_open_on_release_ends_the_run():
    s = started(seen(), release=True, carry_s=0.5)

    def jaw(t: float, command: float, between: bool) -> float:
        if s.status.phase == "release":
            return CUBE_UNITS
        return max(command, CUBE_UNITS) if between else command

    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=jaw)

    assert phases[-2] == "release" and answers[-1].done and not answers[-1].success
    assert answers[-1].note == "szczeka nie otworzyla sie po oddaniu bloku w 3 s (25 mm zamiast 41 mm)"
    assert (phases.count("release") + 1) / s.hz == pytest.approx(3.0, abs=0.06)


def test_release_outside_a_carrying_phase_raises():
    s = started()
    s.act(obs(dict(SPEC.home), t=0.05))
    assert s.status.phase == "wait"
    with pytest.raises(RuntimeError, match=r"grasp: release\(\) outside a carrying phase"):
        s.release()
    s, *_ = grasped()
    assert s.status.phase == "done"
    with pytest.raises(RuntimeError, match=r"grasp: release\(\) outside a carrying phase"):
        s.release()


def test_one_empty_reading_is_no_drop():
    s = started(seen(), carry_s=1.0)
    once: list[float] = []

    def jaw(t: float, command: float, between: bool) -> float:
        if s.status.phase == "carry" and not once:
            once.append(t)
            return command
        return max(command, CUBE_UNITS) if between else command

    trace: list[tuple[float, GraspStatus]] = []
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at(), jaw=jaw, trace=trace)

    assert answers[-1].success, answers[-1].note
    assert s.grasp.drops == 0 and "rise" not in phases[phases.index("carry"):]
    # The empty reading came, inside the carry.
    assert once and any(not grasp.holding for t, grasp in trace if t > once[0] - 1e-9)


def test_the_watch_reads_a_tick_once_when_the_phase_changes_within_it():
    """The tick that arrives at the lift point moves on to the check, and both phases call `carrying` on it:
    an empty reading there counts once."""
    _, _, reference, *_ = grasped()
    s = started(seen(), carry_s=0.5)
    _, phases, joints, t = follow(s, dict(SPEC.home), seen(), reference.index("lift_off") + 1,
                                  block=block_at())
    assert phases[-1] == "lift_off"
    empty = JawReading(q=0.0, cmd=0.0, still=True, squeezing=False, holding=False, open=False, crept=False)
    tick = obs(joints, t=t, objects=seen())

    # The next tick, on which lift_off and then check ask.
    s.status.step += 1
    assert s.carrying(tick, empty) is None and s.carrying(tick, empty) is None
    assert s.grasp.drops == 0 and s._empty_ticks == 1
    s.status.step += 1
    assert s.carrying(tick, empty) is not None
    assert s.grasp.drops == 1 and s.status.phase == "rise"


def test_a_regrasp_that_misses_ends_as_touch_does():
    """The cameras say the block lies 3 cm from where it was picked; it lies 8 cm away."""
    run = knocked(seen_again((XY[0] + 0.03, XY[1])), where=(XY[0] + 0.08, XY[1]))

    assert run.answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"
    assert not run.answers[-1].success and run.s.grasp.recoveries == 1
    assert runs(run.phases).count("recover") == 1


@pytest.mark.parametrize("settings, match", [
    ({"carry_s": -1.0}, "grasp: carry_s must be zero or more"),
    ({"carry_s": NAN}, "grasp: carry_s must be zero or more"),
    ({"wait_s": 0.0}, "grasp: wait_s must be positive"),
    ({"recover_m": 0.0}, "grasp: recover_m must be positive"),
])
def test_recovery_settings_are_validated(settings: dict, match: str):
    with pytest.raises(ValueError, match=match):
        GraspStrategy(GraspSettings(**settings))


def test_the_recovery_defaults_fit_a_drop_and_a_regrasp_in_the_time_limit():
    d = GraspSettings()

    assert (d.carry_s, d.release, d.wait_s, d.recover_m, d.time_limit_s) == (10.0, False, 20.0, 0.15, 120.0)
    # A grasp of about 10 s and its carry, twice, and one wait, with room to spare.
    assert d.time_limit_s >= 2 * (10.0 + d.carry_s) + d.wait_s + 5.0


def test_the_notes_of_the_recovery_phases():
    run = knocked(seen_again())
    notes = {p: [a.note for a, q in zip(run.answers, run.phases, strict=True) if q == p]
             for p in set(run.phases)}

    assert all(n.startswith("niose blok: ") for n in notes["carry"])
    rise = [a.note for a in run.answers[run.drop + 1:run.first("recover", run.drop)]]
    assert rise and all(n.startswith("blok wypadl ze szczeki - w gore: ") for n in rise)
    assert all(n.startswith("blok wypadl, czekam na kamery: ") for n in notes["recover"])
    assert run.s.grasp_verdict() == ("lifted", "blok podniesiony")

    s, answers, phases, *_ = grasped(release=True)
    assert [a.note for a, p in zip(answers, phases, strict=True) if p == "release"][0] == (
        "oddaje blok: otwieram szczeke")
    assert s.grasp_verdict() == ("released", "blok oddany")


def test_the_sighting_kinds():
    s = started()
    joints = pose_at((XY[0], XY[1], 0.04))
    jaw = np.mean(KIN.fingertips(joints), axis=0)
    under = (float(jaw[0]), float(jaw[1]))

    def sighting(source: str, xy: tuple[float, float] = (0.3, 0.1), z: float = 0.0, age_s: float = 0.1,
                 squeezing: bool = False, strategy: GraspStrategy = s, position: np.ndarray | None = None):
        where = np.array([*xy, z], float) if position is None else position
        pose = ObjectPose(where, rot_z(0.3), source, age_s)
        return strategy._sighting(obs(joints, t=5.0, objects={"cube": pose}), squeezing)

    def kind(*args, **kwargs) -> str:
        found = sighting(*args, **kwargs)
        assert found is not None
        return found.kind

    assert kind("w dloni", z=0.05) == IN_HAND
    assert kind("kamery") == SEEN and kind("ostatnie widziane", age_s=2.0) == SEEN
    assert kind("ostatnie widziane", age_s=40.0) == STALE
    assert kind(UPUSZCZONA) == STALE and kind("ostatnie widziane (1 kamera widzi ja gdzie indziej)") == STALE
    assert kind("kamery", z=0.07) == STALE and kind("brak") == STALE
    assert float(np.linalg.norm(np.array([*under, 0.0125]) - jaw)) < 0.06
    assert kind("symulacja", under, z=0.0125, squeezing=True) == IN_HAND
    assert kind("symulacja", under, z=0.0125, squeezing=False) == SEEN
    assert kind("symulacja", (under[0] + 0.2, under[1]), z=0.0125, squeezing=True) == SEEN
    assert kind("", under, z=0.0125) == SEEN
    # A cuboid85 landed on its end under the hand: its centre 4 cm below the tips, at rest height.
    cuboid = started(object_kind="cuboid85", label="cube")
    assert kind("symulacja", under, z=0.0425, squeezing=False, strategy=cuboid) == SEEN
    assert kind("", strategy=started(simulated=False)) == STALE
    assert sighting("kamery", position=np.array([NAN, 0.0, 0.0])) is None
    assert sighting("kamery", position=np.zeros(2)) is None
    found = sighting("kamery", age_s=1.5)
    assert found is not None
    assert found.seen_t == pytest.approx(3.5) and found.xy == (0.3, 0.1) and found.source == "kamery"
    assert found.rotation == pytest.approx(rot_z(0.3))
    assert s._sighting(obs(joints, t=5.0, objects={}), False) is None


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
    # Every lift_off, check and carry tick, the ticks that move on to check and to carry, and the one that
    # ends the run.
    assert len(carrying) == phases.count("lift_off") + phases.count("check") + phases.count("carry") + 3
    assert all(c[2] for c in carrying) and {c[1] for c in carrying} == {"lift_off", "check", "carry"}

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


class LooksBeforeStraddle(GraspStrategy):
    """A wrist look at the straddle hover: `_looking` for `LOOK_S` from its first call, holding the arm,
    then done."""

    LOOK_S = 4.0

    def __init__(self, settings: GraspSettings | None = None):
        super().__init__(settings)
        self.asked: list[tuple[float, bool]] = []

    @property
    def _looking(self) -> bool:
        return bool(self.asked) and self._t - self.asked[0][0] < self.LOOK_S

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        self.asked.append((self._t, settled))
        return self.hold("patrze z otwarta szczeka") if self._looking else None


def test_before_straddle_is_asked_on_the_first_ready_tick_and_the_open_jaw_waits_while_it_looks():
    _, _, plain, *_ = grasped()
    s = started(seen(), strategy=LooksBeforeStraddle)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())
    assert isinstance(s, LooksBeforeStraddle)

    looking = [k for k, a in enumerate(answers) if a.note == "patrze z otwarta szczeka"]
    # The plain run straddles on the tick the look begins: the arm settled, the jaw open, the hand top-down.
    assert looking[0] == plain.index("straddle")
    assert s.asked[0][1] is True
    assert len(s.asked) == len(looking) + 1 and looking == list(range(looking[0], looking[-1] + 1))
    # OPEN's 3 s timeout waits while the look holds the arm.
    assert len(looking) / s.hz == pytest.approx(LooksBeforeStraddle.LOOK_S, abs=0.06)
    assert {phases[k] for k in looking} == {"open"}
    assert all(answers[k].target == answers[looking[0]].target for k in looking)
    # The tick the seam answers None goes on to the straddle.
    assert phases[looking[-1] + 1] == "straddle"
    assert answers[-1].success, answers[-1].note


class LooksOnTheWay(GraspStrategy):
    """A look that runs from the first OPEN tick of each straddle hover and is done at once, before the arm
    has settled there."""

    def __init__(self, settings: GraspSettings | None = None):
        super().__init__(settings)
        self.asked: list[tuple[int, bool]] = []

    @property
    def _looking(self) -> bool:
        return self.status.phase == "open" and self.status.attempt not in {a for a, _ in self.asked}

    def before_straddle(self, obs: Observation, settled: bool) -> Action | None:
        self.asked.append((self.status.attempt, settled))
        return None


@pytest.mark.parametrize("shift", [0.0, -0.010], ids=["one straddle", "a tip stall, then a second"])
def test_before_straddle_is_asked_until_it_returns_none_once_per_straddle_hover(shift: float):
    pose = seen(offset(XY, shift))
    s = started(pose, strategy=LooksOnTheWay)
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    assert isinstance(s, LooksOnTheWay)

    assert answers[-1].success, answers[-1].note
    # Asked on the way, not again once the arm settles at the hover.
    assert s.asked == ([(1, False)] if shift == 0.0 else [(1, False), (2, False)])
    assert runs(phases).count("open") == len(s.asked)


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
    assert all(n.startswith("niose blok: ") for n in notes["carry"])

    pose = seen(offset(XY, -0.010))
    s = started(pose)
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    stalled = [a.note for a, p in zip(answers, phases, strict=True) if p == "straddled"]
    assert stalled and stalled[0] == "szczeka na bloku"
    assert any(a.note.startswith("w gore znad bloku") for a, p in zip(answers, phases, strict=True)
               if p == "rise")
    assert s.grasp_verdict() == ("lifted", "blok podniesiony")


# ------------------------------------------------------------ the grasp library
#: The cube's pose as the box fit gives it, turned by 90 degrees: the same axis-aligned box for the fake arm
#: (`Block` has no yaw of its own here), but not the identity, so library mode composes the library.
YAWED = seen(source="kamery", age_s=0.2, yaw_deg=90.0)
DOWN = (0.0, 0.0, -1.0)
ACROSS_Y = (0.0, 1.0, 0.0)


def grasp(pad: tuple[float, float, float], approach: tuple[float, float, float] = DOWN,
          closing: tuple[float, float, float] = ACROSS_Y, width: float = 0.025, score: float = 0.9) -> dict:
    """One grasp of a library file: the pad point and the axes in the cube's block frame."""
    return {"position": list(pad), "approach": list(approach), "closing": list(closing), "width_m": width,
            "score": score}


def library_dir(tmp_path, grasps: list[dict]) -> str:
    """A folder holding a cube library of `grasps`, for `library_dir`."""
    library = GraspLibrary.from_dict({"format": "soarm-lab-grasps", "version": 1, "kind": "cube",
                                      "half_size": list(CUBE), "generator": {"name": "hand"},
                                      "grasps": grasps})
    save_library(library, tmp_path / "cube.json")
    return str(tmp_path)


def aimed(objects: dict[str, ObjectPose], **settings) -> tuple[GraspStrategy, Action]:
    """A library strategy started and given one tick with `objects`: the tick that takes the pose and aims."""
    s = started(objects, grasp_source="library", **settings)
    return s, s.act(obs(dict(SPEC.home), t=0.05, objects=objects))


def test_the_library_source_is_validated_at_construction(tmp_path):
    with pytest.raises(ValueError, match="grasp: no grasp library for 'cuboid5'"):
        GraspStrategy(GraspSettings(grasp_source="library", object_kind="cuboid5"))
    with pytest.raises(ValueError, match="grasp: grasp_source must be one of geometry, library, is 'x'"):
        GraspStrategy(GraspSettings(grasp_source="x"))
    with pytest.raises(ValueError, match=r"grasp: min_score must lie within \[0, 1\], is 1.5"):
        GraspStrategy(GraspSettings(min_score=1.5))
    (tmp_path / "cube.json").write_text('{"format": "soarm-lab-grasps", "version": 2}')
    with pytest.raises(ValueError, match="grasp: not a grasp library of version 1"):
        GraspStrategy(GraspSettings(grasp_source="library", library_dir=str(tmp_path)))
    # Geometry mode reads no library, whatever the folder says.
    assert GraspStrategy(GraspSettings(library_dir=str(tmp_path))).library is None
    assert len(GraspStrategy(GraspSettings(grasp_source="library")).library.grasps) == 8


@pytest.mark.parametrize("min_score, rejected", [(0.0, "pochylenie 8"), (0.6, "pochylenie 8, wynik 8")])
def test_a_library_grasp_is_chosen_by_score_then_heading_and_the_run_succeeds(min_score: float,
                                                                             rejected: str):
    s = started(YAWED, grasp_source="library", min_score=min_score, carry_s=0.5)
    answers, phases, *_ = follow(s, dict(SPEC.home), YAWED, 1500, block=block_at())
    grasp = s.grasp

    assert answers[-1].success, answers[-1].note
    assert grasp.grasp_source == "library" and grasp.rejected == rejected
    # The other centred grasps (0.91, 0.90 and 0.89) are index 0's own headings: one of each is kept.
    assert grasp.library_index == 0 and grasp.library_score == 0.92
    assert grasp.candidates == (20 if min_score == 0.0 else 16)
    # Of index 0's four headings the one nearest the rest pose's.
    assert grasp.heading_deg == pytest.approx(0.0, abs=0.5)
    assert min(abs(grasp.heading_deg - k * 90.0) for k in range(-2, 3)) < 0.5
    assert s.gap_open_m == pytest.approx(0.041) and s.gap_closed_m == pytest.approx(0.017)
    assert grasp.depth_m == pytest.approx(0.0125) and grasp.offset_xy == pytest.approx((0.0, 0.0))
    assert grasp.skipped == ""
    approach = [a.note for a, p in zip(answers, phases, strict=True) if p == "approach"]
    assert approach and approach[0].endswith("(biblioteka #0, wynik 0.92)")


def test_an_identity_rotation_falls_back_to_geometry_in_library_mode():
    s = started(seen(), grasp_source="library", carry_s=0.5)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())

    assert answers[-1].success, answers[-1].note
    assert s.grasp.grasp_source == "geometry (brak obrotu)"
    assert s.grasp.library_index == -1 and math.isnan(s.grasp.library_score)
    assert math.radians(s.grasp.heading_deg) == pytest.approx(first_heading(), abs=math.radians(0.5))
    approach = [a.note for a, p in zip(answers, phases, strict=True) if p == "approach"]
    assert approach and approach[0].endswith("(geometria: brak obrotu)")
    # Geometry mode says nothing of a source in its notes.
    s, answers, phases, *_ = grasped()
    assert s.grasp.grasp_source == "geometry" and s.library is None
    assert not any("biblioteka" in a.note or "geometria" in a.note for a in answers)


def test_a_library_whose_grasps_all_fail_the_cheap_filters_ends_the_run(tmp_path):
    side = grasp((0.0, 0.0, 0.0), approach=(-1.0, 0.0, 0.0))
    s, a = aimed(YAWED, library_dir=library_dir(tmp_path, [side]))

    assert a.done and not a.success
    assert a.note.startswith("biblioteka chwytow 'cube': zaden z 8 chwytow nie pasuje")
    assert "pochylenie 8" in a.note
    assert s.status.phase == "done" and s.grasp.block_xy is None and s.status.touch_point is None


def test_a_library_grasp_the_jaw_cannot_open_for_is_rejected_as_szczeka(tmp_path):
    wide = grasp((0.0, 0.0, 0.0), width=0.080)
    s = GraspStrategy(GraspSettings(grasp_source="library", library_dir=library_dir(tmp_path, [wide])))
    s.start(run_info(), obs(dict(SPEC.home), objects=YAWED))

    # The short side still fits, so start does not refuse; the grasp's own opening does not.
    assert 0.080 + 2 * s.grasp_settings.open_margin_m > s.kin.jaw_gap(100.0)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=YAWED))
    assert a.done and not a.success
    assert "zaden z 8 chwytow nie pasuje (szczeka 8)" in a.note


def test_a_library_grasp_off_centre_moves_the_straddle_point_and_its_depth(tmp_path):
    folder = library_dir(tmp_path, [grasp((0.005, 0.0, 0.0085))])
    s = started(YAWED, grasp_source="library", library_dir=folder, carry_s=0.5)
    s.act(obs(dict(SPEC.home), t=0.05, objects=YAWED))
    grasp_ = s.grasp
    psi = math.radians(grasp_.heading_deg)
    centre = KIN.jaw_centre(s.cmd_open)

    assert math.hypot(*grasp_.offset_xy) == pytest.approx(0.005, abs=1e-6)
    assert grasp_.depth_m == pytest.approx(0.004, abs=1e-9)
    u = np.array([math.cos(psi), math.sin(psi)])
    expected = np.asarray(XY) + np.asarray(grasp_.offset_xy) - centre[2] * u
    assert s._straddle_xy == pytest.approx(expected, abs=1e-3)
    # The jaw point lies over the pad point; the probe still goes to the block centre.
    assert s.status.touch_point is not None
    assert np.asarray(s.status.touch_point[:2]) == pytest.approx(np.asarray(XY) + along(s, 0.0156), abs=1e-3)
    assert 0.004 - KIN.tip_swing(s.cmd_open) / 2 >= MIN_BITE_M and grasp_.skipped == ""

    answers, *_ = follow(s, dict(SPEC.home), YAWED, 1500, block=block_at())
    assert answers[-1].success, answers[-1].note
    assert grasp_.grasp_point is not None
    expected_z = max(MIN_GRASP_Z, grasp_.top_m - 0.004 + centre[0])
    assert grasp_.grasp_point[2] == pytest.approx(expected_z, abs=1e-3)


def test_a_shallow_library_grasp_is_skipped_by_the_bite_rule(tmp_path):
    """The tips' midpoint 1.5 mm below the top: the trailing tip would sit on the top edge. The four headings
    of the centred grasp are its only executed grasps (fold k flipped is fold k + 2)."""
    s, a = aimed(YAWED, library_dir=library_dir(tmp_path, [grasp((0.0, 0.0, 0.011))]))

    assert a.done and not a.success
    assert a.note.startswith("biblioteka chwytow 'cube': zaden chwyt nie pasuje do szczeki")
    assert "zbyt plytko 4" in a.note and s.grasp.skipped == "zbyt plytko 4"


def test_a_deep_library_grasp_is_clamped_to_the_minimum_grasp_height(tmp_path):
    folder = library_dir(tmp_path, [grasp((0.0, 0.0, -0.0105))])
    s = started(YAWED, grasp_source="library", library_dir=folder, carry_s=0.5)
    answers, phases, *_ = follow(s, dict(SPEC.home), YAWED, 1500, block=block_at())

    assert s.grasp.depth_m == pytest.approx(0.023)
    assert "rise" in phases and s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(MIN_GRASP_Z)


def test_the_openings_follow_the_library_grasps_width(tmp_path):
    """The diagonal of the cube is 35.4 mm across: the jaw opens to it plus the margins and closes to it minus
    the squeeze, and a retarget would move by half of it."""
    diagonal = grasp((0.0, 0.0, 0.0), closing=(math.sqrt(0.5), math.sqrt(0.5), 0.0), width=0.035355)
    s, a = aimed(YAWED, library_dir=library_dir(tmp_path, [diagonal]))

    assert not a.done, a.note
    assert s.gap_open_m == pytest.approx(0.035355 + 0.016)
    assert s.gap_closed_m == pytest.approx(0.035355 - 0.008)
    assert KIN.jaw_gap(s.cmd_open) == pytest.approx(0.051355, abs=1e-6)
    assert KIN.jaw_gap(s.cmd_closed) == pytest.approx(0.027355, abs=1e-6)
    # Along a diagonal of the cube turned by 90 degrees: 45 degrees plus a quarter turn.
    assert min(abs(wrap(math.radians(s.grasp.heading_deg - 45.0 - 90.0 * k))) for k in range(4)) < 1e-6
    xy0 = s._block_xy0
    assert xy0 is not None
    shifted = s._next_shift()
    assert shifted is not None and shifted - xy0 == pytest.approx(along(s, 0.035355 / 2))
    # The next run starts across the short side again.
    s.start(run_info(), obs(dict(SPEC.home)))
    assert s.gap_open_m == pytest.approx(0.041) and s.cmd_open == pytest.approx(KIN.gripper_for_gap(0.041))


def test_the_sideways_retarget_keeps_the_library_grasp_and_shifts_by_half_its_width():
    """The pose 1 cm off toward the fixed jaw: the moving tip lands on the block, the run rises, re-composes
    the library at the shifted point and takes the same grasp half its width toward the moving jaw."""
    pose = seen((XY[0] - 0.010, XY[1]), source="kamery", age_s=0.2, yaw_deg=90.0)
    aims: list[tuple[float, float]] = []
    indices: list[int] = []

    class Aims(GraspStrategy):
        def _aim(self, xy: np.ndarray) -> None:
            super()._aim(xy)
            aims.append((float(xy[0]), float(xy[1])))
            indices.append(self.grasp.library_index)

    s = started(pose, strategy=Aims, grasp_source="library", carry_s=0.5)
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block_at())

    assert answers[-1].success, answers[-1].note
    assert s.grasp.heading_deg == pytest.approx(0.0, abs=0.5)
    assert s.grasp.straddle_stalls == 1 and s.status.attempt == 2
    assert indices == [0, 0] and s.grasp.library_index == 0
    assert np.asarray(aims[1]) - np.asarray(aims[0]) == pytest.approx(along(s, 0.0125), abs=1e-6)
