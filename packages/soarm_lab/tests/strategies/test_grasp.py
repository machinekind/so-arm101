"""Test grasp phases and recovery with a fake supervisor, arm and jaw.

The arm stalls at the block, and a held block follows the fingertips. Jaw models simulate
motion, delay, creep and failures. Phase hooks can drop the block. `Script` supplies the
tracker's poses around a drop. Twin integration and camera tests cover physics and perception.
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

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.grasp_library import GraspLibrary, save_library  # noqa: E402
from soarm_lab.kinematics import IKSolution, RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import (  # noqa: E402
    IN_HAND,
    JAW_REACH_M,
    MIN_BITE_M,
    MIN_GRASP_Z,
    MIN_HOVER_CLEAR_M,
    PROBE_BELOW_M,
    SEEN,
    STACK_CLEAR_M,
    STALE,
    GraspSettings,
    GraspStatus,
    GraspStrategy,
    Sighting,
    grasp_headings,
    heading,
    top_down,
    wrap,
)
from soarm_lab.strategies.jaw import JawReading  # noqa: E402
from soarm_lab.strategies.touch import MIN_TOUCH_Z  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Block position inside the downward reach band [m, base frame].
XY = (0.21, 0.02)
NAN = float("nan")
#: The cube's half sizes [m] and the gripper command whose tip gap is its side [app units].
CUBE = block("cube").half_size
CUBE_UNITS = KIN.gripper_for_gap(2 * CUBE[1])
assert SPEC.gripper_range_q is not None
#: The gripper joint's angle per app unit [rad].
PER_UNIT = (SPEC.gripper_range_q[1] - SPEC.gripper_range_q[0]) / 100
#: Maps time [s], supervisor gripper command [app units] and straddle state
#: to the measured gripper command [app units].
Jaw = Callable[[float, float, bool], float]
#: Fixed poses or poses supplied by observation time [s].
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
    """Return a detector pose without orientation, or a box-fit pose with yaw."""
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
    """Return joint speed limits [deg/s] under the panel cap, with an optional gripper override."""
    return {name: (value if gripper is None else gripper) if name == SPEC.gripper else min(value, cap_deg_s)
            for name, value in SPEC.max_vel.items()}


def tcp(joints: Mapping[str, float]) -> np.ndarray:
    return KIN.tcp(joints)[:3, 3]


def runs(phases: list[str]) -> list[str]:
    """Return phases in order with consecutive repeats removed."""
    return [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]


def contains(seq: list[str], part: list[str]) -> bool:
    """Whether `part` is a subsequence of `seq`."""
    it = iter(seq)
    return all(p in it for p in part)


@dataclass
class Block:
    """A block on the table with the given dimensions and yaw."""

    xy: np.ndarray
    half: tuple[float, float, float]
    yaw: float = 0.0
    #: A gripped block follows the fingertips and no longer stalls the arm.
    gripped: bool = False
    #: False when the block has been removed.
    present: bool = True
    path: list[np.ndarray] = field(default_factory=list)
    #: First observation time after the last drop [s]. NaN until a drop occurs.
    dropped_t: float = NAN

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def local(self, point: np.ndarray) -> np.ndarray:
        """`point`'s x and y in the block's frame [m]."""
        return rot_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    def stalls(self, joints: Mapping[str, float]) -> bool:
        """Whether a fingertip would enter the block at `joints`."""
        if not self.present or self.gripped:
            return False
        sole = KIN.tcp(joints)[2, 3] - KIN.jaw_reach(joints[SPEC.gripper])
        if sole >= self.height:
            return False
        return any(bool(np.all(np.abs(self.local(tip)) < self.half[:2])) for tip in KIN.fingertips(joints))

    def between(self, joints: Mapping[str, float]) -> bool:
        """Whether the tips straddle the block below its top.

        They must lie on opposite sides along one block axis and within its extent along the other."""
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
    """A gripped block that stalls the arm above `rise_m` from the initial grip height."""

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


#: Stage fields are drop delay [s], source, xy [m, base frame], initial age [s] and height [m].
#: A None source hides the pose. None xy uses the block's position.
Stage = tuple[float, str | None, tuple[float, float] | None, float, float]


def stage(delay_s: float, source: str | None, xy: tuple[float, float] | None = None, age_s0: float = 0.0,
          z: float = 0.0) -> Stage:
    return (delay_s, source, xy, age_s0, z)


@dataclass
class Script:
    """Supply tracker poses before and after a drop.

    Use `before` until a stage starts after `Block.dropped_t`. Then use the latest started
    stage and increase its pose age with elapsed time.
    """

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
    """Return a hook that drops the block at `where` on its first `times` calls."""
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
    """Run up to `ticks` ticks, stopping on `done`.

    The supervisor limits arm speed to `cap_deg_s` [deg/s], or follows at once for None.
    The gripper follows its ramped target. The arm follows with lag and a TCP height offset,
    stalling at `block`. `jaw` supplies measured opening, defaulting to the command bounded
    by block width. A held block follows the fingertip midpoint until release or a drop.
    `slip_z` triggers a drop above that TCP height only before the first drop.

    `objects` supplies fixed poses or poses by observation time. `on_phase` hooks run after
    the first tick entering a phase. The next observation records a drop in `block.dropped_t`.
    `commands`, `measured` and `trace` collect commands, measured joints and timestamped grasp
    states. `command` supplies the initial supervisor command, defaulting to measured joints.
    Return actions, phases, final joints and final time.
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
        # Keep the previous jaw angle until the contact model runs.
        arm[g] = joints[g]
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
    """Advance each arm joint toward its target by at most `step_deg` [deg]."""
    out = dict(command)
    for name, value in target.items():
        if step_deg is None or name == SPEC.gripper:
            out[name] = float(value)
        else:
            out[name] = command[name] + float(np.clip(value - command[name], -step_deg, step_deg))
    return out


def held(s: GraspStrategy, command: dict[str, float], offset_z: float) -> dict[str, float]:
    """Return the settled joints with TCP offset `offset_z` [m]. Negative offsets model sag."""
    if offset_z == 0.0:
        return dict(command)
    point = tcp(command) + np.array([0.0, 0.0, offset_z])
    return dict(s.kin.ik_step(point, seed=command, tol=1e-5).joints, gripper=command[SPEC.gripper])


def servo(measured: dict[str, float], settled: dict[str, float], share: float) -> dict[str, float]:
    """Move each measured joint `share` of the way toward its settled position."""
    return {name: measured[name] + share * (value - measured[name]) for name, value in settled.items()}


def along(s: GraspStrategy, metres: float) -> np.ndarray:
    """`metres` along the run's closing heading [m, base frame x and y]."""
    return metres * np.array([math.cos(math.radians(s.grasp.heading_deg)),
                              math.sin(math.radians(s.grasp.heading_deg))])


def first_heading() -> float:
    """Return the rest pose's closing heading [rad], used for poses without orientation."""
    return heading(KIN, dict(SPEC.home, gripper=0.0))


def offset(xy: tuple[float, float], metres: float) -> tuple[float, float]:
    """`xy` moved `metres` along the rest pose's closing heading."""
    psi = first_heading()
    return (xy[0] + metres * math.cos(psi), xy[1] + metres * math.sin(psi))


def grasped(**settings) -> tuple[GraspStrategy, list[Action], list[str], dict[str, float], Block]:
    """Run a complete cube grasp at `XY` with a short carry unless overridden."""
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

    assert s.grasp_settings.probe is False
    assert s.config.touch_z == pytest.approx(2 * CUBE[2] - PROBE_BELOW_M + JAW_REACH_M, abs=1e-4)
    assert s.config.touch_z == pytest.approx(0.0256, abs=1e-4)
    assert s.config.hover_m == 0.04 and s.time_limit_s == 120.0 and s.label == "cube"
    s.start(run_info(), obs(dict(SPEC.home)))
    assert s.config.touch_z == pytest.approx(0.0256, abs=1e-4)
    # A low block's probe stops at the touch floor.
    cuboid = GraspStrategy(GraspSettings(block="cuboid"))
    assert cuboid.config.touch_z == MIN_TOUCH_Z and cuboid.label == "cuboid"
    assert s.grasp_settings.touch_z == 0.0
    assert GraspStrategy(GraspSettings(touch_z=0.03, probe=True)).config.touch_z == 0.03
    # Without the probe, the block dimensions set the hover reference even with an explicit touch_z.
    assert GraspStrategy(GraspSettings(touch_z=0.03)).config.touch_z == pytest.approx(0.0256, abs=1e-4)
    assert GraspStrategy(GraspSettings(label="Red  Cube")).label == "red cube"
    with pytest.raises(ValueError, match="grasp: block must be one of cube, cuboid"):
        GraspStrategy(GraspSettings(block="ball"))


@pytest.mark.parametrize("settings, match", [
    ({"open_margin_m": 0.0}, "grasp: open_margin_m must be positive"),
    ({"squeeze_m": 0.0}, "grasp: squeeze_m must be positive"),
    ({"lift_m": 0.0}, "grasp: lift_m must be positive"),
    ({"hold_s": NAN}, "grasp: hold_s must be positive"),
    ({"grasp_depth": 0.0}, r"grasp: grasp_depth must lie within \(0, 1\]"),
    ({"grasp_depth": 1.5}, r"grasp: grasp_depth must lie within \(0, 1\]"),
    ({"max_attempts": 0}, "grasp: max_attempts must be at least 1"),
    ({"touch_z": 0.01, "probe": True}, r"grasp: touch_z must be zero \(from the block\) or at least 0.02 m"),
    ({"close_jaw": False, "probe": True}, "grasp: close_jaw must be on"),
])
def test_settings_are_validated(settings: dict, match: str):
    with pytest.raises(ValueError, match=match):
        GraspStrategy(GraspSettings(**settings))


def test_without_the_probe_its_touch_z_and_close_jaw_are_not_checked():
    s = GraspStrategy(GraspSettings(touch_z=0.01, close_jaw=False))

    assert s.config.touch_z == pytest.approx(0.0256, abs=1e-4)


@pytest.mark.parametrize("settings, match", [
    ({"open_margin_m": 0.04}, "grasp: jaw gap 0.1090 m is outside the jaw travel"),
    ({"squeeze_m": 0.025}, "grasp: jaw gap 0.0040 m is outside the jaw travel"),
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
    s = started(seen(), probe=True)
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
        # The closed tips' midpoint lies over the block centre.
        assert np.mean(KIN.fingertips(a.target), axis=0)[:2] == pytest.approx(np.asarray(XY), abs=1.5e-3)


@pytest.mark.parametrize("probe", [False, True], ids=["default", "probe"])
def test_the_whole_grasp_on_a_cube_in_free_space(probe: bool):
    start = KIN.from_q(started().q_cmd)
    s, answers, phases, joints, block = grasped(probe=probe)

    probed = ["descend", "touched", "rise"] if probe else []
    assert runs(phases) == ["approach", *probed, "open", "straddle", "close", "lift_off", "check", "carry",
                            "done"]
    if not probe:
        # Without a closed-jaw descent, the hand stays at the hover until the open jaw straddles.
        hover_z = s.config.touch_z + s.config.hover_m
        assert min(tcp(a.target)[2] for a, p in zip(answers, phases, strict=True)
                   if p in ("approach", "open")) >= hover_z - 1e-3
        assert all(a.target["gripper"] == pytest.approx(s.cmd_open)
                   for a, p in zip(answers, phases, strict=True) if p == "straddle")
    last = answers[-1]
    assert last.done and last.success and s.status.success
    assert last.note.startswith("blok w szczekach, podniesiony o")
    assert ", niesiony 0.5 s (szczeka" in last.note
    assert int(last.note.split("podniesiony o ")[1].split(" mm")[0]) == pytest.approx(40, abs=5)
    assert s.grasp.verdict == "lifted" and s.status.attempt == 1
    assert phases.count("carry") == round(0.5 * s.hz) and s.grasp.carried_s == pytest.approx(0.5)
    assert s.grasp.drops == 0 and s.grasp.creeps == 0 and s.grasp.elsewhere is None
    if probe:
        assert s.grasp.top_m == pytest.approx(2 * CUBE[2], abs=0.002)
    else:
        assert math.isnan(s.grasp.top_m)
    top = s.grasp.top_m if probe else 2 * CUBE[2]
    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(max(0.012, top - CUBE[2] + 0.003), abs=1e-3)
    grips = [start["gripper"]] + [a.target["gripper"] for a in answers]
    assert np.max(np.abs(np.diff(grips))) <= 12.0 + 1e-6
    assert answers[-1].target["gripper"] == pytest.approx(KIN.gripper_for_gap(2 * CUBE[1] - 0.008), abs=0.1)
    assert joints["gripper"] == pytest.approx(CUBE_UNITS, abs=0.1)
    assert all(KIN.self_contacts(a.target) == set() for a in answers)
    # The moving jaw pushes the block `open_margin_m` along -u onto the fixed tip.
    assert block.gripped
    assert block.xy - XY == pytest.approx(-along(s, 0.008), abs=1.5e-3)


def test_the_open_jaw_clears_the_block_by_the_margin():
    s, answers, phases, *_ = grasped()
    bottom = [a for a, p in zip(answers, phases, strict=True) if p == "close"][0]
    fixed, moving = KIN.fingertips(dict(bottom.target, gripper=s.cmd_open))
    u = along(s, 1.0)
    v = np.array([-u[1], u[0]])

    for tip, sign in ((fixed, -1), (moving, 1)):
        assert (tip[:2] - XY) @ u == pytest.approx(sign * (CUBE[1] + 0.008), abs=1e-3)
        assert abs((tip[:2] - XY) @ v) < 1e-3


# ------------------------------------------------------------ the close
def late_jaw(open_units: float, delay_s: float) -> Jaw:
    """Model a jaw that waits `delay_s` after a close command before moving."""
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
    """Model a jaw that opens but cannot close again."""
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
        assert answers[-1].note == "szczeka nie ruszyla po rozkazie zamkniecia (45 mm)"
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


class Aims(GraspStrategy):
    """Records every block centre the run aims at."""

    def __init__(self, settings: GraspSettings | None = None):
        super().__init__(settings)
        self.aims: list[tuple[float, float]] = []

    def _aim(self, xy: np.ndarray) -> None:
        super()._aim(xy)
        self.aims.append((float(xy[0]), float(xy[1])))


@pytest.mark.parametrize("sign", [-1, 1], ids=["moving tip lands", "fixed tip lands"])
def test_without_the_probe_a_jaw_tip_on_the_block_straddles_again_half_the_short_side_away(sign: int):
    """After a tip stall, the hand rises and retries toward the moving jaw, then on the other side."""
    pose = seen(offset(XY, sign * 0.010))
    s = started(pose, strategy=Aims)
    block = block_at()
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block)
    assert isinstance(s, Aims)

    assert answers[-1].success and s.grasp.verdict == "lifted", answers[-1].note
    assert not {"descend", "touched"} & set(phases)
    first = np.asarray(offset(XY, sign * 0.010))
    assert np.asarray(s.aims[1]) == pytest.approx(first + along(s, CUBE[1]), abs=1e-6)
    if sign < 0:
        assert s.status.attempt == 2 and s.grasp.straddle_stalls == 1
        assert contains(runs(phases), ["straddle", "straddled", "rise", "approach", "open", "straddle",
                                       "close", "lift_off", "check", "done"])
    else:
        # The fixed tip lands again beside the first aim, so the third straddle is on the other side.
        assert s.status.attempt == 3 and s.grasp.straddle_stalls == 2
        assert np.asarray(s.aims[2]) == pytest.approx(first - along(s, CUBE[1]), abs=1e-6)
    assert np.linalg.norm(block.path[0] - XY) < 1e-9


def test_without_the_probe_an_empty_close_after_a_shift_tries_the_other_side():
    """An empty close after a sideways retry tries the other side of the original aim."""
    pose = seen(offset(XY, 0.025))
    s = started(pose, strategy=Aims, max_attempts=5)
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 6000, block=block_at())
    assert isinstance(s, Aims)

    assert answers[-1].success and s.grasp.verdict == "lifted", answers[-1].note
    first = np.asarray(offset(XY, 0.025))
    shifts = [float((np.asarray(aim) - first) @ along(s, 1.0)) / CUBE[1] for aim in s.aims]
    assert shifts == pytest.approx([0.0, 1.0, -1.0, 2.0, -2.0], abs=1e-6)
    assert s.status.attempt == 5 and s.grasp.straddle_stalls == 2 and s.grasp.empty_closes == 2
    assert any(a.note == "chwyt pusty obok - probuje z drugiej strony" for a in answers)


def test_without_the_probe_an_empty_close_after_a_shift_ends_once_attempts_run_out():
    pose = seen(offset(XY, 0.025))
    s = started(pose, max_attempts=2)
    answers, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block_at())

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note == "wyczerpane proby chwytu (2): chwyt pusty"
    assert s.grasp.verdict == "empty" and s.status.attempt == 2


@pytest.mark.parametrize("sign", [-1, 1], ids=["moving tip lands", "fixed tip lands"])
def test_a_jaw_tip_on_the_block_retargets_sideways_by_half_the_short_side(sign: int):
    """With the probe, a tip stall retries toward the moving jaw first.

    A miss there retries on the opposite side of the original aim."""
    pose = seen(offset(XY, sign * 0.010))
    s = started(pose, strategy=Aims, probe=True)
    block = block_at()
    answers, phases, *_ = follow(s, dict(SPEC.home), pose, 3000, block=block)
    assert isinstance(s, Aims)

    assert answers[-1].success and s.grasp.verdict == "lifted", answers[-1].note
    assert s.grasp.straddle_stalls == 1
    first = np.asarray(offset(XY, sign * 0.010))
    if sign < 0:
        assert s.status.attempt == 2
        assert contains(runs(phases), ["straddle", "straddled", "rise", "approach", "descend", "touched",
                                       "rise", "open", "straddle", "close", "lift_off", "check", "done"])
        assert np.asarray(s.aims[1]) == pytest.approx(first + along(s, CUBE[1]), abs=1e-6)
    else:
        assert s.status.attempt == 3
        # A miss enters rise immediately, so it has no touched tick.
        assert runs(phases).count("descend") == 3 and runs(phases).count("touched") == 2
        assert np.asarray(s.aims[1]) == pytest.approx(first + along(s, CUBE[1]), abs=1e-6)
        assert np.asarray(s.aims[2]) == pytest.approx(first - along(s, CUBE[1]), abs=1e-6)
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
    # The timeout includes planned travel and time for the arm to follow.
    assert phases.count("lift_off") / s.hz == pytest.approx(0.04 / 0.10 + 3.0, abs=0.06)
    assert answers[-1].target["gripper"] == pytest.approx(s.cmd_closed)
    assert s.grasp.verdict == "held" and block.gripped
    assert joints["gripper"] == pytest.approx(CUBE_UNITS, abs=0.1)


def test_a_speed_cap_lowered_during_the_lift_lengthens_its_time_limit():
    """A slower supervisor command extends the lift timeout after targets have reached the lift point."""
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
    assert answers[-1].note == "szczeka nie otworzyla sie w 3 s (21 mm zamiast 45 mm)"


# ------------------------------------------------------------ carrying and recovery
UPUSZCZONA = "ostatnie widziane (upuszczona)"
#: Landing position for most recovery tests [m, base frame].
LANDS = (XY[0] + 0.03, XY[1])


def carried(s: GraspStrategy) -> Callable[[float], Mapping[str, ObjectPose]]:
    """Return the detector pose until closing, then the tracker's in-hand pose."""
    closed: list[bool] = []

    def objects(t: float) -> Mapping[str, ObjectPose]:
        if closed or s.status.phase in ("close", "lift_off", "check", "carry"):
            closed.append(True)
            return seen(source="w dloni", age_s=0.0, z=0.05)
        return seen()

    return objects


@dataclass
class Knocked:
    """A drop-test run with per-tick actions and observations, plus targets recorded at each knock."""

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
        """Return the vertical recovery point at least as high as the hover."""
        z0 = self.at_knock[0][0]
        hover = self.s.config.touch_z + self.s.config.hover_m
        return np.array([z0[0], z0[1], max(hover, z0[2])])


def knocked(stages: list[Stage], phase: str = "carry", where: tuple[float, float] = LANDS, times: int = 1,
            before: Objects | None = None, block: Block | None = None, jaw: Jaw | None = None,
            strategy: type[GraspStrategy] = GraspStrategy, ticks: int = 4000, **settings) -> Knocked:
    """Run a cube grasp with drops at `where` on the first `times` entries into `phase`.

    `stages` supplies poses after each drop. `before` supplies earlier poses, defaulting to `carried`.
    """
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
    """Report the tracker's drop guess, followed by a camera sighting at `xy`."""
    return [stage(0.0, UPUSZCZONA, XY), stage(delay_s, "kamery", xy, 0.1)]


def test_carry_s_zero_ends_the_run_at_the_verified_lift():
    s, answers, phases, *_ = grasped(carry_s=0.0)

    assert answers[-1].success and phases[-2] == "check" and "carry" not in phases
    note = answers[-1].note
    assert note.startswith("blok w szczekach, podniesiony o") and "niesiony 0.0 s" in note
    assert s.grasp.carried_s == 0.0 and s.grasp.verdict == "lifted"


@pytest.mark.parametrize("lift_m", [0.04, 0.065], ids=["lift below the hover", "lift above the hover"])
@pytest.mark.parametrize("phase", ["check", "carry"])
def test_a_block_dropped_mid_carry_and_seen_on_the_table_is_grasped_again(phase: str, lift_m: float):
    """The default regrasp opens at the hover and straddles without probing."""
    run = knocked(seen_again(), phase=phase, lift_m=lift_m)
    s, answers, phases, trace = run.s, run.answers, run.phases, run.trace

    assert answers[-1].success, answers[-1].note
    drop = run.drop
    assert runs(phases[drop:]) == ["rise", "recover", "approach", "open", "straddle", "close", "lift_off",
                                   "check", "carry", "done"]
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1 and s.status.attempt == 2
    # Regrasp uses the camera sighting.
    assert s.grasp.block_xy == pytest.approx(LANDS, abs=1e-9)
    # Recovery rises vertically without lowering an arm already above the hover.
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
    """With the probe on, the jaw closes before the regrasp's measured-top descent."""
    run = knocked(seen_again(), probe=True)
    s, answers = run.s, run.answers
    g = SPEC.joints.index("gripper")

    drop, recover = run.drop, run.first("recover", run.drop)
    grips = [a.target["gripper"] for a in answers[drop - 1:recover + 1]]
    assert grips[0] == pytest.approx(s.cmd_closed)
    assert all(-1e-9 <= b - a <= 12.0 + 1e-6 for a, b in zip(grips, grips[1:], strict=False))
    assert KIN.to_q(answers[recover].target)[g] == pytest.approx(s._recover_open_q)
    assert s._recover_open_q == pytest.approx(SPEC.jaw.open + 2 * SPEC.jaw.margin)
    # The empty jaw follows the command.
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
    # Derive the slip height from the nominal grasp point.
    slip_z = max(0.012, 2 * CUBE[2] - CUBE[2] + 0.003) + 0.02
    script = Script(block, carried(s), [stage(0.0, UPUSZCZONA, XY), stage(0.5, "kamery", None, 0.1)])
    answers, phases, *_ = follow(s, dict(SPEC.home), script, 4000, block=block, slip_z=slip_z)

    assert answers[-1].success, answers[-1].note
    assert contains(runs(phases), ["lift_off", "rise", "recover", "approach", "open", "lift_off", "check",
                                   "carry", "done"])
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1
    assert not any("podczas podnoszenia" in a.note for a in answers)


def test_a_block_that_reappears_as_a_late_last_seen_pose_is_grasped_again():
    """A delayed plain last-seen pose captured after the drop can guide a regrasp."""
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
    # Republishing an old capture must preserve its age.
    before = (lambda t: seen(age_s=8.0 + t)) if case == "before the drop" else None
    run = knocked(stages, where=XY, before=before, wait_s=8.0, max_pose_age_s=60.0)
    s, answers, trace = run.s, run.answers, run.trace
    g = SPEC.joints.index("gripper")

    assert answers[-1].done and not answers[-1].success
    assert answers[-1].note == f"blok zabrany ze szczeki: {why} od 8 s - czekam na czlowieka"
    assert s.grasp.verdict == "taken" and s.grasp.recoveries == 0
    assert 0.0 <= trace[-1][0] - (s.grasp.dropped_t + 8.0) < 1 / s.hz
    assert "approach" not in run.phases[run.drop:]
    # Stop at the safe point with the jaw open.
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
        # Use the fingertip midpoint as the drop origin.
        assert far == pytest.approx(0.2, abs=0.015) and far > s.grasp_settings.recover_m
        assert note == f"blok zabrany: lezy {far * 100:.0f} cm od miejsca upadku - czekam na czlowieka"
    # An unreachable sighting keeps the old aim and safe pose.
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
    """Model small jaw closures that look still per tick but together exceed the creep threshold."""
    calls: list[float] = []

    def model(t: float, command: float, between: bool) -> float:
        if not between:
            return command
        if s.status.phase == phase:
            calls.append(t)
        # The model supplies the next tick's measurement.
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
    # The jaw keeps squeezing throughout the creep.
    assert KIN.to_q({"gripper": CUBE_UNITS})[-1] - 0.035 - s._q(s.cmd_closed) > SPEC.jaw.margin
    if phase == "check":
        # Creep restarts the hold streak.
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
    # Release lets the measured jaw follow the opening command.
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
    assert answers[-1].note == "szczeka nie otworzyla sie po oddaniu bloku w 3 s (29 mm zamiast 45 mm)"
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
    # The carry already observed the empty reading.
    assert once and any(not grasp.holding for t, grasp in trace if t > once[0] - 1e-9)


def test_the_watch_reads_a_tick_once_when_the_phase_changes_within_it():
    """A lift-to-check transition calls `carrying` twice, but counts an empty reading once."""
    _, _, reference, *_ = grasped()
    s = started(seen(), carry_s=0.5)
    _, phases, joints, t = follow(s, dict(SPEC.home), seen(), reference.index("lift_off") + 1,
                                  block=block_at())
    assert phases[-1] == "lift_off"
    empty = JawReading(q=0.0, cmd=0.0, still=True, squeezing=False, holding=False, open=False, crept=False)
    tick = obs(joints, t=t, objects=seen())

    # The next tick runs both lift and check handlers.
    s.status.step += 1
    assert s.carrying(tick, empty) is None and s.carrying(tick, empty) is None
    assert s.grasp.drops == 0 and s._empty_ticks == 1
    s.status.step += 1
    assert s.carrying(tick, empty) is not None
    assert s.grasp.drops == 1 and s.status.phase == "rise"


@pytest.mark.parametrize("probe", [False, True], ids=["default", "probe"])
def test_a_regrasp_that_misses_ends_as_touch_does(probe: bool):
    """A wrong camera pose ends in an empty close by default, or a probe miss with `probe: true`."""
    run = knocked(seen_again((XY[0] + 0.03, XY[1])), where=(XY[0] + 0.08, XY[1]), probe=probe)

    assert run.answers[-1].note == ("brak kontaktu w punkcie dotyku - obiektu tam nie ma" if probe
                                    else "chwyt pusty - blok przesuniety albo obok szczek")
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
    # The time limit covers two grasps and carries plus a recovery wait.
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
    assert float(np.linalg.norm(np.array([*under, CUBE[2]]) - jaw)) < 0.06
    assert kind("symulacja", under, z=CUBE[2], squeezing=True) == IN_HAND
    assert kind("symulacja", under, z=CUBE[2], squeezing=False) == SEEN
    assert kind("symulacja", (under[0] + 0.2, under[1]), z=CUBE[2], squeezing=True) == SEEN
    assert kind("", under, z=CUBE[2]) == SEEN
    # A block resting under the hand is empty when the jaw is not squeezing.
    cuboid = started(block="cuboid", label="cube")
    assert kind("symulacja", under, z=block("cuboid").half_size[0], squeezing=False, strategy=cuboid) == SEEN
    assert kind("", strategy=started(simulated=False)) == STALE
    assert sighting("kamery", position=np.array([NAN, 0.0, 0.0])) is None
    assert sighting("kamery", position=np.zeros(2)) is None
    found = sighting("kamery", age_s=1.5)
    assert found is not None
    assert found.seen_t == pytest.approx(3.5) and found.xy == (0.3, 0.1) and found.source == "kamery"
    assert found.rotation == pytest.approx(rot_z(0.3))
    assert s._sighting(obs(joints, t=5.0, objects={}), False) is None


# ------------------------------------------------------------ headings
@pytest.mark.parametrize("kind, yaw_deg", [("cube", 20.0), ("cuboid", 35.0), ("cuboid", 125.0),
                                           ("cylinder", -60.0)])
def test_a_pose_rotation_puts_the_jaws_across_the_short_side(kind: str, yaw_deg: float):
    pose = seen(label=kind, yaw_deg=yaw_deg, source="kamery", age_s=0.2)
    s = started(pose, block=kind)
    s.act(obs(dict(SPEC.home), t=0.05, objects=pose))
    period = math.degrees(block(kind).yaw_period)
    now = math.degrees(first_heading())
    candidates = [math.degrees(wrap(math.radians(yaw_deg + 90 + k * period)))
                  for k in range(int(360 / period))]
    nearest = min(candidates, key=lambda c: abs(math.degrees(wrap(math.radians(c - now)))))

    assert s.status.phase == "approach", s.status.stopped_because
    assert s.grasp.heading_deg == pytest.approx(nearest, abs=0.5)
    assert s.gap_open_m == pytest.approx(2 * block(kind).half_size[1] + 2 * GraspSettings().open_margin_m)
    if kind == "cuboid" and yaw_deg == 125.0:
        assert s.grasp.heading_deg == pytest.approx(35.0, abs=0.5)
    assert len(grasp_headings(rot_z(math.radians(yaw_deg)), block(kind).yaw_period, 0.0)) == (
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

    s = started(seen(), strategy=Misses, probe=True)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500)

    assert answers[-1].note == "brak kontaktu w punkcie dotyku - obiektu tam nie ma"
    assert not answers[-1].success
    assert "rise" not in phases
    assert misses == [("missed", "brak kontaktu w punkcie dotyku - obiektu tam nie ma")]


def test_without_the_probe_a_missing_block_ends_with_an_empty_close():
    s = started(seen(), strategy=Seams)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500)
    assert isinstance(s, Seams)

    assert runs(phases)[-5:] == ["approach", "open", "straddle", "close", "done"]
    assert answers[-1].note == "chwyt pusty - blok przesuniety albo obok szczek" and not answers[-1].success
    assert s.status.attempt == 1 and ("after_grasp_failed", "empty") in s.calls
    assert not any(c[0] in ("before_descent", "after_miss") for c in s.calls)


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
@pytest.mark.parametrize("probe", [False, True], ids=["default", "probe"])
def test_targets_respect_the_supervisors_speed_limits_including_the_gripper(probe: bool):
    max_vel = capped(20.0, gripper=100.0)
    s = started(seen(), max_vel=max_vel, probe=probe)
    start = KIN.from_q(s.q_cmd)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 6000, block=block_at(), cap_deg_s=20.0)

    assert answers[-1].success, answers[-1].note
    targets = [start] + [a.target for a in answers]
    # A stall holds the measured pose behind the command. The supervisor still limits the command's speed.
    stall = phases.index("touched") if probe else -1
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


@pytest.mark.parametrize("probe", [False, True], ids=["default", "probe"])
def test_the_seams_are_called(probe: bool):
    s = started(seen(), strategy=Seams, probe=probe)
    answers, phases, *_ = follow(s, dict(SPEC.home), seen(), 1500, block=block_at())
    assert isinstance(s, Seams)

    descents = [c for c in s.calls if c[0] == "before_descent"]
    assert descents == ([("before_descent", 1, True)] if probe else [])
    carrying = [c for c in s.calls if c[0] == "carrying"]
    # Include phase transitions and the final tick.
    assert len(carrying) == phases.count("lift_off") + phases.count("check") + phases.count("carry") + 3
    assert all(c[2] for c in carrying) and {c[1] for c in carrying} == {"lift_off", "check", "carry"}

    pose = seen(offset(XY, -0.010))
    s = started(pose, strategy=Seams, probe=probe)
    follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    assert isinstance(s, Seams)
    assert ("after_grasp_failed", "tip") in s.calls
    assert [c[1] for c in s.calls if c[0] == "before_descent"] == ([1, 2] if probe else [])

    pose = seen(offset(XY, 0.010))
    s = started(pose, strategy=Seams, probe=probe)
    follow(s, dict(SPEC.home), pose, 3000, block=block_at())
    assert isinstance(s, Seams)
    missed = ("after_miss", "missed", "brak kontaktu w punkcie dotyku - obiektu tam nie ma")
    assert [c for c in s.calls if c[0] == "after_miss"] == ([missed] if probe else [])

    s = started(seen(), strategy=Seams, probe=probe)
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
    s = started(seen(), strategy=LooksBeforeDescent, probe=True)
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
    # The plain run starts straddling when the look begins.
    assert looking[0] == plain.index("straddle")
    assert s.asked[0][1] is True
    assert len(s.asked) == len(looking) + 1 and looking == list(range(looking[0], looking[-1] + 1))
    # The opening timeout waits while the look holds the arm.
    assert len(looking) / s.hz == pytest.approx(LooksBeforeStraddle.LOOK_S, abs=0.06)
    assert {phases[k] for k in looking} == {"open"}
    assert all(answers[k].target == answers[looking[0]].target for k in looking)
    # Returning None starts the straddle on the same tick.
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
    # The hook finishes before the arm settles at the hover.
    assert s.asked == ([(1, False)] if shift == 0.0 else [(1, False), (2, False)])
    assert runs(phases).count("open") == len(s.asked)


def test_the_notes_per_phase():
    s, answers, phases, *_ = grasped(probe=True)
    notes = {p: [a.note for a, q in zip(answers, phases, strict=True) if q == p] for p in set(phases)}
    assert any(n.startswith("w gore po dotyku") for n in notes["rise"] + notes["touched"])

    s, answers, phases, *_ = grasped()
    notes = {p: [a.note for a, q in zip(answers, phases, strict=True) if q == p] for p in set(phases)}
    assert any(n.startswith("nad obiekt") for n in notes["approach"])
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
#: A nonidentity cube rotation keeps the fake block axis-aligned while enabling library composition.
YAWED = seen(source="kamery", age_s=0.2, yaw_deg=90.0)
DOWN = (0.0, 0.0, -1.0)
ACROSS_Y = (0.0, 1.0, 0.0)


def grasp(pad: tuple[float, float, float], approach: tuple[float, float, float] = DOWN,
          closing: tuple[float, float, float] = ACROSS_Y, width: float = 2 * CUBE[1],
          score: float = 0.9) -> dict:
    """Build one grasp from a pad point and axes in the cube frame."""
    return {"position": list(pad), "approach": list(approach), "closing": list(closing), "width_m": width,
            "score": score}


def library_dir(tmp_path, grasps: list[dict]) -> str:
    """Write a cube library of `grasps` to a test folder."""
    library = GraspLibrary.from_dict({"format": "soarm-lab-grasps", "version": 1, "kind": "cube",
                                      "half_size": list(CUBE), "generator": {"name": "hand"},
                                      "grasps": grasps})
    save_library(library, tmp_path / "cube.json")
    return str(tmp_path)


def aimed(objects: dict[str, ObjectPose], **settings) -> tuple[GraspStrategy, Action]:
    """Start library mode and process one observation to latch and aim at the pose."""
    s = started(objects, grasp_source="library", **settings)
    return s, s.act(obs(dict(SPEC.home), t=0.05, objects=objects))


def test_the_library_source_is_validated_at_construction(tmp_path):
    with pytest.raises(ValueError, match="grasp: no grasp library for 'cuboid'"):
        GraspStrategy(GraspSettings(grasp_source="library", block="cuboid"))
    with pytest.raises(ValueError, match="grasp: grasp_source must be one of geometry, library, is 'x'"):
        GraspStrategy(GraspSettings(grasp_source="x"))
    with pytest.raises(ValueError, match=r"grasp: min_score must lie within \[0, 1\], is 1.5"):
        GraspStrategy(GraspSettings(min_score=1.5))
    (tmp_path / "cube.json").write_text('{"format": "soarm-lab-grasps", "version": 2}')
    with pytest.raises(ValueError, match="grasp: not a grasp library of version 1"):
        GraspStrategy(GraspSettings(grasp_source="library", library_dir=str(tmp_path)))
    # Geometry mode ignores library paths.
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
    # Lower-scored centred entries duplicate the best entry's headings.
    assert grasp.library_index == 0 and grasp.library_score == 0.92
    assert grasp.candidates == (20 if min_score == 0.0 else 16)
    # Prefer the heading nearest the rest pose.
    assert grasp.heading_deg == pytest.approx(0.0, abs=0.5)
    assert min(abs(grasp.heading_deg - k * 90.0) for k in range(-2, 3)) < 0.5
    assert s.gap_open_m == pytest.approx(2 * CUBE[1] + 0.016)
    assert s.gap_closed_m == pytest.approx(2 * CUBE[1] - 0.008)
    assert grasp.depth_m == pytest.approx(CUBE[2]) and grasp.offset_xy == pytest.approx((0.0, 0.0))
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
    # Geometry mode omits the source note.
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

    # Startup width fits, but the selected candidate exceeds travel.
    assert 0.080 + 2 * s.grasp_settings.open_margin_m > s.kin.jaw_gap(100.0)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=YAWED))
    assert a.done and not a.success
    assert "zaden z 8 chwytow nie pasuje (szczeka 8)" in a.note


@pytest.mark.parametrize("probe", [False, True], ids=["nominal top", "probed top"])
def test_a_library_grasp_off_centre_moves_the_straddle_point_and_its_depth(tmp_path, probe: bool):
    folder = library_dir(tmp_path, [grasp((0.005, 0.0, 0.0085))])
    depth = CUBE[2] - 0.0085
    s = started(YAWED, grasp_source="library", library_dir=folder, carry_s=0.5, probe=probe)
    s.act(obs(dict(SPEC.home), t=0.05, objects=YAWED))
    grasp_ = s.grasp
    psi = math.radians(grasp_.heading_deg)
    centre = KIN.jaw_centre(s.cmd_open)

    assert math.hypot(*grasp_.offset_xy) == pytest.approx(0.005, abs=1e-6)
    assert grasp_.depth_m == pytest.approx(depth, abs=1e-9)
    u = np.array([math.cos(psi), math.sin(psi)])
    expected = np.asarray(XY) + np.asarray(grasp_.offset_xy) - centre[2] * u
    assert s._straddle_xy == pytest.approx(expected, abs=1e-3)
    if probe:
        # Offset the straddle while keeping the probe centred.
        assert s.status.touch_point is not None
        probe_xy = np.asarray(XY) + along(s, 0.0156)
        assert np.asarray(s.status.touch_point[:2]) == pytest.approx(probe_xy, abs=1e-3)
    assert depth - KIN.tip_swing(s.cmd_open) / 2 >= MIN_BITE_M and grasp_.skipped == ""

    answers, *_ = follow(s, dict(SPEC.home), YAWED, 1500, block=block_at())
    assert answers[-1].success, answers[-1].note
    assert grasp_.grasp_point is not None
    top = grasp_.top_m if probe else s.top_nominal
    expected_z = max(MIN_GRASP_Z, top - depth + centre[0])
    assert grasp_.grasp_point[2] == pytest.approx(expected_z, abs=1e-3)


def test_a_shallow_library_grasp_is_skipped_by_the_bite_rule(tmp_path):
    """A shallow midpoint leaves the trailing tip at the top edge, so the bite check skips it."""
    s, a = aimed(YAWED, library_dir=library_dir(tmp_path, [grasp((0.0, 0.0, CUBE[2] - 0.0015))]))

    assert a.done and not a.success
    assert a.note.startswith("biblioteka chwytow 'cube': zaden chwyt nie pasuje do szczeki")
    assert "zbyt plytko 4" in a.note and s.grasp.skipped == "zbyt plytko 4"


@pytest.mark.parametrize("probe", [False, True], ids=["nominal top", "probed top"])
def test_a_deep_library_grasp_is_clamped_to_the_minimum_grasp_height(tmp_path, probe: bool):
    folder = library_dir(tmp_path, [grasp((0.0, 0.0, -0.0105))])
    s = started(YAWED, grasp_source="library", library_dir=folder, carry_s=0.5, probe=probe)
    answers, phases, *_ = follow(s, dict(SPEC.home), YAWED, 1500, block=block_at())

    assert s.grasp.depth_m == pytest.approx(CUBE[2] + 0.0105)
    # Only the probe rises from the block before the straddle.
    assert ("rise" in phases) == probe and s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(MIN_GRASP_Z)


def aimed_at(z: float, kind: str = "cube") -> GraspStrategy:
    """Return a strategy that took a box-fit pose with its centre `z` [m] above the table."""
    objects = seen(source="kamery", age_s=0.1, z=z, label=kind, yaw_deg=10.0)
    s = started(objects, block=kind, carry_s=0.5)
    a = s.act(obs(dict(SPEC.home), t=0.05, objects=objects))
    assert not a.done, a.note
    return s


def test_a_pose_near_rest_height_counts_as_on_the_table():
    table, noisy = aimed_at(0.0), aimed_at(CUBE[2] + 0.005)

    assert table.base_z == noisy.base_z == 0.0 and noisy.top_nominal == pytest.approx(2 * CUBE[2])
    assert noisy.grasp.grasp_point == pytest.approx(table.grasp.grasp_point)
    assert noisy.config.touch_z == table.config.touch_z and noisy.config.hover_m == 0.04


def test_a_cube_on_a_cube_is_grasped_at_its_own_height_under_a_hover_the_arm_reaches():
    s = aimed_at(3 * CUBE[2])
    top = 4 * CUBE[2]
    centre = KIN.jaw_centre(s.cmd_open)
    reach = KIN.jaw_reach(s.cmd_open)

    assert s.base_z == pytest.approx(2 * CUBE[2]) and s.top_nominal == pytest.approx(top)
    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(top - CUBE[2] + centre[0])
    # The set hover is out of reach this high, so it comes down but still clears the cube.
    hover = s.straddle_hover[2]
    assert s.config.hover_m < 0.04 and hover - reach >= top + MIN_HOVER_CLEAR_M - 1e-9
    assert s.config.touch_z == pytest.approx(top - PROBE_BELOW_M + KIN.jaw_reach(s.cmd_probe))
    assert s._lift_m <= 0.04 and s.hover_point[2] == pytest.approx(hover)


def test_on_a_cube_a_flat_cuboids_grasp_stays_clear_of_the_cube_and_still_bites():
    s = aimed_at(2 * CUBE[2] + block("cuboid").half_size[2], kind="cuboid")
    base = 2 * CUBE[2]
    top = base + 2 * block("cuboid").half_size[2]
    reach = KIN.jaw_reach(s.cmd_open)

    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] == pytest.approx(base + reach + STACK_CLEAR_M)
    bite = top - (s.grasp.grasp_point[2] - KIN.jaw_centre(s.cmd_open)[0]) - KIN.tip_swing(s.cmd_open) / 2
    assert bite >= MIN_BITE_M and s.grasp.skipped == ""


def test_a_regrasp_after_a_drop_onto_the_table_aims_at_the_table():
    s = aimed_at(3 * CUBE[2])
    s._regrasp(Sighting(SEEN, (XY[0] + 0.03, XY[1]), CUBE[2], np.eye(3), "kamery", 0.1, 0.0))

    assert s.base_z == 0.0 and s.config.hover_m == 0.04 and s.top_nominal == pytest.approx(2 * CUBE[2])
    assert s.grasp.grasp_point is not None and s.grasp.grasp_point[2] < 2 * CUBE[2]


def test_a_cuboid_that_falls_back_onto_its_cube_is_regrasped_clear_of_the_cube():
    half = block("cuboid").half_size[2]
    s = aimed_at(2 * CUBE[2] + half, kind="cuboid")
    base = 2 * CUBE[2]
    seen_back = Sighting(SEEN, (XY[0] + 0.004, XY[1]), base + half, np.eye(3), "kamery", 0.1, 0.0)
    assert s._sighting(obs(dict(SPEC.home), objects=seen(xy=seen_back.xy, z=seen_back.z, label="cuboid",
                                                           source="kamery", age_s=0.1)), False).kind == SEEN
    s._regrasp(seen_back)

    assert s.base_z == pytest.approx(base) and s.top_nominal == pytest.approx(base + 2 * half)
    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] >= base + KIN.jaw_reach(s.cmd_open) + STACK_CLEAR_M - 1e-9


def test_the_openings_follow_the_library_grasps_width(tmp_path):
    """Library width sets opening, squeeze and sideways retry distance."""
    width = 0.041012
    diagonal = grasp((0.0, 0.0, 0.0), closing=(math.sqrt(0.5), math.sqrt(0.5), 0.0), width=width)
    s, a = aimed(YAWED, library_dir=library_dir(tmp_path, [diagonal]))

    assert not a.done, a.note
    assert s.gap_open_m == pytest.approx(width + 0.016)
    assert s.gap_closed_m == pytest.approx(width - 0.008)
    assert KIN.jaw_gap(s.cmd_open) == pytest.approx(width + 0.016, abs=1e-6)
    assert KIN.jaw_gap(s.cmd_closed) == pytest.approx(width - 0.008, abs=1e-6)
    # The selected heading follows a cube diagonal.
    assert min(abs(wrap(math.radians(s.grasp.heading_deg - 45.0 - 90.0 * k))) for k in range(4)) < 1e-6
    xy0 = s._block_xy0
    assert xy0 is not None
    shifted = s._next_shift()
    assert shifted is not None and shifted - xy0 == pytest.approx(along(s, width / 2))
    # Reset the chosen width before the next run.
    s.start(run_info(), obs(dict(SPEC.home)))
    opening = 2 * CUBE[1] + 0.016
    assert s.gap_open_m == pytest.approx(opening)
    assert s.cmd_open == pytest.approx(KIN.gripper_for_gap(opening))


def test_the_sideways_retarget_keeps_the_library_grasp_and_shifts_by_half_its_width():
    """A tip stall recomposes the same library grasp at a retry point half its width toward the moving jaw."""
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
    assert np.asarray(aims[1]) - np.asarray(aims[0]) == pytest.approx(along(s, CUBE[1]), abs=1e-6)
