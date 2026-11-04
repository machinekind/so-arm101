"""Test pick-and-place phases and outcomes with a fake supervisor, arm, jaw and cameras.

The supervisor follows each target at once. A squeezed block takes the jaw's pads flat and
follows the hand. A held block stalls the arm on a support, and a released one lands on the
highest support under its outline. After the lift the world records each fingertip that goes into a
resting block. The cameras may see every block a fixed offset away, as a camera calibration error would.
Twin integration tests cover physics and perception.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, Frame, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.approach_table import ApproachTable, Hit  # noqa: E402
from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.planner.world import overlap, rectangle  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.approach import direction_vector, place_heights  # noqa: E402
from soarm_lab.strategies.grasp import NEEDS_YAW, STACK_CLEAR_M, heading, tilt, wrap  # noqa: E402
from soarm_lab.strategies.grasp_jev import HELD, HELD_OPTIONS, NEXT, NEXT_OPTIONS, GraspJevSettings  # noqa: E402
from soarm_lab.strategies.jev import KEY_ENV  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.pick_place import (  # noqa: E402
    FROM_PICK,
    FROM_TABLE,
    GRIP_SLIP_M,
    MIN_CARRY_CLEAR_M,
    OUTCOMES,
    TABLE_TOP_M,
    PickPlaceSettings,
    PickPlaceStrategy,
    _lean_deg,
    known_yaw,
    yaw_of,
)
from soarm_lab.strategies.wrist import Measurement, WristGraspSettings, fold  # noqa: E402

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
G = SPEC.gripper
assert G is not None
PICK_XY = (0.2, -0.04)
TARGET_XY = (0.2, 0.06)
CUBE = block("cube").half_size
CUBOID = block("cuboid").half_size
#: Pose age of a block the cameras see [s].
SEEN_AGE_S = 0.05
#: A fingertip centre this far inside a resting block counts as a hit, not a touch [m].
HIT_M = 0.0005
#: Where the base's turning axis crosses the table [m, base frame].
PAN_XY = KIN.joint_anchor(SPEC.arm_joints[0], SPEC.home)[:2]


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def run_info() -> RunInfo:
    return RunInfo(robot="so101", joints=tuple(SPEC.joints), gripper=G, limits=dict(SPEC.limits),
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)


@dataclass
class Body:
    """A block with its label, half sizes, centre and yaw."""

    label: str
    half: tuple[float, float, float]
    xy: np.ndarray
    z: float
    yaw: float = 0.0
    #: The cameras see it.
    seen: bool = True
    gripped: bool = False
    #: The centre in the TCP frame and the yaw against the closing heading while gripped.
    local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rel: float = 0.0

    @property
    def top(self) -> float:
        return self.z + self.half[2]

    def frame(self, point: np.ndarray) -> np.ndarray:
        """`point`'s x and y in the block's frame [m]."""
        return rot_z(self.yaw)[:2, :2].T @ (np.asarray(point, float)[:2] - self.xy)

    @property
    def footprint(self) -> tuple[tuple[float, float], ...]:
        return rectangle(float(self.xy[0]), float(self.xy[1]), self.yaw, self.half)

    def inside(self, point: np.ndarray) -> bool:
        """Whether `point` lies more than `HIT_M` inside the block."""
        local = self.frame(point)
        return bool(abs(local[0]) < self.half[0] - HIT_M and abs(local[1]) < self.half[1] - HIT_M
                    and abs(point[2] - self.z) < self.half[2] - HIT_M)

    def straddled(self, joints: Mapping[str, float]) -> int | None:
        """The block axis the tips straddle below the top, or None."""
        tips = KIN.fingertips(joints)
        if any(tip[2] >= self.top or tip[2] < self.z - self.half[2] - 0.01 for tip in tips):
            return None
        a, b = (self.frame(tip) for tip in tips)
        # The closing axis first: tips centred on the block also sit on either side of its other mid-plane.
        for k in sorted((0, 1), key=lambda k: -abs(a[k] - b[k])):
            if a[k] * b[k] < 0 and abs(a[1 - k]) < self.half[1 - k] and abs(b[1 - k]) < self.half[1 - k]:
                return k
        return None


def cube(label: str, xy: tuple[float, float], yaw_deg: float = 0.0, on: float = 0.0) -> Body:
    return Body(label, CUBE, np.array(xy, float), on + CUBE[2], math.radians(yaw_deg))


@dataclass
class World:
    """Blocks, the arm's joints and the supervisor's command."""

    bodies: list[Body]
    joints: dict[str, float] = field(default_factory=lambda: dict(SPEC.home))
    command: dict[str, float] = field(default_factory=lambda: dict(SPEC.home))
    t: float = 0.0
    #: (time, label) of each fingertip that went into a resting block while `watching`.
    hits: list[tuple[float, str]] = field(default_factory=list)
    watching: bool = False
    #: Labels whose poses carry no yaw.
    no_yaw: set[str] = field(default_factory=set)
    #: Where the cameras see every block against where it is [m, base frame x and y].
    offset: tuple[float, float] = (0.0, 0.0)

    def body(self, label: str) -> Body:
        return next(b for b in self.bodies if b.label == label)

    def objects(self) -> dict[str, ObjectPose]:
        out = {}
        for b in self.bodies:
            if b.seen:
                rotation = rot_z(b.yaw) if b.yaw and b.label not in self.no_yaw else np.eye(3)
                source, age = ("w dloni", 0.0) if b.gripped else ("kamery", SEEN_AGE_S)
                xy = b.xy + np.array(self.offset)
                out[b.label] = ObjectPose(np.array([*xy, b.z]), rotation, source, age)
        return out

    def support(self, point: np.ndarray, held: Body) -> float:
        """The highest top under the outline of `held` centred at `point`, the table included [m]."""
        outline = rectangle(float(point[0]), float(point[1]), held.yaw, held.half)
        return max([b.top for b in self.bodies if b is not held and not b.gripped
                    and overlap(outline, b.footprint) > HIT_M] + [0.0])

    def step(self, target: Mapping[str, float]) -> None:
        cmd = dict(self.command, **target)
        new = dict(cmd)
        held = next((b for b in self.bodies if b.gripped), None)
        if held is not None:
            T = KIN.tcp(new)
            centre = T[:3, :3] @ held.local + T[:3, 3]
            if centre[2] - held.half[2] < self.support(centre, held) - 1e-4:
                new = dict(self.joints, **{G: new[G]})
        free = [b for b in self.bodies if not b.gripped]
        straddled = next(((b, k) for b in free if (k := b.straddled(new)) is not None), None)
        if held is not None:
            width = 2 * held.half[1]
            grip = max(cmd[G], KIN.gripper_for_gap(width))
        elif straddled is not None:
            b, k = straddled
            width = 2 * b.half[k]
            grip = max(cmd[G], KIN.gripper_for_gap(width))
        else:
            grip = cmd[G]
        new[G] = grip
        if held is not None and cmd[G] > KIN.gripper_for_gap(2 * held.half[1]) + 0.5:
            held.gripped = False
            held.z = self.support(np.array([*held.xy, 0.0]), held) + held.half[2]
        elif straddled is not None and cmd[G] < grip - 0.5:
            b, k = straddled
            self._grip(b, k, new)
        held = next((b for b in self.bodies if b.gripped), None)
        if held is not None:
            T = KIN.tcp(new)
            centre = T[:3, :3] @ held.local + T[:3, 3]
            held.xy, held.z = centre[:2].copy(), float(centre[2])
            held.yaw = heading(KIN, new) + held.rel
        if self.watching:
            tips = KIN.fingertips(new)
            self.hits += [(self.t, b.label) for b in self.bodies
                          if not b.gripped and any(b.inside(t) for t in tips)]
        self.command, self.joints = cmd, new

    def _grip(self, b: Body, k: int, joints: Mapping[str, float]) -> None:
        """Squeeze `b`: its faces lie flat on the pads and its centre between them."""
        fixed, moving = KIN.fingertips(joints)
        u = (moving - fixed)[:2] / np.linalg.norm((moving - fixed)[:2])
        mid = ((fixed + moving) / 2)[:2]
        b.xy = b.xy + float((mid - b.xy) @ u) * u
        psi = heading(KIN, joints)
        b.rel = round((b.yaw - psi) / (math.pi / 2)) * (math.pi / 2)
        b.yaw = psi + b.rel
        T = KIN.tcp(joints)
        b.local = T[:3, :3].T @ (np.array([*b.xy, b.z]) - T[:3, 3])
        b.gripped = True


Hook = Callable[[World, PickPlaceStrategy], None]


def run(s: PickPlaceStrategy, world: World, seconds: float = 60.0,
        on_phase: dict[str, Hook] | None = None, watch: Hook | None = None,
        frames: Mapping[str, Frame] | None = None) -> tuple[list[Action], list[str]]:
    """Start and tick until done. Return the actions and every phase entered, in order.

    `on_phase` hooks run after the tick that entered their phase, `watch` after every world step. Every
    observation carries `frames`.
    """
    entered: list[str] = []
    enter = s._enter

    def spy(phase: str) -> None:
        entered.append(phase)
        enter(phase)

    s._enter = spy  # type: ignore[method-assign]
    world.watching = False
    s.start(run_info(), Observation(t=0.0, step=0, joints=dict(world.joints), command=dict(world.command),
                                    objects=world.objects()))
    actions = []
    for step in range(round(seconds * s.hz)):
        world.t = step / s.hz
        before = len(entered)
        a = s.act(Observation(t=world.t, step=step, joints=dict(world.joints), command=dict(world.command),
                              objects=world.objects(), frames=frames or {}))
        actions.append(a)
        for phase in entered[before:]:
            # The fake squeeze is not exact, so fingertips count from the lift on.
            world.watching = world.watching or phase == "carry"
            if on_phase and phase in on_phase:
                on_phase[phase](world, s)
        if a.done or s.status.phase == "done":
            break
        world.step(a.target)
        if watch is not None:
            watch(world, s)
    return actions, entered


def jaw_low(joints: Mapping[str, float]) -> float:
    """The jaw's lowest point at `joints` [m above the table]."""
    T = KIN.tcp(joints)
    return float((np.vstack(KIN.jaw_points(joints[G])) @ T[:3, :3].T + T[:3, 3])[:, 2].min())


def built(**settings: Any) -> PickPlaceStrategy:
    settings.setdefault("judge", "none")
    settings.setdefault("wrist_camera", "")
    settings.setdefault("label", "red")
    judge = settings.pop("judge_obj", None)
    return PickPlaceStrategy(PickPlaceSettings(**settings), judge=judge, threaded=False)


def stack(**settings: Any) -> tuple[PickPlaceStrategy, World]:
    """A red cube to pick and a blue cube to place it on."""
    return built(place_on="blue", **settings), World([cube("red", PICK_XY), cube("blue", TARGET_XY, 20.0)])


def tower(xy: tuple[float, float] = TARGET_XY, **settings: Any) -> tuple[PickPlaceStrategy, World]:
    """A red cube to pick and a blue cube on a base cube to place it on."""
    world = World([cube("red", PICK_XY), cube("base", xy), cube("blue", xy, on=2 * CUBE[2])])
    return built(place_on="blue", **settings), world


# ------------------------------------------------------------ settings
def test_its_settings_are_the_grasp_familys_and_the_places():
    names = [f.name for f in dataclasses.fields(PickPlaceSettings)]
    wrist = [f.name for f in dataclasses.fields(WristGraspSettings)]
    jev = [f.name for f in dataclasses.fields(GraspJevSettings)]

    assert names[:len(wrist)] == wrist and set(jev) <= set(names)
    assert names[-29:] == ["judge", "place_on", "place_on_block", "place_x_m", "place_y_m", "place_z_m",
                           "place_yaw_deg", "supports", "align_yaw", "carry_clearance_m", "place_gap_m",
                           "place_speed_mps", "release_margin_m", "release_ease_m", "retreat_m",
                           "retreat_speed_mps", "retreat_clear_m", "fresh_pose_s", "target_wait_s",
                           "verify_wait_s", "place_tol_m", "place_retries", "approach_file",
                           "approach_direction", "approach_lead_m", "pick_shift", "table_check_m",
                           "table_check_runs", "max_table_age_days"]
    defaults = PickPlaceSettings()
    assert defaults.judge == "jev" and defaults.wrist_camera == "wrist" and defaults.time_limit_s == 240.0
    assert defaults.max_tilt_deg == 30.0 and defaults.place_retries == 2
    retreat = (defaults.release_ease_m, defaults.retreat_speed_mps, defaults.retreat_clear_m)
    assert retreat == (0.002, 0.02, 0.02)
    assert (defaults.approach_file, defaults.approach_direction, defaults.pick_shift) == ("", "", "always")
    assert defaults.approach_lead_m == 0.03 and defaults.table_check_m == 0.006
    assert defaults.table_check_runs == 2
    assert OUTCOMES == ("placed", "not_found", "target_not_seen", "pick_failed", "dropped", "fell_off",
                        "target_moved", "unreachable", "stopped")


@pytest.mark.parametrize("settings, message", [
    ({"judge": "claude"}, "judge must be one of jev, none"),
    ({"place_on_block": "ball"}, "place_on_block must be one of cube, cuboid, cylinder"),
    ({"place_speed_mps": 0.0}, "place_speed_mps must be positive"),
    ({"release_margin_m": -0.001}, "release_margin_m must be positive"),
    ({"release_ease_m": 0.004}, "release_ease_m must lie within"),
    ({"release_ease_m": -0.001}, "release_ease_m must lie within"),
    ({"retreat_speed_mps": 0.0}, "retreat_speed_mps must be positive"),
    ({"retreat_clear_m": 0.0}, "retreat_clear_m must be positive"),
    ({"place_retries": -1}, "place_retries must be zero or more"),
    ({"place_gap_m": -0.001}, "place_gap_m must be zero or more"),
    ({"place_x_m": math.nan}, "place_x_m must be finite"),
    ({"place_yaw_deg": math.inf}, "place_yaw_deg must be finite"),
    ({"place_z_m": -0.001}, "place_z_m must be finite and zero or more"),
    ({"place_z_m": 0.029}, "place_z_m 0.029 above the table needs supports"),
    ({"supports": "a", "place_z_m": 0.005}, "supports need place_z_m at their top"),
    ({"supports": "a", "place_z_m": 0.029, "place_on": "blue"}, "supports needs place_on empty"),
    ({"supports": "a, A", "place_z_m": 0.029}, "supports names a block twice"),
    ({"supports": "a, Red", "place_z_m": 0.029}, "supports must name other blocks than label"),
    ({"send_frames": True}, "send_frames needs wrist_camera"),
    ({"place_on": "Red"}, "place_on must name another block than label"),
    ({"approach_direction": "up"}, "approach_direction must be empty or one of out, in, ccw, cw"),
    ({"pick_shift": "sometimes"}, "pick_shift must be one of auto, always, off"),
    ({"approach_lead_m": 0.0}, "approach_lead_m must be positive"),
    ({"table_check_m": math.nan}, "table_check_m must be positive"),
    ({"table_check_runs": 0}, "table_check_runs must be at least 1"),
    ({"max_table_age_days": -1.0}, "max_table_age_days must be zero or more"),
])
def test_bad_settings_are_refused(settings: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=f"pick_place: {message}"):
        built(**settings)


def test_with_judge_none_it_starts_without_a_key_and_follows_the_protocol(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    s = PickPlaceStrategy(PickPlaceSettings(judge="none", wrist_camera=""))
    s.start(run_info(), Observation(t=0.0, step=0, joints=dict(SPEC.home), command={}))
    a = s.act(Observation(t=0.05, step=1, joints=dict(SPEC.home), command={}))

    assert isinstance(a, Action) and set(a.target) == set(SPEC.joints) and not a.done
    assert s.judge.name == "none" and isinstance(s.hz, float)
    s.stop("zatrzymana")
    assert s.place.outcome == "stopped"


def test_with_judge_jev_it_refuses_to_start_without_a_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    s = PickPlaceStrategy(PickPlaceSettings(judge="jev", wrist_camera=""))
    with pytest.raises(RuntimeError, match="brak klucza jev"):
        s.start(run_info(), Observation(t=0.0, step=0, joints=dict(SPEC.home), command={}))


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_a_block_lies_flat_on_any_face_of_its_shape_that_matches_its_bottom():
    """A cube lies flat on any face, a cuboid only on its broad one, a cylinder however it rolls."""
    tipped, quarter = rot_x(math.radians(10.0)), rot_x(math.pi / 2)
    assert _lean_deg(rot_z(0.4), block("cuboid")) == pytest.approx(0.0, abs=1e-6)
    assert _lean_deg(tipped, block("cuboid")) == pytest.approx(10.0)
    assert _lean_deg(quarter, block("cube")) == pytest.approx(0.0, abs=1e-6)
    assert _lean_deg(quarter, block("cuboid")) == pytest.approx(90.0)
    assert _lean_deg(quarter, block("cylinder")) == pytest.approx(0.0, abs=1e-6)
    c, sn = math.cos(0.2), math.sin(0.2)
    pitched = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    assert _lean_deg(pitched, block("cylinder")) == pytest.approx(math.degrees(0.2))
    assert math.isnan(_lean_deg(np.full((3, 3), np.nan), block("cube")))


def test_an_identity_rotation_is_yaw_zero_for_a_target_and_unknown_for_the_picked_block():
    assert yaw_of(np.eye(3)) == 0.0 and yaw_of(rot_z(0.3)) == pytest.approx(0.3)
    assert yaw_of(np.full((3, 3), np.nan)) is None and yaw_of(np.eye(2)) is None
    assert known_yaw(np.eye(3)) is None and known_yaw(np.full((3, 3), np.nan)) is None
    assert known_yaw(rot_z(0.3)) == pytest.approx(0.3)


# ------------------------------------------------------------ placed
def test_a_cube_goes_on_another_cube_through_every_phase():
    s, world = stack()
    actions, phases = run(s, world)
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].done and actions[-1].success and s.place.outcome == "placed"
    assert phases[:3] == ["wait", "approach", "open"]
    assert phases[phases.index("check"):] == ["check", "carry", "target", "carry_up", "carry_over", "latch",
                                              "place", "let_go", "park", "verify"]
    assert not red.gripped and red.z == pytest.approx(blue.top + CUBE[2])
    assert np.linalg.norm(red.xy - blue.xy) < 0.002
    assert abs(fold(red.yaw - blue.yaw, math.pi / 2)) < math.radians(1.0)
    assert s.place.error_m < 0.002 and abs(s.place.height_error_m) < 1e-6
    assert s.status.stopped_because.startswith("blok 'red' odlozony na 'blue': ") and s.grasp.tilt_deg == 0.0
    assert s.place.release_z == pytest.approx(blue.top + 0.002 + GRIP_SLIP_M + s.place.hang_m)
    # The jaw opened just past the block, not to the straddle's opening, and left it without a touch.
    assert s.grasp.gap_cmd_m == pytest.approx(2 * CUBE[1] + 0.008, abs=1e-4)
    assert not s.place.opened_wide and not world.hits


def test_a_cube_goes_to_a_table_spot():
    s = built(place_x_m=0.22, place_y_m=0.08)
    world = World([cube("red", PICK_XY)])
    actions, phases = run(s, world)
    red = world.body("red")

    assert actions[-1].success and s.place.outcome == "placed" and "latch" not in phases
    assert np.linalg.norm(red.xy - np.array([0.22, 0.08])) < 0.002 and red.z == pytest.approx(CUBE[2])
    assert not world.hits
    assert s.status.stopped_because.startswith("blok 'red' odlozony na stol: ")


def test_align_yaw_turns_the_hand_by_the_smallest_equivalent_turn():
    s, world = stack()
    run(s, world)
    turn = s.place.turn_deg

    assert abs(turn) <= 45.0
    assert abs(fold(math.radians(s.place.heading_deg - 90.0 - 20.0), math.pi / 2)) < 1e-6


def test_without_align_yaw_the_hand_keeps_its_heading():
    s, world = stack(align_yaw=False)
    run(s, world)

    assert s.place.turn_deg == 0.0 and s.place.outcome == "placed"


def test_a_target_at_yaw_zero_is_lined_up_with():
    s = built(place_on="blue")
    world = World([cube("red", PICK_XY, 25.0), cube("blue", TARGET_XY)])
    actions, _ = run(s, world)
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.target_yaw_deg == 0.0
    assert abs(fold(red.yaw - blue.yaw, math.pi / 2)) < math.radians(1.0)


def test_a_third_cube_goes_on_two_and_the_fully_open_jaw_leaves_it_untouched():
    s, world = tower()
    actions, phases = run(s, world)
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert red.z == pytest.approx(blue.top + CUBE[2]) and np.linalg.norm(red.xy - blue.xy) < 0.002
    # Out of reach higher up, the carry runs 1 cm over the stack and the descent starts there.
    assert s.place.carry_z - s.place.release_z < 0.01
    assert s.place.opened_wide and s.grasp.gap_cmd_m > 0.08
    assert not world.hits and not world.body("base").gripped
    assert phases[-3:] == ["let_go", "park", "verify"]


def test_a_third_cube_the_open_hand_could_not_leave_is_unreachable_before_the_pick():
    s, world = tower((0.22, 0.0), max_tilt_deg=0.0)
    _, phases = run(s, world)

    assert s.place.outcome == "unreachable" and "open" not in phases
    assert s.status.stopped_because.startswith("po odlozeniu szczeki nie mina bloku")


def square(xy: tuple[float, float]) -> float:
    """The yaw [deg] that sets a block at `xy` square to the arm."""
    return math.degrees(math.atan2(xy[1] - PAN_XY[1], xy[0] - PAN_XY[0]))


def held_tilt(world: World, tilts: list[tuple[float, float]], grip: list[np.ndarray]) -> None:
    """Record how far the held red cube leans from its grip and the hand from straight down [deg]."""
    if world.body("red").gripped:
        R = KIN.tcp(world.joints)[:3, :3]
        grip[:] = grip or [R]
        block = math.degrees(math.acos(min(1.0, float((R @ grip[0].T)[2, 2]))))
        tilts.append((block, math.degrees(tilt(KIN, world.joints))))


def tilted(xy: tuple[float, float], under: int, yaw: float, *, red_yaw: float | None = None,
           posed: bool = True, looked: bool = False, **settings: Any
           ) -> tuple[PickPlaceStrategy, World, list[tuple[float, float]], list[str]]:
    """Place the red cube on a stack of `under` cubes at `xy`. Return the held cube's and the hand's tilts
    and the phases.

    The red cube lies at `red_yaw` [deg], square to the arm by default, with that yaw in its pose when
    `posed`. With `looked`, the wrist camera sends a frame and measures the red cube where it lies."""
    names = [f"base{k}" for k in range(under - 1)] + ["blue"]
    stack = [cube(name, xy, yaw, on=2 * k * CUBE[2]) for k, name in enumerate(names)]
    red = cube("red", PICK_XY, square(PICK_XY) if red_yaw is None else red_yaw)
    s, world = built(place_on="blue", **settings), World([red, *stack], no_yaw=set() if posed else {"red"})
    frames = None
    if looked:
        frames = {"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)}
        s.measure = lambda rgb, joints, ref_xy=None: Measurement(  # type: ignore[method-assign]
            red.xy.copy(), red.yaw, 0.9, 1000, 1.0, np.zeros(2))
    tilts: list[tuple[float, float]] = []
    grip: list[np.ndarray] = []
    _, phases = run(s, world, watch=lambda world, s: held_tilt(world, tilts, grip), frames=frames)
    return s, world, tilts, phases


FAR_XY = (0.24, 0.0)
FOURTH_XY = (0.21, 0.08)


@pytest.mark.parametrize("xy, under", [(FAR_XY, 2), (FOURTH_XY, 3)], ids=["third cube far out", "fourth"])
def test_a_cube_out_of_straight_down_reach_goes_on_its_stack_with_the_hand_tilted(xy, under):
    straight, world, *_ = tilted(xy, under, square(xy), max_tilt_deg=0.0)
    assert straight.place.outcome == "unreachable" and not world.body("red").gripped

    s, world, tilts, _ = tilted(xy, under, square(xy))
    red, blue = world.body("red"), world.body("blue")

    assert s.place.outcome == "placed", s.status.stopped_because
    assert 0.0 < s.grasp.tilt_deg <= 30.0
    assert s.status.stopped_because.endswith(f" (dlon pochylona {s.grasp.tilt_deg:.0f} st.)")
    assert red.z == pytest.approx(blue.top + CUBE[2]) and np.linalg.norm(red.xy - blue.xy) < 0.002
    assert abs(fold(red.yaw - blue.yaw, math.pi / 2)) < math.radians(1.0)
    # The hand kept its tilt and its heading against the arm, so the held cube stayed level.
    assert tilts and max(block for block, _ in tilts) < 2.0
    assert [hand for _, hand in tilts] == pytest.approx([s.grasp.tilt_deg] * len(tilts), abs=1.0)
    assert not world.hits and all(not world.body(f"base{k}").gripped for k in range(under - 1))


def test_without_the_blocks_yaw_and_a_wrist_camera_the_hand_does_not_tilt():
    s, world, tilts, phases = tilted(FAR_XY, 2, square(FAR_XY), posed=False)

    assert s.place.outcome == "unreachable" and "open" not in phases and not tilts and s.grasp.tilt_deg == 0.0
    assert s.status.stopped_because.endswith(f"; {NEEDS_YAW}")


@pytest.mark.parametrize("posed", [True, False], ids=["pose with yaw", "pose without yaw"])
def test_a_tilted_grasp_looks_with_the_wrist_camera(posed):
    pytest.importorskip("cv2")
    s, world, tilts, _ = tilted(FAR_XY, 2, square(FAR_XY), posed=posed, looked=True, wrist_camera="wrist")

    assert s.place.outcome == "placed" and s.grasp.tilt_deg > 0.0, s.status.stopped_because
    assert s.wrist.verdicts == ["centred"] and tilts and max(block for block, _ in tilts) < 2.0


def test_a_tilted_grasp_without_a_frame_keeps_the_poses_yaw_and_never_guesses_one():
    pytest.importorskip("cv2")
    s, world, *_ = tilted(FAR_XY, 2, square(FAR_XY), wrist_camera="wrist")
    assert s.place.outcome == "placed" and s.grasp.tilt_deg > 0.0 and s.wrist.verdicts == ["skipped"]

    s, world, tilts, phases = tilted(FAR_XY, 2, square(FAR_XY), posed=False, wrist_camera="wrist")

    assert s.place.outcome == "unreachable" and "open" in phases and "straddle" not in phases and not tilts
    assert s.status.stopped_because == f"{NEEDS_YAW}, a kamera go nie zmierzyla (brak klatki z kamery wrist)"


@pytest.mark.parametrize("posed", [True, False], ids=["pose with yaw", "pose without yaw"])
def test_a_cube_askew_to_the_arm_is_not_picked_tilted(posed):
    """A known yaw refuses before the pick; a guessed one ends once the wrist camera measures the cube."""
    pytest.importorskip("cv2")
    s, world, tilts, phases = tilted(FAR_XY, 2, square(FAR_XY), red_yaw=square(PICK_XY) + 40.0, posed=posed,
                                     looked=True, wrist_camera="wrist")

    assert s.place.outcome == "unreachable" and "straddle" not in phases and not tilts
    assert np.linalg.norm(world.body("red").xy - PICK_XY) < 1e-9
    if posed:
        assert "open" not in phases and s.wrist.looks == 0
    else:
        assert s.wrist.verdicts == ["skipped"]
        assert s.status.stopped_because.startswith("blok w kamerze obrocony o 40 st. wzgledem ramienia: ")


def test_a_tilted_place_needs_the_target_square_to_the_arm():
    s, world, *_ = tilted(FOURTH_XY, 3, square(FOURTH_XY) + 40.0)

    assert s.place.outcome == "unreachable" and not world.body("red").gripped and s.grasp.tilt_deg == 0.0


def test_a_cube_put_back_from_over_a_stack_is_carried_above_it():
    s, world = tower(target_wait_s=1.0)
    top = world.body("blue").top
    bottoms: list[float] = []

    def over(world: World, s: PickPlaceStrategy) -> None:
        red = world.body("red")
        if s.place.put_back and red.gripped and np.linalg.norm(red.xy - world.body("blue").xy) < 4 * CUBE[0]:
            bottoms.append(red.z - CUBE[2])

    actions, phases = run(s, world, on_phase={"check": stale}, watch=over)
    red = world.body("red")

    assert s.place.outcome == "target_not_seen" and s.place.put_back and not actions[-1].success
    assert phases.count("latch") == 1 and phases.count("carry_over") == 2
    assert bottoms and min(bottoms) >= top + MIN_CARRY_CLEAR_M - 1e-4
    assert not red.gripped and red.z == pytest.approx(CUBE[2]) and np.linalg.norm(red.xy - PICK_XY) < 0.01
    assert not world.hits


def test_a_cube_on_a_cuboid_lines_up_modulo_the_cubes_quarter_turn():
    s = built(place_on="long", place_on_block="cuboid")
    long = Body("long", CUBOID, np.array(TARGET_XY), CUBOID[2], math.radians(-30.0))
    world = World([cube("red", PICK_XY), long])
    actions, _ = run(s, world)
    red = world.body("red")

    assert actions[-1].success and red.z == pytest.approx(2 * CUBOID[2] + CUBE[2])
    assert abs(fold(red.yaw - long.yaw, math.pi / 2)) < math.radians(1.0)
    assert not world.hits


def test_a_target_top_higher_than_planned_stalls_the_descent_and_the_cube_still_lands():
    s = built(place_on="blue")
    base = Body("base", CUBOID, np.array(TARGET_XY), CUBOID[2])
    world = World([cube("red", PICK_XY), base, cube("blue", TARGET_XY, on=2 * CUBOID[2])])
    # The cameras put the target lower than it stands, 6 mm more than the release leaves for a slipped block.
    blue = world.body("blue")
    objects = world.objects

    def lower() -> dict[str, ObjectPose]:
        out = objects()
        if "blue" in out:
            p = out["blue"]
            low = np.array([0.0, 0.0, 0.006 + GRIP_SLIP_M])
            out["blue"] = ObjectPose(p.position - low, p.rotation, p.source, p.age_s)
        return out

    world.objects = lower  # type: ignore[method-assign]
    actions, _ = run(s, world)

    assert s.place.contact and s.place.outcome == "placed" and actions[-1].success
    assert world.body("red").z == pytest.approx(blue.top + CUBE[2])
    assert not world.hits


# ------------------------------------------------------------ leaving the placed block
def leaving(track: list[tuple[str, np.ndarray, float]]) -> Hook:
    """Record the phase, the TCP and the jaw's lowest point from the moment the placing jaw is open."""

    def record(world: World, s: PickPlaceStrategy) -> None:
        if s._placing and s.status.phase in ("let_go", "park") and s._released:
            track.append((s.status.phase, KIN.tcp(world.joints)[:3, 3].copy(), jaw_low(world.joints)))

    return record


@pytest.mark.parametrize("onto", ["stack", "table"])
def test_the_open_hand_eases_2_mm_then_rises_slowly_clear_of_the_cube_before_it_moves_away(onto: str):
    """As on the stand, with a 10 mm release margin: the hand still eases only 2 mm."""
    if onto == "stack":
        s, world = stack(release_margin_m=0.010)
    else:
        s = built(place_x_m=0.22, place_y_m=0.08, release_margin_m=0.010)
        world = World([cube("red", PICK_XY)])
    track: list[tuple[str, np.ndarray, float]] = []
    actions, _ = run(s, world, watch=leaving(track))
    red = world.body("red")
    clear = s.place_settings.retreat_clear_m

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert track and s.place.jaw_clear_m >= clear - 0.001 and not s.place.opened_wide
    opened = track[0][1]
    # Until the jaw clears the cube, the hand moves only the ease across.
    low = [tcp for _, tcp, jaw in track if jaw < red.top + clear - 0.001]
    assert low and max(float(np.linalg.norm(tcp[:2] - opened[:2])) for tcp in low) <= 0.002 + 1e-4
    rises = [b[1][2] - a[1][2] for a, b in zip(track, track[1:], strict=False) if a[0] == b[0] == "let_go"]
    assert max(rises) <= 0.02 / s.hz + 1e-4
    # Near the cube the way to the start position keeps the jaw that high, and the hand about that slow.
    near = [(tcp, jaw) for phase, tcp, jaw in track
            if phase == "park" and float(np.linalg.norm(tcp[:2] - red.xy)) < s._near_m]
    assert near and min(jaw for _, jaw in near) >= red.top + clear - 0.001
    steps = [float(np.linalg.norm(b[0] - a[0])) for a, b in zip(near, near[1:], strict=False)]
    assert max(steps) <= 1.5 * 0.02 / s.hz
    assert not world.hits


def test_an_arm_that_settles_short_on_the_way_up_rises_until_the_open_jaw_clears_the_cube():
    """The pitch joints hang 2.5 deg short once the hand rises: the rise goes on by what is missing. A short
    `retreat_m` plans the rise to just the clearance."""
    s, world = built(place_x_m=0.22, place_y_m=0.08, retreat_m=0.03), World([cube("red", PICK_XY)])
    hung: list[float] = []

    def sag(world: World, s: PickPlaceStrategy) -> None:
        if s._placing and s.status.phase == "let_go" and s._released and s._eased:
            joints = dict(world.joints, shoulder_lift=world.joints["shoulder_lift"] + 2.5,
                          elbow_flex=world.joints["elbow_flex"] + 2.5)
            hung.append(float(KIN.tcp(world.joints)[2, 3] - KIN.tcp(joints)[2, 3]))
            world.joints = joints

    actions, _ = run(s, world, watch=sag)

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert max(hung) > 0.01
    assert s.place.jaw_clear_m >= s.place_settings.retreat_clear_m - 0.001
    assert s._rise_extra > 0.0 and not world.hits


# ------------------------------------------------------------ picked off a stack
def picked_off(lower: Body, **settings: Any) -> tuple[PickPlaceStrategy, World]:
    """A red cube on `lower` at the pick spot and a blue cube to place it on."""
    settings.setdefault("place_on", "blue")
    world = World([lower, cube("red", PICK_XY, on=lower.top), cube("blue", TARGET_XY, 20.0)])
    return built(**settings), world


def lowest_tip(world: World, s: PickPlaceStrategy, lows: list[float], gripped_at: list[np.ndarray]) -> None:
    """Record the fingertips' lowest point while the red cube is not yet held, and where it was gripped."""
    red = world.body("red")
    if not red.gripped and s.status.phase in ("straddle", "close"):
        lows.append(min(float(tip[2]) for tip in KIN.fingertips(world.joints)))
    if red.gripped and not gripped_at:
        gripped_at.append(red.xy.copy())


@pytest.mark.parametrize("lower", [cube("base", PICK_XY), Body("base", CUBOID, np.array(PICK_XY), CUBOID[2],
                                                                math.radians(-30.0))],
                         ids=["off a cube", "off a cuboid"])
def test_a_cube_picked_off_a_block_goes_on_another_and_the_lower_block_stays(lower: Body):
    s, world = picked_off(lower)
    base = world.body("base")
    base_xy, base_z = base.xy.copy(), base.z
    lows: list[float] = []
    gripped_at: list[np.ndarray] = []
    actions, phases = run(s, world, watch=lambda world, s: lowest_tip(world, s, lows, gripped_at))
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert s.base_z == pytest.approx(base.top) and s.place.supports == ("base",)
    # The jaw went down beside the cube and stayed clear of the block under it.
    assert lows and min(lows) > base.top + STACK_CLEAR_M - 0.002
    # The hand followed the closing jaw, so the jaw did not slide the cube across the lower block.
    assert gripped_at and np.linalg.norm(gripped_at[0] - np.asarray(PICK_XY)) < 0.001
    assert red.z == pytest.approx(blue.top + CUBE[2]) and np.linalg.norm(red.xy - blue.xy) < 0.002
    assert np.linalg.norm(base.xy - base_xy) < 1e-9 and base.z == base_z and not base.gripped
    assert s.place.support_shift_m < 1e-9 and not s.place.support_moved
    assert "przesuniety" not in s.status.stopped_because and not world.hits


def test_the_carry_from_a_stack_clears_the_block_it_was_picked_from():
    s, world = picked_off(cube("base", PICK_XY))
    top = world.body("base").top
    bottoms: list[float] = []

    def over(world: World, s: PickPlaceStrategy) -> None:
        red = world.body("red")
        if red.gripped and np.linalg.norm(red.xy - np.asarray(PICK_XY)) < 2 * CUBE[0]:
            bottoms.append(red.z - CUBE[2])

    actions, _ = run(s, world, watch=over)

    assert actions[-1].success and bottoms
    # Above the stack the held cube only rises, and the carry leaves it well above the lower cube.
    assert min(bottoms) >= top - 1e-4 and s.place.carry_z - s.place.hang_m >= top + MIN_CARRY_CLEAR_M


def test_a_cube_put_back_goes_onto_the_block_it_was_picked_from():
    s, world = picked_off(cube("base", PICK_XY), target_wait_s=1.0)

    def hide(world: World, s: PickPlaceStrategy) -> None:
        world.body("blue").seen = False

    actions, _ = run(s, world, on_phase={"check": hide})
    red, base = world.body("red"), world.body("base")

    assert not actions[-1].success and s.place.outcome == "target_not_seen" and s.place.put_back
    assert s.place.target_top_m == pytest.approx(base.top)
    assert not red.gripped and red.z == pytest.approx(base.top + CUBE[2])
    assert np.linalg.norm(red.xy - base.xy) < 0.002 and not world.hits


def test_a_lower_block_that_moved_is_reported_and_the_put_back_follows_it():
    s, world = picked_off(cube("base", PICK_XY))
    pushed: list[bool] = []

    def push(world: World, s: PickPlaceStrategy) -> None:
        if not pushed:
            pushed.append(True)
            world.body("base").xy = world.body("base").xy + np.array([0.0, -0.02])

    def nudge(world: World, s: PickPlaceStrategy) -> None:
        if not s.place.put_back:
            world.body("blue").xy = world.body("blue").xy + np.array([0.0, 0.01])

    actions, _ = run(s, world, seconds=90.0, on_phase={"carry_up": push, "carry_over": nudge})
    red, base = world.body("red"), world.body("base")

    assert s.place.outcome == "target_moved" and s.place.put_back
    assert s.place.support_moved and s.place.support_shift_m == pytest.approx(0.02, abs=1e-6)
    assert s.status.stopped_because.endswith("blok odlozony z powrotem; blok pod podniesionym przesuniety "
                                             "o 20 mm")
    assert not red.gripped and red.z == pytest.approx(base.top + CUBE[2])
    assert np.linalg.norm(red.xy - base.xy) < 0.002 and not world.hits


def test_a_cube_dropped_after_the_pick_off_a_stack_is_picked_again_from_the_table():
    s, world = picked_off(cube("base", PICK_XY))

    def slip(world: World, s: PickPlaceStrategy) -> None:
        if s.grasp.drops == 0:
            red = world.body("red")
            red.gripped = False
            red.xy, red.z = np.array([0.19, 0.0]), CUBE[2]

    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": slip})

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1 and s.base_z == 0.0
    assert world.body("red").z == pytest.approx(world.body("blue").top + CUBE[2])
    assert world.body("base").z == pytest.approx(CUBE[2]) and not s.place.support_moved


def test_a_cuboid_dropped_back_onto_the_cube_it_came_off_is_picked_again_clear_of_that_cube():
    base = cube("base", PICK_XY)
    world = World([base, Body("red", CUBOID, np.array(PICK_XY), base.top + CUBOID[2], math.radians(10.0)),
                   cube("blue", TARGET_XY, 20.0)])
    s = built(block="cuboid", place_on="blue")
    lows: list[float] = []

    def slip(world: World, s: PickPlaceStrategy) -> None:
        if s.grasp.drops == 0:
            red = world.body("red")
            red.gripped = False
            red.xy, red.z = base.xy + np.array([0.003, -0.002]), base.top + CUBOID[2]

    def regrasp_tips(world: World, s: PickPlaceStrategy) -> None:
        if s.grasp.recoveries and not world.body("red").gripped and s.status.phase in ("straddle", "close"):
            lows.append(min(float(tip[2]) for tip in KIN.fingertips(world.joints)))

    actions, _ = run(s, world, seconds=90.0, on_phase={"carry_over": slip}, watch=regrasp_tips)
    red, blue = world.body("red"), world.body("blue")

    assert s.grasp.drops == 1 and s.grasp.recoveries == 1 and s.base_z == pytest.approx(base.top)
    # The regrasp counts from the cube under the cuboid, so the open jaw stays clear of that cube.
    assert s.grasp.grasp_point is not None
    assert s.grasp.grasp_point[2] >= base.top + KIN.jaw_reach(s.cmd_open) + STACK_CLEAR_M - 1e-9
    assert lows and min(lows) > base.top + STACK_CLEAR_M - 0.002 and not world.hits
    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert red.z == pytest.approx(blue.top + CUBOID[2]) and base.z == pytest.approx(CUBE[2])


# ------------------------------------------------------------ outcomes
def test_no_pose_of_the_block_is_not_found():
    s = built(place_x_m=0.22, place_y_m=0.08, wait_for_pose_s=1.0)
    world = World([])
    actions, phases = run(s, world)

    assert actions[-1].done and not actions[-1].success and phases == ["wait"]
    assert s.place.outcome == "not_found"
    assert s.status.stopped_because.startswith("brak uzytecznego polozenia obiektu 'red'")


def test_no_pose_of_the_target_before_the_pick_is_target_not_seen_and_the_arm_stays():
    s = built(place_on="blue", wait_for_pose_s=1.0)
    world = World([cube("red", PICK_XY)])
    actions, phases = run(s, world)

    assert phases == ["wait"] and s.place.outcome == "target_not_seen"
    assert s.status.stopped_because.startswith("brak uzytecznego polozenia celu 'blue'")
    arm = SPEC.arm_joints
    assert {n: actions[-1].target[n] for n in arm} == pytest.approx({n: SPEC.home[n] for n in arm})


def test_a_target_that_vanishes_after_the_lift_sends_the_cube_back_where_it_was_picked():
    s, world = stack(target_wait_s=1.0)

    def hide(world: World, s: PickPlaceStrategy) -> None:
        world.body("blue").seen = False

    actions, phases = run(s, world, on_phase={"check": hide})
    red = world.body("red")

    assert not actions[-1].success and s.place.outcome == "target_not_seen" and s.place.put_back
    assert "verify" not in phases and "place" in phases
    assert s.status.stopped_because == ("brak polozenia celu 'blue' (brak w obserwacji) - blok odlozony z "
                                        "powrotem")
    assert not red.gripped and red.z == pytest.approx(CUBE[2]) and np.linalg.norm(red.xy - PICK_XY) < 0.01


def stale(world: World, s: PickPlaceStrategy) -> None:
    """Hide the blue cube; the cameras then report where it was last seen, 5 s ago."""
    world.body("blue").seen = False
    objects = world.objects
    blue = world.body("blue")

    def old() -> dict[str, ObjectPose]:
        out = objects()
        out["blue"] = ObjectPose(np.array([*blue.xy, blue.z]), rot_z(blue.yaw), "ostatnie widziane", 5.0)
        return out

    world.objects = old  # type: ignore[method-assign]


def test_a_target_not_seen_fresh_on_the_way_or_at_the_hover_is_target_not_seen():
    s, world = stack(target_wait_s=1.0)
    actions, _ = run(s, world, on_phase={"check": stale})

    assert s.place.outcome == "target_not_seen" and s.place.put_back
    assert s.status.stopped_because.startswith("cel 'blue' bez swiezego polozenia od 1 s")


def test_an_empty_close_is_pick_failed():
    s, world = stack()
    # The cameras see the red cube 4 cm away from where it lies.
    world.body("red").xy = np.array([0.2, -0.08])
    objects = world.objects

    def shifted() -> dict[str, ObjectPose]:
        out = objects()
        p = out["red"]
        out["red"] = ObjectPose(np.array([0.2, -0.04, p.position[2]]), p.rotation, p.source, p.age_s)
        return out

    world.objects = shifted  # type: ignore[method-assign]
    actions, _ = run(s, world)

    assert not actions[-1].success and s.place.outcome == "pick_failed"
    assert s.status.stopped_because == "chwyt pusty - blok przesuniety albo obok szczek"


def test_a_cube_taken_away_on_the_way_is_dropped():
    s, world = stack(wait_s=1.0)

    def take(world: World, s: PickPlaceStrategy) -> None:
        red = world.body("red")
        red.gripped, red.seen = False, False
        red.xy, red.z = np.array([0.5, 0.5]), CUBE[2]

    actions, _ = run(s, world, on_phase={"carry_over": take})

    assert not actions[-1].success and s.place.outcome == "dropped" and s.grasp.drops == 1
    assert s.status.stopped_because.startswith("blok zabrany ze szczeki")


def slip(world: World, s: PickPlaceStrategy) -> None:
    """The red cube slips out of the jaw onto the table, once."""
    if s.grasp.drops == 0:
        red = world.body("red")
        red.gripped = False
        red.xy, red.z = np.array([0.19, 0.0]), CUBE[2]


def test_a_cube_dropped_on_the_way_is_picked_again_and_placed():
    s, world = stack()
    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": slip})

    assert actions[-1].success and s.place.outcome == "placed"
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1
    assert s.place.retries == 1 and s.place.retried == ("dropped",)
    assert phases.count("carry_up") == 2
    assert world.body("red").z == pytest.approx(world.body("blue").top + CUBE[2])
    assert s.status.stopped_because.endswith("; ponowienia: 1 z 2")


def test_with_no_retry_left_a_cube_dropped_on_the_way_is_dropped():
    s, world = stack(place_retries=0)
    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": slip})

    assert not actions[-1].success and s.place.outcome == "dropped" and s.grasp.recoveries == 0
    assert s.status.stopped_because == "blok wypadl ze szczeki w drodze" and phases[-1] == "recover"


def push_off(world: World, s: PickPlaceStrategy) -> None:
    """Knock the placed red cube off onto the table beside the blue one."""
    red = world.body("red")
    red.xy, red.z = np.array([0.2, 0.11]), CUBE[2]


def test_a_cube_that_falls_off_after_the_release_is_fell_off():
    s, world = stack(place_retries=0)
    actions, phases = run(s, world, on_phase={"park": push_off})

    assert not actions[-1].success and s.place.outcome == "fell_off" and "retry" not in phases
    assert s.status.stopped_because.startswith("blok 'red' nie lezy na 'blue': ")
    assert s.place.height_error_m == pytest.approx(-2 * CUBE[2]) and s.place.retries == 0


def test_a_cube_that_falls_off_is_picked_again_and_placed():
    s, world = stack()

    def once(world: World, s: PickPlaceStrategy) -> None:
        if s.place.retries == 0:
            push_off(world, s)

    actions, phases = run(s, world, seconds=120.0, on_phase={"park": once})
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert s.place.retries == 1 and s.place.retried == ("fell_off",)
    assert phases.count("retry") == 1 and phases.count("verify") == 2
    assert red.z == pytest.approx(blue.top + CUBE[2]) and np.linalg.norm(red.xy - blue.xy) < 0.002
    # The fake squeeze is not exact, so only the blue cube counts for hits after the first lift.
    assert s.status.stopped_because.endswith("; ponowienia: 1 z 2")
    assert not [hit for hit in world.hits if hit[1] != "red"]
    # The new pick got the whole attempt budget.
    assert s.grasp_settings.max_attempts == s.status.attempt - 1 + 3


def test_a_cube_that_falls_off_a_stack_is_picked_again_with_the_hand_over_it_clear_of_the_stack():
    """Over a stack taller than the new hover, the hand first moves over the fallen cube at a height where
    the jaw clears the stack."""
    s, world = tower()
    blue = world.body("blue")
    notes: list[str] = []
    lows: list[float] = []

    def once(world: World, s: PickPlaceStrategy) -> None:
        if s.place.retries == 0:
            push_off(world, s)

    def note(world: World, s: PickPlaceStrategy) -> None:
        notes.append(s._note())

    def over(world: World, s: PickPlaceStrategy) -> None:
        if s.status.phase == "transit":
            lows.append(jaw_low(world.joints) - blue.top)

    actions, phases = run(s, world, seconds=120.0, on_phase={"park": once, "transit": note}, watch=over)
    red = world.body("red")

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert phases.count("transit") == 1 and s.place.retries == 1
    assert notes and notes[0].startswith("przenosze reke nad blok: ")
    assert notes[0].endswith(" | ponowienie 1 z 2")
    assert lows and min(lows) > 0.02 - 0.001
    assert red.z == pytest.approx(blue.top + CUBE[2]) and np.linalg.norm(red.xy - blue.xy) < 0.002
    assert not [hit for hit in world.hits if hit[1] != "red"]


def test_a_cube_that_keeps_falling_off_is_fell_off_once_the_retries_run_out():
    s, world = stack()
    actions, phases = run(s, world, seconds=180.0, on_phase={"park": push_off})

    assert not actions[-1].success and s.place.outcome == "fell_off"
    assert s.place.retries == 2 and s.place.retried == ("fell_off", "fell_off")
    assert phases.count("retry") == 2 and phases.count("verify") == 3
    assert s.status.stopped_because.startswith("blok 'red' nie lezy na 'blue': ")
    assert s.status.stopped_because.endswith("; ponowienia: 2 z 2")


def test_a_fallen_cube_is_not_picked_again_without_a_fresh_pose_of_the_target():
    s, world = stack(target_wait_s=1.0)

    def hide(world: World, s: PickPlaceStrategy) -> None:
        push_off(world, s)
        world.body("blue").seen = False

    actions, phases = run(s, world, on_phase={"park": hide})

    assert s.place.outcome == "fell_off" and s.place.retries == 0 and phases[-1] == "retry"
    assert s.status.stopped_because.endswith(" - bez ponowienia: cel 'blue' niewidoczny w kamerach")


def test_a_target_knocked_during_the_place_is_target_moved():
    s, world = stack()

    def knock(world: World, s: PickPlaceStrategy) -> None:
        for label in ("red", "blue"):
            world.body(label).xy = world.body(label).xy + np.array([0.0, 0.02])

    actions, _ = run(s, world, on_phase={"park": knock})

    assert not actions[-1].success and s.place.outcome == "target_moved"
    assert s.status.stopped_because == "cel 'blue' przesuniety o 20 mm przy odkladaniu"


def test_a_target_that_keeps_moving_at_the_hover_is_target_moved_and_the_cube_goes_back():
    s, world = stack()

    def nudge(world: World, s: PickPlaceStrategy) -> None:
        world.body("blue").xy = world.body("blue").xy + np.array([0.0, 0.01])

    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": nudge})

    assert s.place.outcome == "target_moved" and s.place.replans == 2 and s.place.put_back
    assert phases.count("latch") == 3 and phases.count("carry_over") == 4
    assert world.body("red").z == pytest.approx(CUBE[2]) and not world.hits


def test_a_target_moved_a_little_at_the_hover_is_followed():
    s, world = stack()

    def nudge(world: World, s: PickPlaceStrategy) -> None:
        if s.place.replans == 0:
            world.body("blue").xy = world.body("blue").xy + np.array([0.0, 0.01])

    actions, _ = run(s, world, seconds=90.0, on_phase={"carry_over": nudge})
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.replans == 1
    assert np.linalg.norm(red.xy - blue.xy) < 0.002


def test_a_table_spot_out_of_reach_is_unreachable_before_the_pick():
    s = built(place_x_m=0.6, place_y_m=0.3)
    world = World([cube("red", PICK_XY)])
    actions, phases = run(s, world)

    assert s.place.outcome == "unreachable" and "open" not in phases
    assert "poza zasiegiem" in s.status.stopped_because and not world.body("red").gripped


def test_a_block_out_of_reach_is_unreachable():
    s = built(place_x_m=0.22, place_y_m=0.08)
    world = World([cube("red", (0.6, 0.3))])
    run(s, world)

    assert s.place.outcome == "unreachable" and "poza zasiegiem" in s.status.stopped_because


def test_a_stop_while_carrying_is_stopped():
    s, world = stack()
    actions, _ = run(s, world, on_phase={"carry_over": lambda world, s: s.stop("zatrzymana")})

    assert s.place.outcome == "stopped" and s.status.phase == "done"


# ------------------------------------------------------------ the judge
def choice(option: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    return {"type": "choice", "choice": option, "confidence": confidence,
            "probabilities": {o: top if o == option else (1 - top) / (n - 1) for o in options}}


class Held:
    """A judge that always sees the block held and lifts it."""

    name = "jev"

    def __init__(self) -> None:
        self.points: list[str] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.points.append(state["point"])
        answers = {HELD: choice("block_held", 0.95, HELD_OPTIONS), NEXT: choice("lift", 0.9, NEXT_OPTIONS)}
        return checked_reply(self.name, answers, "fake", {}, questions)


def test_jev_judges_the_close_and_the_lift_before_the_place():
    judge = Held()
    s, world = stack(judge="jev", judge_obj=judge)
    actions, _ = run(s, world)

    assert actions[-1].success and judge.points == ["close", "lift"]
    assert [d.state["point"] for d in s.judging.decisions] == ["close"] * 2 + ["lift"] * 2


def test_the_notes_are_polish_per_place_phase():
    s, world = stack()
    notes: dict[str, str] = {}

    def note(world: World, s: PickPlaceStrategy) -> None:
        notes[s.status.phase] = s._note()

    run(s, world, on_phase={"carry_up": note, "carry_over": note, "park": note})

    assert notes["carry_over"].startswith("niose blok nad 'blue': ")
    assert notes["carry_up"].startswith("unosze blok do przeniesienia: ")
    assert notes["park"].startswith("odsuwam ramie od bloku: ")
    assert wrap(math.pi / 2) == pytest.approx(math.pi / 2)


# ------------------------------------------------------------ at a pose
def bridge(**settings: Any) -> tuple[PickPlaceStrategy, World]:
    """A red cuboid to lay across cubes 'a' and 'b', which stand a cube's width apart along x."""
    d = 2 * CUBE[0]
    world = World([Body("red", CUBOID, np.array(PICK_XY), CUBOID[2], math.radians(30.0)),
                   cube("a", (TARGET_XY[0] - d, TARGET_XY[1])), cube("b", (TARGET_XY[0] + d, TARGET_XY[1]))])
    settings.setdefault("supports", "a, b")
    s = built(block="cuboid", place_x_m=TARGET_XY[0], place_y_m=TARGET_XY[1], place_z_m=2 * CUBE[2],
              **settings)
    return s, world


def test_a_cube_goes_to_a_table_pose_turned_to_its_yaw():
    s = built(place_x_m=0.22, place_y_m=0.08, place_yaw_deg=30.0)
    world = World([cube("red", PICK_XY)])
    actions, phases = run(s, world)
    red = world.body("red")

    assert actions[-1].success and s.place.outcome == "placed" and "latch" not in phases
    assert np.linalg.norm(red.xy - np.array([0.22, 0.08])) < 0.002 and red.z == pytest.approx(CUBE[2])
    assert abs(fold(red.yaw - math.radians(30.0), math.pi / 2)) < math.radians(1.0)
    assert s.place.target_yaw_deg == pytest.approx(30.0) and s.place.rests_on == ()
    assert s.place.placed_tilt_deg == pytest.approx(0.0)


def test_a_cuboid_goes_across_two_cubes():
    s, world = bridge()
    actions, phases = run(s, world)
    red, a, b = world.body("red"), world.body("a"), world.body("b")

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert phases[phases.index("carry_over"):] == ["carry_over", "latch", "place", "let_go", "park", "verify"]
    assert red.z == pytest.approx(a.top + CUBOID[2]) and np.linalg.norm(red.xy - np.array(TARGET_XY)) < 0.002
    # The long side lies along the line from one cube to the other.
    assert abs(fold(red.yaw, math.pi)) < math.radians(1.0)
    assert s.place.rests_on == ("a", "b") and s.place.target_shift_m < 1e-9
    assert s.status.stopped_because.startswith("blok 'red' odlozony na 'a' i 'b': ")
    assert not a.gripped and not b.gripped and not world.hits


def test_the_jaw_closes_across_the_cube_beside_the_place_not_toward_it():
    """A cube set 1 cm to the left of another: of the headings that line it up, only those that keep the
    jaw off the other cube count, so the jaw closes along x, though closing along y needs a smaller turn."""
    place = (TARGET_XY[0], TARGET_XY[1] + 2 * CUBE[0] + 0.01)
    s = built(place_x_m=place[0], place_y_m=place[1])
    world = World([cube("red", PICK_XY, 40.0), cube("blue", TARGET_XY)])
    actions, _ = run(s, world)

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert abs(math.sin(math.radians(s.place.heading_deg))) < math.sin(math.radians(5.0))
    assert np.linalg.norm(world.body("red").xy - np.array(place)) < 0.002
    assert np.linalg.norm(world.body("blue").xy - np.array(TARGET_XY)) < 1e-9


def test_a_place_with_cubes_on_both_sides_the_jaw_could_close_toward_is_unreachable():
    place = (TARGET_XY[0], TARGET_XY[1] + 2 * CUBE[0] + 0.01)
    d = 2 * CUBE[0] + 0.01
    s = built(place_x_m=place[0], place_y_m=place[1])
    world = World([cube("red", PICK_XY), cube("blue", TARGET_XY), cube("green", (place[0] + d, place[1]))])
    _, phases = run(s, world)

    assert s.place.outcome == "unreachable" and "open" not in phases
    assert "szczeka zahaczylaby o '" in s.status.stopped_because, s.status.stopped_because


def test_supports_without_a_pose_before_the_pick_are_target_not_seen_and_the_arm_stays():
    s, world = bridge(wait_for_pose_s=1.0)
    world.body("b").seen = False
    actions, phases = run(s, world)

    assert phases == ["wait"] and s.place.outcome == "target_not_seen"
    assert s.status.stopped_because.startswith("brak uzytecznego polozenia podpory 'b' (brak w obserwacji)")
    arm = SPEC.arm_joints
    assert {n: actions[-1].target[n] for n in arm} == pytest.approx({n: SPEC.home[n] for n in arm})


def test_supports_moved_a_little_carry_the_pose_with_them():
    s, world = bridge()
    shift = np.array([0.003, -0.002])

    def nudge(world: World, s: PickPlaceStrategy) -> None:
        for name in ("a", "b"):
            world.body(name).xy = world.body(name).xy + shift

    actions, _ = run(s, world, on_phase={"carry_over": nudge})
    red = world.body("red")

    assert actions[-1].success and s.place.outcome == "placed" and s.place.replans == 0
    assert s.place.target_xy == pytest.approx(tuple(np.array(TARGET_XY) + shift), abs=1e-6)
    assert np.linalg.norm(red.xy - (np.array(TARGET_XY) + shift)) < 0.002


def test_supports_moved_apart_send_the_block_back():
    s, world = bridge()

    def push(world: World, s: PickPlaceStrategy) -> None:
        world.body("b").xy = world.body("b").xy + np.array([0.012, 0.0])

    actions, phases = run(s, world, on_phase={"carry_over": push})
    red = world.body("red")

    assert s.place.outcome == "target_moved" and s.place.put_back and "verify" not in phases
    assert s.status.stopped_because.startswith("podpory 'a' i 'b' przesuniete wzgledem siebie o 12 mm")
    assert not red.gripped and np.linalg.norm(red.xy - np.array(PICK_XY)) < 0.01


def test_a_support_knocked_at_the_release_is_target_moved():
    s, world = bridge()

    def knock(world: World, s: PickPlaceStrategy) -> None:
        world.body("a").xy = world.body("a").xy + np.array([-0.015, 0.0])

    run(s, world, on_phase={"park": knock})

    assert s.place.outcome == "target_moved" and s.place.target_shift_m == pytest.approx(0.015, abs=1e-6)
    assert s.status.stopped_because == "podpora 'a' przesunieta o 15 mm przy odkladaniu"


def test_a_block_left_leaning_at_a_pose_fell_off():
    s, world = bridge()
    objects = world.objects

    def lean(world: World, s: PickPlaceStrategy) -> None:
        def leaning() -> dict[str, ObjectPose]:
            out = objects()
            p = out["red"]
            c, sn = math.cos(math.radians(12.0)), math.sin(math.radians(12.0))
            tilted = np.array([[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]])
            out["red"] = ObjectPose(p.position, p.rotation @ tilted, p.source, p.age_s)
            return out

        world.objects = leaning  # type: ignore[method-assign]

    run(s, world, on_phase={"park": lean})

    assert s.place.outcome == "fell_off" and s.place.placed_tilt_deg == pytest.approx(12.0)
    assert s.place.retries == 0
    assert s.status.stopped_because.endswith(", pochylony o 12 st. - bez ponowienia: blok pochylony")


def test_the_latch_note_names_the_supports():
    s, world = bridge()
    notes: dict[str, str] = {}

    def note(world: World, s: PickPlaceStrategy) -> None:
        notes[s.status.phase] = s._note()

    run(s, world, on_phase={"carry_over": note})
    # The latch passes within one tick here, so its note is read with the phase set.
    s.status.phase = "latch"

    assert notes["carry_over"].startswith("niose blok nad 'a' i 'b': ")
    assert s._note() == "sprawdzam podpory 'a' i 'b' w kamerach"


# ------------------------------------------------------------ the approach table
#: The stub table's levels: where a held cube lands on the table, a cube and two cubes [m].
TABLE_LEVELS = [{"surface_m": 0.0, "tcp_z_m": 0.0221}, {"surface_m": 0.029, "tcp_z_m": 0.0511},
                {"surface_m": 0.058, "tcp_z_m": 0.0801}]
#: Corners of the stub table: a grid over the pick and the place [m, base frame].
TABLE_XY = [(x, y) for x in (0.15, 0.2, 0.25) for y in (-0.1, -0.05, 0.0, 0.05, 0.1)]
#: The fake cameras' offset in the table tests [m]; a correction of minus that puts the hand on the block.
SEEN_OFF = (0.006, -0.004)


def approach_file(folder: Path, delta: tuple[float, float] = (-SEEN_OFF[0], -SEEN_OFF[1]),
                  directions: tuple[str, ...] = ("ccw",), preferred: str = "ccw",
                  xy: list[tuple[float, float]] | None = None, **protocol: Any) -> str:
    """A stub approach table with the same correction everywhere, saved in `folder`."""
    table = ApproachTable.new(robot=SPEC.name, stand="test", board={},
                              protocol={"lead_m": 0.03, "carry_clearance_m": 0.03, "place_gap_m": 0.002,
                                        "place_speed_mps": 0.02, **protocol}, wrist={}, levels=TABLE_LEVELS)
    table.data["samples"] = [{"spot": 1, "corner": k, "xy_m": list(p), "level": level, "direction": d,
                              "repeat": r, "delta_m": [*delta, 0.0]}
                             for k, p in enumerate(xy or TABLE_XY) for level in range(3) for d in directions
                             for r in (1, 2)]
    table.data["complete"] = True
    table.data["preferred_direction"] = preferred
    return str(table.save(folder / "approach.json"))


def table_stack(folder: Path, **settings: Any) -> tuple[PickPlaceStrategy, World]:
    """`stack` with the stub table, the cameras `SEEN_OFF` off."""
    s, world = stack(approach_file=approach_file(folder), **settings)
    world.offset = SEEN_OFF
    return s, world


def leg_points(world: World, s: PickPlaceStrategy, path: list[tuple[str, np.ndarray]]) -> None:
    path.append((s.status.phase, s.tcp_cmd.copy()))


def on_leg(points: list[np.ndarray], end: np.ndarray, d: np.ndarray, lead: float) -> bool:
    """Whether `points` run along `d` into `end` over the last `lead` metres, on the line."""
    rel = np.array([p[:2] - end[:2] for p in points])
    along, across = rel @ d, rel @ np.array([-d[1], d[0]])
    return bool(np.all(np.abs(across) < 5e-4) and np.all(along <= 5e-4) and along.min() <= -lead + 5e-4
                and np.all(np.diff(along) >= -1e-5))


def test_the_table_sends_the_hand_where_the_blocks_really_are_and_the_check_stays_in_camera_poses(tmp_path):
    """The cameras see every block 7 mm off. The table's correction moves the pick's first aim and the place's
    hover and release, so the red cube lands on the blue one; the target and the check after the release
    stay where the cameras see them."""
    s, world = table_stack(tmp_path)
    actions, phases = run(s, world)
    red, blue, place = world.body("red"), world.body("blue"), s.place

    assert actions[-1].success and place.outcome == "placed", s.status.stopped_because
    assert np.linalg.norm(red.xy - blue.xy) < 0.002
    assert place.target_xy == pytest.approx(tuple(blue.xy + np.array(SEEN_OFF)), abs=1e-9)
    assert place.correction_from == FROM_TABLE and place.correction_m == pytest.approx((-0.006, 0.004))
    assert place.direction == "ccw" and place.table_problem == "" and place.table_miss == ""
    assert place.error_m < 0.002 and place.lead_m == pytest.approx(0.03)
    assert s.grasp.lead_m == pytest.approx(0.03)
    assert s._first_shifted == pytest.approx((-0.006, 0.004))
    assert phases[phases.index("carry_over"):] == ["carry_over", "latch", "place", "let_go", "park", "verify"]


def test_the_final_leg_runs_along_the_direction_into_the_pick_and_the_place_hovers(tmp_path):
    s, world = table_stack(tmp_path)
    path: list[tuple[str, np.ndarray]] = []
    run(s, world, watch=lambda world, s: leg_points(world, s, path))
    plan = s._place_plan
    assert plan is not None and plan.leg is not None

    # From where the approach stopped, through the hover the jaw opens over.
    k = next(i for i, (phase, _) in enumerate(path) if phase == "open")
    straddle = [path[k - 1][1]] + [p for phase, p in path[k:] if phase == "open"]
    assert s.grasp.block_xy is not None
    d_pick = direction_vector("ccw", s.grasp.block_xy, PAN_XY)
    assert on_leg(straddle, straddle[-1], d_pick, 0.03)
    over = [p for phase, p in path if phase == "carry_over"]
    lead = plan.lead_point
    assert lead is not None
    k = min(range(len(over)), key=lambda i: float(np.linalg.norm(over[i] - lead)))
    assert float(np.linalg.norm(over[k] - lead)) < 5e-4
    assert on_leg(over[k:], plan.hover, plan.leg / 0.03, 0.03 - 5e-4)


def test_without_a_table_or_a_direction_the_place_is_planned_as_before():
    s, world = stack()
    actions, phases = run(s, world)
    plan, place = s._place_plan, s.place
    assert plan is not None and actions[-1].success

    assert plan.leg is None and plan.source == "" and np.all(plan.correction == 0.0)
    assert np.allclose(plan.hover[:2], plan.tcp_xy) and place.correction_from == "" and place.lead_m == 0.0
    assert s.grasp.lead_m == 0.0 and place.direction == "" and place.table_problem == ""
    assert phases[:3] == ["wait", "approach", "open"]
    release, want, _ = place_heights(plan.top_m, place.hang_m, 0.002, GRIP_SLIP_M, TABLE_TOP_M, 0.03,
                                     MIN_CARRY_CLEAR_M, over=s._bottom())
    assert place.release_z == pytest.approx(release) and place.carry_z == pytest.approx(want)


def test_a_direction_without_a_table_runs_the_legs_without_a_correction():
    s, world = stack(approach_direction="cw")
    actions, _ = run(s, world)

    assert actions[-1].success and s.place.direction == "cw" and s.place.correction_from == ""
    assert s.place.lead_m == pytest.approx(0.03) and s.grasp.lead_m == pytest.approx(0.03)
    assert np.linalg.norm(world.body("red").xy - world.body("blue").xy) < 0.002


@pytest.mark.parametrize("problem, expect", [
    ("missing", "tablica podejscia nieczytelna: "),
    ("robot", "tablica podejscia nieuzyta: made on robot 'so100', not 'so101'"),
    ("partial", "tablica podejscia nieuzyta: the calibration did not finish"),
    ("no direction", "tablica podejscia bez kierunku podejscia"),
])
def test_a_table_that_cannot_be_used_leaves_a_note_and_the_run_goes_on(tmp_path, problem, expect):
    path = approach_file(tmp_path, preferred="" if problem == "no direction" else "ccw")
    data = json.loads(Path(path).read_text())
    if problem == "robot":
        data["robot"] = "so100"
    if problem == "partial":
        data["complete"] = False
    Path(path).write_text(json.dumps(data))
    s, world = stack(approach_file=str(tmp_path / "nothing.json") if problem == "missing" else path)
    actions, _ = run(s, world)

    assert actions[-1].success and s.place.table_problem.startswith(expect), s.place.table_problem
    assert s._table is None and s.place.correction_from == "" and s.place.direction == ""


def test_settings_unlike_the_tables_protocol_leave_a_note(tmp_path):
    s, world = stack(approach_file=approach_file(tmp_path, lead_m=0.05))
    world.offset = SEEN_OFF
    actions, _ = run(s, world)

    assert actions[-1].success and s.place.correction_from == FROM_TABLE
    assert s.place.table_note == "tablica zmierzona z innymi ustawieniami: lead_m 0.05/0.03"


def test_a_place_outside_the_table_takes_the_pick_correction_or_none(tmp_path):
    """The table covers only the pick: the place takes the pick's own correction when the wrist camera
    centred the jaws, and none with `pick_shift: off`."""
    pytest.importorskip("cv2")
    near_pick = [(x, y) for x in (0.17, 0.2, 0.23) for y in (-0.07, -0.04, -0.01)]
    path = approach_file(tmp_path, xy=near_pick)
    for shift, source in (("always", FROM_PICK), ("auto", FROM_PICK), ("off", "")):
        s, world = stack(approach_file=path, wrist_camera="wrist", pick_shift=shift)
        world.offset = SEEN_OFF
        red = world.body("red")
        s.measure = lambda rgb, joints, ref_xy=None, red=red: Measurement(  # type: ignore[method-assign]
            red.xy.copy(), red.yaw, 0.9, 1000, 1.0, np.zeros(2))
        actions, _ = run(s, world, frames={"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)})
        place = s.place

        assert place.correction_from == source and place.table_miss == "outside", (shift, place)
        if source:
            assert place.correction_m == pytest.approx((-0.006, 0.004), abs=1e-6)
            assert np.linalg.norm(red.xy - world.body("blue").xy) < 0.002


def test_the_fallback_goes_pick_then_table_without_its_leg_then_none(tmp_path):
    s, world = stack(approach_file=approach_file(tmp_path))
    s.start(run_info(), Observation(t=0.0, step=0, joints=dict(world.joints), command=dict(world.command),
                                    objects=world.objects()))
    xy, z = np.array(TARGET_XY), 0.0511
    table = s._corrections(FROM_TABLE, xy, z)
    assert table is not None and table.source == FROM_TABLE and table.leg_required
    assert table.leg == pytest.approx(0.03 * direction_vector("ccw", xy, PAN_XY))

    s._closed_seen, s._detector_xy, s._closed_aim = True, np.array(PICK_XY), np.array(PICK_XY) + 0.005
    pick = s._corrections("", xy, z)
    assert pick is not None and pick.source == FROM_PICK and pick.delta == pytest.approx((0.005, 0.005))
    assert pick.miss == "lead" and not pick.leg_required

    s._closed_seen = False
    plain = s._corrections("", xy, z)
    assert plain is not None and plain.source == FROM_TABLE and plain.leg is None
    s._putting_back = True
    assert s._corrections(FROM_TABLE, xy, z) is None
    back = s._corrections("", xy, z)
    assert back is not None and back.source == "" and np.all(back.delta == 0.0) and back.leg is None


def test_a_rise_the_arm_reaches_in_height_but_not_across_still_leaves_the_placed_block():
    """After the release the open hand rises; where the arm gets to the height but stays 3 mm off across,
    the run leaves from there instead of waiting for the time limit."""
    s, world = stack()
    drive = s._drive

    def off_across(goal: np.ndarray, step_m: float, *, rotation: np.ndarray | None = None) -> Action:
        if s.status.phase == "let_go" and s._eased:
            action = drive(np.array([goal[0] + 0.003, goal[1], goal[2]]), step_m, rotation=rotation)
            s.status.out_of_reach = True
            return action
        return drive(goal, step_m, rotation=rotation)

    s._drive = off_across  # type: ignore[method-assign]
    actions, phases = run(s, world, seconds=40.0)

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert phases[phases.index("let_go"):][:2] == ["let_go", "park"]
    assert np.linalg.norm(world.body("red").xy - world.body("blue").xy) < 0.002


def checked(path: str, off: tuple[float, float], **settings: Any) -> PickPlaceStrategy:
    """A run with the table at `path` whose pick's own correction lies `off` from the table's [m]."""
    s, world = stack(approach_file=path, **settings)
    s.start(run_info(), Observation(t=0.0, step=0, joints=dict(world.joints), command=dict(world.command),
                                    objects=world.objects()))
    s._pick_hit, s._first_shifted = np.array([-0.006, 0.004]), np.array([-0.006, 0.004])
    s._closed_seen, s._detector_xy = True, np.array(PICK_XY)
    s._closed_aim = np.array(PICK_XY) + np.array([-0.006, 0.004]) + np.array(off)
    s._check_table()
    return s


def test_a_table_two_picks_in_a_row_disagree_with_is_not_trusted_for_the_run(tmp_path):
    """One pick over `table_check_m` keeps the table and says so; the next one in a row with the same table
    sends the run to the pick's own correction. A pick that agrees starts the count again."""
    path = approach_file(tmp_path)
    u, here = np.array([1.0, 0.0]), np.array(PICK_XY)
    s = checked(path, (0.003, 0.0))
    assert s._table_trusted and s.place.table_check_m == pytest.approx(0.003) and s.place.table_note == ""
    assert s._hand_from_table(here, u, here, 0.0) is not None

    s = checked(path, (0.0, 0.007))
    assert s._table_trusted and s.place.table_trusted and s.place.table_check_m == pytest.approx(0.007)
    assert s.place.table_note == "chwyt rozni sie od tablicy o 7 mm (1 raz z rzedu) - zostaje tablica"
    assert isinstance(s._lookup(np.array(TARGET_XY), 0.0511), Hit)
    # The table's correction stays; its account of the block in the hand does not.
    assert s._hand_from_table(here, u, here, 0.0) is None

    s = checked(path, (0.0, 0.008))
    assert not s._table_trusted and not s.place.table_trusted
    assert s.place.table_note == ("tablica nie zgadza sie z chwytem 2 razy z rzedu (teraz o 8 mm) - poprawka "
                                  "z chwytu")
    assert s._lookup(np.array(TARGET_XY), 0.0511) == "check"
    fallback = s._corrections("", np.array(TARGET_XY), 0.0511)
    assert fallback is not None and fallback.source == FROM_PICK and fallback.miss == "check"

    assert checked(path, (0.002, 0.0))._table_trusted
    assert checked(path, (0.0, 0.007))._table_trusted


def test_a_rewritten_table_starts_the_count_again_and_one_run_may_be_enough(tmp_path):
    path = approach_file(tmp_path)
    assert checked(path, (0.0, 0.007))._table_trusted
    data = json.loads(Path(path).read_text())
    data["updated"] = "2030-01-01T00:00:00"
    Path(path).write_text(json.dumps(data))
    assert checked(path, (0.0, 0.007))._table_trusted
    assert not checked(path, (0.0, 0.007), table_check_runs=1)._table_trusted


def test_a_look_that_moved_the_aim_across_the_jaws_does_not_move_the_place(tmp_path):
    """The table's first aim is right and the wrist look then moves the aim 2.5 mm across the jaws: the
    closed jaws leave the cube where it was across them. The place puts the cube where the table's aim says
    it sits in the hand, so it lands on the target."""
    pytest.importorskip("cv2")
    s, world = table_stack(tmp_path, wrist_camera="wrist")
    red = world.body("red")

    def measure(rgb, joints, ref_xy=None):
        psi = heading(KIN, joints)
        across = 0.0025 * np.array([-math.sin(psi), math.cos(psi)])
        return Measurement(red.xy + across, red.yaw, 0.9, 1000, 1.0, np.zeros(2))

    s.measure = measure  # type: ignore[method-assign]
    actions, _ = run(s, world, frames={"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)})
    place = s.place

    assert actions[-1].success and place.outcome == "placed", s.status.stopped_because
    assert place.correction_from == FROM_TABLE and place.table_trusted
    assert place.table_check_m == pytest.approx(0.0025, abs=2e-4)
    assert place.hand_shift_m == pytest.approx(0.0025, abs=2e-4)
    assert np.linalg.norm(red.xy - world.body("blue").xy) < 0.0005


def test_the_pick_correction_keeps_the_look_s_aim_in_the_hand(tmp_path):
    """Where the place takes the pick's own correction, which holds the look's move, the cube stays in the
    hand where the look put it."""
    pytest.importorskip("cv2")
    near_pick = [(x, y) for x in (0.17, 0.2, 0.23) for y in (-0.07, -0.04, -0.01)]
    s, world = stack(approach_file=approach_file(tmp_path, xy=near_pick), wrist_camera="wrist")
    world.offset = SEEN_OFF
    red = world.body("red")
    s.measure = lambda rgb, joints, ref_xy=None: Measurement(  # type: ignore[method-assign]
        red.xy.copy(), red.yaw, 0.9, 1000, 1.0, np.zeros(2))
    actions, _ = run(s, world, frames={"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)})

    assert actions[-1].success and s.place.correction_from == FROM_PICK
    assert s._table_hand is not None and s.place.hand_shift_m == pytest.approx(0.0, abs=1e-4)
    assert np.linalg.norm(red.xy - world.body("blue").xy) < 0.002


def test_without_the_pick_correction_a_disagreeing_table_stays(tmp_path):
    s, world = stack(approach_file=approach_file(tmp_path), pick_shift="off")
    s.start(run_info(), Observation(t=0.0, step=0, joints=dict(world.joints), command=dict(world.command),
                                    objects=world.objects()))
    s._pick_hit, s._first_shifted = np.array([-0.006, 0.004]), np.array([-0.006, 0.004])
    s._closed_seen, s._detector_xy = True, np.array(PICK_XY)
    s._closed_aim = np.array(PICK_XY) + np.array([0.004, 0.004])
    s._check_table()

    assert s._table_trusted and s.place.table_check_m == pytest.approx(0.01)


def test_a_table_aimed_pick_neither_applies_nor_saves_the_residual(tmp_path):
    pytest.importorskip("cv2")
    residual = tmp_path / "residual.json"
    residual.write_text(json.dumps({"mean_m": [0.01, 0.01], "runs": 3}))
    before = residual.read_text()
    s, world = stack(approach_file=approach_file(tmp_path), wrist_camera="wrist", residual_file=str(residual))
    world.offset = SEEN_OFF
    red = world.body("red")
    s.measure = lambda rgb, joints, ref_xy=None: Measurement(  # type: ignore[method-assign]
        red.xy.copy(), red.yaw, 0.9, 1000, 1.0, np.zeros(2))
    actions, _ = run(s, world, frames={"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)})

    assert actions[-1].success and s._table_aimed and not s._saves_residual
    assert s._first_shifted == pytest.approx((-0.006, 0.004)) and s.wrist.residual_m == (0.0, 0.0)
    assert residual.read_text() == before


def test_a_target_moved_a_little_at_the_hover_runs_the_leg_again_without_a_replan(tmp_path):
    s, world = table_stack(tmp_path)
    nudged: list[bool] = []

    def nudge(world: World, s: PickPlaceStrategy) -> None:
        if not nudged:
            nudged.append(True)
            world.body("blue").xy = world.body("blue").xy + np.array([0.0, 0.003])

    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": nudge})
    red, blue = world.body("red"), world.body("blue")

    assert actions[-1].success and s.place.replans == 0, s.status.stopped_because
    assert phases[phases.index("carry_over"):] == ["carry_over", "latch", "carry_over", "place", "let_go",
                                                   "park", "verify"]
    assert np.linalg.norm(red.xy - blue.xy) < 0.002


def test_the_correction_shows_in_the_carry_note(tmp_path):
    s, world = table_stack(tmp_path)
    notes: dict[str, str] = {}

    def note(world: World, s: PickPlaceStrategy) -> None:
        notes[s.status.phase] = s._note()

    run(s, world, on_phase={"carry_over": note})

    assert notes["carry_over"].endswith("; poprawka z tablicy 7 mm (ccw)"), notes["carry_over"]



def test_after_the_wrist_look_moves_the_aim_the_hand_comes_along_the_leg_again():
    """The detector sees the red cube 4 mm off; the wrist camera sees where it is. The hand goes back to the
    start of the leg for the new hover and runs the leg again before it straddles."""
    pytest.importorskip("cv2")
    s, world = stack(approach_direction="ccw", wrist_camera="wrist", close_rounds=0)
    world.offset = (0.004, 0.0)
    red = world.body("red")
    s.measure = lambda rgb, joints, ref_xy=None: Measurement(  # type: ignore[method-assign]
        red.xy.copy(), red.yaw, 0.9, 1000, 1.0, np.zeros(2))
    path: list[tuple[str, np.ndarray]] = []
    actions, _ = run(s, world, watch=lambda world, s: leg_points(world, s, path),
                     frames={"wrist": Frame(np.zeros((48, 64, 3), np.uint8), 0.0)})

    assert actions[-1].success and s.wrist.verdicts == ["centred"] and s.wrist.shift_m > 0.003
    k = next(i for i, (phase, _) in enumerate(path) if phase == "straddle")
    opened = [p for phase, p in path[:k] if phase == "open"]
    assert s.grasp.block_xy is not None
    d = direction_vector("ccw", s.grasp.block_xy, PAN_XY)
    start = opened[-1] - np.array([*(0.03 * d), 0.0])
    j = min(range(len(opened)), key=lambda i: float(np.linalg.norm(opened[i] - start)))
    assert float(np.linalg.norm(opened[j] - start)) < 5e-4
    assert on_leg(opened[j:], opened[-1], d, 0.03 - 5e-4)


def test_a_leg_start_the_arm_does_not_get_to_is_given_up_and_the_run_goes_on(tmp_path):
    s, world = table_stack(tmp_path)
    arrived = s._arrived

    def never_at_a_leg_start(point: np.ndarray) -> bool:
        plan = s._place_plan
        starts = [s._lead_point] + ([plan.lead_point] if plan is not None else [])
        if any(start is not None and np.allclose(point, start) for start in starts):
            return False
        return arrived(point)

    s._arrived = never_at_a_leg_start  # type: ignore[method-assign]
    actions, _ = run(s, world, seconds=120.0)

    assert actions[-1].success and s.place.outcome == "placed", s.status.stopped_because
    assert s.grasp.lead_m == 0.0 and s.place.lead_m == 0.0 and s.place.correction_from == FROM_TABLE
    assert np.linalg.norm(world.body("red").xy - world.body("blue").xy) < 0.002
