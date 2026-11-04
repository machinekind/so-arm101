"""Test pick-and-place phases and outcomes with a fake supervisor, arm, jaw and cameras.

The supervisor follows each target at once. A squeezed block takes the jaw's pads flat and
follows the hand. A held block stalls the arm on a support, and a released one lands on the
support under its centre. After the lift the world records each fingertip that goes into a resting
block. Twin integration tests cover physics and perception.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm

pytest.importorskip("mujoco")

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import heading, wrap  # noqa: E402
from soarm_lab.strategies.grasp_jev import HELD, HELD_OPTIONS, NEXT, NEXT_OPTIONS, GraspJevSettings  # noqa: E402
from soarm_lab.strategies.jev import KEY_ENV  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.pick_place import (  # noqa: E402
    MIN_CARRY_CLEAR_M,
    OUTCOMES,
    PickPlaceSettings,
    PickPlaceStrategy,
    known_yaw,
    yaw_of,
)
from soarm_lab.strategies.wrist import WristGraspSettings, fold  # noqa: E402

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

    def under(self, point: np.ndarray) -> bool:
        """Whether `point` lies over the block's footprint."""
        local = self.frame(point)
        return bool(abs(local[0]) < self.half[0] and abs(local[1]) < self.half[1])

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
        for k in (0, 1):
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

    def body(self, label: str) -> Body:
        return next(b for b in self.bodies if b.label == label)

    def objects(self) -> dict[str, ObjectPose]:
        out = {}
        for b in self.bodies:
            if b.seen:
                rotation = rot_z(b.yaw) if b.yaw else np.eye(3)
                source, age = ("w dloni", 0.0) if b.gripped else ("kamery", SEEN_AGE_S)
                out[b.label] = ObjectPose(np.array([*b.xy, b.z]), rotation, source, age)
        return out

    def support(self, point: np.ndarray, held: Body) -> float:
        """The highest top under `point`, the table included [m]."""
        return max([b.top for b in self.bodies if b is not held and not b.gripped and b.under(point)] + [0.0])

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
        on_phase: dict[str, Hook] | None = None, watch: Hook | None = None) -> tuple[list[Action], list[str]]:
    """Start and tick until done. Return the actions and every phase entered, in order.

    `on_phase` hooks run after the tick that entered their phase, `watch` after every world step.
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
                              objects=world.objects()))
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
    assert names[-15:] == ["judge", "place_on", "place_on_block", "place_x_m", "place_y_m", "align_yaw",
                           "carry_clearance_m", "place_gap_m", "place_speed_mps", "release_margin_m",
                           "retreat_m", "fresh_pose_s", "target_wait_s", "verify_wait_s", "place_tol_m"]
    defaults = PickPlaceSettings()
    assert defaults.judge == "jev" and defaults.wrist_camera == "wrist" and defaults.time_limit_s == 240.0
    assert OUTCOMES == ("placed", "not_found", "target_not_seen", "pick_failed", "dropped", "fell_off",
                        "target_moved", "unreachable", "stopped")


@pytest.mark.parametrize("settings, message", [
    ({"judge": "claude"}, "judge must be one of jev, none"),
    ({"place_on_block": "ball"}, "place_on_block must be one of cube, cuboid, cylinder"),
    ({"place_speed_mps": 0.0}, "place_speed_mps must be positive"),
    ({"release_margin_m": -0.001}, "release_margin_m must be positive"),
    ({"place_gap_m": -0.001}, "place_gap_m must be zero or more"),
    ({"place_x_m": math.nan}, "place_x_m must be finite"),
    ({"send_frames": True}, "send_frames needs wrist_camera"),
    ({"place_on": "Red"}, "place_on must name another block than label"),
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
    assert s.status.stopped_because.startswith("blok 'red' odlozony na 'blue': ")
    assert s.place.release_z == pytest.approx(blue.top + 0.002 + s.place.hang_m)
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
    s, world = tower((0.22, 0.0))
    _, phases = run(s, world)

    assert s.place.outcome == "unreachable" and "open" not in phases
    assert s.status.stopped_because.startswith("po odlozeniu szczeki nie mina bloku")


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
    # The cameras put the target 6 mm lower than it stands.
    blue = world.body("blue")
    objects = world.objects

    def lower() -> dict[str, ObjectPose]:
        out = objects()
        if "blue" in out:
            p = out["blue"]
            out["blue"] = ObjectPose(p.position - np.array([0.0, 0.0, 0.006]), p.rotation, p.source, p.age_s)
        return out

    world.objects = lower  # type: ignore[method-assign]
    actions, _ = run(s, world)

    assert s.place.contact and s.place.outcome == "placed" and actions[-1].success
    assert world.body("red").z == pytest.approx(blue.top + CUBE[2])
    assert not world.hits


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


def test_a_cube_dropped_on_the_way_is_picked_again_and_placed():
    s, world = stack()

    def slip(world: World, s: PickPlaceStrategy) -> None:
        if s.grasp.drops == 0:
            red = world.body("red")
            red.gripped = False
            red.xy, red.z = np.array([0.19, 0.0]), CUBE[2]

    actions, phases = run(s, world, seconds=90.0, on_phase={"carry_over": slip})

    assert actions[-1].success and s.place.outcome == "placed"
    assert s.grasp.drops == 1 and s.grasp.recoveries == 1
    assert phases.count("carry_up") == 2
    assert world.body("red").z == pytest.approx(world.body("blue").top + CUBE[2])


def test_a_cube_that_falls_off_after_the_release_is_fell_off():
    s, world = stack()

    def push(world: World, s: PickPlaceStrategy) -> None:
        red = world.body("red")
        red.xy, red.z = np.array([0.2, 0.11]), CUBE[2]

    actions, _ = run(s, world, on_phase={"park": push})

    assert not actions[-1].success and s.place.outcome == "fell_off"
    assert s.status.stopped_because.startswith("blok 'red' nie lezy na 'blue': ")
    assert s.place.height_error_m == pytest.approx(-2 * CUBE[2])


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
