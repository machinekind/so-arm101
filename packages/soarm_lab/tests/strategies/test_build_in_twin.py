"""Build a tower of three cubes, and a bridge, with `build` through the twin's discovery, runner, supervisor
and physics.

A seed puts the cubes on the table: the blue base where a three-high tower fits with the hand straight
down, green and red anywhere in reach. Each block has its own tracker, fed with simulation poses while the
hand is not near it. The plan comes from `plan_json` or a scripted fake planner, so no model, key or camera
is needed. The simulation's own poses check the tower, and that no block sank into the table, flew,
was dragged or stayed in the jaws. A knocked tower has its middle cube moved to the table while the top
one is carried to it. The bridge sets a cube beside another and lays a cuboid across both.

`SOARM_BUILD_CAMPAIGN=<runs>` also runs that many seeds of each scenario and prints the numbers.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.twin
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as twin_scene  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.perception import CubeDetection, ObjectTracker  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import ObjectPose, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.planner.cli import FakePlanner  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.build import BUILT, BuildSettings, BuildStrategy  # noqa: E402
from soarm_lab.strategies.jev import KEY_ENV  # noqa: E402

NAME = "build"
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: The hand hides a block from the cameras nearer than this [m].
HIDDEN_M = 0.05
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
COLOURS = {"blue": (0.2, 0.4, 0.9, 1.0), "green": (0.2, 0.75, 0.3, 1.0), "red": (0.85, 0.25, 0.2, 1.0),
           "yellow": (0.9, 0.8, 0.2, 1.0)}
NAMES = ("blue", "green", "red")
SPEC = load_robot()
KIN = RobotKinematics(SPEC)
HALF = block("cube").half_size[2]
#: The tower's tolerances by truth: centres over each other [m], heights [m] and tilt [deg].
TOWER_XY_M = 0.008
TOWER_Z_M = 0.002
TOWER_TILT_DEG = 3.0
#: Artefact limits: how far a block may sink below its resting height, its speed, and how far the base
#: may move [m, m/s, m].
SINK_M = 0.0007
SPEED_MPS = 0.5
BASE_SHIFT_M = 0.003


def on(pick: str, target: str) -> dict[str, Any]:
    return {"pick": pick, "place": {"relation": "on", "targets": [target], "side": None, "gap_m": None,
                                    "x_m": None, "y_m": None, "yaw_deg": None}, "why": ""}


def at(pick: str, xy: tuple[float, float]) -> dict[str, Any]:
    return {"pick": pick, "place": {"relation": "at", "targets": [], "side": None, "gap_m": None,
                                    "x_m": round(xy[0], 3), "y_m": round(xy[1], 3), "yaw_deg": None},
            "why": ""}


TOWER = [on("green cube", "blue cube"), on("red cube", "green cube")]
TOWER_JSON = ('[{"pick": "green cube", "place": {"relation": "on", "targets": ["blue cube"]}},'
              ' {"pick": "red cube", "place": {"relation": "on", "targets": ["green cube"]}}]')
TASK = "Build a tower of the three cubes: blue at the bottom, green on it, red on top."


@pytest.fixture
def twin(twin_robots):
    with fresh_twin() as twin:
        yield twin


@contextmanager
def fresh_twin() -> Iterator[Twin]:
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


@dataclass
class Clock:
    t: float = 0.0


@dataclass(frozen=True)
class Placed:
    name: str
    xy: tuple[float, float]
    yaw_deg: float = 0.0
    kind: str = "cube"


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          watch: Callable[[], None] | None = None) -> None:
    """Run twin and strategy ticks on a shared simulated clock until the run ends; `watch` after each twin
    step."""
    strategy_every = max(1, round(1.0 / runner.strategy.hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
            if watch is not None:
                watch()
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
        if not runner.status.running:
            break


def add_blocks(twin: Twin, *blocks: Placed) -> None:
    """Add yawed blocks with stiff contacts and jaw sensors, then connect the simulated arm."""
    twin.configure(objects=[twin_scene.Box(b.name, block(b.kind).half_size, b.xy, rgba=COLOURS[b.name],
                                           mass=block(b.kind).mass) for b in blocks],
                   grasp_sensors=[b.name for b in blocks])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        for b in blocks:
            g = model.geom(b.name).id
            model.geom_solref[g] = STIFF_SOLREF
            model.geom_solimp[g][:2] = STIFF_SOLIMP
            model.geom_priority[g] = 1
            yaw = math.radians(b.yaw_deg)
            adr = model.jnt_qposadr[model.body_jntadr[model.body(b.name).id]]
            data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)


def truth(twin: Twin, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The simulated block's position [m, base frame] and rotation."""
    with twin.lock:
        s = twin.scene
        b = s.model.body(name).id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def tilt_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(rotation[2, 2])))))


def move(twin: Twin, name: str, xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
    """Put a cube flat on the table at `xy`, at rest, as if knocked there."""
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        body = model.body(name).id
        adr = model.jnt_qposadr[model.body_jntadr[body]]
        T = twin.scene.T_base2world
        data.qpos[adr:adr + 3] = T[:3, :3] @ np.array([*xy, HALF]) + T[:3, 3]
        yaw = math.radians(yaw_deg)
        data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        data.qvel[model.body_dofadr[body]:][:6] = 0.0
        mujoco.mj_forward(model, data)


def contact(twin: Twin, name: str) -> list[float]:
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def settle(twin: Twin, seconds: float = 1.0) -> None:
    for _ in range(round(seconds / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)


@dataclass
class Cameras:
    """Tracker poses of every block from visible simulation detections, labelled "<colour> <kind>", a cube
    unless `kinds` says otherwise.

    A cube within `HIDDEN_M` of the TCP is not detected. Only the cube nearest the TCP can be taken by a
    closing jaw, as the twin's stacking tracker does.
    """

    twin: Twin
    clock: Clock
    names: tuple[str, ...]
    kinds: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = self.twin.workspace.spec()
        self.trackers = {n: ObjectTracker.for_spec(spec, hold_s=30.0) for n in self.names}
        assert KIN.spec.gripper is not None
        self.g = KIN.spec.joints.index(KIN.spec.gripper)
        gripper = KIN.spec.gripper
        self.closed = float(KIN.to_q({gripper: self.twin.joint_limits()[gripper][0]})[self.g])

    def __call__(self) -> dict[str, ObjectPose]:
        status = self.twin.status
        measured, command = dict(status.measured), dict(status.command) or dict(status.measured)
        T_tcp = KIN.tcp(measured)
        poses = {name: truth(self.twin, name) for name in self.trackers}
        nearest = min(poses, key=lambda n: float(np.linalg.norm(poses[n][0] - T_tcp[:3, 3])))
        out = {}
        for name, tracker in self.trackers.items():
            position, rotation = poses[name]
            seen = None
            if np.linalg.norm(position - T_tcp[:3, 3]) > HIDDEN_M:
                seen = CubeDetection(position, rotation, area_px=0.0, confidence=1.0, n_cameras=2,
                                     t=self.clock.t)
            grip_q, grip_cmd = float(KIN.to_q(measured)[self.g]), float(KIN.to_q(command)[self.g])
            if name != nearest:
                grip_cmd = grip_q
            got = tracker.update(seen, T_tcp, grip_q, grip_cmd, self.closed, self.clock.t)
            if got is None:
                continue
            age = 0.0 if tracker.source == "w dloni" else max(0.0, self.clock.t - tracker.t_seen)
            label = f"{name} {self.kinds.get(name, 'cube')}"
            out[label] = ObjectPose(np.asarray(got[0], float), np.asarray(got[1], float), tracker.source, age)
        return out


# --------------------------------------------------------------------------- scenes and runs


def polar(r: float, bearing: float) -> tuple[float, float]:
    return (r * math.cos(bearing), r * math.sin(bearing))


def free_spot(rng: np.random.Generator, taken: list[tuple[float, float]], gap: float = 0.07,
              radius: tuple[float, float] = (0.16, 0.24), bearing: float = 0.7) -> tuple[float, float]:
    """A random table spot in reach at least `gap` from every point of `taken` [m], or a little less far
    when there is none."""
    for least in (gap, gap - 0.01, gap - 0.02):
        for _ in range(1000):
            xy = polar(rng.uniform(*radius), rng.uniform(-bearing, bearing))
            if all(math.dist(xy, other) >= least for other in taken):
                return xy
    raise AssertionError("no free spot")


def scene(seed: int) -> list[Placed]:
    """The blue base where a three-high tower fits with the hand down; green and red anywhere in reach."""
    rng = np.random.default_rng(seed)
    blue = (rng.uniform(0.19, 0.21), rng.uniform(-0.06, 0.06))
    placed = [Placed("blue", blue, rng.uniform(0.0, 90.0))]
    for name in ("green", "red"):
        placed.append(Placed(name, free_spot(rng, [p.xy for p in placed]), rng.uniform(0.0, 90.0)))
    return placed


@dataclass
class Result:
    """One build: whether the tower stands by truth, what the strategy said, and the artefacts seen."""

    seed: int
    tower: bool
    why: str
    outcome: str
    success: bool
    steps: int
    replans: int
    retries: int
    sim_s: float
    wall_s: float
    knocked: bool = False
    artefacts: list[str] = field(default_factory=list)
    step_outcomes: list[str] = field(default_factory=list)
    reason: str = ""


def tower_problem(twin: Twin) -> str:
    """Why the cubes are no tower by truth (blue on the table, green on it, red on top); '' if they are."""
    (blue, rb), (green, rg), (red, rr) = (truth(twin, n) for n in NAMES)
    problems = []
    for name, lower, upper, level in (("green", blue, green, 2), ("red", green, red, 3)):
        off = float(np.linalg.norm(upper[:2] - lower[:2]))
        if off > TOWER_XY_M:
            problems.append(f"{name} {off * 1000:.0f} mm off centre")
        if abs(upper[2] - (2 * level - 1) * HALF) > TOWER_Z_M:
            problems.append(f"{name} at {upper[2] * 1000:.0f} mm, level {level}")
    if abs(blue[2] - HALF) > TOWER_Z_M:
        problems.append(f"blue at {blue[2] * 1000:.0f} mm")
    tilts = [tilt_deg(r) for r in (rb, rg, rr)]
    if max(tilts) > TOWER_TILT_DEG:
        problems.append(f"tilted {max(tilts):.1f} deg")
    return "; ".join(problems)


def build_tower(twin: Twin, seed: int, *, planned: bool, knock: bool = False, seconds: float = 700.0,
                **extra: Any) -> Result:
    """Run one seeded build of the tower to its end, then judge it by truth; `extra` goes to the settings."""
    placed = scene(seed)
    add_blocks(twin, *placed)
    settle(twin, 0.5)
    rng = np.random.default_rng(1000 + seed)
    clock = Clock()
    settings: dict[str, Any] = {"judge": "none", "wrist_camera": "", **extra}
    planner = None
    if planned:
        # Without a knock it answers every call with the tower, as a model that sticks to its plan.
        tower = {"steps": TOWER, "done": False, "question": ""}
        planner = FakePlanner([tower] if knock else [tower] * 4)
        settings["task"] = TASK
    else:
        settings["plan_json"] = TOWER_JSON
    strategy = BuildStrategy(BuildSettings(**settings), planner=planner, threaded=False)
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, NAMES))
    lowest = {n: math.inf for n in NAMES}
    fastest = {n: 0.0 for n in NAMES}
    last = {n: truth(twin, n)[0] for n in NAMES}
    base0 = last["blue"].copy()
    base_shift = [0.0]
    knocked: list[float] = []

    def watch() -> None:
        for n in NAMES:
            p = truth(twin, n)[0]
            lowest[n] = min(lowest[n], float(p[2]))
            fastest[n] = max(fastest[n], float(np.linalg.norm(p - last[n])) / TWIN_PERIOD)
            last[n] = p
        base_shift[0] = max(base_shift[0], float(np.linalg.norm(last["blue"] - base0)))
        child = strategy.current_step
        if (knock and not knocked and strategy.status.done == 1 and child is not None
                and getattr(getattr(child, "status", None), "phase", "") == "carry_over"
                and not getattr(getattr(child, "place", None), "put_back", True)):
            red_from = next(p.xy for p in placed if p.name == "red")
            spot = free_spot(rng, [p.xy for p in placed if p.name != "green"])
            move(twin, "green", spot, rng.uniform(0.0, 90.0))
            last["green"] = truth(twin, "green")[0]
            knocked.append(clock.t)
            if planner is not None:
                aside = free_spot(rng, [tuple(base0[:2]), spot, red_from], radius=(0.2, 0.24))
                planner.add({"steps": [at("red cube", aside), *TOWER], "done": False, "question": ""})

    t0 = time.monotonic()
    runner.start(threaded=False)
    drive(twin, runner, seconds, clock, watch=watch)
    wall = time.monotonic() - t0
    if runner.status.running:
        runner.stop()
    settle(twin)
    artefacts = []
    for n in NAMES:
        if lowest[n] < HALF - SINK_M:
            artefacts.append(f"{n} sank to {lowest[n] * 1000:.1f} mm")
        if fastest[n] > SPEED_MPS:
            artefacts.append(f"{n} moved at {fastest[n]:.2f} m/s")
        if max(contact(twin, n)) > 0.5:
            artefacts.append(f"{n} in the jaws")
    if base_shift[0] > BASE_SHIFT_M:
        artefacts.append(f"base moved {base_shift[0] * 1000:.1f} mm")
    why = tower_problem(twin)
    st = strategy.status
    return Result(seed, not why, why, st.outcome, st.success, st.done, st.replans, st.retries, clock.t, wall,
                  bool(knocked), artefacts, [r.outcome.split(":")[0] for r in st.records], st.stopped_because)


# --------------------------------------------------------------------------- discovery


def test_the_twin_finds_build_and_builds_it_from_a_fixed_plan_without_a_key(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    assert strategies()[NAME].value == "soarm_lab.strategies.build:BuildStrategy"
    plugin = load_strategy(NAME)
    fields = {f.name: f for f in plugin.fields}

    assert plugin.cls is BuildStrategy and plugin.settings is BuildSettings
    assert plugin.summary == ("Builds with blocks from a sentence or a fixed plan, one pick_place run per "
                              "step, and replans.")
    assert fields["task"].default == "" and fields["plan_json"].default == ""
    assert fields["planner_cli"].options({}) == ("codex", "claude")
    assert not any(f.required for f in fields.values())
    strategy = plugin.create({"plan_json": TOWER_JSON, "judge": "none", "wrist_camera": ""})
    assert isinstance(strategy, BuildStrategy) and strategy.goal is not None


# --------------------------------------------------------------------------- towers in the simulated twin


def check(result: Result) -> None:
    assert result.tower, result
    assert result.outcome == BUILT and result.success, result
    assert not result.artefacts, result


def test_a_fixed_plan_builds_a_tower_of_three_cubes(twin):
    result = build_tower(twin, 0, planned=False)
    check(result)
    assert result.steps == 2 and result.replans == 0


def test_a_scripted_planner_builds_a_tower_of_three_cubes(twin):
    result = build_tower(twin, 1, planned=True)
    check(result)
    assert result.steps == 2


def test_a_tower_knocked_over_mid_build_is_replanned_and_finished(twin):
    result = build_tower(twin, 2, planned=False, knock=True)
    assert result.knocked
    check(result)
    assert result.replans >= 1


def test_a_scripted_planner_replans_a_knocked_tower(twin):
    result = build_tower(twin, 3, planned=True, knock=True)
    assert result.knocked
    check(result)
    assert result.replans >= 1


# --------------------------------------------------------------------------- a bridge in the simulated twin

BRIDGE_JSON = ('[{"pick": "green cube", "place": {"relation": "next_to", "targets": ["blue cube"],'
               ' "side": "left", "gap_m": 0.015}},'
               ' {"pick": "yellow cuboid",'
               '  "place": {"relation": "across", "targets": ["blue cube", "green cube"]}}]')
BRIDGE_BLOCKS = (Placed("blue", (0.2, 0.04), 10.0), Placed("green", (0.18, -0.11), 30.0),
                 Placed("yellow", (0.2, -0.05), 70.0, "cuboid"))


def test_a_fixed_plan_sets_a_cube_beside_another_and_lays_a_cuboid_across_both(twin):
    """Green goes on the table left of blue, then the cuboid across both: by truth the bridge stands level on
    the two cubes, blue stays put, and no block sinks, flies or stays in the jaws."""
    names = tuple(b.name for b in BRIDGE_BLOCKS)
    rest = {b.name: block(b.kind).half_size[2] for b in BRIDGE_BLOCKS}
    add_blocks(twin, *BRIDGE_BLOCKS)
    settle(twin, 0.5)
    clock = Clock()
    settings = BuildSettings(plan_json=BRIDGE_JSON, judge="none", wrist_camera="")
    strategy = BuildStrategy(settings, threaded=False)
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, names, kinds={"yellow": "cuboid"}))
    blue0 = truth(twin, "blue")[0]
    lowest = {n: math.inf for n in names}
    fastest = {n: 0.0 for n in names}
    last = {n: truth(twin, n)[0] for n in names}

    def watch() -> None:
        for n in names:
            p = truth(twin, n)[0]
            lowest[n] = min(lowest[n], float(p[2]))
            fastest[n] = max(fastest[n], float(np.linalg.norm(p - last[n])) / TWIN_PERIOD)
            last[n] = p

    runner.start(threaded=False)
    drive(twin, runner, 400.0, clock, watch=watch)
    settle(twin)

    st = strategy.status
    assert st.outcome == BUILT and st.success and st.done == 2 and st.replans == 0, st
    (blue, rb), (green, rg), (yellow, ry) = (truth(twin, n) for n in names)
    assert green[1] > blue[1] and green[2] == pytest.approx(HALF, abs=TOWER_Z_M)
    assert abs(float(np.linalg.norm(green[:2] - blue[:2])) - (2 * HALF + 0.015)) < TOWER_XY_M
    assert np.linalg.norm(yellow[:2] - (blue[:2] + green[:2]) / 2) < TOWER_XY_M
    assert yellow[2] == pytest.approx(2 * HALF + rest["yellow"], abs=TOWER_Z_M)
    assert max(tilt_deg(r) for r in (rb, rg, ry)) < TOWER_TILT_DEG
    assert np.linalg.norm(blue - blue0) < BASE_SHIFT_M
    for n in names:
        assert lowest[n] >= rest[n] - SINK_M, (n, lowest[n])
        assert fastest[n] <= SPEED_MPS, (n, fastest[n])
        assert max(contact(twin, n)) < 0.5, n


# --------------------------------------------------------------------------- the seeded campaign

CAMPAIGN = int(os.environ.get("SOARM_BUILD_CAMPAIGN", "0") or 0)
SCENARIOS = {"fixed plan": {"planned": False}, "fake planner": {"planned": True},
             "knocked, fixed plan": {"planned": False, "knock": True},
             "knocked, fake planner": {"planned": True, "knock": True}}
#: Pass bars: successes out of the runs.
BARS = {"fixed plan": 0.8, "fake planner": 0.8, "knocked, fixed plan": 0.6, "knocked, fake planner": 0.6}


@pytest.mark.skipif(CAMPAIGN <= 0, reason="set SOARM_BUILD_CAMPAIGN=<runs> for the seeded campaign")
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_seeded_campaign(twin_robots, scenario, capsys):
    results = []
    for seed in range(100, 100 + CAMPAIGN):
        with fresh_twin() as twin:
            results.append(build_tower(twin, seed, **SCENARIOS[scenario]))
    good = [r for r in results if r.tower and r.success and not r.artefacts]
    with capsys.disabled():
        print(f"\n{scenario}: {len(good)}/{len(results)} towers by truth, built and clean")
        for r in results:
            mark = "ok " if r in good else "BAD"
            print(f"  {mark} seed {r.seed}: tower={r.tower} outcome={r.outcome} steps={r.steps} "
                  f"replans={r.replans} retries={r.retries} sim={r.sim_s:.0f}s wall={r.wall_s:.0f}s "
                  f"knocked={r.knocked} steps={r.step_outcomes} {r.why} {'; '.join(r.artefacts)}"
                  f"{'' if r in good else ' | ' + r.reason}")
    assert len(good) >= math.ceil(BARS[scenario] * len(results))
