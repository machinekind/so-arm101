"""Build a tower of three cubes, and a bridge, with `build` through the twin's discovery, runner, supervisor
and physics.

A seed puts the cubes on the table: the blue base where a three-high tower fits with the hand straight
down, green and red anywhere in reach. Each block has its own tracker, fed with simulation poses while the
hand is not near it. The plan comes from `plan_json` or a scripted fake planner, so no model, key or camera
is needed. The simulation's own poses check the tower, and that no block sank into the table, flew,
was dragged or stayed in the jaws. A knocked tower has its middle cube moved to the table while the top
one is carried to it. The bridge sets a cube beside another and lays a cuboid across both.

`SOARM_CAMPAIGN_RUNS=<runs>` or `SOARM_CAMPAIGN_DIR=<folder>` also runs the seeded campaign: towers, also with
the wrist camera and a camera error, knocked towers and bridges (default 10 seeds each), one JSON file per
run in the folder (`soarm_lab.campaign`).
"""

from __future__ import annotations

import json
import math
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
from soarm_lab.campaign import Run, campaign_dir, jsonable, record, runs_per_scenario  # noqa: E402
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
    closing jaw, as the twin's stacking tracker does. `offset` shifts every detection [m, base frame x and
    y], as a camera calibration error would.
    """

    twin: Twin
    clock: Clock
    names: tuple[str, ...]
    kinds: dict[str, str] = field(default_factory=dict)
    offset: tuple[float, float] = (0.0, 0.0)

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
                seen = CubeDetection(position + np.array([*self.offset, 0.0]), rotation, area_px=0.0,
                                     confidence=1.0, n_cameras=2, t=self.clock.t)
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
    #: pick_place runs started, and the structure's largest centre offset by truth [m] (NaN when a block is
    #: not at its level).
    picks: int = 0
    offset_m: float = math.nan
    #: (simulated time [s], build phase, step, step phase) at each change, each pick_place run's account and
    #: the build's step records.
    timeline: list[tuple[float, str, int, str]] = field(default_factory=list)
    steps_run: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    #: Looks between the open jaw tips that found a block off centre, over every pick_place run, and blocks
    #: that sank into each other (`Recorder.dips`).
    corrections: int = 0
    dips: list[str] = field(default_factory=list)


class Recorder:
    """Watches a build: each block's lowest height and top speed, the deepest contact between two blocks, how
    far the base moved, the phases, and every pick_place run the build starts."""

    def __init__(self, twin: Twin, strategy: BuildStrategy, clock: Clock, names: tuple[str, ...], base: str):
        self.twin, self.strategy, self.clock, self.names = twin, strategy, clock, names
        self.lowest = {n: math.inf for n in names}
        self.fastest = {n: 0.0 for n in names}
        self.last = {n: truth(twin, n)[0] for n in names}
        self.base, self.base0 = base, self.last[base].copy()
        self.base_shift = 0.0
        #: The deepest contact between two blocks [m, negative into each other] and the pair.
        self.deepest, self.pair = 0.0, ("", "")
        self.timeline: list[tuple[float, str, int, str]] = []
        self.children: list[Any] = []
        factory = strategy.step_factory

        def counted(settings: Any) -> Any:
            child = factory(settings)
            self.children.append(child)
            return child

        strategy.step_factory = counted

    def __call__(self) -> None:
        for n in self.names:
            p = truth(self.twin, n)[0]
            self.lowest[n] = min(self.lowest[n], float(p[2]))
            self.fastest[n] = max(self.fastest[n], float(np.linalg.norm(p - self.last[n])) / TWIN_PERIOD)
            self.last[n] = p
        self.base_shift = max(self.base_shift, float(np.linalg.norm(self.last[self.base] - self.base0)))
        with self.twin.lock:
            model, data = self.twin.scene.model, self.twin.scene.data
            for c in data.contact[:data.ncon]:
                pair = (model.geom(c.geom1).name, model.geom(c.geom2).name)
                if set(pair) <= set(self.names) and c.dist < self.deepest:
                    self.deepest, self.pair = float(c.dist), pair
        st, child = self.strategy.status, self.strategy.current_step
        phase = getattr(getattr(child, "status", None), "phase", "") if child is not None else ""
        now = (st.phase, st.index + 1, phase)
        if not self.timeline or self.timeline[-1][1:] != now:
            self.timeline.append((round(self.clock.t, 2), *now))

    def dips(self) -> list[str]:
        """Two blocks that sank into each other, as a block let go above another lands on it. Kept apart from
        `artefacts`, which the tests' checks assert; the campaign records count both."""
        if self.deepest >= -SINK_M:
            return []
        a, b = self.pair
        return [f"{a} and {b} sank {-self.deepest * 1000:.2f} mm into each other"]

    def artefacts(self, rest: dict[str, float]) -> list[str]:
        """Blocks that sank below their resting height on the table, flew, or stayed in the jaws, and a base
        that moved."""
        out = []
        for n in self.names:
            if self.lowest[n] < rest[n] - SINK_M:
                out.append(f"{n} sank {(rest[n] - self.lowest[n]) * 1000:.2f} mm into the table")
            if self.fastest[n] > SPEED_MPS:
                out.append(f"{n} moved at {self.fastest[n]:.2f} m/s")
            if max(contact(self.twin, n)) > 0.5:
                out.append(f"{n} in the jaws")
        if self.base_shift > BASE_SHIFT_M:
            out.append(f"base moved {self.base_shift * 1000:.1f} mm")
        return out

    def corrections(self) -> int:
        """Looks between the open jaw tips that found a block off centre, over every pick_place run."""
        looks = [look for child in self.children for look in child.wrist.close_looks]
        return sum(look.verdict in ("moved", "off") for look in looks)

    def steps_run(self) -> list[dict[str, Any]]:
        """Each pick_place run: its block and place, attempts, outcome, stop reason and place status."""
        out = []
        for child in self.children:
            cfg = child.config
            out.append(jsonable({
                "label": cfg.label, "place_on": cfg.place_on, "supports": cfg.supports,
                "place": (cfg.place_x_m, cfg.place_y_m, cfg.place_z_m, cfg.place_yaw_deg),
                "attempts": child.status.attempt, "outcome": child.place.outcome,
                "why": child.status.stopped_because, "grasp": child.grasp, "place_status": child.place}))
        return out


def tower_offset(twin: Twin) -> float:
    """The largest centre offset between neighbouring cubes of the tower by truth [m]; NaN when a cube is
    not at its level."""
    (blue, _), (green, _), (red, _) = (truth(twin, n) for n in NAMES)
    if any(abs(p[2] - (2 * level - 1) * HALF) > TOWER_Z_M for p, level in ((blue, 1), (green, 2), (red, 3))):
        return math.nan
    return max(float(np.linalg.norm(green[:2] - blue[:2])), float(np.linalg.norm(red[:2] - green[:2])))


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
                offset: tuple[float, float] = (0.0, 0.0), **extra: Any) -> Result:
    """Run one seeded build of the tower to its end, then judge it by truth; `extra` goes to the settings,
    `offset` shifts every detection."""
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
    wrist = ("wrist",) if settings["wrist_camera"] else ()
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, NAMES, offset=offset), cameras=wrist)
    seen = Recorder(twin, strategy, clock, NAMES, "blue")
    base0 = seen.base0
    knocked: list[float] = []

    def watch() -> None:
        seen()
        child = strategy.current_step
        if (knock and not knocked and strategy.status.done == 1 and child is not None
                and getattr(getattr(child, "status", None), "phase", "") == "carry_over"
                and not getattr(getattr(child, "place", None), "put_back", True)):
            red_from = next(p.xy for p in placed if p.name == "red")
            spot = free_spot(rng, [p.xy for p in placed if p.name != "green"])
            move(twin, "green", spot, rng.uniform(0.0, 90.0))
            seen.last["green"] = truth(twin, "green")[0]
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
    why = tower_problem(twin)
    st = strategy.status
    return Result(seed, not why, why, st.outcome, st.success, st.done, st.replans, st.retries, clock.t, wall,
                  bool(knocked), seen.artefacts({n: HALF for n in NAMES}),
                  [r.outcome.split(":")[0] for r in st.records], st.stopped_because, len(seen.children),
                  tower_offset(twin), seen.timeline, seen.steps_run(), [r.to_dict() for r in st.records],
                  seen.corrections(), seen.dips())


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


def bridge_layout(seed: int) -> tuple[list[Placed], str, float]:
    """A seeded bridge: blue in the middle of the reach, the side and gap for green, green and the cuboid
    anywhere in reach clear of each other and of where green and the bridge go. The fixed plan and the gap."""
    rng = np.random.default_rng(seed)
    blue = (rng.uniform(0.19, 0.21), rng.uniform(-0.03, 0.03))
    side = str(rng.choice(["left", "right"]))
    gap = float(rng.uniform(0.01, 0.025))
    placed = [Placed("blue", blue, float(rng.uniform(0.0, 90.0)))]
    away = 1.0 if side == "left" else -1.0
    goal = [(blue[0], blue[1] + away * (2 * HALF + gap)), (blue[0], blue[1] + away * (2 * HALF + gap) / 2)]
    # The cuboid is longer, so it keeps further from where green and the bridge go [m].
    for name, kind, away_m in (("green", "cube", 0.06), ("yellow", "cuboid", 0.08)):
        for _ in range(100_000):
            xy = polar(rng.uniform(0.16, 0.24), rng.uniform(-0.7, 0.7))
            clear = all(math.dist(xy, p.xy) >= 0.07 for p in placed)
            if clear and all(math.dist(xy, g) >= away_m for g in goal):
                break
        else:
            raise AssertionError(f"seed {seed}: no free spot")
        placed.append(Placed(name, xy, float(rng.uniform(0.0, 90.0)), kind))
    across = {"relation": "across", "targets": ["blue cube", "green cube"]}
    plan = [{"pick": "green cube", "place": {"relation": "next_to", "targets": ["blue cube"], "side": side,
                                             "gap_m": round(gap, 3)}},
            {"pick": "yellow cuboid", "place": across}]
    return placed, json.dumps(plan), round(gap, 3)


def build_bridge(twin: Twin, seed: int, seconds: float = 600.0) -> Result:
    """Run one seeded bridge build from its fixed plan to its end, then judge it by truth."""
    placed, plan_json, gap = bridge_layout(seed)
    names = tuple(b.name for b in placed)
    rest = {b.name: block(b.kind).half_size[2] for b in placed}
    add_blocks(twin, *placed)
    settle(twin, 0.5)
    clock = Clock()
    settings = BuildSettings(plan_json=plan_json, judge="none", wrist_camera="")
    strategy = BuildStrategy(settings, threaded=False)
    runner = StrategyRunner(twin, strategy, objects=Cameras(twin, clock, names, kinds={"yellow": "cuboid"}))
    seen = Recorder(twin, strategy, clock, names, "blue")
    t0 = time.monotonic()
    runner.start(threaded=False)
    drive(twin, runner, seconds, clock, watch=seen)
    wall = time.monotonic() - t0
    if runner.status.running:
        runner.stop()
    settle(twin)
    (blue, rb), (green, rg), (yellow, ry) = (truth(twin, n) for n in names)
    problems = []
    beside = float(np.linalg.norm(green[:2] - blue[:2]))
    if abs(beside - (2 * HALF + gap)) > TOWER_XY_M or abs(green[2] - HALF) > TOWER_Z_M:
        problems.append(f"green {beside * 1000:.0f} mm from blue, at {green[2] * 1000:.0f} mm")
    off = float(np.linalg.norm(yellow[:2] - (blue[:2] + green[:2]) / 2))
    level = abs(yellow[2] - (2 * HALF + rest["yellow"])) <= TOWER_Z_M
    if off > TOWER_XY_M or not level:
        problems.append(f"bridge {off * 1000:.1f} mm off, at {yellow[2] * 1000:.1f} mm")
    tilt = max(tilt_deg(r) for r in (rb, rg, ry))
    if tilt > TOWER_TILT_DEG:
        problems.append(f"tilted {tilt:.1f} deg")
    why = "; ".join(problems)
    st = strategy.status
    return Result(seed, not why, why, st.outcome, st.success, st.done, st.replans, st.retries, clock.t, wall,
                  False, seen.artefacts(rest), [r.outcome.split(":")[0] for r in st.records],
                  st.stopped_because, len(seen.children), off if level else math.nan, seen.timeline,
                  seen.steps_run(), [r.to_dict() for r in st.records], seen.corrections(), seen.dips())


#: Runs per scenario without `SOARM_CAMPAIGN_RUNS`; the campaign runs only when either variable is set.
RUNS = 10
CAMPAIGN = campaign_dir() is not None or runs_per_scenario(0) > 0
#: Scenarios: `build_tower` keywords, a camera error [m] in a seeded direction, or the bridge.
SCENARIOS: dict[str, dict[str, Any]] = {
    "tower, fixed plan": {"planned": False},
    "tower, fake planner": {"planned": True},
    "tower, fixed plan, wrist camera": {"planned": False, "wrist_camera": "wrist"},
    "tower, fixed plan, wrist camera, cameras 5 mm off": {"planned": False, "wrist_camera": "wrist",
                                                          "camera_error_m": 0.005},
    "knocked tower, fixed plan": {"planned": False, "knock": True},
    "knocked tower, fake planner": {"planned": True, "knock": True},
    "bridge, fixed plan": {"bridge": True}}
#: Pass bars: the share of runs built by the strategy and by truth, without artefacts. The run with a camera
#: error only measures.
BARS = {"tower, fixed plan": 0.8, "tower, fake planner": 0.8, "tower, fixed plan, wrist camera": 0.8,
        "tower, fixed plan, wrist camera, cameras 5 mm off": 0.0, "knocked tower, fixed plan": 0.6,
        "knocked tower, fake planner": 0.6, "bridge, fixed plan": 0.8}


def as_run(scenario: str, result: Result, settings: dict[str, Any]) -> Run:
    """A campaign record of `result`: the pick_place runs, and the structure's offset by truth. Blocks that
    sank into each other count as artefacts here."""
    artefacts = result.artefacts + result.dips
    if result.outcome != BUILT:
        last = result.step_outcomes[-1] if result.step_outcomes else ""
        reason = f"{result.outcome or 'no outcome'}" + (f" after {last}" if last else "")
    elif not result.tower:
        reason = "built, but not standing by truth"
    else:
        reason = "artefact" if artefacts else ""
    good = result.tower and result.success and not artefacts
    why = "" if good else "; ".join(x for x in (result.why, result.reason) if x)
    details = {"settings": settings, "why": why,
               "outcome": result.outcome, "steps": result.steps, "replans": result.replans,
               "retries": result.retries, "knocked": result.knocked, "step_outcomes": result.step_outcomes,
               "records": result.records, "steps_run": result.steps_run, "timeline": result.timeline}
    return Run(f"build: {scenario}", result.seed, good, result.picks, result.offset_m, reason, artefacts,
               result.sim_s, result.wall_s, details, result.corrections)


@pytest.mark.campaign
@pytest.mark.skipif(not CAMPAIGN, reason="set SOARM_CAMPAIGN_RUNS or SOARM_CAMPAIGN_DIR for the campaign")
@pytest.mark.parametrize("scenario", [
    pytest.param(name, marks=pytest.mark.render) if settings.get("wrist_camera") else name
    for name, settings in SCENARIOS.items()])
def test_seeded_campaign(twin_robots, scenario, capsys):
    settings = dict(SCENARIOS[scenario])
    error = settings.pop("camera_error_m", 0.0)
    results = []
    for seed in range(100, 100 + runs_per_scenario(RUNS)):
        a = np.random.default_rng(10_000 + seed).uniform(0.0, 2 * math.pi)
        offset = (error * math.cos(a), error * math.sin(a))
        with fresh_twin() as twin:
            if settings.get("bridge"):
                result = build_bridge(twin, seed)
            else:
                result = build_tower(twin, seed, offset=offset, **settings)
        record(as_run(scenario, result, {**settings, "camera_offset_m": offset}))
        results.append(result)
    good = [r for r in results if r.tower and r.success and not r.artefacts]
    with capsys.disabled():
        print(f"\n{scenario}: {len(good)}/{len(results)} built by truth and clean")
        for r in results:
            mark = "ok " if r in good else "BAD"
            print(f"  {mark} seed {r.seed}: built={r.tower} outcome={r.outcome} steps={r.steps} "
                  f"picks={r.picks} replans={r.replans} retries={r.retries} sim={r.sim_s:.0f}s "
                  f"wall={r.wall_s:.0f}s knocked={r.knocked} steps={r.step_outcomes} {r.why} "
                  f"{'; '.join(r.artefacts + r.dips)}"
                  f"{'' if r in good else ' | ' + r.reason}")
    assert len(good) >= math.ceil(BARS[scenario] * len(results) - 1e-9)
