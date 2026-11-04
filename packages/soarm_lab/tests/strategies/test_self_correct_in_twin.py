"""Seeded twin campaigns for the self-correcting grasp: `grasp_wrist` with the look between the jaw tips.

Each run puts a cube at a seeded place and yaw, gives the strategy one coarse detector pose off by
`DETECTOR_M` in a seeded direction, and renders the wrist camera. `Backlash` adds hidden play to the
arm's joints. A nudge moves the cube once the hover look is done. Truth poses judge the run: the cube is
lifted and held, the jaw tips' midpoint lay within `BAR_M` of the cube's centre when the jaw last started
closing, and it took at most `MAX_ATTEMPTS` attempts, in at least `BAR_SHARE` of the runs. In every run the
cube stays where it lies until the jaw closes on it, neither pushed by the corrections nor sinking.

`SOARM_CAMPAIGN_RUNS` scales the runs per campaign (default `RUNS`, half for the nudge);
`SOARM_CAMPAIGN_DIR` keeps one JSON file per run (`soarm_lab.campaign`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.twin
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as twin_scene  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy  # noqa: E402
from machinekind_twin.strategy import Action, ObjectPose, Observation, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.campaign import Run, jsonable, record, runs_per_scenario  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import top_down  # noqa: E402
from soarm_lab.strategies.wrist import WristGraspStrategy  # noqa: E402

CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
RED = (0.85, 0.25, 0.2, 1.0)
SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: The arm joints that place the hand, and the pose their play is sized at.
PLAY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
#: The detector's error [m], the nudge after the hover look [m] and the play at the hand [m].
DETECTOR_M = 0.015
NUDGE_M = 0.005
PLAY_M = 0.006
#: The bar: the share of runs lifted with the jaw centred within `BAR_M` in at most `MAX_ATTEMPTS`.
BAR_M = 0.003
MAX_ATTEMPTS = 3
BAR_SHARE = 0.9
#: Runs per campaign without `SOARM_CAMPAIGN_RUNS`.
RUNS = 20
#: The cube may move this much before the close, and sink this much into the table [m].
STILL_M = 0.001
SINK_M = 0.001
#: Where seeded cubes lie [m, base frame].
X_RANGE = (0.17, 0.24)
Y_RANGE = (-0.07, 0.07)


def campaign_runs() -> int:
    return runs_per_scenario(RUNS)


@dataclass
class Clock:
    t: float = 0.0


# --------------------------------------------------------------------------- the hidden play
class Backlash:
    """Hidden play of each joint between the strategy's targets and the simulated arm [deg].

    The strategy sees the motor side: its targets reached, and measured joints that include the play's
    offset. The arm gets the link side, which follows a target only once the target has taken up the play.
    """

    def __init__(self, strategy: Any, play_deg: Mapping[str, float]):
        self.strategy = strategy
        self.play = dict(play_deg)
        self.hz = strategy.hz
        self.time_limit_s = getattr(strategy, "time_limit_s", None)
        self.motor: dict[str, float] = {}
        self.link: dict[str, float] = {}

    def _seen(self, obs: Observation) -> Observation:
        def shifted(values: Mapping[str, float]) -> dict[str, float]:
            return {k: float(v) + self.motor[k] - self.link[k] if k in self.link else float(v)
                    for k, v in values.items()}

        return replace(obs, joints=shifted(obs.joints), command=shifted(obs.command))

    def start(self, run: Any, obs: Observation) -> None:
        self.motor = {k: float(obs.joints[k]) for k in self.play if k in obs.joints}
        self.link = dict(self.motor)
        self.strategy.start(run, obs)

    def act(self, obs: Observation) -> Action:
        action = self.strategy.act(self._seen(obs))
        target = dict(action.target)
        for k, play in self.play.items():
            if k not in target or k not in self.link:
                continue
            x, y = float(target[k]), self.link[k]
            if x > y + play / 2:
                y = x - play / 2
            elif x < y - play / 2:
                y = x + play / 2
            self.motor[k], self.link[k] = x, y
            target[k] = y
        return replace(action, target=target)

    def stop(self, reason: str) -> None:
        self.strategy.stop(reason)


def play_for(metres: float, xy: tuple[float, float] = (0.2, 0.0)) -> dict[str, float]:
    """Play per joint [deg] that moves the hand `metres` across its whole width, at a grasp over `xy`."""
    home = dict(SPEC.home, gripper=0.0)
    sol = KIN.ik(np.array([*xy, 0.02]), top_down(0.0), seed=home)
    assert sol.ok
    joints = dict(sol.joints)
    out = {}
    for name in PLAY_JOINTS:
        step = 0.1
        ahead, back = dict(joints), dict(joints)
        ahead[name] += step
        back[name] -= step
        moved = np.linalg.norm(KIN.tcp(ahead)[:3, 3] - KIN.tcp(back)[:3, 3]) / (2 * step)
        out[name] = metres / float(moved)
    return out


# --------------------------------------------------------------------------- the twin
def add_cube(twin: Twin, xy: tuple[float, float], yaw_deg: float) -> None:
    spec = block("cube")
    twin.configure(objects=[twin_scene.Box("cube", spec.half_size, xy, rgba=RED, mass=spec.mass)],
                   grasp_sensors=["cube"])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        g = model.geom("cube").id
        model.geom_solref[g] = STIFF_SOLREF
        model.geom_solimp[g][:2] = STIFF_SOLIMP
        model.geom_priority[g] = 1
        yaw = math.radians(yaw_deg)
        adr = model.jnt_qposadr[model.body_jntadr[model.body("cube").id]]
        data.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)


def truth(twin: Twin, name: str = "cube") -> np.ndarray:
    with twin.lock:
        s = twin.scene
        Ti = inverse(s.T_base2world)
        return (Ti[:3, :3] @ s.data.xpos[s.model.body(name).id] + Ti[:3, 3]).copy()


def nudge(twin: Twin, d: np.ndarray, name: str = "cube") -> None:
    """Move the block by `d` [m, base frame x and y]."""
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        adr = model.jnt_qposadr[model.body_jntadr[model.body(name).id]]
        dw = twin.scene.T_base2world[:3, :3] @ np.array([d[0], d[1], 0.0])
        data.qpos[adr:adr + 2] += dw[:2]
        mujoco.mj_forward(model, data)


def coarse(twin: Twin, clock: Clock, offset: np.ndarray, age_s: float = 8.0
           ) -> Callable[[], dict[str, ObjectPose]]:
    """One detector pose at table height, off by `offset`, without orientation."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position = truth(twin)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {"cube": ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock, hz: float,
          watch: Callable[[], None] | None = None) -> None:
    strategy_every = max(1, round(1.0 / hz / CLOCK_STEP))
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    for _ in range(round(seconds / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
            if watch is not None:
                watch()
        if not runner.status.running:
            break


# --------------------------------------------------------------------------- one run
@dataclass
class Outcome:
    """One campaign run, judged from truth poses."""

    seed: int
    xy: tuple[float, float]
    yaw_deg: float
    lifted: bool
    attempts: int
    #: The jaw tips' midpoint minus the cube's centre when the jaw started closing [m], each close.
    close_errors_m: list[float] = field(default_factory=list)
    #: The look between the tips: (attempt, round, along mm, across mm, verdict) for each look, and the
    #: truth's along and across [mm] when it looked.
    looks: list[tuple[int, int, float, float, str]] = field(default_factory=list)
    truth_at_looks: list[tuple[float, float]] = field(default_factory=list)
    hover_verdicts: list[str] = field(default_factory=list)
    #: How far the cube moved on the table from the first look between the tips to the last close, and its
    #: lowest centre height before the lift [m].
    moved_m: float = 0.0
    lowest_m: float = math.inf
    reason: str = ""
    sim_s: float = 0.0
    wall_s: float = 0.0
    #: (simulated time [s], attempt, phase) at each change of phase, and the strategy's last status.
    timeline: list[tuple[float, int, str]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def final_error_m(self) -> float:
        return self.close_errors_m[-1] if self.close_errors_m else math.inf

    @property
    def passed(self) -> bool:
        return self.lifted and self.final_error_m <= BAR_M and self.attempts <= MAX_ATTEMPTS


def one_run(seed: int, *, play_m: float = 0.0, nudge_m: float = 0.0, detector_m: float = DETECTOR_M,
            **settings: Any) -> Outcome:
    rng = np.random.default_rng(seed)
    xy = (float(rng.uniform(*X_RANGE)), float(rng.uniform(*Y_RANGE)))
    yaw = float(rng.uniform(0.0, 90.0))
    a, b = rng.uniform(0.0, 2 * math.pi, size=2)
    offset = detector_m * np.array([math.cos(a), math.sin(a)])
    push = nudge_m * np.array([math.cos(b), math.sin(b)])
    started = time.monotonic()
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        add_cube(twin, xy, yaw)
        clock = Clock()
        strategy = load_strategy("grasp_wrist").create({"carry_s": 1.0, **settings})
        assert isinstance(strategy, WristGraspStrategy)
        driven = Backlash(strategy, play_for(play_m, xy)) if play_m else strategy
        runner = StrategyRunner(twin, driven, objects=coarse(twin, clock, offset), cameras=("wrist",))
        out = Outcome(seed, xy, yaw, False, 0)
        last = {"phase": ""}

        anchor: list[np.ndarray] = []

        def watch() -> None:
            phase = strategy.status.phase
            if phase != last["phase"]:
                out.timeline.append((round(clock.t, 2), strategy.status.attempt, phase))
            if phase in ("straddle", "rise", "open", "close") and last["phase"] != "close":
                # From the attempt's first look between the tips to its close.
                where = truth(twin)
                out.lowest_m = min(out.lowest_m, float(where[2]))
                attempt = strategy.status.attempt
                if not anchor and any(c.attempt == attempt for c in strategy.wrist.close_looks):
                    anchor.append(where[:2])
                if anchor:
                    out.moved_m = max(out.moved_m, float(np.linalg.norm(where[:2] - anchor[0])))
                if phase == "close":
                    anchor.clear()
            if len(out.truth_at_looks) < len(strategy.wrist.close_looks):
                tips = KIN.fingertips(dict(twin.status.measured))
                u = (tips[1] - tips[0])[:2] / np.linalg.norm((tips[1] - tips[0])[:2])
                error = (truth(twin) - 0.5 * (tips[0] + tips[1]))[:2]
                out.truth_at_looks.append((round(float(error @ u) * 1000, 2),
                                           round(float(error @ np.array([-u[1], u[0]])) * 1000, 2)))
            if phase == "close" and last["phase"] != "close":
                tips = KIN.fingertips(dict(twin.status.measured))
                mid = 0.5 * (tips[0] + tips[1])
                out.close_errors_m.append(float(np.linalg.norm((mid - truth(twin))[:2])))
            if nudge_m and phase == "straddle" and strategy.wrist.verdicts and "nudged" not in last:
                nudge(twin, push)
                last["nudged"] = "yes"
            last["phase"] = phase

        runner.start(threaded=False)
        drive(twin, runner, 120.0, clock, strategy.hz, watch=watch)
        held = runner.status.stopped_because == "zadanie wykonane" and float(truth(twin)[2]) > 0.04
        for _ in range(round(1.0 / TWIN_PERIOD)):
            twin.step(TWIN_PERIOD)
        out.lifted = held and float(truth(twin)[2]) > 0.04
        out.attempts = strategy.status.attempt
        out.looks = [(c.attempt, c.round, round(c.along_m * 1000, 2), round(c.across_m * 1000, 2), c.verdict)
                     for c in strategy.wrist.close_looks]
        out.hover_verdicts = list(strategy.wrist.verdicts)
        out.reason = strategy.status.stopped_because
        out.sim_s = clock.t
        out.status = jsonable({"run": strategy.status, "grasp": strategy.grasp, "wrist": strategy.wrist})
    finally:
        twin.close()
    out.wall_s = time.monotonic() - started
    return out


def artefacts(o: Outcome) -> list[str]:
    """The cube pushed or sunk before the close."""
    half = block("cube").half_size[2]
    out = []
    if o.moved_m > STILL_M:
        out.append(f"cube pushed {o.moved_m * 1000:.1f} mm before the close")
    if o.lowest_m < half - SINK_M:
        out.append(f"cube sank {(half - o.lowest_m) * 1000:.2f} mm into the table")
    return out


def as_run(name: str, o: Outcome, settings: Mapping[str, Any]) -> Run:
    """A campaign record of `o`: the attempts, and the jaw centre's offset at the last close."""
    seen = artefacts(o)
    if not o.lifted:
        reason = "not lifted"
    elif o.final_error_m > BAR_M:
        reason = f"jaw centre more than {BAR_M * 1000:.0f} mm off"
    elif o.attempts > MAX_ATTEMPTS:
        reason = f"more than {MAX_ATTEMPTS} attempts"
    else:
        reason = "artefact" if seen else ""
    details = {"settings": dict(settings), "why": "" if o.passed else o.reason, **asdict(o)}
    corrections = sum(look[4] in ("moved", "off") for look in o.looks)
    return Run(f"grasp: {name}", o.seed, o.passed and not seen, o.attempts, o.final_error_m, reason, seen,
               o.sim_s, o.wall_s, details, corrections)


def campaign(name: str, seeds: range, **kwargs: Any) -> list[Outcome]:
    outcomes = []
    for seed in seeds:
        outcomes.append(one_run(seed, **kwargs))
        record(as_run(name, outcomes[-1], {"detector_m": DETECTOR_M, **kwargs}))
    return outcomes


def summary(outcomes: list[Outcome]) -> str:
    rows = [f"seed {o.seed}: {'ok' if o.passed else 'MISS'} lifted={o.lifted} attempts={o.attempts} "
            f"errors={[round(e * 1000, 1) for e in o.close_errors_m]} mm looks={o.looks} ({o.reason})"
            for o in outcomes]
    return "\n".join(rows)


def assert_bar(outcomes: list[Outcome]) -> None:
    passed = sum(o.passed for o in outcomes)
    assert passed >= math.ceil(BAR_SHARE * len(outcomes) - 1e-9), summary(outcomes)
    assert not any(artefacts(o) for o in outcomes), [(o.seed, artefacts(o)) for o in outcomes]


# --------------------------------------------------------------------------- the helper
def test_backlash_hides_the_play_from_the_strategy():
    """The strategy sees its own targets reached; the arm lags by half the play after a turn."""

    class Echo:
        hz = 10.0

        def __init__(self) -> None:
            self.seen: list[dict[str, float]] = []
            self.targets = iter([{"elbow_flex": 10.0}, {"elbow_flex": 12.0}, {"elbow_flex": 11.0},
                                 {"elbow_flex": 9.0}])

        def start(self, run, obs) -> None:
            pass

        def act(self, obs) -> Action:
            self.seen.append(dict(obs.joints))
            return Action(target=next(self.targets))

    echo = Echo()
    play = Backlash(echo, {"elbow_flex": 2.0})
    obs = Observation(t=0.0, step=0, joints={"elbow_flex": 10.0}, command={"elbow_flex": 10.0})
    play.start(None, obs)
    sent = []
    for _ in range(4):
        action = play.act(obs)
        sent.append(action.target["elbow_flex"])
        # The arm reaches what it was sent.
        obs = replace(obs, joints={"elbow_flex": sent[-1]}, command={"elbow_flex": sent[-1]})
    # Up 2 deg moves the arm 1 deg less; back down 1 deg stays inside the play; down 2 more moves 1 deg.
    assert sent == [10.0, 11.0, 11.0, 10.0]
    assert [s["elbow_flex"] for s in echo.seen] == [10.0, 10.0, 12.0, 11.0]


def test_the_play_is_sized_at_the_hand():
    play = play_for(PLAY_M)
    assert set(play) == set(PLAY_JOINTS) and all(0.5 < deg < 10.0 for deg in play.values())


# --------------------------------------------------------------------------- the campaigns
@pytest.mark.render
@pytest.mark.campaign
def test_a_cube_seen_1_5_cm_off_is_lifted_centred():
    outcomes = campaign("detector 1.5 cm", range(campaign_runs()))
    assert_bar(outcomes)


@pytest.mark.render
@pytest.mark.campaign
def test_with_6_mm_of_hidden_play_a_cube_seen_1_5_cm_off_is_lifted_centred():
    outcomes = campaign("detector 1.5 cm, play 6 mm", range(100, 100 + campaign_runs()), play_m=PLAY_M)
    assert_bar(outcomes)


@pytest.mark.render
@pytest.mark.campaign
def test_a_cube_nudged_5_mm_after_the_hover_look_is_lifted_centred():
    runs = range(200, 200 + campaign_runs() // 2)
    outcomes = campaign("detector 1.5 cm, nudge 5 mm", runs, nudge_m=NUDGE_M)
    assert_bar(outcomes)


@pytest.mark.render
def test_a_cube_taken_from_between_the_closing_jaws_is_found_again_and_lifted():
    xy = (0.21, 0.02)
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        add_cube(twin, xy, 15.0)
        clock = Clock()
        strategy = load_strategy("grasp_wrist").create({"carry_s": 1.0})
        runner = StrategyRunner(twin, strategy, objects=coarse(twin, clock, np.array([0.008, -0.006])),
                                cameras=("wrist",))
        taken: list[np.ndarray] = []

        def watch() -> None:
            if strategy.status.phase == "close" and not taken:
                # Out sideways along the jaw faces, clear of both jaws.
                psi = math.radians(strategy.grasp.heading_deg)
                d = 0.03 * np.array([-math.sin(psi), math.cos(psi)])
                nudge(twin, d)
                taken.append(truth(twin)[:2])

        runner.start(threaded=False)
        drive(twin, runner, 120.0, clock, strategy.hz, watch=watch)

        assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.wrist)
        assert strategy.grasp.empty_closes == 1 and strategy.status.attempt == 2
        assert strategy.wrist.verdicts == ["centred", "centred"]
        assert truth(twin)[2] > 0.04
    finally:
        twin.close()
