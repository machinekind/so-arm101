"""`grasp` in the twin with servos that settle short under load, as P-only servos do.

`Sag` stands between the strategy and the simulated arm. The loaded pitch joints stop short of a target
that raises the hand, and friction then holds them where they hang until a lowering target comes back
down to them. The strategy sees what a real arm reports: its own targets as the command, and measured
joints short of them.

Each run judges itself by the simulation's truth: the cube lifted and gripped by both jaws.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspStrategy, top_down  # noqa: E402

CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
SPEC = load_robot()
KIN = RobotKinematics(SPEC)
#: Where the cube lies [m, base frame].
CUBE_XY = (0.2, 0.0)
#: How far each loaded pitch joint stops short of a target that raises the hand [deg].
SHORT_DEG = {"shoulder_lift": 2.3, "elbow_flex": 2.2}
#: The simulated time a run may take [s]: well under the strategy's own time limit.
RUN_S = 60.0


@dataclass
class Clock:
    t: float = 0.0


# --------------------------------------------------------------------------- the sagging servos
def lowering(names: tuple[str, ...], xy: tuple[float, float] = CUBE_XY) -> dict[str, float]:
    """+1 for a joint whose angle grows as it lowers the hand, -1 otherwise, at a grasp over `xy`."""
    sol = KIN.ik(np.array([*xy, 0.04]), top_down(0.0), seed=dict(SPEC.home, gripper=0.0))
    assert sol.ok
    joints = dict(sol.joints)
    z = KIN.tcp(joints)[2, 3]
    return {name: 1.0 if KIN.tcp(dict(joints, **{name: joints[name] + 0.5}))[2, 3] < z else -1.0
            for name in names}


class Sag:
    """Loaded pitch joints that settle `short` degrees short of a target that raises the hand.

    A lowering target is followed once it comes down to where the joint hangs, so after a raise the first
    `short` degrees of a lowering move leave the joint where it is. The strategy gets its own targets as
    the command and the measured joints as they are, short of them.
    """

    def __init__(self, strategy: Any, short_deg: Mapping[str, float]):
        self.strategy = strategy
        self.short = dict(short_deg)
        self.sign = lowering(tuple(self.short))
        self.hz = strategy.hz
        self.time_limit_s = getattr(strategy, "time_limit_s", None)
        #: The strategy's last target and what the arm was sent instead [deg].
        self.target: dict[str, float] = {}
        self.sent: dict[str, float] = {}

    def _seen(self, obs: Observation) -> Observation:
        command = {k: float(v) + self.target[k] - self.sent[k] if k in self.sent else float(v)
                   for k, v in obs.command.items()}
        return replace(obs, command=command)

    def start(self, run: Any, obs: Observation) -> None:
        self.target = {k: float(obs.joints[k]) for k in self.short if k in obs.joints}
        self.sent = dict(self.target)
        self.strategy.start(run, obs)

    def act(self, obs: Observation) -> Action:
        action = self.strategy.act(self._seen(obs))
        target = dict(action.target)
        for k, short in self.short.items():
            if k not in target or k not in self.sent:
                continue
            # In the lowering direction: the joint stays between its target and `short` past it.
            sign = self.sign[k]
            x, y = sign * float(target[k]), sign * self.sent[k]
            y = min(max(y, x), x + short)
            self.target[k], self.sent[k] = float(target[k]), sign * y
            target[k] = sign * y
        return replace(action, target=target)

    def stop(self, reason: str) -> None:
        self.strategy.stop(reason)


def test_sag_holds_a_raised_joint_short_until_a_lowering_target_comes_back_to_it():
    class Echo:
        hz = 10.0

        def __init__(self) -> None:
            self.seen: list[Observation] = []
            self.targets = iter([{"elbow_flex": 10.0}, {"elbow_flex": 6.0}, {"elbow_flex": 7.0},
                                 {"elbow_flex": 9.0}, {"elbow_flex": 11.0}])

        def start(self, run, obs) -> None:
            pass

        def act(self, obs) -> Action:
            self.seen.append(obs)
            return Action(target=next(self.targets))

    echo = Echo()
    sag = Sag(echo, {"elbow_flex": 2.0})
    obs = Observation(t=0.0, step=0, joints={"elbow_flex": 10.0}, command={"elbow_flex": 10.0})
    sag.start(None, obs)
    sent = []
    for _ in range(5):
        action = sag.act(obs)
        sent.append(action.target["elbow_flex"])
        # The arm reaches what it was sent.
        obs = replace(obs, joints={"elbow_flex": sent[-1]}, command={"elbow_flex": sent[-1]})
    # A smaller angle raises the hand: 4 deg up stops 2 short, 1 deg back down stays put, then it follows.
    assert sent == [10.0, 8.0, 8.0, 9.0, 11.0]
    # The strategy sees its own target as the command, and the measured joint as it is.
    assert [o.command["elbow_flex"] for o in echo.seen[1:]] == pytest.approx([10.0, 6.0, 7.0, 9.0])
    assert [o.joints["elbow_flex"] for o in echo.seen[1:]] == sent[:4]


def test_the_loaded_pitch_joints_lower_the_hand_as_their_angles_grow():
    assert lowering(tuple(SHORT_DEG)) == {"shoulder_lift": 1.0, "elbow_flex": 1.0}


# --------------------------------------------------------------------------- the twin
@pytest.fixture
def twin(twin_robots):
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


def add_cube(twin: Twin, xy: tuple[float, float] = CUBE_XY) -> None:
    spec = block("cube")
    twin.configure(objects=[twin_scene.Box("cube", spec.half_size, xy, rgba=(0.85, 0.25, 0.2, 1.0),
                                           mass=spec.mass)],
                   grasp_sensors=["cube"])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model = twin.scene.model
        g = model.geom("cube").id
        model.geom_solref[g] = STIFF_SOLREF
        model.geom_solimp[g][:2] = STIFF_SOLIMP
        model.geom_priority[g] = 1


def truth(twin: Twin) -> np.ndarray:
    with twin.lock:
        s = twin.scene
        Ti = inverse(s.T_base2world)
        return (Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3]).copy()


def gripped(twin: Twin) -> bool:
    with twin.lock:
        s = twin.scene
        return all(float(s.data.sensordata[s.model.sensor(f"cube_jaw{k}").adr[0]]) > 0.5 for k in range(2))


def coarse(twin: Twin, clock: Clock, offset: np.ndarray) -> Callable[[], dict[str, ObjectPose]]:
    """One detector pose at table height, off by `offset`, without orientation."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position = truth(twin)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {"cube": ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", 8.0 + clock.t)}

    return objects


def closing_heading() -> float:
    """The rest pose's closing heading [rad], which a pose without orientation keeps."""
    _, closing = KIN.tool_axes(dict(SPEC.home, gripper=0.0))
    return math.atan2(float(closing[1]), float(closing[0]))


@dataclass
class Run:
    strategy: GraspStrategy
    reason: str
    #: (simulated time [s], phase, note) at each change of phase.
    timeline: list[tuple[float, str, str]]
    sim_s: float
    lifted: bool

    def phases(self) -> list[str]:
        return [phase for _, phase, _ in self.timeline]


def run_sagging(twin: Twin, offset: np.ndarray, **settings: Any) -> Run:
    """A default grasp of the cube seen `offset` off, through `Sag`, for at most `RUN_S`."""
    add_cube(twin)
    clock = Clock()
    strategy = load_strategy("grasp").create({"carry_s": 1.0, **settings})
    assert isinstance(strategy, GraspStrategy)
    runner = StrategyRunner(twin, Sag(strategy, SHORT_DEG), objects=coarse(twin, clock, offset))
    timeline: list[tuple[float, str, str]] = []
    strategy_every = round(1.0 / strategy.hz / CLOCK_STEP)
    twin_every = round(TWIN_PERIOD / CLOCK_STEP)
    runner.start(threaded=False)
    for _ in range(round(RUN_S / CLOCK_STEP)):
        k = round(clock.t / CLOCK_STEP) + 1
        clock.t = k * CLOCK_STEP
        if k % twin_every == 0:
            twin.step(TWIN_PERIOD)
        if k % strategy_every == 0 and runner.status.running:
            runner.step_once(clock.t)
            phase = strategy.status.phase
            if not timeline or timeline[-1][1] != phase:
                timeline.append((round(clock.t, 2), phase, runner.status.note))
        if not runner.status.running:
            break
    sim_s = clock.t
    if runner.status.running:
        runner.stop()
    lifted = truth(twin)[2] > 0.04 and gripped(twin)
    return Run(strategy, runner.status.stopped_because, timeline, sim_s, lifted)


def test_a_rise_that_settles_short_after_a_tip_on_the_block_goes_on_to_the_grasp(twin):
    """The moving tip lands on the cube. The rise from there stops short under load: the arm accepts the
    sag and grasps the cube from the side the tip did not land on."""
    psi = closing_heading()
    run = run_sagging(twin, -0.012 * np.array([math.cos(psi), math.sin(psi)]))
    s = run.strategy

    assert run.reason == "zadanie wykonane" and run.lifted, (run.reason, run.timeline)
    assert s.grasp.straddle_stalls == 1 and s.status.attempt == 2, run.timeline
    assert run.phases().count("rise") == 1
    assert run.sim_s < 40.0


def test_a_straddle_from_a_hover_that_hangs_short_is_no_tip_on_the_block(twin):
    """The hover hangs short and friction holds the arm while the descent's command comes down to it: the
    lag that opens up first is not the jaw on the block."""
    run = run_sagging(twin, np.zeros(2))
    s = run.strategy

    assert run.reason == "zadanie wykonane" and run.lifted, (run.reason, run.timeline)
    assert s.grasp.straddle_stalls == 0 and s.status.attempt == 1, run.timeline
    assert "rise" not in run.phases()
