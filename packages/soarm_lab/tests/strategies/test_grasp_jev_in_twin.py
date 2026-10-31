"""The twin finds grasp_jev, builds it without a key, and its decision steps act on a simulated block.

Everything goes through the twin's own code, as in test_grasp_in_twin.py: discovery and strict settings,
the run (`StrategyRunner`), the safety supervisor and the twin's physics on a virtual clock, with the block's
contact sensors against both jaws. Jev is replaced by `GraspContactJudge`, which answers from those sensors:
the probe touched when a jaw touches the block, the block held when both do; no test needs
TYPESAFE_API_KEY. The poses come from the twin's own `ObjectTracker` (`tracked`) or from a coarse pose as the
LLM detector gives it. The wrist-frame test renders the simulated arm's camera and carries `render`.

The twin's robot discovery is pinned to `soarm`'s folder (fixture `twin_robots`).
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
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

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import COLOURS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp_jev import (  # noqa: E402
    HELD_OPTIONS,
    NEXT_OPTIONS,
    GraspJevSettings,
    GraspJevStrategy,
)
from soarm_lab.strategies.jev import IMAGE_SIDE_PX, KEY_ENV  # noqa: E402
from soarm_lab.strategies.judges import Reply, checked_reply  # noqa: E402
from soarm_lab.strategies.touch_jev import CORRECTIONS, VERDICTS  # noqa: E402

NAME = "grasp_jev"
TOUCH_FIELDS = ["label", "touch_z", "hover_m", "wait_for_pose_s", "max_pose_age_s", "close_jaw",
                "need_contact", "hz", "speed_mps", "descend_speed_mps", "contact_gap_m", "contact_band_m",
                "max_tracking_deg", "time_limit_s"]
GRASP_FIELDS = TOUCH_FIELDS + ["object_kind", "open_margin_m", "squeeze_m", "grasp_depth", "lift_m", "hold_s",
                               "max_attempts", "carry_s", "release", "wait_s", "recover_m", "grasp_source",
                               "library_dir", "min_score"]
NEW_FIELDS = ["model", "min_confidence", "min_correction_confidence", "correction_m", "answer_timeout_s",
              "send_frames", "wrist_camera"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the block lies [m, base frame].
BLOCK_XY = (0.2, 0.0)
#: The hand hides the block from the cameras nearer than this [m].
HIDDEN_M = 0.05
#: The block's contact: the time constant twice the 5 ms step (MuJoCo's stiffest) and a high impedance.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
SPEC = load_robot()
KIN = RobotKinematics(SPEC)

Objects = Callable[[], dict[str, ObjectPose]]


@pytest.fixture
def twin(twin_robots):
    twin = Twin(Workspace(robot="so101"), clock=VirtualClock(t0=0.0))
    try:
        yield twin
    finally:
        twin.close()


@dataclass
class Clock:
    """The test's simulated time [s]: the runner's ticks and the tracker both read it."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """The twin's loop at 50 Hz and the strategy at its own rate, on one simulated clock; `watch` after
    every physics step. Stops early once the run is over or `until()` holds."""
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
        if not runner.status.running or (until is not None and until()):
            break


def add_block(twin: Twin, kind: str, xy: tuple[float, float] = BLOCK_XY) -> None:
    """The block `kind` on the table with contact sensors against both jaws, its contacts as stiff as the
    5 ms step allows (test_grasp_in_twin.py says why), then the simulated arm."""
    spec = object_spec(kind)
    twin.configure(objects=[twin_scene.Box(kind, spec.half_size, xy, rgba=COLOURS[kind], mass=spec.mass)],
                   grasp_sensors=[kind])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model = twin.scene.model
        g = model.geom(kind).id
        model.geom_solref[g] = STIFF_SOLREF
        model.geom_solimp[g][:2] = STIFF_SOLIMP
        model.geom_priority[g] = 1


def truth(twin: Twin, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The block's position and rotation in the base frame, as the simulation has them."""
    with twin.lock:
        s = twin.scene
        b = s.model.body(name).id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def contact(twin: Twin, name: str) -> list[float]:
    """The contacts between the block and each jaw right now (fixed, moving)."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def tracked(twin: Twin, name: str, clock: Clock, sources: list[tuple[float, str, float]]) -> Objects:
    """The block as the twin's tracker gives it: the simulation's truth while the hand is farther than
    `HIDDEN_M`, nothing nearer; "w dloni" while the jaw holds it. Each call records (t, source, age)."""
    tracker = ObjectTracker.for_spec(twin.workspace.spec(), hold_s=30.0)
    spec = KIN.spec
    assert spec.gripper is not None
    g = spec.joints.index(spec.gripper)
    closed = float(KIN.to_q({spec.gripper: twin.joint_limits()[spec.gripper][0]})[g])

    def objects() -> dict[str, ObjectPose]:
        status = twin.status
        measured, command = dict(status.measured), dict(status.command) or dict(status.measured)
        T_tcp = KIN.tcp(measured)
        position, rotation = truth(twin, name)
        seen = None
        if np.linalg.norm(position - T_tcp[:3, 3]) > HIDDEN_M:
            seen = CubeDetection(position, rotation, area_px=0.0, confidence=1.0, n_cameras=2, t=clock.t)
        got = tracker.update(seen, T_tcp, float(KIN.to_q(measured)[g]), float(KIN.to_q(command)[g]), closed,
                             clock.t)
        if got is None:
            sources.append((clock.t, tracker.source, float("nan")))
            return {}
        age = 0.0 if tracker.source == "w dloni" else max(0.0, clock.t - tracker.t_seen)
        sources.append((clock.t, tracker.source, age))
        return {name: ObjectPose(np.asarray(got[0], float), np.asarray(got[1], float), tracker.source, age)}

    return objects


def coarse(twin: Twin, name: str, clock: Clock, offset: np.ndarray, age_s: float = 8.0) -> Objects:
    """The block as the LLM detector gives it: seen once, `offset` off in x and y, at the table top, with no
    orientation, "ostatnie widziane" `age_s` old and ageing."""
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position, _ = truth(twin, name)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {name: ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def phases_of(strategy: GraspJevStrategy, phases: list[str]) -> Callable[[], bool]:
    """An `until` that never ends the run and keeps the strategy's phases in `phases`, repeats dropped."""

    def record() -> bool:
        if not phases or phases[-1] != strategy.status.phase:
            phases.append(strategy.status.phase)
        return False

    return record


def closing_heading() -> float:
    """The closing heading of the rest pose [rad]: the one a pose without rotation keeps."""
    _, closing = KIN.tool_axes(dict(KIN.spec.home, gripper=0.0))
    return math.atan2(float(closing[1]), float(closing[0]))


def choice(option: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    return {"type": "choice", "choice": option, "confidence": confidence,
            "probabilities": {o: top if o == option else (1 - top) / (n - 1) for o in options}}


class GraspContactJudge:
    """Answers grasp_jev's questions from the simulation's jaw sensors instead of Jev.

    The probe: touched_object while a jaw touches the block, else missed_object, at `probe_sure`. The close
    and the lift: block_held and lift while both jaws touch it, else nothing_in_jaws and regrasp_wider, at
    `sure` and `nsure`; the first lift-point call answers `lift_once` instead when given. It records the
    states, the questions, the images and each call's point.
    """

    name = "jev"

    def __init__(self, twin: Twin, kind: str = "cube", probe_sure: float = 0.95, sure: float = 0.95,
                 nsure: float = 0.9, lift_once: tuple[str, str] | None = None):
        self.twin, self.kind = twin, kind
        self.probe_sure, self.sure, self.nsure, self.lift_once = probe_sure, sure, nsure, lift_once
        self.states: list[dict[str, Any]] = []
        self.images: list[dict[str, np.ndarray]] = []
        self.points: list[str] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        self.states.append(state)
        self.images.append(dict(images))
        sensors = contact(self.twin, self.kind)
        if "touched" in questions:
            self.points.append("probe")
            verdict = "touched_object" if max(sensors) > 0.5 else "missed_object"
            answers = {"touched": choice(verdict, self.probe_sure, VERDICTS),
                       "correction": choice("aim_at_latest_sighting", 0.9, CORRECTIONS)}
            return checked_reply(self.name, answers, "contact-sensors", {}, questions)
        point = state["point"]
        self.points.append(point)
        held, nxt = ("block_held", "lift") if min(sensors) > 0.5 else ("nothing_in_jaws", "regrasp_wider")
        if point == "lift" and self.lift_once is not None:
            (held, nxt), self.lift_once = self.lift_once, None
        answers = {"held": choice(held, self.sure, HELD_OPTIONS),
                   "next": choice(nxt, self.nsure, NEXT_OPTIONS)}
        return checked_reply(self.name, answers, "contact-sensors", {}, questions)


class Gated:
    """Answers the probe at once and the close once the test opens the gate, from the jaw sensors."""

    name = "jev"

    def __init__(self, twin: Twin):
        self.contact = GraspContactJudge(twin)
        self.gate = threading.Event()
        self.cancels: list[threading.Event] = []

    def check(self) -> None:
        pass

    def ask(self, state, questions, *, images, timeout_s, cancel) -> Reply:
        if "held" in questions:
            self.cancels.append(cancel)
            self.gate.wait(10.0)
        return self.contact.ask(state, questions, images=images, timeout_s=timeout_s, cancel=cancel)


def built(judge: Any, **settings: Any) -> GraspJevStrategy:
    settings.setdefault("carry_s", 1.0)
    return GraspJevStrategy(GraspJevSettings(**settings), judge=judge, threaded=False)


# --------------------------------------------------------------------------- discovery and settings


def test_the_twin_finds_grasp_jev_and_builds_it_without_a_key(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    assert strategies()[NAME].value == "soarm_lab.strategies.grasp_jev:GraspJevStrategy"
    plugin = load_strategy(NAME)
    fields = {field.name: field for field in plugin.fields}

    assert plugin.cls is GraspJevStrategy and plugin.settings is GraspJevSettings
    assert plugin.source == "soarm-lab 0.1.0 (soarm_lab.strategies.grasp_jev:GraspJevStrategy)"
    assert plugin.summary == ("Grasps a block as `grasp` does; Jev judges each probe, close and lift and "
                              "picks the next step.")
    assert list(fields) == GRASP_FIELDS + NEW_FIELDS
    assert fields["wrist_camera"].kind is str and fields["wrist_camera"].default == "wrist"
    assert fields["send_frames"].kind is bool and fields["time_limit_s"].default == 180.0
    assert fields["model"].default == "jev-1.14.0" and not any(field.required for field in fields.values())
    strategy = plugin.create({"object_kind": "cuboid5", "answer_timeout_s": 2.0})
    assert isinstance(strategy, GraspJevStrategy) and strategy.block.kind == "cuboid5"
    with pytest.raises(ValueError, match=r"Nieznane klucze w sekcji grasp_jev: \['wrist_cam'\]"):
        plugin.create({"wrist_cam": "wrist"})


def test_without_the_key_grasp_jev_refuses_the_run_before_the_arm_moves(twin, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    twin.connect("sim", threaded=False)
    runner = StrategyRunner(twin, load_strategy(NAME).create({}))

    with pytest.raises(RuntimeError, match="brak klucza jev"):
        runner.start(threaded=False)
    assert runner.status.stopped_because.startswith(f"strategia odmowila: brak klucza jev: ustaw {KEY_ENV}")
    assert twin.owner is None


# --------------------------------------------------------------------------- grasps in the simulated twin


def test_jev_in_the_loop_lifts_the_cube_and_the_twin_says_w_dloni(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin)
    strategy = built(judge)
    sources: list[tuple[float, str, float]] = []
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, sources))
    closed: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.status.phase == "lift_off" and not closed:
            closed.append(clock.t)

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert judge.points == ["probe", "close", "lift"] and strategy.grasp.verdict == "lifted"
    recorded = [d.state["point"] for d in strategy.judging.decisions]
    assert recorded == ["probe"] * 2 + ["close"] * 2 + ["lift"] * 2
    assert judge.states[1]["close"]["jaw_gap"] == "block_wide"
    assert truth(twin, "cube")[0][2] > 0.04 and all(c > 0.5 for c in contact(twin, "cube"))
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    assert truth(twin, "cube")[0][2] > 0.04 and all(c > 0.5 for c in contact(twin, "cube"))
    in_hand = [t for t, s, _ in sources if s == "w dloni"]
    assert in_hand and in_hand[0] <= closed[0] + 0.5
    assert {s for t, s, _ in sources if t >= in_hand[0]} == {"w dloni"}
    assert strategy.judging.disagreements == 0 and not strategy.judging.needs_review
    assert twin.owner is None


@pytest.mark.render
def test_the_wrist_frame_from_the_simulated_arm_camera_goes_to_the_judge(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin)
    strategy = built(judge, send_frames=True)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []), cameras=("wrist",))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    camera = SPEC.cameras["wrist"]
    stride = math.ceil(max(camera.width, camera.height) / IMAGE_SIDE_PX[1])
    close = judge.points.index("close")
    [(reference, image)] = judge.images[close].items()
    assert image.shape == (math.ceil(camera.height / stride), math.ceil(camera.width / stride), 3)
    assert all(IMAGE_SIDE_PX[0] <= side <= IMAGE_SIDE_PX[1] for side in image.shape[:2])
    state = judge.states[close]
    assert state["camera_views"] == {"wrist": reference} and state["frames_left_out"] == {}
    assert 0.0 <= state["frames_age_s"]["wrist"] <= 1.0
    scaled = f"{camera.width}x{camera.height} -> {image.shape[1]}x{image.shape[0]}"
    assert state["frames_scaled"] == {"wrist": scaled}


def test_an_unsure_answer_at_the_close_stops_the_run_with_the_block_in_the_jaw(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin, probe_sure=0.95, sure=0.4)
    strategy = built(judge)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)
    held = dict(twin.status.measured)
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)

    assert judge.points == ["probe", "close"]
    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because == ("jev niepewny: blok trzymany z pewnoscia 0.40 < 0.80 - ramie "
                                               "stoi, do oceny czlowieka")
    assert all(c > 0.5 for c in contact(twin, "cube"))
    arm = KIN.spec.arm_joints
    assert {n: twin.status.measured[n] for n in arm} == pytest.approx({n: held[n] for n in arm}, abs=0.5)
    assert twin.owner is None


def test_after_the_lift_a_regrasp_sets_the_cube_down_and_lifts_it_again(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin, lift_once=("held_by_edge", "regrasp_wider"))
    strategy = built(judge)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))
    phases: list[str] = []
    lag: list[float] = []
    released: list[tuple[float, list[float]]] = []

    def watch() -> None:
        if not strategy.started:
            return
        phase = strategy.status.phase
        if phase == "lower":
            status = twin.status
            lag.append(max(abs(status.measured[n] - status.command[n]) for n in KIN.spec.arm_joints))
        if phase == "let_go" and strategy._released and not released:
            released.append((float(truth(twin, "cube")[0][2]), contact(twin, "cube")))

    runner.start(threaded=False)
    drive(twin, runner, 90.0, clock, until=phases_of(strategy, phases), watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert strategy.status.attempt == 2 and strategy.gap_open_m > 0.041
    assert judge.points == ["probe", "close", "lift", "probe", "close", "lift"]
    i = phases.index("lower")
    assert phases[i:i + 3] == ["lower", "let_go", "rise"]
    assert released and released[0][0] < 0.02 and all(c < 0.5 for c in released[0][1])
    assert lag and max(lag) < 5.0
    assert truth(twin, "cube")[0][2] > 0.04


def test_a_cube_taken_away_during_the_lift_goes_to_the_recovery_and_is_not_chased(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin)
    strategy = built(judge, wait_s=5.0)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock, until=lambda: strategy.started and strategy.status.phase == "lift_off"
          and strategy.grasp.lifted_m > 0.01)
    with twin.lock:
        model, data = twin.scene.model, twin.scene.data
        adr = model.jnt_qposadr[model.body_jntadr[model.body("cube").id]]
        data.qpos[adr:adr + 3] = (0.0, 0.2, 0.0125)
        data.qvel[model.body_dofadr[model.body("cube").id]:][:6] = 0.0
        mujoco.mj_forward(model, data)
    drive(twin, runner, 30.0, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.grasp.drops == 1 and strategy.grasp.verdict == "taken"
    assert strategy.status.stopped_because.startswith("blok zabrany"), strategy.status.stopped_because
    assert judge.points == ["probe", "close"]
    arm = KIN.spec.arm_joints
    status = twin.status
    assert {n: status.measured[n] for n in arm} == pytest.approx({n: status.command[n] for n in arm}, abs=1.0)


def test_a_coarse_pose_off_along_the_jaws_is_grasped_through_regrasp_wider(twin):
    """1.2 cm off toward the fixed jaw: the moving jaw's open tip lands on the cube. The judge answers
    nothing_in_jaws and regrasp_wider: grasp's sideways retry, the moving tip 16 mm farther out and the
    fixed one where grasp puts it, so the close pushes the cube no farther than grasp's does."""
    add_block(twin, "cube")
    clock = Clock()
    psi = closing_heading()
    off = -0.012 * np.array([math.cos(psi), math.sin(psi)])
    judge = GraspContactJudge(twin)
    strategy = built(judge)
    runner = StrategyRunner(twin, strategy, objects=coarse(twin, "cube", clock, off))
    start, _ = truth(twin, "cube")
    pushed: list[float] = []

    def watch() -> None:
        if strategy.started and strategy.status.phase not in ("lift_off", "check", "carry", "done"):
            pushed.append(float(np.linalg.norm(truth(twin, "cube")[0][:2] - start[:2])))

    runner.start(threaded=False)
    drive(twin, runner, 90.0, clock, watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert strategy.grasp.straddle_stalls == 1 and strategy.status.attempt == 2
    assert strategy.gap_open_m == pytest.approx(0.057) and judge.points[:2] == ["probe", "close"]
    assert judge.states[1]["close"]["ended"] == "jaw_tip_landed_on_block_before_closing"
    assert max(pushed) < 0.015 and truth(twin, "cube")[0][2] > 0.04


def test_a_plain_stop_during_the_close_question(twin):
    add_block(twin, "cube")
    clock = Clock()
    judge = Gated(twin)
    strategy = GraspJevStrategy(GraspJevSettings(), judge=judge, threaded=True)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))

    runner.start(threaded=False)
    try:
        drive(twin, runner, 60.0, clock, until=lambda: bool(judge.cancels))
        assert runner.status.running and strategy.status.phase == "lift_off"
        runner.stop()
    finally:
        judge.gate.set()

    assert runner.status.stopped_because == "zatrzymana"
    assert strategy.status.stopped_because == "zatrzymana" and strategy.status.phase == "done"
    assert judge.cancels[0].is_set()
    close = [json.loads(d.answer)["error"] for d in strategy.judging.decisions if d.state["point"] == "close"]
    assert close == ["cancelled", "cancelled"]
    assert twin.owner is None
