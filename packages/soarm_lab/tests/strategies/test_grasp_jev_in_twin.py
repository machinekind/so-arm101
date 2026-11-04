"""Test grasp judging through the twin's discovery, runner, supervisor and physics.

A fake judge uses simulated jaw contacts. Poses come from the tracker or a coarse detector.
The wrist-frame case requires rendering. No test in this module needs a model key.
`twin_robots` pins discovery to soarm's folder.
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
from machinekind_twin.cli import main as twin_main  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.episode import EpisodeLog, EpisodeOptions  # noqa: E402
from machinekind_twin.kinematics import inverse  # noqa: E402
from machinekind_twin.perception import CubeDetection, ObjectTracker  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategies import load_strategy, strategies  # noqa: E402
from machinekind_twin.strategy import ObjectPose, StrategyRunner  # noqa: E402
from machinekind_twin.workspace import Workspace  # noqa: E402

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
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
                "max_tracking_deg", "sag_m", "stall_s", "time_limit_s"]
GRASP_FIELDS = TOUCH_FIELDS + ["block", "probe", "open_margin_m", "squeeze_m", "grasp_depth",
                               "straddle_push_m", "close_on_stall_m", "lift_m",
                               "hold_s", "max_attempts", "carry_s", "release", "wait_s", "recover_m",
                               "grasp_source", "library_dir", "min_score", "max_tilt_deg",
                               "jaw_gap_offset_m", "jaw_file"]
NEW_FIELDS = ["model", "min_confidence", "min_correction_confidence", "correction_m", "answer_timeout_s",
              "send_frames", "wrist_camera"]
#: The simulated clock's step, and the twin loop's period [s].
CLOCK_STEP = 0.01
TWIN_PERIOD = 0.02
#: Where the block lies [m, base frame].
BLOCK_XY = (0.2, 0.0)
#: The hand hides the block from the cameras nearer than this [m].
HIDDEN_M = 0.05
#: Stiff contact uses a time constant twice the physics step and high impedance.
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
    """Simulated time shared by the runner and tracker [s]."""

    t: float = 0.0


def drive(twin: Twin, runner: StrategyRunner, seconds: float, clock: Clock,
          until: Callable[[], bool] | None = None, watch: Callable[[], None] | None = None) -> None:
    """Run twin and strategy ticks on a shared simulated clock.

    Call `watch` after each physics step. Stop when the run ends or `until()` is true.
    """
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
    """Add a block with stiff contacts and sensors for both jaws, then connect the simulated arm."""
    spec = block(kind)
    blue = (0.2, 0.4, 0.9, 1.0)
    twin.configure(objects=[twin_scene.Box(kind, spec.half_size, xy, rgba=blue, mass=spec.mass)],
                   grasp_sensors=[kind])
    twin.connect("sim", threaded=False)
    with twin.lock:
        model = twin.scene.model
        g = model.geom(kind).id
        model.geom_solref[g] = STIFF_SOLREF
        model.geom_solimp[g][:2] = STIFF_SOLIMP
        model.geom_priority[g] = 1


def truth(twin: Twin, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the simulated block position [m, base frame] and rotation."""
    with twin.lock:
        s = twin.scene
        b = s.model.body(name).id
        Ti = inverse(s.T_base2world)
        position = Ti[:3, :3] @ s.data.xpos[b] + Ti[:3, 3]
        rotation = Ti[:3, :3] @ s.data.xmat[b].reshape(3, 3)
    return position.copy(), rotation.copy()


def contact(twin: Twin, name: str) -> list[float]:
    """Return the contacts with the fixed and moving jaws."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def tracked(twin: Twin, name: str, clock: Clock, sources: list[tuple[float, str, float]]) -> Objects:
    """Return tracker poses from visible simulation detections.

    Hide detections within `HIDDEN_M`. Record (t, source, age) on each call.
    """
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
    """Return a single detector pose shifted by `offset` [m, base frame x and y].

    It has no orientation and starts at age `age_s` [s], which grows with the clock.
    """
    seen: list[np.ndarray] = []

    def objects() -> dict[str, ObjectPose]:
        if not seen:
            position, _ = truth(twin, name)
            seen.append(np.array([position[0] + offset[0], position[1] + offset[1], 0.0]))
        return {name: ObjectPose(seen[0].copy(), np.eye(3), "ostatnie widziane", age_s + clock.t)}

    return objects


def phases_of(strategy: GraspJevStrategy, phases: list[str]) -> Callable[[], bool]:
    """Return an always-false stop predicate that records phase changes."""

    def record() -> bool:
        if not phases or phases[-1] != strategy.status.phase:
            phases.append(strategy.status.phase)
        return False

    return record


def closing_heading() -> float:
    """Return the rest pose closing heading [rad], used for poses without orientation."""
    _, closing = KIN.tool_axes(dict(KIN.spec.home, gripper=0.0))
    return math.atan2(float(closing[1]), float(closing[0]))


def choice(option: str, confidence: float, options: tuple[str, ...]) -> dict[str, Any]:
    n = len(options)
    top = confidence * (1 - 1 / n) + 1 / n
    return {"type": "choice", "choice": option, "confidence": confidence,
            "probabilities": {o: top if o == option else (1 - top) / (n - 1) for o in options}}


class GraspContactJudge:
    """Answer from jaw contacts and record the state, questions, images and decision point.

    A probe touches if either jaw contacts the block. A close holds if both do, selecting lift;
    otherwise select wider regrasp. Supplied confidence values control each answer.
    `lift_once` can override the first lift-point answer.
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
    """Answer the probe immediately and wait for the test gate before answering the close."""

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
    assert plugin.summary == ("Grasps a block as `grasp` does; Jev judges each close and lift and picks the "
                              "next step.")
    assert list(fields) == GRASP_FIELDS + NEW_FIELDS
    assert fields["wrist_camera"].kind is str and fields["wrist_camera"].default == "wrist"
    assert fields["send_frames"].kind is bool and fields["time_limit_s"].default == 180.0
    assert fields["model"].default == "jev-1.14.0" and not any(field.required for field in fields.values())
    strategy = plugin.create({"block": "cuboid", "answer_timeout_s": 2.0})
    assert isinstance(strategy, GraspJevStrategy) and strategy.block.name == "cuboid"
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
    assert judge.points == ["close", "lift"] and strategy.grasp.verdict == "lifted"
    recorded = [d.state["point"] for d in strategy.judging.decisions]
    assert recorded == ["close"] * 2 + ["lift"] * 2
    assert judge.states[0]["close"]["jaw_gap"] == "block_wide"
    assert truth(twin, "cube")[0][2] > 0.04 and all(c > 0.5 for c in contact(twin, "cube"))
    for _ in range(round(1.0 / TWIN_PERIOD)):
        twin.step(TWIN_PERIOD)
    assert truth(twin, "cube")[0][2] > 0.04 and all(c > 0.5 for c in contact(twin, "cube"))
    in_hand = [t for t, s, _ in sources if s == "w dloni"]
    assert in_hand and in_hand[0] <= closed[0] + 0.5
    assert {s for t, s, _ in sources if t >= in_hand[0]} == {"w dloni"}
    assert strategy.judging.disagreements == 0 and not strategy.judging.needs_review
    assert twin.owner is None


def test_the_episode_recording_keeps_every_decision_and_twin_episode_reads_it(twin, tmp_path, capsys):
    add_block(twin, "cube")
    clock = Clock()
    strategy = built(GraspContactJudge(twin))
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []),
                            episode=EpisodeOptions(folder=tmp_path, name="grasp_jev"))

    runner.start(threaded=False)
    drive(twin, runner, 60.0, clock)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert runner.wait_for_episode(5.0) and runner.status.episode is not None
    log = EpisodeLog.load(runner.status.episode)
    assert log.end.code == "done" and log.recording["telemetry_complete"]
    recorded = [d for step in log.steps for d in step["decisions"]]
    kept = strategy.judging.decisions
    assert ([(d["question"], d["state"]["point"]) for d in recorded]
            == [(d.question, d.state["point"]) for d in kept])
    assert [d["answer"] for d in recorded] == [d.answer for d in kept]
    assert [d["confidence"] for d in recorded] == pytest.approx([d.confidence for d in kept], abs=1e-3)

    assert twin_main(["episode", str(log.path)]) == 0
    assert "Model decisions: 4" in capsys.readouterr().out


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

    assert judge.points == ["close"]
    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.status.stopped_because == ("jev niepewny: blok trzymany z pewnoscia 0.40 < 0.80 - ramie "
                                               "stoi, do oceny czlowieka")
    assert all(c > 0.5 for c in contact(twin, "cube"))
    arm = KIN.spec.arm_joints
    assert {n: twin.status.measured[n] for n in arm} == pytest.approx({n: held[n] for n in arm}, abs=0.5)
    assert twin.owner is None


def test_after_the_lift_a_regrasp_sets_the_cube_down_and_lifts_it_again(twin):
    """The released cube stays flat while the fixed jaw clears it before the rise."""
    add_block(twin, "cube")
    clock = Clock()
    judge = GraspContactJudge(twin, lift_once=("held_by_edge", "regrasp_wider"))
    strategy = built(judge)
    runner = StrategyRunner(twin, strategy, objects=tracked(twin, "cube", clock, []))
    phases: list[str] = []
    lag: list[float] = []
    released: list[tuple[float, list[float]]] = []
    tilts: list[float] = []

    def watch() -> None:
        if not strategy.started:
            return
        phase = strategy.status.phase
        if phase == "lower":
            status = twin.status
            lag.append(max(abs(status.measured[n] - status.command[n]) for n in KIN.spec.arm_joints))
        if phase == "let_go" and strategy._released and not released:
            released.append((float(truth(twin, "cube")[0][2]), contact(twin, "cube")))
        if released and phase in ("let_go", "rise"):
            # The released cube stays flat while the open jaw leaves it.
            tilts.append(math.degrees(math.acos(min(1.0, abs(float(truth(twin, "cube")[1][2, 2]))))))

    runner.start(threaded=False)
    drive(twin, runner, 90.0, clock, until=phases_of(strategy, phases), watch=watch)

    assert runner.status.stopped_because == "zadanie wykonane", (strategy.status, strategy.grasp)
    assert strategy.status.attempt == 2 and strategy.gap_open_m > 2 * block("cube").half_size[1] + 0.016
    assert judge.points == ["close", "lift", "close", "lift"]
    i = phases.index("lower")
    assert phases[i:i + 3] == ["lower", "let_go", "rise"]
    assert released and released[0][0] < 0.02 and all(c < 0.5 for c in released[0][1])
    assert tilts and max(tilts) < 2.0
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
        data.qpos[adr:adr + 3] = (0.0, 0.2, block("cube").half_size[2])
        data.qvel[model.body_dofadr[model.body("cube").id]:][:6] = 0.0
        mujoco.mj_forward(model, data)
    drive(twin, runner, 30.0, clock)

    assert runner.status.stopped_because == "strategia skonczyla" and not runner.status.success
    assert strategy.grasp.drops == 1 and strategy.grasp.verdict == "taken"
    assert strategy.status.stopped_because.startswith("blok zabrany"), strategy.status.stopped_because
    assert judge.points == ["close"]
    arm = KIN.spec.arm_joints
    status = twin.status
    assert {n: status.measured[n] for n in arm} == pytest.approx({n: status.command[n] for n in arm}, abs=1.0)


def test_a_coarse_pose_off_along_the_jaws_is_grasped_through_regrasp_wider(twin):
    """A tip stall from a coarse pose recovers with a sideways straddle and wider moving-jaw opening."""
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
    assert strategy.gap_open_m == pytest.approx(2 * block("cube").half_size[1] + 0.032)
    assert judge.points[:1] == ["close"]
    assert judge.states[0]["close"]["ended"] == "jaw_tip_landed_on_block_before_closing"
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
