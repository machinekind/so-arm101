"""`calibrate_approach` without the twin: its settings, the shared approach geometry, the plan of approaches,
the board check and a fake feed that follows every target, for skips, time-outs, partial tables and the notes.

The fake measurement stands in for the wrist camera: it returns the vector from where the TCP is to where the
corner really is, so a camera error shows up as a correction of minus that error."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.boards import BoardSpec, BoardView, board_rotation
from machinekind_twin_api.strategy import ObjectPose, Observation, RunInfo

import soarm
from soarm_lab.approach_table import ApproachTable
from soarm_lab.kinematics import RobotKinematics
from soarm_lab.robot import load_robot
from soarm_lab.strategies.approach import (
    DIRECTIONS,
    carry_heights,
    direction_vector,
    lead_point,
    place_heights,
)
from soarm_lab.strategies.calibrate_approach import (
    CORNERS,
    PARK,
    SAMPLES,
    CalibrateApproachSettings,
    CalibrateApproachStrategy,
    ProbeSpec,
    flip_bias,
    remove_bias,
    rot2,
    tcp_to_corner,
)
from soarm_lab.strategies.grasp import wrap
from soarm_lab.strategies.pick_place import GRIP_SLIP_M, MIN_CARRY_CLEAR_M, TABLE_TOP_M

SPEC = load_robot()
KIN = RobotKinematics(SPEC)
PAN_XY = KIN.joint_anchor("shoulder_pan", SPEC.home)[:2]
HZ = 20.0
BOARD = BoardSpec()
#: A small run: two corners, one level, two directions.
SMALL = {"corners": "1,6", "levels_m": "0.029", "directions": "ccw,cw", "mount_pairs": 0, "verify": False}


def corners(centre=(0.2, 0.0), yaw_deg: float = 0.0, square_m: float = BOARD.square_m) -> np.ndarray:
    """The inner corners (n, 3) [m, base frame] of a board lying face up at `centre`."""
    spec = BoardSpec(square_m=square_m, marker_m=min(BOARD.marker_m, 0.9 * square_m))
    R = board_rotation(math.radians(yaw_deg))
    origin = np.array([*centre, 0.0]) - R @ np.array([*np.array(spec.size_m) / 2, 0.0])
    return spec.corner_obj(range(spec.n_corners)) @ R.T + origin


def sightings(points: np.ndarray, every_s: float = 0.5) -> Callable[[float], dict[str, ObjectPose]]:
    """Perception that sees every corner, with a new capture every `every_s`."""
    def objects(t: float) -> dict[str, ObjectPose]:
        age = t % every_s
        return {f"charuco c{k}": ObjectPose(p.copy(), np.eye(3), "kamery", age) for k, p in enumerate(points)}
    return objects


def wrist(true: np.ndarray, *, blind: set[int] | None = None) -> Callable[..., Any]:
    """A wrist measurement: the true corner minus the TCP where the commanded joints put it."""
    def measure(strategy: CalibrateApproachStrategy, obs: Observation, planned) -> np.ndarray | str:
        if blind and planned.corner in blind:
            return "za malo punktow tablicy (0)"
        return true[planned.corner] - KIN.tcp(dict(obs.joints))[:3, 3]
    return measure


def run_info() -> RunInfo:
    return RunInfo(robot=SPEC.name, joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits={},
                   max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)


class Feed:
    """An arm that reaches every target at once, unless frozen, and the strategy's notes."""

    def __init__(self, strategy: CalibrateApproachStrategy,
                 objects: Callable[[float], Mapping[str, ObjectPose]]):
        self.strategy, self.objects = strategy, objects
        self.joints = dict(SPEC.home, shoulder_pan=-80.0)
        self.notes: list[str] = []
        self.frozen_after: float = math.inf
        self.t = 0.0
        self.done = None

    def observe(self, step: int) -> Observation:
        return Observation(t=self.t, step=step, joints=dict(self.joints), command=dict(self.joints),
                           objects=self.objects(self.t))

    def run(self, seconds: float, until: Callable[[], bool] | None = None) -> None:
        self.strategy.start(run_info(), self.observe(0))
        for step in range(1, round(seconds * HZ)):
            self.t = step / HZ
            action = self.strategy.act(self.observe(step))
            if not self.notes or self.notes[-1] != action.note:
                self.notes.append(action.note)
            if self.t < self.frozen_after:
                self.joints.update(action.target)
            if action.done:
                self.done = action
                self.strategy.stop("zadanie wykonane" if action.success else "strategia skonczyla")
                return
            if until is not None and until():
                return


def built(tmp_path, measure: Any = None, **settings: Any) -> CalibrateApproachStrategy:
    cfg = CalibrateApproachSettings(**{"table_file": str(tmp_path / "approach.json"), **settings})
    return CalibrateApproachStrategy(cfg, measure=measure if measure is not None else False)


# --------------------------------------------------------------------------- settings
@pytest.mark.parametrize("change, words", [
    ({"table_file": ""}, "table_file"),
    ({"directions": "out,up"}, "directions"),
    ({"directions": "out,out"}, "directions"),
    ({"levels_m": "0.029,0.0"}, "levels_m"),
    ({"levels_m": "a"}, "levels_m"),
    ({"corners": "1,x"}, "corners"),
    ({"corners": "8"}, "corners 0 to 7"),
    ({"repeats": 0}, "repeats"),
    ({"looks": 0}, "looks"),
    ({"mount_pairs": -1}, "mount_pairs"),
    ({"lead_m": 0.0}, "lead_m"),
    ({"max_skip_share": 1.0}, "max_skip_share"),
    ({"square_m": 0.03}, "marker_m"),
    ({"wrist_camera": ""}, "wrist_camera"),
])
def test_bad_settings_say_what_is_wrong_in_english(change, words):
    cfg = CalibrateApproachSettings(**{"table_file": "t.json", **change})
    with pytest.raises(ValueError, match=words):
        CalibrateApproachStrategy(cfg)


def test_an_explicit_schedule_only_probes():
    cfg = CalibrateApproachSettings(table_file="t.json")
    with pytest.raises(ValueError, match="schedule"):
        CalibrateApproachStrategy(cfg, schedule=[((0.2, 0.0), 0, "ccw")])
    CalibrateApproachStrategy(CalibrateApproachSettings(), schedule=[((0.2, 0.0), 0, "ccw")], measure=False)


def test_a_probe_takes_its_heading_and_stops_at_its_pick_before_a_straight_carry():
    """A probe with a heading and a pick stop: down at the pick with the pick's heading, up, straight across
    to the leg's start with the hand turning on the way, the leg and down, then still at the heading asked."""
    xy = np.array([0.2, 0.03])
    psi = wrap(math.atan2(xy[1] - PAN_XY[1], xy[0] - PAN_XY[0]) - math.radians(30.0))
    pick = np.array([0.19, -0.06, 0.017])
    stills: list[tuple[float, np.ndarray]] = []
    path: list[np.ndarray] = []

    def on_still(strategy: CalibrateApproachStrategy, obs: Observation, planned) -> None:
        stills.append((planned.psi, KIN.tcp(dict(obs.joints))[:3, 3]))

    strategy = CalibrateApproachStrategy(
        CalibrateApproachSettings(levels_m="0.029"), measure=False, on_still=on_still,
        schedule=[ProbeSpec(xy, 0, "cw", psi, pick, 0.3), (xy, 0, "ccw")])
    feed = Feed(strategy, lambda t: {})
    step = strategy.act

    def act(obs: Observation):
        action = step(obs)
        path.append(KIN.tcp(dict(action.target))[:3, 3].copy())
        return action

    strategy.act = act  # type: ignore[method-assign]
    feed.run(120.0)
    assert feed.done is not None and feed.done.success, feed.notes
    assert len(stills) == 2 and stills[0][0] == pytest.approx(psi)
    release = strategy.level_heights()[0][1]
    assert stills[0][1] == pytest.approx([*xy, release], abs=1e-3)
    at_pick = [k for k, p in enumerate(path) if np.linalg.norm(p - pick) < 1e-3]
    assert at_pick, "the hand never stood at the pick"
    lead = lead_point(xy, direction_vector("cw", xy, PAN_XY), 0.03)
    at_lead = next(k for k, p in enumerate(path) if k > at_pick[-1] and np.linalg.norm(p[:2] - lead) < 1e-3)
    # From over the pick to the leg's start the TCP runs on the straight line between them.
    carry = [p for p in path[at_pick[-1]:at_lead] if p[2] > pick[2] + 0.02]
    start = carry[0]
    line = (lead - start[:2]) / np.linalg.norm(lead - start[:2])
    across = np.array([-line[1], line[0]])
    off_line = [abs(float((p[:2] - start[:2]) @ across)) for p in carry]
    assert max(off_line) < 2e-3


def test_a_level_too_low_for_the_jaw_is_refused(tmp_path):
    strategy = built(tmp_path, hang_m=0.003, levels_m="0.0")
    with pytest.raises(ValueError, match="jaw would come within"):
        strategy.start(run_info(), Observation(t=0.0, step=0, joints=dict(SPEC.home), command={}))


# --------------------------------------------------------------------------- the approach geometry
def test_directions_are_radial_and_tangential_from_the_turning_axis():
    xy = PAN_XY + np.array([0.15, 0.05])
    r = (xy - PAN_XY) / np.linalg.norm(xy - PAN_XY)
    assert direction_vector("out", xy, PAN_XY) == pytest.approx(r)
    assert direction_vector("in", xy, PAN_XY) == pytest.approx(-r)
    ccw = direction_vector("ccw", xy, PAN_XY)
    assert ccw == pytest.approx([-r[1], r[0]]) and r[0] * ccw[1] - r[1] * ccw[0] > 0
    assert direction_vector("cw", xy, PAN_XY) == pytest.approx(-ccw)
    assert lead_point(xy, ccw, 0.03) == pytest.approx(xy - 0.03 * ccw)
    with pytest.raises(ValueError, match="unknown direction"):
        direction_vector("up", xy, PAN_XY)
    with pytest.raises(ValueError, match="away from"):
        direction_vector("out", PAN_XY, PAN_XY)


def test_place_heights_follow_pick_places_rule():
    hang, gap, clearance = 0.017, 0.002, 0.03
    for top, over in ((0.0, 0.0), (0.029, 0.0), (0.058, 0.07)):
        release, want, low = place_heights(top, hang, gap, GRIP_SLIP_M, TABLE_TOP_M, clearance,
                                           MIN_CARRY_CLEAR_M, over)
        floor = max(top, TABLE_TOP_M, over)
        assert release == pytest.approx(top + gap + GRIP_SLIP_M + hang)
        assert low == pytest.approx(floor + MIN_CARRY_CLEAR_M + hang)
        assert want == pytest.approx(max(floor + clearance + hang, low))
    assert carry_heights(0.1, 0.088) == pytest.approx([0.1, 0.095, 0.09, 0.088])
    assert carry_heights(0.1, 0.1) == [0.1]


# --------------------------------------------------------------------------- the plan
def to_samples(strategy: CalibrateApproachStrategy) -> bool:
    return strategy.started and strategy.status.phase == SAMPLES


def test_the_samples_go_corner_by_corner_level_by_level_and_repeat_every_direction(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true), corners="1,6", levels_m="0.0,0.029", mount_pairs=0)
    feed = Feed(strategy, sightings(true))
    feed.run(30.0, until=lambda: to_samples(strategy))
    queue = [strategy.planned, *strategy._queue]
    # An unreachable combination is listed once and stands for every repeat.
    assert all(s["repeat"] == 0 and s["why"] for s in strategy._skipped)
    assert len(queue) + 2 * len(strategy._skipped) == 2 * 2 * len(DIRECTIONS) * 2
    keys = [(p.corner, p.level, p.repeat, p.direction) for p in queue]
    assert keys == sorted(keys, key=lambda k: (keys.index(next(x for x in keys if x[0] == k[0])), k[1], k[2],
                                               DIRECTIONS.index(k[3])))
    # The nearest corner to the parked hand first.
    first = queue[0].corner
    hand = KIN.tcp(feed.joints)[:2, 3]
    assert first == min((1, 6), key=lambda k: float(np.linalg.norm(true[k, :2] - hand)))
    assert all(p.carry_z > p.release_z for p in queue)


def test_a_board_printed_at_another_size_stops_the_run_before_it_moves(tmp_path):
    true = corners(square_m=0.0412)
    strategy = built(tmp_path, wrist(true), **SMALL)
    feed = Feed(strategy, sightings(true))
    feed.run(30.0)
    assert feed.done is not None and not feed.done.success
    assert "rogi tablicy nie pasuja do jej wymiarow (kwadrat 41.2 mm, ustawiono 45.0" in feed.done.note
    parked = dict(SPEC.home, shoulder_pan=-80.0)
    assert {k: feed.joints[k] for k in SPEC.arm_joints} == {k: parked[k] for k in SPEC.arm_joints}


def test_too_few_corners_stop_the_run(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true), corner_wait_s=2.0, **SMALL)
    feed = Feed(strategy, sightings(true[:3]))
    feed.run(10.0)
    assert feed.done is not None and "za malo rogow tablicy: 3 z 8" in feed.done.note
    assert feed.notes[0] == "czekam na rogi tablicy z kamer: 0 z 8 (detektor charuco)"


# --------------------------------------------------------------------------- whole runs on the fake feed
def test_a_small_run_saves_a_complete_table_with_the_cameras_error_as_its_correction(tmp_path):
    true = corners()
    offset = np.array([0.004, -0.003, 0.0])
    strategy = built(tmp_path, wrist(true), **SMALL)
    feed = Feed(strategy, sightings(true + offset))
    feed.run(600.0)
    assert feed.done is not None and feed.done.success, feed.notes[-3:]
    table = ApproachTable.load(tmp_path / "approach.json")
    assert table.data["complete"] and len(table.samples) == 2 * 2 * 2
    for s in table.samples:
        assert np.asarray(s["delta_m"])[:2] == pytest.approx(-offset[:2], abs=3e-4)
    assert table.preferred_direction in ("ccw", "cw")
    assert "rog c" in " ".join(feed.notes) and "powtorzenie 2 z 2 (pomiar 8 z 8)" in " ".join(feed.notes)
    assert feed.done.note.startswith("zapisano tablice podejscia: 8 z 8 pomiarow, kierunek ")
    assert strategy.status.phase != PARK and feed.joints == pytest.approx(dict(SPEC.home, shoulder_pan=-80.0,
                                                                              gripper=feed.joints["gripper"]))


def test_a_corner_the_wrist_camera_cannot_see_is_skipped_and_listed(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true, blind={6}), max_skip_share=0.6, look_timeout_s=0.5, **SMALL)
    feed = Feed(strategy, sightings(true))
    feed.run(600.0)
    assert feed.done is not None and feed.done.success, feed.notes[-3:]
    table = ApproachTable.load(tmp_path / "approach.json")
    assert {s["corner"] for s in table.samples} == {1}
    skipped = [s for s in table.data["skipped"] if s["corner"] == 6]
    assert len(skipped) == 4 and all("tablica niewidoczna w kamerze nadgarstka" in s["why"] for s in skipped)


def test_too_many_skips_stop_the_run_and_keep_a_partial_table(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true, blind={6}), look_timeout_s=0.5, **SMALL)
    feed = Feed(strategy, sightings(true))
    feed.run(600.0)
    assert feed.done is not None and not feed.done.success
    assert "za duzo pominietych pomiarow" in feed.done.note
    partial = tmp_path / "approach.json.partial.json"
    assert partial.exists() and not (tmp_path / "approach.json").exists()
    data = json.loads(partial.read_text())
    assert data["complete"] is False and data["samples"]
    assert ApproachTable(data).lookup(true[1, :2], 0.05, "ccw").reason == "table"


def test_an_arm_that_does_not_arrive_is_skipped_after_the_time_out(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true), move_timeout_s=1.0, **SMALL)
    feed = Feed(strategy, sightings(true))
    feed.frozen_after = 6.0
    feed.run(120.0)
    assert any("ramie nie dojechalo" in s["why"] for s in strategy._skipped)


def test_a_stop_mid_run_keeps_what_was_measured(tmp_path):
    true = corners()
    strategy = built(tmp_path, wrist(true), **SMALL)
    feed = Feed(strategy, sightings(true))
    feed.run(600.0, until=lambda: strategy.calibration.samples >= 2)
    strategy.stop("przerwana: STOP")
    data = json.loads((tmp_path / "approach.json.partial.json").read_text())
    assert data["complete"] is False and len(data["samples"]) >= 2
    assert strategy.calibration.phase == CORNERS or strategy.status.phase == "done"


# --------------------------------------------------------------------------- the TCP against a corner
def test_the_tcp_to_corner_vector_comes_in_base_axes():
    """A camera 0.1 m above a corner looking straight down, the TCP 30 mm under the camera and 2 mm aside."""
    yaw = math.radians(20.0)
    R_base_board = board_rotation(yaw)
    corner_base = np.array([0.2, 0.05, 0.0])
    corner_obj = np.array([0.045, 0.09, 0.0])
    # The camera looks down: its z is the base's -z.
    R_base_cam = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    cam = corner_base + np.array([0.004, -0.003, 0.1])
    R_cb = R_base_cam.T @ R_base_board
    t_cb = R_base_cam.T @ (corner_base - cam) - R_cb @ corner_obj
    view = BoardView(R_cb, t_cb, 0.1, 8, 8, 0)
    tcp = cam + np.array([0.002, 0.0, -0.03])
    p_cam = R_base_cam.T @ (tcp - cam)
    got = tcp_to_corner(view, corner_obj, p_cam, R_base_board)
    assert got == pytest.approx(corner_base - tcp, abs=1e-12)


def test_a_flip_recovers_the_hand_frame_bias_and_the_true_measurement():
    bias = np.array([0.0021, -0.0013])
    truth = np.array([0.004, -0.007])
    psi = math.radians(37.0)
    a = truth + rot2(psi) @ bias
    b = truth + rot2(psi + math.pi) @ bias
    got, mean = flip_bias(a, b, psi)
    assert got == pytest.approx(bias) and mean == pytest.approx(truth)
    assert remove_bias(a, psi, got) == pytest.approx(truth)
