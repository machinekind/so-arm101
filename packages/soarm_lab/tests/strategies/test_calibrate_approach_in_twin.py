"""The approach calibration in the twin: the real `charuco` detector and perception, the rendered wrist
camera, and a seeded campaign that checks the table's corrections at held-out points against the truth.

Each stand (`twin_stand.Stand`) lays a ChArUco board on the table and puts the fixed camera `kamera1` where
its calibration says it is not. The calibration then runs as on the desk, its strategy wrapped in `Backlash`
for hidden play in the arm. Afterwards the same arm, without measuring, approaches held-out points inside
the board's corners: each point where the stand's perception would see a block there, at each level and from
each direction. The truth at each point is the true point minus where the TCP really is, plus what the TCP
lacked of its command; the table's prediction is its lookup at the perceived point. The campaign's bar is on
the distance between the two, pooled over the stands.

`SOARM_CAMPAIGN_RUNS` sets the stands (default `RUNS`); `SOARM_CAMPAIGN_DIR` keeps one JSON file per stand and
each stand's table under `tables/`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.twin
pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")
cv2 = pytest.importorskip("cv2")

from machinekind_twin.strategy import StrategyRunner  # noqa: E402
from twin_stand import THICKNESS_M, Clock, Stand, calibrate, drive, truth_tcp  # noqa: E402

from soarm_lab.approach_table import ApproachTable, Hit  # noqa: E402
from soarm_lab.campaign import Run, record, runs_per_scenario  # noqa: E402
from soarm_lab.stand_errors import Backlash  # noqa: E402
from soarm_lab.strategies.calibrate_approach import (  # noqa: E402
    CalibrateApproachSettings,
    CalibrateApproachStrategy,
    Planned,
)

#: The campaign's errors: play at the hand [m], the fixed camera's error at (0.2, 0) [m] and the wrist
#: camera's offset from its model [m].
PLAY_M = 0.006
CAMERA_M = 0.010
MOUNT_M = (0.002, 0.003)
#: Held-out points per stand, and the bars on the pooled prediction error [m].
PROBES = 10
BAR_MEDIAN_M = 0.0015
BAR_P90_M = 0.0025
#: Stands per campaign without `SOARM_CAMPAIGN_RUNS`.
RUNS = 20
SCENARIO = "approach: table, play 6 mm, camera tilted 10 mm, mount 2-3 mm"
#: The held-out approaches' simulated time limit [s].
PROBE_S = 1800.0


def settings(table: Path | None, **extra: Any) -> CalibrateApproachSettings:
    base: dict[str, Any] = {"table_file": "" if table is None else str(table), "merge": False,
                            "thickness_m": THICKNESS_M}
    return CalibrateApproachSettings(**{**base, **extra})


# --------------------------------------------------------------------------- held-out points
@dataclass
class Probe:
    """One approach to a held-out point: where it really lies and is seen, the truth and the table's word."""

    point: int
    level: int
    direction: str
    true_xy: tuple[float, float]
    seen_xy: tuple[float, float]
    truth_m: tuple[float, float] = (math.nan, math.nan)
    predicted_m: tuple[float, float] = (math.nan, math.nan)
    miss: str = ""

    @property
    def error_m(self) -> float:
        if self.miss or not np.all(np.isfinite(self.truth_m)):
            return math.nan
        return float(np.linalg.norm(np.subtract(self.predicted_m, self.truth_m)))


def held_out(stand: Stand, rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` true points inside the board's inner corners (n, 2) [m, base frame]."""
    s, cols, rows = stand.board.square_m, stand.board.squares_x - 1, stand.board.squares_y - 1
    xy = np.column_stack([rng.uniform(s, cols * s, n), rng.uniform(s, rows * s, n)])
    return stand.board_point(xy)[:, :2]


def probe(stand: Stand, twin, clock: Clock, table: ApproachTable, points: np.ndarray,
          levels: list[float], directions: tuple[str, ...]) -> list[Probe]:
    """Approach each point at each level from each direction without measuring, and compare the table's
    correction with the truth."""
    probes, schedule = [], []
    for k, xy in enumerate(points):
        for level, surface in enumerate(levels):
            # The cameras see a block's top: the target's error is the field at that height.
            seen = stand.seen(np.array([[*xy, surface]]))[0, :2]
            for d in directions:
                probes.append(Probe(k, level, d, (float(xy[0]), float(xy[1])),
                                    (float(seen[0]), float(seen[1]))))
                schedule.append((seen, level, d))
    by_order = iter(probes)

    def on_still(strategy: CalibrateApproachStrategy, obs, planned: Planned) -> None:
        p = next(by_order)
        while (p.level, p.direction) != (planned.level, planned.direction) or not np.allclose(p.seen_xy,
                                                                                              planned.xy):
            p = next(by_order)
        actual = truth_tcp(twin)[:2]
        truth = np.array(p.true_xy) - actual + (strategy.tcp_cmd[:2] - planned.xy)
        p.truth_m = (float(truth[0]), float(truth[1]))
        hit = table.lookup(planned.xy, planned.release_z, planned.direction)
        if isinstance(hit, Hit):
            p.predicted_m = (float(hit.delta_xy[0]), float(hit.delta_xy[1]))
        else:
            p.miss = hit.reason

    strategy = CalibrateApproachStrategy(settings(None, levels_m=",".join(str(z) for z in levels)),
                                         schedule=schedule, measure=False, on_still=on_still)
    driven = Backlash(strategy, stand.play()) if stand.play_m else strategy
    runner = StrategyRunner(twin, driven)
    runner.start(threaded=False)
    drive(twin, runner, PROBE_S, clock)
    if runner.status.running:
        runner.stop()
    return probes


# --------------------------------------------------------------------------- one stand of the campaign
@dataclass
class StandRun:
    seed: int
    reason: str
    complete: bool
    samples: int
    planned: int
    skipped: int
    sim_s: float
    wall_s: float
    focal_error: float
    sample_error_m: list[float]
    verify_m: list[float]
    preferred: str
    spread_m: dict[str, float]
    probes: list[Probe]
    artefacts: list[str]
    probe_sim_s: float = 0.0

    def errors(self, direction: str | None = None, level: int | None = None) -> list[float]:
        return [p.error_m for p in self.probes if math.isfinite(p.error_m)
                and (direction is None or p.direction == direction) and (level is None or p.level == level)]


def stand_run(seed: int, *, tables: Path | None = None) -> StandRun:
    rng = np.random.default_rng(50_000 + seed)
    stand = Stand(seed, camera="tilted", camera_m=CAMERA_M, play_m=PLAY_M,
                  mount_m=float(rng.uniform(*MOUNT_M)))
    twin = stand.twin()
    try:
        clock = Clock()
        cal = calibrate(stand, twin, clock, stand.table_path(tables))
        strategy = cal.strategy
        table = strategy.table
        complete = table is not None and bool(table.data.get("complete")) and strategy.status.success
        st = strategy.calibration
        planned = st.samples + sum(s["kind"] == "sample" for s in strategy._skipped)
        focal = abs(st.focal_px / 1200.0 - 1.0) if math.isfinite(st.focal_px) else math.nan
        probes: list[Probe] = []
        t0 = clock.t
        if complete:
            assert table is not None
            levels = [float(level["surface_m"]) for level in table.levels]
            probes = probe(stand, twin, clock, table, held_out(stand, rng, PROBES), levels,
                           tuple(strategy.directions))
        return StandRun(seed, cal.reason, complete, st.samples, planned, st.skipped, cal.sim_s, cal.wall_s,
                        focal, cal.sample_errors(), list(st.verify_m), st.preferred, dict(st.spread_m),
                        probes, cal.artefacts, clock.t - t0)
    finally:
        twin.close()


def as_run(r: StandRun) -> Run:
    errors = r.errors()
    median = float(np.median(errors)) if errors else math.nan
    p90 = float(np.percentile(errors, 90)) if errors else math.nan
    ok = r.complete and not r.artefacts and math.isfinite(p90) and p90 <= BAR_P90_M
    reason = "" if ok else ("calibration did not finish" if not r.complete else
                            "artefact" if r.artefacts else "prediction off by truth")
    details = {k: v for k, v in asdict(r).items() if k != "probes"}
    details.update(why="" if r.complete else r.reason, p90_m=p90, misses=sum(bool(p.miss) for p in r.probes),
                   probes=[{**asdict(p), "error_m": p.error_m} for p in r.probes])
    return Run(SCENARIO, r.seed, ok, 1, median, reason, r.artefacts, r.sim_s, r.wall_s, details)


def pooled(runs: list[StandRun]) -> str:
    def line(label: str, errors: list[float]) -> str:
        if not errors:
            return f"{label}: none"
        e = np.array(errors) * 1000
        return f"{label}: n {len(e)}, median {np.median(e):.2f} mm, p90 {np.percentile(e, 90):.2f} mm, " \
               f"max {e.max():.2f} mm"

    rows = [f"seed {r.seed}: {'ok' if r.complete else 'INCOMPLETE'} samples {r.samples}/{r.planned} "
            f"sim {r.sim_s / 60:.1f} min, probe median "
            f"{np.median(r.errors()) * 1000 if r.errors() else math.nan:.2f} mm, preferred {r.preferred} "
            f"({r.reason})" for r in runs]
    every = [e for r in runs for e in r.errors()]
    rows.append(line("pooled", every))
    for d in ("out", "in", "ccw", "cw"):
        rows.append(line(f"  {d}", [e for r in runs for e in r.errors(direction=d)]))
    for level in range(3):
        rows.append(line(f"  level {level + 1}", [e for r in runs for e in r.errors(level=level)]))
    rows.append(line("  preferred direction", [e for r in runs for e in r.errors(direction=r.preferred)]))
    rows.append(line("samples against truth", [e for r in runs for e in r.sample_error_m]))
    return "\n".join(rows)


# --------------------------------------------------------------------------- the tests
@pytest.mark.render
def test_the_corners_from_the_real_detector_match_the_stands_error_field(twin_robots):
    """The `charuco` detector through the twin's perception puts each corner where the camera's wrong pose
    puts it, within half a millimetre."""
    stand = Stand(3, camera="tilted", camera_m=CAMERA_M)
    twin = stand.twin()
    try:
        clock = Clock()
        perception, detector = stand.perception(twin, clock)
        for _ in range(4):
            clock.t += 0.5
            twin.step(0.02)
            detector.step_once()
        objects = perception.objects()
        true = stand.corners()
        seen = stand.seen(true)
        got = {k: objects[f"charuco c{k}"] for k in range(stand.board.n_corners)
               if f"charuco c{k}" in objects}
        assert len(got) == stand.board.n_corners, sorted(objects)
        for k, pose in got.items():
            assert pose.source == "kamery"
            assert np.linalg.norm(np.asarray(pose.position)[:2] - seen[k, :2]) < 0.0005, k
        # The field is the camera's error, not noise: about 10 mm at the board.
        assert 0.007 < np.median(np.linalg.norm(seen[:, :2] - true[:, :2], axis=1)) < 0.013
    finally:
        twin.close()


SMOKE = {"corners": "1,6", "levels_m": "0.029", "directions": "ccw,cw", "mount_pairs": 1}


@pytest.mark.render
def test_a_small_calibration_without_errors_measures_next_to_nothing(twin_robots, tmp_path):
    """Two corners, one level, two directions on a perfect stand: every correction within 0.7 mm of zero,
    the wrist camera's focal length within 1 %, and a complete table with Polish notes."""
    stand = Stand(0)
    twin = stand.twin()
    try:
        cal = calibrate(stand, twin, Clock(), tmp_path / "approach.json", **SMOKE)
    finally:
        twin.close()
    strategy = cal.strategy
    st = strategy.calibration
    assert strategy.status.success, (cal.reason, st, strategy._skipped)
    assert not cal.artefacts
    table = ApproachTable.load(tmp_path / "approach.json")
    assert table.data["complete"] and len(table.data["spots"]) == 1
    deltas = [np.linalg.norm(np.asarray(s["delta_m"])[:2]) for s in table.samples]
    assert len(deltas) >= 6 and max(deltas) < 0.0007, deltas
    assert abs(st.focal_px / 1200.0 - 1.0) < 0.01 and st.k_views >= 10
    assert cal.reason.startswith("zapisano tablice podejscia") and "kierunek" in cal.reason
    assert max(cal.sample_errors()) < 0.0005


@pytest.mark.render
def test_a_camera_5_mm_off_gives_corrections_of_minus_its_offset(twin_robots, tmp_path):
    stand = Stand(1, camera="uniform", camera_m=0.005)
    twin = stand.twin()
    try:
        cal = calibrate(stand, twin, Clock(), tmp_path / "approach.json", **SMOKE)
    finally:
        twin.close()
    assert cal.strategy.status.success, cal.reason
    offset = stand.error.offset(np.array([[0.2, 0.0, 0.0]]))[0, :2]
    assert np.linalg.norm(offset) == pytest.approx(0.005, abs=1e-9)
    for s in cal.strategy.samples:
        assert np.linalg.norm(np.asarray(s["delta_m"])[:2] + offset) < 0.0007, s


@pytest.mark.render
def test_a_wrist_camera_mounted_off_its_model_is_measured_out(twin_robots, tmp_path):
    """The rendered wrist camera sits 2.5 mm from where robot.yaml puts it, the arm has 6 mm of play: the
    flipped looks give the mount's bias, and every sample is within 0.5 mm of the truth."""
    stand = Stand(2, play_m=PLAY_M, mount_m=0.0025)
    twin = stand.twin()
    try:
        cal = calibrate(stand, twin, Clock(), tmp_path / "approach.json", **SMOKE)
    finally:
        twin.close()
    st = cal.strategy.calibration
    assert cal.strategy.status.success, cal.reason
    assert st.mount_bias_m and np.linalg.norm(st.mount_bias_m[0]) > 0.001
    errors = cal.sample_errors()
    assert len(errors) >= 6 and max(errors) < 0.0005, errors


@pytest.mark.render
@pytest.mark.campaign
def test_seeded_campaign_the_table_predicts_held_out_corrections(twin_robots):
    """A1: on stands with 6 mm of play, the fixed camera tilted 10 mm off and the wrist camera 2-3 mm off its
    model, the table's correction at held-out points lies within 1.5 mm of the truth (median) and 2.5 mm
    (90th percentile), pooled over the stands, at every level and from every direction."""
    runs = []
    for seed in range(runs_per_scenario(RUNS)):
        runs.append(stand_run(seed))
        record(as_run(runs[-1]))
    report = pooled(runs)
    print(report)
    errors = [e for r in runs for e in r.errors()]
    assert all(r.complete for r in runs), report
    assert not any(r.artefacts for r in runs), report
    assert np.median(errors) <= BAR_MEDIAN_M and np.percentile(errors, 90) <= BAR_P90_M, report
