"""Campaign records: one JSON file per run, read back and summed up per scenario as Markdown."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from soarm_lab.campaign import (
    DIR_ENV,
    RUNS_ENV,
    Run,
    jsonable,
    load,
    record,
    report,
    run_path,
    runs_per_scenario,
    slug,
    summarise,
)
from soarm_lab.cli import main


def run(scenario: str, seed: int, success: bool = True, attempts: int = 1, offset_m: float = 0.001,
        reason: str = "", **kwargs) -> Run:
    return Run(scenario, seed, success, attempts, offset_m, reason, **kwargs)


def test_runs_per_scenario_and_the_folder_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(RUNS_ENV, raising=False)
    monkeypatch.delenv(DIR_ENV, raising=False)
    assert runs_per_scenario(10) == 10
    assert record(run("grasp: a", 0)) is None
    monkeypatch.setenv(RUNS_ENV, "3")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))
    assert runs_per_scenario(10) == 3
    assert record(run("grasp: detector 1.5 cm", 7)) == tmp_path / "grasp-detector-1.5-cm" / "seed-0007.json"


def test_a_scenario_name_becomes_a_folder_name():
    assert slug("pick_place: cube on cube, cameras 5 mm off") == "pick-place-cube-on-cube-cameras-5-mm-off"
    assert run_path(Path("out"), "build: tower", 103) == Path("out/build-tower/seed-0103.json")


@dataclass
class Status:
    phase: str = "done"
    xy: tuple[float, float] = (0.1, 0.2)
    error: float = math.nan
    looks: list[str] = field(default_factory=lambda: ["centred"])


def test_details_become_plain_json():
    data = jsonable({"status": Status(), "array": np.array([1.0, np.inf]), "n": np.int64(3),
                     "f": np.float32(0.5), "set": {2}, "other": Path("x")})
    assert data == {"status": {"phase": "done", "xy": [0.1, 0.2], "error": None, "looks": ["centred"]},
                    "array": [1.0, None], "n": 3, "f": 0.5, "set": [2], "other": "x"}
    json.dumps(data, allow_nan=False)


def test_a_run_is_written_and_read_back(tmp_path):
    written = run("pick_place: cube on cube", 4, False, 2, math.nan, "fell_off",
                  artefacts=["red sank 0.8 mm"], sim_s=12.5, wall_s=1.5, corrections=1,
                  details={"why": "nie lezy", "pose": (0.2, 0.06)})
    path = record(written, tmp_path)
    assert path is not None and json.loads(path.read_text())["offset_m"] is None
    (back,) = load(tmp_path)
    assert math.isnan(back.offset_m)
    back.offset_m = written.offset_m = 0.0
    assert back == Run(**{**written.__dict__, "details": {"why": "nie lezy", "pose": [0.2, 0.06]}})


def test_a_summary_counts_runs_attempts_offsets_and_reasons():
    runs = [run("build: tower", 1, attempts=2, offset_m=0.001),
            run("build: tower", 0, attempts=2, offset_m=0.003),
            run("build: tower", 2, False, 1, math.nan, "stuck after unreachable",
                details={"why": "red at 14 mm"}),
            run("build: tower", 3, False, 3, 0.002, "stuck after unreachable"),
            run("build: tower", 4, False, 2, 0.004, "artefact", artefacts=["red sank 0.9 mm into the table"]),
            run("grasp: play", 0, attempts=1, offset_m=0.0005)]
    grasp, tower = summarise(runs)
    assert grasp.scenario == "grasp: play" and tower.scenario == "build: tower"
    assert (tower.runs, tower.successes) == (5, 2)
    assert tower.attempts == {1: 1, 2: 3, 3: 1} and tower.corrections == {0: 5}
    assert tower.offsets_m == [0.003, 0.001, 0.002, 0.004]
    assert tower.offset_median_m == pytest.approx(0.0025)
    assert tower.offset_p90_m == pytest.approx(np.percentile([1, 2, 3, 4], 90) / 1000)
    assert tower.reasons == {"stuck after unreachable": [2, 3], "artefact": [4]}
    assert tower.whys == ["seed 2: red at 14 mm"]
    assert tower.artefact_runs == 1 and tower.artefacts == ["seed 4: red sank 0.9 mm into the table"]


def test_scenarios_come_grasp_first_and_numbers_in_order():
    names = ["build: tower", "other: thing", "pick_place: cameras 10 mm off", "pick_place: cameras 5 mm off",
             "grasp: b", "grasp: a"]
    assert [s.scenario for s in summarise(run(n, 0) for n in names)] == [
        "grasp: a", "grasp: b", "pick_place: cameras 5 mm off", "pick_place: cameras 10 mm off",
        "build: tower", "other: thing"]


def test_the_report_has_a_row_per_scenario_and_the_failures():
    runs = [run("grasp: detector 1.5 cm", 0, offset_m=0.0002, wall_s=30.0),
            run("grasp: detector 1.5 cm", 1, attempts=3, offset_m=0.0004, wall_s=30.0, corrections=2),
            run("build: tower", 102, False, 2, math.nan, "stuck after unreachable",
                details={"why": "krok 2/2: red cube -> green cube (unreachable)"})]
    text = report(runs, Path("/tmp/campaign"), title="Overnight")
    assert text.startswith("# Overnight\n\nRuns: one JSON file per run under `/tmp/campaign`")
    assert "| grasp: detector 1.5 cm | 2 / 2 | 1: 1, 3: 1 | 0: 1, 2: 1 | 0.3 / 0.4 | 0 | 1.0 |" in text
    assert "| build: tower | 0 / 1 | 2: 1 | 0: 1 | - / - (of 0) | 0 | 0.0 |" in text
    assert "- grasp: attempts are closes until the lift" in text
    assert "- build: attempts are pick_place runs" in text
    assert "- pick_place:" not in text
    assert "## build: tower\n\nFailures:\n\n- 1 x stuck after unreachable (seeds 102)" in text
    assert "- seed 102: krok 2/2: red cube -> green cube (unreachable)" in text
    assert "## grasp" not in text


def test_the_command_writes_the_report(tmp_path, capsys):
    record(run("grasp: a", 0), tmp_path)
    out = tmp_path / "report.md"
    assert main(["campaign", "report", str(tmp_path), "--out", str(out), "--title", "Night"]) == 0
    assert out.read_text().startswith("# Night\n") and capsys.readouterr().out == f"1 runs -> {out}\n"
    assert main(["campaign", "report", str(tmp_path)]) == 0
    assert "| grasp: a | 1 / 1 |" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="no campaign runs"):
        main(["campaign", "report", str(tmp_path / "empty")])
