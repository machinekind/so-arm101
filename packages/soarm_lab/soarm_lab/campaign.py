"""Seeded simulation campaigns: one JSON file per run, and a Markdown report over a campaign folder.

The twin tests marked `campaign` run the scenarios; each run they finish goes through `record`, which
writes `<folder>/<scenario>/seed-<seed>.json` when `DIR_ENV` names a folder. `RUNS_ENV` sets the runs
per scenario. `soarm-lab campaign report <folder>` reads the files back and summarises each scenario:
runs, successes, attempts, corrections, the final offset's median and 90th percentile, failure reasons
and simulation artefacts. `jobs/overnight.sh` runs the whole campaign and writes `report.md`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The folder the runs are written to; unset writes nothing.
DIR_ENV = "SOARM_CAMPAIGN_DIR"
#: Runs per scenario; unset leaves each scenario's own default.
RUNS_ENV = "SOARM_CAMPAIGN_RUNS"
#: Scenario groups in report order, the first word of a scenario's name, with what their attempts and
#: offsets count.
GROUPS = {
    "grasp": "attempts are closes until the lift; the offset is the jaw tips' midpoint from the block's "
             "centre as the jaw last starts closing",
    "pick_place": "attempts are picks; the offset is the placed block's centre from its place",
    "build": "attempts are pick_place runs; the offset is the largest centre offset in the finished "
             "structure",
}


@dataclass
class Run:
    """One seeded run of a scenario, judged by the simulation's own poses."""

    scenario: str
    seed: int
    success: bool
    #: Tries the scenario counts: closes of a grasp, picks of a place, pick_place runs of a build.
    attempts: int
    #: The final offset the scenario measures [m]; NaN when the run got too far to measure it.
    offset_m: float = math.nan
    #: Why the run failed, in a few words shared by runs that failed alike; '' on success. The details
    #: may say more under "why".
    reason: str = ""
    #: Simulation artefacts: blocks that sank, flew, were dragged or stayed in the jaws.
    artefacts: list[str] = field(default_factory=list)
    sim_s: float = 0.0
    wall_s: float = 0.0
    #: Whatever else the scenario keeps: settings, scene, status and timeline.
    details: dict[str, Any] = field(default_factory=dict)
    #: Looks between the open jaw tips that found the block off centre, each followed by a correction of the
    #: aim or a new attempt; 0 without the wrist camera.
    corrections: int = 0


@dataclass
class Summary:
    """One scenario's runs, summed up."""

    scenario: str
    runs: int
    successes: int
    #: Runs per attempt count, and per correction count.
    attempts: dict[int, int]
    corrections: dict[int, int]
    #: Final offsets of the runs that measured one [m].
    offsets_m: list[float]
    #: The failed seeds per reason, the most common reason first, and what each failed run said.
    reasons: dict[str, list[int]]
    whys: list[str]
    #: Runs with at least one artefact, and every artefact seen with its seed.
    artefact_runs: int
    artefacts: list[str]
    wall_s: float

    @property
    def offset_median_m(self) -> float:
        return float(np.median(self.offsets_m)) if self.offsets_m else math.nan

    @property
    def offset_p90_m(self) -> float:
        return float(np.percentile(self.offsets_m, 90)) if self.offsets_m else math.nan


def runs_per_scenario(default: int) -> int:
    """`RUNS_ENV` when set, else `default`."""
    text = os.environ.get(RUNS_ENV, "").strip()
    return int(text) if text else default


def campaign_dir() -> Path | None:
    """The folder named by `DIR_ENV`, or None."""
    text = os.environ.get(DIR_ENV, "").strip()
    return Path(text) if text else None


def slug(text: str) -> str:
    """A folder name for a scenario: lower case, words joined by '-'."""
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def jsonable(value: Any) -> Any:
    """`value` as plain JSON data: dataclasses and mappings as objects, sequences and arrays as lists,
    numpy scalars as numbers, and NaN or infinity as None."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def run_path(folder: Path, scenario: str, seed: int) -> Path:
    return folder / slug(scenario) / f"seed-{seed:04d}.json"


def record(run: Run, folder: Path | None = None) -> Path | None:
    """Write `run` under `folder`, or under `DIR_ENV` when `folder` is None; None when neither is set."""
    folder = folder if folder is not None else campaign_dir()
    if folder is None:
        return None
    path = run_path(folder, run.scenario, run.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(run), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load(folder: Path) -> list[Run]:
    """Every run written under `folder`."""
    runs = []
    for path in sorted(Path(folder).glob("*/seed-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        offset = data.get("offset_m")
        runs.append(Run(scenario=str(data["scenario"]), seed=int(data["seed"]), success=bool(data["success"]),
                        attempts=int(data["attempts"]),
                        offset_m=math.nan if offset is None else float(offset),
                        reason=str(data.get("reason") or ""), artefacts=list(data.get("artefacts") or []),
                        sim_s=float(data.get("sim_s") or 0.0), wall_s=float(data.get("wall_s") or 0.0),
                        corrections=int(data.get("corrections") or 0),
                        details=dict(data.get("details") or {})))
    return runs


def _order(scenario: str) -> tuple[int, list[tuple[int, str]]]:
    """Group order, then the name with its numbers compared as numbers: "5 mm" before "10 mm"."""
    group = scenario.split(":")[0].strip()
    parts = [(int(p), "") if p.isdigit() else (-1, p) for p in re.split(r"(\d+)", scenario)]
    order = list(GROUPS)
    return (order.index(group) if group in order else len(order), parts)


def summarise(runs: Iterable[Run]) -> list[Summary]:
    """One summary per scenario, grasp first, then pick_place, then build."""
    by: dict[str, list[Run]] = {}
    for run in runs:
        by.setdefault(run.scenario, []).append(run)
    out = []
    for scenario in sorted(by, key=_order):
        group = sorted(by[scenario], key=lambda r: r.seed)
        failed = [r for r in group if not r.success]
        reasons: dict[str, list[int]] = {}
        for r in failed:
            reasons.setdefault(r.reason or "no reason given", []).append(r.seed)
        out.append(Summary(
            scenario=scenario, runs=len(group), successes=len(group) - len(failed),
            attempts=dict(sorted(Counter(r.attempts for r in group).items())),
            corrections=dict(sorted(Counter(r.corrections for r in group).items())),
            offsets_m=[r.offset_m for r in group if math.isfinite(r.offset_m)],
            reasons=dict(sorted(reasons.items(), key=lambda item: -len(item[1]))),
            whys=[f"seed {r.seed}: {r.details['why']}" for r in failed if r.details.get("why")],
            artefact_runs=sum(bool(r.artefacts) for r in group),
            artefacts=[f"seed {r.seed}: {a}" for r in group for a in r.artefacts],
            wall_s=sum(r.wall_s for r in group)))
    return out


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _mm(metres: float) -> str:
    return f"{metres * 1000:.1f}" if math.isfinite(metres) else "-"


def report(runs: Iterable[Run], folder: Path | None = None, title: str = "Simulation campaign") -> str:
    """The campaign as Markdown: a table of every scenario, then each scenario's failures and artefacts."""
    summaries = summarise(runs)
    lines = [f"# {title}", ""]
    if folder is not None:
        lines += [f"Runs: one JSON file per run under `{folder}`, `<scenario>/seed-<seed>.json`.", ""]
    lines += ["| scenario | ok / runs | attempts: runs | corrections: runs | final offset median / p90 [mm] "
              "| runs with artefacts | wall [min] |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        attempts = ", ".join(f"{k}: {n}" for k, n in s.attempts.items())
        corrections = ", ".join(f"{k}: {n}" for k, n in s.corrections.items())
        offsets = f"{_mm(s.offset_median_m)} / {_mm(s.offset_p90_m)}"
        if len(s.offsets_m) < s.runs:
            offsets += f" (of {len(s.offsets_m)})"
        lines.append(f"| {s.scenario} | {s.successes} / {s.runs} | {attempts} | {corrections} | {offsets} | "
                     f"{s.artefact_runs} | {s.wall_s / 60:.1f} |")
    groups = {s.scenario.split(":")[0].strip() for s in summaries}
    notes = [f"- {group}: {note}." for group, note in GROUPS.items() if group in groups]
    if notes:
        lines += ["", "A run is ok when it did its task by the simulation's poses, without artefacts. "
                      "Corrections are looks between the open jaw tips that found the block off centre.",
                  *notes]
    for s in summaries:
        if not s.reasons and not s.artefacts:
            continue
        lines += ["", f"## {s.scenario}", ""]
        if s.reasons:
            lines += ["Failures:", ""] + [f"- {len(seeds)} x {reason} (seeds {', '.join(map(str, seeds))})"
                                          for reason, seeds in s.reasons.items()]
        if s.whys:
            lines += ["", "What the failed runs said:", ""] + [f"- {_short(why)}" for why in s.whys]
        if s.artefacts:
            lines += ["", "Artefacts:", ""] + [f"- {a}" for a in s.artefacts]
    return "\n".join(lines) + "\n"
