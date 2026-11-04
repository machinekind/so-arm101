"""The build loop with a scripted pick_place double, a fake table and a fake planner: no arm, twin or model.

The double runs a few ticks, moves its block on the fake table as its settings say (or as its script says),
and ends with the scripted outcome. The table reports every block fresh on every tick.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo

import soarm
from soarm_lab.blocks.classic_world import BLOCKS
from soarm_lab.planner.cli import FakePlanner, LoggedOut, PlannerTimeout, Scripted
from soarm_lab.planner.placing import arm_reach
from soarm_lab.robot import load_robot
from soarm_lab.strategies.build import (
    ACTION_DECISIONS,
    ALREADY_DONE,
    BAD_PLAN,
    BUDGET_SPENT,
    BUILT,
    PLANNER_FAILED,
    QUESTION,
    STEP_STOPPED,
    STUCK,
    BuildSettings,
    BuildStrategy,
    fixed_plan,
)
from soarm_lab.strategies.judges import Decision
from soarm_lab.strategies.pick_place import PickPlaceSettings

SPEC = load_robot()
HOME = {name: float(v) for name, v in SPEC.home.items()}
RUN = RunInfo(robot=SPEC.name, joints=tuple(SPEC.joints), gripper=SPEC.gripper, limits=dict(SPEC.limits),
              max_vel=dict(SPEC.max_vel), folder=soarm.ROBOT_DIR, simulated=True)
HALF = BLOCKS["cube"].half_size[2]
START = {"blue cube": (0.2, 0.0), "green cube": (0.18, 0.1), "red cube": (0.18, -0.1)}


def on(pick: str, target: str) -> dict[str, Any]:
    return {"pick": pick, "place": {"relation": "on", "targets": [target], "side": None, "gap_m": None,
                                    "x_m": None, "y_m": None, "yaw_deg": None}, "why": ""}


TOWER = [on("green cube", "blue cube"), on("red cube", "green cube")]
#: The same tower in the short form plan_json takes.
TOWER_JSON = ('[{"pick": "green cube", "place": {"relation": "on", "targets": ["blue cube"]}},'
              ' {"pick": "red cube", "place": {"relation": "on", "targets": ["green cube"]}}]')


def plan(*steps: dict[str, Any], done: bool = False, question: str = "") -> dict[str, Any]:
    return {"steps": list(steps), "done": done, "question": question}


class Table:
    """Where the blocks are and how they are turned; the cameras see each one fresh on every tick."""

    def __init__(self, start: dict[str, tuple[float, float]] = START):
        self.at = {label: np.array([*xy, HALF]) for label, xy in start.items()}
        self.yaw = {label: 0.0 for label in start}
        self.hidden: set[str] = set()

    def put(self, label: str, xy: tuple[float, float], level: int = 1) -> None:
        self.at[label] = np.array([*xy, (2 * level - 1) * HALF])

    def stack(self, label: str, target: str) -> None:
        self.at[label] = self.at[target] + np.array([0.0, 0.0, 2 * HALF])

    def at_pose(self, label: str, s: PickPlaceSettings) -> None:
        """Set `label` down where the settings' exact pose says, lying as the catalog describes it."""
        lying = BLOCKS[label.split()[-1]].half_size[2]
        self.at[label] = np.array([s.place_x_m, s.place_y_m, s.place_z_m + lying])
        self.yaw[label] = math.radians(s.place_yaw_deg)

    def objects(self) -> dict[str, ObjectPose]:
        return {label: ObjectPose(p.copy(), rot_z(self.yaw.get(label, 0.0)), "kamery", 0.0)
                for label, p in self.at.items() if label not in self.hidden}


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Plays:
    """How one step's run plays out: its outcome after `ticks`, and what it did to the table."""

    outcome: str = "placed"
    #: None does what the settings say for "placed" and nothing otherwise.
    effect: Callable[[Table, PickPlaceSettings], None] | None = None
    ticks: int = 3
    raises: bool = False
    decisions: tuple[Decision, ...] = ()
    #: Joint targets the step ends with, over the start pose.
    ends_at: dict[str, float] = field(default_factory=dict)
    #: Picks of a fallen block the step made before it ended (`PlaceStatus.retries`).
    retries: int = 0


class FakeStep:
    """A pick_place stand-in that plays a script."""

    def __init__(self, settings: PickPlaceSettings, plays: Plays, table: Table):
        self.settings, self.plays, self.table = settings, plays, table
        self.place = SimpleNamespace(outcome="", retries=0)
        self.status = SimpleNamespace(stopped_because="")
        self.stops: list[str] = []
        self.obs: list[Observation] = []
        self._decisions = list(plays.decisions)

    def start(self, run: RunInfo, obs: Observation) -> None:
        self.obs.append(obs)

    def act(self, obs: Observation) -> Action:
        self.obs.append(obs)
        if self.plays.raises:
            raise RuntimeError("ramie nie nadaza za celem")
        if len(self.obs) <= self.plays.ticks:
            return Action(target=HOME, note="niose blok")
        s = self.settings
        if self.plays.effect is not None:
            self.plays.effect(self.table, s)
        elif self.plays.outcome == "placed":
            if s.place_on:
                self.table.stack(s.label, s.place_on)
            else:
                self.table.put(s.label, (s.place_x_m, s.place_y_m))
        self.place.outcome, self.place.retries = self.plays.outcome, self.plays.retries
        self.status.stopped_because = f"{self.plays.outcome} ({s.label})"
        return Action(target={**HOME, **self.plays.ends_at}, done=True,
                      success=self.plays.outcome == "placed", note=self.status.stopped_because)

    def stop(self, reason: str) -> None:
        self.stops.append(reason)

    def drain_decisions(self) -> tuple[Decision, ...]:
        out, self._decisions = tuple(self._decisions), []
        return out


@dataclass
class Steps:
    """The step factory: each step built plays the next script entry."""

    table: Table
    script: list[Plays]
    made: list[FakeStep] = field(default_factory=list)

    def __call__(self, settings: PickPlaceSettings) -> FakeStep:
        assert self.script, f"no scripted step left for {settings.label} -> {settings.place_on}"
        step = FakeStep(settings, self.script.pop(0), self.table)
        self.made.append(step)
        return step

    def moves(self) -> list[tuple[str, str]]:
        return [(s.settings.label, s.settings.place_on or f"({s.settings.place_x_m:.3f}, "
                                                          f"{s.settings.place_y_m:.3f})") for s in self.made]


def build(table: Table, *script: Plays | str, planner: FakePlanner | None = None,
          **settings: Any) -> tuple[BuildStrategy, Steps]:
    steps = Steps(table, [p if isinstance(p, Plays) else Plays(p) for p in script])
    settings.setdefault("judge", "none")
    settings.setdefault("wrist_camera", "")
    if planner is None:
        settings.setdefault("plan_json", TOWER_JSON)
    else:
        settings.setdefault("task", "a tower of 3 cubes")
    strategy = BuildStrategy(BuildSettings(**settings), planner=planner, step_factory=steps, threaded=False)
    return strategy, steps


@dataclass
class Ran:
    action: Action
    notes: list[str]


def run(strategy: BuildStrategy, table: Table, seconds: float = 300.0, *,
        every: Callable[[int], None] | None = None) -> Ran:
    """Tick until the run ends; the time on `obs.t` goes by 1/hz per tick."""
    strategy.start(RUN, Observation(t=0.0, step=0, joints=HOME, command={}, objects=table.objects()))
    notes: list[str] = []
    for k in range(1, round(seconds * strategy.hz)):
        if every is not None:
            every(k)
        action = strategy.act(Observation(t=k / strategy.hz, step=k, joints=HOME, command={},
                                          objects=table.objects()))
        assert set(action.target) == set(SPEC.joints)
        if not notes or notes[-1] != action.note:
            notes.append(action.note)
        if action.done:
            strategy.stop("zadanie wykonane" if action.success else "strategia skonczyla")
            return Ran(action, notes)
    raise AssertionError(f"the run did not end: {strategy.status}")


def tower_stands(table: Table) -> bool:
    blue, green, red = (table.at[k] for k in ("blue cube", "green cube", "red cube"))
    return (np.allclose(green, blue + [0, 0, 2 * HALF]) and np.allclose(red, green + [0, 0, 2 * HALF]))


# --------------------------------------------------------------------------- the happy path


def test_a_fixed_plan_builds_a_tower_step_by_step() -> None:
    table = Table()
    strategy, steps = build(table, "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT, strategy.status
    assert tower_stands(table)
    assert steps.moves() == [("green cube", "blue cube"), ("red cube", "green cube")]
    assert [(s.settings.block, s.settings.place_on_block) for s in steps.made] == [("cube", "cube")] * 2
    assert "krok 1/2: green cube -> blue cube (placed)" in ran.notes
    assert "krok 2/2: red cube -> green cube (placed)" in ran.notes
    assert any(n.startswith("krok 2/2: red cube -> green cube: niose blok") for n in ran.notes)
    assert ran.action.note.startswith("zbudowane: 2 krokow")
    assert [s.stops for s in steps.made] == [["zadanie wykonane"]] * 2
    assert strategy.status.done == 2 and strategy.status.replans == 0 and strategy.decisions == []


def test_each_step_gets_a_fresh_clock_and_the_runs_settings() -> None:
    table = Table()
    strategy, steps = build(table, "placed", "placed", step_time_limit_s=90.0, max_tilt_deg=20.0,
                            focal_px=490.0, tip_pixel_x=632.0, tip_pixel_y=300.0, speed_mps=0.08)

    run(strategy, table)

    for made in steps.made:
        s = made.settings
        assert type(s) is PickPlaceSettings and s.time_limit_s == 90.0 and s.max_tilt_deg == 20.0
        assert (s.focal_px, s.tip_pixel_x, s.tip_pixel_y, s.speed_mps) == (490.0, 632.0, 300.0, 0.08)
        assert s.judge == "none" and s.wrist_camera == ""
        assert made.obs[0].t == made.obs[1].t == 0.0
        assert made.obs[2].t == pytest.approx(1 / strategy.hz) and made.obs[2].step == 1


def test_each_step_gets_the_approach_tables_settings() -> None:
    table = Table()
    strategy, steps = build(table, "placed", "placed", approach_file="~/stands/desk/approach.json",
                            approach_direction="cw", approach_lead_m=0.025, pick_shift="auto",
                            table_check_m=0.005, max_table_age_days=30.0)

    run(strategy, table)

    assert len(steps.made) == 2
    for made in steps.made:
        s = made.settings
        assert (s.approach_file, s.approach_direction, s.pick_shift) == ("~/stands/desk/approach.json", "cw",
                                                                          "auto")
        assert (s.approach_lead_m, s.table_check_m, s.max_table_age_days) == (0.025, 0.005, 30.0)


def test_the_model_plans_off_a_prompt_with_the_world_and_its_call_is_a_decision() -> None:
    table = Table()
    planner = FakePlanner([plan(*TOWER)])
    strategy, steps = build(table, "placed", "placed", planner=planner)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT and tower_stands(table)
    (prompt, schema), = planner.asked
    assert prompt.startswith("You plan moves for a robot arm")
    assert prompt.endswith("Task:\na tower of 3 cubes\n")
    assert '"blue cube": {"kind": "cube"' in prompt
    assert schema["required"] == ["steps", "done", "question"]
    decision, = strategy.decisions
    assert decision.question == "build: plan" and decision.state["prompt"] == prompt
    assert decision.state["prompt_version"] and decision.response_s is not None
    assert '"pick": "green cube"' in decision.answer
    if not ACTION_DECISIONS:
        assert strategy.drain_decisions() == (decision,) and strategy.drain_decisions() == ()


def test_a_task_already_done_ends_without_moving() -> None:
    table = Table()
    table.stack("green cube", "blue cube")
    table.stack("red cube", "green cube")
    planner = FakePlanner([plan(done=True)])
    strategy, steps = build(table, planner=planner)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == ALREADY_DONE and steps.made == []
    assert ran.action.note == "model: zadanie juz wykonane"
    fixed, steps = build(table)
    ran = run(fixed, table)
    assert ran.action.success and fixed.status.outcome == ALREADY_DONE and steps.made == []


def test_a_fixed_plan_skips_steps_that_already_hold() -> None:
    table = Table()
    table.stack("green cube", "blue cube")
    strategy, steps = build(table, "placed")

    ran = run(strategy, table)

    assert ran.action.success and steps.moves() == [("red cube", "green cube")] and tower_stands(table)


def test_a_question_ends_the_run_with_it() -> None:
    table = Table()
    strategy, steps = build(table, planner=FakePlanner([plan(question="Ktory kolor na gorze?")]))

    ran = run(strategy, table)

    assert not ran.action.success and strategy.status.outcome == QUESTION and steps.made == []
    assert strategy.status.question == "Ktory kolor na gorze?"
    assert ran.action.note == "pytanie modelu: Ktory kolor na gorze?"


# --------------------------------------------------------------------------- failures and replans


def drop_at(xy: tuple[float, float]) -> Callable[[Table, PickPlaceSettings], None]:
    return lambda table, s: table.put(s.label, xy)


def test_a_drop_sends_the_model_the_failure_and_its_new_plan_runs() -> None:
    table = Table()
    planner = FakePlanner([plan(*TOWER), plan(*TOWER)])
    strategy, steps = build(table, Plays("dropped", drop_at((0.17, 0.04))), "placed", "placed",
                            planner=planner)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT and tower_stands(table)
    assert strategy.status.replans == 1 and len(planner.asked) == 2
    replan = planner.asked[1][0]
    assert "This step failed:" in replan and '"outcome": "dropped: dropped (green cube)"' in replan
    assert '"green cube": {"kind": "cube", "size_mm": [29, 29, 29], "x_m": 0.17, "y_m": 0.04' in replan
    assert [r.ok for r in strategy.status.records] == [False, True, True]
    assert [d.question for d in strategy.decisions] == ["build: plan", "build: replan"]


def test_a_fixed_plan_replans_from_the_scene_after_a_drop() -> None:
    table = Table()
    strategy, steps = build(table, "placed", Plays("fell_off", drop_at((0.22, 0.06))), "placed")

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT and tower_stands(table)
    assert steps.moves() == [("green cube", "blue cube"), ("red cube", "green cube"),
                             ("red cube", "green cube")]
    assert strategy.status.replans == 1


def test_a_block_not_found_is_waited_for_and_the_step_runs_once_more() -> None:
    table = Table()
    strategy, steps = build(table, "not_found", "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and tower_stands(table)
    assert steps.moves()[:2] == [("green cube", "blue cube")] * 2
    assert strategy.status.retries == 1 and strategy.status.replans == 0
    assert "krok 1/2: green cube -> blue cube (not_found)" in ran.notes


def test_a_step_found_in_place_after_a_late_sighting_counts_as_done() -> None:
    table = Table()

    def placed_unseen(t: Table, s: PickPlaceSettings) -> None:
        t.stack(s.label, s.place_on)

    strategy, steps = build(table, Plays("not_found", placed_unseen), "placed")

    ran = run(strategy, table)

    assert ran.action.success and len(steps.made) == 2 and strategy.status.retries == 0


def test_a_second_miss_of_the_same_step_replans() -> None:
    table = Table()
    strategy, steps = build(table, "target_not_seen", "target_moved", "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.retries == 1 and strategy.status.replans == 1


def test_a_step_whose_pick_place_already_tried_again_is_replanned_not_run_once_more() -> None:
    """pick_place picked the knocked block again and the target moved once more: the build replans."""
    table = Table()
    strategy, steps = build(table, "placed", Plays("target_moved", retries=2), "placed")

    ran = run(strategy, table)

    assert ran.action.success and tower_stands(table)
    assert strategy.status.retries == 0 and strategy.status.replans == 1
    assert strategy.status.place_retries == 2
    assert steps.moves() == [("green cube", "blue cube"), ("red cube", "green cube"),
                             ("red cube", "green cube")]


def test_a_block_that_stays_unseen_is_not_retried() -> None:
    table = Table()
    planner = FakePlanner([plan(*TOWER), plan(question="Gdzie jest green cube?")])
    strategy, steps = build(table, "not_found", planner=planner, wait_for_block_s=2.0)

    def hide(k: int) -> None:
        if strategy.status.phase == "settle":
            table.hidden.add("green cube")

    ran = run(strategy, table, every=hide)

    assert strategy.status.retries == 0 and strategy.status.replans == 1 and len(steps.made) == 1
    assert strategy.status.outcome == QUESTION and not ran.action.success
    assert '"green cube"' not in planner.asked[1][0].split("This step failed:")[0]


def test_an_unreachable_place_asks_the_model_for_another() -> None:
    table = Table()
    planner = FakePlanner([plan(*TOWER), plan(on("red cube", "blue cube"), on("green cube", "red cube"))])
    strategy, steps = build(table, "unreachable", "placed", "placed", planner=planner)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT
    assert steps.moves() == [("green cube", "blue cube"), ("red cube", "blue cube"),
                             ("green cube", "red cube")]
    assert '"outcome": "unreachable: unreachable (green cube)"' in planner.asked[1][0]


def test_a_fixed_plan_tries_an_unreachable_step_once_more_from_the_scene() -> None:
    table = Table()
    strategy, steps = build(table, "unreachable", "placed", "placed")
    ran = run(strategy, table)
    assert ran.action.success and strategy.status.replans == 1 and tower_stands(table)

    table = Table()
    strategy, steps = build(table, "unreachable", "unreachable")
    ran = run(strategy, table)
    assert not ran.action.success and strategy.status.outcome == STUCK and len(steps.made) == 2
    assert ran.action.note == ("krok 1/2: green cube -> blue cube (unreachable) drugi raz: unreachable "
                               "(green cube)")


def test_an_unreachable_step_counts_afresh_after_a_verified_step() -> None:
    table = Table()
    knock = Plays("unreachable", lambda t, s: t.put("green cube", (0.17, -0.05)))
    strategy, steps = build(table, "placed", knock, "placed", "unreachable", "unreachable")
    ran = run(strategy, table)
    assert strategy.status.outcome == STUCK and not ran.action.success
    assert strategy.status.replans == 2 and len(steps.made) == 5


def test_a_placed_step_that_does_not_hold_is_replanned() -> None:
    table = Table()
    strategy, steps = build(table, Plays("placed", lambda t, s: None), "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and tower_stands(table) and strategy.status.replans == 1
    assert strategy.status.records[0].outcome.startswith("placed, but 'green cube' is ")


def test_a_knocked_tower_is_rebuilt_from_the_fixed_plan() -> None:
    """The second step lands, but the cube under it was knocked off the base: the top cube goes aside, then
    the tower is built again."""
    table = Table()

    def knocked(t: Table, s: PickPlaceSettings) -> None:
        t.put("green cube", (0.17, -0.05))
        t.stack("red cube", "green cube")

    strategy, steps = build(table, "placed", Plays("placed", knocked), "placed", "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT and tower_stands(table)
    moves = steps.moves()
    assert moves[:2] == [("green cube", "blue cube"), ("red cube", "green cube")]
    assert moves[2][0] == "red cube" and steps.made[2].settings.place_on == ""
    assert moves[3:] == [("green cube", "blue cube"), ("red cube", "green cube")]
    assert strategy.status.replans == 1
    assert strategy.status.records[2].outcome.startswith("no longer holds: 'green cube' ")


def test_a_tower_knocked_while_a_step_waits_is_noticed_before_the_step_runs_again() -> None:
    table = Table()
    knock = Plays("target_not_seen", lambda t, s: t.put("green cube", (0.17, -0.05)))
    strategy, steps = build(table, "placed", knock, "placed", "placed")

    ran = run(strategy, table)

    assert ran.action.success and tower_stands(table)
    assert steps.moves() == [("green cube", "blue cube"), ("red cube", "green cube"),
                             ("green cube", "blue cube"), ("red cube", "green cube")]
    assert strategy.status.replans == 1


def test_the_replan_budget_ends_the_run() -> None:
    table = Table()
    planner = FakePlanner([plan(*TOWER), plan(*TOWER)])
    strategy, steps = build(table, Plays("dropped", drop_at((0.17, 0.04))),
                            Plays("dropped", drop_at((0.16, 0.0))), planner=planner, max_replans=1)

    ran = run(strategy, table)

    assert not ran.action.success and strategy.status.outcome == BUDGET_SPENT
    assert strategy.status.replans == 1 and len(planner.asked) == 2
    assert ran.action.note.startswith("wyczerpany limit przeplanowan (1): green cube on blue cube: dropped")


def test_two_bad_answers_end_the_run() -> None:
    table = Table()
    planner = FakePlanner([plan(on("purple cube", "blue cube")), plan(*TOWER, *TOWER)])
    strategy, steps = build(table, planner=planner, max_steps=3)

    ran = run(strategy, table)

    assert not ran.action.success and strategy.status.outcome == BAD_PLAN and steps.made == []
    assert strategy.status.bad_answers == 2
    second = planner.asked[1][0]
    assert ("Your previous answer cannot run:\n- step 1 (purple cube on blue cube): no object labelled"
            in second)
    assert ran.action.note == "plan modelu znow bledny: the plan has 4 steps; at most 3 are allowed"
    assert [d.question for d in strategy.decisions] == ["build: plan", "build: fix"]
    assert strategy.decisions[1].state["problems"][0].startswith("step 1 (purple cube on blue cube)")


def test_one_bad_answer_gets_a_second_try() -> None:
    table = Table()
    bad = plan(on("green cube", "blue cube"), on("blue cube", "red cube"))
    planner = FakePlanner([bad, plan(*TOWER)])
    strategy, steps = build(table, "placed", "placed", planner=planner)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.bad_answers == 1 and tower_stands(table)


def test_a_planner_that_fails_ends_the_run_and_a_timeout_gets_a_second_try() -> None:
    table = Table()
    strategy, steps = build(table, planner=FakePlanner([LoggedOut("codex is not logged in")]))
    ran = run(strategy, table)
    assert strategy.status.outcome == PLANNER_FAILED and not ran.action.success
    assert ran.action.note == "planista nie odpowiedzial: codex is not logged in"

    planner = FakePlanner([PlannerTimeout("codex gave no answer in 90 s"), plan(*TOWER)])
    strategy, steps = build(table, "placed", "placed", planner=planner)
    ran = run(strategy, table)
    assert ran.action.success and strategy.status.bad_answers == 1
    assert strategy.decisions[0].answer == "PlannerTimeout: codex gave no answer in 90 s"


# --------------------------------------------------------------------------- the tick and the end


def test_the_model_call_never_blocks_the_tick() -> None:
    # A cold reach cache, as in a new process: start solves it, not a tick.
    arm_reach.cache_clear()
    table = Table()
    planner = FakePlanner([Scripted(answer=plan(*TOWER), delay_s=0.3)])
    steps = Steps(table, [Plays(), Plays()])
    strategy = BuildStrategy(BuildSettings(task="tower", judge="none", wrist_camera=""), planner=planner,
                             step_factory=steps)
    strategy.start(RUN, Observation(t=0.0, step=0, joints=HOME, command={}, objects=table.objects()))
    slowest, k, t0 = 0.0, 0, time.monotonic()
    while strategy.status.phase in ("wait", "plan") and time.monotonic() - t0 < 5.0:
        k += 1
        tick = time.monotonic()
        action = strategy.act(Observation(t=k / 20, step=k, joints=HOME, command={}, objects=table.objects()))
        slowest = max(slowest, time.monotonic() - tick)
        assert not action.done
        time.sleep(0.01)
    assert strategy.status.phase == "step" and slowest < 0.1 and k > 10


def test_stopping_during_a_step_stops_the_step_once_and_the_pending_call() -> None:
    table = Table()
    strategy, steps = build(table, Plays("placed", ticks=10_000))
    strategy.start(RUN, Observation(t=0.0, step=0, joints=HOME, command={}, objects=table.objects()))
    for k in range(1, 10):
        strategy.act(Observation(t=k / 20, step=k, joints=HOME, command={}, objects=table.objects()))
    strategy.stop("zatrzymana")
    assert steps.made[0].stops == ["zatrzymana"]
    assert strategy.status.outcome == STEP_STOPPED and strategy.status.stopped_because == "zatrzymana"


def test_a_step_that_runs_too_long_is_stopped_and_ends_the_build() -> None:
    table = Table()
    strategy, steps = build(table, Plays("placed", ticks=10_000), step_time_limit_s=2.0)

    ran = run(strategy, table)

    assert strategy.status.outcome == STEP_STOPPED and not ran.action.success
    assert steps.made[0].stops == ["limit czasu kroku (2 s)"]


def test_a_step_that_raises_stops_it_and_raises_for_the_twin() -> None:
    table = Table()
    strategy, steps = build(table, Plays("placed", raises=True))

    with pytest.raises(RuntimeError, match="ramie nie nadaza"):
        run(strategy, table)
    strategy.stop("RuntimeError: ramie nie nadaza za celem")

    assert steps.made[0].stops == ["RuntimeError: ramie nie nadaza za celem"]
    assert strategy.status.outcome == STEP_STOPPED
    assert strategy.status.stopped_because == "ramie nie nadaza za celem"


def test_before_a_step_the_arm_turns_back_to_its_start_pose() -> None:
    """A step ends with the wrist turned: the next one starts only once the arm is back where the run
    started, moved at half of each joint's top speed."""
    table = Table()
    turned = {"wrist_roll": HOME["wrist_roll"] + 60.0, "shoulder_pan": HOME["shoulder_pan"] - 20.0}
    strategy, steps = build(table, Plays("placed", ends_at=turned), "placed")
    between: list[dict[str, float]] = []
    strategy.start(RUN, Observation(t=0.0, step=0, joints=HOME, command={}, objects=table.objects()))
    for k in range(1, 2000):
        action = strategy.act(Observation(t=k / strategy.hz, step=k, joints=HOME, command={},
                                          objects=table.objects()))
        if len(steps.made) == 1 and steps.made[0].stops:
            between.append(dict(action.target))
        if action.done:
            break

    assert action.success and len(steps.made) == 2
    # The step starts on the tick the arm is back, so the last target seen here is one tick away.
    rolls = [target["wrist_roll"] - HOME["wrist_roll"] for target in between]
    most = 0.5 * SPEC.max_vel["wrist_roll"] / strategy.hz
    assert rolls[0] == pytest.approx(60.0) and 0.0 < rolls[-1] <= most + 1e-9
    assert all(0.0 <= a - b <= most + 1e-9 for a, b in zip(rolls, rolls[1:], strict=False))
    assert len(rolls) >= 60.0 / most


def test_a_step_that_ends_with_the_hand_low_leaves_it_there_for_the_next() -> None:
    table = Table()
    low = {"shoulder_lift": HOME["shoulder_lift"] + 40.0, "elbow_flex": HOME["elbow_flex"] - 30.0}
    strategy, steps = build(table, Plays("pick_failed", ends_at=low), "placed", "placed")
    strategy.start(RUN, Observation(t=0.0, step=0, joints=HOME, command={}, objects=table.objects()))
    joints = dict(HOME)
    for k in range(1, 2000):
        if steps.made and steps.made[0].stops:
            joints = {**HOME, **low}
        action = strategy.act(Observation(t=k / strategy.hz, step=k, joints=joints, command={},
                                          objects=table.objects()))
        assert "wraca do pozycji startowej" not in action.note
        if len(steps.made) == 2:
            break
    assert steps.made[1].obs[0].joints == {**HOME, **low}


def test_the_steps_decisions_are_kept() -> None:
    table = Table()
    jev = Decision(question="close", answer="{}")
    strategy, steps = build(table, Plays("placed", decisions=(jev,)), "placed")

    run(strategy, table)

    assert strategy.decisions == [jev]
    if not ACTION_DECISIONS:
        assert strategy.drain_decisions() == (jev,)


def test_blocks_not_seen_at_the_start_end_a_fixed_plan() -> None:
    table = Table()
    table.hidden.add("red cube")
    strategy, steps = build(table, wait_for_block_s=1.0)

    ran = run(strategy, table)

    assert strategy.status.outcome == STUCK and ran.action.note == "nie widze blokow planu: red cube"


# --------------------------------------------------------------------------- settings and relations


def test_settings_need_a_task_or_a_plan_and_a_valid_plan() -> None:
    with pytest.raises(ValueError, match="give task"):
        BuildStrategy(BuildSettings(judge="none"))
    with pytest.raises(ValueError, match="plan_json is not JSON"):
        BuildStrategy(BuildSettings(plan_json="[{", judge="none"))
    with pytest.raises(ValueError, match=r"steps\[0\].place.relation must be one of"):
        BuildStrategy(BuildSettings(plan_json='[{"pick": "red cube", "place": {"relation": "over"}}]'))
    with pytest.raises(ValueError, match="max_steps is 1"):
        BuildStrategy(BuildSettings(plan_json=TOWER_JSON, max_steps=1))
    with pytest.raises(ValueError, match="planner: cli must be one of"):
        BuildStrategy(BuildSettings(task="x", planner_cli="gemini"))
    with pytest.raises(ValueError, match="pick_place: place_speed_mps must be positive"):
        BuildStrategy(BuildSettings(plan_json=TOWER_JSON, place_speed_mps=0.0))
    assert fixed_plan(TOWER_JSON).steps == fixed_plan(f'{{"steps": {TOWER_JSON}}}').steps
    assert BuildStrategy(BuildSettings(plan_json=TOWER_JSON)).time_limit_s == 600.0


def test_the_planner_settings_reach_the_cli() -> None:
    strategy = BuildStrategy(BuildSettings(task="x", planner_cli="claude", planner_model="sonnet",
                                           planner_effort="", planner_timeout_s=30.0))
    cli = strategy.planner.settings  # type: ignore[union-attr]
    assert (cli.cli, cli.model, cli.effort, cli.timeout_s) == ("claude", "sonnet", "", 30.0)


def test_a_table_spot_step_goes_to_pick_place_as_a_pose_on_the_table() -> None:
    table = Table()
    spot = {"pick": "red cube", "place": {"relation": "at", "targets": [], "side": None, "gap_m": None,
                                          "x_m": 0.22, "y_m": -0.04, "yaw_deg": 30.0}, "why": ""}
    planner = FakePlanner([plan(spot)])
    strategy, steps = build(table, "placed", planner=planner)

    ran = run(strategy, table)

    assert ran.action.success
    s = steps.made[0].settings
    assert s.place_on == "" and (s.place_x_m, s.place_y_m) == pytest.approx((0.22, -0.04))
    assert s.place_z_m == 0.0 and s.place_yaw_deg == pytest.approx(30.0) and s.supports == ""


def test_a_fixed_plan_sets_a_cube_beside_another_and_lays_a_cuboid_across_both() -> None:
    """next_to and across run as exact poses: the cube on the table beside blue, turned like it, and the
    cuboid resting on both cubes along the line between them; each step's check passes."""
    table = Table({"blue cube": (0.2, -0.03), "green cube": (0.18, 0.1), "yellow cuboid": (0.17, -0.1)})
    table.at["yellow cuboid"][2] = BLOCKS["cuboid"].half_size[2]
    table.yaw["blue cube"] = math.radians(10.0)
    bridge = ('[{"pick": "green cube", "place": {"relation": "next_to", "targets": ["blue cube"],'
              ' "side": "left", "gap_m": 0.015}},'
              ' {"pick": "Yellow Cuboid",'
              '  "place": {"relation": "across", "targets": ["Blue Cube", "green cube"]}}]')
    placed = Plays("placed", lambda t, s: t.at_pose(s.label, s))
    strategy, steps = build(table, placed, placed, plan_json=bridge)

    ran = run(strategy, table)

    assert ran.action.success and strategy.status.outcome == BUILT, strategy.status
    beside, across = (made.settings for made in steps.made)
    assert beside.place_on == "" and beside.supports == "" and beside.place_z_m == 0.0
    assert beside.place_yaw_deg == pytest.approx(10.0)
    assert math.dist((beside.place_x_m, beside.place_y_m), (0.2, -0.03)) == pytest.approx(2 * HALF + 0.015)
    assert across.label == "yellow cuboid" and across.block == "cuboid" and across.place_on == ""
    assert across.supports == "blue cube, green cube" and across.place_z_m == pytest.approx(2 * HALF)
    mid = (table.at["blue cube"][:2] + table.at["green cube"][:2]) / 2
    assert (across.place_x_m, across.place_y_m) == pytest.approx(tuple(mid))
    assert "krok 1/2: green cube -> obok blue cube (left) (placed)" in ran.notes
    assert "krok 2/2: yellow cuboid -> blue cube + green cube (placed)" in ran.notes
    assert [r.ok for r in strategy.status.records] == [True, True] and strategy.status.replans == 0


def test_a_bridge_goes_to_pick_place_as_an_exact_pose() -> None:
    table = Table({"blue cube": (0.2, -0.02), "green cube": (0.2, 0.024), "yellow cuboid": (0.17, -0.1)})
    table.at["yellow cuboid"][2] = BLOCKS["cuboid"].half_size[2]
    across = {"pick": "yellow cuboid", "place": {"relation": "across", "targets": ["blue cube", "green cube"],
                                                 "side": None, "gap_m": None, "x_m": None, "y_m": None,
                                                 "yaw_deg": None}, "why": ""}
    planner = FakePlanner([plan(across)])
    strategy, steps = build(table, Plays("placed", lambda t, s: None), planner=planner)

    run(strategy, table)

    s = steps.made[0].settings
    assert s.place_on == "" and s.supports == "blue cube, green cube"
    assert s.place_z_m == pytest.approx(2 * HALF)
    assert s.place_yaw_deg == pytest.approx(90.0, abs=1.0)
