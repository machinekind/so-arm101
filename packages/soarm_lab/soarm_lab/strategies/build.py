"""Build with blocks from a sentence or a fixed plan: each step is one `pick_place` run inside this run.

`task` is the sentence. The planner (`soarm_lab.planner`) sends the claude or codex CLI the world state
from `obs.objects` and gets back steps in `PLAN_SCHEMA`; the call runs on a thread of its own, so the
tick never waits. A plan is walked through the world its steps meet (`check_steps`) before it runs. A bad
answer goes back once with its problems listed; a second bad one ends the run. `plan_json` is a fixed plan
in the same JSON, or only its list of steps with the unused fields left out: no model is asked.

Each step runs a `PickPlaceStrategy` in-process, built from this run's settings (judge, wrist camera,
speeds and the rest of `PickPlaceSettings`) with the step's block and target. An "on" step targets the
block's label, so pick_place follows that block's pose. Other relations go through `resolve` to a pose
that pick_place places at (`place_x_m`, `place_y_m`, `place_z_m`, `place_yaw_deg`); an "across" step's
targets are its `supports`. The step's block and the time it may take (`step_time_limit_s`) are its
own; the step's clock starts at zero. A step ends with the hand back where it started but the wrist
turned, so before the next one the arm turns back, joint by joint, to the pose it started the run in:
each step starts as a fresh pick_place run. A step that ended with the hand elsewhere, low over the
blocks, leaves it there; pick_place rises first.

After a step the run waits `settle_s`, then up to `wait_for_block_s` for poses captured since the step
ended, and checks with `holds` that the step and every earlier step still standing hold. A step's
outcome decides what follows:

- `placed` and the check passes: the next step;
- `not_found`, `target_not_seen`, `target_moved`: once the blocks are seen again, the same step once
  more, unless it already holds or its pick_place already picked the block again;
- `unreachable`, `dropped`, `fell_off`, `pick_failed` or a failed check: a replan from the world seen
  now. The model gets the steps so far and the failed one, so it can choose another place. A fixed plan
  runs again from its first step that no longer holds, with blocks in the way set aside on free table
  spots (`planner.steps.remaining`); it ends when the same step is unreachable twice in a row;
- `stopped`: the run ends.

A block that falls off its target or out of the jaw is picked and placed again inside the step, up to
`place_retries` times; the step ends `fell_off`, `target_moved` or `dropped` only after those, and the
stop reason the model gets counts them. So the build does not run such a step once more as it stands:
it replans.

At most `max_replans` replans. The run succeeds when the last step's check passes, or when the model
answers that the world already fulfils the task. A model's question ends the run with the question.

Every model call is recorded as a `Decision`, as are the Jev calls of each step. The pinned api has no
`Action.decisions`, so they stay in `decisions` and `drain_decisions`; with a newer api each tick's
`Action.decisions` carries the new ones.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from machinekind_twin_api.strategy import Action, ObjectPose, Observation, RunInfo, Strategy

from ..kinematics import RobotKinematics
from ..planner.cli import (
    CLIS,
    EFFORTS,
    BadAnswer,
    CliPlanner,
    CliSettings,
    Planner,
    PlannerOutcome,
    PlannerRunner,
    PlannerTimeout,
)
from ..planner.prompt import PROMPT_VERSION, StepRecord, build_prompt, system_text
from ..planner.relations import Limits, RelationError, holds, resolve
from ..planner.schema import (
    ACROSS,
    NEXT_TO,
    ON,
    PLACE_SCHEMA,
    PLAN_SCHEMA,
    Plan,
    PlanError,
    Step,
    parse_plan,
)
from ..planner.steps import check_steps, describe, first_unmet, remaining
from ..planner.world import IN_GRIPPER_SOURCE, World, label_key, world_state
from ..robot import load_robot
from .grasp import grip_strength
from .jev import JevJudge
from .judges import Decision, Judge
from .pick_place import (
    DROPPED,
    FELL_OFF,
    JEV,
    NOT_FOUND,
    PICK_FAILED,
    PLACED,
    STOPPED,
    TARGET_MOVED,
    TARGET_NOT_SEEN,
    UNREACHABLE,
    NoJudge,
    PickPlaceSettings,
    PickPlaceStrategy,
)
from .touch import TOUCH_SOURCES

logger = logging.getLogger(__name__)

#: Phases of the run.
WAIT = "wait"
PLAN = "plan"
STEP = "step"
SETTLE = "settle"
RETURN = "return"
DONE = "done"

#: How a build run ended (`BuildStatus.outcome`).
BUILT = "built"
ALREADY_DONE = "already_done"
QUESTION = "question"
BAD_PLAN = "bad_plan"
PLANNER_FAILED = "planner_failed"
BUDGET_SPENT = "budget_spent"
STUCK = "stuck"
STEP_STOPPED = "stopped"
OUTCOMES = (BUILT, ALREADY_DONE, QUESTION, BAD_PLAN, PLANNER_FAILED, BUDGET_SPENT, STUCK, STEP_STOPPED)

#: Step outcomes that wait for the blocks and run the same step once more.
RETRY = (NOT_FOUND, TARGET_NOT_SEEN, TARGET_MOVED)
#: Step outcomes that replan from the world seen now.
REPLAN = (UNREACHABLE, DROPPED, FELL_OFF, PICK_FAILED)

#: Whether the api's `Action` carries decisions.
ACTION_DECISIONS = "decisions" in {f.name for f in dataclasses.fields(Action)}

#: Turning back to the start pose: the share of each joint's top speed, how close the joints come [deg]
#: and the longest wait for them [s]. Only a hand this close to where it started turns back [m].
RETURN_SPEED_SHARE = 0.5
RETURN_TOL_DEG = 1.0
RETURN_WAIT_S = 10.0
RETURN_NEAR_M = 0.03

#: Model calls ended by these errors get the one second try a bad plan gets.
_RETRIED_ERRORS = (BadAnswer, PlannerTimeout)


@dataclass
class BuildSettings(PickPlaceSettings):
    """pick_place's settings, used by every step, and the build's own.

    Each step sets `label`, `block`, `place_on`, `place_on_block` and the place pose itself; the values
    given here for them are not used.
    """

    #: The whole build, model waits included [s]. Zero disables it.
    time_limit_s: float = 600.0
    #: What to build, in a sentence for the model. Ignored with `plan_json`.
    task: str = ""
    #: A fixed plan as JSON: no model is asked. The plan object or its list of steps; unused fields may be
    #: left out.
    plan_json: str = ""
    #: The CLI that plans: codex or claude, logged in with a subscription.
    planner_cli: str = field(default="codex", metadata={"choices": CLIS})
    #: The model as the CLI names it; empty for the CLI's default.
    planner_model: str = "gpt-6-astra"
    #: The model's reasoning effort; empty for the CLI's default.
    planner_effort: str = field(default="low", metadata={"choices": EFFORTS})
    #: The CLI binary; empty finds `planner_cli` on PATH.
    planner_executable: str = ""
    #: How long one model call may take [s].
    planner_timeout_s: float = 90.0
    #: Most steps in one plan.
    max_steps: int = 8
    #: Most replans after a failed step.
    max_replans: int = 3
    #: Most blocks in one stack, the one on the table counting.
    max_stack: int = 3
    #: Wait after each step before checking it [s].
    settle_s: float = 1.0
    #: How long to wait for the blocks to be seen at the start, after a step and before a retry [s].
    wait_for_block_s: float = 15.0
    #: Time limit of one step [s].
    step_time_limit_s: float = 240.0


@dataclass
class BuildStatus:
    """What the build knows about its run, for logs, tests and the panel."""

    #: One of WAIT, PLAN, STEP, SETTLE, DONE.
    phase: str = WAIT
    #: The plan being run and the index of its current step.
    plan: tuple[Step, ...] = ()
    index: int = 0
    #: Every step run, skipped as already done or failed, in order.
    records: list[StepRecord] = field(default_factory=list)
    #: Steps verified in place.
    done: int = 0
    #: Model calls, bad answers, replans and retries of a step.
    asked: int = 0
    bad_answers: int = 0
    replans: int = 0
    retries: int = 0
    #: Picks of a fallen block inside the steps' pick_place runs (`PlaceStatus.retries`), in all.
    place_retries: int = 0
    #: The outcome of the last step run (`pick_place.OUTCOMES`) and its stop reason.
    step_outcome: str = ""
    step_reason: str = ""
    #: One of `OUTCOMES` once the run has ended, '' before.
    outcome: str = ""
    #: The model's question, when it asked one.
    question: str = ""
    success: bool = False
    stopped_because: str = ""


class StepRun(Protocol):
    """What the build needs of a step's strategy: `PickPlaceStrategy`, or a test double."""

    def start(self, run: RunInfo, obs: Observation) -> None: ...

    def act(self, obs: Observation) -> Action: ...

    def stop(self, reason: str) -> None: ...

    def drain_decisions(self) -> tuple[Decision, ...]: ...


StepFactory = Callable[[PickPlaceSettings], StepRun]


def fixed_plan(text: str) -> Plan:
    """The plan in `plan_json`: a plan object or a list of steps; unused fields may be left out.

    Raises `PlanError` when it is no valid plan.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PlanError(f"plan_json is not JSON: {exc}") from exc
    if isinstance(data, list):
        data = {"steps": data}
    if not isinstance(data, dict):
        raise PlanError("plan_json must be a plan object or a list of steps")
    data = {"done": False, "question": "", **data}
    steps = data.get("steps")
    if isinstance(steps, list):
        blank = {key: [] if key == "targets" else None for key in PLACE_SCHEMA["required"]}
        data["steps"] = [{"why": "", **s, "place": {**blank, **s["place"]}}
                         if isinstance(s, dict) and isinstance(s.get("place"), dict) else s for s in steps]
    plan = parse_plan(data)
    if plan.question or plan.done:
        raise PlanError("plan_json must give steps, not done or a question")
    if not plan.steps:
        raise PlanError("plan_json has no steps")
    return plan


class BuildStrategy(Strategy[BuildSettings]):
    """Builds with blocks from a sentence or a fixed plan, one pick_place run per step, and replans."""

    settings = BuildSettings

    def __init__(self, settings: BuildSettings | None = None, *, planner: Planner | None = None,
                 judge: Judge | None = None, step_factory: StepFactory | None = None, threaded: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 blob: Callable[[np.ndarray], float | None] | None = None):
        config = settings if settings is not None else BuildSettings()
        self.config = config
        for name in ("hz", "settle_s", "wait_for_block_s", "time_limit_s"):
            value = getattr(config, name)
            if not (math.isfinite(value) and value >= 0) or (name == "hz" and value == 0):
                raise ValueError(f"build: {name} must be {'positive' if name == 'hz' else 'zero or more'}, "
                                 f"is {value}")
        for name in ("step_time_limit_s", "planner_timeout_s"):
            if not getattr(config, name) > 0:
                raise ValueError(f"build: {name} must be positive, is {getattr(config, name)}")
        for name, least in (("max_steps", 1), ("max_replans", 0), ("max_stack", 1)):
            if getattr(config, name) < least:
                raise ValueError(f"build: {name} must be at least {least}, is {getattr(config, name)}")
        #: Every step's grip; the twin's runner reads it once, after `start`.
        self.grip_strength = grip_strength(config.grip_strength, "build")
        #: The fixed plan, or None when the model plans.
        self.goal: Plan | None = None
        if config.plan_json.strip():
            try:
                self.goal = fixed_plan(config.plan_json)
            except PlanError as exc:
                raise ValueError(f"build: {exc}") from exc
            if len(self.goal.steps) > config.max_steps:
                raise ValueError(f"build: plan_json has {len(self.goal.steps)} steps, max_steps is "
                                 f"{config.max_steps}")
        elif not config.task.strip():
            raise ValueError("build: give task (a sentence) or plan_json (a fixed plan)")
        self.planner: Planner | None = None
        if self.goal is None:
            self.planner = planner if planner is not None else CliPlanner(CliSettings(
                cli=config.planner_cli, model=config.planner_model.strip(), effort=config.planner_effort,
                executable=config.planner_executable.strip(), timeout_s=config.planner_timeout_s))
        self.runner = PlannerRunner(self.planner, threaded=threaded) if self.planner is not None else None
        self.judged = config.judge == JEV
        self.judge: Judge = judge if judge is not None else (
            JevJudge(config.model.strip()) if self.judged else NoJudge())
        self.threaded, self.clock, self.blob = threaded, clock, blob
        self.step_factory: StepFactory = step_factory or self._pick_place
        # Fails here on settings no step can run with.
        PickPlaceStrategy(self._step_settings(label="", place_on=""), judge=NoJudge(), threaded=False)
        self.limits = Limits(max_stack=config.max_stack)
        self.hz = float(config.hz)
        self.time_limit_s = float(config.time_limit_s) if config.time_limit_s > 0 else None
        self.status = BuildStatus()
        #: Every model and judge decision of the run, and those not yet handed out.
        self.decisions: list[Decision] = []
        self._outbox: list[Decision] = []
        self._run: RunInfo | None = None
        self._kin: RobotKinematics | None = None
        self._home_tcp = np.zeros(3)
        self._simulated = False
        self._target: dict[str, float] = {}
        self._child: StepRun | None = None
        self._child_t0 = 0.0
        self._child_step0 = 0
        self._t = 0.0
        self._phase_t0 = 0.0
        #: When the last step ended [s, `obs.t`].
        self._ended_t = 0.0
        self._retried = False
        #: Picks of a fallen block inside the last step's pick_place.
        self._step_retries = 0
        #: The last step that ended unreachable, until a step is verified.
        self._unreachable: Step | None = None
        #: Steps verified in place that later steps have not moved: the structure built so far.
        self._standing: list[Step] = []
        #: The pending model call: its kind, prompt and the problems that sent it back.
        self._call: tuple[str, str, list[str]] = ("", "", [])
        self._second_try = False
        self._failed: StepRecord | None = None

    # ------------------------------------------------------------------ settings of a step
    def _pick_place(self, settings: PickPlaceSettings) -> PickPlaceStrategy:
        return PickPlaceStrategy(settings, judge=self.judge, threaded=self.threaded, clock=self.clock,
                                 blob=self.blob)

    def _step_settings(self, **values: Any) -> PickPlaceSettings:
        """pick_place settings from this run's, with `values` for the step."""
        base = {f.name: getattr(self.config, f.name) for f in dataclasses.fields(PickPlaceSettings)}
        return PickPlaceSettings(**{**base, "time_limit_s": self.config.step_time_limit_s, **values})

    def step_settings(self, step: Step, world: World) -> PickPlaceSettings:
        """pick_place settings for `step` in `world`; raises `RelationError` when it cannot run there."""
        pose = resolve(step, world, limits=self.limits)
        pick = world.get(step.pick)
        assert pick is not None and pick.kind is not None
        values: dict[str, Any] = {"label": pick.label, "block": pick.kind, "place_on": ""}
        if step.place.relation == ON:
            target = world.get(step.place.targets[0])
            assert target is not None and target.kind is not None
            values.update(place_on=target.label, place_on_block=target.kind)
        else:
            # A bridge rests on its two targets; the other relations on the table.
            supports = step.place.targets if step.place.relation == ACROSS else ()
            values.update(place_x_m=pose.x_m, place_y_m=pose.y_m, place_z_m=pose.z_m,
                          place_yaw_deg=math.degrees(pose.yaw_rad),
                          supports=", ".join(label_key(label) for label in supports))
        return self._step_settings(**values)

    # ------------------------------------------------------------------ the run
    def start(self, run: RunInfo, obs: Observation) -> None:
        """Check the judge and solve the build area's reach, then hold the arm where it is until the blocks
        are seen."""
        if self.judged:
            self.judge.check()
        self._run = run
        self._simulated = bool(run.simulated)
        self._kin = RobotKinematics(load_robot(run.folder))
        # The first build area solves the arm's reach (cached per process), so no tick waits for it.
        _ = self._world_of({}).build_area
        self._target = {name: float(obs.command.get(name, obs.joints[name])) for name in run.joints
                        if name in obs.joints}
        #: The arm joints' start pose, and how far each may move in one tick on the way back to it.
        self._home = {name: v for name, v in self._target.items() if name != run.gripper}
        self._return_step = {name: RETURN_SPEED_SHARE * float(run.max_vel.get(name, math.inf)) / self.hz
                             for name in self._home}
        self._home_tcp = self._tcp(obs.joints)
        self.status = BuildStatus()
        self.decisions, self._outbox = [], []
        self._child = None
        self._retried = self._second_try = False
        self._step_retries = 0
        self._failed = None
        self._unreachable = None
        self._standing = []
        self._t = float(obs.t)
        self._enter(WAIT)

    def act(self, obs: Observation) -> Action:
        if self._run is None:
            raise RuntimeError("build: act() przed start()")
        self._t = float(obs.t)
        handlers: dict[str, Callable[[Observation], Action | None]] = {
            WAIT: self._wait, PLAN: self._planning, STEP: self._stepping, SETTLE: self._settle,
            RETURN: self._returning,
            DONE: lambda _: self._hold(self.status.stopped_because)}
        for _ in range(16):
            action = handlers[self.status.phase](obs)
            if action is not None:
                return self._with_decisions(action)
        raise RuntimeError(f"build: no answer in phase {self.status.phase}")

    def stop(self, reason: str) -> None:
        """The twin ended the run: stop a running step and give up on a pending model call."""
        st = self.status
        if self._child is not None:
            child, self._child = self._child, None
            child.stop(reason)
            self._collect(child, None)
        if self.runner is not None:
            self.runner.cancel()
        st.phase = DONE
        if st.stopped_because and (reason in ("zadanie wykonane", "strategia skonczyla")
                                   or reason.endswith(st.stopped_because)):
            return
        st.outcome = st.outcome or STEP_STOPPED
        st.stopped_because = reason or "zatrzymana"

    @property
    def current_step(self) -> StepRun | None:
        """The running step's strategy, or None between steps."""
        return self._child

    def drain_decisions(self) -> tuple[Decision, ...]:
        """The decisions since the last drain, each once."""
        out, self._outbox = tuple(self._outbox), []
        return out

    # ------------------------------------------------------------------ waiting for the blocks
    def _wait(self, obs: Observation) -> Action | None:
        """Hold until the blocks are seen: a fixed plan's every block, or any for the model."""
        seen = {label_key(str(k)) for k, pose in obs.objects.items() if self._usable(pose)}
        if self.goal is not None:
            needed = {label_key(label) for s in self.goal.steps for label in (s.pick, *s.place.targets)}
            missing = sorted(needed - seen)
        else:
            missing = [] if seen else ["bloki"]
        if not missing:
            return self._plan(obs)
        if self._t - self._phase_t0 < self.config.wait_for_block_s:
            return self._hold(f"czekam na bloki w kamerach: {', '.join(missing)}")
        if self.goal is not None:
            return self._finish(STUCK, f"nie widze blokow planu: {', '.join(missing)}")
        return self._plan(obs)

    # ------------------------------------------------------------------ planning
    def _plan(self, obs: Observation) -> Action | None:
        """Plan from the world seen now: the fixed plan's remaining steps, or ask the model."""
        if self.goal is not None:
            try:
                steps = remaining(self.goal.steps, self._poses(obs), self._world_of, limits=self.limits)
            except RelationError as exc:
                return self._finish(STUCK, f"plan nie da sie dokonczyc: {exc}")
            if not steps:
                return self._finish(BUILT if self.status.done else ALREADY_DONE,
                                    "budowla stoi zgodnie z planem", success=True)
            if len(steps) > self.config.max_steps:
                return self._finish(STUCK, f"plan potrzebuje {len(steps)} krokow, max_steps to "
                                           f"{self.config.max_steps}")
            return self._run_plan(tuple(steps), obs)
        failed = self._failed
        history = [r for r in self.status.records if r is not failed]
        message = build_prompt(self.config.task, self._world(obs), history, failed)
        self._ask("replan" if failed is not None else "plan", message, [])
        return None

    def _ask(self, kind: str, message: str, problems: list[str]) -> None:
        """Start a model call off the tick thread."""
        assert self.runner is not None
        prompt = f"{system_text(self.limits)}\n{message}"
        if problems:
            prompt += ("\nYour previous answer cannot run:\n" + "\n".join(f"- {p}" for p in problems)
                       + "\nAnswer again with a plan that avoids these problems.\n")
        self._call = (kind, prompt, problems)
        self.status.asked += 1
        self.runner.start(prompt, PLAN_SCHEMA)
        self._enter(PLAN)

    def _planning(self, obs: Observation) -> Action | None:
        """Wait for the model's answer, then check it and run it, or ask once more."""
        assert self.runner is not None
        outcome = self.runner.poll()
        if outcome is None:
            waited = self._t - self._phase_t0
            return self._hold(f"pytam model o plan ({waited:.0f} s)")
        self._record(outcome)
        kind, _, _ = self._call
        if outcome.error is not None:
            if isinstance(outcome.error, _RETRIED_ERRORS) and not self._second_try:
                return self._again(obs, [str(outcome.error)])
            return self._finish(PLANNER_FAILED, f"planista nie odpowiedzial: {outcome.error}")
        assert outcome.answer is not None
        try:
            plan = parse_plan(outcome.answer)
        except PlanError as exc:
            return self._again(obs, [str(exc)])
        if plan.question:
            self.status.question = plan.question
            return self._finish(QUESTION, f"pytanie modelu: {plan.question}")
        if plan.done:
            return self._finish(BUILT if self.status.done else ALREADY_DONE, "model: zadanie juz wykonane",
                                success=True)
        if not plan.steps:
            return self._again(obs, ["the answer has no steps, is not done and asks nothing"])
        problems = check_steps(plan.steps, self._poses(obs), self._world_of, limits=self.limits,
                               max_steps=self.config.max_steps)
        if problems:
            return self._again(obs, problems)
        logger.info("Plan (%s): %s", kind, "; ".join(describe(s) for s in plan.steps))
        return self._run_plan(plan.steps, obs)

    def _again(self, obs: Observation, problems: list[str]) -> Action | None:
        """Send a bad answer back once with its problems; a second one ends the run."""
        self.status.bad_answers += 1
        if self._second_try:
            return self._finish(BAD_PLAN, f"plan modelu znow bledny: {problems[0]}")
        self._second_try = True
        failed = self._failed
        history = [r for r in self.status.records if r is not failed]
        logger.info("Plan odrzucony: %s", "; ".join(problems))
        self._ask("fix", build_prompt(self.config.task, self._world(obs), history, failed), problems)
        return None

    def _record(self, outcome: PlannerOutcome) -> None:
        kind, prompt, problems = self._call
        if outcome.answer is not None:
            answer = json.dumps(outcome.answer, ensure_ascii=False)
        else:
            answer = f"{type(outcome.error).__name__}: {outcome.error}"
        state: dict[str, Any] = {"call": kind, "prompt_version": PROMPT_VERSION, "prompt": prompt,
                                 "task": self.config.task, "cli": self.config.planner_cli,
                                 "model": self.config.planner_model}
        if problems:
            state["problems"] = list(problems)
        self._add(Decision(question=f"build: {kind}", answer=answer, state=state,
                           response_s=outcome.response_s))

    def _run_plan(self, steps: Sequence[Step], obs: Observation) -> Action | None:
        """Run a new plan; the structure built so far is what of it holds now."""
        st = self.status
        st.plan, st.index = tuple(steps), 0
        world = self._world(obs)
        self._standing = [s for s in self._standing if holds(s, world, self.limits)[0]]
        self._second_try = False
        self._failed = None
        self._retried = False
        return self._next_step(obs)

    # ------------------------------------------------------------------ steps
    def _next_step(self, obs: Observation) -> Action | None:
        """Start the plan's current step, skipping steps that already hold; replan when it cannot run."""
        st = self.status
        world = self._world(obs)
        broken = self._broken(world)
        if broken is not None:
            return self._replan(obs, broken)
        while st.index < len(st.plan):
            step = st.plan[st.index]
            ok, why = holds(step, world, self.limits)
            if not ok:
                break
            self._verified(step, f"already holds: {why}")
        if st.index >= len(st.plan):
            if self.goal is not None:
                unmet = first_unmet(self.goal.steps, world, self.limits)
                if unmet is not None:
                    step = self.goal.steps[unmet]
                    why = holds(step, world, self.limits)[1]
                    return self._replan(obs, StepRecord(step, False, f"no longer holds: {why}"))
            return self._finish(BUILT, f"zbudowane: {self._summary()}", success=True)
        step = st.plan[st.index]
        try:
            settings = self.step_settings(step, world)
        except RelationError as exc:
            return self._replan(obs, StepRecord(step, False, f"cannot run now: {exc}"))
        if self.status.phase != RETURN and self._away_from_home() and self._near_home(obs):
            self._enter(RETURN)
            return None
        child = self.step_factory(settings)
        assert self._run is not None
        child_obs = dataclasses.replace(obs, t=0.0, step=0)
        try:
            child.start(self._run, child_obs)
        except Exception as exc:  # noqa: BLE001 - a step that cannot start ends the build, not the tick
            logger.exception("build: step start failed")
            return self._finish(STEP_STOPPED, f"krok {self._number()} nie wystartowal: {exc}")
        self._child, self._child_t0, self._child_step0 = child, self._t, int(obs.step)
        logger.info("Krok %s: %s", self._number(), describe(step))
        self._enter(STEP)
        return None

    def _away_from_home(self) -> bool:
        return any(abs(self._target.get(name, v) - v) > RETURN_TOL_DEG for name, v in self._home.items())

    def _tcp(self, joints: Mapping[str, float]) -> np.ndarray:
        assert self._kin is not None
        return self._kin.tcp({**self._kin.spec.home, **joints})[:3, 3].copy()

    def _near_home(self, obs: Observation) -> bool:
        """Whether the measured hand is near where it started, so it can turn back without sweeping
        over the blocks."""
        return float(np.linalg.norm(self._tcp(obs.joints) - self._home_tcp)) <= RETURN_NEAR_M

    def _returning(self, obs: Observation) -> Action | None:
        """Turn the arm back to its start pose, then start the step."""
        for name, home in self._home.items():
            now = self._target.get(name, home)
            step = self._return_step[name]
            self._target[name] = now + max(-step, min(step, home - now))
        arrived = all(abs(float(obs.joints.get(name, home)) - home) <= RETURN_TOL_DEG
                      for name, home in self._home.items())
        on_target = not self._away_from_home() and all(
            abs(self._target[name] - home) < 1e-9 for name, home in self._home.items())
        if on_target and (arrived or self._t - self._phase_t0 >= RETURN_WAIT_S):
            return self._next_step(obs)
        return self._hold(f"krok {self._number()}: ramie wraca do pozycji startowej")

    def _stepping(self, obs: Observation) -> Action | None:
        """Tick the step's pick_place; read its outcome when it ends."""
        child = self._child
        assert child is not None
        st = self.status
        step = st.plan[st.index]
        if self._t - self._child_t0 > self.config.step_time_limit_s:
            reason = f"limit czasu kroku ({self.config.step_time_limit_s:.0f} s)"
            self._child = None
            child.stop(reason)
            self._collect(child, None)
            return self._step_ended(step, STOPPED, reason)
        child_obs = dataclasses.replace(obs, t=self._t - self._child_t0,
                                        step=int(obs.step) - self._child_step0)
        try:
            action = child.act(child_obs)
        except Exception as exc:
            self._child = None
            child.stop(f"{type(exc).__name__}: {exc}")
            self._collect(child, None)
            st.step_outcome, st.step_reason = STOPPED, str(exc)
            self._end(STEP_STOPPED, str(exc))
            raise
        self._collect(child, action)
        self._target.update({k: float(v) for k, v in action.target.items()})
        if not action.done:
            return Action(target=dict(self._target), error=action.error,
                          note=f"krok {self._number()}: {self._text(step)}: {action.note}")
        self._child = None
        child.stop("zadanie wykonane" if action.success else "strategia skonczyla")
        self._collect(child, None)
        place = getattr(child, "place", None)
        outcome = str(getattr(place, "outcome", "") or (PLACED if action.success else STOPPED))
        self._step_retries = int(getattr(place, "retries", 0) or 0)
        st.place_retries += self._step_retries
        reason = str(getattr(getattr(child, "status", None), "stopped_because", "") or action.note)
        return self._step_ended(step, outcome, reason)

    def _step_ended(self, step: Step, outcome: str, reason: str) -> Action | None:
        st = self.status
        st.step_outcome, st.step_reason = outcome, reason
        self._ended_t = self._t
        logger.info("Krok %s: %s (%s)", self._number(), outcome, reason)
        if outcome == STOPPED:
            return self._finish(STEP_STOPPED,
                                f"krok {self._number()}: {self._text(step)} ({outcome}): {reason}")
        self._enter(SETTLE)
        return self._hold(f"krok {self._number()}: {self._text(step)} ({outcome})")

    def _settle(self, obs: Observation) -> Action | None:
        """Wait for the blocks to come to rest and be seen again, then decide what follows the step."""
        st = self.status
        step, outcome = st.plan[st.index], st.step_outcome
        cfg = self.config
        waited = self._t - self._phase_t0
        line = f"krok {self._number()}: {self._text(step)} ({outcome})"
        if waited < cfg.settle_s:
            return self._hold(line)
        labels = {label_key(label) for label in (step.pick, *step.place.targets)}
        if outcome == PLACED:
            labels |= {label_key(label) for s in self._standing for label in (s.pick, *s.place.targets)}
        stale = sorted(label for label in labels if not self._fresh(obs, label, self._ended_t))
        if stale and waited < cfg.settle_s + cfg.wait_for_block_s:
            return self._hold(f"{line}: czekam na swieze polozenia ({', '.join(stale)})")
        world = self._world(obs)
        ok, why = holds(step, world, self.limits)
        if outcome == PLACED or (outcome in RETRY and ok):
            if not ok:
                return self._replan(obs, StepRecord(step, False, f"{outcome}, but {why}"))
            self._verified(step, f"{outcome}: {why}")
            broken = self._broken(world)
            if broken is not None:
                return self._replan(obs, broken)
            return self._next_step(obs)
        record = StepRecord(step, False, f"{outcome}: {st.step_reason}")
        if outcome in RETRY:
            # A step whose pick_place already picked the block again is replanned instead.
            if not self._retried and not stale and not self._step_retries:
                self._retried = True
                st.retries += 1
                logger.info("Krok %s: jeszcze raz po %s", self._number(), outcome)
                return self._next_step(obs)
            return self._replan(obs, record)
        if outcome in REPLAN:
            if outcome == UNREACHABLE:
                if self.goal is not None and self._unreachable == step:
                    st.records.append(record)
                    return self._finish(STUCK, f"{line} drugi raz: {st.step_reason}")
                self._unreachable = step
            return self._replan(obs, record)
        st.records.append(record)
        return self._finish(STEP_STOPPED, f"{line}: {st.step_reason}")

    def _replan(self, obs: Observation, failed: StepRecord) -> Action | None:
        """Plan again from the world seen now, within the replan budget."""
        st = self.status
        st.records.append(failed)
        if st.replans >= self.config.max_replans:
            return self._finish(BUDGET_SPENT, f"wyczerpany limit przeplanowan ({self.config.max_replans}): "
                                              f"{describe(failed.step)}: {failed.outcome}")
        st.replans += 1
        logger.info("Przeplanowanie %d: %s: %s", st.replans, describe(failed.step), failed.outcome)
        self._failed = failed
        self._second_try = False
        self._retried = False
        st.plan, st.index = (), 0
        return self._plan(obs)

    def _verified(self, step: Step, why: str) -> None:
        """Count the step as done; it stands, and the steps it moved a block of no longer do."""
        st = self.status
        st.records.append(StepRecord(step, True, why))
        st.done += 1
        st.index += 1
        self._retried = False
        self._unreachable = None
        moved = label_key(step.pick)
        self._standing = [s for s in self._standing
                          if moved not in {label_key(label) for label in (s.pick, *s.place.targets)}]
        self._standing.append(step)

    def _broken(self, world: World) -> StepRecord | None:
        """The first standing step that no longer holds in `world`, as a failed record, or None."""
        for step in self._standing:
            ok, why = holds(step, world, self.limits)
            if not ok:
                return StepRecord(step, False, f"no longer holds: {why}")
        return None

    # ------------------------------------------------------------------ the world
    def _world_of(self, poses: Mapping[str, ObjectPose]) -> World:
        return world_state(poses, levels=self.config.max_stack)

    def _poses(self, obs: Observation) -> dict[str, ObjectPose]:
        """The usable poses of `obs` and those of a block in the gripper."""
        return {str(k): v for k, v in obs.objects.items() if self._usable(v) or v.source == IN_GRIPPER_SOURCE}

    def _world(self, obs: Observation) -> World:
        return self._world_of(self._poses(obs))

    def _usable(self, pose: ObjectPose) -> bool:
        source = str(pose.source or "")
        if not source:
            return self._simulated
        position = np.asarray(pose.position, float)
        return (source in TOUCH_SOURCES and float(pose.age_s) <= self.config.max_pose_age_s
                and position.shape == (3,) and bool(np.all(np.isfinite(position))))

    def _fresh(self, obs: Observation, label: str, since: float) -> bool:
        """Whether `obs` has a usable pose of `label` captured at `since` or later."""
        for key, pose in obs.objects.items():
            if label_key(str(key)) == label and self._usable(pose):
                return float(obs.t) - float(pose.age_s) >= since - 1e-9
        return False

    # ------------------------------------------------------------------ decisions and the end
    def _add(self, decision: Decision) -> None:
        self.decisions.append(decision)
        self._outbox.append(decision)

    def _collect(self, child: StepRun, action: Action | None) -> None:
        """Keep the step's decisions: those its action carries and those it holds back."""
        for decision in (*(getattr(action, "decisions", ()) if action is not None else ()),
                         *child.drain_decisions()):
            self._add(decision)

    def _with_decisions(self, action: Action) -> Action:
        """The action, carrying the decisions since the previous tick when the api has the field."""
        if not ACTION_DECISIONS or not self._outbox:
            return action
        return dataclasses.replace(action, **{"decisions": (*getattr(action, "decisions", ()),
                                                              *self.drain_decisions())})

    def _finish(self, outcome: str, reason: str, success: bool = False) -> Action:
        self._end(outcome, reason, success)
        return Action(target=dict(self._target), done=True, success=success, note=reason)

    def _end(self, outcome: str, reason: str, success: bool = False) -> None:
        st = self.status
        st.phase, st.outcome, st.success, st.stopped_because = DONE, outcome, success, reason
        if self.runner is not None:
            self.runner.cancel()
        logger.info("Budowa zakonczona (%s): %s", outcome, reason)

    def _hold(self, note: str) -> Action:
        return Action(target=dict(self._target), note=note)

    def _enter(self, phase: str) -> None:
        self.status.phase = phase
        self._phase_t0 = self._t

    def _number(self) -> str:
        st = self.status
        return f"{st.done + 1}/{st.done + len(st.plan) - st.index}"

    def _summary(self) -> str:
        st = self.status
        return f"{st.done} krokow, przeplanowan: {st.replans}, powtorzen: {st.retries}"

    @staticmethod
    def _text(step: Step) -> str:
        """The step for the panel: "red cube -> blue cube"."""
        place = step.place
        if place.relation == ON:
            return f"{step.pick} -> {place.targets[0]}"
        if place.relation == ACROSS:
            return f"{step.pick} -> {place.targets[0]} + {place.targets[1]}"
        if place.relation == NEXT_TO:
            return f"{step.pick} -> obok {place.targets[0]} ({place.side})"
        return f"{step.pick} -> stol ({place.x_m or 0.0:.3f}, {place.y_m or 0.0:.3f})"
