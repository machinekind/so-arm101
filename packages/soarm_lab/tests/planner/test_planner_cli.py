"""The planner's CLI runner against a fake claude and codex (fake_cli.py): argv, stdin, environment, schema,
timeouts, cancels and failures; then the runner thread, the fake planner and the schema check. No model,
no network."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from soarm_lab.planner import cli
from soarm_lab.planner.cli import (
    BadAnswer,
    CliFailed,
    CliMissing,
    CliPlanner,
    CliSettings,
    FakePlanner,
    LoggedOut,
    PlannerCancelled,
    PlannerError,
    PlannerRunner,
    PlannerTimeout,
    Scripted,
    UsageLimit,
    check_answer,
    validate,
)

FAKE = Path(__file__).with_name("fake_cli.py")

NUM = {"type": "number"}
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["steps", "note"],
    "properties": {
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["block", "x_mm", "y_mm"],
            "properties": {"block": {"type": "string"}, "x_mm": NUM, "y_mm": NUM}}},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
ANSWER = {"steps": [{"block": "red cube", "x_mm": 120.0, "y_mm": -40.0}], "note": None}
PROMPT = "Stack the red cube on the green one."
MODELS = {"claude": "sonnet", "codex": "gpt-6-astra"}


class Fake:
    def __init__(self, bin_dir: Path, log: Path) -> None:
        self.bin = bin_dir
        self.claude = str(bin_dir / "claude")
        self.codex = str(bin_dir / "codex")
        self._log = log

    def runs(self) -> list[dict[str, Any]]:
        if not self._log.is_file():
            return []
        return [json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines()]

    def last(self) -> dict[str, Any]:
        return [r for r in self.runs() if "argv" in r][-1]

    def planner(self, who: str, **settings: Any) -> CliPlanner:
        settings.setdefault("model", MODELS[who])
        return CliPlanner(CliSettings(cli=who, executable=getattr(self, who), **settings))


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body = FAKE.read_text(encoding="utf-8")
    for name in ("claude", "codex"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_ANSWER", json.dumps(ANSWER))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.setenv("FAKE_AUTH", "subscription")
    monkeypatch.delenv(cli.KEEP_ENV, raising=False)
    return Fake(bin_dir, log)


def after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int, within_s: float = 3.0) -> bool:
    deadline = time.monotonic() + within_s
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not alive(pid)


def logged(fake: Fake, key: str, within_s: float = 5.0) -> int:
    """The value of the first log entry with `key`, waiting for the fake to write it."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        for run in fake.runs():
            if key in run:
                return int(run[key])
        time.sleep(0.02)
    raise AssertionError(f"the fake never logged {key!r}")


# The CLIs' argv, input and environment ----------------------------------------------------------------

def test_claude_gets_the_prompt_on_stdin_and_the_schema_in_its_argv(fake: Fake):
    got = fake.planner("claude", effort="low").ask(PROMPT, SCHEMA)

    assert got == ANSWER
    run = fake.last()
    argv = run["argv"]
    assert argv[0] == fake.claude
    assert "-p" in argv and "--bare" not in argv
    assert after(argv, "--output-format") == "json"
    assert json.loads(after(argv, "--json-schema")) == SCHEMA
    assert after(argv, "--tools") == ""
    assert after(argv, "--max-turns") == str(cli.CLAUDE_MAX_TURNS)
    assert {"--safe-mode", "--no-session-persistence", "--strict-mcp-config"} <= set(argv)
    assert after(argv, "--permission-prompts") == "none"
    assert after(argv, "--model") == "sonnet" and after(argv, "--effort") == "low"
    assert run["stdin"] == PROMPT
    # A fresh call directory, removed afterwards.
    assert Path(run["cwd"]).name.startswith("soarm-planner-")
    assert not Path(run["cwd"]).exists()


def test_codex_gets_the_prompt_on_stdin_and_the_schema_as_a_file(fake: Fake):
    got = fake.planner("codex", effort="low").ask(PROMPT, SCHEMA)

    assert got == ANSWER
    run = fake.last()
    argv = run["argv"][1:]
    call = Path(run["cwd"])
    assert argv[:4] == ["exec", "--json", "-o", str(Path(after(argv, "--cd")) / "last.txt")]
    assert {"--skip-git-repo-check", "--ephemeral", "--ignore-user-config"} <= set(argv)
    assert after(argv, "--sandbox") == "read-only"
    assert Path(after(argv, "--cd")).resolve() == call.resolve()
    assert after(argv, "-m") == "gpt-6-astra"
    assert after(argv, "-c") == 'model_reasoning_effort="low"'
    assert argv[-2:] == ["--", "-"]
    assert run["schema"] == SCHEMA
    assert run["stdin"] == PROMPT
    assert not call.exists()


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_without_model_or_effort_neither_is_passed(fake: Fake, who: str):
    fake.planner(who, model="", effort="").ask(PROMPT, SCHEMA)

    argv = fake.last()["argv"]
    assert not {"--model", "-m", "--effort", "-c"} & set(argv)


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_api_keys_and_the_calling_session_never_reach_the_cli(fake: Fake, monkeypatch: pytest.MonkeyPatch,
                                                               who: str):
    stripped = {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "CLAUDECODE",
                "CMUX_SURFACE_ID", "CLAUDE_CODE_SESSION_ID"}
    for name in stripped:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-token")

    fake.planner(who).ask(PROMPT, SCHEMA)

    keys = set(fake.last()["env_keys"])
    assert not keys & stripped
    assert "CLAUDE_CODE_OAUTH_TOKEN" in keys
    assert {r["auth"] for r in fake.runs() if "auth" in r} == {who}


def test_clean_env_strips_exactly_the_listed_names(monkeypatch: pytest.MonkeyPatch):
    for name in cli.STRIPPED_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("KEEP_ME", "y")

    env = cli.clean_env()

    assert not set(cli.STRIPPED_ENV) & set(env)
    assert env["KEEP_ME"] == "y"


def test_the_cli_is_found_on_path(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(fake.bin))

    assert CliPlanner(CliSettings(cli="codex")).ask(PROMPT, SCHEMA) == ANSWER
    assert fake.last()["argv"][0] == fake.codex


def test_the_login_is_checked_once(fake: Fake):
    planner = fake.planner("claude")

    assert planner.check() == "claude.ai (max)"
    planner.ask(PROMPT, SCHEMA)
    planner.ask(PROMPT, SCHEMA)

    assert sum("auth" in r for r in fake.runs()) == 1
    assert sum("argv" in r for r in fake.runs()) == 2


def test_keep_copies_each_call_directory(fake: Fake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    keep = tmp_path / "keep"
    monkeypatch.setenv(cli.KEEP_ENV, str(keep))

    fake.planner("claude").ask(PROMPT, SCHEMA)
    fake.planner("codex").ask(PROMPT, SCHEMA)

    codex_dir, claude_dir = sorted(keep.iterdir(), key=lambda p: not (p / "last.txt").exists())
    files = {"prompt.txt", "argv.json", "schema.json", "stdout.txt", "stderr.txt"}
    assert {p.name for p in claude_dir.iterdir()} == files
    assert {p.name for p in codex_dir.iterdir()} == files | {"last.txt"}
    assert (claude_dir / "prompt.txt").read_text(encoding="utf-8") == PROMPT
    assert json.loads((claude_dir / "argv.json").read_text(encoding="utf-8"))[0] == fake.claude
    assert json.loads((codex_dir / "schema.json").read_text(encoding="utf-8")) == SCHEMA
    assert json.loads((codex_dir / "last.txt").read_text(encoding="utf-8")) == ANSWER
    assert not Path(fake.last()["cwd"]).exists()


# Failures ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_hanging_cli_times_out_and_its_whole_process_group_is_killed(
        fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str):
    monkeypatch.setenv("FAKE_MODE", "child")
    t0 = time.monotonic()

    with pytest.raises(PlannerTimeout, match="no answer in 1 s"):
        fake.planner(who, timeout_s=1.0).ask(PROMPT, SCHEMA)

    assert time.monotonic() - t0 < 5.0
    assert gone(logged(fake, "pid"))
    assert gone(logged(fake, "grandchild"))
    assert not Path(fake.last()["cwd"]).exists()


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_cancel_mid_call_kills_the_cli_within_a_second(fake: Fake, monkeypatch: pytest.MonkeyPatch,
                                                         who: str):
    monkeypatch.setenv("FAKE_MODE", "sleep")
    cancel = threading.Event()
    fired: list[float] = []

    def fire() -> None:
        logged(fake, "sleeping")
        fired.append(time.monotonic())
        cancel.set()

    timer = threading.Thread(target=fire)
    timer.start()
    with pytest.raises(PlannerCancelled):
        fake.planner(who, timeout_s=30.0).ask(PROMPT, SCHEMA, cancel=cancel)
    timer.join()

    assert time.monotonic() - fired[0] < 1.0
    assert gone(logged(fake, "pid"))


@pytest.mark.parametrize(("who", "mode", "error", "message"), [
    ("claude", "garbage", BadAnswer, "not JSON"),
    ("claude", "no_structured", BadAnswer, "no structured output"),
    ("claude", "exit1", CliFailed, "claude: exit 1: boom: something broke"),
    ("claude", "limit", UsageLimit, "subscription limit reached"),
    ("claude", "logged_out", LoggedOut, "Not logged in.*run `claude` and log in"),
    ("claude", "expired", LoggedOut, "OAuth token has expired"),
    ("codex", "garbage", BadAnswer, "not JSON"),
    ("codex", "no_structured", BadAnswer, "empty answer"),
    ("codex", "exit1", CliFailed, "codex: exit 1: boom: something broke"),
    ("codex", "limit", UsageLimit, "subscription limit reached"),
    ("codex", "logged_out", LoggedOut, "Not logged in.*run `codex login`"),
    ("codex", "turn_failed", CliFailed, "the model refused"),
])
def test_unusable_calls_raise_a_clear_error(fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str, mode: str,
                                            error: type[PlannerError], message: str):
    monkeypatch.setenv("FAKE_MODE", mode)

    with pytest.raises(error, match=message):
        fake.planner(who).ask(PROMPT, SCHEMA)


def test_a_codex_retry_notice_alone_is_not_a_failure(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "retry_notice")

    assert fake.planner("codex").ask(PROMPT, SCHEMA) == ANSWER


@pytest.mark.parametrize("who", ["claude", "codex"])
@pytest.mark.parametrize(("answer", "message"), [
    ({"steps": []}, "missing required key 'note'"),
    ({"steps": [{"block": "red cube", "x_mm": "far", "y_mm": 0}], "note": None},
     r"answer.steps\[0\].x_mm: expected number, got string"),
    ({"steps": [], "note": None, "extra": 1}, "unexpected key 'extra'"),
])
def test_an_answer_that_breaks_the_schema_is_a_bad_answer(fake: Fake, monkeypatch: pytest.MonkeyPatch,
                                                         who: str, answer: dict[str, Any], message: str):
    monkeypatch.setenv("FAKE_ANSWER", json.dumps(answer))

    with pytest.raises(BadAnswer, match=message):
        fake.planner(who).ask(PROMPT, SCHEMA)


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_logged_out_cli_is_refused_before_any_question(fake: Fake, monkeypatch: pytest.MonkeyPatch,
                                                         who: str):
    monkeypatch.setenv("FAKE_AUTH", "logged_out")
    hint = "run `claude` and log in" if who == "claude" else "run `codex login`"

    with pytest.raises(LoggedOut, match=f"not logged in; {hint}"):
        fake.planner(who).ask(PROMPT, SCHEMA)

    assert not [r for r in fake.runs() if "argv" in r]


@pytest.mark.parametrize("who", ["claude", "codex"])
def test_a_cli_on_an_api_key_is_refused(fake: Fake, monkeypatch: pytest.MonkeyPatch, who: str):
    monkeypatch.setenv("FAKE_AUTH", "apikey")

    with pytest.raises(LoggedOut, match="not a subscription"):
        fake.planner(who).check()


def test_a_missing_cli_is_cli_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(CliMissing, match="codex not found on PATH"):
        CliPlanner(CliSettings(cli="codex")).ask(PROMPT, SCHEMA)
    with pytest.raises(CliMissing, match="not found or not executable"):
        CliPlanner(CliSettings(cli="claude", model="", executable=str(tmp_path / "nope"))).check()


def test_a_cli_that_cannot_start_is_cli_missing(tmp_path: Path):
    with pytest.raises(CliMissing, match="could not be started"):
        cli.run_cli([str(tmp_path / "nope")], stdin="", cwd=tmp_path, timeout_s=5, cancel=threading.Event())


def test_a_call_cancelled_before_it_starts_runs_nothing(fake: Fake):
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(PlannerCancelled):
        fake.planner("claude").ask(PROMPT, SCHEMA, cancel=cancel)

    assert not [r for r in fake.runs() if "argv" in r]


@pytest.mark.parametrize(("change", "message"), [
    ({"cli": "gemini"}, "cli must be one of"),
    ({"effort": "huge"}, "effort must be one of"),
    ({"timeout_s": 0.0}, "timeout_s must be a positive"),
    ({"timeout_s": float("nan")}, "timeout_s must be a positive"),
    ({"cli": "claude"}, "is an OpenAI model and cli is claude"),
    ({"model": "opus"}, "is a Claude model and cli is codex"),
])
def test_settings_no_call_can_work_with_are_refused(change: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        validate(CliSettings(**change))
    with pytest.raises(ValueError, match=message):
        CliPlanner(CliSettings(**change))


# The runner thread -------------------------------------------------------------------------------------

def test_the_runner_polls_none_until_the_answer_then_gives_it_once():
    gate = threading.Event()

    class Gated:
        def ask(self, prompt: str, schema: Any, *, cancel: threading.Event | None = None) -> dict[str, Any]:
            gate.wait(5.0)
            return dict(ANSWER)

    runner = PlannerRunner(Gated())
    runner.start(PROMPT, SCHEMA)

    assert runner.pending and runner.poll() is None
    with pytest.raises(RuntimeError, match="already pending"):
        runner.start(PROMPT, SCHEMA)
    gate.set()
    runner.wait(5.0)
    outcome = runner.poll()
    assert outcome is not None and outcome.answer == ANSWER and outcome.error is None
    assert outcome.response_s >= 0.0
    assert runner.poll() is None and not runner.pending


def test_the_runner_reports_errors_and_planner_bugs_as_the_outcome():
    class Broken:
        def ask(self, prompt: str, schema: Any, *, cancel: threading.Event | None = None) -> dict[str, Any]:
            raise KeyError("oops")

    runner = PlannerRunner(FakePlanner([UsageLimit("spent")]), threaded=False)
    runner.start(PROMPT, SCHEMA)
    outcome = runner.poll()
    assert outcome is not None and isinstance(outcome.error, UsageLimit) and outcome.answer is None

    runner = PlannerRunner(Broken(), threaded=False)
    runner.start(PROMPT, SCHEMA)
    outcome = runner.poll()
    assert outcome is not None and type(outcome.error) is PlannerError
    assert "KeyError" in str(outcome.error)


def test_the_runner_measures_on_its_clock():
    clock = [10.0]

    class Slow:
        def ask(self, prompt: str, schema: Any, *, cancel: threading.Event | None = None) -> dict[str, Any]:
            clock[0] += 2.5
            return dict(ANSWER)

    runner = PlannerRunner(Slow(), threaded=False, clock=lambda: clock[0])
    runner.start(PROMPT, SCHEMA)
    outcome = runner.poll()
    assert outcome is not None and outcome.response_s == pytest.approx(2.5)


def test_the_runner_copies_the_schema():
    planner = FakePlanner([ANSWER])
    runner = PlannerRunner(planner, threaded=False)
    schema = json.loads(json.dumps(SCHEMA))

    runner.start(PROMPT, schema)
    schema["required"].append("later")

    assert planner.asked[0][1] == SCHEMA


def test_a_cancelled_call_is_dropped_and_the_runner_can_start_again():
    planner = FakePlanner([Scripted(answer=ANSWER, delay_s=30.0), {"steps": [], "note": "again"}])
    runner = PlannerRunner(planner)
    runner.start(PROMPT, SCHEMA)
    t0 = time.monotonic()

    runner.cancel()
    runner.wait(5.0)

    assert time.monotonic() - t0 < 2.0
    assert runner.poll() is None and not runner.pending
    runner.start(PROMPT, SCHEMA)
    runner.wait(5.0)
    outcome = runner.poll()
    assert outcome is not None and outcome.answer == {"steps": [], "note": "again"}


def test_cancelling_the_runner_kills_the_cli(fake: Fake, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_MODE", "child")
    runner = PlannerRunner(fake.planner("codex", timeout_s=30.0))
    runner.start(PROMPT, SCHEMA)
    grandchild = logged(fake, "grandchild")

    runner.cancel()

    assert runner.poll() is None and not runner.pending
    assert gone(logged(fake, "pid"))
    assert gone(grandchild)
    runner.wait(5.0)
    assert runner.poll() is None


# The fake planner and the schema check -----------------------------------------------------------------

def test_the_fake_planner_plays_its_script_in_order():
    other = {"steps": [], "note": "b"}
    planner = FakePlanner([ANSWER, CliFailed("exit 1"), Scripted(answer=other)])

    assert planner.ask("first", SCHEMA) == ANSWER
    with pytest.raises(CliFailed):
        planner.ask("second", SCHEMA)
    assert planner.ask("third", SCHEMA) == other
    with pytest.raises(PlannerError, match="no scripted step left"):
        planner.ask("fourth", SCHEMA)
    assert [prompt for prompt, _ in planner.asked] == ["first", "second", "third", "fourth"]
    planner.add(ANSWER)
    assert planner.ask("fifth", SCHEMA) == ANSWER


def test_the_fake_planner_checks_answers_against_the_schema():
    with pytest.raises(BadAnswer, match="missing required key 'note'"):
        FakePlanner([{"steps": []}]).ask(PROMPT, SCHEMA)


def test_the_fake_planner_waits_and_a_cancel_ends_the_wait():
    planner = FakePlanner([ANSWER, ANSWER], delay_s=0.2)
    t0 = time.monotonic()
    assert planner.ask(PROMPT, SCHEMA) == ANSWER
    assert time.monotonic() - t0 >= 0.2

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(PlannerCancelled):
        planner.ask(PROMPT, SCHEMA, cancel=cancel)


def test_a_scripted_step_needs_exactly_one_outcome():
    with pytest.raises(ValueError, match="exactly one"):
        Scripted()
    with pytest.raises(ValueError, match="exactly one"):
        Scripted(answer={}, error=PlannerError("x"))


STEP = {"type": "object", "additionalProperties": False, "required": ["block", "on", "level"],
        "properties": {"block": {"type": "string", "enum": ["red", "green"]},
                       "on": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                       "level": {"type": "integer"}}}
PLAN = {"$defs": {"step": STEP}, "type": "object", "additionalProperties": False,
        "required": ["steps", "done"],
        "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                       "done": {"type": "boolean"}}}


@pytest.mark.parametrize(("answer", "problem"), [
    ({"steps": [{"block": "red", "on": None, "level": 0}], "done": False}, None),
    ({"steps": [{"block": "red", "on": "green", "level": 1.0}], "done": True}, None),
    ([], "not a JSON object"),
    ({"steps": [], "done": 1}, "answer.done: expected boolean, got integer"),
    ({"steps": [{"block": "blue", "on": None, "level": 0}], "done": False}, "is not one of"),
    ({"steps": [{"block": "red", "on": 3, "level": 0}], "done": False}, "matches none of its anyOf"),
    ({"steps": [{"block": "red", "on": None, "level": True}], "done": False},
     "expected integer, got boolean"),
    ({"steps": [{"block": "red", "on": None, "level": 0.5}], "done": False},
     "expected integer, got number"),
    ({"steps": [{"block": "red", "on": None}], "done": False}, r"answer.steps\[0\]: missing required key"),
])
def test_the_schema_check(answer: Any, problem: str | None):
    if problem is None:
        assert check_answer(answer, PLAN) is answer
    else:
        with pytest.raises(BadAnswer, match=problem):
            check_answer(answer, PLAN)


def test_an_unknown_ref_is_a_bad_answer():
    with pytest.raises(BadAnswer, match="unknown \\$ref"):
        check_answer({"a": 1}, {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}})
