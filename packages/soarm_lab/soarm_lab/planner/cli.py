"""Ask the claude or codex CLI for one JSON answer, run headless on the subscription it is logged in with.

No API keys, SDKs or HTTP: each call is one subprocess of the locally installed CLI. Its environment has API
keys, base-URL overrides and a calling Claude Code session's variables removed, so the CLI answers on its
own login. Each call runs in a fresh directory that holds the prompt, argv, schema, stdout and stderr as
files, never pipes; with `SOARM_PLANNER_KEEP=<dir>` a copy of it is kept there. The CLI gets a process
session of its own, and a timeout or a cancel kills the whole process group.

`CliPlanner.ask` blocks. `PlannerRunner` runs any `Planner` on a worker thread, so a strategy tick can poll
it without waiting. `FakePlanner` stands in for a planner in tests.

The CLI flags follow the twin's LLM detector (`machinekind_twin_llm.cli`), without images.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "CLIS", "EFFORTS", "KEEP_ENV", "STRIPPED_ENV", "BadAnswer", "CliFailed", "CliMissing", "CliPlanner",
    "CliSettings", "FakePlanner", "LoggedOut", "Planner", "PlannerCancelled", "PlannerError",
    "PlannerOutcome", "PlannerRunner", "PlannerTimeout", "Scripted", "UsageLimit", "check_answer",
    "check_login", "claude_command", "clean_env", "codex_command", "find_cli", "run_cli", "validate",
]

logger = logging.getLogger(__name__)

#: The CLIs a planner can ask.
CLIS = ("codex", "claude")
#: Reasoning efforts passed to the CLI; "" = the CLI's default.
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")
#: claude's model aliases; any "claude-..." name counts too.
CLAUDE_ALIASES = ("sonnet", "opus", "haiku", "fable")
#: Set it to a directory to keep a copy of every call's files there.
KEEP_ENV = "SOARM_PLANNER_KEEP"
#: Removed from the CLI's environment: API keys and base URLs (the login decides), a key the CLI has no use
#: for, and the variables of a calling Claude Code or cmux session. `CLAUDE_CODE_OAUTH_TOKEN` stays: it is
#: the subscription token of `claude setup-token`.
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "TYPESAFE_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
    "CMUX_SURFACE_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_EXECPATH", "CLAUDE_PID",
    "CLAUDE_EFFORT",
)
#: How long the login check may take [s].
LOGIN_TIMEOUT_S = 15.0
#: claude's turn cap for one answer: structured output may take more than one turn.
CLAUDE_MAX_TURNS = 3

#: How often a running CLI is checked for a cancel or the deadline [s].
_POLL_S = 0.1
#: Output lines that say nothing about why a call failed: MCP and OAuth start-up chatter.
_NOISE = (
    re.compile(r"\bmcp\b", re.IGNORECASE),
    re.compile(r"oauth", re.IGNORECASE),
    re.compile(r"^\s*Reading (prompt|additional input) from stdin", re.IGNORECASE),
)
_LIMIT = re.compile(r"usage limit|rate limit|quota|\b429\b", re.IGNORECASE)
_LOGIN = re.compile(r"not logged in|log ?in\b", re.IGNORECASE)
_API_KEY = re.compile(r"api[\s_-]?key", re.IGNORECASE)


class PlannerError(RuntimeError):
    """The planner gave no usable answer."""


class CliMissing(PlannerError):
    """The CLI is not installed or could not be started."""


class LoggedOut(PlannerError):
    """The CLI is not logged in, or is logged in with an API key instead of a subscription."""


class UsageLimit(PlannerError):
    """The subscription's usage or rate limit is reached."""


class PlannerTimeout(PlannerError):
    """No answer in time; the CLI's process group was killed."""


class PlannerCancelled(PlannerError):
    """The call was cancelled; the CLI's process group was killed."""


class CliFailed(PlannerError):
    """The CLI exited with an error or reported a failed turn."""


class BadAnswer(PlannerError):
    """The answer is not JSON or does not match the schema."""


@dataclass
class CliSettings:
    """Which CLI the planner asks, and how."""

    #: Which locally installed CLI answers: "codex" or "claude".
    cli: str = "codex"
    #: The model as the CLI names it ("gpt-6-astra", "sonnet", "claude-opus-5-5"); "" = the CLI's default.
    model: str = "gpt-6-astra"
    #: Reasoning effort: low, medium, high, xhigh, max; "" = the CLI's default.
    effort: str = "low"
    #: The CLI binary; "" = `cli` found on PATH. Point it at ~/.local/bin/<cli> to bypass terminal shims.
    executable: str = ""
    #: Kill the CLI after this many seconds.
    timeout_s: float = 90.0


def validate(settings: CliSettings) -> None:
    """Raise ValueError for settings no CLI call can work with."""
    s = settings
    if s.cli not in CLIS:
        raise ValueError(f"planner: cli must be one of {', '.join(CLIS)}, not {s.cli!r}")
    if s.effort not in EFFORTS:
        raise ValueError(f"planner: effort must be one of {', '.join(e or repr(e) for e in EFFORTS)}, "
                         f"not {s.effort!r}")
    if not (math.isfinite(s.timeout_s) and s.timeout_s > 0):
        raise ValueError(f"planner: timeout_s must be a positive number of seconds, got {s.timeout_s}")
    model = s.model.strip().lower()
    if s.cli == "claude" and model.startswith("gpt-"):
        raise ValueError(f"planner: model {s.model!r} is an OpenAI model and cli is claude; use a claude "
                         "model (sonnet, opus, ...) or \"\" for claude's default")
    if s.cli == "codex" and (model in CLAUDE_ALIASES or model.startswith("claude-")):
        raise ValueError(f"planner: model {s.model!r} is a Claude model and cli is codex; use an OpenAI "
                         "model (gpt-6-astra, ...) or \"\" for codex's default")


class Planner(Protocol):
    """Anything that answers a prompt with a JSON object matching a schema."""

    def ask(self, prompt: str, schema: Mapping[str, Any], *,
            cancel: threading.Event | None = None) -> dict[str, Any]:
        """The answer, blocking. Raises PlannerError when there is no usable one, and PlannerCancelled
        soon after `cancel` is set."""
        ...


# Schema check ------------------------------------------------------------------------------------------

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
}


def _type_name(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPES[name](value):
            return name
    return type(value).__name__


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: true and 1 differ."""
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _resolve(root: Mapping[str, Any], ref: str) -> Any:
    """The node a local `$ref` ("#/$defs/step") points at, or None."""
    if not ref.startswith("#"):
        return None
    node: Any = root
    for part in ref[1:].split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _problem(value: Any, node: Any, root: Mapping[str, Any], path: str, depth: int = 0) -> str | None:
    """The first way `value` breaks `node`, or None."""
    if not isinstance(node, Mapping):
        return None
    if depth > 64:
        return f"{path}: schema nests too deep"
    ref = node.get("$ref")
    if isinstance(ref, str):
        target = _resolve(root, ref)
        if target is None:
            return f"{path}: unknown $ref {ref!r}"
        found = _problem(value, target, root, path, depth + 1)
        if found:
            return found
    options = node.get("anyOf")
    if isinstance(options, list) and options:
        if all(_problem(value, option, root, path, depth + 1) for option in options):
            return f"{path}: {_type_name(value)} matches none of its anyOf options"
    kind = node.get("type")
    if kind is not None:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        if not any(_TYPES.get(k, lambda v: True)(value) for k in kinds):
            return f"{path}: expected {' or '.join(kinds)}, got {_type_name(value)}"
    if "enum" in node and not any(_equal(value, e) for e in node["enum"]):
        return f"{path}: {value!r} is not one of {node['enum']!r}"
    if "const" in node and not _equal(value, node["const"]):
        return f"{path}: {value!r} is not {node['const']!r}"
    if isinstance(value, dict):
        props = node.get("properties") or {}
        for key in node.get("required") or ():
            if key not in value:
                return f"{path}: missing required key {key!r}"
        extra = node.get("additionalProperties")
        for key, item in value.items():
            if key in props:
                found = _problem(item, props[key], root, f"{path}.{key}", depth + 1)
            elif extra is False:
                found = f"{path}: unexpected key {key!r}"
            else:
                found = _problem(item, extra, root, f"{path}.{key}", depth + 1)
            if found:
                return found
    if isinstance(value, list):
        items = node.get("items")
        for i, item in enumerate(value):
            found = _problem(item, items, root, f"{path}[{i}]", depth + 1)
            if found:
                return found
    return None


def check_answer(answer: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """`answer` when it is a JSON object matching `schema`, else BadAnswer naming the first problem.

    Checks type, enum, const, anyOf, local $ref, properties, required, additionalProperties and items;
    other keywords are not checked.
    """
    if not isinstance(answer, dict):
        raise BadAnswer(f"answer is not a JSON object but {_type_name(answer)}")
    found = _problem(answer, schema, schema, "answer")
    if found:
        raise BadAnswer(f"answer does not match the schema: {found}")
    return answer


# Running a CLI -----------------------------------------------------------------------------------------

def clean_env() -> dict[str, str]:
    """This process's environment without `STRIPPED_ENV`."""
    return {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}


def find_cli(settings: CliSettings) -> str:
    """The absolute path of the CLI the settings name, or CliMissing."""
    if settings.executable:
        given = str(Path(settings.executable).expanduser())
        exe = shutil.which(given)
        if exe is None:
            raise CliMissing(f"{given} not found or not executable; fix `executable` or leave it empty")
    else:
        exe = shutil.which(settings.cli)
        if exe is None:
            raise CliMissing(f"{settings.cli} not found on PATH; install it or set `executable`")
    # Calls change the working directory; keep the executable's own (symlink) name.
    return str(Path(exe).absolute())


def _hint(name: str) -> str:
    return "run `codex login`" if name == "codex" else "run `claude` and log in"


def _tail(text: str, limit: int = 600) -> str:
    """The last `limit` characters of a CLI's output, without its start-up noise."""
    lines = [line for line in text.splitlines() if line.strip() and not any(p.search(line) for p in _NOISE)]
    return "\n".join(lines)[-limit:].strip()


def _failure(name: str, detail: str) -> PlannerError:
    """The error for a failed call: a spent subscription, a lost login, or any other failure."""
    if _LIMIT.search(detail):
        return UsageLimit(f"{name}: subscription limit reached: {detail}")
    if _LOGIN.search(detail):
        return LoggedOut(f"{name}: {detail}; {_hint(name)}")
    return CliFailed(f"{name}: {detail}")


def _kill(proc: subprocess.Popen[Any]) -> None:
    """Terminate the process group, give its leader up to 2 s to exit, then kill whatever is left."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(2)
    except subprocess.TimeoutExpired:
        pass
    # The leader exiting does not mean its children accepted SIGTERM.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def run_cli(cmd: Sequence[str], *, stdin: str, cwd: Path, timeout_s: float, cancel: threading.Event,
            name: str = "") -> tuple[int, str, str]:
    """Run `cmd` in `cwd` with `stdin` as its input; return (exit code, stdout, stderr).

    Input and output go through files in `cwd` (`prompt.txt`, `stdout.txt`, `stderr.txt`), never pipes.
    Raises PlannerCancelled within 0.1 s of `cancel` being set, PlannerTimeout after `timeout_s`, and
    CliMissing when the command cannot be started.
    """
    name = name or Path(cmd[0]).name
    if cancel.is_set():
        raise PlannerCancelled(f"{name}: cancelled")
    src, out, err = cwd / "prompt.txt", cwd / "stdout.txt", cwd / "stderr.txt"
    src.write_text(stdin, encoding="utf-8")
    with (open(src, encoding="utf-8") as fi, open(out, "w", encoding="utf-8") as fo,
          open(err, "w", encoding="utf-8") as fe):
        try:
            proc = subprocess.Popen(list(cmd), stdin=fi, stdout=fo, stderr=fe, cwd=cwd, env=clean_env(),
                                    start_new_session=True)
        except OSError as e:
            raise CliMissing(f"{name} could not be started: {e}") from e
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                proc.wait(timeout=_POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel.is_set():
                _kill(proc)
                raise PlannerCancelled(f"{name}: cancelled")
            if time.monotonic() > deadline:
                _kill(proc)
                raise PlannerTimeout(f"{name} gave no answer in {timeout_s:g} s")
    return proc.returncode, out.read_text(encoding="utf-8"), err.read_text(encoding="utf-8")


@contextmanager
def _call_dir() -> Iterator[Path]:
    """A fresh directory for one call, removed afterwards; first copied to `$SOARM_PLANNER_KEEP` if set."""
    call = Path(tempfile.mkdtemp(prefix="soarm-planner-"))
    try:
        yield call
    finally:
        keep = os.environ.get(KEEP_ENV)
        if keep:
            try:
                dest = Path(keep).expanduser() / f"{time.strftime('%Y%m%d-%H%M%S')}-{call.name}"
                shutil.copytree(call, dest, dirs_exist_ok=True)
            except OSError as e:
                logger.warning("planner: could not keep %s: %s", call, e)
        shutil.rmtree(call, ignore_errors=True)


def check_login(cli: str, exe: str, *, timeout_s: float = LOGIN_TIMEOUT_S) -> str:
    """How `cli` is logged in, e.g. "claude.ai (max)" or "Logged in using ChatGPT".

    Raises LoggedOut when it is not logged in or uses an API key, CliMissing when it cannot be run, and
    PlannerTimeout when the check hangs.
    """
    args = [exe, "auth", "status"] if cli == "claude" else [exe, "login", "status"]
    try:
        with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=clean_env(), start_new_session=True) as r:
            try:
                stdout, stderr = r.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired as e:
                _kill(r)
                raise PlannerTimeout(f"{cli} login check gave no answer in {timeout_s:g} s") from e
    except OSError as e:
        raise CliMissing(f"{cli} could not be run: {e}") from e
    text = _tail(stdout + "\n" + stderr, 300)
    logged_out = LoggedOut(f"{cli} is not logged in; {_hint(cli)}")
    if cli == "claude":
        try:
            status = json.loads(stdout)
        except ValueError:
            status = None
        if isinstance(status, dict):
            method = str(status.get("authMethod") or "")
            source = str(status.get("apiKeySource") or "")
            if _API_KEY.search(method) or source:
                raise LoggedOut(f"claude uses {method or source}, not a subscription; {_hint(cli)}")
            if r.returncode != 0 or not status.get("loggedIn", True):
                raise logged_out
            plan = status.get("subscriptionType")
            return (method or "logged in") + (f" ({plan})" if plan else "")
    if r.returncode != 0:
        raise logged_out
    if _API_KEY.search(text):
        raise LoggedOut(f"{cli} uses an API key, not a subscription: {text}; {_hint(cli)}")
    return text.splitlines()[0] if text else "logged in"


def _events(stdout: str) -> list[dict[str, Any]]:
    """The JSON objects of an NDJSON stream, skipping lines that are not JSON objects."""
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def claude_command(exe: str, schema: Mapping[str, Any], *, model: str = "", effort: str = "") -> list[str]:
    """claude's argv for one question read from stdin: no tools, no session saved, no user setup.

    `--bare` is never passed: it turns the subscription login off.
    """
    return [exe, "-p", "--output-format", "json", "--tools", "", "--max-turns", str(CLAUDE_MAX_TURNS),
            "--json-schema", json.dumps(schema), "--no-session-persistence", "--safe-mode",
            "--strict-mcp-config", "--permission-prompts", "none",
            *(["--model", model] if model else []), *(["--effort", effort] if effort else [])]


def codex_command(exe: str, call: Path, *, model: str = "", effort: str = "") -> list[str]:
    """codex's argv for one question read from stdin; the schema is `call/schema.json`, the answer
    `call/last.txt`. `--ignore-user-config` drops the user's config.toml and keeps the login."""
    return [exe, "exec", "--json", "-o", str(call / "last.txt"), "--skip-git-repo-check", "--ephemeral",
            "--ignore-user-config", "--sandbox", "read-only", "--cd", str(call), "--color", "never",
            "--output-schema", str(call / "schema.json"),
            *(["-m", model] if model else []),
            *(["-c", f"model_reasoning_effort={json.dumps(effort)}"] if effort else []),
            "--", "-"]


def _claude_answer(code: int, stdout: str, stderr: str) -> Any:
    """The structured answer of claude's `result` object."""
    try:
        parsed: Any = json.loads(stdout)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        results = [e for e in parsed if isinstance(e, dict) and e.get("type") == "result"]
        parsed = results[-1] if results else None
    result = parsed if isinstance(parsed, dict) else None
    if result is not None and (result.get("is_error") or result.get("subtype") != "success"):
        # The CLI's own final message: no start-up noise to drop, and it may name OAuth.
        detail = str(result.get("result") or result.get("errors") or result.get("subtype")).strip()[-600:]
        raise _failure("claude", f"exit {code}: {detail}" if code else detail)
    if code != 0:
        raise _failure("claude", f"exit {code}: {_tail(stderr) or _tail(stdout)}")
    if result is None:
        raise BadAnswer(f"claude answer is not JSON: {_tail(stdout, 200)}")
    if "structured_output" not in result:
        raise BadAnswer(f"claude gave no structured output: {_tail(str(result.get('result') or ''), 200)}")
    return result["structured_output"]


def _codex_answer(code: int, stdout: str, stderr: str, last: Path) -> Any:
    """codex's last message, parsed. An `error` event alone may be a retry notice; a failed turn or a
    non-zero exit is a failure."""
    errors: list[str] = []
    failed = False
    message = ""
    for event in _events(stdout):
        kind = event.get("type")
        if kind in ("turn.failed", "error"):
            error = event.get("error")
            detail = error.get("message") if isinstance(error, dict) else event.get("message")
            errors.append(str(detail or kind))
            failed = failed or kind == "turn.failed"
        item = event.get("item")
        if kind == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
            message = str(item.get("text") or "")
    said = "; ".join(dict.fromkeys(errors))
    if failed or code != 0:
        detail = said or _tail(stderr) or _tail(stdout)
        raise _failure("codex", f"exit {code}: {detail}" if code else detail)
    text = (last.read_text(encoding="utf-8") if last.is_file() else "").strip() or message.strip()
    if not text:
        raise _failure("codex", said) if said else BadAnswer("codex gave an empty answer")
    try:
        return json.loads(text)
    except ValueError as e:
        raise BadAnswer(f"codex answer is not JSON: {text[:200]}") from e


class CliPlanner:
    """Asks the claude or codex CLI, headless on its subscription login, for answers matching a schema.

    The first `ask` finds the CLI and checks its login (`check`); later calls reuse that. Each call starts
    a fresh CLI process, so calls share no conversation.
    """

    def __init__(self, settings: CliSettings | None = None):
        self.settings = settings if settings is not None else CliSettings()
        validate(self.settings)
        self._exe = ""
        self._login = ""
        self._lock = threading.Lock()

    def check(self) -> str:
        """How the CLI is logged in; finds it and checks the login on the first call only.

        Raises CliMissing, LoggedOut or PlannerTimeout. Start-up code can call it to fail early.
        """
        with self._lock:
            if not self._login:
                exe = find_cli(self.settings)
                login = check_login(self.settings.cli, exe)
                logger.info("planner: %s at %s, logged in as %s", self.settings.cli, exe, login)
                self._exe, self._login = exe, login
            return self._login

    def ask(self, prompt: str, schema: Mapping[str, Any], *,
            cancel: threading.Event | None = None) -> dict[str, Any]:
        """The CLI's answer to `prompt`, checked against `schema`; blocks for up to `timeout_s`.

        `schema` should be strict (every object closed and all its properties required) so both CLIs take
        it. Raises a PlannerError subclass when there is no usable answer.
        """
        s = self.settings
        cancel = cancel if cancel is not None else threading.Event()
        t0 = time.monotonic()
        try:
            self.check()
            with _call_dir() as call:
                (call / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
                if s.cli == "claude":
                    cmd = claude_command(self._exe, schema, model=s.model, effort=s.effort)
                else:
                    cmd = codex_command(self._exe, call, model=s.model, effort=s.effort)
                (call / "argv.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
                code, stdout, stderr = run_cli(cmd, stdin=prompt, cwd=call, timeout_s=s.timeout_s,
                                               cancel=cancel, name=s.cli)
                if s.cli == "claude":
                    raw = _claude_answer(code, stdout, stderr)
                else:
                    raw = _codex_answer(code, stdout, stderr, call / "last.txt")
            answer = check_answer(raw, schema)
        except PlannerError as e:
            logger.info("planner: %s failed after %.1f s: %s: %s", s.cli, time.monotonic() - t0,
                        type(e).__name__, e)
            raise
        logger.info("planner: %s answered in %.1f s", s.cli, time.monotonic() - t0)
        return answer


# Not blocking the tick ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerOutcome:
    """How one call ended: an answer or an error, and how long it took on the runner's clock [s]."""

    answer: dict[str, Any] | None
    error: PlannerError | None
    response_s: float


@dataclass(eq=False)
class _Ticket:
    """One call: only the current ticket's outcome is kept."""

    cancel: threading.Event


class PlannerRunner:
    """Runs a planner's `ask` on a thread of its own, one call at a time, for a tick that must not wait.

    `start` begins a call, `poll` returns its outcome once (None while it runs), `cancel` gives up on it:
    the planner sees its cancel event (a CLI is killed) and the late outcome is dropped. `threaded=False`
    runs the call inline in `start`, for tests; the outcome still waits for `poll`.
    """

    def __init__(self, planner: Planner, *, threaded: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        self.planner = planner
        self.threaded = threaded
        self._clock = clock
        self._lock = threading.Lock()
        self._ticket: _Ticket | None = None
        self._outcome: PlannerOutcome | None = None
        self._thread: threading.Thread | None = None

    @property
    def pending(self) -> bool:
        """Whether a call was started and its outcome not yet collected."""
        with self._lock:
            return self._ticket is not None

    def start(self, prompt: str, schema: Mapping[str, Any]) -> None:
        """Start a call; the schema is copied. Raises RuntimeError while another call is pending."""
        schema = copy.deepcopy(dict(schema))
        with self._lock:
            if self._ticket is not None:
                raise RuntimeError("planner: a call is already pending")
            ticket = self._ticket = _Ticket(threading.Event())
            self._outcome = None
        if not self.threaded:
            self._run(ticket, prompt, schema)
            return
        self._thread = threading.Thread(target=self._run, args=(ticket, prompt, schema), name="planner-ask",
                                        daemon=True)
        self._thread.start()

    def _run(self, ticket: _Ticket, prompt: str, schema: Mapping[str, Any]) -> None:
        t0 = self._clock()
        answer, error = None, None
        try:
            answer = self.planner.ask(prompt, schema, cancel=ticket.cancel)
        except PlannerError as e:
            error = e
        except Exception as e:  # noqa: BLE001 - a planner bug becomes the call's error
            logger.exception("planner: ask raised")
            error = PlannerError(f"planner raised {type(e).__name__}: {e}")
        outcome = PlannerOutcome(answer=answer, error=error, response_s=self._clock() - t0)
        with self._lock:
            if ticket is self._ticket:
                self._outcome = outcome

    def poll(self) -> PlannerOutcome | None:
        """The finished call's outcome, once; None while it runs or when nothing was started."""
        with self._lock:
            outcome, self._outcome = self._outcome, None
            if outcome is not None:
                self._ticket = None
        return outcome

    def cancel(self) -> None:
        """Give up on the pending call, if any: set its cancel event and drop its outcome. Never waits."""
        with self._lock:
            ticket, self._ticket = self._ticket, None
            self._outcome = None
        if ticket is not None:
            ticket.cancel.set()

    def wait(self, timeout_s: float | None = None) -> None:
        """Wait for the last call's thread, if any (tests and shutdown)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)


# Tests -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scripted:
    """One scripted call of a `FakePlanner`: an answer or an error, after a delay [s]."""

    answer: Mapping[str, Any] | None = None
    error: PlannerError | None = None
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        if (self.answer is None) == (self.error is None):
            raise ValueError("Scripted needs exactly one of answer and error")


#: One step of a `FakePlanner` script.
Step = Mapping[str, Any] | PlannerError | Scripted


class FakePlanner:
    """A `Planner` for tests: each `ask` plays the next scripted step, without a CLI.

    A step is an answer (a mapping), a PlannerError to raise, or a `Scripted` with its own delay; bare steps
    wait `delay_s`. A delay ends early with PlannerCancelled when the call is cancelled. Answers are checked
    against the schema as the CLI's are. `asked` holds each call's (prompt, schema).
    """

    def __init__(self, script: Iterable[Step] = (), *, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.asked: list[tuple[str, dict[str, Any]]] = []
        self._script: deque[Step] = deque(script)
        self._lock = threading.Lock()

    def add(self, *steps: Step) -> None:
        """Append steps to the script."""
        with self._lock:
            self._script.extend(steps)

    def ask(self, prompt: str, schema: Mapping[str, Any], *,
            cancel: threading.Event | None = None) -> dict[str, Any]:
        with self._lock:
            self.asked.append((prompt, copy.deepcopy(dict(schema))))
            step = self._script.popleft() if self._script else None
        if step is None:
            raise PlannerError("fake planner: no scripted step left")
        if not isinstance(step, Scripted):
            if isinstance(step, PlannerError):
                step = Scripted(error=step, delay_s=self.delay_s)
            else:
                step = Scripted(answer=step, delay_s=self.delay_s)
        cancel = cancel if cancel is not None else threading.Event()
        cancelled = cancel.wait(step.delay_s) if step.delay_s > 0 else cancel.is_set()
        if cancelled:
            raise PlannerCancelled("fake planner: cancelled")
        if step.error is not None:
            raise step.error
        return check_answer(copy.deepcopy(dict(step.answer or {})), schema)
