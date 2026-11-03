"""A stand-in for the claude and codex CLIs: the tests copy it into a bin folder under both names, with a
shebang naming the test's own Python.

It answers as the CLI it is named after and logs what it got, so a test sees what the real CLI would have
seen. No model, no network.

Environment:
    FAKE_LOG     a file; one JSON line per run: argv, stdin, cwd, env_keys, schema, pid
    FAKE_ANSWER  the answer as JSON text (default: {})
    FAKE_MODE    ok | sleep | child | garbage | no_structured | exit1 | limit | logged_out | expired (claude)
                 | turn_failed | retry_notice (codex)
    FAKE_AUTH    subscription | apikey | logged_out
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NAME = Path(sys.argv[0]).name
ARGS = sys.argv[1:]
MODE = os.environ.get("FAKE_MODE", "ok")
AUTH = os.environ.get("FAKE_AUTH", "subscription")


def log(entry: dict) -> None:
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def flag(name: str) -> str | None:
    return ARGS[ARGS.index(name) + 1] if name in ARGS else None


def answer_text() -> str:
    return os.environ.get("FAKE_ANSWER", "{}")


def auth_status() -> int:
    log({"auth": NAME})
    if NAME == "claude":
        status = {
            "subscription": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
            "apikey": {"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"},
            "logged_out": {"loggedIn": False, "authMethod": "none"},
        }[AUTH]
        print(json.dumps(status, indent=2))
        return 1 if AUTH == "logged_out" else 0
    text = {"subscription": "Logged in using ChatGPT",
            "apikey": "Logged in using an API key - sk-proj-***ABCD",
            "logged_out": "Not logged in"}[AUTH]
    print(text, file=sys.stderr)  # codex prints its status on stderr
    return 1 if AUTH == "logged_out" else 0


def act_out() -> None:
    """The modes that never answer: a crash, a hang, a hang with a grandchild."""
    if MODE == "exit1":
        print("boom: something broke", file=sys.stderr)
        sys.exit(1)
    if MODE == "child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        log({"grandchild": child.pid})
    if MODE in ("sleep", "child"):
        log({"sleeping": os.getpid()})
        time.sleep(60)


def claude(stdin: str) -> int:
    schema = flag("--json-schema")
    log({"argv": sys.argv, "stdin": stdin, "cwd": os.getcwd(), "env_keys": sorted(os.environ),
         "schema": json.loads(schema) if schema else None, "pid": os.getpid()})
    act_out()
    if MODE == "garbage":
        print("this is not json at all")
        return 0
    result: dict = {"type": "result", "subtype": "success", "is_error": False, "result": "done",
                    "structured_output": json.loads(answer_text())}
    if MODE == "no_structured":
        del result["structured_output"]
    if MODE == "limit":
        result.update(is_error=True, result="Claude AI usage limit reached|1792718400")
    if MODE == "logged_out":
        result.update(is_error=True, result="Not logged in - Please run /login")
    if MODE == "expired":
        result.update(is_error=True, result="OAuth token has expired. Please run /login")
    print(json.dumps(result))
    return 1 if MODE in ("limit", "logged_out", "expired") else 0


def codex(stdin: str) -> int:
    path = flag("--output-schema")
    schema = json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else None
    log({"argv": sys.argv, "stdin": stdin, "cwd": os.getcwd(), "env_keys": sorted(os.environ),
         "schema": schema, "pid": os.getpid()})
    act_out()
    print(json.dumps({"type": "thread.started", "thread_id": "fake"}))
    print(json.dumps({"type": "turn.started"}))
    failures = {"turn_failed": "the model refused", "limit": "You've hit your usage limit.",
                "logged_out": "Not logged in. Please run codex login"}
    if MODE in failures:
        print(json.dumps({"type": "error", "message": failures[MODE]}))
        print(json.dumps({"type": "turn.failed", "error": {"message": failures[MODE]}}))
        return 1
    if MODE == "retry_notice":
        print(json.dumps({"type": "error", "message": "Reconnecting... 1/5"}))
    text = {"garbage": "this is not json", "no_structured": ""}.get(MODE, answer_text())
    item = {"id": "item_0", "type": "agent_message", "text": text}
    print(json.dumps({"type": "item.completed", "item": item}))
    print(json.dumps({"type": "turn.completed", "usage": {}}))
    last = flag("-o")
    if last:
        Path(last).write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    if ARGS[:2] in (["auth", "status"], ["login", "status"]):
        return auth_status()
    stdin = sys.stdin.read()
    return claude(stdin) if NAME == "claude" else codex(stdin)


if __name__ == "__main__":
    sys.exit(main())
