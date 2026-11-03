"""One real question to a real model through the installed CLI, on your subscription.

Skipped unless `SOARM_PLANNER_LIVE=1`. Optional: `SOARM_PLANNER_CLI` (codex or claude, default codex),
`SOARM_PLANNER_MODEL` (default gpt-6-astra for codex, the CLI's own default for claude) and
`SOARM_PLANNER_EXECUTABLE` (e.g. ~/.local/bin/codex, to bypass terminal shims).

    SOARM_PLANNER_LIVE=1 uv run pytest -q packages/soarm_lab/tests/planner/test_planner_live.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from soarm_lab.planner.cli import CliPlanner, CliSettings

pytestmark = pytest.mark.skipif(os.environ.get("SOARM_PLANNER_LIVE") != "1",
                                reason="set SOARM_PLANNER_LIVE=1 to ask a real model")

COLOURS = ["red", "green", "blue"]
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["bottom_to_top", "count"],
    "properties": {"bottom_to_top": {"type": "array", "items": {"type": "string", "enum": COLOURS}},
                   "count": {"type": "integer"}}}


def test_a_real_model_answers_in_the_schema():
    cli = os.environ.get("SOARM_PLANNER_CLI", "codex")
    model = os.environ.get("SOARM_PLANNER_MODEL", "gpt-6-astra" if cli == "codex" else "")
    executable = os.environ.get("SOARM_PLANNER_EXECUTABLE", "")
    planner = CliPlanner(CliSettings(cli=cli, model=model, executable=executable, timeout_s=300.0))

    answer = planner.ask("Three blocks lie on a table: red, green and blue. Stack them into one tower with "
                         "blue at the bottom and red on top. List the colours from bottom to top and give "
                         "the number of blocks.", SCHEMA)

    assert answer == {"bottom_to_top": ["blue", "green", "red"], "count": 3}
