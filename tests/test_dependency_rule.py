"""`soarm` imports neither the twin (`lerobot_mp`) nor `soarm_lab`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import SOARM_ROOT

FORBIDDEN = ("lerobot_mp", "soarm_lab")


def imported_modules(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", sorted((SOARM_ROOT / "soarm").rglob("*.py")), ids=lambda p: p.name)
def test_soarm_does_not_import_the_twin_or_the_lab(path: Path) -> None:
    bad = {m for m in imported_modules(path) if m.split(".")[0] in FORBIDDEN}
    assert not bad
