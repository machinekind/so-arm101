"""`soarm` imports neither the twin (`lerobot_mp`) nor `soarm_lab`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import soarm

FORBIDDEN = ("lerobot_mp", "soarm_lab")
SOURCES = sorted(Path(soarm.__file__).parent.rglob("*.py"))


def imports(path: Path) -> tuple[set[str], list[int]]:
    """Absolute modules imported by `path` and the levels of its relative imports."""
    modules, levels = set(), []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
            elif node.level:
                levels.append(node.level)
    return modules, levels


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_soarm_does_not_import_the_twin_or_the_lab(path: Path) -> None:
    modules, levels = imports(path)
    assert {m for m in modules if m.split(".")[0] in FORBIDDEN} == set()
    # `from ..base import ...` (the twin's layout) would reach outside the package.
    assert all(level == 1 for level in levels)
