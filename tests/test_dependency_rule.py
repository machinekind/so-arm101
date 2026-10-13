"""Neither package imports the twin (`lerobot_mp`); `soarm` never imports `soarm_lab`."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import soarm
import soarm_lab

#: Package -> top-level modules its source may not import.
FORBIDDEN = {
    "soarm": ("lerobot_mp", "soarm_lab"),
    "soarm_lab": ("lerobot_mp",),
}
ROOTS = {"soarm": Path(soarm.__file__).parent, "soarm_lab": Path(soarm_lab.__file__).parent}
SOURCES = [(package, path) for package, root in ROOTS.items() for path in sorted(root.rglob("*.py"))]


def imports(path: Path) -> tuple[set[str], list[int]]:
    """Absolute modules imported by `path` and the levels of its relative imports.

    Every import counts, also one under `if TYPE_CHECKING:` or inside a function.
    """
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


def depth(path: Path, root: Path) -> int:
    """How many packages deep `path` lies below the package root (`rl/env.py` -> 1)."""
    return len(path.relative_to(root).parts) - 1


@pytest.mark.parametrize("package, path", SOURCES, ids=lambda p: p if isinstance(p, str) else p.name)
def test_no_forbidden_imports(package: str, path: Path) -> None:
    modules, levels = imports(path)
    assert {m for m in modules if m.split(".")[0] in FORBIDDEN[package]} == set()
    # A relative import may not reach above its own package (`from ..base import ...`: the twin's layout).
    assert all(level <= 1 + depth(path, ROOTS[package]) for level in levels)


def test_the_lab_imports_nothing_from_the_twin_at_runtime() -> None:
    """Importing every lab module (strategy, environments, CLI) loads no `lerobot_mp` module."""
    script = (
        "import sys\n"
        "import soarm_lab.cli, soarm_lab.sim, soarm_lab.strategy\n"
        "import soarm_lab.rl, soarm_lab.rl.env, soarm_lab.rl.evaluate, soarm_lab.rl.ppo\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot_mp'))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
