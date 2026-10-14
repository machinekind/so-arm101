"""`soarm` imports neither the twin (`lerobot_mp`) nor `soarm_lab`; it imports the twin's contracts,
`lerobot_mp_api`. A test that imports the twin is marked `twin`; the others run without it, also one that
imports the contracts."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import soarm

#: Top-level modules soarm's source may not import.
FORBIDDEN = ("lerobot_mp", "soarm_lab")
#: The twin's plugin contracts: allowed, and no twin.
CONTRACTS = "lerobot_mp_api"
ROOT = Path(soarm.__file__).parent
TESTS = Path(__file__).parent


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


def top(module: str) -> str:
    """The top-level package of a dotted module name: `lerobot_mp_api.backend` -> `lerobot_mp_api`."""
    return module.split(".")[0]


def forbidden(modules: set[str]) -> set[str]:
    return {m for m in modules if top(m) in FORBIDDEN}


def test_the_contracts_are_not_the_twin() -> None:
    """The check compares whole package names: `lerobot_mp_api` starts with `lerobot_mp` and is allowed."""
    assert forbidden({"lerobot_mp", "lerobot_mp.robot.base", "soarm_lab.cli"}) == {
        "lerobot_mp", "lerobot_mp.robot.base", "soarm_lab.cli"}
    assert forbidden({CONTRACTS, f"{CONTRACTS}.backend", "soarm.feetech"}) == set()
    assert f"{CONTRACTS}.backend" in imports(ROOT / "feetech.py")[0]


@pytest.mark.parametrize("path", sorted(ROOT.rglob("*.py")), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_forbidden_imports(path: Path) -> None:
    modules, levels = imports(path)
    assert forbidden(modules) == set()
    # A relative import may not reach above its own package (`from ..base import ...`: the twin's layout).
    assert all(level <= len(path.relative_to(ROOT).parts) for level in levels)


def twin_imports(path: Path) -> list[str]:
    """The `lerobot_mp` modules `path` imports, by an import statement or `pytest.importorskip`."""
    modules = imports(path)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "importorskip"
                and node.args and isinstance(node.args[0], ast.Constant)):
            modules.add(str(node.args[0].value))
    return sorted(m for m in modules if top(m) == "lerobot_mp")


def marked_twin(path: Path) -> bool:
    """The module's `pytestmark` holds `pytest.mark.twin`."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "pytestmark" for t in node.targets):
            marks = ast.walk(node.value)
            return any(isinstance(mark, ast.Attribute) and mark.attr == "twin" for mark in marks)
    return False


def test_a_test_that_imports_the_twin_is_marked_twin() -> None:
    found = {path: twin_imports(path) for path in sorted(TESTS.rglob("test_*.py"))}
    assert any(found.values())                                # the check sees the twin's imports
    # A test that imports only the contracts needs no marker.
    assert CONTRACTS in {top(m) for m in imports(TESTS / "test_entry_points.py")[0]}
    assert found[TESTS / "test_entry_points.py"] == []
    unmarked = [path.relative_to(TESTS).as_posix() for path, modules in found.items()
                if modules and not marked_twin(path)]
    assert unmarked == []


#: pytest in a fresh interpreter where `import lerobot_mp` fails as if the twin were not installed;
#: `import lerobot_mp_api` works.
WITHOUT_TWIN = ("import sys; sys.modules['lerobot_mp'] = None\n"
                "import pytest; sys.exit(pytest.main(sys.argv[1:]))")


def test_the_tests_not_marked_twin_collect_without_the_twin(pytestconfig: pytest.Config) -> None:
    """`pytest -m "not twin"` where the twin cannot be imported: no test module fails to import."""
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", "-m", "not twin", str(TESTS)]
    out = subprocess.run([sys.executable, "-c", WITHOUT_TWIN, *args], capture_output=True, text=True,
                         cwd=pytestconfig.rootpath)
    assert out.returncode == 0, out.stdout + out.stderr
