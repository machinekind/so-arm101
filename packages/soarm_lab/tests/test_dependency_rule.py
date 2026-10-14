"""`soarm_lab` imports no twin (`lerobot_mp`), also not at runtime, and running a policy needs no torch
(`soarm-lab[torch]` is the legacy trainer's extra). A test that imports the twin is marked `twin`; the
others run without it."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import soarm_lab

#: Top-level modules the lab's source may not import.
FORBIDDEN = ("lerobot_mp",)
ROOT = Path(soarm_lab.__file__).parent
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


@pytest.mark.parametrize("path", sorted(ROOT.rglob("*.py")), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_forbidden_imports(path: Path) -> None:
    modules, levels = imports(path)
    assert {m for m in modules if m.split(".")[0] in FORBIDDEN} == set()
    # A relative import may not reach above its own package (`from ..base import ...`: the twin's layout).
    assert all(level <= len(path.relative_to(ROOT).parts) for level in levels)


def test_the_lab_imports_nothing_from_the_twin_at_runtime() -> None:
    """Importing every lab module (strategy, environments, CLI) loads no `lerobot_mp` module."""
    script = (
        "import importlib.util, sys\n"
        "import soarm_lab.cli, soarm_lab.sim, soarm_lab.strategies, soarm_lab.strategies.policy\n"
        "import soarm_lab.rl, soarm_lab.rl.env, soarm_lab.rl.evaluate\n"
        "if importlib.util.find_spec('torch'):\n"
        "    import soarm_lab.rl.ppo, soarm_lab.rl.torch_policy\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot_mp'))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


#: What a twin user, `soarm-lab eval` and `soarm-lab policies` load.
RUNTIME = ("soarm_lab", "soarm_lab.strategies", "soarm_lab.strategies.policy", "soarm_lab.rl.policy",
           "soarm_lab.station", "soarm_lab.sim", "soarm_lab.cli", "soarm_lab.rl", "soarm_lab.rl.env",
           "soarm_lab.rl.evaluate")
#: Imports the runtime, loads both bundled policies, runs a strategy tick, `policies` and `eval`.
RUN_POLICIES = f"""
import sys
from types import SimpleNamespace
import {", ".join(RUNTIME)}
import soarm
from soarm_lab.cli import main
from soarm_lab.robot import load_robot
from soarm_lab.rl.policy import Policy, find_policy
from soarm_lab.strategies.policy import PolicySettings, PolicyStrategy

for name in ("reach-v3", "lift-v3"):
    pol = Policy.load(find_policy(name))
    assert pol.act([0.0] * pol.meta.obs_dim).shape == (pol.meta.act_dim,)
s = PolicyStrategy(PolicySettings(policy="reach-v3"))
run = SimpleNamespace(robot="so101", folder=soarm.ROBOT_DIR, limits={{}}, simulated=True)
obs = SimpleNamespace(joints=dict(load_robot().home), command={{}}, goal=None, objects={{}}, t=0.0)
s.start(run, obs)
assert s.act(obs).target
assert main(["policies"]) == 0
assert main(["eval", "reach-v3", "--episodes", "1"]) == 0
"""
#: Without torch, the trainer and the converter name the extra.
TORCH_COMMANDS = """
for argv in (["train", "--iters", "1"], ["convert", "policy.pt"]):
    try:
        main(argv)
    except ImportError as exc:
        assert "install soarm-lab[torch]" in str(exc), exc
    else:
        raise AssertionError(f"soarm-lab {argv[0]} ran without torch")
"""


@pytest.mark.parametrize("torch_missing", [False, True], ids=["torch-installed", "torch-missing"])
def test_running_a_policy_needs_no_torch(tmp_path: Path, torch_missing: bool) -> None:
    """The runtime loads no torch module, and works where torch cannot be imported at all.

    `sys.modules["torch"] = None` makes `import torch` fail as if torch were not installed.
    """
    script = RUN_POLICIES + (TORCH_COMMANDS if torch_missing else "")
    if torch_missing:
        script = 'import sys\nsys.modules["torch"] = None\n' + script
    script += "print('TORCH', sorted(m for m, v in sys.modules.items() if v and m.split('.')[0] == 'torch'))"
    env = {**os.environ, "SOARM_LAB_POLICIES": str(tmp_path / "policies")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "TORCH []", out.stdout


def twin_imports(path: Path) -> list[str]:
    """The `lerobot_mp` modules `path` imports, by an import statement or `pytest.importorskip`."""
    modules = imports(path)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "importorskip"
                and node.args and isinstance(node.args[0], ast.Constant)):
            modules.add(str(node.args[0].value))
    return sorted(m for m in modules if m.split(".")[0] == "lerobot_mp")


def marked_twin(path: Path) -> bool:
    """The module's `pytestmark` holds `pytest.mark.twin`."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "pytestmark" for t in node.targets):
            marks = ast.walk(node.value)
            return any(isinstance(mark, ast.Attribute) and mark.attr == "twin" for mark in marks)
    return False


def test_a_test_that_imports_the_twin_is_marked_twin() -> None:
    assert twin_imports(TESTS / "conftest.py")                # the check sees an import inside a function
    found = {path: twin_imports(path) for path in sorted(TESTS.rglob("test_*.py"))}
    unmarked = [path.relative_to(TESTS).as_posix() for path, modules in found.items()
                if modules and not marked_twin(path)]
    assert unmarked == []


#: pytest in a fresh interpreter where `import lerobot_mp` fails as if the twin were not installed.
WITHOUT_TWIN = ("import sys; sys.modules['lerobot_mp'] = None\n"
                "import pytest; sys.exit(pytest.main(sys.argv[1:]))")


def test_the_tests_not_marked_twin_collect_without_the_twin(pytestconfig: pytest.Config) -> None:
    """`pytest -m "not twin"` where the twin cannot be imported: no test module fails to import."""
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", "-m", "not twin", str(TESTS)]
    out = subprocess.run([sys.executable, "-c", WITHOUT_TWIN, *args], capture_output=True, text=True,
                         cwd=pytestconfig.rootpath)
    assert out.returncode == 0, out.stdout + out.stderr
