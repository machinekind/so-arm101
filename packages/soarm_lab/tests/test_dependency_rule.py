"""`soarm_lab` imports no twin (`machinekind_twin`), also not at runtime; it imports the twin's contracts,
`machinekind_twin_api`. Running a policy needs neither jax nor torch: jax is the trainer's extra
(`soarm-lab[train]`), torch the converter's (`soarm-lab[torch]`). A test that imports the twin is marked
`twin`; the others run without it, also one that imports the contracts."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import soarm_lab

#: Top-level modules the lab's source may not import.
FORBIDDEN = ("machinekind_twin",)
#: The twin's plugin contracts: allowed, and no twin.
CONTRACTS = "machinekind_twin_api"
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


def top(module: str) -> str:
    """The top-level package of a dotted module name:
    `machinekind_twin_api.strategy` -> `machinekind_twin_api`."""
    return module.split(".")[0]


def forbidden(modules: set[str]) -> set[str]:
    return {m for m in modules if top(m) in FORBIDDEN}


def test_the_contracts_are_not_the_twin() -> None:
    """The check compares whole package names: `machinekind_twin_api` starts with `machinekind_twin`
    and is allowed."""
    assert forbidden({"machinekind_twin", "machinekind_twin.strategy"}) == {
        "machinekind_twin", "machinekind_twin.strategy"}
    assert forbidden({CONTRACTS, f"{CONTRACTS}.strategy", "soarm"}) == set()
    assert f"{CONTRACTS}.strategy" in imports(ROOT / "strategies" / "policy.py")[0]


@pytest.mark.parametrize("path", sorted(ROOT.rglob("*.py")), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_forbidden_imports(path: Path) -> None:
    modules, levels = imports(path)
    assert forbidden(modules) == set()
    # A relative import may not reach outside `soarm_lab`.
    assert all(level <= len(path.relative_to(ROOT).parts) for level in levels)


def test_the_lab_imports_nothing_from_the_twin_at_runtime() -> None:
    """Importing every lab module (strategy, environments, CLI) loads no `machinekind_twin` module.

    It loads the contracts, `machinekind_twin_api`: modules count by their exact top-level name.
    """
    script = (
        "import importlib.util, sys\n"
        "import soarm_lab.cli, soarm_lab.sim, soarm_lab.strategies, soarm_lab.strategies.policy\n"
        "import soarm_lab.strategies.ik, soarm_lab.strategies.dream, soarm_lab.dream_run\n"
        "import soarm_lab.rl, soarm_lab.rl.env, soarm_lab.rl.evaluate\n"
        "if all(importlib.util.find_spec(m) for m in ('jax', 'optax', 'mujoco_warp')):\n"
        "    import soarm_lab.rl.ppo, soarm_lab.rl.jax_policy, soarm_lab.rl.batch\n"
        "if importlib.util.find_spec('torch'):\n"
        "    import soarm_lab.rl.torch_policy\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'machinekind_twin'))\n"
        "print('machinekind_twin_api.strategy' in sys.modules)\n"
        "print('dream' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    # The Dream client loads only when a dream run starts (`strategies.dream.open_client`).
    assert out.stdout.split() == ["[]", "True", "False"]


#: What a twin user, `soarm-lab eval` and `soarm-lab policies` load.
RUNTIME = ("soarm_lab", "soarm_lab.strategies", "soarm_lab.strategies.policy", "soarm_lab.strategies.ik",
           "soarm_lab.strategies.dream",
           "soarm_lab.rl.policy", "soarm_lab.station", "soarm_lab.sim", "soarm_lab.cli", "soarm_lab.rl",
           "soarm_lab.rl.env", "soarm_lab.rl.evaluate")
#: Imports the runtime, loads both bundled policies, runs a strategy tick, `policies` and `eval`.
RUN_POLICIES = f"""
import sys
from machinekind_twin_api.strategy import Observation, RunInfo
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
spec = load_robot()
run = RunInfo(robot="so101", joints=tuple(spec.joints), gripper=spec.gripper, limits={{}},
              max_vel=dict(spec.max_vel), folder=soarm.ROBOT_DIR, simulated=True)
obs = Observation(t=0.0, step=0, joints=dict(spec.home), command={{}})
s.start(run, obs)
assert s.act(obs).target
assert main(["policies"]) == 0
assert main(["eval", "reach-v3", "--episodes", "1"]) == 0
"""
#: The training and conversion stacks: running a policy loads none of them.
HEAVY = ("jax", "jaxlib", "optax", "torch")
#: Without jax the trainer names its extra, without torch the converter names its own.
EXTRA_COMMANDS = """
for argv, extra in ((["train", "--iters", "1"], "train"), (["convert", "policy.pt"], "torch")):
    try:
        main(argv)
    except ImportError as exc:
        assert f"install soarm-lab[{extra}]" in str(exc), exc
    else:
        raise AssertionError(f"soarm-lab {argv[0]} ran without soarm-lab[{extra}]")
"""


@pytest.mark.parametrize("missing", [False, True], ids=["extras-installed", "extras-missing"])
def test_running_a_policy_needs_no_jax_and_no_torch(tmp_path: Path, missing: bool) -> None:
    """The runtime loads no jax, optax or torch module, and works where they cannot be imported at all.

    `sys.modules["jax"] = None` makes `import jax` fail as if jax were not installed; the same for
    optax and torch.
    """
    script = RUN_POLICIES + (EXTRA_COMMANDS if missing else "")
    if missing:
        script = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in HEAVY) + script
    script += f"print('HEAVY', sorted(m for m, v in sys.modules.items() if v and m.split('.')[0] in {HEAVY}))"
    env = {**os.environ, "SOARM_LAB_POLICIES": str(tmp_path / "policies")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[-1] == "HEAVY []", out.stdout


def twin_imports(path: Path) -> list[str]:
    """The `machinekind_twin` modules `path` imports, by an import statement or `pytest.importorskip`."""
    modules = imports(path)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "importorskip"
                and node.args and isinstance(node.args[0], ast.Constant)):
            modules.add(str(node.args[0].value))
    return sorted(m for m in modules if top(m) == "machinekind_twin")


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
    # A test that imports only the contracts needs no marker.
    assert CONTRACTS in {top(m) for m in imports(TESTS / "strategies" / "test_policy.py")[0]}
    assert found[TESTS / "strategies" / "test_policy.py"] == []
    unmarked = [path.relative_to(TESTS).as_posix() for path, modules in found.items()
                if modules and not marked_twin(path)]
    assert unmarked == []


#: pytest in a fresh interpreter where `import machinekind_twin` fails as if the twin were not installed;
#: `import machinekind_twin_api` works.
WITHOUT_TWIN = ("import sys; sys.modules['machinekind_twin'] = None\n"
                "import pytest; sys.exit(pytest.main(sys.argv[1:]))")


def test_the_tests_not_marked_twin_collect_without_the_twin(pytestconfig: pytest.Config) -> None:
    """`pytest -m "not twin"` where the twin cannot be imported: no test module fails to import."""
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", "-m", "not twin", str(TESTS)]
    out = subprocess.run([sys.executable, "-c", WITHOUT_TWIN, *args], capture_output=True, text=True,
                         cwd=pytestconfig.rootpath)
    assert out.returncode == 0, out.stdout + out.stderr
