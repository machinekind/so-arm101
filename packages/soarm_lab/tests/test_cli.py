"""The `soarm-lab` command: policies and evaluation on CPU, without torch.

Training needs a GPU (rl/test_batch.py). Converting a torch checkpoint: rl/test_policy_format.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

pytest.importorskip("mujoco")

from soarm_lab.cli import main  # noqa: E402
from soarm_lab.station import STATIONS_DIR  # noqa: E402


@pytest.fixture(autouse=True)
def _own_policies_dir(tmp_path, monkeypatch):
    """Policies trained on this machine (workspace/policies) stay out of the tests."""
    monkeypatch.setenv("SOARM_LAB_POLICIES", str(tmp_path / "policies"))


def test_policies_lists_the_bundled_ones(capsys):
    assert main(["policies"]) == 0
    out = capsys.readouterr().out
    assert "reach-v3" in out and "lift-v3" in out
    assert out.count("bazowa") == 2


def test_eval_runs_a_bundled_policy_by_name(capsys):
    assert main(["eval", "reach-v3", "--episodes", "2"]) == 0
    out = capsys.readouterr().out
    assert "reach-v3/policy.npz" in out and "'success': 1.0" in out


def test_eval_on_an_exported_station_folder(tmp_path, capsys):
    folder = tmp_path / "desk"
    shutil.copytree(STATIONS_DIR / "sim", folder)
    assert main(["eval", "lift-v3", "--episodes", "1", "--station", str(folder)]) == 0
    assert "'episodes': 1" in capsys.readouterr().out


def _blocks_policy(tmp_path, evals=None) -> str:
    """Save a policy with two variants and zero weights in SOARM_LAB_POLICIES."""
    from dataclasses import asdict

    from soarm_lab.rl.policy import Policy, PolicyMeta
    from soarm_lab.rl.tasks import make_task

    task = make_task("lift", variants=("cube", "cylinder"), block="cube")
    meta = PolicyMeta(asdict(task), [8], task.obs_dim, task.act_dim, evals=evals or {}, created="2030-01-01")
    Policy(meta).save(tmp_path / "policies" / "two-blocks" / "policy.npz")
    return "two-blocks"


def test_policies_shows_the_mean_and_the_worst_variant(tmp_path, capsys):
    evals = {"cpu_rand_cube": {"success": 0.8}, "cpu_rand_cylinder": {"success": 0.4},
             "cpu_cube": {"success": 1.0}, "cpu_cylinder": {"success": 1.0}}
    _blocks_policy(tmp_path, evals)
    assert main(["policies"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "two-blocks" in x)
    assert "warianty z rand. srednio 60.0%, najslabszy cylinder 40.0%" in line


def test_policies_falls_back_to_the_nominal_variants(tmp_path, capsys):
    _blocks_policy(tmp_path, {"cpu_cube": {"success": 1.0}, "cpu_cylinder": {"success": 0.5}})
    assert main(["policies"]) == 0
    out = capsys.readouterr().out
    assert "warianty bez rand. srednio 75.0%, najslabszy cylinder 50.0%" in out
    assert out.count("warianty") == 1                                 # Bundled policies have no variants.


def test_eval_on_one_variant(tmp_path, capsys):
    name = _blocks_policy(tmp_path)
    assert main(["eval", name, "--episodes", "1", "--variant", "cylinder"]) == 0
    out = capsys.readouterr().out
    assert "policy.npz (cylinder): " in out and "'variant': 'cylinder'" in out
    assert main(["eval", name, "--episodes", "1"]) == 0
    assert "'variant'" not in capsys.readouterr().out


def test_eval_of_a_name_that_is_no_variant_is_refused(tmp_path):
    name = _blocks_policy(tmp_path)
    with pytest.raises(ValueError, match="'cuboid' is not a variant"):
        main(["eval", name, "--episodes", "1", "--variant", "cuboid"])
    with pytest.raises(ValueError, match="reach has no variants"):
        main(["eval", "reach-v3", "--episodes", "1", "--variant", "cube"])


def test_a_policy_without_variants_evaluates_on_any_block(capsys):
    assert main(["eval", "lift-v3", "--episodes", "1", "--variant", "cuboid"]) == 0
    assert "'variant': 'cuboid'" in capsys.readouterr().out
    with pytest.raises(ValueError, match="unknown block"):
        main(["eval", "lift-v3", "--episodes", "1", "--variant", "mug"])


def test_an_unknown_policy_names_the_known_ones():
    with pytest.raises(FileNotFoundError, match="znane: .*reach-v3"):
        main(["eval", "reach-v9"])


def test_a_torch_checkpoint_says_how_to_convert_it(tmp_path, capsys):
    """A policy trained before policy.npz: `eval` and `policies` name `soarm-lab convert`."""
    folder = tmp_path / "policies" / "reach-old"                    # SOARM_LAB_POLICIES, see the fixture
    folder.mkdir(parents=True)
    (folder / "policy.pt").write_bytes(b"not read")
    with pytest.raises(ValueError, match="soarm-lab convert .*reach-old/policy.pt"):
        main(["eval", "reach-old"])
    assert main(["policies"]) == 0
    assert "soarm-lab convert " in capsys.readouterr().out


def test_the_console_script_answers():
    out = subprocess.run([sys.executable, "-m", "soarm_lab.cli", "--help"], capture_output=True, text=True,
                         check=True)
    assert "train" in out.stdout and "eval" in out.stdout and "policies" in out.stdout
