"""The `soarm-lab` command: policies and evaluation on CPU, without torch.

Training needs a GPU (test_lab_batch.py). Converting a torch checkpoint: test_lab_policy_format.py.
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
