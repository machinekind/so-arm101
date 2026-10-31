"""The `soarm-lab` command: policies and evaluation on CPU, without torch, and the grasp libraries' files.

Training needs a GPU (rl/test_batch.py). Converting a torch checkpoint: rl/test_policy_format.py. The grasp
library's own tests: test_grasp_library.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("mujoco")

from soarm_lab.cli import main  # noqa: E402
from soarm_lab.grasp_library import GraspLibrary  # noqa: E402
from soarm_lab.paths import REPO_ROOT, source_checkout  # noqa: E402
from soarm_lab.rl.objects import KINDS  # noqa: E402
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
    assert "grasps" in out.stdout


def test_grasps_meshes_writes_one_obj_per_kind(tmp_path, capsys):
    assert main(["grasps", "meshes", "--out", str(tmp_path)]) == 0

    assert all((tmp_path / f"{kind}.obj").is_file() for kind in KINDS)
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / f"{kind}.obj") for kind in KINDS]
    assert main(["grasps", "meshes", "--out", str(tmp_path / "two"), "--kinds", "cube,cylinder"]) == 0
    assert sorted(p.name for p in (tmp_path / "two").iterdir()) == ["cube.obj", "cylinder.obj"]


def test_grasps_convert_writes_a_library_from_an_isaac_grasp_file(tmp_path, capsys):
    if not source_checkout():
        pytest.skip("needs a source checkout (tools/graspgenx)")
    import yaml

    down = {"w": 0.0, "xyz": [0.70710678, 0.70710678, 0.0]}
    source = tmp_path / "cube.yml"
    source.write_text(yaml.safe_dump({"format": "isaac_grasp", "format_version": 1.0, "grasps": {
        "grasp_0": {"confidence": 0.6, "position": [0.0, -0.0344, 0.0736], "orientation": down},
        "grasp_1": {"confidence": 0.8, "position": [0.0, -0.0344, 0.0786], "orientation": down},
        "grasp_2": {"confidence": 0.9, "position": [0.0, -0.0344, 0.1736], "orientation": down}}}))
    out = tmp_path / "lib" / "cube.json"
    config = REPO_ROOT / "tools/graspgenx/so101/config.json"

    assert main(["grasps", "convert", "cube", str(source), "--out", str(out), "--gripper", str(config),
                 "--generator", "test", "--commit", "abc"]) == 0
    library = GraspLibrary.from_dict(json.loads(out.read_text()))
    assert len(library.grasps) == 2 and [g.score for g in library.grasps] == [0.8, 0.6]
    assert library.generator["name"] == "test" and library.generator["commit"] == "abc"
    assert library.generator["version"] == "" and library.generator["dropped"] == "1 of 3"
    assert capsys.readouterr().out == (f"cube: 2 grasps -> {out} (1 of 3 dropped: pad point outside the "
                                       "block)\n")
    with pytest.raises(FileNotFoundError, match="gripper config not found: .*nope.json; pass --gripper"):
        main(["grasps", "convert", "cube", str(source), "--gripper", str(tmp_path / "nope.json")])
    with pytest.raises(FileNotFoundError, match="GraspGenX output not found"):
        main(["grasps", "convert", "cube", str(tmp_path / "nope.yml"), "--gripper", str(config)])
    with pytest.raises(SystemExit):
        main(["grasps", "convert", "ball", str(source)])


def test_grasps_show_lists_the_bundled_cube_library(capsys):
    assert main(["grasps", "show", "cube"]) == 0
    out = capsys.readouterr().out
    grasps = [line for line in out.splitlines() if line.startswith("#")]

    assert "  name: hand" in out and "  dropped: 0 of 8" in out
    assert len(grasps) == 8
    assert grasps[0] == ("#0 score 0.92 pad (0.0, 0.0, 0.0) mm approach (0.00, 0.00, -1.00) "
                         "closing (0.00, 1.00, 0.00) width 25.0 mm")
    assert grasps[4].startswith("#4 score 0.85 pad (5.0, 0.0, 8.5) mm")
