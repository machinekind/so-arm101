"""The `soarm-lab` command: policies and evaluation on CPU, without torch.

Training on the GPU: rl/test_batch.py; here the training flags, and a slow CPU smoke run of the trainer
(marked `slow`, it needs the `train` extra). Converting a torch checkpoint: rl/test_policy_format.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest

pytest.importorskip("mujoco")

from soarm_lab.cli import _describe, _perception_flags, _train_plan, main, parser  # noqa: E402
from soarm_lab.rl.randomize import Dynamics  # noqa: E402
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


def test_the_train_flags_shape_the_perception_and_the_jaw():
    a = parser().parse_args(["train", "--pose-bias-m", "0.02", "--pose-delay", "4,20", "--pose-period", "4",
                             "--latch-m", "0.05", "--grip-close-s", "1.3"])
    assert a.pose_delay == (4, 20) and a.pose_period == 4 and a.grip_close_s == 1.3
    rand, evals = _train_plan(replace(Dynamics(), grip_close_s=1.3), False, 1.0, _perception_flags(a))
    assert (rand.pose_bias_m, rand.cube_min_delay, rand.cube_delay, rand.cube_period, rand.latch_m) == \
        (0.02, 4, 20, 4, 0.05)
    assert rand.grip_close == (0.8, 1.25)
    plan = {key: r for _, key, r in evals}
    assert plan["cpu_rand"] is rand                                    # evaluated with the same perception
    nominal = plan["cpu_nominal"]                                      # the ideal one, the measured jaw
    assert (nominal.pose_bias_m, nominal.cube_period, nominal.centre.grip_close_s) == (0.0, 1, 1.3)
    assert parser().parse_args(["train", "--pose-delay", "7"]).pose_delay == (0, 7)
    assert _perception_flags(parser().parse_args(["train"])) == {}     # nothing given: the randomization's
    for bad in (["--pose-delay", "5,3"], ["--pose-delay", "-1"], ["--pose-period", "0"]):
        with pytest.raises(SystemExit):
            parser().parse_args(["train", *bad])
    text = _describe(rand)
    assert "zamykanie chwytaka 1.30 s" in text and "odswiezanie co 1-4 takt." in text
    assert "przesuniecie do 2 cm" in text and "opoznienie pozy 4-20 takt." in text and "< 5 cm" in text


@pytest.mark.slow
def test_a_cpu_smoke_run_of_the_trainer_from_lift_v3(tmp_path, capsys):
    """`train --init lift-v3` with every perception flag, 2 iterations on 4 Warp worlds on the CPU."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")
    from soarm_lab.rl.policy import Policy

    argv = ["train", "--init", "lift-v3", "--kinds", "cube", "--device", "cpu", "--envs", "4", "--iters", "2",
            "--eval-episodes", "1", "--pose-bias-m", "0.02", "--pose-delay", "2,6", "--pose-period", "4",
            "--latch-m", "0.05", "--grip-close-s", "1.3", "--out", str(tmp_path), "--name", "smoke"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Trening lift (klocki: cube)" in out
    assert "zamykanie chwytaka 1.30 s" in out and "odswiezanie co 1-4 takt." in out
    pol = Policy.load(tmp_path / "smoke" / "policy.npz")
    assert pol.meta.obs_dim == 40 and pol.task.shape_conditioned
    r = pol.meta.randomization
    assert (r["pose_bias_m"], r["cube_min_delay"], r["cube_delay"], r["cube_period"], r["latch_m"]) == \
        (0.02, 2, 6, 4, 0.05)
    assert r["grip_close"] == [0.8, 1.25] and r["centre"]["grip_close_s"] == 1.3
    assert set(pol.meta.evals) >= {"gpu", "cpu_nominal", "cpu_rand", "cpu_cube", "cpu_rand_cube"}
    config = json.loads((tmp_path / "smoke" / "config.json").read_text(encoding="utf-8"))
    assert config["kinds"] == ["cube"]
