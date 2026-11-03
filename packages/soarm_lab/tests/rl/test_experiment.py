"""Training composition, stored tasks and Hydra's command line behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

pytest.importorskip("hydra")
pytest.importorskip("jax")
pytest.importorskip("optax")

from hydra import compose, initialize_config_module  # noqa: E402
from hydra.errors import ConfigCompositionException  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from soarm_lab.rl import tasks  # noqa: E402
from soarm_lab.rl.experiment import composed_task, randomization_plan, training_task  # noqa: E402
from soarm_lab.rl.policy import Policy, find_policy  # noqa: E402
from soarm_lab.rl.randomize import describe  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402


def config(*overrides):
    with initialize_config_module(version_base="1.3", config_module="soarm_lab.conf"):
        return compose(config_name="train", overrides=list(overrides))


def options(*overrides):
    return OmegaConf.to_object(config(*overrides).randomization)


@pytest.mark.parametrize("name", tasks.TASKS)
@pytest.mark.parametrize("custom_station", [False, True])
def test_composed_task_matches_the_station_factory(name, custom_station):
    spec = load_robot()
    if custom_station:
        spec = replace(spec, joints=spec.joints + ("extra",),
                       desk=replace(spec.desk, goal_radius=(0.2, 0.4), goal_height=(0.1, 0.3),
                                    cube_radius=(0.25, 0.35), cube_bearing=(-0.5, 0.6)))
    cfg = config(f"task={name}")
    task = composed_task(cfg, spec)
    assert task == tasks.make_task(name, spec=spec)
    for value in asdict(task).values():
        assert not isinstance(value, list)
    assert type(task) is type(instantiate(cfg.task, _convert_="object"))


def test_task_and_ppo_tuple_overrides_are_tuples():
    cfg = config("task=pickplace", "task.target_radius=[0.18,0.24]", "ppo.hidden=[32,16]")
    task = composed_task(cfg, load_robot())
    assert task.target_radius == (0.18, 0.24)
    assert task.n_joints == len(load_robot().joints)
    ppo = instantiate(cfg.ppo, _convert_="object")
    assert tuple(ppo.hidden) == (32, 16)


@pytest.mark.parametrize("override", ["ppo.iteratons=1", "task.cube_hlaf=0.02", "randomization.spead=0",
                                      "randomization.pose_bias=0.02", "eval_episdodes=1"])
def test_unknown_keys_are_rejected(override):
    with pytest.raises(ConfigCompositionException):
        config("task=lift", override)


@pytest.mark.parametrize("name,iterations", [("reach", 150), ("lift", 300), ("pickplace", 300)])
def test_standard_experiment_sizes(name, iterations):
    cfg = config(f"experiment={name}")
    assert cfg.task.name == name
    assert cfg.name == name
    assert cfg.ppo.iterations == iterations


def test_fine_tuning_uses_the_stored_task_and_its_overrides():
    cfg = config("init=lift-v3", "ppo=fine_tune", "+task_overrides.episode_steps=250")
    init, task = training_task(cfg, load_robot())
    assert init is not None
    assert task == replace(Policy.load(find_policy("lift-v3")).task, episode_steps=250)
    with pytest.raises(ValueError, match="differs"):
        training_task(cfg, load_robot(), task_selected=True)
    cfg = config("experiment=lift_finetune")
    init, task = training_task(cfg, load_robot(), task_selected=True)
    assert task == init.task
    assert cfg.ppo.init_std == 0.25
    assert cfg.ppo.critic_warmup == 30


def test_the_lift_blocks_experiment_composes():
    cfg = config("experiment=lift_blocks")
    assert cfg.name == "lift_blocks" and cfg.init == "lift-v3"
    assert cfg.ppo.iterations == 300 and cfg.ppo.widen_init is True
    assert cfg.ppo.init_std == 0.25 and cfg.ppo.critic_warmup == 30
    init, task = training_task(cfg, load_robot(), task_selected=True)
    assert init is not None and task.name == "lift"
    variants = ("cube", "cuboid", "cylinder")
    assert (task.variants, task.block) == (variants, "cube")
    assert task == replace(init.task, variants=variants, block="cube")
    assert task.obs_dim == init.meta.obs_dim + len(variants) + 3


def test_the_late_pose_experiment_extends_the_block_training():
    cfg, blocks = config("experiment=lift_late_pose"), config("experiment=lift_blocks")
    assert cfg.name == "lift_late_pose" and cfg.init == blocks.init == "lift-v3"
    assert cfg.ppo == blocks.ppo and cfg.task_overrides == blocks.task_overrides
    assert cfg.task == blocks.task
    assert cfg.ppo.iterations == 300 and cfg.ppo.widen_init is True and cfg.ppo.init_std == 0.25
    assert config("experiment=lift_late_pose", "station=stations/desk1").station == "stations/desk1"
    rand, evals = randomization_plan(OmegaConf.to_object(cfg.randomization), {})
    assert (rand.pose_bias_m, rand.cube_min_delay, rand.cube_delay, rand.cube_period, rand.latch_m) == \
        (0.03, 4, 40, 4, 0.05)
    assert rand.centre.grip_close_s == 1.3 and rand.grip_close == (0.8, 1.25)
    assert [key for _, key, _ in evals] == ["cpu_nominal", "cpu_rand"]


def test_the_randomization_keys_are_off_by_default():
    assert options().changes() == {}
    assert options("experiment=lift_blocks").changes() == {}
    rand, _ = randomization_plan(options(), {})
    assert rand.centre.grip_close_s == 0.0 and rand.grip_close == (1.0, 1.0)
    assert (rand.pose_bias_m, rand.cube_min_delay, rand.latch_m) == (0.0, 0, 0.0)


def test_the_randomization_keys_shape_the_perception_and_the_jaw():
    chosen = options("randomization.pose_bias_m=0.02", "randomization.pose_delay=[4,20]",
                     "randomization.pose_period=4", "randomization.latch_m=0.05",
                     "randomization.grip_close_s=1.3")
    assert chosen.pose_delay == [4, 20] and chosen.pose_period == 4 and chosen.grip_close_s == 1.3
    assert chosen.changes() == {"pose_bias_m": 0.02, "cube_min_delay": 4, "cube_delay": 20,
                                "cube_period": 4, "latch_m": 0.05}
    rand, evals = randomization_plan(chosen, {})
    assert (rand.pose_bias_m, rand.cube_min_delay, rand.cube_delay, rand.cube_period, rand.latch_m) == \
        (0.02, 4, 20, 4, 0.05)
    assert rand.centre.grip_close_s == 1.3 and rand.grip_close == (0.8, 1.25)
    plan = {key: r for _, key, r in evals}
    assert plan["cpu_rand"] is rand
    nominal = plan["cpu_nominal"]
    assert (nominal.pose_bias_m, nominal.cube_period, nominal.centre.grip_close_s) == (0.0, 1, 1.3)
    text = describe(rand)
    assert "zamykanie chwytaka 1.30 s" in text and "odswiezanie co 1-4 takt." in text
    assert "przesuniecie do 2 cm" in text and "opoznienie pozy 4-20 takt." in text and "< 5 cm" in text
    assert options("randomization.pose_delay=[7]").changes() == {"cube_min_delay": 0, "cube_delay": 7}


def test_the_closing_time_overrides_the_stations():
    measured = {"grip_close_s": 2.0}
    assert randomization_plan(options(), measured)[0].centre.grip_close_s == 2.0
    rand, _ = randomization_plan(options("randomization.grip_close_s=1.3"), measured)
    assert rand.centre.grip_close_s == 1.3
    rand, _ = randomization_plan(options("randomization.grip_close_s=0"), measured)
    assert rand.centre.grip_close_s == 0.0 and rand.grip_close == (1.0, 1.0)


BAD_RANDOMIZATION = {"randomization.pose_delay=[5,3]": "0 <= min <= max",
                     "randomization.pose_delay=[-1]": "0 <= min <= max",
                     "randomization.pose_delay=[1,2,3]": r"\[max\] or \[min, max\]",
                     "randomization.pose_period=0": "at least 1 tick",
                     "randomization.pose_bias_m=-0.01": "pose_bias_m must be zero or more",
                     "randomization.latch_m=-0.01": "latch_m must be zero or more",
                     "randomization.grip_close_s=-1": "grip_close_s must be zero or more"}


@pytest.mark.parametrize("override,message", BAD_RANDOMIZATION.items())
def test_bad_randomization_values_are_refused(override, message):
    with pytest.raises(ValueError, match=message):
        options(override)


@pytest.mark.parametrize("override", ["randomization.pose_delay=[4.5,40]", "randomization.pose_period=often"])
def test_randomization_values_of_the_wrong_type_are_rejected(override):
    with pytest.raises(ConfigCompositionException):
        config(override)


def test_the_command_refuses_bad_randomization_values_before_writing_anything(tmp_path):
    args = ["train", "randomization.pose_delay=[5,3]"]
    result = subprocess.run([sys.executable, "-m", "soarm_lab.cli", *args], cwd=tmp_path, capture_output=True,
                            text=True)
    assert result.returncode == 1 and "0 <= min <= max" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_a_cpu_smoke_run_of_the_trainer_from_lift_v3(tmp_path):
    """A short CPU training run saves the perception and gripper settings with the policy."""
    pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")
    args = ["experiment=lift_blocks", "task_overrides.variants=[cube]", "device=cpu", "ppo.num_envs=4",
            "ppo.iterations=2", "eval_episodes=1", "randomization.pose_bias_m=0.02",
            "randomization.pose_delay=[2,6]", "randomization.pose_period=4", "randomization.latch_m=0.05",
            "randomization.grip_close_s=1.3", f"out={tmp_path}", "name=smoke"]
    result = subprocess.run([sys.executable, "-m", "soarm_lab.cli", "train", *args], cwd=tmp_path,
                            env={**os.environ, "MUJOCO_GL": "disable"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Trening lift (warianty: cube)" in out
    assert "zamykanie chwytaka 1.30 s" in out and "odswiezanie co 1-4 takt." in out
    pol = Policy.load(tmp_path / "smoke" / "policy.npz")
    assert pol.meta.obs_dim == 37 and pol.task.variants == ("cube",)
    r = pol.meta.randomization
    assert (r["pose_bias_m"], r["cube_min_delay"], r["cube_delay"], r["cube_period"], r["latch_m"]) == \
        (0.02, 2, 6, 4, 0.05)
    assert r["grip_close"] == [0.8, 1.25] and r["centre"]["grip_close_s"] == 1.3
    assert set(pol.meta.evals) >= {"gpu", "cpu_nominal", "cpu_rand", "cpu_cube", "cpu_rand_cube"}
    saved = json.loads((tmp_path / "smoke" / "config.json").read_text(encoding="utf-8"))
    assert saved["variants"] == ["cube"]


def test_variants_compose_through_the_task_group_too():
    cfg = config("task=pickplace", "task.variants=[cube,cylinder]", "task.block=cylinder")
    task = composed_task(cfg, load_robot())
    assert (task.variants, task.block) == (("cube", "cylinder"), "cylinder")
    for overrides in (["task.block=arch"], ["task.variants=[cube,arch]", "task.block=cube"]):
        with pytest.raises(Exception, match="unknown block 'arch'"):
            composed_task(config("task=lift", *overrides), load_robot())


def test_unknown_stored_task_override_is_rejected():
    cfg = config("init=lift-v3", "ppo=fine_tune", "+task_overrides.cube_hlaf=0.02")
    with pytest.raises(ValueError, match="unknown task overrides"):
        training_task(cfg, load_robot())


@pytest.mark.parametrize("args", [["--cfg", "job"], ["--help"],
                                  ["task=lift", "ppo.iteratons=1"]])
def test_hydra_commands_leave_no_files(tmp_path, args):
    result = subprocess.run([sys.executable, "-m", "soarm_lab.cli", "train", *args],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (1 if "ppo.iteratons=1" in args else 0), result.stderr
    assert list(tmp_path.iterdir()) == []
    if args == ["--help"]:
        assert "experiment" in result.stdout


def test_training_preserves_cwd_and_saves_only_in_the_policy_folder(tmp_path):
    out = tmp_path / "policies"
    script = """
from pathlib import Path
from soarm_lab.cli import main
from soarm_lab.rl import experiment
from soarm_lab.rl.policy import Policy, PolicyMeta
from dataclasses import asdict

cwd = Path.cwd()
def fake_train(task, cfg, **kwargs):
    assert Path.cwd() == cwd
    assert isinstance(cfg.hidden, tuple)
    path = kwargs['out_dir']
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text('{}')
    return Policy(PolicyMeta(asdict(task), [], task.obs_dim, task.act_dim))
experiment.train = fake_train
experiment.evaluate = lambda *a, **kw: {'success': 0.0}
assert main(['train', 'task=lift', 'name=test', 'out=policies', 'randomization.spread=0']) == 0
assert Path.cwd() == cwd
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env={**os.environ, "MUJOCO_GL": "disable"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policies"]
    assert sorted(p.name for p in (out / "test").iterdir()) == [
        "config.json", "experiment.yaml", "meta.json", "policy.npz"]
    text = (out / "test" / "experiment.yaml").read_text()
    assert "name: lift" in text and "spread: 0.0" in text


def test_variants_train_on_the_mixed_env_and_evaluate_each_variant(tmp_path):
    script = """
import json
from dataclasses import asdict
from soarm_lab.cli import main
from soarm_lab.rl import experiment
from soarm_lab.rl.policy import Policy, PolicyMeta
from soarm_lab.rl.ppo import Progress

def fake_train(task, cfg, **kwargs):
    assert kwargs["make_env"] is experiment.mixed_env
    assert kwargs["init_name"] == "lift-v3"
    assert cfg.widen_init
    kwargs["on_progress"](Progress(iteration=1, iterations=1, status="gotowe",
                                   success_by_variant={"cube": 0.9, "cylinder": 0.1, "cuboid": 0.5}))
    return Policy(PolicyMeta(asdict(task), [], task.obs_dim, task.act_dim))

calls = []
def fake_evaluate(policy, episodes, *, randomization, station, variant=None):
    calls.append(variant)
    return {"success": 0.5}

experiment.train = fake_train
experiment.evaluate = fake_evaluate
assert main(["train", "experiment=lift_blocks", "name=blocks", "out=policies", "device=cpu"]) == 0
print("CALLS", json.dumps(calls))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env={**os.environ, "MUJOCO_GL": "disable"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "najslabszy cylinder 10.0%" in result.stdout
    assert "(warianty: cube, cuboid, cylinder)" in result.stdout
    # Reuse the evaluation of the policy's default variant.
    variants = ("cuboid", "cylinder")
    calls = json.loads(result.stdout.split("CALLS ")[1])
    assert calls == [None, None, *(v for v in variants for _ in range(2))]
    meta = json.loads((tmp_path / "policies" / "blocks" / "meta.json").read_text())
    assert set(meta["evals"]) == {"cpu_nominal", "cpu_rand", *(f"cpu_{v}" for v in ("cube", *variants)),
                                  *(f"cpu_rand_{v}" for v in ("cube", *variants))}
    assert meta["evals"]["cpu_cube"] == {"success": 0.5, "variant": "cube"}
    assert meta["task"]["variants"] == ["cube", *variants] and meta["task"]["block"] == "cube"


def test_no_variants_train_on_the_plain_env(tmp_path):
    script = """
from dataclasses import asdict
from soarm_lab.cli import main
from soarm_lab.rl import experiment
from soarm_lab.rl.policy import Policy, PolicyMeta

def fake_train(task, cfg, **kwargs):
    assert kwargs["make_env"] is None and kwargs["init_name"] == ""
    return Policy(PolicyMeta(asdict(task), [], task.obs_dim, task.act_dim))

experiment.train = fake_train
experiment.evaluate = lambda *a, **kw: {"success": 0.0}
assert main(["train", "task=lift", "name=plain", "out=policies", "device=cpu"]) == 0
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env={**os.environ, "MUJOCO_GL": "disable"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "warianty" not in result.stdout


@pytest.mark.parametrize("missing", ["hydra", "omegaconf", "jax", "optax"])
def test_missing_training_dependency_names_the_extra(tmp_path, missing):
    script = f"""
import sys
sys.modules[{missing!r}] = None
from soarm_lab.cli import main
try:
    main(['train', '--cfg', 'job'])
except ImportError as exc:
    assert 'install soarm-lab[train]' in str(exc), exc
else:
    raise AssertionError('train ran without its dependency')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("overrides,accepted", [
    (["init=lift-v3"], False),
    (["init=lift-v3", "ppo=default"], False),
    (["init=lift-v3", "ppo=fine_tune"], True),
    (["experiment=lift_finetune"], True),
    (["experiment=lift_late_pose"], True),
    (["experiment=lift_finetune", "ppo=default"], False),
])
def test_init_requires_the_fine_tuning_ppo_group(tmp_path, overrides, accepted):
    script = """
import sys
from soarm_lab.cli import main
from soarm_lab.rl import experiment
experiment.execute = lambda cfg, **kwargs: print('ACCEPTED')
main(sys.argv[1:])
"""
    result = subprocess.run([sys.executable, "-c", script, "train", *overrides],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (0 if accepted else 1), result.stderr
    assert ("ACCEPTED" in result.stdout) == accepted
    if not accepted:
        assert "init requires ppo=fine_tune" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_a_quoted_tilde_station_composes_and_resolves(tmp_path, monkeypatch):
    from soarm_lab.station import resolve

    monkeypatch.setenv("HOME", str(tmp_path))
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "station.json").write_text('{}')
    (folder / "scene.xml").write_text('<mujoco/>')
    cfg = config('station="~/x"')
    assert cfg.station == "~/x"
    assert resolve(cfg.station) == folder
