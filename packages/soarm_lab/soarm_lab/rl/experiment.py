"""Compose and run training experiments with Hydra."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:
    raise ImportError("Training needs Hydra: install soarm-lab[train]") from exc

from ..paths import policies_dir
from ..robot import RobotSpec
from ..station import Station
from . import tasks
from .evaluate import evaluate
from .policy import POLICY_FILE, Policy, find_policy, variant_key
from .ppo import PPOConfig, train
from .randomize import Dynamics, describe, train_plan


@dataclass
class RandomizationConfig:
    spread: float = 1.0


@dataclass
class TrainConfig:
    task: Any = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    seed: int = 0
    name: str | None = None
    out: str | None = None
    station: str = "sim"
    eval_episodes: int = 50
    stop_file: str | None = None
    init: str | None = None
    device: str = "cuda:0"
    task_overrides: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(group="ppo_schema", name="default", node=PPOConfig)
    cs.store(group="randomization_schema", name="default", node=RandomizationConfig)
    for name in tasks.TASKS:
        schema = OmegaConf.structured(tasks.make_task(name))
        OmegaConf.set_readonly(schema, False)
        cs.store(group="task_schema", name=name, node=schema)


register_configs()


def composed_task(cfg: DictConfig, spec: RobotSpec) -> tasks.TaskConfig:
    """Apply station defaults, then the composed task's changes."""
    value = instantiate(cfg.task, _convert_="object")
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in asdict(value).items()}
    default = asdict(tasks.make_task(data["name"]))
    changes = {k: v for k, v in data.items() if k != "name" and v != default[k]}
    return tasks.make_task(data["name"], spec=spec, **changes)


def training_task(cfg: DictConfig, spec: RobotSpec, *, task_selected: bool = False):
    task = composed_task(cfg, spec)
    init = Policy.load(find_policy(cfg.init)) if cfg.init else None
    if init is not None:
        if task_selected and type(task) is not type(init.task):
            raise ValueError("selected task differs from the initial policy's task")
        task = init.task
    overrides = OmegaConf.to_container(cfg.task_overrides, resolve=True)
    owned = {f.name for f in fields(task)} - {"name"}
    unknown = set(overrides) - owned
    if unknown:
        raise ValueError(f"unknown task overrides: {', '.join(sorted(unknown))}")
    task = replace(task, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})
    return init, task


def mixed_env(task: tasks.LiftConfig, num_envs: int, **kwargs):
    """Build a `MixedBatchEnv` from the task variants using `BatchEnv` arguments."""
    from .mixed import MixedBatchEnv

    return MixedBatchEnv(task.variant_tasks(), num_envs, **kwargs)


def execute(a: DictConfig, *, task_selected: bool = False) -> int:
    st = Station.load(a.station)
    init, task = training_task(a, st.spec(), task_selected=task_selected)
    name = a.name or f"{task.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(a.out or policies_dir()) / name
    dyn = Dynamics.from_dict(st.dynamics)
    rand, evals = train_plan(dyn, a.randomization.spread)
    cfg = instantiate(a.ppo, _convert_="object")
    cfg.hidden = tuple(cfg.hidden)
    if init is not None:
        print(f"Start z polityki {a.init} ({init.task.name})")
    variants = task.variants if isinstance(task, tasks.LiftConfig) else ()
    listed = f" (warianty: {', '.join(variants)})" if variants else ""
    print(f"Trening {task.name}{listed} na stanowisku {st.folder}: {a.ppo.num_envs} swiatow x {a.ppo.iterations} iteracji -> {out}")
    print(f"  {describe(rand)}{'' if a.randomization.spread == 0 else f', rozrzut x{a.randomization.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            worst = ""
            if p.success_by_variant:
                variant = min(p.success_by_variant, key=p.success_by_variant.__getitem__)
                worst = f"  najslabszy {variant} {p.success_by_variant[variant]:5.1%}"
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}{worst}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

    stop = None
    if a.stop_file:
        # Save the current policy when the stop file appears.
        import threading

        stop = threading.Event()
        flag = Path(a.stop_file)

        def watch():
            while not stop.is_set():
                if flag.exists():
                    stop.set()
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(a, out / "experiment.yaml", resolve=True)
    init_name = find_policy(a.init).parent.name if a.init else ""
    pol = train(task, cfg, station=st, randomization=rand, out_dir=out, on_progress=show, stop=stop,
                init=init, device=a.device, make_env=mixed_env if variants else None, init_name=init_name)

    def report(label, res):
        rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
        print(f"  {label:{27 if variants else 17}s} sukces {res['success']:5.1%}  ({json.dumps(rounded)})")

    print("Ewaluacja na CPU (zwykle MuJoCo, inny silnik niz w treningu):")
    for label, key, r in evals:
        res = evaluate(pol, a.eval_episodes, randomization=r, station=st)
        pol.meta.evals[key] = res
        report(label, res)
    for variant in variants:
        for label, key, r in evals:
            res = {**pol.meta.evals[key], "variant": variant} if variant == task.block else \
                evaluate(pol, a.eval_episodes, randomization=r, station=st, variant=variant)
            pol.meta.evals[variant_key(key, variant)] = res
            report(f"{variant} {label}", res)
    print(f"Zapisano {pol.save(out / POLICY_FILE)}")
    return 0


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runtime = HydraConfig.get()
    if cfg.init and runtime.runtime.choices["ppo"] != "fine_tune":
        raise ValueError("init requires ppo=fine_tune to preserve the fine-tuning settings")
    overrides = runtime.overrides.task
    selected = any(o.startswith(("task=", "experiment=")) for o in overrides)
    execute(cfg, task_selected=selected)


def run(argv: list[str]) -> int:
    previous = sys.argv
    try:
        sys.argv = ["soarm-lab train", *argv]
        main()
    finally:
        sys.argv = previous
    return 0
