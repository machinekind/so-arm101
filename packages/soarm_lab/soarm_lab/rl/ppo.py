"""PPO on the GPU batch environment: short, plain torch, no RL library.

    pol = train("reach", PPOConfig(iterations=200), station=Station.load("sim"))
    pol.save("workspace/policies/reach/policy.npz")

The legacy torch trainer: it needs the `soarm-lab[torch]` and `soarm-lab[train]` (MuJoCo Warp)
extras. The lab's training stack is JAX and MuJoCo Warp; a JAX trainer will replace this one, but it
is not written yet. This one trains a `torch_policy.TorchPolicy`, and returns and saves the numpy
policy `policy.Policy` (`policy.npz`, with the critic for fine-tuning).

The classic recipe of rsl_rl / legged_gym, tuned for thousands of parallel worlds: short rollouts
(`horizon` steps per world), GAE, a few epochs over minibatches, a learning rate adapted to the KL,
and environments that ended on the time limit get the state value added to their reward instead of
zero (otherwise the critic learns that the world ends with the episode, after 5 s for reach, and on
the arm it does not end).

Progress goes to `on_progress(Progress)` and, with `out_dir`, to `progress.json` in the run's folder.
Interrupting: `stop` (threading.Event); the policy as it is at that point is returned, and saved with
`out_dir`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..station import Station
from . import task as tk
from ._torch import torch
from .batch import BatchEnv
from .policy import POLICY_FILE, Policy, PolicyMeta
from .randomize import Randomization
from .torch_policy import TorchPolicy, load_layers, mlp

nn = torch.nn


@dataclass
class PPOConfig:
    num_envs: int = 4096
    horizon: int = 24
    iterations: int = 300
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    desired_kl: float = 0.01
    entropy: float = 0.002
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 0.6
    #: Fine-tuning a policy saved without a critic: this many first iterations train only the critic
    #: (the actor holds), see `train`.
    critic_warmup: int = 0
    hidden: tuple[int, ...] = (256, 256, 128)
    seed: int = 0
    #: Save a checkpoint every this many iterations (0 = only at the end).
    save_every: int = 50


class ActorCritic(nn.Module):
    def __init__(self, meta: PolicyMeta, init_std: float):
        super().__init__()
        self.policy = TorchPolicy(meta)
        self.critic = mlp([meta.obs_dim, *meta.hidden], 1)
        self.log_std = nn.Parameter(torch.full((meta.act_dim,), float(np.log(init_std))))

    def dist(self, obs_n: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.policy.actor(obs_n), self.log_std.exp())

    def value(self, obs_n: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_n).squeeze(-1)


@dataclass
class Progress:
    iteration: int = 0
    iterations: int = 0
    steps: int = 0
    fps: float = 0.0
    reward: float = 0.0
    success: float = 0.0
    episode_len: float = 0.0
    lr: float = 0.0
    std: float = 0.0
    elapsed: float = 0.0
    status: str = "uczenie"
    history: list[dict[str, float]] = field(default_factory=list)


def train(task: str | tk.TaskConfig, cfg: PPOConfig | None = None, *, station: Station | None = None,
          randomization: Randomization | None = None, out_dir: str | Path | None = None,
          on_progress: Callable[[Progress], None] | None = None,
          stop: threading.Event | None = None, device: str = "cuda:0", init: Policy | None = None) -> Policy:
    """Trains a policy for `task`; `init` starts from an existing one (actor, normalizer, critic if saved).

    `init` is for fine-tuning: after identifying the arm's dynamics, after a change of randomization
    or perception, instead of discovering the grasp from scratch.
    """
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)
    station = station or Station.load()
    spec = station.spec()
    tsk = task if isinstance(task, tk.TaskConfig) else tk.make_task(task, spec)
    if init is not None:
        if init.meta.robot != spec.name:
            raise RuntimeError(f"polityka uczona na {init.meta.robot}, stanowisko ma {spec.name}")
        tsk = init.task                                # the task it was trained on
    rand = randomization if randomization is not None else Randomization()
    env = BatchEnv(tsk, cfg.num_envs, station=station, randomization=rand, device=device, seed=cfg.seed)
    meta = PolicyMeta(task=asdict(tsk), hidden=list(cfg.hidden), obs_dim=tsk.obs_dim, act_dim=tsk.act_dim,
                      randomization=_rand_dict(rand), robot=spec.name)
    if init is not None:
        meta.hidden = list(init.meta.hidden)
        meta.notes = f"douczana z polityki z {init.meta.created} ({init.meta.steps / 1e6:.0f} mln krokow)"
    ac = ActorCritic(meta, cfg.init_std).to(device)
    warmup = 0
    if init is not None:
        ac.policy.load_numpy(init)
        if init.critic is not None:
            load_layers(ac.critic, init.critic)
        else:
            warmup = cfg.critic_warmup
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    lr = cfg.lr
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"ppo": asdict(cfg), "task": asdict(tsk),
                                                     "randomization": meta.randomization}, indent=2),
                                         encoding="utf-8")

    N, T = cfg.num_envs, cfg.horizon
    obs = env.reset()
    buf_obs = torch.zeros(T, N, tsk.obs_dim, device=device)
    buf_act = torch.zeros(T, N, tsk.act_dim, device=device)
    buf_logp = torch.zeros(T, N, device=device)
    buf_val = torch.zeros(T, N, device=device)
    buf_rew = torch.zeros(T, N, device=device)
    buf_done = torch.zeros(T, N, device=device)
    ep_ret = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    prog = Progress(iterations=cfg.iterations)
    t_start = time.perf_counter()
    recent_succ: list[float] = []
    recent_ret: list[float] = []
    recent_len: list[float] = []

    for it in range(1, cfg.iterations + 1):
        t0 = time.perf_counter()
        # Fine-tuning a policy saved without a critic starts with a new critic: its advantages are
        # noise at first, and with little exploration (init_std 0.25) it takes long to tell "at the
        # goal" from "a few mm off". Measured on reach-v1: 98% success up to iteration 60, then a
        # drop to 9%, and the deterministic policy drifted off the goal (3 -> 29 mm at the end of
        # the episode). The actor, together with the input normalizer, holds until the critic has
        # learned. A policy with a saved critic needs no warm-up.
        warming = it <= warmup
        # ---------------------------------------------------------- rollout
        with torch.no_grad():
            for t in range(T):
                if not warming:
                    ac.policy.norm.update(obs)
                obs_n = ac.policy.norm(obs)
                dist = ac.dist(obs_n)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = ac.value(obs_n)
                obs, rew, done, info = env.step(act)
                rew = rew + cfg.gamma * val * info["time_outs"].float()
                buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = obs_n, act, logp, val
                buf_rew[t], buf_done[t] = rew, done.float()
                ep_ret += rew
                ep_len += 1
                if done.any():
                    recent_succ += info["success"][done].float().tolist()
                    recent_ret += ep_ret[done].tolist()
                    recent_len += ep_len[done].tolist()
                    ep_ret[done] = 0.0
                    ep_len[done] = 0.0
            last_val = ac.value(ac.policy.norm(obs))
            adv = torch.zeros_like(buf_rew)
            gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nxt = last_val if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = buf_rew[t] + cfg.gamma * nxt * nonterm - buf_val[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + buf_val

        # ------------------------------------------------------------ update
        b_obs, b_act = buf_obs.reshape(T * N, -1), buf_act.reshape(T * N, -1)
        b_logp, b_val = buf_logp.reshape(-1), buf_val.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        mb = T * N // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N, device=device)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                dist = ac.dist(b_obs[idx])
                logp = dist.log_prob(b_act[idx]).sum(-1)
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - b_logp[idx])
                pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
                v = ac.value(b_obs[idx])
                v_clip = b_val[idx] + torch.clamp(v - b_val[idx], -cfg.clip, cfg.clip)
                vl = torch.max((v - b_ret[idx]) ** 2, (v_clip - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                if warming:
                    loss = cfg.value_coef * vl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                    opt.step()
                    continue
                loss = pg + cfg.value_coef * vl - cfg.entropy * ent
                with torch.no_grad():
                    # KL between the old and the new distribution (Schulman's approximation).
                    kl = ((ratio - 1) - (logp - b_logp[idx])).mean().item()
                if cfg.desired_kl > 0:
                    if kl > 2.0 * cfg.desired_kl:
                        lr = max(1e-5, lr / 1.5)
                    elif 0.0 < kl < 0.5 * cfg.desired_kl:
                        lr = min(1e-2, lr * 1.5)
                    for g in opt.param_groups:
                        g["lr"] = lr
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    ac.log_std.clamp_(np.log(0.05), np.log(1.5))

        # ---------------------------------------------------------- progress
        dt = time.perf_counter() - t0
        keep = max(cfg.num_envs, 200)
        recent_succ, recent_ret, recent_len = recent_succ[-keep:], recent_ret[-keep:], recent_len[-keep:]
        prog.iteration, prog.steps = it, it * T * N
        prog.fps = T * N / dt
        prog.success = float(np.mean(recent_succ)) if recent_succ else 0.0
        prog.reward = float(np.mean(recent_ret)) if recent_ret else 0.0
        prog.episode_len = float(np.mean(recent_len)) if recent_len else 0.0
        prog.lr, prog.std = lr, float(ac.log_std.detach().exp().mean())
        prog.elapsed = time.perf_counter() - t_start
        prog.history.append({"it": it, "steps": prog.steps, "success": prog.success, "reward": prog.reward})
        stopping = stop is not None and stop.is_set()
        if it == cfg.iterations or stopping:
            prog.status = "przerwane" if stopping and it < cfg.iterations else "gotowe"
        if on_progress:
            on_progress(prog)
        if out:
            (out / "progress.json").write_text(json.dumps(asdict(prog)), encoding="utf-8")
            if cfg.save_every and it % cfg.save_every == 0:
                _finish(ac, prog, out / POLICY_FILE)
        if stopping:
            break

    return _finish(ac, prog, out / POLICY_FILE if out else None)


def _finish(ac: ActorCritic, prog: Progress, path: Path | None) -> Policy:
    pol = ac.policy.to_numpy(critic=ac.critic)
    pol.meta.steps, pol.meta.iterations = prog.steps, prog.iteration
    pol.meta.success = prog.success
    pol.meta.evals["gpu"] = {"success": prog.success, "reward": prog.reward, "steps": prog.steps}
    if path is not None:
        pol.save(path)
    return pol


def _rand_dict(r: Randomization) -> dict[str, Any]:
    d = asdict(r)
    # Dynamics.to_dict writes an infinite band as "inf": asdict would give Infinity in
    # config.json / meta.json, which strict JSON parsers (a browser's) reject.
    d["centre"] = r.centre.to_dict()
    return d
