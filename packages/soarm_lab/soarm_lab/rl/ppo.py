"""PPO on the GPU batch environment: short, plain JAX and optax, no RL library.

    pol = train("reach", PPOConfig(iterations=200), station=Station.load("sim"))
    pol.save("workspace/policies/reach/policy.npz")

Needs the `soarm-lab[train]` extra (JAX, optax, MuJoCo Warp), no torch. It trains the actor, the critic
and the observation normalizer as JAX arrays (`jax_policy`), and returns and saves the numpy policy
`policy.Policy` (`policy.npz`, with the critic for fine-tuning).

The classic recipe of rsl_rl / legged_gym, tuned for thousands of parallel worlds: short rollouts
(`horizon` steps per world), GAE, a few epochs over minibatches, a learning rate adapted to the KL,
and environments that ended on the time limit get the state value added to their reward instead of
zero (otherwise the critic learns that the world ends with the episode, after 5 s for reach, and on
the arm it does not end). It is step for step the algorithm of the torch trainer it replaced.

The environment (`batch.BatchEnv`, or another with its interface: `make_env`) steps eagerly, so the
rollout is a Python loop over the `horizon` steps. What runs around a step is jitted: sampling the
actions (`_act`) and bootstrapping the reward (`_record`). So is the whole update, GAE and every
epoch's minibatches (`_update`). The rollout's arrays are collected in lists and stacked once per
iteration.

Progress goes to `on_progress(Progress)` and, with `out_dir`, to `progress.json` in the run's folder.
Interrupting: `stop` (threading.Event); the policy as it is at that point is returned, and saved with
`out_dir`. An environment of several object kinds (`mixed.MixedBatchEnv`) marks every world's kind in
`info["kind"]`, and `Progress.success_by_kind` gives the success per kind.

Fine-tuning (`init`) keeps the policy's layers. One change of shape is allowed: a shape-conditioned
task (`TaskConfig.shape_conditioned`) observes `objects.DESCRIPTOR_DIM` more inputs, the object's
descriptor, than a policy trained on one object. Such a policy is widened (`_widen_layers`,
`jax_policy.widen_norm`): the first layer of the actor and of the critic gets zero weights for the
new inputs, the normalizer gets mean 0 and variance 1 for them. It starts by doing exactly what it
did (lift-v3 grasps the cube) and learns what the descriptor means from there.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np

from ..station import Station
from . import jax_policy as jp
from . import task as tk
from .jax_policy import jax, jnp
from .objects import DESCRIPTOR_DIM
from .policy import POLICY_FILE, Policy, PolicyMeta
from .randomize import Randomization

try:
    import optax
except ImportError as exc:
    raise ImportError(jp.MISSING) from exc

#: The bounds of the KL-adapted learning rate and of the exploration's standard deviation.
LR_MIN, LR_MAX = 1e-5, 1e-2
STD_MIN, STD_MAX = 0.05, 1.5
_LOG_2PI = float(np.log(2.0 * np.pi))
#: torch's Adam: betas (0.9, 0.999), eps 1e-8; the learning rate is applied in `_adam`.
_ADAM = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)


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


class EnvLike(Protocol):
    """What `train` needs of an environment: the interface of `batch.BatchEnv`.

    Worlds reset themselves inside `step`; `info` holds bool arrays (N,) "time_outs" (episodes to
    bootstrap with the state value) and "success". An environment of several object kinds adds
    "kind", an int array (N,) of indices into its `kinds` (names), for `Progress.success_by_kind`.
    """

    num_envs: int

    @property
    def obs_dim(self) -> int: ...

    @property
    def act_dim(self) -> int: ...

    def reset(self) -> jax.Array: ...

    def step(self, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]: ...


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
    #: The success of the recent episodes per object kind, when the environment reports `info["kind"]`
    #: (`mixed.MixedBatchEnv`); empty otherwise.
    success_by_kind: dict[str, float] = field(default_factory=dict)


class _State(NamedTuple):
    """What training changes, as one pytree."""

    #: "actor" and "critic" (dense layers as in `policy.npz`) and "log_std" (act_dim,): the Gaussian's
    #: state-independent log standard deviation.
    params: dict[str, Any]
    #: Adam's moments per group, "policy" (actor and log_std) and "critic": a critic warm-up steps the
    #: critic only, and the actor's first step after it is Adam's first, as in torch.
    opt: dict[str, Any]
    norm: jp.Norm
    lr: jax.Array


class _Rollout(NamedTuple):
    """`horizon` steps of every world, stacked: (T, N, ...)."""

    obs: jax.Array          # normalized, as the policy saw it
    act: jax.Array
    logp: jax.Array
    val: jax.Array
    rew: jax.Array          # with the time-outs bootstrapped
    done: jax.Array         # float


def train(task: str | tk.TaskConfig, cfg: PPOConfig | None = None, *, station: Station | None = None,
          randomization: Randomization | None = None, out_dir: str | Path | None = None,
          on_progress: Callable[[Progress], None] | None = None,
          stop: threading.Event | None = None, device: str = "cuda:0", init: Policy | None = None,
          make_env: Callable[..., EnvLike] | None = None, init_name: str = "") -> Policy:
    """Trains a policy for `task`; `init` starts from an existing one (actor, normalizer, critic if saved).

    `init` is for fine-tuning: after identifying the arm's dynamics, after a change of randomization
    or perception, instead of discovering the grasp from scratch. It trains on the task `init` was
    trained on, unless `task` is a `TaskConfig` of the same name: then on that one (the CLI turns on
    `shape_conditioned` and overrides `episode_steps` this way). A shape-conditioned task widens a
    policy trained without the descriptor (see the module docstring). `init_name` (`--init`) goes
    into `meta.notes`.

    `device` is "cuda:0" ("cuda:k", "cuda") or "cpu": the JAX device, and the same string goes to the
    environment. `make_env` builds the environment, called as `batch.BatchEnv` (the default):
    `make_env(task, num_envs, station=, randomization=, device=, seed=)`.
    """
    cfg = cfg or PPOConfig()
    dev = jax_device(device)
    key = jax.random.PRNGKey(cfg.seed)
    station = station or Station.load()
    spec = station.spec()
    tsk = task if isinstance(task, tk.TaskConfig) else tk.make_task(task, spec)
    if init is not None:
        if init.meta.robot != spec.name:
            raise RuntimeError(f"polityka uczona na {init.meta.robot}, stanowisko ma {spec.name}")
        if not isinstance(task, tk.TaskConfig):
            tsk = init.task                            # the task it was trained on
        elif tsk.name != init.task.name:
            raise ValueError(f"the init policy was trained on {init.task.name}, the task is {tsk.name}")
    rand = randomization if randomization is not None else Randomization()
    if make_env is None:
        from .batch import BatchEnv

        make_env = BatchEnv
    env = make_env(tsk, cfg.num_envs, station=station, randomization=rand, device=device, seed=cfg.seed)
    #: The names of `info["kind"]`'s indices (`mixed.MixedBatchEnv.kinds`); empty for one object.
    kinds = tuple(getattr(env, "kinds", ()) or ())
    meta = PolicyMeta(task=asdict(tsk), hidden=list(cfg.hidden), obs_dim=env.obs_dim, act_dim=env.act_dim,
                      randomization=_rand_dict(rand), robot=spec.name)
    # A policy trained without the object's descriptor, on a task that observes it: widened.
    extra = 0
    if init is not None:
        meta.hidden = list(init.meta.hidden)
        source = f"{init_name}, " if init_name else ""
        meta.notes = (f"douczana z polityki z {init.meta.created} ({source}{init.meta.steps / 1e6:.0f} mln "
                      "krokow)")
        if tsk.shape_conditioned and meta.obs_dim - init.meta.obs_dim == DESCRIPTOR_DIM:
            extra = DESCRIPTOR_DIM
            meta.notes += (f"; obserwacja poszerzona z {init.meta.obs_dim} do {meta.obs_dim} wejsc o opis "
                           "obiektu, nowe wejscia z zerowymi wagami")
    key, k_actor, k_critic = jax.random.split(key, 3)
    sizes = [meta.obs_dim, *meta.hidden]
    norm = jp.Norm.new(meta.obs_dim)
    actor = jp.init_mlp(k_actor, [*sizes, meta.act_dim])
    critic = jp.init_mlp(k_critic, [*sizes, 1])
    warmup = 0
    if init is not None:
        norm, init_actor, init_critic = jp.from_policy(init)
        actor = _widen_layers("actor", init_actor, actor, extra)
        if extra:
            norm = jp.widen_norm(norm, meta.obs_dim)
        if init_critic is not None:
            critic = _widen_layers("critic", init_critic, critic, extra)
        else:
            warmup = cfg.critic_warmup
    log_std = jnp.full(meta.act_dim, np.log(cfg.init_std), jnp.float32)
    params = {"actor": actor, "log_std": log_std, "critic": critic}
    opt = {"policy": _ADAM.init({"actor": actor, "log_std": log_std}), "critic": _ADAM.init(critic)}
    state = jax.device_put(_State(params, opt, norm, jnp.asarray(cfg.lr, jnp.float32)), dev)
    key = jax.device_put(key, dev)
    update = jax.jit(partial(_update, cfg=cfg), static_argnames="warming")
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        config = {"ppo": asdict(cfg), "task": asdict(tsk), "randomization": meta.randomization}
        if kinds:
            config["kinds"] = list(kinds)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    N, T = cfg.num_envs, cfg.horizon
    obs = env.reset()
    ep_ret = jax.device_put(jnp.zeros(N, jnp.float32), dev)
    ep_len = jax.device_put(jnp.zeros(N, jnp.float32), dev)
    prog = Progress(iterations=cfg.iterations)
    t_start = time.perf_counter()
    recent_succ: list[float] = []
    recent_ret: list[float] = []
    recent_len: list[float] = []
    recent_kind: list[int] = []                        # with `info["kind"]`: the kind of each episode

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
        steps, ends = [], []
        for _ in range(T):
            norm, obs_n, act, logp, val, key = _act(state.params, state.norm, obs, key, update=not warming)
            state = state._replace(norm=norm)
            obs, rew, done, info = env.step(act)
            rew, done_f, ep_ret, ep_len, ret_end, len_end = _record(val, rew, done, info["time_outs"], ep_ret,
                                                                    ep_len, cfg.gamma)
            steps.append((obs_n, act, logp, val, rew, done_f))
            end = (done, info["success"], ret_end, len_end)
            ends.append((*end, info["kind"]) if "kind" in info else end)
        roll = _Rollout(*(jnp.stack(x) for x in zip(*steps)))

        # ------------------------------------------------------------ update
        key, k_update = jax.random.split(key)
        state = update(state, roll, obs, k_update, warming=warming)
        # The episodes that ended in this rollout, in the order they ended.
        done_h, succ_h, ret_h, len_h, *kind_h = (np.asarray(jnp.stack(x)) for x in zip(*ends))
        done_h = done_h.astype(bool)
        recent_succ += succ_h[done_h].astype(float).tolist()
        recent_ret += ret_h[done_h].tolist()
        recent_len += len_h[done_h].tolist()
        if kind_h:
            recent_kind += kind_h[0][done_h].astype(int).tolist()
        jax.block_until_ready(state)

        # ---------------------------------------------------------- progress
        dt = time.perf_counter() - t0
        keep = max(cfg.num_envs, 200)
        recent_succ, recent_ret, recent_len = recent_succ[-keep:], recent_ret[-keep:], recent_len[-keep:]
        recent_kind = recent_kind[-keep:]
        prog.iteration, prog.steps = it, it * T * N
        prog.fps = T * N / dt
        prog.success = float(np.mean(recent_succ)) if recent_succ else 0.0
        prog.reward = float(np.mean(recent_ret)) if recent_ret else 0.0
        prog.episode_len = float(np.mean(recent_len)) if recent_len else 0.0
        prog.lr, prog.std = float(state.lr), float(jnp.exp(state.params["log_std"]).mean())
        prog.elapsed = time.perf_counter() - t_start
        if recent_kind:
            prog.success_by_kind = _success_by_kind(recent_succ, recent_kind, kinds)
        prog.history.append({"it": it, "steps": prog.steps, "success": prog.success, "reward": prog.reward,
                             **{f"success_{k}": v for k, v in prog.success_by_kind.items()}})
        stopping = stop is not None and stop.is_set()
        if it == cfg.iterations or stopping:
            prog.status = "przerwane" if stopping and it < cfg.iterations else "gotowe"
        if on_progress:
            on_progress(prog)
        if out:
            (out / "progress.json").write_text(json.dumps(asdict(prog)), encoding="utf-8")
            if cfg.save_every and it % cfg.save_every == 0:
                _finish(meta, state, prog, out / POLICY_FILE)
        if stopping:
            break

    return _finish(meta, state, prog, out / POLICY_FILE if out else None)


def jax_device(device: str) -> jax.Device:
    """The JAX device of "cuda:k" (CUDA's device k, as `batch.BatchEnv` reads it), "cuda" (k = 0), "cpu"."""
    return jp.devices(device)[0]


# ------------------------------------------------------------------- the algorithm
def log_prob(act: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """The Gaussian's log density of the actions (N, act_dim), summed over the action's dimensions."""
    return jnp.sum(-0.5 * ((act - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * _LOG_2PI, -1)


def entropy(log_std: jax.Array) -> jax.Array:
    """The Gaussian's entropy, summed over the action's dimensions (the same in every state)."""
    return jnp.sum(0.5 + 0.5 * _LOG_2PI + log_std)


def value(params: dict[str, Any], obs_n: jax.Array) -> jax.Array:
    """The critic's state value (N,) of normalized observations."""
    return jp.mlp(params["critic"], obs_n)[..., 0]


def gae(rew: jax.Array, val: jax.Array, done: jax.Array, last_val: jax.Array, gamma: float,
        lam: float) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over a rollout (T, N): (advantages, returns = advantages + values).

    `done` (float) cuts the sums where an episode ended; `last_val` is the value of the state after
    the last step.
    """
    def back(carry, step):
        adv_next, val_next = carry
        r, v, d = step
        nonterm = 1.0 - d
        delta = r + gamma * val_next * nonterm - v
        adv = delta + gamma * lam * nonterm * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(back, (jnp.zeros_like(last_val), last_val), (rew, val, done), reverse=True)
    return adv, adv + val


def adapt_lr(lr: jax.Array, kl: jax.Array, desired_kl: float) -> jax.Array:
    """The learning rate after a minibatch with this KL: /1.5 above 2 x `desired_kl`, x1.5 below half
    of it (and above 0), within [`LR_MIN`, `LR_MAX`]."""
    down = jnp.maximum(LR_MIN, lr / 1.5)
    up = jnp.minimum(LR_MAX, lr * 1.5)
    return jnp.where(kl > 2.0 * desired_kl, down, jnp.where((kl > 0.0) & (kl < 0.5 * desired_kl), up, lr))


@partial(jax.jit, static_argnames="update")
def _act(params: dict[str, Any], norm: jp.Norm, obs: jax.Array, key: jax.Array, *, update: bool):
    """A rollout step's policy: (norm, normalized obs, sampled action, its log density, value, next key).

    With `update` the normalizer takes in `obs` first.
    """
    obs = jnp.asarray(obs, jnp.float32)
    if update:
        norm = jp.norm_update(norm, obs)
    obs_n = jp.normalize(norm, obs)
    mean = jp.mlp(params["actor"], obs_n)
    key, k_noise = jax.random.split(key)
    act = mean + jnp.exp(params["log_std"]) * jax.random.normal(k_noise, mean.shape, mean.dtype)
    return norm, obs_n, act, log_prob(act, mean, params["log_std"]), value(params, obs_n), key


@jax.jit
def _record(val, rew, done, time_outs, ep_ret, ep_len, gamma):
    """A step's reward with the time-outs bootstrapped, and the episodes' sums.

    (reward, done as float, return and length so far with the ended episodes zeroed, the return and
    length of every world's episode at this step: those of the ended ones count).
    """
    rew = jnp.asarray(rew, jnp.float32) + gamma * val * time_outs.astype(jnp.float32)
    ep_ret, ep_len = ep_ret + rew, ep_len + 1.0
    return (rew, done.astype(jnp.float32), jnp.where(done, 0.0, ep_ret), jnp.where(done, 0.0, ep_len),
            ep_ret, ep_len)


def _update(state: _State, roll: _Rollout, last_obs: jax.Array, key: jax.Array, *, cfg: PPOConfig,
            warming: bool) -> _State:
    """GAE and `cfg.epochs` passes over `cfg.minibatches` random minibatches of the rollout.

    `warming` trains the critic only: the actor, its log_std and the learning rate hold.
    """
    norm = state.norm
    last_val = value(state.params, jp.normalize(norm, jnp.asarray(last_obs, jnp.float32)))
    adv, ret = gae(roll.rew, roll.val, roll.done, last_val, cfg.gamma, cfg.lam)
    batch = tuple(x.reshape(-1, *x.shape[2:]) for x in (roll.obs, roll.act, roll.logp, roll.val, adv, ret))
    n = batch[0].shape[0]
    mb = n // cfg.minibatches

    def loss_fn(params, obs, act, logp_old, val_old, a, ret):
        logp = log_prob(act, jp.mlp(params["actor"], obs), params["log_std"])
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)               # torch's std: unbiased
        log_ratio = logp - logp_old
        ratio = jnp.exp(log_ratio)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        v = value(params, obs)
        v_clip = val_old + jnp.clip(v - val_old, -cfg.clip, cfg.clip)
        vl = jnp.maximum((v - ret) ** 2, (v_clip - ret) ** 2).mean()
        if warming:
            return cfg.value_coef * vl, 0.0
        # KL between the old and the new distribution (Schulman's approximation).
        kl = jax.lax.stop_gradient(((ratio - 1) - log_ratio).mean())
        return pg + cfg.value_coef * vl - cfg.entropy * entropy(params["log_std"]), kl

    def minibatch(carry, idx):
        params, opt, lr = carry
        (_, kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *(x[idx] for x in batch))
        if not warming and cfg.desired_kl > 0:
            lr = adapt_lr(lr, kl, cfg.desired_kl)
        grads = _clip_grad_norm(grads, cfg.max_grad_norm)
        critic, opt_critic = _adam(params["critic"], grads["critic"], opt["critic"], lr)
        params, opt = {**params, "critic": critic}, {**opt, "critic": opt_critic}
        if not warming:
            pi = {"actor": params["actor"], "log_std": params["log_std"]}
            pi, opt_pi = _adam(pi, {"actor": grads["actor"], "log_std": grads["log_std"]}, opt["policy"], lr)
            log_std = jnp.clip(pi["log_std"], np.log(STD_MIN), np.log(STD_MAX))
            params, opt = {**params, "actor": pi["actor"], "log_std": log_std}, {**opt, "policy": opt_pi}
        return (params, opt, lr), None

    def epoch(carry, k):
        perm = jax.random.permutation(k, n)[:mb * cfg.minibatches].reshape(cfg.minibatches, mb)
        return jax.lax.scan(minibatch, carry, perm)[0], None

    (params, opt, lr), _ = jax.lax.scan(epoch, (state.params, state.opt, state.lr),
                                        jax.random.split(key, cfg.epochs))
    return _State(params, opt, norm, lr)


def _clip_grad_norm(grads: Any, max_norm: float) -> Any:
    """torch's `clip_grad_norm_`: every gradient scaled by max_norm / (global norm + 1e-6), at most 1."""
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    coef = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * coef, grads)


def _adam(params: Any, grads: Any, opt: Any, lr: jax.Array) -> tuple[Any, Any]:
    """One Adam step with the learning rate `lr`."""
    updates, opt = _ADAM.update(grads, opt)
    return jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates), opt


def _widen_layers(name: str, layers: jp.JaxLayers, like: jp.JaxLayers, extra: int = 0) -> jp.JaxLayers:
    """The `init` policy's layers, checked against the shapes the environment needs (those of `like`).

    With `extra` > 0 the environment's observation may be `extra` inputs wider than the policy's:
    the first layer then gets zero weights for them (`jax_policy.widen_inputs`). Any other difference
    raises.
    """
    got = [tuple(w.shape) for w, _ in layers]
    want = [tuple(w.shape) for w, _ in like]
    if got == want:
        return layers
    narrower = [(want[0][0], want[0][1] - extra), *want[1:]]
    if extra > 0 and got == narrower:
        return jp.widen_inputs(layers, want[0][1])
    raise ValueError(f"the {name} of the init policy has layers {got}, the environment needs {want}")


def _success_by_kind(success: list[float], kind: list[int], names: tuple[str, ...]) -> dict[str, float]:
    """The mean success per object kind of the recent episodes (`success` and `kind` of equal length).

    A kind is named by `names[index]`, or by its index when the environment gives no names.
    """
    succ, idx = np.asarray(success, float), np.asarray(kind, int)
    return {(names[k] if 0 <= k < len(names) else str(k)): float(succ[idx == k].mean()) for k in np.unique(idx)}


def _finish(meta: PolicyMeta, state: _State, prog: Progress, path: Path | None) -> Policy:
    pol = jp.to_policy(meta, state.norm, state.params["actor"], state.params["critic"])
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
