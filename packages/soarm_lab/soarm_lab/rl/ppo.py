"""PPO training with JAX and optax on a batch environment.

    pol = train("reach", PPOConfig(iterations=200), station=Station.load("sim"))
    pol.save("workspace/policies/reach/policy.npz")

Requires the `soarm-lab[train]` extra. Training updates the actor, critic, and observation
normalizer. It returns a NumPy `policy.Policy` with the critic saved for fine-tuning.

Each rollout has `horizon` steps per world. Updates use GAE, clipped PPO minibatches, and a
learning rate adapted to KL divergence. Time-limit truncations bootstrap the state value.

`make_env` can replace `batch.BatchEnv` with another environment that implements `EnvLike`.
The rollout steps the environment eagerly. Action sampling, reward bootstrapping, and updates
are JIT compiled.

`on_progress` receives `Progress` after each iteration. `out_dir` enables checkpoints and
`progress.json`. Setting `stop` returns the current policy and saves it when `out_dir` is set.
Variant environments report indices in `info["variant"]` for `Progress.success_by_variant`.

`init` resumes from a saved actor, normalizer, and optional critic. `PPOConfig.widen_init`
allows extra observation inputs appended after the saved inputs. Their actor and critic weights
start at zero. Their normalizer entries start with mean 0 and variance 1.
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
from . import tasks as tk
from .jax_policy import jax, jnp
from .policy import POLICY_FILE, Policy, PolicyMeta
from .randomize import Randomization

try:
    import optax
except ImportError as exc:
    raise ImportError(jp.MISSING) from exc

#: Bounds for the adaptive learning rate and exploration standard deviation.
LR_MIN, LR_MAX = 1e-5, 1e-2
STD_MIN, STD_MAX = 0.05, 1.5
_LOG_2PI = float(np.log(2.0 * np.pi))
#: `_adam` applies the learning rate separately from moment scaling.
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
    #: Initial iterations reserved for learning a missing critic while the actor stays fixed.
    critic_warmup: int = 0
    hidden: tuple[int, ...] = (256, 256, 128)
    seed: int = 0
    #: Checkpoint interval in iterations; zero saves only the final policy.
    save_every: int = 50
    #: Allow appended observation inputs when loading `init`, with zero initial weights.
    widen_init: bool = False


class EnvLike(Protocol):
    """The environment interface required by `train`.

    `step` resets done worlds. `info` contains (N,) bool arrays for "success" and "time_outs".
    "time_outs" marks episodes whose value must be bootstrapped. Variant environments also provide
    a (N,) integer "variant" array indexing their `variants` names.
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
    #: Recent episode success rates grouped by `info["variant"]`.
    success_by_variant: dict[str, float] = field(default_factory=dict)


class _State(NamedTuple):
    """Keep trainable parameters, optimizer moments, normalization, and learning rate in one pytree."""

    #: Actor and critic layers plus the action log standard deviation (act_dim,).
    params: dict[str, Any]
    #: Separate optimizer moments let the critic warm up without advancing the actor's optimizer.
    opt: dict[str, Any]
    norm: jp.Norm
    lr: jax.Array


class _Rollout(NamedTuple):
    """Store `horizon` rollout steps with leading dimensions (T, N)."""

    obs: jax.Array          # Normalized policy inputs.
    act: jax.Array
    logp: jax.Array
    val: jax.Array
    rew: jax.Array          # Includes bootstrapped time-out values.
    done: jax.Array         # Float done flags.


def train(task: str | tk.TaskConfig, cfg: PPOConfig | None = None, *, station: Station | None = None,
          randomization: Randomization | None = None, out_dir: str | Path | None = None,
          on_progress: Callable[[Progress], None] | None = None,
          stop: threading.Event | None = None, device: str = "cuda:0", init: Policy | None = None,
          make_env: Callable[..., EnvLike] | None = None, init_name: str = "") -> Policy:
    """Train `task`, optionally starting from `init` weights and normalization.

    With `init`, a task name uses the saved task configuration. An explicit `TaskConfig` must have
    the same task name as `init`. `cfg.widen_init` permits appended observation inputs.
    `init_name` records the source in `meta.notes`.

    `device` selects "cpu" or a CUDA device such as "cuda:0" for both JAX and the environment.
    `make_env` defaults to `batch.BatchEnv` and must accept
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
            tsk = init.task
        elif tsk.name != init.task.name:
            raise ValueError(f"the init policy was trained on {init.task.name}, the task is {tsk.name}")
    rand = randomization if randomization is not None else Randomization()
    if make_env is None:
        from .batch import BatchEnv

        make_env = BatchEnv
    env = make_env(tsk, cfg.num_envs, station=station, randomization=rand, device=device, seed=cfg.seed)
    variants = tuple(getattr(env, "variants", ()))
    meta = PolicyMeta(task=asdict(tsk), hidden=list(cfg.hidden), obs_dim=env.obs_dim, act_dim=env.act_dim,
                      randomization=_rand_dict(rand), robot=spec.name)
    extra = 0
    if init is not None:
        meta.hidden = list(init.meta.hidden)
        source = f"{init_name}, " if init_name else ""
        meta.notes = f"douczana z polityki z {init.meta.created} ({source}{init.meta.steps / 1e6:.0f} mln krokow)"
        if cfg.widen_init and meta.obs_dim > init.meta.obs_dim:
            extra = meta.obs_dim - init.meta.obs_dim
            meta.notes += f"; obserwacja poszerzona z {init.meta.obs_dim} do {meta.obs_dim} wejsc, nowe wejscia z zerowymi wagami"
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
        if variants:
            config["variants"] = list(variants)
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
    recent_variant: list[int] = []

    for it in range(1, cfg.iterations + 1):
        t0 = time.perf_counter()
        # Freeze the actor and normalizer during critic warm-up to protect the learned behavior.
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
            ends.append((*end, info["variant"]) if "variant" in info else end)
        roll = _Rollout(*(jnp.stack(x) for x in zip(*steps)))

        # ------------------------------------------------------------ update
        key, k_update = jax.random.split(key)
        state = update(state, roll, obs, k_update, warming=warming)
        # Aggregate completed episodes in rollout order.
        done_h, succ_h, ret_h, len_h, *variant_h = (np.asarray(jnp.stack(x)) for x in zip(*ends))
        done_h = done_h.astype(bool)
        recent_succ += succ_h[done_h].astype(float).tolist()
        recent_ret += ret_h[done_h].tolist()
        recent_len += len_h[done_h].tolist()
        if variant_h:
            recent_variant += variant_h[0][done_h].astype(int).tolist()
        jax.block_until_ready(state)

        # ---------------------------------------------------------- progress
        dt = time.perf_counter() - t0
        keep = max(cfg.num_envs, 200)
        recent_succ, recent_ret, recent_len = recent_succ[-keep:], recent_ret[-keep:], recent_len[-keep:]
        recent_variant = recent_variant[-keep:]
        prog.iteration, prog.steps = it, it * T * N
        prog.fps = T * N / dt
        prog.success = float(np.mean(recent_succ)) if recent_succ else 0.0
        prog.reward = float(np.mean(recent_ret)) if recent_ret else 0.0
        prog.episode_len = float(np.mean(recent_len)) if recent_len else 0.0
        prog.lr, prog.std = float(state.lr), float(jnp.exp(state.params["log_std"]).mean())
        prog.elapsed = time.perf_counter() - t_start
        if recent_variant:
            prog.success_by_variant = _success_by_variant(recent_succ, recent_variant, variants)
        prog.history.append({"it": it, "steps": prog.steps, "success": prog.success, "reward": prog.reward,
                             **{f"success_{k}": v for k, v in prog.success_by_variant.items()}})
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
    """Resolve "cpu", "cuda" (device 0), or "cuda:k" to a JAX device."""
    return jp.devices(device)[0]


# ------------------------------------------------------------------- the algorithm
def log_prob(act: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Return Gaussian action log densities (N,) summed over action dimensions."""
    return jnp.sum(-0.5 * ((act - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * _LOG_2PI, -1)


def entropy(log_std: jax.Array) -> jax.Array:
    """Return Gaussian entropy summed over action dimensions."""
    return jnp.sum(0.5 + 0.5 * _LOG_2PI + log_std)


def value(params: dict[str, Any], obs_n: jax.Array) -> jax.Array:
    """Return critic values (N,) for normalized observations (N, obs_dim)."""
    return jp.mlp(params["critic"], obs_n)[..., 0]


def gae(rew: jax.Array, val: jax.Array, done: jax.Array, last_val: jax.Array, gamma: float,
        lam: float) -> tuple[jax.Array, jax.Array]:
    """Return GAE advantages and returns with shape (T, N).

    `done` is a float array that stops accumulation at episode boundaries.
    `last_val` (N,) gives the value after the final step.
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
    """Adjust the learning rate toward `desired_kl` within [`LR_MIN`, `LR_MAX`]."""
    down = jnp.maximum(LR_MIN, lr / 1.5)
    up = jnp.minimum(LR_MAX, lr * 1.5)
    return jnp.where(kl > 2.0 * desired_kl, down, jnp.where((kl > 0.0) & (kl < 0.5 * desired_kl), up, lr))


@partial(jax.jit, static_argnames="update")
def _act(params: dict[str, Any], norm: jp.Norm, obs: jax.Array, key: jax.Array, *, update: bool):
    """Return (norm, normalized obs, sampled action, log density, value, next key).

    `update` includes the current observations in the normalizer before sampling.
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
    """Bootstrap time-out rewards and accumulate episode returns and lengths.

    Return (reward, float done, running return, running length, final return, final length).
    Running totals reset in done worlds; final totals are valid at episode ends.
    """
    rew = jnp.asarray(rew, jnp.float32) + gamma * val * time_outs.astype(jnp.float32)
    ep_ret, ep_len = ep_ret + rew, ep_len + 1.0
    return (rew, done.astype(jnp.float32), jnp.where(done, 0.0, ep_ret), jnp.where(done, 0.0, ep_len),
            ep_ret, ep_len)


def _update(state: _State, roll: _Rollout, last_obs: jax.Array, key: jax.Array, *, cfg: PPOConfig,
            warming: bool) -> _State:
    """Update parameters over shuffled rollout minibatches using GAE targets.

    `warming` freezes the actor, exploration variance, and learning rate while training the critic.
    """
    norm = state.norm
    last_val = value(state.params, jp.normalize(norm, jnp.asarray(last_obs, jnp.float32)))
    adv, ret = gae(roll.rew, roll.val, roll.done, last_val, cfg.gamma, cfg.lam)
    batch = tuple(x.reshape(-1, *x.shape[2:]) for x in (roll.obs, roll.act, roll.logp, roll.val, adv, ret))
    n = batch[0].shape[0]
    mb = n // cfg.minibatches

    def loss_fn(params, obs, act, logp_old, val_old, a, ret):
        logp = log_prob(act, jp.mlp(params["actor"], obs), params["log_std"])
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        log_ratio = logp - logp_old
        ratio = jnp.exp(log_ratio)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        v = value(params, obs)
        v_clip = val_old + jnp.clip(v - val_old, -cfg.clip, cfg.clip)
        vl = jnp.maximum((v - ret) ** 2, (v_clip - ret) ** 2).mean()
        if warming:
            return cfg.value_coef * vl, 0.0
        # Estimate KL divergence between the old and current action distributions.
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
    """Scale gradients together so their global norm is at most `max_norm`."""
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    coef = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * coef, grads)


def _adam(params: Any, grads: Any, opt: Any, lr: jax.Array) -> tuple[Any, Any]:
    """Apply an Adam update with learning rate `lr` and return parameters and optimizer state."""
    updates, opt = _ADAM.update(grads, opt)
    return jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates), opt


def _widen_layers(name: str, layers: jp.JaxLayers, like: jp.JaxLayers, extra: int = 0) -> jp.JaxLayers:
    """Match saved layer shapes to `like`, allowing `extra` appended inputs with zero weights."""
    got = [tuple(w.shape) for w, _ in layers]
    want = [tuple(w.shape) for w, _ in like]
    if got == want:
        return layers
    narrower = [(want[0][0], want[0][1] - extra), *want[1:]]
    if extra > 0 and got == narrower:
        return jp.widen_inputs(layers, want[0][1])
    raise ValueError(f"the {name} of the init policy has layers {got}, the environment needs {want}")


def _success_by_variant(success: list[float], variant: list[int], names: tuple[str, ...]) -> dict[str, float]:
    """Group episode success by variant name, falling back to the index for unknown names."""
    succ, idx = np.asarray(success, float), np.asarray(variant, int)
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
    # Encode infinite bounds as strings so configuration files remain valid JSON.
    d["centre"] = r.centre.to_dict()
    return d
