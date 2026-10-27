"""The policy in JAX: what the PPO trainer (`ppo.py`) optimizes, and `policy.Policy` from it and back.

Needs `soarm-lab[train]`. Running a policy needs numpy only (`policy.Policy`); this module holds the
same network as JAX arrays, so the trainer can differentiate and jit it, and turns it into that policy
and back (fine-tuning starts from a `policy.npz`):

    actor = init_mlp(key, [obs_dim, *hidden, act_dim])      # [(weight (n_out, n_in), bias), ...]
    norm = norm_update(Norm.new(obs_dim), obs)               # the running mean and variance
    mean_action = mlp(actor, normalize(norm, obs))           # = Policy.act(obs) before its clip to [-1, 1]

The layers keep the layout of `policy.npz` (`x @ weight.T + bias`, ELU after every layer but the last),
so `Policy.act` in numpy gives this actor's mean to float32 precision. The weights start as torch's
`nn.Linear` does (uniform in +-1/sqrt(n_in), weights and biases), and the normalizer merges a batch
exactly as `torch_policy.Normalizer.update`: training behaves as with the torch trainer it replaces.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .policy import OBS_CLIP, VAR_EPS, Layers, Policy, PolicyMeta

#: The error without jax.
MISSING = ("jax is not installed. The PPO trainer (soarm-lab train) needs it: install soarm-lab[train] "
           "(pip install 'soarm-lab[train]'; on Linux it brings the CUDA 12 build of jax). "
           "Running a policy (the strategy, soarm-lab eval) needs numpy only.")

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(MISSING) from exc

def devices(device: str) -> tuple[jax.Device, str]:
    """The JAX device and the Warp device for `device`: "cuda", "cuda:<k>" or "cpu"."""
    name = str(device)
    if name == "cpu":
        return jax.devices("cpu")[0], "cpu"
    if name == "cuda" or name.startswith("cuda:"):
        k = int(name.partition(":")[2] or 0)
        try:
            gpus = jax.devices("gpu")
        except RuntimeError as exc:
            raise RuntimeError(f"JAX sees no CUDA GPU for device {name!r}: install soarm-lab[train] on Linux "
                               "(jax[cuda12]), or run MuJoCo Warp on the CPU: soarm-lab train --device cpu "
                               "(device='cpu')") from exc
        # A Warp array's DLPack device id is its CUDA ordinal, and JAX places what it reads by
        # `local_hardware_id`: the same number.
        match = [d for d in gpus if d.local_hardware_id == k]
        if not match:
            raise RuntimeError(f"no CUDA device {k} in JAX (it has {len(gpus)})")
        return match[0], f"cuda:{k}"
    raise ValueError(f"unknown device {device!r}: 'cuda', 'cuda:<k>' or 'cpu'")


#: Dense layers as JAX arrays, (weight (n_out, n_in), bias (n_out,)), as `policy.Layers`.
JaxLayers = list[tuple[jax.Array, jax.Array]]


class Norm(NamedTuple):
    """The observations' running mean and variance (a pytree): updated during training, fixed after."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array
    clip: jax.Array

    @staticmethod
    def new(dim: int, clip: float = OBS_CLIP) -> Norm:
        """No observations seen: mean 0, variance 1, count 1e-4 (as `torch_policy.Normalizer`)."""
        return Norm(jnp.zeros(dim, jnp.float32), jnp.ones(dim, jnp.float32), jnp.asarray(1e-4, jnp.float32),
                    jnp.asarray(clip, jnp.float32))


def norm_update(norm: Norm, x: jax.Array) -> Norm:
    """The statistics with a batch `x` (n, dim) merged in: `torch_policy.Normalizer.update`'s formula.

    The batch variance is the biased one (divided by n), and the count starts at 1e-4.
    """
    b_mean, b_var, n = x.mean(0), x.var(0), x.shape[0]
    delta = b_mean - norm.mean
    tot = norm.count + n
    mean = norm.mean + delta * n / tot
    var = (norm.var * norm.count + b_var * n + delta**2 * norm.count * n / tot) / tot
    return Norm(mean, var, tot, norm.clip)


def normalize(norm: Norm, x: jax.Array) -> jax.Array:
    """As `Policy.normalize`: clip((x - mean) / sqrt(var + 1e-8), -clip, clip)."""
    return jnp.clip((x - norm.mean) / jnp.sqrt(norm.var + VAR_EPS), -norm.clip, norm.clip)


def init_mlp(key: jax.Array, sizes: list[int]) -> JaxLayers:
    """Dense layers for `sizes` (n_in, ..., n_out), initialised as torch's `nn.Linear`.

    torch draws the weight with Kaiming-uniform (a = sqrt(5)) and the bias uniformly, both within
    +-1/sqrt(n_in).
    """
    layers = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        kw, kb = jax.random.split(k)
        bound = 1.0 / np.sqrt(a)
        layers.append((jax.random.uniform(kw, (b, a), jnp.float32, -bound, bound),
                       jax.random.uniform(kb, (b,), jnp.float32, -bound, bound)))
    return layers


def mlp(layers: JaxLayers, x: jax.Array) -> jax.Array:
    """The MLP of `policy.run_mlp`: x @ weight.T + bias, ELU (alpha 1) after every layer but the last."""
    for i, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if i < len(layers) - 1:
            x = jax.nn.elu(x)
    return x


def to_policy(meta: PolicyMeta, norm: Norm, actor: JaxLayers, critic: JaxLayers | None = None) -> Policy:
    """The numpy policy with these weights (sharing `meta`), and the critic's if given."""
    pol = Policy(meta)
    pol.obs_mean, pol.obs_var = _array(norm.mean), _array(norm.var)
    pol.obs_count, pol.obs_clip = _array(norm.count)[()], _array(norm.clip)[()]
    pol.actor = to_layers(actor)
    pol.critic = to_layers(critic) if critic is not None else None
    return pol


def from_policy(pol: Policy) -> tuple[Norm, JaxLayers, JaxLayers | None]:
    """The numpy policy's normalizer, actor and critic (None if it was saved without one) as JAX arrays."""
    norm = Norm(*(jnp.asarray(np.asarray(v, np.float32)) for v in (pol.obs_mean, pol.obs_var, pol.obs_count,
                                                                     pol.obs_clip)))
    return norm, from_layers(pol.actor), from_layers(pol.critic) if pol.critic is not None else None


def to_layers(layers: JaxLayers) -> Layers:
    return [(_array(weight), _array(bias)) for weight, bias in layers]


def from_layers(layers: Layers) -> JaxLayers:
    return [(jnp.asarray(np.asarray(weight, np.float32)), jnp.asarray(np.asarray(bias, np.float32)))
            for weight, bias in layers]


def _array(x: jax.Array) -> np.ndarray:
    return np.array(x, np.float32, copy=True)
