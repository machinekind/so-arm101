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

A policy can start training on a wider observation than it was trained on (`widen_inputs`,
`widen_norm`): new inputs appended at the end, with zero weights and the statistics of nothing seen.
The widened network gives the same action as before for any value of the new inputs. `ppo.train`
uses it to warm-start a shape-conditioned lift policy from `lift-v3`.
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


def widen_inputs(layers: JaxLayers, n_in: int) -> JaxLayers:
    """`layers` for `n_in` inputs: the first weight gets zero columns for the new inputs, appended last.

    The other layers and every bias stay as they are, so the layers compute the same for any value
    of the new inputs until training moves their weights.
    """
    (weight, bias), rest = layers[0], layers[1:]
    extra = n_in - weight.shape[1]
    if extra < 0:
        raise ValueError(f"the layers take {weight.shape[1]} inputs, more than {n_in}")
    weight = jnp.concatenate([weight, jnp.zeros((weight.shape[0], extra), weight.dtype)], axis=1)
    return [(weight, bias), *rest]


def widen_norm(norm: Norm, dim: int) -> Norm:
    """The normalizer for `dim` inputs: the new entries appended last with mean 0 and variance 1.

    That is `Norm.new` for the new entries; the count stays. It weighs what was seen against each new
    batch, so after a long training (lift-v3: 118 million observations) the new entries' statistics
    hardly move, and they go into the network about as they come.
    """
    extra = dim - norm.mean.shape[0]
    if extra < 0:
        raise ValueError(f"the normalizer has {norm.mean.shape[0]} entries, more than {dim}")
    return Norm(jnp.concatenate([norm.mean, jnp.zeros(extra, norm.mean.dtype)]),
                jnp.concatenate([norm.var, jnp.ones(extra, norm.var.dtype)]), norm.count, norm.clip)


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
