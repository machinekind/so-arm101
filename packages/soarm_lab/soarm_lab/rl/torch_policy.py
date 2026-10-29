"""The policy in torch, as the torch trainer saved it: the converter of `policy.pt` to `policy.npz`.

Needs `soarm-lab[torch]`, and only `soarm-lab convert` uses it: running a policy needs numpy only
(`policy.Policy`), and training is JAX (`ppo`, `jax_policy`). This module reads a torch checkpoint
(`policy.pt`, the format before `policy.npz`) into torch modules, turns them into that policy and
back, and writes the numpy file:

    soarm-lab convert workspace/policies/reach-x/policy.pt      # policy.npz and meta.json next to it

A torch checkpoint is `torch.save({"meta": asdict(PolicyMeta), "state": TorchPolicy.state_dict(),
"critic": critic.state_dict()})`. `policy.npz` keeps the same arrays: `nn.Linear`'s (n_out, n_in) weight
and its bias per layer, the normalizer's buffers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._torch import torch
from .policy import POLICY_FILE, TORCH_FILE, Layers, Policy, PolicyMeta

nn = torch.nn


def mlp(sizes: list[int], out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    layers.append(nn.Linear(sizes[-1], out))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """The observations' running mean and variance: updated during training (`update`), fixed after."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


class TorchPolicy(nn.Module):
    """`policy.Policy` as a torch module: what the torch trainer optimized and saved in `policy.pt`."""

    def __init__(self, meta: PolicyMeta):
        super().__init__()
        self.meta = meta
        self.norm = Normalizer(meta.obs_dim)
        self.actor = mlp([meta.obs_dim, *meta.hidden], meta.act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """The deterministic action (the distribution's mean), clipped to [-1, 1]."""
        x = torch.as_tensor(np.asarray(obs, np.float32), device=self.norm.mean.device)
        single = x.ndim == 1
        a = self(x[None] if single else x).clamp(-1.0, 1.0).cpu().numpy()
        return a[0] if single else a

    def to_numpy(self, critic: nn.Sequential | None = None) -> Policy:
        """The numpy policy with these weights (sharing `meta`), and the critic's if given."""
        pol = Policy(self.meta)
        pol.obs_mean, pol.obs_var = _array(self.norm.mean), _array(self.norm.var)
        pol.obs_count, pol.obs_clip = _array(self.norm.count)[()], np.float32(self.norm.clip)
        pol.actor = layers_of(self.actor)
        pol.critic = layers_of(critic) if critic is not None else None
        return pol

    @torch.no_grad()
    def load_numpy(self, pol: Policy) -> None:
        """The numpy policy's normalizer and actor into this module (on its device)."""
        for buf, value in ((self.norm.mean, pol.obs_mean), (self.norm.var, pol.obs_var),
                           (self.norm.count, pol.obs_count)):
            buf.copy_(torch.as_tensor(np.asarray(value, np.float32)))
        self.norm.clip = float(pol.obs_clip)
        load_layers(self.actor, pol.actor)

    @staticmethod
    def from_numpy(pol: Policy) -> TorchPolicy:
        tp = TorchPolicy(pol.meta)
        tp.load_numpy(pol)
        return tp.eval()


def _array(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=True)


def linears(seq: nn.Sequential) -> list[nn.Linear]:
    return [m for m in seq if isinstance(m, nn.Linear)]


def layers_of(seq: nn.Sequential) -> Layers:
    """The dense layers of an `mlp` as (weight (n_out, n_in), bias)."""
    return [(_array(lin.weight), _array(lin.bias)) for lin in linears(seq)]


@torch.no_grad()
def load_layers(seq: nn.Sequential, layers: Layers) -> None:
    """Dense layers (weight, bias) into an `mlp` of the same sizes."""
    lins = linears(seq)
    if len(lins) != len(layers):
        raise ValueError(f"{len(layers)} layers for an MLP of {len(lins)}")
    for lin, (weight, bias) in zip(lins, layers, strict=True):
        lin.weight.copy_(torch.as_tensor(np.asarray(weight, np.float32)))
        lin.bias.copy_(torch.as_tensor(np.asarray(bias, np.float32)))


def load_checkpoint(path: str | Path) -> tuple[TorchPolicy, nn.Sequential | None]:
    """A torch checkpoint (`policy.pt`): the policy and its critic (None in checkpoints saved without one)."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = PolicyMeta(**blob["meta"])
    tp = TorchPolicy(meta)
    tp.load_state_dict(blob["state"])
    critic = None
    if blob.get("critic") is not None:
        critic = mlp([meta.obs_dim, *meta.hidden], 1)
        critic.load_state_dict(blob["critic"])
    return tp.eval(), critic


def convert(path: str | Path, out: str | Path | None = None) -> Path:
    """A torch checkpoint (`policy.pt` or its folder) as `policy.npz` and `meta.json`.

    `out` is the `policy.npz` to write, by default next to the checkpoint; `meta.json` goes next to it,
    from the checkpoint's own meta. Returns the `policy.npz`.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TORCH_FILE
    tp, critic = load_checkpoint(path)
    return tp.to_numpy(critic).save(Path(out) if out is not None else path.with_name(POLICY_FILE))
