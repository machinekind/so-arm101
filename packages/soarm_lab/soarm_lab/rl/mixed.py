"""Task variants share one environment interface with equal numbers of worlds.

    env = MixedBatchEnv(task.variant_tasks(), 4096, device="cuda:0")
    obs = env.reset()                         # (N, obs_dim)
    obs, rew, done, info = env.step(action)   # action (N, act_dim)

Each variant needs its own `BatchEnv` because collision geometry is fixed within a batch.
Batches step in sequence. Arrays group worlds in variant order.
`info["variant"]` has shape (N,) and indexes `env.variants`.
`num_envs` must be divisible by the number of variants.
Requires the `soarm-lab[train]` extra.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..station import Station
from . import tasks as tk
from .batch import BatchEnv
from .jax_policy import jax, jnp
from .randomize import Randomization

#: Seed spacing gives each variant a separate random stream.
SEED_STRIDE = 1009

Array = jax.Array


class MixedBatchEnv:
    """Combine task variants with matching observation and action dimensions."""

    def __init__(self, tasks: Sequence[tk.TaskConfig], num_envs: int = 4096, *, station: Station | None = None,
                 randomization: Randomization | None = None, device: str = "cuda:0", seed: int = 0):
        tasks = list(tasks)
        if not tasks:
            raise ValueError("MixedBatchEnv needs at least one task variant")
        names = [task.variant for task in tasks]
        if None in names:
            raise ValueError(f"these tasks have no variants: {[t.name for t in tasks if t.variant is None]}")
        if len(set(names)) != len(names):
            raise ValueError(f"a variant repeats in {names}")
        num_envs = int(num_envs)
        if num_envs % len(tasks):
            raise ValueError(f"num_envs ({num_envs}) must divide by the number of variants ({len(tasks)})")
        station = station or Station.load()
        #: `info["variant"]` indexes these names.
        self.variants = names
        self.num_envs = num_envs
        self.per_variant = n = num_envs // len(tasks)
        self.envs = [BatchEnv(task, n, station=station, randomization=randomization, device=device,
                              seed=seed + SEED_STRIDE * i)
                     for i, task in enumerate(tasks)]
        first = self.envs[0]
        if any(e.obs_dim != first.obs_dim or e.act_dim != first.act_dim for e in self.envs):
            raise ValueError("the variants observe or act differently: "
                             f"{[(e.obs_dim, e.act_dim) for e in self.envs]}")
        self.device = first.device
        #: Each world has an int32 variant index in this (N,) array.
        self.variant = jax.device_put(jnp.repeat(jnp.arange(len(tasks), dtype=jnp.int32), n), self.device)

    def reset(self) -> Array:
        """Reset all worlds and return observations (N, obs_dim) grouped by variant."""
        return jnp.concatenate([env.reset() for env in self.envs])

    def step(self, action: Array) -> tuple[Array, Array, Array, dict[str, Array]]:
        """Step all worlds as in `BatchEnv.step` and add each world's variant index to `info`."""
        action = jax.device_put(jnp.asarray(action), self.device)
        n = self.per_variant
        out = [env.step(action[i * n:(i + 1) * n]) for i, env in enumerate(self.envs)]
        obs, rew, done = (jnp.concatenate([o[j] for o in out]) for j in range(3))
        info = {k: jnp.concatenate([o[3][k] for o in out]) for k in out[0][3]}
        info["variant"] = self.variant
        return obs, rew, done, info

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def act_dim(self) -> int:
        return self.envs[0].act_dim
