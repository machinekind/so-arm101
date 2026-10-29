"""Several object kinds in one training run: one `batch.BatchEnv` per kind behind one env's interface.

    env = MixedBatchEnv(task, 4096, kinds=("cube", "cuboid5", "cuboid85", "cylinder"), device="cuda:0")
    obs = env.reset()                         # (4096, obs_dim): 1024 worlds per kind, in the order of `kinds`
    obs, rew, done, info = env.step(action)   # info["kind"] (4096,): each world's index into `kinds`

This is the environment of a shape-conditioned training (`soarm-lab train --kinds ...`): one policy,
one PPO minibatch, every kind at once. The task is shape-conditioned (`TaskConfig.shape_conditioned`),
so every world's observation ends with its object's descriptor (`objects.descriptor`); the kind's
yaw fold comes from the object table inside each `BatchEnv`.

Why one environment per kind: a `BatchEnv` is one compiled MuJoCo model copied into every world.
Randomization changes model fields per world (masses, friction: `put_model(batch_sizes=)`), not the
geometry: the geoms, their types and parts, and with them the collision functions MuJoCo Warp
compiles and launches, are the model's. A cylinder needs other collision kernels than a box, and
Warp cannot mix the geoms of different objects in one model without compiling another. So each kind
gets its own `BatchEnv` (its own scene, contact budget and captured CUDA graph). This class splits
the action by kind, steps each environment, and concatenates what they return. The environments
step one after the other; each has `num_envs // len(kinds)` worlds, so `num_envs` must divide by the
number of kinds.

Needs the `soarm-lab[train]` extra, as `batch`.
"""

from __future__ import annotations

from dataclasses import replace

from ..station import Station
from . import task as tk
from .batch import BatchEnv
from .jax_policy import jax, jnp
from .objects import KINDS, object_spec
from .randomize import Randomization

#: The seed offset between the kinds' environments (a prime, so the kinds' draws do not line up).
SEED_STRIDE = 1009

Array = jax.Array


class MixedBatchEnv:
    """`len(kinds)` batch environments of one task, one per object kind, as one environment."""

    def __init__(self, task: str | tk.TaskConfig = "lift", num_envs: int = 4096, *,
                 kinds: tuple[str, ...] | list[str] = KINDS, station: Station | None = None,
                 randomization: Randomization | None = None, device: str = "cuda:0", seed: int = 0):
        kinds = tuple(kinds)
        if not kinds:
            raise ValueError("MixedBatchEnv needs at least one object kind")
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"an object kind repeats in {kinds}")
        for kind in kinds:
            object_spec(kind)                          # an unknown kind raises with the known ones
        num_envs = int(num_envs)
        if num_envs % len(kinds):
            raise ValueError(f"num_envs ({num_envs}) must divide by the number of object kinds ({len(kinds)})")
        station = station or Station.load()
        base = task if isinstance(task, tk.TaskConfig) else tk.make_task(task, spec=station.spec())
        if base.cube is None:
            raise ValueError(f"the task {base.name} has no object to vary")
        #: The task as given: its `object_kind` is the policy's own (the one `meta.task` saves).
        self.task = base
        self.kinds = kinds
        self.num_envs = num_envs
        #: The worlds of each kind: world i is of kind `kinds[i // per_kind]`.
        self.per_kind = n = num_envs // len(kinds)
        #: One `BatchEnv` per kind, in the order of `kinds`.
        self.envs = [BatchEnv(replace(base, object_kind=kind), n, station=station, randomization=randomization,
                              device=device, seed=seed + SEED_STRIDE * i)
                     for i, kind in enumerate(kinds)]
        first = self.envs[0]
        if any(e.obs_dim != first.obs_dim or e.act_dim != first.act_dim for e in self.envs):
            raise ValueError("the kinds' environments observe or act differently: "
                             f"{[(e.obs_dim, e.act_dim) for e in self.envs]}")
        self.device = first.device
        #: Each world's index into `kinds` (N,) int32, the `info["kind"]` of every step.
        self.kind = jax.device_put(jnp.repeat(jnp.arange(len(kinds), dtype=jnp.int32), n), self.device)

    def reset(self) -> Array:
        """New episodes in every world: the observations (N, obs_dim), the kinds one after another."""
        return jnp.concatenate([env.reset() for env in self.envs])

    def step(self, action: Array) -> tuple[Array, Array, Array, dict[str, Array]]:
        """One policy tick in every world: (obs, rew, done, info), as `BatchEnv.step`, plus `info["kind"]`."""
        action = jax.device_put(jnp.asarray(action), self.device)
        n = self.per_kind
        out = [env.step(action[i * n:(i + 1) * n]) for i, env in enumerate(self.envs)]
        obs, rew, done = (jnp.concatenate([o[j] for o in out]) for j in range(3))
        info = {k: jnp.concatenate([o[3][k] for o in out]) for k in out[0][3]}
        info["kind"] = self.kind
        return obs, rew, done, info

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def act_dim(self) -> int:
        return self.envs[0].act_dim
