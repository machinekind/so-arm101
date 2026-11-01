"""Mixed task variants run on Warp's CPU device.

These tests require the `train` extra and compile each scene's physics kernels.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")

import jax.numpy as jnp  # noqa: E402

from soarm_lab.rl import tasks as tk  # noqa: E402
from soarm_lab.rl.mixed import MixedBatchEnv  # noqa: E402
from soarm_lab.rl.randomize import Randomization  # noqa: E402

#: Box and cylinder variants require different collision geometry.
VARIANTS = ("cuboid", "cylinder")
N = 8


@pytest.fixture(scope="module")
def task(station):
    return tk.make_task("lift", spec=station.spec(), variants=VARIANTS, block="cuboid")


@pytest.fixture(scope="module")
def env(task, station):
    return MixedBatchEnv(task.variant_tasks(), N, station=station, device="cpu", seed=3)


def test_the_variants_step_as_one_environment(env, task):
    assert env.variants == list(VARIANTS) and env.num_envs == N and env.per_variant == N // 2
    assert env.obs_dim == 33 + 5 == tk.OBS_DIMS["lift"] + 5 and env.act_dim == 6
    assert [e.task.block for e in env.envs] == list(VARIANTS)
    obs = env.reset()
    assert isinstance(obs, jax.Array) and obs.shape == (N, env.obs_dim) and obs.devices() == {env.device}
    key = jax.random.key(0)
    for _ in range(10):
        key, k = jax.random.split(key)
        obs, rew, done, info = env.step(jax.random.uniform(k, (N, env.act_dim), minval=-1.0, maxval=1.0))
    assert obs.shape == (N, env.obs_dim) and rew.shape == (N,) and done.shape == (N,)
    assert done.dtype == jnp.bool_
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()
    assert set(info) == {"success", "time_outs", "broken", "finished", "height", "variant"}
    assert all(v.shape == (N,) for v in info.values())
    np.testing.assert_array_equal(info["variant"], [0, 0, 0, 0, 1, 1, 1, 1])
    assert info["variant"].dtype == jnp.int32
    # Each world receives its variant descriptor.
    want = np.stack([task.with_variant(VARIANTS[k]).descriptor() for k in np.asarray(info["variant"])])
    np.testing.assert_array_equal(np.asarray(obs[:, -5:]), want)
    # Each variant retains its own collision geometry.
    geoms = [int(e.scene.model.geom_type[e.cube_geoms[0]]) for e in env.envs]
    assert geoms == [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER]


def test_each_variant_draws_its_own_sizes(task, station):
    rand = replace(Randomization.none(), block_tolerance_mm=5.0)
    env = MixedBatchEnv(task.variant_tasks(), 4, station=station, randomization=rand, device="cpu", seed=3)
    obs = np.asarray(env.reset())
    sizes = [np.asarray(e.s["size"]) for e in env.envs]
    for i, (inner, size) in enumerate(zip(env.envs, sizes, strict=True)):
        half = np.asarray(inner.task.block_spec.half_size)
        assert (obs[2 * i:2 * i + 2, -3:] == inner.task.descriptor()[-3:]).all()
        assert (np.abs(2000.0 * half * (size - 1.0)) <= 5.0 + 1e-4).all()       # Side changes [mm].
    assert np.array_equal(sizes[1][:, 1], sizes[1][:, 2])                   # The cylinder stays round.
    assert not np.array_equal(sizes[0][:, 1], sizes[0][:, 2])
    obs, rew, done, info = env.step(jnp.zeros((4, env.act_dim)))
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()


def test_the_worlds_must_divide_by_the_variants(task, station):
    with pytest.raises(ValueError, match="must divide by the number of variants"):
        MixedBatchEnv(task.variant_tasks(), 9, station=station, device="cpu")


def test_the_tasks_must_be_variants(task, station):
    with pytest.raises(ValueError, match="at least one"):
        MixedBatchEnv([], 8, station=station, device="cpu")
    with pytest.raises(ValueError, match="a variant repeats"):
        MixedBatchEnv([task, task], 8, station=station, device="cpu")
    with pytest.raises(ValueError, match="have no variants"):
        MixedBatchEnv([tk.make_task("reach")], 8, station=station, device="cpu")
    with pytest.raises(ValueError, match="observe or act differently"):
        MixedBatchEnv([task, tk.make_task("lift", spec=station.spec())], 8, station=station, device="cpu")
