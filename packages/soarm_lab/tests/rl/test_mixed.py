"""`mixed.MixedBatchEnv`: one batch environment per wooden block behind one environment's interface.

On Warp's CPU device (`device="cpu"`), like the CPU tests of rl/test_batch.py; the first run compiles
the kernels of each block's scene. Skipped without the `train` extra (jax, mujoco-warp).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed (the `train` extra)")

import jax.numpy as jnp  # noqa: E402

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.rl.mixed import MixedBatchEnv  # noqa: E402
from soarm_lab.rl.objects import DESCRIPTOR_DIM, descriptor  # noqa: E402

#: Two kinds with different collision functions: box and cylinder.
KINDS = ("cuboid5", "cylinder")
N = 8


@pytest.fixture(scope="module")
def env(station):
    task = replace(tk.make_task("lift", spec=station.spec()), object_kind="cuboid5", shape_conditioned=True)
    return MixedBatchEnv(task, N, kinds=KINDS, station=station, device="cpu", seed=3)


def test_the_kinds_step_as_one_environment(env):
    assert env.kinds == KINDS and env.num_envs == N and env.per_kind == N // 2
    assert env.obs_dim == 40 == tk.OBS_DIMS["lift"] + DESCRIPTOR_DIM and env.act_dim == 6
    assert [e.task.object_kind for e in env.envs] == list(KINDS)
    assert env.task.object_kind == "cuboid5"                        # the policy's own kind, as given
    obs = env.reset()
    assert isinstance(obs, jax.Array) and obs.shape == (N, 40) and obs.devices() == {env.device}
    key = jax.random.key(0)
    for _ in range(10):
        key, k = jax.random.split(key)
        obs, rew, done, info = env.step(jax.random.uniform(k, (N, env.act_dim), minval=-1.0, maxval=1.0))
    assert obs.shape == (N, 40) and rew.shape == (N,) and done.shape == (N,) and done.dtype == jnp.bool_
    assert jnp.isfinite(obs).all() and jnp.isfinite(rew).all() and not info["broken"].any()
    assert set(info) == {"success", "time_outs", "broken", "finished", "height", "kind"}
    assert all(v.shape == (N,) for v in info.values())
    np.testing.assert_array_equal(info["kind"], [0, 0, 0, 0, 1, 1, 1, 1])
    assert info["kind"].dtype == jnp.int32
    # Every world's observation ends with its own block's descriptor.
    want = np.stack([descriptor(KINDS[k]) for k in np.asarray(info["kind"])])
    np.testing.assert_array_equal(np.asarray(obs[:, -DESCRIPTOR_DIM:]), want)
    # The kinds are different scenes: a box geom and a cylinder geom.
    geoms = [int(e.scene.model.geom_type[e.cube_geoms[0]]) for e in env.envs]
    assert geoms == [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER]


def test_the_worlds_must_divide_by_the_kinds(station):
    task = replace(tk.make_task("lift"), shape_conditioned=True)
    with pytest.raises(ValueError, match="must divide by the number of object kinds"):
        MixedBatchEnv(task, 9, kinds=KINDS, station=station, device="cpu")
    with pytest.raises(ValueError, match="unknown object kind 'pyramid'"):
        MixedBatchEnv(task, 8, kinds=("cube", "pyramid"), station=station, device="cpu")
    with pytest.raises(ValueError, match="has no object"):
        MixedBatchEnv(tk.make_task("reach"), 8, kinds=KINDS, station=station, device="cpu")
