"""The wooden blocks: their table, how they lie in the bundled station, the height of their lowest point."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from soarm_lab.kinematics import inverse  # noqa: E402
from soarm_lab.rl import objects as ob  # noqa: E402

#: Random rotations per kind in the brute-force check of `bottom_height`.
ROTATIONS = 20
#: Points on each rim of a cylinder in the brute force: the lowest of them is at most
#: r * (1 - cos(pi / RIM)) = 5 nm above the true lowest point for a 12.5 mm radius.
RIM = 3600


def _rot(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = np.zeros((n, 9))
    for k in range(n):
        mujoco.mju_quat2Mat(out[k], q[k])
    return out.reshape(n, 3, 3)


def _scene(station, kind: str):
    model = station.build(objects=[ob.scene_object(kind, xy=(0.2, 0.1))], grasp_sensors=["cube"])
    data = mujoco.MjData(model)
    body = model.body("cube").id
    return model, data, body, model.jnt_qposadr[model.body_jntadr[body]]


def _pose(station, data, body) -> tuple[np.ndarray, np.ndarray]:
    """The body's position and rotation in the base frame."""
    T = inverse(station.T_base2world)
    return T[:3, :3] @ data.xpos[body] + T[:3, 3], T[:3, :3] @ data.xmat[body].reshape(3, 3)


def test_the_table_is_one_marked_place():
    """Every number is a placeholder until `MEASURED`; each spec comes from the table and is sane."""
    assert isinstance(ob.MEASURED, bool)
    assert tuple(ob.BLOCKS) == ob.KINDS == ("cube", "cuboid5", "cuboid85", "cylinder")
    for kind in ob.KINDS:
        spec = ob.object_spec(kind)
        _, size_mm, mass_g = ob.BLOCKS[kind]
        assert spec.kind == kind
        assert spec.mass == pytest.approx(mass_g / 1000) and spec.mass > 0
        assert spec.half_size == pytest.approx(np.asarray(size_mm) / 2000)
        assert spec.parts and all(p.kind in ("box", "cylinder") for p in spec.parts)
        assert spec.yaw_period in (np.pi / 2, np.pi)
    assert ob.object_spec("cube").yaw_period == np.pi / 2
    with pytest.raises(ValueError, match="unknown object kind 'arch'"):
        ob.object_spec("arch")


@pytest.mark.parametrize("kind", ob.KINDS)
def test_the_half_size_is_the_bounding_box_of_the_parts(kind):
    """With the body's x, y or z vertical, the lowest point is that half size below the origin."""
    spec = ob.object_spec(kind)
    down = {0: _rot(1, np.pi / 2), 1: _rot(0, np.pi / 2), 2: np.eye(3)}      # body x / y / z vertical
    for axis, R in down.items():
        low = ob.bottom_height(np, kind, np.zeros((1, 3)), R[None])[0]
        assert low == pytest.approx(-spec.half_size[axis], abs=1e-12)


def test_the_descriptor_is_the_kind_and_the_scaled_half_sizes():
    assert ob.DESCRIPTOR_DIM == len(ob.KINDS) + 3
    for i, kind in enumerate(ob.KINDS):
        d = ob.descriptor(kind)
        assert d.dtype == np.float32 and d.shape == (ob.DESCRIPTOR_DIM,)
        assert d[: len(ob.KINDS)].tolist() == [float(k == i) for k in range(len(ob.KINDS))]
        np.testing.assert_allclose(d[len(ob.KINDS):], np.asarray(ob.object_spec(kind).half_size) / 0.05,
                                   rtol=1e-6)


@pytest.mark.parametrize("kind", ob.KINDS)
def test_every_kind_settles_on_the_table(station, kind):
    """Dropped from 5 mm with a yaw, the block lies on the table within 0.5 s: its lowest point at the
    table top (MuJoCo's soft contact lets it sink 0.1-0.2 mm), still, no NaN. The cylinder lies on its
    side and stays there (see `station.CYLINDER_CCD_TOLERANCE`)."""
    model, data, body, a = _scene(station, kind)
    mujoco.mj_forward(model, data)
    data.qpos[a + 2] += 0.005
    data.qpos[a + 3:a + 7] = [np.cos(0.35), 0.0, 0.0, np.sin(0.35)]
    mujoco.mj_step(model, data, nstep=int(round(0.5 / model.opt.timestep)))
    mujoco.mj_kinematics(model, data)
    pos, rot = _pose(station, data, body)
    v = model.jnt_dofadr[model.body_jntadr[body]]
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert abs(ob.bottom_height(np, kind, pos[None], rot[None])[0]) < 0.001
    assert np.linalg.norm(data.qvel[v:v + 3]) < 0.001                  # [m/s]
    assert rot[2, 2] == pytest.approx(1.0, abs=1e-3)                   # still lying as it was put down


@pytest.mark.parametrize("kind", ob.KINDS)
def test_bottom_height_is_the_lowest_point_of_the_geoms(station, kind):
    """Against brute force on the COMPILED geoms: every corner of a box, points on both rims of a cylinder."""
    model, data, body, a = _scene(station, kind)
    T = inverse(station.T_base2world)
    geoms = np.flatnonzero(model.geom_bodyid == body)
    t = np.linspace(0.0, 2 * np.pi, RIM, endpoint=False)
    rng = np.random.default_rng(ob.KINDS.index(kind))
    for R in _random_rotations(rng, ROTATIONS):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        data.qpos[a + 3:a + 7] = q
        mujoco.mj_kinematics(model, data)
        lowest = np.inf
        for g in geoms:
            size = model.geom_size[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * size
            else:
                assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
                rim = np.c_[size[0] * np.cos(t), size[0] * np.sin(t)]
                local = np.r_[np.c_[rim, np.full(RIM, size[1])], np.c_[rim, np.full(RIM, -size[1])]]
            world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            lowest = min(lowest, float((world @ T[:3, :3].T + T[:3, 3])[:, 2].min()))
        pos, rot = _pose(station, data, body)
        assert ob.bottom_height(np, kind, pos[None], rot[None])[0] == pytest.approx(lowest, abs=1e-7)


def test_numpy_and_jax_compute_the_same_bottom_height():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(0)
    rot = _random_rotations(rng, 64)
    pos = rng.normal(0.0, 0.1, (64, 3))
    for kind in ob.KINDS:
        a = ob.bottom_height(np, kind, pos, rot)
        b = ob.bottom_height(jnp, kind, jnp.asarray(pos, jnp.float32), jnp.asarray(rot, jnp.float32))
        assert a.shape == b.shape == (64,) and b.dtype == np.float32
        np.testing.assert_allclose(np.asarray(b), a, rtol=1e-5, atol=1e-6)
