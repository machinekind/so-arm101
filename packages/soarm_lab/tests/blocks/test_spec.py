"""Block geometry, scene placement, and lowest-point height."""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from soarm_lab import blocks  # noqa: E402
from soarm_lab.blocks import WOOD, BlockSpec, block, bottom_height, box_block, cylinder_block, scene_object  # noqa: E402
from soarm_lab.kinematics import inverse  # noqa: E402
from soarm_lab.station import Part  # noqa: E402

#: Include all catalog entries and the default cube.
NAMES = ("cube3cm", *blocks.classic_world.BLOCKS)
#: Number of sampled rotations for each geometry check.
ROTATIONS = 20
#: Rim samples bound the lowest-point error by r * (1 - cos(pi / RIM)).
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


def _scene(station, spec: BlockSpec):
    model = station.build(objects=[scene_object(spec, "cube", (0.2, 0.1))], grasp_sensors=["cube"])
    data = mujoco.MjData(model)
    body = model.body("cube").id
    return model, data, body, model.jnt_qposadr[model.body_jntadr[body]]


def _pose(station, data, body) -> tuple[np.ndarray, np.ndarray]:
    """Return the body pose in the base frame."""
    T = inverse(station.T_base2world)
    return T[:3, :3] @ data.xpos[body] + T[:3, 3], T[:3, :3] @ data.xmat[body].reshape(3, 3)


def test_the_constructors_take_millimetres_and_grams():
    box = box_block("b", (50.0, 30.0, 20.0), 12.0, yaw_period=math.pi)
    assert box.half_size == pytest.approx((0.025, 0.015, 0.01))
    assert box.mass == pytest.approx(0.012)
    assert [p.kind for p in box.parts] == ["box"]
    assert box.parts[0].size == box.half_size
    cyl = cylinder_block("c", 50.0, 20.0, 12.0, yaw_period=math.pi)
    assert cyl.half_size == pytest.approx((0.025, 0.01, 0.01))
    assert [p.kind for p in cyl.parts] == ["cylinder"]
    assert cyl.parts[0].size == pytest.approx((0.01, 0.025, 0.0))        # radius, half length


def test_a_spec_is_frozen():
    with pytest.raises(AttributeError):
        block("cube").mass = 1.0


def test_scene_object_is_a_box_of_the_spec():
    spec = block("cuboid")
    box = scene_object(spec, "thing", (0.1, 0.2))
    assert (box.name, box.size, box.xy, box.rgba, box.mass, box.parts) == (
        "thing", spec.half_size, (0.1, 0.2), WOOD, spec.mass, spec.parts)
    assert box.free and not box.mocap


@pytest.mark.parametrize("name", NAMES)
def test_the_half_size_is_the_bounding_box_of_the_parts(name):
    """Each body-axis half extent matches the lowest point when that axis is vertical."""
    spec = block(name)
    down = {0: _rot(1, np.pi / 2), 1: _rot(0, np.pi / 2), 2: np.eye(3)}      # body x / y / z vertical
    for axis, R in down.items():
        low = bottom_height(np, spec, np.zeros((1, 3)), R[None])[0]
        assert low == pytest.approx(-spec.half_size[axis], abs=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_every_block_settles_on_the_table(station, name):
    """A dropped block settles on the table with finite state and retains its initial roll and pitch."""
    spec = block(name)
    model, data, body, a = _scene(station, spec)
    mujoco.mj_forward(model, data)
    data.qpos[a + 2] += 0.005
    data.qpos[a + 3:a + 7] = [np.cos(0.35), 0.0, 0.0, np.sin(0.35)]
    mujoco.mj_step(model, data, nstep=int(round(0.5 / model.opt.timestep)))
    mujoco.mj_kinematics(model, data)
    pos, rot = _pose(station, data, body)
    v = model.jnt_dofadr[model.body_jntadr[body]]
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert abs(bottom_height(np, spec, pos[None], rot[None])[0]) < 0.001
    assert np.linalg.norm(data.qvel[v:v + 3]) < 0.001                  # [m/s]
    assert rot[2, 2] == pytest.approx(1.0, abs=1e-3)                   # Initial roll and pitch are preserved.


def _two_parts() -> BlockSpec:
    """Build a compound body with offset box and cylinder parts."""
    axis_x = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
    parts = (Part("box", (0.0125, 0.0125, 0.0125), pos=(-0.0175, 0.002, -0.001)),
             Part("cylinder", (0.0125, 0.0175, 0.0), pos=(0.0125, -0.002, 0.001), quat=axis_x))
    return BlockSpec("two", (0.03, 0.0145, 0.0135), 0.02, parts, np.pi)


@pytest.mark.parametrize("spec", [block(name) for name in NAMES] + [_two_parts()], ids=lambda s: s.name)
def test_bottom_height_is_the_lowest_point_of_the_geoms(station, spec):
    """Compare height with box corners and cylinder rim samples from compiled geoms."""
    model, data, body, a = _scene(station, spec)
    T = inverse(station.T_base2world)
    geoms = np.flatnonzero(model.geom_bodyid == body)
    assert len(geoms) == len(spec.parts)
    t = np.linspace(0.0, 2 * np.pi, RIM, endpoint=False)
    rng = np.random.default_rng(len(spec.name))
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
        assert bottom_height(np, spec, pos[None], rot[None])[0] == pytest.approx(lowest, abs=1e-7)


def test_numpy_and_jax_compute_the_same_bottom_height():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(0)
    rot = _random_rotations(rng, 64)
    pos = rng.normal(0.0, 0.1, (64, 3))
    for spec in [block(name) for name in NAMES] + [_two_parts()]:
        a = bottom_height(np, spec, pos, rot)
        b = bottom_height(jnp, spec, jnp.asarray(pos, jnp.float32), jnp.asarray(rot, jnp.float32))
        assert a.shape == b.shape == (64,) and b.dtype == np.float32
        np.testing.assert_allclose(np.asarray(b), a, rtol=1e-5, atol=1e-6)


def test_a_part_without_a_height_is_refused():
    spec = BlockSpec("ball", (0.01,) * 3, 0.01, (Part("sphere", (0.01, 0.0, 0.0)),), np.pi)
    with pytest.raises(ValueError, match="no height for a 'sphere' part"):
        bottom_height(np, spec, np.zeros((1, 3)), np.eye(3)[None])
