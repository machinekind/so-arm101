"""Block geometry, scene placement, and lowest-point height."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from soarm_lab import blocks  # noqa: E402
from soarm_lab.blocks import (  # noqa: E402
    WOOD,
    BlockSpec,
    Geometry,
    block,
    bottom_height,
    box_block,
    can_stand,
    check_scalable,
    cylinder_block,
    end_yaw_period,
    is_on_end,
    part_scale,
    quat_on_end,
    rest_height,
    rotation_on_end,
    scale_from_mm,
    scaled_geometry,
    scene_object,
    symmetric_scale,
    yaw_on_end,
)
from soarm_lab.kinematics import inverse  # noqa: E402
from soarm_lab.station import Box, Part  # noqa: E402

#: Include all catalog entries and the default cube.
NAMES = ("cube3cm", *blocks.classic_world.BLOCKS)
#: Number of sampled rotations for each geometry check.
ROTATIONS = 20
#: Rim samples bound the lowest-point error by r * (1 - cos(pi / RIM)).
RIM = 3600
#: A turn of 45 degrees about z.
_TURNED = (float(np.cos(np.pi / 8)), 0.0, 0.0, float(np.sin(np.pi / 8)))
#: Body-axis scale factors for each test block. Cylinder radial scales must match.
SCALES = {"cube3cm": (0.9, 1.1, 1.2), "cube": (1.15, 0.85, 1.05), "cuboid": (0.85, 1.1, 1.15),
          "cylinder": (0.88, 1.13, 1.13)}


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


def _scaled_object(spec: BlockSpec, scale) -> Box:
    """Build an independent reference object at `scale` with constant density."""
    half = tuple(float(h * s) for h, s in zip(spec.half_size, scale, strict=True))
    part = spec.parts[0]
    if part.kind == "box":
        parts = (Part("box", half),)
    else:
        parts = (Part("cylinder", (half[1], half[0], 0.0), quat=part.quat),)
    return Box("cube", half, (0.2, 0.1), rgba=WOOD, mass=spec.mass * float(np.prod(scale)), parts=parts)


def _scene(station, spec: BlockSpec, scale=None):
    obj = scene_object(spec, "cube", (0.2, 0.1)) if scale is None else _scaled_object(spec, scale)
    model = station.build(objects=[obj], grasp_sensors=["cube"])
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


#: Nominal blocks, a compound block and scaled blocks for the lowest-point checks.
LOWEST_CASES = [*((block(name), None) for name in NAMES), (_two_parts(), None),
                *((block(name), SCALES[name]) for name in NAMES)]


@pytest.mark.parametrize("spec,scale", LOWEST_CASES,
                         ids=[f"{s.name}-{'scaled' if c else 'own'}" for s, c in LOWEST_CASES])
def test_bottom_height_is_the_lowest_point_of_the_geoms(station, spec, scale):
    """Compare height with box corners and cylinder rim samples from compiled geoms."""
    model, data, body, a = _scene(station, spec, scale)
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
        s = None if scale is None else np.asarray(scale)[None]
        assert bottom_height(np, spec, pos[None], rot[None], s)[0] == pytest.approx(lowest, abs=1e-7)


def test_numpy_and_jax_compute_the_same_bottom_height():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(0)
    rot = _random_rotations(rng, 64)
    pos = rng.normal(0.0, 0.1, (64, 3))
    scale = rng.uniform(0.8, 1.2, (64, 3))
    for spec in [block(name) for name in NAMES] + [_two_parts()]:
        a = bottom_height(np, spec, pos, rot)
        b = bottom_height(jnp, spec, jnp.asarray(pos, jnp.float32), jnp.asarray(rot, jnp.float32))
        assert a.shape == b.shape == (64,) and b.dtype == np.float32
        np.testing.assert_allclose(np.asarray(b), a, rtol=1e-5, atol=1e-6)
        a = bottom_height(np, spec, pos, rot, scale)
        b = bottom_height(jnp, spec, jnp.asarray(pos, jnp.float32), jnp.asarray(rot, jnp.float32),
                          jnp.asarray(scale, jnp.float32))
        assert a.shape == b.shape == (64,) and b.dtype == np.float32
        np.testing.assert_allclose(np.asarray(b), a, rtol=1e-5, atol=1e-6)


def test_a_part_without_a_height_is_refused():
    spec = BlockSpec("ball", (0.01,) * 3, 0.01, (Part("sphere", (0.01, 0.0, 0.0)),), np.pi)
    with pytest.raises(ValueError, match="no height for a 'sphere' part"):
        bottom_height(np, spec, np.zeros((1, 3)), np.eye(3)[None])


def test_part_scale_maps_the_body_axes_onto_the_part():
    """Body-axis scales map to the part frame in both NumPy and JAX."""
    s = np.array([[1.1, 0.9, 1.2], [0.8, 1.0, 1.3]])
    box, cyl = block("cuboid").parts[0], block("cylinder").parts[0]
    assert np.array_equal(part_scale(box, s), s)
    assert np.array_equal(part_scale(cyl, s), s[:, ::-1])
    assert part_scale(cyl, s[0]).tolist() == [1.2, 0.9, 1.1]
    jnp = pytest.importorskip("jax.numpy")
    assert np.array_equal(np.asarray(part_scale(cyl, jnp.asarray(s, jnp.float32))),
                          s[:, ::-1].astype(np.float32))
    with pytest.raises(ValueError, match="not axis-aligned"):
        part_scale(Part("box", (0.01, 0.01, 0.01), quat=_TURNED), s)


def _compiled(model, body: int) -> dict[str, np.ndarray]:
    """Read geometry and inertia fields from the compiled reference block."""
    (g,) = np.flatnonzero(model.geom_bodyid == body)
    assert model.body_bvhnum[body] == 1
    return {"size": model.geom_size[g], "rbound": model.geom_rbound[g], "aabb": model.geom_aabb[g, 3:],
            "inertia": model.body_inertia[body], "bvh": model.bvh_aabb[model.body_bvhadr[body], 3:],
            "centre": np.r_[model.geom_aabb[g, :3], model.bvh_aabb[model.body_bvhadr[body], :3]]}


def _assert_matches(geo: Geometry, compiled: dict[str, np.ndarray]) -> None:
    for name in ("size", "rbound", "aabb", "inertia"):
        np.testing.assert_allclose(getattr(geo, name)[0], compiled[name], rtol=1e-9, atol=0, err_msg=name)
    np.testing.assert_allclose(geo.aabb[0], compiled["bvh"], rtol=1e-9, atol=0)
    assert not compiled["centre"].any()


@pytest.mark.parametrize("name", NAMES)
def test_scaled_geometry_at_scale_one_is_the_compiled_block(station, name):
    spec = block(name)
    model, _, body, _ = _scene(station, spec)
    _assert_matches(scaled_geometry(spec, np.ones((1, 3)), spec.mass), _compiled(model, body))
    assert model.body_mass[body] == pytest.approx(spec.mass, rel=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_scaled_geometry_is_the_compiled_scaled_block(station, name):
    """Computed geometry and inertia match a reference object compiled at the scaled dimensions."""
    spec = block(name)
    scale = np.asarray(SCALES[name])
    mass = spec.mass * float(np.prod(scale))
    model, _, body, _ = _scene(station, spec, scale)
    _assert_matches(scaled_geometry(spec, scale[None], np.array([mass])), _compiled(model, body))
    assert model.body_mass[body] == pytest.approx(mass, rel=1e-12)
    nominal, _, b0, _ = _scene(station, spec)
    np.testing.assert_allclose(model.body_iquat[body], nominal.body_iquat[b0], atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_every_catalog_block_can_scale(name):
    check_scalable(block(name))


def test_scaled_geometry_refuses_what_it_cannot_scale():
    with pytest.raises(ValueError, match="radial scales must be equal"):
        scaled_geometry(block("cylinder"), [[1.0, 1.1, 0.9]], 0.02)
    with pytest.raises(ValueError, match="one-part block"):
        scaled_geometry(_two_parts(), [[1.0, 1.0, 1.0]], 0.02)
    cuboid, size, one = block("cuboid"), (0.01, 0.01, 0.01), [[1.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="centred on its body"):
        scaled_geometry(replace(cuboid, parts=(Part("box", size, pos=(0.01, 0.0, 0.0)),)), one, 0.02)
    with pytest.raises(ValueError, match="not axis-aligned"):
        scaled_geometry(replace(cuboid, parts=(Part("box", size, quat=_TURNED),)), one, 0.02)
    with pytest.raises(ValueError, match="no size rule for a 'sphere' part"):
        scaled_geometry(replace(cuboid, parts=(Part("sphere", size),)), one, 0.02)


def test_the_scale_keeps_the_symmetry_the_spec_declares():
    """Scale sharing follows part axes and yaw symmetry independently of the block name."""
    s = np.array([[1.1], [0.9], [1.2]])
    square = box_block("a", (30.0, 30.0, 50.0), 10.0, yaw_period=math.pi / 2)
    assert symmetric_scale(np, square, *s).tolist() == [[1.1, 1.1, 1.2]]
    cuboid = box_block("b", (50.0, 30.0, 20.0), 10.0, yaw_period=math.pi)
    assert symmetric_scale(np, cuboid, *s).tolist() == [[1.1, 0.9, 1.2]]
    assert symmetric_scale(np, replace(block("cube"), yaw_period=math.pi), *s).tolist() == [[1.1, 0.9, 1.2]]
    lying = cylinder_block("c", 50.0, 20.0, 10.0, yaw_period=math.pi)
    assert symmetric_scale(np, lying, *s).tolist() == [[1.1, 0.9, 0.9]]
    about_x = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    along_y = replace(lying, parts=(Part("cylinder", (0.01, 0.025, 0.0), quat=about_x),))
    assert symmetric_scale(np, along_y, *s).tolist() == [[1.1, 0.9, 1.1]]
    standing = replace(lying, parts=(Part("cylinder", (0.01, 0.025, 0.0)),))
    assert symmetric_scale(np, standing, *s).tolist() == [[1.1, 1.1, 1.2]]
    with pytest.raises(ValueError, match="one-part block"):
        symmetric_scale(np, _two_parts(), *s)


def test_side_changes_in_millimetres_scale_the_catalog_blocks_symmetrically():
    """Side changes [mm] give scale factors of 1 + d / L. Shared axes follow the block's symmetry."""
    t = 2.0
    d = np.array([[t, -t], [-t, t], [t, -t]])                   # (body axis, world) side changes [mm]
    # Expected sides [mm] share x/y changes for cubes and y/z changes for cylinders.
    want = {"cube": [[31.0, 31.0, 31.0], [27.0, 27.0, 27.0]],
            "cuboid": [[62.0, 27.0, 16.0], [58.0, 31.0, 12.0]],
            "cylinder": [[62.0, 27.0, 27.0], [58.0, 31.0, 31.0]]}
    assert set(want) == set(blocks.classic_world.BLOCKS)
    for name, sides in want.items():
        spec = block(name)
        nominal = 2000.0 * np.asarray(spec.half_size)               # [mm]
        scale = scale_from_mm(np, spec, *d)
        assert scale.shape == (2, 3)
        np.testing.assert_allclose(nominal * scale, sides, rtol=1e-12, err_msg=name)
        assert scale_from_mm(np, spec, *np.zeros((3, 2))).tolist() == [[1.0, 1.0, 1.0]] * 2
    cube, cylinder = scale_from_mm(np, block("cube"), *d), scale_from_mm(np, block("cylinder"), *d)
    assert np.array_equal(cube[:, 0], cube[:, 1]) and np.array_equal(cylinder[:, 1], cylinder[:, 2])


def test_a_block_can_stand_when_x_exceeds_z():
    assert [name for name in NAMES if can_stand(block(name))] == ["cuboid", "cylinder"]
    assert can_stand(box_block("b", (50.0, 30.0, 20.0), 10.0, yaw_period=math.pi))
    assert not can_stand(box_block("b", (30.0, 30.0, 30.0), 10.0, yaw_period=math.pi / 2))
    assert not can_stand(box_block("b", (20.0, 30.0, 40.0), 10.0, yaw_period=math.pi))


def test_the_end_symmetry_comes_from_the_shape_not_the_name():
    # Round ends look alike at every yaw.
    # Square box ends repeat every quarter turn. Other box ends repeat every half turn.
    period = {name: end_yaw_period(block(name)) for name in NAMES}
    assert period == {"cube3cm": math.pi / 2, "cube": math.pi / 2, "cuboid": math.pi, "cylinder": 0.0}
    square_end = box_block("b", (60.0, 25.0, 25.0), 10.0, yaw_period=math.pi)
    assert end_yaw_period(square_end) == math.pi / 2
    assert end_yaw_period(replace(square_end, name="cylinder")) == math.pi / 2
    assert end_yaw_period(replace(block("cuboid"), name="cylinder")) == math.pi
    assert end_yaw_period(replace(block("cylinder"), name="cuboid")) == 0.0
    oblong = box_block("b", (50.0, 30.0, 20.0), 10.0, yaw_period=math.pi / 2)
    assert end_yaw_period(oblong) == math.pi
    # With body x up, a transverse cylinder has a rectangular outline from above.
    across = replace(block("cylinder"), parts=(Part("cylinder", (0.0125, 0.025, 0.0)),))
    assert end_yaw_period(across) == math.pi
    assert end_yaw_period(_two_parts()) == 2.0 * math.pi
    off_centre = Part("box", (0.03, 0.01, 0.01), pos=(0.0, 0.002, 0.0))
    assert end_yaw_period(replace(block("cuboid"), parts=(off_centre,))) == 2.0 * math.pi


@pytest.mark.parametrize("name", NAMES)
def test_a_block_on_its_end_rests_on_the_table(name):
    spec = block(name)
    n = 8
    yaw = np.linspace(-3.0, 3.0, n)
    for scale in (np.ones((n, 3)), np.tile(SCALES[name], (n, 1))):
        z = rest_height(np, spec, scale, np.ones(n, bool))
        np.testing.assert_array_equal(z, spec.half_size[0] * scale[:, 0])
        pos, R = np.c_[np.zeros((n, 2)), z], rotation_on_end(np, yaw)
        np.testing.assert_allclose(bottom_height(np, spec, pos, R, scale), 0.0, atol=1e-15)
        lying = spec.half_size[2] * scale[:, 2]
        np.testing.assert_array_equal(rest_height(np, spec, scale), lying)
        np.testing.assert_array_equal(rest_height(np, spec, scale, np.zeros(n, bool)), lying)
        some = np.arange(n) % 2 == 0
        np.testing.assert_array_equal(rest_height(np, spec, scale, some), np.where(some, z, lying))


def test_the_pose_on_end_points_body_x_up():
    yaw = np.random.default_rng(4).uniform(-np.pi, np.pi, 16)
    R = rotation_on_end(np, yaw)
    np.testing.assert_allclose(R[:, :, 0], np.tile([0.0, 0.0, 1.0], (16, 1)), atol=1e-15)
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.tile(np.eye(3), (16, 1, 1)), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(yaw_on_end(np, R), yaw, atol=1e-14)
    np.testing.assert_allclose(R[0], _rot(2, yaw[0]) @ _rot(1, np.pi / 2), atol=1e-15)
    assert is_on_end(np, R).all()


def test_the_quaternion_on_end_is_the_rotation_on_end():
    yaw = np.random.default_rng(5).uniform(-np.pi, np.pi, 16)
    q = np.c_[np.cos(yaw / 2), np.zeros(16), np.zeros(16), np.sin(yaw / 2)]
    turned = quat_on_end(np, q)
    np.testing.assert_allclose(np.linalg.norm(turned, axis=1), 1.0)
    want = rotation_on_end(np, yaw)
    for k in range(16):
        got = np.zeros(9)
        mujoco.mju_quat2Mat(got, turned[k])
        np.testing.assert_allclose(got.reshape(3, 3), want[k], atol=1e-14)


def test_numpy_and_jax_compute_the_same_pose_on_end():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(6)
    n = 32
    yaw, on_end = rng.uniform(-np.pi, np.pi, n), rng.integers(0, 2, n).astype(bool)
    scale = rng.uniform(0.8, 1.2, (n, 3))
    q = rng.normal(size=(n, 4))
    f = jnp.float32

    def close(a, b):
        assert b.dtype == np.float32 and a.shape == b.shape
        np.testing.assert_allclose(np.asarray(b), a, rtol=1e-5, atol=1e-6)

    close(rest_height(np, block("cuboid"), scale, on_end),
          rest_height(jnp, block("cuboid"), jnp.asarray(scale, f), jnp.asarray(on_end)))
    close(rotation_on_end(np, yaw), rotation_on_end(jnp, jnp.asarray(yaw, f)))
    close(quat_on_end(np, q), quat_on_end(jnp, jnp.asarray(q, f)))
    R = _random_rotations(rng, n)
    assert np.array_equal(is_on_end(np, R), np.asarray(is_on_end(jnp, jnp.asarray(R, f))))
    close(yaw_on_end(np, R), yaw_on_end(jnp, jnp.asarray(R, f)))
