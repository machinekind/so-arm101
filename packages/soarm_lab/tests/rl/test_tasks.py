"""Task ownership and metadata compatibility."""

from dataclasses import asdict, fields

import numpy as np
import pytest

from soarm_lab.blocks import block
from soarm_lab.rl import tasks
from soarm_lab.rl.policy import Policy, PolicyMeta, find_policy


@pytest.mark.parametrize("name", tasks.TASKS)
def test_old_flat_metadata_ignores_unowned_fields(name):
    full = {}
    for other in tasks.TASKS:
        full.update(asdict(tasks.make_task(other)))
    full.update(asdict(tasks.make_task(name)))
    full = {k: list(v) if isinstance(v, tuple) else v for k, v in full.items()}
    task = tasks.task_from_dict(full)
    assert task == tasks.make_task(name)
    meta = PolicyMeta(full, [], task.obs_dim, task.act_dim)
    assert meta.task_config() == task
    assert isinstance(asdict(task), dict)
    assert asdict(task)["name"] == name


@pytest.mark.parametrize("name", tasks.TASKS)
def test_missing_fields_use_factory_defaults(name):
    assert tasks.task_from_dict({"name": name}) == tasks.make_task(name)


@pytest.mark.parametrize("data", [{}, {"name": None}, {"name": "unknown"}])
def test_missing_or_unknown_names_are_rejected(data):
    with pytest.raises(ValueError, match="task name"):
        tasks.task_from_dict(data)


def test_task_configs_own_only_their_fields():
    reach = {f.name for f in fields(tasks.ReachConfig)}
    lift = {f.name for f in fields(tasks.LiftConfig)}
    place = {f.name for f in fields(tasks.PickPlaceConfig)}
    assert "cube_half" not in reach | lift | place
    assert "block" not in reach and "variants" not in reach
    assert "block" in lift and "variants" in place
    assert "goal_radius" not in lift | place
    assert "target_radius" not in reach | lift
    assert "target_radius" in place


@pytest.mark.parametrize("name", ["reach-v3", "lift-v3"])
def test_bundled_metadata_loads(name):
    policy = Policy.load(find_policy(name))
    assert policy.task == tasks.task_from_dict(policy.meta.task)
    assert policy.task.obs_dim == policy.meta.obs_dim
    assert policy.task.act_dim == policy.meta.act_dim


@pytest.mark.parametrize("name", ["lift", "pickplace"])
def test_a_stored_3cm_cube_loads_as_its_block(name):
    data = {"name": name, "cube_half": 0.015, "cube_mass": 0.03}
    assert tasks.task_from_dict(data) == tasks.make_task(name, block="cube3cm")


@pytest.mark.parametrize("name", ["lift", "pickplace"])
@pytest.mark.parametrize("stored", [{"cube_half": 0.02, "cube_mass": 0.03},
                                    {"cube_half": 0.015, "cube_mass": 0.05},
                                    {"cube_half": 0.02}, {"cube_mass": 0.05}])
def test_another_stored_cube_is_refused(name, stored):
    with pytest.raises(ValueError, match="no block to load it as"):
        tasks.task_from_dict({"name": name, **stored})


def test_the_stored_cube_fields_mean_nothing_for_reach():
    assert tasks.task_from_dict({"name": "reach", "cube_half": 0.02}) == tasks.make_task("reach")


def test_a_stored_block_and_variants_load():
    task = tasks.make_task("lift", variants=("cube", "cylinder"), block="cylinder")
    stored = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(task).items()}
    assert tasks.task_from_dict(stored) == task


def test_the_default_block_is_the_3cm_cube():
    for name in ("lift", "pickplace"):
        task = tasks.make_task(name)
        assert (task.block, task.variants) == ("cube3cm", ())
        spec = task.block_spec
        assert (spec.half_size, spec.mass, spec.yaw_period) == ((0.015,) * 3, 0.03, np.pi / 2)
    with pytest.raises(ValueError, match="unknown block 'arch'"):
        tasks.make_task("lift", block="arch")
    with pytest.raises(ValueError, match="unknown block 'arch'"):
        tasks.make_task("lift", variants=("cube", "arch"), block="cube")
    with pytest.raises(TypeError):
        tasks.make_task("reach", variants=("cube",))


def test_variants_must_hold_the_block_once():
    with pytest.raises(ValueError, match="not one of the variants"):
        # Variants must include the selected block.
        tasks.make_task("lift", variants=("cube", "cylinder"))
    with pytest.raises(ValueError, match="a variant repeats"):
        tasks.make_task("lift", variants=("cube", "cube"), block="cube")


@pytest.mark.parametrize("name", ["lift", "pickplace"])
def test_the_observation_grows_by_the_variants_and_the_half_sizes(name):
    plain = tasks.make_task(name)
    for variants in (("cube",), ("cube", "cylinder"), ("cube", "cuboid", "cylinder")):
        task = tasks.make_task(name, variants=variants, block="cube")
        assert task.obs_dim == plain.obs_dim + len(variants) + 3


def test_the_descriptor_is_the_one_hot_and_the_scaled_half_sizes():
    variants = ("cube", "cuboid", "cylinder")
    for i, name in enumerate(variants):
        d = tasks.make_task("lift", variants=variants, block=name).descriptor()
        assert d.dtype == np.float32 and d.shape == (len(variants) + 3,)
        assert d[: len(variants)].tolist() == [float(k == i) for k in range(len(variants))]
        np.testing.assert_allclose(d[len(variants):], np.asarray(block(name).half_size) / 0.05, rtol=1e-6)


def test_the_descriptor_follows_the_tasks_own_variants():
    """Descriptor indices follow the task variant order."""
    d = tasks.make_task("lift", variants=("cylinder", "cube"), block="cube").descriptor()
    assert d[:2].tolist() == [0.0, 1.0] and d.shape == (5,)


def test_variant_tasks_and_with_variant():
    task = tasks.make_task("lift", variants=("cube", "cylinder"), block="cylinder")
    assert task.variant == "cylinder"
    assert [t.block for t in task.variant_tasks()] == ["cube", "cylinder"]
    assert all(t.variants == task.variants for t in task.variant_tasks())
    with pytest.raises(ValueError, match="'cuboid' is not a variant"):
        task.with_variant("cuboid")
    assert tasks.make_task("lift").with_variant("cuboid").block == "cuboid"
    with pytest.raises(ValueError, match="unknown block"):
        tasks.make_task("lift").with_variant("mug")
    assert tasks.make_task("reach").variant is None


STANDING_BLOCKS = ("cuboid", "cylinder")


def test_standing_starts_are_off_by_default_and_old_files_load_with_zero():
    for name in ("lift", "pickplace"):
        assert tasks.make_task(name).standing_fraction == 0.0
        assert tasks.task_from_dict({"name": name}).standing_fraction == 0.0
        assert not tasks.make_task(name, block="cuboid").stands
    assert "standing_fraction" not in Policy.load(find_policy("lift-v3")).meta.task
    assert Policy.load(find_policy("lift-v3")).task.standing_fraction == 0.0
    with pytest.raises(TypeError):
        tasks.make_task("reach", standing_fraction=0.5)


def test_a_stored_standing_fraction_loads():
    task = tasks.make_task("pickplace", block="cylinder", standing_fraction=0.25)
    stored = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(task).items()}
    assert tasks.task_from_dict(stored) == task
    assert all(t.standing_fraction == 0.25 for t in tasks.make_task(
        "lift", variants=("cube", "cuboid"), block="cube", standing_fraction=0.25).variant_tasks())


@pytest.mark.parametrize("name", ["lift", "pickplace"])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_the_standing_fraction_must_be_a_probability(name, value):
    with pytest.raises(ValueError, match="standing_fraction must be within"):
        tasks.make_task(name, standing_fraction=value)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_the_standing_fraction_accepts_its_range(value):
    assert tasks.make_task("lift", standing_fraction=value).standing_fraction == value


def test_only_a_block_that_can_stand_takes_the_new_path(square_end):
    periods = {"cube3cm": None, "cube": None, "cuboid": np.pi, square_end: np.pi / 2, "cylinder": 0.0}
    for name, period in periods.items():
        task = tasks.make_task("lift", block=name, standing_fraction=0.25)
        assert task.end_period == period and task.stands == (period is not None)
        assert tasks.make_task("lift", block=name).end_period is None


def _old_sample_cubes(task, rng, n):
    """Reproduce reset sampling before standing starts were added."""
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.block_spec.half_size[2])], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    return pos, np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)


@pytest.mark.parametrize("name,fraction", [("cube3cm", 0.0), ("cube", 0.0), ("cuboid", 0.0),
                                           ("cylinder", 0.0), ("cube3cm", 1.0), ("cube", 0.5)])
def test_a_reset_that_cannot_stand_draws_and_computes_as_before(name, fraction):
    task = tasks.make_task("lift", block=name, standing_fraction=fraction)
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    got, want = tasks.sample_cubes(task, a, 8), _old_sample_cubes(task, b, 8)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    assert a.bit_generator.state == b.bit_generator.state


def test_standing_draws_after_the_shared_draws():
    n = 64
    task = tasks.make_task("lift", block="cuboid", standing_fraction=0.5)
    a, b = np.random.default_rng(12), np.random.default_rng(12)
    pos, quat = tasks.sample_cubes(task, a, n)
    plain_pos, plain_quat = _old_sample_cubes(task, b, n)
    flags = b.uniform(size=n) < 0.5
    assert a.bit_generator.state == b.bit_generator.state
    assert 0 < flags.sum() < n
    assert np.array_equal(pos[:, :2], plain_pos[:, :2])
    half = task.block_spec.half_size
    assert pos[:, 2].tolist() == np.where(flags, half[0], half[2]).tolist()
    assert np.array_equal(quat[~flags], plain_quat[~flags])
    w, x, y, z = quat.T
    body_x_z = 2.0 * (x * z - w * y)
    np.testing.assert_allclose(body_x_z, np.where(flags, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(np.arctan2(x, w)[flags], np.arctan2(plain_quat[:, 3], plain_quat[:, 0])[flags])


@pytest.mark.parametrize("fraction,lying,standing", [(1.0, False, True), (0.0, True, False)])
def test_the_ends_of_the_standing_fraction(fraction, lying, standing):
    for name in STANDING_BLOCKS:
        task = tasks.make_task("lift", block=name, standing_fraction=fraction)
        pos, _ = tasks.sample_cubes(task, np.random.default_rng(13), 32)
        half = task.block_spec.half_size
        assert (pos[:, 2] == half[0]).all() == standing and (pos[:, 2] == half[2]).all() == lying


def test_the_sampled_height_scales_with_the_world():
    task = tasks.make_task("lift", block="cuboid", standing_fraction=1.0)
    size = np.random.default_rng(14).uniform(0.8, 1.2, (8, 3))
    pos, _ = tasks.sample_cubes(task, np.random.default_rng(15), 8, size)
    assert pos[:, 2].tolist() == (task.block_spec.half_size[0] * size[:, 0]).tolist()
    lying = tasks.make_task("lift", block="cuboid")
    pos, _ = tasks.sample_cubes(lying, np.random.default_rng(15), 8, size)
    assert pos[:, 2].tolist() == (lying.block_spec.half_size[2] * size[:, 2]).tolist()
