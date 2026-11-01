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


