"""Task ownership and metadata compatibility."""

from dataclasses import asdict, fields

import pytest

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
    assert "cube_half" not in reach
    assert "goal_radius" not in lift | place
    assert "target_radius" not in reach | lift
    assert "target_radius" in place


@pytest.mark.parametrize("name", ["reach-v3", "lift-v3"])
def test_bundled_metadata_loads(name):
    policy = Policy.load(find_policy(name))
    assert policy.task == tasks.task_from_dict(policy.meta.task)
    assert policy.task.obs_dim == policy.meta.obs_dim
    assert policy.task.act_dim == policy.meta.act_dim
