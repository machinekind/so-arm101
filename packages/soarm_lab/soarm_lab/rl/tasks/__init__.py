"""Task registry and policy metadata loading."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ...blocks import block
from ...robot import RobotSpec
from .base import (
    Limits,
    apply_action,
    fold_yaw,
    grip_is_open,
    limit_cost,
    normalize_q,
    sample_start,
    success_streak,
)
from .lift import LiftConfig, sample_cubes
from .pickplace import TARGET_TRIES, PickPlaceConfig, carry_latch, sample_targets
from .reach import ReachConfig, sample_goals

TaskConfig = ReachConfig | LiftConfig | PickPlaceConfig
TASKS = ("reach", "lift", "pickplace")
_CONFIGS = {"reach": ReachConfig, "lift": LiftConfig, "pickplace": PickPlaceConfig}


def make_task(name: str = "reach", spec: RobotSpec | None = None, **overrides) -> TaskConfig:
    """Build a task scaled to the arm, then apply overrides."""
    if name not in _CONFIGS:
        raise ValueError(f"nieznane zadanie {name!r}; dostepne: {', '.join(TASKS)}")
    base = _CONFIGS[name]()
    if spec is not None:
        base = base.for_spec(spec)
    return replace(base, **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})


def _stored_cube(data: dict[str, Any]) -> dict[str, str]:
    """Map legacy cube dimensions and mass to the default block, rejecting unmatched values."""
    spec = block(LiftConfig().block)
    want = {"cube_half": spec.half_size[2], "cube_mass": spec.mass}
    stored = {k: data[k] for k in want if k in data}
    if not stored:
        return {}
    if any(value != want[k] for k, value in stored.items()):
        raise ValueError(f"the stored task has {stored}, which is not the block {spec.name} ({want}); "
                         "there is no block to load it as")
    return {"block": spec.name}


def task_from_dict(data: dict[str, Any]) -> TaskConfig:
    """Load owned fields and fill missing ones from the task factory."""
    name = data.get("name")
    if name not in _CONFIGS:
        raise ValueError(f"unknown task name: {name!r}")
    owned = {f.name for f in fields(_CONFIGS[name])} - {"name"}
    kept = {k: v for k, v in data.items() if k in owned}
    if "block" in owned and "block" not in data:
        kept.update(_stored_cube(data))
    return make_task(name, **kept)


OBS_DIMS = {name: make_task(name).obs_dim for name in TASKS}

__all__ = ["TASKS", "TARGET_TRIES", "OBS_DIMS", "TaskConfig", "ReachConfig", "LiftConfig", "PickPlaceConfig",
           "Limits", "make_task", "task_from_dict", "apply_action", "fold_yaw",
           "grip_is_open", "limit_cost", "normalize_q", "sample_start", "success_streak", "carry_latch",
           "sample_targets", "sample_goals", "sample_cubes"]
