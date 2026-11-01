"""Task registry and policy metadata loading."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

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


def task_from_dict(data: dict[str, Any]) -> TaskConfig:
    """Load owned fields and fill missing ones from the task factory."""
    name = data.get("name")
    if name not in _CONFIGS:
        raise ValueError(f"unknown task name: {name!r}")
    owned = {f.name for f in fields(_CONFIGS[name])} - {"name"}
    return make_task(name, **{k: v for k, v in data.items() if k in owned})


OBS_DIMS = {name: make_task(name).obs_dim for name in TASKS}

__all__ = ["TASKS", "TARGET_TRIES", "OBS_DIMS", "TaskConfig", "ReachConfig", "LiftConfig", "PickPlaceConfig",
           "Limits", "make_task", "task_from_dict", "apply_action", "fold_yaw",
           "grip_is_open", "limit_cost", "normalize_q", "sample_start", "success_streak", "carry_latch",
           "sample_targets", "sample_goals", "sample_cubes"]
