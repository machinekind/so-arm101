"""Strategy adapters that the twin's `StrategyRunner` drives. Stubs.

A strategy has `start(twin, robot)`, `act(obs)` and `stop()`, as planned in the twin's
docs/PLAN_GENERIC_TWIN.md. `act` returns joint targets in degrees (gripper 0..100) or None to hold.
The twin's `Strategy`, `Observation` and `StrategyRunner` contracts do not exist on its main branch yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TorchPolicyStrategy:
    """Runs a trained RL policy. Comes from the twin's `twin/rl/runner.py` `PolicyRunner` (#146)."""

    def __init__(self, policy_path: str) -> None:
        self.policy_path = policy_path

    def start(self, twin: Any, robot: str) -> None:
        """Load the policy and build its task from the policy file, as `PolicyRunner` does."""
        raise NotImplementedError

    def act(self, obs: Any) -> Mapping[str, float] | None:
        """One policy step: `task.observe`, the actor, then `task.apply_action`."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release the policy."""
        raise NotImplementedError


class IKStrategy:
    """Scripted IK planner: TCP waypoints, IK per waypoint, collision check per segment. New code."""

    def start(self, twin: Any, robot: str) -> None:
        """Plan the waypoints for this episode."""
        raise NotImplementedError

    def act(self, obs: Any) -> Mapping[str, float] | None:
        """Ramp toward the IK solution of the current waypoint."""
        raise NotImplementedError

    def stop(self) -> None:
        """Drop the plan."""
        raise NotImplementedError
