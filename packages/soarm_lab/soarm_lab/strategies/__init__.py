"""Strategies: the lab's adapters for the twin, one module per approach.

Each module wraps one part of the lab and imports from it: `policy` runs an RL policy from `rl`.
`packages/soarm_lab/pyproject.toml` registers each strategy under the `lerobot_mp.strategies` entry point
group. The twin finds it there and runs it through its `StrategyRunner`.
"""

from .policy import PolicySettings, PolicyStrategy

__all__ = ["PolicySettings", "PolicyStrategy"]
