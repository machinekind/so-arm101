"""Strategies: the lab's adapters for the twin, one module per approach.

Each module wraps one part of the lab and imports from it: `policy` runs an RL policy from `rl`, `ik`
drives the TCP to a goal with `kinematics`. `packages/soarm_lab/pyproject.toml` registers each strategy
under the `machinekind_twin.strategies` entry point group. The twin finds it there and runs it through its
`StrategyRunner`.
"""

from .ik import IKSettings, IKStrategy
from .policy import PolicySettings, PolicyStrategy

__all__ = ["IKSettings", "IKStrategy", "PolicySettings", "PolicyStrategy"]
