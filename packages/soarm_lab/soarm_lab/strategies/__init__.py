"""Strategies: the lab's adapters for the twin, one module per approach.

Each module wraps one part of the lab and imports from it: `policy` runs an RL policy from `rl`, `ik`
drives the TCP to a goal with `kinematics`, `touch` touches an object from `obs.objects` with `kinematics`,
`dream` runs Dream Labs' Dream-1 model over the network.
`packages/soarm_lab/pyproject.toml` registers each strategy under the `machinekind_twin.strategies` entry
point group. The twin finds it there and runs it through its
`StrategyRunner`.
"""

from .dream import DreamSettings, DreamStrategy
from .ik import IKSettings, IKStrategy
from .policy import PolicySettings, PolicyStrategy
from .touch import TouchSettings, TouchStrategy

__all__ = ["DreamSettings", "DreamStrategy", "IKSettings", "IKStrategy", "PolicySettings", "PolicyStrategy",
           "TouchSettings", "TouchStrategy"]
