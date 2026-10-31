"""Strategies: the lab's adapters for the twin, one module per approach.

Each module wraps one part of the lab and imports from it: `policy` runs an RL policy from `rl`, `ik`
drives the TCP to a goal with `kinematics`, `touch` touches an object from `obs.objects` with `kinematics`,
`touch_jev` adds a decision step after each descent, asking TypeSafe's Jev model (`judges`, `jev`) over
the network, `grasp` picks a block up after `touch`'s probe (`jaw` holds the shared jaw rule), and
`dream` runs Dream Labs' Dream-1 model over the network.
`packages/soarm_lab/pyproject.toml` registers each strategy under the `machinekind_twin.strategies` entry
point group. The twin finds it there and runs it through its
`StrategyRunner`.
"""

from .dream import DreamSettings, DreamStrategy
from .grasp import GraspSettings, GraspStrategy
from .ik import IKSettings, IKStrategy
from .jaw import JawMonitor, JawReading
from .policy import PolicySettings, PolicyStrategy
from .touch import TouchSettings, TouchStrategy
from .touch_jev import JevTouchSettings, JevTouchStrategy

__all__ = ["DreamSettings", "DreamStrategy", "GraspSettings", "GraspStrategy", "IKSettings", "IKStrategy",
           "JawMonitor", "JawReading", "JevTouchSettings", "JevTouchStrategy", "PolicySettings",
           "PolicyStrategy", "TouchSettings", "TouchStrategy"]
