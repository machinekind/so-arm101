"""Strategies: the lab's adapters for the twin, one module per approach.

Each module wraps one part of the lab and imports from it: `policy` runs an RL policy from `rl`, `ik`
drives the TCP to a goal with `kinematics`, `touch` touches an object from `obs.objects` with `kinematics`,
`touch_jev` adds a decision step after each descent, asking TypeSafe's Jev model (`judges`, `jev`) over
the network. `grasp` picks up a block from above. `jaw` supplies the shared contact rule.
`dream` runs Dream Labs' Dream-1 model over the network.
`grasp_wrist` (`wrist`) refines the grasp aim and yaw from the wrist camera.
`grasp_jev` asks Jev after each close and lift through `judges.Consultation`.
With `probe: true`, it also asks after each probe.
`pick_place` picks a block and places it on another block or at a pose; `build` runs one `pick_place` per
step of a plan. `calibrate_approach` measures how the arm lands on what the stand's cameras see, into the
approach table that `pick_place` reads; `approach` holds the final-leg directions and place heights both use.
`packages/soarm_lab/pyproject.toml` registers each strategy under the `machinekind_twin.strategies` entry
point group. The twin finds it there and runs it through its
`StrategyRunner`.
"""

from .dream import DreamSettings, DreamStrategy
from .grasp import GraspSettings, GraspStrategy
from .grasp_jev import GraspJevSettings, GraspJevStrategy
from .ik import IKSettings, IKStrategy
from .jaw import JawMonitor, JawReading
from .policy import PolicySettings, PolicyStrategy
from .touch import TouchSettings, TouchStrategy
from .touch_jev import JevTouchSettings, JevTouchStrategy
from .wrist import WristGraspSettings, WristGraspStrategy, WristStatus

__all__ = ["DreamSettings", "DreamStrategy", "GraspJevSettings", "GraspJevStrategy", "GraspSettings",
           "GraspStrategy", "IKSettings", "IKStrategy", "JawMonitor", "JawReading", "JevTouchSettings",
           "JevTouchStrategy", "PolicySettings", "PolicyStrategy", "TouchSettings", "TouchStrategy",
           "WristGraspSettings", "WristGraspStrategy", "WristStatus"]
