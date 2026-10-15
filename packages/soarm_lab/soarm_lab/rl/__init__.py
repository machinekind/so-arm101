"""Reinforcement learning for the SO-101: tasks, CPU and GPU environments, PPO, policies.

The twin identifies the servo dynamics; the result arrives as data in the exported `station.json`.
Importing this package registers the Gymnasium environments:

    SoArmLab/Reach-v0       the TCP drives to a point (the goal is sampled from reachable ones)
    SoArmLab/Lift-v0        grasp and lift a 3 cm cube
    SoArmLab/PickPlace-v0   carry the cube onto a target on the table and release it
"""

from __future__ import annotations

import gymnasium as gym

for _id, _task in (("SoArmLab/Reach-v0", "reach"), ("SoArmLab/Lift-v0", "lift"),
                   ("SoArmLab/PickPlace-v0", "pickplace")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="soarm_lab.rl.env:ArmEnv", kwargs={"task": _task})
