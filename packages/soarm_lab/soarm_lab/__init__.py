"""Simulation, training, and policy strategies for the SO-101.

`station` reads exported twin stations. `sim` builds scenes with task objects.
`robot` and `kinematics` provide the arm model and coordinate conversions.
`blocks` provides named geometry catalogs. `rl` provides tasks, environments, PPO, and evaluation.
`strategies` implements the twin's strategy API. `cli` exposes the `soarm-lab` commands.

Policy inference uses NumPy. Training requires the `soarm-lab[train]` extra.
The `soarm-lab[torch]` extra converts older PyTorch checkpoints.
The package depends on the twin's plugin API and receives station data from the twin.
"""
