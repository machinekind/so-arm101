"""SO-101 training: RL tasks, CPU and GPU environments, PPO, the policies and their strategy for the twin.

* `station`: a measured station as the twin exports it (`lerobot-twin export <folder>`), read with
  plain `mujoco`; `stations/sim` is bundled.
* `robot`, `kinematics`: the SO-101 description from `soarm` and its kinematics (degrees <-> MJCF).
* `sim`: the compiled station scene with the task's objects.
* `rl`: tasks, the Gymnasium environment (CPU), the MuJoCo Warp batch (GPU), PPO, policies, evaluation,
  domain randomization. `policies/` holds the base policies `reach-v3` and `lift-v3`.
* `strategy`: a policy as a strategy of the twin (entry point `lerobot_mp.strategies`: `policy`).
* `cli`: the `soarm-lab` command (`train`, `eval`, `policies`).

Nothing here imports the twin (`lerobot_mp`). The twin hands the station over as data and runs the
strategy through its own `StrategyRunner`.
"""
