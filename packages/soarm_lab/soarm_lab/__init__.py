"""SO-101 training: RL tasks, CPU and GPU environments, PPO, the policies and their strategy for the twin.

* `station`: a measured station as the twin exports it (`twin export <folder>`), read with
  plain `mujoco`; `stations/sim` is bundled.
* `robot`, `kinematics`: the SO-101 description from `soarm` and its kinematics (degrees <-> MJCF).
* `sim`: the compiled station scene with the task's objects.
* `rl`: tasks, the Gymnasium environment (CPU), the MuJoCo Warp batch (GPU), PPO, policies, evaluation,
  domain randomization. `policies/` holds the base policies `reach-v3` and `lift-v3` as `policy.npz`.
* `strategies`: the lab's strategies for the twin, one module per approach; `strategies.policy` runs a
  policy (entry point `machinekind_twin.strategies`: `policy`).
* `cli`: the `soarm-lab` command (`train`, `eval`, `policies`, `convert`).

Running a policy needs numpy only. Training needs the extra `soarm-lab[train]`: PPO in JAX and optax
(`rl.ppo`, `rl.jax_policy`) on MuJoCo Warp (`rl.batch`). torch is the optional extra `soarm-lab[torch]`,
only for the converter of old torch checkpoints (`rl.torch_policy`).

Nothing here imports the twin (`machinekind_twin`). The twin hands the station over as data and runs the
strategy through its own `StrategyRunner`. The strategy follows the twin's plugin contracts,
`machinekind_twin_api.strategy`.
"""
