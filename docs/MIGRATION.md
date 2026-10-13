# Migration from the twin

Two tickets move the robot and the training out of the twin (`digital_twin_training`, package `lerobot_mp`) into this repository.

- **#145** B1. Robot folder and Feetech backend plugin. Done when `pip install soarm` is enough for the twin to drive an SO-101, and the twin no longer holds `robot/so101/`.
- **#146** B2. Training (`rl/`) moves from the twin to this repository. Only `rl/sysid.py` stays in the twin.

Paths on the left are relative to the twin's repository root. Paths on the right are relative to this repository.

## Files

| Twin file | Destination | Ticket | Status |
|---|---|---|---|
| `src/lerobot_mp/robot/__init__.py` | stays in the twin; its docstring stops naming `so101/` | #145 | twin PR pending |
| `src/lerobot_mp/robot/so101/__init__.py` | `packages/soarm/soarm/__init__.py` | #145 | copied; twin PR pending |
| `src/lerobot_mp/robot/so101/robot/` (`robot.yaml`, `so101.xml`, `assets/*.stl`, `LICENSE`, `README.md`) | `packages/soarm/soarm/robot/` | #145 | copied; twin PR pending |
| `src/lerobot_mp/robot/so101/feetech.py` | `packages/soarm/soarm/feetech.py` | #145 | copied; twin PR pending |
| `src/lerobot_mp/robot/so101/lerobot_backend.py` | `packages/soarm/soarm/lerobot_backend.py` | #145 | copied; twin PR pending |
| `src/lerobot_mp/robot/so101/serial_bridge.py` | `packages/soarm/soarm/serial_bridge.py` | #145 | copied; twin PR pending |
| `from_mapping` and `_coerce` in `src/lerobot_mp/twin/config.py` | a copy in `packages/soarm/soarm/settings.py`; the twin keeps its own | #145 | copied |
| `src/lerobot_mp/twin/rl/__init__.py` | `packages/soarm_lab/soarm_lab/rl/__init__.py` (Gymnasium ids `SoArmLab/*-v0`) | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/batch.py` | `packages/soarm_lab/soarm_lab/rl/batch.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/env.py` | `packages/soarm_lab/soarm_lab/rl/env.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/evaluate.py` | `packages/soarm_lab/soarm_lab/rl/evaluate.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/policy.py` | `packages/soarm_lab/soarm_lab/rl/policy.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/ppo.py` | `packages/soarm_lab/soarm_lab/rl/ppo.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/randomize.py` | `packages/soarm_lab/soarm_lab/rl/randomize.py` (a copy: `Dynamics` and `UNFITTED_RANGES` also stay in the twin, next to sysid) | #146 | copied; twin PR pending |
| `src/lerobot_mp/twin/rl/runner.py` | `PolicyRunner` becomes `PolicyStrategy` in `packages/soarm_lab/soarm_lab/strategy.py`, run by the twin's `StrategyRunner` | #146 | rewritten; twin PR pending |
| `src/lerobot_mp/twin/rl/task.py` | `packages/soarm_lab/soarm_lab/rl/task.py` | #146 | moved; twin PR pending |
| `src/lerobot_mp/twin/rl/sysid.py` | stays in the twin | #146 | - |
| `src/lerobot_mp/twin/kinematics.py` | a copy in `packages/soarm_lab/soarm_lab/kinematics.py`; the twin keeps its own | #146 | copied |
| `RobotSpec` and `load_robot` in `src/lerobot_mp/twin/robots.py` | the fields the lab uses, in `packages/soarm_lab/soarm_lab/robot.py`; the twin keeps its own | #146 | copied |
| `Scene`, `Box`, `CameraView`, `build` in `src/lerobot_mp/twin/scene.py` | `packages/soarm_lab/soarm_lab/sim.py` and `station.py`, built from the exported `scene.xml`; the twin keeps its own | #146 | copied |
| `_train`, `_eval`, `_policies`, `_train_plan`, `_describe` in `src/lerobot_mp/twin/cli.py` | `packages/soarm_lab/soarm_lab/cli.py`, the `soarm-lab` command | #146 | moved; twin PR pending |
| `assets/policies/reach-v3/` (`meta.json`, `policy.pt`) | `packages/soarm_lab/soarm_lab/policies/reach-v3/` (`meta.json`, `policy.npz` converted from `policy.pt`) | #146 | converted; twin PR pending |
| `assets/policies/lift-v3/` (`meta.json`, `policy.pt`) | `packages/soarm_lab/soarm_lab/policies/lift-v3/` (`meta.json`, `policy.npz` converted from `policy.pt`) | #146 | converted; twin PR pending |
| `lerobot-twin export` of `examples/twin.sim.json` (branch `scene-export`) | `packages/soarm_lab/soarm_lab/stations/sim/` (`scene.xml`, `station.json`; meshes from `soarm`) | #146 | copied |
| `tests/test_feetech.py` | `tests/test_feetech.py` | #145 | copied; twin PR pending |
| `tests/test_serial_bridge.py` | `tests/test_serial_bridge.py` | #145 | copied; twin PR pending |
| `tests/test_lerobot_backend_guard.py` | `tests/test_lerobot_backend_guard.py` | #145 | copied; twin PR pending |
| `tests/test_twin_rl.py` | `tests/test_lab_rl.py` | #146 | ported; twin PR pending |
| `tests/test_twin_policies.py` | `tests/test_lab_policies.py` | #146 | ported; twin PR pending |
| `tests/test_twin_runner.py` | `tests/test_lab_runner.py` (the strategy's guards) | #146 | ported; twin PR pending |
| `tests/test_twin_lift_from_cameras.py` | `tests/test_lab_lift_from_cameras.py` | #146 | ported; twin PR pending |

The twin keeps `robot/base.py`, `robot/factory.py`, `twin/rl/sysid.py` and `robot/a1x/`.
`robot/a1x/` drives the Galaxea A1X and is not part of these tickets.

## Twin side of #145

This repository registers the robot `so101` and the backends `feetech` and `lerobot`.
The twin's `main` still registers the same names from its own copy.
With both installed, the twin's discovery raises on the duplicates.
A twin PR still has to do the cut:

- Delete `src/lerobot_mp/robot/so101/`.
- In the twin's `pyproject.toml`, remove the `so101`, `feetech` and `lerobot` entry points,
  the `lerobot-mp-bridge` script and the `lerobot_mp.robot.so101` package data.
- Delete the three moved tests. Rework `tests/test_robot_plugins.py` and `tests/test_twin_robots.py`,
  which name the twin's `so101` package.
- Other twin tests load the robot `so101` by name. After the cut they need `soarm` installed.

After that PR, this repository's tests can cover discovery through `robot_folders()` and `backends()`.

## Things that block a clean move

These come from the twin's imports on its `main` branch.

1. Resolved with #146. `twin/rl/sysid.py` stays in the twin and imports `Dynamics` and `UNFITTED_RANGES`
   from `twin/rl/randomize.py`; the twin's cut moves both next to sysid. `soarm_lab` keeps its own copy of
   `randomize.py` and reads the sysid result from `station.json` (`dynamics`), so it never imports the twin.
2. Resolved with #145. The backends take the twin's `BackendContext` and build their settings with a local copy of `from_mapping`.
3. Resolved before #145. `robot/factory.py` finds backends through the `lerobot_mp.backends` entry points.
4. Twin callers of `rl/`: `twin/cli.py` (train, eval and policy commands), `twin/ui/jobs.py` and `twin/ui/app.py`.
   They lose those features or reach them through the Strategy contract. They cannot import `soarm_lab`.
   See "Twin side of #146".
5. Resolved with #146. The bundled policies live in `soarm_lab/policies/`; `rl/policy.py` looks there and in the
   station's `workspace/policies` (`find_policy`).
6. `pyproject.toml` of the twin: the `robot`, `feetech` and `bridge` extras and the `lerobot-mp-bridge` script move to `soarm`.
   `soarm` now has the `feetech` and `lerobot` extras and the `soarm-bridge` script. The twin still has its own.
   The `train` extra moves to `soarm-lab` and becomes `mujoco-warp` and `jax`. mujoco and gymnasium are base
   dependencies there. torch is the optional extra `torch`.

## #146: training

`packages/soarm_lab` is the twin's `twin/rl/` without `sysid.py`, the base policies and the training commands.
It imports no `lerobot_mp`. Its tests may, because the twin is a dev dependency.

The twin hands the measured station over as data. `lerobot-twin export <folder>` writes `scene.xml` (the twin's
scene from `MjSpec`), `assets/` (the arm's meshes) and `station.json` (the workspace record).
`soarm_lab.station.Station` reads the folder with plain `mujoco`, adds the task's objects and jaw sensors, and compiles.
The bundled `stations/sim` is that export of the twin's simulated example station, without the meshes.
`tests/test_station.py` checks that its compiled model matches the twin's own scene build to 1e-5
(the XML keeps 6 significant digits).

### Policy format and torch

The lab's training stack is JAX and MuJoCo Warp. The JAX trainer is a separate ticket.
torch is no longer a dependency of `soarm-lab`.

- Base dependencies are `soarm`, numpy, PyYAML, mujoco and gymnasium.
  The strategy, `soarm-lab eval`, `soarm-lab policies` and the CPU environments import no torch.
  `tests/test_dependency_rule.py` checks this in a fresh interpreter, once with torch made unimportable.
- The extra `train` is `mujoco-warp` and `jax`. Nothing uses jax yet.
- The extra `torch` holds torch for the legacy PPO trainer (`rl/ppo.py`, `rl/batch.py`) and for
  `soarm-lab convert`. Those modules raise "install soarm-lab[torch]" without it.
  The workspace's dev group installs `soarm-lab[torch]`, so the tests still cover the torch trainer.
- A policy is `policy.npz` next to `meta.json`. `policy.npz` holds the normalizer's buffers, the layer sizes
  and the actor's and critic's weights in float32. `packages/soarm_lab/README.md` describes the keys.
- `Policy` in `rl/policy.py` is plain numpy. It has no `to`, `eval`, `state_dict` or `critic_state`.
  `critic` holds the critic's layers. The torch modules (`TorchPolicy`, `Normalizer`, `mlp`) moved to
  `rl/torch_policy.py`.
- The PPO trainer trains `TorchPolicy` and saves and returns a numpy `Policy` with its critic.
  `--init` fine-tunes from a `policy.npz`.
- `Policy.load` refuses a `policy.pt` and names `soarm-lab convert <policy.pt>`.
  `soarm-lab policies` lists the folders that hold only a `policy.pt`.
- The base policies were converted with `soarm-lab convert`. Every array equals the twin's checkpoint.
  `meta.json` is the twin's file, unchanged. On 5000 random observations, one at a time, the numpy actions
  match torch's to 6e-7. In batches of 5000 they match to 1.1e-6.

### Twin imports and their replacements

| Twin import (in the moved code) | Replacement in `soarm_lab` |
|---|---|
| `..workspace.Workspace` (`rl/env.py`, `rl/batch.py`, `rl/ppo.py`, `rl/evaluate.py`) | `station.Station`: `Station.load(folder or "sim")`; `spec()`, `dynamics`, `cameras`, `T_base2world`, `home`, `build(...)` |
| `Workspace.scene_config()` + `scene.build` | `sim.SceneConfig(station)` + `sim.build`: `scene.xml` through `MjSpec.from_file`, plus objects, sensors, extra cameras |
| `.. scene` as `sc` (`Scene`, `SceneConfig`, `Box`, `CameraView`, `build`) | `soarm_lab.sim` as `sc`; `Box` and `CameraView` live in `station.py` |
| `..kinematics.RobotKinematics, inverse, pose` (`rl/env.py`, `rl/task.py`, `rl/runner.py`) | `soarm_lab.kinematics` (a copy) |
| `.robots.RobotSpec` (`kinematics.py`), `..robots.RobotSpec` (`rl/task.py`) | `soarm_lab.robot.RobotSpec`, read from `soarm.ROBOT_DIR/robot.yaml` with `soarm.settings.from_mapping` |
| `..robots.robot_config` (`rl/task.py`, `Limits.of`) | `RobotSpec.limits` from `robot.yaml` (what `robot_config` gives without a config file) |
| `...paths.data_path` (`rl/policy.py`, `DEFAULT_DIR`) | `soarm_lab.paths.policies_dir()`: `workspace/policies` of the checkout, or `SOARM_LAB_POLICIES` |
| `..robots.REPO_ROOT` (`rl/policy.py`, `bundled_dir`) | the package's `soarm_lab/policies/` |
| the `twin` object in `rl/runner.py` (`workspace.spec()`, `status.measured`, `status.command`, `status.simulated`, `joint_limits()`, `claim`, `set_engaged`, `set_target`, `hold_measured`, `release`) | the twin's `StrategyRunner`; the strategy reads `RunInfo` (`robot`, `folder`, `limits`, `simulated`) and `Observation` (`joints`, `command`, `goal`, `objects`, `t`) by attribute |
| `PolicyRunner(cube_provider=..., cube_source=...)` | `obs.objects["cube"]`: an `ObjectPose` with its `source` |
| `PolicyRunner.goal` (set by the panel) | `obs.goal`; without one, the settings' `goal_x`, `goal_y`, `goal_z` |
| Gymnasium ids `LeRobotMP/Twin*-v0` | `SoArmLab/Reach-v0`, `SoArmLab/Lift-v0`, `SoArmLab/PickPlace-v0` |
| `lerobot-twin train/eval/policies --workspace` | `soarm-lab train/eval/policies --station` |

Behaviour that changes on purpose:

- A config file's `joints:` table (`LEROBOT_MP_CONFIG`) no longer narrows the training limits. They come from `robot.yaml`.
- The strategy's lag guard ends the run with the measured pose as its last target; a squeezing gripper keeps its
  command. `PolicyRunner` called `Twin.hold_measured`, which also allows for the servos' braking distance.
  The contract has no "hold the measured pose" answer. The twin's own tracking guard (STOP) applies on top.
- `test_lab_policies.py` allows 70 degrees of joint travel after the first lift success, not 55. The exported scene
  differs from the twin's by XML rounding, and contacts amplify it. The twin measures 44-54 degrees over four blocks of
  30 seeds, the lab 26-59, and both have a p90 of 25.8. lift-v2 measured 72.
  Numpy inference differs from torch's by float32 rounding, which changes the trajectories too. Over twelve blocks of
  30 seeds the lab measures a maximum of 26-65 degrees with torch and 26-65 with numpy. The torch maximum passes 60 in
  three blocks and the numpy maximum in two. The p90 stays at 25.8-28.8.
- The twin's `policy.pt` files do not load in the lab. `soarm-lab convert` turns them into `policy.npz`.
- `test_twin_runner.py`'s race between `claim` and the clutch is the `StrategyRunner`'s now.
  `tests/test_strategy_contract.py` runs it on the twin's runner with the lab's strategy.

### Twin side of #146

A twin PR still has to do the cut:

- Delete `src/lerobot_mp/twin/rl/` except `sysid.py`.
  `sysid.py` imports `Dynamics` and `UNFITTED_RANGES` from `rl/randomize.py`. Move both next to sysid first.
- Delete the training UI: the Training tab, the Policies tab (`PolicyRunner`), the eval job (`ui/jobs.py`, `run_eval`),
  and the panel's uses of `rl.task` (random cube placement, the pickplace target pad, goal projection).
- Delete `_train`, `_eval`, `_policies`, `_train_plan` and `_describe` from `twin/cli.py`, and their parser entries.
- Delete `assets/policies/`.
- Delete `tests/test_twin_rl.py`, `test_twin_policies.py`, `test_twin_runner.py` and `test_twin_lift_from_cameras.py`.
- Drop the `train` extra. Drop gymnasium and torch from the `twin` extra if nothing else needs them.
- Merge `lerobot-twin export` (branch `scene-export`). This repository's lock still pins the twin's `main` at d4e2024.
- Pass `goal` and `objects` from the Strategies tab to `StrategyRunner`: the cube tracker as `ObjectPose` with its
  source. Today the tab passes neither, so a lift or pickplace policy run as a strategy never sees the cube.
