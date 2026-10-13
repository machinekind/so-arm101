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
| `src/lerobot_mp/twin/rl/__init__.py` | `packages/soarm_lab/training/rl/__init__.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/batch.py` | `packages/soarm_lab/training/rl/batch.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/env.py` | `packages/soarm_lab/training/rl/env.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/evaluate.py` | `packages/soarm_lab/training/rl/evaluate.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/policy.py` | `packages/soarm_lab/training/rl/policy.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/ppo.py` | `packages/soarm_lab/training/rl/ppo.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/randomize.py` | `packages/soarm_lab/training/rl/randomize.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/runner.py` | `packages/soarm_lab/training/rl/runner.py`; `PolicyRunner` becomes `TorchPolicyStrategy` in `packages/soarm_lab/soarm_lab/strategies.py` | #146 | not moved yet |
| `src/lerobot_mp/twin/rl/task.py` | `packages/soarm_lab/training/rl/task.py` | #146 | not moved yet |
| `assets/policies/reach-v3/` (`meta.json`, `policy.pt`) | `packages/soarm_lab/policies/reach-v3/` (`meta.json` and a Hugging Face link; weights to Hugging Face) | #146 | not moved yet |
| `assets/policies/lift-v3/` (`meta.json`, `policy.pt`) | `packages/soarm_lab/policies/lift-v3/` (`meta.json` and a Hugging Face link; weights to Hugging Face) | #146 | not moved yet |
| `tests/test_feetech.py` | `tests/test_feetech.py` | #145 | copied; twin PR pending |
| `tests/test_serial_bridge.py` | `tests/test_serial_bridge.py` | #145 | copied; twin PR pending |
| `tests/test_lerobot_backend_guard.py` | `tests/test_lerobot_backend_guard.py` | #145 | copied; twin PR pending |
| `tests/test_twin_rl.py` | `tests/test_lab_rl.py` | #146 | not moved yet |
| `tests/test_twin_policies.py` | `tests/test_lab_policies.py` | #146 | not moved yet |
| `tests/test_twin_runner.py` | `tests/test_lab_runner.py` | #146 | not moved yet |
| `tests/test_twin_lift_from_cameras.py` | `tests/test_lab_lift_from_cameras.py` | #146 | not moved yet |

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

1. `twin/rl/sysid.py` stays in the twin but imports `Dynamics` and `UNFITTED_RANGES` from `twin/rl/randomize.py`.
   Rule 2 forbids the twin to import `soarm_lab`. `Dynamics` and `UNFITTED_RANGES` stay in the twin,
   and `training/rl/randomize.py` imports them from there.
2. Resolved with #145. The backends take the twin's `BackendContext` and build their settings with a local copy of `from_mapping`.
3. Resolved before #145. `robot/factory.py` finds backends through the `lerobot_mp.backends` entry points.
4. Twin callers of `rl/`: `twin/cli.py` (train, eval and policy commands), `twin/ui/jobs.py` and `twin/ui/app.py`.
   They lose those features or reach them through the Strategy contract. They cannot import `soarm_lab`.
5. `twin/rl/policy.py` finds bundled policies under the twin's `assets/policies/`. The bundled lookup moves with the policies.
6. `pyproject.toml` of the twin: the `robot`, `feetech` and `bridge` extras and the `lerobot-mp-bridge` script move to `soarm`.
   `soarm` now has the `feetech` and `lerobot` extras and the `soarm-bridge` script. The twin still has its own.
   The `train` extra moves to `soarm-lab`.
