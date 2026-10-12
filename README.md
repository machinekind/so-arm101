# so-arm101

The SO-101 robot and everything machinekind learns or measures with it.
The SO-101 is LeRobot's SO-ARM101 with Feetech STS3215 servos.
The demo on 2026-11-11 uses two SO-101 arms, one mounted inverted.
They pick an object from a known set, hand it from one arm to the other and place it on the quadruped Wojtek.
A second robot, the tri-arm, is the same arm with a hand in place of the gripper.
This repository is one uv workspace with two Python packages.
`soarm` is the robot. `soarm-lab` holds the rig, perception, training and evaluation.
The digital twin ([digital_twin_training](https://github.com/machinekind/digital_twin_training), package `lerobot_mp`)
measures the rig and runs strategies through its safety supervisor.

## Rules

1. The twin holds no robot and no training. It keeps the scene, cameras, calibration, perception primitives,
   runtime, safety, sysid and the contracts `Observation`, `Strategy`, `StrategyRunner` and `EpisodeLog`.
2. Dependency direction: the twin imports `soarm`. `soarm_lab` imports the twin and `soarm`.
   `soarm` imports neither the twin nor `soarm_lab`. The twin never imports `soarm_lab`.

## Layout

```
so-arm101/
  packages/
    soarm/                  the robot; no twin import; the twin loads it as a plugin
      robot/
        so101/              MJCF, meshes, robot.yaml
        tri-arm/            robot.yaml of the arm with a hand (draft)
      end_effectors/
        gripper/            jaw meshes, opening range, pad geometry
        hand/               finger joints, open and closed poses, grasp detection
      mounts/               table/ inverted/ wrist_cam/ top_cam/, one transform.yaml each
      soarm/                Python package: spec, backend, feetech, bridge, lerobot, servo, leader, calibration, cli
      arms/                 one folder per physical arm
      embodiment/           embodiment descriptors
    soarm_lab/              everything learned or measured; imports the twin and soarm
      rig/                  demo table, cameras, object set, protocol, handover, Wojtek
      perception/           which object from the set, and where on the table
      training/
        rl/                 PPO, GPU batch env, tasks, policy format, randomization, curricula
        planner/            geometric grasp, GraspGenX adapter
        vla/                recording recipe, LeRobot config, fine-tune configs, Cosmos, automatic demos
      eval/                 run_sim.py, run_real.py, results/, compare.md
      policies/             meta.json and Hugging Face links only
      soarm_lab/            Python package: strategies.py
  docs/                     first contact, conventions, safety, migration
  tests/                    pytest
```

## State

`soarm.spec` is real: `soarm.so101()` loads and validates `robot/so101/robot.yaml`.
The robot entry point and the backend entry points are registered.
The backends, the servo tools, the `soarm` command and everything in `soarm_lab` are stubs.
Code moves here from the twin with tickets #145 and #146. [docs/MIGRATION.md](docs/MIGRATION.md) tracks it.

## Install

```
uv sync
```

This installs both packages in editable mode with pytest and ruff. It does not install the twin or torch.

Extras:

```
uv sync --extra feetech      # pyserial, to drive servos
uv sync --extra lerobot      # LeRobot 0.6 backend (Python 3.12+)
uv sync --extra twin         # the twin from GitHub
uv sync --extra train        # torch, MuJoCo, MuJoCo Warp
```

## Test

```
uv run pytest
uv run ruff check .
```

Tests that need MuJoCo skip when it is not installed.

## Docs

- [docs/first-contact.md](docs/first-contact.md): plug in an arm for the first time, safely.
- [docs/safety.md](docs/safety.md): safety rules.
- [docs/conventions.md](docs/conventions.md): units, joint names, robot.yaml schema, entry points, dependency rule.
- [docs/MIGRATION.md](docs/MIGRATION.md): what moves from the twin, where, and with which ticket.

## License

Apache-2.0. See [LICENSE](LICENSE).
The SO-101 MJCF and meshes come from MuJoCo Menagerie under Apache-2.0. See `packages/soarm/robot/so101/`.
