# so-arm101

The SO-101 robot and everything machinekind learns or measures with it.
The SO-101 is LeRobot's SO-ARM101 with Feetech STS3215 servos.
The demo on 2026-11-11 uses two SO-101 arms, one mounted inverted.
They pick an object from a known set, hand it from one arm to the other and place it on the quadruped Wojtek.
A second robot, the tri-arm, is the same arm with a hand in place of the gripper.
This repository is one uv workspace with two Python packages:

- `soarm` is the robot: its robot folder and the hardware backends.
- `soarm-lab` (package `soarm_lab`) is the training: RL tasks, the CPU and GPU environments, PPO,
  evaluation, the base policies, and a strategy that runs a policy in the twin.

The digital twin ([digital_twin_training](https://github.com/machinekind/digital_twin_training), package `lerobot_mp`)
measures the rig and runs strategies through its safety supervisor.

## Rules

1. The twin holds no robot and no training. It keeps the scene, cameras, calibration, perception primitives,
   runtime, safety, sysid and the contracts `Observation`, `Strategy`, `StrategyRunner` and `EpisodeLog`.
2. Dependency direction: the twin imports `soarm`. `soarm_lab` imports `soarm` and never the twin.
   `soarm` imports neither the twin nor `soarm_lab`. The twin never imports `soarm_lab`.
3. The twin hands the measured station to `soarm_lab` as data: `lerobot-twin export <folder>`.
   It runs the lab's policies as a strategy it finds through an entry point.

## Layout

```
so-arm101/
  packages/
    soarm/                  the robot; no twin import; the twin loads it as a plugin
      soarm/                Python package
        robot/              robot.yaml, MJCF and meshes of the SO-101
        feetech.py          backend `feetech`: Feetech STS3215 servos over a serial port or socket://
        lerobot_backend.py  backend `lerobot`: the arm through LeRobot
        serial_bridge.py    COM <-> TCP bridge, the `soarm-bridge` command
        settings.py         strict builder for the backends' config sections
    soarm_lab/              the training; no twin import; the twin runs its strategy as a plugin
      soarm_lab/            Python package
        station.py          an exported station (scene.xml + station.json), read with plain MuJoCo
        stations/sim/       the bundled simulated station
        robot.py            robot.yaml of soarm as a dataclass
        kinematics.py       degrees <-> MJCF, TCP forward kinematics, IK
        sim.py              the compiled scene with the task's objects
        rl/                 tasks, Gymnasium env (CPU), MuJoCo Warp batch (GPU), PPO, policies, evaluation
        policies/           base policies reach-v3 and lift-v3
        strategy.py         a policy as a twin strategy (entry point `policy`)
        cli.py              the `soarm-lab` command: train, eval, policies
  docs/                     first contact, conventions, safety, migration
  tests/                    pytest
```

## The station: from the twin to the lab

The twin measures the station: where the arm stands on the table, the cameras and the arm's dynamics.
`lerobot-twin export <folder>` writes it as data:

- `scene.xml`: the MuJoCo scene with floor, table, lights, the arm under the prefix `robot/`,
  the cameras with their full intrinsics, and the arm's physics options;
- `assets/`: the arm's meshes;
- `station.json`: the twin's station record (table, cameras, identified dynamics, home pose).

`soarm_lab.station.Station.load(folder)` reads the folder with plain `mujoco`.
It adds the task's cube, target and jaw contact sensors, then trains or evaluates on that scene.
`soarm-lab train --station <folder>` and `soarm-lab eval --station <folder>` take the folder.
Without `--station`, the lab uses `sim`, the simulated station bundled with the package.

The twin runs a trained policy through the `lerobot_mp.strategies` entry point `policy`.
It appears in the panel's Strategies tab.
See [packages/soarm_lab/README.md](packages/soarm_lab/README.md).

## State

`packages/soarm/soarm/` is the twin's `src/lerobot_mp/robot/so101/`, moved here with ticket #145.
The robot folder is the twin's, byte for byte. The backends no longer import the twin. Their behaviour is unchanged.
`packages/soarm_lab/` is the twin's `twin/rl/` without `sysid.py`, moved here with ticket #146,
together with the base policies and the training commands of `lerobot-twin`.
The twin still holds its own copies and registers the same robot and backend names (`so101`, `feetech`, `lerobot`).
Until the twin drops its copy, its robot and backend discovery fails when both packages are installed.
The servo tools and a `soarm` command come later.
[docs/MIGRATION.md](docs/MIGRATION.md) tracks the move.

## Install

```
uv sync
```

This installs `soarm` and `soarm-lab` in editable mode with pyserial, MuJoCo, Gymnasium, torch, pytest, ruff
and the twin from GitHub.
The tests run both packages through the twin's own loaders and contracts.
The twin repository is private, so `uv sync` needs GitHub access to it.

GPU training needs MuJoCo Warp and a CUDA build of torch:

```
uv sync --extra train
```

Extras of `soarm`:

```
pip install "soarm[feetech]"   # pyserial: the feetech backend and soarm-bridge
pip install "soarm[lerobot]"   # LeRobot 0.6 backend (Python 3.12+)
```

Extras of `soarm-lab`:

```
pip install "soarm-lab[train]" # mujoco-warp: GPU training
```

## Test

```
uv run pytest
uv run ruff check .
```

Tests that need MuJoCo skip when it is not installed.
Tests marked `render` skip where MuJoCo cannot open an OpenGL context. On a headless Linux machine set `MUJOCO_GL=egl`.
GPU tests skip without `mujoco-warp` or CUDA.

## Docs

- [docs/first-contact.md](docs/first-contact.md): plug in an arm for the first time, safely.
- [docs/safety.md](docs/safety.md): safety rules.
- [docs/conventions.md](docs/conventions.md): units, joint names, the dependency rule and where the contracts live.
- [docs/MIGRATION.md](docs/MIGRATION.md): what moves from the twin, where, and with which ticket.
- [packages/soarm_lab/README.md](packages/soarm_lab/README.md): stations, training, evaluation and the policy strategy.

## License

Apache-2.0. See [LICENSE](LICENSE).
The SO-101 MJCF and meshes come from MuJoCo Menagerie under Apache-2.0. See `packages/soarm/soarm/robot/`.
