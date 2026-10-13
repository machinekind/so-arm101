# so-arm101

The SO-101 robot and everything machinekind learns or measures with it.
The SO-101 is LeRobot's SO-ARM101 with Feetech STS3215 servos.
The demo on 2026-11-11 uses two SO-101 arms, one mounted inverted.
They pick an object from a known set, hand it from one arm to the other and place it on the quadruped Wojtek.
A second robot, the tri-arm, is the same arm with a hand in place of the gripper.
This repository is one uv workspace with one Python package, `soarm`, which is the robot.
A second package, `soarm-lab`, comes with ticket #146. It will hold the rig, perception, training and evaluation.
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
      soarm/                Python package
        robot/              robot.yaml, MJCF and meshes of the SO-101
        feetech.py          backend `feetech`: Feetech STS3215 servos over a serial port or socket://
        lerobot_backend.py  backend `lerobot`: the arm through LeRobot
        serial_bridge.py    COM <-> TCP bridge, the `soarm-bridge` command
        settings.py         strict builder for the backends' config sections
  docs/                     first contact, conventions, safety, migration
  tests/                    pytest
```

## State

`packages/soarm/soarm/` is the twin's `src/lerobot_mp/robot/so101/`, moved here with ticket #145.
The robot folder is the twin's, byte for byte. The backends no longer import the twin. Their behaviour is unchanged.
The twin still holds its own copy of that package and registers the same names (`so101`, `feetech`, `lerobot`).
Until the twin drops its copy, its robot and backend discovery fails when both packages are installed.
The servo tools and a `soarm` command come later. Training moves here with #146.
[docs/MIGRATION.md](docs/MIGRATION.md) tracks the move.

## Install

```
uv sync
```

This installs `soarm` in editable mode with pyserial, pytest, ruff and the twin from GitHub.
The tests run the package through the twin's own loader and backend contract.
The twin repository is private, so `uv sync` needs GitHub access to it.

Extras of `soarm`:

```
pip install "soarm[feetech]"   # pyserial: the feetech backend and soarm-bridge
pip install "soarm[lerobot]"   # LeRobot 0.6 backend (Python 3.12+)
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
- [docs/conventions.md](docs/conventions.md): units, joint names, the dependency rule and where the contracts live.
- [docs/MIGRATION.md](docs/MIGRATION.md): what moves from the twin, where, and with which ticket.

## License

Apache-2.0. See [LICENSE](LICENSE).
The SO-101 MJCF and meshes come from MuJoCo Menagerie under Apache-2.0. See `packages/soarm/soarm/robot/`.
