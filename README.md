# so-arm101

The SO-101 robot and everything machinekind learns or measures with it.
The SO-101 is LeRobot's SO-ARM101 with Feetech STS3215 servos.
The demo on 2026-11-11 uses two SO-101 arms, one mounted inverted.
They pick an object from a known set, hand it from one arm to the other and place it on the quadruped Wojtek.
This repository is one uv workspace with two Python packages:

- `soarm` is the robot: its robot folder and the hardware backends.
- `soarm-lab` (package `soarm_lab`) is the training: RL tasks, the CPU and GPU environments, PPO,
  evaluation, the base policies, and a strategy that runs a policy in the twin.
  The training stack is JAX and MuJoCo Warp. Until the JAX trainer is written, training runs a legacy
  torch trainer. Running a policy needs numpy only.

The digital twin ([digital_twin_training](https://github.com/machinekind/digital_twin_training), package `lerobot_mp`)
measures the rig and runs strategies through its safety supervisor.
The twin loads both packages through entry points.
Both packages import the twin's plugin contracts, `lerobot-mp-api` (package `lerobot_mp_api`), and never the twin.
The contracts need only numpy.
[docs/conventions.md](docs/conventions.md) has the full dependency rule.

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
      tests/                pytest, mirroring soarm/
    soarm_lab/              the training; no twin import; the twin runs its strategy as a plugin
      soarm_lab/            Python package
        station.py          an exported station (scene.xml + station.json), read with plain MuJoCo
        stations/sim/       the bundled simulated station
        robot.py            robot.yaml of soarm as a dataclass
        kinematics.py       degrees <-> MJCF, TCP forward kinematics, IK
        sim.py              the compiled scene with the task's objects
        rl/                 tasks, Gymnasium env (CPU), MuJoCo Warp batch (GPU), PPO, policies, evaluation
        policies/           base policies reach-v3 and lift-v3 (policy.npz + meta.json)
        strategies/         strategies for the twin; policy.py runs a policy (entry point `policy`)
        cli.py              the `soarm-lab` command: train, eval, policies, convert
      tests/                pytest, mirroring soarm_lab/
  docs/                     first contact, conventions, safety
  conftest.py               the pytest markers `twin` and `render`
```

## How the twin uses the packages

The twin finds both packages through entry points.
`lerobot_mp_api.entry_points` names the groups: `ROBOTS`, `BACKENDS` and `STRATEGIES`.

| Group | Name | Target |
|---|---|---|
| `lerobot_mp.robots` | `so101` | `soarm:ROBOT_DIR`, the robot folder |
| `lerobot_mp.backends` | `feetech` | `soarm.feetech:create` |
| `lerobot_mp.backends` | `lerobot` | `soarm.lerobot_backend:create` |
| `lerobot_mp.strategies` | `policy` | `soarm_lab.strategies.policy:PolicyStrategy` |

The panel offers the robot `so101` with the `feetech` and `lerobot` connections.
A trained policy runs as the strategy `policy` in the panel's Strategies tab.
See [packages/soarm/README.md](packages/soarm/README.md) and [packages/soarm_lab/README.md](packages/soarm_lab/README.md).

The twin also registers an SO-101 of its own under the names `so101`, `feetech` and `lerobot`.
With both installed, its robot and backend discovery fails on the duplicate names.

## Get a station

The lab trains and evaluates on a station the twin measured: the table and where the arm stands on it,
the cameras and the arm's dynamics. Export it from the twin's environment:

```
lerobot-twin export ~/stations/desk1
```

The folder holds:

- `scene.xml`: the MuJoCo scene with floor, table, lights, the arm under the prefix `robot/`,
  the cameras with their full intrinsics, and the arm's physics options;
- `assets/`: the arm's meshes;
- `station.json`: the station record (table, cameras, identified dynamics, home pose).

Pass the folder to the lab:

```
soarm-lab train --task reach --station ~/stations/desk1
soarm-lab eval lift-v3 --station ~/stations/desk1
```

Without `--station`, the lab uses `sim`, the simulated station bundled with the package.
In Python, `soarm_lab.station.Station.load(folder)` reads the folder with plain `mujoco`.

## Install

```
uv sync
```

This installs `soarm` and `soarm-lab` in editable mode with `lerobot-mp-api`, pyserial, MuJoCo, Gymnasium,
pytest, ruff and the twin from GitHub.
It also installs torch, so the tests cover the legacy torch trainer and `soarm-lab convert`.
`soarm-lab` itself does not depend on torch.
The twin repository is private, so `uv sync` needs GitHub access to it.

Both packages depend on `lerobot-mp-api>=0.1,<0.2`. The package is not published yet.
It lives in the twin's repository under `packages/lerobot_mp_api`.
`[tool.uv.sources]` in `pyproject.toml` installs it from there at a pinned commit, and the twin from the same commit.
pip ignores that table. Outside the workspace, install the contracts from git first:

```
pip install "lerobot-mp-api @ git+https://github.com/machinekind/digital_twin_training.git@40ee4527f9b19a43a5a139c091ecf854b5c6db1f#subdirectory=packages/lerobot_mp_api"
```

GPU training needs MuJoCo Warp.
Until the JAX trainer exists, it runs the legacy torch trainer and needs a CUDA build of torch:

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
pip install "soarm-lab[train]" # mujoco-warp and jax: GPU training (the JAX trainer is not written yet)
pip install "soarm-lab[torch]" # torch: the legacy PPO trainer and soarm-lab convert
```

Without extras, `soarm-lab` runs policies, evaluates them and runs the strategy in the twin.

## Test

```
uv run pytest -q                       # both packages
uv run pytest -q -m "not twin"         # without the twin
uv run pytest -q packages/soarm_lab    # one package
uv run ruff check .
uv run pyright                         # the strategies, against the twin's contracts
```

Each package keeps its tests in its own `tests/` folder.
The test files mirror the package's modules: `packages/soarm_lab/tests/rl/test_env.py` tests `soarm_lab/rl/env.py`.
Tests that run a package through the twin's own loaders, backend factory and runner are marked `twin`.
They skip when the twin is not installed.
Tests that need MuJoCo skip when it is not installed. Tests that need torch skip without it.
Tests marked `render` skip where MuJoCo cannot open an OpenGL context. On a headless Linux machine set `MUJOCO_GL=egl`.
GPU tests skip without `mujoco-warp` or CUDA.
[docs/conventions.md](docs/conventions.md#tests) has the test conventions.

## Docs

- [docs/first-contact.md](docs/first-contact.md): plug in an arm for the first time, safely.
- [docs/safety.md](docs/safety.md): safety rules.
- [docs/conventions.md](docs/conventions.md): units, joint names, the dependency rule and where the contracts live.
- [packages/soarm_lab/README.md](packages/soarm_lab/README.md): stations, training, evaluation and the policy strategy.

## License

Apache-2.0. See [LICENSE](LICENSE).
The SO-101 MJCF and meshes come from MuJoCo Menagerie under Apache-2.0. See `packages/soarm/soarm/robot/`.
