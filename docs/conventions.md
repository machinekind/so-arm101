# Conventions

## Dependency rule

- Both packages import the twin's plugin contracts, `lerobot_mp_api`. Neither imports the twin, `lerobot_mp`.
- `soarm_lab` may import `soarm`. `soarm` never imports `soarm_lab`.
- The twin imports `soarm`. It never imports `soarm_lab`.

The twin and `soarm_lab` meet only through data and entry points:

- the twin exports the measured station as a folder (`lerobot-twin export`): `scene.xml`, `assets/`, `station.json`;
- the twin finds the lab's strategy under the `lerobot_mp.strategies` entry point and runs it through its `StrategyRunner`.

The contracts are a base dependency of both packages, `lerobot-mp-api>=0.1,<0.2`. They need only numpy.
The twin is a dev dependency. Tests that run a package through the twin's loaders, factory and runner are
marked `twin` and skip when it is not installed. A test that imports only `lerobot_mp_api` needs no marker.
Each package's `tests/test_dependency_rule.py` checks every import in the package's source, also one under
`TYPE_CHECKING`. It checks that every test module that imports `lerobot_mp` is marked `twin`.
The checks compare whole package names, so `lerobot_mp_api` never counts as `lerobot_mp`.
The test also collects the package's tests with `-m "not twin"` where `lerobot_mp` cannot be imported.
The lab's test also imports every lab module in a fresh interpreter. It checks that `lerobot_mp_api` got
loaded and no `lerobot_mp` module did.

Running a policy needs no torch.
The strategy, the policy file, evaluation and the `soarm-lab` commands other than `train` and `convert`
import numpy, never torch.
torch is the optional extra `soarm-lab[torch]` for the legacy torch trainer and `soarm-lab convert`.
`packages/soarm_lab/tests/test_dependency_rule.py` runs the bundled policies in a fresh interpreter and
checks that no torch module got loaded, also with torch made unimportable.

The twin holds no robot and no training. It keeps the scene, cameras, calibration, perception primitives,
runtime, safety, sysid, `StrategyRunner` and `EpisodeLog`.

## Tests

Each package keeps its tests next to its source, in `packages/soarm/tests/` and `packages/soarm_lab/tests/`.
A test file mirrors the module it tests. `soarm_lab/rl/env.py` is tested in `tests/rl/test_env.py`.
A test that runs a module through the twin's own code may get a file of its own next to the module's test file.
`tests/strategies/test_policy_in_twin.py` sits next to `tests/strategies/test_policy.py`.

A test module that imports the twin declares `pytestmark = pytest.mark.twin`.
It calls `pytest.importorskip("lerobot_mp")` before its twin imports, so it skips without the twin.
The root `conftest.py` skips every test marked `twin` when the twin is not installed.
It also skips tests marked `render` where MuJoCo cannot open an OpenGL context.

```
uv run pytest -q                       # both packages
uv run pytest -q -m "not twin"         # without the twin
uv run pytest -q packages/soarm_lab    # one package
```

Fixtures shared by a package's tests live in its `tests/conftest.py`.
pytest imports test modules with `--import-mode=importlib`, so both packages can have test files of the same name.
A test module cannot import another test module.
A helper that several test modules use is a module of its own on pytest's `pythonpath` (`pyproject.toml`).
`packages/soarm/tests/fake_bus.py` is one.

## Units

| Quantity | Unit |
|---|---|
| Arm joints in `robot.yaml`, backends and strategies | degrees |
| Gripper | 0..100, 0 closed, 100 open |
| Joint speed limits | degrees per second (gripper: units per second) |
| MJCF joints | radians (hinge) or metres (slide) |
| Lengths and positions | metres |
| Servo positions | ticks, 0..4095 per turn |

These are LeRobot's units. Conversion to MJCF units happens in `RobotKinematics` and nowhere else:
the twin's `lerobot_mp.twin.kinematics` and `soarm_lab.kinematics`.
Joint zero sits at mid-range, tick 2048 (`center_ticks`). The MJCF uses the same zero
(LeRobot's new calibration convention).

## Joint names

SO-101 joints, from the base to the gripper, with their servo IDs:

| ID | Joint |
|---|---|
| 1 | `shoulder_pan` |
| 2 | `shoulder_lift` |
| 3 | `elbow_flex` |
| 4 | `wrist_flex` |
| 5 | `wrist_roll` |
| 6 | `gripper` |

Names are identical in `robot.yaml`, the MJCF, LeRobot and the twin.

## Contracts

The twin owns the contracts these packages follow.
The plugin contracts are the package `lerobot-mp-api` (see the root README for how it installs).

- The `robot.yaml` schema is `RobotSpec` in `lerobot_mp.twin.robots`. `load_robot(folder)` checks a folder strictly.
- The backend contract is `lerobot_mp_api.backend`: `BackendContext`, `BackendPlugin`, `RobotBackend` and `RobotInfo`.
  A backend factory `create(ctx)` takes a `BackendContext` and returns the backend.
  `FeetechArm` and `LeRobotArm` subclass `RobotBackend`, so the twin uses them without its `PluginBackend` wrapper.
- `lerobot_mp_api.entry_points` names the entry point groups: `ROBOTS`, `BACKENDS` and `STRATEGIES`.
  `packages/soarm/pyproject.toml` registers the robot folder under `lerobot_mp.robots` (`ROBOTS`)
  and the backend factories under `lerobot_mp.backends` (`BACKENDS`).
- The strategy contract is `lerobot_mp_api.strategy`: `Strategy`, `RunInfo`, `Observation`, `ObjectPose` and `Action`.
  The twin builds a strategy's settings dataclass strictly (`lerobot_mp.twin.strategies`).
  `soarm_lab.strategies.policy.PolicyStrategy` takes `RunInfo` and `Observation` and returns an `Action`.
- `packages/soarm_lab/pyproject.toml` registers that strategy under `lerobot_mp.strategies` (`STRATEGIES`) as `policy`.
- The station export is `Workspace.export` in the twin (`lerobot-twin export`). `soarm_lab.station.Station` reads it.
  `soarm_lab.robot.RobotSpec` reads only the `robot.yaml` fields the lab uses and skips the others.
