# Conventions

## Dependency rule

- Neither package imports the twin (`lerobot_mp`).
- `soarm_lab` may import `soarm`. `soarm` never imports `soarm_lab`.
- The twin imports `soarm`. It never imports `soarm_lab`.

The twin and `soarm_lab` meet only through data and entry points:

- the twin exports the measured station as a folder (`lerobot-twin export`): `scene.xml`, `assets/`, `station.json`;
- the twin finds the lab's strategy under the `lerobot_mp.strategies` entry point and runs it through its `StrategyRunner`.

The twin is a dev dependency. The tests in `tests/twin/` run both packages through its loaders, contracts
and runner, and skip when it is not installed. No other test imports it.
`tests/test_dependency_rule.py` checks every import in both packages' source, also one under `TYPE_CHECKING`.
It also imports every lab module in a fresh interpreter and checks that no `lerobot_mp` module got loaded,
and it checks that no test outside `tests/twin/` imports `lerobot_mp`.

Running a policy needs no torch.
The strategy, the policy file, evaluation and the `soarm-lab` commands other than `train` and `convert`
import numpy, never torch.
torch is the optional extra `soarm-lab[torch]` for the legacy torch trainer and `soarm-lab convert`.
`tests/test_dependency_rule.py` runs the bundled policies in a fresh interpreter and checks that no torch
module got loaded, also with torch made unimportable.

The twin holds no robot and no training. It keeps the scene, cameras, calibration, perception primitives,
runtime, safety, sysid and the contracts `Observation`, `Strategy`, `StrategyRunner` and `EpisodeLog`.

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

The twin owns the contracts this package follows.

- The `robot.yaml` schema is `RobotSpec` in `lerobot_mp.twin.robots`. `load_robot(folder)` checks a folder strictly.
- The backend contract is `BackendPlugin` and `BackendContext` in `lerobot_mp.robot.base`.
  A backend factory `create(ctx)` takes a `BackendContext` and returns the backend.
- `packages/soarm/pyproject.toml` registers the robot folder under the `lerobot_mp.robots` entry point group
  and the backend factories under `lerobot_mp.backends`.
- The strategy contract is `Strategy`, `Observation`, `RunInfo` and `Action` in `lerobot_mp.twin.strategy`,
  and the strict settings dataclass in `lerobot_mp.twin.strategies`.
  `soarm_lab.strategies.policy.PolicyStrategy` follows it by attributes, without importing it.
- `packages/soarm_lab/pyproject.toml` registers that strategy under `lerobot_mp.strategies` as `policy`.
- The station export is `Workspace.export` in the twin (`lerobot-twin export`). `soarm_lab.station.Station` reads it.
  `soarm_lab.robot.RobotSpec` reads only the `robot.yaml` fields the lab uses and skips the others.
