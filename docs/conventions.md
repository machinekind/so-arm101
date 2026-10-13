# Conventions

## Dependency rule

- `soarm` imports neither the twin (`lerobot_mp`) nor `soarm_lab`.
- `soarm_lab` imports the twin and `soarm`.
- The twin imports `soarm`. It never imports `soarm_lab`.

`tests/test_dependency_rule.py` checks the first rule.

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

These are LeRobot's units. Conversion to MJCF units happens in the twin's `RobotKinematics` and nowhere else.
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
