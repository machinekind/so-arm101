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

## robot.yaml

One file per robot folder in `packages/soarm/robot/<name>/`. `soarm.load(path)` reads it into a `RobotSpec`.
The fields mirror the twin's `RobotSpec` in `lerobot_mp/twin/robots.py`, plus `status`, `backend` and `hardware`.

The loader rejects unknown keys at every level, missing required keys and wrong types.
Every joint named in `gripper`, `roll_joint`, `home`, `limits`, `max_vel`, `wave_ranges`, `mirror_joints`,
`home_stages` and `hardware.servo_ids` must be in `joints`.
`limits`, `max_vel` and `hardware.servo_ids` must list every joint.

| Key | Required | Type | Meaning |
|---|---|---|---|
| `status` | no | `ready` or `draft` | strict tests skip drafts; default `ready` |
| `name` | yes | string | robot ID |
| `title` | yes | string | display name |
| `mjcf` | yes | path | arm-only MJCF, relative to the robot folder |
| `joints` | yes | list of names | application joints in order, names as in the MJCF |
| `gripper` | yes | name or null | the joint that runs in 0..100 |
| `tcp_site` | yes | name | MJCF site of the tool centre point |
| `hand_body` | yes | name | MJCF body that carries the calibration card |
| `base_body` | yes | name | MJCF body of the base frame |
| `backend` | yes | `module:Class` | default backend for this robot |
| `home` | yes | joint: value | rest pose |
| `limits` | yes | joint: [low, high] | application limits, narrower than the MJCF ranges |
| `max_vel` | yes | joint: value | largest commanded speed |
| `tcp_approach` | no | [x, y, z] | approach direction in the TCP frame |
| `tcp_closing` | no | [x, y, z] | jaw closing axis in the TCP frame |
| `fingertips` | no | [geom, geom] | MJCF geoms at the two jaw tips |
| `roll_joint` | no | name or null | joint that rolls the tool about the approach direction |
| `wave_ranges` | no | joint: [low, high] | ranges for the calibration wave |
| `onboard_cameras` | no | list of names | MJCF cameras mounted on the arm |
| `jaw_bodies` | no | [body, body] | fixed and moving jaw bodies |
| `gripper_range_q` | no | [closed, open] or null | gripper 0..100 mapped linearly in MJCF units; null maps through servo ticks |
| `mirror_joints` | no | joint: [source, ratio] | MJCF joints coupled to an application joint |
| `home_stages` | no | list of joint lists | homing in stages; empty means all joints at once |
| `jaw` | yes | mapping | `still`, `margin`, `creep`, `open` in MJCF units of the gripper joint |
| `desk` | yes | mapping | where the arm works on the table, base frame, metres and radians |
| `hardware` | yes | mapping | servo bus settings, below |

`hardware` keys, all required. The values come from the twin's `RobotConfig`:

| Key | Meaning |
|---|---|
| `kind` | arm type passed to LeRobot (`so101` or `so100`) |
| `servo_ids` | joint: Feetech servo ID |
| `baudrate` | serial port speed |
| `center_ticks` | tick of joint zero |
| `gripper_closed_ticks` | tick of gripper 0, or null |
| `gripper_open_ticks` | tick of gripper 100, or null |
| `use_degrees` | joints in degrees in LeRobot |
| `max_relative_target` | largest step from the current position in one command, degrees, or null |
| `read_state_hz` | how often the backend reads joint positions |
| `torque_off_on_exit` | release torque on disconnect |

Per-arm values (port, serial number, calibration, EEPROM limits) go to `packages/soarm/arms/<id>/arm.yaml`.

## Backend interface

`soarm.backend.Backend` is a `Protocol` with `connect`, `disconnect`, `read_joints`, `send_joints`,
`is_connected`, `link_silent`, `faults` and `joint_limits`.

It has the same shape as the twin's `RobotBackend` ABC in `lerobot_mp/robot/base.py`.
`soarm` restates the shape because it cannot import the twin.
The twin's ABC also has `info` and methods with default bodies: `gripper_ticks`, `calibration_warnings`,
`efforts`, `velocities` and `step`. The backends implement them when their code moves here (#145).

## Plugin entry points

`packages/soarm/pyproject.toml` declares them. The twin finds them with
`importlib.metadata.entry_points(group=...)`.

| Group | Name | Object |
|---|---|---|
| `lerobot_mp.robots` | `so101` | `soarm.spec:so101`, returns a `RobotSpec` |
| `lerobot_mp.backends` | `feetech` | `soarm.feetech:FeetechArm` |
| `lerobot_mp.backends` | `bridge` | `soarm.bridge:BridgeArm` |
| `lerobot_mp.backends` | `lerobot` | `soarm.lerobot:LeRobotArm` |

## Embodiment descriptors

`packages/soarm/embodiment/<name>.json`, schema `soarm-embodiment/0` (draft):

- `name`: embodiment ID.
- `robot`: robot folder name.
- `end_effector`: folder name in `end_effectors/`.
- `state.joints`, `action.joints`: joint order, equal to the robot's `joints`.
- `state.units`, `action.units`: units per joint group.
- `action.mode`: `absolute` joint targets.

## Mounts

`packages/soarm/mounts/<mount>/transform.yaml` holds `parent`, `child`, `pos` in metres,
`quat` as `[w, x, y, z]` and `cad`.
