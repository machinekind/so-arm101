# soarm

The SO-101 robot as a Python package. Distribution name `soarm`, import name `soarm`.

It holds the robot description and the hardware backends.
It imports neither the twin (`lerobot_mp`) nor `soarm_lab`.
The twin loads it as a plugin through entry points.

## Folders

| Folder | Holds |
|---|---|
| `robot/so101/` | MJCF, meshes and `robot.yaml` of the SO-101 with the stock gripper |
| `robot/tri-arm/` | `robot.yaml` of the SO-101 with a hand (draft) |
| `end_effectors/` | gripper and hand data |
| `mounts/` | transform of each mount (table, inverted, wrist camera, top camera) |
| `embodiment/` | embodiment descriptors for VLA tooling |
| `arms/` | one folder per physical arm (not in the wheel) |
| `soarm/` | the Python package |

`robot/`, `end_effectors/`, `mounts/` and `embodiment/` ship inside the wheel under `soarm/`.

## Python package

| Module | State | Holds |
|---|---|---|
| `spec.py` | working | `RobotSpec`, `load(path)`, `so101()` |
| `backend.py` | working | `Backend` protocol, same shape as the twin's `RobotBackend` |
| `cli.py` | skeleton | the `soarm` command; subcommands exit with code 2 |
| `feetech.py` | stub | servo bus driver and `FeetechArm` |
| `bridge.py` | stub | arm over the network: `serve` and `BridgeArm` |
| `lerobot.py` | stub | `LeRobotArm` over LeRobot |
| `servo.py` | stub | scan, inspect, EEPROM limits, torque off |
| `leader.py` | stub | read-only leader arm |
| `calibration.py` | stub | LeRobot calibration to ticks |

## Use

```python
import soarm

spec = soarm.so101()
spec.joints        # ('shoulder_pan', ..., 'gripper')
spec.mjcf_path     # .../robot/so101/so101.xml
```

## Entry points

| Group | Name | Object |
|---|---|---|
| `lerobot_mp.robots` | `so101` | `soarm.spec:so101` |
| `lerobot_mp.backends` | `feetech` | `soarm.feetech:FeetechArm` |
| `lerobot_mp.backends` | `bridge` | `soarm.bridge:BridgeArm` |
| `lerobot_mp.backends` | `lerobot` | `soarm.lerobot:LeRobotArm` |

## Extras

- `feetech`: pyserial, for the Feetech and bridge backends.
- `lerobot`: LeRobot 0.6 with Feetech support. Needs Python 3.12 or newer.
- `dev`: pytest and ruff.
