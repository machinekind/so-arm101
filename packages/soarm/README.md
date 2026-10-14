# soarm

The SO-101 robot as a package for the digital twin (`lerobot_mp`).
It holds the robot folder and the hardware backends.
It imports the twin's plugin contracts, `lerobot_mp_api`, and never the twin.
The twin finds both through entry points.

## Contents

| Path | Holds |
|---|---|
| `soarm/robot/` | `robot.yaml`, the MJCF `so101.xml` and its meshes, from MuJoCo Menagerie |
| `soarm/__init__.py` | `ROBOT_DIR`, the path of the robot folder |
| `soarm/feetech.py` | backend `feetech`: STS3215 servos over a serial port, or over `socket://` through the bridge |
| `soarm/lerobot_backend.py` | backend `lerobot`: the arm through LeRobot's SO follower |
| `soarm/serial_bridge.py` | the `soarm-bridge` command, which serves the arm's serial port over TCP |
| `soarm/settings.py` | `from_mapping`, the strict builder for the `feetech:` and `lerobot:` config sections |

The twin's `lerobot_mp.twin.robots` defines the `robot.yaml` schema.
`lerobot_mp_api.backend` defines the backend contract.
Each backend module has a `create(ctx)` factory that takes a `BackendContext`.
`FeetechArm` and `LeRobotArm` subclass `RobotBackend`.

## Entry points

```toml
[project.entry-points."lerobot_mp.robots"]
so101 = "soarm:ROBOT_DIR"

[project.entry-points."lerobot_mp.backends"]
feetech = "soarm.feetech:create"
lerobot = "soarm.lerobot_backend:create"

[project.scripts]
soarm-bridge = "soarm.serial_bridge:main"
```

## Dependencies

`lerobot-mp-api>=0.1,<0.2` is the only base dependency. It needs only numpy.

## Extras

- `feetech`: pyserial, for the `feetech` backend and `soarm-bridge`.
- `lerobot`: LeRobot 0.6 with Feetech support. It needs Python 3.12 or newer.
