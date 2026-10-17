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
| `soarm/panel.py` | the `soarm-panel` command: a browser test panel that moves the arm through the `feetech` backend |
| `soarm/panel.html` | the panel's page: plain HTML, CSS and JavaScript, no external assets |
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
soarm-panel = "soarm.panel:main"
```

## Dependencies

`lerobot-mp-api>=0.1,<0.2` is the only base dependency. It needs only numpy.

## Extras

- `feetech`: pyserial, for the `feetech` backend and `soarm-bridge`.
- `panel`: pyserial and pyyaml, for `soarm-panel`.
- `lerobot`: LeRobot 0.6 with Feetech support. It needs Python 3.12 or newer.

## Test panel

`soarm-panel` moves the arm from a browser page.
Use it to try a freshly assembled arm, to find a servo's limits, or to check that a port and a baud rate work.
It talks to the servos through the `feetech` backend:

```
soarm-panel --port /dev/ttyACM0
soarm-panel --port COM5 --listen 127.0.0.1:8765 --no-browser
```

Each joint has a column with its measured position, a slider for the target, and minus and plus buttons.
The step is 1, 5 or 15 degrees. The gripper moves 10 units per press and has Close and Open.
The speed is 10, 30 or 60 degrees per second, never above the joint's `max_vel` in `robot.yaml`.
Home, STOP and Torque act on the whole arm.
The header shows the port, the loop rate, the torque state, servo faults and the last target cut to a limit.

Targets stay inside the joint's limit from `robot.yaml`, narrowed to the servo's own EEPROM limit.
A target outside it is cut to the limit, and the page says so.

STOP, or Esc on the page, holds the arm where it stands, with torque on.
Closing the page does the same, and so does losing the page for 1.5 s while a joint is moving.
Ctrl-C ends the command and leaves torque on.
Only the Torque button turns torque off. The arm then drops under its own weight, so support it first.
Turning torque back on first sets each servo's goal to where the arm stands, so the arm does not jump.

The panel listens on 127.0.0.1 unless `--listen` says otherwise. Anyone who can reach the port can move the arm.
`tests/test_panel.py` runs it on the fake bus from `tests/fake_bus.py`, without hardware.
