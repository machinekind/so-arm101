# soarm

Hardware control and the robot model for the SO-101 arm. This package provides
Feetech STS3215 and LeRobot backends, a browser control panel, and a serial-to-TCP
bridge. It also registers the robot and backends as plugins for the
[Machinekind digital twin](https://github.com/machinekind/twin).

## Get started

From the repository root, follow the [setup instructions](../../README.md#get-started)
and read the [safety rules](../../docs/safety.md) before powering the arm.

```bash
uv sync
uv run soarm-panel
```

The panel opens in your browser with joint and gripper controls, measured positions,
Home, STOP and torque controls. It automatically selects a single USB-serial device;
otherwise, specify the port:

```bash
uv run soarm-panel --port /dev/ttyACM0
uv run soarm-panel --port COM5 --no-browser
```

Targets are constrained by both the robot's configured limits and the servo's
EEPROM limits. STOP or Esc holds the measured pose with torque on. Closing the
page or losing its heartbeat for 1.5 seconds also stops motion; Ctrl-C exits while
keeping torque on. Support the arm before using the Torque button to release it.
The panel listens on localhost by default; anyone who can reach its port can move the arm.

See the [first-contact guide](../../docs/first-contact.md) for assembly checks and calibration.

## Serial bridge

Run the bridge in one terminal, then the panel in another to use the port over TCP:

```bash
uv run soarm-bridge --port COM5 --listen 127.0.0.1:5555
uv run soarm-panel --port socket://127.0.0.1:5555
```

For remote access, bind to the host's network address and restrict clients with
`--allow <client-ip>`. The Feetech backend accepts `socket://host:port` directly.

## Servo gains

Each STS3215 servo holds its joint with its own position loop and three gains:

- **P** sets how hard the servo pulls toward its goal for a given error.
- **D** damps the motion and calms overshoot.
- **I** slowly pushes away an error that stays. It is usually 0.

The servos keep the gains in their own memory. The factory P is 32. LeRobot writes P 16 to stop
the arm shaking, so an arm set up with LeRobot holds softly. Under load, a joint stops short of its
target, and a small correction may not move it at all. A higher P holds the arm closer to its target
under load; too high and a joint starts to shake. The shoulder and the elbow carry the arm against
gravity, so they gain the most. Raise P a step at a time and watch for shaking.

Set the gains per joint in the `feetech:` section of the twin config:

```yaml
feetech:
  p_gain: {shoulder_lift: 40, elbow_flex: 40, wrist_flex: 40}
```

`d_gain` and `i_gain` work the same way. Values are whole numbers: P 1..254, D and I 0..254.
A joint left out keeps its servo's own value, and without gains in the config nothing is written.
The backend writes the gains at every connect, after it sets each goal to where the arm stands and
before torque on, so the arm does not move. It leaves the EEPROM lock alone: with the lock on, a
written value lasts until power off, and the next connect writes it again. The backend logs the
gains each servo had at connect and the ones it set, and warns when a servo did not keep one.
A read-only arm only logs them. `soarm-panel` and `soarm-zero` use the same section with
`--config twin.yaml`.

## Integration and dependencies

The twin discovers robot `so101` and backends `feetech` and `lerobot` through
entry points. The [robot folder](soarm/robot/README.md) contains its MuJoCo model,
meshes and `robot.yaml` configuration.

The twin can also watch an SO-101 it does not drive, such as a second arm on the
stand. `feetech` then opens the arm read-only: only PING, READ and SYNC READ reach
the servos, so it never sets torque or goals, also not on connect or disconnect.
`lerobot` has no read-only mode.

The base dependency is `machinekind-twin-api`. Optional extras provide:

- `feetech`: serial communication and the bridge.
- `panel`: the browser control panel.
- `lerobot`: the LeRobot backend, requiring Python 3.12+.

The workspace's default `uv sync` includes `feetech` and `panel`.
See [conventions](../../docs/conventions.md) for units and plugin contracts,
and [soarm-lab](../soarm_lab/README.md) for simulation and training.

## License

[Apache-2.0](LICENSE), with project attribution in [NOTICE](NOTICE).
The bundled robot model and meshes retain their upstream license and
[third-party notices](THIRD_PARTY_NOTICES).
