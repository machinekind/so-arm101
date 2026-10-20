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

## Integration and dependencies

The twin discovers robot `so101` and backends `feetech` and `lerobot` through
entry points. The [robot folder](soarm/robot/README.md) contains its MuJoCo model,
meshes and `robot.yaml` configuration.

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
