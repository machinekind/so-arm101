# Arms

One folder per physical arm. The folder name is the arm ID, for example `arms/left-follower/`.

A robot folder in `robot/` describes a model. An arm folder describes one unit of that model:
its serial number, port, EEPROM limits, gripper ticks, LeRobot calibration and system identification.

Each arm folder holds:

- `arm.yaml`: identity and measured values. Start from `example/arm.yaml`.
- `calibration.json`: the LeRobot calibration file of this arm, if it has one.
- `sysid/dynamics.json`: the servo dynamics fitted by the twin's sysid for this arm.
- `sysid/raw/`: raw sysid recordings. Git ignores this folder.

Values in `arm.yaml` override the defaults in the robot's `robot.yaml` for this arm only.
No loader reads `arm.yaml` yet.
Arm folders are not shipped in the wheel.
