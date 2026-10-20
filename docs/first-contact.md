# First contact with an arm

How to plug in an SO-101 for the first time, safely. Read [safety.md](safety.md) first.

The read-only leader reader in galaxeo-manipulators (`so101_feetech.py`) reads the servos without torque.
The twin's `twin ui` drives the arm through this package's `feetech` backend.

## 1. Before power

- Put the arm on the table in its rest pose. Nothing else within reach.
- Check that the servo IDs run 1 to 6 from the base to the gripper. LeRobot's motor setup assigns them.
- Have the motor power plug within reach. Unplugging it is your stop.

## 2. Plug in, read only

- Plug in USB, then motor power. The servos stay limp. Nothing enables torque yet.
- Find the port. macOS: `/dev/cu.usbmodem*`. Linux: `/dev/ttyACM*`. Windows: `COM<n>`.
- With the read-only reader, check that every ID from 1 to 6 answers at 1 Mbaud.
- Check the supply voltage. Voltage under 9 V means the motor supply is off or wrong.

## 3. Zero and directions

- Joint zero is tick 2048. Each servo decides which pose that is: it subtracts the homing offset stored in its EEPROM.
- LeRobot's calibration (`lerobot-calibrate`) stores that offset at its "middle of its range of motion" step.
  Hold the model's zero pose there: base facing forward, upper arm vertical, forearm level and pointing forward,
  wrist in line with the forearm, jaws one above the other.
  Any other pose puts the twin's 3D arm off the real one by the same angles.
- LeRobot turns torque on at connect without first setting each servo's goal to where it stands.
  An arm moved by hand since its last session lunges toward that session's pose, and a servo can drop its reply.
  Before `lerobot-calibrate`, support the arm and press Torque on, then off, in `soarm-panel`: torque on sets every goal to where the arm stands.
  Quit the panel and start the calibration without moving the arm.
- Move each joint by hand and watch the reading in the read-only reader.
  Compare the direction of each joint with the twin's 3D view.

## 4. Measure this arm

- `gripper_closed_ticks` and `gripper_open_ticks`: close the jaw by hand and read the gripper position, then open it and read again.
  Put both in the `feetech:` section of the twin's config file.
  Wrong gripper ticks make the twin squeeze too hard or leave the jaw open.
- EEPROM angle limits: the `feetech` backend reads them at connect.
  It warns when they are tighter than the joint limits in `robot.yaml`.

## 5. First motion

- Connect through the twin: run `twin ui`, pick the robot `so101`, then the `feetech` connection, and type the port.
  The backend sets each servo's goal to where the arm stands, then enables torque.
  The twin's safety supervisor checks every command.
- Start from the twin's simulation of the same task. Then switch the backend to the real arm.
- Keep one hand on the motor power plug.
- `soarm-panel` moves the arm from a browser page with a slider per joint (see `packages/soarm/README.md`).
  It finds the port itself when the arm is the only USB-serial device.

## 6. Done

- Bring the arm to its rest pose through the twin.
- The `feetech` backend keeps torque on when it disconnects.
  It releases torque only with `torque_off_on_exit: true` in its config section. Support the arm before you set that.
  Otherwise leave torque on and unplug the motor power while holding the arm.
