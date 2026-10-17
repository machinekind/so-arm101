# First contact with an arm

How to plug in an SO-101 for the first time, safely. Read [safety.md](safety.md) first.

The read-only leader reader in galaxeo-manipulators (`so101_feetech.py`) reads the servos without torque.
The twin's `lerobot-twin ui` drives the arm through this package's `feetech` backend.

## 1. Before power

- Put the arm on the table in its rest pose. Nothing else within reach.
- Check that the servo IDs run 1 to 6 from the base to the gripper. LeRobot's motor setup assigns them.
- Have the motor power plug within reach. Unplugging it is your stop.

## 2. Plug in, read only

- Plug in USB, then motor power. The servos stay limp. Nothing enables torque yet.
- Find the port. macOS: `/dev/cu.usbmodem*`. Linux: `/dev/ttyACM*`. Windows: `COM<n>`.
- With the read-only reader, check that every ID from 1 to 6 answers at 1 Mbaud.
- Check the supply voltage. Voltage under 9 V means the motor supply is off or wrong.

## 3. Check directions by hand

- Move each joint by hand and watch the reading in the read-only reader.
- Joint zero is mid-range, tick 2048. Compare the direction of each joint with the twin's 3D view.

## 4. Measure this arm

- `gripper_closed_ticks` and `gripper_open_ticks`: close the jaw by hand and read the gripper position, then open it and read again.
  Put both in the `feetech:` section of the twin's config file.
  Wrong gripper ticks make the twin squeeze too hard or leave the jaw open.
- EEPROM angle limits: the `feetech` backend reads them at connect.
  It warns when they are tighter than the joint limits in `robot.yaml`.

## 5. First motion

- Connect through the twin: run `lerobot-twin ui`, pick the robot `so101`, then the `feetech` connection, and type the port.
  The backend sets each servo's goal to where the arm stands, then enables torque.
  The twin's safety supervisor checks every command.
- Start from the twin's simulation of the same task. Then switch the backend to the real arm.
- Keep one hand on the motor power plug.
- Without the twin, `soarm-panel --port <port>` moves the joints from the browser at 30 degrees per second by default (see `packages/soarm/README.md`).

## 6. Done

- Bring the arm to its rest pose through the twin.
- The `feetech` backend keeps torque on when it disconnects.
  It releases torque only with `torque_off_on_exit: true` in its config section. Support the arm before you set that.
  Otherwise leave torque on and unplug the motor power while holding the arm.
