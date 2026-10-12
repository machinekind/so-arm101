# First contact with an arm

How to plug in an SO-101 for the first time, safely. Read [safety.md](safety.md) first.

The `soarm` subcommands below are not implemented yet. They print "not implemented" and exit with code 2.
Until ticket #145 lands, the read-only leader reader in galaxeo-manipulators (`so101_feetech.py`)
and the twin's `feetech` backend do the same jobs.

## 1. Before power

- Put the arm on the table in its rest pose. Nothing else within reach.
- Check that the servo IDs run 1 to 6 from the base to the gripper. LeRobot's motor setup assigns them.
- Have the motor power plug within reach. Unplugging it is your stop.

## 2. Plug in, read only

- Plug in USB, then motor power. The servos stay limp. Nothing enables torque yet.
- Find the port. macOS: `/dev/cu.usbmodem*`. Linux: `/dev/ttyACM*`. Windows: `COM<n>`.
- `soarm scan --port <port>`: every ID from 1 to 6 must answer at 1 Mbaud.
- `soarm inspect --port <port> --id <n>`: voltage, temperature, position and EEPROM limits of one servo.
  Voltage under 9 V means the motor supply is off or wrong.

## 3. Check directions by hand

- Move each joint by hand and watch the reading (`soarm leader --port <port>` prints all joints).
- Joint zero is mid-range, tick 2048. Compare the direction of each joint with the twin's 3D view.

## 4. Measure this arm

Create `packages/soarm/arms/<arm-id>/` from `arms/example/` and fill in `arm.yaml`:

- `serial_number` and `port`.
- `eeprom_limits`: `soarm limits --port <port>`.
- `gripper_closed_ticks` and `gripper_open_ticks`: close the jaw by hand and read the gripper position, then open it and read again.
  Wrong gripper ticks make the twin squeeze too hard or leave the jaw open.

## 5. First motion

- Connect through the twin. Its safety supervisor checks every command.
- Start from the twin's simulation of the same task. Then switch the backend to the real arm.
- Keep one hand on the motor power plug.

## 6. Done

- Bring the arm to its rest pose through the twin.
- Support the arm before you release torque (`soarm torque-off --port <port>`), or leave torque on and unplug the motor power while holding the arm.
