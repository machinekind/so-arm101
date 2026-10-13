# Safety

An SO-101 is small, but it moves fast and its jaw pinches. These rules apply to every arm in this repository.

## Rules

1. Every autonomous motion goes through the twin's `SafetySupervisor`. It enforces position limits, speed limits and STOP.
   Backends in `soarm` move servos only when the twin tells them to.
2. The servo tools that come later read only. The one exception is a command that releases torque.
3. Know your stop before you power the arm. The fastest stop is the motor power plug.
4. Support the arm before torque goes off. A limp arm drops under its own weight.
   For that reason `torque_off_on_exit` is `false` by default.
5. Keep hands, cables and cameras out of reach during the first run of any new strategy.
6. Run a new strategy in simulation first. Then run it on the real arm at reduced speed.
7. Never run two arms autonomously in the same space unless the twin checks both for collisions.

## Facts the rules rest on

- Each servo stores its own angle limits in EEPROM. It clips every command to them without telling anyone.
  The `feetech` backend reads them at connect and warns when they are tighter than the joint limits.
- A servo in protection (overload, overheating, voltage) releases torque but keeps answering.
  Without its error bits it looks healthy. Backends report these bits through `faults()`.
- The limits in `robot.yaml` are narrower than the mechanical ranges in the MJCF. Tests check that.
- The `lerobot` backend turns off LeRobot's `max_relative_target`. Its 12 degree clamp hid collisions from the twin's tracking guard.
- The `feetech` backend writes goal = present position before it enables torque.
  The goal register keeps the value of the last session, so enabling torque first would make the arm jump.
- Below about 9 V the servos still answer (the board runs from USB) but cannot move.
  The `feetech` backend reports this at connect.
- The network bridge listening on `0.0.0.0` without `--allow` lets anyone who reaches the port move the arm.
