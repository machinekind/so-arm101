# Gripper

The stock SO-101 gripper: one fixed jaw and one moving jaw on servo 6.

This folder will hold the jaw meshes, the opening range and the pad geometry.
The meshes in use today are in `robot/so101/assets/` (`moving_jaw_so101_v1.stl`, `wrist_roll_follower_so101_*`).
The opening range in use today is in `robot/so101/robot.yaml`:
`gripper_closed_ticks` (0) and `gripper_open_ticks` (100) under `hardware`.
