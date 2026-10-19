# SO-101 robot model

The SO-101 MuJoCo model, meshes and [robot configuration](robot.yaml) used by
soarm, soarm-lab and the Machinekind digital twin. The configuration defines
joint names, limits, the home pose and gripper mapping.

The MJCF and meshes come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/robotstudio_so101)
(commit `ac6b2b0`), under [Apache-2.0](LICENSE). Menagerie derives the model from
[TheRobotStudio's SO-101 simulation](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),
adding collision geometry, manipulation solver settings and a wrist camera.

The model follows LeRobot's new calibration convention, with joint zero at the
calibrated midpoint. Arm angles convert from degrees to MJCF radians; the gripper's
0–100 range maps to the jaw angles configured in `robot.yaml`.
See [conventions](../../../../docs/conventions.md) for units and calibration details.
