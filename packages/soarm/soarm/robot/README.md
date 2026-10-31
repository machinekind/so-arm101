# SO-101 robot model

The SO-101 MuJoCo model, meshes and [robot configuration](robot.yaml) used by
soarm, soarm-lab and the Machinekind digital twin. The configuration defines
joint names, limits, the home pose and gripper mapping.

The MJCF and meshes come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/robotstudio_so101)
(commit `ac6b2b0`), under the original
[Apache-2.0 license](LICENSE). `so101.xml` and all 19 meshes in `assets/` are
unchanged from that commit. The upstream README includes the attribution
"developed by I2RT Robotics".

Menagerie derives the model from
[TheRobotStudio's SO-101 simulation](https://github.com/TheRobotStudio/SO-ARM100/tree/aec17bbc256d1a7342d53aaa4950595d4c30b40d/Simulation/SO101),
adding collision geometry, manipulation solver settings and a wrist camera.
See [third-party notices](../../THIRD_PARTY_NOTICES) for the pinned sources and
attribution. `robot.yaml` is this project's configuration.

The model follows LeRobot's new calibration convention, with joint zero at the
calibrated midpoint. Arm angles convert from degrees to MJCF radians; the gripper's
0–100 range maps to the jaw angles configured in `robot.yaml`.
See [conventions](../../../../docs/conventions.md) for units and calibration details.

`robot.yaml` also declares the MJCF's wrist camera for the twin, under `cameras: wrist`: its pose on
`camera_mount` as T_cam2body in OpenCV axes, its vertical field of view and the MJCF's 1920x1080. The
comment above the entry converts the MJCF's values. The twin renders the camera on a simulated arm and
serves the workspace camera named `wrist` on a real one; soarm-lab's `grasp_wrist` strategy reads the
entry through `soarm_lab.kinematics.RobotKinematics.camera`.
