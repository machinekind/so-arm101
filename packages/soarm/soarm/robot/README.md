# SO-101 robot model

The SO-101 MuJoCo model, meshes and [robot configuration](robot.yaml) used by
soarm, soarm-lab and the Machinekind digital twin. The configuration defines
joint names, limits, the home pose and gripper mapping.

The MJCF and meshes come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/robotstudio_so101)
(commit `ac6b2b0`), under the original
[Apache-2.0 license](LICENSE). All 19 meshes in `assets/` are unchanged from
that commit. Machinekind modified `so101.xml` from the Menagerie original.
Solver iterations increased from 10 to 20, and line search iterations from 20
to 50. The gripper's contact geoms use `solimp="0.9 0.995 0.001"`.
These settings limit jaw penetration and help the solver converge on stiff
contacts. The upstream README includes the attribution "developed by I2RT Robotics".

Menagerie derives the model from
[TheRobotStudio's SO-101 simulation](https://github.com/TheRobotStudio/SO-ARM100/tree/aec17bbc256d1a7342d53aaa4950595d4c30b40d/Simulation/SO101),
adding collision geometry, manipulation solver settings and a wrist camera.
See [third-party notices](../../THIRD_PARTY_NOTICES) for the pinned sources and
attribution. `robot.yaml` is this project's configuration.

The model follows LeRobot's new calibration convention, with joint zero at the
calibrated midpoint. Arm angles convert from degrees to MJCF radians; the gripper's
0–100 range maps to the jaw angles configured in `robot.yaml`.
See [conventions](../../../../docs/conventions.md) for units and calibration details.
