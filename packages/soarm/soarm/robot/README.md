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

`robot.yaml` declares the wrist camera on `camera_mount` as `cameras: wrist`.
Its `pose` is T_cam2body in OpenCV axes. Field of view and image size define its intrinsics.
The twin renders it in simulation and serves the workspace camera named `wrist` on a real arm.
`RobotKinematics.camera` exposes its pose to soarm-lab strategies.

## Overhead camera mount

`fixtures/overhead_cam_32` is TheRobotStudio's
[overhead camera mount for a 32x32 mm UVC module](https://github.com/TheRobotStudio/SO-ARM100/tree/c8af7c4b7abe191b149c618bb36044425f4ad159/Optional/Overhead_Cam_Mount_32x32_UVC_Module)
as a fixture of the digital twin. Its pole stands on a foot that joins the plate the arm
stands on. The head holds the camera module looking down, tilted 25 degrees forward.

- Parts: `bottom`, `middle` and `top`. The full design, `[bottom, middle, top]`, is 0.56 m
  tall. Without the middle part, `[bottom, top]`, it is 0.41 m.
- Colliders: boxes fitted to each part. The top's camera head box also holds the module's
  board and lens.
- Camera site `camera`: the module's lens, with the top of the image forward.
- Placements `left` and `right`: the pole on the arm's left or right, as the design puts it.
  The SO-101's base sits on the arm plate by its four screw holes, and the plate's tabs join
  the foot's. The pole's axis then stands 2 mm behind the base origin and 11.6 cm to the
  side. The arm stands on the 7.2 mm plate, so the foot is 9.6 mm below the base origin.

Two parts on the arm's left, with `kamera3` on the mount, in `stand.json`:

```json
"fixtures": [
  {"name": "slup", "fixture": "so101/overhead_cam_32", "placement": "left", "parts": ["bottom", "top"]}
],
"cameras": [
  {"name": "kamera3", "source": "1", "width": 1920, "height": 1080, "fixture": "slup"}
]
```

Where the pole stands elsewhere, `xyz` (its foot in the arm's base frame, metres) and
`yaw_deg` replace `placement`. The twin's documentation, "Fixtures in the scene", lists
every key. `tools/overhead_cam_mount.py` in the repository builds the meshes and
`fixture.yaml` from the upstream STL files.
