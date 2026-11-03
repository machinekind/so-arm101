# GraspGenX block libraries

`grasp_source: library` uses offline grasps for each block. This folder supplies the SO-101
gripper description and generation recipe. The bundled cube library is hand-written.
No GraspGenX run has been made. Plan each GPU run and get an explicit go before starting.

## Environment and downloads

[GraspGenX](https://github.com/NVlabs/GraspGenX) is pinned to
`b9429097728cb1c430dd78b92edf17ba318aad03`. It samples grasps with a diffusion model and
scores them with a discriminator for a gripper defined by URDF and config.json.
Its code is Apache-2.0 with an NVIDIA copyright line. Checkpoints from
`adithyamurali/GraspGenXModel` use the NVIDIA Open Model License.

First import clones weights into `<clone>/ext/graspgenx_checkpoints/release/{gen,dis}`.
`GRASPGENX_CHECKPOINT_DIR` overrides this path. It also clones the 315 MB
`adithyamurali/gripper_descriptions` dataset. An override through `GRASPGENX_GRIPPER_CFG_DIR`
must exist and contain `gripper_descriptions/assets/x_grippers/`. If authentication is
requested, run `hf auth login` in the shell. Keep its token there. This setup does not read
`.env` for Hugging Face credentials. Checkpoints stay outside the repository and releases.

Run GraspGenX in its own clone and venv on a CUDA GPU. Inference is CUDA-only. Its project
requires Python 3.10 or later, `torch>=2.1,<2.7`, `numpy==1.26.4` and `trimesh==4.5.3`.
This workspace uses NumPy 2.2 and a `torch>=2.7` extra. The lab exchanges mesh and grasp
files with that environment.

## SO-101 gripper description

`so101/config.json` supplies the ten keys written by the upstream configuration wizard.
`base_rotation = Rx(180 deg)` aligns the MJCF gripper body to +Z approach and +X closing
toward the moving jaw. Values below use this aligned frame in metres. Tip positions refer
to `fixed_jaw_sph_tip1` and `moving_jaw_sph_tip1` at the arm's home pose.

| Command | Fixed tip | Moving tip | Tip gap | Moving tip behind the fixed one |
|---|---|---|---|---|
| 0 (closed) | (-0.0081, 0, 0.1010) | (0.0008, 0.0004, 0.0999) | 9.0 mm | 1.1 mm |
| 35 | same | (0.0296, 0.0004, 0.1018) | 37.7 mm | -0.8 mm |
| 42 | same | (0.0353, 0.0004, 0.1009) | 43.4 mm | 0.1 mm |
| 50 | same | (0.0418, 0.0004, 0.0993) | 49.9 mm | 1.7 mm |
| 75 | same | (0.0607, 0.0004, 0.0911) | 69.5 mm | 9.9 mm |
| 100 (open) | same | (0.0769, 0.0004, 0.0783) | 88.0 mm | 22.7 mm |

| Key | Value | Where it comes from |
|---|---|---|
| `open`, `close` | `gripper` 0.9541361, -0.0951068 rad | Servo commands 100 and 0 give 88.0 and 9.0 mm travel |
| `sweep_volume.extents`, `.offset` | 0.085 x 0.020 x 0.057 at (0.0344, 0, 0.0736) | the open box: the tips' x span (-0.0081..0.0769), the pads' width (`fixed_jaw_box7` is 20 mm), the fixed pads' z span (`fixed_jaw_box4..7`, 0.045..0.1022) |
| `sweep_volume.extents2`, `.offset2` | 0.050 x 0.020 x 0.057 at (0.0169, 0, 0.0736) | the half-open box: the same at command 50 (x -0.0081..0.0418) |
| `fingertip` | (0.0344, 0, 0.0736) | the open box's centre, as the wizard writes it (`fingertip = sv_offset`) |
| `standoff` | (0, 0.0285) | (0, the open box's z extent / 2), as the wizard writes it |
| `bbox` | (-0.0356, -0.0240, -0.0189) .. (0.0824, 0.0280, 0.1048) | every geom of the gripper and moving-jaw bodies, visual meshes included, the camera mount left out, jaw open |
| `type` | `revolute_2f` | the moving jaw is a hinge: its tip swings 22.7 mm back along the approach as it opens, which a `parallel_2f` tip would not |
| `symmetric` | `false` | One jaw is fixed. A half turn swaps the pushing jaw. The lab adds this variant. The wizard defaults to true for antipodal grippers. |
| `base_rotation` | diag(1, -1, -1, 1) | Rx(180 deg): the body's -z (toward the tips) becomes +Z, its +x (toward the moving jaw) stays +X |
| `links` | world, gripper, moving_jaw | the URDF's links |

`gripper.urdf` converts the MJCF gripper subtree to URDF with a revolute joint. It excludes
the camera mount and primitive collision geoms. The installer copies its six referenced
meshes. The header preserves provenance and identifies the changes under Apache-2.0.
See [THIRD_PARTY_NOTICES](../../packages/soarm/THIRD_PARTY_NOTICES).

At the pinned commit, `get_gripper_info` reads fingertip depth, bbox width and depth,
sweep volumes, standoff, type and symmetry. `open`, `close`, `links` and `base_rotation`
serve the wizard and viewer. Missing `points.json`,
`proc_gripper_only_pointnet_vae_repr.json`, `tsdf.npy`, `coll_mesh.obj` and `vis_mesh.obj`
use dummy values or meshes. Check for five warnings and usable output on the first run.

Install and inspect the description before committing a generated library:

    bash tools/graspgenx/install_gripper.sh <clone>
    cd <clone>
    uv run python scripts/list_grippers.py
    uv run python scripts/vis_gripper_desc.py --gripper so101 --root <clone>/assets/x_grippers --port 8081

The blue open box must span the tips and pads, and the orange box covers the half-open
sweep. The animation must swing the moving jaw. Installation uses
`<clone>/assets/x_grippers/so101/`, searched after the gripper_descriptions checkout and
before `proc_grippers/`. The custom gripper does not require the dataset description.

## Generation

`jobs/grasps.sh` performs the file workflow below on an x86_64 CUDA box. gpu-ops syncs this
checkout and pulls `workspace/grasps` through `OPS_WATCH_PULL` in `.gpu-ops.example`.
DGX aarch64 CUDA wheels for torch<2.7 are unverified. A vast.ai box through gpu-ops is the
fallback.

1. Export meshes from `blocks.block`:

       uv run soarm-lab grasps meshes --out workspace/grasps/meshes

2. Clone `https://github.com/NVlabs/GraspGenX.git`, check out the pinned commit, then run
   `uv python pin 3.11 && uv sync` inside the clone.
3. Install and inspect the description as above.
4. Generate grasps per block inside the clone:

       uv run python scripts/demo_object_mesh.py --mesh_file <abs>/workspace/grasps/meshes/<block>.obj \
           --mesh_scale 1.0 --gripper_name so101 --num_grasps 400 --return_topk --topk_num_grasps 50 \
           --no-visualization --output_file <abs>/workspace/grasps/raw/<block>.yml

   The upstream README mentions `--moe_obb_density dense` for top grasps along cuboid long
   sides. The pinned script has no such flag. Check `--help` on the box before using it.
5. Convert in this repository:

       uv run soarm-lab grasps convert <block> workspace/grasps/raw/<block>.yml --generator graspgenx \
           --commit b9429097728cb1c430dd78b92edf17ba318aad03 --checkpoint release

6. Review with `uv run soarm-lab grasps show <block> --library workspace/grasps`.
7. Copy reviewed JSON into `packages/soarm_lab/soarm_lab/grasps/` with generator metadata.

## Frames

A GraspGenX pose maps its aligned gripper frame into the mesh frame. The root is the URDF
root, +Z is approach and +X is closing. The demo centres sampled points by translation and
maps results back, preserving the block mesh frame. This body frame is centred on the
block with x along its long side and z up at rest.

The TCP site has x approach and z closing. Its aligned-frame transform is:

    T_aligned_from_tcp = B @ inverse(T_gripper_body) @ T_tcp =
        [[0,  0, 1, 0.0120],
         [0, -1, 0, 0.0002],
         [1,  0, 0, 0.0981],
         [0,  0, 0, 1     ]]

Here B is `base_rotation`. This proper rotation has determinant +1 and maps TCP approach
(1, 0, 0) to +Z and closing (0, 0, 1) to +X. The block-frame TCP pose is
`T_grasp @ T_aligned_from_tcp`.

The library pad point is `T_grasp @ sweep_volume.offset`, the open box centre 27.4 mm behind
the fixed tip. The strategy places the fingertip midpoint there. Using the tips' original
point would put most grasps of a 25 mm block beyond its bottom.

## Regeneration and the example

Regenerate after catalog dimensions change, the gripper description changes or a new
GraspGenX commit is pinned. The loader checks stored half sizes against the catalog.

`examples/cube.yml` contains eight hand-written grasps in isaac_grasp format with scores
out of order. Its conversion is the bundled cube.json, checked by a test:

    uv run soarm-lab grasps convert cube tools/graspgenx/examples/cube.yml \
        --out packages/soarm_lab/soarm_lab/grasps/cube.json --generator hand \
        --note 'hand-written example for the tests; a GraspGenX run replaces it (tools/graspgenx/README.md)'
