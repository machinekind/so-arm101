# GraspGenX: the grasp libraries of the wooden blocks

The `grasp` strategy can take its grasps from a library per block (`grasp_source: library`, see
[the lab README](../../packages/soarm_lab/README.md#grasp-library)). The libraries are computed offline by
[GraspGenX](https://github.com/NVlabs/GraspGenX) on the block meshes. This folder holds what GraspGenX needs
for the SO-101 and the recipe to run it. No run has been made yet: the bundled `cube.json` is the
hand-written example below, and every run needs an explicit go.

## What GraspGenX is and where it runs

GraspGenX (NVlabs, pinned commit `b9429097728cb1c430dd78b92edf17ba318aad03`) is a diffusion grasp sampler
with a discriminator that scores each grasp, for any gripper described by a URDF and a `config.json`. Its
code is Apache-2.0 with an NVIDIA copyright line. Its weights, the Hugging Face repository
`adithyamurali/GraspGenXModel`, are under the NVIDIA Open Model License: they are cloned into
`<clone>/ext/graspgenx_checkpoints/release/{gen,dis}` on first import (override with
`GRASPGENX_CHECKPOINT_DIR`). The Hugging Face dataset `adithyamurali/gripper_descriptions` (315 MB) is
cloned too (override with `GRASPGENX_GRIPPER_CFG_DIR`, which must then exist and hold
`gripper_descriptions/assets/x_grippers/`). If a clone asks for credentials, log in with `hf auth login` in
the shell: the token stays there, never in this repository, and `.env` is not read for it. The weights are
never committed or redistributed.

GraspGenX runs in its own clone and venv on a CUDA GPU, never in this workspace. Inference is CUDA-only, and
its pyproject pins `torch>=2.1,<2.7`, `numpy==1.26.4` and `trimesh==4.5.3` (Python 3.10 or later), against
this workspace's numpy 2.2 and the `torch>=2.7` extra. The lab reads only the files GraspGenX writes: a data
boundary like `twin export`.

## The SO-101 description: `so101/`

`config.json` is hand-written, with the ten keys GraspGenX's wizard (`scripts/gripper_config_wizard.py`)
writes. Its frame is the MJCF `gripper` body frame turned by `base_rotation` = Rx(180 deg), so that +Z is
the approach (toward the tips) and +X the closing direction (toward the moving jaw). All values are in
metres. They come from a read-only MuJoCo probe of `so101.xml` at the arm's home pose; a tip is the centre
of `fixed_jaw_sph_tip1` or `moving_jaw_sph_tip1`.

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
| `open`, `close` | `gripper` 0.9541361, -0.0951068 rad | commands 100 and 0: the servo's travel (88.0 and 9.0 mm), not the MJCF range |
| `sweep_volume.extents`, `.offset` | 0.085 x 0.020 x 0.057 at (0.0344, 0, 0.0736) | the open box: the tips' x span (-0.0081..0.0769), the pads' width (`fixed_jaw_box7` is 20 mm), the fixed pads' z span (`fixed_jaw_box4..7`, 0.045..0.1022) |
| `sweep_volume.extents2`, `.offset2` | 0.050 x 0.020 x 0.057 at (0.0169, 0, 0.0736) | the half-open box: the same at command 50 (x -0.0081..0.0418) |
| `fingertip` | (0.0344, 0, 0.0736) | the open box's centre, as the wizard writes it (`fingertip = sv_offset`) |
| `standoff` | (0, 0.0285) | (0, the open box's z extent / 2), as the wizard writes it |
| `bbox` | (-0.0356, -0.0240, -0.0189) .. (0.0824, 0.0280, 0.1048) | every geom of the gripper and moving-jaw bodies, visual meshes included, the camera mount left out, jaw open |
| `type` | `revolute_2f` | the moving jaw is a hinge: its tip swings 22.7 mm back along the approach as it opens, which a `parallel_2f` tip would not |
| `symmetric` | `false` | one jaw is fixed, so a grasp turned 180 deg about the approach pushes the block with the other jaw; the lab adds that twin itself. GraspGenX documents the flag for antipodal grippers; the wizard's default is true |
| `base_rotation` | diag(1, -1, -1, 1) | Rx(180 deg): the body's -z (toward the tips) becomes +Z, its +x (toward the moving jaw) stays +X |
| `links` | world, gripper, moving_jaw | the URDF's links |

`gripper.urdf` is the gripper subtree of `so101.xml` written as URDF: the root link `gripper` is the MJCF
`gripper` body frame, the joint `gripper` is revolute, the camera mount and the primitive collision geoms
are left out, and the six meshes it names are copied by the installer. Its header records the provenance
(Apache-2.0, see `packages/soarm/THIRD_PARTY_NOTICES`).

What GraspGenX reads of a description, in `graspgenx/x_grippers.py` at the pinned commit: `get_gripper_info`
loads `config.json` and takes `fingertip` (its z is the gripper's depth), `bbox` (width
`bbox[1][0] - bbox[0][0]`, depth `bbox[1][2]`), `sweep_volume` (`extents`, `offset`, `extents2`,
`offset2`), `standoff`, `type` and `symmetric`. `open`, `close`, `links` and `base_rotation` serve the
wizard and the viewer. `points.json`, `proc_gripper_only_pointnet_vae_repr.json`, `tsdf.npy`,
`coll_mesh.obj` and `vis_mesh.obj` are optional: each missing one prints a `[WARNING] ... not found for
so101, using dummy values.` (`... using dummy mesh.` for the two meshes) and gets a dummy. To confirm on the
first run: five `[WARNING]` lines, and grasps still come out.

Install the description into a clone and check it by eye before any library is committed:

    bash tools/graspgenx/install_gripper.sh <clone>
    cd <clone>
    uv run python scripts/list_grippers.py
    uv run python scripts/vis_gripper_desc.py --gripper so101 --root <clone>/assets/x_grippers --port 8081

The blue open box must span the tips and the pads, the orange half-open box half of it, and the animation
must swing the moving jaw. The installer writes `<clone>/assets/x_grippers/so101/`, the second place
GraspGenX looks for a gripper (after the gripper_descriptions checkout, before `proc_grippers/`), so the
gripper_descriptions dataset is not needed for it.

## The run

`jobs/grasps.sh` does these steps; with gpu-ops it runs on a rented box and `workspace/grasps` comes back
(`OPS_WATCH_PULL` in `.gpu-ops.example`). Plan the run first and start it only on an explicit go.

1. The block meshes, from `rl/objects.py`: `uv run soarm-lab grasps meshes --out workspace/grasps/meshes`.
2. The clone: `git clone https://github.com/NVlabs/GraspGenX.git`, check out the pinned commit, then
   `uv python pin 3.11 && uv sync` inside it.
3. The description: `bash tools/graspgenx/install_gripper.sh <clone>`, and the visual check above.
4. Per block, in the clone:

       uv run python scripts/demo_object_mesh.py --mesh_file <abs>/workspace/grasps/meshes/<kind>.obj \
           --mesh_scale 1.0 --gripper_name so101 --num_grasps 400 --return_topk --topk_num_grasps 50 \
           --no-visualization --output_file <abs>/workspace/grasps/raw/<kind>.yml

   GraspGenX's README also documents `--moe_obb_density dense` (top-down grasps along a box's long side),
   worth a try on the cuboids if the default set is poor; `demo_object_mesh.py` at the pinned commit has no
   such flag, so check `--help` on the box first.
5. The conversion, in this repository:
   `uv run soarm-lab grasps convert <kind> workspace/grasps/raw/<kind>.yml --generator graspgenx --commit
   b9429097728cb1c430dd78b92edf17ba318aad03 --checkpoint release`.
6. The review: `uv run soarm-lab grasps show <kind> --library workspace/grasps`.
7. Copy the reviewed `<kind>.json` into `packages/soarm_lab/soarm_lab/grasps/` and commit it with its
   generator fields filled.

On the DGX (aarch64) the CUDA wheels of torch<2.7 are unverified; a vast.ai box through gpu-ops is the
fallback, and the payload assumes an x86_64 CUDA box.

## Frames

A GraspGenX grasp is the pose of its aligned gripper frame (+Z approach, +X closing, origin at the URDF
root) in the mesh's frame. `demo_object_mesh.py` centres the sampled points by a translation only and maps
the grasps back with its inverse, so the mesh frame is the block frame of `rl/objects.py`: the block's
centre, x its long side, z up as it lies on the table.

The strategy's TCP is the `gripperframe` site: x the approach, z the closing direction. In the aligned frame
it is, on the SO-101,

    T_aligned_from_tcp = B @ inverse(T_gripper_body) @ T_tcp =
        [[0,  0, 1, 0.0120],
         [0, -1, 0, 0.0002],
         [1,  0, 0, 0.0981],
         [0,  0, 0, 1     ]]

with B = `base_rotation`. Its rotation is proper (det +1) and maps the TCP's approach (1, 0, 0) to +Z and its
closing (0, 0, 1) to +X. A grasp's TCP pose in the block frame is `T_grasp @ T_aligned_from_tcp`: its x
column is the approach and its z column the closing direction the library stores.

The library's point is the PAD POINT, `T_grasp` applied to `sweep_volume.offset`: the centre of the open
box, the slab of the block GraspGenX puts between the pads, 27.4 mm behind the fixed tip. The strategy puts
its JAW POINT, the fingertips' midpoint, there: an executed grasp is GraspGenX's shifted back along the
approach by that distance. Taking the tips' own point instead would put most grasps of a 25 mm block past its
bottom.

## When to regenerate

* `rl.objects.MEASURED` turns True or any number of `BLOCKS` changes: the loader refuses a library whose
  `half_size` differs from the table;
* the gripper description changes;
* another GraspGenX commit is pinned.

## The hand-written example

`examples/cube.yml` is eight cube grasps written by hand in GraspGenX's own format, out of score order on
purpose. The bundled `packages/soarm_lab/soarm_lab/grasps/cube.json` is its conversion, and a test pins
that:

    uv run soarm-lab grasps convert cube tools/graspgenx/examples/cube.yml \
        --out packages/soarm_lab/soarm_lab/grasps/cube.json --generator hand \
        --note 'hand-written example for the tests; a GraspGenX run replaces it (tools/graspgenx/README.md)'
