#!/usr/bin/env bash
# Compute the grasp libraries of the wooden blocks with GraspGenX on a CUDA GPU box: the gpu-ops payload of
# tools/graspgenx/README.md, which explains each step.
#
# GraspGenX runs in a clone and a venv of its own: it pins torch<2.7 and numpy 1.26, which cannot share this
# workspace's environment. Its checkpoints (NVIDIA Open Model License) download into the clone on first
# import and stay on the box; only the grasp files come back. The dispatcher (gpu-ops, out of tree) syncs
# this checkout, runs this file from the repository root and pulls workspace/grasps back
# (OPS_WATCH_PULL in .gpu-ops). Without gpu-ops: `bash jobs/grasps.sh` in a checkout on a CUDA box.
#
#   bash jobs/grasps.sh                       # every block, 400 grasps sampled, the best 50 kept
#   KINDS=cube NUM_GRASPS=200 bash jobs/grasps.sh
#
# The reviewed JSON files are copied into packages/soarm_lab/soarm_lab/grasps/ by hand and committed.
#
: "${GRASPGENX_DIR:=$HOME/graspgenx}"     # the GraspGenX clone, cloned when absent
: "${GRASPGENX_REV:=b9429097728cb1c430dd78b92edf17ba318aad03}"   # the pinned commit
: "${KINDS:=cube cuboid5 cuboid85 cylinder}"   # space-separated: the blocks of rl/objects.py
: "${NUM_GRASPS:=400}"                    # grasps sampled per block
: "${TOPK:=50}"                           # the best this many are written
: "${GRASPS_OUT:=workspace/grasps}"       # meshes/, raw/ (GraspGenX's YAML) and <kind>.json
set -euo pipefail

cd "$(dirname "$0")/.."
root="$(pwd)"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi missing')"
# The lab's base dependencies are enough here (numpy, PyYAML, mujoco): no jax, no torch.
uv sync
mkdir -p "$GRASPS_OUT/meshes" "$GRASPS_OUT/raw"
uv run soarm-lab grasps meshes --out "$GRASPS_OUT/meshes"

if [ ! -d "$GRASPGENX_DIR/.git" ]; then
    git clone https://github.com/NVlabs/GraspGenX.git "$GRASPGENX_DIR"
fi
git -C "$GRASPGENX_DIR" fetch origin
git -C "$GRASPGENX_DIR" checkout --detach "$GRASPGENX_REV"
(cd "$GRASPGENX_DIR" && uv python pin 3.11 && uv sync)
bash tools/graspgenx/install_gripper.sh "$GRASPGENX_DIR"

for kind in $KINDS; do
    echo "== $kind"
    (cd "$GRASPGENX_DIR" && uv run python scripts/demo_object_mesh.py \
        --mesh_file "$root/$GRASPS_OUT/meshes/$kind.obj" --mesh_scale 1.0 --gripper_name so101 \
        --num_grasps "$NUM_GRASPS" --return_topk --topk_num_grasps "$TOPK" \
        --no-visualization --output_file "$root/$GRASPS_OUT/raw/$kind.yml")
    uv run soarm-lab grasps convert "$kind" "$GRASPS_OUT/raw/$kind.yml" --out "$GRASPS_OUT/$kind.json" \
        --generator graspgenx --commit "$GRASPGENX_REV" --checkpoint release
done

echo "== the libraries"
for kind in $KINDS; do
    uv run soarm-lab grasps show "$kind" --library "$GRASPS_OUT"
done
