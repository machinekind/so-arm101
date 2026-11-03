#!/usr/bin/env bash
# Generate block grasp libraries in a separate GraspGenX environment on a CUDA box.
# See tools/graspgenx/README.md for setup, licensing and review before generation.
# gpu-ops runs this from the repository root and pulls workspace/grasps via OPS_WATCH_PULL.
# Review the resulting JSON before copying it into the package.
#
#   bash jobs/grasps.sh
#   BLOCKS=cube NUM_GRASPS=200 bash jobs/grasps.sh
#
: "${GRASPGENX_DIR:=$HOME/graspgenx}"     # the GraspGenX clone, cloned when absent
: "${GRASPGENX_REV:=b9429097728cb1c430dd78b92edf17ba318aad03}"   # the pinned commit
: "${BLOCKS:=cube cuboid cylinder}"   # space-separated: soarm_lab/blocks/classic_world.py
: "${NUM_GRASPS:=400}"                    # grasps sampled per block
: "${TOPK:=50}"                           # the best this many are written
: "${GRASPS_OUT:=workspace/grasps}"       # meshes/, raw/ YAML and block JSON files
set -euo pipefail

cd "$(dirname "$0")/.."
root="$(pwd)"
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi missing')"
# Mesh export and conversion use the lab's base environment.
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

for kind in $BLOCKS; do
    echo "== $kind"
    (cd "$GRASPGENX_DIR" && uv run python scripts/demo_object_mesh.py \
        --mesh_file "$root/$GRASPS_OUT/meshes/$kind.obj" --mesh_scale 1.0 --gripper_name so101 \
        --num_grasps "$NUM_GRASPS" --return_topk --topk_num_grasps "$TOPK" \
        --no-visualization --output_file "$root/$GRASPS_OUT/raw/$kind.yml")
    uv run soarm-lab grasps convert "$kind" "$GRASPS_OUT/raw/$kind.yml" --out "$GRASPS_OUT/$kind.json" \
        --generator graspgenx --commit "$GRASPGENX_REV" --checkpoint release
done

echo "== the libraries"
for kind in $BLOCKS; do
    uv run soarm-lab grasps show "$kind" --library "$GRASPS_OUT"
done
