#!/usr/bin/env bash
# Installs the SO-101 gripper description into a GraspGenX clone as the gripper `so101`: config.json,
# gripper.urdf and the six STL meshes the URDF names, copied from the robot package (packages/soarm), which
# keeps the only copy of the meshes in this repository.
#
#   bash tools/graspgenx/install_gripper.sh ~/graspgenx
#
# GraspGenX looks for a gripper in its gripper_descriptions checkout first, then in
# <assets>/x_grippers/<name>/ (`--assets_dir`, by default <clone>/assets), then in proc_grippers/: the
# description lands in the second place, so the gripper_descriptions dataset is not needed for it.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <GraspGenX clone>" >&2
    exit 2
fi
clone="$1"
if [ ! -d "$clone" ]; then
    echo "not a folder: $clone" >&2
    exit 1
fi
here="$(cd "$(dirname "$0")" && pwd)"
assets="$here/../../packages/soarm/soarm/robot/assets"
dest="$clone/assets/x_grippers/so101"
meshes=(
    sts3215_03a_v1
    wrist_roll_follower_so101_v1
    wrist_roll_follower_so101_gripper_part0_v1
    moving_jaw_so101_v1
    moving_jaw_so101_gripper_part0_v1
    moving_jaw_so101_gripper_part1_v1
)

mkdir -p "$dest/meshes"
cp "$here/so101/config.json" "$here/so101/gripper.urdf" "$dest/"
for mesh in "${meshes[@]}"; do
    cp "$assets/$mesh.stl" "$dest/meshes/"
done
echo "so101 -> $dest"
