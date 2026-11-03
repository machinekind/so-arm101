#!/usr/bin/env bash
# Copy the SO-101 config, URDF and referenced meshes into a GraspGenX clone.
# Meshes come from packages/soarm, which retains their provenance and license.
#
#   bash tools/graspgenx/install_gripper.sh ~/graspgenx
#
# Install under assets/x_grippers, between the dataset and proc_grippers search paths.
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
