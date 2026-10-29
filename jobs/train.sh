#!/usr/bin/env bash
# Train policies with the JAX trainer on a GPU box: the gpu-ops payload of this repository.
#
# The dispatcher (gpu-ops, out of tree) syncs this checkout to the machine, creates the venv from
# uv.lock, activates it, and runs this file from the repository root. Trained policies land in
# workspace/policies/<NAME>/ (policy.npz, meta.json with the GPU and CPU evaluations), which the
# dispatcher pulls back. Without gpu-ops: `bash jobs/train.sh` in a checkout with a CUDA GPU.
#
#   TASKS="reach lift" bash jobs/train.sh      # the bundled policies' training sizes, one after the other
#   TASKS=lift ITERS=50 bash jobs/train.sh      # a short run
#   TASKS=lift KINDS=cube,cuboid5,cuboid85,cylinder INIT=lift-v3 ITERS=300 bash jobs/train.sh
#                                               # one policy for four wooden blocks, widened from lift-v3
#
: "${TASKS:=reach}"               # space-separated: reach, lift, pickplace
: "${ITERS:=}"                    # iterations per task; empty: reach 150, lift 300, pickplace 300 (the bundled policies' sizes)
: "${ENVS:=4096}"                 # worlds in parallel
: "${EVAL_EPISODES:=50}"          # CPU episodes per evaluation after training
: "${NAME_PREFIX:=poc}"           # policies are saved as workspace/policies/<NAME_PREFIX>-<task>
: "${INIT:=}"                     # fine-tune from this policy (name or policy.npz); empty: from scratch
: "${KINDS:=}"                    # wooden blocks, comma-separated (see rl/objects.py); empty: the one cube
: "${EPISODE_STEPS:=}"            # episode length in policy ticks; empty: the task's (lift 200)
: "${GPUS:=1}"
: "${ON_FAILURE:=stops at the first task that fails; the finished tasks keep their policies}"
# JAX would reserve 75 % of the GPU's memory at start; MuJoCo Warp's worlds live in the rest.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# No rendering here, so no OpenGL context.
export MUJOCO_GL="${MUJOCO_GL:-disable}"
set -euo pipefail

cd "$(dirname "$0")/.."
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi missing')"
# The dispatcher syncs the default dependency set; training needs the `train` extra on top.
uv sync --extra train
uv run --extra train python -c "import jax; print('jax', jax.__version__, [d.platform for d in jax.devices()])"

for task in $TASKS; do
    iters="$ITERS"
    if [ -z "$iters" ]; then
        case "$task" in
            reach) iters=150 ;;
            *) iters=300 ;;
        esac
    fi
    echo "== $task: $ENVS worlds x $iters iterations"
    extra=()
    [ -n "$INIT" ] && extra+=(--init "$INIT")
    [ -n "$KINDS" ] && extra+=(--kinds "$KINDS")
    [ -n "$EPISODE_STEPS" ] && extra+=(--episode-steps "$EPISODE_STEPS")
    uv run --extra train soarm-lab train --task "$task" --envs "$ENVS" --iters "$iters" \
        --eval-episodes "$EVAL_EPISODES" --name "$NAME_PREFIX-$task" ${extra[@]+"${extra[@]}"}
done

echo "== policies (the bundled ones are the baseline)"
uv run --extra train soarm-lab policies
