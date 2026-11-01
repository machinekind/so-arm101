#!/usr/bin/env bash
# Run named training experiments from the repository root.
# EXPERIMENTS="reach lift" OVERRIDES="ppo.iterations=50" bash jobs/train.sh
: "${EXPERIMENTS:=reach}"
: "${OVERRIDES:=}"
: "${GPUS:=1}"
: "${ON_FAILURE:=stops at the first experiment that fails; finished runs keep their policies}"
# Leave GPU memory available for MuJoCo Warp.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Training needs no OpenGL context.
export MUJOCO_GL="${MUJOCO_GL:-disable}"
set -euo pipefail

cd "$(dirname "$0")/.."
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi missing')"
uv sync --extra train
uv run --extra train python -c "import jax; print('jax', jax.__version__, [d.platform for d in jax.devices()])"

read -r -a experiments <<< "$EXPERIMENTS"
read -r -a overrides <<< "$OVERRIDES"
for experiment in "${experiments[@]}"; do
    echo "== $experiment"
    uv run --extra train soarm-lab train "experiment=$experiment" ${overrides[@]+"${overrides[@]}"}
done

echo "== policies (the bundled ones are the baseline)"
uv run --extra train soarm-lab policies
