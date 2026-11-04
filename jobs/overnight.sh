#!/usr/bin/env bash
# The seeded simulation campaign: the self-correcting grasp, pick_place and build in the twin's simulation,
# one JSON file per run, then report.md over them. Simulation only: no arm, no model, no key. The grasp and
# wrist camera scenarios render, so it needs OpenGL: a desktop, or MUJOCO_GL=egl on a headless box.
#
#   bash jobs/overnight.sh
#   RUNS=50 OUT=workspace/campaign/tonight bash jobs/overnight.sh
#   CAMPAIGN_GROUPS="grasp build" RUNS=5 bash jobs/overnight.sh
: "${RUNS:=30}"                                          # seeds per scenario (the grasp nudge runs half)
: "${OUT:=workspace/campaign/$(date +%Y-%m-%d-%H%M)}"    # the runs' JSON files, the logs and report.md
: "${CAMPAIGN_GROUPS:=grasp pick_place build}"           # scenario groups, in this order
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# No live judge or model: the scenarios use fakes, and a key in the environment is not passed on.
unset TYPESAFE_API_KEY
export SOARM_CAMPAIGN_DIR="$OUT" SOARM_CAMPAIGN_RUNS="$RUNS"

tests=packages/soarm_lab/tests/strategies
status=0
for group in $CAMPAIGN_GROUPS; do
    case "$group" in
        grasp) file="$tests/test_self_correct_in_twin.py" ;;
        pick_place) file="$tests/test_pick_place_in_twin.py" ;;
        build) file="$tests/test_build_in_twin.py" ;;
        *) echo "unknown group: $group (grasp, pick_place, build)" >&2; exit 2 ;;
    esac
    echo "== $group: $RUNS seeds per scenario -> $OUT"
    start=$(date +%s)
    # A scenario below its bar fails its test; the others still run and every run is kept.
    if ! uv run --group twin pytest -q -p no:cacheprovider -m campaign -s "$file" 2>&1 | tee "$OUT/$group.log"; then
        status=1
    fi
    echo "== $group: $(( ($(date +%s) - start) / 60 )) min"
done

uv run soarm-lab campaign report "$OUT" --out "$OUT/report.md" \
    --title "Simulation campaign, $RUNS seeds per scenario, $(git rev-parse --short HEAD)"
exit "$status"
