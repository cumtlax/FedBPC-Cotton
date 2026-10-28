#!/usr/bin/env bash
# Run one FedCAP-Cotton configuration on seeds 1/2/3 concurrently.
# Usage:
#   bash scripts/run_parallel_seeds.sh METHOD OUTPUT_ROOT [extra run_final args...]
# Env: PYTHON, DEVICE=cuda, FEATURE_SET=no-expression, TEST_EVAL=false, SEEDS="1 2 3"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
FEATURE_SET="${FEATURE_SET:-no-expression}"
TEST_EVAL="${TEST_EVAL:-false}"
SEEDS="${SEEDS:-1 2 3}"
METHOD="${1:?method required}"; shift
OUT="${1:?output root required}"; shift
mkdir -p "$OUT"
pids=()
for seed in $SEEDS; do
  echo "[$(date '+%H:%M:%S')] start $METHOD seed=$seed"
  "$PY" tools/run_final.py "$METHOD" "$seed" 1 "$FEATURE_SET" \
    --device "$DEVICE" --test-eval "$TEST_EVAL" \
    --output-root "$OUT" "$@" \
    > "$OUT/seed${seed}.console.log" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo "[$(date '+%H:%M:%S')] all seeds finished rc=$rc; results under $OUT"
exit "$rc"
