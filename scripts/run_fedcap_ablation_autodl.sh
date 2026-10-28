#!/usr/bin/env bash
# FedCAP-Cotton main-version ablation queue (prepared; run only on explicit request).
#
# Stages:
#   S1: main version, cross-cluster row sharing OFF, class-support alpha=0.5
#   S2: same as S1 but alpha=0
#   A0: single cluster (no learned grouping)
#   B1: probe-only clustering (profile_weight=0), seed 1 only
#   B2: probe budget sensitivity 64/128, seed 1 only
#
# Usage:
#   STAGES="S1 S2 A0 B1 B2" MAXJOBS=2 bash scripts/run_fedcap_ablation_autodl.sh
#   STAGES="S1 S2" bash scripts/run_fedcap_ablation_autodl.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
MAXJOBS="${MAXJOBS:-2}"
STAGES="${STAGES:-S1 S2 A0 B1 B2}"
OUT="${OUT:-CottonExperiments/ablation}"
TEST_EVAL="${TEST_EVAL:-false}"
mkdir -p "$OUT"

LAUNCHER="scripts/run_final.sh"
MAIN="configs/FedCAP_CottonMain.yaml"
NOSUPPORT="configs/FedCAP_CottonMain_NoSupport.yaml"

run_one() {
  local name="$1"; shift
  local out="$OUT/$name"
  mkdir -p "$out"
  echo "[$(date '+%H:%M:%S')] START $name"
  "$PY" tools/run_final.py "$@" \
    --test-eval "$TEST_EVAL" --device "$DEVICE" --output-root "$out" \
    > "$OUT/$name.console.log" 2>&1
  echo "[$(date '+%H:%M:%S')] END $name rc=$?"
}

wait_slot() {
  while (( $(jobs -rp | wc -l) >= MAXJOBS )); do
    wait -n
  done
}

want() { [[ " $STAGES " == *" $1 "* ]]; }

if want S1; then
  wait_slot; run_one S1 FedCAP 1 3 --fedcap-config "$MAIN" &
fi
if want S2; then
  wait_slot; run_one S2 FedCAP 1 3 --fedcap-config "$NOSUPPORT" &
fi
if want A0; then
  wait_slot; run_one A0 FedCAP 1 3 --fedcap-config "$MAIN" \
    --fedcap-arg '--max-clusters 1' &
fi
if want B1; then
  wait_slot; run_one B1 FedCAP 1 1 --fedcap-config "$MAIN" \
    --fedcap-arg '--profile-weight 0' &
fi
if want B2; then
  wait_slot; run_one B2-probe64 FedCAP 1 1 --fedcap-config "$MAIN" \
    --fedcap-arg '--probe-budget 64' &
  wait_slot; run_one B2-probe128 FedCAP 1 1 --fedcap-config "$MAIN" \
    --fedcap-arg '--probe-budget 128' &
fi

wait
echo "=== ablation queue done $(date '+%F %T'); results under $OUT ==="
"$PY" tools/collect_results.py --root "$OUT" || true
