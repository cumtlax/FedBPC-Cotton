#!/usr/bin/env bash
# FedCAP-Cotton: 对比算法 × 3 seeds（自包含；只需 FedCAPCotton 工程）
# 用法：bash scripts/run_baselines_autodl.sh
# 环境变量：PYTHON=python  OUT=runs_baselines  SEEDS="1 2 3"  DEVICE=cuda
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUT="${OUT:-runs_baselines}"
SEEDS="${SEEDS:-1 2 3}"
DEVICE="${DEVICE:-cuda}"
ALGOS="FedAvg FedProx SCAFFOLD FedFM AdaptiveFedAvg Flash pFedMe Ditto FedRep FedBABU FedPAC IFCA LocalOnly Centralized"

# Check the optional native solver before spending time on the preceding methods.
echo "=== FedPAC solver preflight ==="
"$PY" -c "from utils.fedpac_dependencies import check_fedpac_runtime; print(check_fedpac_runtime())"

for seed in $SEEDS; do
  mkdir -p "$OUT/seed$seed"
  for algo in $ALGOS; do
    echo "=== baseline seed=$seed algo=$algo ==="
    console_log="$(mktemp "$OUT/seed$seed/$algo.XXXXXX.log")"
    "$PY" methods/CottonBaselines.py \
      --algorithm "$algo" \
      --config configs/CottonPaper_Baselines.yaml \
      --cotton-feature-set no-expression \
      --seed "$seed" --rounds 100 --device "$DEVICE" \
      --output-root "$OUT/seed$seed" "$@" 2>&1 | tee "$console_log"
  done
  echo "=== FedCCFA seed=$seed ==="
  console_log="$(mktemp "$OUT/seed$seed/FedCCFA.XXXXXX.log")"
  "$PY" methods/FedCAP.py \
    --algorithm FedCCFA --dataset cotton30 \
    --config configs/FedCCFA_CottonPaper.yaml \
    --cotton-feature-set no-expression \
    --cotton-test-eval false --cotton-cross-source-eval false \
    --seed "$seed" --rounds 100 --device "$DEVICE" \
    --output-root "$OUT/seed$seed" "$@" 2>&1 | tee "$console_log"
  echo "=== FedDrift seed=$seed ==="
  console_log="$(mktemp "$OUT/seed$seed/FedDrift.XXXXXX.log")"
  "$PY" methods/FedCAP.py \
    --algorithm FedDrift --dataset cotton30 \
    --config configs/FedDrift_Cotton90.yaml \
    --cotton-feature-set no-expression \
    --cotton-test-eval false --cotton-cross-source-eval false \
    --seed "$seed" --rounds 100 --device "$DEVICE" \
    --output-root "$OUT/seed$seed" "$@" 2>&1 | tee "$console_log"
done

echo "== done. results under $OUT/seed{1,2,3}/log/ =="
