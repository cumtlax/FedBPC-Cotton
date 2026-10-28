#!/usr/bin/env bash
# FedCAP-Cotton fixed paper protocol launcher.
# Usage: bash scripts/run_final.sh METHOD [FIRST_SEED=1] [REPEATS=1] [FEATURE_SET=no-expression]
# Examples:
#   bash scripts/run_final.sh LocalOnly 1 1
#   bash scripts/run_final.sh FedAvg 1 3
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/tools/run_final.py" "$@"
