# FedBPC-Cotton

Code and processed cotton data for **Behavioral clustering for federated prediction of differentially expressed genes across cotton experimental contrasts**.

FedBPC is the method proposed in the paper. The implementation retains the historical `FedCAP` class names, configuration names and command-line identifier for compatibility. Use `FedCAP` when running the commands below.

## Contents

- `entities/`, `utils/`, `methods/`: FedBPC, baseline implementations and their shared dependencies.
- `configs/`: algorithm settings; `FedCAP_CottonMain_NoSupport.yaml` is the paper's uniform-head-aggregation configuration (`class_support_alpha: 0`). The older `FedCAP_CottonMain.yaml` enables support weighting and is not the paper's final configuration.
- `data/cotton/`: processed feature tables, contrast-specific DEG labels and gene-family identifiers.
- `scripts/`, `tools/`: training launchers, data checks, data preparation and result collection scripts.
- `tests/`: source-code and data checks.

Training logs, experiment outputs, model checkpoints, manuscript files, figures and raw sequencing files are excluded. No experiment results or per-model prediction files are distributed in this repository.

## Installation

Use Python 3.10 or newer and install a compatible PyTorch / torchvision pair for your CPU or CUDA environment. Then:

```bash
python -m pip install -r requirements_autodl.txt
# Optional, for the source tests:
python -m pip install pytest
python tools/check_data.py --clients 90 --feature-set no-expression
```

## Final-test runs

Run from the repository root. These commands use seeds 1, 2 and 3, 90 virtual clients, 100 communication rounds and 20% participation. All methods use the `no-expression` feature setting. The launcher prints the complete resolved command; `--dry-run` prints it without training. Omit `--test-eval true` for validation-only runs.

```bash
# Proposed FedBPC: historical CLI identifier FedCAP, final uniform-head setting
python tools/run_final.py FedCAP 1 3 no-expression --fedcap-config configs/FedCAP_CottonMain_NoSupport.yaml --test-eval true

# Main-paper comparison methods, fixed settings (no parameter search)
python tools/run_final.py FedFM 1 3 no-expression --test-eval true
python tools/run_final.py FedCCFA 1 3 no-expression --test-eval true
python tools/run_final.py FedProx 1 3 no-expression --test-eval true
python tools/run_final.py FedAvg 1 3 no-expression --test-eval true
python tools/run_final.py IFCA 1 3 no-expression --test-eval true
python tools/run_final.py FedRep 1 3 no-expression --test-eval true
python tools/run_final.py FedAvgWeighted 1 3 no-expression --test-eval true
python tools/run_final.py Centralized 1 3 no-expression --test-eval true
python tools/run_final.py LocalOnly 1 3 no-expression --test-eval true
```

Add `--device cpu` if needed. Runs create local output files; `.gitignore` excludes them from version control. Other experimental algorithms and older scripts/configurations are retained as code, not as additional main-paper comparisons. `scripts/run_baselines_autodl.sh` defaults to validation and a broader algorithm collection; use the explicit commands above for the main-paper comparison.

## Reproducibility notes

The runtime loader derives the shared gene-level 70/15/15 train/validation/test split from the seed, fits standardization using training genes only, and creates 30 virtual clients per contrast. Test genes do not occur in training or validation. The family map is effectively one family per retained dataset gene; this is gene separation, not a guarantee of homologous-family separation.

The primary experiment uses 20 active regulatory, sequence and structural features. To preserve the supplied implementation and historical initialization, the TSV schema and neural-network interface retain 23 feature columns; three expression-derived columns are masked to zero in `no-expression` mode. `all23` is not the paper's primary feature setting. See `data/README.md` for the exact columns.

This release contains the local code and configurations, not a claim that every present-day default reconstructs every historical run. In particular, the original FedCCFA result summaries did not retain all algorithm-specific runtime values. The paper reports this limitation; no missing values have been inferred from test results.

`tools/prepare_cotton_data.py` requires the original upstream processed-data project and is not needed to use the included TSV tables. Raw RNA-seq, ATAC-seq and ChIP-seq reads/BAMs are not included. No raw-sequencing accession or software/data licence is inferred by this release.
