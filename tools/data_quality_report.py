# -*- coding: utf-8 -*-
"""Data-quality report for the FedCAP-Cotton training data (data/cotton)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.cotton_dataset import (active_feature_names, distribute_cotton)  # noqa: E402

DATA = ROOT / "data" / "cotton"
CLIENTS = {
    "c1_nitrogen": DATA / "clients" / "c1_cotton_nitrogen" / "train_ready.tsv",
    "c2_mutant": DATA / "clients" / "c2_cotton_mutant" / "train_ready.tsv",
    "c3_salt": DATA / "clients" / "c3_cotton_salt" / "train_ready.tsv",
}
EXPR = ("expr_mean_log2cpm", "expr_sd_log2cpm", "expr_detected_frac")
report = {"generated": "scripts/data_quality_report.py", "data_root": str(DATA)}

frames = {k: pd.read_csv(p, sep="\t", index_col="gene_id") for k, p in CLIENTS.items()}
feature_cols = [c for c in next(iter(frames.values())).columns
                if c not in ("label", "log2FoldChange", "padj")]
assert len(feature_cols) == 23
report["schema"] = {
    "rows": {k: int(len(f)) for k, f in frames.items()},
    "unique_genes": {k: bool(f.index.is_unique) for k, f in frames.items()},
    "feature_columns": feature_cols,
    "feature_order_identical": all(list(f[feature_cols].columns) == feature_cols for f in frames.values()),
}

# finiteness / ranges / constants
stats = {}
for k, f in frames.items():
    X = f[feature_cols].to_numpy(dtype=float)
    stats[k] = {
        "nan": int(np.isnan(X).sum()),
        "inf": int(np.isinf(X).sum()),
        "constant_columns": [c for c in feature_cols if f[c].nunique(dropna=False) <= 1],
        "min": {c: float(f[c].min()) for c in feature_cols},
        "max": {c: float(f[c].max()) for c in feature_cols},
    }
report["feature_quality"] = stats

# labels / threshold consistency
lab = {}
for k, f in frames.items():
    rule = ((f["padj"] < 0.05) & (f["log2FoldChange"].abs() > 1.0)).astype(int)
    lab[k] = {
        "positives": int(f["label"].sum()),
        "prevalence": round(float(f["label"].mean()), 4),
        "binary": bool(f["label"].isin([0, 1]).all()),
        "matches_padj05_fc1_rule": bool((rule == f["label"].astype(int)).all()),
        "label_disagreements": int((rule != f["label"].astype(int)).sum()),
    }
report["labels"] = lab

# gene-set identity and shared-feature identity across sources
keys = list(frames)
same_genes = all(set(frames[keys[0]].index) == set(frames[k].index) for k in keys[1:])
report["cross_source"] = {"same_gene_set": bool(same_genes)}
shared = [c for c in feature_cols if c not in EXPR]
mismatch = {}
for k in keys[1:]:
    a = frames[keys[0]].loc[frames[keys[0]].index.intersection(frames[k].index), shared]
    b = frames[k].loc[a.index, shared]
    mismatch[k] = int((~np.isclose(a.to_numpy(float), b.to_numpy(float), atol=1e-9)).sum())
report["cross_source"]["shared_feature_mismatches_vs_c1"] = mismatch
expr_diff = {}
for k in keys[1:]:
    a = frames[keys[0]].loc[frames[keys[0]].index.intersection(frames[k].index), list(EXPR)]
    b = frames[k].loc[a.index, list(EXPR)]
    expr_diff[k] = int((~np.isclose(a.to_numpy(float), b.to_numpy(float), atol=1e-9)).any(axis=1).sum())
report["cross_source"]["genes_with_source_specific_expr"] = expr_diff

# family map
fam = pd.read_csv(DATA / "metadata" / "gene_families.tsv", sep="\t")
gene_union = set().union(*[set(f.index) for f in frames.values()])
fam_in = fam[fam.gene_id.isin(gene_union)]
missing = gene_union - set(fam.gene_id)
fallback = int((fam_in.family_id == fam_in.gene_id).sum())
sizes = fam_in.groupby("family_id").size()
report["families"] = {
    "rows_in_map": int(len(fam)),
    "genes_in_dataset": int(len(gene_union)),
    "genes_without_family": int(len(missing)),
    "families_for_dataset": int(sizes.size),
    "singleton_families": int((sizes == 1).sum()),
    "singleton_fraction": round(float((sizes == 1).mean()), 6),
    "self_family_fallback_rows": fallback,
    "max_family_size": int(sizes.max()),
    "family_split_effective_gene_level": bool(sizes.max() == 1),
    "note": ("family IDs are singletons among dataset genes because the 171 two-member "
             "families pair GhChr with non-GhChr annotations that are not in this dataset"),
}

# loader-level clients and split
train, test, global_tests = distribute_cotton(
    client_num=90, seed=1, root=DATA, feature_set="no-expression", include_test=True)
train_v, test_v, gv, validation, gval = distribute_cotton(
    client_num=90, seed=1, root=DATA, feature_set="no-expression",
    include_test=True, return_validation=True, return_source_validation=True)
sizes = [len(train[i]) for i in range(90)]
pos = [int(np.asarray(train[i].targets).sum()) for i in range(90)]
min_class = min(int(np.bincount(np.asarray(train[i].targets), minlength=2).min()) for i in range(90))
excluded_zero = bool(all(np.all(np.asarray(train[i].data)[:, [j for j, name in enumerate(feature_cols)
                                          if name in EXPR]] == 0) for i in range(90)))
# family-disjoint checks
fam_disjoint = True
for src in range(3):
    tr_f = set()
    for j in range(30):
        tr_f |= set(train[src * 30 + j].family_ids)
    te_f = set(test[src * 30].family_ids)
    va_f = set(validation[src * 30].family_ids)
    if (tr_f & te_f) or (tr_f & va_f) or (te_f & va_f):
        fam_disjoint = False
report["loader_clients"] = {
    "n_train_clients": len(train), "n_test_clients": len(test),
    "train_size_min": int(min(sizes)), "train_size_max": int(max(sizes)),
    "positives_min": int(min(pos)), "positives_max": int(max(pos)),
    "min_class_samples_any_client": min_class,
    "expr_columns_zeroed": excluded_zero,
    "active_features": len(active_feature_names(feature_cols, "no-expression")),
    "network_width": int(np.asarray(train[0].data).shape[1]),
    "train_samples_per_source": [int(sum(len(train[src * 30 + j]) for j in range(30))) for src in range(3)],
    "validation_sizes": [int(len(validation[src * 30])) for src in range(3)],
    "test_sizes": [int(len(test[src * 30])) for src in range(3)],
    "global_test_sizes": [int(len(g)) for g in global_tests],
    "family_disjoint_train_val_test": bool(fam_disjoint),
    "min_family_count_across_splits": None,
}

# single-feature leakage check (train genes of c1, RAW TSV values)
from sklearn.metrics import roc_auc_score
f = frames["c1_nitrogen"]
y = f["label"].to_numpy()
auc = {}
for c in feature_cols:
    x = f[c].to_numpy(float)
    if np.unique(x).size > 1:
        auc[c] = round(float(roc_auc_score(y, x)), 4)
report["single_feature_auc_c1"] = dict(sorted(auc.items(), key=lambda kv: -kv[1]))
report["single_feature_auc_note"] = ("expr_* features are same-assay leakage; "
                                     "non-expression features are the honest signal")

out = ROOT / "data" / "quality_report_20261018.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("schema", "labels", "cross_source", "families", "loader_clients")},
                 ensure_ascii=False, indent=1))
print("top leakage AUC:", list(report["single_feature_auc_c1"].items())[:5])
print("bottom signal AUC:", list(report["single_feature_auc_c1"].items())[-5:])
print("report ->", out)
