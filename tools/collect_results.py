# -*- coding: utf-8 -*-
"""Collect baseline/FedCAP run results into a flat table.

Scans an output root for result-*/summary-* JSON files, extracts the main
validation metrics, and writes:
  <root>/results_summary.csv
  <root>/flat/<method>_seed<N>.json     (copy of the run JSON)

Usage:
  python scripts/collect_results.py --root runs_baselines
"""
import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

METRICS = [
    "validation_macro_source_average_precision",
    "validation_worst_source_average_precision",
    "validation_worst_client_average_precision",
    "validation_local_accuracy",
    "validation_local_balanced_accuracy",
    "validation_local_f1",
    "validation_local_auroc",
    "validation_local_average_precision",
    "validation_global_accuracy",
    "validation_global_balanced_accuracy",
    "validation_global_f1",
    "validation_global_auroc",
    "validation_global_average_precision",
    "validation_generalized_accuracy",
    "validation_generalized_balanced_accuracy",
    "validation_generalized_f1",
    "validation_generalized_auroc",
    "validation_generalized_average_precision",
    "test_macro_source_average_precision",
    "test_worst_source_average_precision",
    "test_worst_client_average_precision",
    "test_local_accuracy",
    "test_local_balanced_accuracy",
    "test_local_f1",
    "test_local_auroc",
    "test_local_average_precision",
    "test_global_accuracy",
    "test_global_balanced_accuracy",
    "test_global_f1",
    "test_global_auroc",
    "test_global_average_precision",
    "test_generalized_accuracy",
    "test_generalized_balanced_accuracy",
    "test_generalized_f1",
    "test_generalized_auroc",
    "test_generalized_average_precision",
]


def parse_path(p):
    parts = Path(p).parts
    seed = None
    method = None
    clients = None
    for part in parts:
        m = re.fullmatch(r"seed-(\d+)", part)
        if m:
            seed = int(m.group(1))
        m = re.fullmatch(r"clients-(\d+)", part)
        if m:
            clients = int(m.group(1))
    if "cotton30" in parts:
        i = parts.index("cotton30")
        if i + 1 < len(parts):
            candidate = parts[i + 1]
            # FedCAP variants put the method name before cotton30
            method = parts[i - 1] if candidate.startswith("clients-") else candidate
    return method, seed, clients


def clean_method(name):
    if not name:
        return name
    for suffix in ("-ValidationOnly", "-NoExpression", "-All23"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def flatten_metrics(data):
    """Expose the nested validation_metrics block as flat validation_/test_ keys.

    Cotton test-eval summaries store metrics under ``validation_metrics`` even
    when the evaluated split is the locked test family set; the prefix follows
    ``cotton_test_evaluated``.
    """
    merged = dict(data)
    nested = data.get("validation_metrics")
    if isinstance(nested, dict):
        prefix = "test_" if data.get("cotton_test_evaluated") else "validation_"
        for key, value in nested.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            flat = key if key.startswith(("validation_", "test_")) else prefix + key
            merged.setdefault(flat, value)
    return merged


def first_present(row, keys):
    for key in keys:
        if key in row:
            return row[key]
    return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_baselines")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    flat = root / "flat"
    flat.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    candidates = sorted(list(root.rglob("result-*.json")) + list(root.rglob("summary-*.json")))
    for p in candidates:
        # prefer result-* over summary-* for the same run folder
        run_dir = p.parent
        if run_dir in seen:
            continue
        sibling_result = list(run_dir.glob("result-*.json"))
        chosen = sibling_result[0] if sibling_result else p
        if chosen != p:
            continue
        try:
            data = json.load(open(chosen, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = flatten_metrics(data)
        method, seed, clients = parse_path(chosen)
        method = clean_method(method)
        row = {"method": method or (data.get("algorithm") or "?"),
               "seed": seed, "clients": clients,
               "file": str(chosen.relative_to(root)),
               "feature_set": data.get("cotton_feature_set", ""),
               "rounds": data.get("rounds", "")}
        for m in METRICS:
            if m in metrics and isinstance(metrics[m], (int, float)):
                row[m] = round(float(metrics[m]), 6)
        rows.append(row)
        seen.add(run_dir)
        flat_name = "%s_seed%s.json" % (row["method"], row["seed"])
        if (flat / flat_name).exists():
            flat_name = "%s_seed%s_%s.json" % (row["method"], row["seed"], chosen.stem)
        shutil.copyfile(chosen, flat / flat_name)

    rows.sort(key=lambda r: (str(r["method"]), r["seed"] or 0))
    out = Path(a.out) if a.out else root / "results_summary.csv"
    if rows:
        fields = ["method", "seed", "clients", "feature_set", "rounds", "file"] + \
                 [m for m in METRICS if any(m in r for r in rows)]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print("runs collected:", len(rows))
    print("summary ->", out)
    print("flat copies ->", flat)
    for r in rows[:10]:
        ap = first_present(r, ("validation_macro_source_average_precision",
                               "test_macro_source_average_precision"))
        bal = first_present(r, ("validation_local_balanced_accuracy",
                                "test_local_balanced_accuracy"))
        split = "test" if "test_macro_source_average_precision" in r else "validation"
        print("  %-24s seed=%s [%s] AP=%s balACC=%s" % (
            r["method"], r["seed"], split, ap, bal))


if __name__ == "__main__":
    main()
