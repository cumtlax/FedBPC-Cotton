# -*- coding: utf-8 -*-
"""Main-version (no cross-cluster row sharing) config and launcher checks."""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN = "FedCAP_CottonMain.yaml"
NOSUPPORT = "FedCAP_CottonMain_NoSupport.yaml"
BASE = "FedCAP_CottonNPZ.yaml"

CORE_KEYS = (
    "algorithm", "dataset", "client_num", "sample_ratio", "rounds", "num_classes",
    "cotton_feature_set", "probe_training", "probe_budget", "probe_steps", "probe_lr",
    "reference_strategy", "reference_refresh", "profile_head", "profile_weight",
    "profile_samples_per_class", "head_weights", "cluster_method", "min_cluster_size",
    "max_candidate_clusters", "max_total_clusters", "silhouette_threshold",
    "regroup_confirmations", "class_support_count_cap", "rep_epochs", "clf_epochs",
    "rep_lr", "clf_lr", "batch_size", "deterministic",
)


def load(name):
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_main_config_disables_row_sharing_but_keeps_core():
    cfg, base = load(MAIN), load(BASE)
    assert cfg["head_sharing"] == "none"
    assert cfg["class_support_alpha"] == 0.5
    for key in CORE_KEYS:
        assert cfg.get(key) == base.get(key), (key, cfg.get(key), base.get(key))
    assert cfg["cotton_test_eval"] is False
    assert cfg["cotton_root"] == "data/cotton"


def test_nosupport_config_only_changes_alpha():
    main, nosupport = load(MAIN), load(NOSUPPORT)
    assert nosupport["head_sharing"] == "none"
    assert nosupport["class_support_alpha"] == 0.0
    for key in main:
        if key == "class_support_alpha":
            continue
        assert main[key] == nosupport.get(key), key


def test_launcher_defaults_to_main_config_and_accepts_ablation_flags():
    common = [sys.executable, str(ROOT / "tools" / "run_final.py"),
              "FedCAP", "1", "1", "--dry-run", "--rounds", "1", "--clients", "9",
              "--device", "cpu"]
    result = subprocess.run(common, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "configs/FedCAP_CottonMain.yaml" in result.stdout

    extra = common + ["--fedcap-config", "configs/FedCAP_CottonMain_NoSupport.yaml",
                      "--fedcap-arg", "--max-clusters 1"]
    result = subprocess.run(extra, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "FedCAP_CottonMain_NoSupport.yaml" in result.stdout
    assert "--max-clusters 1" in result.stdout


def test_fedcap_only_flags_rejected_for_baselines():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_final.py"),
         "FedAvg", "1", "1", "--dry-run", "--fedcap-config", "configs/FedCAP_CottonMain.yaml"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "only supported by FedCAP" in result.stderr
