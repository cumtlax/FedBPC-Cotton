# -*- coding: utf-8 -*-
"""Static-FL conformance checks for the main FedCAP-Cotton configuration."""
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN = {
    "client_num": 90, "sample_ratio": 0.2, "rounds": 100,
    "drift_pattern": False, "num_classes": 2,
    "cotton_feature_set": "no-expression",
    "probe_training": "legacy", "probe_budget": 100, "probe_steps": 5, "probe_lr": 0.1,
    "reference_strategy": "average_reset", "reference_refresh": 10,
    "profile_head": "reference", "profile_weight": 0.5,
    "profile_samples_per_class": 64, "head_weights": "uniform",
    "head_sharing": "none",
    "min_class_samples": 2, "cluster_method": "agglomerative",
    "min_cluster_size": 3, "max_candidate_clusters": 5, "max_total_clusters": 5,
    "silhouette_threshold": 0.25, "regroup_confirmations": 1,
    "body_aggregation_step": 1.0,
    "class_support_alpha": 0.5, "class_support_count_cap": 0,
    "rep_epochs": 5, "clf_epochs": 1, "rep_lr": 0.01, "clf_lr": 0.1,
}
DISABLED = {
    "merge_guard": "none", "profile_merge_guard": "none", "label_merge_guard": "none",
    "body_prox_mu": 0.0, "head_calibration_ratio": 0.0,
}
REMOVED_KEYS = (
    "class_share_rule", "class_share_eps", "class_share_min_separation",
    "class_share_rho", "class_share_min_witnesses", "class_share_witness_policy",
    "class_share_label_mode", "class_share_task_gate", "class_share_centering",
)

def load(name):
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_main_config_matches_maiN():
    cfg = load("FedCAP_CottonMain.yaml")
    for key, value in MAIN.items():
        assert cfg.get(key) == value, (key, cfg.get(key), value)
    for key, value in DISABLED.items():
        assert cfg.get(key) == value, (key, cfg.get(key), value)
    assert cfg.get("deterministic") is True
    assert cfg.get("cotton_root") == "data/cotton"

def test_default_config_matches_main():
    for name in ("FedCAP_cotton.yaml", "FedCAP.yaml", "FedCAP_CottonNPZ.yaml"):
        cfg = load(name)
        assert cfg.get("head_sharing") == "none", name
        assert cfg.get("cotton_feature_set") == "no-expression", name
        for key in MAIN:
            if key in ("client_num","sample_ratio","rounds"):
                continue
            assert cfg.get(key) == MAIN[key], (name, key)

def test_removed_sharing_keys_are_absent():
    for name in ("FedCAP_CottonMain.yaml","FedCAP_CottonNPZ.yaml","FedCAP_cotton.yaml","FedCAP.yaml"):
        cfg = load(name)
        for key in REMOVED_KEYS:
            assert key not in cfg, (name, key)

def test_cotton_data_layout():
    root = ROOT / "data" / "cotton"
    for name in ("c1_cotton_nitrogen", "c2_cotton_mutant", "c3_cotton_salt"):
        assert (root / "clients" / name / "train_ready.tsv").is_file()
    assert (root / "metadata" / "gene_families.tsv").is_file()

def test_idea_switch_consistency():
    main = load("FedCAP_CottonMain.yaml")
    assert main["head_sharing"] == "none"
    assert main["drift_pattern"] is False
    assert main["cotton_cross_source_eval"] is False

def test_default_entry_config_exists():
    cfg = load("FedCAP.yaml")
    assert cfg.get("dataset") == "cotton30"
    assert cfg.get("cotton_feature_set") == "no-expression"

def test_all_kept_configs_agree():
    main = load("FedCAP_CottonMain.yaml")
    for name in ("FedCAP_cotton.yaml", "FedCAP.yaml", "FedCAP_CottonNPZ.yaml"):
        cfg = load(name)
        for key in MAIN:
            assert cfg.get(key) == main.get(key), (name, key)
