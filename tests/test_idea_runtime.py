"""Behavioral checks for Idea-v3 controls; training fixtures are synthetic only."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entities.FedCAP import FedCAPClient, FedCAPServer
from entities.FedCAPCore import descriptor_distances
from entities.FedCAPClassSupport import class_support_weights
from entities.CottonWeightedFedAvg import train_class_weights
from utils import fedcap_experiment as experiment
from utils import cotton_baselines as baselines
from utils.cotton_collaboration import head_difference_report
from utils.cotton_dataset import EXPRESSION_FEATURES, FEATURE_GROUPS, distribute_cotton
from utils.fedcap_output import variant_name
from utils.fedcap_reproducibility import training_fingerprint

FEATURES = tuple(n for group in FEATURE_GROUPS.values() for n in group) + EXPRESSION_FEATURES


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, source, identity, counts=(6, 4), split="train"):
        self.concept_id = source
        self.targets = np.repeat(np.arange(2), counts).astype(np.int64)
        self.data = torch.randn(len(self.targets), 23,
                                generator=torch.Generator().manual_seed(identity + 19))
        self.data[:, -3:] = 0
        self.gene_ids = np.asarray([f"{split}-{identity}-{i}" for i in range(len(self.targets))])
        self.family_ids = self.gene_ids.copy()
        self.input_features = FEATURES[:-3]
        self.excluded_features = EXPRESSION_FEATURES
        self.split = split
        self.transform = None
        self.manifest = dict(features=list(FEATURES), active_features=list(self.input_features),
                             excluded_model_inputs=list(EXPRESSION_FEATURES),
                             data_root="synthetic", split="synthetic_gene_disjoint", seed=1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])


def config(name="FedCAP_CottonNPZ.yaml", *extra):
    parser = experiment.build_parser()
    options = parser.parse_args(["--config", str(ROOT / "configs" / name),
                                "--clients", "6", "--sample-ratio", "1", "--device", "cpu", *extra])
    return experiment.resolve_config(options, parser)


def make_clients(args, server):
    return [FedCAPClient(i, args, TinyDataset(i // 2, i), TinyDataset(i // 2, i, split="validation"),
                         server.initial) for i in range(6)]


def synthetic_loader(client_num, seed, root, **kwargs):
    train = {i: TinyDataset(i // 2, i) for i in range(client_num)}
    validation = {i: TinyDataset(i // 2, i + 100, split="validation") for i in train}
    test = {i: TinyDataset(i // 2, i + 200, split="test") for i in train} if kwargs["include_test"] else {}

    def full(parts):
        result = []
        for source in range(3):
            members = [parts[i] for i in parts if i // 2 == source]
            ds = deepcopy(members[0])
            ds.data = torch.cat([d.data for d in members])
            for attr in ("targets", "gene_ids", "family_ids"):
                setattr(ds, attr, np.concatenate([getattr(d, attr) for d in members]))
            result.append(ds)
        return result
    return train, test, full(test) if test else [], validation, full(validation)


def test_profile_only_falls_back_to_update_distance():
    rows = [dict(version=0, profile=np.asarray([[0.9, 0.1], [0.1, 0.9]]) + i * 0.01,
                 valid=np.asarray(v), delta=np.asarray([float(i * 3)]))
            for i, v in enumerate(([True, False], [False, True], [True, False]))]
    distance = descriptor_distances(rows, 1.0)
    # Probe pair distances are 3,6,3; the nonzero median is 3.
    assert distance[0, 1] == pytest.approx(1.0)
    assert distance[1, 2] == pytest.approx(1.0)


def test_full_class_support_does_not_cap_at_profile_budget():
    np.testing.assert_allclose(class_support_weights([64, 256], 0.5, 0), [5/12, 7/12])
    np.testing.assert_allclose(class_support_weights([64, 256], 0.5, 64), [0.5, 0.5])
    np.testing.assert_allclose(class_support_weights([0, 0], 0.5, 0), [0.5, 0.5])
    np.testing.assert_allclose(class_support_weights([1, 999], 0, 0), [0.5, 0.5])


def test_reference_refresh_uses_current_body_and_all_client_heads():
    args = config()
    server = FedCAPServer(args)
    clients = make_clients(args, server)
    anchor = deepcopy(server.reference.state_dict())
    server.body_state = {k: v + 1 for k, v in server.body_state.items()}
    private = {k: v + 3 for k, v in server.initial.head.state_dict().items()}
    server.retained_heads[5] = private
    server.membership[5] = None
    server.cache[0] = dict(round=0, version=0)
    server.refresh_reference(clients, 9)
    assert server.version == 0 and server.cache
    assert all(torch.equal(v, server.reference.state_dict()[k]) for k, v in anchor.items())
    server.refresh_reference(clients, 10)
    assert server.version == 1 and not server.cache
    assert all(torch.equal(v, server.reference.body.state_dict()[k]) for k, v in server.body_state.items())
    for k, v in server.initial.head.state_dict().items():
        torch.testing.assert_close(server.reference.head.state_dict()[k], (5*v+private[k])/6)


def test_probe_keeps_reference_and_formal_model_unchanged():
    args = config()
    server = FedCAPServer(args)
    client = make_clients(args, server)[0]
    reference = deepcopy(server.reference.state_dict())
    formal = deepcopy(client.model.state_dict())
    descriptor = client.describe(server.reference, 0, 0)
    with torch.no_grad():
        p = server.reference(client.train_set.data).softmax(1).numpy()
    for category in (0, 1):
        expected = p[client.train_set.targets == category].mean(0)
        np.testing.assert_allclose(descriptor["row_profile"][category], expected, atol=1e-6)
    assert descriptor["profile_unique_counts"] == [6, 4]
    assert descriptor["sample_counts"] == [50, 50]
    assert descriptor["balanced_probe_steps"] == 5
    assert np.linalg.norm(descriptor["delta"]) > 0
    assert all(torch.equal(v, server.reference.state_dict()[k]) for k, v in reference.items())
    assert all(torch.equal(v, client.model.state_dict()[k]) for k, v in formal.items())


def test_decision_diagnostic_ignores_common_logit_offset():
    a = dict(weight=torch.tensor([[0., 0.], [1., 1.]]), bias=torch.tensor([0., 1.]))
    b = {k: v + 2 for k, v in a.items()}
    result = head_difference_report({0: a, 1: b})
    assert result["identical_head_pairs"] == 0
    assert result["identical_decision_pairs"] == 1


def test_data_fingerprint_includes_gene_and_group_identities():
    dataset = TinyDataset(0, 0)
    original = training_fingerprint({0: dataset})
    changed = deepcopy(dataset)
    changed.gene_ids[0] = "other-gene"
    assert original["sha256"] != training_fingerprint({0: changed})["sha256"]
    changed = deepcopy(dataset)
    changed.family_ids[0] = "other-group"
    assert original["sha256"] != training_fingerprint({0: changed})["sha256"]
    assert original["identities_complete"] is True


def test_weighted_fedavg_uses_only_training_counts_and_matches_initialization(tmp_path):
    weights = train_class_weights([0] * 8 + [1] * 2)
    torch.testing.assert_close(weights, torch.tensor([0.625, 2.5]))
    p = baselines.parser()
    args = baselines.resolve_config(p.parse_args([
        "--algorithm", "FedAvgWeighted", "--config", str(ROOT / "configs/FedAvgWeighted_Cotton.yaml"),
        "--clients", "6", "--device", "cpu", "--output-root", str(tmp_path)]))
    args["tensorboard_dir"] = str(tmp_path / "events")
    train = {i: TinyDataset(i // 2, i, (8, 2)) for i in range(6)}
    # Deliberately reverse evaluation labels: weights must still be [0.625,2.5].
    test = {i: TinyDataset(i // 2, i, (2, 8), "test") for i in range(6)}
    torch.manual_seed(41)
    clients, server = baselines.build(args, train, test)
    try:
        assert all(torch.equal(c.criterion.weight.cpu(), weights) for c in clients)
        torch.manual_seed(41)
        cap = FedCAPServer(config())
        left = torch.nn.utils.parameters_to_vector(server.model.parameters())
        right = torch.nn.utils.parameters_to_vector(cap.initial.parameters())
        assert torch.equal(left, right)
    finally:
        server.writer.close()


@pytest.mark.parametrize("evaluator", [experiment.evaluate, baselines.evaluate])
def test_evaluation_preserves_rng_and_all_module_modes(evaluator):
    model = torch.nn.Sequential(torch.nn.Linear(23, 2), torch.nn.Dropout())
    model.train(); model[1].eval()
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    evaluator(model, TinyDataset(0, 0), "cpu", True)
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())


def test_validation_defaults_and_launcher_dry_run(tmp_path):
    import subprocess
    import yaml
    for name in ("FedCAP.yaml", "FedCAP_cotton.yaml", "FedCAP_CottonNPZ.yaml"):
        values = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert values["cotton_test_eval"] is False
    result = subprocess.run([sys.executable, str(ROOT / "tools/run_final.py"), "FedCAP", "1", "1",
                             "no-expression", "--dry-run", "--device", "cpu",
                             "--output-root", str(tmp_path / "not-created")],
                            check=True, capture_output=True, text=True)
    assert "FedCAP_CottonMain.yaml" in result.stdout
    assert not (tmp_path / "not-created").exists()


def test_default_cotton_baseline_entries_use_current_protocol():
    p = experiment.build_parser()
    for algorithm in ("FedCCFA", "FedDrift"):
        args = experiment.resolve_config(p.parse_args(["--algorithm", algorithm, "--device", "cpu"]), p)
        assert args["client_num"] == 90 and args["sample_ratio"] == 0.2 and args["rounds"] == 100
        assert args["cotton_feature_set"] == "no-expression" and args["cotton_test_eval"] is False


def test_baseline_configs_and_yaml_data_path_are_portable(tmp_path, monkeypatch):
    p = baselines.parser()
    for algorithm in (*baselines.REGISTRY, *baselines.REFERENCE_BASELINES):
        args = baselines.resolve_config(p.parse_args(["--algorithm", algorithm, "--device", "cpu"]))
        assert args["cotton_feature_set"] == "no-expression" and not args["cotton_test_eval"]
        assert args["client_num"] == 90 and args["rounds"] == 100
    monkeypatch.chdir(tmp_path)
    args = baselines.resolve_config(p.parse_args([
        "--algorithm", "FedAvgWeighted", "--config", str(ROOT / "configs/FedAvgWeighted_Cotton.yaml")]))
    assert Path(args["cotton_root"]) == ROOT / "data/cotton"


def test_real_loader_gene_split_masks_and_test_switch():
    args = dict(client_num=90, seed=1, root=ROOT / "data/cotton", feature_set="no-expression",
                return_validation=True, return_source_validation=True)
    off = distribute_cotton(**args, include_test=False)
    on = distribute_cotton(**args, include_test=True)
    sets = [set(np.concatenate([d.gene_ids for d in on[index].values()])) for index in (0, 3, 1)]
    assert [len(s) for s in sets] == [43579, 9339, 9339]
    assert not any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    for index in (0, 3):
        for identity, ds in off[index].items():
            other = on[index][identity]
            assert np.array_equal(ds.gene_ids, other.gene_ids)
            assert torch.equal(ds.data, other.data)
            assert np.array_equal(ds.targets, other.targets)
            assert not torch.count_nonzero(ds.data[:, -3:])
    train_ids = np.concatenate([d.gene_ids for d in on[0].values()])
    f = pd.read_csv(ROOT / "data/cotton/clients/c1_cotton_nitrogen/train_ready.tsv", sep="\t", index_col="gene_id")
    expected_mean = f.loc[f.index.isin(set(train_ids)), list(FEATURES)].mean().to_numpy()
    # Expression features differ by source, so compare shared non-expression channels.
    np.testing.assert_allclose(on[0][0].manifest["scaler_mean"][:20], expected_mean[:20], rtol=1e-10)


@pytest.mark.parametrize("name", ["FedCAP_CottonMain.yaml", "FedCAP_CottonNPZ.yaml"])
def test_synthetic_end_to_end_controls(name, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(experiment, "distribute_cotton", synthetic_loader)
    calls = []
    evaluate = experiment.evaluate_client_split
    def observe(clients, local, sources, *args, **kwargs):
        calls.append(sources[0].split)
        return evaluate(clients, local, sources, *args, **kwargs)
    monkeypatch.setattr(experiment, "evaluate_client_split", observe)
    monkeypatch.setattr(sys, "argv", ["FedCAP.py", "--config", str(ROOT / "configs" / name),
        "--clients", "6", "--rounds", "2", "--sample-ratio", "1", "--eval-interval", "1",
        "--rep-epochs", "1", "--clf-epochs", "1", "--device", "cpu", "--output-root", str(tmp_path)])
    experiment.main()
    summaries = list(tmp_path.rglob("summary-*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["status"] == "completed" and not summary["cotton_test_evaluated"]
    assert calls == ["validation", "validation"]
    rows = [json.loads(l) for l in next(tmp_path.rglob("rounds-*.jsonl")).read_text().splitlines()]
    assert all("cotton_source_composition" in row for row in rows)


def test_explicit_test_runs_once_at_final_round(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(experiment, "distribute_cotton", synthetic_loader)
    calls = []
    evaluate = experiment.evaluate_client_split
    def observe(clients, local, sources, *args, **kwargs):
        calls.append(sources[0].split)
        return evaluate(clients, local, sources, *args, **kwargs)
    monkeypatch.setattr(experiment, "evaluate_client_split", observe)
    monkeypatch.setattr(sys, "argv", ["FedCAP.py", "--config", str(ROOT / "configs/FedCAP_CottonNPZ.yaml"),
        "--clients", "6", "--rounds", "2", "--sample-ratio", "1", "--eval-interval", "1",
        "--rep-epochs", "1", "--clf-epochs", "1", "--device", "cpu", "--cotton-test-eval", "true",
        "--output-root", str(tmp_path)])
    experiment.main()
    assert calls == ["validation", "validation", "test"]


def test_weighted_baseline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines, "distribute_cotton", synthetic_loader)
    p = baselines.parser()
    args = baselines.resolve_config(p.parse_args([
        "--algorithm", "FedAvgWeighted", "--config", str(ROOT / "configs/FedAvgWeighted_Cotton.yaml"),
        "--clients", "6", "--rounds", "2", "--sample-ratio", "1", "--eval-interval", "1",
        "--epochs", "1", "--rep-epochs", "1", "--clf-epochs", "1", "--device", "cpu",
        "--output-root", str(tmp_path)]))
    summary = baselines.run(args)
    assert summary["status"] == "completed" and not summary["cotton_test_evaluated"]
    provenance = json.loads(next(tmp_path.rglob("data-*.json")).read_text())
    assert len(provenance["local_train_class_weights"]) == 6
