"""Read-only Cotton reports; source-specific and union TEST scopes stay distinct."""
import numpy as np
from torch.utils.data import ConcatDataset

from entities.FedCAPProfile import isolated_profile_rng


SOURCE_NAMES = ("nitrogen", "mutant", "salt")
METRICS = ("accuracy", "balanced_accuracy", "f1", "auroc", "average_precision")


def _mean(values):
    present = [value for value in values if value is not None]
    if present and not np.isfinite(present).all():
        raise FloatingPointError("Non-finite Cotton report metric")
    return float(np.mean(present)) if present else None


def cotton_ap_summary(source_reports, client_average_precision=()):
    """Return explicitly named equal-source and worst-case AP summaries."""
    names = SOURCE_NAMES
    source_values = [source_reports[name]["generalized_average_precision"]
                     for name in names]
    if any(value is None for value in source_values):
        macro_source = worst_source = None
    else:
        source_values = np.asarray(source_values, dtype=float)
        if not np.isfinite(source_values).all():
            raise FloatingPointError("Non-finite Cotton source AP")
        macro_source = float(np.mean(source_values))
        worst_source = float(np.min(source_values))
    client_values = [value for value in client_average_precision if value is not None]
    if client_values and not np.isfinite(client_values).all():
        raise FloatingPointError("Non-finite Cotton client AP")
    return {
        "macro_source_average_precision": macro_source,
        "worst_source_average_precision": worst_source,
        "worst_client_average_precision": float(np.min(client_values)) if client_values else None,
    }


def cotton_source_report(clients, metrics, global_tests, local_datasets=None,
                         evaluation_scope="own_source_full_test"):
    """Average client metrics within each real source, not an inferred cluster."""
    by_id = {row["client_id"]: row for row in metrics}
    if len(by_id) != len(metrics) or set(by_id) != {client.id for client in clients}:
        raise ValueError("Cotton source report requires exactly one row per client")
    if len(global_tests) != 3 or [data.concept_id for data in global_tests] != [0, 1, 2]:
        raise ValueError("Cotton reports require the three ordered source test sets")
    result = {}
    split_name = "validation" if "validation" in evaluation_scope else "test"
    for source, name in enumerate(SOURCE_NAMES):
        group = [client for client in clients if client.train_set.concept_id == source]
        if not group or any(client.train_set.concept_id != source or
                            client.global_test_id != source for client in group):
            raise ValueError("Cotton train/local/full test source IDs must agree")
        rows = [by_id[client.id] for client in group]
        full_test = global_tests[source]
        local_size = sum(len(local_datasets[client.id] if local_datasets is not None
                              else client.test_set) for client in group)
        report = dict(client_ids=sorted(client.id for client in group), clients=len(group),
                      train_samples=sum(len(client.train_set) for client in group),
                      **{f"local_{split_name}_samples": local_size},
                      **{f"source_{split_name}_samples": len(full_test)},
                      **{f"source_{split_name}_positive_fraction": float(np.mean(np.asarray(full_test.targets) == 1))},
                      evaluation_scope=evaluation_scope, metric_averaging="equal_clients")
        for scope in ("local", "global"):
            for metric in METRICS:
                report[("generalized" if scope == "global" else scope) + "_" + metric] = _mean(
                    [row[f"{scope}_{metric}"] for row in rows])
        result[name] = report
    if sum(report["clients"] for report in result.values()) != len(clients):
        raise ValueError("Unknown Cotton source ID")
    return result


def cotton_union_report(clients, global_tests, evaluator, device, seed, model_key=None):
    """Final-only TEST diagnostic. Never controls training, sharing or selection.

    Every model is evaluated on the same concatenation of three source tests.
    Sources therefore contribute by TEST sample count inside each model's metric.
    RNG and all module train/eval flags are restored, even if evaluation fails.
    model_key may deduplicate identical global-body/cluster-head predictions.
    """
    union = ConcatDataset(global_tests)
    rows, cache = [], {}
    with isolated_profile_rng(seed, device):
        for client in clients:
            key = model_key(client.id) if model_key is not None else client.id
            if key not in cache:
                modes = [(module, module.training) for module in client.model.modules()]
                try:
                    cache[key] = evaluator(client.model, union, device, True)
                finally:
                    for module, mode in modes:
                        module.training = mode
            rows.append(dict(client_id=client.id, **{
                "cross_source_" + metric: cache[key][metric] for metric in METRICS}))
    report = dict(evaluation_scope="union_of_three_source_tests", diagnostic_only=True,
                  evaluation_timing="after_final_round", metric_averaging="equal_clients",
                  within_model_source_weighting="test_sample_count", clients=len(clients),
                  unique_models_evaluated=len(cache), test_samples=len(union),
                  source_test_samples=[len(data) for data in global_tests])
    for metric in METRICS:
        report["generalized_" + metric] = _mean([row["cross_source_" + metric] for row in rows])
    report["generalized_client_std"] = float(np.std([row["cross_source_accuracy"] for row in rows]))
    return report, rows
