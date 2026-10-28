"""CPU postprocessing of locked, full-source Cotton predictions; no training."""

import csv
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from utils.cotton_dataset import COTTON_SOURCE_NAMES, distribute_cotton
from utils.cotton_predictions import POLICY, dataset_signature, file_sha256


def load_score_manifest(path, allow_smoke=False):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if (
        meta.get("policy") != POLICY
        or meta.get("status") != "completed"
        or meta.get("evaluation_split") != "test"
        or meta.get("evaluation_scope") != "own_source_full_test"
        or meta.get("labels_included") is not False
    ):
        raise ValueError(f"Not a completed full-source TEST export: {path}")
    if meta.get("smoke_test") and not allow_smoke:
        raise ValueError(
            "Smoke-test exports are excluded; pass --allow-smoke-test for functional checks only"
        )
    name = meta["score_file"]
    if Path(name).name != name:
        raise ValueError("Score file must be a sibling of its manifest")
    scores = path.parent / name
    if file_sha256(scores) != meta["score_sha256"]:
        raise ValueError("Score file checksum mismatch")
    stamp = path.stem.rsplit("-", 1)[-1]
    summary = json.loads(
        (path.parent / f"summary-{stamp}.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("status") != "completed"
        or summary.get("run_completed") is not True
        or summary.get("training_completed") is not True
        or summary.get("cotton_test_evaluated") is not True
        or summary.get("evaluation_split") != "test"
        or summary.get("cotton_source_score_manifest") != path.name
        or summary.get("cotton_source_score_file") != name
        or summary.get("algorithm") != meta["algorithm"]
        or summary.get("seed") != meta["seed"]
        or summary.get("client_num") != meta["client_num"]
        or summary.get("cotton_feature_set") != meta["feature_set"]
    ):
        raise ValueError(
            "Missing/mismatched successful run summary; a score export alone is not a completed run"
        )
    meta["training_data_sha256"] = summary["training_data_sha256"]
    return meta, scores


def score_one_run(meta, score_path, source_datasets, ks):
    if (
        not ks
        or any(not isinstance(k, int) or k < 1 for k in ks)
        or len(set(ks)) != len(ks)
    ):
        raise ValueError("K values must be distinct positive integers")
    if len(source_datasets) != 3 or [d.concept_id for d in source_datasets] != [
        0,
        1,
        2,
    ]:
        raise ValueError("Three ordered source datasets are required")
    for source, dataset in enumerate(source_datasets):
        if dataset_signature(dataset) != meta["source_signatures"][str(source)]:
            raise ValueError(
                "Current data/split/labels differ from the scored source; do not mix dataset versions"
            )
        if max(ks) > len(dataset):
            raise ValueError(f"K={max(ks)} exceeds source {source} size {len(dataset)}")
    models = {m["model_id"]: m for m in meta["models"]}
    if len(models) != len(meta["models"]):
        raise ValueError("Duplicate model IDs in manifest")
    per_client, top_candidates, seen_models, seen_clients = [], [], set(), set()
    rows_seen = 0
    with gzip.open(score_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for model_id, block in itertools.groupby(
            reader, key=lambda row: row["model_id"]
        ):
            if model_id not in models or model_id in seen_models:
                raise ValueError("Unknown, repeated or non-contiguous model block")
            seen_models.add(model_id)
            model = models[model_id]
            source = int(model["source_id"])
            dataset = source_datasets[source]
            records = list(block)
            rows_seen += len(records)
            ids = [row["gene_id"] for row in records]
            positions = {str(gene): i for i, gene in enumerate(dataset.gene_ids)}
            if (
                len(ids) != len(set(ids))
                or len(ids) != len(dataset)
                or set(ids) != set(positions)
                or model["rows"] != len(dataset)
            ):
                raise ValueError(
                    "Missing/duplicate/foreign genes in full-source scores"
                )
            values, labels = [], []
            for row in records:
                i = positions[row["gene_id"]]
                value = float(row["positive_probability"])
                if (
                    int(row["source_id"]) != source
                    or row["family_id"] != str(dataset.family_ids[i])
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Invalid source, family ID or probability")
                values.append(value)
                labels.append(int(dataset.targets[i]))
            labels = np.asarray(labels)
            ap = (
                float(average_precision_score(labels, values)) if labels.any() else None
            )
            saved_ap = model["metrics"]["average_precision"]
            if (ap is None) != (saved_ap is None) or (
                ap is not None
                and not math.isclose(ap, saved_ap, rel_tol=0, abs_tol=1e-7)
            ):
                raise ValueError("Recomputed AP differs from the run's evaluation AP")
            # Ties never depend on labels, CSV order or a test-selected threshold.
            order = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))
            prevalence = float(labels.mean())
            base = {
                "algorithm": meta["algorithm"],
                "seed": meta["seed"],
                "feature_set": meta["feature_set"],
                "evaluation_split": "test",
                "source_id": source,
                "source_name": COTTON_SOURCE_NAMES[source],
                "model_id": model_id,
                "source_samples": len(dataset),
                "positive_fraction": prevalence,
                "average_precision": ap,
            }
            client_ids = model["client_ids"]
            if not client_ids or len(client_ids) != len(set(client_ids)):
                raise ValueError("Invalid model/client mapping")
            for client in client_ids:
                if (
                    client in seen_clients
                    or client < 0
                    or client >= meta["client_num"]
                    or client // (meta["client_num"] // 3) != source
                ):
                    raise ValueError("Client mapping duplicates or crosses sources")
                seen_clients.add(client)
                for k in sorted(ks):
                    hits = int(labels[order[:k]].sum())
                    precision = hits / k
                    per_client.append(
                        dict(
                            base,
                            client_id=client,
                            k=k,
                            hits=hits,
                            precision_at_k=precision,
                            random_expected_hits=k * prevalence,
                            lift=precision / prevalence if prevalence else None,
                        )
                    )
            for rank, i in enumerate(order[: max(ks)], 1):
                top_candidates.append(
                    {
                        "algorithm": meta["algorithm"],
                        "seed": meta["seed"],
                        "feature_set": meta["feature_set"],
                        "source_id": source,
                        "source_name": COTTON_SOURCE_NAMES[source],
                        "model_id": model_id,
                        "client_ids": ";".join(map(str, client_ids)),
                        "rank": rank,
                        "gene_id": ids[i],
                        "family_id": records[i]["family_id"],
                        "positive_probability": values[i],
                    }
                )
    if (
        seen_models != set(models)
        or seen_clients != set(range(meta["client_num"]))
        or rows_seen != meta["score_rows"]
    ):
        raise ValueError("Incomplete model/client/row coverage")
    return per_client, top_candidates


def _mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def aggregate_candidates(rows):
    groups = {}
    for row in rows:
        key = (
            row["algorithm"],
            row["seed"],
            row["feature_set"],
            row["source_id"],
            row["k"],
        )
        groups.setdefault(key, []).append(row)
    source_rows = []
    metrics = (
        "hits",
        "precision_at_k",
        "random_expected_hits",
        "lift",
        "average_precision",
    )
    for (algorithm, seed, feature_set, source, k), group in sorted(groups.items()):
        if len({r["client_id"] for r in group}) != len(group):
            raise ValueError("Duplicate client results; do not pool repeated runs")
        source_rows.append(
            dict(
                algorithm=algorithm,
                seed=seed,
                feature_set=feature_set,
                source_id=source,
                source_name=COTTON_SOURCE_NAMES[source],
                k=k,
                clients=len(group),
                **{key: _mean(group, key) for key in metrics},
            )
        )
    macro_groups = {}
    for row in source_rows:
        key = (row["algorithm"], row["seed"], row["feature_set"], row["k"])
        macro_groups.setdefault(key, []).append(row)
    macro_rows = []
    for (algorithm, seed, feature_set, k), group in sorted(macro_groups.items()):
        if {r["source_id"] for r in group} != {0, 1, 2}:
            raise ValueError("Macro averaging requires all three sources")
        macro_rows.append(
            dict(
                algorithm=algorithm,
                seed=seed,
                feature_set=feature_set,
                k=k,
                **{key: _mean(group, key) for key in metrics},
            )
        )
    return source_rows, macro_rows


def write_candidate_reports(
    manifest_paths, cotton_root, output_dir, ks=(20, 50, 100), allow_smoke=False
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            "Choose a new output directory; previous analyses are never overwritten"
        )
    all_clients, all_tops, runs, signatures, identities, datasets = (
        [],
        [],
        [],
        {},
        set(),
        {},
    )
    for path in sorted(set(map(Path, manifest_paths))):
        meta, scores = load_score_manifest(path, allow_smoke)
        identity = (meta["algorithm"], meta["seed"], meta["feature_set"])
        if identity in identities:
            raise ValueError(
                f"Multiple runs for {identity}; explicitly select one, never silently choose the best/latest"
            )
        identities.add(identity)
        cache_key = (meta["seed"], meta["feature_set"], meta["client_num"])
        if cache_key not in datasets:
            _, _, sources = distribute_cotton(
                meta["client_num"],
                meta["seed"],
                cotton_root,
                feature_set=meta["feature_set"],
                include_test=True,
            )
            datasets[cache_key] = sources
        clients, tops = score_one_run(meta, scores, datasets[cache_key], list(ks))
        match_key = (meta["seed"], meta["feature_set"])
        signature = (
            meta["source_signatures"],
            meta["training_data_sha256"],
            meta["client_num"],
        )
        if match_key in signatures and signatures[match_key] != signature:
            raise ValueError(
                "Methods have different data fingerprints or client counts"
            )
        signatures[match_key] = signature
        all_clients.extend(clients)
        all_tops.extend(tops)
        runs.append(
            {
                "manifest": str(Path(path).resolve()),
                "manifest_sha256": file_sha256(path),
                "algorithm": meta["algorithm"],
                "seed": meta["seed"],
                "feature_set": meta["feature_set"],
            }
        )
    if not runs:
        raise ValueError("No source score manifests selected")
    source_rows, macro_rows = aggregate_candidates(all_clients)
    deltas = []
    for scope, records in (("source", source_rows), ("macro", macro_rows)):
        references = {
            (r["seed"], r["feature_set"], r.get("source_id"), r["k"]): r
            for r in records
            if r["algorithm"] == "FedCCFA"
        }
        for row in records:
            reference = references.get(
                (row["seed"], row["feature_set"], row.get("source_id"), row["k"])
            )
            if row["algorithm"] == "FedCCFA" or reference is None:
                continue
            deltas.append(
                {
                    "algorithm": row["algorithm"],
                    "reference": "FedCCFA",
                    "scope": scope,
                    "seed": row["seed"],
                    "feature_set": row["feature_set"],
                    "source_id": row.get("source_id"),
                    "k": row["k"],
                    "delta_hits": row["hits"] - reference["hits"],
                    "delta_precision_at_k": row["precision_at_k"]
                    - reference["precision_at_k"],
                }
            )
    summaries, groups = [], {}
    for scope, records in (("source", source_rows), ("macro", macro_rows)):
        for row in records:
            groups.setdefault(
                (
                    scope,
                    row["algorithm"],
                    row["feature_set"],
                    row.get("source_id"),
                    row["k"],
                ),
                [],
            ).append(row)
    for (scope, algorithm, feature_set, source, k), group in sorted(
        groups.items(), key=str
    ):
        item = {
            "scope": scope,
            "algorithm": algorithm,
            "feature_set": feature_set,
            "source_id": source,
            "k": k,
            "n_seeds": len(group),
            "seeds": ";".join(str(r["seed"]) for r in group),
        }
        for key in ("hits", "precision_at_k", "lift", "average_precision"):
            values = [r[key] for r in group if r[key] is not None]
            item[key + "_mean"] = float(np.mean(values)) if values else None
            item[key + "_sample_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summaries.append(item)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in (
        ("per-client.csv", all_clients),
        ("per-source.csv", source_rows),
        ("macro.csv", macro_rows),
        ("across-seeds.csv", summaries),
        ("top-candidates-unlabelled.csv", all_tops),
    ):
        with (output_dir / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (output_dir / "delta-vs-FedCCFA.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "algorithm",
            "reference",
            "scope",
            "seed",
            "feature_set",
            "source_id",
            "k",
            "delta_hits",
            "delta_precision_at_k",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(deltas)
    result = {
        "status": "completed",
        "evaluation_split": "test",
        "ks": list(ks),
        "runs": runs,
        "averaging": "equal clients within source, then equal three sources; no probability ensemble",
        "tie_policy": "descending score then ascending gene_id",
        "labels_in_candidate_file": False,
        "note": "Descriptive computational evaluation, not independent biological validation; model runs are not biological replicates.",
    }
    with (output_dir / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result
