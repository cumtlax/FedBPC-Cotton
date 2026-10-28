"""Opt-in, label-free full-source scores from the existing evaluation pass.

Identical evaluation models are stored once; the manifest preserves every
client's membership so postprocessing retains equal-client/equal-source weights.
No model selection, fitting, random sampling or extra forward pass occurs here.
"""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from utils.cotton_dataset import COTTON_SOURCE_NAMES

POLICY = "cotton_full_source_scores_v1"
FIELDS = ("model_id", "source_id", "gene_id", "family_id", "positive_probability")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_signature(dataset):
    """Bind ordered source rows and labels without publishing label values."""
    digest = hashlib.sha256()
    ids = [str(value) for value in dataset.gene_ids]
    families = [str(value) for value in dataset.family_ids]
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(dataset):
        raise ValueError("Source scores require nonempty, unique gene IDs")
    if len(families) != len(ids) or len(dataset.targets) != len(ids):
        raise ValueError("Source IDs, families and targets have unequal lengths")
    if not np.isin(dataset.targets, [0, 1]).all():
        raise ValueError("Source targets must be binary")
    for gene, family, label in zip(ids, families, dataset.targets):
        digest.update(
            (json.dumps([gene, family, int(label)], ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
        )
    inputs = np.ascontiguousarray(dataset.data)
    input_digest = hashlib.sha256()
    input_digest.update(str(inputs.dtype).encode())
    input_digest.update(json.dumps(list(inputs.shape)).encode())
    input_digest.update(inputs.tobytes())
    return {
        "rows": len(ids),
        "ordered_rows_sha256": digest.hexdigest(),
        "inputs_sha256": input_digest.hexdigest(),
    }


def gene_prediction_rows(dataset, scores):
    if len(scores) != len(dataset):
        raise ValueError("Prediction count does not match the dataset")
    return [
        {
            "gene_id": str(gene),
            "family_id": str(family),
            "positive_probability": float(score),
        }
        for gene, family, score in zip(dataset.gene_ids, dataset.family_ids, scores)
    ]


class CottonSourceScoreWriter:
    def __init__(self, folder, stamp, args, source_datasets):
        self.folder = Path(folder)
        self.score_path = self.folder / f"cotton-source-scores-{stamp}.csv.gz"
        self.meta_path = self.folder / f"cotton-source-scores-{stamp}.json"
        if self.score_path.exists() or self.meta_path.exists():
            raise FileExistsError("Refusing to overwrite source score artifacts")
        if args["client_num"] < 3 or args["client_num"] % 3:
            raise ValueError("Cotton client count must be a positive multiple of three")
        if len(source_datasets) != 3 or [d.concept_id for d in source_datasets] != [
            0,
            1,
            2,
        ]:
            raise ValueError("Expected the three ordered full-source datasets")
        self.sources = source_datasets
        self.meta = {
            "policy": POLICY,
            "status": "incomplete",
            "labels_included": False,
            "algorithm": args["algorithm"],
            "seed": args["seed"],
            "client_num": args["client_num"],
            "feature_set": args["cotton_feature_set"],
            "smoke_test": bool(args.get("smoke_test", False)),
            "evaluation_split": "test",
            "evaluation_scope": "own_source_full_test",
            "metric_averaging": "equal_clients_then_equal_sources",
            "score_file": self.score_path.name,
            "source_signatures": {
                str(i): dataset_signature(d) for i, d in enumerate(source_datasets)
            },
            "models": [],
        }
        self.by_key = {}
        self.clients = set()
        self.rows = 0
        self.summary = None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.score_path, "xt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
        self.writer.writeheader()
        return self

    def add_model(self, key, source_id, predictions, metrics):
        if key in self.by_key:
            raise ValueError("Evaluation model registered twice")
        dataset = self.sources[source_id]
        if len(predictions) != len(dataset):
            raise ValueError("Full-source prediction count mismatch")
        model_id = f"model-{len(self.by_key):04d}"
        item = {
            "model_id": model_id,
            "source_id": source_id,
            "source_name": COTTON_SOURCE_NAMES[source_id],
            "client_ids": [],
            "metrics": metrics,
            "rows": len(dataset),
        }
        for index, row in enumerate(predictions):
            if row["gene_id"] != str(dataset.gene_ids[index]) or row[
                "family_id"
            ] != str(dataset.family_ids[index]):
                raise ValueError("Prediction IDs do not match source dataset order")
            score = float(row["positive_probability"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid positive-class probability")
            self.writer.writerow(dict(model_id=model_id, source_id=source_id, **row))
            self.rows += 1
        self.by_key[key] = item
        self.meta["models"].append(item)

    def add_client(self, key, source_id, client_id):
        item = self.by_key[key]
        if (
            item["source_id"] != source_id
            or client_id in self.clients
            or client_id < 0
            or client_id >= self.meta["client_num"]
            or client_id // (self.meta["client_num"] // 3) != source_id
        ):
            raise ValueError("Incorrect or duplicate client/source assignment")
        self.clients.add(client_id)
        item["client_ids"].append(int(client_id))

    def __exit__(self, exc_type, exc, traceback):
        self.handle.close()
        if exc_type is None:
            if (
                len(self.clients) != self.meta["client_num"]
                or any(not item["client_ids"] for item in self.meta["models"])
                or {item["source_id"] for item in self.meta["models"]} != {0, 1, 2}
            ):
                raise ValueError("Source score export is missing clients or sources")
            self.meta.update(
                status="completed",
                score_rows=self.rows,
                score_sha256=file_sha256(self.score_path),
            )
            with self.meta_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    self.meta, handle, ensure_ascii=False, indent=2, allow_nan=False
                )
            self.summary = {
                "cotton_source_score_file": self.score_path.name,
                "cotton_source_score_manifest": self.meta_path.name,
                "cotton_source_score_rows": self.rows,
                "cotton_source_score_policy": POLICY,
                "cotton_source_score_labels_included": False,
            }
        return False
