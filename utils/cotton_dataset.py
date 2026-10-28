"""Read-only Cotton loader: global family split and training-only scaling."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


EXPRESSION_FEATURES = (
    "expr_mean_log2cpm",
    "expr_sd_log2cpm",
    "expr_detected_frac",
)
FEATURE_GROUPS = {
    "atac": ("atac_prom_HN_AS98", "atac_prom_HN_WT", "atac_prom_LN_AS98",
             "atac_prom_LN_WT", "atac_prom_mean"),
    "chip": ("drebind_occ_4", "drebind_occ_5", "drebind_occ_6",
             "drebind_occ_mean", "drebind_occ_sd"),
    "sequence": ("gc", "DRE_full", "DRE_core", "ABRE", "G_box", "GCC_box"),
    "structure": ("gene_len", "n_exon_blocks", "exon_len_total", "exon_fraction"),
}
FEATURE_ABLATIONS = {
    "sequence-structure": ("sequence", "structure"),
    "sequence-structure-atac": ("sequence", "structure", "atac"),
    "sequence-structure-chip": ("sequence", "structure", "chip"),
}
FEATURE_SETS = ("all23", "no-expression", *FEATURE_ABLATIONS)
COTTON_SOURCE_NAMES = ("nitrogen", "mutant", "salt")


def active_feature_names(features, feature_set):
    """Named input masks; old all23/no-expression semantics are unchanged."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    if feature_set == "all23":
        return tuple(features)
    if not set(EXPRESSION_FEATURES).issubset(features):
        raise ValueError("Non-expression feature sets require all three expr_* columns")
    if feature_set == "no-expression":
        return tuple(name for name in features if name not in EXPRESSION_FEATURES)
    wanted = {name for group in FEATURE_ABLATIONS[feature_set] for name in FEATURE_GROUPS[group]}
    if not wanted.issubset(features):
        raise ValueError(f"Missing feature columns: {sorted(wanted - set(features))}")
    return tuple(name for name in features if name in wanted)


def resolve_cotton_roots(root=None):
    """Prefer checkout-local data; retain the legacy project-level fallback.

    An explicit root is authoritative: a missing/invalid explicit directory
    must fail rather than silently train on another dataset.
    """
    checkout = Path(__file__).resolve().parents[1]
    candidates = ([Path(root).expanduser().absolute().resolve()] if root else
                  [checkout / "data" / "cotton", checkout / "data" / "cotton30",
                   checkout.parent / "_data"])
    for candidate in candidates:
        family_files = (candidate / "work" / "gene_families.tsv",
                        candidate / "metadata" / "gene_families.tsv")
        if (candidate / "clients").is_dir() and any(p.is_file() for p in family_files):
            data_root = candidate
            break
    else:
        raise FileNotFoundError(
            "Cotton data root must contain clients/ and either work/gene_families.tsv "
            "or metadata/gene_families.tsv; checked: "
            + ", ".join(str(p) for p in candidates)
            + "; pass --cotton-root to select a dataset explicitly"
        )
    client_root = data_root / "clients"
    family_candidates = (data_root / "work" / "gene_families.tsv",
                         data_root / "metadata" / "gene_families.tsv")
    family_path = next((path for path in family_candidates if path.is_file()), None)
    if not client_root.is_dir() or family_path is None:
        raise FileNotFoundError(
            "Cotton data root must contain clients/ and either work/gene_families.tsv "
            "or metadata/gene_families.tsv; pass --cotton-root for a standalone copy"
        )
    return data_root, client_root, family_path


class CottonDataset(Dataset):
    def __init__(self, frame, features, scaler, concept_id, feature_set="all23"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        inputs = scaler.transform(frame[features]).astype(np.float32)
        active = active_feature_names(features, feature_set)
        excluded = [i for i, name in enumerate(features) if name not in active]
        if excluded:
            # Keep the 23-input MLP and fitted training-only scaler unchanged.
            inputs[:, excluded] = 0.0
        self.data = torch.from_numpy(inputs)
        self.targets = frame.label.to_numpy(dtype=np.int64, copy=True)
        self.gene_ids = frame.index.to_numpy(copy=True)
        self.family_ids = frame.family_id.to_numpy(copy=True)
        self.concept_id = concept_id
        self.feature_set = feature_set
        self.input_features = active
        self.excluded_features = tuple(name for name in features
                                      if name not in self.input_features)
        self.transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])


def distribute_cotton(client_num, seed, root=None, return_validation=False,
                      feature_set="all23", return_source_validation=False,
                      include_test=True):
    if client_num < 3 or client_num % 3:
        raise ValueError("Cotton client_num must be a positive multiple of 3")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    if return_source_validation and not return_validation:
        raise ValueError("return_source_validation requires return_validation=True")
    if not include_test and not return_validation:
        raise ValueError("A validation-only load requires return_validation=True")
    data_root, client_root, family_path = resolve_cotton_roots(root)
    families = pd.read_csv(family_path, sep="\t", index_col="gene_id")
    if not families.index.is_unique or families.family_id.isna().any():
        raise ValueError("Invalid gene family mapping")
    frames, features = [], None
    for name in ("c1_cotton_nitrogen", "c2_cotton_mutant", "c3_cotton_salt"):
        frame = pd.read_csv(client_root / name / "train_ready.tsv", sep="\t", index_col="gene_id")
        cols = [c for c in frame if c not in ("label", "log2FoldChange", "padj")]
        if len(cols) != 23 or len(set(cols)) != 23 or not frame.index.is_unique:
            raise ValueError(f"Invalid feature schema or gene IDs in {name}")
        if not set(EXPRESSION_FEATURES).issubset(cols):
            raise ValueError(f"Missing expression-derived features in {name}")
        if features is not None and cols != features:
            raise ValueError("Cotton sources must use identical feature order")
        features = cols
        if not np.isfinite(frame[cols].to_numpy()).all() or not frame.label.isin([0, 1]).all():
            raise ValueError(f"Invalid model inputs in {name}")
        frame = frame.join(families[["family_id"]], validate="one_to_one")
        if frame.family_id.isna().any():
            raise ValueError("Missing family IDs")
        frames.append(frame)
    rng = np.random.default_rng(seed)
    unique = np.unique(np.concatenate([f.family_id.to_numpy() for f in frames]))
    rng.shuffle(unique)
    a, b = int(len(unique) * .7), int(len(unique) * .85)
    splits = [set(unique[:a]), set(unique[a:b]), set(unique[b:])]
    training = [f[f.family_id.isin(splits[0])] for f in frames]
    scaler = StandardScaler().fit(pd.concat(training)[features])
    train, test, global_tests = {}, {}, []
    per_source = client_num // 3
    for source, frame in enumerate(frames):
        if include_test:
            source_test = frame[frame.family_id.isin(splits[2])]
            global_tests.append(CottonDataset(source_test, features, scaler, source, feature_set))
        destinations = ((0, train), (2, test)) if include_test else ((0, train),)
        for split_index, destination in destinations:
            # Independent streams: creating TEST must never change a later
            # source's training allocation. Versioned because v1 shared rng.
            allocation_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 710003, source, split_index]))
            subset = frame[frame.family_id.isin(splits[split_index])]
            # Whole families stay together, with positive-rich families assigned first.
            family_ids, inverse = np.unique(subset.family_id.to_numpy(), return_inverse=True)
            family_sizes = np.bincount(inverse)
            family_positives = np.bincount(inverse, weights=subset.label.to_numpy())
            order = allocation_rng.permutation(len(family_ids))
            order = order[np.argsort(-family_positives[order], kind="stable")]
            owners, loads = np.zeros(len(family_ids), dtype=int), np.zeros(per_source)
            positives = np.zeros(per_source)
            for family in order:
                target = int(np.argmin(positives if family_positives[family] else loads))
                owners[family] = target
                loads[target] += family_sizes[family]
                positives[target] += family_positives[family]
            row_owners = owners[inverse]
            for j in range(per_source):
                bucket = subset.iloc[np.flatnonzero(row_owners == j)]
                if bucket.empty:
                    raise ValueError("Too many virtual clients for family split")
                data = CottonDataset(bucket, features, scaler, source, feature_set)
                if split_index == 0 and np.bincount(data.targets, minlength=2).min() < 5:
                    raise ValueError("Each Cotton training client needs at least 5 samples per class")
                destination[source * per_source + j] = data
    active_features = list(active_feature_names(features, feature_set))
    manifest = {"data_root": str(data_root.resolve()), "feature_set": feature_set,
                "features": features, "active_features": active_features,
                "excluded_model_inputs": [name for name in features if name not in active_features],
                "split": "family-disjoint 70/15/15", "seed": seed,
                "allocation_policy": "independent_source_split_rng_v2",
                "test_materialized": bool(include_test),
                "family_counts": [len(s) for s in splits], "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "train_sizes": [len(train[i]) for i in train],
                "validation_sizes": [int(f.family_id.isin(splits[1]).sum()) for f in frames]}
    for data in train.values():
        data.manifest = manifest
    if return_validation:
        # Build validation only AFTER the historical train/test allocation.
        # A separate RNG keeps existing partitions bit-for-bit unchanged.
        validation = {}
        global_validation_tests = []
        val_rng = np.random.default_rng(seed + 971003)
        for source, frame in enumerate(frames):
            subset = frame[frame.family_id.isin(splits[1])]
            global_validation_tests.append(
                CottonDataset(subset, features, scaler, source, feature_set)
            )
            ids, inverse = np.unique(subset.family_id.to_numpy(), return_inverse=True)
            sizes = np.bincount(inverse)
            positive_counts = np.bincount(inverse, weights=subset.label.to_numpy())
            order = val_rng.permutation(len(ids))
            order = order[np.argsort(-positive_counts[order], kind="stable")]
            owners = np.zeros(len(ids), dtype=int)
            loads, positives = np.zeros(per_source), np.zeros(per_source)
            for family in order:
                target = int(np.argmin(positives if positive_counts[family] else loads))
                owners[family] = target
                loads[target] += sizes[family]
                positives[target] += positive_counts[family]
            for j in range(per_source):
                bucket = subset.iloc[np.flatnonzero(owners[inverse] == j)]
                if bucket.empty:
                    raise ValueError("Too many Cotton clients for validation family split")
                validation[source * per_source + j] = CottonDataset(
                    bucket, features, scaler, source, feature_set
                )
        if return_source_validation:
            return train, test, global_tests, validation, global_validation_tests
        return train, test, global_tests, validation
    return train, test, global_tests
