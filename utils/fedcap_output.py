"""Readable, configuration-derived log folders; no training state is changed."""
from pathlib import Path
import re


_NUMBERED_RUN = re.compile(
    r"(?:config|data|provenance|rounds|profiles|results|checkpoint|summary|error|traceback|cotton-gene-scores|"
    r"numerical|mechanism|mechanism-first-alarm|stages|cotton-source-scores)-([0-9]+)\.[^.]+(?:\.gz)?$|\.run-([0-9]+)$")


def cotton_feature_suffix(args):
    feature_set = args.get("cotton_feature_set", "all23")
    if feature_set == "all23":
        return ""
    if feature_set == "no-expression":
        return "-NoExpression"
    return "-Features-" + feature_set


def reserve_run(folder):
    """Atomically reserve a short, folder-local ID, including failed runs.

    Persistent markers prevent concurrent launches from choosing the same ID.
    Existing numbered artifacts also count, so copying results without their
    markers cannot cause reuse. Historical parameter/timestamp names are ignored.
    This helper never touches training RNG or rewrites existing artifacts.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    last = 0
    for path in folder.iterdir():
        match = _NUMBERED_RUN.fullmatch(path.name)
        if match:
            last = max(last, int(next(group for group in match.groups() if group is not None)))
    number = last + 1
    while True:
        stamp = f"{number:03d}"
        try:
            with (folder / f".run-{stamp}").open("x", encoding="utf-8") as stream:
                stream.write("Reserved run ID; keep this marker to prevent reuse.\n")
        except FileExistsError:
            number += 1
            continue
        return stamp


def dataset_folder(dataset):
    """Cotton aliases identify one dataset, including its output location."""
    name = dataset.lower()
    return "cotton30" if name == "cotton" else name


def variant_name(args):
    """Classify effective switches, not a recipe label that may be stale."""
    if args["algorithm"] != "FedCAP":
        return args["algorithm"]
    if args.get("diagnostic_head_sharing", "none") == "oracle-class":
        name = "FedCAP-OracleRows"
    elif args.get("head_sharing", "none") == "profile":
        rule = args.get("class_share_rule", "witness")
        independent = args.get("profile_samples_per_class", 0) > 0
        name = ("FedCAP-Sampling" if independent else "FedCAP-Centroid") if rule == "centroid" else (
            "FedCAP-WitnessSampling" if independent else "FedCAP-Witness")
    else:
        name = "FedCAP-V3"
    if args.get("class_share_centering", "none") == "head-mean":
        name += "-Centered"
    if args.get("class_share_task_gate", "none") != "none":
        name += "-TaskGate"
    guard = args.get("profile_merge_guard", "none")
    if guard == "conflict":
        name += "-Conflict"
    elif guard == "evidence":
        name += "-Evidence"
        if args.get("profile_merge_memory_rounds", 10) > 0:
            name += "-Memory"
    if args.get("merge_guard", "none") == "loss":
        name += "-LossGuard"
    if args.get("cluster_method", "agglomerative") == "hdbscan":
        name += "-HDBSCAN"
    if args.get("descriptor_cache_policy") == "current_round_same_global_body_v1":
        name += "-Fresh"
    if args.get("profile_head", "reference") == "none":
        name += "-BodyProfile"
    if args.get("profile_weight") == 0:
        name += "-ClassifierOnly"
    if args.get("body_prox_mu", 0) > 0:
        name += f"-BodyProx{args['body_prox_mu']:g}"
    if args.get("head_calibration_ratio", 0) > 0:
        name += f"-HeadCal{args['head_calibration_ratio']:g}"
    if args.get("class_support_alpha", 0) > 0:
        name += f"-ClassSupport{args['class_support_alpha']:g}"
        cap = args.get("class_support_count_cap", 0)
        if cap != 64:
            name += "-CapFull" if cap == 0 else f"-Cap{cap}"
    if args.get("class_support_scope", "all") == "cross-cluster":
        name += "-CrossOnly"
    if args.get("class_share_witness_policy") == "consensus":
        name += f"-Consensus{args.get('class_share_consensus_ratio', 0.8):g}"
    if args.get("collaboration_policy") == "fresh-verified":
        name += "-VerifiedV2"
    if args.get("body_aggregation_step", 1.0) != 1.0:
        name += f"-BodyStep{args['body_aggregation_step']:g}"
    if args.get("class_share_label_mode") == "aligned":
        name += "-LabelAligned"
    if args.get("label_merge_guard") == "cross-label":
        name += "-MergeConsistent"
    if args.get("cotton_grouping") == "source":
        name += "-FixedSource"
    if args.get("head_sharing") == "profile" and args.get("class_share_rho", 1.0) != 1.0:
        name += f"-Rho{args['class_share_rho']:g}"
    if args.get("dataset", "").lower() in ("cotton", "cotton30"):
        name += cotton_feature_suffix(args)
        if not args.get("cotton_test_eval", True):
            name += "-ValidationOnly"
    return name


def output_folder(args):
    """FedCAP variants own a top-level folder; baseline paths stay unchanged."""
    dataset = dataset_folder(args["dataset"])
    root = Path(args["output_root"]) / "log"
    if args["algorithm"] == "FedCAP":
        root = root / variant_name(args) / dataset
    else:
        method = args["algorithm"]
        if dataset == "cotton30":
            method += cotton_feature_suffix(args)
        if dataset == "cotton30" and not args.get("cotton_test_eval", True):
            method += "-ValidationOnly"
        root = root / dataset / method
    drift = f"drift-{str(args['drift_pattern']).lower()}"
    if args.get("drift_class_count", 6) != 6 and args["drift_pattern"] not in (False, "false"):
        drift += f"-classes{args['drift_class_count']}"
    return root / f"clients-{args['client_num']}" / drift / f"seed-{args['seed']}"


def summary_paths(root, dataset, drift, clients, seed):
    """Read unified log/, with read-only compatibility for historical Cotton."""
    dataset = dataset_folder(dataset)
    log_roots = [Path(root) / "log"]
    if dataset == "cotton30":
        log_roots.append(Path(root) / "log_cotton")
    tail = Path(f"clients-{clients}") / f"drift-{str(drift).lower()}" / f"seed-{seed}"
    paths = []
    for log_root in log_roots:
        legacy = log_root / dataset / "FedCAP" / tail
        paths.extend(legacy.glob("summary-*.json"))
        for variant in sorted(log_root.glob("FedCAP-*")):
            paths.extend((variant / dataset / tail).glob("summary-*.json"))
    return sorted(set(paths))
