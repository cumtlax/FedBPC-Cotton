"""FedCAP runner and Cotton/FedCCFA/FedDrift comparison entry point."""
import argparse
import csv
from contextlib import ExitStack, nullcontext
from copy import copy, deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from entities.FedCAP import (FedCAPClient, FedCAPServer, CCFA_PROBE_POLICY,
                             LEGACY_PROBE_POLICY, measurement_policies,
                             validate_body_aggregation_step, body_aggregation_policy, BODY_STEP_SUFFIX)
from entities.FedCAPOracleClassShare import FedCAPOracleClassShareServer, oracle_label_maps_for_round
from entities.FedCAPStability import RegroupConfirmation
from entities.FedCAPMergeGuard import MergeLossGuard
from entities.FedCAPProfileMerge import make_profile_merge_guard
from entities.FedCAPProfileShare import (
    FedCAPProfileShareServer, validate_label_mode, LABEL_ALIGNMENT_SUFFIX,
    LABEL_ALIGNMENT_POLICY, ALIGNED_HEAD_AGGREGATION, validate_label_merge_guard,
    LABEL_MERGE_SUFFIX, LABEL_MERGE_POLICY)
from entities.FedCAPProfileConsensus import (
    validate_consensus, POLICY as CONSENSUS_POLICY, DONOR_POLICY as CONSENSUS_DONOR_POLICY,
    IMPLEMENTATION_SUFFIX as CONSENSUS_SUFFIX)
from entities.FedCAPRowTaskGate import validate_task_gate, POLICY as TASK_GATE_POLICY
from entities.FedCAPHeadCentering import (
    validate_head_centering, POLICY as CENTERING_POLICY, IMPLEMENTATION_SUFFIX as CENTERING_SUFFIX)
from entities.FedCAPProfile import (
    LEGACY_POLICY, UNIQUE_POLICY, profile_budget, summarize_profiles)
from utils.fedcap_profile_ablation import MODES as PROFILE_ABLATIONS, apply_profile_ablation
from utils.fedcap_output import output_folder, reserve_run
from utils.fedcap_body_proximal import POLICY as BODY_PROX_POLICY, validate_mu, proximal_summary
from utils.fedcap_head_calibration import (POLICY as HEAD_CALIBRATION_POLICY,
                                           COMMUNICATION_POLICY, validate_ratio, calibration_summary)
from entities.FedCAPClassSupport import (
    validate_class_support, validate_support_scope, validate_count_cap, support_policy, implementation_suffix,
    SUPPORT_SCOPES, CROSS_ONLY_SUFFIX,
    COUNT_CAP as CLASS_SUPPORT_CAP, DEFAULT_COUNT_CAP)
from utils.fedcap_dataset_reports import (cotton_ap_summary, cotton_source_report,
                                         cotton_union_report)
from utils.fedcap_numerics import nonfinite_names, record_failures
from utils.fedcap_mechanism import MechanismDiagnostics
from utils.fedcap_stages import StageDiagnostics, validate_stages
from utils.cotton_collaboration import preserve_model_modes
from utils.cotton_dataset import (distribute_cotton, FEATURE_SETS as COTTON_FEATURE_SETS,
                                 COTTON_SOURCE_NAMES)
from utils.cotton_predictions import CottonSourceScoreWriter
from utils.cotton_paths import cotton_output_root
from utils.gen_dataset import distribute_dataset, dataset_partition_settings, CIFAR100_MEAN, CIFAR100_STD
from utils.drift import sudden_drift, incremental_drift, label_swap_pairs
from utils.fedcap_reproducibility import (
    POLICY as REPRODUCIBILITY_POLICY, configure_determinism, runtime_fingerprint,
    source_fingerprint, training_fingerprint, state_fingerprint,
    whole_head_fingerprint, rng_fingerprint)

ROOT = Path(__file__).resolve().parents[1]
CAP_ALGORITHMS = ("FedCAP",)
WHOLE_HEAD_ALGORITHMS = ("FedCAP",)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@preserve_model_modes
def evaluate(model, dataset, device, cotton, return_predictions=False):
    model.eval()
    truths, probabilities = [], []
    gene_predictions, offset = [], 0
    with torch.no_grad():
        # DataLoader iterator construction draws a base seed even without
        # shuffling. Keep evaluation frequency out of the training RNG stream.
        for batch, (x, y) in enumerate(DataLoader(
                dataset, batch_size=1024,
                generator=torch.Generator().manual_seed(0) if cotton else None), start=1):
            x = x.to(device)
            if nonfinite_names({"inputs": x}):
                raise FloatingPointError(f"Non-finite evaluation inputs: batch={batch}")
            logits = model(x)
            if nonfinite_names({"logits": logits}):
                raise FloatingPointError(f"Non-finite evaluation logits: batch={batch}")
            probability = logits.softmax(1)
            if nonfinite_names({"probabilities": probability}):
                raise FloatingPointError(f"Non-finite evaluation probabilities: batch={batch}")
            probabilities.append(probability.cpu().numpy())
            truths.append(y.numpy())
            if return_predictions:
                if not hasattr(dataset, "gene_ids"):
                    raise ValueError("Gene-level predictions require a CottonDataset with gene_ids")
                ids = dataset.gene_ids[offset:offset + len(y)]
                families = getattr(dataset, "family_ids", [None] * len(dataset))[offset:offset + len(y)]
                gene_predictions.extend(
                    {"gene_id": str(gene_id), "family_id": str(family_id),
                     "positive_probability": float(score)}
                    for gene_id, family_id, score in zip(
                        ids, families, probability[:, 1].cpu().tolist())
                )
                offset += len(y)
    y, p = np.concatenate(truths), np.concatenate(probabilities)
    result = {"accuracy": float(accuracy_score(y, p.argmax(1)))}
    if cotton:
        result.update(balanced_accuracy=float(balanced_accuracy_score(y, p.argmax(1))),
                      f1=float(f1_score(y, p.argmax(1), zero_division=0)),
                      auroc=float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None,
                      average_precision=float(average_precision_score(y, p[:, 1])) if (y == 1).any() else None)
    return (result, gene_predictions) if return_predictions else result


def evaluate_client_split(clients, local_datasets, source_datasets, device, cotton,
                          model_key=None, capture_local_predictions=False,
                          source_score_writer=None):
    """Evaluate client models on an explicitly supplied split, never an implicit test set."""
    rows, source_cache, predictions = [], {}, []
    for client in clients:
        local_result = evaluate(client.model, local_datasets[client.id], device, cotton,
                                 return_predictions=capture_local_predictions)
        if capture_local_predictions:
            local_metrics, client_predictions = local_result
            predictions.extend({"client_id": client.id, "source_id": client.global_test_id,
                                "source_name": COTTON_SOURCE_NAMES[client.global_test_id],
                                **row}
                               for row in client_predictions)
        else:
            local_metrics = local_result
        key = ((model_key(client.id) if model_key is not None else client.id),
               client.global_test_id)
        if key not in source_cache:
            source_result = evaluate(client.model, source_datasets[client.global_test_id],
                                     device, cotton, return_predictions=source_score_writer is not None)
            if source_score_writer is not None:
                source_metrics, source_predictions = source_result
                source_score_writer.add_model(key, client.global_test_id, source_predictions, source_metrics)
                source_cache[key] = source_metrics
            else:
                source_cache[key] = source_result
        if source_score_writer is not None:
            source_score_writer.add_client(key, client.global_test_id, client.id)
        rows.append({"client_id": client.id, "source_id": client.global_test_id,
                     **{"local_" + name: value for name, value in local_metrics.items()},
                     **{"global_" + name: value for name, value in source_cache[key].items()}})
    if not rows:
        raise ValueError("Cannot evaluate an empty client split")
    if capture_local_predictions:
        predictions.sort(key=lambda row: (row["source_id"], row["client_id"],
                                          -row["positive_probability"], row["gene_id"]))
        ranks = {}
        for row in predictions:
            key = (row["source_id"], row["client_id"])
            ranks[key] = ranks.get(key, 0) + 1
            row["rank_within_client"] = ranks[key]
    return (rows, predictions) if capture_local_predictions else rows


def mean_client_metrics(rows, scope):
    """Equal-client averages for metrics present in evaluation rows."""
    prefix = scope + "_"
    metric_names = sorted({key[len(prefix):] for row in rows for key in row
                           if key.startswith(prefix)})
    return {metric: float(np.mean([row[prefix + metric] for row in rows
                                   if row[prefix + metric] is not None]))
            for metric in metric_names
            if any(row[prefix + metric] is not None for row in rows)}


def cotton_evaluation_clients(args, clients, server):
    """Match shared-body evaluation without overwriting training state."""
    if (args["algorithm"] != "FedCCFA" or
            args.get("shared_body_evaluation", "stored") != "latest_global"):
        return clients
    views = []
    for client in clients:
        view = copy(client)
        view.model = deepcopy(client.model)
        view.model.hidden_layers.load_state_dict(server.model.hidden_layers.state_dict())
        views.append(view)
    return views


def baseline_step(server, clients, selected, args, round_index):
    # Same training/aggregation order as methods/FedCCFA.py; no new adapter or gradient rescaling.
    server.send_params(selected)
    descriptors = {}
    for client in selected:
        client.update_label_distribution()
        if args["balanced_epochs"] > 0:
            client.balance_train()
            descriptors[client.id] = deepcopy(client.get_clf_parameters())
        if not args["clustered_protos"]:
            client.global_protos = deepcopy(server.global_protos)
        client.train_with_protos(round_index)
        if args["balanced_epochs"] == 0:
            descriptors[client.id] = deepcopy(client.get_clf_parameters())
    server.aggregate_rep(selected)
    server.aggregate_protos(selected)
    server.send_rep_params(selected)
    merged = server.merge_classifiers(descriptors)
    for label, groups in merged.items():
        for ids in groups:
            group = [c for c in selected if c.id in ids]
            params = server.aggregate_label_params(label, group)
            proto = server.aggregate_label_protos(label, group)
            for client in group:
                rows = [p[label] for name, p in client.model.named_parameters() if name in client.clf_keys]
                torch.nn.utils.vector_to_parameters(params, rows)
                client.set_label_params(label, rows)
                client.global_protos[label] = proto.clone()
    for client in selected:
        client.p_clf_params = deepcopy(client.get_clf_parameters())
    return sum(bool(groups) for groups in merged.values())


def build_parser():
    parser = argparse.ArgumentParser(description="FedCAP research experiments; Cotton remains static")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--algorithm", choices=(*CAP_ALGORITHMS, "FedCCFA", "FedDrift"))
    parser.add_argument("--dataset", choices=("CIFAR10", "CIFAR100", "Fashion-MNIST", "CINIC-10", "cotton30"))
    parser.add_argument("--clients", type=int)
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--drift", choices=("false", "sudden", "incremental", "recurrent"))
    parser.add_argument("--drift-class-count", type=int,
                        help="Optional image scenario: union of affected labels across three groups; multiple of 6, default 6; 60 for CIFAR100 is NOT the paper scenario")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--deterministic", choices=("true", "false"),
                        help="Opt-in strict deterministic kernels; unsupported operations raise errors, never silently downgrade")
    parser.add_argument("--cotton-root", type=Path)
    parser.add_argument("--cotton-grouping", choices=("inferred", "source"),
                        help="Cotton FedCAP: inferred clusters or fixed known-source heads (row sharing must be off)")
    parser.add_argument("--cotton-feature-set", choices=COTTON_FEATURE_SETS,
                        help="Cotton only: all23; no-expression (20 active); sequence-structure (10); sequence-structure-atac/chip (15). Preserve the 23-input network.")
    parser.add_argument("--cotton-test-eval", choices=("true", "false"),
                        help="Cotton: default validation-only; explicitly enable final test after settings are locked")
    parser.add_argument("--cotton-export-source-scores", choices=("true", "false"),
                        help="Opt-in label-free full-source scores at final Cotton test; no extra training")
    parser.add_argument("--cotton-cross-source-eval", choices=("true", "false"),
                        help="Cotton only: additional final TEST-union diagnostic, no training/gating changes; source-specific primary metrics stay unchanged")
    parser.add_argument("--output-root", type=Path, help="Parent directory for log/ (all datasets); default is repository root")
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--profile-weight", type=float,
                        help="Clustering distance weight in [0, 1]: 0=probe delta only, 1=prediction profile only; overrides YAML")
    parser.add_argument("--profile-head", choices=("reference", "none"),
                        help="FedCAP: reference=existing softmax profile (default); none=current-body per-feature 8-bin histograms, no reference-head forward")
    parser.add_argument("--profile-ablation", choices=PROFILE_ABLATIONS,
                        help="FedCAP matched 64-row recipes: sampling, conflict, evidence (80%% coverage, memory=0), memory (same coverage, memory=10). Contradictory explicit switches are rejected")
    parser.add_argument("--profile-samples-per-class", type=int,
                        help="FedCAP only: independent TRAIN rows for sharing/merge evidence; 0=legacy draw. Client-distance profile and SGD stay unchanged")
    parser.add_argument("--numerical-diagnostics", choices=("true", "false"),
                        help="FedCAP clients: check tensors/gradients/SGD state and save recent batch history on failure; diagnostics do not change training")
    parser.add_argument("--numerical-history-size", type=int,
                        help="Completed batches retained per local head/body optimization when diagnostics are on; default 20, 0=last batch only")
    parser.add_argument("--mechanism-diagnostics", choices=("true", "false"),
                        help="Read-only body/head transition and same-batch alternative-head diagnostics; FedCAP only")
    parser.add_argument("--mechanism-start-round", type=int,
                        help="First one-based round for detailed mechanism traces; default 1")
    parser.add_argument("--stage-diagnostics", choices=("true", "false"),
                        help="FedCAP forward-only fixed TRAIN panel before/after local training, body/head aggregation and row sharing; never a gate")
    parser.add_argument("--stage-start-round", type=int,
                        help="First one-based round for TRAIN stage measurements; default 1")
    parser.add_argument("--stage-interval", type=int,
                        help="Measure every N rounds from stage-start-round; default 1")
    parser.add_argument("--stage-samples-per-class", type=int,
                        help="Fixed distinct TRAIN examples per class for stage diagnostics only; default 64; does not change probe/profile budgets")
    parser.add_argument("--body-aggregation-step", type=float,
                        help="FedCAP server BODY only: old + step * (sample-weighted mean - old), in (0,1]; 1=exact old replacement (default), 0.5=half step; local SGD/heads unchanged")
    parser.add_argument("--body-prox-mu", type=float,
                        help="FedCAP formal BODY only: CE + mu/2 * squared parameter distance to this round's downloaded body; 0=old training (default)")
    parser.add_argument("--head-calibration-ratio", type=float,
                        help="FedCAP only: fraction of the EXISTING head SGD budget moved after global body aggregation; 0=old schedule; adds a body download/exchange")
    parser.add_argument("--class-support-alpha", type=float,
                        help="FedCAP formal head aggregation: uniform/count-support blend in [0,1]; 0=original arithmetic, 0.5=equal mixture")
    parser.add_argument("--class-support-count-cap", type=int,
                        help="TRAIN class-count cap for support weights: default 0=CapFull; 64 restores the old cap; independent of the 64-example profile budget")
    parser.add_argument("--class-support-scope", choices=SUPPORT_SCOPES,
                        help="FedCAP support weighting: all=old cluster/cross-cluster rows (default); cross-cluster=uniform whole heads within clusters, support weights only across clusters")
    parser.add_argument("--collaboration-policy", choices=("legacy", "fresh-verified"),
                        help="Revision of existing FedCAP collaboration: legacy=old decisions; fresh-verified=current-draw topology verification, witnessed donors, fixed 0.2 separation; no new model/SGD")
    parser.add_argument("--max-clusters", type=int, help="Compatibility alias: set both candidate and total cluster limits")
    parser.add_argument("--diagnostic-head-sharing", choices=("none", "oracle-class"),
                        help="FedCAP-only diagnostic: oracle-class expands compatible head rows across inferred clusters; NOT a fair benchmark")
    parser.add_argument("--head-sharing", choices=("none", "profile"),
                        help="FedCAP: share compatible class rows across inferred clusters using fresh training profiles")
    parser.add_argument("--class-share-eps", type=float,
                        help="Maximum profile total-variation distance for extra row sharing; default 0.1")
    parser.add_argument("--class-share-min-witnesses", type=int,
                        help="Fresh clients with a valid class profile required per source cluster; default 2")
    parser.add_argument("--class-share-rule", choices=("witness", "centroid"),
                        help="Profile sharing: strict witness distances (default) or cluster centroids with leave-one-out consistency")
    parser.add_argument("--class-share-label-mode", choices=("same", "aligned"),
                        help="same=original same-index row sharing; aligned=opt-in fresh TRAIN profile matching across local label indices (image FedCAP only)")
    parser.add_argument("--label-merge-guard", choices=("none", "cross-label"),
                        help="Opt-in aligned rows: veto raw whole-head merges when fresh row matches require different local label indices; reuse sharing thresholds")
    parser.add_argument("--class-share-witness-policy", choices=("strict", "consensus"),
                        help="Centroid rows: all-witness consistency (old default), or majority LOO + verified donors")
    parser.add_argument("--class-share-consensus-ratio", type=float,
                        help="Required fraction of ORIGINAL valid witnesses, in (0.5,1]; consensus default 0.8")
    parser.add_argument("--class-share-centering", choices=("none", "head-mean"),
                        help="Opt-in centroid rows: subtract donor head means, share, restore each receiver's pre-sharing offset; independent of profile merge protection; none preserves old arithmetic")
    parser.add_argument("--class-share-task-gate", choices=("none", "train-loss"),
                        help="Optional receiver-side TRAIN task check for centroid row sharing")
    parser.add_argument("--class-share-task-tolerance", type=float,
                        help="Absolute allowed macro/shared-class CE increase; default 0.01")
    parser.add_argument("--class-share-rho", type=float,
                        help="Current-round row sharing strength in [0,1]: 0=cluster only, 1=old full sharing (default)")
    parser.add_argument("--cluster-method", choices=("agglomerative", "hdbscan"))
    parser.add_argument("--regroup-confirmations", type=int,
                        help="FedCAP whole-client split/merge evidence rounds: 1=original v3, 2=confirm twice")
    parser.add_argument("--merge-guard", choices=("none", "loss"),
                        help="FedCAP: veto whole-head merges that increase balanced training loss")
    parser.add_argument("--merge-loss-tolerance", type=float,
                        help="Allowed source-cluster relative loss increase; default 0.05 (5%%)")
    parser.add_argument("--merge-min-witnesses", type=int,
                        help="Fresh selected clients required in each merging old source cluster; default 2")
    parser.add_argument("--profile-merge-guard", choices=("none", "conflict", "evidence"),
                        help="FedCAP: old conflict veto or positive-evidence tri-state merge gate with short conflict memory")
    parser.add_argument("--profile-merge-eps", type=float,
                        help="Conflict threshold after subtracting both observed profile radii; default 0.1")
    parser.add_argument("--profile-merge-min-witnesses", type=int,
                        help="Fresh valid clients per class and old source in a merge proposal; default 2")
    parser.add_argument("--profile-merge-min-coverage", type=float,
                        help="evidence only: fraction of all classes requiring reliable positive compatibility; default 1")
    parser.add_argument("--profile-merge-memory-rounds", type=int,
                        help="evidence only: lifetime of remembered source/label conflicts; 0=disable memory, default 10")
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--hdbscan-min-samples", type=int)
    parser.add_argument("--max-candidate-clusters", type=int)
    parser.add_argument("--max-total-clusters", type=int)
    parser.add_argument("--reference-strategy", choices=("average_reset", "largest_cluster", "fixed"))
    parser.add_argument("--probe-budget", type=int)
    parser.add_argument("--probe-training", choices=("fedccfa", "legacy"),
                        help="FedCAP balanced SGD: fedccfa (default) or historical cached-feature fixed steps")
    parser.add_argument("--balanced-epochs", type=int,
                        help="Balanced classifier epochs, same meaning as FedCCFA; default 5")
    parser.add_argument("--balanced-clf-lr", type=float,
                        help="Balanced classifier SGD learning rate; default 0.1")
    parser.add_argument("--probe-steps", type=int, help="Legacy FedCAP probe only: number of SGD updates")
    parser.add_argument("--probe-lr", type=float, help="Legacy probe LR; also accepted as a balanced-clf-lr alias in fedccfa mode")
    parser.add_argument("--rep-epochs", type=int)
    parser.add_argument("--clf-epochs", type=int)
    parser.add_argument("--epochs", type=int, help="FedDrift full-model local epochs")
    parser.add_argument("--lr", type=float, help="FedDrift full-model learning rate")
    parser.add_argument("--detection-threshold", type=float)
    return parser


def resolve_config(options, parser):
    """Resolve command line > supplied YAML > algorithm/dataset defaults."""
    if options.cotton_root is not None:
        options.cotton_root = options.cotton_root.expanduser().absolute().resolve()
    if options.output_root is not None:
        options.output_root = options.output_root.resolve()
    # Read the selected/default file first so edits to its dataset or algorithm
    # are respected even when neither flag was passed on the command line.
    source_name = (options.algorithm or "FedCAP") + ("_cotton" if options.dataset == "cotton30" else "")
    if options.dataset is None and options.algorithm in ("FedCCFA", "FedDrift"):
        source_name = options.algorithm + "_cotton"
    source_path = options.config or ROOT / "configs" / f"{source_name}.yaml"
    with source_path.open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream)
    if not isinstance(supplied, dict):
        parser.error("Config must be a YAML mapping")
    algorithm = options.algorithm or supplied.get("algorithm", "FedCAP")
    if algorithm not in (*CAP_ALGORITHMS, "FedCCFA", "FedDrift"):
        parser.error("Unsupported algorithm in config")
    dataset = options.dataset or supplied.get("dataset", "CIFAR10")
    if dataset not in ("CIFAR10", "CIFAR100", "Fashion-MNIST", "CINIC-10", "cotton", "cotton30"):
        parser.error("Unsupported dataset in config")
    cotton = dataset.lower() in ("cotton", "cotton30")
    if cotton:
        dataset = "cotton30"
    config_path = ROOT / "configs" / (algorithm + ("_cotton" if cotton else "") + ".yaml")
    with config_path.open(encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream)
    args = {**defaults, **supplied, "algorithm": algorithm, "dataset": dataset}
    if args.get("cotton_variant") is not None or any(key.startswith("cotton_lite_") for key in args):
        parser.error("The CottonLite variant has been withdrawn; select a retained FedCAP/FedCCFA configuration")
    # Resolve before main changes cwd. CLI > YAML > loader's checkout fallback.
    if cotton:
        cotton_root = options.cotton_root or args.get("cotton_root")
        if cotton_root:
            p = Path(cotton_root).expanduser()
            # Portable: resolve relative data roots against the project root,
            # independent of the invocation cwd (AutoDL vs local).
            if not p.is_absolute():
                p = ROOT / p
            args["cotton_root"] = str(p.resolve())
        else:
            args["cotton_root"] = None
    args["config_path"] = str(source_path.resolve())
    if cotton:
        args["output_root"] = str(cotton_output_root(
            options.output_root or args.get("output_root"),
            from_cli=options.output_root is not None))
    else:
        args["output_root"] = str(Path(options.output_root or args.get("output_root", ROOT)).resolve())
    mappings = {"clients": "client_num", "drift": "drift_pattern"}
    for key in ("dataset", "clients", "sample_ratio", "alpha", "drift", "drift_class_count", "seed", "rounds", "device",
                "cotton_feature_set", "cotton_grouping",
                "profile_weight", "probe_budget", "probe_steps", "rep_epochs", "clf_epochs",
                "max_candidate_clusters", "max_total_clusters", "reference_strategy", "eval_interval",
                "epochs", "lr", "detection_threshold", "probe_lr", "probe_training", "profile_head",
                "balanced_epochs", "balanced_clf_lr", "cluster_method",
                "min_cluster_size", "hdbscan_min_samples",
                "diagnostic_head_sharing", "regroup_confirmations", "merge_guard",
                "merge_loss_tolerance", "merge_min_witnesses", "head_sharing",
                "class_share_eps", "class_share_min_witnesses", "class_share_rule", "class_share_rho", "class_share_label_mode", "label_merge_guard",
                "class_share_witness_policy", "class_share_consensus_ratio",
                "profile_merge_guard", "profile_merge_eps", "profile_merge_min_witnesses",
                "profile_merge_min_coverage", "profile_merge_memory_rounds", "profile_samples_per_class",
                "numerical_history_size", "body_aggregation_step", "body_prox_mu", "head_calibration_ratio", "class_support_alpha", "class_support_count_cap", "class_support_scope", "collaboration_policy", "class_share_task_gate",
                "class_share_task_tolerance", "mechanism_start_round", "class_share_centering"):
        value = getattr(options, key)
        if value is not None:
            args[mappings.get(key, key)] = False if key == "drift" and value == "false" else value
    if options.deterministic is not None:
        args["deterministic"] = options.deterministic == "true"
    if options.cotton_cross_source_eval is not None:
        args["cotton_cross_source_eval"] = options.cotton_cross_source_eval == "true"
    if not isinstance(args.get("cotton_cross_source_eval", False), bool):
        parser.error("cotton_cross_source_eval must be true or false")
    if not cotton and args.get("cotton_cross_source_eval", False):
        parser.error("--cotton-cross-source-eval is only supported by Cotton")
    if cotton:
        args.setdefault("cotton_cross_source_eval", False)
        args.setdefault("cotton_feature_set", "all23")
        args.setdefault("cotton_test_eval", False)
        args.setdefault("cotton_export_source_scores", False)
        if options.cotton_export_source_scores is not None:
            args["cotton_export_source_scores"] = options.cotton_export_source_scores == "true"
        if not isinstance(args["cotton_export_source_scores"], bool):
            parser.error("cotton_export_source_scores must be true or false")
        if args["cotton_feature_set"] not in COTTON_FEATURE_SETS:
            parser.error(f"cotton_feature_set must be one of {COTTON_FEATURE_SETS}")
        if options.cotton_test_eval is not None:
            args["cotton_test_eval"] = options.cotton_test_eval == "true"
        if not isinstance(args["cotton_test_eval"], bool):
            parser.error("cotton_test_eval must be true or false")
        if args.get("cotton_cross_source_eval", False) and not args["cotton_test_eval"]:
            parser.error("Cotton TEST-union evaluation requires cotton_test_eval: true")
        if args["cotton_export_source_scores"] and not args["cotton_test_eval"]:
            parser.error("Source score export requires cotton_test_eval: true")
    else:
        args.pop("cotton_cross_source_eval", None)
        args.pop("cotton_feature_set", None)
        args.pop("cotton_test_eval", None)
        args.pop("cotton_export_source_scores", None)
        if options.cotton_export_source_scores is not None:
            parser.error("--cotton-export-source-scores is Cotton-only")
        if options.cotton_feature_set is not None or options.cotton_test_eval is not None:
            parser.error("--cotton-feature-set and --cotton-test-eval are Cotton-only")
    # Old saved configs may request full profile dumps. They no longer enable
    # disk output; profile measurement and the in-memory evidence are unchanged.
    args.pop("record_profiles", None)
    args.pop("profiles_file", None)
    if options.numerical_diagnostics is not None:
        args["numerical_diagnostics"] = options.numerical_diagnostics == "true"
    if options.mechanism_diagnostics is not None:
        args["mechanism_diagnostics"] = options.mechanism_diagnostics == "true"
    if not isinstance(args.get("mechanism_diagnostics", False), bool):
        parser.error("mechanism_diagnostics in YAML must be true or false")
    if args.get("mechanism_diagnostics", False):
        if algorithm != "FedCAP":
            parser.error("Mechanism diagnostics are only supported by FedCAP")
        if options.numerical_diagnostics == "false":
            parser.error("Mechanism diagnostics require numerical diagnostics")
        args["numerical_diagnostics"] = True
        args.setdefault("mechanism_start_round", 1)
        start_round = args["mechanism_start_round"]
        if isinstance(start_round, bool) or not isinstance(start_round, int) or start_round < 1:
            parser.error("mechanism_start_round must be a positive integer")
    elif options.mechanism_start_round is not None or "mechanism_start_round" in args:
        parser.error("--mechanism-start-round requires --mechanism-diagnostics true")
    if not isinstance(args.get("numerical_diagnostics", False), bool):
        parser.error("numerical_diagnostics in YAML must be true or false")
    if algorithm not in CAP_ALGORITHMS:
        if args.get("numerical_diagnostics", False):
            parser.error("--numerical-diagnostics is only supported by FedCAP clients")
        args.pop("numerical_diagnostics", None)
        if options.numerical_history_size is not None or "numerical_history_size" in args:
            parser.error("Numerical history options are only supported by FedCAP clients; baseline training is unchanged")
    else:
        args.setdefault("numerical_history_size", 20)
        history_size = args["numerical_history_size"]
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 0:
            parser.error("numerical_history_size must be a nonnegative integer")
    apply_profile_ablation(args, options, parser)
    if algorithm == "FedCAP":
        try:
            args["collaboration_policy"] = RegroupConfirmation.validate_policy(
                args.get("collaboration_policy", "legacy"))
        except ValueError as error:
            parser.error(str(error))
    else:
        if options.collaboration_policy is not None or args.get("collaboration_policy", "legacy") != "legacy":
            parser.error("--collaboration-policy is only supported by FedCAP; baselines are unchanged")
        args.pop("collaboration_policy", None)
    args.setdefault("deterministic", False)
    if not isinstance(args["deterministic"], bool):
        parser.error("deterministic in YAML must be true or false, not a string or number")
    if algorithm in CAP_ALGORITHMS:
        # Existing YAML and old commands remain supported, with an explicit hard
        # total limit as well as the per-measurement candidate limit.
        legacy = supplied.get("max_clusters", defaults.get("max_clusters"))
        if legacy is not None:
            for key in ("max_candidate_clusters", "max_total_clusters"):
                if key not in supplied and getattr(options, key) is None:
                    args[key] = legacy
        if options.max_clusters is not None:
            if options.max_candidate_clusters is not None or options.max_total_clusters is not None:
                parser.error("Do not combine --max-clusters with the two explicit cluster limits")
            args["max_candidate_clusters"] = args["max_total_clusters"] = options.max_clusters
        args.pop("max_clusters", None)
        args.setdefault("max_candidate_clusters", 5)
        args.setdefault("max_total_clusters", 5)
        args.setdefault("reference_strategy", "largest_cluster"
                        if args.get("probe_training", "fedccfa") == "fedccfa" else "average_reset")
        args.setdefault("cluster_method", "agglomerative")
        args.setdefault("hdbscan_min_samples", 2)
        if args["cluster_method"] not in ("agglomerative", "hdbscan"):
            parser.error("Invalid FedCAP cluster_method")
    elif any(getattr(options, key) is not None for key in
             ("cluster_method", "min_cluster_size", "hdbscan_min_samples")):
        parser.error("Clustering options are only supported by FedCAP")
    if cotton:
        if args["drift_pattern"] not in (False, "false"):
            parser.error("Cotton uses static source labels; temporal label swaps are not supported")
        args.update(model="MLP", num_classes=2, partition="family")
    else:
        args.update(model="CNN", num_classes=100 if dataset == "CIFAR100" else 10)
    for key in ("dataset_extension", "dataset_label_space", "train_class_min_samples", "test_class_min_samples",
                "normalization_mean", "normalization_std", "dataset_drift_policy"):
        args.pop(key, None)  # Do not retain CIFAR100 metadata when reusing a saved config on another dataset.
    if dataset == "CIFAR100":
        args.update(dataset_extension="cifar100_fine100_v1", dataset_label_space="fine_labels",
                    **dataset_partition_settings(dataset), normalization_mean=list(CIFAR100_MEAN),
                    normalization_std=list(CIFAR100_STD),
                    dataset_drift_policy="existing_pairs_1_2__3_4__5_6_v1")
    if cotton:
        if options.drift_class_count is not None or args.get("drift_class_count", 6) != 6:
            parser.error("Cotton does not support simulated label swaps")
        args.pop("drift_class_count", None)
        args.pop("drift_label_pairs", None)
    else:
        args.setdefault("drift_class_count", 6)
        try:
            pairs = label_swap_pairs(args["drift_class_count"], args["num_classes"])
        except ValueError as error:
            parser.error(str(error))
        args["drift_label_pairs"] = [[list(pair) for pair in group] for group in pairs]
        if args["drift_class_count"] != 6:
            args["dataset_drift_policy"] = "repeated_disjoint_three_pair_blocks_v1"
    args.setdefault("eval_interval", 20)
    positive_integers = ["client_num", "rounds", "batch_size"]
    if algorithm in CAP_ALGORITHMS:
        positive_integers += ["probe_budget", "reference_refresh", "descriptor_ttl",
                              "max_candidate_clusters", "max_total_clusters", "min_cluster_size", "min_class_samples",
                              "hdbscan_min_samples"]
        args.pop("adapter_steps", None)
        if algorithm in CAP_ALGORITHMS:
            for key in ("adapter_steps", "adapter_lr", "adapter_bottleneck"):
                args.pop(key, None)
            if args.get("probe_init", "reference_head") != "reference_head":
                parser.error("FedCAP requires probe_init: reference_head")
            args["probe_init"] = "reference_head"
            args.setdefault("probe_training", "fedccfa")
            if args["probe_training"] not in ("fedccfa", "legacy"):
                parser.error("FedCAP probe_training must be fedccfa or legacy")
            from entities.FedCAPProfile import profile_metadata
            try:
                args.update(profile_metadata(args))
            except ValueError as error:
                parser.error(str(error))
            args["balanced_probe_policy"] = (CCFA_PROBE_POLICY if args["probe_training"] == "fedccfa"
                                             else LEGACY_PROBE_POLICY)
            args["balanced_probe_init"] = ("server_initial_head" if args["probe_training"] == "fedccfa"
                                           else "reference_head")
            args["implementation"] = FedCAPServer.implementation
            if algorithm in WHOLE_HEAD_ALGORITHMS:
                args.setdefault("head_weights", "uniform")
                if args["head_weights"] not in ("uniform", "samples"):
                    parser.error("FedCAP head_weights must be uniform or samples")
                args.update(body_aggregation="global", head_aggregation="cluster", initialization="fedccfa_order")
                args["cluster_identity_policy"] = FedCAPServer.cluster_identity_policy
            learning_rates = ["rep_lr", "clf_lr"]
            positive_integers.append("probe_steps")
            args.setdefault("probe_lr", 0.1)
            learning_rates.append("probe_lr")
            if args["probe_training"] == "fedccfa":
                args.setdefault("balanced_epochs", 5)
                args.setdefault("balanced_clf_lr", 0.1)
                positive_integers.append("balanced_epochs")
                learning_rates.append("balanced_clf_lr")
                if options.probe_lr is not None and options.balanced_clf_lr is None:
                    args["balanced_clf_lr"] = options.probe_lr
            for key in learning_rates:
                if not isinstance(args[key], (int, float)) or isinstance(args[key], bool) or not np.isfinite(args[key]) or args[key] <= 0:
                    parser.error(f"{key} must be finite and positive")
    if options.probe_lr is not None and algorithm not in CAP_ALGORITHMS:
        parser.error("--probe-lr is only supported by FedCAP")
    if options.probe_training is not None and algorithm not in CAP_ALGORITHMS:
        parser.error("--probe-training is only supported by FedCAP")
    if options.profile_head is not None and algorithm not in CAP_ALGORITHMS:
        parser.error("--profile-head is only supported by FedCAP")
    if algorithm not in CAP_ALGORITHMS:
        args.pop("profile_head", None)
        args.pop("profile_representation", None)
    if algorithm == "FedDrift" and (options.balanced_epochs is not None or options.balanced_clf_lr is not None):
        parser.error("Balanced classifier options are not supported by FedDrift")
    if algorithm in WHOLE_HEAD_ALGORITHMS:
        args.setdefault("regroup_confirmations", 1)
        confirmations = args["regroup_confirmations"]
        if not isinstance(confirmations, int) or isinstance(confirmations, bool) or confirmations < 1:
            parser.error("regroup_confirmations must be a positive integer")
        if algorithm != "FedCAP" and confirmations != 1:
            parser.error("Regroup confirmation is currently only supported by FedCAP")
        if args["reference_strategy"] != "fixed" and confirmations > args["reference_refresh"]:
            parser.error("regroup_confirmations cannot exceed reference_refresh; evidence resets on refresh")
        args["regroup_confirmation_policy"] = (RegroupConfirmation.verified_policy
                                               if args.get("collaboration_policy") == "fresh-verified"
                                               else RegroupConfirmation.policy)
        if confirmations > 1:
            args["implementation"] = "fedcap_shared_body_v3_stable_v1"
    elif options.regroup_confirmations is not None or "regroup_confirmations" in args:
        parser.error("--regroup-confirmations is only supported by whole-head FedCAP")
    merge_keys = ("merge_guard", "merge_loss_tolerance", "merge_min_witnesses")
    if algorithm in WHOLE_HEAD_ALGORITHMS:
        args.setdefault("merge_guard", "none")
        args.setdefault("merge_loss_tolerance", 0.05)
        args.setdefault("merge_min_witnesses", 2)
        try:
            guard = MergeLossGuard(*(args[key] for key in merge_keys))
        except ValueError as error:
            parser.error(str(error))
        args["merge_guard_policy"] = guard.policy
        if guard.mode != "none":
            if algorithm != "FedCAP":
                parser.error("Merge loss guard is currently only supported by FedCAP")
            args["implementation"] = guard.implementation
    elif any(getattr(options, key) is not None for key in merge_keys) or args.get("merge_guard", "none") != "none":
        parser.error("Merge loss guard is only supported by FedCAP")
    for key in positive_integers:
        if not isinstance(args[key], int) or isinstance(args[key], bool) or args[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if dataset == "CIFAR100" and args["client_num"] > 500:
        parser.error("CIFAR100 supports at most 500 clients with one TRAIN sample/class/client")
    if dataset == "CIFAR100" and algorithm in CAP_ALGORITHMS and args["probe_budget"] < 100:
        parser.error("CIFAR100 FedCAP requires probe_budget >= 100 to cover all present fine classes")
    if cotton and (args["client_num"] < 3 or args["client_num"] % 3):
        parser.error("Cotton client_num must be a positive multiple of 3 (e.g. 30); 100 is unsupported")
    if algorithm in CAP_ALGORITHMS and args["cluster_method"] == "hdbscan" and args["min_cluster_size"] < 2:
        parser.error("HDBSCAN min_cluster_size must be at least 2")
    epoch_keys = ["epochs"] if algorithm == "FedDrift" else ["rep_epochs", "clf_epochs"]
    for key in ["seed", "eval_interval", *epoch_keys]:
        if not isinstance(args[key], int) or isinstance(args[key], bool) or args[key] < 0:
            parser.error(f"{key} must be a nonnegative integer")
    if options.device is None and args["device"] == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA unavailable; using CPU", flush=True)
        args["device"] = "cpu"
    count = int(args["client_num"] * args["sample_ratio"])
    if args["rounds"] < 1 or not 0 < args["sample_ratio"] <= 1 or count < 1:
        parser.error("Invalid rounds, sample ratio, or participant count")
    if algorithm == "FedCCFA" and count < 3:
        parser.error("Original FedCCFA MADD requires at least 3 participants")
    if algorithm == "FedCCFA" and args.get("oracle"):
        parser.error("This comparison runner uses inferred clusters, not oracle labels")
    if algorithm in CAP_ALGORITHMS:
        weight = args["profile_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 <= weight <= 1:
            parser.error("profile_weight must be a finite number in [0, 1]")
        args["profile_weight"] = float(weight)
        if not -1 <= args["silhouette_threshold"] <= 1:
            parser.error("Invalid FedCAP silhouette threshold")
        if args["reference_strategy"] not in ("average_reset", "largest_cluster", "fixed"):
            parser.error("Invalid reference strategy")
    elif options.profile_weight is not None:
        parser.error("--profile-weight is only supported by FedCAP variants")
    if algorithm == "FedDrift":
        for key in ("lr", "detection_threshold"):
            if not isinstance(args[key], (int, float)) or isinstance(args[key], bool) or not np.isfinite(args[key]) or args[key] <= 0:
                parser.error(f"{key} must be finite and positive")
        if options.rep_epochs is not None or options.clf_epochs is not None:
            parser.error("FedDrift trains the full model; use --epochs instead")
    # These are derived metadata, not switches. Reusing a saved diagnostic
    # config with --diagnostic-head-sharing none must really disable it.
    for key in ("diagnostic_case", "diagnostic_only", "oracle_information_used"):
        args.pop(key, None)
    sharing = args.get("diagnostic_head_sharing", "none")
    if sharing not in ("none", "oracle-class"):
        parser.error("Invalid diagnostic_head_sharing")
    head_sharing = args.get("head_sharing", "none")
    if head_sharing not in ("none", "profile"):
        parser.error("Invalid head_sharing")
    if head_sharing == "profile":
        if algorithm != "FedCAP":
            parser.error("Profile head sharing is currently only supported by FedCAP")
        if sharing != "none" or args.get("merge_guard", "none") != "none":
            parser.error("Test profile head sharing without oracle sharing or merge loss guard")
        epsilon = args.get("class_share_eps", 0.1)
        witnesses = args.get("class_share_min_witnesses", 2)
        rule = args.get("class_share_rule", "witness")
        rho = args.get("class_share_rho", 1.0)
        if (isinstance(rho, bool) or not isinstance(rho, (int, float)) or
                not np.isfinite(rho) or not 0 <= rho <= 1):
            parser.error("class_share_rho must be finite in [0, 1]")
        if not isinstance(rule, str) or rule not in FedCAPProfileShareServer.implementations:
            parser.error("class_share_rule must be witness or centroid")
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or
                not np.isfinite(epsilon) or not 0 < epsilon < 0.5):
            parser.error("class_share_eps must be finite and strictly between 0 and 0.5")
        if not isinstance(witnesses, int) or isinstance(witnesses, bool) or witnesses < 2:
            parser.error("class_share_min_witnesses must be an integer of at least 2")
        if args.get("collaboration_policy") == "fresh-verified" and (
                rule != "centroid" or args.get("class_share_witness_policy", "strict") != "strict"):
            parser.error("fresh-verified row sharing requires the existing centroid + strict witness rule")
        separation = 0.2 if args.get("collaboration_policy") == "fresh-verified" else 2 * epsilon
        args.update(head_sharing="profile", class_share_eps=float(epsilon), class_share_rho=float(rho),
                    class_share_min_witnesses=witnesses, class_share_min_separation=separation,
                    class_share_rule=rule,
                    implementation=FedCAPProfileShareServer.implementations[rule],
                    head_aggregation=FedCAPProfileShareServer.head_aggregation,
                    profile_share_policy=FedCAPProfileShareServer.policies[rule])
        if rho != 1.0:
            args["implementation"] += "_shrink_v1"
    elif (options.class_share_eps is not None or options.class_share_min_witnesses is not None or
          options.class_share_rule is not None or options.class_share_rho is not None or
          options.class_share_witness_policy is not None or options.class_share_consensus_ratio is not None):
        parser.error("--class-share-* options require --head-sharing profile")
    elif options.head_sharing is not None and algorithm != "FedCAP":
        parser.error("--head-sharing is only supported by FedCAP")
    else:
        # Derived metadata from a saved profile-sharing config must disappear
        # when the optional aggregation is explicitly disabled. Inactive
        # module settings must not make v3/Reentry training configs diverge.
        for key in ("head_sharing", "class_share_eps", "class_share_min_witnesses",
                    "class_share_rule", "class_share_rho", "profile_share_policy", "class_share_min_separation",
                    "class_share_witness_policy", "class_share_consensus_ratio", "profile_donor_policy"):
            args.pop(key, None)
        # Disabling a saved sharing config must also disable its derived
        # centering mode; an explicit non-none CLI request remains an error.
        if options.class_share_centering is None:
            args.pop("class_share_centering", None)
    try:
        task_gate, task_tolerance = validate_task_gate(args)
    except ValueError as error:
        parser.error(str(error))
    if task_gate != "none":
        args.update(class_share_task_gate=task_gate, class_share_task_tolerance=task_tolerance,
                    task_gate_policy=TASK_GATE_POLICY)
        args["implementation"] += "_task_gate_v1"
    else:
        if options.class_share_task_tolerance is not None:
            parser.error("--class-share-task-tolerance requires --class-share-task-gate train-loss")
        for key in ("class_share_task_gate", "class_share_task_tolerance", "task_gate_policy"):
            args.pop(key, None)
    profile_merge_keys = ("profile_merge_guard", "profile_merge_eps", "profile_merge_min_witnesses")
    profile_merge = args.get("profile_merge_guard", "none")
    evidence_keys = ("profile_merge_min_coverage", "profile_merge_memory_rounds")
    if profile_merge != "evidence":
        if any(getattr(options, key) is not None for key in evidence_keys):
            parser.error("--profile-merge-min-coverage/memory-rounds require --profile-merge-guard evidence")
        for key in evidence_keys:
            args.pop(key, None)
    try:
        profile_merge_module = make_profile_merge_guard(args)
    except ValueError as error:
        parser.error(str(error))
    if profile_merge != "none":
        if algorithm != "FedCAP" or sharing != "none" or args.get("merge_guard", "none") != "none":
            parser.error("Profile merge guard requires FedCAP without oracle sharing or merge loss guard")
        args.update(profile_merge_guard=profile_merge, profile_merge_eps=profile_merge_module.epsilon,
                    profile_merge_min_witnesses=profile_merge_module.min_witnesses,
                    profile_merge_policy=profile_merge_module.policy)
        args["implementation"] += profile_merge_module.implementation_suffix
        if profile_merge == "evidence":
            args.update(profile_merge_min_coverage=profile_merge_module.min_coverage,
                        profile_merge_memory_rounds=profile_merge_module.memory_rounds)
    else:
        if options.profile_merge_eps is not None or options.profile_merge_min_witnesses is not None:
            parser.error("--profile-merge-eps/min-witnesses require --profile-merge-guard conflict or evidence")
        if options.profile_merge_guard is not None and algorithm != "FedCAP":
            parser.error("--profile-merge-guard is only supported by FedCAP")
        for key in (*profile_merge_keys, "profile_merge_policy"):
            args.pop(key, None)
    try:
        centering = validate_head_centering(args)
    except ValueError as error:
        parser.error(str(error))
    args.pop("head_centering_policy", None)  # Recompute saved metadata.
    if centering != "none":
        args.update(class_share_centering=centering, head_centering_policy=CENTERING_POLICY)
        args["implementation"] += CENTERING_SUFFIX
    else:
        args.pop("class_share_centering", None)
    try:
        witness_policy, consensus_ratio = validate_consensus(args)
    except ValueError as error:
        parser.error(str(error))
    args.pop("profile_donor_policy", None)  # Saved derived metadata cannot enable sharing.
    if witness_policy == "consensus":
        args.update(class_share_witness_policy=witness_policy, class_share_consensus_ratio=consensus_ratio,
                    profile_share_policy=CONSENSUS_POLICY, profile_donor_policy=CONSENSUS_DONOR_POLICY)
    else:
        if options.class_share_consensus_ratio is not None:
            parser.error("--class-share-consensus-ratio requires --class-share-witness-policy consensus")
        for key in ("class_share_witness_policy", "class_share_consensus_ratio"):
            args.pop(key, None)
    if sharing == "oracle-class":
        if args.get("merge_guard", "none") != "none":
            parser.error("Do not combine merge loss guard with oracle class sharing")
        if args.get("regroup_confirmations", 1) != 1:
            parser.error("Do not combine regroup confirmation with oracle class sharing")
        if algorithm != "FedCAP" or cotton:
            parser.error("Oracle head sharing is diagnostic-only for image FedCAP; Cotton and other algorithms are unsupported")
        if args["drift_pattern"] not in (False, "false", "sudden", "incremental", "recurrent"):
            parser.error("Unsupported oracle diagnostic drift schedule")
        args.update(diagnostic_case="oracle-class-share", diagnostic_only=True,
                    oracle_information_used=True,
                    implementation=FedCAPOracleClassShareServer.implementation,
                    head_aggregation=FedCAPOracleClassShareServer.head_aggregation)
    elif (algorithm == "FedCAP" and args["cluster_method"] == "agglomerative" and
          args["max_candidate_clusters"] == args["max_total_clusters"] == 1):
        args.update(diagnostic_case="single-cluster", diagnostic_only=True,
                    oracle_information_used=False)
    if args.get("diagnostic_case") and options.output_root is None and "output_root" not in supplied:
        name = "FedCAP-OracleClassShare" if sharing == "oracle-class" else "FedCAP-SingleCluster"
        args["output_root"] = str((cotton_output_root() if cotton else ROOT) / "diagnostics" / name)
    if algorithm == "FedCAP":
        args.setdefault("profile_samples_per_class", 0)
        try:
            limit = profile_budget(args)
        except ValueError as error:
            parser.error(str(error))
        if 0 < limit < args["min_class_samples"]:
            parser.error("profile_samples_per_class must be 0 or at least min_class_samples")
        args["profile_sample_policy"] = UNIQUE_POLICY if limit else LEGACY_POLICY
        args["profile_distance_policy"] = LEGACY_POLICY
        if not limit and options.profile_samples_per_class is None:
            # Inactive measurement metadata must not break the existing v3 /
            # Reentry same-training-config invariant or alter old saved configs.
            for key in ("profile_samples_per_class", "profile_sample_policy", "profile_distance_policy"):
                args.pop(key, None)
    else:
        if (options.profile_samples_per_class is not None or args.get("profile_samples_per_class", 0) != 0):
            parser.error("Independent profile sampling is currently only supported by FedCAP")
        for key in ("profile_samples_per_class", "profile_sample_policy", "profile_distance_policy"):
            args.pop(key, None)
    if algorithm == "FedCAP":
        try:
            args["head_calibration_ratio"] = validate_ratio(args.get("head_calibration_ratio", 0))
        except ValueError as error:
            parser.error(str(error))
        enabled = args["head_calibration_ratio"] > 0
        if enabled and args["clf_epochs"] < 1:
            parser.error("Head calibration requires clf_epochs >= 1; it does not add training steps")
        if enabled and (args.get("mechanism_diagnostics", False) or
                        args.get("diagnostic_head_sharing", "none") != "none"):
            parser.error("Head calibration does not support mechanism traces or oracle head sharing")
        args["head_calibration_policy"] = HEAD_CALIBRATION_POLICY if enabled else "none"
        args["head_calibration_communication_policy"] = COMMUNICATION_POLICY if enabled else "none"
    else:
        if options.head_calibration_ratio is not None or args.get("head_calibration_ratio", 0) != 0:
            parser.error("--head-calibration-ratio is only supported by FedCAP; baseline training is unchanged")
        for key in ("head_calibration_ratio", "head_calibration_policy", "head_calibration_communication_policy"):
            args.pop(key, None)
    validate_stages(args, options, parser)
    if algorithm == "FedCAP":
        try:
            args["class_support_alpha"] = validate_class_support(args)
            args["class_support_scope"] = validate_support_scope(args)
        except ValueError as error:
            parser.error(str(error))
        enabled = args["class_support_alpha"] > 0
        args["class_support_count_cap"] = validate_count_cap(args.get("class_support_count_cap", DEFAULT_COUNT_CAP))
        args["class_support_policy"] = support_policy(args["class_support_count_cap"]) if enabled else "none"
        if enabled:
            args["implementation"] += implementation_suffix(args["class_support_count_cap"])
        if args["class_support_scope"] == "cross-cluster":
            args["implementation"] += CROSS_ONLY_SUFFIX
    else:
        if (options.class_support_alpha is not None or options.class_support_count_cap is not None or
                options.class_support_scope is not None or args.get("class_support_scope", "all") != "all" or
                args.get("class_support_alpha", 0) != 0 or args.get("class_support_count_cap", CLASS_SUPPORT_CAP) != CLASS_SUPPORT_CAP):
            parser.error("Class support options are only supported by FedCAP")
        for key in ("class_support_alpha", "class_support_policy", "class_support_count_cap", "class_support_scope"):
            args.pop(key, None)
    if algorithm == "FedCAP":
        if witness_policy == "consensus":
            args["implementation"] += CONSENSUS_SUFFIX
        try:
            args["body_prox_mu"] = validate_mu(args.get("body_prox_mu", 0.0))
        except ValueError as error:
            parser.error(str(error))
        args["body_prox_policy"] = BODY_PROX_POLICY if args["body_prox_mu"] > 0 else "none"
        args["clustering_signal"] = ("balanced_classifier_delta_only" if args["profile_weight"] == 0
                                     else "profile_only" if args["profile_weight"] == 1
                                     else "balanced_classifier_delta_and_profile")
        args["profile_usage"] = ("class_row_sharing_only" if args["profile_weight"] == 0
                                 and args.get("head_sharing") == "profile"
                                 and args.get("profile_merge_guard", "none") == "none"
                                 else "configured_clustering_sharing_and_merge_checks")
    else:
        if options.body_prox_mu is not None or args.get("body_prox_mu", 0) != 0:
            parser.error("--body-prox-mu is only supported by FedCAP; baseline training is unchanged")
        for key in ("body_prox_mu", "body_prox_policy", "clustering_signal", "profile_usage"):
            args.pop(key, None)
    # Recompute on reuse of saved configs; these cannot enable an unsafe cache.
    if algorithm == "FedCAP":
        args.update(measurement_policies(args))
        if args["collaboration_policy"] == "fresh-verified":
            args["implementation"] += "_fresh_verified_v2"
            if args.get("head_sharing") == "profile":
                args["profile_share_policy"] = "fresh_train_centroid_verified_donors_fixed_separation_v2"
                args["profile_donor_policy"] = "fresh_accepted_witnesses_only_v2"
    else:
        for key in ("descriptor_cache_policy", "profile_body_policy"):
            args.pop(key, None)
    # Recompute after all implementation overrides, including saved configs.
    if algorithm == "FedCAP":
        try:
            args["body_aggregation_step"] = validate_body_aggregation_step(args.get("body_aggregation_step", 1.0))
        except ValueError as error:
            parser.error(str(error))
        args["body_aggregation_policy"] = body_aggregation_policy(args["body_aggregation_step"])
        if args["body_aggregation_step"] != 1.0:
            if args.get("diagnostic_head_sharing", "none") != "none":
                parser.error("Body aggregation step ablation does not support oracle head sharing")
            args["implementation"] += BODY_STEP_SUFFIX
    else:
        if options.body_aggregation_step is not None or args.get("body_aggregation_step", 1.0) != 1.0:
            parser.error("--body-aggregation-step is only supported by FedCAP; baselines are unchanged")
        for key in ("body_aggregation_step", "body_aggregation_policy"):
            args.pop(key, None)
    if options.head_sharing == "none" and options.class_share_label_mode is None:
        args.pop("class_share_label_mode", None)
    try:
        label_mode = validate_label_mode(args)
    except ValueError as error:
        parser.error(str(error))
    if algorithm != "FedCAP" and options.class_share_label_mode is not None:
        parser.error("--class-share-label-mode is only supported by FedCAP")
    if label_mode == "aligned":
        args.update(class_share_label_mode=label_mode, label_alignment_policy=LABEL_ALIGNMENT_POLICY,
                    head_aggregation=ALIGNED_HEAD_AGGREGATION)
        args["implementation"] += LABEL_ALIGNMENT_SUFFIX
    else:
        for key in ("class_share_label_mode", "label_alignment_policy"):
            args.pop(key, None)
    if (options.head_sharing == "none" or options.class_share_label_mode == "same") and options.label_merge_guard is None:
        args.pop("label_merge_guard", None)
    try:
        label_guard = validate_label_merge_guard(args)
    except ValueError as error:
        parser.error(str(error))
    if algorithm != "FedCAP" and options.label_merge_guard is not None:
        parser.error("--label-merge-guard is only supported by FedCAP")
    if label_guard != "none":
        args.update(label_merge_guard=label_guard, label_merge_policy=LABEL_MERGE_POLICY)
        args["implementation"] += LABEL_MERGE_SUFFIX
    else:
        for key in ("label_merge_guard", "label_merge_policy"):
            args.pop(key, None)
    grouping = args.get("cotton_grouping", "inferred")
    if grouping not in ("inferred", "source"):
        parser.error("cotton_grouping must be inferred or source")
    if grouping == "source":
        if not cotton or algorithm != "FedCAP":
            parser.error("Fixed-source grouping is only supported by Cotton FedCAP")
        if (args.get("head_sharing", "none") != "none" or
                args.get("collaboration_policy", "legacy") != "legacy" or
                args.get("diagnostic_head_sharing", "none") != "none" or
                any(args.get(k, "none") != "none" for k in
                    ("merge_guard", "profile_merge_guard", "label_merge_guard"))):
            parser.error("Fixed-source control requires head_sharing=none and no merge/oracle guards")
        if args["max_total_clusters"] < 3 or args["max_candidate_clusters"] < 3:
            parser.error("Fixed-source control requires room for all three source heads")
        args["implementation"] += "_known_source_v1"
    if options.cotton_grouping is not None and (not cotton or algorithm != "FedCAP"):
        parser.error("--cotton-grouping is Cotton FedCAP-only")
    if cotton and algorithm == "FedCAP":
        args["cotton_grouping"] = grouping
    args["smoke_test"] = args["rounds"] == 1 or any(args[key] == 0 for key in epoch_keys)
    return args


def feddrift_step(server, clients, selected, args):
    """Preserve FedDrift detection -> merge -> train -> aggregate ordering."""
    from utils.models import get_model
    for client in selected:
        client.clustering(server.global_models)
    detected = sum(c.cluster_identity is None for c in selected)
    if len(server.global_models) > 1:
        # The existing merger deletes model-list entries and renumbers members.
        # Remember identities so inactive clients cannot retain invalid indices.
        old_models = list(server.global_models)
        old_members = {c.id: c.cluster_identity for c in clients}
        server.merge_clusters(selected)
        surviving = {id(model): i for i, model in enumerate(server.global_models)}
        remap = {i: surviving[id(model)] for i, model in enumerate(old_models) if id(model) in surviving}
        for client in selected:
            if old_members[client.id] is not None:
                remap[old_members[client.id]] = client.cluster_identity
        selected_ids = {c.id for c in selected}
        for client in clients:
            if client.id not in selected_ids:
                client.cluster_identity = remap[old_members[client.id]]
    for client in selected:
        if client.cluster_identity is None:
            server.global_models.append(get_model(args["dataset"], args["model"]).to(args["device"]))
            client.cluster_identity = len(server.global_models) - 1
    server.send_params(selected)
    for client in selected:
        client.train()
    server.aggregate_with_clustering(selected)
    server.send_params(clients)
    # Freeze previous labels: a later in-place label swap must not change history.
    for client in clients:
        client.prev_train_set = deepcopy(client.train_set)
    return len({c.cluster_identity for c in clients}), detected


def run_stamp(args):
    """Reserve a unique run name before data loading or model initialization."""
    return reserve_run(output_folder(args))


def synchronize_cotton_initialization(args, clients, server):
    """Match the actual prediction model, including FedDrift's separate cluster copy."""
    if not args.get("cotton_common_initialization", False):
        return
    if args["algorithm"] not in ("FedCCFA", "FedDrift"):
        return
    state = server.model.state_dict()
    if args["algorithm"] == "FedDrift":
        server.global_models[0].load_state_dict(state)
    for client in clients:
        client.model.load_state_dict(state)


def main(required_algorithm=None):
    parser = build_parser()
    options = parser.parse_args()
    if required_algorithm is not None:
        if options.algorithm not in (None, required_algorithm):
            parser.error(f"This entry point only supports {required_algorithm}")
        options.algorithm = required_algorithm
    args = resolve_config(options, parser)
    folder, stamp = output_folder(args), run_stamp(args)
    run_context = dict(round=None, last_completed_round=0, phase="setup", client_id=None,
                       selected=[], completed_train_clients=[], training_completed=False)
    server_state = {}
    with record_failures(folder, stamp, args, run_context,
                         server_provider=lambda: server_state.get("server")):
        folder.mkdir(parents=True, exist_ok=True)
        run_experiment(args, options, folder, stamp, run_context, server_state)


def run_experiment(args, options, folder, stamp, run_context, server_state):
    """One guarded run; opt-in training interventions are recorded explicitly."""
    algorithm = args["algorithm"]
    cotton = args["dataset"].lower() in ("cotton", "cotton30")
    count = int(args["client_num"] * args["sample_ratio"])
    run_context["phase"] = "runtime_setup"
    configure_determinism(args["deterministic"], args["device"])
    provenance = dict(policy=REPRODUCIBILITY_POLICY, deterministic_requested=args["deterministic"],
                      runtime=runtime_fingerprint(args["device"]), source=source_fingerprint(ROOT))
    os.chdir(ROOT / "methods")  # Preserve legacy artifact cwd; image data now uses an absolute root.
    seed_all(args["seed"])
    run_context["phase"] = "data_loading"
    validation = validation_sources = None
    if cotton:
        train, test, global_tests, validation, validation_sources = distribute_cotton(
            args["client_num"], args["seed"], args.get("cotton_root"),
            return_validation=True, feature_set=args["cotton_feature_set"],
            return_source_validation=True,
            include_test=args["cotton_test_eval"])
    else:
        train, test, global_tests = distribute_dataset(args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"])
    provenance["training_data"] = training_fingerprint(train)
    if cotton:
        provenance["validation_data"] = training_fingerprint(validation)
        provenance["manifest"] = train[0].manifest
    seed_all(args["seed"])
    client_evaluation_sets = (test if not cotton or args["cotton_test_eval"]
                              else validation)
    if algorithm in ("FedCCFA", "FedDrift"):
        # TensorBoard's file backend can fail on Windows when event filenames
        # extend the full experiment hierarchy beyond the legacy path limit.
        # Short log IDs are folder-local; TensorBoard needs its own unique path
        # across datasets/settings that can each have a run numbered 001.
        event_stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f") + f"-{os.getpid()}"
        args["tensorboard_dir"] = str(Path(args["output_root"]) / "runs" / algorithm / event_stamp)
    # Exclusive file creation prevents accidental overwrite even when rerunning a seed.
    run_context["phase"] = "save_config"
    with (folder / f"config-{stamp}.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(args, stream, allow_unicode=True)
    if cotton:
        with (folder / f"data-{stamp}.json").open("x", encoding="utf-8") as stream:
            json.dump(train[0].manifest, stream, indent=2)
    print("Config | " + json.dumps(args), flush=True)
    print(f"[output] {folder} | files suffix={stamp}", flush=True)
    if cotton:
        print(f"[cotton-inputs] feature_set={args['cotton_feature_set']} "
              f"active={len(train[0].input_features)}/23; raw TSV unchanged", flush=True)
        print("[cotton-evaluation] validation used for checkpoints/selection; "
              f"test_eval={args['cotton_test_eval']} (test only at final locked evaluation); "
              f"test_union={args['cotton_cross_source_eval']}", flush=True)
    elif args["drift_class_count"] != 6:
        print(f"[scenario-extension] union affected labels=1..{args['drift_class_count']}; "
              f"{args['drift_class_count'] // 3} labels per client group after drift; "
              "new scenario, not the paper's 6-label setting; compare baselines with the SAME flag", flush=True)
    if algorithm in CAP_ALGORITHMS:
        if args["probe_training"] == "fedccfa":
            print(f"[balanced-probe] FedCCFA: current global body (frozen), fixed server initial head; "
                  f"min(min present-class count, 5) samples/class, no replacement; TRAIN transforms, shuffle; "
                  f"epochs={args['balanced_epochs']} lr={args['balanced_clf_lr']:g}; "
                  "profile sampling and formal training unchanged; isolated probe RNG", flush=True)
        else:
            print("[balanced-probe] legacy: cached eval features, fixed SGD steps", flush=True)
        if args["class_support_alpha"] > 0:
            cap = args["class_support_count_cap"]
            count_formula = f"min(full TRAIN class count,{cap})" if cap else "full TRAIN class count (uncapped)"
            aggregation_scope = ("cluster whole heads: uniform; cross-cluster rows"
                                 if args["class_support_scope"] == "cross-cluster" else "formal cluster/cross-cluster rows")
            print(f"[class-support] alpha={args['class_support_alpha']:g}; "
                  f"scope={args['class_support_scope']}; {aggregation_scope}: uniform + sqrt({count_formula}); "
                  "all-zero support falls back to uniform; sample support is not a confidence certificate", flush=True)
        if args.get("class_share_witness_policy") == "consensus":
            print(f"[profile-consensus] ratio={args['class_share_consensus_ratio']:g} of ORIGINAL valid witnesses; "
                  "repeat leave-one-out checks, unchanged TV/separation bounds; only accepted witnesses donate "
                  "cross-cluster rows; formal cluster/body aggregation unchanged; no test/oracle evidence", flush=True)
        if args["probe_training"] == "fedccfa":
            print("[measurement-consistency] current-round probes/profiles on the same global body; "
                  "old descriptors excluded, cluster identities retained; "
                  + ("anchor-head=disabled" if args["profile_head"] == "none" else
                     f"anchor-head={args['reference_strategy']} (TRAIN mass, never test accuracy)"), flush=True)
        print(f"[profile] head={args['profile_head']} representation={args['profile_representation']}; "
              "TRAIN only; head-free histogram TV thresholds are experimental, not equivalent to softmax thresholds", flush=True)
        print(f"[signal-roles] clustering={args['clustering_signal']}; profiles={args['profile_usage']}", flush=True)
        if args["body_aggregation_step"] != 1.0:
            print(f"[body-aggregation-step] step={args['body_aggregation_step']:g}; "
                  "previous global BODY + step * (sample-weighted client mean - previous BODY); "
                  "heads/probes/local SGD unchanged; experimental server update, not a validated gain", flush=True)
        if args.get("collaboration_policy") == "fresh-verified":
            print("[collaboration-revision] fresh-verified: splits/merges require independent current-draw "
                  "clustering plus >=2 fresh clients per affected source/destination; only accepted class "
                  "witnesses donate cross-cluster rows; separation fixed at 0.2; "
                  "probe/body/head SGD and communication unchanged; UNKNOWN topology keeps old heads", flush=True)
        if args["body_prox_mu"] > 0:
            print(f"[body-proximal] mu={args['body_prox_mu']:g}; anchor=this round's downloaded global body; "
                  "CE + mu/2 * squared parameter L2 distance, then existing SGD; "
                  "heads/probes unchanged; train_loss includes the penalty, body_ce/body_prox report it separately", flush=True)
        if args["head_calibration_ratio"] > 0:
            print(f"[head-calibration] ratio={args['head_calibration_ratio']:g}; split existing head SGD steps; "
                  "resume sampler and head momentum; download aggregated body before remaining head steps; "
                  "BODY parameters/buffers frozen, then cluster heads and profile rows; "
                  "extra body download + formal exchange (head upload is deferred, not duplicated)", flush=True)
        print("[numerical-safety] always on: non-finite training inputs/logits, gradients, "
              "parameters and SGD momentum fail fast; detailed history remains a separate diagnostic", flush=True)
    if args.get("numerical_diagnostics", False):
        print("[numerical-checks] enabled: failed client/round/batch, tensors, gradients, SGD momentum; "
              f"recent completed batches={args['numerical_history_size']}; "
              "failed-batch snapshot is diagnostic only, not a resumable checkpoint", flush=True)
    if args.get("class_share_centering", "none") != "none":
        print("[head-centering] head-mean: canonical donor rows + each receiver's pre-sharing offset; "
              "no task gate or extra SGD; applied only when cross-cluster rows are shared", flush=True)
    if args.get("oracle_information_used"):
        print("[diagnostic] ORACLE CLASS SHARING: uses simulated true label mappings, not test accuracy. "
              "Inferred client clusters remain unchanged. NOT a deployable algorithm or a fair benchmark.", flush=True)
    elif args.get("diagnostic_case") == "single-cluster":
        print("[diagnostic] SINGLE CLUSTER: same v3 training, clustering capped at one group.", flush=True)
    run_context["phase"] = "model_initialization"
    if algorithm in CAP_ALGORITHMS:
        server_type, client_type = FedCAPServer, FedCAPClient
        if args.get("diagnostic_head_sharing") == "oracle-class":
            server_type = FedCAPOracleClassShareServer
        elif args.get("head_sharing", "none") == "profile":
            server_type = FedCAPProfileShareServer
        if args.get("cotton_grouping") == "source":
            from entities.FedCAPCotton import FedCAPSourceServer
            server_type = FedCAPSourceServer
        server = server_type(args)
        server_state["server"] = server
        clients = [client_type(i, args, train[i], client_evaluation_sets[i], server.initial)
                   for i in train]
    elif algorithm == "FedCCFA":
        from entities.FedCCFA import FedCCFAClient, FedCCFAServer
        clients = [FedCCFAClient(i, args, train[i], client_evaluation_sets[i],
                                 getattr(train[i], "concept_id", 0)) for i in train]
        server = FedCCFAServer(args)
        server_state["server"] = server
        server.get_client_data_size(clients)
        server.clf_keys = list(server.model.state_dict())[-2:]
        for client in clients:
            client.clf_keys = server.clf_keys
    else:
        from entities.FedDrift import FedDriftClient, FedDriftServer
        clients = [FedDriftClient(i, args, train[i], client_evaluation_sets[i],
                                  getattr(train[i], "concept_id", 0)) for i in train]
        for client in clients:
            client.prev_train_set = deepcopy(client.train_set)
        server = FedDriftServer(args)
        server_state["server"] = server
        server.get_client_data_size(clients)
    if cotton:
        for client in clients:
            client.validation_set = validation[client.id]
        synchronize_cotton_initialization(args, clients, server)
    initial = server.initial if algorithm in CAP_ALGORITHMS else server.model if algorithm == "FedCCFA" else server.global_models[0]
    provenance["initial_model_sha256"] = state_fingerprint(initial.state_dict())
    if cotton:
        provenance["initialization_policy"] = (
            "actual_prediction_model_from_server_v2" if args.get("cotton_common_initialization", False)
            and algorithm in ("FedCCFA", "FedDrift") else
            "shared_server_initial_v1" if algorithm in CAP_ALGORITHMS else "method_native")
    provenance["initial_rng_sha256"] = rng_fingerprint(args["device"])
    provenance_path = folder / f"provenance-{stamp}.json"
    run_context["phase"] = "save_provenance"
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    print(f"[reproducibility] strict={args['deterministic']} "
          f"source={provenance['source']['sha256'][:12]} "
          f"train={provenance['training_data']['sha256'][:12]} "
          f"initial={provenance['initial_model_sha256'][:12]}", flush=True)
    if args.get("class_share_label_mode") == "aligned":
        print("[label-alignment] fresh TRAIN profile row matching; extend same-label groups only; "
              "complete-link, one local row per cluster; mapped donor counts and receiver slots; "
              "no oracle, no relabeling, local training unchanged", flush=True)
    if args.get("label_merge_guard") == "cross-label":
        print("[label-merge-guard] before whole-head averaging: fresh cross-index row matches veto "
              "incompatible source merges; existing Evidence/confirmation still apply; "
              "no match is not a compatibility certificate", flush=True)
    head_transitions_header = " | exit/rejoin | split/merge" if algorithm in WHOLE_HEAD_ALGORITHMS else ""
    reentry_header = ""
    oracle_header = " | oracle_shared_labels" if args.get("oracle_information_used") else ""
    stability_header = " | confirm | ref_refresh | cache_coverage | pair_change" if algorithm in WHOLE_HEAD_ALGORITHMS else ""
    merge_header = " | merge_check | merge_loss_change" if args.get("merge_guard", "none") == "loss" else ""
    profile_header = " | shared_labels | share_groups | share_rho" if args.get("head_sharing", "none") == "profile" else ""
    if args.get("class_share_label_mode") == "aligned":
        profile_header += " | aligned_groups | aligned_rows"
    if args.get("label_merge_guard") == "cross-label":
        profile_header += " | label_merge_blocked/checked"
    consensus_header = " | consensus_recovered/trimmed | verified_donors/excluded" if args.get("class_share_witness_policy") == "consensus" else ""
    task_header = " | task_full/half/reject | task_s" if args.get("class_share_task_gate", "none") != "none" else ""
    profile_merge_header = " | profile_merge_blocked/checked" if args.get("profile_merge_guard", "none") != "none" else ""
    if args.get("profile_merge_guard") == "evidence":
        profile_merge_header += " | merge_unknown | merge_memory"
    quality_header = " | profile_rows | profile_unique | profile_std | profile_extra_s" if args.get("profile_samples_per_class", 0) else ""
    proximal_header = " | body_ce | body_prox | body_drift_l2" if args.get("body_prox_mu", 0) > 0 else ""
    calibration_header = " | head_pre/post | head_cal_ce | head_cal_s" if args.get("head_calibration_ratio", 0) > 0 else ""
    print(f"Round | clusters/merged_labels | candidates | regroup | cached | noise | held_heads{head_transitions_header}{reentry_header}{oracle_header}{stability_header}{merge_header}{profile_header}{consensus_header}{task_header}{profile_merge_header}{quality_header}{proximal_header}{calibration_header} | silhouette | train_loss | local_acc | generalized_acc | validation_macro_source_AP | validation_worst_source_AP | drift | seconds", flush=True)
    trajectory_path = folder / f"rounds-{stamp}.jsonl"
    extra_profile_seconds = 0.0
    proximal_totals = proximal_summary([])
    calibration_records = []
    cotton_gene_predictions = None
    cotton_source_score_info = None
    # Dedicated selector matches CottonBaselines and cannot be advanced by
    # algorithm-specific random sampling or model construction.
    cotton_selector = random.Random(args["seed"]) if cotton else None
    with ExitStack() as streams:
        run_context["phase"] = "open_round_logs"
        trajectory = streams.enter_context(trajectory_path.open("x", encoding="utf-8"))
        mechanism = (streams.enter_context(MechanismDiagnostics(folder, stamp, args))
                     if args.get("mechanism_diagnostics", False) else None)
        stages = (streams.enter_context(StageDiagnostics(folder, stamp, args))
                  if args.get("stage_diagnostics", False) else None)
        for round_index in range(args["rounds"]):
            run_context.update(round=round_index + 1, phase="drift", client_id=None,
                               selected=[], completed_train_clients=[])
            start, event = time.perf_counter(), False
            if args["drift_pattern"] == "sudden" and round_index == 100 or args["drift_pattern"] == "recurrent" and round_index in (100, 150):
                sudden_drift(clients, global_tests, round_index, args["drift_class_count"])
                event = True
            elif args["drift_pattern"] == "incremental" and round_index in (100, 110, 120):
                incremental_drift(clients, global_tests, round_index, args["drift_class_count"])
                event = True
            selected = (cotton_selector.sample(clients, count) if cotton
                        else random.sample(clients, count))
            run_context.update(selected=[c.id for c in selected], phase="regroup")
            record = {"round": round_index + 1, "selected": [c.id for c in selected], "drift": event}
            if args.get("diagnostic_case"):
                record.update(diagnostic_case=args["diagnostic_case"], diagnostic_only=True,
                              oracle_information_used=args["oracle_information_used"])
            if algorithm in CAP_ALGORITHMS:
                if args.get("diagnostic_head_sharing") == "oracle-class":
                    server.set_oracle_label_maps(oracle_label_maps_for_round(
                        (c.id for c in clients), args["num_classes"], args["drift_pattern"], round_index,
                        args["drift_class_count"]))
                score, cached = server.regroup(clients, selected, round_index)
                run_context["phase"] = "sync_before_train"
                if mechanism is not None:
                    mechanism.observe_sync(server, selected, round_index, "before_local_training")
                server.sync(selected)
                if stages is not None:
                    run_context["phase"] = "stage_before_local"
                    stages.begin_round(server, selected, round_index)
                train_losses = []
                for client in selected:
                    run_context.update(phase="local_train", client_id=client.id)
                    train_losses.append(client.train(round_index))
                    run_context["completed_train_clients"].append(client.id)
                    if stages is not None:
                        run_context["phase"] = "stage_after_local"
                        stages.observe_local(server, client)
                loss = float(np.mean(train_losses))
                if proximal_header:
                    prox_record = proximal_summary(c.last_body_prox_stats for c in selected)
                    record["body_proximal"] = dict(mu=args["body_prox_mu"], **prox_record)
                    proximal_totals = proximal_summary([proximal_totals, prox_record])
                run_context.update(phase="aggregate", client_id=None)
                if calibration_header:
                    server._head_calibration_context = run_context
                if stages is None:
                    server.aggregate(selected)
                else:
                    # The profile-sharing server observes the exact cluster-only
                    # result inside aggregate(), before replacing any class row.
                    server._stage_diagnostics = stages if stages.active(round_index) else None
                    try:
                        server.aggregate(selected)
                        if args.get("head_sharing", "none") != "profile":
                            stages.observe_aggregation(server, selected)
                    finally:
                        del server._stage_diagnostics
                    run_context["phase"] = "stage_after_sharing"
                    stage_record = stages.finish_round(server, selected)
                    if stage_record is not None:
                        record["stage_diagnostics"] = stage_record
                if calibration_header:
                    del server._head_calibration_context
                    calibrated = dict(server.last_head_calibration_stats)
                    record["head_calibration"] = calibrated
                    calibration_records.extend(c.last_head_calibration_stats for c in selected)
                    loss = float(np.mean([c.last_calibrated_train_loss for c in selected]))
                run_context["phase"] = "sync_after_train"
                if mechanism is not None:
                    mechanism.observe_sync(server, selected, round_index, "after_server_aggregation")
                server.sync(clients)
                groups = server.cluster_count()
                record.update(clusters=groups, cached=cached, silhouette=score, train_loss=loss,
                              membership=server.membership.copy(),
                              reference_version=server.version)
                record.update(server.last_regroup_stats)
                if cotton:
                    from utils.cotton_collaboration import source_composition
                    record["cotton_source_composition"] = source_composition(server, clients)
                    record["cotton_grouping"] = args.get("cotton_grouping", "inferred")
                if args.get("diagnostic_head_sharing") == "oracle-class":
                    record.update(server.last_oracle_stats)
                if args.get("head_sharing", "none") == "profile":
                    record.update(server.last_profile_stats)
                record["profile_weight"] = args["profile_weight"]
                record["body_aggregation_step"] = args["body_aggregation_step"]
                record["body_aggregation_policy"] = args["body_aggregation_policy"]
                record.update(body_prox_mu=args["body_prox_mu"], body_prox_policy=args["body_prox_policy"],
                              clustering_signal=args["clustering_signal"], profile_usage=args["profile_usage"])
                record.update(head_calibration_ratio=args["head_calibration_ratio"],
                              head_calibration_policy=args["head_calibration_policy"])
                if args["class_support_alpha"] > 0:
                    record["class_support"] = server.last_class_support_stats.copy()
                record["class_support_scope"] = args["class_support_scope"]
                if quality_header:
                    record.update(summarize_profiles(selected, server.cache, round_index, server.version),
                                  profile_sample_policy=args["profile_sample_policy"],
                                  profile_samples_per_class=args["profile_samples_per_class"])
                    extra_profile_seconds += record["profile_extra_forward_seconds"]
            elif algorithm == "FedCCFA":
                run_context["phase"] = "fedccfa_step"
                groups = baseline_step(server, clients, selected, args, round_index)
                score, cached, loss = 0., 0, None
                record["merged_labels"] = groups
            else:
                run_context["phase"] = "feddrift_step"
                groups, detected = feddrift_step(server, clients, selected, args)
                score, cached, loss = 0., 0, None
                record.update(clusters=groups, detected_clients=detected,
                              membership={c.id: int(c.cluster_identity) for c in clients})
            local, generalized = None, None
            metrics = None
            validation_metrics = None
            test_metrics = None
            run_context.update(phase="evaluation", client_id=None)
            if (args["eval_interval"] > 0 and (round_index + 1) % args["eval_interval"] == 0) or round_index == args["rounds"] - 1:
                if cotton:
                    evaluation_clients = cotton_evaluation_clients(args, clients, server)
                    validation_metrics = evaluate_client_split(
                        evaluation_clients, validation, validation_sources, args["device"], True,
                        server.model_key if algorithm in CAP_ALGORITHMS else None)
                    validation_local = mean_client_metrics(validation_metrics, "local")
                    validation_global = mean_client_metrics(validation_metrics, "global")
                    record.update({"validation_local_" + key: value
                                   for key, value in validation_local.items()})
                    record.update({"validation_global_" + key: value
                                   for key, value in validation_global.items()})
                    record.update(validation_generalized_accuracy=validation_global["accuracy"],
                                  evaluation_split="validation")
                    validation_source_report = cotton_source_report(
                        clients, validation_metrics, validation_sources, validation,
                        evaluation_scope="own_source_full_validation")
                    record["validation_sources"] = validation_source_report
                    record.update({"validation_" + key: value for key, value in
                                   cotton_ap_summary(
                                       validation_source_report,
                                       [row["global_average_precision"]
                                        for row in validation_metrics]).items()})
                    local = validation_local["accuracy"]
                    generalized = validation_global["accuracy"]
                    metrics = validation_metrics
                    if round_index == args["rounds"] - 1 and args["cotton_test_eval"]:
                        export = (CottonSourceScoreWriter(folder, stamp, args, global_tests)
                                  if args["cotton_export_source_scores"] else nullcontext(None))
                        with export as score_writer:
                            test_metrics, cotton_gene_predictions = evaluate_client_split(
                                evaluation_clients, test, global_tests, args["device"], True,
                                server.model_key if algorithm in CAP_ALGORITHMS else None,
                                capture_local_predictions=True, source_score_writer=score_writer)
                        if score_writer is not None:
                            cotton_source_score_info = score_writer.summary
                        test_local = mean_client_metrics(test_metrics, "local")
                        test_global = mean_client_metrics(test_metrics, "global")
                        record.update({"test_local_" + key: value for key, value in test_local.items()})
                        record.update({"test_global_" + key: value for key, value in test_global.items()})
                        record.update(test_generalized_accuracy=test_global["accuracy"],
                                      test_evaluated=True)
                        test_source_report = cotton_source_report(
                            clients, test_metrics, global_tests,
                            evaluation_scope="own_source_full_test")
                        record["test_sources"] = test_source_report
                        record.update({"test_" + key: value for key, value in
                                       cotton_ap_summary(
                                           test_source_report,
                                           [row["global_average_precision"]
                                            for row in test_metrics]).items()})
                        metrics = test_metrics
                        print("[final-test] evaluated once after the configured run; "
                              f"generalized_accuracy={test_global['accuracy']:.6f}; "
                              f"global_AP={test_global.get('average_precision', float('nan')):.6f}", flush=True)
                    else:
                        record["test_evaluated"] = False
                else:
                    metrics = evaluate_client_split(
                        clients, {client.id: client.test_set for client in clients}, global_tests,
                        args["device"], False,
                        server.model_key if algorithm in CAP_ALGORITHMS else None)
                    local = float(np.mean([m["local_accuracy"] for m in metrics]))
                    generalized = float(np.mean([m["global_accuracy"] for m in metrics]))
                    record.update(local_accuracy=local, generalized_accuracy=generalized)
                if algorithm in WHOLE_HEAD_ALGORITHMS:
                    # Use the evaluated split for both Cotton and image runs.
                    # Cotton's final split may be test or validation only.
                    head_group_metrics = {}
                    for name, personal in (("cluster", False), ("retained", True)):
                        group_metrics = [m for m in metrics if (m["client_id"] in server.retained_heads) == personal]
                        head_group_metrics[name] = {
                            "clients": len(group_metrics),
                            "local_accuracy": float(np.mean([m["local_accuracy"] for m in group_metrics])) if group_metrics else None,
                            "generalized_accuracy": float(np.mean([m["global_accuracy"] for m in group_metrics])) if group_metrics else None,
                        }
                    record["head_group_metrics"] = head_group_metrics
            if args["deterministic"] and algorithm in WHOLE_HEAD_ALGORITHMS:
                record["model_sha256"] = whole_head_fingerprint(server)
                record["rng_sha256"] = rng_fingerprint(args["device"])
            record["seconds"] = time.perf_counter() - start
            run_context.update(phase="save_round_log", client_id=None)
            trajectory.write(json.dumps(record) + "\n")
            trajectory.flush()
            run_context["last_completed_round"] = round_index + 1
            if round_index == args["rounds"] - 1:
                run_context["training_completed"] = True
            fmt = lambda value: "-" if value is None else f"{value:.4f}"
            candidates = record.get("candidate_clusters", "-")
            regroup = record.get("regroup_status", ("accept" if record["regroup_accepted"] else "defer-cap") if "regroup_accepted" in record else "-")
            if record.get("oracle_information_used"):
                regroup += "+oracle-rows"
            head_transitions = (f" | {record['detached_heads']}/{record['rejoined_heads']} | {record['split_events']}/{record['merge_events']}"
                                if algorithm in WHOLE_HEAD_ALGORITHMS else "")
            reentry = ""
            oracle_output = f" | {record['oracle_shared_labels']}/{args['num_classes']}" if args.get("oracle_information_used") else ""
            stability_output = (f" | {record['confirmation_count']}/{record['confirmation_required']}"
                                f" | {int(record['reference_refreshed'])} | {record['cache_coverage']:.2f}"
                                f" | {record['proposal_pair_disagreement']:.3f}"
                                if algorithm in WHOLE_HEAD_ALGORITHMS else "")
            merge_output = (f" | {record['merge_guard_reason']} | {fmt(record['merge_guard_max_relative_loss_increase'])}"
                            if merge_header else "")
            profile_output = (f" | {record['profile_shared_labels']}/{args['num_classes']}"
                              f" | {record['profile_cross_cluster_row_groups']} | {record['class_share_rho']:g}" if profile_header else "")
            if args.get("class_share_label_mode") == "aligned":
                profile_output += f" | {record['aligned_groups']} | {record['aligned_receiver_rows']}"
            if args.get("label_merge_guard") == "cross-label":
                profile_output += f" | {record['label_merge_blocked_candidates']}/{record['label_merge_checked_candidates']}"
            consensus_output = (f" | {record['consensus_recovered_rows']}/{record['consensus_trimmed_witnesses']}"
                                f" | {record['consensus_verified_donors']}/{record['consensus_excluded_donors']}"
                                if consensus_header else "")
            profile_merge_output = (f" | {record['profile_merge_blocked_candidates']}/{record['profile_merge_checked_candidates']}"
                                    if profile_merge_header else "")
            if args.get("profile_merge_guard") == "evidence":
                profile_merge_output += (f" | {record['profile_merge_unknown_source_pairs']}"
                                         f" | {record['profile_merge_memory_pairs']}")
            quality_output = (f" | {record['profile_valid_rows']}/{record['profile_total_rows']}"
                              f" | {fmt(record['profile_mean_unique_samples'])} | {fmt(record['profile_mean_row_std'])}"
                              f" | {record['profile_extra_forward_seconds']:.2f}" if quality_header else "")
            proximal_output = (f" | {prox_record['body_ce_mean']:.4f} | {prox_record['body_prox_mean']:.6f}"
                               f" | {prox_record['body_distance_l2_mean']:.4f}" if proximal_header else "")
            calibration_output = (f" | {calibrated['pre_steps']}/{calibrated['post_steps']}"
                                  f" | {calibrated['post_ce_mean']:.4f} | {calibrated['seconds']:.2f}"
                                  if calibration_header else "")
            task_output = (f" | {record['task_gate_full']}/{record['task_gate_half']}/{record['task_gate_rejected']}"
                           f" | {record['task_gate_seconds']:.2f}" if task_header else "")
            validation_macro_ap = record.get("validation_macro_source_average_precision")
            validation_worst_ap = record.get("validation_worst_source_average_precision")
            print(f"{round_index+1:03d}/{args['rounds']} | {groups} | {candidates} | {regroup} | {cached} | {record.get('noise_clients', '-')} | {record.get('retained_heads', '-')}{head_transitions}{reentry}{oracle_output}{stability_output}{merge_output}{profile_output}{consensus_output}{task_output}{profile_merge_output}{quality_output}{proximal_output}{calibration_output} | {score:.3f} | {fmt(loss)} | {fmt(local)} | {fmt(generalized)} | {fmt(validation_macro_ap)} | {fmt(validation_worst_ap)} | {'yes' if event else '-'} | {record['seconds']:.1f}", flush=True)
    run_context.update(training_completed=True, phase="save_results", client_id=None)
    cross_source = None
    if cotton and args.get("cotton_cross_source_eval", False):
        if not args["cotton_test_eval"]:
            raise RuntimeError("Cross-source TEST evaluation cannot run in validation-only mode")
        run_context["phase"] = "cotton_final_cross_source_evaluation"
        cross_source, cross_rows = cotton_union_report(
            evaluation_clients, global_tests, evaluate, args["device"], args["seed"],
            server.model_key if algorithm in CAP_ALGORITHMS else None)
        by_id = {row["client_id"]: row for row in cross_rows}
        for row in metrics:
            row.update(by_id[row["client_id"]])
        print("[cotton-cross-source] " + json.dumps(cross_source), flush=True)
        run_context["phase"] = "save_results"
    if cotton:
        for row in metrics:
            row["evaluation_split"] = "test" if args["cotton_test_eval"] else "validation"
    with (folder / f"results-{stamp}.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    cotton_gene_predictions_path = None
    if cotton_gene_predictions is not None:
        cotton_gene_predictions_path = folder / f"cotton-gene-scores-{stamp}.csv"
        with cotton_gene_predictions_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cotton_gene_predictions[0]))
            writer.writeheader()
            writer.writerows(cotton_gene_predictions)
    if cotton:
        evaluation_local = mean_client_metrics(metrics, "local")
        evaluation_global = mean_client_metrics(metrics, "global")
        validation_local = mean_client_metrics(validation_metrics, "local")
        validation_global = mean_client_metrics(validation_metrics, "global")
        summary = {"smoke_test": args["smoke_test"],
                   "cotton_test_evaluated": bool(args["cotton_test_eval"]),
                   "evaluation_split": "test" if args["cotton_test_eval"] else "validation"}
        for key, value in validation_local.items():
            summary["validation_local_" + key] = value
        for key, value in validation_global.items():
            summary["validation_global_" + key] = value
        summary["validation_generalized_accuracy"] = validation_global["accuracy"]
        if args["cotton_test_eval"]:
            summary.update(local_accuracy=evaluation_local["accuracy"],
                           generalized_accuracy=evaluation_global["accuracy"],
                           local_client_std=float(np.std([m["local_accuracy"] for m in metrics])),
                           generalized_client_std=float(np.std([m["global_accuracy"] for m in metrics])))
            for key, value in evaluation_local.items():
                summary["local_" + key] = value
            for key, value in evaluation_global.items():
                summary["global_" + key] = value
        else:
            summary.update(validation_local_accuracy=evaluation_local["accuracy"],
                           validation_generalized_accuracy=evaluation_global["accuracy"],
                           validation_local_client_std=float(np.std([m["local_accuracy"] for m in metrics])),
                           validation_generalized_client_std=float(np.std([m["global_accuracy"] for m in metrics])))
    else:
        summary = {"local_accuracy": local, "generalized_accuracy": generalized,
                   "smoke_test": args["smoke_test"],
                   "local_client_std": float(np.std([m["local_accuracy"] for m in metrics])),
                   "generalized_client_std": float(np.std([m["global_accuracy"] for m in metrics]))}
    summary.update(deterministic=args["deterministic"], reproducibility_policy=REPRODUCIBILITY_POLICY,
                   provenance_file=provenance_path.name, source_sha256=provenance["source"]["sha256"],
                   training_data_sha256=provenance["training_data"]["sha256"],
                   initial_model_sha256=provenance["initial_model_sha256"])
    if not cotton:
        summary.update(drift_class_count=args["drift_class_count"], drift_label_pairs=args["drift_label_pairs"],
                       drift_pattern=args["drift_pattern"],
                       scenario_extension=args["drift_class_count"] != 6,
                       affected_labels_union=args["drift_class_count"] if args["drift_pattern"] not in (False, "false") else 0)
    if algorithm in CAP_ALGORITHMS:
        summary["numerical_diagnostics"] = args.get("numerical_diagnostics", False)
        if mechanism is not None:
            summary.update(mechanism.summary())
        if stages is not None:
            summary.update(stages.summary())
        summary["numerical_history_size"] = args["numerical_history_size"]
    if algorithm in CAP_ALGORITHMS:
        summary["profile_weight"] = args["profile_weight"]
        summary["collaboration_policy"] = args.get("collaboration_policy", "legacy")
    if algorithm == "FedCAP":
        summary.update(body_prox_mu=args["body_prox_mu"], body_prox_policy=args["body_prox_policy"],
                       clustering_signal=args["clustering_signal"], profile_usage=args["profile_usage"])
        summary.update(head_calibration_ratio=args["head_calibration_ratio"],
                       head_calibration_policy=args["head_calibration_policy"],
                       head_calibration_communication_policy=args["head_calibration_communication_policy"])
        summary.update(class_support_alpha=args["class_support_alpha"],
                       class_support_scope=args["class_support_scope"],
                       class_support_policy=args["class_support_policy"])
        if args["class_support_alpha"] > 0:
            summary["class_support_count_cap"] = args["class_support_count_cap"]
            summary["class_support_last_round"] = server.last_class_support_stats.copy()
        if calibration_header:
            summary["head_calibration"] = calibration_summary(calibration_records)
            summary["head_calibration"]["additional_formal_exchanges"] = args["rounds"]
        if proximal_header:
            summary["body_proximal"] = dict(mu=args["body_prox_mu"], **proximal_totals)
        summary.update(profile_head=args["profile_head"], profile_representation=args["profile_representation"],
                       profile_samples_per_class=args.get("profile_samples_per_class", 0),
                       profile_sample_policy=args.get("profile_sample_policy", LEGACY_POLICY),
                       profile_distance_policy=args.get("profile_distance_policy", LEGACY_POLICY),
                       profile_extra_forward_seconds_total=extra_profile_seconds)
    if algorithm in WHOLE_HEAD_ALGORITHMS:
        if args.get("profile_ablation"):
            summary.update(profile_ablation=args["profile_ablation"],
                           profile_ablation_policy=args["profile_ablation_policy"])
        summary.update(probe_type="balanced_classifier", probe_init=args["probe_init"],
                       probe_training=args["probe_training"],
                       balanced_probe_policy=args["balanced_probe_policy"],
                       balanced_probe_init=args["balanced_probe_init"],
                       balanced_epochs=args.get("balanced_epochs"),
                       balanced_clf_lr=args.get("balanced_clf_lr"),
                       implementation=args["implementation"], cluster_method=args["cluster_method"],
                       noise_policy="retain_personal_head", body_aggregation="global", head_aggregation="cluster",
                       head_weights=server.head_weights, initialization="fedccfa_order",
                       retained_heads=len(server.retained_heads),
                       cluster_identity_policy=server.cluster_identity_policy,
                       head_group_metrics=head_group_metrics)
        summary.update(server.regroup_summary())
        if args.get("head_sharing", "none") == "profile":
            summary.update(server.profile_summary())
    if cotton:
        evaluation_sources = global_tests if args["cotton_test_eval"] else validation_sources
        if algorithm == "FedCAP":
            from utils.cotton_collaboration import source_composition
            summary["cotton_grouping"] = args.get("cotton_grouping", "inferred")
            summary["cotton_source_composition"] = source_composition(server, clients)
        summary["cotton_feature_set"] = args["cotton_feature_set"]
        if cotton_source_score_info is not None:
            summary.update(cotton_source_score_info)
        summary_evaluation_scope = ("own_source_full_test" if args["cotton_test_eval"]
                                    else "own_source_full_validation")
        source_report = cotton_source_report(
            clients, metrics, evaluation_sources,
            None if args["cotton_test_eval"] else validation,
            evaluation_scope=summary_evaluation_scope)
        summary.update(cotton_evaluation_scope=summary_evaluation_scope,
                       cotton_sources=source_report,
                       cotton_cross_source_eval=(args["cotton_cross_source_eval"]
                                                 if args["cotton_test_eval"] else False))
        ap_prefix = "" if args["cotton_test_eval"] else "validation_"
        summary.update({ap_prefix + key: value for key, value in
                        cotton_ap_summary(
                            source_report,
                            [row["global_average_precision"] for row in metrics]).items()})
        if cotton_gene_predictions_path is not None:
            summary.update(cotton_gene_score_file=cotton_gene_predictions_path.name,
                           cotton_gene_score_rows=len(cotton_gene_predictions),
                           cotton_gene_score_labels_included=False)
        if cross_source is not None:
            summary["cotton_cross_source"] = cross_source
    if args.get("diagnostic_case"):
        summary.update(diagnostic_case=args["diagnostic_case"], diagnostic_only=True,
                       oracle_information_used=args["oracle_information_used"])
    if args.get("diagnostic_head_sharing") == "oracle-class":
        summary.update(server.diagnostic_summary())
    run_context["phase"] = "close_writer"
    if algorithm in ("FedCCFA", "FedDrift"):
        server.writer.close()
    # Publish completion only after training, results and log writers succeed.
    # No model weights or full profile matrices are saved by this runner.
    run_context["phase"] = "save_summary"
    summary.update(status="completed", training_completed=True, run_completed=True,
                   algorithm=args["algorithm"], dataset=args["dataset"], seed=args["seed"],
                   client_num=args["client_num"], sample_ratio=args["sample_ratio"],
                   rounds=args["rounds"])
    summary_json = json.dumps(summary, indent=2, allow_nan=False)
    with (folder / f"summary-{stamp}.json").open("x", encoding="utf-8") as stream:
        stream.write(summary_json)
    run_context["phase"] = "completed"
    print("[finished] " + json.dumps(summary), flush=True)
