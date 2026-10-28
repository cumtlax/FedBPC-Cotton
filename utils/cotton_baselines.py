"""Cotton adapters: original algorithms, common data and explicit evaluation.

No source identity enters training or clustering. It selects the proper TEST
scope only. Evaluations use disposable views and preserve all training RNGs.
"""
import argparse
import csv
from contextlib import nullcontext
import importlib
import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import ConcatDataset, DataLoader

from entities.FedCAPProfile import isolated_profile_rng
from utils.cotton_dataset import distribute_cotton, FEATURE_SETS as COTTON_FEATURE_SETS
from utils.cotton_predictions import CottonSourceScoreWriter, gene_prediction_rows
from utils.cotton_paths import cotton_output_root
from utils.cotton_collaboration import preserve_model_modes
from utils.models import get_model
from utils.fedcap_dataset_reports import (METRICS, cotton_ap_summary,
                                         cotton_source_report, cotton_union_report)
from utils.fedcap_numerics import record_failures
from utils.fedcap_output import output_folder, reserve_run
from utils.fedcap_reproducibility import (configure_determinism, runtime_fingerprint,
                                        source_fingerprint, state_fingerprint,
                                        training_fingerprint)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    "FedAvg": ("base", "Client", "Server"),
    "FedAvgWeighted": ("CottonWeightedFedAvg", "WeightedFedAvgClient", "Server"),
    "FedProx": ("FedProx", "FedProxClient", "FedProxServer"),
    "SCAFFOLD": ("SCAFFOLD", "SCAFFOLDClient", "ScaffoldServer"),
    "FedFM": ("FedFM", "FedFMClient", "FedFMServer"),
    "AdaptiveFedAvg": ("AdaptiveFedAvg", "AdaptiveFedAvgClient", "AdaptiveFedAvgServer"),
    "Flash": ("Flash", "FlashClient", "FlashServer"),
    "pFedMe": ("pFedMe", "pFedMeClient", "pFedMeServer"),
    "Ditto": ("Ditto", "DittoClient", "DittoServer"),
    "FedRep": ("FedRep", "FedRepClient", "FedRepServer"),
    "FedBABU": ("FedBABU", "FedBABUClient", "FedBABUServer"),
    "FedPAC": ("FedPAC", "FedPACClient", "FedPACServer"),
    "IFCA": ("IFCA", "IFCAClient", "IFCAServer"),
}
GLOBAL_METHODS = {"FedAvg", "FedProx", "SCAFFOLD", "FedFM", "AdaptiveFedAvg", "Flash"}
REFERENCE_BASELINES = {"LocalOnly", "Centralized"}
GLOBAL_METHODS.add("Centralized")
GLOBAL_METHODS.add("FedAvgWeighted")


class _NullWriter:
    def close(self):
        pass


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false")


def parser():
    p = argparse.ArgumentParser(description="Original paper baselines on static Cotton")
    p.add_argument("--algorithm", choices=sorted(set(REGISTRY) | REFERENCE_BASELINES), required=True)
    p.add_argument("--config", help="Shared Cotton overrides, not the original image YAML")
    p.add_argument("--dataset", choices=("cotton", "cotton30"), default="cotton30")
    p.add_argument("--drift", choices=("false",), default="false")
    for flag, kind in (("clients", int), ("sample-ratio", float), ("alpha", float),
                       ("seed", int), ("rounds", int), ("eval-interval", int),
                       ("epochs", int), ("rep-epochs", int), ("clf-epochs", int),
                       ("batch-size", int), ("lr", float), ("rep-lr", float),
                       ("clf-lr", float), ("cluster-num", int),
                       ("momentum", float), ("weight-decay", float)):
        p.add_argument("--" + flag, type=kind)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--cotton-root")
    p.add_argument("--cotton-feature-set", choices=COTTON_FEATURE_SETS)
    p.add_argument("--cotton-test-eval", type=boolean)
    p.add_argument("--cotton-export-source-scores", type=boolean)
    p.add_argument("--output-root")
    p.add_argument("--deterministic", type=boolean)
    p.add_argument("--cotton-cross-source-eval", type=boolean)
    p.add_argument("--shared-body-evaluation", choices=("latest_global", "stored"),
                   help="FedRep/FedPAC/FedBABU evaluation copies only; default latest_global; stored restores the old evaluation protocol")
    p.add_argument("--show-config", type=boolean, default=False)
    return p


def resolve_config(options):
    reference = options.algorithm in REFERENCE_BASELINES
    native_algorithm = "FedAvg" if options.algorithm == "FedAvgWeighted" else options.algorithm
    original = None if reference else ROOT / "configs" / (native_algorithm + ".yaml")
    shared = Path(options.config) if options.config else ROOT / "configs/CottonPaper_Baselines.yaml"
    if original is None:
        args = {}
    else:
        with original.open(encoding="utf-8") as stream:
            args = yaml.safe_load(stream)
    with shared.open(encoding="utf-8") as stream:
        overrides = yaml.safe_load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("Cotton config must be a mapping")
    if overrides.get("cotton_variant") is not None:
        raise ValueError("cotton_variant is reserved for the FedCAP runner, not Cotton baseline methods")
    if not reference and overrides.get("algorithm", options.algorithm) != options.algorithm:
        raise ValueError("Config algorithm differs from --algorithm")
    optimizer_keys = ("lr", "rep_lr", "clf_lr", "momentum", "weight_decay")
    native_optimizer = {key: args[key] for key in optimizer_keys if key in args}
    optimizer_policy = overrides.get("optimizer_policy", "shared-overrides")
    if optimizer_policy not in ("shared-overrides", "algorithm-defaults"):
        raise ValueError("optimizer_policy must be shared-overrides or algorithm-defaults")
    args.update(overrides)
    if optimizer_policy == "algorithm-defaults":
        args.update(native_optimizer)
    args["optimizer_policy"] = optimizer_policy
    args["native_optimizer_defaults"] = native_optimizer
    args["algorithm"] = options.algorithm
    mapping = {"clients": "client_num", "drift": "drift_pattern"}
    for key, value in vars(options).items():
        if value is not None and key not in ("config", "show_config"):
            args[mapping.get(key, key)] = value
    cotton_root = args.get("cotton_root")
    if cotton_root:
        data_path = Path(cotton_root).expanduser()
        if not data_path.is_absolute():
            data_path = (Path.cwd() if options.cotton_root is not None else ROOT) / data_path
        args["cotton_root"] = str(data_path.resolve())
    else:
        args["cotton_root"] = None
    args["output_root"] = str(cotton_output_root(
        args.get("output_root"), from_cli=options.output_root is not None))
    if str(args["drift_pattern"]).lower() != "false":
        raise ValueError("Cotton uses a static family split; temporal drift is unsupported")
    args.update(dataset="cotton30", model="MLP", num_classes=2, drift_pattern=False,
                partition="family", output_root=str(Path(args.get("output_root", ROOT)).resolve()),
                config_path=str(shared.resolve()),
                algorithm_config_path=(str(original.resolve()) if original is not None else None),
                implementation=("cotton_reference_baselines_v1" if reference
                                else "cotton_paper_baselines_v2"),
                evaluation_scope="own_source_full_test", alpha_used_for_partition=False,
                evaluation_rng_policy="isolated_disposable_models_v1",
                smoke_test=args["rounds"] < 20)
    args.setdefault("cotton_feature_set", "all23")
    args.setdefault("cotton_test_eval", False)
    args["training_loss_policy"] = ("local_train_inverse_frequency_ce_v1"
                                    if options.algorithm == "FedAvgWeighted" else "method_native")
    args.setdefault("cotton_export_source_scores", False)
    if not isinstance(args["cotton_export_source_scores"], bool):
        raise ValueError("cotton_export_source_scores must be true or false")
    if args["cotton_export_source_scores"] and not args["cotton_test_eval"]:
        raise ValueError("Source score export requires cotton_test_eval=true")
    args.setdefault("momentum", 0.9)
    args.setdefault("weight_decay", 1e-5)
    if not np.isfinite(args["momentum"]) or not 0 <= args["momentum"] < 1:
        raise ValueError("momentum must be finite and in [0, 1)")
    if not np.isfinite(args["weight_decay"]) or args["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if options.algorithm == "SCAFFOLD" and args["momentum"] != 0:
        raise ValueError(
            "Cotton SCAFFOLD uses the standard SGD control-variate update and requires "
            "momentum=0; use its native optimizer defaults or --momentum 0")
    if args["cotton_feature_set"] not in COTTON_FEATURE_SETS:
        raise ValueError(f"cotton_feature_set must be one of {COTTON_FEATURE_SETS}")
    if not isinstance(args["cotton_test_eval"], bool):
        raise ValueError("cotton_test_eval must be true or false")
    if args.get("cotton_cross_source_eval", False) and not args["cotton_test_eval"]:
        raise ValueError("Cotton TEST-union evaluation requires cotton_test_eval=true")
    if args["client_num"] < 3 or args["client_num"] % 3:
        raise ValueError("Cotton clients must be a positive multiple of three")
    if not 0 < args["sample_ratio"] <= 1 or int(args["client_num"] * args["sample_ratio"]) < 1:
        raise ValueError("sample_ratio must select at least one client")
    for key in ("rounds", "batch_size", "epochs", "rep_epochs", "clf_epochs", "eval_interval"):
        if not isinstance(args[key], int) or args[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("lr", "rep_lr", "clf_lr"):
        if not np.isfinite(args[key]) or args[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    if args["seed"] < 0 or args.get("cluster_num", 1) < 1:
        raise ValueError("Invalid seed or cluster_num")
    body_evaluation = args.get("shared_body_evaluation", "latest_global")
    if body_evaluation not in ("latest_global", "stored"):
        raise ValueError("shared_body_evaluation must be latest_global or stored")
    args["shared_body_evaluation"] = body_evaluation
    args["evaluation_body_policy"] = (
        body_evaluation if options.algorithm in ("FedRep", "FedPAC", "FedBABU") else "method_native")
    args["evaluation_model"] = (
        "pooled_centralized" if options.algorithm == "Centralized" else
        "independent_local" if options.algorithm == "LocalOnly" else
        "global" if options.algorithm in GLOBAL_METHODS else
        "cluster" if options.algorithm == "IFCA" else
        "personal_p_model" if options.algorithm == "Ditto" else
        "personal_inner_theta" if options.algorithm == "pFedMe" else
        "disposable_" + body_evaluation + "_full_model_finetune" if options.algorithm == "FedBABU" else
        body_evaluation + "_body_personal_head" if options.algorithm in ("FedRep", "FedPAC") else
        "stored_client_model")
    if options.algorithm == "pFedMe":
        args["personalized_evaluation_policy"] = "last_inner_theta_outer_w_upload_v1"
    if options.algorithm == "FedPAC":
        args["head_solver_policy"] = "float64_gram_psd_simplex_local_fallback_v2"
    if options.algorithm == "Flash":
        args["early_stopping_scope"] = "family_isolated_validation_not_test"
    if options.algorithm == "AdaptiveFedAvg":
        args["adaptive_client_policy"] = "existing_AdaptiveFedAvgClient_not_legacy_base_Client"
    if options.algorithm == "FedBABU":
        args["evaluation_finetune_epochs"] = args["epochs"]
    if options.algorithm == "FedFM":
        args["alignment_start_round"] = 21
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(args, train, test, validation=None):
    module_name, client_name, server_name = REGISTRY[args["algorithm"]]
    module = importlib.import_module("entities." + module_name)
    clients = [getattr(module, client_name)(i, args, train[i], test[i], train[i].concept_id)
               for i in range(args["client_num"])]
    server = getattr(module, server_name)(args)
    server.get_client_data_size(clients)
    if args.get("cotton_common_initialization", False):
        # The server is constructed after N clients, as in FedCCFA/FedCAP.
        # Native random multi-cluster IFCA initialization stays diverse.
        if args["algorithm"] == "IFCA":
            server.global_models[0].load_state_dict(server.model.state_dict())
        for client in clients:
            client.model.load_state_dict(server.model.state_dict())
            if hasattr(client, "p_model"):
                client.p_model.load_state_dict(server.model.state_dict())
    if args["algorithm"] in ("FedRep", "FedBABU", "FedPAC"):
        keys = list(server.model.state_dict())[-2:]
        server.clf_keys = keys
        for client in clients:
            client.clf_keys = keys
    if args["algorithm"] == "Flash":
        if validation is None:
            server.writer.close()
            raise ValueError("Flash requires the independent Cotton validation partition")
    if validation is not None:
        for client in clients:
            client.validation_set = validation[client.id]
    return clients, server


def build_reference(args, train, test):
    """Initialize matched LocalOnly/Centralized reference models."""
    if args.get("cotton_common_initialization", False):
        for _ in range(args["client_num"]):
            get_model(args["dataset"], args["model"])
    initial = get_model(args["dataset"], args["model"]).to(args["device"])
    clients = [SimpleNamespace(id=i, args=args, train_set=train[i], test_set=test[i],
                               global_test_id=train[i].concept_id, model=deepcopy(initial))
               for i in range(args["client_num"])]
    server = SimpleNamespace(model=deepcopy(initial), writer=_NullWriter())
    return clients, server


def reference_step(args, clients, server, selected, round_index):
    """Run compute-matched centralized or independent-local SGD for one round."""
    epochs = int(args.get("reference_local_epochs", args.get("rep_epochs", args.get("epochs", 5)) +
                          args.get("clf_epochs", 0)))
    if epochs < 1:
        raise ValueError("reference_local_epochs must be positive")
    lr = float(args.get("lr", args.get("rep_lr", 0.01)))
    optimizer_kwargs = dict(lr=lr, momentum=float(args.get("momentum", 0.9)),
                            weight_decay=float(args.get("weight_decay", 1e-5)))
    if args["algorithm"] == "Centralized":
        dataset = ConcatDataset([client.train_set for client in selected])
        model = server.model
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
        client_id = 0
    else:
        losses = []
        for client in selected:
            optimizer = torch.optim.SGD(client.model.parameters(), **optimizer_kwargs)
            losses.extend(_train_reference_model(
                client.model, client.train_set, optimizer,
                args, epochs, args["seed"] + round_index * 100003 + client.id))
        return float(np.mean(losses)) if losses else 0.0
    losses = _train_reference_model(model, dataset, optimizer, args, epochs,
                                    args["seed"] + round_index * 100003 + client_id)
    return float(np.mean(losses)) if losses else 0.0


def _train_reference_model(model, dataset, optimizer, args, epochs, seed):
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True,
                        generator=generator)
    losses = []
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(args["device"])
            targets = targets.to(args["device"], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reference-baseline loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("Non-finite reference-baseline gradient")
            optimizer.step()
            assert_finite_model(model)
            losses.append(float(loss.detach().cpu()))
    return losses


def assert_finite_model(model):
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite Cotton model tensor: " + name)


def step(args, server, selected, round_index, context):
    """Exactly the operation order of the original per-method entry points."""
    method = args["algorithm"]
    if method in ("FedRep", "FedPAC"):
        server.send_rep_params(selected)
    elif method != "IFCA":
        server.send_params(selected)
    variances, means = [], []
    for client in selected:
        context.update(phase="local_training", client_id=client.id)
        if method == "SCAFFOLD":
            client.train(server.global_c)
        elif method == "FedFM":
            client.update_label_distribution()
            client.train_with_anchors(server.global_anchors)
        elif method == "FedPAC":
            client.update_label_distribution()
            v, h = client.statistics_extraction()
            variances.append(deepcopy(v))
            means.append(deepcopy(h))
            client.train_with_protos(server.global_protos)
        else:
            if method == "IFCA":
                client.clustering(server.global_models)
            elif method == "pFedMe":
                client.local_params = deepcopy(list(server.model.parameters()))
            client.train()
            if method == "Ditto":
                client.personalized_train(server.model)
                assert_finite_model(client.p_model)
            elif method == "pFedMe":
                if any(not torch.isfinite(value).all() for value in client.personalized_state.values()):
                    raise FloatingPointError("Non-finite pFedMe personalized theta")
        assert_finite_model(client.model)
    context.update(phase="aggregation", client_id=None)
    if method in ("SCAFFOLD", "Flash", "Ditto"):
        server.aggregate_by_updates(selected)
        if method == "SCAFFOLD":
            server.update_global_c(selected)
    elif method in ("FedRep", "FedBABU", "FedPAC"):
        server.aggregate_rep(selected)
        if method == "FedPAC":
            server.aggregate_protos(selected)
    elif method == "IFCA":
        server.aggregate_with_clustering(selected)
    else:
        server.aggregate_by_params(selected)
        if method == "FedFM":
            server.aggregate_anchors(selected)
    if method in ("FedRep", "FedPAC"):
        server.send_rep_params(selected)
    elif method not in ("Ditto", "pFedMe"):
        server.send_params(selected)
    if method == "AdaptiveFedAvg":
        lr = float(server.cal_adaptive_lr(round_index))
        if not np.isfinite(lr) or lr <= 0:
            raise FloatingPointError("Invalid AdaptiveFedAvg learning rate")
        for client in selected:
            client.lr = lr
    if method == "FedPAC":
        weights = server.get_head_agg_weight(variances, means, len(selected))
        heads = [[p for name, p in c.model.named_parameters() if name in server.clf_keys]
                 for c in selected]
        # Independent snapshots avoid earlier receiver writes changing later donors.
        vectors = [parameters_to_vector(head).detach().clone() for head in heads]
        for index, head in enumerate(heads):
            if weights[index] is None:
                continue  # documented local-classifier fallback of FedPAC
            if not np.isfinite(weights[index]).all():
                raise FloatingPointError("Invalid FedPAC QP weights")
            vector_to_parameters(server.aggregate_clf(vectors, weights[index]), head)
    assert_finite_model(server.model)
    for model in getattr(server, "global_models", []):
        assert_finite_model(model)
    for client in selected:
        assert_finite_model(client.model)


@preserve_model_modes
def evaluate(model, dataset, device, cotton=True, return_predictions=False):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for inputs, target in DataLoader(
                dataset, batch_size=1024, shuffle=False,
                generator=torch.Generator().manual_seed(0)):
            if not torch.isfinite(inputs).all():
                raise FloatingPointError("Non-finite evaluation inputs")
            logits = model(inputs.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            probabilities.append(logits.softmax(1).cpu().numpy())
            labels.append(target.numpy())
    if not labels:
        raise ValueError("Empty Cotton evaluation set")
    y, p = np.concatenate(labels), np.concatenate(probabilities)
    predicted = p.argmax(1)
    result = dict(accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                auroc=float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p[:, 1])) if np.any(y == 1) else None)
    return (result, gene_prediction_rows(dataset, p[:, 1])) if return_predictions else result


def evaluation_views(args, clients, server):
    """Never synchronize/fine-tune persistent training models for evaluation."""
    views = []
    common = deepcopy(server.model) if args["algorithm"] in GLOBAL_METHODS else None
    clusters = {}
    for client in clients:
        if common is not None:
            model = common
        elif args["algorithm"] == "IFCA":
            key = client.cluster_identity
            if key not in clusters:
                clusters[key] = deepcopy(server.global_models[key])
            model = clusters[key]
        elif args["algorithm"] == "Ditto":
            model = deepcopy(client.p_model)
        elif args["algorithm"] == "pFedMe":
            model = client.personalized_model()
        elif args["algorithm"] == "FedBABU":
            # Preserve the repository's whole-model fine-tuning budget, but
            # discard it at intermediate evaluations to avoid extra training.
            shadow = SimpleNamespace(**vars(client))
            shadow.model = deepcopy(server.model if args.get("shared_body_evaluation", "latest_global") == "latest_global"
                                    else client.model)
            type(client).fine_tune(shadow)
            model = shadow.model
        else:
            model = deepcopy(client.model)
            if (args["algorithm"] in ("FedRep", "FedPAC") and
                    args.get("shared_body_evaluation", "latest_global") == "latest_global"):
                # Keep the personal classifier, replace only shared body/buffers.
                state = model.state_dict()
                for name, value in server.model.state_dict().items():
                    if name not in server.clf_keys:
                        state[name] = value.detach().clone()
                model.load_state_dict(state)
        assert_finite_model(model)
        views.append(SimpleNamespace(id=client.id, train_set=client.train_set,
                                     test_set=client.test_set, global_test_id=client.global_test_id,
                                     model=model))
    return views


def report(args, clients, server, global_tests, round_number, final=False,
           local_datasets=None, evaluation_split="test", source_score_writer=None):
    with isolated_profile_rng(args["seed"] + 811 * round_number, args["device"]):
        views = evaluation_views(args, clients, server)
        rows, cache = [], {}
        for client in views:
            local_dataset = (local_datasets[client.id] if local_datasets is not None
                             else client.test_set)
            local = evaluate(client.model, local_dataset, args["device"])
            key = (id(client.model), client.global_test_id)
            if key not in cache:
                full_result = evaluate(client.model, global_tests[client.global_test_id], args["device"],
                                       return_predictions=source_score_writer is not None)
                if source_score_writer is not None:
                    full_metrics, predictions = full_result
                    source_score_writer.add_model(key, client.global_test_id, predictions, full_metrics)
                    cache[key] = full_metrics
                else:
                    cache[key] = full_result
            if source_score_writer is not None:
                source_score_writer.add_client(key, client.global_test_id, client.id)
            global_metrics = cache[key]
            rows.append(dict(client_id=client.id, source_id=client.global_test_id,
                             **{"local_" + k: v for k, v in local.items()},
                             **{"global_" + k: v for k, v in global_metrics.items()}))
        source_metrics = cotton_source_report(
            views, rows, global_tests, local_datasets,
            evaluation_scope=f"own_source_full_{evaluation_split}")
        result = dict(evaluation_model=args["evaluation_model"],
                      evaluation_body_policy=args["evaluation_body_policy"],
                      cotton_sources=source_metrics,
                      **cotton_ap_summary(
                          source_metrics, [row["global_average_precision"] for row in rows]))
        for scope in ("local", "global"):
            for metric in METRICS:
                values = [row[scope + "_" + metric] for row in rows if row[scope + "_" + metric] is not None]
                result[scope + "_" + metric] = float(np.mean(values)) if values else None
            result[scope + "_client_std"] = float(np.std([row[scope + "_accuracy"] for row in rows]))
        result["generalized_accuracy"] = result["global_accuracy"]
        result["generalized_client_std"] = result["global_client_std"]
        result["cotton_evaluation_scope"] = f"own_source_full_{evaluation_split}"
        if final and evaluation_split == "test" and args.get("cotton_cross_source_eval", True):
            union, cross_rows = cotton_union_report(
                views, global_tests, evaluate, args["device"], args["seed"],
                model_key=lambda client_id: id(views[client_id].model))
            result["cotton_cross_source"] = union
            for row, cross in zip(rows, cross_rows):
                row.update({key: value for key, value in cross.items() if key != "client_id"})
    return result, rows


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(args):
    folder = output_folder(args)
    stamp = reserve_run(folder)
    # Short independent TensorBoard path avoids Windows nested-path limits.
    args["tensorboard_dir"] = str(Path(args["output_root"]) / "runs" / "cotton-baselines" /
                                   args["algorithm"] / ("c%d-s%d-%s-%s" % (
                                       args["client_num"], args["seed"], stamp,
                                       hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12])))
    with (folder / ("config-" + stamp + ".yaml")).open("x", encoding="utf-8") as stream:
        yaml.safe_dump(args, stream, allow_unicode=True, sort_keys=False)
    print("Config | " + json.dumps(args, ensure_ascii=False), flush=True)
    print("[output] %s | files suffix=%s" % (folder, stamp), flush=True)
    context = dict(phase="setup", last_completed_round=0, training_completed=False)
    server = None
    started = time.perf_counter()
    try:
        with record_failures(folder, stamp, args, context, server_provider=lambda: server):
            # Check method-specific dependencies before loading data. The CLI
            # additionally preloads FedPAC's native solver before PyTorch.
            context["phase"] = "dependency_import"
            fedpac_runtime = None
            if args["algorithm"] == "FedPAC":
                from utils.fedpac_dependencies import check_fedpac_runtime
                fedpac_runtime = check_fedpac_runtime()
            if args["algorithm"] not in REFERENCE_BASELINES:
                importlib.import_module("entities." + REGISTRY[args["algorithm"]][0])
            configure_determinism(args["deterministic"], args["device"])
            if args["device"] == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; pass --device cpu for a smoke test")
            seed_all(args["seed"])
            context["phase"] = "data_loading"
            train, test, global_tests, validation, validation_sources = distribute_cotton(
                args["client_num"], args["seed"], args.get("cotton_root"),
                return_validation=True, feature_set=args["cotton_feature_set"],
                return_source_validation=True,
                include_test=args["cotton_test_eval"])
            source_files = ["methods/CottonBaselines.py", 
                            "utils/cotton_predictions.py",
                            "utils/cotton_baselines.py", "utils/cotton_dataset.py", "utils/cotton_paths.py",
                            "utils/cotton_collaboration.py",
                            "utils/fedcap_reproducibility.py", "utils/fedcap_numerics.py",
                            "entities/FedCAPProfile.py", "utils/metric.py",
                            "utils/fedcap_dataset_reports.py",
                            "utils/fedcap_output.py", "utils/models.py", "entities/base.py",
                            "utils/cotton_predictions.py"]
            # Keep compatibility with the actual shared/algorithm YAML paths.
            source_files = [path for path in source_files if (ROOT / path).is_file()]
            for config_key in ("config_path", "algorithm_config_path"):
                config_path = args.get(config_key)
                if config_path:
                    try:
                        source_files.append(Path(config_path).resolve().relative_to(ROOT).as_posix())
                    except ValueError:
                        raise ValueError(f"Config must be inside the project for source fingerprinting: {config_path}")
            if args["algorithm"] not in REFERENCE_BASELINES:
                source_files.append("entities/" + REGISTRY[args["algorithm"]][0] + ".py")
            if args["algorithm"] == "FedPAC":
                source_files.append("utils/fedpac_dependencies.py")
            provenance = dict(runtime=runtime_fingerprint(args["device"]),
                              training_data=training_fingerprint(train), manifest=train[0].manifest,
                              validation_data=training_fingerprint(validation),
                              source=source_fingerprint(ROOT, files=sorted(set(source_files))))
            seed_all(args["seed"])
            context["phase"] = "initialization"
            if fedpac_runtime is not None:
                provenance["fedpac_solver_runtime"] = fedpac_runtime
            client_evaluation_sets = test if args["cotton_test_eval"] else validation
            if args["algorithm"] in REFERENCE_BASELINES:
                clients, server = build_reference(args, train, client_evaluation_sets)
            else:
                clients, server = build(args, train, client_evaluation_sets, validation)
            provenance["initial_model_sha256"] = state_fingerprint(server.model.state_dict())
            if args["algorithm"] == "FedAvgWeighted":
                provenance["local_train_class_weights"] = {
                    str(c.id): c.criterion.weight.detach().cpu().tolist() for c in clients}
            if args["algorithm"] == "IFCA":
                provenance["initial_cluster_models"] = [state_fingerprint(m.state_dict()) for m in server.global_models]
            write_json(folder / ("data-" + stamp + ".json"), provenance)
            selector = random.Random(args["seed"])
            selected_count = int(args["client_num"] * args["sample_ratio"])
            align_weight = args.get("lambda")
            print("Round | selected | validation_local_acc | validation_generalized_acc | validation_macro_source_AP | validation_worst_source_AP | seconds", flush=True)
            validation_metrics = None
            validation_rows = None
            metrics = None
            rows = None
            source_score_info = None
            with (folder / ("rounds-" + stamp + ".jsonl")).open("x", encoding="utf-8") as stream:
                for index in range(args["rounds"]):
                    tick = time.perf_counter()
                    selected = selector.sample(clients, selected_count)
                    context.update(round=index + 1, selected_client_ids=[c.id for c in selected])
                    reference_train_loss = None
                    if args["algorithm"] == "FedFM":
                        args["lambda"] = 0 if index < 20 else align_weight
                    try:
                        if args["algorithm"] in REFERENCE_BASELINES:
                            reference_train_loss = reference_step(args, clients, server, selected, index)
                        else:
                            step(args, server, selected, index, context)
                    finally:
                        if args["algorithm"] == "FedFM":
                            args["lambda"] = align_weight
                    context["last_completed_round"] = index + 1
                    final = index + 1 == args["rounds"]
                    entry = dict(round=index + 1, selected_client_ids=[c.id for c in selected],
                                 drift=False, train_loss=None)
                    if reference_train_loss is not None:
                        entry["reference_train_loss"] = reference_train_loss
                    if args["algorithm"] == "IFCA":
                        entry["membership"] = {c.id: int(c.cluster_identity) for c in clients}
                    if args["algorithm"] == "FedPAC":
                        entry["head_solver"] = server.last_head_solver_stats
                        entry["head_solver_fallbacks"] = sum(row["fallback"] for row in server.last_head_solver_stats)
                    # Original entities do not return a loss; do not invent it.
                    if final or (index + 1) % args["eval_interval"] == 0:
                        context["phase"] = "evaluation"
                        validation_metrics, validation_rows = report(
                            args, clients, server, validation_sources, index + 1,
                            final=False, local_datasets=validation,
                            evaluation_split="validation")
                        entry.update({"validation_" + key: value
                                      for key, value in validation_metrics.items()})
                        entry["evaluation_split"] = "validation"
                        if final and args["cotton_test_eval"]:
                            export = (CottonSourceScoreWriter(folder, stamp, args, global_tests)
                                      if args["cotton_export_source_scores"] else nullcontext(None))
                            with export as score_writer:
                                metrics, rows = report(
                                    args, clients, server, global_tests, index + 1,
                                    final=True, local_datasets=test, evaluation_split="test",
                                    source_score_writer=score_writer)
                            if score_writer is not None:
                                source_score_info = score_writer.summary
                            entry.update({"test_" + key: value for key, value in metrics.items()})
                            entry["evaluation_split"] = "validation+test"
                            entry["test_evaluated"] = True
                        else:
                            metrics, rows = validation_metrics, validation_rows
                            entry["test_evaluated"] = False
                    entry["elapsed_s"] = time.perf_counter() - tick
                    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    acc = "%.4f" % validation_metrics["local_accuracy"] if validation_metrics else "-"
                    gen = "%.4f" % validation_metrics["global_accuracy"] if validation_metrics else "-"
                    macro_ap = ("%.4f" % validation_metrics["macro_source_average_precision"]
                                if validation_metrics and validation_metrics["macro_source_average_precision"] is not None
                                else "-")
                    worst_ap = ("%.4f" % validation_metrics["worst_source_average_precision"]
                                if validation_metrics and validation_metrics["worst_source_average_precision"] is not None
                                else "-")
                    print("%03d/%d | %d | %s | %s | %s | %s | %.1f" %
                          (index + 1, args["rounds"], selected_count, acc, gen, macro_ap,
                           worst_ap, entry["elapsed_s"]), flush=True)
            context.update(phase="saving", training_completed=True)
            if metrics is None or rows is None or validation_metrics is None:
                raise RuntimeError("Final validation/evaluation did not run")
            for row in rows:
                row["evaluation_split"] = "test" if args["cotton_test_eval"] else "validation"
            with (folder / ("results-" + stamp + ".csv")).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if args["cotton_test_eval"]:
                summary = dict(metrics)
                summary["validation_metrics"] = validation_metrics
            else:
                summary = {"validation_" + key: value for key, value in validation_metrics.items()}
            summary.update(cotton_test_evaluated=bool(args["cotton_test_eval"]),
                           evaluation_split="test" if args["cotton_test_eval"] else "validation",
                           status="completed", run_completed=True, training_completed=True,
                           completed_rounds=args["rounds"], algorithm=args["algorithm"], dataset="cotton30",
                           seed=args["seed"], client_num=args["client_num"], sample_ratio=args["sample_ratio"],
                           drift_pattern=False, smoke_test=args["smoke_test"], config=args.copy(),
                           elapsed_s=time.perf_counter() - started,
                           cotton_cross_source_eval=args["cotton_cross_source_eval"],
                           deterministic=args["deterministic"], provenance_file="data-" + stamp + ".json",
                           source_sha256=provenance["source"]["sha256"],
                           training_data_sha256=provenance["training_data"]["sha256"],
                           initial_model_sha256=provenance["initial_model_sha256"])
            context["phase"] = "close_writer"
            summary["cotton_feature_set"] = args["cotton_feature_set"]
            if source_score_info is not None:
                summary.update(source_score_info)
            server.writer.close()
            context["phase"] = "save_summary"
            write_json(folder / ("summary-" + stamp + ".json"), summary)
            print("[completed] " + str(folder / ("summary-" + stamp + ".json")), flush=True)
            return summary
    finally:
        if server is not None:
            server.writer.close()


def main(argv=None):
    p = parser()
    options = p.parse_args(argv)
    try:
        args = resolve_config(options)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    if options.show_config:
        print(json.dumps(args, ensure_ascii=False, indent=2))
        return
    run(args)
