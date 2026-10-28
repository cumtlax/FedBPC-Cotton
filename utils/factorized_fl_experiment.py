"""Factorized-FL baseline on shared datasets, drift schedules and metrics."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from entities.FactorizedFL import (
    FactorizedFLClient, FactorizedFLServer, UPSTREAM_COMMIT, factorize_model)
from entities.FedDAA import isolated_rng
from utils.cotton_dataset import distribute_cotton
from utils.drift import sudden_drift, incremental_drift, label_swap_pairs
from utils.fedcap_experiment import evaluate, mean_client_metrics
from utils.fedcap_numerics import record_failures
from utils.fedcap_output import output_folder, reserve_run
from utils.fedcap_reproducibility import (
    configure_determinism, runtime_fingerprint, source_fingerprint, state_fingerprint, training_fingerprint)
from utils.gen_dataset import distribute_dataset
from utils.models import get_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("entities/FactorizedFL.py", "utils/factorized_fl_experiment.py", "methods/FactorizedFL.py",
                "configs/FactorizedFL.yaml", "utils/gen_dataset.py", "utils/models.py", "utils/drift.py",
                "utils/cotton_dataset.py", "utils/fedcap_experiment.py", "utils/fedcap_numerics.py",
                "utils/fedcap_output.py", "utils/fedcap_reproducibility.py", "entities/FedDAA.py")


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def build_parser():
    parser = argparse.ArgumentParser(description="Factorized-FL unified-protocol baseline")
    parser.add_argument("--algorithm", choices=("FactorizedFL",), default="FactorizedFL")
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=("Fashion-MNIST", "CIFAR10", "CIFAR100", "CINIC-10", "cotton30", "cotton"))
    parser.add_argument("--drift", choices=("false", "sudden", "incremental", "recurrent"))
    for flag, kind in (("clients", int), ("sample-ratio", float), ("alpha", float), ("seed", int),
                       ("rounds", int), ("epochs", int), ("batch-size", int), ("lr", float),
                       ("momentum", float), ("weight-decay", float), ("eval-interval", int),
                       ("drift-class-count", int), ("factorized-rank", int), ("factorized-l1", float),
                       ("factorized-tau", float), ("factorized-similarity-scale", float),
                       ("factorized-min-neighbors", int)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--factorized-variant", choices=("alpha", "beta"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--deterministic", type=boolean)
    parser.add_argument("--cotton-root")
    parser.add_argument("--cotton-feature-set", choices=("all23", "no-expression"))
    parser.add_argument("--cotton-test-eval", type=boolean)
    parser.add_argument("--output-root")
    return parser


def resolve_config(options, parser):
    source = Path(options.config) if options.config else ROOT / "configs/FactorizedFL.yaml"
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        parser.error(f"Config is not a file: {source}")
    with source.open(encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    if not isinstance(args, dict):
        parser.error("Config must be a mapping")
    allowed = {action.dest for action in parser._actions} | {"client_num", "drift_pattern", "partition", "model"}
    if set(args) - allowed:
        parser.error(f"Unknown Factorized-FL config fields: {sorted(set(args) - allowed)}")
    if args.get("algorithm", "FactorizedFL") != "FactorizedFL":
        parser.error("This config does not specify FactorizedFL")
    for name, value in vars(options).items():
        if value is not None and name not in ("config", "algorithm"):
            args[{"clients": "client_num", "drift": "drift_pattern"}.get(name, name)] = value
    args["algorithm"] = "FactorizedFL"
    args.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    args.setdefault("drift_class_count", 6)
    args["output_root"] = str(Path(args.get("output_root", ROOT)).resolve())
    args["config_path"] = str(source.resolve())
    cotton = args.get("dataset", "").lower() in ("cotton", "cotton30")
    if cotton:
        args.update(dataset="cotton30", model="MLP", partition="family")
    if args.get("dataset") not in ("Fashion-MNIST", "CIFAR10", "CIFAR100", "CINIC-10", "cotton30"):
        parser.error("Invalid dataset")
    args["num_classes"] = 2 if cotton else 100 if args["dataset"] == "CIFAR100" else 10
    for name in ("client_num", "rounds", "epochs", "batch_size", "eval_interval", "factorized_rank", "factorized_min_neighbors"):
        value = args.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            parser.error(f"{name} must be a positive integer")
    if isinstance(args.get("seed"), bool) or not isinstance(args.get("seed"), int) or args["seed"] < 0:
        parser.error("seed must be a nonnegative integer")
    for name in ("lr", "alpha", "sample_ratio", "momentum", "weight_decay", "factorized_l1",
                 "factorized_tau", "factorized_similarity_scale"):
        if not isinstance(args.get(name), (int, float)) or not np.isfinite(args[name]):
            parser.error(f"{name} must be finite")
    if (not 0 < args["sample_ratio"] <= 1 or int(args["client_num"] * args["sample_ratio"]) < 1
            or args["lr"] <= 0 or args["alpha"] <= 0):
        parser.error("Invalid sampling ratio, learning rate or Dirichlet alpha")
    if (not 0 <= args["momentum"] < 1 or args["weight_decay"] < 0 or args["factorized_l1"] < 0
            or not -1 <= args["factorized_tau"] <= 1 or args["factorized_similarity_scale"] < 0):
        parser.error("Invalid optimizer or Factorized-FL matching settings")
    if args.get("factorized_variant") not in ("alpha", "beta"):
        parser.error("factorized_variant must be alpha or beta")
    if args["device"] not in ("cpu", "cuda") or args["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("Requested device is unavailable")
    for name in ("deterministic", "cotton_test_eval"):
        if not isinstance(args.get(name), bool):
            parser.error(f"{name} must be true or false")
    if args.get("cotton_feature_set") not in ("all23", "no-expression"):
        parser.error("Invalid cotton_feature_set")
    if not cotton and (args.get("model") != "CNN" or args.get("partition") not in ("homo", "hetero")):
        parser.error("Image baseline requires CNN and homo/hetero partition")
    if args.get("drift_pattern") not in (False, "false", "sudden", "incremental", "recurrent"):
        parser.error("Invalid drift pattern")
    args["drift_pattern"] = False if args["drift_pattern"] == "false" else args["drift_pattern"]
    if cotton and (args["client_num"] < 3 or args["client_num"] % 3 or args["drift_pattern"] is not False):
        parser.error("Cotton requires a multiple of 3 clients and drift=false")
    if not cotton:
        try:
            args["drift_label_pairs"] = label_swap_pairs(args["drift_class_count"], args["num_classes"])
        except ValueError as error:
            parser.error(str(error))
    args.update(implementation="factorized_fl_unified_protocol_v1", upstream_commit=UPSTREAM_COMMIT,
                original_paper_protocol=False, head_policy="persistent_private_no_reset",
                matching_policy="last_body_v_cosine_cached_uploads",
                aggregation_snapshot_policy="immutable_pre_round_donors",
                l1_policy="differentiable_parameter_l1", evaluation_model="method_native_last_local",
                gradient_clipping="none", backbone_policy="project_topology_factorized_layers",
                smoke_test=args["rounds"] < (100 if cotton else 200))
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_split(clients, local_sets, sources, args):
    rows = []
    with isolated_rng(args["device"]):
        for client in clients:
            model = client.model.to(args["device"])
            try:
                local = evaluate(model, local_sets[client.id], args["device"], args["dataset"] == "cotton30")
                generalized = evaluate(model, sources[client.global_test_id], args["device"], args["dataset"] == "cotton30")
                rows.append(dict(client_id=client.id, source_id=client.global_test_id,
                                 **{"local_" + k: v for k, v in local.items()},
                                 **{"global_" + k: v for k, v in generalized.items()}))
            finally:
                model.cpu()
    return rows


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run(args, folder, stamp, context, server_state):
    configure_determinism(args["deterministic"], args["device"])
    seed_all(args["seed"])
    cotton = args["dataset"] == "cotton30"
    context["phase"] = "data_loading"
    if cotton:
        train, test, sources, validation, validation_sources = distribute_cotton(
            args["client_num"], args["seed"], args.get("cotton_root"), return_validation=True,
            return_source_validation=True, feature_set=args["cotton_feature_set"], include_test=args["cotton_test_eval"])
        save_json(folder / f"data-{stamp}.json", train[0].manifest)
    else:
        train, test, sources = distribute_dataset(args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"])
    seed_all(args["seed"])
    model = factorize_model(get_model(args["dataset"], args["model"]), args["factorized_rank"], args["factorized_l1"])
    server = FactorizedFLServer(model, args)
    server_state["server"] = server
    clients = [FactorizedFLClient(i, train[i], test.get(i), model,
                                  i // (args["client_num"] // 3) if cotton else 0)
               for i in range(args["client_num"])]
    provenance = dict(source=source_fingerprint(ROOT, SOURCE_FILES), training_data=training_fingerprint(train),
                      initial_model_sha256=state_fingerprint(model.state_dict()), runtime=runtime_fingerprint(args["device"]),
                      upstream_commit=UPSTREAM_COMMIT, trainable_parameters=sum(p.numel() for p in model.parameters()))
    save_json(folder / f"provenance-{stamp}.json", provenance)
    print("Config | " + json.dumps(args), flush=True)
    print(f"[output] {folder} | files suffix={stamp}", flush=True)
    print("[Factorized-FL] project-backbone adaptation, private classifier heads; no clipping; "
          "all downloads read frozen pre-round uploads; test data never used for matching", flush=True)
    print("Round | cached_clients | train_loss | local_acc | generalized_acc | seconds", flush=True)
    sampler = random.Random(args["seed"])
    total_steps, rows, start_run = 0, None, time.perf_counter()
    with (folder / f"rounds-{stamp}.jsonl").open("x", encoding="utf-8") as trajectory:
        for index in range(args["rounds"]):
            start, event = time.perf_counter(), False
            context.update(round=index + 1, phase="drift", client_id=None, completed_train_clients=[])
            pattern = args["drift_pattern"]
            if (pattern == "sudden" and index == 100 or pattern == "recurrent" and index in (100, 150)):
                sudden_drift(clients, sources, index, args["drift_class_count"])
                event = True
            elif pattern == "incremental" and index in (100, 110, 120):
                incremental_drift(clients, sources, index, args["drift_class_count"])
                event = True
            selected = sampler.sample(clients, int(args["client_num"] * args["sample_ratio"]))
            context.update(phase="train", selected=[c.id for c in selected])
            training, neighbors = [], {}
            for client in selected:
                context["client_id"] = client.id
                state, donor_ids, weights = server.personalized_body(client.id)
                client.receive(state)
                neighbors[str(client.id)] = dict(clients=donor_ids, weights=weights)
                training.append(client.train(args, index))
                context["completed_train_clients"].append(client.id)
            # Uploads are committed only after ALL local training is complete.
            server.update(selected)
            steps = sum(t["optimizer_steps"] for t in training)
            total_steps += steps
            record = dict(round=index + 1, selected=[c.id for c in selected], drift=event,
                          cached_clients=len(server.uploads), matching=neighbors, optimizer_steps=steps,
                          **{k: float(np.mean([t[k] for t in training])) for k in ("train_loss", "train_ce", "train_l1")})
            local = generalized = None
            if (index + 1) % args["eval_interval"] == 0 or index + 1 == args["rounds"]:
                context["phase"] = "evaluation"
                rows = evaluate_split(clients, validation if cotton else test, validation_sources if cotton else sources, args)
                local, generalized = mean_client_metrics(rows, "local")["accuracy"], mean_client_metrics(rows, "global")["accuracy"]
                record.update(local_accuracy=local, generalized_accuracy=generalized,
                              evaluation_split="validation" if cotton else "test")
            record["seconds"] = time.perf_counter() - start
            trajectory.write(json.dumps(record, allow_nan=False) + "\n")
            trajectory.flush()
            context["last_completed_round"] = index + 1
            print(f"{index + 1:03d}/{args['rounds']} | {len(server.uploads)} | {record['train_loss']:.4f} | "
                  f"{local if local is not None else '-'} | {generalized if generalized is not None else '-'} | {record['seconds']:.1f}", flush=True)
    context.update(training_completed=True, phase="final_evaluation")
    validation_rows = rows if cotton else None
    if cotton and args["cotton_test_eval"]:
        rows = evaluate_split(clients, test, sources, args)
        print("[final-test] Cotton test evaluated once after the configured run", flush=True)
    prefix = "validation_" if cotton and not args["cotton_test_eval"] else ""
    summary = dict(algorithm="FactorizedFL", implementation=args["implementation"], dataset=args["dataset"],
                   seed=args["seed"], rounds=args["rounds"], completed_rounds=args["rounds"], run_completed=True,
                   training_completed=True, original_paper_protocol=False, factorized_variant=args["factorized_variant"],
                   smoke_test=args["smoke_test"], optimizer_steps=total_steps, deterministic=args["deterministic"],
                   evaluation_split="validation" if prefix else "test", drift_pattern=args["drift_pattern"],
                   trainable_parameters=provenance["trainable_parameters"], upstream_commit=UPSTREAM_COMMIT,
                   training_data_sha256=provenance["training_data"]["sha256"],
                   source_sha256=provenance["source"]["sha256"], initial_model_sha256=provenance["initial_model_sha256"],
                   seconds=time.perf_counter() - start_run)
    for scope in ("local", "global"):
        for key, value in mean_client_metrics(rows, scope).items():
            summary[prefix + scope + "_" + key] = value
    summary[prefix + "generalized_accuracy"] = mean_client_metrics(rows, "global")["accuracy"]
    summary[prefix + "local_client_std"] = float(np.std([r["local_accuracy"] for r in rows]))
    summary[prefix + "generalized_client_std"] = float(np.std([r["global_accuracy"] for r in rows]))
    if cotton:
        summary["cotton_test_evaluated"] = args["cotton_test_eval"]
        for scope in ("local", "global"):
            for key, value in mean_client_metrics(validation_rows, scope).items():
                summary["validation_" + scope + "_" + key] = value
        summary["validation_generalized_accuracy"] = summary["validation_global_accuracy"]
    context["phase"] = "save_results"
    with (folder / f"results-{stamp}.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(folder / f"summary-{stamp}.json", summary)
    print("[completed] " + json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = build_parser()
    args = resolve_config(parser.parse_args(argv), parser)
    folder = output_folder(dict(args, algorithm="FactorizedFL-" + args["factorized_variant"].title()))
    stamp = reserve_run(folder)
    context = dict(round=None, last_completed_round=0, phase="setup", training_completed=False,
                   client_id=None, selected=[], completed_train_clients=[])
    server_state = {}
    with record_failures(folder, stamp, args, context, server_provider=lambda: server_state.get("server")):
        with (folder / f"config-{stamp}.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(args, stream, allow_unicode=True)
        return run(args, folder, stamp, context, server_state)
