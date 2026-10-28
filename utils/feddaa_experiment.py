"""Unified-protocol FedDAA baseline; deliberately independent of FedCAP runner."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from entities.FedDAA import FedDAAClient, FedDAAServer, ProbabilityMixture, UPSTREAM_COMMIT, isolated_rng
from utils.cotton_dataset import distribute_cotton
from utils.drift import sudden_drift, incremental_drift, label_swap_pairs
from utils.fedcap_experiment import evaluate, mean_client_metrics
from utils.fedcap_numerics import record_failures
from utils.fedcap_output import output_folder, reserve_run
from utils.fedcap_reproducibility import (
    configure_determinism, runtime_fingerprint, source_fingerprint,
    state_fingerprint, training_fingerprint)
from utils.gen_dataset import distribute_dataset, resolve_image_data_root
from utils.models import get_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("entities/FedDAA.py", "utils/feddaa_experiment.py", "methods/FedDAA.py",
                "configs/FedDAA.yaml", "utils/gen_dataset.py", "utils/models.py",
                "utils/drift.py", "utils/cotton_dataset.py", "utils/fedcap_reproducibility.py",
                "utils/fedcap_experiment.py", "utils/fedcap_numerics.py", "utils/fedcap_output.py")


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def build_parser():
    parser = argparse.ArgumentParser(description="FedDAA under the repository's unified protocol")
    parser.add_argument("--algorithm", choices=("FedDAA",), default="FedDAA")
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=("Fashion-MNIST", "CIFAR10", "CIFAR100", "CINIC-10", "cotton30", "cotton"))
    for flag, kind in (("clients", int), ("sample-ratio", float), ("alpha", float),
                       ("seed", int), ("rounds", int), ("epochs", int), ("batch-size", int),
                       ("lr", float), ("momentum", float), ("weight-decay", float),
                       ("eval-interval", int), ("drift-class-count", int),
                       ("feddaa-initial-clusters", int), ("feddaa-max-clusters", int),
                       ("feddaa-time-step-rounds", int)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--drift", choices=("false", "sudden", "incremental", "recurrent"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--deterministic", type=boolean)
    parser.add_argument("--cotton-root")
    parser.add_argument("--data-root", help="Image data root; default repository/data; relative paths are repository-relative")
    parser.add_argument("--cotton-feature-set", choices=("all23", "no-expression"))
    parser.add_argument("--cotton-test-eval", type=boolean)
    parser.add_argument("--output-root")
    return parser


def resolve_config(options, parser):
    source = Path(options.config) if options.config else ROOT / "configs/FedDAA.yaml"
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        parser.error(f"FedDAA config is not a file: {source}")
    with source.open(encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    if not isinstance(args, dict):
        parser.error("FedDAA config must be a mapping")
    allowed = {action.dest for action in parser._actions} | {"client_num", "drift_pattern", "partition", "model"}
    if set(args) - allowed:
        parser.error(f"Unknown FedDAA config fields: {sorted(set(args) - allowed)}")
    for key, value in vars(options).items():
        if value is not None and key not in ("config", "algorithm"):
            args[{"clients": "client_num", "drift": "drift_pattern"}.get(key, key)] = value
    if args.get("algorithm", "FedDAA") != "FedDAA":
        parser.error("FedDAA entry cannot run another algorithm")
    args["algorithm"] = "FedDAA"
    args.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    args.setdefault("drift_class_count", 6)
    args.setdefault("output_root", str(ROOT))
    args["output_root"] = str(Path(args["output_root"]).resolve())
    args["config_path"] = str(source.resolve())
    cotton = args.get("dataset", "").lower() in ("cotton", "cotton30")
    if cotton and args.get("data_root") is not None:
        parser.error("--data-root is for image datasets; use --cotton-root for Cotton")
    if not cotton:
        args["data_root"] = str(resolve_image_data_root(args.get("data_root")))
    if cotton:
        args["dataset"], args["model"], args["partition"] = "cotton30", "MLP", "family"
    args["num_classes"] = 2 if cotton else 100 if args.get("dataset") == "CIFAR100" else 10
    if args.get("dataset") not in ("Fashion-MNIST", "CIFAR10", "CIFAR100", "CINIC-10", "cotton30"):
        parser.error("Unsupported FedDAA dataset")
    for field in ("client_num", "rounds", "epochs", "batch_size", "eval_interval",
                  "feddaa_initial_clusters", "feddaa_max_clusters", "feddaa_time_step_rounds"):
        value = args.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            parser.error(f"{field} must be a positive integer")
    if args["feddaa_initial_clusters"] > args["feddaa_max_clusters"]:
        parser.error("initial clusters must not exceed maximum clusters")
    if args.get("seed", -1) < 0:
        parser.error("seed must be nonnegative")
    if not 0 < args.get("sample_ratio", 0) <= 1 or int(args["client_num"] * args["sample_ratio"]) < 1:
        parser.error("sample ratio must select at least one client and be at most 1")
    for name in ("lr", "alpha"):
        if not np.isfinite(args.get(name, np.nan)) or args[name] <= 0:
            parser.error(f"{name} must be finite and positive")
    if not 0 <= args.get("momentum", -1) < 1 or not np.isfinite(args.get("weight_decay", np.nan)) or args["weight_decay"] < 0:
        parser.error("Invalid momentum or weight_decay")
    if args.get("partition") not in ("hetero", "homo", "family") or (not cotton and args.get("partition") == "family"):
        parser.error("Invalid dataset partition")
    if not cotton and args.get("model") != "CNN":
        parser.error("Image FedDAA requires the shared CNN architecture")
    if args.get("device") not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda")
    if args["device"] == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    for field in ("deterministic", "cotton_test_eval"):
        if not isinstance(args.get(field), bool):
            parser.error(f"{field} must be true or false")
    if args.get("cotton_feature_set") not in ("all23", "no-expression"):
        parser.error("Invalid cotton_feature_set")
    drift = args.get("drift_pattern")
    if drift not in (False, "false", "sudden", "incremental", "recurrent"):
        parser.error("Invalid drift_pattern")
    if cotton and (args["client_num"] < 3 or args["client_num"] % 3 or drift not in (False, "false")):
        parser.error("Cotton requires a multiple of 3 clients and drift=false")
    if not cotton:
        try:
            args["drift_label_pairs"] = label_swap_pairs(args["drift_class_count"], args["num_classes"])
        except ValueError as error:
            parser.error(str(error))
    args["drift_pattern"] = False if drift == "false" else drift
    args.update(implementation="feddaa_unified_protocol_v1", upstream_commit=UPSTREAM_COMMIT,
                prototype_policy="first_expert_raw_logits_per_class_train_only_v1",
                prototype_participation="all_clients_at_fixed_time_slots",
                history_policy="previous_raw_slot_plus_current_if_rdld_clean",
                history_input_policy="reused_static_inputs_frozen_slot_labels",
                model_resize_policy="retain_expert_indices_seed_new_experts",
                expert_aggregation="full_model_train_sample_count_not_mixture_weight",
                evaluation_model="personalized_probability_mixture_train_fitted",
                gradient_clipping="none", original_paper_protocol=False,
                smoke_test=args["rounds"] < (100 if cotton else 200))
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model_factory(args):
    def factory(expert):
        # Expert 0 has exactly the seed used by the other project methods.
        # Distinct seed+expert is essential: identical experts would never
        # escape uniform FedRC responsibilities.
        with isolated_rng(args["device"]):
            torch.manual_seed(args["seed"] + expert)
            return get_model(args["dataset"], args["model"]).cpu().float()
    return factory


def evaluate_split(server, clients, local_sets, source_sets, cotton):
    rows, cache = [], {}
    device = server.args["device"]
    with isolated_rng(device):
        try:
            for model in server.models:
                model.to(device).eval()
            for client in clients:
                mixture = ProbabilityMixture(server.models, client.mixture).to(device).eval()
                local = evaluate(mixture, local_sets[client.id], device, cotton)
                # Only train-fitted weights, never select models using test loss.
                key = (tuple(client.mixture.tolist()), client.global_test_id)
                if key not in cache:
                    cache[key] = evaluate(mixture, source_sets[client.global_test_id], device, cotton)
                rows.append(dict(client_id=client.id, source_id=client.global_test_id,
                                 **{"local_" + k: v for k, v in local.items()},
                                 **{"global_" + k: v for k, v in cache[key].items()}))
        finally:
            for model in server.models:
                model.cpu()
    return rows


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run_experiment(args, folder, stamp, context, server_state):
    configure_determinism(args["deterministic"], args["device"])
    seed_all(args["seed"])
    context["phase"] = "data_loading"
    cotton = args["dataset"] == "cotton30"
    validation = validation_sources = None
    if cotton:
        train, test, sources, validation, validation_sources = distribute_cotton(
            args["client_num"], args["seed"], args.get("cotton_root"), return_validation=True,
            feature_set=args["cotton_feature_set"], return_source_validation=True,
            include_test=args["cotton_test_eval"])
        save_json(folder / f"data-{stamp}.json", train[0].manifest)
    else:
        print(f"[data] image root={args['data_root']}", flush=True)
        train, test, sources = distribute_dataset(args["dataset"], args["client_num"],
                                                   args["partition"], args["alpha"], args["seed"],
                                                   data_root=args["data_root"])
    seed_all(args["seed"])
    # Evaluation transforms are deterministic preprocessing, not fitted on test.
    eval_transform = None if cotton else sources[0].transform
    factory = make_model_factory(args)
    server = FedDAAServer(factory, args, eval_transform)
    server_state["server"] = server
    clients = [FedDAAClient(i, train[i], test.get(i), getattr(train[i], "source", 0))
               for i in range(args["client_num"])]
    if cotton:
        for client in clients:
            client.global_test_id = client.id // (args["client_num"] // 3)
    provenance = dict(source=source_fingerprint(ROOT, SOURCE_FILES),
                      training_data=training_fingerprint(train),
                      initial_model_sha256=state_fingerprint(server.models[0].state_dict()),
                      initial_expert_sha256=[state_fingerprint(m.state_dict()) for m in server.models],
                      runtime=runtime_fingerprint(args["device"]), upstream_commit=UPSTREAM_COMMIT)
    save_json(folder / f"provenance-{stamp}.json", provenance)
    print("Config | " + json.dumps(args), flush=True)
    print(f"[output] {folder} | files suffix={stamp}", flush=True)
    print("[FedDAA] unified-protocol adaptation, NOT the original paper benchmark; "
          "raw-logit prototypes + KMeans/silhouette + RDLD + FedRC responsibilities", flush=True)
    print("[participation] local SGD uses sample_ratio; prototypes use ALL clients at fixed time slots; "
          "clean clients train on previous+current slot (static inputs may repeat); no clipping", flush=True)
    print("Round | experts | slot_refresh | detected | history | train_loss | local_acc | generalized_acc | seconds", flush=True)
    sampler = random.Random(args["seed"])
    count = int(args["client_num"] * args["sample_ratio"])
    started = time.perf_counter()
    steps_total, slots, rows = 0, [], None
    with (folder / f"rounds-{stamp}.jsonl").open("x", encoding="utf-8") as trajectory:
        for index in range(args["rounds"]):
            start, event = time.perf_counter(), False
            context.update(round=index + 1, phase="drift", client_id=None,
                           selected=[], completed_train_clients=[])
            pattern = args["drift_pattern"]
            if (pattern == "sudden" and index == 100 or
                    pattern == "recurrent" and index in (100, 150)):
                sudden_drift(clients, sources, index, args["drift_class_count"])
                event = True
            elif pattern == "incremental" and index in (100, 110, 120):
                incremental_drift(clients, sources, index, args["drift_class_count"])
                event = True
            refresh = index % args["feddaa_time_step_rounds"] == 0
            if refresh:
                context["phase"] = "prototype_and_history"
                slots.append(dict(round=index + 1, **server.start_slot(
                    clients, index // args["feddaa_time_step_rounds"])))
            selected = sampler.sample(clients, count)
            context.update(phase="training", selected=[client.id for client in selected])
            loss, steps = server.train_round(clients, selected, index, context)
            steps_total += steps
            record = dict(round=index + 1, selected=[c.id for c in selected], drift=event,
                          clusters=len(server.models), slot_refresh=refresh, train_loss=loss,
                          optimizer_steps=steps, train_samples=sum(len(c.training) for c in selected),
                          detected_clients=server.last_slot["detected_clients"],
                          history_clients=server.last_slot["history_clients"],
                          client_mixture_weights={str(c.id): c.mixture.tolist() for c in clients})
            if refresh:
                record["slot"] = server.last_slot
            local = generalized = None
            if (index + 1) % args["eval_interval"] == 0 or index + 1 == args["rounds"]:
                context["phase"] = "evaluation"
                rows = evaluate_split(server, clients, validation if cotton else test,
                                       validation_sources if cotton else sources, cotton)
                local = mean_client_metrics(rows, "local")["accuracy"]
                generalized = mean_client_metrics(rows, "global")["accuracy"]
                record.update(local_accuracy=local, generalized_accuracy=generalized,
                              evaluation_split="validation" if cotton else "test")
            record["seconds"] = time.perf_counter() - start
            trajectory.write(json.dumps(record, allow_nan=False) + "\n")
            trajectory.flush()
            context["last_completed_round"] = index + 1
            print(f"{index + 1:03d}/{args['rounds']} | {len(server.models)} | {int(refresh)} | "
                  f"{len(server.last_slot['detected_clients'])} | {len(server.last_slot['history_clients'])} | "
                  f"{loss:.4f} | {local if local is not None else '-'} | "
                  f"{generalized if generalized is not None else '-'} | {record['seconds']:.1f}", flush=True)
    context.update(training_completed=True, phase="final_evaluation")
    validation_rows = rows if cotton else None
    if cotton and args["cotton_test_eval"]:
        rows = evaluate_split(server, clients, test, sources, True)
        print("[final-test] Cotton test evaluated once after the configured run", flush=True)
    local_metrics, global_metrics = mean_client_metrics(rows, "local"), mean_client_metrics(rows, "global")
    prefix = "validation_" if cotton and not args["cotton_test_eval"] else ""
    summary = dict(algorithm="FedDAA", dataset=args["dataset"], seed=args["seed"],
                   rounds=args["rounds"], completed_rounds=args["rounds"],
                   training_completed=True, run_completed=True, smoke_test=args["smoke_test"],
                   implementation=args["implementation"], upstream_commit=UPSTREAM_COMMIT,
                   original_paper_protocol=False, clusters=len(server.models),
                   drift_pattern=args["drift_pattern"], num_classes=args["num_classes"],
                   time_slots=len(slots), optimizer_steps=steps_total,
                   prototype_client_uploads=sum(slot["prototype_clients"] for slot in slots),
                   evaluation_split="validation" if prefix else "test",
                   training_data_sha256=provenance["training_data"]["sha256"],
                   initial_model_sha256=provenance["initial_model_sha256"],
                   source_sha256=provenance["source"]["sha256"],
                   deterministic=args["deterministic"], seconds=time.perf_counter() - started)
    for scope, values in (("local", local_metrics), ("global", global_metrics)):
        for name, value in values.items():
            summary[prefix + scope + "_" + name] = value
    summary[prefix + "generalized_accuracy"] = global_metrics["accuracy"]
    summary[prefix + "local_client_std"] = float(np.std([r["local_accuracy"] for r in rows]))
    summary[prefix + "generalized_client_std"] = float(np.std([r["global_accuracy"] for r in rows]))
    if cotton:
        summary["cotton_test_evaluated"] = args["cotton_test_eval"]
        for scope in ("local", "global"):
            for name, value in mean_client_metrics(validation_rows, scope).items():
                summary["validation_" + scope + "_" + name] = value
        summary["validation_generalized_accuracy"] = summary["validation_global_accuracy"]
    context["phase"] = "save_results"
    with (folder / f"results-{stamp}.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Diagnostic snapshot, not a resumable checkpoint: data/RNG/optimizer
    # histories are intentionally NOT claimed to be recoverable from this file.
    with (folder / f"checkpoint-{stamp}.pt").open("xb") as stream:
        torch.save(dict(resumable=False, models=[m.state_dict() for m in server.models],
                        label_priors=server.label_priors,
                        mixtures={c.id: c.mixture for c in clients}, config=args), stream)
    save_json(folder / f"summary-{stamp}.json", summary)
    print("[completed] " + json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = build_parser()
    args = resolve_config(parser.parse_args(argv), parser)
    folder = output_folder(args)
    stamp = reserve_run(folder)
    context = dict(round=None, last_completed_round=0, phase="setup", client_id=None,
                   selected=[], completed_train_clients=[], training_completed=False)
    server_state = {}
    with record_failures(folder, stamp, args, context,
                         server_provider=lambda: server_state.get("server")):
        # Record effective arguments even when data loading subsequently fails.
        with (folder / f"config-{stamp}.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(args, stream, allow_unicode=True)
        return run_experiment(args, folder, stamp, context, server_state)
