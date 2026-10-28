"""Opt-in, TRAIN-only stage measurements; never a gate or optimizer.

Fixed raw indices and the same materialized inputs/labels are reused within
each measured round. Labels are re-read after drift. Metrics describe a small
training diagnostic panel, NOT held-out Local/Generalized accuracy. Only
scalar/class summaries are written; examples and features remain local.
"""
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from entities.FedCAPProfile import isolated_profile_rng, unique_profile_indices


POLICY = "read_only_fixed_train_stage_panel_v1"
STAGES = ("before_local_training", "after_local_training",
          "after_body_aggregation", "after_cluster_aggregation", "after_row_sharing")
TRANSITIONS = (("local_training", STAGES[0], STAGES[1]),
               ("body_aggregation", STAGES[1], STAGES[2]),
               ("cluster_head_aggregation", STAGES[2], STAGES[3]),
               ("row_sharing", STAGES[3], STAGES[4]))
METRICS = ("sample_ce", "macro_ce", "sample_accuracy", "balanced_accuracy",
           "train_prior_ce", "train_prior_accuracy")
OPTIONS = dict(stage_start_round=1, stage_interval=1, stage_samples_per_class=64)


def validate_stages(args, options, parser):
    if options.stage_diagnostics is not None:
        args["stage_diagnostics"] = options.stage_diagnostics == "true"
    enabled = args.get("stage_diagnostics", False)
    if not isinstance(enabled, bool):
        parser.error("stage_diagnostics in YAML must be true or false")
    if enabled:
        if args["algorithm"] != "FedCAP":
            parser.error("Stage diagnostics are only supported by FedCAP")
        if args.get("diagnostic_head_sharing", "none") != "none":
            parser.error("Stage diagnostics do not support oracle head sharing")
        for key, default in OPTIONS.items():
            value = getattr(options, key)
            value = args.get(key, default) if value is None else value
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                parser.error(f"{key} must be a positive integer")
            args[key] = value
        args["stage_diagnostics_policy"] = POLICY
        if args.get("head_calibration_ratio", 0) > 0:
            args["stage_diagnostics_policy"] = "read_only_fixed_train_stage_panel_head_calibration_v1"
    else:
        # Explicit disabling of a saved diagnostic config is supported.
        if any(getattr(options, key) is not None for key in OPTIONS):
            parser.error("Stage panel options require --stage-diagnostics true")
        for key in (*OPTIONS, "stage_diagnostics_policy"):
            args.pop(key, None)
        args.pop("stage_diagnostics", None)


def _hash(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _head_stats(head):
    weight, bias = head["weight"].detach().double(), head["bias"].detach().double()
    finite = bool(torch.isfinite(weight).all() and torch.isfinite(bias).all())
    return dict(finite_head=finite,
                row_weight_l2=weight.norm(dim=1).cpu().tolist() if finite else None,
                row_bias=bias.cpu().tolist() if finite else None)


class StageDiagnostics:
    def __init__(self, folder, stamp, args):
        self.path = Path(folder) / f"stages-{stamp}.jsonl"
        self.args, self.stream = args, None
        self.calibration = args.get("head_calibration_ratio", 0) > 0
        self.policy = args.get("stage_diagnostics_policy", "read_only_fixed_train_stage_panel_head_calibration_v1"
                               if self.calibration else POLICY)
        self.stages = (STAGES[:3] + ("after_head_calibration",) + STAGES[3:]
                       if self.calibration else STAGES)
        self.transitions = ((TRANSITIONS[:2] +
                             (("head_calibration", STAGES[2], "after_head_calibration"),
                              ("cluster_head_aggregation", "after_head_calibration", STAGES[3])) +
                             TRANSITIONS[3:]) if self.calibration else TRANSITIONS)
        self.indices, self.panels, self.measures = {}, {}, {}
        self.rounds, self.records, self.seconds = 0, 0, 0.0
        self.round_index = None
        self.aggregation_observed = False
        self.body_observed = False
        self.transition_totals = {name: dict(measured_clients=0, loss_increased_clients=0,
                                            loss_decreased_clients=0, loss_unchanged_clients=0,
                                            unavailable_clients=0)
                                  for name, _, _ in self.transitions}

    def __enter__(self):
        self.stream = self.path.open("x", encoding="utf-8")
        print(f"[stage-diagnostics] {self.path} | fixed TRAIN panel; forward-only; "
              "not test accuracy or an acceptance rule", flush=True)
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()
        self.panels.clear()

    def active(self, round_index):
        start = self.args["stage_start_round"]
        return round_index + 1 >= start and (round_index + 1 - start) % self.args["stage_interval"] == 0

    def _write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        self.records += 1

    def _seed(self, client):
        return (self.args["seed"] + client.id + 982451653) % (2 ** 63 - 1)

    def begin_round(self, server, selected, round_index):
        self.round_index, self.aggregation_observed = round_index, False
        self.body_observed = False
        self.panels, self.measures = {}, {}
        if not self.active(round_index):
            return
        started = time.perf_counter()
        for client in selected:
            seed = self._seed(client)
            # Freeze only raw indices, not labels; recurrent/incremental label
            # updates must be reflected on the exact same examples.
            if client.id not in self.indices:
                self.indices[client.id] = unique_profile_indices(
                    client.train_set.targets, self.args["stage_samples_per_class"],
                    self.args["num_classes"], seed)
            indices = self.indices[client.id]
            dataset = copy(client.train_set)
            if self.args["dataset"] in ("CIFAR10", "CIFAR100", "CINIC-10"):
                # Reuse preprocessing ONLY, never test examples or labels.
                dataset.transform = deepcopy(client.test_set.transform)
            elif hasattr(dataset, "transform"):
                dataset.transform = deepcopy(dataset.transform)
            with isolated_profile_rng(seed, self.args["device"]):
                loader = DataLoader(Subset(dataset, indices), batch_size=self.args["batch_size"],
                                    shuffle=False, generator=torch.Generator().manual_seed(seed))
                batches = [(x.detach().cpu().clone(), y.detach().cpu().long().clone()) for x, y in loader]
            labels = torch.cat([y for _, y in batches]).numpy()
            self.panels[client.id] = dict(
                batches=batches, index_sha256=_hash(indices), label_sha256=_hash(labels),
                samples=len(labels), sample_counts=np.bincount(labels, minlength=self.args["num_classes"]),
                train_counts=np.bincount(client.train_set.targets, minlength=self.args["num_classes"]))
            self.measures[client.id] = {}
            self._observe(client, client.model.body.state_dict(),
                          {STAGES[0]: client.model.head.state_dict()}, server.model_key(client.id))
        self.seconds += time.perf_counter() - started

    def observe_local(self, server, client):
        if client.id not in self.panels:
            return
        started = time.perf_counter()
        self._observe(client, client.model.body.state_dict(),
                      {STAGES[1]: client.model.head.state_dict()}, server.model_key(client.id))
        self.seconds += time.perf_counter() - started

    def observe_aggregation(self, server, selected):
        """Call after global-body/cluster-head means, BEFORE any row sharing.

        Without calibration, the body-only stage is a counterfactual: new
        global body + the local head. With calibration it is recorded earlier,
        before the real body download/head SGD; cluster replacement is then
        measured relative to the newly calibrated local head.
        """
        if not self.panels:
            return
        if self.aggregation_observed:
            raise RuntimeError("Stage aggregation was already observed this round")
        started = time.perf_counter()
        for client in selected:
            heads = {} if self.body_observed else {STAGES[2]: client.model.head.state_dict()}
            heads[STAGES[3]] = server.head_state(client.id)
            if self.calibration and "after_head_calibration" not in self.measures[client.id]:
                raise RuntimeError("Missing post-global-body head calibration measurement")
            self._observe(client, server.body_state,
                          heads,
                          server.model_key(client.id))
        self.aggregation_observed = True
        self.seconds += time.perf_counter() - started

    def observe_body_aggregation(self, server, selected):
        """Actual downloaded body + pre-calibration local head, before head SGD."""
        if not self.panels:
            return
        if not self.calibration or self.body_observed:
            raise RuntimeError("Unexpected/duplicate calibrated body measurement")
        started = time.perf_counter()
        for client in selected:
            self._observe(client, server.body_state, {STAGES[2]: client.model.head.state_dict()},
                          server.model_key(client.id))
        self.body_observed = True
        self.seconds += time.perf_counter() - started

    def observe_head_calibration(self, server, client):
        if client.id not in self.panels:
            return
        if not self.body_observed:
            raise RuntimeError("Head calibration must follow body aggregation")
        started = time.perf_counter()
        self._observe(client, server.body_state, {"after_head_calibration": client.model.head.state_dict()},
                      server.model_key(client.id))
        self.seconds += time.perf_counter() - started

    def finish_round(self, server, selected):
        if not self.panels:
            return None
        if not self.aggregation_observed:
            raise RuntimeError("Missing pre-row-sharing stage measurement")
        started = time.perf_counter()
        for client in selected:
            self._observe(client, server.body_state, {STAGES[4]: server.head_state(client.id)},
                          server.model_key(client.id))
        transitions = {}
        for name, before, after in self.transitions:
            paired = [row for row in self.measures.values() if row[before]["finite"] and row[after]["finite"]]
            available = {metric: [row for row in paired if row[before][metric] is not None
                                  and row[after][metric] is not None] for metric in METRICS}
            transitions[name] = dict(
                paired_clients=len(paired),
                metric_paired_clients={metric: len(rows) for metric, rows in available.items()},
                **{f"delta_{metric}": float(np.mean([row[after][metric] - row[before][metric]
                                                     for row in available[metric]])) if available[metric] else None
                   for metric in METRICS})
            # Means can hide a minority harmed by otherwise useful sharing.
            # These are descriptive TRAIN measurements, never an accept/veto
            # rule and never a substitute for Local/Generalized test accuracy.
            rows = available["train_prior_ce"]
            deltas = [row[after]["train_prior_ce"] - row[before]["train_prior_ce"] for row in rows]
            counts = dict(measured_clients=len(deltas),
                          loss_increased_clients=sum(value > 1e-8 for value in deltas),
                          loss_decreased_clients=sum(value < -1e-8 for value in deltas),
                          loss_unchanged_clients=sum(abs(value) <= 1e-8 for value in deltas),
                          unavailable_clients=len(selected) - len(deltas))
            transitions[name].update(**counts,
                                     largest_loss_increase=max(0.0, max(deltas)) if deltas else None)
            for key, value in counts.items():
                self.transition_totals[name][key] += value
        stages = {}
        for stage in self.stages:
            finite = [row[stage] for row in self.measures.values() if row[stage]["finite"]]
            stages[stage] = dict(finite_clients=len(finite),
                                metric_clients={metric: sum(row[metric] is not None for row in finite)
                                                for metric in METRICS},
                                **{metric: float(np.mean([row[metric] for row in finite if row[metric] is not None]))
                                   if any(row[metric] is not None for row in finite) else None
                                   for metric in METRICS})
        record = dict(kind="round_summary", round=self.round_index + 1, policy=self.policy,
                      clients=len(selected), stages=stages, transitions=transitions,
                      shared_row_groups=server.last_profile_stats.get("profile_cross_cluster_row_groups", 0)
                      if hasattr(server, "last_profile_stats") else 0,
                      metric_scope="TRAIN_panel_not_test_accuracy")
        self._write(record)
        self.rounds += 1
        self.seconds += time.perf_counter() - started
        def fmt(name):
            value = transitions[name]["delta_train_prior_ce"]
            return "n/a" if value is None else f"{value:+.4f}"
        print(f"[stages] round={self.round_index + 1} TRAIN prior-CE change: "
              f"body={fmt('body_aggregation')} "
              + (f"calibration={fmt('head_calibration')} " if self.calibration else "")
              + f"head={fmt('cluster_head_aggregation')} "
              f"rows={fmt('row_sharing')} (positive=worse; no gate)", flush=True)
        self.panels.clear()
        return record

    def _observe(self, client, body_state, heads, model_key):
        panel, classes = self.panels[client.id], self.args["num_classes"]
        results = {stage: dict(losses=torch.zeros(classes, dtype=torch.float64),
                               correct=torch.zeros(classes, dtype=torch.float64), finite=True,
                               feature_norm=0.0) for stage in heads}
        with isolated_profile_rng(self._seed(client), self.args["device"]), torch.no_grad():
            model = deepcopy(client.model).eval()
            # Copies must not trigger hooks closing over a live numerical monitor.
            for module in model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                module._backward_hooks.clear()
            model.body.load_state_dict(body_state)
            for x, y in panel["batches"]:
                features = model.body(x.to(self.args["device"]))
                labels = y.to(self.args["device"])
                feature_norm = float(features.detach().double().norm(dim=1).sum())
                for stage, head in heads.items():
                    model.head.load_state_dict(head)
                    logits = model.head(features)
                    loss = F.cross_entropy(logits, labels, reduction="none")
                    result = results[stage]
                    if not bool(torch.isfinite(features).all() and torch.isfinite(loss).all()):
                        result["finite"] = False
                        continue
                    result["feature_norm"] += feature_norm
                    correct = (logits.argmax(1) == labels).cpu().double()
                    result["losses"].scatter_add_(0, y, loss.cpu().double())
                    result["correct"].scatter_add_(0, y, correct)
        supported = panel["sample_counts"] > 0
        for stage, result in results.items():
            stats = _head_stats(heads[stage])
            finite = result["finite"] and stats["finite_head"]
            metrics = {key: None for key in METRICS}
            ce, acc = np.zeros(classes), np.zeros(classes)
            if finite:
                counts = panel["sample_counts"][supported]
                ce[supported] = result["losses"].numpy()[supported] / counts
                acc[supported] = result["correct"].numpy()[supported] / counts
                prior = panel["train_counts"] / panel["train_counts"].sum()
                # Label permutations preserve coverage. If a different drift
                # introduces a class absent from the fixed panel, its prior
                # metric is unknown, never silently imputed as zero loss.
                full_coverage = not panel["train_counts"][~supported].any()
                metrics = dict(sample_ce=float(result["losses"].sum() / panel["samples"]),
                               macro_ce=float(ce[supported].mean()),
                               sample_accuracy=float(result["correct"].sum() / panel["samples"]),
                               balanced_accuracy=float(acc[supported].mean()),
                               train_prior_ce=float(np.dot(prior, ce)) if full_coverage else None,
                               train_prior_accuracy=float(np.dot(prior, acc)) if full_coverage else None)
            measured = dict(finite=finite, **metrics)
            self.measures[client.id][stage] = measured
            self._write(dict(
                kind="client_stage", round=self.round_index + 1, stage=stage, client_id=client.id,
                model_key=list(model_key), policy=self.policy, samples=panel["samples"],
                sample_counts=panel["sample_counts"].tolist(), train_class_counts=panel["train_counts"].tolist(),
                supported_train_mass=float(panel["train_counts"][supported].sum() / panel["train_counts"].sum()),
                index_sha256=panel["index_sha256"], label_sha256=panel["label_sha256"],
                class_ce=[float(ce[c]) if finite and supported[c] else None for c in range(classes)],
                class_accuracy=[float(acc[c]) if finite and supported[c] else None for c in range(classes)],
                mean_feature_l2=result["feature_norm"] / panel["samples"] if finite else None,
                counterfactual=stage == STAGES[2] and not self.calibration, **stats, **measured))

    def summary(self):
        return dict(stage_diagnostics=True, stage_diagnostics_policy=self.policy,
                    stage_file=self.path.name, stage_measured_rounds=self.rounds,
                    stage_records=self.records, stage_seconds=self.seconds,
                    stage_transition_totals=deepcopy(self.transition_totals),
                    stage_metric_scope="TRAIN_panel_not_test_accuracy")
