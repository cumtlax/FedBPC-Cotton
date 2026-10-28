"""Read-only diagnostics for body/head transitions; never repair an update.

Alarm thresholds are diagnostic heuristics, not acceptance rules. Counterfactual
heads are evaluated on copies, using the exact current TRAIN batch and RNG.
They can identify a suspect transition, but do not prove a causal mechanism.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from utils.fedcap_numerics import cpu_copy, norm_summary

POLICY = "read_only_body_head_transition_v1"


def rng_state(device):
    device = torch.device(device)
    result = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch=torch.get_rng_state().clone())
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], torch.device(device))


@contextmanager
def preserve_rng(device):
    saved = rng_state(device)
    try:
        yield
    finally:
        restore_rng(saved, device)


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def state_change(before, after):
    return norm_summary({key: after[key].detach().double() - before[key].to(after[key]).double()
                         for key in after})


def counterfactual_heads(monitor, heads):
    """Repeat a single backward on copies; no SGD, calibration or test data."""
    device = monitor.client.args["device"]
    result = {}
    start = getattr(monitor, "forward_rng", None) or rng_state(device)
    with preserve_rng(device), torch.enable_grad():
        for label, state in heads.items():
            restore_rng(start, device)
            model = deepcopy(monitor.model)
            # The numerical hook closes over the real monitor. Copies must not
            # call it or overwrite the live batch's diagnostic tensors.
            for module in model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                module._backward_hooks.clear()
            model.head.load_state_dict(state)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("body."))
            model.zero_grad(set_to_none=True)
            logits = model(monitor.tensors["inputs"])
            loss = F.cross_entropy(logits, monitor.tensors["targets"])
            proximal = getattr(monitor, "body_proximal", None)
            if proximal is not None:
                # Compare alternative heads under the SAME current body objective.
                # penalty() has no counters or mutation; diagnostic replays stay read-only.
                loss = loss + proximal.penalty(model)
            if torch.isfinite(loss):
                loss.backward()
                gradients = {name: p.grad for name, p in model.named_parameters()
                             if p.requires_grad and p.grad is not None}
                norms = norm_summary(gradients)
                result[label] = dict(loss=float(loss.detach()),
                                     gradient_l2=finite_number(norms["l2"]),
                                     gradient_max_abs=finite_number(norms["max_abs"]),
                                     finite_gradients=all(bool(torch.isfinite(g).all()) for g in gradients.values()))
            else:
                result[label] = dict(loss=None, finite_loss=False)
    return result


class MechanismDiagnostics:
    def __init__(self, folder, stamp, args):
        self.folder, self.stamp, self.args = Path(folder), stamp, args
        self.path = self.folder / f"mechanism-{stamp}.jsonl"
        self.stream = None
        self.records, self.alarms, self.first_alarm = 0, 0, None

    def __enter__(self):
        self.stream = self.path.open("x", encoding="utf-8")
        print(f"[mechanism-diagnostics] {self.path} | read-only; no gradient rescaling, rollback or skipped updates", flush=True)
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()

    def active(self, round_index):
        return round_index + 1 >= self.args.get("mechanism_start_round", 1)

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        self.records += 1

    def observe_sync(self, server, clients, round_index, stage):
        if not self.active(round_index):
            return
        for client in clients:
            incoming = server.head_state(client.id)
            previous = cpu_copy(client.model.head.state_dict())
            key = server.model_key(client.id)
            options = dict(before_download=previous, downloaded=cpu_copy(incoming))
            # A pre-sharing head is valid only if regroup has NOT changed the
            # corresponding final head. Never relabel stale pre-regroup evidence.
            shared = getattr(server, "_mechanism_shared_heads", {}).get(key)
            baseline = getattr(server, "_mechanism_cluster_heads", {}).get(key)
            valid_baseline = shared is not None and baseline is not None and all(
                torch.equal(shared[k].to(incoming[k]), incoming[k]) for k in incoming)
            if valid_baseline:
                options["cluster_only_before_row_share"] = baseline
            if stage == "before_local_training":
                client._mechanism_diagnostics = self
                client._mechanism_heads = options
                client._mechanism_transition = dict(
                    cluster_key=list(key), round=round_index + 1,
                    cluster_only_available=valid_baseline,
                    row_share_source_round=getattr(server, "_mechanism_share_round", None) if valid_baseline else None,
                    regroup_status=server.last_regroup_stats.get("regroup_status"))
            self.write(dict(kind="sync", stage=stage, round=round_index + 1, client_id=client.id,
                            cluster_key=list(key), head_change=state_change(previous, incoming),
                            body_change=state_change(client.model.body.state_dict(), server.body_state),
                            cluster_only_available=valid_baseline))

    def phase(self, client, stage):
        self.write(dict(kind="phase", stage=stage, round=client._training_round_index + 1,
                        client_id=client.id,
                        body=norm_summary(dict(client.model.body.named_parameters())),
                        head=norm_summary(dict(client.model.head.named_parameters())),
                        head_change_from_download=state_change(client._mechanism_heads["downloaded"],
                                                              client.model.head.state_dict())))

    def before_step(self, monitor):
        if monitor.context["component"] != "body":
            return
        parameters = {n: p for n, p in monitor.parameters().items() if p.requires_grad}
        body_norm = norm_summary(parameters)["l2"]
        updates = {}
        # Diagnostics of the existing SGD formula, not a replacement optimizer.
        for group in monitor.optimizer.param_groups:
            for name, p in parameters.items():
                if not any(p is member for member in group["params"]) or p.grad is None:
                    continue
                direction = p.grad.detach().double() + group["weight_decay"] * p.detach().double()
                buffer = monitor.optimizer.state.get(p, {}).get("momentum_buffer")
                if buffer is not None and group["momentum"]:
                    direction = group["momentum"] * buffer.detach().double() + (1 - group["dampening"]) * direction
                updates[name] = group["lr"] * direction
        update_norm = norm_summary(updates)["l2"]
        ratio = update_norm / max(body_norm, 1e-12)
        loss = float(monitor.tensors["loss"])
        gradient = monitor.raw_gradients["l2"]
        alarm = loss >= 100 or gradient >= 1e4 or ratio >= 1
        record = dict(kind="body_step", **monitor.context, loss=finite_number(loss),
                      body_l2=finite_number(body_norm), gradient_l2=finite_number(gradient),
                      predicted_update_l2=finite_number(update_norm), update_body_ratio=finite_number(ratio),
                      alarm=alarm, transition=getattr(monitor.client, "_mechanism_transition", None))
        if monitor.context["step"] == 1 or alarm and self.first_alarm is None:
            heads = dict(getattr(monitor.client, "_mechanism_heads", {}))
            heads["current_after_local_head_training"] = cpu_copy(monitor.model.head.state_dict())
            record["head_comparisons"] = counterfactual_heads(monitor, heads)
        if alarm:
            self.alarms += 1
            if self.first_alarm is None:
                filename = f"mechanism-first-alarm-{self.stamp}.pt"
                snapshot = dict(format=POLICY, resumable=False, record=record,
                                model_state=cpu_copy(monitor.model.state_dict()),
                                optimizer_state=cpu_copy(monitor.optimizer.state_dict()),
                                gradients=cpu_copy(monitor.gradients()),
                                tensors=cpu_copy(monitor.tensors),
                                alternative_heads=cpu_copy(heads),
                                forward_rng=cpu_copy(monitor.forward_rng))
                with (self.folder / filename).open("xb") as stream:
                    torch.save(snapshot, stream)
                self.first_alarm = filename
                record["snapshot"] = filename
                print(f"[mechanism-first-alarm] client={monitor.client.id} round={monitor.context['round']} "
                      f"body_step={monitor.context['step']} loss={loss:.5g} grad={gradient:.5g} "
                      f"update/body={ratio:.5g}; saved {filename}; training continues unchanged", flush=True)
        self.write(record)

    def summary(self):
        return dict(mechanism_diagnostics=True, mechanism_policy=POLICY,
                    mechanism_file=self.path.name, mechanism_records=self.records,
                    mechanism_alarm_steps=self.alarms, mechanism_first_alarm=self.first_alarm)
