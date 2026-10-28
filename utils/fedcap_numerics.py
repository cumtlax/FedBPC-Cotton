"""Opt-in numerical history for FedCAP failure diagnostics.

Diagnostics never modify an update; non-finite gradients always fail rather
than being replaced or rescaled.
"""
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import traceback

import torch


class NumericalTrainingError(FloatingPointError):
    def __init__(self, details, snapshot=None):
        self.details, self.snapshot = details, snapshot
        kind = "loss" if details["stage"] == "loss" else "tensor"
        super().__init__(
            f"Non-finite FedCAP {kind}: client={details['client_id']} "
            f"round={details['round']} component={details['component']} "
            f"epoch={details['epoch']} batch={details['batch']} "
            f"stage={details['stage']} tensors={details['nonfinite_tensors']}")


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def tensor_stats(value):
    """Finite-only reductions keep NaN/Inf out of the JSON itself."""
    value = value.detach()
    finite = torch.isfinite(value)
    selected = value[finite].double()
    return dict(
        shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
        elements=value.numel(), nan_count=int(torch.isnan(value).sum().item()),
        inf_count=int(torch.isinf(value).sum().item()),
        finite_min=float(selected.min().item()) if selected.numel() else None,
        finite_max=float(selected.max().item()) if selected.numel() else None,
        finite_max_abs=float(selected.abs().max().item()) if selected.numel() else None,
        finite_l2=float(selected.norm().item()) if selected.numel() else None)


def nonfinite_names(tensors):
    # One host synchronization per device, rather than one per parameter.
    devices = {}
    for name, tensor in tensors.items():
        if tensor is not None and tensor.numel():
            devices.setdefault(tensor.device, []).append((name, tensor))
    bad = []
    for items in devices.values():
        flags = torch.stack([torch.isfinite(value).all() for _, value in items]).cpu().tolist()
        bad.extend(name for (name, _), finite in zip(items, flags) if not finite)
    return bad


def norm_summary(tensors):
    values = [value.detach().double() for value in tensors.values()
              if value is not None and value.numel()]
    if not values:
        return dict(l2=0.0, max_abs=0.0)
    return dict(l2=float(torch.stack([value.square().sum() for value in values]).sum().sqrt().item()),
                max_abs=float(torch.stack([value.abs().max() for value in values]).max().item()))




class TrainingNumerics:
    def __init__(self, client, model, optimizer, component, round_index):
        self.client, self.model, self.optimizer = client, model, optimizer
        self.enabled = bool(client.args.get("numerical_diagnostics", False))
        history_size = client.args.get("numerical_history_size", 20)
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 0:
            raise ValueError("numerical_history_size must be a nonnegative integer")
        self.history = deque(maxlen=history_size)
        self.initial_state = None
        self.raw_gradients = None
        self.context = dict(client_id=client.id, component=component,
                            round=round_index + 1 if round_index is not None else None,
                            epoch=None, batch=None, step=0,
                            lr=optimizer.param_groups[0]["lr"])
        self.tensors, self.previous = {}, None

    def parameters(self):
        return dict(self.model.named_parameters())

    def gradients(self):
        return {name: parameter.grad for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None}

    def momentum(self):
        return {name: self.optimizer.state[parameter]["momentum_buffer"]
                for name, parameter in self.model.named_parameters()
                if "momentum_buffer" in self.optimizer.state.get(parameter, {})}

    def check(self, stage, tensors):
        bad = nonfinite_names(tensors)
        if not bad:
            return
        details = dict(
            **self.context, stage=stage, nonfinite_tensors=bad,
            numerical_diagnostics=self.enabled, last_finite_step=self.previous,
            initial_state=self.initial_state, recent_steps=list(self.history),
            numerical_history_size=self.history.maxlen,
            tensors={name: tensor_stats(value) for name, value in self.tensors.items()},
            parameters={name: tensor_stats(value) for name, value in self.parameters().items()},
            buffers={name: tensor_stats(value) for name, value in self.model.named_buffers()},
            gradients={name: tensor_stats(value) for name, value in self.gradients().items()},
            momentum={name: tensor_stats(value) for name, value in self.momentum().items()},
            client_class_counts=[int((torch.as_tensor(self.client.train_set.targets) == category).sum())
                                 for category in range(self.client.args["num_classes"])],
            settings={name: self.client.args.get(name) for name in
                      ("algorithm", "dataset", "seed", "rep_lr", "clf_lr", "momentum", "weight_decay",
                       "head_sharing", "class_share_rho", "class_share_eps", "profile_ablation",
                       "body_prox_mu", "body_prox_policy", "numerical_history_size")})
        if hasattr(self.model, "head") and hasattr(self.model.head, "weight"):
            details["head_rows"] = {
                str(category): tensor_stats(row) for category, row in enumerate(self.model.head.weight)}
        snapshot = None
        if self.enabled:
            snapshot = dict(format="fedcap_failed_batch_v1", resumable=False,
                            context=dict(self.context, stage=stage),
                            tensors=cpu_copy(self.tensors), model_state=cpu_copy(self.model.state_dict()),
                            gradients=cpu_copy(self.gradients()), optimizer_state=cpu_copy(self.optimizer.state_dict()),
                            initial_state=self.initial_state, recent_steps=list(self.history),
                            torch_rng_state=torch.get_rng_state().clone())
            proximal = getattr(self, "body_proximal", None)
            if proximal is not None:
                snapshot["body_proximal"] = dict(mu=proximal.mu, anchor=cpu_copy(proximal.anchor))
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                snapshot["cuda_rng_state"] = torch.cuda.get_rng_state(device).cpu().clone()
        raise NumericalTrainingError(details, snapshot)

    @contextmanager
    def capture_features(self):
        hook = None
        if self.enabled and hasattr(self.model, "body"):
            def capture(_module, _inputs, output):
                self.tensors["features"] = output.detach()
                self.check("features", {"features": output})
            hook = self.model.body.register_forward_hook(capture)
        try:
            yield
        finally:
            if hook is not None:
                hook.remove()

    def remember(self, loss):
        if self.enabled:
            self.previous = dict(**self.context, loss=float(loss),
                                 gradients=self.raw_gradients or norm_summary(self.gradients()),
                                 parameters=norm_summary(self.parameters()),
                                 momentum=norm_summary(self.momentum()),
                                 logits=norm_summary({"logits": self.tensors["logits"]}))
            if "features" in self.tensors:
                self.previous["features"] = norm_summary({"features": self.tensors["features"]})
            self.history.append(self.previous)

    def remember_initial_state(self):
        if self.enabled:
            self.initial_state = dict(**self.context, parameters=norm_summary(self.parameters()),
                                      body=norm_summary({k: v for k, v in self.parameters().items() if k.startswith("body.")}),
                                      head=norm_summary({k: v for k, v in self.parameters().items() if k.startswith("head.")}),
                                      momentum=norm_summary(self.momentum()))

    def prepare_step(self):
        """Fail fast on non-finite gradients; record opt-in diagnostic norms.

        Gradients are never clipped or rescaled here.
        """
        gradients = self.gradients()
        self.check("gradients", gradients)
        if not self.enabled:
            return
        self.raw_gradients = norm_summary(gradients)
        observer = getattr(self.client, "_mechanism_diagnostics", None)
        if observer is not None:
            observer.before_step(self)


@contextmanager
def record_failures(folder, stamp, args, context, server=None, server_provider=None):
    """Record setup, training, or saving failures; preserve the original error.

    A provider permits installing the guard before the server is constructed.
    Usage/configuration errors from argparse remain terminal-only errors.
    """
    try:
        yield
    except Exception as error:
        if server_provider is not None:
            server = server_provider()
        details = getattr(error, "details", None)
        report = dict(status="failed", exception_type=type(error).__name__, message=str(error),
                      failure_phase=context.get("phase"),
                      training_completed=bool(context.get("training_completed", False)),
                      run_completed=False,
                      algorithm=args["algorithm"], dataset=args["dataset"], seed=args["seed"],
                      config=cpu_copy(args),
                      run_context=cpu_copy(context), numerical=details,
                      reference_version=getattr(server, "version", None),
                      membership=getattr(server, "membership", None),
                      last_regroup=getattr(server, "last_regroup_stats", {}),
                      last_profile_sharing=getattr(server, "last_profile_stats", {}))
        folder = Path(folder)
        if args.get("mechanism_diagnostics", False):
            report["mechanism_file"] = f"mechanism-{stamp}.jsonl"
            alarm = folder / f"mechanism-first-alarm-{stamp}.pt"
            report["mechanism_first_alarm"] = alarm.name if alarm.exists() else None
        trace = traceback.format_exc()
        artifact_errors = {}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"traceback-{stamp}.txt").open("x", encoding="utf-8") as stream:
                stream.write(trace)
        except Exception as save_error:
            artifact_errors["traceback"] = str(save_error)
        snapshot = getattr(error, "snapshot", None)
        if snapshot is not None:
            try:
                snapshot_path = folder / f"numerical-{stamp}.pt"
                with snapshot_path.open("xb") as stream:
                    torch.save(snapshot, stream)
                report["numerical_snapshot"] = snapshot_path.name
            except Exception as save_error:
                artifact_errors["numerical_snapshot"] = str(save_error)
        if artifact_errors:
            report["artifact_errors"] = artifact_errors
        try:
            path = folder / f"error-{stamp}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            print(f"[failed] {error}\n[error-report] {path}", file=sys.stderr, flush=True)
        except Exception as save_error:
            # Logging problems must not hide the training error.
            print(f"[error-report-write-failed] {save_error}; original error: {error}",
                  file=sys.stderr, flush=True)
        if artifact_errors:
            print(f"[error-artifact-write-failed] {artifact_errors}", file=sys.stderr, flush=True)
        raise
