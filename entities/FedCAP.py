"""Prediction profiles plus isolated, balanced classifier probes.

Production models use one global body and cluster-specific whole heads.
The balanced probe can use FedCCFA training without changing profile,
distance, or formal local-training kernels. Uncertain heads are held separately
rather than creating extra historical concept clusters.
"""
from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from entities import FedCAPCore
from entities.FedCAPCore import (ClusterClient, ClusterServer, average_states,
                                balanced_indices, descriptor_distances, cluster_descriptors)
from torch.utils.data import DataLoader, Subset
from utils.models import get_model
from utils.fedcap_body_proximal import BodyProximal, validate_mu
from utils.fedcap_head_calibration import (HeadCalibrationSession, validate_ratio,
                                          calibration_summary, state_bytes)
from entities.FedCAPStability import RegroupConfirmation
from entities.FedCAPMergeGuard import MergeLossGuard
from entities.FedCAPProfileMerge import make_profile_merge_guard
from entities.FedCAPProfile import (LEGACY_POLICY, ProfileMoments, profile_budget,
                                   measure_unique_profile, isolated_profile_rng,
                                   profile_head, profile_width, profile_values, profile_metadata)
from entities.FedCAPClassSupport import (
    validate_class_support, validate_support_scope, class_support_weights, current_class_counts,
    CROSS_ONLY_SUFFIX,
    DEFAULT_COUNT_CAP as CLASS_SUPPORT_CAP, support_policy, implementation_suffix)


CCFA_PROBE_POLICY = "fedccfa_min5_train_epochs_global_body_fixed_head_v1"
LEGACY_PROBE_POLICY = "cached_eval_features_fixed_steps_v1"
FRESH_CACHE_POLICY = "current_round_same_global_body_v1"
LEGACY_CACHE_POLICY = "reference_version_ttl_v1"
BODY_STEP_SUFFIX = "_body_aggregation_step_v1"


def validate_body_aggregation_step(value):
    """Server BODY step only; one preserves the historical full replacement."""
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not np.isfinite(value) or not 0 < value <= 1):
        raise ValueError("body_aggregation_step must be finite in (0, 1]")
    return float(value)


def body_aggregation_policy(step):
    return ("sample_weighted_mean_v1" if step == 1 else
            "previous_global_toward_sample_weighted_mean_v1")


def stepped_body_state(previous, averaged, step):
    """Interpolate the BODY only, without mutating donors or the old anchor.

    Non-floating buffers take their new aggregated values rather than a
    fractional update. Current CNN/MLP bodies contain floating parameters only.
    The endpoint returns the original average without any extra arithmetic.
    """
    step = validate_body_aggregation_step(step)
    if step == 1:
        return averaged
    if previous.keys() != averaged.keys():
        raise ValueError("BODY aggregation states have different keys")
    result = {}
    with torch.no_grad():
        for key, value in averaged.items():
            anchor = previous[key]
            if (anchor.shape != value.shape or anchor.dtype != value.dtype or
                    anchor.device != value.device):
                raise ValueError(f"Incompatible BODY aggregation tensor: {key}")
            if value.is_floating_point() or value.is_complex():
                updated = anchor + step * (value - anchor)
                if not (torch.isfinite(anchor).all() and torch.isfinite(value).all() and
                        torch.isfinite(updated).all()):
                    raise FloatingPointError(f"Non-finite stepped BODY aggregation: {key}")
                result[key] = updated
            else:
                result[key] = value.clone()
    return result


def measurement_policies(args):
    """Derived metadata, not switches that can disable comparability checks."""
    fresh = args.get("probe_training", "fedccfa") == "fedccfa"
    return dict(descriptor_cache_policy=FRESH_CACHE_POLICY if fresh else LEGACY_CACHE_POLICY,
                profile_body_policy=("current_global_body_head_free_v1" if args.get("profile_head", "reference") == "none" else
                                     "current_global_body_interval_anchor_head_v1" if fresh
                                     else "periodic_reference_body_v1"))


def ccfa_balanced_indices(targets):
    """FedCCFA's min(min class count, 5) draw, without replacement.

    Iterate actual labels so a Cotton client missing class 0 is also supported.
    On contiguous labels the draw/order matches FedCCFA.class_balance_sample.
    """
    labels = np.asarray(targets)
    classes, counts = np.unique(labels, return_counts=True)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Balanced classifier requires nonempty training labels")
    count = min(int(counts.min()), 5)
    indices = []
    for category in classes:
        indices.extend(np.random.choice(np.flatnonzero(labels == category), count, replace=False))
    np.random.shuffle(indices)
    return indices


class BalancedBodyHeadModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.body = deepcopy(base.hidden_layers)
        self.head = deepcopy(base.fc)

    def forward(self, x, return_features=False):
        features = self.body(x)
        outputs = self.head(features)
        return (outputs, features) if return_features else outputs


class FedCAPClient(ClusterClient):
    def __init__(self, client_id, args, train_set, test_set, initial):
        super().__init__(client_id, args, train_set, test_set, initial)
        self._balanced_head_state = deepcopy(initial.head.state_dict())
        self._balanced_body_state = None
        self._head_calibration_session = None
        self.last_head_calibration_stats = None

    def set_balanced_probe_reference(self, body_state, head_state):
        # Read-only during describe; never load this head into the formal model.
        self._balanced_body_state = body_state
        self._balanced_head_state = head_state

    def train_ccfa_probe(self, round_index):
        """Mirror FedCCFA balanced SGD; isolate extra RNG from other branches."""
        seed = (self.args["seed"] + 100003 * round_index + self.id + 15485863) % (2 ** 63 - 1)
        with isolated_profile_rng(seed, self.args["device"]):
            probe = deepcopy(self.model).train()
            if self._balanced_body_state is not None:
                probe.body.load_state_dict(self._balanced_body_state)
            probe.head.load_state_dict(self._balanced_head_state)
            for parameter in probe.body.parameters():
                parameter.requires_grad_(False)
            for parameter in probe.head.parameters():
                parameter.requires_grad_(True)
            before = torch.nn.utils.parameters_to_vector(probe.head.parameters()).detach().clone()
            indices = ccfa_balanced_indices(self.train_set.targets)
            # Keep the TRAIN transform, including CIFAR/CINIC augmentation.
            loader = DataLoader(Subset(self.train_set, indices),
                                batch_size=self.args["batch_size"], shuffle=True)
            optimizer = torch.optim.SGD(probe.head.parameters(),
                                       lr=self.args.get("balanced_clf_lr", 0.1),
                                       momentum=self.args["momentum"],
                                       weight_decay=self.args["weight_decay"])
            steps = 0
            for _ in range(self.args.get("balanced_epochs", 5)):
                for x, y in loader:
                    outputs = probe(x.to(self.args["device"]))
                    loss = nn.functional.cross_entropy(outputs, y.to(self.args["device"], dtype=torch.long))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite balanced classifier probe loss")
                    optimizer.zero_grad()
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in probe.head.parameters()):
                        raise FloatingPointError("Non-finite balanced classifier probe gradient")
                    optimizer.step()
                    steps += 1
            delta = torch.nn.utils.parameters_to_vector(probe.head.parameters()).detach() - before
        counts = np.bincount(np.asarray(self.train_set.targets)[indices], minlength=self.args["num_classes"])
        return delta, dict(balanced_probe_policy=CCFA_PROBE_POLICY,
                           balanced_probe_init="server_initial_head",
                           balanced_probe_body_round=round_index,
                           balanced_probe_epochs=self.args.get("balanced_epochs", 5),
                           balanced_probe_lr=self.args.get("balanced_clf_lr", 0.1),
                           balanced_probe_steps=steps, balanced_probe_sample_counts=counts.tolist())

    def row_task_losses(self, body_state, heads, round_index):
        """Optional local receiver calibration; upload scalar summaries only.

        No training examples, labels per example, or feature vectors leave the
        client. All three heads reuse one frozen current-body forward pass.
        """
        from entities.FedCAPRowTaskGate import calibration_features, class_losses
        body = deepcopy(self.model.body).eval()
        body.load_state_dict(body_state)
        features, labels = calibration_features(self, body, round_index)
        return dict(class_losses=[class_losses(head, features, labels, self.args["num_classes"])
                                  for head in heads],
                    class_counts=np.bincount(self.train_set.targets, minlength=self.args["num_classes"]),
                    samples=len(labels))

    def describe(self, reference, round_index, version):
        if self.args.get("probe_training", "fedccfa") not in ("fedccfa", "legacy"):
            raise ValueError("Unknown balanced classifier training mode")
        independent_profile = profile_budget(self.args) > 0
        loader = self.balanced_loader(round_index)
        probe = deepcopy(reference).eval()
        for parameter in probe.body.parameters():
            parameter.requires_grad_(False)
        for parameter in probe.head.parameters():
            parameter.requires_grad_(True)

        classes = self.args["num_classes"]
        width = profile_width(probe, self.args)
        profile = torch.zeros(classes, width)
        sampled_counts = torch.zeros(classes, dtype=torch.long)
        moments = ProfileMoments(classes, width)
        raw_indices, offset = loader.dataset.indices, 0
        batches = []
        # Preserve the original profile draw and reference. Only legacy probe
        # training reuses these cached eval features.
        with torch.no_grad():
            for x, y in loader:
                features = probe.body(x.to(self.args["device"])).detach()
                probabilities = profile_values(probe, features, self.args)
                for label in y.unique():
                    mask = y == label
                    profile[label] += probabilities[mask].sum(0)
                    sampled_counts[label] += mask.sum()
                moments.update(probabilities.numpy(), y.numpy(), raw_indices[offset:offset + len(y)])
                offset += len(y)
                batches.append((features, y.to(self.args["device"], dtype=torch.long)))
        profile /= sampled_counts.clamp_min(1)[:, None]
        before = torch.nn.utils.parameters_to_vector(probe.head.parameters()).detach().clone()
        if self.args.get("probe_training", "fedccfa") == "fedccfa":
            delta, probe_metadata = self.train_ccfa_probe(round_index)
        else:
            delta = self.train_legacy_probe(probe, batches, before)
            probe_metadata = dict(balanced_probe_policy=LEGACY_PROBE_POLICY,
                                  balanced_probe_init="reference_head",
                                  balanced_probe_steps=self.args["probe_steps"],
                                  balanced_probe_sample_counts=sampled_counts.tolist())
        if not torch.isfinite(delta).all() or not torch.isfinite(profile).all():
            raise FloatingPointError("Non-finite balanced classifier descriptor")
        actual_counts = np.bincount(self.train_set.targets, minlength=classes)
        metadata = dict(**moments.metadata(), profile_sample_counts=sampled_counts.tolist(),
                        profile_sample_policy=LEGACY_POLICY, profile_extra_forward_seconds=0.0)
        valid = (actual_counts >= self.args["min_class_samples"]) & (sampled_counts.numpy() > 0)
        row_evidence = {}
        if independent_profile:
            # Extra forward passes use the original PROFILE reference, never
            # the new balanced head or current-body training copy.
            row_profile, metadata = measure_unique_profile(self, reference, round_index)
            row_evidence = dict(row_profile=row_profile,
                                row_valid=np.asarray(metadata["profile_unique_counts"]) >= self.args["min_class_samples"])
        return {"profile": profile.numpy(), "valid": valid, **row_evidence,
                "delta": delta.cpu().numpy(), "round": round_index, "version": version,
                "profile_body_round": (round_index if self.args.get("probe_training", "fedccfa") == "fedccfa"
                                       else None),
                **measurement_policies(self.args), **profile_metadata(self.args),
                "class_counts": actual_counts.tolist(), "sample_counts": sampled_counts.tolist(),
                "probe_sample_counts": probe_metadata["balanced_probe_sample_counts"],
                **metadata, **probe_metadata,
                "probe_type": "balanced_classifier", "probe_init": "reference_head"}

    def train_legacy_probe(self, probe, batches, before):
        """Historical kernel retained for controlled old/new comparisons."""
        optimizer = torch.optim.SGD(probe.head.parameters(), lr=self.args["probe_lr"],
                                    momentum=self.args["momentum"], weight_decay=self.args["weight_decay"])
        # C0 is this reference version's head, identical for every client.
        # Only the detached head copy is adapted, for exactly probe_steps updates.
        probe.head.train()
        for step in range(self.args["probe_steps"]):
            features, labels = batches[step % len(batches)]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(probe.head(features), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite balanced classifier probe loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in probe.head.parameters()):
                raise FloatingPointError("Non-finite balanced classifier probe gradient")
            optimizer.step()
        delta = torch.nn.utils.parameters_to_vector(probe.head.parameters()).detach() - before
        return delta


    def merge_calibration_losses(self, body_state, original_head, candidate_head, round_index):
        """Compare two heads on the same balanced TRAIN sample and current global body.

        Copies only; eval/no_grad and RNG restoration leave local training intact.
        The deterministic sampler reuses this round's probe indices (including
        repeats of scarce classes). These are not held-out validation examples.
        """
        device = torch.device(self.args["device"])
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            model = deepcopy(self.model).eval()
            model.body.load_state_dict(body_state)
            model.head.load_state_dict(original_head)
            candidate = deepcopy(model.head)
            candidate.load_state_dict(candidate_head)
            original_sum, candidate_sum, samples = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in self.balanced_loader(round_index):
                    features = model.body(x.to(device))
                    labels = y.to(device, dtype=torch.long)
                    original_sum += nn.functional.cross_entropy(model.head(features), labels, reduction="sum").item()
                    candidate_sum += nn.functional.cross_entropy(candidate(features), labels, reduction="sum").item()
                    samples += len(labels)
            return dict(original_loss=original_sum / samples if samples else float("inf"),
                        candidate_loss=candidate_sum / samples if samples else float("inf"), samples=samples)

    def train(self, round_index):
        self._training_round_index = round_index
        mu = validate_mu(self.args.get("body_prox_mu", 0.0))
        ratio = validate_ratio(self.args.get("head_calibration_ratio", 0.0))
        if self._head_calibration_session is not None:
            raise RuntimeError("Previous round's head calibration was not completed")
        proximal = BodyProximal(self.model, mu) if mu > 0 else None
        self.last_body_prox_stats = None
        self.last_head_calibration_stats = None
        observer = getattr(self, "_mechanism_diagnostics", None)
        if observer is not None:
            observer.phase(self, "before_head_training")
        loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        if ratio > 0:
            session = HeadCalibrationSession(self, loader, round_index)
            self._head_calibration_session = session
            losses = (self.optimize(self.model, "head", session.take(session.pre),
                                    steps=session.pre, optimizer=session.optimizer) if session.pre else [])
            session.stats.update(pre_steps=len(losses), pre_loss_sum=sum(losses))
        else:
            losses = self.optimize(self.model, "head", loader, self.args["clf_epochs"])
        if observer is not None:
            observer.phase(self, "after_head_training")
        if proximal is None:
            # Preserve the old loss, SGD calls and RNG path exactly at mu=0.
            losses += self.optimize(self.model, "body", loader, self.args["rep_epochs"])
        else:
            losses += self.optimize(self.model, "body", loader, self.args["rep_epochs"], body_proximal=proximal)
            self.last_body_prox_stats = proximal.summary(self.model)
        if observer is not None:
            observer.phase(self, "after_body_training")
        if ratio > 0:
            session.losses.extend(losses)
        return float(np.mean(losses)) if losses else 0.0

    def calibrate_head(self, body_state):
        session = self._head_calibration_session
        if session is None or session.round_index != self._training_round_index:
            raise RuntimeError("Missing current-round head calibration session")
        started = time.perf_counter()
        self.model.body.load_state_dict(body_state)
        losses = self.optimize(self.model, "head", session.take(session.post),
                               steps=session.post, optimizer=session.optimizer, freeze_body_eval=True)
        session.losses.extend(losses)
        session.stats.update(post_steps=len(losses), post_loss_sum=sum(losses),
                             seconds=time.perf_counter() - started,
                             extra_body_download_bytes=state_bytes(body_state),
                             deferred_head_upload_bytes=state_bytes(self.model.head.state_dict()))
        calibration_summary([session.stats])  # Exact per-client, not just aggregate, budget check.
        self.last_head_calibration_stats = dict(session.stats)
        self.last_calibrated_train_loss = float(np.mean(session.losses))
        self._head_calibration_session = None  # Release optimizer/loader before the next round.



class FedCAPServer(ClusterServer):
    implementation = "fedcap_shared_body_v3"
    cluster_identity_policy = "stable_overlap_v1"

    def __init__(self, args):
        self.args = args
        self.body_aggregation_step = validate_body_aggregation_step(args.get("body_aggregation_step", 1.0))
        self.class_support_alpha = validate_class_support(args)
        self.class_support_scope = validate_support_scope(args)
        self.class_support_count_cap = args.get("class_support_count_cap", CLASS_SUPPORT_CAP)
        self.class_support_round = None
        self._class_support_counts = {}
        self.last_class_support_stats = {}
        # The untouched FedCCFA runner constructs N client models before its
        # server. Consume the same CPU initialization draws, without retaining
        # unused models, so the server starts from identical weights/RNG state.
        for _ in range(args["client_num"]):
            get_model(args["dataset"], args["model"])
        self.initial = BalancedBodyHeadModel(get_model(args["dataset"], args["model"])).to(args["device"])
        self.reference = deepcopy(self.initial)
        self.body_state = deepcopy(self.initial.body.state_dict())
        # states contains heads only; bodies can never diverge across clusters.
        self.states = {0: deepcopy(self.initial.head.state_dict())}
        self.membership = {i: 0 for i in range(args["client_num"])}
        self.retained_heads = {}
        self.cache, self.version, self.next_id = {}, 0, 1
        self.head_weights = args.get("head_weights", "uniform")
        if self.head_weights not in ("uniform", "samples"):
            raise ValueError("head_weights must be uniform or samples")
        self.reference_strategy = args.get("reference_strategy", "largest_cluster"
                                           if args.get("probe_training", "fedccfa") == "fedccfa" else "average_reset")
        if self.reference_strategy not in ("average_reset", "largest_cluster", "fixed"):
            raise ValueError("Unknown reference_strategy")
        self.reference_source_cluster = None
        self.reference_source_client = None
        self.measurement_policies = measurement_policies(args)
        self.last_regroup_stats = {}
        self.collaboration_policy = RegroupConfirmation.validate_policy(args.get("collaboration_policy", "legacy"))
        self.confirmation = RegroupConfirmation(args.get("regroup_confirmations", 1), self.collaboration_policy)
        if self.reference_strategy != "fixed" and self.confirmation.required > args["reference_refresh"]:
            raise ValueError("regroup_confirmations cannot exceed the reference refresh interval")
        self.regroup_diagnostics = dict(reference_refresh_events=0,
                                       discarded_stale_descriptors=0,
                                       deferred_confirmation_rounds=0,
                                       confirmed_transition_rounds=0,
                                       committed_split_events=0, committed_merge_events=0)
        if self.collaboration_policy == "fresh-verified":
            self.regroup_diagnostics.update(topology_fresh_checked_rounds=0,
                                           topology_fresh_rejected_rounds=0)
        if self.confirmation.required > 1:
            self.implementation = "fedcap_shared_body_v3_stable_v1"
        self.merge_guard = MergeLossGuard(args.get("merge_guard", "none"),
                                         args.get("merge_loss_tolerance", 0.05),
                                         args.get("merge_min_witnesses", 2))
        if self.merge_guard.mode != "none":
            self.implementation = self.merge_guard.implementation
        self.regroup_diagnostics.update(merge_guard_checked_rounds=0, merge_guard_blocked_loss_rounds=0,
                                       merge_guard_deferred_evidence_rounds=0, merge_guard_evaluated_clients=0)
        self.profile_merge_guard = make_profile_merge_guard(args)
        if self.profile_merge_guard.mode != "none":
            if args.get("algorithm", "FedCAP") != "FedCAP" or self.merge_guard.mode != "none":
                raise ValueError("Profile merge guard requires FedCAP without merge loss guard")
            self.implementation += self.profile_merge_guard.implementation_suffix
        self.profile_merge_diagnostics = dict(profile_merge_checked_rounds=0,
                                              profile_merge_blocked_rounds=0,
                                              profile_merge_blocked_candidate_events=0)
        if self.profile_merge_guard.mode == "evidence":
            self.profile_merge_diagnostics.update(profile_merge_unknown_pair_events=0,
                                                 profile_merge_memory_blocked_pair_events=0,
                                                 profile_merge_compatible_pair_events=0,
                                                 profile_merge_conflicting_pair_events=0,
                                                 profile_merge_memory_released_pair_events=0,
                                                 profile_merge_memory_expired_pair_events=0)
        if self.class_support_alpha > 0:
            self.implementation += implementation_suffix(self.class_support_count_cap)
        if self.class_support_scope == "cross-cluster":
            self.implementation += CROSS_ONLY_SUFFIX
        if self.collaboration_policy == "fresh-verified":
            self.implementation += "_fresh_verified_v2"

        if self.body_aggregation_step != 1.0:
            self.implementation += BODY_STEP_SUFFIX

    @staticmethod
    def match_cluster_ids(groups, old_members, old_ids):
        """Maximize member overlap without treating private heads as clusters.

        Labels produced by clustering are temporary. Match all groups jointly
        to persistent identities; zero-overlap matches must get a new identity.
        No cached descriptors from another reference version are compared.
        """
        old_ids = sorted(old_ids)
        if not groups or not old_ids:
            return {}
        overlap = np.array([[sum(old_members[i] == identity for i in group)
                             for identity in old_ids] for group in groups], dtype=np.int64)
        positions, columns = linear_sum_assignment(-overlap)
        return {int(position): old_ids[column] for position, column in zip(positions, columns)
                if overlap[position, column] > 0}

    def refresh_reference(self, clients, round_index):
        # Head-free measurement uses the current body, never an anchor head.
        if profile_head(self.args) == "none":
            return
        if self.reference_strategy == "fixed" or not round_index or round_index % self.args["reference_refresh"]:
            return
        if self.reference_strategy == "average_reset":
            head = average_states([self.head_state(c.id) for c in clients],
                                  [len(c.train_set) for c in clients])
            self.reference_source_cluster = None
            self.reference_source_client = None
        else:
            masses = {identity: 0 for identity in self.states}
            for client in clients:
                if client.id not in self.retained_heads:
                    masses[self.membership[client.id]] += len(client.train_set)
            if masses:
                identity = max(masses, key=lambda key: (masses[key], -key))
                head = self.states[identity]
                self.reference_source_cluster = identity
                self.reference_source_client = None
            else:
                # All clients may be noise. There is still a valid common probe
                # reference, but no inferred concept cluster to select.
                # Never average conflicting personal heads as a fallback for
                # the single-head anchor. Select by TRAIN mass, not accuracy.
                client = max(clients, key=lambda c: (len(c.train_set), -c.id))
                head = self.head_state(client.id)
                self.reference_source_cluster = None
                self.reference_source_client = client.id
        self.reference.body.load_state_dict(self.body_state)
        self.reference.head.load_state_dict(head)
        self.version += 1
        self.cache.clear()

    def filter_label_merges(self, groups, selected, round_index):
        """Optional row-correspondence veto, before constructing merged heads."""
        return groups, {}

    def partition_descriptors(self, clients, ids):
        """Grouping hook; the fixed-source Cotton control changes only this policy."""
        return FedCAPCore.cluster_descriptors([self.cache[i] for i in ids], self.args)

    def regroup(self, clients, selected, round_index):
        self.class_support_round = round_index
        fresh_measurements = self.args.get("probe_training", "fedccfa") == "fedccfa"
        stale = (sum(d.get("round") != round_index for d in self.cache.values())
                 if fresh_measurements else 0)
        previous_version = self.version
        self.refresh_reference(clients, round_index)
        refreshed = self.version != previous_version
        if fresh_measurements:
            # Classifier updates on B_t cannot be compared with updates on
            # B_(t-1), even when the profile anchor-head version is unchanged.
            # Keep historical memberships/heads, not obsolete measurements.
            self.cache = {i: d for i, d in self.cache.items() if d.get("round") == round_index}
            self.reference.body.load_state_dict(self.body_state)
        for client in selected:
            if fresh_measurements and isinstance(client, FedCAPClient):
                client.set_balanced_probe_reference(self.body_state, self.initial.head.state_dict())
            self.cache[client.id] = client.describe(self.reference, round_index, self.version)
            if fresh_measurements:
                descriptor = self.cache[client.id]
                # Mock clients may omit tags, but a supplied mismatched tag
                # must never silently enter the clustering matrix.
                for key in ("balanced_probe_body_round", "profile_body_round"):
                    if key in descriptor and descriptor[key] != round_index:
                        raise ValueError(f"Stale FedCAP {key}: client={client.id} round={round_index}")
        self.cache = {i: d for i, d in self.cache.items()
                      if d["version"] == self.version and
                      (d["round"] == round_index if fresh_measurements else
                       0 <= round_index - d["round"] <= self.args["descriptor_ttl"])}
        ids = sorted(self.cache)
        distance_stats = (FedCAPCore.descriptor_scale_stats([self.cache[i] for i in ids])
                          if all(all(key in self.cache[i] for key in ("profile", "valid", "delta"))
                                 for i in ids) else {})
        labels, score = self.partition_descriptors(clients, ids)
        if np.asarray(labels).shape != (len(ids),):
            raise ValueError("FedCAP cluster labels must match the current descriptor count")
        groups = [set(ids[j] for j in np.flatnonzero(labels == label))
                  for label in np.unique(labels) if label >= 0]
        groups.sort(key=lambda group: (-len(group), min(group)))
        raw_candidates = len(groups)
        groups, profile_merge_stats = self.profile_merge_guard.filter_groups(
            groups, self.membership, {c.id for c in selected}, self.cache,
            round_index, self.version, self.args["num_classes"])
        groups, label_merge_stats = self.filter_label_merges(groups, selected, round_index)
        noise = {ids[j] for j in np.flatnonzero(labels < 0)}
        assigned = set().union(*groups) if groups else set()
        old_members = self.membership.copy()
        old_retained = set(self.retained_heads)
        heads = {c.id: self.head_state(c.id) for c in clients}
        sizes = {c.id: len(c.train_set) for c in clients}
        proposed_members, proposed_states = old_members.copy(), self.states.copy()
        retained = self.retained_heads.copy()
        matches = self.match_cluster_ids(groups, old_members, self.states)
        next_id = self.next_id

        # Splits/merges must not pass a new concept's averaged head to an
        # unobserved member just because its old numeric identity was reused.
        destinations, merge_positions, affected = {}, [], set()
        for position, group in enumerate(groups):
            # A private head returning to a cluster is not a concept merge.
            # Only simultaneous membership in distinct old clusters counts.
            sources = {old_members[i] for i in group if old_members[i] is not None}
            if len(sources) > 1:
                merge_positions.append(position)
                affected.update(sources)
            for i in group:
                identity = old_members[i]
                if identity is not None:
                    destinations.setdefault(identity, set()).add(position)
        split_ids = {i for i, positions in destinations.items() if len(positions) > 1}
        affected.update(split_ids)
        for client in clients:
            i = client.id
            if i in noise or (i not in assigned and old_members[i] in affected):
                retained[i] = deepcopy(heads[i])
                proposed_members[i] = None

        candidate_heads = {}
        for position, group in enumerate(groups):
            identity = matches.get(position)
            if identity is None:
                identity, next_id = next_id, next_id + 1
            ordered = sorted(group)
            # Avoid averaging many identical copies on an unchanged cluster.
            if len({self.model_key(i) for i in group}) == 1:
                proposed_states[identity] = deepcopy(heads[ordered[0]])
            else:
                weights = [1 if self.head_weights == "uniform" else sizes[i] for i in ordered]
                proposed_states[identity] = average_states([heads[i] for i in ordered], weights)
            candidate_heads[position] = proposed_states[identity]
            for i in group:
                proposed_members[i] = identity
                retained.pop(i, None)
        used = {identity for identity in proposed_members.values() if identity is not None}
        proposed_states = {i: head for i, head in proposed_states.items() if i in used}
        maximum = self.args["max_total_clusters"]
        cap_allowed = len(proposed_states) <= maximum and len(groups) <= self.args["max_candidate_clusters"]
        new_groups = any(all(old_members[i] is None for i in group) for group in groups)
        changed = bool(split_ids or merge_positions or new_groups)
        fresh_check = {}
        if self.collaboration_policy == "fresh-verified":
            fresh_check = dict(topology_fresh_checked=False, topology_fresh_verified=True,
                               topology_fresh_reason="unchanged" if not changed else "cluster-cap")
            if changed and cap_allowed:
                # The cached proposal is NOT re-used as its own confirmation.
                # Same existing algorithm/signals, this round's descriptors only.
                fresh_ids = sorted(c.id for c in selected if c.id in self.cache and
                                   self.cache[c.id]["round"] == round_index and
                                   self.cache[c.id]["version"] == self.version)
                fresh_labels, fresh_score = FedCAPCore.cluster_descriptors(
                    [self.cache[i] for i in fresh_ids], self.args)
                if np.asarray(fresh_labels).shape != (len(fresh_ids),):
                    raise ValueError("Fresh cluster labels must match the descriptor count")
                fresh_check = self.confirmation.verify_fresh_partition(
                    groups, old_members, {i: int(label) for i, label in zip(fresh_ids, fresh_labels)})
                fresh_check.update(topology_fresh_clients=len(fresh_ids),
                                   topology_fresh_clusters=len({int(label) for label in fresh_labels if label >= 0}),
                                   topology_fresh_silhouette=float(fresh_score))
                self.regroup_diagnostics["topology_fresh_checked_rounds"] += 1
                self.regroup_diagnostics["topology_fresh_rejected_rounds"] += int(
                    not fresh_check["topology_fresh_verified"])
        confirmation = self.confirmation.observe(
            groups, {c.id for c in selected}, self.version, round_index,
            changed, cap_allowed,
            fresh_verified=fresh_check.get("topology_fresh_verified", True))
        accepted = cap_allowed and confirmation["transition_confirmed"]
        guard = self.merge_guard.result("disabled" if self.merge_guard.mode == "none" else "awaiting-topology")
        if accepted:
            guard = self.merge_guard.check(groups, merge_positions, old_members, selected,
                                           self.body_state, heads, candidate_heads, round_index)
            accepted = guard["merge_guard_passed"]
            if not accepted:
                self.confirmation.pending = None
        if accepted:
            self.membership, self.states, self.next_id = proposed_members, proposed_states, next_id
            self.retained_heads = retained
        else:
            # Defer the proposed split/merge atomically. Noise isolation is
            # independent of that transition and must not contaminate old heads.
            for i in noise:
                self.retained_heads[i] = deepcopy(heads[i])
                self.membership[i] = None
            used = {i for i in self.membership.values() if i is not None}
            self.states = {i: head for i, head in self.states.items() if i in used}
        current_retained = set(self.retained_heads)
        detached = sorted(current_retained - old_retained)
        rejoined = sorted(old_retained - current_retained)
        pair_changes, compared_pairs = RegroupConfirmation.relation_changes(groups, old_members)
        ages = [round_index - self.cache[i]["round"] for i in ids]
        status = "accept" if accepted else "defer-cap" if not cap_allowed else "defer-confirmation"
        if cap_allowed and not fresh_check.get("topology_fresh_verified", True):
            status = "defer-fresh-partition"
        if guard["merge_guard_checked"] and not guard["merge_guard_passed"]:
            status = "defer-merge-evidence" if guard["merge_guard_reason"] == "insufficient-fresh-witnesses" else "defer-merge-loss"
        self.regroup_diagnostics["reference_refresh_events"] += int(refreshed)
        self.regroup_diagnostics["discarded_stale_descriptors"] += stale
        self.regroup_diagnostics["deferred_confirmation_rounds"] += int(status == "defer-confirmation")
        self.regroup_diagnostics["confirmed_transition_rounds"] += int(accepted and confirmation["topology_change_proposed"])
        self.regroup_diagnostics["committed_split_events"] += len(split_ids) if accepted else 0
        self.regroup_diagnostics["committed_merge_events"] += len(merge_positions) if accepted else 0
        self.regroup_diagnostics["merge_guard_checked_rounds"] += int(guard["merge_guard_checked"])
        self.regroup_diagnostics["merge_guard_blocked_loss_rounds"] += int(status == "defer-merge-loss")
        self.regroup_diagnostics["merge_guard_deferred_evidence_rounds"] += int(status == "defer-merge-evidence")
        self.regroup_diagnostics["merge_guard_evaluated_clients"] += guard["merge_guard_evaluated_clients"]
        self.last_regroup_stats = dict(
            candidate_clusters=len(groups), proposed_total_clusters=len(proposed_states),
            regroup_accepted=accepted, max_total_clusters=maximum,
            reference_strategy=self.reference_strategy, reference_source_cluster=self.reference_source_cluster,
            reference_source_client=self.reference_source_client,
            cluster_method=self.args.get("cluster_method", "agglomerative"),
            noise_clients=len(noise), noise_client_ids=sorted(noise), noise_policy="retain_personal_head",
            retained_heads=len(self.retained_heads), retained_client_ids=sorted(self.retained_heads),
            detached_heads=len(detached), detached_client_ids=detached,
            rejoined_heads=len(rejoined), rejoined_client_ids=rejoined,
            split_events=len(split_ids) if accepted else 0,
            merge_events=len(merge_positions) if accepted else 0,
            proposed_split_cluster_ids=sorted(split_ids), proposed_merge_groups=merge_positions,
            cluster_identity_policy=self.cluster_identity_policy,
            probe_type="balanced_classifier", probe_init="reference_head", body_aggregation="global",
            head_aggregation="cluster", head_weights=self.head_weights, cluster_assignment="inferred",
            implementation=self.implementation, initialization="fedccfa_order")
        self.last_regroup_stats.update(
            **self.measurement_policies,
            **profile_metadata(self.args),
            discarded_stale_descriptors_this_round=stale,
            measurement_body_round=round_index if fresh_measurements else None,
            probe_training=self.args.get("probe_training", "fedccfa"),
            balanced_probe_policy=(CCFA_PROBE_POLICY if self.args.get("probe_training", "fedccfa") == "fedccfa"
                                   else LEGACY_PROBE_POLICY),
            **confirmation, regroup_status=status, regroup_confirmation_policy=self.confirmation.policy,
            reference_refreshed=refreshed, cache_coverage=len(ids) / len(clients),
            fresh_descriptors=len(selected), cached_descriptor_mean_age=float(np.mean(ages)) if ages else 0.0,
            proposal_pair_changes=pair_changes, proposal_compared_pairs=compared_pairs,
            proposal_pair_disagreement=pair_changes / compared_pairs if compared_pairs else 0.0)
        self.last_regroup_stats.update(distance_stats)
        self.last_regroup_stats.update(fresh_check)
        self.last_regroup_stats.update(guard)
        if profile_merge_stats:
            blocked = profile_merge_stats["profile_merge_blocked_candidates"]
            self.profile_merge_diagnostics["profile_merge_checked_rounds"] += int(
                profile_merge_stats["profile_merge_checked_candidates"] > 0)
            self.profile_merge_diagnostics["profile_merge_blocked_rounds"] += int(blocked > 0)
            self.profile_merge_diagnostics["profile_merge_blocked_candidate_events"] += blocked
            if self.profile_merge_guard.mode == "evidence":
                self.profile_merge_diagnostics["profile_merge_unknown_pair_events"] += profile_merge_stats["profile_merge_unknown_source_pairs"]
                self.profile_merge_diagnostics["profile_merge_memory_blocked_pair_events"] += profile_merge_stats["profile_merge_memory_blocked_source_pairs"]
                for event, statistic in (
                        ("compatible_pair", "compatible_source_pairs"),
                        ("conflicting_pair", "conflicting_source_pairs"),
                        ("memory_released_pair", "memory_released_pairs"),
                        ("memory_expired_pair", "memory_expired_pairs")):
                    self.profile_merge_diagnostics[f"profile_merge_{event}_events"] += profile_merge_stats[f"profile_merge_{statistic}"]
            self.last_regroup_stats.update(profile_merge_stats, raw_candidate_clusters=raw_candidates)
            if blocked:
                self.last_regroup_stats["regroup_status"] += "+profile-filter"
        if label_merge_stats:
            self.last_regroup_stats.update(label_merge_stats, raw_candidate_clusters=raw_candidates)
            if label_merge_stats["label_merge_blocked_candidates"]:
                self.last_regroup_stats["regroup_status"] += "+label-filter"
        return float(score), len(self.cache)

    def regroup_summary(self):
        result = dict(self.regroup_diagnostics, regroup_confirmations=self.confirmation.required,
                    regroup_confirmation_policy=self.confirmation.policy,
                    merge_guard=self.merge_guard.mode, merge_guard_policy=self.merge_guard.policy,
                    merge_loss_tolerance=self.merge_guard.tolerance, merge_min_witnesses=self.merge_guard.min_witnesses)
        result.update(body_aggregation_step=self.body_aggregation_step,
                      body_aggregation_policy=body_aggregation_policy(self.body_aggregation_step))
        result.update(probe_training=self.args.get("probe_training", "fedccfa"),
                      **self.measurement_policies,
                      reference_strategy=self.reference_strategy,
                      reference_source_cluster=self.reference_source_cluster,
                      reference_source_client=self.reference_source_client,
                      balanced_probe_policy=(CCFA_PROBE_POLICY if self.args.get("probe_training", "fedccfa") == "fedccfa"
                                             else LEGACY_PROBE_POLICY))
        if self.collaboration_policy == "fresh-verified":
            result["collaboration_policy"] = self.collaboration_policy
        if self.profile_merge_guard.mode != "none":
            result.update(self.profile_merge_diagnostics, profile_merge_guard=self.profile_merge_guard.mode,
                          profile_merge_policy=self.profile_merge_guard.policy,
                          profile_merge_eps=self.profile_merge_guard.epsilon,
                          profile_merge_min_witnesses=self.profile_merge_guard.min_witnesses)
            if self.profile_merge_guard.mode == "evidence":
                result.update(profile_merge_min_coverage=self.profile_merge_guard.min_coverage,
                              profile_merge_memory_rounds=self.profile_merge_guard.memory_rounds,
                              profile_merge_memory_pairs=len(self.profile_merge_guard.conflict_memory))
        return result

    def head_state(self, client_id):
        if client_id in self.retained_heads:
            return self.retained_heads[client_id]
        return self.states[self.membership[client_id]]

    def model_key(self, client_id):
        # Retained clients need separate evaluations, even if their previous
        # cluster identities happened to agree.
        return ("retained", client_id) if client_id in self.retained_heads else ("cluster", self.membership[client_id])

    def cluster_count(self):
        return len(self.states)

    def aggregate(self, selected):
        if not selected:
            return
        support_alpha = getattr(self, "class_support_alpha", 0.0)
        if support_alpha > 0:
            active = [c for c in selected if c.id not in self.retained_heads and
                      self.membership[c.id] is not None]
            self._class_support_counts = current_class_counts(
                active, self.cache, self.class_support_round, self.version, self.args["num_classes"])
            self.last_class_support_stats = dict(
                alpha=self.class_support_alpha, policy=support_policy(self.class_support_count_cap),
                count_cap=self.class_support_count_cap, count_source="fresh_full_train_class_counts",
                cluster_rows=0, cross_cluster_rows=0, zero_support_rows=0, max_donor_weight=0.0)
            if self.class_support_scope == "cross-cluster":
                self.last_class_support_stats.update(scope=self.class_support_scope, cluster_whole_heads=0)
        if self.body_aggregation_step == 1.0:
            # Keep the old full-replacement arithmetic exactly, not an
            # equivalent old + 1 * (mean - old) with different rounding.
            self.body_state = average_states([c.model.body.state_dict() for c in selected],
                                             [len(c.train_set) for c in selected])
        else:
            averaged_body = average_states([c.model.body.state_dict() for c in selected],
                                            [len(c.train_set) for c in selected])
            self.body_state = stepped_body_state(self.body_state, averaged_body,
                                                 self.body_aggregation_step)
        if self.args.get("head_calibration_ratio", 0) > 0:
            observer = getattr(self, "_stage_diagnostics", None)
            if observer is not None:
                observer.observe_body_aggregation(self, selected)
            context = getattr(self, "_head_calibration_context", None)
            for client in selected:
                if context is not None:
                    context.update(phase="head_calibration", client_id=client.id)
                client.calibrate_head(self.body_state)
                if observer is not None:
                    observer.observe_head_calibration(self, client)
            self.last_head_calibration_stats = calibration_summary(c.last_head_calibration_stats for c in selected)
            if context is not None:
                context.update(phase="head_aggregation", client_id=None)
        for identity in self.states:
            group = [c for c in selected if c.id not in self.retained_heads and self.membership[c.id] == identity]
            if group:
                if support_alpha == 0 or self.class_support_scope == "cross-cluster":
                    weights = [1 if self.head_weights == "uniform" else len(c.train_set) for c in group]
                    self.states[identity] = average_states([c.model.head.state_dict() for c in group], weights)
                    if support_alpha > 0:
                        self.last_class_support_stats["cluster_whole_heads"] += 1
                else:
                    head = deepcopy(group[0].model.head.state_dict())
                    if set(head) != {"weight", "bias"}:
                        raise ValueError("Class support requires an affine weight/bias head")
                    for category in range(self.args["num_classes"]):
                        rows = [{key: c.model.head.state_dict()[key][category]
                                 for key in ("weight", "bias")} for c in group]
                        weights = self.support_weights(group, category, "cluster")
                        row = average_states(rows, weights)
                        for key in ("weight", "bias"):
                            head[key][category].copy_(row[key])
                    self.states[identity] = head
        for client in selected:
            if client.id in self.retained_heads:
                self.retained_heads[client.id] = deepcopy(client.model.head.state_dict())

    def support_weights(self, clients, category, scope):
        counts = [self._class_support_counts[c.id][category] for c in clients]
        weights = class_support_weights(counts, self.class_support_alpha, self.class_support_count_cap)
        stats = self.last_class_support_stats
        stats[scope + "_rows"] += 1
        stats["zero_support_rows"] += int(not any(counts))
        stats["max_donor_weight"] = max(stats["max_donor_weight"], float(weights.max()))
        return weights.tolist()

    def sync(self, clients):
        for client in clients:
            client.model.body.load_state_dict(self.body_state)
            client.model.head.load_state_dict(self.head_state(client.id))

    def checkpoint(self):
        cpu = lambda state: {k: value.detach().cpu().clone() for k, value in state.items()}
        return dict(args=self.args, format=self.implementation,
                    cluster_identity_policy=self.cluster_identity_policy, global_body=cpu(self.body_state),
                    cluster_head_states={i: cpu(head) for i, head in self.states.items()},
                    retained_heads={i: cpu(head) for i, head in self.retained_heads.items()},
                    membership=self.membership.copy(), next_id=self.next_id, reference_version=self.version,
                    reference_state=cpu(self.reference.state_dict()),
                    balanced_probe_initial_head=cpu(self.initial.head.state_dict()),
                    regroup_confirmation=self.confirmation.checkpoint(),
                    merge_guard=self.merge_guard.checkpoint(),
                    profile_merge_guard=self.profile_merge_guard.checkpoint(),
                    class_support_alpha=self.class_support_alpha,
                    class_support_scope=self.class_support_scope,
                    class_support_count_cap=self.class_support_count_cap,
                    class_support_stats=self.last_class_support_stats.copy(),
                    regroup_diagnostics=self.regroup_summary())
