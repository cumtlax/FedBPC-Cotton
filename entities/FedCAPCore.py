from copy import copy, deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from utils.fedcap_numerics import TrainingNumerics


def average_states(states, weights):
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    return {k: sum(s[k] * float(w) for s, w in zip(states, weights)) for k in states[0]}


def balanced_indices(targets, budget, classes, rng):
    # Constant total budget; absent classes masked, scarce classes sampled with replacement.
    targets = np.asarray(targets)
    present = [c for c in range(classes) if np.any(targets == c)]
    if budget < len(present) or not present:
        raise ValueError("Budget must cover all present classes")
    indices = []
    for j, c in enumerate(present):
        count = budget // len(present) + (j < budget % len(present))
        pool = np.flatnonzero(targets == c)
        indices.extend(rng.choice(pool, count, replace=count > len(pool)).tolist())
    rng.shuffle(indices)
    return indices


class ClusterClient:
    def __init__(self, client_id, args, train_set, test_set, initial):
        self.id, self.args = client_id, args
        self.train_set, self.test_set = train_set, test_set
        self.global_test_id = getattr(train_set, "concept_id", 0)
        self.model = deepcopy(initial)

    def balanced_loader(self, round_index):
        rng = np.random.default_rng(self.args["seed"] + 100003 * round_index + self.id)
        ids = balanced_indices(self.train_set.targets, self.args["probe_budget"], self.args["num_classes"], rng)
        # Deterministic evaluation transform, but labels always read from live training data.
        dataset = copy(self.train_set)
        if self.args["dataset"] in ("CIFAR10", "CIFAR100", "CINIC-10"):
            dataset.transform = self.test_set.transform
        return DataLoader(Subset(dataset, ids), batch_size=self.args["batch_size"], shuffle=False)

    def optimize(self, model, component, loader, epochs=1, steps=None, body_proximal=None,
                 optimizer=None, freeze_body_eval=False):
        if body_proximal is not None and component != "body":
            raise ValueError("Proximal training is restricted to the formal body")
        if freeze_body_eval and component != "head":
            raise ValueError("Frozen-body evaluation is restricted to head calibration")
        model.train()
        if freeze_body_eval:
            model.body.eval()  # Freeze BN buffers/dropout as well as parameters.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(component + "."))
        lr = self.args[{"body": "rep_lr", "head": "clf_lr"}[component]]
        if optimizer is None:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                        momentum=self.args["momentum"], weight_decay=self.args["weight_decay"])
        elif {id(p) for group in optimizer.param_groups for p in group["params"]} != {
                id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("Reused optimizer parameters must match the trained component")
        monitor = TrainingNumerics(self, model, optimizer, component, getattr(self, "_training_round_index", None))
        monitor.body_proximal = body_proximal
        losses, done = [], 0

        def loss_values():
            # Ordinary loss logging is read once per optimize() call instead
            # of synchronizing CUDA on every batch. No averaging or rounding:
            # callers still receive the original ordered list of Python floats.
            if monitor.enabled or not losses:
                return losses
            return torch.stack(losses).cpu().tolist()

        monitor.check("before_training", {**monitor.parameters(), **dict(model.named_buffers())})
        monitor.remember_initial_state()
        with monitor.capture_features():
            for epoch in range(epochs):
                for batch, (x, y) in enumerate(loader):
                    x, y = x.to(self.args["device"]), y.to(self.args["device"], dtype=torch.long)
                    monitor.context.update(epoch=epoch + 1, batch=batch + 1, step=done + 1)
                    monitor.raw_gradients = None
                    monitor.tensors = {"inputs": x.detach(), "targets": y.detach()}
                    monitor.check("inputs", monitor.tensors)
                    optimizer.zero_grad()
                    if getattr(self, "_mechanism_diagnostics", None) is not None:
                        from utils.fedcap_mechanism import rng_state
                        monitor.forward_rng = rng_state(self.args["device"])
                    logits = model(x)
                    monitor.tensors["logits"] = logits.detach()
                    monitor.check("logits", {"logits": logits})
                    loss = nn.functional.cross_entropy(logits, y)
                    if body_proximal is not None:
                        ce = loss
                        penalty = body_proximal.penalty(model)
                        monitor.tensors.update(cross_entropy=ce.detach(), body_prox_penalty=penalty.detach())
                        monitor.check("body_objective", {"cross_entropy": ce, "body_prox_penalty": penalty})
                        loss = ce + penalty
                    monitor.tensors["loss"] = loss.detach()
                    monitor.check("loss", {"loss": loss})
                    loss.backward()
                    monitor.prepare_step()
                    optimizer.step()
                    monitor.check("parameters_after_step", {**monitor.parameters(), **dict(model.named_buffers())})
                    monitor.check("momentum_after_step", monitor.momentum())
                    if body_proximal is not None:
                        body_proximal.observe(ce, penalty)
                    if monitor.enabled:
                        # Error history keeps its original per-step contents.
                        losses.append(loss.item())
                        monitor.remember(losses[-1])
                    else:
                        # Detach scalars; never retain computation graphs.
                        losses.append(loss.detach())
                    done += 1
                    if steps is not None and done >= steps:
                        return loss_values()
        return loss_values()

def _descriptor_components(descriptors):
    n = len(descriptors)
    profiles, updates = np.zeros((n, n)), np.zeros((n, n))
    missing = np.zeros((n, n), dtype=bool)
    if len({d["version"] for d in descriptors}) > 1:
        raise ValueError("Cannot compare different reference versions")
    modes = {d.get("profile_head", "reference") for d in descriptors}
    shapes = {np.asarray(d["profile"]).shape for d in descriptors}
    if len(modes) > 1 or len(shapes) > 1:
        raise ValueError("Cannot compare incompatible profile representations or shapes")
    for key in ("balanced_probe_policy", "balanced_probe_body_round", "profile_body_round"):
        tagged = [d[key] for d in descriptors if key in d]
        if tagged and (len(tagged) != n or len(set(tagged)) > 1):
            raise ValueError(f"Cannot compare incompatible {key} measurements")
    for i in range(n):
        for j in range(i):
            a, b = descriptors[i], descriptors[j]
            valid = a["valid"] & b["valid"]
            if valid.any():
                profiles[i, j] = profiles[j, i] = np.sqrt(np.mean((a["profile"][valid] - b["profile"][valid]) ** 2))
            else:
                missing[i, j] = missing[j, i] = True
            updates[i, j] = updates[j, i] = np.sqrt(np.mean((a["delta"] - b["delta"]) ** 2))
    return profiles, updates, missing


def descriptor_scale_stats(descriptors):
    """Report raw signal scales; median normalization is not confidence."""
    profiles, updates, missing = _descriptor_components(descriptors)
    upper = np.triu_indices(len(descriptors), 1)
    result = dict(clustering_profile_missing_pairs=int(missing[upper].sum()),
                  clustering_compared_pairs=len(upper[0]))
    for name, matrix in (("profile", profiles), ("delta", updates)):
        values = matrix[upper]
        positive = values[values > 1e-12]
        result[f"clustering_{name}_raw_median"] = float(np.median(positive)) if len(positive) else 0.0
    return result


def descriptor_distances(descriptors, weight):
    n = len(descriptors)
    profiles, updates, missing = _descriptor_components(descriptors)
    for matrix in ((updates,) if weight == 0 else (profiles, updates)):
        positive = matrix[np.triu_indices(n, 1)]
        positive = positive[positive > 1e-12]
        if len(positive):
            matrix /= np.median(positive)
    if weight == 0:
        # Do not let even an invalid/NaN unused profile contaminate classifier-
        # only distances via IEEE 0 * NaN. Row-sharing eligibility stays separate.
        return updates
    # At the profile-only endpoint, an unused update must not contaminate
    # valid profile pairs via 0 * NaN. Missing-profile pairs still use it.
    distance = profiles.copy() if weight == 1 else weight * profiles + (1 - weight) * updates
    # No common valid classes: use the probe update, not a fake zero profile.
    distance[missing] = updates[missing]
    return distance


def cluster_descriptors(descriptors, args):
    n = len(descriptors)
    distances = descriptor_distances(descriptors, args["profile_weight"])
    method = args.get("cluster_method", "agglomerative")
    if method not in ("agglomerative", "hdbscan"):
        raise ValueError("Unknown FedCAP cluster_method")
    best, score = np.zeros(n, dtype=int), args["silhouette_threshold"]
    if not n or distances.max() <= 1e-12:
        return best, 0.
    if method == "hdbscan":
        # No Euclidean embedding: use the same fused distances as the baseline.
        from sklearn.cluster import HDBSCAN
        minimum = args["min_cluster_size"]
        min_samples = args.get("hdbscan_min_samples", 2)
        if n < minimum or n < min_samples:
            return np.full(n, -1, dtype=int), 0.
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("HDBSCAN needs finite nonnegative distances")
        labels = HDBSCAN(metric="precomputed", min_cluster_size=minimum,
                         min_samples=min_samples, cluster_selection_method="eom",
                         allow_single_cluster=True, copy=True).fit_predict(distances)
        valid = labels >= 0
        # Diagnostic only; do not use silhouette to override density clustering.
        count = len(np.unique(labels[valid]))
        score = silhouette_score(distances[np.ix_(valid, valid)], labels[valid],
                                 metric="precomputed") if 1 < count < valid.sum() else 0.
        return labels, float(score)
    maximum = args.get("max_candidate_clusters", args.get("max_clusters", 5))
    for k in range(2, min(maximum, n - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distances)
        if np.bincount(labels).min() < args["min_cluster_size"]:
            continue
        candidate = silhouette_score(distances, labels, metric="precomputed")
        if candidate > score:
            best, score = labels, candidate
    return best, float(score) if best.max(initial=0) else 0.


class ClusterServer:
    def regroup(self, clients, selected, round_index):
        self.refresh_reference(clients, round_index)
        for client in selected:
            self.cache[client.id] = client.describe(self.reference, round_index, self.version)
        self.cache = {i: d for i, d in self.cache.items()
                      if d["version"] == self.version and round_index - d["round"] <= self.args["descriptor_ttl"]}
        ids = sorted(self.cache)
        labels, score = cluster_descriptors([self.cache[i] for i in ids], self.args)
        old_members, old_states = self.membership.copy(), self.states.copy()
        maximum = self.args.get("max_total_clusters", self.args.get("max_clusters", 5))
        if len(old_states) > maximum:
            raise ValueError("Existing clusters exceed max_total_clusters; cannot safely merge without evidence")
        unused = set(old_states)
        # -1 is uncertainty, not a shared concept cluster. Retain old assignments.
        groups = [set(ids[j] for j in np.flatnonzero(labels == label))
                  for label in np.unique(labels) if label >= 0]
        # Match stable identities by member overlap; each old identity used at most once.
        groups.sort(key=lambda group: (-len(group), min(group)))
        sizes = {c.id: len(c.train_set) for c in clients}
        proposed_members, proposed_states, next_id = old_members.copy(), old_states.copy(), self.next_id
        for group in groups:
            identity = max(unused, key=lambda k: (sum(old_members[i] == k for i in group), -k), default=None)
            if identity is None or not any(old_members[i] == identity for i in group):
                identity, next_id = next_id, next_id + 1
            else:
                unused.remove(identity)
            # A split retains its parent state; a merge averages previous states of its members.
            proposed_states[identity] = average_states([old_states[old_members[i]] for i in sorted(group)],
                                                       [sizes[i] for i in sorted(group)])
            for i in group:
                proposed_members[i] = identity
        proposed_states = {k: v for k, v in proposed_states.items() if k in set(proposed_members.values())}
        candidate_limit = self.args.get("max_candidate_clusters", self.args.get("max_clusters", 5))
        accepted = len(proposed_states) <= maximum and len(groups) <= candidate_limit
        if accepted:
            self.membership, self.states, self.next_id = proposed_members, proposed_states, next_id
        # Reject the entire transition if retained old clusters plus new clusters
        # exceed the cap. Keep training old clusters; never force-merge stale ones.
        self.last_regroup_stats = {"candidate_clusters": len(groups), "proposed_total_clusters": len(proposed_states),
                                   "regroup_accepted": accepted, "max_total_clusters": maximum,
                                   "reference_strategy": self.reference_strategy,
                                   "cluster_method": self.args.get("cluster_method", "agglomerative"),
                                   "noise_clients": int(np.sum(labels < 0)),
                                   "noise_client_ids": [ids[j] for j in np.flatnonzero(labels < 0)],
                                   "noise_policy": "retain_previous",
                                   "reference_source_cluster": self.reference_source_cluster}
        return score, len(self.cache)

    def aggregate(self, selected):
        for identity in set(self.membership[c.id] for c in selected):
            group = [c for c in selected if self.membership[c.id] == identity]
            self.states[identity] = average_states([c.model.state_dict() for c in group],
                                                   [len(c.train_set) for c in group])

    def sync(self, clients):
        for client in clients:
            client.model.load_state_dict(self.states[self.membership[client.id]])
