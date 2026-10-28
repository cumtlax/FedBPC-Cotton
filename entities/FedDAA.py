"""FedDAA port of the authors' NCD, RDLD and FedRC adaptation routines.

Source: LeegerPENG/FedDAA-Dynamic-Client-Clustering-for-Concept-Drift-Adaptation-in-Federated-Learning
Commit: f79e6165f030cd4569b28e8877cddea03e06e869 (Apache-2.0).
Unmodified sources and license: third_party/feddaa/upstream/.
The unified-protocol changes are documented in third_party/feddaa/INTEGRATION.md.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset


UPSTREAM_COMMIT = "f79e6165f030cd4569b28e8877cddea03e06e869"


@contextmanager
def isolated_rng(device="cpu"):
    """Read-only panels/evaluation must not change augmentation or SGD RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    target = torch.device(device)
    devices = ([target.index if target.index is not None else torch.cuda.current_device()]
               if target.type == "cuda" else [])
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def snapshot(dataset, transform=None):
    """Share immutable inputs, copy labels BEFORE a future in-place label swap."""
    frozen = copy(dataset)
    frozen.targets = np.asarray(dataset.targets, dtype=np.int64).copy()
    if transform is not None:
        frozen.transform = transform
    return frozen


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index


def dataset_targets(dataset):
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([dataset_targets(part) for part in dataset.datasets])
    return np.asarray(dataset.targets, dtype=np.int64)


def checked(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite FedDAA {name}")
    return tensor


def output_prototype(model, dataset, num_classes, device, batch_size):
    """Classwise RAW LOGITS, not softmax, as in get_first_model_output_prototype.

    Missing rows stay zero, including binary/CIFAR100 extensions. No test data.
    """
    was_training = model.training
    original_device = next(model.parameters()).device
    sums = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.long)
    try:
        model.to(device).eval()
        with isolated_rng(device), torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                logits = checked(model(checked(x.to(device, dtype=torch.float32), "prototype inputs")),
                                 "prototype logits").cpu().double()
                if logits.shape != (len(y), num_classes):
                    raise ValueError("FedDAA prototype output dimension does not match num_classes")
                sums.index_add_(0, y.long(), logits)
                counts += torch.bincount(y.long(), minlength=num_classes)
        return (sums / counts.clamp_min(1)[:, None]).numpy()
    finally:
        model.to(original_device).train(was_training)


def determine_clusters(prototypes, max_clusters=9):
    """Authors' KMeans(k=2..9), maximum silhouette, Euclidean flattened logits.

    Only degenerate/small inputs fall back to one cluster; no added threshold.
    Pin n_init=10 to avoid sklearn-version-dependent defaults.
    """
    x = np.asarray(prototypes, dtype=np.float64).reshape(len(prototypes), -1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("FedDAA requires nonempty finite prototypes")
    limit = min(max_clusters, len(x) - 1, len(np.unique(x, axis=0)))
    best, scores = None, {}
    for k in range(2, limit + 1):
        fit = KMeans(n_clusters=k, random_state=0, n_init=10).fit(x)
        if len(np.unique(fit.labels_)) != k:
            continue
        score = float(silhouette_score(x, fit.labels_))
        scores[str(k)] = score
        if best is None or score > best[0]:
            best = (score, fit)
    if best is None:
        return np.zeros(len(x), dtype=int), x.mean(0, keepdims=True), scores
    return best[1].labels_, best[1].cluster_centers_, scores


def detect_real_drift(previous, current, centers):
    """Recompute both prototypes with ONE current model, assign to SAME centers."""
    old = np.asarray(previous).reshape(len(previous), -1)
    new = np.asarray(current).reshape(len(current), -1)
    old_labels = ((old[:, None] - centers[None]) ** 2).sum(2).argmin(1)
    new_labels = ((new[:, None] - centers[None]) ** 2).sum(2).argmin(1)
    return old_labels != new_labels


def responsibilities(losses, mixture, label_priors, targets):
    """FedRC E step: q(c|x,y) proportional to pi(c)*p_c(y|x)/p_c(y)."""
    tiny = torch.finfo(losses.dtype).tiny
    log_scores = (mixture.clamp_min(tiny).log()[:, None] - losses
                  - label_priors[:, targets].clamp_min(1e-8).log())
    return checked(log_scores.softmax(dim=0), "responsibilities")


class ProbabilityMixture(nn.Module):
    """Probability ensemble, matching upstream hard_cluster=False evaluation."""
    def __init__(self, models, weights):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("weights", weights.detach().clone())

    def forward(self, x):
        probabilities = sum(w * model(x).softmax(1)
                            for w, model in zip(self.weights, self.models))
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


class FedDAAClient:
    def __init__(self, identity, train_set, test_set, source_id=0):
        self.id, self.train_set, self.test_set = identity, train_set, test_set
        self.global_test_id = source_id
        self.previous = None
        self.current = None
        self.training = None
        self.mixture = None
        self.label_counts = None
        self.optimizer_states = {}
        self.real_drift = False

    def start_slot(self, current, real_drift, clusters, classes):
        self.previous, self.current = self.current, current
        self.real_drift = bool(real_drift)
        self.training = (current if self.previous is None or real_drift else
                         ConcatDataset([self.previous, current]))
        self.mixture = torch.full((clusters,), 1.0 / clusters)
        # Match upstream reset at each slot; these are class pseudocounts until
        # the client participates, NOT artificial training samples.
        self.label_counts = torch.full((clusters, classes), 1.0 / classes)
        self.optimizer_states = {}

    def train(self, server, args, round_index):
        device, batch_size = args["device"], args["batch_size"]
        targets = torch.from_numpy(dataset_targets(self.training))
        measurement = self.training
        if isinstance(measurement, ConcatDataset):
            measurement = ConcatDataset([snapshot(part, server.eval_transform)
                                         for part in measurement.datasets])
        else:
            measurement = snapshot(measurement, server.eval_transform)
        losses = []
        # Identical deterministic inputs for all experts; never use validation
        # or test labels to infer membership or responsibilities.
        with isolated_rng(device), torch.no_grad():
            for model in server.models:
                model.to(device).eval()
                parts = []
                try:
                    for x, y in DataLoader(measurement, batch_size=batch_size, shuffle=False):
                        logits = checked(model(x.to(device, dtype=torch.float32)), "E-step logits")
                        parts.append(F.cross_entropy(logits, y.to(device), reduction="none").cpu())
                finally:
                    model.cpu()
                losses.append(torch.cat(parts))
        q = responsibilities(torch.stack(losses), self.mixture, server.label_priors, targets)
        self.mixture = q.mean(1)
        self.label_counts = torch.zeros_like(server.label_priors)
        for expert in range(len(server.models)):
            self.label_counts[expert].scatter_add_(0, targets, q[expert])
        total_loss, steps = 0.0, 0
        states = []
        for expert, global_model in enumerate(server.models):
            model = deepcopy(global_model).to(device).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"],
                                        momentum=args["momentum"], weight_decay=args["weight_decay"])
            if expert in self.optimizer_states:
                optimizer.load_state_dict(self.optimizer_states[expert])
            generator = torch.Generator().manual_seed(
                args["seed"] + 1000003 * round_index + 1009 * self.id + expert)
            loader = DataLoader(IndexedDataset(self.training), batch_size=batch_size,
                                shuffle=True, generator=generator, drop_last=False)
            for _ in range(args["epochs"]):
                for x, y, indices in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = checked(model(checked(x.to(device, dtype=torch.float32), "training inputs")),
                                     "training logits")
                    loss_vec = F.cross_entropy(logits, y.to(device), reduction="none")
                    # Upstream divides by B, not sum(q): preserve expert mass.
                    loss = checked((loss_vec * q[expert, indices].to(device)).mean(), "loss")
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            checked(parameter.grad, "gradient")
                    optimizer.step()
                    for parameter in model.parameters():
                        checked(parameter, "parameter")
                    total_loss += float(loss.detach())
                    steps += 1
            opt_state = optimizer.state_dict()
            for state in opt_state["state"].values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        checked(value, "optimizer state")
                        state[key] = value.cpu()
            self.optimizer_states[expert] = opt_state
            states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            del optimizer, model
        return states, len(self.training), total_loss / max(steps, 1), steps


class FedDAAServer:
    def __init__(self, model_factory, args, eval_transform=None):
        self.model_factory, self.args, self.eval_transform = model_factory, args, eval_transform
        self.models = [model_factory(i) for i in range(args["feddaa_initial_clusters"])]
        self.label_priors = torch.full((len(self.models), args["num_classes"]),
                                       1.0 / len(self.models))
        self.centers = None
        self.last_slot = {}

    def start_slot(self, clients, slot):
        args = self.args
        frozen = [snapshot(client.train_set) for client in clients]
        scores, detected, labels = {}, np.zeros(len(clients), dtype=bool), None
        if slot:
            model = self.models[0]
            current = [output_prototype(model, snapshot(data, self.eval_transform),
                                        args["num_classes"], args["device"], args["batch_size"])
                       for data in frozen]
            old = [output_prototype(model, snapshot(client.current, self.eval_transform),
                                    args["num_classes"], args["device"], args["batch_size"])
                   for client in clients]
            labels, self.centers, scores = determine_clusters(current, args["feddaa_max_clusters"])
            detected = detect_real_drift(old, current, self.centers)
            clusters = len(self.centers)
            # Keep existing expert indices; extra experts use seed+expert just
            # like upstream get_learners_ensemble. No oracle concept matching.
            self.models = (self.models[:clusters] +
                           [self.model_factory(i) for i in range(len(self.models), clusters)])
        for client, data, drift in zip(clients, frozen, detected):
            client.start_slot(data, drift, len(self.models), args["num_classes"])
        self.label_priors = torch.full((len(self.models), args["num_classes"]),
                                       1.0 / len(self.models))
        self.last_slot = dict(time_slot=slot, clusters=len(self.models),
                              detected_clients=[c.id for c, d in zip(clients, detected) if d],
                              history_clients=[c.id for c in clients if isinstance(c.training, ConcatDataset)],
                              prototype_clients=len(clients) if slot else 0,
                              prototype_assignments=None if labels is None else labels.tolist(),
                              silhouette_scores=scores)
        return self.last_slot

    def train_round(self, clients, selected, round_index, context):
        totals = None
        sample_total, losses, steps = 0, [], 0
        for client in selected:
            context["client_id"] = client.id
            states, size, loss, client_steps = client.train(self, self.args, round_index)
            if totals is None:
                totals = [{key: value.double() * size if value.is_floating_point() else value.clone()
                           for key, value in state.items()} for state in states]
            else:
                for total, state in zip(totals, states):
                    for key, value in state.items():
                        if value.is_floating_point():
                            total[key].add_(value.double(), alpha=size)
            sample_total += size
            losses.append(loss)
            steps += client_steps
            context["completed_train_clients"].append(client.id)
        # Upstream CentralizedAggregator averages full experts by TRAIN size,
        # NOT by mixture weights and NOT by an IFCA hard assignment.
        for model, total in zip(self.models, totals):
            for key, value in model.state_dict().items():
                if value.is_floating_point():
                    total[key] = checked((total[key] / sample_total).to(value.dtype), "aggregate")
            model.load_state_dict(total)
        counts = torch.stack([client.label_counts for client in clients]).sum(0).clamp_min(1e-8)
        self.label_priors = counts / counts.sum(1, keepdim=True)
        return float(np.mean(losses)), steps
