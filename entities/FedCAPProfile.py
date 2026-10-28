"""RNG-isolated TRAIN-only profile measurement; no optimizer or sharing rules.

Zero budget preserves the legacy profile from the balanced probe draw. A
positive budget takes at most that many distinct examples of each present
class. Variance describes observed prediction variation, NOT a confidence
interval, a semantic certificate, or a new aggregation weight.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


LEGACY_POLICY = "shared_balanced_probe_draw_v1"
UNIQUE_POLICY = "independent_unique_train_rows_v1"
FEATURE_BINS = 8


def profile_head(args):
    mode = args.get("profile_head", "reference")
    if mode not in ("reference", "none"):
        raise ValueError("profile_head must be reference or none")
    if mode == "none" and args.get("probe_training", "fedccfa") != "fedccfa":
        raise ValueError("Head-free profiles require current-body fedccfa probes")
    return mode


def profile_metadata(args):
    mode = profile_head(args)
    return dict(profile_head=mode, profile_representation=(
        "reference_softmax_v1" if mode == "reference" else "body_tanh_marginal_histogram8_v1"))


def profile_width(model, args):
    return args["num_classes"] if profile_head(args) == "reference" else model.head.in_features * FEATURE_BINS


def profile_values(model, features, args):
    """Fixed marginal histograms, not feature-mean prototypes.

    Tanh bounds finite activations; eight equal bins per feature coordinate.
    Each sample contributes 1/D mass per coordinate. No learned projection.
    Tails saturate and correlations are lost; this is an experimental signal.
    """
    if profile_head(args) == "reference":
        return model.head(features).softmax(1).cpu()
    if features.ndim != 2 or not features.shape[1] or not torch.isfinite(features).all():
        raise FloatingPointError("Invalid body features for head-free profile")
    bins = ((features.tanh() + 1) * (FEATURE_BINS / 2)).long().clamp(0, FEATURE_BINS - 1)
    values = torch.nn.functional.one_hot(bins, FEATURE_BINS).to(features.dtype)
    return (values.flatten(1) / features.shape[1]).cpu()


def valid_profile_descriptor(profile, valid, descriptor, classes):
    """Both representations contain nonnegative unit-mass supported rows."""
    mode = descriptor.get("profile_head", "reference")
    shape_ok = (profile.ndim == 2 and profile.shape[0] == classes and
                (profile.shape[1] == classes if mode == "reference" else
                 mode == "none" and profile.shape[1] > 0 and profile.shape[1] % FEATURE_BINS == 0))
    valid_rows = (shape_ok and valid.shape == (classes,) and valid.dtype == np.bool_ and
                  np.isfinite(profile).all() and not (profile < 0).any() and
                  np.allclose(profile[valid].sum(axis=1), 1.0, atol=1e-5))
    if valid_rows and mode == "none":
        dimensions = profile.shape[1] // FEATURE_BINS
        marginals = profile.reshape(classes, dimensions, FEATURE_BINS).sum(axis=2)
        return np.allclose(marginals[valid], 1.0 / dimensions, atol=1e-6)
    return valid_rows


def profile_budget(args):
    value = args.get("profile_samples_per_class", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("profile_samples_per_class must be a nonnegative integer")
    return value


def unique_profile_indices(targets, limit, classes, seed):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Independent profiles require a positive per-class limit")
    labels = np.asarray(targets)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer) or
            np.any(labels < 0) or np.any(labels >= classes)):
        raise ValueError("Invalid training labels for profile measurement")
    rng = np.random.default_rng(seed)
    indices = []
    for category in range(classes):
        pool = np.flatnonzero(labels == category)
        count = min(len(pool), limit)
        if count:
            indices.extend(rng.choice(pool, count, replace=False).tolist())
    rng.shuffle(indices)
    return indices


class ProfileMoments:
    """Moments of distinct raw examples, even if the legacy draw repeats them."""

    def __init__(self, classes, width=None):
        self.counts = np.zeros(classes, dtype=np.int64)
        self.sums = np.zeros((classes, classes if width is None else width), dtype=np.float64)
        self.squares = np.zeros_like(self.sums)
        self.seen = set()

    def update(self, probabilities, labels, indices):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if probabilities.shape != (len(labels), self.sums.shape[1]) or len(indices) != len(labels):
            raise ValueError("Profile probabilities, labels and raw indices do not match")
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite prediction profile")
        for probability, category, index in zip(probabilities, labels, indices):
            if int(index) in self.seen:
                continue
            self.seen.add(int(index))
            self.counts[category] += 1
            self.sums[category] += probability
            self.squares[category] += probability * probability

    def metadata(self):
        variance = np.zeros_like(self.sums)
        valid = self.counts > 1
        counts = self.counts[valid, None]
        variance[valid] = np.maximum(
            (self.squares[valid] - self.sums[valid] ** 2 / counts) / (counts - 1), 0)
        return dict(profile_unique_counts=self.counts.tolist(),
                    profile_variance=variance.tolist(),
                    profile_variance_valid=valid.tolist(),
                    profile_row_std=np.sqrt(variance.mean(axis=1)).tolist())


@contextmanager
def isolated_profile_rng(seed, device):
    """Extra loaders/transforms cannot advance the formal-training RNG stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            # Seed only the generators saved by fork_rng, not every visible GPU.
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def measure_unique_profile(client, reference, round_index):
    """Use the common PRE-adaptation reference, never the locally fitted head."""
    limit, classes = profile_budget(client.args), client.args["num_classes"]
    seed = (client.args["seed"] + 100003 * round_index + client.id + 791900001) % (2 ** 63 - 1)
    indices = unique_profile_indices(client.train_set.targets, limit, classes, seed)
    dataset = copy(client.train_set)
    # Transform policy matches the legacy profile loader; labels stay live.
    if client.args["dataset"] in ("CIFAR10", "CIFAR100", "CINIC-10"):
        dataset.transform = client.test_set.transform
    width = profile_width(reference, client.args)
    moments = ProfileMoments(classes, width)
    profile = torch.zeros(classes, width)
    counts = torch.zeros(classes, dtype=torch.long)
    started = time.perf_counter()
    with isolated_profile_rng(seed, client.args["device"]), torch.no_grad():
        model = deepcopy(reference).eval()
        loader = DataLoader(Subset(dataset, indices), batch_size=client.args["batch_size"],
                            shuffle=False, generator=torch.Generator().manual_seed(seed))
        offset = 0
        for x, y in loader:
            features = model.body(x.to(client.args["device"]))
            probabilities = profile_values(model, features, client.args)
            for label in y.unique():
                mask = y == label
                profile[label] += probabilities[mask].sum(0)
                counts[label] += mask.sum()
            moments.update(probabilities.numpy(), y.numpy(), indices[offset:offset + len(y)])
            offset += len(y)
    profile /= counts.clamp_min(1)[:, None]
    return profile.numpy(), dict(
        **moments.metadata(), **profile_metadata(client.args), profile_sample_counts=counts.tolist(),
        profile_sample_policy=UNIQUE_POLICY,
        profile_extra_forward_seconds=time.perf_counter() - started)


def summarize_profiles(selected, cache, round_index, version):
    """Small round statistics without copying/serializing profile matrices.

    Only current, same-reference evidence is counted. Independent row validity
    takes precedence over clustering validity, as it does in sharing/merge gates.
    """
    unique, variation = [], []
    valid_rows = total_rows = 0
    extra_seconds = 0.0
    for client in selected:
        descriptor = cache.get(client.id)
        if (descriptor is None or descriptor.get("round") != round_index or
                descriptor.get("version") != version or "profile" not in descriptor):
            continue
        valid = descriptor.get("row_valid", descriptor.get("valid", []))
        valid_rows += sum(valid)
        total_rows += len(valid)
        unique.extend(n for n in descriptor.get("profile_unique_counts", []) if n > 0)
        variation.extend(s for s, valid in zip(descriptor.get("profile_row_std", []),
                                             descriptor.get("profile_variance_valid", [])) if valid)
        extra_seconds += descriptor.get("profile_extra_forward_seconds", 0.0)
    return dict(profile_valid_rows=int(valid_rows), profile_total_rows=total_rows,
                profile_mean_unique_samples=float(np.mean(unique)) if unique else None,
                profile_mean_row_std=float(np.mean(variation)) if variation else None,
                profile_extra_forward_seconds=extra_seconds)
