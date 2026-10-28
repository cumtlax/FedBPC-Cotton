"""TRAIN class-count weights for formal FedCAP head aggregation only."""
import numpy as np


POLICY = "capped_sqrt_class_count_uniform_mixture_v1"
IMPLEMENTATION_SUFFIX = "_class_support_v1"
COUNT_CAP = 64
# Keep COUNT_CAP for historical implementation identifiers; new runs use CapFull.
DEFAULT_COUNT_CAP = 0
SUPPORT_SCOPES = ("all", "cross-cluster")
CROSS_ONLY_SUFFIX = "_class_support_cross_only_v1"


def validate_support_scope(args):
    """Old all-row weighting is the default; opt-in keeps whole heads uniform."""
    scope = args.get("class_support_scope", "all")
    if not isinstance(scope, str) or scope not in SUPPORT_SCOPES:
        raise ValueError("class_support_scope must be all or cross-cluster")
    if scope == "cross-cluster":
        alpha = args.get("class_support_alpha", 0.0)
        if (args.get("algorithm", "FedCAP") != "FedCAP" or
                isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
                not np.isfinite(alpha) or not 0 < alpha <= 1 or
                args.get("head_weights", "uniform") != "uniform" or
                args.get("head_sharing", "none") != "profile" or
                args.get("diagnostic_head_sharing", "none") != "none"):
            raise ValueError("cross-cluster support requires FedCAP, positive alpha, uniform heads and profile row sharing")
    return scope


def validate_count_cap(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("class_support_count_cap must be a nonnegative integer (0=uncapped)")
    return value


def support_policy(count_cap):
    validate_count_cap(count_cap)
    return POLICY if count_cap else "sqrt_full_train_class_count_uniform_mixture_v1"


def implementation_suffix(count_cap):
    validate_count_cap(count_cap)
    if count_cap == COUNT_CAP:
        return IMPLEMENTATION_SUFFIX
    return IMPLEMENTATION_SUFFIX + ("_uncapped" if count_cap == 0 else f"_cap{count_cap}")


def validate_class_support(args):
    validate_count_cap(args.get("class_support_count_cap", DEFAULT_COUNT_CAP))
    alpha = args.get("class_support_alpha", 0.0)
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
            not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("class_support_alpha must be finite in [0, 1]")
    if alpha > 0:
        if args.get("algorithm", "FedCAP") != "FedCAP":
            raise ValueError("Class support aggregation is only supported by FedCAP")
        if args.get("head_weights", "uniform") != "uniform":
            raise ValueError("Class support aggregation requires head_weights: uniform")
        if args.get("diagnostic_head_sharing", "none") != "none":
            raise ValueError("Class support aggregation does not support oracle head sharing")
    validate_support_scope(args)
    return float(alpha)


def class_support_weights(counts, alpha, count_cap=DEFAULT_COUNT_CAP):
    """Counts indicate support, not calibrated statistical confidence.

    Keep a uniform component, including donors without positive examples.
    The cap is independent of the profile sampling budget (0 means no cap). Entirely
    absent classes fall back to uniform, including at alpha=1.
    """
    validate_class_support({"class_support_alpha": alpha, "class_support_count_cap": count_cap})
    counts = np.asarray(counts, dtype=np.float64)
    if (counts.ndim != 1 or not counts.size or not np.isfinite(counts).all() or
            (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError("Class support counts must be finite nonnegative integers")
    uniform = np.full(counts.size, 1.0 / counts.size)
    if alpha == 0:
        return uniform
    support = np.sqrt(np.minimum(counts, count_cap) if count_cap else counts)
    total = support.sum()
    if total == 0:
        return uniform
    return (1 - alpha) * uniform + alpha * support / total


def current_class_counts(clients, cache, round_index, version, classes):
    """Use full TRAIN counts uploaded this round, never profile draw counts."""
    rows = {}
    for client in clients:
        descriptor = cache.get(client.id)
        if (round_index is None or descriptor is None or
                descriptor.get("round") != round_index or descriptor.get("version") != version):
            raise ValueError(f"Class support requires fresh counts: client={client.id}")
        counts = np.asarray(descriptor.get("class_counts", []), dtype=np.float64)
        if (counts.shape != (classes,) or not np.isfinite(counts).all() or
                (counts < 0).any() or (counts != np.floor(counts)).any() or
                counts.sum() != len(client.train_set)):
            raise ValueError(f"Invalid full TRAIN class counts: client={client.id}")
        rows[client.id] = counts
    return rows
