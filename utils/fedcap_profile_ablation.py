"""Opt-in matched recipes; no new model, optimizer, or aggregation kernel.

Evidence and memory use the SAME 80% coverage requirement. Full coverage
already recertifies every remembered class, making memory redundant across
rounds. Partial coverage is an ablation, not a semantic safety guarantee.
"""
from copy import deepcopy


POLICY = "independent_rows_merge_factorization_v1"
MODES = ("sampling", "conflict", "evidence", "memory")
METADATA_KEYS = ("profile_ablation", "profile_ablation_policy")


def recipe(mode):
    if mode not in MODES:
        raise ValueError("Unknown profile ablation")
    values = dict(profile_samples_per_class=64,
                  head_sharing="profile", class_share_rule="centroid",
                  class_share_rho=1.0, class_share_eps=0.1,
                  class_share_min_witnesses=2, cluster_method="agglomerative",
                  profile_weight=0.5, deterministic=True, merge_guard="none",
                  diagnostic_head_sharing="none", head_weights="uniform",
                  profile_merge_guard="none" if mode == "sampling" else
                  "conflict" if mode == "conflict" else "evidence")
    if mode != "sampling":
        values.update(profile_merge_eps=0.1, profile_merge_min_witnesses=2)
    if mode in ("evidence", "memory"):
        values.update(profile_merge_min_coverage=0.8,
                      profile_merge_memory_rounds=10 if mode == "memory" else 0)
    return values


def apply_profile_ablation(args, options, parser):
    """Expand only an explicitly requested CLI recipe; saved labels are data.

    Old CLI/YAML switches keep their original precedence when no recipe is
    selected. Reject contradictory explicit switches rather than silently
    changing what a named experiment means.
    """
    for key in METADATA_KEYS:
        args.pop(key, None)
    mode = options.profile_ablation
    if mode is None:
        return
    if args["algorithm"] != "FedCAP":
        parser.error("--profile-ablation is only supported by FedCAP")
    for key, expected in recipe(mode).items():
        explicit = getattr(options, key, None)
        if key == "deterministic" and explicit is not None:
            explicit = explicit == "true"
        if explicit is not None and explicit != expected:
            parser.error(f"--{key.replace('_', '-')} conflicts with --profile-ablation {mode}; "
                         "use the individual switches for a custom experiment")
        args[key] = deepcopy(expected)
    args.update(profile_ablation=mode, profile_ablation_policy=POLICY)


def evidence_stamp(args):
    """Human-readable distinction between evidence with/without memory."""
    return f"-pme-m{args.get('profile_merge_memory_rounds', 10)}"
