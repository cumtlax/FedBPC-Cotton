"""Removed 2026-10-19: profile merge guards are not used by the main method."""
class _NoProfileMergeGuard:
    mode = "none"
    implementation_suffix = ""
    epsilon = 0.0
    min_witnesses = 0
    policy = "none"
    min_coverage = 0.0
    memory_rounds = 0
    conflict_memory = {}

    def filter_groups(self, groups, *args, **kwargs):
        return groups, {}

    def checkpoint(self):
        return {"mode": "none"}

def make_profile_merge_guard(args):
    if args.get("profile_merge_guard", "none") != "none":
        raise ValueError("profile merge guard has been removed from the main branch")
    return _NoProfileMergeGuard()
