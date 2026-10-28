"""Removed 2026-10-19: cross-cluster class-row sharing is not part of the main method."""
LABEL_ALIGNMENT_SUFFIX = ""
LABEL_ALIGNMENT_POLICY = "none"
ALIGNED_HEAD_AGGREGATION = "none"
LABEL_MERGE_SUFFIX = ""
LABEL_MERGE_POLICY = "none"

class FedCAPProfileShareServer:
    implementations = {}
    policies = {}
    head_aggregation = "none"
    def __init__(self, *args, **kwargs):
        raise RuntimeError("class-row sharing has been removed from the main FedCAP-Cotton branch")

def validate_label_mode(args):
    if args.get("class_share_label_mode", "same") != "same":
        raise ValueError("class-row sharing has been removed")
    return "same"

def validate_label_merge_guard(args):
    if args.get("label_merge_guard", "none") != "none":
        raise ValueError("class-row sharing has been removed")
    return "none"
