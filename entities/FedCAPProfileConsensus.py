"""Removed 2026-10-19 with the class-row sharing branch."""
POLICY = "none"
DONOR_POLICY = "none"
IMPLEMENTATION_SUFFIX = ""

def validate_consensus(args):
    if args.get("class_share_witness_policy", "strict") == "consensus":
        raise ValueError("class-row sharing has been removed")
    return "strict", None
