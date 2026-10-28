"""Removed 2026-10-19 with the class-row sharing branch."""
POLICY = "none"
IMPLEMENTATION_SUFFIX = ""

def validate_head_centering(args):
    if args.get("class_share_centering", "none") != "none":
        raise ValueError("class-row sharing has been removed")
    return "none"
