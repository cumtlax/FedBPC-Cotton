"""Removed 2026-10-19 with the class-row sharing branch."""
POLICY = "none"

def validate_task_gate(args):
    if args.get("class_share_task_gate", "none") != "none":
        raise ValueError("class-row sharing has been removed")
    return "none", None

def calibration_features(*args, **kwargs):
    raise RuntimeError("row task gate has been removed")

def class_losses(*args, **kwargs):
    raise RuntimeError("row task gate has been removed")
