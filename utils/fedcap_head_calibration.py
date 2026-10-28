"""Split the existing head SGD budget around global BODY aggregation.

Opt-in only. The head sampler is resumed, not restarted, and its SGD momentum
is retained across the two phases. This is a training intervention, not a
read-only diagnostic or a guarantee of improved test accuracy.
"""
from itertools import islice
import math

import torch


POLICY = "split_head_budget_aggregate_body_v1"
COMMUNICATION_POLICY = "body_upload_then_body_download_head_upload_v1"


def validate_ratio(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("head_calibration_ratio must be finite in [0, 1] (0=old training)")
    return float(value)


def split_steps(total, ratio):
    ratio = validate_ratio(ratio)
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("Head SGD budget must be a nonnegative integer")
    post = math.ceil(total * ratio)
    return total - post, post


def _batches(loader, epochs):
    for _ in range(epochs):
        yield from loader


def state_bytes(state):
    return sum(value.numel() * value.element_size() for value in state.values())


class HeadCalibrationSession:
    def __init__(self, client, loader, round_index):
        self.round_index = round_index
        self.ratio = validate_ratio(client.args.get("head_calibration_ratio", 0))
        self.total = len(loader) * client.args["clf_epochs"]
        self.pre, self.post = split_steps(self.total, self.ratio)
        if self.post < 1:
            raise ValueError("Head calibration requires a nonzero head training budget")
        self.batches = _batches(loader, client.args["clf_epochs"])
        self.optimizer = torch.optim.SGD(
            client.model.head.parameters(), lr=client.args["clf_lr"],
            momentum=client.args["momentum"], weight_decay=client.args["weight_decay"])
        self.losses = []
        self.stats = dict(client_id=client.id, round=round_index + 1, planned_steps=self.total,
                          pre_steps=0, post_steps=0, pre_loss_sum=0., post_loss_sum=0.,
                          seconds=0., extra_body_download_bytes=0, deferred_head_upload_bytes=0)

    def take(self, steps):
        return islice(self.batches, steps)


def calibration_summary(stats):
    records = list(stats)
    keys = ("planned_steps", "pre_steps", "post_steps", "pre_loss_sum", "post_loss_sum",
            "seconds", "extra_body_download_bytes", "deferred_head_upload_bytes")
    result = {key: sum(row[key] for row in records) for key in keys}
    result["client_rounds"] = len(records)
    result["budget_preserved"] = result["planned_steps"] == result["pre_steps"] + result["post_steps"]
    if not result["budget_preserved"]:
        raise RuntimeError("Head calibration changed the formal head SGD budget")
    result["post_ce_mean"] = result["post_loss_sum"] / result["post_steps"] if result["post_steps"] else 0.
    # The normal head upload is deferred, NOT transmitted twice. There is one
    # extra body download and one extra formal exchange per communication round.
    result["additional_head_upload_bytes"] = 0
    return result
