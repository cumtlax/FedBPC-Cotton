"""Optional FedCAP BODY-only proximal objective, anchored to this round's download.

No gradient rescaling, extra SGD, head adaptation or test data is introduced here. The
disabled branch never constructs an anchor or adds a zero tensor to the loss.
"""
import math
import torch

POLICY = "round_download_body_parameter_l2_v1"


def validate_mu(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("body_prox_mu must be finite and nonnegative (0=disabled)")
    return float(value)


class BodyProximal:
    def __init__(self, model, mu):
        self.mu = validate_mu(mu)
        if self.mu == 0:
            raise ValueError("Do not construct a proximal anchor when mu=0")
        # Called once at train() entry, AFTER server.sync(), BEFORE local SGD.
        # Buffers and head parameters are deliberately excluded.
        self.anchor = {name: p.detach().clone() for name, p in model.body.named_parameters()}
        if not self.anchor:
            raise ValueError("Body proximal training requires body parameters")
        self.steps = 0
        self.ce_sum = self.penalty_sum = 0.0

    def penalty(self, model):
        parameters = dict(model.body.named_parameters())
        if parameters.keys() != self.anchor.keys():
            raise ValueError("Proximal anchor and body parameters do not match")
        return (self.mu / 2) * sum((p - self.anchor[name]).square().sum()
                                   for name, p in parameters.items())

    def observe(self, ce, penalty):
        self.steps += 1
        self.ce_sum += float(ce.detach())
        self.penalty_sum += float(penalty.detach())

    def summary(self, model):
        with torch.no_grad():
            squared = sum((p.detach().double() - self.anchor[name].double()).square().sum()
                          for name, p in model.body.named_parameters())
        return dict(steps=self.steps, ce_sum=self.ce_sum, penalty_sum=self.penalty_sum,
                    client_rounds=1, body_distance_l2_sum=math.sqrt(float(squared)),
                    body_distance_l2_max=math.sqrt(float(squared)))


def proximal_summary(records):
    """Step-weighted body losses; equally weighted participating client-round drift."""
    rows = list(records)
    result = {key: sum(row[key] for row in rows) for key in
              ("steps", "ce_sum", "penalty_sum", "client_rounds", "body_distance_l2_sum")}
    result["body_distance_l2_max"] = max((row["body_distance_l2_max"] for row in rows), default=0.0)
    result["body_ce_mean"] = result["ce_sum"] / result["steps"] if result["steps"] else 0.0
    result["body_prox_mean"] = result["penalty_sum"] / result["steps"] if result["steps"] else 0.0
    result["body_distance_l2_mean"] = (result["body_distance_l2_sum"] / result["client_rounds"]
                                      if result["client_rounds"] else 0.0)
    return result
