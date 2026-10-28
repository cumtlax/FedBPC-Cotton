"""Small, RNG-neutral diagnostics. Source IDs never feed inferred grouping decisions."""
from itertools import combinations
from functools import wraps

import torch


def preserve_model_modes(evaluator):
    """Evaluation is observational even for mixed train/eval submodules."""
    @wraps(evaluator)
    def wrapped(model, *args, **kwargs):
        modes = [(module, module.training) for module in model.modules()]
        try:
            return evaluator(model, *args, **kwargs)
        finally:
            for module, mode in modes:
                module.training = mode
    return wrapped


def head_difference_report(heads):
    pairs = []
    for left, right in combinations(sorted(heads), 2):
        a, b = heads[left], heads[right]
        aw, bw = a["weight"].detach().double().cpu(), b["weight"].detach().double().cpu()
        ab, bb = a["bias"].detach().double().cpu(), b["bias"].detach().double().cpu()
        if aw.shape[0] != 2 or bw.shape != aw.shape:
            raise ValueError("Cotton head diagnostics require matching binary affine heads")
        margin = (aw[1] - aw[0]) - (bw[1] - bw[0])
        bias_margin = (ab[1] - ab[0]) - (bb[1] - bb[0])
        pairs.append(dict(
            cluster_ids=[int(left), int(right)],
            parameter_l2=float(torch.sqrt(((aw-bw)**2).sum() + ((ab-bb)**2).sum())),
            decision_l2=float(torch.sqrt((margin**2).sum() + bias_margin**2)),
            identical_head=bool(torch.equal(aw, bw) and torch.equal(ab, bb)),
            identical_decision=bool(torch.count_nonzero(margin) == 0 and bias_margin == 0)))
    return dict(clusters=len(heads), pairs=pairs,
                identical_head_pairs=sum(p["identical_head"] for p in pairs),
                identical_decision_pairs=sum(p["identical_decision"] for p in pairs))


def source_composition(server, clients):
    groups = {}
    retained = {"0": 0, "1": 0, "2": 0}
    for client in clients:
        source = str(int(client.train_set.concept_id))
        identity = server.membership[client.id]
        if identity is None:
            retained[source] += 1
        else:
            counts = groups.setdefault(str(identity), {"0": 0, "1": 0, "2": 0})
            counts[source] += 1
    return dict(clusters=groups, retained=retained, diagnostic_only=True)
