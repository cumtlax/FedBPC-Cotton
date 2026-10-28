"""Independent Factorized-FL implementation under the project backbone layout.

Reference: Jeong & Hwang, NeurIPS 2022; wyjeong/Factorized-FL,
a21a3e1d43f9a16b7c9a1285483bfc39799a8ca7.
See third_party/factorized_fl/INTEGRATION.md for source/protocol differences.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


UPSTREAM_COMMIT = "a21a3e1d43f9a16b7c9a1285483bfc39799a8ca7"


def finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite Factorized-FL {label}")
    return value


class FactorizedLinear(nn.Module):
    def __init__(self, layer, rank=1, threshold=5e-4):
        super().__init__()
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.threshold = threshold
        self.uu = nn.Parameter(torch.empty(layer.in_features, rank))
        self.vv = nn.Parameter(torch.empty(rank, layer.out_features))
        self.mask = nn.Parameter(torch.empty(layer.in_features, layer.out_features))
        self.bias = nn.Parameter(torch.empty(layer.out_features)) if layer.bias is not None else None
        for parameter in (self.uu, self.vv, self.mask):
            nn.init.xavier_uniform_(parameter)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias[None])

    def effective_weight(self):
        residual = (self.mask if self.training else
                    self.mask.masked_fill(self.mask.abs() < self.threshold, 0))
        return (self.uu @ self.vv + residual).T

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class FactorizedConv2d(nn.Module):
    def __init__(self, layer, rank=1, threshold=5e-4):
        super().__init__()
        if layer.groups != 1 or layer.padding_mode != "zeros":
            raise ValueError("Factorized-FL port supports ungrouped, zero-padded convolutions only")
        self.in_channels, self.out_channels = layer.in_channels, layer.out_channels
        self.kernel_size, self.stride = layer.kernel_size, layer.stride
        self.padding, self.dilation = layer.padding, layer.dilation
        self.threshold = threshold
        kernel_elements = layer.kernel_size[0] * layer.kernel_size[1]
        self.uu = nn.Parameter(torch.empty(kernel_elements, rank))
        self.vv = nn.Parameter(torch.empty(rank, layer.out_channels * layer.in_channels))
        self.mask = nn.Parameter(torch.zeros(kernel_elements, layer.out_channels * layer.in_channels))
        self.bias = nn.Parameter(torch.empty(layer.out_channels)) if layer.bias is not None else None
        nn.init.xavier_uniform_(self.uu)
        nn.init.xavier_uniform_(self.vv)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias[None])

    def effective_weight(self):
        residual = (self.mask if self.training else
                    self.mask.masked_fill(self.mask.abs() < self.threshold, 0))
        # Deliberately retain the authors' flattened layout, not an einsum
        # interpretation with different channel/kernel axes.
        return (self.uu @ self.vv + residual).reshape(
            self.out_channels, self.in_channels, *self.kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias,
                        self.stride, self.padding, self.dilation)


def factorize_model(model, rank=1, threshold=5e-4):
    """Preserve layer topology/activations; replace Conv/Linear including head."""
    if rank < 1 or threshold < 0:
        raise ValueError("Invalid factorization rank or sparsity threshold")
    for name, layer in list(model.named_children()):
        if isinstance(layer, nn.Linear):
            replacement = FactorizedLinear(layer, rank, threshold)
        elif isinstance(layer, nn.Conv2d):
            replacement = FactorizedConv2d(layer, rank, threshold)
        else:
            replacement = factorize_model(layer, rank, threshold)
        setattr(model, name, replacement)
    if not hasattr(model, "hidden_layers") or not hasattr(model, "fc"):
        # Recursive calls on Sequential have no body/head by design.
        return model
    return model


def mask_penalty(model):
    # named_parameters retains autograd. Upstream uses state_dict(), which
    # normally detaches tensors and can silently eliminate this gradient.
    return sum(parameter.abs().sum() for name, parameter in model.named_parameters()
               if name.endswith("mask"))


def matching_vector(body):
    vectors = [parameter for name, parameter in body.named_parameters() if name.endswith("vv")]
    if not vectors:
        raise ValueError("Factorized-FL body has no factor coefficients for matching")
    return vectors[-1].detach().cpu().flatten().clone()


def cloned_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def shared_parameter(name, variant):
    # Alpha shares u plus ordinary biases, as upstream aggr='u'; beta shares
    # u,v,residual,bias in the encoder (aggr='uvm'). Heads remain private.
    return variant == "beta" or not (name.endswith("vv") or name.endswith("mask"))


class FactorizedFLClient:
    def __init__(self, identity, train_set, test_set, initial_model, source_id=0):
        self.id, self.train_set, self.test_set = identity, train_set, test_set
        self.global_test_id = source_id
        self.model = deepcopy(initial_model).cpu()
        self.optimizer_state = None

    def receive(self, body_state):
        self.model.hidden_layers.load_state_dict(body_state)

    def train(self, args, round_index):
        model = self.model.to(args["device"]).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"],
                                    momentum=args["momentum"], weight_decay=args["weight_decay"])
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        loader = DataLoader(self.train_set, batch_size=args["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(
                                args["seed"] + 1000003 * round_index + self.id))
        loss_total, ce_total, penalty_total, steps = 0.0, 0.0, 0.0, 0
        for _ in range(args["epochs"]):
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = finite(model(finite(x.to(args["device"], dtype=torch.float32), "inputs")), "logits")
                ce = F.cross_entropy(logits, y.to(args["device"]))
                penalty = args["factorized_l1"] * mask_penalty(model)
                loss = finite(ce + penalty, "loss")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        finite(parameter.grad, "gradient")
                optimizer.step()
                for parameter in model.parameters():
                    finite(parameter, "parameter")
                loss_total += float(loss.detach())
                ce_total += float(ce.detach())
                penalty_total += float(penalty.detach())
                steps += 1
        self.optimizer_state = optimizer.state_dict()
        for state in self.optimizer_state["state"].values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = finite(value, "optimizer state").cpu()
        self.model.cpu()
        return dict(train_loss=loss_total / max(steps, 1),
                    train_ce=ce_total / max(steps, 1),
                    train_l1=penalty_total / max(steps, 1), optimizer_steps=steps)


class FactorizedFLServer:
    def __init__(self, initial_model, args):
        self.initial = cloned_state(initial_model.hidden_layers)
        self.args = args
        self.uploads = {}

    def personalized_body(self, identity):
        """Pure function of frozen uploads: never rewrite donors on download."""
        if not self.uploads:
            return {k: v.clone() for k, v in self.initial.items()}, [], []
        identities = sorted(self.uploads)
        if identity not in self.uploads:
            # Authors' cold-start fallback averages complete cached encoders.
            masses = torch.tensor([self.uploads[i]["size"] for i in identities], dtype=torch.float64)
            weights = masses / masses.sum()
            base = self.initial
            cold_start = True
        else:
            target = self.uploads[identity]["vector"]
            similarities = F.cosine_similarity(target[None], torch.stack(
                [self.uploads[i]["vector"] for i in identities]), dim=1)
            similarities[identities.index(identity)] = 1.0
            ordering = sorted(range(len(identities)), key=lambda i: (-float(similarities[i]), identities[i]))
            eligible = int((similarities >= self.args["factorized_tau"]).sum())
            count = min(len(identities), max(eligible, self.args["factorized_min_neighbors"]))
            chosen = ordering[:count]
            weights = (similarities[chosen].double() * self.args["factorized_similarity_scale"]).softmax(0)
            identities = [identities[i] for i in chosen]
            base = self.uploads[identity]["body"]
            cold_start = False
        result = {key: value.clone() for key, value in base.items()}
        for key, value in base.items():
            if (cold_start or shared_parameter(key, self.args["factorized_variant"])) and value.is_floating_point():
                result[key] = finite(sum(self.uploads[i]["body"][key].double() * weight
                                         for i, weight in zip(identities, weights)).to(value.dtype), "aggregate")
        return result, identities, weights.tolist()

    def update(self, selected):
        for client in selected:
            self.uploads[client.id] = dict(body=cloned_state(client.model.hidden_layers),
                                           vector=matching_vector(client.model.hidden_layers),
                                           size=len(client.train_set))
