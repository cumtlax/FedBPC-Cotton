"""FedAvg with inverse-frequency weights derived exclusively from local TRAIN labels."""
import numpy as np
import torch

from entities.base import Client, Server


def train_class_weights(targets, classes=2):
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=classes)
    if len(counts) != classes or (counts <= 0).any():
        raise ValueError("Weighted FedAvg requires every class in the local training set")
    return torch.as_tensor(counts.sum() / (classes * counts), dtype=torch.float32)


class WeightedFedAvgClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)
        weights = train_class_weights(train_set.targets, args["num_classes"]).to(args["device"])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
