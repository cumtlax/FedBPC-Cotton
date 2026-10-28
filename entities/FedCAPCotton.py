"""Known-source control for Idea v3; probe and formal SGD kernels are unchanged."""
from copy import deepcopy

import numpy as np

from entities.FedCAP import FedCAPServer


class FedCAPSourceServer(FedCAPServer):
    def __init__(self, args):
        super().__init__(args)
        self.implementation += "_known_source_v1"
        self._source_membership = None

    def regroup(self, clients, selected, round_index):
        sources = {c.id: int(c.train_set.concept_id) for c in clients}
        if set(sources.values()) != {0, 1, 2}:
            raise ValueError("Fixed-source Cotton grouping requires sources 0, 1 and 2")
        if self._source_membership is None:
            # Assign ALL clients before the first partial draw. Otherwise an
            # unseen client could be detached by an apparent initial split.
            self._source_membership = sources
            self.membership = sources.copy()
            self.states = {s: deepcopy(self.initial.head.state_dict()) for s in (0, 1, 2)}
            self.retained_heads = {}
            self.next_id = 3
        elif sources != self._source_membership:
            raise ValueError("Source membership changed during a static Cotton run")
        # Keep identical measurements/RNG work to the inferred-group control.
        result = super().regroup(clients, selected, round_index)
        self.last_regroup_stats.update(cotton_grouping="source", cluster_assignment="known_source")
        return result

    def partition_descriptors(self, clients, ids):
        return np.asarray([self._source_membership[i] for i in ids], dtype=int), 0.0

    def regroup_summary(self):
        return dict(super().regroup_summary(), cotton_grouping="source", cluster_assignment="known_source")
