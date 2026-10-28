"""Confirm whole-client regrouping using fresh evidence on common clients."""


class RegroupConfirmation:
    policy = "common_clients_fresh_evidence_v1"
    verified_policy = "common_clients_independent_fresh_partition_v2"

    @staticmethod
    def validate_policy(value):
        if value not in ("legacy", "fresh-verified"):
            raise ValueError("collaboration_policy must be legacy or fresh-verified")
        return value

    def __init__(self, required=1, collaboration_policy="legacy"):
        if not isinstance(required, int) or isinstance(required, bool) or required < 1:
            raise ValueError("regroup_confirmations must be a positive integer")
        self.required = required
        self.collaboration_policy = self.validate_policy(collaboration_policy)
        if self.collaboration_policy == "fresh-verified":
            self.policy = self.verified_policy
        self.pending = None

    @staticmethod
    def verify_fresh_partition(groups, old_members, fresh_labels):
        """Verify a cache-based proposal using an independently clustered draw.

        No labels/concept IDs or new model evaluations are used. Require two
        freshly described clients from EACH merging source/split destination,
        not merely one fresh client in the combined candidate group. Compare
        relations rather than arbitrary numeric clustering labels. Noise or
        absent source evidence means unknown, never approval.
        """
        assignment = {i: position for position, group in enumerate(groups) for i in group}
        result = dict(topology_fresh_checked=True, topology_fresh_verified=False,
                      topology_fresh_compared_pairs=0, topology_fresh_disagreements=0,
                      topology_fresh_missing_sources=[])
        destinations = {}
        missing = []
        for position, group in enumerate(groups):
            sources = {old_members[i] for i in group if old_members[i] is not None}
            for source in sources:
                destinations.setdefault(source, []).append(position)
            if len(sources) > 1:
                for source in sorted(sources):
                    count = sum(i in fresh_labels and fresh_labels[i] >= 0
                                for i in group if old_members[i] == source)
                    if count < 2:
                        missing.append(dict(candidate=position, source=source, fresh_clients=count))
            elif not sources:
                count = sum(i in fresh_labels and fresh_labels[i] >= 0 for i in group)
                if count < 2:
                    missing.append(dict(candidate=position, source=None, fresh_clients=count))
        for source, positions in destinations.items():
            if len(positions) < 2:
                continue
            for position in positions:
                count = sum(i in fresh_labels and fresh_labels[i] >= 0
                            for i in groups[position] if old_members[i] == source)
                if count < 2:
                    missing.append(dict(candidate=position, source=source, fresh_clients=count))
        if missing:
            return dict(result, topology_fresh_missing_sources=missing,
                        topology_fresh_reason="insufficient-fresh-source-support")
        ids = sorted(set(assignment) & set(fresh_labels))
        if len(ids) < 2 or any(fresh_labels[i] < 0 for i in ids):
            return dict(result, topology_fresh_reason="insufficient-fresh-partition")
        differences = sum((assignment[i] == assignment[j]) != (fresh_labels[i] == fresh_labels[j])
                          for position, i in enumerate(ids) for j in ids[:position])
        pairs = len(ids) * (len(ids) - 1) // 2
        return dict(result, topology_fresh_compared_pairs=pairs,
                    topology_fresh_disagreements=differences,
                    topology_fresh_verified=differences == 0,
                    topology_fresh_reason="verified" if differences == 0 else "fresh-partition-disagrees")

    @staticmethod
    def relation_changes(groups, old_members):
        assignment = {i: position for position, group in enumerate(groups) for i in group}
        ids = sorted(i for i in assignment if old_members[i] is not None)
        changes, pairs = 0, 0
        for position, i in enumerate(ids):
            for j in ids[:position]:
                changes += ((assignment[i] == assignment[j]) !=
                            (old_members[i] == old_members[j]))
                pairs += 1
        return changes, pairs

    def observe(self, groups, fresh_ids, version, round_index, changed, cap_allowed,
                fresh_verified=True):
        """Numeric cluster IDs do not matter; partitions must agree on overlap.

        Require at least two common clients in each group and 50% coverage of
        the smaller proposal. Each group must also receive a freshly described
        common client, so repeatedly clustering an unchanged cache is not proof.
        Reference refresh, conflicting evidence and gaps restart confirmation.
        """
        assignment = {i: position for position, group in enumerate(groups) for i in group}
        result = dict(confirmation_required=self.required, confirmation_count=0,
                      confirmation_common_clients=0, confirmation_overlap_coverage=0.0,
                      topology_change_proposed=bool(changed), transition_confirmed=True)
        if not cap_allowed:
            self.pending = None
            return dict(result, transition_confirmed=False, confirmation_reason="cluster-cap")
        if not fresh_verified:
            self.pending = None
            return dict(result, transition_confirmed=False, confirmation_reason="unverified-fresh-partition")
        if self.required == 1:
            self.pending = None
            return dict(result, confirmation_count=int(changed), confirmation_reason="disabled")
        if not changed:
            self.pending = None
            return dict(result, confirmation_reason="unchanged")

        previous = self.pending
        count, reason = 1, "first-evidence"
        if previous is not None:
            if previous["version"] != version:
                reason = "reference-refresh"
            elif previous["round"] + 1 != round_index:
                reason = "round-gap"
            else:
                common = set(assignment) & set(previous["assignment"])
                coverage = len(common) / max(1, min(len(assignment), len(previous["assignment"])))
                result.update(confirmation_common_clients=len(common),
                              confirmation_overlap_coverage=coverage)
                intersections = {}
                for i in common:
                    pair = (previous["assignment"][i], assignment[i])
                    intersections.setdefault(pair, set()).add(i)
                old_groups = set(previous["assignment"].values())
                new_groups = set(assignment.values())
                old_counts = {k: sum(a == k for a, _ in intersections) for k in old_groups}
                new_counts = {k: sum(b == k for _, b in intersections) for k in new_groups}
                consistent = (len(old_groups) == len(new_groups) and
                              all(n == 1 for n in old_counts.values()) and
                              all(n == 1 for n in new_counts.values()))
                sufficient = coverage >= 0.5 and all(len(ids) >= 2 for ids in intersections.values())
                fresh = bool(intersections) and all(ids & set(fresh_ids) for ids in intersections.values())
                if not consistent:
                    reason = "conflicting-partition"
                elif not sufficient:
                    reason = "insufficient-overlap"
                elif not fresh:
                    reason = "insufficient-fresh-evidence"
                else:
                    count, reason = previous["count"] + 1, "consistent-evidence"
        confirmed = count >= self.required
        self.pending = None if confirmed else dict(assignment=assignment, version=version,
                                                   round=round_index, count=count)
        return dict(result, confirmation_count=count, confirmation_reason=reason,
                    transition_confirmed=confirmed)

    def checkpoint(self):
        pending = None if self.pending is None else {
            **self.pending, "assignment": self.pending["assignment"].copy()}
        return dict(policy=self.policy, required=self.required, pending=pending)
