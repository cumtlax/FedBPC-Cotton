import numpy as np


def label_swap_pairs(class_count=6, classes=None):
    """Three disjoint concept groups; the historical setting changes labels 1..6.

    Expanded scenarios repeat the same three-pair block. class_count is the
    UNION of affected labels across groups, not the number changed per client.
    """
    if (isinstance(class_count, bool) or not isinstance(class_count, int) or
            class_count < 6 or class_count % 6):
        raise ValueError("drift_class_count must be a positive multiple of 6")
    if classes is not None and class_count >= classes:
        raise ValueError("drift_class_count must be smaller than num_classes (label 0 stays unchanged)")
    return tuple(tuple((start + 2 * group, start + 2 * group + 1)
                       for start in range(1, class_count, 6)) for group in range(3))


def _client_group(client_id):
    return 0 if client_id % 10 < 3 else 1 if client_id % 10 < 6 else 2


def drift_dataset(dataset, class_a, class_b):
    """
    Perform concept drift by swapping two labels class_a and class_b.
    :param dataset: The dataset to be drifted.
    :param class_a: The drifted label.
    :param class_b: The drifted label.
    :return:
    """
    class_a_indices = np.where(np.array(dataset.targets) == class_a)[0]
    class_b_indices = np.where(np.array(dataset.targets) == class_b)[0]
    for i in class_a_indices:
        dataset.targets[i] = -1
    for i in class_b_indices:
        dataset.targets[i] = class_a
    class_a_indices = np.where(np.array(dataset.targets) == -1)[0]
    for i in class_a_indices:
        dataset.targets[i] = class_b


def sudden_drift(clients, global_test_sets, _round, class_count=6):
    print(f"Sudden drift occurs at {_round}")

    pairs = label_swap_pairs(class_count)
    for group, swaps in enumerate(pairs):
        for a, b in swaps:
            drift_dataset(global_test_sets[group + 1], a, b)

    for client in clients:
        group = _client_group(client.id)
        for a, b in pairs[group]:
            drift_dataset(client.train_set, a, b)
            drift_dataset(client.test_set, a, b)
        client.global_test_id = group + 1


def incremental_drift(clients, global_test_sets, _round, class_count=6):
    print(f"Incremental drift occurs at {_round}")

    pairs = label_swap_pairs(class_count)
    if _round not in (100, 110, 120):
        return
    active_group = (_round - 100) // 10
    for a, b in pairs[active_group]:
        drift_dataset(global_test_sets[active_group + 1], a, b)

    for client in clients:
        if _client_group(client.id) == active_group:
            for a, b in pairs[active_group]:
                drift_dataset(client.train_set, a, b)
                drift_dataset(client.test_set, a, b)
            client.global_test_id = active_group + 1
