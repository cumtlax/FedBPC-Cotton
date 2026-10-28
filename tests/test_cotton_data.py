# -*- coding: utf-8 -*-
"""Load the cotton root through the FedCAP loader and check client invariants."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from utils.cotton_dataset import distribute_cotton  # noqa: E402


def test_cotton_clients():
    train, test, global_tests = distribute_cotton(
        client_num=9, seed=1, root=ROOT / "data" / "cotton",
        feature_set="no-expression", include_test=True)
    assert len(train) == 9 and len(test) == 9 and len(global_tests) == 3
    for i in range(9):
        assert train[i].data.shape[1] == 23
        counts = np.bincount(np.asarray(train[i].targets), minlength=2)
        assert counts.min() >= 5
    # family-disjoint: no family crosses train/test within a source
    for i in range(0, 9, 3):
        assert not (set(train[i].family_ids) & set(test[i].family_ids))

def test_family_split_is_effectively_gene_level():
    """Documented limitation: among dataset genes every family is a singleton."""
    import csv
    from collections import Counter
    fam = {}
    with open(ROOT / "data" / "cotton" / "metadata" / "gene_families.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fam[row["gene_id"]] = row["family_id"]
    genes = set()
    with open(ROOT / "data" / "cotton" / "clients" / "c1_cotton_nitrogen" / "train_ready.tsv",
              encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            genes.add(row["gene_id"])
    sizes = Counter(fam[g] for g in genes)
    assert max(sizes.values()) == 1, "family map changed: update data/README claims"
