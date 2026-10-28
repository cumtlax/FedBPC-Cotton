# -*- coding: utf-8 -*-
"""Data self-check: load cotton virtual clients through the FedCAP loader."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from utils.cotton_dataset import (COTTON_SOURCE_NAMES, distribute_cotton,  # noqa: E402
                                  resolve_cotton_roots)


def main(client_num=9, feature_set="no-expression"):
    data_root, client_root, family_path = resolve_cotton_roots(ROOT / "data" / "cotton")
    print("data_root:", data_root)
    print("client_root:", client_root)
    print("family:", family_path)
    train, test, global_tests = distribute_cotton(
        client_num=client_num, seed=1, root=ROOT / "data" / "cotton",
        feature_set=feature_set, include_test=True)
    print("train clients:", len(train), "| test clients:", len(test),
          "| global test sets:", len(global_tests))
    sizes = [len(train[i]) for i in range(client_num)]
    pos = [int(np.asarray(train[i].targets).sum()) for i in range(client_num)]
    print("train sizes:", sizes)
    print("train positives:", pos)
    print("network input width:", train[0].data.shape[1])
    print("active features:", len(train[0].input_features), "| excluded:",
          len(train[0].excluded_features))
    print("manifest data_root:", train[0].manifest["data_root"])
    for i in range(client_num):
        counts = np.bincount(np.asarray(train[i].targets), minlength=2)
        assert counts.min() >= 5, "client %d has <5 samples in a class: %s" % (i, counts)
    for g in global_tests:
        print("global test size: %6d | source: %s" % (
            len(g), COTTON_SOURCE_NAMES[g.concept_id]))
    print("OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=9)
    ap.add_argument("--feature-set", default="no-expression")
    a = ap.parse_args()
    main(a.clients, a.feature_set)
