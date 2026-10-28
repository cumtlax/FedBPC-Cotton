# -*- coding: utf-8 -*-
"""Build the FedCAP-compatible cotton root from the extracted full-genome dataset.

Inputs  (external production project, NOT shipped with this training package):
  --source  E:\argi\cotton_pfl\dataset_full  (source_c{1,2,3}.csv)
  --family  E:\argi\cotton_pfl\metadata\gene_families.tsv

The generated data/cotton/ TSV layout IS included in the package; running this
script is only needed when regenerating from the production project.

Outputs (read by utils/cotton_dataset.py):
  data/cotton/clients/c1_cotton_nitrogen/train_ready.tsv
  data/cotton/clients/c2_cotton_mutant/train_ready.tsv
  data/cotton/clients/c3_cotton_salt/train_ready.tsv
  data/cotton/metadata/gene_families.tsv
"""
import csv
import os
import shutil
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.environ.get("COTTON_SOURCE", r"E:\argi\cotton_pfl\dataset_full")
DEFAULT_FAM = os.environ.get("COTTON_FAMILY", r"E:\argi\cotton_pfl\metadata\gene_families.tsv")
SRC = DEFAULT_SRC
FAM = DEFAULT_FAM
DST = os.path.join(BASE, "data", "cotton")

SOURCES = {
    "c1": "c1_cotton_nitrogen",
    "c2": "c2_cotton_mutant",
    "c3": "c3_cotton_salt",
}
FEATURES = [
    "atac_prom_HN_AS98", "atac_prom_HN_WT", "atac_prom_LN_AS98", "atac_prom_LN_WT",
    "atac_prom_mean",
    "drebind_occ_4", "drebind_occ_5", "drebind_occ_6", "drebind_occ_mean", "drebind_occ_sd",
    "gc", "DRE_full", "DRE_core", "ABRE", "G_box", "GCC_box",
    "gene_len", "n_exon_blocks", "exon_len_total", "exon_fraction",
    "expr_mean_log2cpm", "expr_sd_log2cpm", "expr_detected_frac",
]
META_COLS = ["label", "log2FoldChange", "padj"]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    global SRC, FAM
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild data/cotton from the production dataset")
    ap.add_argument("--source", default=DEFAULT_SRC, help="dataset_full directory (source_c*.csv)")
    ap.add_argument("--family", default=DEFAULT_FAM, help="gene_families.tsv")
    a = ap.parse_args(argv)
    SRC, FAM = a.source, a.family
    if not os.path.isdir(SRC):
        raise SystemExit("source not found: %s\nThis training package already ships data/cotton; "
                         "only run this script with --source/--family pointing at the production project." % SRC)
    if not os.path.isfile(FAM):
        raise SystemExit("family map not found: %s" % FAM)
    os.makedirs(os.path.join(DST, "metadata"), exist_ok=True)
    families = {}
    with open(FAM, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            families[row["gene_id"]] = row["family_id"]
    n_mapped_families = len(families)
    n_fallback = 0

    header = None
    dataset_genes = set()
    for key, dirname in SOURCES.items():
        rows = load_csv(os.path.join(SRC, "source_%s.csv" % key))
        outdir = os.path.join(DST, "clients", dirname)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "train_ready.tsv")
        missing_family = 0
        with open(outpath, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            cols = ["gene_id"] + FEATURES + META_COLS
            if header is None:
                header = cols
            elif header != cols:
                raise RuntimeError("feature schema mismatch across sources")
            w.writerow(cols)
            for r in rows:
                gid = r["gene_id"]
                if gid not in families:
                    # New-only genes absent from the legacy family map fall back
                    # to a singleton family (documented in README).
                    families[gid] = gid
                    n_fallback += 1
                dataset_genes.add(gid)
                vals = [gid] + [r[c] for c in FEATURES] + [r[c] for c in META_COLS]
                w.writerow(vals)
        print("%-18s rows=%6d missing_family=%d -> %s" % (dirname, len(rows), missing_family, outpath))

    fam_out = os.path.join(DST, "metadata", "gene_families.tsv")
    with open(fam_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "family_id"])
        for gid in sorted(families):
            w.writerow([gid, families[gid]])
    print("family map ->", fam_out, "| legacy rows:", n_mapped_families, "| singleton fallback:", n_fallback)
    sizes = Counter(families[g] for g in dataset_genes)
    multi = sum(1 for v in sizes.values() if v > 1)
    print("[family-check] dataset genes=%d families=%d multi-member families=%d"
          % (len(dataset_genes), len(sizes), multi))
    if multi == 0:
        print("[family-check][WARN] all families are singletons among dataset genes: "
              "the family split is effectively gene-disjoint (see data/README.md)")
    print("done")


if __name__ == "__main__":
    main()
