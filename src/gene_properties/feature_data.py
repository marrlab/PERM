#!/usr/bin/env python
"""Per-gene sequence and annotation features, labelled with the paper's gene sets.

The features themselves (expression, CpG island promoters, promoter CpG content,
publication and GO-term counts, STRING degree, protein domains, gnomAD constraint,
gene length) are precomputed per dataset in the feature matrices read here. What
this module adds is the LABEL of each gene, and it takes that label from the figure
code rather than from any gene-list file:

    memory      >= 2 of the 4 methods, each method cut to its own top 300 first
                (figures.characterisation_table, tier >= 2: the genes of Fig 1d
                and Suppl Fig 1d, identical to build_consensus.py's merged set)
    background  genes in none of the four methods' top 300 (the paper's
                non-memory definition), from the current method lists
    other       called by at least one method but not memory; excluded

Reusing the figure code is the point: an earlier version labelled genes from a
gene-list file built on a different consensus definition (top 300 of the uncapped
ranking at >= 3 of 5 methods), which shared only ~75% of its genes with the paper's
sets. The precomputed matrices still carry those old labels; they are dropped here
so nothing downstream can use them by accident.
"""
# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_matching as FM                                         # noqa: E402

# the precomputed per-gene feature matrices; override with PERM_FEATURES
FEATURES = os.environ.get("PERM_FEATURES", os.path.join(P.OUTPUTS, "gene_properties"))
OUT_DIR = os.path.join(P.OUTPUTS, "gene_properties")

ORDER = ["T-lineage", "Harmange", "LARRY", "Bandler"]
PAPER_NAME = {"T-lineage": "Michaels", "Harmange": "Harmange",
              "LARRY": "Weinreb", "Bandler": "Bandler"}
TOP_N = 300

# labels baked into the precomputed matrices on the OLD gene set; never used
_STALE = ["tier", "is_top300_min3", "is_top300_min4", "is_top100_min3",
          "is_top100_min4", "is_unanimous_n5", "n_methods", "median_rank",
          "in_consensus"]


def memory_labels(top_n=TOP_N, datasets=ORDER):
    """(dataset, gene, set) for every memory and background gene, from the figures."""
    os.environ.setdefault("MEMGENE_TOP_N", str(top_n))
    sys.path.insert(0, P.LIB)
    sys.path.insert(0, P.FIGURES)
    import figures as F
    T = F.characterisation_table(datasets=list(datasets), tiers=(2,), top_n=top_n,
                                 verbose=False)
    T = T.assign(set=np.where(T.group == "background", "background", "memory"))
    return T[["dataset", "gene", "set"]].drop_duplicates(["dataset", "gene"])


def load(features_dir=FEATURES, top_n=TOP_N, datasets=ORDER):
    """Feature matrix (tiers A and C) for the four datasets, with a `set` column."""
    M = FM.prep(pd.read_csv(f"{features_dir}/feature_matrix_tierA.csv"))
    M = M[M.dataset.isin(datasets)]
    parts = []
    for sp in ("human", "mouse"):
        C = pd.read_csv(f"{features_dir}/feature_tierC_promoter_{sp}.csv")
        C = C.drop_duplicates("gene_upper").set_index("gene_upper")
        C = C.drop(columns=[c for c in ("chrom", "strand", "tss") if c in C.columns])
        parts.append(M[M.species == sp].set_index("gene_upper")
                     .join(C, how="left").reset_index())
    M = pd.concat(parts, ignore_index=True)
    M["log_cgi_dist"] = np.log10(M.cgi_dist + 1)
    for w in (500, 1000, 2000):
        M[f"cgi_prom_{w}"] = M[f"cgi_prom_{w}"].eq(True).astype(float)

    # promoter sequence (tier C) and isoforms / conservation (tier D)
    parts = []
    for sp in ("human", "mouse"):
        S = (pd.read_csv(f"{features_dir}/feature_tierC_promseq_{sp}.csv")
               .drop_duplicates("gene_upper").set_index("gene_upper"))
        T = (pd.read_csv(f"{features_dir}/feature_tierD_{sp}.csv")
               .drop_duplicates("gene_upper").set_index("gene_upper"))
        parts.append(M[M.species == sp].set_index("gene_upper")
                     .join(S, how="left").join(T, how="left").reset_index())
    M = pd.concat(parts, ignore_index=True)

    # derived columns
    M["log_isoforms"] = np.log10(M.n_isoforms)
    M["log_intron_len"] = np.log10(M.total_intron_len + 1)
    M["log_gene_span"] = np.log10(M.gene_span)
    M["log_tx_span"] = np.log10(M.tx_span)
    for a in ("bp", "mf", "cc", "noniea"):
        M[f"log_go_{a}"] = np.log10(M[f"n_go_{a}"] + 1)
    M["is_protein_coding"] = M.gene_type.eq("protein_coding").astype(float)

    M = M.drop(columns=[c for c in _STALE if c in M.columns])
    lab = memory_labels(top_n=top_n, datasets=datasets)
    M = M.merge(lab, on=["dataset", "gene"], how="left")
    M["set"] = M["set"].fillna("other")

    n = M.groupby(["dataset", "set"]).size().unstack(fill_value=0)
    want = lab[lab.set == "memory"].groupby("dataset").size()
    for d in datasets:
        got = int(n.loc[d, "memory"]) if d in n.index else 0
        if got != int(want.get(d, 0)):
            raise SystemExit(f"{d}: {got} memory genes joined to the feature matrix, "
                             f"{int(want.get(d, 0))} in the figures' set")
    return M
