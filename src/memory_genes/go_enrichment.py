#!/usr/bin/env python
"""GO over-representation for the consensus memory genes, against an
EXPRESSION-MATCHED null as well as the plain transcriptome.

Why the matched null. Memory genes are strongly expression-biased and strongly
cell-type-specific -- the two confounds figures 2f / suppl 5a exist to control.
A plain ORA against the measured transcriptome therefore returns the abundant
and the lineage-restricted gene families whatever the biology: on the four
datasets it gives 144-489 terms at q<0.05, topped by 'neutrophil degranulation'
in BOTH haematopoietic systems and by the ribosome in melanoma. Those are
predictions of the confound, not findings. So every term is scored twice:

  naive   Fisher's exact test of the memory set against the measured universe,
          BH-corrected over terms. This is the conventional number.
  matched the same overlap compared with control sets of the SAME SIZE redrawn
          from the SAME expression bins (50 bins, n_draws draws), which is the
          null used elsewhere in this project (figures._cross_null). A term is
          reported as surviving only if it also beats this.

Species. The per-dataset consensus lists are in NATIVE symbols (T-lineage and
Harmange human, LARRY and Bandler mouse); only the cross_dataset/recurrence
*_allspecies files are ortholog-mapped. --map-mouse (default) maps the mouse
datasets to human so the human GO gmt applies, which costs genes (LARRY 300 ->
273, universe 25,289 -> 16,873). Pass --gmt-mouse with a mouse GO gmt to run
those two natively instead, which is the better analysis; GO term IDs are
species-agnostic, so terms still compare across datasets either way.

Usage
-----
  python go_enrichment.py                       # all four, GO BP, 1000 draws
  python go_enrichment.py --datasets T-lineage --draws 200
  python go_enrichment.py --gmt $PERM_DATA/GO_Molecular_Function_2021.gmt \
                          --out gene_lists/go_mf_min2_top300_matched.csv
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, P.FIGURES)
sys.path.insert(0, P.LIB)

GMT_BP = f"{P.DATA}/GO_Biological_Process_2021.gmt"

# The gene set MUST be the one the figures use. build_consensus.py reuses the
# figure code itself: every method is cut to its own top N FIRST, then the methods
# are counted. That is what "consensus memory genes (>=2 of 4 methods, top 300)"
# means in the paper.
#
# Do NOT point this at a list ranked over the UNCAPPED method lists (the old
# top300_by_dataset.csv). The two definitions differ by about one gene in six
# (Jaccard 0.81-0.89 per dataset), so the enrichment would describe a gene set
# the paper never reports.
LISTS = os.path.join(P.MEMGENES_ROOT, "gene_lists", "figure_consensus_top300.csv")
OUT = os.path.join(P.MEMGENES_ROOT, "gene_lists", "go_bp_min2_top300_native.csv")

# figure_consensus_top300.csv keys datasets by first author; figures.DATASETS
# uses the dataset names
DATASET_KEY = {"Michaels": "T-lineage", "Harmange": "Harmange",
               "Weinreb": "LARRY", "Bandler": "Bandler"}


def native_mean_expr(F, label):
    """Mean expression per gene in the dataset's OWN symbols.

    figures._mean_expr does the same chunked pass but maps mouse -> human and
    reindexes onto a shared universe, which is wrong here: the bins have to be
    built over the dataset's whole measured transcriptome, before any ortholog
    loss, or the background the controls are drawn from is already filtered.
    """
    import anndata as ad
    cfg = F.DATASETS[label]
    A = ad.read_h5ad(cfg["h5ad"], backed="r")
    n, s = A.n_obs, np.zeros(A.n_vars)
    for X, _, _ in A.chunked_X(4000):
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        s += np.asarray(X.sum(0)).ravel()
    ids = (A.var[cfg["var_col"]].astype(str).values if cfg["var_col"]
           else np.array(list(map(str, A.var_names))))
    A.file.close()
    # a symbol can appear twice in var (var_names_make_unique); keep its max, as
    # figures._mean_expr does, so the two copies do not land in different bins
    return pd.DataFrame({"id": ids, "m": s / n}).groupby("id")["m"].max()


def matched_draws(bins, mem_idx, n_bins, n_draws, rng):
    """n_draws control sets with the memory set's own expression-bin profile."""
    memb = [np.where(bins == b)[0] for b in range(n_bins)]
    prof = np.bincount(bins[mem_idx], minlength=n_bins)
    out = np.empty((n_draws, len(mem_idx)), dtype=np.int64)
    for i in range(n_draws):
        out[i] = np.concatenate([
            rng.choice(memb[b], c, replace=c > len(memb[b]))
            for b, c in enumerate(prof) if c])
    return out


def run_one(F, label, mem_native, gmt, map_mouse, n_bins, n_draws, seed,
            min_term, max_term, verbose=True):
    """map_mouse=False means `gmt` is already in this dataset's own symbols."""
    from marson_KO import enrich_against_gmt

    expr = native_mean_expr(F, label).dropna()
    expr = expr[expr > 0]                      # unmeasured genes cannot be binned
    mem = [g for g in dict.fromkeys(mem_native) if g in expr.index]

    # ---- symbols the gmt speaks ------------------------------------------
    if map_mouse and F.DATASETS[label].get("species") == "mouse":
        o = F.ortho_map(label, verbose=False)
        keep = [g for g in expr.index if g in o]
        expr = expr.loc[keep]
        expr.index = [F.human_symbol(o[g]) for g in keep]
        expr = expr.groupby(level=0).max()
        mem = [F.human_symbol(o[g]) for g in mem if g in o]
    elif F.DATASETS[label].get("species") == "human":
        expr.index = [F.human_symbol(g) for g in expr.index]
        expr = expr.groupby(level=0).max()
        mem = [F.human_symbol(g) for g in mem]
    mem = [g for g in dict.fromkeys(mem) if g in expr.index]

    uni = list(expr.index)
    pos = {g: i for i, g in enumerate(uni)}
    mem_idx = np.array([pos[g] for g in mem])
    bins = pd.qcut(expr.rank(method="first"), n_bins, labels=False).to_numpy()

    # ---- terms x genes incidence over this universe ----------------------
    terms, rows, cols = [], [], []
    for t, genes in gmt.items():
        hit = [pos[g] for g in genes if g in pos]
        if min_term <= len(hit) <= max_term:
            rows += [len(terms)] * len(hit)
            cols += hit
            terms.append(t)
    T = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)),
                      shape=(len(terms), len(uni)))

    k_obs = np.asarray(T[:, mem_idx].sum(1)).ravel()
    if n_draws:
        rng = np.random.default_rng(seed)
        draws = matched_draws(bins, mem_idx, n_bins, n_draws, rng)
        K = np.empty((n_draws, len(terms)), np.int32)
        for i in range(n_draws):
            K[i] = np.asarray(T[:, draws[i]].sum(1)).ravel()
        k_null = K.mean(0)
        # one-sided empirical p with the +1 correction, so p is never 0
        p_matched = (1 + (K >= k_obs[None, :]).sum(0)) / (n_draws + 1)
    else:
        k_null = np.full(len(terms), np.nan)
        p_matched = np.full(len(terms), np.nan)

    naive = (enrich_against_gmt(mem, gmt, uni, min_term_size=min_term,
                                max_term_size=max_term, top_n=10 ** 9)
             .set_index("term"))

    from statsmodels.stats.multitest import multipletests
    df = pd.DataFrame({"dataset": label, "term": terms,
                       "n_memory": len(mem), "universe": len(uni),
                       "term_size": np.asarray(T.sum(1)).ravel(),
                       "k_observed": k_obs,
                       "k_matched_mean": k_null,
                       "fold_vs_matched": np.where(k_null > 0, k_obs / np.maximum(k_null, 1e-9),
                                                   np.nan),
                       "p_matched": p_matched})
    df["q_matched"] = (multipletests(df.p_matched, method="fdr_bh")[1]
                       if n_draws else np.nan)
    df["p_naive"] = df.term.map(naive["p_raw"]).astype(float)
    df["q_naive"] = df.term.map(naive["q_bh"]).astype(float)
    df["overlap_genes"] = df.term.map(naive["overlap_genes"])
    df = df.sort_values("p_naive" if not n_draws else "p_matched").reset_index(drop=True)

    if verbose:
        print(f"=== {label}: {len(mem)} memory genes, universe {len(uni)}, "
              f"{len(terms)} terms tested   q<0.05: {int((df.q_naive < .05).sum())}")
        for _, r in df[df.q_naive < .05].head(8).iterrows():
            print(f"    q={r.q_naive:.1e}  {int(r.k_observed):3d}/{int(r.term_size):3d}"
                  f"  {r.term[:64]}")
        print()
    return df


def crossdataset_overlap(table, out, alpha=0.05):
    """Which terms recur across datasets: one row per term, its q and overlap count
    in each dataset, and in how many datasets it is significant.

    Built from a HUMAN-symbol table (every dataset on the same human ontology, mouse
    datasets mapped through orthologs), because a cross-dataset comparison needs one
    term space. The universe is the terms MEASURABLE IN EVERY DATASET, i.e. testable
    (10-500 genes in that dataset's measured transcriptome) in all of them, so a term
    can only fail to recur because it was not enriched, never because it could not
    be tested. A term counts in a dataset if q_naive < alpha there.
    """
    d = pd.read_csv(table)
    order = [c for c in DATASET_KEY.values() if c in set(d.dataset)]
    present = d.groupby("term").dataset.nunique()
    measurable = present.index[present == len(order)]
    d = d[d.term.isin(measurable)]
    q = d.pivot(index="term", columns="dataset", values="q_naive")
    k = d.pivot(index="term", columns="dataset", values="k_observed")
    res = pd.concat([q[order].add_prefix("q_"), k[order].add_prefix("k_")], axis=1)
    res["n_datasets_sig"] = (q[order] < alpha).sum(axis=1)
    res.insert(0, "ontology", res.index.str.extract(r"^GO(BP|MF|CC)_",
                                                     expand=False).fillna(""))
    res = (res.assign(_worst=q[order].max(axis=1))
              .sort_values(["n_datasets_sig", "_worst"], ascending=[False, True])
              .drop(columns="_worst"))
    res.index.name = "term"
    res.to_csv(out)
    n = res.n_datasets_sig
    print(f"{len(res)} terms | significant in all {len(order)}: {int((n == len(order)).sum())}"
          f" | in >= {len(order) - 1}: {int((n >= len(order) - 1).sum())}  -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmt", default=GMT_BP,
                    help="gmt for the human datasets (and for mouse when "
                         "--gmt-mouse is not given, via orthologs)")
    ap.add_argument("--gmt-mouse", default=None,
                    help="gmt in MOUSE symbols; mouse datasets then run natively "
                         "with no ortholog mapping")
    ap.add_argument("--lists", default=LISTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--draws", type=int, default=0,
                    help="expression-matched control draws; 0 = plain ORA only")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-term", type=int, default=10)
    ap.add_argument("--max-term", type=int, default=500)
    ap.add_argument("--no-map-mouse", action="store_true",
                    help="keep mouse symbols (needs a mouse gmt via --gmt)")
    ap.add_argument("--overlap-from", default=None,
                    help="build the cross-dataset overlap table from this HUMAN-symbol "
                         "result table and write it to --out; no enrichment is run")
    a = ap.parse_args()

    if a.overlap_from:
        crossdataset_overlap(a.overlap_from, a.out)
        return

    os.environ.setdefault("MEMGENE_TOP_N", "300")
    import figures as F
    from marson_KO import load_gmt

    gmt_h = load_gmt(a.gmt)
    gmt_m = load_gmt(a.gmt_mouse) if a.gmt_mouse else None
    print(f"human gmt: {os.path.basename(a.gmt)} ({len(gmt_h)} terms)")
    print("mouse gmt: " + (f"{os.path.basename(a.gmt_mouse)} ({len(gmt_m)} terms), "
                            "mouse datasets run NATIVELY"
                            if gmt_m else "none, mouse datasets mapped to human"))
    print(f"matched draws: {a.draws or 'none (plain ORA)'}\n")

    # Long format: one row per (dataset, group, gene). Take the MERGED view and
    # the >=2-method call, which is exactly the figures' gene set. The per-group
    # rows (T1/T2, E/M, E13/E14, 2.0/4.0/6.0) are the supplementary view and are
    # deliberately not pooled in here.
    lists = pd.read_csv(a.lists)
    need = {"dataset", "group", "gene", "consensus_ge2"}
    missing = need - set(lists.columns)
    if missing:
        sys.exit(f"{a.lists} is missing {sorted(missing)}. Expected the output of "
                 "build_consensus.py; a wide per-dataset list is the wrong input "
                 "(see the note beside LISTS).")
    lists = lists[(lists["group"] == "merged") & (lists["consensus_ge2"])]
    out = []
    for ds, sub in lists.groupby("dataset", sort=False):
        label = DATASET_KEY.get(str(ds))
        if label is None:
            print("skip (no DATASETS key):", ds)
            continue
        if a.datasets and label not in a.datasets:
            continue
        mem = sorted(set(sub["gene"].astype(str)))
        print(f"{label}: {len(mem)} consensus genes (>=2 of 4 methods, top 300)")
        native = gmt_m is not None and F.DATASETS[label].get("species") == "mouse"
        out.append(run_one(F, label, mem, gmt_m if native else gmt_h,
                           map_mouse=not native, n_bins=a.bins, n_draws=a.draws,
                           seed=a.seed, min_term=a.min_term, max_term=a.max_term))

    ALL = pd.concat(out, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ALL.to_csv(a.out, index=False)
    print(f"{len(ALL)} rows -> {a.out}")


if __name__ == "__main__":
    main()
