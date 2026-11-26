#!/usr/bin/env python
"""
prepare_weinstock.py - donor-corrected CPM LFC and per-gene responsiveness for the
Weinstock CRISPR T-cell Perturb-seq screen (GSE271788).

The sixth main screen. The other five go through prepare_screen.py; this one does
not, because its input is a bulk-style count matrix rather than a 10x h5ad and its
LFC is computed within donor. Replaces the build cells of weinstock_analysis.ipynb;
the notebook's diagnostic plots are deliberately not reproduced.

Why donor-corrected rather than pooled
--------------------------------------
This screen is donor-batch confounded: a pooled LFC compares a perturbation's
cells against ALL control cells regardless of donor, so donor composition leaks
into every effect. compute_donor_corrected_lfc instead compares each perturbation
to the control cells of the SAME donor and then aggregates. That is the recipe
marson_per_donor.py later generalises to two nuisance axes (donor x stimulation
state), and it is the same idea as the within-sub-group LFC prepare_screen.py uses
for Shi (cell type) and Bowness (replicate).

The notebook also computed a pooled LFC, but only to show that confound in a
figure. Nothing downstream reads it, so it is not written here.

The gene symbols are CACHED
---------------------------
marson_KO.load_weinstock_adata maps Ensembl ids to symbols by querying MyGene
live, with no cache, so the gene universe is whatever MyGene returns on the day
it runs. Rebuilding the published tables in Nov 2026 gave 45,509 genes against
their 45,511: every value on the 45,463 shared genes is identical, but 46 names
appeared and 48 vanished because MyGene had updated (IL1F10 became IL38, new
antisense names like ANKRD11-DT arrived). That makes the published universe
unreproducible by anyone running the script later.

So the mapping is resolved once and cached to --symbol-map. Delete that file to
re-query. The same pattern is used for Xu in prepare_screen.py.

Outputs (in $PERM_DATA/):
  weinstock_lfc_cpm_donor_corrected.csv   per-KO x gene LFC, computed within donor
  weinstock_gene_summary_cpm.csv          per-gene responsiveness (mean_abs_lfc, ...)
  weinstock_gene_ctrl_expr.csv            baseline control expression as MEAN CPM,
                                          because load_weinstock_adata CPM-normalises
                                          the matrix before anything else. The other
                                          five screens write mean raw counts per cell,
                                          so the same numeric floor is NOT the same
                                          threshold here as it is there.
"""
import argparse
import os
import sys

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

sys.path.insert(0, P.LIB)          # shared perturb-seq libraries
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from marson_KO import (load_weinstock_adata, compute_donor_corrected_lfc,  # noqa: E402
                       summarize_gene_response)


def symbol_map(ens_ids, path):
    """Ensembl -> HGNC symbol, cached so the gene universe is reproducible."""
    if os.path.exists(path):
        sym = pd.read_csv(path, index_col=0).squeeze("columns")
        print(f"symbol map: {len(sym)} cached ids <- {path}", flush=True)
        return sym
    print("no cached symbol map; querying MyGene (needs internet). The result is "
          f"written to {path} and reused from then on.", flush=True)
    import mygene
    res = mygene.MyGeneInfo().querymany(list(ens_ids), scopes="ensembl.gene",
                                        fields="symbol", species="human",
                                        as_dataframe=True)
    sym = res["symbol"].dropna()
    sym = sym[~sym.index.duplicated()]
    sym.to_csv(path)
    print(f"symbol map: {len(sym)}/{len(ens_ids)} mapped -> {path}", flush=True)
    return sym


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=f"{P.DATA}/GSE271788_dedup_counts.txt.gz",
                    help="the GEO dedup count matrix")
    ap.add_argument("--out-dir", default=P.DATA)
    ap.add_argument("--symbol-map", default=f"{P.DATA}/weinstock_ensembl2symbol.csv",
                    help="cached Ensembl->symbol map; built on first run")
    ap.add_argument("--control-label", default="control")
    ap.add_argument("--perturbation-col", default="perturbation")
    ap.add_argument("--large-change-threshold", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"reading {args.counts}", flush=True)
    # map_symbols=False, then the cached map: the library's own mapping queries
    # MyGene every time. These three lines are the library's, unchanged, so the
    # result is identical given the same map.
    adata = load_weinstock_adata(args.counts, map_symbols=False)
    sym = symbol_map(adata.var_names.tolist(), args.symbol_map)
    adata = adata[:, adata.var_names.isin(sym.index)].copy()
    adata.var_names = sym.reindex(adata.var_names).astype(str).values
    adata.var_names_make_unique()
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)

    # 1. within-donor LFC
    lfc, per_donor = compute_donor_corrected_lfc(adata, control_label=args.control_label)
    lfc.to_csv(f"{args.out_dir}/weinstock_lfc_cpm_donor_corrected.csv")
    print(f"donor-corrected LFC: {lfc.shape}  ({len(per_donor)} donors)", flush=True)

    # 2. per-gene responsiveness
    gs = summarize_gene_response(lfc, large_change_threshold=args.large_change_threshold)
    gs.to_csv(f"{args.out_dir}/weinstock_gene_summary_cpm.csv")

    # 3. baseline control expression, the covariate the matched backgrounds draw on
    ctrl = adata[adata.obs[args.perturbation_col] == args.control_label]
    expr = pd.Series(np.asarray(ctrl.X.mean(axis=0)).ravel(), index=adata.var_names)
    expr.to_csv(f"{args.out_dir}/weinstock_gene_ctrl_expr.csv")

    print(f"\nFINAL: {lfc.shape[0]} KOs x {lfc.shape[1]} genes, "
          f"{len(expr):,} genes with control expression "
          f"({ctrl.n_obs:,} control cells)")
    print("wrote weinstock_lfc_cpm_donor_corrected.csv, weinstock_gene_summary_cpm.csv, "
          f"weinstock_gene_ctrl_expr.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()
