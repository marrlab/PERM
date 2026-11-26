"""
Per-gene covariates for the responsiveness matching, computed on Marson CONTROL cells.

Why control cells: detection rate and dispersion are baseline properties of a gene.
Computing them on perturbed cells would let the perturbation response leak into the
variable we are trying to match on.

Source: $PERM_RAW/ctrl.h5ad
  296,479 cells x 13,992 genes, CSR int16 RAW counts, var_names = HGNC symbols
  (same gene space as marson_gene_summary_cpm.csv / marson_gene_ctrl_expr.csv).
  obs['dataset'] carries 19 values; the 12 Marson2025_D{1-4}_{Rest,Stim8hr,Stim48hr}
  groups are kept, the Replogle/XAtlas controls are dropped.

Outputs (to $PERM_DATA/marson/):
  marson_gene_covariates.csv           pooled over the 12 Marson control groups
  marson_gene_covariates_by_group.csv  same quantities per donor x timepoint

Columns:
  n_cells         cells the gene was measured in (constant per group)
  mean_cpm        mean CPM across cells
  var_cpm         variance of CPM across cells
  detection_rate  fraction of cells with a non-zero count      <- the dropout axis
  nb_phi          method-of-moments NB overdispersion, (var-mean)/mean^2
  disp_resid      log10(var) minus the binned-median log10(var) at that mean
                  -> "more variable than other genes of the same abundance"

nb_phi and disp_resid are two views of the same thing. nb_phi is interpretable but
blows up at low mean; disp_resid is expression-detrended by construction and is the
one to match on. Both are written so the choice stays auditable.

Single streaming pass, accumulating per-gene sums; peak memory is one chunk.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../..")))  # repo root
from perm_paths import DATA, RAW  # noqa: E402
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

CTRL_H5AD = RAW + '/ctrl.h5ad'
OUT_DIR = DATA + '/marson'
MARSON_PREFIX = 'Marson2025_'


def _accumulate(X_chunk):
    """Return (nnz_per_gene, sum_cpm, sumsq_cpm) for one chunk of raw counts."""
    X = sp.csr_matrix(X_chunk, dtype=np.float64)

    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0                      # keep empty cells from producing NaN
    inv = sp.diags(1e6 / lib)
    C = inv @ X                              # CPM, still sparse

    nnz = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)
    s1 = np.asarray(C.sum(axis=0)).ravel()
    s2 = np.asarray(C.multiply(C).sum(axis=0)).ravel()
    return nnz, s1, s2


def _finalize(nnz, s1, s2, n_cells, n_bins=50):
    """Turn the accumulators into the covariate table."""
    mean = s1 / n_cells
    var = (s2 - n_cells * mean ** 2) / max(n_cells - 1, 1)
    var = np.maximum(var, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(mean > 0, (var - mean) / mean ** 2, np.nan)

    # expression-detrended dispersion: residual from the binned mean-variance trend
    disp_resid = np.full_like(mean, np.nan)
    ok = (mean > 0) & (var > 0)
    if ok.sum() > n_bins:
        lm, lv = np.log10(mean[ok]), np.log10(var[ok])
        # rank-based bins so every bin holds the same number of genes
        bins = pd.qcut(pd.Series(lm).rank(method='first'), n_bins, labels=False).values
        trend = pd.Series(lv).groupby(bins).transform('median').values
        disp_resid[ok] = lv - trend

    return pd.DataFrame({
        'n_cells': n_cells,
        'mean_cpm': mean,
        'var_cpm': var,
        'detection_rate': nnz / n_cells,
        'nb_phi': phi,
        'disp_resid': disp_resid,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=CTRL_H5AD)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--chunk', type=int, default=20000, help='cells per chunk')
    ap.add_argument('--n-bins', type=int, default=50,
                    help='mean-expression bins for the dispersion trend')
    args = ap.parse_args()

    a = ad.read_h5ad(args.h5ad, backed='r')
    genes = a.var_names.to_numpy()
    # obs['dataset'] is categorical -> .to_numpy() gives an object array, which
    # np.char.startswith cannot dispatch on. Do the prefix test in pandas.
    ds_ser = a.obs['dataset'].astype(str)
    is_marson = ds_ser.str.startswith(MARSON_PREFIX).to_numpy()
    ds = ds_ser.to_numpy()
    groups = sorted(set(ds[is_marson]))
    print(f"{args.h5ad}: {a.shape[0]:,} cells; "
          f"{is_marson.sum():,} Marson control cells in {len(groups)} groups", flush=True)
    if is_marson.sum() == 0:
        raise SystemExit(f"no cells with dataset prefix {MARSON_PREFIX!r}")

    G = len(genes)
    pooled = [np.zeros(G, np.int64), np.zeros(G), np.zeros(G)]
    per_group = {}

    for grp in groups:
        idx = np.flatnonzero(ds == grp)          # sorted, required for backed slicing
        acc = [np.zeros(G, np.int64), np.zeros(G), np.zeros(G)]
        for start in range(0, len(idx), args.chunk):
            sl = idx[start:start + args.chunk]
            nnz, s1, s2 = _accumulate(a.X[sl])
            for k, v in enumerate((nnz, s1, s2)):
                acc[k] += v
                pooled[k] += v
        per_group[grp] = _finalize(*acc, n_cells=len(idx), n_bins=args.n_bins)
        per_group[grp].index = genes
        print(f"  {grp:<28s} {len(idx):>7,} cells  "
              f"median detection {per_group[grp].detection_rate.median():.3f}", flush=True)

    n_total = int(is_marson.sum())
    out = _finalize(*pooled, n_cells=n_total, n_bins=args.n_bins)
    out.index = genes
    out.index.name = 'gene'

    p1 = f"{args.out_dir}/marson_gene_covariates.csv"
    out.to_csv(p1)
    print(f"\nwrote {p1}  {out.shape}", flush=True)

    by = pd.concat(per_group, names=['group', 'gene'])
    p2 = f"{args.out_dir}/marson_gene_covariates_by_group.csv"
    by.to_csv(p2)
    print(f"wrote {p2}  {by.shape}", flush=True)

    print("\npooled summary (genes above the 0.1 CPM floor):")
    hi = out[out.mean_cpm > 0.1]
    print(hi[['mean_cpm', 'detection_rate', 'nb_phi', 'disp_resid']]
          .describe().to_string(float_format=lambda x: f'{x:.4g}'))
    print(f"\ngenes above floor: {len(hi):,} of {len(out):,}")
    for c in ('detection_rate', 'disp_resid'):
        r = hi[['mean_cpm', c]].dropna().corr(method='spearman').iloc[0, 1]
        print(f"Spearman(log mean_cpm, {c}) = {r:+.3f}")


if __name__ == '__main__':
    main()
