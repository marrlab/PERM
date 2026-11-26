#!/usr/bin/env python
"""Power-Seek memory-gene calling on the CONTROL cells of ANY perturb-seq h5ad.

Generic replacement for the per-dataset runners (powerseek_{marson,frangieh,velten,
perturbfate,hsc}_control.py). Same faithful counts-native prep, same untouched
algorithm (mem_genes_benchmark/utils.powerseek_memory_gene_calling); only the
dataset-specific bits are now arguments:

  --ctrl-col / --ctrl-values   which cells are non-targeting controls
                               (or --ctrl-nonempty: any non-empty value, as Velten)
  --group-col / --group        restrict to one biological group (donor x condition,
                               replicate, drug arm ...). Groups are NEVER pooled here --
                               pick one per task.
  --symbol-map                 Ensembl -> symbol CSV (columns ensembl, gene_symbol);
                               columns sharing a symbol are SUMMED, unmapped dropped,
                               exactly as prepare_melanoma.collapse_to_symbol.

Dataset traps handled automatically:
  * X layout is read from the h5ad: CSR (cell-major), CSC (gene-major, Frangieh --
    a CSR row-reader returns garbage there without erroring) or dense.
  * the var index is whatever var.attrs['_index'] names (Frangieh: 'gene_symbol').
  * X must be RAW COUNTS. Non-integer X aborts (override: --allow-noninteger), because
    a log1p(CP10k) matrix would run silently and give a wrong answer. Do NOT use
    utils.build_powerseek_input on counts -- it starts with expm1.

Example (Velten rep2, as the old runner):
  powerseek_control.py --h5ad .../velten_perturb.h5ad --name velten \
      --ctrl-col control_group --ctrl-nonempty --group-col replicate --group rep2 \
      --min-ctrl-expr 0.1 --out-dir ...

Run one group per job; the README lists the groups used in the paper.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import os
# setdefault, so the sbatch's export wins. 1 thread is right for a SMALL input
# (n_cells <= n_genes -> tiny cells x cells eigendecompositions, where thread
# overhead dominates); on ~18k control cells the algorithm switches to the
# 3000x3000 genes x genes branch and multithreaded BLAS pays for itself.
# Results can differ in the last bits with thread count -- reproduce with the same one.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time, argparse, unicodedata
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix, issparse

BENCH = P.MEMORY_GENES
sys.path.insert(0, BENCH)
from utils import _powerseek_outlier_count, powerseek_memory_gene_calling

# Their filter is a literal `> 200 cells` on a 3000 x 7590 input = 2.6% of cells.
# On ALL control cells we use their literal 200 -- no deviation. Only when
# subsampling does the absolute number have to become their fraction, or 200 of
# 800 cells would mean a 25% detection floor they never applied.
MIN_CELLS_ORIGINAL = 200
MIN_CELL_FRAC = 200 / 7590


def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


def obs_column(f, col):
    """One obs column as an array of str, for categorical (both anndata encodings) or plain."""
    o = f["obs"]
    if col not in o:
        raise SystemExit(f"obs has no column '{col}'; have {sorted(o.keys())}")
    node = o[col]
    if isinstance(node, h5py.Group):                          # anndata >= 0.8 categorical
        cats = np.array([str(_dec(c)) for c in node["categories"][:]], dtype=object)
        codes = node["codes"][:]
    elif "__categories" in o and col in o["__categories"]:    # anndata < 0.8 categorical
        cats = np.array([str(_dec(c)) for c in o["__categories"][col][:]], dtype=object)
        codes = node[:]
    else:
        return np.array([str(_dec(v)) for v in node[:]], dtype=object)
    out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")   # -1 = NaN -> ''
    return out.astype(object)


def control_cell_index(path, ctrl_col, ctrl_values, ctrl_nonempty, group_col, group):
    with h5py.File(path, "r") as f:
        c = obs_column(f, ctrl_col)
        if ctrl_nonempty:
            m = np.array([v.strip() != "" for v in c])
        else:
            missing = [v for v in ctrl_values if v not in set(c)]
            if missing:
                raise SystemExit(f"control value(s) {missing} not in obs['{ctrl_col}']")
            m = np.isin(c, ctrl_values)
        if group_col:
            g = obs_column(f, group_col)
            if group not in set(g):
                raise SystemExit(f"group '{group}' not in obs['{group_col}']: {sorted(set(g))[:30]}")
            m &= g == group
    return np.flatnonzero(m)


def _genes(f):
    vg = f["var"]
    key = _dec(vg.attrs.get("_index", "_index"))
    return np.array([_dec(g) for g in vg[key][:]])


def x_layout(path):
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            return "dense"
        enc = _dec(X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", "")))
        if "csr" in enc:
            return "csr"
        if "csc" in enc:
            return "csc"
    raise SystemExit(f"cannot tell the X layout of {path} (encoding-type '{enc}')")


def read_rows_csr(path, rows):
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        X = f["X"]
        n_genes = int(X.attrs["shape"][1])
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]
        d, i, ip = [], [], [0]
        for s, e in zip(indptr[rows], indptr[rows + 1]):
            d.append(data[s:e]); i.append(indices[s:e]); ip.append(ip[-1] + (e - s))
        genes = _genes(f)
    return csr_matrix((np.concatenate(d).astype(np.float64), np.concatenate(i), np.array(ip)),
                      shape=(len(rows), n_genes)), genes


def read_rows_csc(path, rows):
    """Dense (cells x genes) float64 for `rows`, scanning the CSC gene columns."""
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        X = f["X"]
        n_genes = int(X.attrs["shape"][1])
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]
        out = np.zeros((len(rows), n_genes), dtype=np.float64)
        for j in range(n_genes):
            s, e = int(indptr[j]), int(indptr[j + 1])
            if e == s:
                continue
            idx = indices[s:e]
            k = np.searchsorted(rows, idx)
            np.clip(k, 0, len(rows) - 1, out=k)
            hit = rows[k] == idx                      # which nonzeros belong to our cells
            if hit.any():
                out[k[hit], j] = data[s:e][hit]
            if j % 5000 == 0:
                print(f"    gene column {j}/{n_genes}", flush=True)
        genes = _genes(f)
    return out, genes


def read_rows_dense(path, rows):
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        out = f["X"][rows, :].astype(np.float64)       # h5py needs increasing indices
        genes = _genes(f)
    return out, genes


READERS = {"csr": read_rows_csr, "csc": read_rows_csc, "dense": read_rows_dense}


def _values(M):
    return M.data if issparse(M) else M[M != 0]


def _colmean(M):
    return np.asarray(M.mean(axis=0)).ravel()


def collapse_to_symbol(M, genes, symbol_map):
    """Sum Ensembl columns sharing a gene symbol; drop unmapped (as prepare_melanoma does)."""
    sym = pd.read_csv(symbol_map).set_index("ensembl")["gene_symbol"].reindex(genes)
    valid = sym.notna().to_numpy()
    uniq, inv = np.unique(sym[valid].astype(str).to_numpy(), return_inverse=True)
    onehot = csr_matrix((np.ones(inv.size), (np.arange(inv.size), inv)), shape=(inv.size, len(uniq)))
    print(f"  symbol map: {int(valid.sum())}/{len(genes)} ids mapped -> {len(uniq)} symbols", flush=True)
    return (csr_matrix(M)[:, valid] @ onehot).tocsr(), uniq


def build_input(M, genes, min_cells, n_top_cv):
    """Power-Seek's DataPrepV2 on raw counts (their step 1 is counts, not log1p)."""
    df = pd.DataFrame((M.toarray() if issparse(M) else M).T, index=genes)    # genes x cells
    df = df[(df != 0).sum(axis=1) > 0]
    df = df.divide(df.sum(axis=0).astype(np.float64), axis=1).astype("float32")
    df = df[(df != 0).sum(axis=1) > min_cells]
    print(f"  detection filter (> {min_cells} cells): {df.shape[0]} genes", flush=True)
    df = df[df.apply(lambda r: _powerseek_outlier_count(r.to_numpy()) == 0, axis=1)]
    print(f"  outlier filter: {df.shape[0]} genes", flush=True)
    cv = df.std(axis=1) / df.mean(axis=1)
    return {"expr": df.iloc[np.argsort(-cv.to_numpy())].head(n_top_cv).astype(np.float64)}


def _safe(s):
    """ASCII, filename-safe ('IFNγ' -> 'IFNg')."""
    s = s.replace("γ", "g")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--name", required=True, help="dataset name for the output files, e.g. velten")
    ap.add_argument("--ctrl-col", required=True, help="obs column holding the control label")
    cg = ap.add_mutually_exclusive_group(required=True)
    cg.add_argument("--ctrl-values", nargs="+", help="value(s) of --ctrl-col that mark controls")
    cg.add_argument("--ctrl-nonempty", action="store_true",
                    help="controls = any non-empty value of --ctrl-col (Velten's control_group)")
    ap.add_argument("--group-col", default=None, help="obs column defining the groups; omit = all cells")
    ap.add_argument("--group", default=None, help="the one group to run")
    ap.add_argument("--symbol-map", default=None,
                    help="CSV with columns ensembl,gene_symbol; collapse ids to symbols first")
    ap.add_argument("--allow-noninteger", action="store_true",
                    help="skip the raw-counts check (only if you know X is counts-like)")
    ap.add_argument("--ref-ctrl-expr", default=None,
                    help="optional per-gene control-mean CSV to check the reader against "
                         "(e.g. <ds>_gene_ctrl_expr.csv); prints r and max abs diff")
    ap.add_argument("--n-cells", type=int, default=0,
                    help="0 (default) = every control cell of the group; >0 subsamples")
    ap.add_argument("--half", type=int, default=0, choices=[0, 1, 2],
                    help="split-half reproducibility: 1 or 2 = one of two DISJOINT blocks of "
                         "--n-cells cells (default n-cells = half of the controls)")
    ap.add_argument("--min-cells", type=int, default=None,
                    help="detection filter; default = their literal 200 on a full run, "
                         "or 2.6%% of the cells when subsampling")
    ap.add_argument("--downsample-frac", type=float, default=0.0,
                    help="binomial thinning of the counts (depth experiment); 0 = off")
    ap.add_argument("--min-ctrl-expr", type=float, default=0.0,
                    help="restrict the INPUT gene set to genes with mean control expression "
                         "above this (counts/cell) BEFORE Power-Seek's prep. 0 = off (the "
                         "authors' prep). 0.1 = the project-wide detection floor, which makes "
                         "the universe testable in the responsiveness readout.")
    ap.add_argument("--n-top-cv", type=int, default=3000)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if bool(a.group_col) != bool(a.group):
        ap.error("--group-col and --group go together")
    os.makedirs(a.out_dir, exist_ok=True)
    label = a.group if a.group else a.name

    t = time.time()
    idx_all = control_cell_index(a.h5ad, a.ctrl_col, a.ctrl_values, a.ctrl_nonempty,
                                 a.group_col, a.group)
    print(f"{label}: {len(idx_all):,} control cells ({time.time() - t:.0f}s)", flush=True)
    if len(idx_all) == 0:
        raise SystemExit("no control cells selected")
    if a.n_cells and len(idx_all) < a.n_cells:
        raise SystemExit(f"only {len(idx_all)} control cells, need {a.n_cells}")

    if a.half:
        # one shuffle, two disjoint blocks -> the halves share no cells
        n = a.n_cells or len(idx_all) // 2
        perm = np.random.default_rng(a.seed).permutation(idx_all)
        if len(perm) < 2 * n:
            raise SystemExit(f"need {2 * n} cells for two halves, have {len(perm)}")
        idx = perm[(a.half - 1) * n: a.half * n]
        tag_n = f"{n}h{a.half}"
        min_cells = a.min_cells if a.min_cells is not None else int(MIN_CELL_FRAC * n)
    elif a.n_cells:
        # same seed across groups -> the subsample is not a source of between-group difference
        idx, tag_n = np.random.default_rng(a.seed).choice(idx_all, a.n_cells, replace=False), str(a.n_cells)
        min_cells = a.min_cells if a.min_cells is not None else int(MIN_CELL_FRAC * a.n_cells)
    else:
        idx, tag_n = idx_all, "all"
        min_cells = a.min_cells if a.min_cells is not None else MIN_CELLS_ORIGINAL

    t = time.time()
    layout = x_layout(a.h5ad)
    M, genes = READERS[layout](a.h5ad, idx)
    vals = _values(M)
    if not a.allow_noninteger and vals.size and not np.array_equal(vals, np.round(vals)):
        raise SystemExit("X is not integer-valued -- Power-Seek's prep needs RAW counts. "
                         "Point --h5ad at the counts, or pass --allow-noninteger if you are sure.")
    print(f"  X layout {layout}, max value {vals.max() if vals.size else 0:g}", flush=True)
    if a.symbol_map:
        M, genes = collapse_to_symbol(M, genes, a.symbol_map)   # before any filtering
    if a.downsample_frac:
        # binomial thinning: emulate a shallower library WITHOUT changing which cells
        # are present, so depth is the only thing that differs from the full-depth arm
        M = csr_matrix(M)
        rs = np.random.default_rng(a.seed + 1000 * a.half)
        M.data = rs.binomial(M.data.astype(np.int64), a.downsample_frac).astype(np.float64)
        M.eliminate_zeros()
        ng = np.asarray((M > 0).sum(axis=1)).ravel(); um = np.asarray(M.sum(axis=1)).ravel()
        print(f"  downsampled to frac={a.downsample_frac}: genes/cell median={np.median(ng):.0f}, "
              f"UMIs/cell median={np.median(um):.0f}", flush=True)
    print(f"  matrix {M.shape} ({time.time() - t:.0f}s)", flush=True)
    print(f"  min_cells = {min_cells} "
          f"({100 * min_cells / M.shape[0]:.1f}% of cells; theirs was 2.6%)", flush=True)

    if a.ref_ctrl_expr and os.path.exists(a.ref_ctrl_expr):
        # end-to-end reader check: our per-gene control mean must reproduce the file the
        # responsiveness pipeline was built from. An indexing bug would show up here.
        ref = pd.read_csv(a.ref_ctrl_expr, index_col=0).iloc[:, 0]
        ours = pd.Series(_colmean(M), index=genes)
        common = ref.index.intersection(ours.index)
        r = np.corrcoef(ref[common], ours[common])[0, 1]
        md = float((ref[common] - ours[common]).abs().max())
        print(f"  reader check vs {os.path.basename(a.ref_ctrl_expr)}: r={r:.6f}, "
              f"max_abs_diff={md:.3e} over {len(common)} genes", flush=True)

    if a.min_ctrl_expr > 0:
        # Applied BEFORE their prep: their detection+CV filter was designed for an
        # already-filtered Cell Ranger matrix. On a transcriptome-wide input the CV
        # ranking fills with sparse genes (CV = sd/mean is inflated at tiny means),
        # and the resulting universe sits below the responsiveness floor.
        keep = _colmean(M) > a.min_ctrl_expr
        print(f"  expression floor > {a.min_ctrl_expr}: {int(keep.sum())} of {len(genes)} "
              "genes kept as the Power-Seek input", flush=True)
        M, genes = M[:, keep], genes[keep]
    inp = build_input(M, genes, min_cells, a.n_top_cv)
    print(f"  Power-Seek input {inp['expr'].shape}", flush=True)
    del M

    t = time.time()
    out = powerseek_memory_gene_calling(inp, top_n=a.top_n)
    print(f"  done in {(time.time() - t) / 60:.1f} min, power-law n = {out['n_eigenvalues']}", flush=True)

    floor_tag = f"_floor{a.min_ctrl_expr:g}" if a.min_ctrl_expr > 0 else ""
    floor_tag += f"_ds{a.downsample_frac:g}" if a.downsample_frac else ""
    grp_tag = f"_{_safe(a.group)}" if a.group else ""
    tag = f"{_safe(a.name)}{grp_tag}_n{tag_n}_cv{a.n_top_cv}{floor_tag}"
    mem = out["memory_genes"].copy()
    mem.insert(0, a.group_col or "dataset", label)
    mem.to_csv(f"{a.out_dir}/powerseek_control_{tag}.csv", index_label="gene")
    out["delta"].to_csv(f"{a.out_dir}/powerseek_control_delta_{tag}.csv", index_label="gene")
    print(f"  saved -> {a.out_dir}/powerseek_control_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
