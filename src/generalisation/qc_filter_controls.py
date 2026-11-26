#!/usr/bin/env python
"""Step 1 of power-seek_run: QC-filter the CONTROL cells of one raw perturb-seq h5ad.

One row of datasets.tsv per call (--row, 0-based, or --name). Per file:
  1. control cells = EXACT match of obs[ctrl_col] against ctrl_values (never a substring:
     `CTRL` is a real gene in Marson / XAtlas / Mali, and a guard refuses it as a label)
  2. read ONLY those rows of X (CSR / CSC / dense auto-detected, block-wise, so the
     100-350 GB files never sit in memory)
  3. per-cell QC metrics on ALL genes of the file
  4. the ExPert pipeline's MAD rule (ExPert-data src/canonical.py qc_mask_from_metrics),
     ported one-to-one. ONE deliberate difference: thresholds are computed on this file's
     CONTROL cells, not on all candidate cells -- essential-gene knockdowns make sick
     cells that would widen the MADs, and Power-Seek only ever sees the controls.
       drop if > 5 MAD (either side) on log1p total counts, log1p genes, % top-50 genes
       drop if > 3 MAD on % mito, or % mito > 20 (cancer) / 12 (non-cancer)
  5. gene ids -> symbols (symbol_col, or the var index); ids sharing a symbol are SUMMED
     (as powerseek_control.collapse_to_symbol); ids without a symbol keep their id
  6. counts stay RAW -- Power-Seek's prep normalises itself

Repro mode (repro_from != '-'): no QC. Takes exactly the old run's control cells and
genes from repro_from, in the old order (cell order changes floating-point sums), and
checks the counts are identical to the old file entry by entry.

Outputs
  $PERM_SCRATCH/power-seek_run/qc/<name>_ctrl_qc.h5ad   QC-passed controls, raw counts,
        symbol genes, obs['is_ctrl'] == 'ctrl'   (bulk -> scratch, home quota is 80 G)
  results/qc/<name>_qc_metrics.csv.gz   every control cell: metrics, which rule failed it
  results/qc/<name>_thresholds.json     medians / MADs / cutoffs, counts, gene bookkeeping
  results/qc/status/<name>.status       one-line progress record, read by progress.sh
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import os, sys, time, json, argparse, traceback
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix, coo_matrix, vstack
from scipy.stats import median_abs_deviation

# Configuration (datasets.tsv, tasks.tsv) ships with the code in this directory;
# results go to the data tree, where figures.py reads them (PS_RUN/results).
RUN_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(RUN_DIR, "datasets.tsv")
RESULTS = os.path.join(P.MEMGENES_ROOT, "power-seek_run", "results")
OUT_H5AD_DIR = f"{P.SCRATCH}/power-seek_run/qc"
OUT_QC_DIR = os.path.join(RESULTS, "qc")
STATUS_DIR = os.path.join(OUT_QC_DIR, "status")
# repro mode only: old symbol -> Ensembl (the colleague's map, built from datasets that
# carry both ids; gene_name, ensembl_id, source)
GENE_ID_MAP = f"{P.RAW}/ExPert-data-pipeline/resources/vocab/gene_id_map.tsv"

# ExPert-data canonical.py / statics.py
QC_NMADS, QC_MT_NMADS, QC_PERCENT_TOP = 5, 3, 50
MT_PERCENT_CANCER, MT_PERCENT_NORMAL = 20, 12


# --------------------------------------------------------------------------- status
class Status:
    """One TSV line, rewritten atomically: progress.sh reads it."""
    COLS = ["name", "state", "phase", "pct", "n_ctrl", "n_pass", "started", "updated", "msg"]

    def __init__(self, name):
        os.makedirs(STATUS_DIR, exist_ok=True)
        self.path = os.path.join(STATUS_DIR, f"{name}.status")
        self.f = dict(name=name, state="running", phase="start", pct="", n_ctrl="", n_pass="",
                      started=time.strftime("%Y-%m-%d %H:%M:%S"), updated="", msg="")
        self._last = 0.0
        self.set()

    def set(self, force=True, **kw):
        self.f.update({k: str(v) for k, v in kw.items()})
        now = time.time()
        if not force and now - self._last < 20:          # throttle progress writes
            return
        self._last = now
        self.f["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\t".join(self.f[c].replace("\t", " ").replace("\n", " ")
                               for c in self.COLS) + "\n")
        os.replace(tmp, self.path)


def log(msg):
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} | peak {rss:.1f} GB] {msg}", flush=True)


# --------------------------------------------------------------------------- h5ad io
def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


def index_key(g):
    return _dec(g.attrs.get("_index", "_index"))


def read_col(f, grp, col):
    """obs/var column as a numpy array (str for categoricals and strings).
    Handles anndata >= 0.8 categoricals, the old `__categories` encoding, plain arrays."""
    g = f[grp]
    if col not in g:
        raise SystemExit(f"{grp} has no column '{col}'; have {sorted(g.keys())}")
    node = g[col]
    if isinstance(node, h5py.Group):
        cats = np.array([str(_dec(c)) for c in node["categories"][:]], dtype=object)
        codes = node["codes"][:]
    elif "__categories" in g and col in g["__categories"]:
        cats = np.array([str(_dec(c)) for c in g["__categories"][col][:]], dtype=object)
        codes = node[:]
    else:
        v = node[:]
        return np.array([str(_dec(x)) for x in v], dtype=object) if v.dtype.kind in "OSU" else v
    out = np.full(len(codes), "", dtype=object)          # code -1 (NaN) -> ''
    ok = codes >= 0
    out[ok] = cats[codes[ok]]
    return out


def x_layout(X):
    if isinstance(X, h5py.Dataset):
        return "dense"
    enc = _dec(X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", "")))
    for k in ("csr", "csc"):
        if k in enc:
            return k
    raise SystemExit(f"cannot tell the X layout (encoding-type '{enc}')")


def read_csr_rows(X, rows, st, gap=4_000_000, max_span=100_000_000):
    """rows (sorted) of a CSR X. Nearby rows are coalesced into one contiguous read of at
    most `max_span` entries, so a 4.5M-row file costs a few hundred reads, not 200k."""
    n_var = int(X.attrs["shape"][1])
    indptr = X["indptr"][:].astype(np.int64)
    data, indices = X["data"], X["indices"]
    starts, ends = indptr[rows], indptr[rows + 1]
    lens = ends - starts
    out_d, out_i = [], []
    i, n = 0, len(rows)
    while i < n:
        a, j = starts[i], i
        while j + 1 < n and starts[j + 1] - ends[j] <= gap and ends[j + 1] - a <= max_span:
            j += 1
        b = ends[j]
        d, ix = data[a:b], indices[a:b]
        ls = lens[i:j + 1]
        before = np.concatenate([[0], np.cumsum(ls)[:-1]])
        pos = np.repeat(starts[i:j + 1] - a - before, ls) + np.arange(ls.sum())
        out_d.append(d[pos].astype(np.float32))
        out_i.append(ix[pos].astype(np.int32))
        i = j + 1
        st.set(force=False, phase="read X", pct=f"{100 * i / n:.0f}")
    ip = np.concatenate([[0], np.cumsum(lens)])
    return csr_matrix((np.concatenate(out_d), np.concatenate(out_i), ip), shape=(n, n_var))


def read_dense_rows(X, rows, st, target_elems=200_000_000):
    """rows (sorted) of a dense X, read in row bands aligned to the HDF5 chunks
    (row-by-row access on a compressed chunked dataset would decompress each chunk
    hundreds of times)."""
    n_obs, n_var = X.shape
    chunk = X.chunks[0] if X.chunks else 1
    band = max(chunk, (target_elems // n_var) // chunk * chunk)
    parts = []
    for s in range(0, n_obs, band):
        e = min(s + band, n_obs)
        sel = rows[(rows >= s) & (rows < e)]
        if sel.size:
            parts.append(csr_matrix(X[s:e, :][sel - s].astype(np.float32)))
        st.set(force=False, phase="read X", pct=f"{100 * e / n_obs:.0f}")
    return vstack(parts).tocsr()


def read_csc_rows(X, rows, n_obs, st, max_nnz=200_000_000):
    """rows (sorted) of a gene-major CSC X (Mali): scan gene-column blocks and keep the
    entries whose row is a control. A CSR row-reader on CSC returns garbage silently."""
    n_var = int(X.attrs["shape"][1])
    indptr = X["indptr"][:].astype(np.int64)
    data, indices = X["data"], X["indices"]
    rowmap = np.full(n_obs, -1, dtype=np.int64)
    rowmap[rows] = np.arange(len(rows))
    R, C, V = [], [], []
    j = 0
    while j < n_var:
        k = int(np.searchsorted(indptr, indptr[j] + max_nnz, side="right")) - 1
        k = min(max(k, j + 1), n_var)
        a, b = indptr[j], indptr[k]
        if b > a:
            r = rowmap[indices[a:b]]
            hit = r >= 0
            if hit.any():
                cols = np.repeat(np.arange(j, k, dtype=np.int32), np.diff(indptr[j:k + 1]))
                R.append(r[hit].astype(np.int32))
                C.append(cols[hit])
                V.append(data[a:b][hit].astype(np.float32))
        j = k
        st.set(force=False, phase="read X", pct=f"{100 * j / n_var:.0f}")
    M = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                   shape=(len(rows), n_var))
    return M.tocsr()


def read_rows(path, rows, st):
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        X = f["X"]
        layout = x_layout(X)
        if layout == "csr":
            M = read_csr_rows(X, rows, st)
        elif layout == "dense":
            M = read_dense_rows(X, rows, st)
        else:
            n_obs = len(f["obs"][index_key(f["obs"])])
            M = read_csc_rows(X, rows, n_obs, st)
    M.sum_duplicates()
    return M, layout


# --------------------------------------------------------------------------- QC
def qc_metrics(M, mito_mask):
    """Per-cell scalars, as ExPert streaming.stream_qc_metrics (computed in float64)."""
    n = M.shape[0]
    d = M.data.astype(np.float64)
    lens = np.diff(M.indptr)
    row_of = np.repeat(np.arange(n), lens)
    total = np.bincount(row_of, weights=d, minlength=n)
    n_genes = np.bincount(row_of[d > 0], minlength=n)
    m = mito_mask[M.indices]
    mito = np.bincount(row_of[m], weights=d[m], minlength=n)
    top = np.zeros(n)
    for j in range(n):
        lo, hi = M.indptr[j], M.indptr[j + 1]
        if hi - lo <= QC_PERCENT_TOP:
            top[j] = d[lo:hi].sum()
        else:
            top[j] = np.partition(d[lo:hi], -QC_PERCENT_TOP)[-QC_PERCENT_TOP:].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_mt = np.where(total > 0, mito / total * 100.0, 0.0)
        pct_top = np.where(total > 0, top / total * 100.0, 0.0)
    return pd.DataFrame({"total_counts": total, "n_genes_by_counts": n_genes,
                         f"pct_counts_in_top_{QC_PERCENT_TOP}_genes": pct_top,
                         "pct_counts_mt": pct_mt})


def _outlier(v, nmads):
    med, mad = np.median(v), median_abs_deviation(v)
    return (v < med - nmads * mad) | (med + nmads * mad < v)


def qc_mask(met, cancer):
    """ExPert canonical.qc_mask_from_metrics, with each failing rule kept separately."""
    mt_per = MT_PERCENT_CANCER if cancer else MT_PERCENT_NORMAL
    log_total = np.log1p(met["total_counts"].to_numpy())
    log_genes = np.log1p(met["n_genes_by_counts"].to_numpy().astype(np.float64))
    pct_top = met[f"pct_counts_in_top_{QC_PERCENT_TOP}_genes"].to_numpy()
    pct_mt = met["pct_counts_mt"].to_numpy()
    fails = pd.DataFrame({
        "fail_total_counts": _outlier(log_total, QC_NMADS),
        "fail_n_genes": _outlier(log_genes, QC_NMADS),
        "fail_pct_top50": _outlier(pct_top, QC_NMADS),
        "fail_mt_mad": _outlier(pct_mt, QC_MT_NMADS),
        "fail_mt_cap": pct_mt > mt_per,
    }, index=met.index)
    keep = ~fails.any(axis=1).to_numpy()
    thr = {"mt_per": mt_per, "nmads": QC_NMADS, "mt_nmads": QC_MT_NMADS,
           "percent_top": QC_PERCENT_TOP, "thresholds_computed_on": "control cells of this file"}
    for nm, v, k in (("log1p_total_counts", log_total, QC_NMADS),
                     ("log1p_n_genes_by_counts", log_genes, QC_NMADS),
                     (f"pct_counts_in_top_{QC_PERCENT_TOP}_genes", pct_top, QC_NMADS),
                     ("pct_counts_mt", pct_mt, QC_MT_NMADS)):
        med, mad = float(np.median(v)), float(median_abs_deviation(v))
        thr[nm] = {"median": med, "mad": mad, "lo": med - k * mad, "hi": med + k * mad}
    return keep, fails, thr


# --------------------------------------------------------------------------- genes
def collapse_symbols(M, names):
    """Sum columns sharing a name; first-occurrence order. Returns (M, names, n_merged)."""
    names = np.asarray(names, dtype=object)
    uniq, first, inv = np.unique(names.astype(str), return_index=True, return_inverse=True)
    if len(uniq) == len(names):
        return M, names, 0
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    onehot = csr_matrix((np.ones(len(names), dtype=np.float32),
                         (np.arange(len(names)), rank[inv])), shape=(len(names), len(uniq)))
    return (M @ onehot).tocsr(), uniq[order].astype(object), len(names) - len(uniq)


# --------------------------------------------------------------------------- main
def load_row(a):
    df = pd.read_csv(DATASETS, sep="\t", dtype=str, keep_default_na=False)
    if a.name:
        hit = df[df["name"] == a.name]
        if hit.empty:
            raise SystemExit(f"no dataset '{a.name}' in {DATASETS}")
        return hit.iloc[0]
    if not 0 <= a.row < len(df):
        raise SystemExit(f"--row {a.row} out of range (0..{len(df) - 1})")
    return df.iloc[a.row]


def run(r, st):
    name, path, cancer = r["name"], os.path.expandvars(r["path"]), r["cancer"] == "y"
    ctrl_values = [v for v in r["ctrl_values"].split(",") if v]
    repro = r["repro_from"] not in ("", "-")
    if any(v.strip().upper() == "CTRL" for v in ctrl_values):
        raise SystemExit("'CTRL' is a real gene (chymotrypsin-like), never a control label")
    t0 = time.time()
    info = {"name": name, "path": path, "ctrl_col": r["ctrl_col"], "ctrl_values": ctrl_values,
            "cancer": cancer, "mode": "repro" if repro else "qc"}

    # ---- obs: controls ------------------------------------------------------------
    st.set(phase="select controls")
    with h5py.File(path, "r") as f:
        barcodes = read_col(f, "obs", index_key(f["obs"])).astype(str)
        labels = read_col(f, "obs", r["ctrl_col"]).astype(str)
        gene_ids = read_col(f, "var", index_key(f["var"])).astype(str)
        if r["symbol_col"] not in ("", "-"):
            sym = read_col(f, "var", r["symbol_col"]).astype(str)
            sym = np.where((sym == "") | (sym == "nan"), gene_ids, sym)
        else:
            sym = gene_ids
    per_label = {v: int((labels == v).sum()) for v in ctrl_values}
    missing = [v for v, c in per_label.items() if c == 0]
    if missing:
        raise SystemExit(f"control label(s) {missing} not found in obs['{r['ctrl_col']}']")
    is_ctrl = np.isin(labels, ctrl_values)
    group_col = r.get("group_col", "-")
    grp_vals = None
    if group_col not in ("", "-"):
        # one biological group per run (Wong condition, Mali batch), as Marson is one
        # donor x condition per file: QC thresholds AND Power-Seek both see one group only
        with h5py.File(path, "r") as f:
            grp_vals = read_col(f, "obs", group_col).astype(str)
        if r["group"] not in set(grp_vals):
            raise SystemExit(f"group '{r['group']}' not in obs['{group_col}']: {sorted(set(grp_vals))}")
        n_all = int(is_ctrl.sum())
        is_ctrl &= grp_vals == r["group"]
        info.update(group_col=group_col, group=r["group"], n_ctrl_all_groups=n_all)
        per_label = {v: int(((labels == v) & is_ctrl).sum()) for v in ctrl_values}
    info.update(n_cells_file=int(len(labels)), n_ctrl_per_label=per_label)
    log(f"{name}: {int(is_ctrl.sum()):,} control cells of {len(labels):,} {per_label}")

    if repro:
        group = name[:-len("_repro")] if name.endswith("_repro") else name
        with h5py.File(r["repro_from"], "r") as g:
            old_bc = read_col(g, "obs", index_key(g["obs"])).astype(str)
            old_pert = read_col(g, "obs", "perturbation").astype(str)
            old_ds = read_col(g, "obs", "dataset").astype(str)
            old_genes = read_col(g, "var", index_key(g["var"])).astype(str)
        old_rows = np.flatnonzero((old_pert == "control") & (old_ds == group))   # old runner order
        want = np.array([b.rsplit(";", 1)[0] for b in old_bc[old_rows]])
        pos = pd.Series(np.arange(len(barcodes)), index=barcodes)
        if not pos.index.is_unique:
            raise SystemExit("raw barcodes are not unique -- cannot map the old cells")
        lost = ~pd.Index(want).isin(pos.index)
        if lost.any():
            raise SystemExit(f"{int(lost.sum())} old control barcodes not in the raw file")
        rows_old_order = pos[want].to_numpy()
        not_ntc = int((~is_ctrl[rows_old_order]).sum())
        info.update(repro_from=r["repro_from"], repro_group=group,
                    n_old_cells=int(len(want)), n_old_genes=int(len(old_genes)),
                    old_cells_not_labelled_control_in_raw=not_ntc)
        log(f"repro: {len(want):,} old cells ({not_ntc} not {ctrl_values} in raw), "
            f"{len(old_genes):,} old genes")
        rows = np.sort(rows_old_order)
    else:
        rows = np.flatnonzero(is_ctrl)
    st.set(n_ctrl=len(rows))

    # ---- X ------------------------------------------------------------------------
    st.set(phase="read X", pct=0)
    M, layout = read_rows(path, rows, st)
    vals = M.data
    if vals.size and not np.array_equal(vals, np.round(vals)):
        raise SystemExit("X is not integer-valued -- these must be RAW counts")
    info.update(x_layout=layout, x_max=float(vals.max()) if vals.size else 0.0,
                n_genes_file=int(M.shape[1]))
    log(f"read {M.shape} ({layout}), nnz {M.nnz:,}, max {info['x_max']:g}")

    # ---- metrics + QC ---------------------------------------------------------------
    st.set(phase="qc metrics", pct="")
    mito_mask = np.char.startswith(np.char.upper(sym.astype(str)), "MT-")
    if not mito_mask.any():
        raise SystemExit("no MT- genes found: % mito would be 0 and the mito filter a no-op")
    info["mito_genes"] = sorted(sym[mito_mask].tolist())
    met = qc_metrics(M, mito_mask)
    met.index = barcodes[rows]
    met.insert(0, "ctrl_label", labels[rows])
    if grp_vals is not None:
        met.insert(1, group_col, grp_vals[rows])
    if repro:
        keep = np.ones(len(rows), dtype=bool)
        fails = pd.DataFrame(index=met.index)
        thr = {"note": "repro mode: no QC applied"}
    else:
        keep, fails, thr = qc_mask(met, cancer)
        info["n_fail_per_rule"] = {c: int(fails[c].sum()) for c in fails.columns}
    info.update(n_ctrl=int(len(rows)), n_pass=int(keep.sum()), thresholds=thr)
    st.set(n_pass=int(keep.sum()))
    log(f"QC: {int(keep.sum()):,} of {len(rows):,} pass "
        f"({100 * (1 - keep.mean()):.1f}% removed) {info.get('n_fail_per_rule', '')}")
    os.makedirs(OUT_QC_DIR, exist_ok=True)
    pd.concat([met, fails], axis=1).assign(qc_pass=keep).to_csv(
        os.path.join(OUT_QC_DIR, f"{name}_qc_metrics.csv.gz"), index_label="barcode")

    # ---- genes -> symbols -------------------------------------------------------------
    st.set(phase="symbols")
    M = M[keep]
    cells = barcodes[rows][keep]
    obs = met[keep].copy()                       # positional, so duplicate barcodes stay 1:1
    info["n_duplicate_barcodes"] = int(len(cells) - len(set(cells)))

    if not repro:
        M, genes, n_merged = collapse_symbols(M, sym)
        info.update(n_genes_symbols=int(len(genes)), n_ids_merged_into_shared_symbols=int(n_merged))
        log(f"genes: {info['n_genes_file']:,} ids -> {len(genes):,} symbols ({n_merged} merged)")
    else:
        # old cell order and old gene order, then compare entry by entry with the old X.
        # The old file's 13,992-gene axis is a cross-dataset SYMBOL-keyed union: it holds
        # OLDER symbols from other datasets (AARS, HIST1H1B) next to Marson's own (AARS1,
        # H1-5), and genes Marson never measured (A1BG). Marson's counts sit only under its
        # own symbols, so the old file has those older-symbol columns ALL ZERO for Marson
        # (verified 2026-10-15: all 108 such columns; zeroing them gives 0 differing
        # entries). So: same raw symbol -> that column, everything else -> all-zero.
        # The Ensembl route (old symbol -> gene_id_map -> raw id) is only RECORDED, as the
        # list of genes whose Marson counts the old file never carried.
        order = pd.Series(np.arange(len(cells)), index=cells)[want].to_numpy()
        M, cells, obs = M[order], cells[order], obs.iloc[order]
        idmap = pd.read_csv(GENE_ID_MAP, sep="\t", dtype=str).drop_duplicates("gene_name")
        sym2ens = dict(zip(idmap["gene_name"], idmap["ensembl_id"]))
        by_sym = {s: i for i, s in reversed(list(enumerate(sym.astype(str))))}
        by_ens = {e: i for i, e in enumerate(gene_ids.astype(str))}
        cols, how, renamed = [], {"symbol": 0, "zero_old_symbol_renamed": 0, "zero_absent": 0}, []
        old_set = set(old_genes.astype(str))
        for gname in old_genes:
            if gname in by_sym:
                cols.append(by_sym[gname]); how["symbol"] += 1
                continue
            cols.append(-1)
            ens = sym2ens.get(gname)
            if ens in by_ens:                    # Marson measured it, under a newer symbol
                how["zero_old_symbol_renamed"] += 1
                new = str(sym[by_ens[ens]])
                renamed.append({"old_symbol": str(gname), "marson_symbol": new,
                                "marson_symbol_in_old_axis": new in old_set})
            else:
                how["zero_absent"] += 1
        cols = np.array(cols)
        ok = cols >= 0
        S = csr_matrix((np.ones(int(ok.sum()), dtype=np.float32), (cols[ok], np.flatnonzero(ok))),
                       shape=(M.shape[1], len(old_genes)))
        M, genes = (M @ S).tocsr(), old_genes.astype(object)
        lost = [d for d in renamed if not d["marson_symbol_in_old_axis"]]
        info["repro_gene_mapping"] = dict(how, renamed=renamed,
                                          n_renamed_marson_symbol_missing_from_old_axis=len(lost))
        log(f"repro genes: {how}; renamed genes whose Marson symbol is NOT in the old axis "
            f"(Marson counts absent from the old file): {len(lost)}")
        st.set(phase="repro check", pct=0)
        with h5py.File(r["repro_from"], "r") as g:
            M_old = read_csr_rows(g["X"], old_rows, st)        # old_rows are sorted
        diff = (M_old.astype(np.float64) - M.astype(np.float64)).tocsr()
        diff.eliminate_zeros()
        bad_genes = np.unique(diff.indices)
        info["repro_check"] = {
            "identical": bool(diff.nnz == 0), "n_entries_differ": int(diff.nnz),
            "n_cells_differ": int((np.diff(diff.indptr) > 0).sum()),
            "n_genes_differ": int(len(bad_genes)),
            "genes_differ_examples": [str(genes[i]) for i in bad_genes[:30]]}
        log(f"REPRO CHECK vs old X: {info['repro_check']}")

    # ---- write --------------------------------------------------------------------------
    st.set(phase="write h5ad")
    import anndata as ad
    obs.insert(0, "is_ctrl", "ctrl")
    A = ad.AnnData(X=M.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes.astype(str))))
    os.makedirs(OUT_H5AD_DIR, exist_ok=True)
    out = os.path.join(OUT_H5AD_DIR, f"{name}_ctrl_qc.h5ad")
    A.write_h5ad(out + ".tmp")
    os.replace(out + ".tmp", out)
    info.update(out_h5ad=out, out_shape=list(A.shape), minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(OUT_QC_DIR, f"{name}_thresholds.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    log(f"wrote {out} {A.shape} in {info['minutes']} min")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--row", type=int, help="0-based data row of datasets.tsv (= SLURM_ARRAY_TASK_ID)")
    g.add_argument("--name", help="dataset name in datasets.tsv")
    ap.add_argument("--force", action="store_true", help="redo even if already done")
    a = ap.parse_args()
    r = load_row(a)
    name = r["name"]
    done = os.path.join(OUT_QC_DIR, f"{name}_thresholds.json")
    if os.path.exists(done) and os.path.exists(os.path.join(OUT_H5AD_DIR, f"{name}_ctrl_qc.h5ad")) \
            and not a.force:
        log(f"{name}: already done ({done}); --force to redo")
        return
    st = Status(name)
    try:
        info = run(r, st)
        msg = ("REPRO identical" if info.get("repro_check", {}).get("identical")
               else "REPRO DIFFERS" if "repro_check" in info else "")
        st.set(state="done", phase="done", pct=100, msg=msg)
    except BaseException as e:
        st.set(state="failed", msg=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
