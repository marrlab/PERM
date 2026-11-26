#!/usr/bin/env python
"""Step 3 of power-seek_run: per-perturbation pseudobulk of one dataset row.

Same row of datasets.tsv as steps 1-2 (--row / --name), so the label column, control
labels, symbol column and group (Wong condition, Mali batch) are the ones Power-Seek used.

Per row:
  1. cells = every cell with a label in obs[ctrl_col] (empty / NaN = no guide called ->
     dropped), inside the row's group if it has one, passing the optional `cell_filter`
  2. read X in blocks with step 1's readers (CSR / CSC / dense)
  3. per-cell QC. CONTROL cells always use EXACTLY step 1's cutoffs
     (results/qc/<name>_thresholds.json); check: controls passing == step 1's n_pass, so
     the LFC control is the very cell set Power-Seek ran on. KNOCKOUT cells, --ko-qc:
       all      (default) the same MAD rule, thresholds computed on ALL cells in scope
                (knockouts + controls) -- as the ExPert pipeline built the main-paper
                Marson file. Needs one extra read pass for the metrics.
       controls step 1's control cutoffs. Tested on HepG2 2026-10-15: removes the
                STRONGEST knockouts (POLRMT, TEFM, RPL*: the knockout itself moves %mito /
                top-50 share out of the control range); 162 of 2,160 KOs fell below 20 cells.
       minimal  only clearly broken cells: total counts or genes below step 1's lower
                cutoff, or %mito above the fixed cap (20 cancer / 12 other).
       none     no QC (as the main-paper Xu / Frangieh / Bowness pseudobulks).
     --ctrl-qc same: controls get the SAME rule as the knockouts (symmetric, as every
     main-paper pseudobulk); the step-1 check is then reported, not enforced.
     Default step1 (controls = Power-Seek's exact cells).
  4. sum raw counts per label; all control labels -> one group 'control'
  5. genes -> symbols exactly as step 1 (ids sharing a symbol summed); the gene order is
     checked against the step-1 h5ad = the Power-Seek input
  6. write marson_KO.save_pseudobulk format (MEAN raw counts per cell + group sizes), so
     marson_KO.lfc_from_pseudobulk runs on it unchanged

cell_filter (datasets.tsv, '-' = none): comma-separated `col=value` / `col!=value`,
all must hold, compared as strings, e.g. `low_quality!=True`.

Outputs
  $PERM_SCRATCH/power-seek_run/pseudobulk/<name>_pseudobulk.npz   (bulk -> scratch)
  results/pseudobulk/<name>_label_counts.csv   per label: cells, cells passing QC
  results/pseudobulk/<name>_pseudobulk.json    bookkeeping + the control check
  results/pseudobulk/status/<name>.status      progress line (bash progress.sh 3)
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
from scipy.sparse import csr_matrix

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN_DIR)
sys.path.insert(0, P.LIB)
import qc_filter_controls as q                      # readers, metrics, symbols, Status
from marson_KO import save_pseudobulk

# results go to the data tree, where figures.py reads them (PS_RUN/results)
RESULTS = os.path.join(P.MEMGENES_ROOT, "power-seek_run", "results")
OUT_NPZ_DIR = f"{P.SCRATCH}/power-seek_run/pseudobulk"
OUT_DIR = os.path.join(RESULTS, "pseudobulk")
CONTROL = "control"
NNZ_PER_BLOCK = {"csr": 300_000_000, "dense": 300_000_000, "csc": 1_000_000_000}
# CSC (Mali): every block rescans the whole gene-major file, so blocks are big


def parse_filter(s):
    conds = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "!=" in part:
            c, v = part.split("!=", 1); conds.append((c.strip(), v.strip(), False))
        elif "=" in part:
            c, v = part.split("=", 1); conds.append((c.strip(), v.strip(), True))
        else:
            raise SystemExit(f"cell_filter term '{part}' is not col=value / col!=value")
    return conds


def qc_keep(met, thr):
    """Step 1's rule with step 1's numbers: keep inside [lo, hi] on all four, mito <= cap."""
    v = {"log1p_total_counts": np.log1p(met["total_counts"].to_numpy()),
         "log1p_n_genes_by_counts": np.log1p(met["n_genes_by_counts"].to_numpy().astype(np.float64)),
         f"pct_counts_in_top_{q.QC_PERCENT_TOP}_genes": met[f"pct_counts_in_top_{q.QC_PERCENT_TOP}_genes"].to_numpy(),
         "pct_counts_mt": met["pct_counts_mt"].to_numpy()}
    keep = np.ones(len(met), dtype=bool)
    for k, x in v.items():                 # _outlier is strict (<, >), so the bounds pass
        keep &= (x >= thr[k]["lo"]) & (x <= thr[k]["hi"])
    keep &= v["pct_counts_mt"] <= thr["mt_per"]
    return keep


def qc_keep_minimal(met, thr):
    """Only clearly broken cells fail: too few counts / genes (step 1's lower cutoff), or
    %mito over the fixed cap. No upper bounds, no two-sided %mito / top-50 MAD."""
    lt = np.log1p(met["total_counts"].to_numpy())
    lg = np.log1p(met["n_genes_by_counts"].to_numpy().astype(np.float64))
    return ((lt >= thr["log1p_total_counts"]["lo"]) & (lg >= thr["log1p_n_genes_by_counts"]["lo"])
            & (met["pct_counts_mt"].to_numpy() <= thr["mt_per"]))


def collapse_dense(sums, names):
    """q.collapse_symbols for a dense (groups x genes) array: same first-occurrence order,
    ids sharing a symbol summed. (A sparse copy of an 18k x 38k dense matrix costs ~3x.)"""
    names = np.asarray(names, dtype=object).astype(str)
    uniq, first, inv = np.unique(names, return_index=True, return_inverse=True)
    if len(uniq) == len(names):
        return sums, names.astype(object), 0
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    onehot = csr_matrix((np.ones(len(names)), (np.arange(len(names)), rank[inv])),
                        shape=(len(names), len(uniq)))
    return (onehot.T @ sums.T).T, uniq[order].astype(object), len(names) - len(uniq)


def blocks(path, rows, layout):
    """Split sorted rows into blocks of ~NNZ_PER_BLOCK stored entries."""
    budget = NNZ_PER_BLOCK[layout]
    with h5py.File(path, "r") as f:
        X = f["X"]
        if layout == "csr":
            ip = X["indptr"][:].astype(np.int64)
            per_row = ip[rows + 1] - ip[rows]
        else:
            n_obs = len(f["obs"][q.index_key(f["obs"])])
            nnz = X["data"].shape[0] if layout == "csc" else n_obs * X.shape[1] * 0.25
            per_row = np.full(len(rows), max(1, int(nnz / n_obs)))
    cut = np.searchsorted(np.cumsum(per_row), np.arange(budget, per_row.sum() + budget, budget))
    edges = np.unique(np.concatenate([[0], np.minimum(cut + 1, len(rows)), [len(rows)]]))
    return [rows[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]


def variant_tag(ko_qc, ctrl_qc):
    return f"koqc-{ko_qc}" + ("" if ctrl_qc == "step1" else f"_ctrlqc-{ctrl_qc}")


def run(r, st, ko_qc, ctrl_qc="step1"):
    name, path = r["name"], os.path.expandvars(r["path"])
    vtag = variant_tag(ko_qc, ctrl_qc)
    tag = f"{name}_pseudobulk_{vtag}"
    t0 = time.time()
    thr_json = json.load(open(os.path.join(RESULTS, "qc", f"{name}_thresholds.json")))
    thr = thr_json["thresholds"]
    ctrl_values = [v for v in r["ctrl_values"].split(",") if v]
    info = {"name": name, "path": path, "label_col": r["ctrl_col"], "control_labels": ctrl_values,
            "ko_qc": ko_qc, "ctrl_qc": ctrl_qc,
            "control_qc_thresholds_from": (f"results/qc/{name}_thresholds.json" if ctrl_qc == "step1"
                                           else "same rule as the knockout cells")}

    # ---- which cells ----------------------------------------------------------------
    st.set(phase="select cells")
    with h5py.File(path, "r") as f:
        labels = q.read_col(f, "obs", r["ctrl_col"]).astype(str)
        gene_ids = q.read_col(f, "var", q.index_key(f["var"])).astype(str)
        sym = (q.read_col(f, "var", r["symbol_col"]).astype(str)
               if r["symbol_col"] not in ("", "-") else gene_ids)
        sym = np.where((sym == "") | (sym == "nan"), gene_ids, sym)
        use = (labels != "") & (labels != "nan")
        info["n_cells_file"], info["n_cells_no_label"] = int(len(labels)), int((~use).sum())
        if r.get("group_col", "-") not in ("", "-"):
            g = q.read_col(f, "obs", r["group_col"]).astype(str)
            use &= g == r["group"]
            info.update(group_col=r["group_col"], group=r["group"])
        for c, v, eq in parse_filter(r.get("cell_filter", "-") if r.get("cell_filter", "-") != "-" else ""):
            x = q.read_col(f, "obs", c).astype(str)
            use &= (x == v) if eq else (x != v)
            info.setdefault("cell_filter", []).append(f"{c}{'=' if eq else '!='}{v}")
    is_ctrl = np.isin(labels, ctrl_values)
    grp_label = np.where(is_ctrl, CONTROL, labels)
    rows = np.flatnonzero(use)
    cats, codes_all = np.unique(grp_label[rows], return_inverse=True)
    n_groups = len(cats)
    log_ = q.log
    log_(f"{name}: {len(rows):,} labelled cells in scope, {n_groups:,} labels "
         f"({int(is_ctrl[rows].sum()):,} control)")
    st.set(n_ctrl=len(rows))

    # ---- stream, QC, accumulate --------------------------------------------------------
    with h5py.File(path, "r") as f:
        layout = q.x_layout(f["X"])
    mito = np.char.startswith(np.char.upper(sym.astype(str)), "MT-")
    if not mito.any():
        raise SystemExit("no MT- genes: the mito cutoffs cannot be applied")
    sums = np.zeros((n_groups, len(gene_ids)), dtype=np.float64)
    n_in = np.bincount(codes_all, minlength=n_groups)
    n_pass = np.zeros(n_groups, dtype=np.int64)
    blks = blocks(path, rows, layout)
    log_(f"X layout {layout}: {len(blks)} blocks")

    n_pass_ = 2 if ko_qc == "all" else 1

    class Sub:                                   # per-block reader progress -> overall %
        def __init__(self, k, p, label): self.k, self.p, self.label = k, p, label
        def set(self, force=False, **kw):
            if "pct" in kw and kw.get("phase") == "read X":
                done = self.p * len(blks) + self.k + float(kw["pct"]) / 100
                st.set(force=False, phase=f"{self.label} {self.k + 1}/{len(blks)}",
                       pct=f"{100 * done / (n_pass_ * len(blks)):.0f}")

    ctrl_rows = is_ctrl[rows]
    met_all = None
    if ko_qc == "all":
        # pass 1: QC metrics of every cell in scope -> thresholds over ALL of them
        parts = []
        for k, b in enumerate(blks):
            M, _ = q.read_rows(path, b, Sub(k, 0, "metrics"))
            parts.append(q.qc_metrics(M, mito))
            del M
        met_all = pd.concat(parts, ignore_index=True)
        _, _, thr_ko = q.qc_mask(met_all, r["cancer"] == "y")
        thr_ko["thresholds_computed_on"] = "ALL cells in scope (knockouts + controls)"
        info["ko_qc_thresholds"] = thr_ko
        log_(f"knockout-cell thresholds from all {len(met_all):,} cells: " + ", ".join(
            f"{k} [{v['lo']:.2f}, {v['hi']:.2f}]" for k, v in thr_ko.items() if isinstance(v, dict)))

    for k, b in enumerate(blks):
        M, _ = q.read_rows(path, b, Sub(k, n_pass_ - 1, "block"))
        vals = M.data
        if vals.size and not np.array_equal(vals, np.round(vals)):
            raise SystemExit("X is not integer-valued -- these must be RAW counts")
        i0 = np.searchsorted(rows, b)                     # positions of this block in `rows`
        met = met_all.iloc[i0].reset_index(drop=True) if met_all is not None else q.qc_metrics(M, mito)
        keep_step1 = qc_keep(met, thr)                    # step 1's control cutoffs
        keep_ko = (keep_step1 if ko_qc == "controls" else
                   qc_keep(met, thr_ko) if ko_qc == "all" else
                   qc_keep_minimal(met, thr) if ko_qc == "minimal" else
                   np.ones(len(met), dtype=bool))         # none
        keep_ctrl = keep_step1 if ctrl_qc == "step1" else keep_ko
        keep = np.where(ctrl_rows[i0], keep_ctrl, keep_ko)
        c = codes_all[i0]
        n_pass += np.bincount(c[keep], minlength=n_groups)
        G = csr_matrix((np.ones(int(keep.sum())), (c[keep], np.arange(int(keep.sum())))),
                       shape=(n_groups, int(keep.sum())))
        sums += (G @ M[keep]).toarray()
        log_(f"block {k + 1}/{len(blks)}: {len(b):,} cells, {int(keep.sum()):,} pass QC")
        del M

    # ---- checks ---------------------------------------------------------------------------
    ci = int(np.flatnonzero(cats == CONTROL)[0])
    info.update(n_cells_in_scope=int(len(rows)), n_cells_pass_qc=int(n_pass.sum()),
                pct_removed=round(100 * (1 - n_pass.sum() / len(rows)), 2),
                n_ctrl_pass=int(n_pass[ci]), n_ctrl_pass_step1=int(thr_json["n_pass"]),
                controls_match_step1=bool(n_pass[ci] == thr_json["n_pass"]))
    log_(f"QC: {n_pass.sum():,} of {len(rows):,} cells pass ({info['pct_removed']}% removed); "
         f"controls {n_pass[ci]:,} vs step 1 {thr_json['n_pass']:,}")
    if ctrl_qc == "step1" and not info["controls_match_step1"]:
        raise SystemExit(f"control cells passing QC ({n_pass[ci]}) != step 1 ({thr_json['n_pass']}): "
                         "the LFC control would not be Power-Seek's cell set")

    # ---- genes -> symbols, identical to the Power-Seek input -------------------------------
    st.set(phase="symbols", pct="")
    sums, genes, n_merged = collapse_dense(sums, sym)
    with h5py.File(os.path.join(q.OUT_H5AD_DIR, f"{name}_ctrl_qc.h5ad"), "r") as f:
        ps_genes = q.read_col(f, "var", q.index_key(f["var"])).astype(str)
    if not np.array_equal(ps_genes, genes.astype(str)):
        raise SystemExit("gene axis differs from the step-1 h5ad (the Power-Seek input)")
    info.update(n_genes=int(len(genes)), n_ids_merged_into_shared_symbols=int(n_merged))

    # ---- write ----------------------------------------------------------------------------
    st.set(phase="write")
    ok = n_pass > 0
    means = sums[ok] / n_pass[ok][:, None]
    pb = pd.DataFrame(means.astype(np.float32), index=pd.Index(cats[ok], name="group"),
                      columns=pd.Index(genes.astype(str)))
    sizes = pd.Series(n_pass[ok], index=pb.index, name="n_cells")
    os.makedirs(OUT_NPZ_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_NPZ_DIR, f"{tag}.npz")
    save_pseudobulk(pb, sizes, out)
    lc = pd.DataFrame({"label": cats, "n_cells": n_in, "n_cells_pass_qc": n_pass,
                       "is_control": cats == CONTROL})
    lc.to_csv(os.path.join(OUT_DIR, f"{name}_label_counts_{vtag}.csv"), index=False)
    ko = lc[~lc.is_control & (lc.n_cells >= 20)]
    info["ko_loss"] = {"n_kos_ge20_cells": int(len(ko)),
                       "median_pct_removed": round(float(100 * (1 - ko.n_cells_pass_qc / ko.n_cells).median()), 1),
                       "n_kos_losing_over_half": int(((ko.n_cells_pass_qc / ko.n_cells) < 0.5).sum()),
                       "n_kos_falling_below_20": int((ko.n_cells_pass_qc < 20).sum())}
    log_(f"knockouts (>=20 cells before QC): {info['ko_loss']}")
    info.update(out_npz=out, n_groups=int(ok.sum()), minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(OUT_DIR, f"{tag}.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    log_(f"wrote {out}: {pb.shape[0]:,} groups x {pb.shape[1]:,} genes in {info['minutes']} min")
    st.set(n_pass=int(n_pass.sum()))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--row", type=int, help="0-based data row of datasets.tsv (= SLURM_ARRAY_TASK_ID)")
    g.add_argument("--name", help="dataset name in datasets.tsv")
    ap.add_argument("--ko-qc", choices=["all", "controls", "minimal", "none"], default="all",
                    help="QC for KNOCKOUT cells; see top")
    ap.add_argument("--ctrl-qc", choices=["step1", "same"], default="step1",
                    help="QC for CONTROL cells: step 1's cutoffs, or the same rule as the knockouts")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = q.load_row(a)
    name = r["name"]
    if r.get("run_powerseek", "y") == "n" or r.get("repro_from", "-") not in ("", "-"):
        q.log(f"{name}: not part of the analysis (run_powerseek=n or repro row) -- skipped")
        return
    vtag = variant_tag(a.ko_qc, a.ctrl_qc)
    if os.path.exists(os.path.join(OUT_DIR, f"{name}_pseudobulk_{vtag}.json")) and not a.force:
        q.log(f"{name}: already done ({vtag}); --force to redo")
        return
    q.STATUS_DIR = os.path.join(OUT_DIR, "status")
    st = q.Status(f"{name}__{vtag}")              # one file per variant; progress.sh shows newest
    try:
        info = run(r, st, a.ko_qc, a.ctrl_qc)
        kl = info["ko_loss"]
        st.set(state="done", phase="done", pct=100,
               msg=f"{vtag}: {info['n_groups']} groups, {kl['n_kos_falling_below_20']} KOs <20 cells")
    except BaseException as e:
        st.set(state="failed", msg=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
