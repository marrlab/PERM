"""
marson_per_group.py - Marson LFC + responsiveness summaries split by TIMEPOINT
(Rest / Stim8hr / Stim48hr) and by DONOR (D1..D4), from the donor/condition-
resolved pseudobulk.

Why this exists: `marson_lfc_cpm.csv` (the file every KO_analysis plot uses) is
built from `marson_pseudobulk.npz`, which groups cells by PERTURBATION ONLY --
all 4 donors x 3 timepoints pooled into one group per KO. So the published
responsiveness numbers are a cell-count-weighted POOL, not a mean of per-timepoint
(or per-donor) LFCs, and neither axis can be recovered from that CSV. Both can be
recovered from `marson_pseudobulk_donor.npz` (perturbation x donor x condition),
which is what this script does -- in one pass, since the expensive step (the
within-group LFC) is shared.

Recipe (identical to marson_per_donor.py, only the collapse differs):
  raw pseudobulk -> CPM per group -> LFC vs the control of the SAME
  (donor, condition) -> cell-count-weighted mean over the OTHER axis
  -> per (KO, timepoint) and per (KO, donor).
Self-KO entries are masked to NaN within each group.

Complements marson_per_donor.py rather than replacing it: that script answers
"was pooling donors safe?" and writes the donor-corrected LFC matrix; this one
writes the small per-group gene summaries the responsiveness plots need.

Gene summaries use the KOs present in ALL groups of an axis, so a group-vs-group
responsiveness comparison is not confounded by a different KO set per group.
`--no-common-kos` disables that.

Baseline expression is written PER GROUP (cell-weighted control mean, same units
as marson_gene_ctrl_expr.csv -- verified identical when pooled over all groups),
because Rest and Stim48hr have very different expression profiles and the
expression-matched comparison must match WITHIN group.

Data only -- no figures. Plotting stays in the analysis notebook.

Outputs (to --out-dir, default $PERM_DATA/marson):
  marson_gene_summary_cpm_{Rest,Stim8hr,Stim48hr,D1..D4}.csv   per-gene responsiveness
  marson_gene_ctrl_expr_{...}.csv                              per-gene control expression
  marson_{timepoint,donor}_group_summary.csv                   QC table per axis
  marson_{timepoint,donor}_lfc_correlation.csv                 group-group LFC correlation
  marson_lfc_cpm_{...}.npz                                     only with --save-lfc; float32,
                                                               ~355 MB/timepoint, ~175 MB/donor

Run:  python marson_per_group.py                 # both axes
      python marson_per_group.py --axes timepoint
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

# shared perturb-seq libs live outside this dir
MARSON_DIR = P.LIB
sys.path.insert(0, MARSON_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marson_KO import summarize_gene_response          # noqa: E402
from marson_per_donor import load, lfc_within, collapse  # noqa: E402

NPZ = f'{P.DATA}/marson/marson_pseudobulk_donor.npz'
OUT_DIR = f'{P.DATA}/marson'
CONTROL = 'control'

# axis name -> (pseudobulk index level, preferred group order)
AXES = {
    'timepoint': ('condition', ['Rest', 'Stim8hr', 'Stim48hr']),
    'donor':     ('donor',     ['D1', 'D2', 'D3', 'D4']),
}


def ctrl_expr_per_group(pb, counts, level):
    """Cell-weighted mean control expression per (group, gene).

    Same units as marson_gene_ctrl_expr.csv (mean raw counts/cell), so the >0.1
    detection floor used everywhere else transfers unchanged.
    """
    ctrl = pb.xs(CONTROL, level='perturbation')
    w = counts.xs(CONTROL, level='perturbation').reindex(ctrl.index).values.astype(np.float64)
    num = pd.DataFrame(ctrl.values * w[:, None], index=ctrl.index, columns=ctrl.columns) \
        .groupby(level=level, observed=True).sum()
    den = pd.Series(w, index=ctrl.index).groupby(level=level, observed=True).sum()
    return num.div(den, axis=0)                        # group x gene


def save_lfc_npz(mat, path):
    """Per-group LFC as float32 npz, NOT csv.

    The pooled `marson_lfc_cpm.csv` is 1.6 GB and slow to parse; there are 7 of
    these. float32 npz is ~4x smaller and loads in seconds, which matters because
    the analysis notebook runs on the login node under an 8 GB cgroup cap.
    """
    np.savez(path,
             values=mat.values.astype(np.float32),
             index=mat.index.to_numpy().astype(str),
             columns=mat.columns.to_numpy().astype(str))
    print(f"    wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)


def order_groups(present, preferred):
    """Preferred order first, then anything unexpected, so a renamed group still shows up."""
    return [g for g in preferred if g in present] + sorted(present - set(preferred))


def run_axis(axis, lfc_g, cnt_g, expr, args):
    level, preferred = AXES[axis]
    by_group = collapse(lfc_g, cnt_g, ['perturbation', level])
    n_sub = cnt_g.groupby(level=['perturbation', level], observed=True).size()
    n_cells = cnt_g.groupby(level=['perturbation', level], observed=True).sum()

    groups = order_groups(set(by_group.index.get_level_values(level)), preferred)
    per_group = {}
    for g in groups:
        sub = by_group.xs(g, level=level)
        shared = sub.index.intersection(sub.columns)
        if len(shared):                                     # mask self-KO within group
            sub.values[sub.index.get_indexer(shared), sub.columns.get_indexer(shared)] = np.nan
        per_group[g] = sub
    del by_group; gc.collect()
    print(f"[{axis}] per-group LFC:", {g: per_group[g].shape for g in groups}, flush=True)

    common = sorted(set.intersection(*[set(per_group[g].index) for g in groups]))
    print(f"[{axis}] KOs present in all {len(groups)} groups: {len(common):,} "
          f"(per group: {[per_group[g].shape[0] for g in groups]})", flush=True)

    rows = []
    for g in groups:
        mat = per_group[g] if args.no_common_kos else per_group[g].loc[common]
        gs = summarize_gene_response(mat, large_change_threshold=args.large_change_threshold)
        gs.to_csv(f'{args.out_dir}/marson_gene_summary_cpm_{g}.csv')
        e = expr.loc[g].rename('mean_ctrl_expr')
        e.index.name = 'gene'
        e.to_csv(f'{args.out_dir}/marson_gene_ctrl_expr_{g}.csv')
        if args.save_lfc:
            save_lfc_npz(mat, f'{args.out_dir}/marson_lfc_cpm_{g}.npz')
        rows.append(dict(
            group=g,
            n_KOs=per_group[g].shape[0],
            n_KOs_used=mat.shape[0],
            n_cells=int(n_cells.xs(g, level=level).sum()),
            mean_subgroups_per_KO=float(n_sub.xs(g, level=level).mean()),
            mean_abs_lfc=float(np.nanmean(np.abs(mat.values))),
            median_abs_lfc=float(np.nanmedian(np.abs(mat.values))),
            n_genes_expr_gt_0p1=int((e > 0.1).sum()),
            mean_ctrl_expr=float(e.mean()),
        ))
        print(f"  [{axis}] {rows[-1]}", flush=True)
        del mat, gs; gc.collect()

    summary = pd.DataFrame(rows).set_index('group')
    summary.to_csv(f'{args.out_dir}/marson_{axis}_group_summary.csv')
    print(f"\n[{axis}] summary:\n{summary.round(4).to_string()}", flush=True)

    # ---- reproducibility across groups (same KO set, same genes) ----
    R = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for a in groups:
        for b in groups:
            A = per_group[a].loc[common].values.ravel()
            B = per_group[b].loc[common].values.ravel()
            m = np.isfinite(A) & np.isfinite(B)
            R.loc[a, b] = np.corrcoef(A[m], B[m])[0, 1]
    R.to_csv(f'{args.out_dir}/marson_{axis}_lfc_correlation.csv')
    print(f"\n[{axis}] group-group LFC correlation:\n{R.round(3).to_string()}", flush=True)

    del per_group; gc.collect()
    return summary, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--axes', nargs='+', default=list(AXES), choices=list(AXES))
    ap.add_argument('--min-cells', type=int, default=20)
    ap.add_argument('--large-change-threshold', type=float, default=1.0)
    ap.add_argument('--save-lfc', action='store_true',
                    help='Also write the full per-group LFC matrices as float32 npz '
                         '(~355 MB per timepoint, ~175 MB per donor). Needed for the '
                         'per-KO analyses in per_donor_or_tp.ipynb.')
    ap.add_argument('--no-common-kos', action='store_true',
                    help='Summarise each group on its own KO set instead of the intersection.')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pb, counts = load(args.npz)
    print(f"pseudobulk {pb.shape} | {int(counts.sum()):,} cells", flush=True)

    expr = {ax: ctrl_expr_per_group(pb, counts, AXES[ax][0]) for ax in args.axes}

    lfc_g, cnt_g = lfc_within(pb, counts, args.min_cells)
    print(f"within (donor,condition) LFC: {lfc_g.shape[0]:,} rows", flush=True)
    del pb; gc.collect()

    for ax in args.axes:
        run_axis(ax, lfc_g, cnt_g, expr[ax], args)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
