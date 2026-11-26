"""
marson_per_cell.py - Marson responsiveness tables for all 12 donor x timepoint cells.

`marson_per_group.py` collapses one axis at a time (per timepoint = 4 donors pooled;
per donor = 3 timepoints pooled). This does neither: each of the 12 (donor, timepoint)
combinations is kept separate, which is exactly the UN-COLLAPSED `lfc_within` output --
every KO already sits against the control cells of its own donor AND its own
stimulation state. No new modelling, just no averaging.

Point of it: show whether the memory-gene responsiveness effect is present in every
individual cell. If it is, pooling for downstream analyses is justified on power
grounds rather than assumed. If some cells invert, pooling is hiding heterogeneity.

Power caveat baked into the design: cells hold 213k-638k KO cells (41-133 cells per
KO, a ~3x spread, D4_Rest weakest and D2_Rest strongest), against 5.07M pooled. Effect
sizes here are attenuated by noise relative to the pooled analysis, and unequally so --
compare cells with that in mind, and read the per-cell n printed into the QC table.

Gene summaries use the 1,377 KOs present in ALL 12 cells by default, so a cell-to-cell
comparison is not confounded by a different KO set (--no-common-kos to disable, which
gives each cell its own 4,159-5,626 KOs and more power per cell).

Outputs (to --out-dir, default $PERM_DATA/marson):
  marson_gene_summary_cpm_{D1_Rest..D4_Stim48hr}.csv   per-gene responsiveness
  marson_gene_ctrl_expr_{...}.csv                      per-cell baseline expression
  marson_cell_summary.csv                              QC per cell

Run:  python marson_per_cell.py
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

MARSON_DIR = P.LIB
sys.path.insert(0, MARSON_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marson_KO import summarize_gene_response            # noqa: E402
from marson_per_donor import load, lfc_within            # noqa: E402

NPZ = f'{P.DATA}/marson/marson_pseudobulk_donor.npz'
OUT_DIR = f'{P.DATA}/marson'
CONTROL = 'control'
DONORS = ['D1', 'D2', 'D3', 'D4']
TIMEPOINTS = ['Rest', 'Stim8hr', 'Stim48hr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--min-cells', type=int, default=20)
    ap.add_argument('--large-change-threshold', type=float, default=1.0)
    ap.add_argument('--no-common-kos', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pb, counts = load(args.npz)
    print(f"pseudobulk {pb.shape} | {int(counts.sum()):,} cells", flush=True)

    # per-cell baseline expression: the control row of that donor x timepoint, which
    # is already a single group -- no weighting needed, and the same units as
    # marson_gene_ctrl_expr.csv (mean raw counts per cell)
    ctrl = pb.xs(CONTROL, level='perturbation')

    lfc_g, cnt_g = lfc_within(pb, counts, args.min_cells)
    print(f"within (donor,condition) LFC: {lfc_g.shape[0]:,} rows", flush=True)
    del pb; gc.collect()

    cells = [(d, t) for d in DONORS for t in TIMEPOINTS]
    present = {}
    for d, t in cells:
        idx = ((lfc_g.index.get_level_values('donor') == d) &
               (lfc_g.index.get_level_values('condition') == t))
        present[(d, t)] = set(lfc_g.index.get_level_values('perturbation')[idx])
    common = sorted(set.intersection(*present.values()))
    print(f"KOs per cell: {min(len(v) for v in present.values())}-"
          f"{max(len(v) for v in present.values())} | in all 12: {len(common):,}", flush=True)

    rows = []
    for d, t in cells:
        name = f'{d}_{t}'
        sub = lfc_g.xs((d, t), level=('donor', 'condition'))
        if not args.no_common_kos:
            sub = sub.loc[common]
        shared = sub.index.intersection(sub.columns)
        if len(shared):                                   # mask self-KO within the cell
            sub.values[sub.index.get_indexer(shared), sub.columns.get_indexer(shared)] = np.nan

        gs = summarize_gene_response(sub, large_change_threshold=args.large_change_threshold)
        gs.to_csv(f'{args.out_dir}/marson_gene_summary_cpm_{name}.csv')

        e = ctrl.loc[(d, t)].rename('mean_ctrl_expr')
        e.index.name = 'gene'
        e.to_csv(f'{args.out_dir}/marson_gene_ctrl_expr_{name}.csv')

        n_cells = int(cnt_g.xs((d, t), level=('donor', 'condition')).reindex(sub.index).sum())
        rows.append(dict(cell=name, donor=d, timepoint=t,
                         n_KOs_available=len(present[(d, t)]), n_KOs_used=sub.shape[0],
                         n_cells=n_cells, cells_per_KO=round(n_cells / max(sub.shape[0], 1), 1),
                         mean_abs_lfc=float(np.nanmean(np.abs(sub.values))),
                         median_abs_lfc=float(np.nanmedian(np.abs(sub.values))),
                         n_genes_expr_gt_0p1=int((e > 0.1).sum())))
        print(f"  {name}: {rows[-1]}", flush=True)
        del sub, gs; gc.collect()

    summary = pd.DataFrame(rows).set_index('cell')
    summary.to_csv(f'{args.out_dir}/marson_cell_summary.csv')
    print(f"\n{summary.round(4).to_string()}", flush=True)
    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
