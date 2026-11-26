"""
marson_per_donor.py - per-donor Marson LFC from the donor/condition-resolved
pseudobulk, plus the figures that show whether pooling donors was safe.

Reads `$PERM_DATA/marson/marson_pseudobulk_donor.npz` (built by
build_marson_donor_pseudobulk.py: perturbation x donor x condition, mean RAW counts).

LFC is computed WITHIN (donor, condition) -- each perturbation against the control
cells of the SAME donor AND the SAME stimulation state -- then aggregated. That is
the Weinstock recipe (compute_donor_corrected_lfc) generalised to two nuisance axes.

Unlike the HSC screen, Marson is NOT confounded: every donor received essentially
the whole library (~4.2-5.6k perturbations with >=20 cells per donor x condition),
so pooling is not structurally fatal here -- but it is now measurable rather than
assumed, which is the point.

Outputs (to --out-dir, default $PERM_DATA/):
  marson_lfc_cpm_donor_{D1..D4}.csv   per-donor LFC (KO x gene)
  marson_lfc_cpm_donorcorrected.csv   mean over donors of the within-donor LFC
  marson_donor_summary.csv            per-donor QC / reproducibility table
  plots/marson_per_donor/*.png

Run:  python marson_per_donor.py
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NPZ = f'{P.DATA}/marson/marson_pseudobulk_donor.npz'
OUT_DIR = f'{P.DATA}/marson'
PLOT_DIR = f'{P.PLOTS}/marson_per_donor'
CONTROL = 'control'


def load(npz):
    z = np.load(npz, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays(
        [z['perts'], z['donors'], z['conditions']],
        names=['perturbation', 'donor', 'condition'])
    pb = pd.DataFrame(z['values'], index=idx, columns=z['genes'])
    return pb, pd.Series(z['counts'], index=idx, name='n_cells')


def lfc_within(pb, counts, min_cells, pseudocount=1.0):
    """CPM LFC of each (KO, donor, condition) vs the control of the SAME
    (donor, condition). Mirrors mold's within-cell-type recipe."""
    mat = pb.values.astype(np.float32)
    lib = mat.sum(1, keepdims=True); lib[lib == 0] = 1.0
    mat = mat / lib * 1e6                                   # CPM per group
    ko = pb.index.get_level_values(0).values.astype(str)
    key = np.array([f"{d}|{c}" for d, c in zip(pb.index.get_level_values(1),
                                               pb.index.get_level_values(2))])
    ctrl_row = {k: i for i, (g, k) in enumerate(zip(ko, key)) if g == CONTROL}
    keep = np.array([i for i in range(len(ko))
                     if ko[i] != CONTROL and key[i] in ctrl_row
                     and counts.iloc[i] >= min_cells])
    ref = np.array([ctrl_row[key[i]] for i in keep])
    lfc = np.log2((mat[keep] + pseudocount) / (mat[ref] + pseudocount))
    out = pd.DataFrame(lfc, columns=pb.columns, index=pd.MultiIndex.from_arrays(
        [ko[keep], pb.index.get_level_values(1)[keep], pb.index.get_level_values(2)[keep]],
        names=['perturbation', 'donor', 'condition']))
    return out, counts.iloc[keep]


def collapse(lfc, counts, level):
    """Cell-count-weighted mean of the within-group LFC over the given index level(s)."""
    w = counts.reindex(lfc.index).values.astype(np.float64)[:, None]
    num = pd.DataFrame(lfc.values * w, index=lfc.index, columns=lfc.columns) \
        .groupby(level=level, observed=True).sum()
    den = pd.Series(w.ravel(), index=lfc.index).groupby(level=level, observed=True).sum()
    out = num.div(den, axis=0)
    if isinstance(level, str) or len(np.atleast_1d(level)) == 1:
        shared = out.index.intersection(out.columns)
        if len(shared):                                     # mask self-KO
            out.values[out.index.get_indexer(shared), out.columns.get_indexer(shared)] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=NPZ)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--plot-dir', default=PLOT_DIR)
    ap.add_argument('--min-cells', type=int, default=20)
    ap.add_argument('--save-per-donor', action='store_true',
                    help='Also write the four per-donor LFC CSVs (~370 MB each).')
    args = ap.parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    pb, counts = load(args.npz)
    print(f"pseudobulk {pb.shape} | {int(counts.sum()):,} cells", flush=True)

    lfc_g, cnt_g = lfc_within(pb, counts, args.min_cells)
    print(f"within (donor,condition) LFC: {lfc_g.shape[0]:,} rows", flush=True)

    per_donor = {d: collapse(lfc_g.xs(d, level='donor'),
                             cnt_g.xs(d, level='donor'), 'perturbation')
                 for d in sorted(set(lfc_g.index.get_level_values('donor')))}
    donors = sorted(per_donor)
    print("per-donor LFC:", {d: per_donor[d].shape for d in donors}, flush=True)

    # donor-corrected = mean over donors (equal weight per donor)
    common = sorted(set.intersection(*[set(v.index) for v in per_donor.values()]))
    dc = sum(per_donor[d].loc[common] for d in donors) / len(donors)
    dc.to_csv(f'{args.out_dir}/marson_lfc_cpm_donorcorrected.csv')
    print(f"donor-corrected LFC: {dc.shape} (KOs in all {len(donors)} donors)", flush=True)
    if args.save_per_donor:
        for d in donors:
            per_donor[d].to_csv(f'{args.out_dir}/marson_lfc_cpm_donor_{d}.csv')

    # ---- reproducibility across donors ----
    R = pd.DataFrame(index=donors, columns=donors, dtype=float)
    for i, a in enumerate(donors):
        for b in donors:
            A = per_donor[a].loc[common].values.ravel()
            B = per_donor[b].loc[common].values.ravel()
            m = np.isfinite(A) & np.isfinite(B)
            R.loc[a, b] = np.corrcoef(A[m], B[m])[0, 1]
    print("\ndonor-donor LFC correlation:\n", R.round(3).to_string(), flush=True)

    off = R.values[~np.eye(len(donors), dtype=bool)]
    print(f"mean off-diagonal r = {off.mean():.3f}", flush=True)

    summary = pd.DataFrame({
        'n_KOs': {d: per_donor[d].shape[0] for d in donors},
        'n_cells': {d: int(counts.xs(d, level='donor').sum()) for d in donors},
        'mean_abs_lfc': {d: float(np.nanmean(np.abs(per_donor[d].values))) for d in donors},
        'mean_r_to_other_donors': {d: float(R.loc[d].drop(d).mean()) for d in donors},
    })
    summary.to_csv(f'{args.out_dir}/marson_donor_summary.csv')
    print("\n", summary.round(3).to_string(), flush=True)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    im = ax[0].imshow(R.values.astype(float), vmin=min(0.0, float(off.min())), vmax=1, cmap='viridis')
    ax[0].set_xticks(range(len(donors)), donors); ax[0].set_yticks(range(len(donors)), donors)
    for i in range(len(donors)):
        for j in range(len(donors)):
            ax[0].text(j, i, f"{R.values[i,j]:.2f}", ha='center', va='center',
                       color='w' if R.values[i, j] < 0.7 else 'k', fontsize=10)
    ax[0].set_title('Donor-donor LFC correlation'); fig.colorbar(im, ax=ax[0], shrink=.8)

    ax[1].bar(donors, summary['mean_abs_lfc'], color='steelblue')
    ax[1].set_ylabel('mean |LFC|'); ax[1].set_title('Effect size per donor')
    for i, d in enumerate(donors):
        ax[1].text(i, summary['mean_abs_lfc'][d], f"n={summary['n_KOs'][d]:,}",
                   ha='center', va='bottom', fontsize=9)

    # per-condition effect size, pooled over donors
    by_cond = collapse(lfc_g, cnt_g, ['perturbation', 'condition'])
    conds = sorted(set(by_cond.index.get_level_values('condition')))
    vals = [np.nanmean(np.abs(by_cond.xs(c, level='condition').values)) for c in conds]
    ax[2].bar(conds, vals, color='indianred')
    ax[2].set_ylabel('mean |LFC|'); ax[2].set_title('Effect size per stimulation condition')
    plt.tight_layout()
    p = f'{args.plot_dir}/marson_per_donor.png'
    plt.savefig(p, dpi=200, bbox_inches='tight')
    print(f"\nwrote {p}", flush=True)


if __name__ == '__main__':
    main()
