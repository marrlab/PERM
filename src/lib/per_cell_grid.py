"""
per_cell_grid.py - the 4x3 donor x timepoint grid of memory-gene responsiveness.

One panel per (donor, timepoint) cell, each showing the memory gene set against its
expression-matched non-memory background, matched WITHIN that cell. The question the
grid answers: is the responsiveness effect present in every individual cell, or is
the pooled result carried by a subset?

Built from marson_per_cell.py's tables. Reads only the small per-cell gene-summary
and control-expression CSVs, so it runs on the login node.

READING THE GRID -- the cells are NOT equally powered. They hold 41-133 cells per KO
(D4_Rest weakest, D2_Rest strongest, ~3x spread), so delta is attenuated unevenly and
a smaller delta in D4 is expected from noise alone. Compare each panel against zero,
not against each other; the summary panel plots delta against cells-per-KO so that
relationship is visible rather than implicit.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import os

import matplotlib.pyplot as plt          # no matplotlib.use('Agg') at import
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from marson_KO import (load_memory_gene_set, build_consensus_gene_sets,
                       compare_gene_responsiveness_expr_matched, cliffs_delta, cliffs_ci,
                       median_ci)

DATA = f'{P.DATA}/marson'
PLOT_DIR = f'{P.PLOTS}/per_group'
MEMDIR = f'{P.MEMGENES_ROOT}/data/T-lineage'
FILES = {'CORAL': 'coral_memory_genes_T1_T2.csv', 'GEMLI': 'gemli_memory_genes_T1_T2.csv',
         'MOLD': 'mold_memory_genes_T1_T2.csv', 'POWERSEEK': 'powerseek_memory_genes_T1_T2.csv',
         'SCG': 'scg_memory_genes_T1_T2.csv'}
# Same value as figures.EXPR_FLOOR. Genes at or below this mean control CPM are
# dropped before anything else: below it |LFC| tracks the denominator rather than
# real responsiveness. Pass floor=None to reproduce results from before 2026-09-16.
EXPR_FLOOR = 0.1
DONORS = ['D1', 'D2', 'D3', 'D4']
TIMEPOINTS = ['Rest', 'Stim8hr', 'Stim48hr']
MEM_COLOR, BG_COLOR = '#DC143C', '#1F77B4'
NEUTRAL = '#3F4A4F'


def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = p[o] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[o[i]] = min(q[o[i]], q[o[i + 1]])
    return np.minimum(q, 1)


def _star(q):
    return '***' if q < 1e-3 else '**' if q < 1e-2 else '*' if q < 5e-2 else 'n.s.'


def _ratio_ci(a, b, n_boot=500, seed=0):
    """Bootstrap 95% CI for median(a)/median(b) -- resamples both groups jointly."""
    rng = np.random.default_rng(seed)
    r = [np.median(rng.choice(a, len(a), True)) / np.median(rng.choice(b, len(b), True))
         for _ in range(n_boot)]
    return tuple(np.percentile(r, [2.5, 97.5]))


def score_one(gs, expr, method='SCG', top_n=500, timepoint='merge',
              metric='median_abs_lfc', n_bins=30, n_iter=20, bg_top_n=None,
              floor=EXPR_FLOOR):
    """Memory vs strict-background comparison for ONE dataset (a cell, or pooled).

    Shared by cell_table and pooled_row so the forest plot's pooled estimate is
    computed exactly like the per-cell ones -- same background rule, same matching,
    same metric. Anything else and the reference row would not be comparable.
    """
    # ONE expression rule across every panel (2026-09-16): drop genes at or below
    # `floor` mean control CPM BEFORE the universe is fixed, so the memory set is the
    # top N among EXPRESSED genes and the background is drawn from the same pool.
    # Identical to figures.load_marson(floor=...). This module had no floor at all
    # until then, while Fig 2d/2e on the SAME data did.
    if floor is not None:
        keep = expr.index[expr > floor]
        gs = gs.loc[gs.index.intersection(keep)]
        expr = expr.loc[expr.index.intersection(gs.index)]
    # bg_top_n: depth at which "called by no method" is judged (figures.BG_TOP_N);
    # None keeps the old behaviour, the memory-gene cutoff itself
    _, counts = build_consensus_gene_sets(
        min_methods=3, timepoint=timepoint, top_n=bg_top_n or top_n, merge_mode='union',
        memgene_dir=MEMDIR, method_files=FILES, universe=gs.index, verbose=False)
    bg = counts.index[counts == 0].tolist()
    bgset = set(bg)
    mem = [g for g in load_memory_gene_set(method, timepoint, top_n, 'union',
                                           MEMDIR, FILES, restrict_to=gs.index)
           if g not in bgset]
    frame = gs.loc[list(dict.fromkeys(mem + bg))]
    m = compare_gene_responsiveness_expr_matched(
        frame, expr, mem, metric=metric, n_bins=n_bins, n_iter=n_iter)
    a, b = np.asarray(m['scg_values']), np.asarray(m['non_scg_values'])
    m_med, m_lo, m_hi = median_ci(a)
    b_med, b_lo, b_hi = median_ci(b)
    d_lo, d_hi = cliffs_ci(a, b)
    r_lo, r_hi = _ratio_ci(a, b)
    return dict(n_mem=len(mem), n_bg=len(bg),
                mem=float(m_med), mem_lo=float(m_lo), mem_hi=float(m_hi),
                bg=float(b_med), bg_lo=float(b_lo), bg_hi=float(b_hi),
                ratio=float(m_med / b_med), ratio_lo=r_lo, ratio_hi=r_hi,
                d=cliffs_delta(a, b), d_lo=d_lo, d_hi=d_hi,
                p=mannwhitneyu(a, b, alternative='greater')[1]), (a, b)


def pooled_row(method='SCG', top_n=500, timepoint='merge', metric='median_abs_lfc',
               n_bins=30, n_iter=20, floor=EXPR_FLOOR):
    """The same comparison on the POOLED Marson tables -- the reference row."""
    gs = pd.read_csv(f'{DATA}/marson_gene_summary_cpm.csv', index_col=0)
    expr = pd.read_csv(f'{DATA}/marson_gene_ctrl_expr.csv', index_col=0).iloc[:, 0]
    row, _ = score_one(gs, expr, method, top_n, timepoint, metric, n_bins, n_iter,
                       floor=floor)
    row['cell'] = 'POOLED (all 12)'
    return row


def cell_table(method='SCG', top_n=500, timepoint='merge', metric='median_abs_lfc',
               n_bins=30, n_iter=20, donors=DONORS, timepoints=TIMEPOINTS, bg_top_n=None,
               floor=EXPR_FLOOR):
    """One row per (donor, timepoint) cell: memory vs matched background.

    `timepoint` is the MEMORY-GENE FILE's sister-pair timepoint (T1/T2/merge), not a
    Marson stimulation timepoint -- both concepts appear in this grid.
    """
    rows, values = [], {}
    for d in donors:
        for t in timepoints:
            name = f'{d}_{t}'
            gs = pd.read_csv(f'{DATA}/marson_gene_summary_cpm_{name}.csv', index_col=0)
            expr = pd.read_csv(f'{DATA}/marson_gene_ctrl_expr_{name}.csv',
                               index_col=0).iloc[:, 0]
            row, (a, b) = score_one(gs, expr, method, top_n, timepoint,
                                    metric, n_bins, n_iter, bg_top_n, floor)
            values[name] = (a, b)              # kept so the grid can show distributions
            rows.append(dict(cell=name, donor=d, timepoint=t, **row))
            print(f"  {name}: ratio={row['ratio']:.3f}  δ={row['d']:+.3f}  "
                  f"n_mem={row['n_mem']}", flush=True)
    T = pd.DataFrame(rows)
    T['q'] = _bh(T['p'])                       # BH across the 12 cells
    T['sig'] = T['q'].map(_star)
    qc = pd.read_csv(f'{DATA}/marson_cell_summary.csv', index_col=0)
    T = T.merge(qc[['n_cells', 'cells_per_KO', 'n_KOs_used']], left_on='cell',
                right_index=True, how='left')
    # the per-gene distributions ride along so plot_cell_grid can draw boxen panels
    # instead of two summary bars. .attrs does not survive a to_csv round-trip, so a
    # table reloaded from disk falls back to bars.
    T.attrs['values'] = values
    T.attrs['metric'] = metric
    return T


def plot_cell_grid(T, method='SCG', metric=None, style='bar', save=True, FS=12,
                   donors=DONORS, timepoints=TIMEPOINTS, ylim_pct=99):
    """4x3 grid: memory vs matched background in every donor x timepoint cell.

    style : 'bar' | 'boxen' | 'violin'
        'bar' (default) is the clean version -- two medians per cell, easiest to read
        across 12 panels. 'boxen'/'violin' draw the full per-gene distributions, which
        show honestly that the two groups overlap heavily and the shift is modest;
        worth a look before quoting the bars, and a good supplementary panel. Both
        need T.attrs['values'], which cell_table attaches -- a table reloaded from CSV
        has lost it and falls back to bars.
    ylim_pct : upper percentile for the shared y-limit in the distribution styles, so
        a few extreme genes do not flatten all 12 panels.
    """
    vals = T.attrs.get('values')
    metric = metric or T.attrs.get('metric', 'median_abs_lfc')
    if style != 'bar' and not vals:
        print("no per-gene values on this table (reloaded from CSV?) -- drawing bars")
        style = 'bar'

    fig, axes = plt.subplots(len(donors), len(timepoints),
                             figsize=(3.6 * len(timepoints), 3.2 * len(donors)),
                             sharey=True, squeeze=False)
    idx = T.set_index('cell')
    metric_label = metric.replace('_', ' ').replace('abs lfc', '|LFC|')
    order = ['memory', 'matched bg']            # memory on the left, as in the grid
    palette = {'memory': MEM_COLOR, 'matched bg': BG_COLOR}
    if style == 'bar':
        ymax = max(idx['mem'].max(), idx['bg'].max()) * 1.38
    else:
        ymax = max(np.percentile(np.concatenate(v), ylim_pct) for v in vals.values()) * 1.30

    for i, d in enumerate(donors):
        for j, t in enumerate(timepoints):
            ax = axes[i][j]
            name = f'{d}_{t}'
            r = idx.loc[name]
            if style == 'bar':
                ax.bar([0, 1], [r['mem'], r['bg']], width=0.62,
                       color=[MEM_COLOR, BG_COLOR], edgecolor='k', linewidth=0.6)
                ax.set_xticks([0, 1], ['memory', 'matched\nbg'], fontsize=FS - 3)
            else:
                a, b = vals[name]
                df = pd.DataFrame({'value': np.concatenate([a, b]),
                                   'Group': ['memory'] * len(a) + ['matched bg'] * len(b)})
                plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
                kw = {} if style == 'boxen' else dict(cut=0, inner='box')
                plotter(data=df, x='Group', y='value', order=order, hue='Group',
                        hue_order=order, palette=palette, legend=False, ax=ax, **kw)
                ax.set_xlabel('')
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(['memory', 'matched\nbg'], fontsize=FS - 3)
            ax.set_ylim(0, ymax)
            ax.annotate(f"δ={r['d']:+.3f} {r['sig']}", (0.5, 0.98),
                        xycoords='axes fraction', ha='center', va='top',
                        fontsize=FS - 1, fontweight='bold')
            ax.annotate(f"{r['cells_per_KO']:.0f} cells/KO", (0.5, 0.87),
                        xycoords='axes fraction', ha='center', va='top',
                        fontsize=FS - 4, color=NEUTRAL)
            ax.set_ylabel(f'{d}\n{metric_label}' if j == 0 else '', fontsize=FS - 1)
            if i == 0:
                ax.set_title(t, fontsize=FS + 2)
            ax.tick_params(labelsize=FS - 3)
            ax.grid(axis='y', alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

    n_sig = int((T['q'] < 0.05).sum())
    fig.suptitle(f'{method} memory genes vs expression-matched background — '
                 f'all {len(T)} donor x timepoint cells\n'
                 f'effect positive and significant in {n_sig}/{len(T)} cells '
                 '(BH across cells)', fontsize=FS + 5, y=1.0)
    fig.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        p = f'{PLOT_DIR}/cell_grid_{method}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'wrote {p}')
    return fig


def plot_cell_forest(T, method='SCG', pooled=None, save=True, FS=13,
                     donors=DONORS, timepoints=TIMEPOINTS):
    """The 12 cells as a forest plot -- one row, one estimate, one CI.

    Deliberately the least information that still makes the argument: is the effect
    present in every donor x timepoint cell, and does the pooled estimate represent
    them? The paired-dot and grid versions show more but bury that question.

    x is the RATIO of medians (memory / matched background), not Cliff's delta,
    because "memory genes are 1.35x as responsive" is directly readable and its null
    is a line at 1.0. `pooled` is a pooled_row() dict, drawn as a diamond.
    """
    T = T.copy()
    T['_tp'] = pd.Categorical(T['timepoint'], categories=timepoints, ordered=True)
    T = T.sort_values(['donor', '_tp'], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(T))

    h = 0.42 * len(T) + (2.2 if pooled else 1.6)
    fig, ax = plt.subplots(figsize=(8.2, h))
    tp_color = {t: c for t, c in zip(timepoints, ['#0072B2', '#E69F00', '#009E73'])}

    ax.axvline(1.0, color='k', lw=1.4, ls='--', zorder=1)
    for yi, r in T.iterrows():
        ax.plot([r['ratio_lo'], r['ratio_hi']], [yi, yi], color=tp_color[r['timepoint']],
                lw=2.2, solid_capstyle='round', zorder=2)
        ax.scatter(r['ratio'], yi, s=95, color=tp_color[r['timepoint']],
                   edgecolor='w', linewidth=1.2, zorder=3)

    labels = list(T['cell'])
    if pooled:
        yp = len(T) + 0.6
        ax.plot([pooled['ratio_lo'], pooled['ratio_hi']], [yp, yp], color='k', lw=2.6,
                solid_capstyle='round', zorder=2)
        ax.scatter(pooled['ratio'], yp, marker='D', s=150, color='k', zorder=3)
        ax.axhline(len(T) - 0.3, color='0.8', lw=1)
        y = np.append(y, yp)
        labels.append(pooled['cell'])

    ax.set_yticks(y, labels, fontsize=FS - 2)
    ax.set_ylim(-0.7, y.max() + 0.7)
    ax.set_xlabel('responsiveness ratio   (memory / expression-matched background)',
                  fontsize=FS)
    ax.tick_params(axis='x', labelsize=FS - 2)
    ax.grid(axis='x', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=9, label=t) for t, c in tp_color.items()]
    ax.legend(handles=handles, fontsize=FS - 3, frameon=False, ncol=3,
              loc='lower right', bbox_to_anchor=(1.0, -0.02))
    ax.set_title(f'{method} memory genes: responsiveness in every donor × timepoint cell',
                 fontsize=FS + 2, pad=12)

    fig.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        p = f'{PLOT_DIR}/cell_forest_{method}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'wrote {p}')
    return fig


def plot_cell_summary(T, method='SCG', save=True, FS=13):
    """Why pooling is defensible: every cell's delta with CI, and delta vs power.

    Left  - the 12 deltas with 95% CIs and the pooled value; if every CI clears zero,
            pooling is combining like with like rather than averaging over a mixture.
    Right - delta against cells per KO. Attenuation predicts an upward trend, which is
            the power argument for pooling stated as data rather than assertion.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2), squeeze=False)
    ax = ax.ravel()
    # chronological, not alphabetical -- plain sorting puts Stim48hr before Stim8hr
    T = T.copy()
    T['_tp'] = pd.Categorical(T['timepoint'], categories=TIMEPOINTS, ordered=True)
    T = T.sort_values(['donor', '_tp']).reset_index(drop=True)
    x = np.arange(len(T))

    # the two groups plotted directly (same red/blue as the boxen grid), points only --
    # no connecting lines, because the cells are separate experiments, not a series
    ax[0].errorbar(x - 0.10, T['mem'], yerr=[T['mem'] - T.mem_lo, T.mem_hi - T['mem']],
                   fmt='o', ms=9, color=MEM_COLOR, ecolor=MEM_COLOR, capsize=4,
                   elinewidth=1.5, linestyle='none', label='memory gene set', zorder=3)
    ax[0].errorbar(x + 0.10, T['bg'], yerr=[T['bg'] - T.bg_lo, T.bg_hi - T['bg']],
                   fmt='s', ms=9, color=BG_COLOR, ecolor=BG_COLOR, capsize=4,
                   elinewidth=1.5, linestyle='none', label='expr-matched non-memory',
                   zorder=3)
    for xi, r in T.iterrows():
        ax[0].annotate(r['sig'], (xi - 0.10, r['mem_hi']), textcoords='offset points',
                       xytext=(0, 7), ha='center', fontsize=FS,
                       fontweight='bold', color='k' if r['sig'] != 'n.s.' else 'grey')
    ax[0].set_xticks(x, T['cell'], rotation=55, ha='right', fontsize=FS - 3)
    ax[0].set_xlim(-0.6, len(T) - 0.4)
    ax[0].set_ylabel(f'{T.attrs.get("metric", "median_abs_lfc").replace("_", " ")} (CPM)',
                     fontsize=FS)
    ax[0].set_title('Every donor × timepoint cell', fontsize=FS + 2)
    ax[0].legend(fontsize=FS - 3, frameon=False, loc='upper left', ncol=2)

    for a in ax:
        a.tick_params(labelsize=FS - 2)
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)

    fig.suptitle(f'{method}: memory vs expression-matched background in every '
                 'donor × timepoint cell', fontsize=FS + 3, y=1.02)
    fig.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        p = f'{PLOT_DIR}/cell_grid_summary_{method}.png'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'wrote {p}')
    return fig
