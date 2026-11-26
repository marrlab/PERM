"""Threshold sweep for sister-concordant genes (SCGs).

The canonical SCG call is `concordance_freq >= 0.80` (findingSCGs.py, `is_biased`).
`concordance_freq` -- the fraction of the n_iterations reshuffles in which the gene
came out BH-significant -- is computed WITHOUT reference to that threshold, so the
whole 0->1 sweep comes out of a SINGLE pass of the statistics. This script therefore:

  1. runs `run_heritability_stats` once per seed (the expensive part),
  2. dumps the full continuous per-gene table for every tested gene,
  3. re-thresholds that table over a grid and writes tidy count/overlap tables.

Nothing about the SCG definition itself is changed: clone_size window defaults to
[2, 2] (true sister pairs, the canonical definition) and the statistic is imported
from findingSCGs.py rather than reimplemented.

Multiple seeds are supported because `concordance_freq` is itself a Monte-Carlo
estimate: the gene count at a given threshold has sampling noise, which matters most
near freq = 1.0 where the counts are small.

Outputs (per dataset, `<stem> = <dataset><_tag>`):
  scg_concordance_full_<stem>.csv    long: seed,timepoint,gene,...,concordance_freq
  scg_threshold_counts_<stem>.csv    long: seed,threshold,timepoint,n_scg,...
  scg_threshold_overlap_<stem>.csv   long: seed,threshold,n_T1,n_T2,n_union,...

Example:
  python sweep_thresholds_SCGs.py \
      --data T-lineage $PERM_DATA/datasets/T-lineage_processed.h5ad \
      --memory_path $PERM_DATA/potential_heritable_genes_filtered.csv \
      --output_dir $PERM_OUTPUTS/scg_threshold_sweep \
      --n_seeds 3

The bottom half of this file READS those three tables back for figures and the
summary report, so diff_thresholds_SCGs.ipynb is just a few calls:

  from sweep_thresholds_SCGs import load_sweep, plot_counts_and_score, threshold_report
  sw  = load_sweep('T-lineage')
  fig = plot_counts_and_score(sw)
  rep = threshold_report(sw)
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
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from findingSCGs import classify_and_standardize, run_heritability_stats

# 0.05-spaced grid over the full range, so the canonical 0.80 is one point on a
# curve rather than a special case. Endpoint 1.00 = "significant in every reshuffle".
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def concordance_table(adata, her_ensg_set, seed, n_iterations, min_clone_size, max_clone_size):
    """One statistics pass -> long per-gene table with the continuous score."""
    np.random.seed(seed)
    adata, tps, time_col, _ = classify_and_standardize(adata)
    res = run_heritability_stats(
        adata, her_ensg_set, tps, time_col,
        n_iterations=n_iterations,
        min_clone_size=min_clone_size, max_clone_size=max_clone_size,
    )
    frames = []
    for tp, df in res.items():
        out = df.copy()
        out.insert(0, 'timepoint', tp)
        out.insert(0, 'seed', seed)
        # is_biased here reflects findingSCGs' hardcoded default; the sweep
        # recomputes the call at every threshold, so drop it to avoid confusion.
        out = out.drop(columns=['is_biased'])
        frames.append(out.rename_axis('gene').reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sweep(full, thresholds):
    """Re-threshold the continuous score. Returns (per-timepoint counts, overlap)."""
    counts, overlap = [], []
    for seed in sorted(full['seed'].unique()):
        sub = full[full['seed'] == seed]
        timepoints = sorted(sub['timepoint'].unique())
        for thr in thresholds:
            hit = sub['concordance_freq'] >= thr
            sets = {}
            for tp in timepoints:
                m = sub['timepoint'] == tp
                scg = sub.loc[m & hit]
                sets[tp] = set(scg['gene'])
                counts.append({
                    'seed': seed, 'threshold': thr, 'timepoint': tp,
                    'n_tested': int(m.sum()),
                    'n_scg': len(scg),
                    'frac_scg': len(scg) / max(int(m.sum()), 1),
                    'n_scg_heritable': int(scg['is_heritable'].sum()),
                    'median_lfc_ratio': float(scg['lfc_ratio'].median()) if len(scg) else np.nan,
                })
            # pairwise overlap; with 2 timepoints this is the single T1-vs-T2 row
            for a, b in itertools.combinations(timepoints, 2):
                inter = sets[a] & sets[b]
                union = sets[a] | sets[b]
                overlap.append({
                    'seed': seed, 'threshold': thr, 'tp_a': a, 'tp_b': b,
                    'n_a': len(sets[a]), 'n_b': len(sets[b]),
                    'n_union': len(union), 'n_intersect': len(inter),
                    'jaccard': len(inter) / len(union) if union else np.nan,
                })
    return pd.DataFrame(counts), pd.DataFrame(overlap)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs=2, action='append', metavar=('NAME', 'PATH'),
                   required=True, help='Dataset name and h5ad path (repeatable).')
    p.add_argument('--memory_path', type=str, required=True,
                   help='CSV of potential heritable genes (fills is_heritable).')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--n_seeds', type=int, default=1,
                   help='Independent reshuffle seeds (0..n-1). >1 gives an error band.')
    p.add_argument('--n_iterations', type=int, default=100,
                   help='Reshuffles per run; also the resolution of concordance_freq.')
    p.add_argument('--min_clone_size', type=int, default=2)
    p.add_argument('--max_clone_size', type=int, default=2,
                   help='2 = canonical true-sister definition. Do not widen casually.')
    p.add_argument('--tag', type=str, default='',
                   help="Label inserted into filenames (e.g. 'clones2-8').")
    p.add_argument('--thresholds', type=float, nargs='+', default=None,
                   help='Override the default 0.05..1.00 grid.')
    args = p.parse_args()

    thresholds = np.round(np.asarray(args.thresholds), 4) if args.thresholds else DEFAULT_THRESHOLDS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    her_ensg_set = set(
        pd.read_csv(args.memory_path)['gene_id'].dropna().astype(str).tolist()
    )
    print(f"Loaded {len(her_ensg_set)} potential heritable gene IDs.")

    for name, path in args.data:
        stem = f"{name}_{args.tag}" if args.tag else name
        print(f"\n>>> {name}  (clone_size {args.min_clone_size}-{args.max_clone_size}, "
              f"{args.n_seeds} seed(s) x {args.n_iterations} iterations)")
        per_seed = []
        for seed in range(args.n_seeds):
            print(f"  -- seed {seed}")
            adata = sc.read_h5ad(path)
            per_seed.append(concordance_table(
                adata, her_ensg_set, seed, args.n_iterations,
                args.min_clone_size, args.max_clone_size))
            del adata
            gc.collect()

        full = pd.concat(per_seed, ignore_index=True)
        if full.empty:
            print("  !! no timepoint yielded >=3 within-clone pairs; nothing written.")
            continue

        counts, ovl = sweep(full, thresholds)
        full.to_csv(out_dir / f'scg_concordance_full_{stem}.csv', index=False)
        counts.to_csv(out_dir / f'scg_threshold_counts_{stem}.csv', index=False)
        ovl.to_csv(out_dir / f'scg_threshold_overlap_{stem}.csv', index=False)
        print(f"  wrote 3 tables to {out_dir} (stem: {stem})")
        print(counts[counts['seed'] == 0]
              .pivot(index='threshold', columns='timepoint', values='n_scg')
              .to_string())


# =========================================================================== #
# Reading the sweep back: figures + report
# =========================================================================== #
# Everything below consumes the three CSVs written above. Kept in this module
# rather than a notebook so the file layout is defined in exactly one place.

SWEEP_DIR = Path(f'{P.OUTPUTS}/scg_threshold_sweep')
PLOTS_DIR = Path(f'{P.PLOTS}/scg_thresholds')
CANON = 0.80                       # the threshold findingSCGs.py actually uses
TP_COLORS = {'T1': '#2563eb', 'T2': '#d97706',
             'A': '#2563eb', 'B': '#d97706',
             'E': '#2563eb', 'M': '#d97706'}
CONVENTIONAL = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

PLOT_RC = {'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
           'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6}
FONTSIZE = 15          # base text size; every other size is derived from it


def _rc(fontsize=FONTSIZE):
    """PLOT_RC with all text sizes scaled off one base size, so a single
    `fontsize=` argument moves everything together instead of leaving tick
    labels or legends behind at the matplotlib defaults."""
    return {**PLOT_RC,
            'font.size': fontsize,
            'axes.titlesize': fontsize + 1,
            'axes.labelsize': fontsize + 1,
            'xtick.labelsize': fontsize - 1,
            'ytick.labelsize': fontsize - 1,
            'legend.fontsize': fontsize - 1,
            'legend.title_fontsize': fontsize - 1,
            'figure.titlesize': fontsize + 2}


def load_sweep(dataset, sweep_dir=SWEEP_DIR):
    """Read the three sweep tables for one dataset stem.

    Returns a dict with 'full' (every tested gene, continuous score), 'counts'
    (seed x threshold x timepoint) and 'overlap' (seed x threshold), plus the
    'dataset', 'timepoints' and 'seeds' present.
    """
    sweep_dir = Path(sweep_dir)
    sw = {
        'dataset': dataset,
        'full':    pd.read_csv(sweep_dir / f'scg_concordance_full_{dataset}.csv'),
        'counts':  pd.read_csv(sweep_dir / f'scg_threshold_counts_{dataset}.csv'),
        'overlap': pd.read_csv(sweep_dir / f'scg_threshold_overlap_{dataset}.csv'),
    }
    sw['timepoints'] = sorted(sw['counts']['timepoint'].unique())
    sw['seeds'] = sorted(sw['full']['seed'].unique())
    return sw


def _band(df, value):
    """Mean / min / max of `value` across seeds, indexed by threshold."""
    g = df.groupby('threshold')[value]
    return g.mean(), g.min(), g.max()


def seed_jaccard_table(sw):
    """Set-level reproducibility: at each threshold, do two independent reshuffle
    seeds call the SAME genes? The seed band on the count curve only shows that
    the COUNTS agree, which is much weaker than agreeing on membership.
    """
    full, seeds = sw['full'], sw['seeds']
    thresholds = sorted(sw['counts']['threshold'].unique())
    rows = []
    for tp in sw['timepoints']:
        piv = (full[full.timepoint == tp]
               .pivot(index='gene', columns='seed', values='concordance_freq'))
        for thr in thresholds:
            sets = {s: set(piv.index[piv[s] >= thr]) for s in seeds}
            js = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                  for a, b in itertools.combinations(seeds, 2) if sets[a] | sets[b]]
            rows.append({'timepoint': tp, 'threshold': thr,
                         'seed_jaccard': float(np.mean(js)) if js else np.nan})
    return pd.DataFrame(rows)


def plot_counts_and_score(sw, canon=CANON, tp_colors=None, plots_dir=PLOTS_DIR,
                          fontsize=FONTSIZE, figsize=(12.5, 5.0),
                          save=True, show=True):
    """Two panels: (A) how many SCGs survive each threshold, with a seed band,
    and (B) where the continuous score actually sits.

    Returns the Figure.
    """
    import matplotlib.pyplot as plt

    counts, full = sw['counts'], sw['full']
    tps, seeds, dataset = sw['timepoints'], sw['seeds'], sw['dataset']
    tp_colors = tp_colors or TP_COLORS
    colors = {tp: tp_colors.get(tp, f'C{i}') for i, tp in enumerate(tps)}

    with plt.rc_context(_rc(fontsize)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # A -- how many SCGs survive each threshold
        ax = axes[0]
        for tp in tps:
            m, lo, hi = _band(counts[counts.timepoint == tp], 'n_scg')
            ax.plot(m.index, m.values, color=colors[tp], lw=2.4, label=tp)
            ax.fill_between(m.index, lo.values, hi.values, color=colors[tp],
                            alpha=0.18, lw=0)
            if canon in m.index:
                ax.annotate(f'{m.loc[canon]:.0f}', (canon, m.loc[canon]),
                            textcoords='offset points', xytext=(6, 6),
                            color=colors[tp], fontweight='bold', fontsize=fontsize - 1)
        ax.axvline(canon, color='#6b7280', ls='--', lw=1.2)
        ax.text(canon, ax.get_ylim()[1], f' canonical {canon:.2f}', color='#6b7280',
                va='top', fontsize=fontsize - 2)
        ax.set(xlabel='concordance-frequency threshold', ylabel='SCGs called',
               title='A  SCG count vs threshold')
        ax.legend(frameon=False, title='timepoint')

        # B -- where the continuous score actually sits.
        # concordance_freq only takes values k/100, and 0.05-spaced edges land
        # exactly on many of them; numpy then assigns those genes to whichever
        # side floating point happens to put the edge. Offset the edges by half
        # the underlying grid so no value can sit on one -- the last bin then
        # cleanly isolates the freq == 1.00 spike.
        ax = axes[1]
        edges = np.arange(-0.005, 1.0451, 0.05)
        for tp in tps:
            v = full[(full.seed == seeds[0]) & (full.timepoint == tp)]['concordance_freq']
            ax.hist(v, bins=edges, histtype='step', lw=2.4, color=colors[tp], label=tp)
        ax.axvline(canon, color='#6b7280', ls='--', lw=1.2)
        ax.set_yscale('log')
        ax.set_xlim(-0.03, 1.06)
        ax.set(xlabel='concordance_freq', ylabel='genes (log)',
               title='B  Distribution of the continuous score')
        ax.legend(frameon=False, title='timepoint')

        fig.suptitle(f'{dataset} — how many SCGs, and how the score is distributed '
                     f'(clone_size 2, {len(seeds)} seeds x 100 reshuffles)', y=1.02)
        fig.tight_layout()

        if save:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            out = plots_dir / f'{dataset}_threshold_counts_and_score.png'
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(out)
        if show:
            plt.show()
    return fig


def threshold_report(sw, stab=None, conventional=CONVENTIONAL, plots_dir=PLOTS_DIR,
                     save=True):
    """Counts, T1/T2 overlap and seed reproducibility at the conventional cuts."""
    counts, ovl = sw['counts'], sw['overlap']
    if stab is None:
        stab = seed_jaccard_table(sw)

    tbl = (counts[counts.threshold.isin(conventional)]
           .groupby(['threshold', 'timepoint'])
           .agg(n_scg_mean=('n_scg', 'mean'), n_scg_sd=('n_scg', 'std'),
                frac=('frac_scg', 'mean'))
           .round(3).unstack('timepoint'))
    # unstack leaves 2-level columns (stat, timepoint); flatten so the
    # single-level overlap/stability columns can be joined on
    tbl.columns = [f'{stat}_{tp}' for stat, tp in tbl.columns]

    st = (stab[stab.threshold.isin(conventional)]
          .pivot(index='threshold', columns='timepoint', values='seed_jaccard')
          .round(3).add_prefix('seed_jaccard_'))
    st.columns.name = None

    o = (ovl[ovl.threshold.isin(conventional)]
         .groupby('threshold')[['n_intersect', 'jaccard']].mean().round(3))

    report = tbl.join(o).join(st)
    if save:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(plots_dir / f"{sw['dataset']}_threshold_report.csv")
    return report


def marginal_genes(sw, timepoint=None, canon=CANON, n=15):
    """Genes the canonical cut keeps but a perfect-reproducibility cut would drop.

    Also prints the set sizes at freq == 1.00 / >= canon / >= 0.50, which is the
    quickest way to see how much of the SCG set rests on marginal calls.
    """
    full, seeds = sw['full'], sw['seeds']
    s0 = full[full.seed == seeds[0]]
    for tp in sw['timepoints']:
        d = s0[s0.timepoint == tp].set_index('gene')['concordance_freq']
        core, canon_set, loose = (set(d.index[d >= 1.00]), set(d.index[d >= canon]),
                                  set(d.index[d >= 0.50]))
        print(f'{tp}:  freq==1.00 {len(core):>5}   >={canon:.2f} {len(canon_set):>5}   '
              f'>=0.50 {len(loose):>5}   |  lost {canon:.2f}->1.00: '
              f'{len(canon_set - core)}   gained {canon:.2f}->0.50: {len(loose - canon_set)}')

    tp = timepoint or sw['timepoints'][0]
    d = s0[s0.timepoint == tp].set_index('gene')
    marginal = d[(d.concordance_freq >= canon) & (d.concordance_freq < 1.0)]
    return (marginal.sort_values('lfc_ratio')
            .head(n)[['concordance_freq', 'lfc_ratio', 'sister_lfc', 'random_lfc']])


# --------------------------------------------------------------------------- #
# Does the SCG set stay perturbation-responsive as the threshold moves?
# --------------------------------------------------------------------------- #
# The threshold controls a size/purity trade-off (panel A): loosening it adds
# genes, tightening it keeps only those significant in every reshuffle. Neither
# tells you whether the ADDED genes are real. This does: at each threshold the
# SCG set is compared against an EXPRESSION-MATCHED control set on perturb-seq
# knockout responsiveness. If a looser cut were just adding noise, its curve
# would collapse toward the matched control.
#
# The matching procedure is deliberately identical to
# marson_KO.plot_responsiveness_matched (bin baseline expression into n_bins
# quantiles, draw controls with the SCG set's per-bin counts, repeat n_perm
# times for a null of medians) so numbers here are comparable to that figure.
# It is reimplemented rather than imported because that function draws its own
# two-panel figure per gene set; here we need one violin panel per threshold.

MARSON_GENE_SUMMARY = Path(f'{P.DATA}/marson/marson_gene_summary_cpm.csv')
MARSON_CTRL_EXPR = Path(f'{P.DATA}/marson/marson_gene_ctrl_expr.csv')
DEFAULT_EVAL_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 1.00)


def _as_series(obj, path_default=None):
    """Accept a path, a DataFrame (first column) or a Series."""
    if obj is None:
        obj = path_default
    if isinstance(obj, (str, Path)):
        obj = pd.read_csv(obj, index_col=0)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.dropna()


def responsiveness_across_thresholds(
        sw, thresholds=DEFAULT_EVAL_THRESHOLDS, timepoint=None, seed=None,
        gene_summary=None, gene_expr=None, metric='mean_abs_lfc',
        n_bins=10, n_iter=20, n_perm=200, random_seed=0,
        perturb_label='Marson', ylabel='Mean |LFC| across KOs',
        fontsize=FONTSIZE, plots_dir=PLOTS_DIR, save=True, show=True):
    """SCG vs expression-matched controls, at several concordance thresholds.

    One violin pair per threshold: the SCG set called at that cut, and an
    expression-matched control set of the same size drawn from the genes it did
    NOT call. Returns (fig, stats) where `stats` has one row per threshold with
    the medians, the ratio, the permutation z / p and the set size.

    sw          : dict from load_sweep()
    timepoint   : which timepoint's SCG set to score (default: the first)
    seed        : which reshuffle seed's scores to use (default: the first)
    gene_summary: per-gene responsiveness table (path / DataFrame); the `metric`
                  column is used. Defaults to the Marson CPM summary.
    gene_expr   : baseline control expression per gene, used for the matching.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import norm

    rng = np.random.default_rng(random_seed)
    tp = timepoint or sw['timepoints'][0]
    sd = sw['seeds'][0] if seed is None else seed

    if gene_summary is None:
        gene_summary = MARSON_GENE_SUMMARY
    if isinstance(gene_summary, (str, Path)):
        gene_summary = pd.read_csv(gene_summary, index_col=0)
    resp = gene_summary[metric].dropna()          # per-gene responsiveness
    expr = _as_series(gene_expr, MARSON_CTRL_EXPR)  # baseline expr, for matching

    scores = (sw['full'].query('timepoint == @tp and seed == @sd')
              .set_index('gene')['concordance_freq'])
    measured = resp.index.intersection(expr.index)
    scores = scores.reindex(measured).dropna()
    ebins = pd.Series(pd.qcut(expr.loc[measured], n_bins, labels=False,
                              duplicates='drop'), index=measured)

    rows, plot_rows = [], []
    for thr in thresholds:
        mem = list(scores.index[scores >= thr])
        if len(mem) < 10:
            print(f'  threshold {thr:.2f}: only {len(mem)} genes measured -- skipped')
            continue
        pool = measured.difference(mem)
        per_bin = ebins.loc[mem].value_counts()
        binpool = {b: ebins.index[(ebins == b) & ebins.index.isin(pool)].to_numpy()
                   for b in per_bin.index}

        sorted_draws, null_medians = [], []
        for _ in range(max(n_perm, n_iter)):
            samp = []
            for b, need in per_bin.items():
                take = min(int(need), len(binpool[b]))
                if take:
                    samp.extend(rng.choice(binpool[b], size=take, replace=False))
            v = np.sort(resp[samp].values)
            sorted_draws.append(v)
            null_medians.append(np.median(v))
        null_medians = np.asarray(null_medians)

        L = min(len(d) for d in sorted_draws[:n_iter])
        bg_vals = np.mean([d[:L] for d in sorted_draws[:n_iter]], axis=0)
        mem_vals = resp[mem].values

        mem_med, null_mu = float(np.median(mem_vals)), float(null_medians.mean())
        null_sd = float(null_medians.std(ddof=1))
        z = (mem_med - null_mu) / null_sd if null_sd > 0 else np.nan
        rows.append({'threshold': thr, 'n_scg': len(mem_vals),
                     'scg_median': mem_med, 'matched_median': null_mu,
                     'ratio': mem_med / null_mu if null_mu else np.nan,
                     'z': z, 'p': float(norm.sf(z)) if np.isfinite(z) else np.nan,
                     'beats': int((null_medians < mem_med).sum()), 'n_perm': len(null_medians)})
        lab = f'{thr:.2f}'
        plot_rows += [{'threshold': lab, metric: v, 'Group': 'SCG'} for v in mem_vals]
        plot_rows += [{'threshold': lab, metric: v, 'Group': 'expr-matched control'}
                      for v in bg_vals]

    stats = pd.DataFrame(rows)
    plot_df = pd.DataFrame(plot_rows)

    with plt.rc_context(_rc(fontsize)):
        fig, ax = plt.subplots(figsize=(2.3 * max(len(stats), 1) + 3.0, 6.2))
        sns.violinplot(data=plot_df, x='threshold', y=metric, hue='Group',
                       order=[f'{t:.2f}' for t in stats.threshold],
                       hue_order=['expr-matched control', 'SCG'],
                       palette=['steelblue', 'crimson'], cut=0, density_norm='width',
                       inner='quartile', linewidth=1.1, ax=ax)
        # annotate each pair with its ratio and set size
        ymax = plot_df[metric].quantile(0.995)
        for i, r in stats.reset_index(drop=True).iterrows():
            star = '*' if r['p'] < 0.05 else ''
            ax.text(i, ymax, f"{r['ratio']:.2f}x{star}\nn={int(r['n_scg'])}",
                    ha='center', va='bottom', fontsize=fontsize - 2, fontweight='bold',
                    color='crimson' if r['ratio'] > 1 else '#6b7280')
        ax.set_ylim(top=ymax * 1.32)
        ax.set(xlabel='concordance-frequency threshold', ylabel=ylabel)
        ax.set_title(f"{sw['dataset']} SCGs vs expression-matched controls — "
                     f"{perturb_label} knockout responsiveness ({tp}, seed {sd})")
        ax.legend(frameon=False, loc='upper left')
        fig.tight_layout()

        if save:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            out = plots_dir / f"{sw['dataset']}_{perturb_label}_responsiveness_by_threshold.png"
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(out)
        if show:
            plt.show()
    return fig, stats


if __name__ == '__main__':
    main()
