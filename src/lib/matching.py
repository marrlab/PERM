"""
Multi-covariate matched comparison for the memory-gene responsiveness benchmark.

`marson_KO.compare_gene_responsiveness_expr_matched` matches on ONE variable
(baseline expression) using quantile bins. That is the right tool for one covariate
and the wrong one for three: k quantile bins per covariate gives k**p cells, which
empty out fast.

This module matches on any number of covariates by nearest-neighbour sampling in a
rank-standardised covariate space, and — the part that matters — reports how well
the match actually balanced, so a failed match is visible instead of silent.

Typical use (Fig 2d, the "what survives" ladder):

    from matching import matched_comparison, matching_ladder

    cov = pd.read_csv(f'{P.DATA}/marson/marson_gene_covariates.csv',
                      index_col=0)
    cov['expr'] = gene_expr                      # the existing baseline expression

    lad = matching_ladder(
        gene_summary, cov, memory_genes, background,
        steps={'expression':            ['expr'],
               '+ detection':           ['expr', 'detection_rate'],
               '+ dispersion':          ['expr', 'disp_resid'],
               'all three':             ['expr', 'detection_rate', 'disp_resid']},
        metric='median_abs_lfc')

The output has one row per step with Cliff's delta + CI, the ratio, and the worst
post-match standardised mean difference across covariates.

INTERPRETATION WARNING, keep it attached to the figure. Matching on a covariate is
only a confound control if the covariate is not itself part of what "memory gene"
means. Between-clone variability is arguably constitutive of memory, so matching on
dispersion may be over-adjustment rather than control -- the same argument that
applies to lineage restriction. Report the ladder, do not silently pick the most
conservative rung.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.spatial import cKDTree

from marson_KO import cliffs_delta, cliffs_ci


def _rank_standardise(df):
    """Rank-transform each column to [0,1] then z-score.

    Rank first because these covariates are heavily skewed (expression spans orders
    of magnitude, nb_phi has a long tail); a plain z-score would let one outlier
    dominate the distance and match on nothing else.
    """
    R = df.rank(method='average') / (len(df) + 1)
    return (R - R.mean()) / R.std(ddof=0)


def _smd(a, b):
    """Standardised mean difference; the usual balance diagnostic."""
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return 0.0 if sd == 0 else (np.mean(a) - np.mean(b)) / sd


def matched_comparison(gene_summary, covariates, memory_genes, background,
                       match_on, metric='median_abs_lfc', n_iter=20, k=10,
                       random_seed=0, alternative='greater'):
    """Compare memory genes against a covariate-matched background.

    gene_summary : DataFrame with `metric` as a column, indexed by gene
    covariates   : DataFrame of per-gene covariates, indexed by gene
    memory_genes : iterable of gene symbols
    background   : iterable of gene symbols (the strict "called by no method" set)
    match_on     : list of covariate column names
    n_iter       : independent matched draws; values are pooled across draws
    k            : each memory gene is matched to a random one of its k nearest
                   unused background genes. k=1 is the tightest match but gives
                   the same background every iteration, so the CI would understate
                   sampling variability.

    Returns a dict with the same keys as compare_gene_responsiveness_expr_matched
    ('scg_values', 'non_scg_values', 'mann_whitney_p', 'n_scg'), plus 'balance'
    (a DataFrame of SMD before/after per covariate) and 'cliffs_d'/'d_lo'/'d_hi'.
    """
    match_on = list(match_on)
    mem = [g for g in dict.fromkeys(memory_genes)]
    bg = [g for g in dict.fromkeys(background) if g not in set(mem)]

    df = covariates[match_on].join(gene_summary[[metric]], how='inner').dropna()
    mem = [g for g in mem if g in df.index]
    bg = [g for g in bg if g in df.index]
    if len(mem) < 10:
        raise ValueError(f"only {len(mem)} memory genes survive the covariate join")
    if len(bg) < len(mem):
        raise ValueError(f"background ({len(bg)}) smaller than memory set ({len(mem)})")

    Z = _rank_standardise(df[match_on])
    Zm, Zb = Z.loc[mem].to_numpy(), Z.loc[bg].to_numpy()
    vals = df[metric]
    rng = np.random.default_rng(random_seed)

    tree = cKDTree(Zb)
    kq = min(k, len(bg))
    # order memory genes by how isolated they are, so the hardest ones pick first;
    # matching them last would leave only poor partners and inflate the imbalance
    _, nn_all = tree.query(Zm, k=kq)
    nn_all = np.asarray(nn_all).reshape(len(mem), -1)   # k=1 returns a flat array
    isolation = np.linalg.norm(Zb[nn_all[:, -1]] - Zm, axis=1)
    order = np.argsort(-isolation)

    bg_arr = np.asarray(bg)
    picked_per_iter = []
    for it in range(n_iter):
        used = set()
        chosen = np.empty(len(mem), dtype=object)
        for i in order:
            cands = [c for c in nn_all[i] if c not in used]
            # Widen only if forced, and keep widening -- a single k*5 step can still
            # come up empty when memory genes cluster in one region of covariate
            # space. len(bg) >= len(mem) is checked above, so an unused partner
            # always exists; the last resort just finds it.
            kk = kq
            while not cands and kk < len(bg):
                kk = min(kk * 5, len(bg))
                extra = np.atleast_1d(tree.query(Zm[i], k=kk)[1])
                cands = [c for c in extra if c not in used]
            if not cands:
                cands = [c for c in range(len(bg)) if c not in used]
            pick = int(rng.choice(cands))
            used.add(pick)
            chosen[i] = bg_arr[pick]
        picked_per_iter.append(chosen)

    mem_vals = vals.loc[mem].to_numpy()
    bg_vals = np.concatenate([vals.loc[list(p)].to_numpy() for p in picked_per_iter])

    # p per iteration on matched n, then the median -- pooling all draws into one
    # test would treat n_iter re-uses of the same background pool as independent
    ps = [mannwhitneyu(mem_vals, vals.loc[list(p)].to_numpy(),
                       alternative=alternative)[1] for p in picked_per_iter]

    rows = []
    last = list(picked_per_iter[-1])
    for c in match_on:
        rows.append(dict(covariate=c,
                         smd_before=_smd(Z.loc[mem, c].to_numpy(), Z.loc[bg, c].to_numpy()),
                         smd_after=_smd(Z.loc[mem, c].to_numpy(), Z.loc[last, c].to_numpy())))
    balance = pd.DataFrame(rows).set_index('covariate')

    d = cliffs_delta(mem_vals, bg_vals)
    d_lo, d_hi = cliffs_ci(mem_vals, bg_vals)
    return dict(scg_values=mem_vals, non_scg_values=bg_vals,
                mann_whitney_p=float(np.median(ps)), n_scg=len(mem), n_bg=len(bg),
                cliffs_d=d, d_lo=d_lo, d_hi=d_hi, balance=balance,
                ratio=float(np.median(mem_vals) / np.median(bg_vals)),
                memory_genes=list(mem), matched_genes=last, match_on=match_on)


def matching_ladder(gene_summary, covariates, memory_genes, background, steps,
                    metric='median_abs_lfc', n_iter=20, k=10, random_seed=0):
    """Run matched_comparison once per step; return one tidy row per step.

    `steps` is an ordered {label: [covariate names]} mapping. The point of the
    ladder is the TREND across rungs, not any single rung.
    """
    out = []
    for label, cols in steps.items():
        r = matched_comparison(gene_summary, covariates, memory_genes, background,
                               match_on=cols, metric=metric, n_iter=n_iter, k=k,
                               random_seed=random_seed)
        out.append(dict(step=label, matched_on=', '.join(cols), n=r['n_scg'],
                        mem=float(np.median(r['scg_values'])),
                        bg=float(np.median(r['non_scg_values'])),
                        ratio=r['ratio'], cliffs_d=r['cliffs_d'],
                        d_lo=r['d_lo'], d_hi=r['d_hi'], p=r['mann_whitney_p'],
                        worst_smd=float(r['balance'].smd_after.abs().max())))
    return pd.DataFrame(out)


def win_rate(delta):
    """Cliff's delta as the plain-English quantity people actually understand:
    the share of random memory-vs-background pairings the memory gene wins."""
    return (delta + 1) / 2


def plot_rung(result, covariates, gene_summary, metric='median_abs_lfc',
              title=None, ref_delta=None, savefig=None):
    """One self-contained figure for a single matching rung.

    Left panels  : each matched covariate, memory vs its matched partners. These
                   should overlap almost completely -- that IS the evidence the
                   match worked, and it belongs in the figure rather than in a
                   number the reader has to trust.
    Right panel  : the responsiveness distributions, i.e. the difference that
                   survives after those covariates were equalised.

    ref_delta : optional delta from the expression-only rung, drawn as a reference
                so the reader can see how much this covariate removed.
    """
    import matplotlib.pyplot as plt

    mem, bg = result['memory_genes'], list(result['matched_genes'])
    cols = list(result['match_on'])
    n = len(cols) + 1

    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 1, 4.6))
    axes = np.atleast_1d(axes)

    def _box(ax, a, b, ylabel, log=False):
        ax.boxplot([a, b], widths=.55, showfliers=False,
                   medianprops=dict(color='k', lw=1.6))
        for i, (v, col) in enumerate(((a, 'crimson'), (b, 'steelblue')), start=1):
            jit = np.random.default_rng(0).normal(i, .055, len(v))
            ax.scatter(jit, v, s=4, alpha=.18, color=col, linewidths=0)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['memory', 'matched\nbackground'])
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale('log')

    for j, c in enumerate(cols):
        v = covariates[c]
        _box(axes[j], v.loc[mem].dropna().to_numpy(), v.loc[bg].dropna().to_numpy(),
             c, log=(c == 'expr'))
        smd = result['balance'].loc[c, 'smd_after']
        axes[j].set_title(f"{c}\nmatched  (SMD {smd:+.3f})", fontsize=10)

    v = gene_summary[metric]
    _box(axes[-1], v.loc[mem].dropna().to_numpy(), v.loc[bg].dropna().to_numpy(),
         metric.replace('_', ' '))
    d, lo, hi = result['cliffs_d'], result['d_lo'], result['d_hi']
    axes[-1].set_title(f"responsiveness\ndelta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]", fontsize=10)

    sub = (f"memory gene more responsive in {win_rate(d):.0%} of random pairings"
           "   (50% = no effect)")
    if ref_delta is not None:
        sub += f"\nexpression-only rung was {ref_delta:+.3f} "\
               f"({win_rate(ref_delta):.0%})  ->  this rung removes "\
               f"{100 * (1 - d / ref_delta):.0f}% of the effect"
    fig.suptitle((title or 'matched comparison') + f"\n{sub}", fontsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches='tight')
    return fig


def plot_ladder(ladder, ax=None, title=None, smd_warn=0.10):
    """Cliff's delta with CI per rung. Rungs whose match failed are marked.

    smd_warn : a rung with |SMD| above this after matching is not actually matched;
               it is drawn hollow so it cannot be read as a controlled result.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(1.9 * len(ladder) + 3, 5))

    x = np.arange(len(ladder))
    bad = ladder.worst_smd.abs() > smd_warn
    ax.errorbar(x[~bad], ladder.cliffs_d[~bad],
                yerr=[(ladder.cliffs_d - ladder.d_lo)[~bad],
                      (ladder.d_hi - ladder.cliffs_d)[~bad]],
                fmt='o', color='crimson', capsize=5, markersize=9, label='matched')
    if bad.any():
        ax.errorbar(x[bad], ladder.cliffs_d[bad],
                    yerr=[(ladder.cliffs_d - ladder.d_lo)[bad],
                          (ladder.d_hi - ladder.cliffs_d)[bad]],
                    fmt='o', mfc='white', color='crimson', capsize=5, markersize=9,
                    label=f'imbalanced (|SMD| > {smd_warn})')

    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ladder.step, rotation=20, ha='right')
    ax.set_ylabel("Cliff's delta  (memory vs matched background)")
    ax.set_title(title or 'Effect size under progressively stricter matching')
    ax.legend(frameon=False)
    return ax
