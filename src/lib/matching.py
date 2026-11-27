"""
Multi-covariate matched comparison for the memory-gene responsiveness benchmark.

`marson_KO.compare_gene_responsiveness_expr_matched` matches on ONE variable
(baseline expression) using quantile bins. That is the right tool for one covariate
and the wrong one for three: k quantile bins per covariate gives k**p cells, which
empty out fast.

This module matches on any number of covariates by nearest-neighbour sampling in a
rank-standardised covariate space, and — the part that matters — reports how well
the match actually balanced, so a failed match is visible instead of silent.

Typical use (one rung of the Fig. 2f matching panels):

    from matching import matched_comparison

    cov = pd.read_csv(f'{P.DATA}/marson/marson_gene_covariates.csv', index_col=0)
    cov['expr'] = gene_expr                      # the baseline expression
    res = matched_comparison(gene_summary, cov, memory_genes, background,
                             match_on=['expr', 'detection_rate'])

The result carries Cliff's delta + CI, the ratio, and the worst post-match
standardised mean difference across covariates.

INTERPRETATION WARNING, keep it attached to the figure. Matching on a covariate is
only a confound control if the covariate is not itself part of what "memory gene"
means. Between-clone variability is arguably constitutive of memory, so matching on
dispersion may be over-adjustment rather than control -- the same argument that
applies to lineage restriction. Report every rung, do not silently pick the most
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

