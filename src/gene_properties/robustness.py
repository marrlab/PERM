#!/usr/bin/env python
"""Uncertainty, significance and stress-testing for the matched feature results.

WHY THIS IS SEPARATE FROM matching.py. Effect sizes are the primary output, but an
effect with no interval is not a finding -- a reader cannot tell -29 points from
noise, and the guide's section 3e asked for multiple-testing correction that was
never applied. This adds both, plus the robustness checks that decide whether the
CpG-island result is real or an artefact of one arbitrary choice.

TWO INDEPENDENT UNCERTAINTY ESTIMATES, deliberately:

  BOOTSTRAP CI   resample the memory genes with replacement and RE-MATCH each
      draw, so the interval carries the variability of the matching itself, not
      just of the outcome. Re-matching per draw matters: a bootstrap that reuses
      one fixed matched set understates the interval, because the choice of
      controls is itself a source of variance.

  STRATIFIED PERMUTATION p   shuffle the memory/background label WITHIN expression
      deciles, so every permuted set has the memory genes' expression distribution
      by construction. The null is therefore "no association beyond expression",
      which is exactly the hypothesis at issue -- an unstratified shuffle would
      test "no association at all" and would be trivially rejected by the
      expression difference alone.

BH-FDR is applied across features WITHIN a dataset, not across the whole grid.
The features are correlated (GO count, domains and degree all track annotation
depth), so treating 13 x 5 as independent tests would over-correct.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

N_BOOT = 400
N_PERM = 2000
# 50 STRATA, not 10. With 10 the permutation null under-corrects for expression and
# inflates z for features whose matched estimate is ~0: Harmange protein partners
# reads z=+4.0 at 10 bins, +1.0 at 50 and -0.4 at 100, against a matched estimate of
# -0.04. The CpG-island effect is unmoved by the same sweep (-17.6 / -16.8 / -16.6),
# which is itself the cleanest evidence it is real rather than a resolution artefact.
# 50 is the same bin count the project's expression matching already standardised on.
N_DECILES = 50
RNG = 0


def _match_idx(fg_cov, bg_cov, k=10):
    """Indices of the k nearest background rows for each foreground row."""
    mu, sd = bg_cov.mean(0), bg_cov.std(0, ddof=1)
    sd[sd == 0] = 1
    tree = cKDTree((bg_cov - mu) / sd)
    _, idx = tree.query((fg_cov - mu) / sd, k=min(k, len(bg_cov)))
    return np.atleast_2d(idx)


def matched_effect(fg_v, fg_cov, bg_v, bg_cov, k=10, kind="mean"):
    """Difference in the outcome between memory genes and their matched controls."""
    idx = _match_idx(fg_cov, bg_cov, k).ravel()
    ctrl = bg_v[idx]
    ok = np.isfinite(ctrl)
    if not ok.any() or not np.isfinite(fg_v).any():
        return np.nan
    d = np.nanmean(fg_v) - ctrl[ok].mean()
    if kind == "pct":                      # binary outcome -> percentage points
        return 100 * d
    pooled = np.sqrt((np.nanvar(fg_v, ddof=1) + ctrl[ok].var(ddof=1)) / 2)
    return d / pooled if pooled > 0 else np.nan


def bootstrap_ci(fg_v, fg_cov, bg_v, bg_cov, k=10, n=N_BOOT, kind="mean", seed=RNG):
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    m = len(fg_v)
    for i in range(n):
        s = rng.integers(0, m, m)
        out[i] = matched_effect(fg_v[s], fg_cov[s], bg_v, bg_cov, k, kind)
    out = out[np.isfinite(out)]
    if len(out) < 20:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_test(values, is_mem, expr, n=N_PERM, deciles=N_DECILES, seed=RNG):
    """Stratified permutation -> (p_normal, p_empirical, z) from ONE null."""
    ok = np.isfinite(values) & np.isfinite(expr)
    v, m, e = values[ok], is_mem[ok], expr[ok]
    if m.sum() < 20 or (~m).sum() < 20:
        return np.nan
    q = pd.qcut(pd.Series(e), deciles, labels=False, duplicates="drop").to_numpy()
    obs = v[m].mean() - v[~m].mean()
    rng = np.random.default_rng(seed)
    lab = m.copy()
    null = np.empty(n)
    for i in range(n):
        p = lab.copy()
        for b in np.unique(q):
            j = np.where(q == b)[0]
            p[j] = rng.permutation(lab[j])
        null[i] = v[p].mean() - v[~p].mean()
    # COMPARE obs TO THE NULL DISTRIBUTION, never |obs| to |null|. The stratified
    # null is NOT centred on zero: it reproduces the memory genes' expression
    # profile, and expression predicts most of these features, so the null centre
    # is the expression-driven effect. For T-lineage CpG islands the null sits at
    # +30.8 pts (sd 1.8) while the observed is -1.3 -- about 18 SD away. The
    # magnitude test called that p = 1.0 (|obs| is SMALLER than |null|), i.e. it
    # reported the strongest effect in the analysis as non-significant. Two-sided
    # p against the null's own location is the correct statistic.
    # P COMES FROM THE Z, NOT FROM THE EMPIRICAL TAIL. The empirical two-sided p
    # cannot go below 1/(n+1), so with n=1000 it floors at 1e-3 -- and 33 of 56
    # feature x dataset cells hit that floor at once. BH then multiplies floored
    # values by n/rank, every q lands near 1.4e-3, and EVERY star prints as "**":
    # the significance was reporting the permutation count, not the evidence, while
    # the underlying z ranged from -17.1 to +18.3.
    #
    # The null here is a difference of means over ~200 vs ~20,000 genes, and it is
    # normal to a good approximation (skew and excess kurtosis both |.|<0.5,
    # Shapiro p>0.03 on every feature checked), so the normal tail is the honest
    # high-resolution p. The empirical p is returned alongside for reference.
    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 0 else np.nan
    p_norm = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    left, right = (null <= obs).mean(), (null >= obs).mean()
    p_emp = max(min(1.0, 2 * min(left, right)), 1.0 / (n + 1))
    return p_norm, p_emp, float(z)


def bh(p):
    """Benjamini-Hochberg, NaN-safe, returns q in the input order."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    o = np.argsort(v)
    n = len(v)
    adj = np.empty(n)
    prev = 1.0
    for r in range(n - 1, -1, -1):
        prev = min(prev, v[o[r]] * n / (r + 1))
        adj[o[r]] = prev
    q[ok] = adj
    return q
