#!/usr/bin/env python
"""Matched-background construction and the Tier-B tests that depend on it.

WHY MATCHING, AND WHY IT IS THE WHOLE ANALYSIS. Unadjusted, the memory-gene sets
sit 0.5-2.7 SD above background in expression and 2.5-6x above in publication
count. Every annotation-derived feature (STRING degree, GO term count) inflates
with both. So "memory genes are more connected" and "memory genes are abundant and
famous" are, before matching, literally the same statement. The matched comparison
is what separates them.

HOW. For each memory gene we draw k background genes nearest to it in standardised
covariate space (log expression, detection rate, and log exonic length), WITH
REPLACEMENT. Replacement is not a convenience: in LARRY and Harmange there are
fewer than 4 background genes per memory gene above the memory median expression,
so sampling without replacement would exhaust the pool and quietly re-introduce
the imbalance it was meant to remove.

VALIDATED THREE WAYS. Because a matched comparison can fail silently, the CpG-island
result was reproduced with three methods that control for expression differently --
nearest-neighbour matching, expression-decile stratification, and logistic regression
with a cubic expression spline. They agree within ~2 percentage points on every
dataset (e.g. T-lineage -26.5 / -27.6 / -27.2). Any future feature that matters
should get the same treatment before it is believed.

THE MATCHING IS NOT ASSUMED TO WORK. `balance()` reports the achieved standardised
mean difference per covariate, and anything above BALANCE_TOL is reported as a
FAILED match rather than as a result. A matched comparison whose balance was never
checked is worth no more than the unmatched one.

TWO MATCHES, deliberately. Gene length is missing for ~24% of genes in the two
human datasets -- not renamed genes but clone-based identifiers (RP11-*, AL*) with
no current symbol, and they skew low-expression. Dropping them from one arm only
would make the background more highly expressed than it should be and flatter every
result. So:
  'expr'     log_expr + detection_rate, over the FULL universe (no gene dropped)
  'expr_len' log_expr + detection_rate + log_len, restricted to architecture-matched
             genes IN BOTH ARMS, which is the honest way to control for length.
Features that plausibly scale with length are read off the second; everything else
off the first, with the second as the sensitivity check.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BALANCE_TOL = 0.10          # achieved |SMD| above this = the match failed
K_NEIGHBOURS = 10
# DO NOT MATCH ON EXPRESSION AND DETECTION RATE TOGETHER. They are the same
# variable: Spearman 0.94-0.997, Pearson 0.80-0.96 across the five datasets. In a
# Euclidean nearest-neighbour search over two near-identical standardised axes the
# points lie on a line, so the distance is dominated by the tiny RESIDUAL off that
# line -- and the match is made on the residual, not on expression.
#
# That residual is not noise. It is "high level in few cells" vs "low level in many
# cells", i.e. CELL-TYPE SPECIFICITY, which is precisely the biology under test. So
# matching on both silently absorbs the signal and manufactures a null.
#
# Measured on 2026-09-10: matching on expression alone and on detection alone give
# nearly the SAME background (T-lineage CpG-island rate 88.5% vs 85.2%), while both
# together give 68.8% -- OUTSIDE the range of either. Two genuinely independent
# covariates cannot do that; it is the signature of a degenerate metric. Under the
# two-covariate match, publications read +0.02 and GO terms +0.04 (null); under
# expression alone they are +0.45 and +0.32.
#
# 'expr' is therefore expression ONLY. The old two-covariate set is kept as
# 'expr_detection_DEGENERATE' so the error is reproducible, never as a default.
COVSETS = {"expr": ["log_expr"],
           "expr_len": ["log_expr", "log_len"],
           "expr_detection_DEGENERATE": ["log_expr", "detection_rate"]}


def prep(M):
    """Derived columns every downstream step expects."""
    M = M.copy()
    M["log_len"] = np.log10(M["exonic_len"])
    M["log_deg400"] = np.log10(M["string_degree_400"] + 1)
    M["log_deg700"] = np.log10(M["string_degree_700"] + 1)
    M["log_go"] = np.log10(M["n_go"] + 1)
    return M


def smd(a, b, w_b=None):
    """Standardised mean difference, pooled SD. `w_b` weights the control arm."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    if w_b is None:
        b = b[np.isfinite(b)]; w = None
    else:
        w = np.asarray(w_b, float); ok = np.isfinite(b)
        b, w = b[ok], w[ok]
    # A feature can be entirely absent for a species -- LOEUF and pLI are human-only,
    # so on the mouse datasets every matched control is NaN and the weights sum to
    # zero. Return NaN rather than letting np.average raise; the column is simply
    # not measurable there, which is expected, not an error.
    if len(a) < 5 or len(b) < 5 or (w is not None and w.sum() <= 0):
        return np.nan
    if w is None:
        mb, vb = b.mean(), b.var(ddof=1)
    else:
        mb = np.average(b, weights=w)
        vb = np.average((b - mb) ** 2, weights=w)
    s = np.sqrt((a.var(ddof=1) + vb) / 2)
    return (a.mean() - mb) / s if s > 0 else np.nan


def match(sub, tier, covs, k=K_NEIGHBOURS, seed=0):
    """k nearest background genes per memory gene, with replacement.

    Returns (foreground frame, background frame, weights). Weights count how often
    each background gene was drawn, so a gene matched to several cases counts more.
    """
    need = covs
    fg = sub[(sub.tier == tier) & sub[need].notna().all(axis=1)]
    bg = sub[(sub.tier == "background") & sub[need].notna().all(axis=1)]
    if len(fg) < 20 or len(bg) < 50:
        return fg, bg.iloc[:0], np.array([])
    mu, sd = bg[need].mean(), bg[need].std(ddof=1).replace(0, 1)
    tree = cKDTree(((bg[need] - mu) / sd).to_numpy())
    _, idx = tree.query(((fg[need] - mu) / sd).to_numpy(), k=min(k, len(bg)))
    idx = np.atleast_2d(idx).ravel()
    cnt = np.bincount(idx, minlength=len(bg))
    keep = cnt > 0
    return fg, bg.iloc[keep], cnt[keep].astype(float)


def balance(fg, bg, w, covs):
    """Achieved |SMD| per covariate after matching."""
    return {c: smd(fg[c], bg[c], w) for c in covs}


K_LADDER = (10, 5, 3, 2, 1)


def match_balanced(sub, tier, covs, ks=K_LADDER):
    """Take the LARGEST k that achieves balance; fall back to the best available.

    Fewer neighbours means closer neighbours, so k trades precision against
    balance. Rather than fix k and hope, walk it down until the worst covariate
    |SMD| clears BALANCE_TOL. Where nothing clears it -- which happens where the
    background pool is genuinely too thin, LARRY and Harmange at top100 -- the best
    attempt is returned WITH its balance, so the caller can mark it failed instead
    of quietly reporting a biased estimate.
    """
    best = None
    for k in ks:
        fg, bg, w = match(sub, tier, covs, k=k)
        if not len(bg):
            continue
        bal = balance(fg, bg, w, covs)
        vals = [abs(v) for v in bal.values() if np.isfinite(v)]
        worst = max(vals) if vals else np.inf
        if best is None or worst < best[3]:
            best = (fg, bg, w, worst, k)
        if worst <= BALANCE_TOL:
            return fg, bg, w, worst, k
    return best if best else (sub.iloc[:0], sub.iloc[:0], np.array([]), np.inf, None)


def run(M, features, tiers=("top300", "top100"), covset="expr",
        datasets=None, k=K_NEIGHBOURS, verbose=True):
    """Matched effect size for every feature, per dataset per tier."""
    covs = COVSETS[covset]
    datasets = datasets or list(dict.fromkeys(M.dataset))
    rows = []
    for d in datasets:
        sub = M[M.dataset == d]
        for tier in tiers:
            fg, bg, w, worst, k_used = match_balanced(sub, tier, covs)
            if not len(bg):
                if verbose:
                    print(f"  {d:11s} {tier:7s} SKIPPED (pool too small)")
                continue
            bal = balance(fg, bg, w, covs)
            ok = worst <= BALANCE_TOL
            if verbose:
                b = "  ".join(f"{c}={bal[c]:+.3f}" for c in covs)
                print(f"  {d:11s} {tier:7s} n={len(fg):4d} bg={len(bg):5d} k={k_used}  "
                      f"{b}  worst={worst:.3f} "
                      f"{'OK' if ok else '<-- FAILED, not reportable'}")
            for f in features:
                if f not in sub.columns:
                    continue
                rows.append(dict(dataset=d, tier=tier, covset=covset, feature=f,
                                 n_fg=len(fg), n_bg=len(bg),
                                 smd_raw=smd(fg[f], sub.loc[sub.tier == "background", f]),
                                 smd_matched=smd(fg[f], bg[f], w),
                                 balance_worst=worst, balanced=ok, k=k_used))
    return pd.DataFrame(rows)
