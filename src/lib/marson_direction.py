"""
marson_direction.py - direction (signed-LFC) analysis of memory genes on the
Marson perturb-seq knockouts.

Separate from the cross-method benchmark (marson_MGs_benchmark): here we take the
genes ALL methods agree are memory genes and ask which WAY they move under
perturbation (up / down / no change), versus non-memory genes.

Main entry points:
  consensus_and_background(...)   -> genes called by all methods / no method
  plot_leaveout_kos(...)          -> responsiveness with the strongest KOs removed
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from marson_KO import (
    build_consensus_gene_sets, MEMGENE_DIR, METHOD_FILES,
    load_gmt, enrich_against_gmt,
)


def consensus_and_background(
    timepoint='merge', top_n=500, merge_mode='union',
    memgene_dir=MEMGENE_DIR, method_files=METHOD_FILES,
    universe=None, methods=None, verbose=True,
):
    """
    Genes called by ALL methods (consensus memory) and by NO method (non-memory).

    Returns (consensus_genes, nonmemory_genes, counts), where `counts` is the
    per-gene method-agreement count over `universe` (0 = called by no method).
    """
    methods = methods or list(method_files)
    n = len(methods)
    _, counts = build_consensus_gene_sets(
        min_methods=n, exactly=True, timepoint=timepoint, top_n=top_n,
        merge_mode=merge_mode, memgene_dir=memgene_dir, method_files=method_files,
        methods=methods, universe=universe, verbose=False,
    )
    consensus = counts.index[counts == n].tolist()
    nonmem = counts.index[counts == 0].tolist()
    if verbose:
        print(f"consensus (all {n} methods): {len(consensus)} genes | "
              f"non-memory (0 methods): {len(nonmem)} genes")
    return consensus, nonmem, counts


def plot_leaveout_kos(
    lfc_df, memory_genes, metric='signed', background_genes=None, gene_expr=None,
    n_bins=10, n_bg=2000, step=25, random_seed=0, fontsize=20, figsize=(11, 6.5),
    title=None, save_path=None, show=True,
):
    """
    Is the SCG mean driven by a few perturbations? Per KO, impact =
    mean |LFC| over the memory set; the most-impactful KOs are removed progressively.

    metric='signed' : plots MEAN SIGNED LFC (down-lean). Stays below 0 -> downward
                      lean not skewed by a few perturbations. (needs the down-direction
                      context to read.)
    metric='abs'    : plots MEAN |LFC| (responsiveness/magnitude), with a non-memory
                      reference line (from background_genes). SCG line staying above
                      non-memory -> higher responsiveness not driven by a few big KOs.
                      This is the version matching "where LFC is big for SCGs".
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    sub = lfc_df[mem]
    ranked = sub.abs().mean(axis=1).sort_values(ascending=False).index   # strongest KOs first
    ks = list(range(0, len(sub) - 50, step))

    bg_sub = None
    if metric == 'abs' and background_genes is not None:
        if gene_expr is not None:            # expression-matched non-memory reference
            pool = _match_expression(mem, background_genes, gene_expr, measured,
                                     n_bins=n_bins, rng=rng)
        else:
            pool = [g for g in background_genes if g in measured and g not in set(mem)]
            if len(pool) > n_bg:
                pool = list(rng.choice(pool, size=n_bg, replace=False))
        bg_sub = lfc_df[pool]

    mem_curve, bg_curve, signed = [], [], []
    for k in ks:
        keep = ranked[k:]
        mb = sub.loc[keep].values; mb = mb[np.isfinite(mb)]
        signed.append(mb.mean()); mem_curve.append(np.abs(mb).mean())
        if bg_sub is not None:
            bb = bg_sub.loc[keep].values; bb = bb[np.isfinite(bb)]
            bg_curve.append(np.abs(bb).mean())

    fig, ax = plt.subplots(figsize=figsize)
    if metric == 'abs':
        ax.plot(ks, mem_curve, '-o', color='crimson', ms=3, lw=2.5, label='Memory genes')
        if bg_sub is not None:
            _lab = ('Expression-matched non-memory' if gene_expr is not None
                    else 'Non-memory')
            ax.plot(ks, bg_curve, '-o', color='steelblue', ms=3, lw=2.5, label=_lab)
        ax.set_ylabel('mean |LFC| of (remaining) KOs', fontsize=fontsize)
    else:
        ax.plot(ks, signed, '-o', color='crimson', ms=3, lw=2.5, label='Memory genes')
        ax.axhline(0, color='k', ls='--', lw=1.5)
        ax.set_ylabel('mean signed LFC of (remaining) KOs', fontsize=fontsize)

    ax.set_xlabel('Most-impactful KOs removed (n)', fontsize=fontsize)
    ax.set_title(title or 'Memory-gene responsiveness is not driven by a few strong perturbations',
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 4, frameon=False)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    out = {'ko_removed': ks, 'mean_signed': signed, 'mean_abs': mem_curve}
    if bg_curve:
        out['bg_mean_abs'] = bg_curve
    return pd.DataFrame(out)


def _match_expression(mem, background_genes, gene_expr, measured, n_bins=10, rng=None, reps=1):
    """
    Return non-memory genes EXPRESSION-MATCHED to `mem`: bin all genes into n_bins
    baseline-expression quantiles, and per bin draw the same number of non-memory genes
    as there are memory genes. `reps` draws are concatenated (reps=1 = one matched set,
    size ~len(mem)). Used to make any background expression-matched.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    expr = gene_expr.reindex(measured).dropna()
    mem_e = [g for g in mem if g in expr.index]
    poolset = set(g for g in background_genes if g in measured and g not in set(mem))
    ebins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'), index=expr.index)
    per_bin = ebins.loc[mem_e].value_counts()
    binpool = {b: ebins.index[(ebins == b) & (ebins.index.isin(poolset))].to_numpy()
               for b in per_bin.index}
    out = []
    for _ in range(reps):
        for b, need in per_bin.items():
            out.extend(rng.choice(binpool[b], size=min(int(need), len(binpool[b])), replace=False))
    return out

