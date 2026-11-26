"""
marson_direction.py - direction (signed-LFC) analysis of memory genes on the
Marson perturb-seq knockouts.

Separate from the cross-method benchmark (marson_MGs_benchmark): here we take the
genes ALL methods agree are memory genes and ask which WAY they move under
perturbation (up / down / no change), versus non-memory genes.

Main entry points:
  consensus_and_background(...)          -> genes called by all methods / no method
  plot_mirrored_direction_density(...)   -> the split (mirrored) density figure
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


def _direction_values(lfc_df, genes, mode='pool'):
    """Signed-LFC values for `genes`, dropping NaNs (self-KO entries)."""
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    if mode == 'pool':
        v = lfc_df[cols].values.ravel()          # one value per (gene, KO)
    elif mode == 'per_ko':
        v = lfc_df[cols].mean(axis=1).values     # net direction per perturbation
    else:
        raise ValueError(f"mode must be 'pool' or 'per_ko', got {mode!r}")
    return v[np.isfinite(v)]


def direction_buckets(v, thr=0.05):
    """(% up > thr, % down < -thr, % near-zero |LFC| <= thr) for a value array."""
    return (100 * (v > thr).mean(),
            100 * (v < -thr).mean(),
            100 * (np.abs(v) <= thr).mean())


def plot_mirrored_direction_density(
    lfc_df, memory_genes, background_genes, mode='pool',
    gene_expr=None, n_bins=10, thr=0.05, clip=(-3, 3), grid=400, max_points=60000,
    mem_label='Memory', bg_label='Non-memory',
    mem_color='crimson', bg_color='steelblue',
    fontsize=18, figsize=(7, 9), title=None, save_path=None, show=True,
    random_seed=0,
):
    """
    Mirrored (split) density of signed LFC: non-memory fills the left half, memory
    the right, LFC on the y-axis, dashed line at 0. Each half is area-normalised, so
    the two groups compare on SHAPE despite very different gene counts.

    mode : 'pool'   -> one value per (gene, KO)  (spread of individual responses)
           'per_ko' -> mean over the set per KO  (net direction PER perturbation)
    gene_expr : if given, the non-memory side is EXPRESSION-MATCHED to the memory genes
                (binned into n_bins) instead of using all background genes.

    Annotations per side: % up (> thr), % near-zero (|LFC| <= thr), % down (< -thr).
    Returns a per-group stats DataFrame.
    """
    rng = np.random.default_rng(random_seed)
    if gene_expr is not None:
        background_genes = _match_expression(memory_genes, background_genes, gene_expr,
                                             lfc_df.columns, n_bins=n_bins, rng=rng)
    vm = _direction_values(lfc_df, memory_genes, mode)
    vb = _direction_values(lfc_df, background_genes, mode)

    ys = np.linspace(clip[0], clip[1], grid)

    def dens(v):
        s = v if v.size <= max_points else rng.choice(v, max_points, replace=False)
        return gaussian_kde(np.clip(s, *clip))(ys)

    dm, db = dens(vm), dens(vb)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_betweenx(ys, 0,  dm, color=mem_color, alpha=0.6,
                     label=f'{mem_label} (n={len(memory_genes)})')
    ax.fill_betweenx(ys, 0, -db, color=bg_color, alpha=0.6,
                     label=f'{bg_label} (n={len(background_genes)})')
    ax.axhline(0, color='k', ls='--', lw=2)               # LFC = 0 (no change)
    ax.axvline(0, color='k', lw=1)
    ax.axhline( thr, color='lightgray', lw=1.2, zorder=1)  # near-zero band
    ax.axhline(-thr, color='lightgray', lw=1.2, zorder=1)

    _t = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.02, thr + 0.02, f'<{thr:g}', transform=_t, ha='left', va='bottom',
            color='gray', fontsize=fontsize - 7)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.1f}'))
    ax.set_ylabel('LFC (per gene × KO)' if mode == 'pool' else 'LFC (net per KO)',
                  fontsize=fontsize)
    ax.set_xlabel('density', fontsize=fontsize)
    ax.set_ylim(*clip)
    ax.text(0.02, 0.98, bg_label, transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize - 3, color=bg_color, fontweight='bold')
    ax.text(0.98, 0.98, mem_label, transform=ax.transAxes, ha='right', va='top',
            fontsize=fontsize - 3, color=mem_color, fontweight='bold')
    ax.tick_params(labelsize=fontsize - 4)
    if title:
        ax.set_title(title, fontsize=fontsize - 2)

    # per-side quadrant percentages: up (top) / near-zero (mid) / down (bottom)
    box = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')

    def annot(x, v, color):
        up, dn, fl = direction_buckets(v, thr)
        ax.text(x, 0.83, f'{up:.0f}%', transform=ax.transAxes, ha='center',
                fontsize=fontsize - 3, color=color, fontweight='bold', bbox=box)
        ax.text(x, 0.50, f'{fl:.0f}%', transform=ax.transAxes, ha='center',
                fontsize=fontsize - 5, color='dimgray', fontweight='bold', bbox=box)
        ax.text(x, 0.17, f'{dn:.0f}%', transform=ax.transAxes, ha='center',
                fontsize=fontsize - 3, color=color, fontweight='bold', bbox=box)

    annot(0.22, vb, bg_color)   # non-memory (left)
    annot(0.78, vm, mem_color)  # memory (right)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mirrored direction density saved to: {save_path}")
    if show:
        plt.show()

    rows = []
    for lab, v in [(mem_label, vm), (bg_label, vb)]:
        up, dn, fl = direction_buckets(v, thr)
        rows.append({'group': lab, 'n_values': int(v.size),
                     'median_lfc': float(np.median(v)),
                     'pct_up': up, 'pct_near0': fl, 'pct_down': dn})
    return pd.DataFrame(rows).set_index('group')


# ---------------------------------------------------------------------------
# YG's question: do individual shared memory genes go the SAME direction
# consistently across perturbations, and do subsets move TOGETHER?
#   gene_direction_consistency  -> per-gene up/down consistency across KOs
#   plot_gene_codirection_clustermap -> gene-gene correlation across KOs, clustered
# ---------------------------------------------------------------------------
def gene_direction_consistency(lfc_df, genes, thr=0.0):
    """
    Per-gene directional consistency across all KOs (signed LFC).

    Returns a DataFrame indexed by gene, sorted by mean_signed_lfc:
      n_obs           : KOs contributing (self-KO NaNs excluded)
      frac_up         : fraction of KOs with LFC >  thr
      frac_down       : fraction of KOs with LFC < -thr
      mean_signed_lfc : average signed LFC across KOs
      consistency     : max(frac_up, frac_down) -> 0.5 = coin flip, 1 = always one way
      direction       : 'up'/'down' by the sign of the mean
    """
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    sub = lfc_df[cols]
    n = sub.notna().sum(axis=0)
    up = (sub > thr).sum(axis=0)
    dn = (sub < -thr).sum(axis=0)
    out = pd.DataFrame({
        'n_obs': n.astype(int),
        'frac_up': up / n,
        'frac_down': dn / n,
        'mean_signed_lfc': sub.mean(axis=0),
    })
    out['consistency'] = out[['frac_up', 'frac_down']].max(axis=1)
    out['direction'] = np.where(out['mean_signed_lfc'] >= 0, 'up', 'down')
    return out.sort_values('mean_signed_lfc')


def plot_gene_codirection_clustermap(
    lfc_df, genes, min_periods=100, cluster_method='average', n_clusters=None,
    show_direction=True, dir_vmax=0.5, dir_cmap='PRGn',
    cmap='RdBu_r', figsize=(13, 13), fontsize=6, title=None,
    save_path=None, show=True,
):
    """
    Cluster shared memory genes by whether they MOVE TOGETHER across perturbations.

    Each gene is its vector of signed LFC over all KOs; we correlate every gene pair
    (Pearson, pairwise-complete obs) and cluster the correlation matrix. Red blocks
    on the diagonal = groups of genes that consistently move the SAME direction
    together; blue = anti-correlated (one up while the other goes down).

    NOTE: correlation removes each gene's mean, so the heatmap shows CO-MOVEMENT,
    not up/down. Set show_direction=True to add a side strip coloured by each gene's
    mean signed LFC (red=up, blue=down) so you can tell whether a co-moving module
    is an up-group or a down-group.

    Returns (corr, modules, clustergrid):
      corr    : gene x gene correlation DataFrame
      modules : Series of cluster labels per gene (None unless n_clusters is set)
      grid    : the seaborn ClusterGrid
    """
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    sub = lfc_df[cols]
    corr = sub.corr(min_periods=min_periods).fillna(0.0)

    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0                      # enforce symmetry for squareform
    Z = linkage(squareform(dist, checks=False), method=cluster_method)

    # optional side strip: each gene's mean signed LFC (direction) as a colour
    row_colors = None
    if show_direction:
        mean_signed = sub.mean(axis=0).reindex(corr.index)
        norm = plt.Normalize(-dir_vmax, dir_vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=dir_cmap)
        row_colors = pd.Series(
            [sm.to_rgba(v) for v in mean_signed.values],
            index=corr.index, name='mean LFC (up=green, down=purple)',
        )

    grid = sns.clustermap(
        corr, row_linkage=Z, col_linkage=Z, cmap=cmap, center=0,
        vmin=-1, vmax=1, figsize=figsize,
        xticklabels=True, yticklabels=True,
        row_colors=row_colors,
        cbar_kws={'label': 'Pearson r across KOs'},
    )
    grid.ax_heatmap.tick_params(labelsize=fontsize)

    # second colorbar for the direction strip (distinct from the correlation one)
    if show_direction:
        sm.set_array([])
        pos = grid.ax_cbar.get_position()
        cax = grid.figure.add_axes([pos.x0, pos.y0 - 0.13, pos.width, 0.02])
        cb = grid.figure.colorbar(sm, cax=cax, orientation='horizontal',
                                  extend='both')
        cb.set_label('mean LFC per gene\n(green=up, purple=down)',
                     fontsize=fontsize + 2)
        cb.ax.tick_params(labelsize=fontsize)

    if title:
        grid.figure.suptitle(title, fontsize=fontsize + 10, y=1.02)

    modules = None
    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        modules = pd.Series(labels, index=corr.index, name='module').sort_values()

    if save_path:
        grid.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Co-direction clustermap saved to: {save_path}")
    if show:
        plt.show()
    return corr, modules, grid


def _mean_offdiag_corr(mat):
    """Mean off-diagonal Pearson r for a KOs x genes matrix (NaNs filled per column)."""
    M = np.asarray(mat, dtype=float).copy()
    col_mean = np.nanmean(M, axis=0)
    nan_i, nan_j = np.where(np.isnan(M))
    M[nan_i, nan_j] = np.take(col_mean, nan_j)      # fill self-KO NaNs with gene mean
    c = np.corrcoef(M, rowvar=False)
    iu = np.triu_indices(c.shape[0], k=1)
    return float(np.nanmean(c[iu]))


def module_direction_summary(lfc_df, genes, modules, min_periods=100):
    """
    Per co-movement module: size, up/down counts, mean direction, mean per-gene
    consistency, and within-module mean correlation (how tightly the module co-moves).

    `modules` is the Series returned by plot_gene_codirection_clustermap(n_clusters=..).
    """
    cons = gene_direction_consistency(lfc_df, genes)
    tab = cons.join(modules)
    corr = lfc_df[[g for g in genes if g in lfc_df.columns]].corr(min_periods=min_periods)

    rows = []
    for m, sub in tab.groupby('module'):
        gs = [g for g in sub.index if g in corr.index]
        if len(gs) > 1:
            cm = corr.loc[gs, gs].values
            iu = np.triu_indices(len(gs), k=1)
            within_r = float(np.nanmean(cm[iu]))
        else:
            within_r = np.nan
        rows.append({
            'module': m, 'n': len(sub),
            'n_up': int((sub['direction'] == 'up').sum()),
            'n_down': int((sub['direction'] == 'down').sum()),
            'mean_lfc': float(sub['mean_signed_lfc'].mean()),
            'mean_consistency': float(sub['consistency'].mean()),
            'within_module_r': within_r,
        })
    return pd.DataFrame(rows).set_index('module')


def codirection_vs_random(
    lfc_df, genes, universe=None, gene_expr=None, n_bins=10, n_iter=200, random_seed=0,
):
    """
    Is the gene set's CO-MOVEMENT above chance? Compares the set's mean pairwise
    correlation across KOs against same-size random sets drawn from `universe`
    (genes in `genes` are excluded from the pool).

    gene_expr : if given, the null sets are EXPRESSION-MATCHED to `genes` (binned into
                n_bins) instead of drawn uniformly at random.

    Returns dict: observed_mean_r, null_mean_r, null_std_r, z, p_empirical (two-sided),
    and the null array. A high positive z means the genes co-move more than random.
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    genes = [g for g in genes if g in measured]
    k = len(genes)

    obs = _mean_offdiag_corr(lfc_df[genes].values)

    universe = list(measured) if universe is None else list(universe)
    pool = np.array([g for g in universe if g in measured and g not in set(genes)])

    null = np.empty(n_iter)
    for i in range(n_iter):
        if gene_expr is not None:
            samp = _match_expression(genes, universe, gene_expr, measured, n_bins=n_bins, rng=rng)
        else:
            samp = list(rng.choice(pool, size=k, replace=False))
        null[i] = _mean_offdiag_corr(lfc_df[samp].values)

    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 0 else np.nan
    p = float((np.abs(null - mu) >= abs(obs - mu)).mean())
    return {'n_genes': k, 'observed_mean_r': obs, 'null_mean_r': mu,
            'null_std_r': sd, 'z': z, 'p_empirical': p, 'null': null}


def plot_direction_consistency_scatter(
    lfc_df, genes, thr=0.0, consistency_line=0.8, dir_vmax=0.5,
    label_n=5, symmetric_x=True, xpad=0.05, fontsize=14, figsize=(10, 7.5),
    title=None, save_path=None, show=True,
):
    """
    One point per gene: x = mean signed LFC (direction), y = consistency (fraction
    of KOs in the gene's majority direction). Colour = mean LFC (green=up, purple=
    down), matching the clustermap strip. This is the direct answer to "do some
    genes always go the same direction, and which way?" -- a cloud far left/high =
    consistently down, far right/high = consistently up, low y = flip-floppy.

    Labels the `label_n` most EXTREME genes on each side (most negative and most
    positive mean signed LFC) -- the outliers, not the central blob.
    symmetric_x mirrors the x-limits around 0 so the (often empty) positive side is
    shown. Returns the per-gene consistency table.
    """
    cons = gene_direction_consistency(lfc_df, genes, thr=thr)
    x = cons['mean_signed_lfc'].values
    y = cons['consistency'].values

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=x, cmap='PRGn', vmin=-dir_vmax, vmax=dir_vmax,
                    s=45, edgecolor='k', linewidth=0.3, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('mean LFC (up=green, down=purple)', fontsize=fontsize - 1)

    if symmetric_x:
        m = np.nanmax(np.abs(x)) * (1 + xpad)
        ax.set_xlim(-m, m)

    ax.axvline(0, color='k', lw=1)
    ax.axhline(consistency_line, color='gray', ls='--', lw=1.5)
    ax.text(ax.get_xlim()[1], consistency_line, f'  {int(consistency_line*100)}% consistent',
            va='bottom', ha='right', color='gray', fontsize=fontsize - 4)

    # region hints
    ax.text(0.02, 0.98, 'consistently\nDOWN', transform=ax.transAxes, ha='left',
            va='top', color='purple', fontweight='bold', fontsize=fontsize)
    ax.text(0.98, 0.98, 'consistently\nUP', transform=ax.transAxes, ha='right',
            va='top', color='green', fontweight='bold', fontsize=fontsize)

    # label the most EXTREME genes: furthest left (most negative) and furthest right
    most_neg = cons.nsmallest(label_n, 'mean_signed_lfc')
    most_pos = cons.nlargest(label_n, 'mean_signed_lfc')
    for g, r in pd.concat([most_neg, most_pos]).iterrows():
        ax.annotate(g, (r['mean_signed_lfc'], r['consistency']),
                    fontsize=fontsize - 5, fontweight='bold', xytext=(4, 3),
                    textcoords='offset points', zorder=4)

    n_cons = int((cons['consistency'] >= consistency_line).sum())
    n_up = int(((cons['consistency'] >= consistency_line) & (cons['direction'] == 'up')).sum())
    n_dn = n_cons - n_up
    ax.set_xlabel('mean LFC across KOs  (direction)', fontsize=fontsize)
    ax.set_ylabel('consistency  (fraction of KOs in majority direction)', fontsize=fontsize)
    ax.set_title(title or (f'Per-gene direction consistency (n={len(cons)})\n'
                           f'{n_cons} genes >= {int(consistency_line*100)}% consistent '
                           f'({n_dn} down, {n_up} up)'), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Direction-consistency scatter saved to: {save_path}")
    if show:
        plt.show()
    return cons


# ---------------------------------------------------------------------------
# Flip the view: which KNOCKOUTS drive the memory set, and do many memory genes
# react to the SAME knockouts (broad regulators)?
#   rank_ko_drivers        -> per-KO breadth/strength/direction over the memory set
#   plot_ko_driver_overview-> how broad each KO is + top driver KOs
#   plot_ko_scg_heatmap    -> top-driver KOs x memory genes, to see the agreement
# ---------------------------------------------------------------------------
def rank_ko_drivers(lfc_df, genes, thr=0.5):
    """
    For each knockout (row), summarise its effect on the memory-gene set (columns):

      n_genes          : memory genes measured in that KO
      n_react          : memory genes with |LFC| > thr (a 'reaction')
      frac_react       : n_react / n_genes  (how BROADLY the KO hits the set)
      mean_abs_lfc     : mean |LFC| over the set (how STRONGLY it hits)
      mean_signed_lfc  : net direction over the set
      frac_up_of_react : of the reacting genes, fraction that went up
      direction        : 'up'/'down' by sign of mean_signed_lfc

    Sorted by frac_react (broadest drivers first).
    """
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    sub = lfc_df[cols]
    absv = sub.abs()
    n = sub.notna().sum(axis=1)
    n_react = (absv > thr).sum(axis=1)
    up_react = (sub > thr).sum(axis=1)
    out = pd.DataFrame({
        'n_genes': n,
        'n_react': n_react,
        'frac_react': n_react / n.replace(0, np.nan),
        'mean_abs_lfc': absv.mean(axis=1),
        'mean_signed_lfc': sub.mean(axis=1),
        'frac_up_of_react': up_react / n_react.replace(0, np.nan),
    })
    out['direction'] = np.where(out['mean_signed_lfc'] >= 0, 'up', 'down')
    return out.sort_values('frac_react', ascending=False)


def plot_ko_driver_overview(
    drivers, top_n=25, thr=0.5, fontsize=14, figsize=(16, 7),
    title=None, save_path=None, show=True,
):
    """
    Left: distribution of frac_react across ALL KOs -- are there a few broad drivers
    (long right tail) or does every KO hit a similar small fraction?
    Right: the top_n broadest driver KOs, bars coloured by net direction
    (purple=down, green=up). Answers 'which KO genes drive the set, and how many'.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    fr = drivers['frac_react'].dropna() * 100
    ax.hist(fr, bins=40, color='steelblue', alpha=0.85)
    ax.set_xlabel(f'% of memory genes reacting (|LFC| > {thr}) per KO', fontsize=fontsize)
    ax.set_ylabel('number of KOs', fontsize=fontsize)
    ax.set_title('Breadth of each KO’s effect on the memory set', fontsize=fontsize + 1)
    for q in (25, 50):
        nko = int((fr >= q).sum())
        ax.axvline(q, color='gray', ls=':', lw=1)
        ax.text(q, ax.get_ylim()[1] * 0.9, f' {nko} KOs\n hit >={q}%',
                fontsize=fontsize - 4, color='dimgray', va='top')
    ax.tick_params(labelsize=fontsize - 3)

    ax = axes[1]
    top = drivers.head(top_n).iloc[::-1]
    colors = ['green' if d == 'up' else 'purple' for d in top['direction']]
    ax.barh(range(len(top)), top['frac_react'].values * 100, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=fontsize - 4)
    ax.set_xlabel('% of memory genes reacting', fontsize=fontsize)
    ax.set_title(f'Top {top_n} KO drivers (purple=down, green=up)', fontsize=fontsize + 1)
    ax.tick_params(axis='x', labelsize=fontsize - 3)

    if title:
        fig.suptitle(title, fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KO-driver overview saved to: {save_path}")
    if show:
        plt.show()


def plot_ko_scg_heatmap(
    lfc_df, genes, drivers=None, top_n=30, thr=0.5, vmax=2.0,
    fontsize=6, figsize=(16, 10), title=None, save_path=None, show=True,
):
    """
    Heatmap of the top_n broadest-driver KOs (rows) x memory genes (columns), signed
    LFC. Columns are clustered so genes that react together sit together: a vertical
    red/blue stripe = a memory gene that reacts across MANY driver KOs; a shared
    block = a set of memory genes that agree in responding to the same knockouts.
    Returns the (KO x gene) matrix shown.
    """
    if drivers is None:
        drivers = rank_ko_drivers(lfc_df, genes, thr=thr)
    top_kos = drivers.head(top_n).index
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    mat = lfc_df.loc[top_kos, cols]

    grid = sns.clustermap(
        mat.fillna(0.0), cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
        row_cluster=True, col_cluster=True, figsize=figsize,
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'signed LFC'},
    )
    grid.ax_heatmap.tick_params(labelsize=fontsize)
    grid.ax_heatmap.set_xlabel('memory genes')
    grid.ax_heatmap.set_ylabel('driver KOs')
    if title:
        grid.figure.suptitle(title, fontsize=fontsize + 12, y=1.02)
    if save_path:
        grid.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KO x SCG heatmap saved to: {save_path}")
    if show:
        plt.show()
    return mat


def ko_driver_specificity(lfc_df, genes, thr=0.5):
    """
    Per-KO: does the knockout hit the MEMORY set more than expected given how broadly
    it hits the whole genome? For each KO compares memory reaction rate to the
    genome-wide rate with a hypergeometric test.

    Columns:
      frac_react_memory : memory genes reacting (|LFC|>thr) / memory genes measured
      frac_react_global : ALL genes reacting / all genes measured (KO's overall breadth)
      enrichment        : observed / expected memory reactors  (1 = like the genome)
      hyperg_p          : one-sided P(memory reactors >= observed) under the genome rate
    Sorted by enrichment (most memory-specific first).
    """
    from scipy.stats import hypergeom

    mem_cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    react = (lfc_df > thr) | (lfc_df < -thr)        # |LFC|>thr, NaN -> False

    N = lfc_df.notna().sum(axis=1)                  # measured genes per KO
    K = react.sum(axis=1)                           # genome-wide reactors per KO
    m = lfc_df[mem_cols].notna().sum(axis=1)        # memory genes measured
    k = react[mem_cols].sum(axis=1)                 # memory reactors

    exp = m * K / N.replace(0, np.nan)              # expected memory reactors
    out = pd.DataFrame({
        'frac_react_memory': k / m.replace(0, np.nan),
        'frac_react_global': K / N.replace(0, np.nan),
        'enrichment': k / exp.replace(0, np.nan),
        'hyperg_p': [hypergeom.sf(int(kk) - 1, int(NN), int(KK), int(mm))
                     if mm > 0 and KK > 0 else np.nan
                     for kk, NN, KK, mm in zip(k, N, K, m)],
    })
    return out.sort_values('enrichment', ascending=False)


def plot_ko_driver_specificity(
    spec, drivers=None, top_n_label=15, baseline_col='frac_react_global',
    baseline_label='% of ALL genes reacting per KO',
    fontsize=13, figsize=(9, 8), title=None, save_path=None, show=True,
):
    """
    Scatter: x = baseline reaction rate per KO, y = % of MEMORY genes reacting.
    The dashed identity line = 'memory reacts just like the baseline' (no specificity).
    Points ABOVE = KOs that hit memory genes MORE than expected (memory-specific);
    ON the line = generic breadth. Colour = log2 enrichment. If `drivers` is given,
    the top_n_label broadest drivers are labelled.

    baseline_col : 'frac_react_global' (vs whole genome) or 'frac_react_matched'
                   (vs expression-matched random genes).
    """
    x = spec[baseline_col] * 100
    y = spec['frac_react_memory'] * 100
    c = np.log2(spec['enrichment'].replace([np.inf, -np.inf], np.nan))

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=c, cmap='coolwarm', vmin=-1, vmax=1,
                    s=16, alpha=0.6, zorder=2)
    cb = fig.colorbar(sc); cb.set_label('log2 enrichment (memory vs expected)',
                                        fontsize=fontsize - 1)
    lim = float(np.nanmax([x.max(), y.max()])) * 1.02
    ax.plot([0, lim], [0, lim], 'k--', lw=1.5, zorder=1,
            label='memory = baseline (no specificity)')

    if drivers is not None:
        for ko in drivers.head(top_n_label).index:
            if ko in spec.index:
                ax.annotate(ko, (x[ko], y[ko]), fontsize=fontsize - 5,
                            fontweight='bold', xytext=(3, 2),
                            textcoords='offset points', zorder=3)

    ax.set_xlabel(baseline_label, fontsize=fontsize)
    ax.set_ylabel('% of MEMORY genes reacting per KO', fontsize=fontsize)
    ax.set_title(title or 'Per-KO reaction rate: memory vs baseline',
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KO-driver specificity saved to: {save_path}")
    if show:
        plt.show()
    return spec


def ko_driver_specificity_exprmatched(
    lfc_df, genes, gene_expr, thr=0.5, n_bins=10, n_iter=50, random_seed=0,
):
    """
    Expression-matched version of ko_driver_specificity: for each KO, compares the
    memory set's reaction rate to random gene sets MATCHED for baseline expression
    (so 'memory reacts more' can't just be 'memory is higher-expressed').

    Genes are binned into n_bins expression quantiles; each of n_iter background sets
    draws, per bin, the same number of NON-memory genes as there are memory genes in
    that bin. Per-KO frac_react is averaged over the draws.

    Returns per-KO DataFrame:
      frac_react_memory  : memory reactors / memory measured
      frac_react_matched : mean reactors / measured over the expr-matched sets
      enrichment         : frac_react_memory / frac_react_matched (1 = no specificity)
    Sorted by enrichment.
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    expr = gene_expr.reindex(measured).dropna()

    mem = [g for g in genes if g in expr.index]
    mem_set = set(mem)
    bins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'),
                     index=expr.index)

    # per bin: how many memory genes, and the non-memory pool to match from
    mem_per_bin = bins.loc[mem].value_counts()
    pool_per_bin = {b: bins.index[(bins == b) & (~bins.index.isin(mem_set))].to_numpy()
                    for b in mem_per_bin.index}

    react = (lfc_df > thr) | (lfc_df < -thr)            # |LFC|>thr, NaN -> False
    notna = lfc_df.notna()

    def frac_react(cols):
        cols = list(cols)
        return react[cols].sum(axis=1) / notna[cols].sum(axis=1).replace(0, np.nan)

    fr_mem = frac_react(mem)

    fr_bg_sum = np.zeros(len(lfc_df))
    for _ in range(n_iter):
        samp = []
        for b, need in mem_per_bin.items():
            pool = pool_per_bin[b]
            take = min(need, len(pool))
            samp.extend(rng.choice(pool, size=take, replace=False))
        fr_bg_sum += frac_react(samp).values
    fr_bg = pd.Series(fr_bg_sum / n_iter, index=lfc_df.index)

    out = pd.DataFrame({
        'frac_react_memory': fr_mem,
        'frac_react_matched': fr_bg,
        'enrichment': fr_mem / fr_bg.replace(0, np.nan),
    })
    return out.sort_values('enrichment', ascending=False)


def plot_ko_specificity_comparison(
    spec_global, spec_em, drivers=None, top_n_label=12,
    fontsize=13, figsize=(16, 8), title=None, save_path=None, show=True,
):
    """
    Side-by-side specificity scatters: LEFT = memory vs whole genome, RIGHT = memory
    vs expression-matched random genes. Shared axes so the drop toward the identity
    line (= how much of the enrichment was just expression) is visible at a glance.
    Each panel's title shows its median enrichment.
    """
    panels = [
        (spec_global, 'frac_react_global',  '% of ALL genes reacting per KO',        'vs whole genome'),
        (spec_em,     'frac_react_matched', '% of EXPRESSION-MATCHED genes reacting', 'vs expression-matched'),
    ]

    lim = 0.0
    for spec, col, _, _ in panels:
        lim = max(lim, float((spec[col] * 100).max()), float((spec['frac_react_memory'] * 100).max()))
    lim *= 1.02

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    sc = None
    for ax, (spec, col, xlabel, sub) in zip(axes, panels):
        x = spec[col] * 100
        y = spec['frac_react_memory'] * 100
        c = np.log2(spec['enrichment'].replace([np.inf, -np.inf], np.nan))
        sc = ax.scatter(x, y, c=c, cmap='coolwarm', vmin=-1, vmax=1, s=14, alpha=0.6, zorder=2)
        ax.plot([0, lim], [0, lim], 'k--', lw=1.5, zorder=1)
        ax.set_title(f'{sub}\nmedian enrichment = {spec["enrichment"].median():.2f}',
                     fontsize=fontsize + 1)
        ax.set_xlabel(xlabel, fontsize=fontsize - 1)
        if drivers is not None:
            for ko in drivers.head(top_n_label).index:
                if ko in spec.index:
                    ax.annotate(ko, (x[ko], y[ko]), fontsize=fontsize - 6, fontweight='bold',
                                xytext=(3, 2), textcoords='offset points', zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.tick_params(labelsize=fontsize - 3)

    axes[0].set_ylabel('% of MEMORY genes reacting per KO', fontsize=fontsize)
    fig.colorbar(sc, ax=list(axes), shrink=0.8,
                 label='log2 enrichment (memory vs baseline)')
    if title:
        fig.suptitle(title, fontsize=fontsize + 4)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Specificity comparison saved to: {save_path}")
    if show:
        plt.show()


def ko_specificity_volcano_stats(
    lfc_df, genes, gene_expr, thr=0.5, n_bins=10, n_iter=100, random_seed=0,
):
    """
    Per-KO enrichment of memory reaction vs an expression-matched null, WITH a
    significance estimate. For each KO we keep the mean and std of frac_react over
    n_iter expression-matched background draws, then z-score the observed memory
    frac_react against that null (normal two-sided p).

    Returns per-KO DataFrame (sorted by z):
      frac_react_memory, matched_mean, matched_std,
      log2_enrichment = log2(memory / matched_mean),
      z, p, neglog10p
    """
    from scipy.stats import norm

    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    expr = gene_expr.reindex(measured).dropna()

    mem = [g for g in genes if g in expr.index]
    mem_set = set(mem)
    bins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'), index=expr.index)
    mem_per_bin = bins.loc[mem].value_counts()
    pool_per_bin = {b: bins.index[(bins == b) & (~bins.index.isin(mem_set))].to_numpy()
                    for b in mem_per_bin.index}

    react = (lfc_df > thr) | (lfc_df < -thr)
    notna = lfc_df.notna()

    def frac_react(cols):
        cols = list(cols)
        return (react[cols].sum(axis=1) / notna[cols].sum(axis=1).replace(0, np.nan)).values

    fr_mem = frac_react(mem)
    s = np.zeros(len(lfc_df)); s2 = np.zeros(len(lfc_df))
    for _ in range(n_iter):
        samp = []
        for b, need in mem_per_bin.items():
            pool = pool_per_bin[b]
            samp.extend(rng.choice(pool, size=min(need, len(pool)), replace=False))
        fr = frac_react(samp)
        s += fr; s2 += fr * fr

    mean = s / n_iter
    std = np.sqrt(np.clip(s2 / n_iter - mean ** 2, 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (fr_mem - mean) / np.where(std > 0, std, np.nan)
        log2e = np.log2(fr_mem / np.where(mean > 0, mean, np.nan))
    p = 2 * norm.sf(np.abs(z))

    out = pd.DataFrame({
        'frac_react_memory': fr_mem, 'matched_mean': mean, 'matched_std': std,
        'log2_enrichment': log2e, 'z': z, 'p': p,
        'neglog10p': -np.log10(np.clip(p, 1e-300, None)),
    }, index=lfc_df.index)
    return out.sort_values('z', ascending=False)


def plot_ko_specificity_volcano(
    vol, top_n_label=15, alpha_bonf=0.05, fontsize=13, figsize=(10, 8),
    title=None, save_path=None, show=True,
):
    """
    Volcano of per-KO memory specificity: x = log2 enrichment (memory vs expression-
    matched), y = -log10 p (vs the matched null). A memory-specific DRIVER SET appears
    as a separated cluster far to the RIGHT; a uniform offset appears as one blob
    centred at the median log2 enrichment. Bonferroni line + median line drawn;
    the top_n_label most enriched significant KOs are labelled.
    """
    x = vol['log2_enrichment']; y = vol['neglog10p']
    p_thr = alpha_bonf / len(vol)
    sig = vol['p'] < p_thr
    med = float(np.nanmedian(x))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[~sig], y[~sig], s=10, color='lightgray', alpha=0.5, zorder=1, label='n.s.')
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s=16, color='crimson', alpha=0.6,
               zorder=2, label='memory-enriched')
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s=16, color='steelblue', alpha=0.6,
               zorder=2, label='memory-depleted')
    ax.axvline(0, color='k', lw=1)
    ax.axvline(med, color='green', ls=':', lw=1.5, label=f'median log2 = {med:+.2f}')
    ax.axhline(-np.log10(p_thr), color='gray', ls='--', lw=1,
               label=f'Bonferroni p<{p_thr:.0e}')

    lab = vol[sig & (x > 0)].sort_values('log2_enrichment', ascending=False).head(top_n_label)
    for ko, r in lab.iterrows():
        ax.annotate(ko, (r['log2_enrichment'], r['neglog10p']), fontsize=fontsize - 6,
                    fontweight='bold', xytext=(3, 2), textcoords='offset points', zorder=3)

    ax.set_xlabel('log2 enrichment (memory vs expression-matched)', fontsize=fontsize)
    ax.set_ylabel('-log10 p  (vs matched null)', fontsize=fontsize)
    ax.set_title(title or 'Per-KO memory specificity (volcano)', fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 4, loc='upper left')
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Specificity volcano saved to: {save_path}")
    if show:
        plt.show()
    return vol


def plot_gene_net_direction(
    lfc_df, genes, thr=0.0, consistent_at=0.6, label_n=8,
    fontsize=15, figsize=(13, 6.5), title=None, save_path=None, show=True,
):
    """
    Ranked 'waterfall' of per-gene net direction across all KOs -- the intuitive
    answer to 'do genes go the same direction consistently?'

    For each gene, net = frac_up - frac_down over all KOs:
        +1 = up in EVERY knockout, -1 = down in every knockout, 0 = half-and-half.
    Genes are sorted by net; bars coloured green (up) / purple (down). Dotted lines
    mark the +/- consistent_at 'consistently one direction' threshold. The title
    counts how many genes are consistently down vs up. A few extreme genes labelled.
    """
    cons = gene_direction_consistency(lfc_df, genes, thr=thr)
    net = (cons['frac_up'] - cons['frac_down']).sort_values()
    vals = net.values
    colors = ['green' if v > 0 else 'purple' for v in vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(vals)), vals, width=1.0, color=colors, linewidth=0)
    ax.axhline(0, color='k', lw=1)
    ax.axhline( consistent_at, color='green', ls=':', lw=1.3)
    ax.axhline(-consistent_at, color='purple', ls=':', lw=1.3)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, len(vals))

    n_down = int((vals <= -consistent_at).sum())
    n_up = int((vals >= consistent_at).sum())
    ax.text(0.01, 0.06, f'consistently DOWN\n({n_down} genes)', transform=ax.transAxes,
            color='purple', fontweight='bold', fontsize=fontsize - 2, va='bottom')
    ax.text(0.99, 0.94, f'consistently UP\n({n_up} genes)', transform=ax.transAxes,
            color='green', fontweight='bold', fontsize=fontsize - 2, ha='right', va='top')

    # label the most extreme genes on each end
    for i in range(min(label_n, len(vals))):
        ax.text(i, vals[i] - 0.02, net.index[i], rotation=90, ha='center', va='top',
                fontsize=fontsize - 6, fontweight='bold')
    for i in range(max(0, len(vals) - label_n), len(vals)):
        if vals[i] > 0:
            ax.text(i, vals[i] + 0.02, net.index[i], rotation=90, ha='center',
                    va='bottom', fontsize=fontsize - 6, fontweight='bold')

    ax.set_xlabel('genes (sorted by net direction)', fontsize=fontsize)
    ax.set_ylabel('net direction\n(-1 = always down,  +1 = always up)', fontsize=fontsize)
    ax.set_title(title or f'Direction of each shared memory gene across KOs '
                          f'(n={len(net)})', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize - 3)
    ax.set_xticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Net-direction waterfall saved to: {save_path}")
    if show:
        plt.show()
    return net


def plot_codirection_per_ko(
    lfc_df, genes, background_pool, direction='down', thr=0.0, n_iter=50,
    gene_expr=None, n_expr_bins=10, random_seed=0, bins=40, fontsize=15,
    figsize=(11, 6), title=None, save_path=None, show=True,
):
    """
    Intuitive 'do they move together?' plot. For each knockout, compute the % of the
    memory genes that go `direction` (down or up). If the genes move together, most
    knockouts push a LARGE share the same way (histogram piled to the right). A baseline
    set (grey, averaged over n_iter draws) is overlaid: if memory looks like it, the
    together-ness is just the global trend, not special.

    gene_expr : if given (per-gene baseline expression), the grey baseline is drawn
        EXPRESSION-MATCHED to the memory genes (binned into n_expr_bins), controlling
        for memory genes being highly expressed. If None, it's a plain random set.

    Returns (mem_frac_per_ko, bg_frac_per_ko).
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    mem = [g for g in genes if g in measured]

    def frac_dir(cols):
        sub = lfc_df[list(cols)]
        n = sub.notna().sum(axis=1).replace(0, np.nan)
        cnt = (sub < -thr).sum(axis=1) if direction == 'down' else (sub > thr).sum(axis=1)
        return (cnt / n).values * 100

    mem_frac = frac_dir(mem)
    pool_all = [g for g in background_pool if g in measured and g not in set(mem)]

    if gene_expr is not None:
        expr = gene_expr.reindex(measured).dropna()
        mem_e = [g for g in mem if g in expr.index]
        ebins = pd.Series(pd.qcut(expr, n_expr_bins, labels=False, duplicates='drop'),
                          index=expr.index)
        mem_per_bin = ebins.loc[mem_e].value_counts()
        pool_set = set(pool_all)
        pool_per_bin = {b: ebins.index[(ebins == b) & (ebins.index.isin(pool_set))].to_numpy()
                        for b in mem_per_bin.index}
        def draw():
            samp = []
            for b, need in mem_per_bin.items():
                p = pool_per_bin[b]
                samp.extend(rng.choice(p, size=min(need, len(p)), replace=False))
            return samp
        bg_label = 'expression-matched set'
    else:
        pool = np.array(pool_all)
        def draw():
            return rng.choice(pool, size=len(mem), replace=False)
        bg_label = 'random matched-size set'

    bg = np.zeros(len(lfc_df))
    for _ in range(n_iter):
        bg += frac_dir(draw())
    bg /= n_iter

    col = 'purple' if direction == 'down' else 'green'
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(mem_frac, bins=bins, color=col, alpha=0.6,
            label=f'memory genes (n={len(mem)})')
    ax.hist(bg, bins=bins, color='lightgray', alpha=0.75, label=bg_label)
    ax.axvline(50, color='k', ls='--', lw=1.5)
    ax.text(50, ax.get_ylim()[1] * 0.97, ' half', va='top', color='gray', fontsize=fontsize - 4)

    mm, mb = float(np.nanmedian(mem_frac)), float(np.nanmedian(bg))
    ax.axvline(mm, color=col, lw=2); ax.axvline(mb, color='gray', lw=2)
    ax.set_xlabel(f'% of genes going {direction.upper()} per knockout', fontsize=fontsize)
    ax.set_ylabel('number of knockouts', fontsize=fontsize)
    ax.set_title(title or (f'% of memory genes going {direction} per knockout\n'
                           f'median: memory {mm:.0f}%  vs  baseline {mb:.0f}%'),
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Co-direction per-KO figure saved to: {save_path}")
    if show:
        plt.show()
    return mem_frac, bg


def plot_memory_ko_heatmap(
    lfc_df, genes, vmax=2.0, cluster_rows=True, sort_cols=True,
    figsize=(15, 9), fontsize=7, title=None, save_path=None, show=True,
):
    """
    The raw picture: memory genes (rows) x ALL perturbations (columns), coloured by
    signed LFC (red = up, blue = down). Shows directly, per gene and per knockout,
    how up/down each response is. Rows optionally clustered (co-behaving genes group);
    columns optionally sorted by their mean effect on the memory set.

    Expect: mostly blue with red (up) events SCATTERED throughout -- up happens
    everywhere but as a minority, with no solid-red row (no consistently-up gene).
    Returns the ordered (genes x KOs) matrix.
    """
    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    matf = lfc_df[cols].T.fillna(0.0)                     # genes x KOs

    if cluster_rows:
        from scipy.cluster.hierarchy import linkage, leaves_list
        Z = linkage(matf.values, method='average')
        matf = matf.iloc[leaves_list(Z)]
    else:
        matf = matf.loc[matf.mean(axis=1).sort_values().index]
    if sort_cols:
        matf = matf[matf.mean(axis=0).sort_values().index]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matf.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    fig.colorbar(im, ax=ax, label='signed LFC  (red = up, blue = down)')
    ax.set_yticks(range(len(matf)))
    ax.set_yticklabels(matf.index, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_xlabel(f'{matf.shape[1]} perturbations (KOs), sorted by mean effect',
                  fontsize=fontsize + 6)
    ax.set_ylabel('memory genes', fontsize=fontsize + 6)
    ax.set_title(title or f'Every memory gene × every perturbation (n={len(matf)} genes)',
                 fontsize=fontsize + 7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Memory × KO heatmap saved to: {save_path}")
    if show:
        plt.show()
    return matf


def gene_response_profile(lfc_df, genes, thr=0.5, broad_at=0.40, weak_peak=0.5):
    """
    Per-gene response profile across all KOs -- answers 'which genes respond broadly /
    weakly / bimodally'. For each gene:
      n_obs        : KOs measured
      breadth      : fraction of KOs with a real reaction (|LFC| > thr)
      mean_abs_lfc : average |LFC| (overall strength)
      peak_abs_lfc : 95th-percentile |LFC| (how strong its biggest reactions are)
      bimodality   : Sarle's bimodality coefficient of |LFC| (>0.555 ~ bimodal)
      response_class :
        'broad'     -> reacts in >= broad_at of KOs
        'selective' -> big peak but narrow breadth (strong to SOME KOs, not others = bimodal)
        'weak'      -> even its 95th-pct |LFC| < weak_peak (barely reacts to anything)
    Sorted by breadth (broadest first).
    """
    from scipy.stats import skew, kurtosis

    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    sub = lfc_df[cols]
    absv = sub.abs()

    n = sub.notna().sum(axis=0)
    breadth = (absv > thr).sum(axis=0) / n.replace(0, np.nan)
    mean_abs = absv.mean(axis=0)
    peak = absv.quantile(0.95, axis=0)
    a = absv.values
    with np.errstate(all='ignore'):
        s = skew(a, axis=0, nan_policy='omit')
        k = kurtosis(a, axis=0, fisher=False, nan_policy='omit')
        bc = (s ** 2 + 1) / np.where(k > 0, k, np.nan)

    prof = pd.DataFrame({
        'n_obs': n.astype(int), 'breadth': breadth, 'mean_abs_lfc': mean_abs,
        'peak_abs_lfc': peak, 'bimodality': pd.Series(bc, index=absv.columns),
    })

    def classify(r):
        if r['peak_abs_lfc'] < weak_peak:
            return 'weak'
        if r['breadth'] >= broad_at:
            return 'broad'
        return 'selective'
    prof['response_class'] = prof.apply(classify, axis=1)
    return prof.sort_values('breadth', ascending=False)


def plot_gene_response_profile(
    profile, broad_at=0.40, weak_peak=0.5, label_n=6, fontsize=14,
    figsize=(11, 8), title=None, save_path=None, show=True,
):
    """
    Scatter: x = breadth (% of KOs a gene reacts to), y = peak |LFC| (strength of its
    biggest reactions). Cleanly separates the three classes:
      top-right = BROAD (reacts to most KOs), top-left = SELECTIVE/bimodal (reacts
      strongly but only to some KOs), bottom = WEAK (barely reacts to anything).
    Coloured by response_class; a few genes labelled per class. Shows the responders
    are spread across many genes (not a couple).
    """
    palette = {'broad': 'teal', 'selective': 'darkorange', 'weak': 'lightgray'}
    x = profile['breadth'] * 100
    y = profile['peak_abs_lfc']

    fig, ax = plt.subplots(figsize=figsize)
    for cls, c in palette.items():
        m = profile['response_class'] == cls
        ax.scatter(x[m], y[m], s=40, color=c, alpha=0.7, edgecolor='k',
                   linewidth=0.3, label=f'{cls} (n={int(m.sum())})')
    ax.axvline(broad_at * 100, color='gray', ls=':', lw=1.2)
    ax.axhline(weak_peak, color='gray', ls=':', lw=1.2)

    ax.text(0.98, 0.97, 'BROAD', transform=ax.transAxes, ha='right', va='top',
            color='teal', fontweight='bold', fontsize=fontsize)
    ax.text(0.02, 0.97, 'SELECTIVE\n(bimodal)', transform=ax.transAxes, ha='left',
            va='top', color='darkorange', fontweight='bold', fontsize=fontsize)
    ax.text(0.5, 0.02, 'WEAK', transform=ax.transAxes, ha='center', va='bottom',
            color='gray', fontweight='bold', fontsize=fontsize)

    for cls in ('broad', 'selective'):
        sub = profile[profile['response_class'] == cls].head(label_n)
        for g, r in sub.iterrows():
            ax.annotate(g, (r['breadth'] * 100, r['peak_abs_lfc']), fontsize=fontsize - 6,
                        fontweight='bold', xytext=(3, 2), textcoords='offset points')

    ax.set_xlabel('how OFTEN it reacts  (% of knockouts with |LFC| > threshold)', fontsize=fontsize)
    ax.set_ylabel('how STRONG its reactions get  (|LFC| in its strongest 5% of KOs)',
                  fontsize=fontsize)
    ax.set_title(title or 'Per-gene response profile across KOs', fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3, loc='center right')
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gene response profile saved to: {save_path}")
    if show:
        plt.show()
    return profile


def response_class_go_enrichment(
    profile, gmt_path, background, classes=('broad', 'selective', 'weak'),
    min_term_size=5, max_term_size=500, top_n=10,
):
    """
    GO Biological Process over-representation for each response class (broad /
    selective / weak) from gene_response_profile, using the offline Fisher test
    (enrich_against_gmt) against `background` (use all measured genes = lfc_df.columns).
    Returns {class: enrichment DataFrame or None}.
    """
    gmt = load_gmt(gmt_path)
    out = {}
    for cls in classes:
        genes = profile.index[profile['response_class'] == cls].tolist()
        try:
            out[cls] = enrich_against_gmt(genes, gmt, background,
                                          min_term_size=min_term_size,
                                          max_term_size=max_term_size, top_n=top_n)
        except ValueError as e:
            print(f"{cls}: {e}")
            out[cls] = None
    return out


def plot_response_class_go(
    enrich_by_class, top_n=8, fontsize=11, figsize=(12, 12),
    title=None, save_path=None, show=True,
):
    """
    Stacked GO-term comparison, one row per response class: horizontal bars of
    -log10 FDR for the top terms, so you can read what BROAD vs SELECTIVE vs WEAK
    responders are, functionally. Red line = FDR 0.05. Returns the figure's axes.
    """
    colors = {'broad': 'teal', 'selective': 'darkorange', 'weak': 'gray'}
    classes = list(enrich_by_class)
    fig, axes = plt.subplots(len(classes), 1, figsize=figsize)
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        df = enrich_by_class.get(cls)
        c = colors.get(cls, 'steelblue')
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, f'{cls}: no enriched GO terms', transform=ax.transAxes,
                    ha='center', va='center', fontsize=fontsize, color=c)
            ax.axis('off')
            continue
        d = df.head(top_n).iloc[::-1]
        yy = np.arange(len(d))
        vals = -np.log10(d['q_bh'].clip(lower=1e-300).values)
        ax.barh(yy, vals, color=c, alpha=0.85)
        # annotate each bar with the number of overlapping genes
        for yi, v, k in zip(yy, vals, d['n_overlap'].values):
            ax.text(v + 0.03, yi, f'{int(k)} genes', va='center',
                    fontsize=fontsize - 4, color='dimgray')
        ax.set_yticks(yy)
        labels = [_pretty_term(t) for t in d['term']]
        ax.set_yticklabels(labels, fontsize=fontsize - 1)
        ax.axvline(-np.log10(0.05), color='red', ls='--', lw=1.3)
        ax.set_xlim(0, vals.max() * 1.28)
        n = int(d['query_size'].iloc[0])
        ax.set_title(f'{cls.upper()} responders (n={n})', fontsize=fontsize + 1,
                     color=c, fontweight='bold', loc='left')
        ax.set_xlabel('significance  (–log10 FDR;  right of red dashes = FDR < 0.05)',
                      fontsize=fontsize - 2)
        ax.tick_params(labelsize=fontsize - 3)

    fig.suptitle(title or 'Pathway enrichment by response class',
                 fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GO-by-class figure saved to: {save_path}")
    if show:
        plt.show()
    return axes


def plot_class_go_dotplot(
    profile, gmt_path, background, classes=('broad', 'selective', 'weak'),
    min_term_size=5, max_term_size=500, top_per_class=6, fontsize=11,
    figsize=(9, 12), title=None, save_path=None, show=True,
):
    """
    One combined dot-plot: rows = pathways, columns = response classes. Dot COLOUR =
    significance (-log10 FDR), dot SIZE = number of class genes in the pathway. Shows,
    for every category, which of broad / selective / weak is enriched in it. Rows are
    the union of each class's top `top_per_class` terms. Returns the long DataFrame.
    """
    gmt = load_gmt(gmt_path)
    recs, chosen = {}, set()
    for cls in classes:
        genes = profile.index[profile['response_class'] == cls].tolist()
        try:
            df = enrich_against_gmt(genes, gmt, background, min_term_size=min_term_size,
                                    max_term_size=max_term_size, top_n=None)
        except ValueError:
            df = None
        recs[cls] = df
        if df is not None:
            chosen |= set(df.head(top_per_class)['term'])

    rows = []
    for cls, df in recs.items():
        if df is None:
            continue
        for _, r in df[df['term'].isin(chosen)].iterrows():
            rows.append({'term': r['term'], 'class': cls, 'n_overlap': int(r['n_overlap']),
                         'neglogq': -np.log10(max(r['q_bh'], 1e-300))})
    long = pd.DataFrame(rows)
    if long.empty:
        print("No enriched terms to plot.")
        return long

    order = long.groupby('term')['neglogq'].max().sort_values().index.tolist()
    term_idx = {t: i for i, t in enumerate(order)}
    cls_idx = {c: i for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=figsize)
    for yi in range(len(order)):
        ax.axhline(yi, color='0.92', lw=0.8, zorder=0)
    sc = ax.scatter(
        [cls_idx[c] for c in long['class']], [term_idx[t] for t in long['term']],
        s=long['n_overlap'] * 45, c=long['neglogq'], cmap='Reds', vmin=0,
        edgecolor='k', linewidth=0.4, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('significance (–log10 FDR)', fontsize=fontsize)
    cb.ax.axhline(-np.log10(0.05), color='blue', lw=1.5)   # FDR 0.05 mark on the colorbar

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.upper() for c in classes], fontsize=fontsize + 1, fontweight='bold')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty_term(t) for t in order], fontsize=fontsize - 1)
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(-0.7, len(order) - 0.3)

    # size legend
    for k in (1, 3, 5):
        ax.scatter([], [], s=k * 45, c='gray', edgecolor='k', linewidth=0.4, label=f'{k} genes')
    ax.legend(title='genes in pathway', loc='lower left', bbox_to_anchor=(1.02, 0),
              fontsize=fontsize - 2, title_fontsize=fontsize - 1, frameon=False)

    ax.set_title(title or 'Pathway enrichment across response classes\n'
                          '(blue mark on colorbar = FDR 0.05)', fontsize=fontsize + 2)
    ax.tick_params(length=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class GO dot-plot saved to: {save_path}")
    if show:
        plt.show()
    return long


# Human-readable names for the 50 MSigDB Hallmark 2020 categories (keys lowercased).
HALLMARK_LABELS = {
    'tnf-alpha signaling via nf-kb': 'TNF-α / NF-κB inflammation',
    'hypoxia': 'Low-oxygen (hypoxia) response',
    'cholesterol homeostasis': 'Cholesterol balance',
    'mitotic spindle': 'Cell division (spindle)',
    'wnt-beta catenin signaling': 'Wnt signaling',
    'tgf-beta signaling': 'TGF-β signaling',
    'il-6/jak/stat3 signaling': 'IL-6 / JAK-STAT3 signaling',
    'dna repair': 'DNA repair',
    'g2-m checkpoint': 'Cell cycle (G2/M)',
    'apoptosis': 'Programmed cell death',
    'notch signaling': 'Notch signaling',
    'adipogenesis': 'Fat-cell development',
    'estrogen response early': 'Estrogen response (early)',
    'estrogen response late': 'Estrogen response (late)',
    'androgen response': 'Androgen (hormone) response',
    'myogenesis': 'Muscle development',
    'protein secretion': 'Protein secretion',
    'interferon alpha response': 'Interferon-α (antiviral)',
    'interferon gamma response': 'Interferon-γ (immune activation)',
    'apical junction': 'Cell-cell junctions',
    'apical surface': 'Cell surface (apical)',
    'hedgehog signaling': 'Hedgehog signaling',
    'complement': 'Complement (innate immunity)',
    'unfolded protein response': 'Unfolded-protein stress',
    'pi3k/akt/mtor signaling': 'PI3K-AKT-mTOR signaling',
    'mtorc1 signaling': 'mTOR growth signaling',
    'e2f targets': 'Cell cycle (E2F proliferation)',
    'myc targets v1': 'MYC growth targets',
    'myc targets v2': 'MYC growth targets (II)',
    'epithelial mesenchymal transition': 'EMT / tissue remodeling',
    'inflammatory response': 'Inflammation',
    'xenobiotic metabolism': 'Detox / drug metabolism',
    'fatty acid metabolism': 'Fatty-acid metabolism',
    'oxidative phosphorylation': 'Mitochondrial energy (OXPHOS)',
    'glycolysis': 'Glycolysis (sugar metabolism)',
    'reactive oxygen species pathway': 'Oxidative stress (ROS)',
    'p53 pathway': 'p53 stress response',
    'uv response up': 'UV/DNA-damage response (up)',
    'uv response dn': 'UV/DNA-damage response (down)',
    'angiogenesis': 'Blood-vessel growth',
    'heme metabolism': 'Red-blood-cell / heme',
    'coagulation': 'Blood coagulation',
    'il-2/stat5 signaling': 'IL-2 / STAT5 (T-cell) signaling',
    'bile acid metabolism': 'Bile-acid metabolism',
    'peroxisome': 'Peroxisome metabolism',
    'allograft rejection': 'Immune rejection / T-cell activation',
    'spermatogenesis': 'Sperm development',
    'kras signaling up': 'KRAS growth signaling (up)',
    'kras signaling dn': 'KRAS signaling (down)',
    'pancreas beta cells': 'Pancreatic β-cell genes',
}


def _pretty_term(term, maxlen=42):
    """Human-readable pathway label: map Hallmark names, else clean GO strings."""
    clean = term.split(' (GO:')[0].replace('_', ' ').strip()
    return HALLMARK_LABELS.get(clean.lower(), clean)[:maxlen]


def plot_leaveout_genes(
    lfc_df, memory_genes, background_genes, background_mode='all', gene_expr=None,
    n_bins=10, n_iter=30, step=2, random_seed=0, fontsize=14, figsize=(10, 6),
    title=None, save_path=None, show=True,
):
    """
    Is 'memory genes are more responsive' driven by a few genes? Per gene,
    responsiveness = mean |LFC| across KOs. Plots the MEDIAN responsiveness of the
    memory set as the top-k most-responsive memory genes are progressively removed,
    against a non-memory reference. If the memory line stays above it, NOT a few genes.

    background_mode : 'all'     -> grey line = median responsiveness over ALL
                                   non-memory genes (the Non-SCG boxen median)
                      'matched' -> expression-matched sampled median (needs gene_expr)
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    mem_resp = lfc_df[mem].abs().mean(axis=0)               # per-gene responsiveness
    ranked = mem_resp.sort_values(ascending=False).index    # most responsive first
    pool_all = [g for g in background_genes if g in measured and g not in set(mem)]

    if background_mode == 'matched' and gene_expr is not None:
        expr = gene_expr.reindex(measured).dropna()
        mem_e = [g for g in mem if g in expr.index]
        ebins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'), index=expr.index)
        per_bin = ebins.loc[mem_e].value_counts()
        pset = set(pool_all)
        binpool = {b: ebins.index[(ebins == b) & (ebins.index.isin(pset))].to_numpy() for b in per_bin.index}
        def draw():
            s = []
            for b, need in per_bin.items():
                s.extend(rng.choice(binpool[b], size=min(need, len(binpool[b])), replace=False))
            return s
        bg_meds = [np.median(lfc_df[draw()].abs().mean(axis=0)) for _ in range(n_iter)]
        bg_med = float(np.median(bg_meds))
        bg_lo, bg_hi = np.percentile(bg_meds, [5, 95])
        bg_lab = 'expression-matched non-memory median'
    else:
        # median responsiveness over ALL non-memory genes (the Non-SCG median)
        bg_resp = lfc_df[pool_all].abs().mean(axis=0).values
        bg_med = float(np.median(bg_resp))
        boot = [np.median(rng.choice(bg_resp, size=len(bg_resp), replace=True)) for _ in range(200)]
        bg_lo, bg_hi = np.percentile(boot, [5, 95])
        bg_lab = 'non-memory median'

    ks = list(range(0, len(mem) - 5, step))
    curve = [np.median(mem_resp.loc[ranked[k:]]) for k in ks]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, curve, '-o', color='crimson', ms=3, label='memory (top-k removed)')
    ax.axhspan(bg_lo, bg_hi, color='gray', alpha=0.25)
    ax.axhline(bg_med, color='gray', lw=2, label=bg_lab)
    ax.set_xlabel('number of most-responsive memory genes removed', fontsize=fontsize)
    ax.set_ylabel('median responsiveness (mean |LFC|)\nof remaining memory genes', fontsize=fontsize)
    ax.set_title(title or 'Remaining SCG responsiveness while removing the most-responsive genes',
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return pd.DataFrame({'k_removed': ks, 'memory_median': curve}).assign(bg_median=bg_med)


def plot_leaveout_kos(
    lfc_df, memory_genes, metric='signed', background_genes=None, gene_expr=None,
    n_bins=10, n_bg=2000, step=25, random_seed=0, fontsize=20, figsize=(11, 6.5),
    title=None, save_path=None, show=True,
):
    """
    Is the SCG mean driven by a few perturbations? (Madeline). Per KO, impact =
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


def plot_top_scg_ko_check(
    lfc_df, memory_genes, background_genes, top_q=0.20, gene_expr=None, n_bins=10,
    n_bg=3000, random_seed=0, fontsize=13, figsize=(14, 6),
    title=None, save_path=None, show=True,
):
    """
    Madeline: for the perturbations where SCGs respond most (top `top_q` by memory
    mean |LFC|), do the OTHER genes also have higher-than-average LFC? If yes, those
    KOs are just globally strong perturbations (not memory-specific).

    gene_expr : if given, the 'other genes' are EXPRESSION-MATCHED to the memory genes
                (binned into n_bins) instead of a random sample of all non-memory.

    Left: per-KO scatter of non-memory impact (x) vs memory impact (y), top-q memory
    KOs highlighted; Pearson r annotated. Right: mean |LFC| of memory and non-memory
    genes in the top-q SCG KOs vs the other KOs (dashed = overall non-memory average).
    Returns a stats dict.
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    if gene_expr is not None:                    # expression-matched 'other genes'
        bg_pool = _match_expression(mem, background_genes, gene_expr, measured,
                                    n_bins=n_bins, rng=rng, reps=5)
    else:
        bg_pool = [g for g in background_genes if g in measured and g not in set(mem)]
        if len(bg_pool) > n_bg:                  # representative sample of "other genes"
            bg_pool = list(rng.choice(bg_pool, size=n_bg, replace=False))

    mem_impact = lfc_df[mem].abs().mean(axis=1)          # per-KO memory response
    bg_impact = lfc_df[bg_pool].abs().mean(axis=1)       # per-KO non-memory response

    thr = mem_impact.quantile(1 - top_q)
    top = mem_impact >= thr
    stats = {
        'pearson_r': float(np.corrcoef(mem_impact, bg_impact)[0, 1]),
        'bg_overall': float(bg_impact.mean()),
        'bg_top': float(bg_impact[top].mean()),
        'bg_rest': float(bg_impact[~top].mean()),
        'mem_top': float(mem_impact[top].mean()),
        'mem_rest': float(mem_impact[~top].mean()),
    }
    stats['bg_fold_top_vs_overall'] = stats['bg_top'] / stats['bg_overall']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(bg_impact[~top], mem_impact[~top], s=8, color='lightgray', alpha=0.5,
                label='other 80% KOs')
    ax1.scatter(bg_impact[top], mem_impact[top], s=10, color='crimson', alpha=0.6,
                label=f'top {int(top_q*100)}% memory-response KOs')
    ax1.set_xlabel('non-memory response per KO  (mean |LFC|)', fontsize=fontsize)
    ax1.set_ylabel('memory response per KO  (mean |LFC|)', fontsize=fontsize)
    ax1.set_title(f'Memory vs non-memory response per knockout\nPearson r = {stats["pearson_r"]:.2f}',
                  fontsize=fontsize + 1)
    ax1.legend(fontsize=fontsize - 3)
    ax1.tick_params(labelsize=fontsize - 3)

    groups = ['non-memory\ngenes', 'memory\ngenes']
    top_vals = [stats['bg_top'], stats['mem_top']]
    rest_vals = [stats['bg_rest'], stats['mem_rest']]
    xpos = np.arange(2)
    ax2.bar(xpos - 0.2, top_vals, width=0.4, color='crimson', alpha=0.8,
            label=f'top {int(top_q*100)}% memory-response KOs')
    ax2.bar(xpos + 0.2, rest_vals, width=0.4, color='steelblue', alpha=0.8, label='other KOs')
    ax2.axhline(stats['bg_overall'], color='gray', ls='--', lw=1.5,
                label='overall non-memory avg')
    ax2.set_xticks(xpos); ax2.set_xticklabels(groups, fontsize=fontsize)
    ax2.set_ylabel('mean |LFC|', fontsize=fontsize)
    ax2.set_title(f'non-memory in top memory-response KOs is {stats["bg_fold_top_vs_overall"]:.1f}× overall',
                  fontsize=fontsize + 1)
    ax2.legend(fontsize=fontsize - 3)
    ax2.tick_params(labelsize=fontsize - 3)

    fig.suptitle(title or 'Response in top-20% memory-response knockouts: memory and non-memory genes',
                 fontsize=fontsize + 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return stats


def perturbed_covariance(adata, perturbed_genes, do_normalize=True):
    """
    Ben / fluctuation-dissipation: in unperturbed (resting-control) cells, each gene's
    covariance with the perturbed-gene set. Uses the identity mean_j cov(x_i, x_j) =
    cov(x_i, Pbar), where Pbar = mean expression of the perturbed genes per cell, so
    only ONE covariance vector is needed (no gene x gene matrix).

    adata : cells x genes AnnData of RESTING CONTROL cells (raw counts ok).
    Returns per-gene DataFrame: cov_perturbed, corr_perturbed, mean_expr, var_expr.
    """
    import scanpy as sc
    import scipy.sparse as sp

    ad = adata.copy()
    if do_normalize:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    X = ad.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)
    N = X.shape[0]
    genes = np.asarray(ad.var_names)
    gidx = {g: i for i, g in enumerate(genes)}

    Pcols = [gidx[g] for g in perturbed_genes if g in gidx]
    Pbar = X[:, Pcols].mean(axis=1)
    Pc = Pbar - Pbar.mean()                       # centred meta-gene
    cov = (X.T @ Pc) / N                           # cov(gene, Pbar); Pc centred so no X-centering
    std = X.std(axis=0)
    corr = cov / (std * Pc.std() + 1e-12)

    return pd.DataFrame({
        'cov_perturbed': cov, 'corr_perturbed': corr,
        'mean_expr': X.mean(axis=0), 'var_expr': std ** 2,
    }, index=pd.Index(genes, name='gene'))


def plot_perturbed_covariance(
    cov_df, memory_genes, metric='corr_perturbed', gene_summary=None,
    style='boxen', fontsize=15, figsize=(14, 6), title=None, save_path=None, show=True,
):
    """
    Compare SCG vs non-SCG baseline covariance-with-perturbed-genes (Ben's question),
    and test the fluctuation-dissipation link.

    Left: SCG vs non-SCG distribution of `metric` (corr_perturbed = expression-robust;
    cov_perturbed = Ben's literal covariance), Mann-Whitney p.
    Right (if gene_summary given): per-gene baseline covariance vs LFC responsiveness
    (mean_abs_lfc) -- a positive trend supports fluctuation-dissipation (equilibrium
    covariance predicts perturbation response). Returns a stats dict.
    """
    from scipy.stats import mannwhitneyu

    df = cov_df.dropna(subset=[metric])
    is_scg = df.index.isin(set(memory_genes))
    scg = df.loc[is_scg, metric].values
    non = df.loc[~is_scg, metric].values
    p = mannwhitneyu(scg, non, alternative='two-sided')[1]
    stats = {'metric': metric, 'scg_median': float(np.median(scg)),
             'nonscg_median': float(np.median(non)), 'mwu_p': float(p), 'n_scg': int(is_scg.sum())}

    ncol = 2 if gene_summary is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize if ncol == 2 else (7, 6))
    axes = np.atleast_1d(axes)

    plot_df = pd.DataFrame({metric: np.concatenate([scg, non]),
                            'grp': ['SCG'] * len(scg) + ['Non-SCG'] * len(non)})
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    plotter(data=plot_df, x='grp', y=metric, order=['Non-SCG', 'SCG'],
            palette={'Non-SCG': 'steelblue', 'SCG': 'crimson'}, ax=axes[0])
    axes[0].set_title(f'Baseline covariance with perturbed genes\nMann-Whitney p={p:.1e}',
                      fontsize=fontsize)
    axes[0].set_xlabel(''); axes[0].set_ylabel(metric, fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize - 2)

    if gene_summary is not None:
        resp = gene_summary['mean_abs_lfc'].reindex(df.index)
        ok = resp.notna()
        r = float(np.corrcoef(df.loc[ok, metric], resp[ok])[0, 1])
        stats['fdt_pearson_r'] = r
        ax = axes[1]
        ax.scatter(df.loc[ok & ~is_scg, metric], resp[ok & ~is_scg], s=6,
                   color='lightgray', alpha=0.4, label='Non-SCG')
        ax.scatter(df.loc[ok & is_scg, metric], resp[ok & is_scg], s=14,
                   color='crimson', alpha=0.6, label='SCG')
        ax.set_xlabel(f'baseline {metric}', fontsize=fontsize)
        ax.set_ylabel('LFC responsiveness (mean |LFC|)', fontsize=fontsize)
        ax.set_title(f'Fluctuation-dissipation test\nPearson r = {r:.2f}', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 3); ax.tick_params(labelsize=fontsize - 2)

    fig.suptitle(title or 'SCG vs non-SCG covariance with perturbed genes (resting cells)',
                 fontsize=fontsize + 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return stats


def rank_memory_specific_kos(vol, drivers, min_memory_breadth=0.15):
    """
    Rank KOs by how MEMORY-SPECIFIC their effect is, not by raw breadth -- this
    demotes the essential/global knockouts (they hit everything, so low enrichment).

    vol      : per-KO stats from ko_specificity_volcano_stats (log2_enrichment vs the
               expression-matched null, neglog10p, frac_react_memory).
    drivers  : rank_ko_drivers output (for direction / mean_signed_lfc).
    Keeps only KOs that actually move a meaningful fraction of memory genes
    (frac_react_memory >= min_memory_breadth), then sorts by log2_enrichment.
    """
    df = vol[vol['frac_react_memory'] >= min_memory_breadth].copy()
    df = df.join(drivers[['frac_react', 'mean_signed_lfc', 'direction']], how='left')
    return df.sort_values('log2_enrichment', ascending=False)


def plot_memory_specific_drivers(
    ranked, top_n=25, fontsize=13, figsize=(10, 9), title=None, save_path=None, show=True,
):
    """
    Top memory-SPECIFIC driver KOs (from rank_memory_specific_kos): bars = log2
    enrichment over expression-matched, coloured by direction (purple=down, green=up),
    annotated with the % of memory genes each moves. These are the candidate memory
    regulators left after the essential/global KOs drop out.
    """
    top = ranked.head(top_n).iloc[::-1]
    colors = ['green' if d == 'up' else 'purple' for d in top['direction']]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top)), top['log2_enrichment'].values, color=colors, alpha=0.85)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r['log2_enrichment'] + 0.02, i, f"{r['frac_react_memory']*100:.0f}% mem",
                va='center', fontsize=fontsize - 5, color='dimgray')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=fontsize - 3)
    ax.set_xlabel('log2 enrichment over expression-matched  (memory-specificity)', fontsize=fontsize)
    ax.set_title(title or 'Memory-specific driver KOs (essential/global KOs removed)',
                 fontsize=fontsize + 1)
    ax.set_xlim(0, top['log2_enrichment'].max() * 1.25)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return top


def plot_responsiveness_floor(
    lfc_df, memory_genes, background_genes, gene_expr, floor=0.1, n_bins=10,
    n_iter=50, random_seed=0, fontsize=14, figsize=(10, 6),
    title=None, save_path=None, show=True,
):
    """
    Madeline: does including 0/1-count genes in the background inflate the memory-vs-
    non-memory responsiveness gap? Compares median responsiveness (mean |LFC| per gene)
    of memory genes against three background definitions:
      - ALL non-memory (includes unexpressed genes -> dilutes baseline)
      - non-memory with baseline expr > floor (zero-expression genes removed)
      - expression-matched non-memory (binned to the memory genes' expression)
    The memory/background ratio above each bar shows how the gap shrinks once the
    low-expression genes are handled. Returns a stats dict.
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    resp = lfc_df.abs().mean(axis=0)                         # per-gene responsiveness
    expr = gene_expr.reindex(measured)

    mem = [g for g in memory_genes if g in measured]
    mem_med = float(np.median(resp[mem]))
    pool_all = [g for g in background_genes if g in measured and g not in set(mem)]
    bg_all = float(np.median(resp[pool_all]))
    pool_floor = [g for g in pool_all if expr.get(g, 0) > floor]
    bg_floor = float(np.median(resp[pool_floor]))

    # expression-matched (memory genes binned; draw same-size non-memory per bin)
    e = expr.dropna()
    mem_e = [g for g in mem if g in e.index]
    ebins = pd.Series(pd.qcut(e, n_bins, labels=False, duplicates='drop'), index=e.index)
    per_bin = ebins.loc[mem_e].value_counts()
    pset = set(pool_all)
    binpool = {b: ebins.index[(ebins == b) & (ebins.index.isin(pset))].to_numpy() for b in per_bin.index}
    meds = []
    for _ in range(n_iter):
        samp = []
        for b, need in per_bin.items():
            samp.extend(rng.choice(binpool[b], size=min(need, len(binpool[b])), replace=False))
        meds.append(np.median(resp[samp]))
    bg_match = float(np.median(meds))

    n_mem_below = int((expr.reindex(mem) <= floor).sum())
    labels = ['Memory', 'Non-mem\n(all)', f'Non-mem\n(expr>{floor})', 'Non-mem\n(expr-matched)']
    vals = [mem_med, bg_all, bg_floor, bg_match]
    colors = ['crimson', 'lightgray', 'gray', 'steelblue']

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(4), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=fontsize - 3)
    for i in (1, 2, 3):                                       # memory/background ratio
        ax.text(i, vals[i] / 2, f'{mem_med / vals[i]:.1f}×', ha='center',
                color='white', fontweight='bold', fontsize=fontsize)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=fontsize - 1)
    ax.set_ylabel('median responsiveness (mean |LFC|)', fontsize=fontsize)
    ax.set_title(title or 'Memory-vs-non-memory gap shrinks once low-expression genes are handled',
                 fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()

    stats = {'memory_median': mem_med, 'nonmem_all': bg_all, 'nonmem_floor': bg_floor,
             'nonmem_matched': bg_match, 'ratio_all': mem_med / bg_all,
             'ratio_floor': mem_med / bg_floor, 'ratio_matched': mem_med / bg_match,
             'n_nonmem_all': len(pool_all), 'n_nonmem_floor': len(pool_floor),
             'n_memory_below_floor': n_mem_below}
    return stats


def plot_responsiveness_boxen_matched(
    lfc_df, memory_genes, background_genes, gene_expr, n_bins=10, n_iter=20,
    n_perm=200, mem_name='memory gene', bg_name='non-memory gene', random_seed=0,
    style='boxen', fontsize=15, figsize=(7, 6), title=None, save_path=None, show=True,
):
    """
    Responsiveness (mean |LFC| per gene across KOs): memory genes vs EXPRESSION-
    MATCHED non-memory genes (binned to the memory genes' baseline expression).

    Draws n_perm matched same-size sets. The plotted non-memory distribution is the
    mean of the first n_iter draws' sorted values (equal n, stable). Significance is a
    proper EMPIRICAL p: where the memory median falls among the n_perm matched-draw
    medians (one-sided, memory > matched) -- valid and stable, unlike an MWU on the
    averaged distribution.
    """
    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    resp = lfc_df.abs().mean(axis=0)
    expr = gene_expr.reindex(measured).dropna()

    mem = [g for g in memory_genes if g in measured and g in expr.index]
    pool = set(g for g in background_genes if g in measured and g not in set(mem))
    ebins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'), index=expr.index)
    per_bin = ebins.loc[mem].value_counts()
    binpool = {b: ebins.index[(ebins == b) & (ebins.index.isin(pool))].to_numpy() for b in per_bin.index}

    n_perm = max(n_perm, n_iter)
    sorted_draws, null_medians = [], []
    for _ in range(n_perm):
        samp = []
        for b, need in per_bin.items():
            samp.extend(rng.choice(binpool[b], size=min(need, len(binpool[b])), replace=False))
        v = np.sort(resp[samp].values)
        sorted_draws.append(v); null_medians.append(np.median(v))
    null_medians = np.array(null_medians)

    # plotted distribution = mean of first n_iter draws' sorted values (equal n, stable)
    L = min(len(d) for d in sorted_draws[:n_iter])
    bg_vals = np.mean([d[:L] for d in sorted_draws[:n_iter]], axis=0)

    from scipy.stats import norm
    mem_vals = resp[mem].values
    mem_med = float(np.median(mem_vals))
    n_beaten = int((null_medians < mem_med).sum())                        # matched sets memory beats
    emp_p = float((np.sum(null_medians >= mem_med) + 1) / (n_perm + 1))   # memory > matched
    null_mu, null_sd = float(null_medians.mean()), float(null_medians.std(ddof=1))
    z = (mem_med - null_mu) / null_sd if null_sd > 0 else np.nan          # permutation z-score
    z_p = float(norm.sf(z))                                               # one-sided normal-approx p
    ratio = mem_med / null_mu

    bg_label = f'{bg_name}\n(expr-matched)'
    mem_label = f'{mem_name}'
    plot_df = pd.DataFrame({
        'Mean |LFC| across KOs': np.concatenate([bg_vals, mem_vals]),
        'Group': [bg_label] * len(bg_vals) + [mem_label] * len(mem_vals),
    })
    order = [bg_label, mem_label]
    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    plotter(data=plot_df, x='Group', y='Mean |LFC| across KOs', order=order,
            palette=['steelblue', 'crimson'], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Mean |LFC| across KOs', fontsize=fontsize + 4)
    top = title or 'Memory vs expression-matched non-memory responsiveness'
    ax.set_title(f'{top}\n{ratio:.2f}× more responsive,  z = {z:.1f} (p = {z_p:.1e})\n'
                 f'n({mem_name})={len(mem_vals)}, n({bg_name})={len(bg_vals)}',
                 fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return {'memory_median': mem_med, 'matched_median': null_mu,
            'ratio': float(ratio), 'z_score': float(z), 'z_p': z_p,
            'empirical_p': emp_p, 'beats_n_of_perm': f'{n_beaten}/{n_perm}',
            'n_perm': n_perm, 'n': len(mem_vals)}


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


def plot_expr_vs_response(
    lfc_df, gene_expr, memory_genes, mem_label='SCG', expr_floor=1e-3, keep_all=False,
    n_ebins=25, scg_size=30, fontsize=14, figsize=(15, 6),
    title=None, save_path=None, show=True,
):
    """
    Does baseline expression drive response magnitude, and do SCGs beat the trend?

    Left  : density cloud of all genes (x = log10 baseline expression, y = response =
            mean |LFC| across KOs), SCGs overlaid in crimson, with the binned-median
            trend over all genes. SCGs sitting ABOVE the trend = they respond more than
            expression predicts (the expression-matched residual, shown visually).
    Right : per-KO Pearson r between log-expression and |LFC| across genes -> one value
            per knockout; the distribution shows how consistently high-expression genes
            respond more strongly.

    Returns a stats dict (median per-KO r; SCG vs non-SCG median residual above trend).
    """
    measured = lfc_df.columns
    absA = lfc_df.abs()
    resp = absA.mean(axis=0)                                   # per-gene response
    expr = gene_expr.reindex(measured)

    df = pd.DataFrame({'expr': expr, 'resp': resp})
    if keep_all:                                              # clip zeros/low to the left edge
        df['expr'] = df['expr'].fillna(0).clip(lower=expr_floor)
        df = df.dropna(subset=['resp'])
    else:                                                     # drop unexpressed (default)
        df = df.dropna()
        df = df[df['expr'] > expr_floor]
    is_scg = df.index.isin(set(memory_genes))
    logx = np.log10(df['expr'].values)
    y = df['resp'].values

    # binned-median trend over all genes + per-gene residual above it
    bins = np.linspace(logx.min(), logx.max(), n_ebins)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(logx, bins) - 1, 0, len(centers) - 1)
    med = np.array([np.median(y[idx == b]) if np.any(idx == b) else np.nan
                    for b in range(len(centers))])
    ok = ~np.isnan(med)
    trend_at = np.interp(logx, centers[ok], med[ok])
    resid = y - trend_at

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Panel A: cloud + SCGs + trend ---
    ax1.hexbin(logx[~is_scg], y[~is_scg], gridsize=55, cmap='Greys', bins='log', mincnt=1)
    ax1.scatter(logx[is_scg], y[is_scg], s=scg_size, color='crimson', alpha=0.6,
                edgecolor='black', linewidth=0.3, label=f'{mem_label} (n={int(is_scg.sum())})')
    ax1.plot(centers[ok], med[ok], color='black', lw=2.5, label='median (all genes)')
    ax1.set_xlabel('log10 baseline expression', fontsize=fontsize)
    ax1.set_ylabel('response  (mean |LFC| across KOs)', fontsize=fontsize)
    ax1.set_title(f'Expression vs response  ({mem_label} in red)', fontsize=fontsize + 1)
    ax1.legend(fontsize=fontsize - 3)
    ax1.tick_params(labelsize=fontsize - 3)

    # --- Panel B: per-KO correlation of log-expr vs |LFC| ---
    A = absA[df.index].values                                 # KO x expressed genes
    ec = logx - logx.mean()
    Ac = np.nan_to_num(A - np.nanmean(A, axis=1, keepdims=True))
    ko_r = (Ac @ ec) / (np.sqrt((Ac ** 2).sum(axis=1)) * np.sqrt((ec ** 2).sum()) + 1e-12)
    med_r = float(np.nanmedian(ko_r))
    ax2.hist(ko_r, bins=40, color='steelblue', alpha=0.85)
    ax2.axvline(0, color='k', ls='--', lw=1.5)
    ax2.axvline(med_r, color='crimson', lw=2, label=f'median r = {med_r:.2f}')
    ax2.set_xlabel('Pearson r (log-expr vs |LFC|) per knockout', fontsize=fontsize)
    ax2.set_ylabel('number of knockouts', fontsize=fontsize)
    ax2.set_title('Do high-expression genes respond more, per KO?', fontsize=fontsize + 1)
    ax2.legend(fontsize=fontsize - 3)
    ax2.tick_params(labelsize=fontsize - 3)

    stats = {'median_ko_r': med_r,
             'scg_resid_median': float(np.median(resid[is_scg])),
             'nonscg_resid_median': float(np.median(resid[~is_scg])),
             'n_scg': int(is_scg.sum()), 'n_genes': int(len(df))}
    ax1.text(0.02, 0.97, f"{mem_label} median above trend: {stats['scg_resid_median']:+.3f}",
             transform=ax1.transAxes, va='top', fontsize=fontsize - 3, color='crimson')

    fig.suptitle(title or 'Baseline expression vs perturbation response', fontsize=fontsize + 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return stats


def cluster_memory_response_profiles(
    lfc_df, genes, response_class=None, metric='cosine', n_neighbors=15,
    resolution=1.0, random_seed=0, fontsize=13, figsize=(15, 6),
    title=None, save_path=None, show=True,
):
    """
    Cluster memory genes by their perturbation-response profiles m_i = (LFC_i1..LFC_in)
    to find groups responding to the same perturbations. Builds a cosine kNN graph over
    the gene x KO LFC matrix and runs Leiden + UMAP.

    Shows a UMAP coloured by Leiden cluster (left) and, if `response_class` is given
    (from gene_response_profile), by broad/selective/weak (right) -- so you can see
    whether the response-profile clusters ARE the functional classes.

    Returns (labels Series, AnnData with the embedding).
    """
    import scanpy as sc
    import anndata as ad

    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    M = lfc_df[cols].T.fillna(0.0)                          # genes x KOs
    A = ad.AnnData(M.values.astype('float32'))
    A.obs_names = list(M.index)
    sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep='X', metric=metric,
                    random_state=random_seed)
    try:
        sc.tl.leiden(A, resolution=resolution, random_state=random_seed,
                     flavor='igraph', n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(A, resolution=resolution, random_state=random_seed)
    sc.tl.umap(A, random_state=random_seed)

    labels = pd.Series(A.obs['leiden'].astype(str).values, index=M.index, name='cluster')
    umap = A.obsm['X_umap']

    ncol = 2 if response_class is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize if ncol == 2 else (7.5, 6))
    axes = np.atleast_1d(axes)

    for c in sorted(labels.unique(), key=lambda s: int(s)):
        m = labels.values == c
        axes[0].scatter(umap[m, 0], umap[m, 1], s=28, alpha=0.8, label=f'{c} (n={int(m.sum())})')
    axes[0].set_title(f'Leiden clusters (resolution={resolution})', fontsize=fontsize + 1)
    axes[0].legend(fontsize=fontsize - 5, markerscale=1.3, ncol=2, loc='best')
    axes[0].set_xlabel('UMAP1', fontsize=fontsize); axes[0].set_ylabel('UMAP2', fontsize=fontsize)
    axes[0].tick_params(labelsize=fontsize - 3)

    if response_class is not None:
        rc = response_class.reindex(M.index)
        pal = {'broad': 'teal', 'selective': 'darkorange', 'weak': 'lightgray'}
        for cls, col in pal.items():
            m = (rc.values == cls)
            axes[1].scatter(umap[m, 0], umap[m, 1], s=28, color=col, alpha=0.8, label=cls)
        axes[1].set_title('Coloured by response class', fontsize=fontsize + 1)
        axes[1].legend(fontsize=fontsize - 3, loc='best')
        axes[1].set_xlabel('UMAP1', fontsize=fontsize); axes[1].set_ylabel('UMAP2', fontsize=fontsize)
        axes[1].tick_params(labelsize=fontsize - 3)

    fig.suptitle(title or 'Memory genes clustered by KO-response profile', fontsize=fontsize + 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return labels, A


def plot_cluster_ko_signature(
    lfc_df, labels, value='signed', top_kos=40, min_cluster=3,
    cluster_kos=True, vmax=None, fontsize=9, figsize=(14, 6),
    title=None, save_path=None, show=True,
):
    """
    Given gene->cluster labels (from cluster_memory_response_profiles OR from any
    hierarchical cut), show WHICH set of KOs defines each cluster: a cluster x KO
    heatmap of the mean response, so you can read off "these genes all respond to
    this set of perturbations". Answers Ida's actual question, past the UMAP.

    value : 'signed' = mean signed LFC per (cluster, KO)  (red up / blue down)
            'abs'    = mean |LFC| per (cluster, KO)        (magnitude of response)
    top_kos     : keep the N KOs with the most variance ACROSS clusters (the ones
                  that actually separate clusters); None keeps all.
    min_cluster : drop clusters smaller than this (usually singletons/noise).
    cluster_kos : hierarchically order the KO columns so co-responding KOs sit together.

    Returns the (cluster x KO) mean-response DataFrame, columns ordered as plotted.
    """
    labels = pd.Series(labels).astype(str)
    genes = lfc_df.columns[lfc_df.columns.isin(set(labels.index))]
    M = lfc_df[genes].T                                    # genes x KOs
    labels = labels.reindex(M.index).dropna()
    M = M.loc[labels.index]

    keep = labels.value_counts()
    keep = keep[keep >= min_cluster].index
    M, labels = M.loc[labels.isin(keep)], labels[labels.isin(keep)]

    signed = M.groupby(labels).mean()                      # cluster x KO, mean signed LFC
    mat = signed.abs() if value == 'abs' else signed
    mat = mat.reindex(sorted(mat.index, key=lambda s: int(s) if s.isdigit() else s))

    if top_kos is not None and mat.shape[1] > top_kos:
        sel = mat.var(axis=0).sort_values(ascending=False).index[:top_kos]
        mat = mat[sel]

    if cluster_kos and mat.shape[1] > 2:
        from scipy.cluster.hierarchy import leaves_list
        Zc = linkage(mat.T.fillna(0.0).values, method='average')
        order = mat.columns[leaves_list(Zc)]
        mat = mat[order]

    counts = labels.value_counts()
    yl = [f'{c} (n={counts[c]})' for c in mat.index]
    center = 0 if value == 'signed' else None
    cmap = 'RdBu_r' if value == 'signed' else 'magma'
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(mat.values), 98)) or 1.0
    vmin = -vmax if value == 'signed' else 0

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat.values, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                yticklabels=yl, xticklabels=list(mat.columns),
                cbar_kws={'label': ('mean signed LFC' if value == 'signed'
                                    else 'mean |LFC|')})
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=fontsize - 2)
    ax.set_yticklabels(yl, rotation=0, fontsize=fontsize)
    ax.set_xlabel(f'KO ({mat.shape[1]} most cluster-discriminating)', fontsize=fontsize + 1)
    ax.set_ylabel('gene cluster', fontsize=fontsize + 1)
    ax.set_title(title or 'Response signature per memory-gene cluster',
                 fontsize=fontsize + 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()

    for c in mat.index:                                    # top driver KOs per cluster
        drivers = signed.loc[c, mat.columns].abs().sort_values(ascending=False).index[:6]
        print(f"cluster {c} (n={counts[c]}): "
              + ", ".join(f"{k}{signed.loc[c, k]:+.2f}" for k in drivers))
    return mat


def plot_covariance_vs_response(
    cov_df, gene_summary, memory_genes, cov_metric='cov_perturbed',
    resp_metric='mean_abs_lfc', mem_label='SCG', scg_size=35, n_ebins=25,
    clip_pct=(1, 99), fontsize=14, figsize=(9, 7), title=None, save_path=None, show=True,
):
    """
    Fluctuation-dissipation view: baseline covariance-with-perturbed-genes (x, at rest)
    vs perturbation responsiveness (y = mean |LFC|), one dot per gene. A positive trend
    = equilibrium covariance predicts response (FDT). SCGs highlighted; sitting in the
    high-covariance / high-response corner (and above the trend) means covariance
    explains their responsiveness. Returns stats (Pearson r; SCG median above trend).
    """
    x = cov_df[cov_metric]
    y = gene_summary[resp_metric]
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    is_scg = df.index.isin(set(memory_genes))
    xv, yv = df['x'].values, df['y'].values
    r = float(np.corrcoef(xv, yv)[0, 1])

    lo, hi = np.percentile(xv, clip_pct)
    bins = np.linspace(lo, hi, n_ebins)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(xv, bins) - 1, 0, len(centers) - 1)
    med = np.array([np.median(yv[idx == b]) if np.any(idx == b) else np.nan
                    for b in range(len(centers))])
    ok = ~np.isnan(med)
    trend_at = np.interp(xv, centers[ok], med[ok])
    resid = yv - trend_at

    fig, ax = plt.subplots(figsize=figsize)
    ax.hexbin(xv[~is_scg], yv[~is_scg], gridsize=55, cmap='Greys', bins='log', mincnt=1)
    ax.scatter(xv[is_scg], yv[is_scg], s=scg_size, color='crimson', alpha=0.6,
               edgecolor='black', linewidth=0.3, label=f'{mem_label} (n={int(is_scg.sum())})')
    ax.plot(centers[ok], med[ok], color='black', lw=2.5, label='median (all genes)')
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f'baseline {cov_metric} with perturbed genes (rest)', fontsize=fontsize)
    ax.set_ylabel(f'responsiveness  ({resp_metric})', fontsize=fontsize)
    ax.set_title(title or f'Fluctuation–dissipation: covariance vs response  '
                          f'(Pearson r = {r:.2f})', fontsize=fontsize + 1)
    ax.text(0.02, 0.97, f"{mem_label} median above trend: {np.median(resid[is_scg]):+.3f}",
            transform=ax.transAxes, va='top', color='crimson', fontsize=fontsize - 3)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return {'pearson_r': r, 'scg_resid_median': float(np.median(resid[is_scg])),
            'nonscg_resid_median': float(np.median(resid[~is_scg])), 'n_scg': int(is_scg.sum())}


def plot_covariance_vs_random(
    cov_df, memory_genes, background_genes, gene_expr=None, metric='cov_perturbed',
    quantity_label='covariance with perturbed genes', n_perm=1000, n_bins=10,
    agg='mean', mem_label='SCG', random_seed=0,
    fontsize=14, figsize=(9, 6), title=None, save_path=None, show=True,
):
    """
    Ben's question, directly: do SCGs have higher AVERAGE covariance with the perturbed
    genes than randomly selected non-SCGs? Compares the SCG mean(/median) covariance to
    the distribution of the same statistic over `n_perm` random same-size non-SCG sets.
    If gene_expr is given, the random sets are EXPRESSION-MATCHED (the fair version).

    One histogram: null distribution (grey) + SCG value (red line), annotated with how
    many random sets the SCG value beats, plus a permutation z-score and p.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(random_seed)
    v = cov_df[metric].dropna()
    measured = v.index
    mem = [g for g in memory_genes if g in measured]
    aggf = np.mean if agg == 'mean' else np.median
    obs = float(aggf(v[mem].values))

    null = np.empty(n_perm)
    matched = gene_expr is not None
    kind = 'expression-matched' if matched else 'random'
    if matched:
        for i in range(n_perm):
            samp = _match_expression(mem, background_genes, gene_expr, measured, n_bins=n_bins, rng=rng)
            null[i] = aggf(v[samp].values)
    else:
        pool = np.array([g for g in background_genes if g in measured and g not in set(mem)])
        for i in range(n_perm):
            samp = rng.choice(pool, size=len(mem), replace=False)
            null[i] = aggf(v[samp].values)

    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else np.nan
    p = float(norm.sf(z))
    beats = int((null < obs).sum())
    verdict = 'HIGHER' if obs > mu else 'LOWER'

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(null, bins=40, color='lightgray', edgecolor='none',
            label=f'{n_perm} {kind} non-{mem_label} sets')
    ax.axvline(mu, color='gray', ls=':', lw=2, label=f'{kind} sets: average = {mu:.4f}')
    ax.axvline(obs, color='crimson', ls='--', lw=2.5, label=f'{mem_label}: average = {obs:.4f}')
    ax.set_xlabel(f'{agg} {quantity_label}', fontsize=fontsize)
    ax.set_ylabel(f'number of {kind} gene sets', fontsize=fontsize)
    ax.set_title(title or (f'Do {mem_label} have higher {quantity_label} than {kind} sets?\n'
                           f'{mem_label} are {verdict}: {obs:.4f} vs {mu:.4f}  '
                           f'(above {beats}/{n_perm} sets, z = {z:.1f})'),
                 fontsize=fontsize)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return {'observed': obs, 'null_mean': mu, 'z': float(z), 'p': p,
            'beats': f'{beats}/{n_perm}', 'ratio': obs / mu if mu else np.nan}


def plot_gene_leaveone_contribution(
    lfc_df, memory_genes, agg='median', top_n=30, fontsize=13, figsize=(9, 10),
    title=None, save_path=None, show=True,
):
    """
    Which individual memory genes prop up the elevated responsiveness? For each memory
    gene x, contribution = agg(all memory genes) - agg(memory genes without x), where
    agg is the median (or mean) of per-gene responsiveness (mean |LFC| across KOs).
    Positive = removing x LOWERS the aggregate (x was pulling it up). Ranks genes by
    contribution and shows the top_n. Returns the full contribution Series.
    """
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    resp = lfc_df[mem].abs().mean(axis=0)                   # per-gene responsiveness
    vals = resp.values
    aggf = np.median if agg == 'median' else np.mean
    all_agg = float(aggf(vals))

    contrib = pd.Series(
        {g: all_agg - float(aggf(np.delete(vals, i))) for i, g in enumerate(mem)},
        name='contribution').sort_values(ascending=False)

    top = contrib.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top)), top.values, color='crimson', alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=fontsize - 3)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel(f'contribution to {agg} responsiveness\n'
                  f'[{agg}(all) − {agg}(without gene)]', fontsize=fontsize)
    ax.set_title(title or f'Which memory genes prop up the {agg} responsiveness?',
                 fontsize=fontsize + 1)
    ax.tick_params(axis='x', labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return contrib


def plot_leaveout_by_contribution(
    lfc_df, memory_genes, background_genes, agg='median', label_n=10,
    fontsize=13, figsize=(12, 6), title=None, save_path=None, show=True,
):
    """
    Leave-out curve, but genes are removed in order of their INDIVIDUAL contribution
    to the aggregate responsiveness [agg(all) - agg(without gene)], biggest first -- so
    the largest steps are front-loaded. The top `label_n` contributors are labelled on
    the curve, and the non-memory median is drawn for reference. Returns the per-gene
    contribution Series (descending).
    """
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    resp = lfc_df[mem].abs().mean(axis=0)                    # per-gene responsiveness
    aggf = np.median if agg == 'median' else np.mean
    all_agg = float(aggf(resp.values))

    contrib = pd.Series(
        {g: all_agg - float(aggf(resp.drop(g).values)) for g in mem}
    ).sort_values(ascending=False)
    order = list(contrib.index)                             # biggest contributor first

    ks = list(range(len(order) - 1))
    curve = [float(aggf(resp[order[k:]].values)) for k in ks]

    pool = [g for g in background_genes if g in measured and g not in set(mem)]
    bg = float(np.median(lfc_df[pool].abs().mean(axis=0).values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, curve, '-o', color='crimson', ms=3, label='memory (biggest contributor removed first)')
    ax.axhline(bg, color='gray', lw=2, label='non-memory median')
    for i in range(min(label_n, len(order))):
        ax.annotate(order[i], (ks[i], curve[i]), fontsize=fontsize - 5, rotation=40,
                    xytext=(2, 6), textcoords='offset points', color='darkred')
    ax.set_xlabel('number of top-contributing memory genes removed', fontsize=fontsize)
    ax.set_ylabel(f'{agg} responsiveness (mean |LFC|)\nof remaining memory genes', fontsize=fontsize)
    ax.set_title(title or 'Leave-out ordered by contribution (biggest first)', fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return contrib


def plot_contribution_scatter(
    lfc_df, memory_genes, agg='median', label_n=15, fontsize=13, figsize=(11, 7),
    title=None, save_path=None, show=True,
):
    """
    Scatter to identify which memory genes prop up the elevated responsiveness.
    x = each gene's own responsiveness (mean |LFC| across KOs).
    y = its contribution to the aggregate = agg(all) - agg(without that gene).
    The top `label_n` contributors are highlighted/labelled. Returns the contribution
    Series (descending).
    """
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    resp = lfc_df[mem].abs().mean(axis=0)
    aggf = np.median if agg == 'median' else np.mean
    all_agg = float(aggf(resp.values))
    contrib = pd.Series({g: all_agg - float(aggf(resp.drop(g).values)) for g in mem},
                        name='contribution')

    top = contrib.sort_values(ascending=False).head(label_n).index
    is_top = contrib.index.isin(top)
    r = float(np.corrcoef(resp.values, contrib.values)[0, 1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(resp[~is_top], contrib[~is_top], s=25, color='lightgray', alpha=0.6, label='memory genes')
    ax.scatter(resp[is_top], contrib[is_top], s=45, color='crimson', alpha=0.8,
               edgecolor='black', linewidth=0.3, label=f'top {label_n} contributors')
    for g in top:
        ax.annotate(g, (resp[g], contrib[g]), fontsize=fontsize - 5, fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('gene responsiveness  (mean |LFC| across KOs)', fontsize=fontsize)
    ax.set_ylabel(f'contribution to {agg} responsiveness\n[{agg}(all) − {agg}(without gene)]',
                  fontsize=fontsize)
    ax.set_title(title or f'Which memory genes drive the {agg} responsiveness?  (r = {r:.2f})',
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return contrib.sort_values(ascending=False)


def plot_median_without_scatter(
    lfc_df, memory_genes, label_n=15, fontsize=13, figsize=(11, 7),
    title=None, save_path=None, show=True,
):
    """
    'With vs without' view: for each memory gene, y = median responsiveness of the set
    WITHOUT that gene; the horizontal line = median responsiveness WITH all genes. Genes
    whose 'without' median falls BELOW the line were propping the median up (removing
    them lowers it). x = the gene's own responsiveness (so dots spread out). The
    strongest contributors (lowest y) are labelled. Returns median_without per gene.
    """
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    resp = lfc_df[mem].abs().mean(axis=0)
    med_all = float(np.median(resp.values))
    without = pd.Series({g: float(np.median(resp.drop(g).values)) for g in mem},
                        name='median_without')

    lowest = without.sort_values().head(label_n).index          # biggest contributors
    is_low = without.index.isin(lowest)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(resp[~is_low], without[~is_low], s=25, color='lightgray', alpha=0.6,
               label='memory genes')
    ax.scatter(resp[is_low], without[is_low], s=45, color='crimson', alpha=0.8,
               edgecolor='black', linewidth=0.3, label=f'top {label_n} contributors')
    ax.axhline(med_all, color='black', lw=2, ls='--',
               label=f'median WITH all genes = {med_all:.3f}')
    for g in lowest:
        ax.annotate(g, (resp[g], without[g]), fontsize=fontsize - 5, fontweight='bold',
                    xytext=(3, -3), textcoords='offset points')
    ax.set_xlabel('gene responsiveness  (mean |LFC| across KOs)', fontsize=fontsize)
    ax.set_ylabel('median responsiveness of memory set\nWITHOUT that gene', fontsize=fontsize)
    ax.set_title(title or 'Which memory genes prop up the median? (below the line = drivers)',
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return without.sort_values()


def plot_ranked_responsiveness(
    lfc_df, memory_genes, background_genes, label_n=15, fontsize=13, figsize=(12, 6),
    title=None, save_path=None, show=True,
):
    """
    All memory genes ranked by their own responsiveness (mean |LFC| across KOs), with
    the non-memory median as a reference line. Fraction of memory genes ABOVE the line
    = the robustness answer (many genes elevated, not a few); the top `label_n` genes
    are labelled = the strongest contributors. Returns responsiveness Series (desc).
    """
    measured = lfc_df.columns
    mem = [g for g in memory_genes if g in measured]
    resp = lfc_df[mem].abs().mean(axis=0).sort_values(ascending=False)
    pool = [g for g in background_genes if g in measured and g not in set(mem)]
    bg_med = float(np.median(lfc_df[pool].abs().mean(axis=0).values))

    above = int((resp.values > bg_med).sum())
    ranks = np.arange(len(resp))

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(ranks, bg_med, resp.values, where=resp.values > bg_med,
                    color='crimson', alpha=0.25)
    ax.plot(ranks, resp.values, color='crimson', lw=2, label='memory genes (ranked)')
    ax.axhline(bg_med, color='gray', lw=2, ls='--', label=f'non-memory median = {bg_med:.3f}')
    for i in range(min(label_n, len(resp))):
        ax.annotate(resp.index[i], (ranks[i], resp.values[i]), fontsize=fontsize - 6,
                    rotation=40, xytext=(2, 4), textcoords='offset points', color='darkred')
    ax.set_xlabel('memory genes ranked by responsiveness', fontsize=fontsize)
    ax.set_ylabel('responsiveness  (mean |LFC| across KOs)', fontsize=fontsize)
    ax.set_title(title or (f'{above}/{len(resp)} memory genes ({100*above/len(resp):.0f}%) '
                           f'above the non-memory median'), fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return resp


def plot_within_group_vs_random(
    adata, memory_genes, background_genes, gene_expr=None, kind='correlation',
    n_perm=1000, n_bins=10, do_normalize=True, mem_label='memory genes',
    random_seed=0, fontsize=14, figsize=(9, 6), title=None, save_path=None, show=True,
):
    """
    Ben (v2): do SCGs covary more WITH EACH OTHER (as a group) than random gene groups?
    Computes the average pairwise correlation (kind='correlation') or covariance
    (kind='covariance') AMONG the memory genes, in resting cells, and compares it to the
    same statistic over n_perm random same-size sets. If gene_expr is given, the random
    sets are expression-matched. Uses the meta-gene identity (fast, no full k x k matrix).

    Returns a stats dict (observed, null_mean, z, p, beats, ratio).
    """
    import scanpy as sc
    import scipy.sparse as sp
    from scipy.stats import norm

    rng = np.random.default_rng(random_seed)
    ad = adata.copy()
    if do_normalize:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    X = X.astype(np.float32)
    mu, sd = X.mean(0), X.std(0)
    genes = np.asarray(ad.var_names)
    gidx = {g: i for i, g in enumerate(genes)}

    if kind == 'correlation':
        Z = (X - mu) / (sd + 1e-9)

        def avg_pair(idx):
            k = len(idx)
            zbar = Z[:, idx].mean(1)
            return (zbar.var() * k * k - k) / (k * (k - 1))
        var_per = None
    else:
        def avg_pair(idx):
            k = len(idx)
            xbar = X[:, idx].mean(1)
            return (xbar.var() * k * k - (sd[idx] ** 2).sum()) / (k * (k - 1))
        var_per = sd ** 2

    mem = [g for g in memory_genes if g in gidx]
    obs = float(avg_pair([gidx[g] for g in mem]))

    matched = gene_expr is not None
    null = np.empty(n_perm)
    if matched:
        for i in range(n_perm):
            samp = _match_expression(mem, background_genes, gene_expr, genes, n_bins=n_bins, rng=rng)
            null[i] = avg_pair([gidx[g] for g in samp])
    else:
        pool = np.array([g for g in background_genes if g in gidx and g not in set(mem)])
        for i in range(n_perm):
            samp = rng.choice(pool, size=len(mem), replace=False)
            null[i] = avg_pair([gidx[g] for g in samp])

    kind_lab = 'expression-matched' if matched else 'random'
    mu_n, sd_n = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu_n) / sd_n if sd_n > 0 else np.nan
    p = float(norm.sf(z))
    beats = int((null < obs).sum())
    verdict = 'HIGHER' if obs > mu_n else 'LOWER'

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(null, bins=40, color='lightgray', edgecolor='none',
            label=f'{n_perm} {kind_lab} gene sets')
    ax.axvline(mu_n, color='gray', ls=':', lw=2, label=f'{kind_lab} sets: average = {mu_n:.4f}')
    ax.axvline(obs, color='crimson', ls='--', lw=2.5, label=f'{mem_label}: average = {obs:.4f}')
    ax.set_xlabel(f'average within-group {kind} (resting cells)', fontsize=fontsize)
    ax.set_ylabel(f'number of {kind_lab} gene sets', fontsize=fontsize)
    ax.set_title(title or (f'Do {mem_label} covary more with each other than {kind_lab} sets?\n'
                           f'{mem_label} are {verdict}: {obs:.4f} vs {mu_n:.4f}  '
                           f'(above {beats}/{n_perm}, z = {z:.1f})'), fontsize=fontsize)
    ax.legend(fontsize=fontsize - 3)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return {'observed': obs, 'null_mean': mu_n, 'z': float(z), 'p': p,
            'beats': f'{beats}/{n_perm}', 'ratio': obs / mu_n if mu_n else np.nan}


# ---------------------------------------------------------------------------
# Multi-resolution response-profile clustering, per method, genes OR KOs
# (Ida: same plot per method on the top-N genes, several resolutions, plus the
#  transposed view where each dot is a gene KO.)
# ---------------------------------------------------------------------------

def cluster_response_profiles(
    lfc_df, genes, axis='gene', resolutions=(0.25, 0.5, 0.75, 1.0),
    metric='cosine', n_neighbors=15, random_seed=0, kos=None, verbose=True,
):
    """
    Cluster response profiles at SEVERAL Leiden resolutions on ONE shared kNN
    graph / UMAP, so the resolutions are directly comparable (same coordinates,
    only the colouring changes).

    axis='gene' : dots are memory genes, features are the KOs -- which genes
                  respond to the same set of perturbations.
    axis='ko'   : dots are KO targets, features are the memory genes -- which
                  perturbations elicit a similar response across the memory set.

    kos : optional subset of lfc_df rows (KO targets) to use, as features when
          axis='gene' and as the dots themselves when axis='ko'.

    Returns (table, AnnData). `table` is indexed by gene / KO and holds UMAP1,
    UMAP2 and one `cluster_res<r>` column per resolution.
    """
    import scanpy as sc
    import anndata as ad

    cols = lfc_df.columns[lfc_df.columns.isin(set(genes))]
    rows = lfc_df.index if kos is None else lfc_df.index[lfc_df.index.isin(set(kos))]
    sub = lfc_df.loc[rows, cols]
    M = (sub.T if axis == 'gene' else sub).fillna(0.0)      # dots x features
    if verbose:
        print(f"axis={axis}: {M.shape[0]} dots x {M.shape[1]} features")

    A = ad.AnnData(M.values.astype('float32'))
    A.obs_names = [str(x) for x in M.index]
    sc.pp.neighbors(A, n_neighbors=min(n_neighbors, max(2, A.n_obs - 1)),
                    use_rep='X', metric=metric, random_state=random_seed)
    sc.tl.umap(A, random_state=random_seed)

    table = pd.DataFrame(A.obsm['X_umap'], index=M.index, columns=['UMAP1', 'UMAP2'])
    for r in resolutions:
        key = f'leiden_{r}'
        try:
            sc.tl.leiden(A, resolution=r, key_added=key, random_state=random_seed,
                         flavor='igraph', n_iterations=2, directed=False)
        except TypeError:
            sc.tl.leiden(A, resolution=r, key_added=key, random_state=random_seed)
        table[f'cluster_res{r}'] = A.obs[key].astype(str).values
        if verbose:
            print(f"  resolution {r}: {table[f'cluster_res{r}'].nunique()} clusters")
    table.index.name = 'gene' if axis == 'gene' else 'ko'
    return table, A


def response_cluster_table(
    lfc_df, genes, axis='gene', resolutions=(0.25, 0.5, 0.75, 1.0), thr=0.5,
    broad_at=0.40, weak_peak=0.5, rank_order=None, kos=None, **kw,
):
    """
    cluster_response_profiles(...) + the matching per-dot statistics, i.e. the
    table to hand out as a spreadsheet.

    axis='gene' -> joins gene_response_profile (breadth / mean_abs_lfc /
                   peak_abs_lfc / bimodality / response_class).
    axis='ko'   -> joins rank_ko_drivers (n_react / frac_react / mean_abs_lfc /
                   mean_signed_lfc / direction) computed over `genes`.

    rank_order : the method's ranked gene list (best first); adds `method_rank`
                 and sorts by it, so the sheet reads in the method's own order.
    """
    table, A = cluster_response_profiles(
        lfc_df, genes, axis=axis, resolutions=resolutions, kos=kos, **kw)

    if axis == 'gene':
        stats = gene_response_profile(lfc_df, genes, thr=thr, broad_at=broad_at,
                                      weak_peak=weak_peak)
    else:
        stats = rank_ko_drivers(lfc_df, genes, thr=thr)

    out = table.join(stats)
    if rank_order is not None:
        rk = pd.Series({g: i + 1 for i, g in enumerate(rank_order)}, dtype='float')
        out.insert(0, 'method_rank', rk.reindex(out.index).values)
        out = out.sort_values('method_rank', na_position='last')
    return out, A


def cluster_membership_summary(table, resolutions=None, class_col='response_class',
                               max_members=None):
    """
    One row per (resolution, cluster) with the member names spelled out -- the
    sheet you actually read when the UMAP dots have no labels.
    """
    cols = ([f'cluster_res{r}' for r in resolutions] if resolutions is not None
            else [c for c in table.columns if c.startswith('cluster_res')])
    rows = []
    for col in cols:
        res = col.replace('cluster_res', '')
        for cl, idx in table.groupby(col, sort=True).groups.items():
            members = list(idx)
            rec = {'resolution': res, 'cluster': cl, 'n': len(members)}
            if class_col in table.columns:
                comp = table.loc[members, class_col].value_counts()
                for k in ('broad', 'selective', 'weak'):
                    rec[f'n_{k}'] = int(comp.get(k, 0))
            shown = members[:max_members] if max_members else members
            rec['members'] = ', '.join(map(str, shown))
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_profile_cluster_grid(
    table, resolutions=None, class_col='response_class', ncols=None, s=26,
    fontsize=12, figsize=None, title=None, save_path=None, show=True,
):
    """
    Same UMAP, one panel per Leiden resolution (+ a final panel coloured by
    response class / KO direction if `class_col` is present).
    """
    cols = ([f'cluster_res{r}' for r in resolutions] if resolutions is not None
            else [c for c in table.columns if c.startswith('cluster_res')])
    panels = list(cols) + ([class_col] if class_col and class_col in table.columns else [])
    ncols = ncols or min(len(panels), 3)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (5.2 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    x, y = table['UMAP1'].values, table['UMAP2'].values
    pal = {'broad': 'teal', 'selective': 'darkorange', 'weak': 'lightgray',
           'up': 'firebrick', 'down': 'steelblue'}

    for ax, col in zip(axes, panels):
        vals = table[col].astype(str)
        try:
            order = sorted(vals.unique(), key=int)
        except ValueError:
            order = [k for k in pal if k in set(vals)] or sorted(vals.unique())
        for v in order:
            m = (vals.values == v)
            ax.scatter(x[m], y[m], s=s, alpha=0.8, color=pal.get(v),
                       label=f'{v} (n={int(m.sum())})')
        ax.set_title(col.replace('cluster_res', 'Leiden resolution ')
                     if col.startswith('cluster_res') else col.replace('_', ' '),
                     fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 5, markerscale=1.2, ncol=2, loc='best')
        ax.set_xlabel('UMAP1', fontsize=fontsize)
        ax.set_ylabel('UMAP2', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 3)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return fig


def plot_labeled_umap(
    table, cluster_col='cluster_res1.0', label_top=6, label_by='mean_abs_lfc',
    labels=None, s=42, fontsize=12, label_fontsize=7, figsize=(11, 9),
    title=None, save_path=None, show=True,
):
    """
    One large UMAP with the dots NAMED -- the top `label_top` members per cluster
    ranked by `label_by` (or exactly the names in `labels`).
    """
    x, y = table['UMAP1'].values, table['UMAP2'].values
    vals = table[cluster_col].astype(str)
    try:
        order = sorted(vals.unique(), key=int)
    except ValueError:
        order = sorted(vals.unique())

    fig, ax = plt.subplots(figsize=figsize)
    for v in order:
        m = (vals.values == v)
        ax.scatter(x[m], y[m], s=s, alpha=0.85, label=f'{v} (n={int(m.sum())})')

    if labels is not None:
        show_idx = [g for g in labels if g in table.index]
    elif label_top and label_by in table.columns:
        show_idx = (table.groupby(cluster_col, sort=False)[label_by]
                    .nlargest(label_top).index.get_level_values(-1).tolist())
    else:
        show_idx = list(table.index)

    for g in show_idx:
        ax.annotate(str(g), (table.at[g, 'UMAP1'], table.at[g, 'UMAP2']),
                    fontsize=label_fontsize, xytext=(3, 3),
                    textcoords='offset points', alpha=0.9)

    ax.set_title(title or f'Response-profile UMAP ({cluster_col})', fontsize=fontsize + 2)
    ax.legend(fontsize=fontsize - 4, markerscale=1.2, ncol=2, loc='best')
    ax.set_xlabel('UMAP1', fontsize=fontsize); ax.set_ylabel('UMAP2', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"saved: {save_path}")
    if show:
        plt.show()
    return fig


def write_cluster_workbook(sheets, path, verbose=True):
    """
    Write {sheet_name: DataFrame} to one .xlsx. If neither openpyxl nor
    xlsxwriter is installed, fall back to one CSV per sheet in a sibling folder
    (so the run never dies just because of the Excel engine).
    """
    engine = None
    for cand in ('openpyxl', 'xlsxwriter'):
        try:
            __import__(cand); engine = cand; break
        except ImportError:
            continue

    if engine is None:
        base = os.path.splitext(path)[0]
        os.makedirs(base, exist_ok=True)
        for name, df in sheets.items():
            df.to_csv(os.path.join(base, f'{name}.csv'))
        if verbose:
            print(f"no openpyxl/xlsxwriter in this env -> wrote {len(sheets)} CSVs to {base}")
        return base

    with pd.ExcelWriter(path, engine=engine) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=str(name)[:31])
    if verbose:
        print(f"saved: {path}  ({len(sheets)} sheets, engine={engine})")
    return path
