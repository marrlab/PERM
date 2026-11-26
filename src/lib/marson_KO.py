"""
marson_KO.py - knockout analysis on the Marson perturb-seq dataset.

First-pass question: when we knock out *any* gene, do SCGs (sister-concordant
genes, treated here as *measured features*) change more or less than other
genes?

Pipeline:
  1. annotate_scg_genes(adata, scg_list)         -> adata.var['is_scg']
  2. compute_pseudobulk_lfc(adata, ...)           -> DataFrame[KOs x genes] of LFC vs control
  3. summarize_gene_response(lfc_df)              -> per-gene |LFC| summary stats
  4. compare_gene_responsiveness(summary, scgs)   -> SCG vs Non-SCG test (MWU + bootstrap)
  5. plot_gene_responsiveness(results)            -> figure
  6. compute_per_ko_scg_bias(lfc_df, scgs)        -> per-KO mean |LFC| over SCG vs Non-SCG
  7. plot_per_ko_scg_bias(per_ko_df)              -> figure

Notes on LFC math:
  - We compute pseudobulk (mean expression per KO group) via one sparse mat-mul.
  - If adata.X looks log-normalized (auto-detected by max value), LFC is the
    mean-difference in log-space, divided by ln(2) to report in approximate
    log2 units. This is the standard log1p shortcut.
  - If adata.X looks like raw / normalized counts, LFC = log2((mean_KO + eps) /
    (mean_ctrl + eps)) using a small pseudocount.
  - LFC[X, X] is masked to NaN: the KO'd gene's own collapse is a technical
    artifact, not a phenotypic response.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm

KO_PLOTS_DIR = f'{P.PLOTS}/KO_analysis'
os.makedirs(KO_PLOTS_DIR, exist_ok=True)

# Memory-gene benchmark: four ranked method outputs for the T-lineage dataset.
# Each CSV is pre-ranked (best gene first) and has a 'timepoint' column (T1/T2).
MEMGENE_DIR = f'{P.MEMGENES_ROOT}/data/T-lineage'
# FALLBACK ONLY: figures.py passes its own METHOD_FILES to every call. Kept in step
# with it (2026-09-16): the four methods used in the paper, SCG = the clones 2-8 run.
# Was CORAL/GEMLI/MOLD/SCG with the canonical SCG, i.e. it had the excluded Mold
# METHOD and was missing Power-Seek entirely.
METHOD_FILES = {
    'CORAL': 'coral_memory_genes_T1_T2.csv',
    'GEMLI': 'gemli_memory_genes_T1_T2.csv',
    'Power-Seek': 'powerseek_memory_genes_T1_T2.csv',
    'SCG':   'scg_memory_genes_clones2-8_T1_T2.csv',
}


def annotate_scg_genes(adata, scg_strict_list):
    """
    Mark measured genes as SCG / Non-SCG in adata.var (in place).

    Parameters
    ----------
    adata : AnnData
        Expression object. adata.var_names should be gene symbols matching the
        names in scg_strict_list.
    scg_strict_list : list / array-like of str
        SCG gene symbols.

    Returns
    -------
    adata (modified in place, also returned for chaining).
    """
    is_scg = adata.var_names.isin(scg_strict_list)
    adata.var['is_scg'] = is_scg
    n_scg = int(is_scg.sum())
    n_total = adata.n_vars
    print(
        f"Annotated {n_total:,} measured genes: "
        f"{n_scg:,} SCG, {n_total - n_scg:,} Non-SCG."
    )
    return adata


def compute_pseudobulk_matrix(
    adata,
    control_label='control',
    perturbation_col='perturbation',
    eff_threshold=None,
    chunk_size=200_000,
):
    """
    Mean raw expression per perturbation group -- the expensive step.

    This is the only step that needs the full adata in memory. Cache the result
    with save_pseudobulk(); everything downstream (library-size normalisation,
    pseudocount choice, LFC, min-cells filtering) is then cheap to redo from the
    small cached table without re-reading the full matrix.

    Nothing is filtered or normalised here: ALL groups are returned, including
    the control and (if eff_threshold is used) an '__excluded__' group, so that
    downstream choices stay reversible.

    Returns
    -------
    pb : DataFrame [groups x genes], mean expression per cell, raw units.
    group_sizes : Series [groups], number of cells contributing to each group.
    """
    if perturbation_col not in adata.obs.columns:
        raise KeyError(f"adata.obs missing '{perturbation_col}'.")

    # Optional efficiency filter (controls always kept)
    cell_mask = np.ones(adata.n_obs, dtype=bool)
    pert_raw = adata.obs[perturbation_col].astype(str).values
    if eff_threshold is not None:
        if 'eff_score' not in adata.obs.columns:
            print("Warning: eff_threshold given but 'eff_score' not in obs. Skipping filter.")
        else:
            eff = adata.obs['eff_score'].fillna(-np.inf).values
            cell_mask = (eff >= eff_threshold) | (pert_raw == control_label)
            print(
                f"After eff_score >= {eff_threshold}: "
                f"{cell_mask.sum():,} / {adata.n_obs:,} cells kept "
                "(controls always included).", flush=True
            )

    pert = np.where(cell_mask, pert_raw, '__excluded__')
    categories, codes = np.unique(pert, return_inverse=True)
    n_groups = len(categories)
    n_cells = adata.n_obs

    if control_label not in categories:
        raise ValueError(f"Control label '{control_label}' not found in {perturbation_col}.")

    group_sizes = np.bincount(codes, minlength=n_groups)
    safe_sizes = np.where(group_sizes > 0, group_sizes, 1).astype(np.float32)

    n_genes = adata.n_vars
    sums = np.zeros((n_groups, n_genes), dtype=np.float32)

    n_chunks = int(np.ceil(n_cells / chunk_size))
    print(
        f"Accumulating pseudobulk in {n_chunks} chunks of <= {chunk_size:,} cells "
        f"({n_groups} groups x {n_genes} genes)...", flush=True
    )
    X = adata.X
    for start in tqdm(range(0, n_cells, chunk_size), total=n_chunks):
        end = min(start + chunk_size, n_cells)
        chunk_codes = codes[start:end]
        chunk_n = end - start
        chunk_ind = sp.csr_matrix(
            (np.ones(chunk_n, dtype=np.float32),
             (chunk_codes, np.arange(chunk_n))),
            shape=(n_groups, chunk_n),
        )
        chunk_sum = chunk_ind @ X[start:end]
        if sp.issparse(chunk_sum):
            chunk_sum = chunk_sum.toarray()
        sums += np.asarray(chunk_sum, dtype=np.float32)

    pb = pd.DataFrame(
        sums / safe_sizes[:, None],
        index=pd.Index(categories.astype(str), name='group'),
        columns=adata.var_names.astype(str),
    )
    sizes = pd.Series(group_sizes, index=pb.index, name='n_cells')
    print(f"Pseudobulk: {pb.shape[0]} groups x {pb.shape[1]} genes.", flush=True)
    return pb, sizes


def save_pseudobulk(pb, group_sizes, path):
    """Cache a pseudobulk table + group sizes to a compressed .npz (fast to reload)."""
    np.savez_compressed(
        path,
        values=pb.values.astype(np.float32),
        index=pb.index.values.astype(str),
        columns=pb.columns.values.astype(str),
        sizes=group_sizes.reindex(pb.index).values.astype(np.int64),
    )
    print(f"Saved pseudobulk to {path} ({pb.shape[0]} groups x {pb.shape[1]} genes).")


def load_pseudobulk(path):
    """Reload a pseudobulk cached by save_pseudobulk(). Returns (pb, group_sizes)."""
    z = np.load(path, allow_pickle=False)
    pb = pd.DataFrame(z['values'], index=pd.Index(z['index'], name='group'),
                      columns=z['columns'])
    sizes = pd.Series(z['sizes'], index=pb.index, name='n_cells')
    return pb, sizes


def lfc_from_pseudobulk(
    pb,
    group_sizes=None,
    min_cells_per_ko=20,
    control_label='control',
    normalize='group',
    cpm_target=1e6,
    pseudocount=1.0,
    assume_log_normalized=None,
    mask_self_ko=True,
):
    """
    LFC vs control from a (cached) pseudobulk table -- the cheap step.

    normalize : 'group' or None
        'group' : scale each group's summed profile to `cpm_target` (CPM) before
            the log2 ratio. This is the standard bulk/pseudobulk workflow: it
            removes per-KO sequencing-depth differences, and on a CPM scale a
            pseudocount of ~1 (= 1 count per million) is a real detection floor,
            so weakly-detected genes shrink toward 0 instead of exploding.
        None : use the raw per-cell means as-is. Reproduces the original
            behaviour; note that with raw counts + pseudocount=1e-3 the genes
            whose control mean sits near the pseudocount get a hugely inflated
            |LFC| (they looked ~5x more 'responsive' than well-expressed genes).

    Note: scaling summed counts weights each cell by its depth, which is not
    identical to per-cell normalize_total + log1p + mean (equal weight per cell).
    The depth-weighted version is the standard one for pseudobulk DE.
    """
    genes = pb.columns.astype(str)
    mat = pb.values.astype(np.float64)
    idx_str = pb.index.values.astype(str)

    if normalize == 'group':
        # n_cells cancels in the ratio, so CPM from the per-cell means is
        # identical to CPM from the summed counts.
        lib = mat.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        mat = mat / lib * cpm_target
        assume_log_normalized = False
        print(f"normalize='group': per-group CPM (target {cpm_target:g}); "
              f"max = {mat.max():.1f}; pseudocount = {pseudocount:g}.")
    elif normalize is not None and normalize is not False:
        raise ValueError(f"normalize must be 'group' or None, got {normalize!r}")

    if assume_log_normalized is None:
        max_val = float(mat.max())
        assume_log_normalized = max_val < 25
        print(f"Auto-detected log-normalization: {assume_log_normalized} "
              f"(max = {max_val:.2f}).")

    if control_label not in idx_str:
        raise ValueError(f"Control label '{control_label}' not in pseudobulk index.")
    control_idx = int(np.where(idx_str == control_label)[0][0])
    control_expr = mat[control_idx]

    if assume_log_normalized:
        # log1p(natural) -> mean-difference ~ ln-fold-change; convert to log2.
        lfc = (mat - control_expr[None, :]) / np.log(2.0)
    else:
        lfc = np.log2((mat + pseudocount) / (control_expr[None, :] + pseudocount))

    keep = np.ones(len(idx_str), dtype=bool)
    keep[control_idx] = False
    if '__excluded__' in idx_str:
        keep[int(np.where(idx_str == '__excluded__')[0][0])] = False
    if group_sizes is not None:
        keep &= (group_sizes.reindex(pb.index).values >= min_cells_per_ko)

    lfc_df = pd.DataFrame(lfc[keep], index=pd.Index(idx_str[keep], name='KO_Gene'),
                          columns=genes)

    if mask_self_ko:
        shared = lfc_df.index.intersection(lfc_df.columns)
        if len(shared) > 0:
            row_idx = lfc_df.index.get_indexer(shared)
            col_idx = lfc_df.columns.get_indexer(shared)
            lfc_df.values[row_idx, col_idx] = np.nan
            print(f"Masked self-KO LFC for {len(shared)} genes (set to NaN).")

    print(f"Returned LFC matrix: {lfc_df.shape[0]} KOs x {lfc_df.shape[1]} genes "
          f"(min {min_cells_per_ko} cells/KO).")
    return lfc_df


def compute_pseudobulk_lfc(
    adata,
    min_cells_per_ko=20,
    control_label='control',
    perturbation_col='perturbation',
    eff_threshold=None,
    pseudocount=1e-3,
    assume_log_normalized=None,
    mask_self_ko=True,
    chunk_size=200_000,
    normalize=None,
):
    """
    Per-KO pseudobulk vs control in one shot, returned as a log-fold-change frame.

    Thin wrapper over compute_pseudobulk_matrix() + lfc_from_pseudobulk(). Prefer
    calling those two directly and caching the pseudobulk with save_pseudobulk():
    the pseudobulk is the only step that needs the full adata, so caching it lets
    you re-do the normalisation / pseudocount / min-cells choices in seconds.

    Defaults reproduce the original raw-count behaviour (normalize=None,
    pseudocount=1e-3). Pass normalize='group' for CPM-normalised LFC, in which
    case you also want pseudocount=1.0.

    Returns
    -------
    DataFrame, shape (n_KOs, n_genes), index=KO gene name, columns=measured genes.
    """
    pb, sizes = compute_pseudobulk_matrix(
        adata,
        control_label=control_label,
        perturbation_col=perturbation_col,
        eff_threshold=eff_threshold,
        chunk_size=chunk_size,
    )
    return lfc_from_pseudobulk(
        pb,
        sizes,
        min_cells_per_ko=min_cells_per_ko,
        control_label=control_label,
        normalize=normalize,
        pseudocount=pseudocount,
        assume_log_normalized=assume_log_normalized,
        mask_self_ko=mask_self_ko,
    )


def summarize_gene_response(lfc_df, large_change_threshold=1.0):
    """
    Per-gene |LFC| summary across all KO experiments.

    Returns
    -------
    DataFrame indexed by gene with columns:
      - mean_abs_lfc   : mean of |LFC| across all KOs
      - median_abs_lfc : median of |LFC|
      - max_abs_lfc    : max |LFC| over KOs
      - frac_large     : fraction of KOs where |LFC| > large_change_threshold
      - std_lfc        : std of signed LFC across KOs
      - n_obs          : number of KOs contributing (excludes NaN self-KO entries)
    """
    abs_lfc = lfc_df.abs()
    summary = pd.DataFrame(index=lfc_df.columns)
    summary['mean_abs_lfc'] = abs_lfc.mean(axis=0, skipna=True)
    summary['median_abs_lfc'] = abs_lfc.median(axis=0, skipna=True)
    summary['max_abs_lfc'] = abs_lfc.max(axis=0, skipna=True)
    summary['frac_large'] = (abs_lfc > large_change_threshold).sum(axis=0) / abs_lfc.notna().sum(axis=0)
    summary['std_lfc'] = lfc_df.std(axis=0, skipna=True)
    summary['n_obs'] = abs_lfc.notna().sum(axis=0)
    summary.index.name = 'gene'
    return summary


def compare_gene_responsiveness(
    gene_summary,
    scg_genes,
    metric='mean_abs_lfc',
    n_bootstrap=1000,
    alternative='two-sided',
    random_seed=42,
    match_non_scg=False,
):
    """
    Compare SCG vs Non-SCG distributions of a per-gene responsiveness metric.

    - Mann-Whitney U test (alternative: 'two-sided', 'greater', 'less')
        where 'greater' means SCG metric > Non-SCG metric.
    - Bootstrap null: sample random non-SCG gene sets of size |SCG|, compute
        the group mean. Empirical p:
            'greater'   : P(null_mean >= scg_mean)
            'less'      : P(null_mean <= scg_mean)
            'two-sided' : P(|null_mean - non_scg_mean| >= |scg_mean - non_scg_mean|)

    Parameters
    ----------
    match_non_scg : bool, default False
        If True, draw a single random subset of Non-SCG genes equal in size to
        the SCG set (|SCG|), and use that balanced subset for the displayed
        distribution and the Mann-Whitney test. The full Non-SCG pool is still
        used to build the bootstrap null. Draw is reproducible via random_seed.
    """
    rng = np.random.default_rng(random_seed)

    is_scg = gene_summary.index.isin(scg_genes)
    scg_vals = gene_summary.loc[is_scg, metric].dropna().values
    non_scg_all = gene_summary.loc[~is_scg, metric].dropna().values

    if len(scg_vals) == 0 or len(non_scg_all) == 0:
        raise ValueError(
            f"Empty group: SCG={len(scg_vals)}, Non-SCG={len(non_scg_all)}."
        )

    n_scg = len(scg_vals)
    if n_scg > len(non_scg_all):
        raise ValueError(
            f"|SCG| ({n_scg}) exceeds |Non-SCG| ({len(non_scg_all)}) for bootstrap."
        )

    # Non-SCG set used for the displayed distribution + Mann-Whitney. When
    # match_non_scg is on, this is a random |SCG|-sized subset; the full pool
    # (non_scg_all) is kept for the bootstrap null below.
    if match_non_scg:
        non_scg_vals = rng.choice(non_scg_all, size=n_scg, replace=False)
    else:
        non_scg_vals = non_scg_all

    _, p_mwu = mannwhitneyu(scg_vals, non_scg_vals, alternative=alternative)

    null_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        null_means[i] = rng.choice(non_scg_all, size=n_scg, replace=False).mean()

    scg_mean = float(scg_vals.mean())
    non_scg_mean = float(non_scg_vals.mean())
    # Bootstrap null is drawn from the full Non-SCG pool, so center the two-sided
    # comparison on the full-pool mean (equals non_scg_mean when match_non_scg=False).
    non_scg_pool_mean = float(non_scg_all.mean())

    if alternative == 'greater':
        p_boot = float((null_means >= scg_mean).mean())
    elif alternative == 'less':
        p_boot = float((null_means <= scg_mean).mean())
    else:
        observed_diff = abs(scg_mean - non_scg_pool_mean)
        p_boot = float((np.abs(null_means - non_scg_pool_mean) >= observed_diff).mean())

    return {
        'metric': metric,
        'alternative': alternative,
        'scg_mean': scg_mean,
        'non_scg_mean': non_scg_mean,
        'scg_median': float(np.median(scg_vals)),
        'non_scg_median': float(np.median(non_scg_vals)),
        'n_scg': int(n_scg),
        'n_non_scg': int(len(non_scg_vals)),
        'mann_whitney_p': float(p_mwu),
        'bootstrap_p': p_boot,
        'null_means': null_means,
        'scg_values': scg_vals,
        'non_scg_values': non_scg_vals,
    }


def plot_gene_responsiveness(
    results,
    metric_label=None,
    figsize=(15, 6),
    save_path=None,
    show=True,
    style='violin',
    fontsize=18,
    scg_label='SCG gene',
    bg_label='Non-SCG gene',
    title=None,
):
    """
    Two-panel figure: per-gene distribution (left) + bootstrap null (right).

    Parameters
    ----------
    style : {'violin', 'ecd', 'density', 'boxen'}
        Left-panel rendering:
          - 'violin' : KDE violin + strip; original default.
          - 'ecdf'   : empirical CDF curves for SCG and Non-SCG. Best when group
                       sizes differ by orders of magnitude; visual gap between
                       curves at any x is the Kolmogorov-Smirnov statistic at
                       that point. K-S two-sided p is added to the title.
          - 'density': area-normalized KDE curves overlaid on the same x-axis.
                       Most intuitive shift-of-peak view; obscures sample-size
                       imbalance.
          - 'boxen'  : letter-value (boxen) plot; nested quantile boxes that
                       capture tail behavior better than violins/boxplots for
                       large samples.
    """
    if metric_label is None:
        metric_label = results['metric']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    scg_vals = results['scg_values']
    non_scg_vals = results['non_scg_values']
    order = [scg_label, bg_label]
    palette = {bg_label: 'steelblue', scg_label: 'crimson'}

    extra_title = ''
    if style == 'violin':
        plot_df = pd.DataFrame({
            'value': np.concatenate([scg_vals, non_scg_vals]),
            'Group': [scg_label] * len(scg_vals) + [bg_label] * len(non_scg_vals),
        })
        sns.violinplot(
            data=plot_df, x='Group', y='value', order=order,
            palette='magma', inner='box', ax=ax1, cut=0,
        )
        sns.stripplot(
            data=plot_df, x='Group', y='value', order=order,
            color='black', size=2, alpha=0.2, ax=ax1, jitter=0.25,
        )
        ax1.set_ylabel(metric_label)
        ax1.set_xlabel('')

    elif style == 'ecdf':
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(scg_vals, non_scg_vals)
        non_scg_sorted = np.sort(non_scg_vals)
        scg_sorted = np.sort(scg_vals)
        ax1.plot(
            non_scg_sorted,
            np.arange(1, len(non_scg_sorted) + 1) / len(non_scg_sorted),
            color=palette[bg_label], lw=2,
            label=f'{bg_label} (n={len(non_scg_sorted)})',
        )
        ax1.plot(
            scg_sorted,
            np.arange(1, len(scg_sorted) + 1) / len(scg_sorted),
            color=palette[scg_label], lw=2,
            label=f'{scg_label} (n={len(scg_sorted)})',
        )
        ax1.set_xlabel(metric_label)
        ax1.set_ylabel('Cumulative fraction of genes')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        extra_title = f'\nK-S stat = {ks_stat:.3f}, K-S p = {ks_p:.2e}'

    elif style == 'density':
        sns.kdeplot(
            non_scg_vals, ax=ax1, color=palette[bg_label], lw=2,
            fill=True, alpha=0.3, label=f'{bg_label} (n={len(non_scg_vals)})',
            common_norm=False, cut=0,
        )
        sns.kdeplot(
            scg_vals, ax=ax1, color=palette[scg_label], lw=2,
            fill=True, alpha=0.3, label=f'{scg_label} (n={len(scg_vals)})',
            common_norm=False, cut=0,
        )
        ax1.axvline(np.mean(non_scg_vals), color=palette[bg_label],
                    linestyle=':', lw=1)
        ax1.axvline(np.mean(scg_vals), color=palette[scg_label],
                    linestyle='--', lw=1)
        ax1.set_xlabel(metric_label)
        ax1.set_ylabel('Density (area = 1 per group)')
        ax1.legend(loc='best')

    elif style == 'boxen':
        plot_df = pd.DataFrame({
            'value': np.concatenate([scg_vals, non_scg_vals]),
            'Group': [scg_label] * len(scg_vals) + [bg_label] * len(non_scg_vals),
        })
        sns.boxenplot(
            data=plot_df, x='Group', y='value', order=order,
            palette=palette, ax=ax1,
        )
        ax1.set_ylabel(metric_label)
        ax1.set_xlabel('')

    else:
        raise ValueError(
            f"Unknown style='{style}'. Choose from 'violin', 'ecd', 'density', 'boxen'."
        )

    first_line = title if title is not None else f'{metric_label} per measured gene'
    ax1.set_title(
        f'{first_line}\n'
        f'Mann-Whitney ({results["alternative"]}) p = {results["mann_whitney_p"]:.2e}\n'
        f'n({scg_label})={results["n_scg"]}, n({bg_label})={results["n_non_scg"]}'
        f'{extra_title}'
    )

    ax2.hist(results['null_means'], bins=40, color='lightgray',
             edgecolor='black', alpha=0.85)
    ax2.axvline(results['scg_mean'], color='crimson', linestyle='--', linewidth=2,
                label=f'SCG group mean = {results["scg_mean"]:.3f}')
    ax2.axvline(results['non_scg_mean'], color='steelblue', linestyle=':', linewidth=1.5,
                label=f'Non-SCG overall mean = {results["non_scg_mean"]:.3f}')
    ax2.set_title(
        f'Bootstrap null: random {results["n_scg"]}-gene sets from Non-SCG\n'
        f'Empirical p ({results["alternative"]}) = {results["bootstrap_p"]:.4f}'
    )
    ax2.set_xlabel(f'Mean {metric_label} of random gene set')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='best')

    # Enlarge every text element on both panels (titles, axis labels, tick
    # labels, legends) so the figure stays readable in a slide / poster.
    for ax in (ax1, ax2):
        ax.title.set_fontsize(fontsize + 2)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(fontsize - 2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)


def plot_scg_response_gap_by_ko_context(
    lfc_df,
    scg_genes,
    metric='mean_abs_lfc',
    figsize=(7, 6),
    save_path=None,
    show=True,
):
    """
    Q3 summary: is the SCG-gene response advantage specific to SCG knockouts?

    Splits KOs into SCG-KOs and non-SCG-KOs, and for each context computes the
    gap = mean responsiveness of SCG genes - mean responsiveness of non-SCG genes
    (responsiveness = per-gene mean |LFC| across that KO context). Two roughly
    equal bars => the SCG advantage is intrinsic, not SCG-module-specific (a null
    for the "SCGs are a connected module" hypothesis).

    Returns dict with the two gaps and the per-context group means.
    """
    scg_set = set(scg_genes)
    is_scg_col = lfc_df.columns.isin(scg_set)
    if is_scg_col.sum() == 0 or (~is_scg_col).sum() == 0:
        raise ValueError("Need both SCG and non-SCG genes among measured features.")

    contexts = {
        'SCG knockouts': lfc_df.loc[lfc_df.index.isin(scg_set)],
        'Non-SCG knockouts': lfc_df.loc[~lfc_df.index.isin(scg_set)],
    }

    gaps, scg_means, non_means, n_kos = {}, {}, {}, {}
    for label, sub in contexts.items():
        abs_lfc = sub.abs()
        per_gene = abs_lfc.mean(axis=0, skipna=True)  # mean |LFC| per gene in this context
        scg_means[label] = float(per_gene[is_scg_col].mean())
        non_means[label] = float(per_gene[~is_scg_col].mean())
        gaps[label] = scg_means[label] - non_means[label]
        n_kos[label] = int(sub.shape[0])

    labels = list(contexts.keys())
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [f'{l}\n(n={n_kos[l]} KOs)' for l in labels],
        [gaps[l] for l in labels],
        color=['crimson', 'steelblue'], edgecolor='black', alpha=0.85,
    )
    for bar, l in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'+{gaps[l]:.4f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('SCG-gene minus Non-SCG-gene\nmean |LFC| (response advantage)')
    ratio = gaps[labels[0]] / gaps[labels[1]] if gaps[labels[1]] != 0 else float('nan')
    ax.set_title(
        'SCG response advantage is the same regardless of what is knocked out\n'
        f'ratio (SCG-KO / Non-SCG-KO context) = {ratio:.2f}x  => no SCG-specific module',
        fontsize=10,
    )
    ax.set_ylim(0, max(gaps.values()) * 1.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q3 gap figure saved to: {save_path}")
    if show:
        plt.show()

    return {
        'gap_scg_ko': gaps[labels[0]],
        'gap_non_scg_ko': gaps[labels[1]],
        'ratio': ratio,
        'scg_means': scg_means,
        'non_scg_means': non_means,
        'n_kos': n_kos,
    }


def compute_per_ko_scg_bias(lfc_df, scg_genes):
    """
    For each KO, compute the mean |LFC| over SCG genes vs over Non-SCG genes.

    Returns DataFrame indexed by KO with columns:
      - mean_abs_lfc_scg
      - mean_abs_lfc_non_scg
      - scg_minus_non_scg : positive => SCGs respond more to this KO
      - log2_ratio        : log2(mean_scg / mean_non_scg)
      - n_scg_obs, n_non_scg_obs : valid gene counts (after NaN masking)
    """
    is_scg = lfc_df.columns.isin(scg_genes)
    if is_scg.sum() == 0 or (~is_scg).sum() == 0:
        raise ValueError(
            "Need both SCG and Non-SCG genes among measured features "
            f"(got {int(is_scg.sum())} SCG, {int((~is_scg).sum())} Non-SCG)."
        )
    abs_lfc = lfc_df.abs()
    scg_means = abs_lfc.loc[:, is_scg].mean(axis=1, skipna=True)
    non_scg_means = abs_lfc.loc[:, ~is_scg].mean(axis=1, skipna=True)

    df = pd.DataFrame({
        'mean_abs_lfc_scg': scg_means,
        'mean_abs_lfc_non_scg': non_scg_means,
        'scg_minus_non_scg': scg_means - non_scg_means,
        'log2_ratio': np.log2((scg_means + 1e-9) / (non_scg_means + 1e-9)),
        'n_scg_obs': abs_lfc.loc[:, is_scg].notna().sum(axis=1),
        'n_non_scg_obs': abs_lfc.loc[:, ~is_scg].notna().sum(axis=1),
    })
    df.index.name = 'KO_Gene'
    return df


def compute_ko_effect_size(lfc_df, scg_genes=None):
    """
    Per-KO effect-size summary derived from the pseudobulk LFC matrix.

    For each KO (row of lfc_df), the effect size is the mean |LFC| across all
    measured genes. NaNs from self-KO masking are ignored.

    Parameters
    ----------
    lfc_df : DataFrame
        Output of compute_pseudobulk_lfc. Rows = KOs, columns = measured genes.
    scg_genes : iterable of str or None
        If provided, additionally compute mean |LFC| restricted to SCG genes
        and to non-SCG genes (useful sanity check vs Q1's per_ko_df).

    Returns
    -------
    DataFrame indexed by KO_Gene with columns:
      - mean_abs_lfc           : mean |LFC| across all measured genes (the
                                  per-KO effect size used for ranking)
      - n_gene_obs             : number of non-NaN gene entries used
      - mean_abs_lfc_scg       : (if scg_genes given) mean |LFC| over SCG genes
      - mean_abs_lfc_nonscg    : (if scg_genes given) mean |LFC| over non-SCG genes
    """
    abs_lfc = lfc_df.abs()
    out = pd.DataFrame(index=lfc_df.index)
    out['mean_abs_lfc'] = abs_lfc.mean(axis=1, skipna=True)
    out['n_gene_obs'] = abs_lfc.notna().sum(axis=1)
    if scg_genes is not None:
        is_scg_col = lfc_df.columns.isin(scg_genes)
        if is_scg_col.sum() == 0:
            print("Warning: no SCG genes overlap with lfc_df columns.")
        out['mean_abs_lfc_scg'] = abs_lfc.loc[:, is_scg_col].mean(axis=1, skipna=True)
        out['mean_abs_lfc_nonscg'] = abs_lfc.loc[:, ~is_scg_col].mean(axis=1, skipna=True)
    out.index.name = 'KO_Gene'
    return out


def compare_top_scg_effect(
    ko_effect_df,
    scg_genes,
    top_k=20,
    metric='mean_abs_lfc',
    n_bootstrap=1000,
    alternative='greater',
    random_seed=42,
):
    """
    Within the SCG-KO group, rank KOs by their effect size (`metric`), take
    the top_k, and compare against three reference groups:

      (a) Other SCG KOs        : the remaining SCG-KOs (rank > k within SCGs).
      (b) All non-SCG KOs      : every KO not in `scg_genes`.
      (c) Top_k non-SCG KOs    : the top_k non-SCGs ranked the same way
                                  (tail-to-tail control — head-to-head test
                                  of the upper tails of both groups).

    Parameters
    ----------
    ko_effect_df : DataFrame
        Output of compute_ko_effect_size; indexed by KO gene name.
    scg_genes : iterable of str
        Unordered SCG gene symbols (e.g. tcell_genes_all).
    top_k : int
        How many top SCG-KOs (by `metric`) to take.
    metric : str
        Column in ko_effect_df used to rank and compare.
    n_bootstrap : int
        Number of random size-k draws from the non-SCG pool for the empirical null.
    alternative : {'greater', 'less', 'two-sided'}
        Direction of the test ('greater' => top_k SCG > reference).

    Returns
    -------
    dict with keys: metric, top_k, alternative, n_top, n_other_scg, n_non_scg,
    top_mean / median, *_mean / *_median for each group, *_values arrays,
    top_genes (list), top_table (DataFrame of top genes + values),
    p_top_vs_other_scg, p_top_vs_nonscg, p_top_vs_top_nonscg,
    bootstrap_p_vs_nonscg, null_means.
    """
    scg_set = set(scg_genes)
    df = ko_effect_df.copy()
    df['is_scg'] = df.index.isin(scg_set)

    scg_sorted = df.loc[df['is_scg']].sort_values(metric, ascending=False)
    non_sorted = df.loc[~df['is_scg']].sort_values(metric, ascending=False)

    if len(scg_sorted) == 0:
        raise ValueError("No SCG-KOs found in ko_effect_df (check scg_genes).")
    if len(non_sorted) == 0:
        raise ValueError("No Non-SCG KOs found in ko_effect_df.")

    top_scg = scg_sorted.head(top_k)
    other_scg = scg_sorted.iloc[top_k:]
    top_non = non_sorted.head(top_k)

    n_scg_total = len(scg_sorted)
    n_non_total = len(non_sorted)
    scg_pct = len(top_scg) / n_scg_total
    k_non_pct = max(1, int(round(scg_pct * n_non_total)))
    top_non_pct = non_sorted.head(k_non_pct)

    top_vals = top_scg[metric].values
    other_vals = other_scg[metric].values
    non_vals = non_sorted[metric].values
    top_non_vals = top_non[metric].values
    top_non_pct_vals = top_non_pct[metric].values

    n_top_actual = len(top_vals)
    if n_top_actual < top_k:
        print(
            f"Note: only {n_top_actual} SCG-KOs available (requested top_k={top_k})."
        )

    out = {
        'metric': metric,
        'top_k': n_top_actual,
        'top_k_requested': top_k,
        'alternative': alternative,
        'n_top': int(n_top_actual),
        'n_other_scg': int(len(other_vals)),
        'n_non_scg': int(len(non_vals)),
        'n_top_non_scg': int(len(top_non_vals)),
        'n_top_non_scg_pct': int(len(top_non_pct_vals)),
        'scg_top_percentile': float(scg_pct),
        'top_mean': float(np.mean(top_vals)),
        'other_scg_mean': float(np.mean(other_vals)) if len(other_vals) else np.nan,
        'non_scg_mean': float(np.mean(non_vals)),
        'top_non_scg_mean': float(np.mean(top_non_vals)) if len(top_non_vals) else np.nan,
        'top_non_scg_pct_mean': float(np.mean(top_non_pct_vals)) if len(top_non_pct_vals) else np.nan,
        'top_median': float(np.median(top_vals)),
        'other_scg_median': float(np.median(other_vals)) if len(other_vals) else np.nan,
        'non_scg_median': float(np.median(non_vals)),
        'top_non_scg_median': float(np.median(top_non_vals)) if len(top_non_vals) else np.nan,
        'top_non_scg_pct_median': float(np.median(top_non_pct_vals)) if len(top_non_pct_vals) else np.nan,
        'top_values': top_vals,
        'other_scg_values': other_vals,
        'non_scg_values': non_vals,
        'top_non_scg_values': top_non_vals,
        'top_non_scg_pct_values': top_non_pct_vals,
        'top_genes': list(top_scg.index),
        'top_table': top_scg[[metric]].reset_index().rename(
            columns={metric: 'effect_size'}
        ),
        'top_non_scg_genes': list(top_non.index),
        'top_non_scg_pct_genes': list(top_non_pct.index),
    }

    # Within-SCG: top-k vs the rest of the SCGs (descriptive — top was sorted out).
    if len(other_vals) > 0:
        _, p1 = mannwhitneyu(top_vals, other_vals, alternative=alternative)
        out['p_top_vs_other_scg'] = float(p1)

    # Broader baseline: top-k SCG vs all non-SCG.
    _, p2 = mannwhitneyu(top_vals, non_vals, alternative=alternative)
    out['p_top_vs_nonscg'] = float(p2)

    # Tail-to-tail (size-matched): top-k SCG vs top-k non-SCG.
    if len(top_non_vals) > 0:
        _, p3 = mannwhitneyu(top_vals, top_non_vals, alternative=alternative)
        out['p_top_vs_top_nonscg'] = float(p3)

    # Tail-to-tail (percentile-matched): top-k SCG vs top-k% non-SCG. The fair test.
    if len(top_non_pct_vals) > 0:
        _, p4 = mannwhitneyu(top_vals, top_non_pct_vals, alternative=alternative)
        out['p_top_vs_top_nonscg_pct'] = float(p4)

    # Bootstrap null: random size-k subsets from non-SCG pool.
    rng = np.random.default_rng(random_seed)
    if n_top_actual > len(non_vals):
        raise ValueError(
            f"top_k ({n_top_actual}) exceeds non-SCG pool ({len(non_vals)})."
        )
    null_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        null_means[i] = rng.choice(non_vals, size=n_top_actual, replace=False).mean()
    if alternative == 'greater':
        boot_p = float((null_means >= out['top_mean']).mean())
    elif alternative == 'less':
        boot_p = float((null_means <= out['top_mean']).mean())
    else:
        diff = abs(out['top_mean'] - out['non_scg_mean'])
        boot_p = float((np.abs(null_means - out['non_scg_mean']) >= diff).mean())
    out['bootstrap_p_vs_nonscg'] = boot_p
    out['null_means'] = null_means

    return out


def plot_top_scg_effect(
    results,
    metric_label='Per-KO mean |LFC|',
    figsize=(15, 6),
    save_path=None,
    show=True,
):
    """
    Two-panel summary of the top-k SCG-effect comparison.

      Left  : boxen + strip across four groups:
                Non-SCG (all) / Other SCG / Top-k Non-SCG / Top-k SCG.
              Title shows MWU p-values for the three meaningful contrasts.
      Right : bootstrap-null histogram with the actual top-k SCG mean marked.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    k = results['top_k']
    k_pct = results.get('n_top_non_scg_pct', 0)
    pct = results.get('scg_top_percentile', 0.0)
    label_non = f'Non-SCG (all)\n(n={results["n_non_scg"]})'
    label_other = f'Other SCG\n(rank > {k} within SCG, n={results["n_other_scg"]})'
    label_top_non = f'Top {k} Non-SCG\n(size-matched, n={results["n_top_non_scg"]})'
    label_top_non_pct = f'Top {k_pct} Non-SCG\n(percentile-matched {pct:.1%}, n={k_pct})'
    label_top_scg = f'Top {k} SCG\n(n={results["n_top"]})'

    rows = []
    for label, vals in [
        (label_non, results['non_scg_values']),
        (label_other, results['other_scg_values']),
        (label_top_non, results['top_non_scg_values']),
        (label_top_non_pct, results.get('top_non_scg_pct_values', np.array([]))),
        (label_top_scg, results['top_values']),
    ]:
        if len(vals) == 0:
            continue
        for v in vals:
            rows.append({'Group': label, 'value': v})
    plot_df = pd.DataFrame(rows)
    order = [g for g in [label_non, label_other, label_top_non,
                         label_top_non_pct, label_top_scg]
             if g in plot_df['Group'].unique()]

    sns.boxenplot(
        data=plot_df, x='Group', y='value', order=order,
        palette='magma', ax=ax1,
    )
    sns.stripplot(
        data=plot_df, x='Group', y='value', order=order,
        color='black', size=2.5, alpha=0.45, ax=ax1, jitter=0.25,
    )

    title_lines = [f'Top-{k} SCG KOs (ranked by {metric_label})']
    if 'p_top_vs_top_nonscg_pct' in results:
        title_lines.append(
            f'Top SCG vs Top {pct:.1%} Non-SCG (percentile-matched): '
            f'MWU ({results["alternative"]}) p = {results["p_top_vs_top_nonscg_pct"]:.2e}   [fair test]'
        )
    if 'p_top_vs_top_nonscg' in results:
        title_lines.append(
            f'Top SCG vs Top Non-SCG (size-matched): MWU ({results["alternative"]}) '
            f'p = {results["p_top_vs_top_nonscg"]:.2e}'
        )
    title_lines.append(
        f'Top SCG vs all Non-SCG: MWU ({results["alternative"]}) '
        f'p = {results["p_top_vs_nonscg"]:.2e}'
    )
    if 'p_top_vs_other_scg' in results:
        title_lines.append(
            f'Top SCG vs Other SCG: MWU ({results["alternative"]}) '
            f'p = {results["p_top_vs_other_scg"]:.2e} (descriptive — sorted)'
        )
    ax1.set_title('\n'.join(title_lines), fontsize=9)
    ax1.set_ylabel(metric_label)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelsize=8)

    ax2.hist(results['null_means'], bins=40, color='lightgray',
             edgecolor='black', alpha=0.85)
    ax2.axvline(results['top_mean'], color='crimson', linestyle='--', linewidth=2.5,
                label=f'Top-{k} SCG mean = {results["top_mean"]:.3f}')
    ax2.axvline(results['non_scg_mean'], color='steelblue', linestyle=':', linewidth=1.5,
                label=f'Non-SCG overall mean = {results["non_scg_mean"]:.3f}')
    ax2.set_title(
        f'Bootstrap null: random {k}-gene sets from Non-SCG\n'
        f'Empirical p ({results["alternative"]}) = {results["bootstrap_p_vs_nonscg"]:.4f}'
    )
    ax2.set_xlabel(f'Mean {metric_label} of random gene set')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)


def plot_top_scg_genes(
    results,
    metric_label='Per-KO mean |LFC|',
    figsize=(8, 8),
    save_path=None,
    show=True,
):
    """
    Horizontal bar chart of the top-k SCG KO genes with their effect sizes,
    plus a dashed reference line at the Non-SCG overall mean.
    """
    top_table = results['top_table'].sort_values('effect_size', ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_table['KO_Gene'], top_table['effect_size'],
            color='crimson', alpha=0.8, edgecolor='black')
    ax.axvline(results['non_scg_mean'], color='steelblue', linestyle=':', lw=1.5,
               label=f'Non-SCG overall mean = {results["non_scg_mean"]:.3f}')
    ax.axvline(results['top_non_scg_mean'], color='goldenrod', linestyle='--', lw=1.5,
               label=f'Top-{results["top_k"]} Non-SCG mean = {results["top_non_scg_mean"]:.3f}')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('')
    ax.set_title(f'Top-{results["top_k"]} SCG knockouts by {metric_label}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top-genes figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, ax


def plot_per_ko_scg_bias(
    per_ko_df,
    figsize=(14, 6),
    save_path=None,
    show=True,
):
    """Scatter + histogram of per-KO SCG bias, with Wilcoxon signed-rank test."""
    diffs = per_ko_df['scg_minus_non_scg'].dropna()
    stat, p = wilcoxon(diffs, alternative='two-sided')
    n_above = int((diffs > 0).sum())
    n_below = int((diffs < 0).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(
        per_ko_df['mean_abs_lfc_non_scg'],
        per_ko_df['mean_abs_lfc_scg'],
        s=8, alpha=0.4, color='steelblue',
    )
    lim = float(np.nanmax([
        per_ko_df['mean_abs_lfc_non_scg'].max(),
        per_ko_df['mean_abs_lfc_scg'].max(),
    ]))
    ax1.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Mean |LFC| over Non-SCG genes (per KO)')
    ax1.set_ylabel('Mean |LFC| over SCG genes (per KO)')
    ax1.set_title(
        'Per-KO SCG vs Non-SCG response\n'
        f'{n_above}/{len(diffs)} KOs above the diagonal ({n_above/len(diffs):.1%}) '
        '=> SCGs respond more'
    )

    ax2.hist(diffs, bins=60, color='lightgray', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(diffs.median(), color='crimson', linewidth=2,
                label=f'median = {diffs.median():.3f}')
    ax2.set_xlabel('Mean |LFC| (SCG) - Mean |LFC| (Non-SCG), per KO')
    ax2.set_ylabel('Number of KOs')
    ax2.set_title(
        'Per-KO SCG bias\n'
        f'KOs with SCG > Non-SCG: {n_above}/{len(diffs)}  ({n_above/len(diffs):.1%})\n'
        f'Wilcoxon signed-rank p = {p:.2e}'
    )
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-KO bias figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)

    return {
        'wilcoxon_p': float(p),
        'median_diff': float(diffs.median()),
        'n_above': n_above,
        'n_below': n_below,
        'n_total': int(len(diffs)),
    }


# ---------------------------------------------------------------------------
# #2 — In-degree (responsiveness) vs out-degree (driving power)
# ---------------------------------------------------------------------------

def compute_in_out_degree(lfc_df, scg_genes):
    """
    For each gene that is BOTH measured and knocked out, compute its position in
    the perturbation network:

      - in_degree  : mean |LFC| of this gene across all KOs  (responsiveness —
                     how much the gene moves when *others* are knocked out).
      - out_degree : mean |LFC| this gene's own KO induces across all genes
                     (driving power — how much knocking *it* out moves everything).

    The responder-not-driver hypothesis predicts SCGs cluster in the
    high-in / low-out corner (network 'sinks').

    Returns DataFrame indexed by gene: in_degree, out_degree, is_scg.
    """
    abs_lfc = lfc_df.abs()
    in_degree = abs_lfc.mean(axis=0, skipna=True)   # per measured gene (column)
    out_degree = abs_lfc.mean(axis=1, skipna=True)  # per KO (row)

    genes_both = lfc_df.columns.intersection(lfc_df.index)
    if len(genes_both) == 0:
        raise ValueError("No genes are both measured and knocked out — cannot plot in/out degree.")

    df = pd.DataFrame(index=genes_both)
    df['in_degree'] = in_degree.loc[genes_both].values
    df['out_degree'] = out_degree.loc[genes_both].values
    df['is_scg'] = df.index.isin(set(scg_genes))
    df.index.name = 'gene'
    print(
        f"{len(genes_both)} genes measured AND knocked out "
        f"({int(df['is_scg'].sum())} SCG, {int((~df['is_scg']).sum())} non-SCG)."
    )
    return df


def plot_in_out_degree(
    degree_df,
    figsize=(10, 10),
    label_top_drivers=8,
    save_path=None,
    show=True,
):
    """
    Joint plot of in-degree (responsiveness) vs out-degree (driving power) with
    marginal density curves for SCG vs non-SCG on each axis.

    The marginals make the mean/median divergence visible: dashed lines mark the
    group means. Titles report BOTH the rank-based Mann-Whitney (two-sided) and
    the mean-based Welch t-test for each axis, since the SCG responsiveness signal
    lives in the mean/tail rather than the median.

    Returns a dict of per-axis group means and both p-values.
    """
    from scipy.stats import ttest_ind
    import matplotlib.gridspec as gridspec

    scg = degree_df[degree_df['is_scg']]
    non = degree_df[~degree_df['is_scg']]
    c_scg, c_non = '#d6336c', '#adb5bd'

    mean_in_scg, mean_in_non = scg['in_degree'].mean(), non['in_degree'].mean()
    mean_out_scg, mean_out_non = scg['out_degree'].mean(), non['out_degree'].mean()
    _, p_in_mwu = mannwhitneyu(scg['in_degree'], non['in_degree'], alternative='two-sided')
    _, p_in_t = ttest_ind(scg['in_degree'], non['in_degree'], equal_var=False)
    _, p_out_mwu = mannwhitneyu(scg['out_degree'], non['out_degree'], alternative='two-sided')
    _, p_out_t = ttest_ind(scg['out_degree'], non['out_degree'], equal_var=False)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05, figure=fig,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # --- main scatter ---
    ax.scatter(non['out_degree'], non['in_degree'], s=7, alpha=0.30,
               color=c_non, edgecolor='none', rasterized=True,
               label=f'Non-SCG (n={len(non)})')
    ax.scatter(scg['out_degree'], scg['in_degree'], s=16, alpha=0.55,
               color=c_scg, edgecolor='none', rasterized=True,
               label=f'SCG (n={len(scg)})')
    ax.set_xlabel("Out-degree: driving power\n(mean |LFC| this gene's KO induces)")
    ax.set_ylabel("In-degree: responsiveness\n(mean |LFC| of this gene across all KOs)")
    ax.legend(loc='upper right', framealpha=0.9, markerscale=1.5)
    ax.grid(alpha=0.15)

    if label_top_drivers:
        for gene, row in degree_df.nlargest(label_top_drivers, 'out_degree').iterrows():
            ax.annotate(gene, (row['out_degree'], row['in_degree']),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 2), textcoords='offset points')

    # --- top marginal: out-degree ---
    for data, c in [(non['out_degree'], c_non), (scg['out_degree'], c_scg)]:
        sns.kdeplot(x=data, ax=ax_top, color=c, fill=True, alpha=0.3,
                    lw=1.8, cut=0, common_norm=False)
    ax_top.axvline(mean_out_non, color=c_non, ls='--', lw=1.2)
    ax_top.axvline(mean_out_scg, color=c_scg, ls='--', lw=1.2)
    ax_top.set_ylabel('')
    ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False)
    for s in ['left', 'right', 'top']:
        ax_top.spines[s].set_visible(False)
    ax_top.set_title(
        f'Out-degree (driving):  SCG {mean_out_scg:.3f}  vs  non-SCG {mean_out_non:.3f}\n'
        f't p={p_out_t:.1e}   MWU p={p_out_mwu:.1e}',
        fontsize=9,
    )

    # --- right marginal: in-degree ---
    for data, c in [(non['in_degree'], c_non), (scg['in_degree'], c_scg)]:
        sns.kdeplot(y=data, ax=ax_right, color=c, fill=True, alpha=0.3,
                    lw=1.8, cut=0, common_norm=False)
    ax_right.axhline(mean_in_non, color=c_non, ls='--', lw=1.2)
    ax_right.axhline(mean_in_scg, color=c_scg, ls='--', lw=1.2)
    ax_right.set_xlabel('')
    ax_right.set_xticks([])
    ax_right.tick_params(labelleft=False)
    for s in ['bottom', 'right', 'top']:
        ax_right.spines[s].set_visible(False)

    fig.suptitle(
        'Perturbation-network position of SCGs   (dashed = group means)\n'
        f'In-degree (responsiveness):  SCG {mean_in_scg:.3f}  vs  non-SCG {mean_in_non:.3f}'
        f'   —   t p={p_in_t:.1e}   MWU p={p_in_mwu:.1e}',
        fontsize=11, y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"In/out-degree figure saved to: {save_path}")
    if show:
        plt.show()

    return {
        'mean_in_scg': float(mean_in_scg),
        'mean_in_non_scg': float(mean_in_non),
        'mean_out_scg': float(mean_out_scg),
        'mean_out_non_scg': float(mean_out_non),
        'p_in_degree_mwu': float(p_in_mwu),
        'p_in_degree_ttest': float(p_in_t),
        'p_out_degree_mwu': float(p_out_mwu),
        'p_out_degree_ttest': float(p_out_t),
    }


# ---------------------------------------------------------------------------
# #1 — Regulator discovery: which KOs drive the SCG program?
# ---------------------------------------------------------------------------

# Convenience set for *visual* annotation only — NOT exhaustive. For a real
# enrichment test pass your own complete TF list to test_regulator_enrichment.
CANONICAL_TCELL_TFS = {
    'TBX21', 'GATA3', 'RORC', 'FOXP3', 'BCL6', 'PRDM1', 'BATF', 'BATF3',
    'IRF4', 'IRF1', 'STAT1', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'RUNX1', 'RUNX3', 'TCF7', 'LEF1', 'ID2', 'ID3', 'EOMES', 'ZEB2', 'TOX',
    'NR4A1', 'NR4A2', 'NR4A3', 'EGR2', 'IKZF1', 'IKZF2', 'IKZF3', 'GFI1',
    'MAF', 'AHR', 'FOXO1', 'KLF2', 'NFATC1', 'NFATC2', 'NFKB1', 'REL', 'RELA',
}


def rank_scg_regulators(lfc_df, scg_genes):
    """
    Rank knockouts by how strongly they perturb the SCG program.

    For each KO computes:
      - effect_on_scg     : mean |LFC| over SCG genes (effect on the SCG program)
      - effect_on_nonscg  : mean |LFC| over non-SCG genes (overall background effect)
      - scg_specificity   : effect_on_scg - effect_on_nonscg  (preferential SCG hit,
                            controls for the KO simply being strong overall)
      - is_scg_ko         : whether the knocked-out gene is itself an SCG

    Sorted by effect_on_scg descending. The top rows are candidate upstream
    regulators of the SCG program.
    """
    abs_lfc = lfc_df.abs()
    is_scg_col = lfc_df.columns.isin(set(scg_genes))
    if is_scg_col.sum() == 0 or (~is_scg_col).sum() == 0:
        raise ValueError("Need both SCG and non-SCG genes among measured features.")

    df = pd.DataFrame({
        'effect_on_scg': abs_lfc.loc[:, is_scg_col].mean(axis=1, skipna=True),
        'effect_on_nonscg': abs_lfc.loc[:, ~is_scg_col].mean(axis=1, skipna=True),
    })
    df['scg_specificity'] = df['effect_on_scg'] - df['effect_on_nonscg']
    df['is_scg_ko'] = df.index.isin(set(scg_genes))
    df = df.sort_values('effect_on_scg', ascending=False)
    df.index.name = 'KO_Gene'
    return df


def test_regulator_enrichment(regulator_df, category_set, top_n=30,
                              rank_by='effect_on_scg'):
    """
    Fisher's exact test: are the top_n SCG regulators enriched for a gene category
    (e.g. transcription factors)?

    Parameters
    ----------
    regulator_df : DataFrame
        Output of rank_scg_regulators.
    category_set : set of str
        Gene symbols in the category of interest (e.g. a full TF list).
    top_n : int
        Size of the 'top regulators' group.
    rank_by : str
        Column to rank by before taking the top_n.

    Returns
    -------
    dict with counts, hit rates, Fisher p, odds ratio, and the top genes that
    fall in the category.
    """
    from scipy.stats import fisher_exact

    ranked = regulator_df.sort_values(rank_by, ascending=False)
    category_set = set(category_set)
    in_cat = ranked.index.isin(category_set)
    top_mask = np.zeros(len(ranked), dtype=bool)
    top_mask[:top_n] = True

    a = int((top_mask & in_cat).sum())          # top & in category
    b = int((top_mask & ~in_cat).sum())         # top & not
    c = int((~top_mask & in_cat).sum())         # rest & in category
    d = int((~top_mask & ~in_cat).sum())        # rest & not
    odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')

    return {
        'top_n': top_n,
        'n_top_in_category': a,
        'top_hit_rate': a / top_n if top_n else float('nan'),
        'background_hit_rate': (a + c) / len(ranked) if len(ranked) else float('nan'),
        'fisher_p': float(p),
        'fisher_odds_ratio': float(odds),
        'top_category_genes': list(ranked.index[:top_n][in_cat[:top_n]]),
    }


def rank_based_enrichment(regulator_df, category_set, rank_by='scg_specificity'):
    """
    Cutoff-free enrichment: instead of an arbitrary top-N, ask whether genes in
    `category_set` sit systematically higher in the full ranking than genes
    outside it (Mann-Whitney on rank position).

    More robust than test_regulator_enrichment — no top_n to choose, uses every KO.

    Returns dict: median rank of category vs non-category genes (rank 1 = strongest
    regulator), MWU p (one-sided: category ranked higher), n in/out category.
    """
    ranked = regulator_df.sort_values(rank_by, ascending=False).copy()
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    in_cat = ranked.index.isin(set(category_set))

    cat_ranks = ranked.loc[in_cat, 'rank'].values
    non_ranks = ranked.loc[~in_cat, 'rank'].values
    if len(cat_ranks) == 0 or len(non_ranks) == 0:
        raise ValueError("Category has no members (or all members) among the KOs.")

    # 'less' => category genes have SMALLER rank numbers (closer to the top).
    _, p = mannwhitneyu(cat_ranks, non_ranks, alternative='less')

    return {
        'rank_by': rank_by,
        'n_in_category': int(in_cat.sum()),
        'n_out_category': int((~in_cat).sum()),
        'median_rank_in_category': float(np.median(cat_ranks)),
        'median_rank_out_category': float(np.median(non_ranks)),
        'n_total': len(ranked),
        'mwu_p_category_higher': float(p),
    }


def load_gmt(gmt_path):
    """
    Parse an Enrichr/MSigDB .gmt file into {term_name: set(genes)}.
    GMT line format: term <tab> description <tab> gene1 <tab> gene2 ...
    """
    gene_sets = {}
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {g.strip() for g in parts[2:] if g.strip()}
            if genes:
                gene_sets[term] = genes
    return gene_sets


def enrich_against_gmt(
    query_genes,
    gmt,
    background_genes,
    min_term_size=5,
    max_term_size=500,
    top_n=20,
):
    """
    Offline over-representation analysis (no internet, no gseapy).

    For each gene set in `gmt`, runs a one-sided Fisher's exact test asking
    whether `query_genes` overlap the term more than expected given the
    `background_genes` universe. Benjamini-Hochberg FDR across terms.

    Parameters
    ----------
    query_genes : iterable of str
        Your gene list of interest (e.g. top SCG regulators).
    gmt : dict
        {term: set(genes)} from load_gmt.
    background_genes : iterable of str
        The universe to test against — use the set of all knocked-out genes
        (or all measured genes), NOT the whole genome, or p-values inflate.
    min_term_size, max_term_size : int
        Restrict to terms with a sensible number of background genes.
    top_n : int
        Number of top terms (by q-value) to return.

    Returns
    -------
    DataFrame: term, n_overlap, term_size_in_bg, query_size, odds_ratio,
    p_raw, q_bh, overlap_genes — sorted by q_bh.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    background = set(background_genes)
    query = set(query_genes) & background           # restrict query to background
    n_bg = len(background)
    n_query = len(query)
    if n_query == 0:
        raise ValueError("No query genes overlap the background universe.")

    rows = []
    for term, genes in gmt.items():
        term_bg = genes & background                 # term genes present in universe
        m = len(term_bg)
        if m < min_term_size or m > max_term_size:
            continue
        overlap = query & term_bg
        k = len(overlap)
        if k == 0:
            continue
        # 2x2: in-term/out-term  x  in-query/out-query
        a = k
        b = n_query - k
        c = m - k
        d = n_bg - n_query - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        rows.append({
            'term': term,
            'n_overlap': k,
            'term_size_in_bg': m,
            'query_size': n_query,
            'odds_ratio': float(odds),
            'p_raw': float(p),
            'overlap_genes': ','.join(sorted(overlap)),
        })

    if not rows:
        raise ValueError("No terms passed the size filter with non-zero overlap.")

    df = pd.DataFrame(rows)
    df['q_bh'] = multipletests(df['p_raw'].values, method='fdr_bh')[1]
    df = df.sort_values('q_bh').reset_index(drop=True)
    return df.head(top_n) if top_n else df


def plot_scg_regulators(
    regulator_df,
    top_n=30,
    tf_set=CANONICAL_TCELL_TFS,
    figsize=(9, 10),
    save_path=None,
    show=True,
):
    """
    Horizontal bar chart of the top_n KOs by their effect on the SCG program.
    Bars whose KO gene is in `tf_set` are colored to flag candidate TFs; a
    dashed line marks the genome-wide median effect-on-SCG for reference.
    """
    top = regulator_df.head(top_n).sort_values('effect_on_scg', ascending=True)
    tf_set = set(tf_set) if tf_set else set()
    colors = ['darkgreen' if g in tf_set else 'crimson' for g in top.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index, top['effect_on_scg'], color=colors,
            alpha=0.85, edgecolor='black')
    median_effect = regulator_df['effect_on_scg'].median()
    ax.axvline(median_effect, color='steelblue', linestyle=':', lw=1.5,
               label=f'median KO effect on SCGs = {median_effect:.3f}')
    ax.set_xlabel('Effect on SCG program (mean |LFC| over SCG genes)')
    ax.set_ylabel('')
    n_tf = sum(1 for g in top.index if g in tf_set)
    ax.set_title(
        f'Top {top_n} candidate regulators of the SCG program\n'
        f'(green = in canonical T-cell TF set; {n_tf}/{top_n} flagged)'
    )
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regulator figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, ax


# ---------------------------------------------------------------------------
# q6 — Do SCG knockouts shift cell FATE, not just expression?
# ---------------------------------------------------------------------------

def compute_fate_composition_shift(
    adata,
    fate_col='cls_label',
    perturbation_col='perturbation',
    control_label='control',
    min_cells_per_ko=20,
    scg_genes=None,
):
    """
    Per-KO cell-fate composition shift from control.

    For each knockout, computes the fraction of its cells in each fate category
    (`fate_col`), then the total-variation distance from the control composition:
        TV = 0.5 * sum_over_states | frac_KO(state) - frac_control(state) |
    TV ranges 0 (identical to control) to 1 (completely different fate makeup).

    Parameters
    ----------
    adata : AnnData
        Cell-level object with `perturbation_col` and `fate_col` in .obs.
    fate_col : str
        Categorical cell-state / fate label (e.g. 'cls_label').
    control_label : str
        Value of perturbation_col marking control cells.
    min_cells_per_ko : int
        Drop KOs with fewer cells than this.
    scg_genes : iterable of str or None
        If given, adds an 'is_scg' column.

    Returns
    -------
    DataFrame indexed by KO: n_cells, tv_distance, (is_scg), plus one
    frac_<state> column per fate state.

    Note
    ----
    `adata` may be an AnnData OR a plain obs DataFrame (handy when you've
    clustered a backed object and only have the metadata table in memory).
    """
    obs = adata.obs if hasattr(adata, 'obs') else adata
    for col in (fate_col, perturbation_col):
        if col not in obs.columns:
            raise KeyError(f"obs missing '{col}'.")

    ct = pd.crosstab(obs[perturbation_col], obs[fate_col])
    n_cells = ct.sum(axis=1)

    if control_label not in ct.index:
        raise ValueError(f"Control label '{control_label}' not in {perturbation_col}.")
    ctrl_frac = ct.loc[control_label] / ct.loc[control_label].sum()

    # Keep control for the reference, filter the rest by cell count.
    keep = (n_cells >= min_cells_per_ko)
    keep[control_label] = True
    ct = ct[keep]

    frac = ct.div(ct.sum(axis=1), axis=0)
    tv = 0.5 * (frac - ctrl_frac).abs().sum(axis=1)

    df = pd.DataFrame({'n_cells': ct.sum(axis=1), 'tv_distance': tv})
    df = df.join(frac.add_prefix('frac_'))
    df = df.drop(index=control_label, errors='ignore')
    if scg_genes is not None:
        df['is_scg'] = df.index.isin(set(scg_genes))
    df.index.name = 'KO_Gene'
    return df.sort_values('tv_distance', ascending=False)


def compare_fate_shift_scg_vs_nonscg(
    shift_df,
    n_bootstrap=1000,
    alternative='greater',
    random_seed=42,
):
    """
    Compare per-KO fate-shift (tv_distance) between SCG-KOs and non-SCG-KOs.
    Mann-Whitney + size-matched bootstrap null (mirrors compare_gene_responsiveness).
    Requires an 'is_scg' column on shift_df.
    """
    if 'is_scg' not in shift_df.columns:
        raise KeyError("shift_df needs an 'is_scg' column — pass scg_genes to compute_fate_composition_shift.")

    scg_vals = shift_df.loc[shift_df['is_scg'], 'tv_distance'].dropna().values
    non_vals = shift_df.loc[~shift_df['is_scg'], 'tv_distance'].dropna().values
    if len(scg_vals) == 0 or len(non_vals) == 0:
        raise ValueError(f"Empty group: SCG={len(scg_vals)}, non-SCG={len(non_vals)}.")

    _, p_mwu = mannwhitneyu(scg_vals, non_vals, alternative=alternative)

    rng = np.random.default_rng(random_seed)
    n_scg = len(scg_vals)
    if n_scg > len(non_vals):
        raise ValueError(f"|SCG| ({n_scg}) exceeds |non-SCG| ({len(non_vals)}).")
    null_means = np.array([
        rng.choice(non_vals, size=n_scg, replace=False).mean()
        for _ in range(n_bootstrap)
    ])
    scg_mean = float(scg_vals.mean())
    non_mean = float(non_vals.mean())
    if alternative == 'greater':
        p_boot = float((null_means >= scg_mean).mean())
    elif alternative == 'less':
        p_boot = float((null_means <= scg_mean).mean())
    else:
        p_boot = float((np.abs(null_means - non_mean) >= abs(scg_mean - non_mean)).mean())

    return {
        'scg_mean': scg_mean,
        'non_scg_mean': non_mean,
        'scg_median': float(np.median(scg_vals)),
        'non_scg_median': float(np.median(non_vals)),
        'n_scg': n_scg,
        'n_non_scg': len(non_vals),
        'mann_whitney_p': float(p_mwu),
        'bootstrap_p': p_boot,
        'alternative': alternative,
        'null_means': null_means,
        'scg_values': scg_vals,
        'non_scg_values': non_vals,
    }


def plot_fate_shift(
    shift_df,
    results=None,
    style='ecdf',
    figsize=(15, 6),
    save_path=None,
    show=True,
):
    """
    Two-panel fate-shift summary.
      Left  : SCG vs non-SCG distribution of per-KO tv_distance ('ecd' or 'boxen').
      Right : bootstrap null (if `results` from compare_fate_shift_scg_vs_nonscg given).
    """
    scg = shift_df.loc[shift_df['is_scg'], 'tv_distance'].dropna()
    non = shift_df.loc[~shift_df['is_scg'], 'tv_distance'].dropna()
    c_scg, c_non = '#d6336c', '#adb5bd'

    if results is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
        ax2 = None

    if style == 'ecdf':
        for data, c, lab in [(non, c_non, f'Non-SCG (n={len(non)})'),
                             (scg, c_scg, f'SCG (n={len(scg)})')]:
            xs = np.sort(data.values)
            ax1.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2, label=lab)
        ax1.set_xlabel('Per-KO fate shift (TV distance from control)')
        ax1.set_ylabel('Cumulative fraction of KOs')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
    else:  # boxen
        plot_df = pd.DataFrame({
            'tv': np.concatenate([non.values, scg.values]),
            'Group': ['Non-SCG'] * len(non) + ['SCG'] * len(scg),
        })
        sns.boxenplot(data=plot_df, x='Group', y='tv', order=['Non-SCG', 'SCG'],
                      palette={'Non-SCG': c_non, 'SCG': c_scg}, ax=ax1)
        ax1.set_ylabel('Per-KO fate shift (TV distance from control)')
        ax1.set_xlabel('')

    title = 'Cell-fate shift: SCG-KOs vs non-SCG-KOs'
    if results is not None:
        title += (f'\nMWU ({results["alternative"]}) p = {results["mann_whitney_p"]:.2e}'
                  f'   |   SCG mean {results["scg_mean"]:.3f} vs non-SCG {results["non_scg_mean"]:.3f}')
    ax1.set_title(title, fontsize=10)

    if ax2 is not None:
        ax2.hist(results['null_means'], bins=40, color='lightgray',
                 edgecolor='black', alpha=0.85)
        ax2.axvline(results['scg_mean'], color=c_scg, ls='--', lw=2.5,
                    label=f'SCG mean = {results["scg_mean"]:.3f}')
        ax2.axvline(results['non_scg_mean'], color='steelblue', ls=':', lw=1.5,
                    label=f'Non-SCG mean = {results["non_scg_mean"]:.3f}')
        ax2.set_title(
            f'Bootstrap null: random {results["n_scg"]}-gene sets from non-SCG\n'
            f'Empirical p ({results["alternative"]}) = {results["bootstrap_p"]:.4f}'
        )
        ax2.set_xlabel('Mean fate shift of random KO set')
        ax2.set_ylabel('Frequency')
        ax2.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fate-shift figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig


def fate_enrichment_by_group(
    shift_df,
    fate_states=None,
):
    """
    Which fate states do SCG-KOs push toward vs non-SCG-KOs, on average?

    Compares the mean per-KO fraction in each fate state between SCG-KOs and
    non-SCG-KOs. Uses the frac_<state> columns from compute_fate_composition_shift.

    Returns DataFrame: state, scg_mean_frac, non_scg_mean_frac, diff, mwu_p (BH).
    """
    from statsmodels.stats.multitest import multipletests

    if 'is_scg' not in shift_df.columns:
        raise KeyError("shift_df needs 'is_scg'.")
    frac_cols = [c for c in shift_df.columns if c.startswith('frac_')]
    if fate_states is not None:
        frac_cols = [f'frac_{s}' for s in fate_states if f'frac_{s}' in frac_cols]

    scg = shift_df[shift_df['is_scg']]
    non = shift_df[~shift_df['is_scg']]
    rows = []
    for col in frac_cols:
        s, n = scg[col].dropna(), non[col].dropna()
        try:
            _, p = mannwhitneyu(s, n, alternative='two-sided')
        except ValueError:
            p = 1.0
        rows.append({
            'state': col.replace('frac_', ''),
            'scg_mean_frac': float(s.mean()),
            'non_scg_mean_frac': float(n.mean()),
            'diff': float(s.mean() - n.mean()),
            'p_raw': float(p),
        })
    df = pd.DataFrame(rows)
    df['q_bh'] = multipletests(df['p_raw'].values, method='fdr_bh')[1]
    return df.sort_values('diff', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Head-to-head: compare a second gene list (e.g. top XGBoost T-lineage genes)
# against the SCGs — are they MORE or LESS responsive than the SCGs?
#
# Unlike the functions above (which test one set vs *everything else*), these
# pit two named gene sets directly against each other.
# ---------------------------------------------------------------------------

def load_ranked_gene_list(
    csv_path,
    top_n=200,
    gene_col='gene',
    rank_col='rank',
    restrict_to=None,
    flag_col=None,
    flag_value=True,
):
    """
    Load the top-N genes from a ranked feature-importance CSV (e.g. the
    XGBoost T-lineage importances at data/T-lineage/xgb_feature_importances.csv).

    Parameters
    ----------
    csv_path : str
        Path to the CSV. Must contain `gene_col` and (if `rank_col` is used)
        a rank column. Lower rank number = more important.
    top_n : int or None
        How many top genes to return after any filtering. None => return all.
    gene_col : str
        Column holding gene symbols.
    rank_col : str or None
        Column to sort ascending by before taking the head. If None, the CSV's
        existing row order is trusted.
    restrict_to : iterable of str or None
        If given, the list is first filtered to genes present in this set
        (e.g. the perturbations actually knocked out, or the measured genes),
        THEN the top-N are taken. This makes "the top 200 genes I can actually
        test" well-defined and keeps the size honest.
    flag_col : str or None
        Optional boolean column to pre-filter on (e.g. 'is_tcell_scg'). Rows are
        kept where flag_col == flag_value before ranking/top-N.
    flag_value : any
        Value `flag_col` must equal to keep the row.

    Returns
    -------
    list of str : the selected gene symbols, in rank order.
    """
    df = pd.read_csv(csv_path)
    if gene_col not in df.columns:
        raise KeyError(f"'{gene_col}' not in {csv_path}. Columns: {list(df.columns)}")

    if flag_col is not None:
        if flag_col not in df.columns:
            raise KeyError(f"flag_col '{flag_col}' not in CSV. Columns: {list(df.columns)}")
        df = df[df[flag_col] == flag_value]

    if rank_col is not None and rank_col in df.columns:
        df = df.sort_values(rank_col, ascending=True)

    if restrict_to is not None:
        restrict_set = set(restrict_to)
        before = len(df)
        df = df[df[gene_col].isin(restrict_set)]
        print(
            f"restrict_to filter: {len(df)}/{before} ranked genes are present "
            "in the provided set (e.g. perturbations / measured genes)."
        )

    genes = df[gene_col].astype(str).tolist()
    if top_n is not None:
        genes = genes[:top_n]
    print(f"Loaded {len(genes)} genes from {os.path.basename(csv_path)} "
          f"(top_n={top_n}).")
    return genes


def compare_two_gene_sets(
    values,
    set_a,
    set_b,
    label_a='Set A',
    label_b='Set B',
    exclude_overlap=True,
    n_perm=10000,
    alternative='two-sided',
    random_seed=42,
):
    """
    Directly compare two named gene sets on a per-entity responsiveness value.

    This is the head-to-head analogue of `compare_gene_responsiveness` (which
    tests one set vs the background). Here both groups are explicit gene sets,
    so you can ask e.g. "are the top XGBoost T-lineage genes MORE or LESS
    responsive than the SCGs?"

    Parameters
    ----------
    values : pd.Series
        Per-entity metric indexed by gene (or KO) name. Examples:
          - gene_summary['mean_abs_lfc']   -> responsiveness (in-degree)
          - ko_effect_df['mean_abs_lfc']   -> driving power   (out-degree)
        NaNs are dropped automatically.
    set_a, set_b : iterable of str
        The two gene sets. `alternative='greater'` tests set_a > set_b.
    label_a, label_b : str
        Display names (used by the plotting helper and in the returned dict).
    exclude_overlap : bool
        The two sets often overlap (e.g. many top T-lineage genes are also SCGs).
        If True (default), genes in BOTH sets are dropped so the comparison is
        between the *exclusive* members of each set — a cleaner contrast. If
        False, overlapping genes are counted in both groups.
    n_perm : int
        Number of label permutations for the empirical null on the mean
        difference (set_a mean - set_b mean).
    alternative : {'two-sided', 'greater', 'less'}
        'greater' => set_a is more responsive than set_b.
    random_seed : int
        RNG seed for the permutation null.

    Returns
    -------
    dict : means/medians/n for each set, Mann-Whitney p, permutation p,
    observed mean difference, the permutation-null differences, and the raw
    value arrays (for plotting).
    """
    vals = values.dropna()
    a_set = set(set_a) & set(vals.index)
    b_set = set(set_b) & set(vals.index)

    overlap = a_set & b_set
    n_overlap = len(overlap)
    if exclude_overlap and n_overlap:
        a_set = a_set - overlap
        b_set = b_set - overlap
        print(f"Excluded {n_overlap} genes present in both sets "
              "(comparing exclusive members).")
    elif n_overlap:
        print(f"Note: {n_overlap} genes are in BOTH sets and counted in each "
              "(exclude_overlap=False).")

    a_vals = vals.loc[sorted(a_set)].values
    b_vals = vals.loc[sorted(b_set)].values
    if len(a_vals) == 0 or len(b_vals) == 0:
        raise ValueError(
            f"Empty group after filtering: {label_a}={len(a_vals)}, "
            f"{label_b}={len(b_vals)}. Check the gene names match `values.index`."
        )

    _, p_mwu = mannwhitneyu(a_vals, b_vals, alternative=alternative)

    # Permutation null: pool both groups, reshuffle labels, recompute the
    # difference of group means. Distribution-free and handles unequal sizes.
    pooled = np.concatenate([a_vals, b_vals])
    total = pooled.sum()
    n_a, n_b = len(a_vals), len(b_vals)
    observed_diff = float(a_vals.mean() - b_vals.mean())

    rng = np.random.default_rng(random_seed)
    perm_sums = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_sums[i] = rng.choice(pooled, size=n_a, replace=False).sum()
    perm_diff = perm_sums / n_a - (total - perm_sums) / n_b

    if alternative == 'greater':
        p_perm = float((perm_diff >= observed_diff).mean())
    elif alternative == 'less':
        p_perm = float((perm_diff <= observed_diff).mean())
    else:
        p_perm = float((np.abs(perm_diff) >= abs(observed_diff)).mean())

    direction = (
        f"{label_a} > {label_b}" if observed_diff > 0
        else f"{label_a} < {label_b}"
    )
    print(
        f"{label_a} mean = {a_vals.mean():.4f} (n={n_a})   "
        f"{label_b} mean = {b_vals.mean():.4f} (n={n_b})   "
        f"=> {direction}\n"
        f"Mann-Whitney ({alternative}) p = {p_mwu:.2e}   "
        f"permutation p = {p_perm:.4f}"
    )

    return {
        'label_a': label_a,
        'label_b': label_b,
        'alternative': alternative,
        'a_mean': float(a_vals.mean()),
        'b_mean': float(b_vals.mean()),
        'a_median': float(np.median(a_vals)),
        'b_median': float(np.median(b_vals)),
        'n_a': n_a,
        'n_b': n_b,
        'n_overlap': n_overlap,
        'exclude_overlap': exclude_overlap,
        'observed_diff': observed_diff,
        'mann_whitney_p': float(p_mwu),
        'permutation_p': p_perm,
        'perm_diff': perm_diff,
        'a_values': a_vals,
        'b_values': b_vals,
    }


def plot_two_gene_set_comparison(
    results,
    value_label='Mean |LFC| across KOs',
    style='ecdf',
    figsize=(15, 6),
    save_path=None,
    show=True,
):
    """
    Two-panel figure for `compare_two_gene_sets`.

      Left  : distribution of the metric for set A vs set B
                ('ecd' — best when sizes differ; or 'boxen').
      Right : permutation null of the mean difference (A - B), with the
                observed difference marked. If the line sits in the tail, the
                two sets differ more than label-shuffling would produce.
    """
    a_vals = results['a_values']
    b_vals = results['b_values']
    label_a, label_b = results['label_a'], results['label_b']
    c_a, c_b = '#2b8cbe', '#d6336c'   # set A = teal, set B = crimson

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if style == 'ecdf':
        for data, c, lab in [
            (b_vals, c_b, f'{label_b} (n={len(b_vals)})'),
            (a_vals, c_a, f'{label_a} (n={len(a_vals)})'),
        ]:
            xs = np.sort(data)
            ax1.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2, label=lab)
        ax1.set_xlabel(value_label)
        ax1.set_ylabel('Cumulative fraction of genes')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
    else:  # boxen
        plot_df = pd.DataFrame({
            'value': np.concatenate([b_vals, a_vals]),
            'Group': [label_b] * len(b_vals) + [label_a] * len(a_vals),
        })
        sns.boxenplot(data=plot_df, x='Group', y='value', order=[label_b, label_a],
                      palette={label_b: c_b, label_a: c_a}, ax=ax1)
        sns.stripplot(data=plot_df, x='Group', y='value', order=[label_b, label_a],
                      color='black', size=2, alpha=0.25, ax=ax1, jitter=0.25)
        ax1.set_ylabel(value_label)
        ax1.set_xlabel('')

    overlap_note = (
        f'overlap excluded (n={results["n_overlap"]})' if results['exclude_overlap']
        else f'overlap kept in both (n={results["n_overlap"]})'
    )
    ax1.set_title(
        f'{value_label}: {label_a} vs {label_b}\n'
        f'Mann-Whitney ({results["alternative"]}) p = {results["mann_whitney_p"]:.2e}   |   '
        f'{label_a} mean {results["a_mean"]:.3f} vs {label_b} mean {results["b_mean"]:.3f}\n'
        f'{overlap_note}',
        fontsize=10,
    )

    ax2.hist(results['perm_dif'], bins=40, color='lightgray',
             edgecolor='black', alpha=0.85)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1)
    ax2.axvline(results['observed_dif'], color='crimson', linestyle='--', linewidth=2.5,
                label=f'observed diff = {results["observed_diff"]:.3f}')
    ax2.set_title(
        f'Permutation null: shuffled {label_a}/{label_b} labels\n'
        f'Empirical p ({results["alternative"]}) = {results["permutation_p"]:.4f}'
    )
    ax2.set_xlabel(f'Mean difference ({label_a} - {label_b})')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Two-set comparison figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Multi-group KO comparison: several named knockout groups vs the non-SCG
# background ("the rest") — the driving-power framing, generalized.
#
# This is the version of `compare_top_scg_effect` that keeps your two named
# sets (e.g. T-lineage-classifier-top and SCG) AND the Non-SCG background all
# in one figure, like the analysis you had before.
# ---------------------------------------------------------------------------

def compare_ko_effect_groups(
    ko_effect_df,
    group_sets,
    value_col='mean_abs_lfc',
    background_label='Non-SCG (rest)',
    n_bootstrap=1000,
    alternative='two-sided',
    random_seed=42,
):
    """
    Compare per-KO effect size across several named KO groups + a background.

    The background group is built automatically: every KO that is NOT in any of
    the provided `group_sets`. That is your "Non-SCG (rest)" reference — the
    same role non-SCG played in compare_top_scg_effect.

    Each named group is tested against the background two ways:
      - Mann-Whitney U (rank test, fine with very unequal group sizes), and
      - a size-matched bootstrap null: draw random same-size subsets from the
        background and compare the group's mean to that null distribution. This
        is the statistically correct way to compare a small group to a big
        background — you do NOT need equal group sizes.

    Parameters
    ----------
    ko_effect_df : DataFrame
        Output of compute_ko_effect_size; indexed by KO gene name.
    group_sets : dict {label: iterable of str}
        The named knockout groups, e.g.
            {'T-lineage classifier top': tlineage_top,
             'T-lineage SCG':            scg_strict_list}
        A gene may appear in more than one group (overlap is kept); the
        background excludes the UNION of all groups.
    value_col : str
        Column of ko_effect_df to compare (default per-KO mean |LFC|).
    background_label : str
        Display name for the auto-built "rest" group.
    n_bootstrap : int
        Random size-matched draws from the background for each group's null.
    alternative : {'two-sided', 'greater', 'less'}
        'greater' => the named group drives more than the background.
    random_seed : int
        RNG seed for the bootstrap.

    Returns
    -------
    dict with:
      - 'order'           : group labels in plot order (named groups, then background)
      - 'values'/'means'/'medians'/'n' : per-group, keyed by label
      - 'background_label'
      - 'alternative', 'value_col'
      - 'tests' : {group_label: {'mean', 'mwu_p', 'bootstrap_p', 'null_means'}}
                  one entry per NAMED group, each vs the background.
    """
    if value_col not in ko_effect_df.columns:
        raise KeyError(f"'{value_col}' not in ko_effect_df. Columns: {list(ko_effect_df.columns)}")

    vals = ko_effect_df[value_col].dropna()
    index_set = set(vals.index)

    resolved = {}          # label -> sorted list of KOs present in vals
    union = set()
    for label, genes in group_sets.items():
        present = set(genes) & index_set
        if len(present) == 0:
            print(f"Warning: group '{label}' has no KOs present in ko_effect_df — skipped.")
            continue
        resolved[label] = sorted(present)
        union |= present

    if not resolved:
        raise ValueError("None of the provided group_sets matched any KO in ko_effect_df.")

    background = sorted(index_set - union)
    if len(background) == 0:
        raise ValueError("Background is empty — the groups cover every KO. Nothing to compare against.")

    bg_vals = vals.loc[background].values

    values, means, medians, n = {}, {}, {}, {}
    for label, kos in resolved.items():
        v = vals.loc[kos].values
        values[label] = v
        means[label] = float(v.mean())
        medians[label] = float(np.median(v))
        n[label] = len(v)
    values[background_label] = bg_vals
    means[background_label] = float(bg_vals.mean())
    medians[background_label] = float(np.median(bg_vals))
    n[background_label] = len(bg_vals)

    rng = np.random.default_rng(random_seed)
    tests = {}
    for label, v in values.items():
        if label == background_label:
            continue
        _, p_mwu = mannwhitneyu(v, bg_vals, alternative=alternative)
        k = len(v)
        null_means = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            null_means[i] = rng.choice(bg_vals, size=k, replace=False).mean()
        g_mean = means[label]
        if alternative == 'greater':
            p_boot = float((null_means >= g_mean).mean())
        elif alternative == 'less':
            p_boot = float((null_means <= g_mean).mean())
        else:
            diff = abs(g_mean - means[background_label])
            p_boot = float((np.abs(null_means - means[background_label]) >= diff).mean())
        tests[label] = {
            'mean': g_mean,
            'mwu_p': float(p_mwu),
            'bootstrap_p': p_boot,
            'null_means': null_means,
        }
        print(
            f"{label:<28} mean={g_mean:.3f} (n={k})   vs {background_label} "
            f"mean={means[background_label]:.3f} (n={len(bg_vals)})   "
            f"MWU p={p_mwu:.2e}   bootstrap p={p_boot:.4f}"
        )

    order = list(resolved.keys()) + [background_label]
    return {
        'order': order,
        'values': values,
        'means': means,
        'medians': medians,
        'n': n,
        'background_label': background_label,
        'alternative': alternative,
        'value_col': value_col,
        'tests': tests,
    }


def plot_ko_effect_groups(
    results,
    value_label='Per-KO mean |LFC|',
    figsize=(16, 6),
    save_path=None,
    show=True,
):
    """
    Two-panel summary for `compare_ko_effect_groups`.

      Left  : boxen + strip across every group (named groups then background).
      Right : size-matched bootstrap null for each NAMED group, with that
                group's observed mean marked. A line in the tail => that group
                drives differently than a random same-size set of background KOs.
    """
    order = results['order']
    bg_label = results['background_label']
    values = results['values']

    # named groups get colors from a palette; background is gray
    named = [l for l in order if l != bg_label]
    palette_colors = sns.color_palette('Set1', n_colors=max(len(named), 1))
    color_map = {l: palette_colors[i] for i, l in enumerate(named)}
    color_map[bg_label] = '#adb5bd'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    rows = []
    for label in order:
        for v in values[label]:
            rows.append({'Group': f'{label}\n(n={results["n"][label]})', 'value': v})
    plot_df = pd.DataFrame(rows)
    ordered_labels = [f'{l}\n(n={results["n"][l]})' for l in order]
    box_palette = {f'{l}\n(n={results["n"][l]})': color_map[l] for l in order}

    sns.boxenplot(data=plot_df, x='Group', y='value', order=ordered_labels,
                  palette=box_palette, ax=ax1)
    sns.stripplot(data=plot_df, x='Group', y='value', order=ordered_labels,
                  color='black', size=2, alpha=0.25, ax=ax1, jitter=0.25)
    ax1.set_ylabel(value_label)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelsize=8)

    title_lines = [f'{value_label}: named KO groups vs {bg_label}']
    for label in named:
        t = results['tests'][label]
        title_lines.append(
            f'{label} vs {bg_label}: MWU ({results["alternative"]}) '
            f'p={t["mwu_p"]:.2e}, bootstrap p={t["bootstrap_p"]:.4f}'
        )
    ax1.set_title('\n'.join(title_lines), fontsize=9)

    # right: overlay each named group's bootstrap null + observed mean
    for label in named:
        t = results['tests'][label]
        ax2.hist(t['null_means'], bins=40, color=color_map[label], alpha=0.35,
                 edgecolor='none', label=f'{label} null (n={results["n"][label]})')
        ax2.axvline(t['mean'], color=color_map[label], linestyle='--', linewidth=2.5,
                    label=f'{label} mean = {t["mean"]:.3f}')
    ax2.axvline(results['means'][bg_label], color='black', linestyle=':', linewidth=1.5,
                label=f'{bg_label} mean = {results["means"][bg_label]:.3f}')
    ax2.set_title(
        f'Size-matched bootstrap null: random draws from {bg_label}\n'
        f'(each null matches its group size; alternative={results["alternative"]})'
    )
    ax2.set_xlabel(f'Mean {value_label} of random {bg_label} set')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KO-effect group figure saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Set overlap: Venn diagram of two gene lists (e.g. SCGs vs classifier-top)
# ---------------------------------------------------------------------------

def plot_gene_set_venn(
    set_a,
    set_b,
    label_a='Set A',
    label_b='Set B',
    restrict_to=None,
    color_a='#d6336c',
    color_b='#2b8cbe',
    figsize=(8, 8),
    title=None,
    save_path=None,
    show=True,
):
    """
    Two-set Venn diagram of gene-set overlap.

    Parameters
    ----------
    set_a, set_b : iterable of str
        The two gene lists to compare.
    label_a, label_b : str
        Circle labels.
    restrict_to : iterable of str or None
        Optional shared universe to intersect BOTH sets with first (e.g. the
        perturbations, the measured genes, or the 2000 ranked classifier
        features). Without it, the Venn uses the raw lists — note this makes
        the larger list's circle dominate (e.g. the full 1384-gene SCG set vs a
        top-200 classifier list leaves ~1257 SCG-only). With restrict_to set to
        a shared feature space the comparison is more apples-to-apples.
    color_a, color_b : str
        Circle colors (default crimson = set A, teal = set B).
    title : str or None
        Custom title; defaults to a summary with overlap count and Jaccard index.

    Returns
    -------
    dict : counts (only_a, only_b, both), jaccard, and the sorted overlap genes.
    """
    from matplotlib_venn import venn2

    a = set(set_a)
    b = set(set_b)
    if restrict_to is not None:
        universe = set(restrict_to)
        a &= universe
        b &= universe
        print(f"restrict_to applied: {label_a} -> {len(a)}, {label_b} -> {len(b)} "
              f"(within shared universe of {len(universe)}).")

    only_a = len(a - b)
    only_b = len(b - a)
    both = len(a & b)
    union = len(a | b)
    jaccard = both / union if union else float('nan')

    fig, ax = plt.subplots(figsize=figsize)
    v = venn2(
        subsets=(only_a, only_b, both),
        set_labels=(f'{label_a}\n(n={len(a)})', f'{label_b}\n(n={len(b)})'),
        set_colors=(color_a, color_b),
        alpha=0.55,
        ax=ax,
    )
    # readable subset counts
    for sid in ('10', '01', '11'):
        lbl = v.get_label_by_id(sid)
        if lbl is not None:
            lbl.set_fontsize(13)
            lbl.set_fontweight('bold')

    if title is None:
        frac_a = both / len(a) if len(a) else float('nan')
        frac_b = both / len(b) if len(b) else float('nan')
        title = (
            f'{label_a} vs {label_b}\n'
            f'overlap = {both}   (Jaccard = {jaccard:.3f})\n'
            f'{both}/{len(a)} of {label_a} ({frac_a:.0%})   |   '
            f'{both}/{len(b)} of {label_b} ({frac_b:.0%})'
        )
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Venn diagram saved to: {save_path}")
    if show:
        plt.show()

    return {
        'only_a': only_a,
        'only_b': only_b,
        'both': both,
        'jaccard': jaccard,
        'overlap_genes': sorted(a & b),
    }


# ---------------------------------------------------------------------------
# Classifier importance curve + offline pathway enrichment for the top genes
# ---------------------------------------------------------------------------

def plot_importance_curve(
    importance_df,
    top_n=80,
    gene_col='gene',
    importance_col='importance',
    rank_col='rank',
    mark_sets=None,
    annotate_genes=True,
    figsize=(18, 6),
    title=None,
    save_path=None,
    show=True,
):
    """
    Line plot of classifier feature importance across the top-N genes, with the
    points (and gene labels) colored by which overlap set they belong to.

    Parameters
    ----------
    importance_df : DataFrame or str
        The xgb_feature_importances table, or a path to its CSV. Needs
        `gene_col` and `importance_col`; `rank_col` is used to order if present.
    top_n : int
        How many top genes to show.
    mark_sets : dict {label: iterable of str} or None
        Overlap groups to color. A gene is colored by the FIRST set it matches
        (dict order = priority); genes in none are gray 'Other'. If None and an
        'is_tcell_scg' column exists, defaults to {'T-lineage SCG': those genes}.
    annotate_genes : bool
        Put gene names as (rotated) x-tick labels, colored by overlap set.

    Returns
    -------
    dict : per-category counts and the membership table (gene, rank,
    importance, category).
    """
    if isinstance(importance_df, str):
        importance_df = pd.read_csv(importance_df)
    df = importance_df.copy()
    if rank_col in df.columns:
        df = df.sort_values(rank_col, ascending=True)
    else:
        df = df.sort_values(importance_col, ascending=False)
    top = df.head(top_n).reset_index(drop=True)

    if mark_sets is None:
        if 'is_tcell_scg' in top.columns:
            mark_sets = {'T-lineage SCG': set(top.loc[top['is_tcell_scg'] == True, gene_col])}
        else:
            mark_sets = {}
    mark_sets = {k: set(v) for k, v in mark_sets.items()}

    labels = list(mark_sets.keys())
    palette = sns.color_palette('tab10', n_colors=max(len(labels), 1))
    color_map = {lab: palette[i] for i, lab in enumerate(labels)}
    color_map['Other'] = '#c8c8c8'

    def category(g):
        for lab, s in mark_sets.items():
            if g in s:
                return lab
        return 'Other'

    top['category'] = top[gene_col].map(category)
    point_colors = top['category'].map(color_map).tolist()

    x = np.arange(1, len(top) + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, top[importance_col].values, color='#888888', lw=1.2, zorder=1, alpha=0.8)
    ax.scatter(x, top[importance_col].values, c=point_colors, s=45,
               edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xlim(0.5, len(top) + 0.5)
    ax.set_ylabel('XGBoost feature importance')
    ax.set_xlabel('Rank' if not annotate_genes else '')
    ax.grid(axis='y', alpha=0.25)

    if annotate_genes:
        ax.set_xticks(x)
        ax.set_xticklabels(top[gene_col].values, rotation=90, fontsize=7)
        for tick, cat in zip(ax.get_xticklabels(), top['category']):
            tick.set_color('black' if cat == 'Other' else color_map[cat])

    # legend with per-category counts
    counts = top['category'].value_counts().to_dict()
    handles = []
    for lab in labels + ['Other']:
        n = counts.get(lab, 0)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[lab], markeredgecolor='black',
                                  markersize=9, label=f'{lab} (n={n})'))
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    if title is None:
        overlap_bits = '   '.join(f'{lab}: {counts.get(lab, 0)}/{len(top)}' for lab in labels)
        title = f'Feature importance across top {len(top)} genes\n{overlap_bits}'
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Importance curve saved to: {save_path}")
    if show:
        plt.show()

    return {
        'counts': counts,
        'table': top[[gene_col, rank_col, importance_col, 'category']]
                 if rank_col in top.columns else top[[gene_col, importance_col, 'category']],
    }


def plot_enrichment_bar(
    enrich_df,
    top_n=15,
    term_col='term',
    q_col='q_bh',
    overlap_col='n_overlap',
    figsize=(11, 8),
    title='Pathway enrichment (GO Biological Process)',
    save_path=None,
    show=True,
):
    """
    Horizontal bar chart of the top enriched terms from `enrich_against_gmt`,
    ranked by -log10(BH q-value). Bars annotated with the overlap gene count.
    """
    d = enrich_df.sort_values(q_col, ascending=True).head(top_n).iloc[::-1].copy()
    d['neg_log10_q'] = -np.log10(d[q_col].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(d[term_col], d['neg_log10_q'], color='#3b7dd8',
                   edgecolor='black', alpha=0.85)
    if overlap_col in d.columns:
        for bar, k in zip(bars, d[overlap_col]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f'  {int(k)}', va='center', ha='left', fontsize=8)
    ax.axvline(-np.log10(0.05), color='crimson', linestyle='--', lw=1.2,
               label='q = 0.05')
    ax.set_xlabel('-log10(BH q-value)   (number = overlapping genes)')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enrichment bar chart saved to: {save_path}")
    if show:
        plt.show()
    else:
        return fig, ax


# ---------------------------------------------------------------------------
# Memory-gene benchmark: compare the per-gene responsiveness of several ranked
# memory-gene method outputs (CORAL / GEMLI / MOLD / SCG) against a shared
# Non-SCG background, in one figure.
# ---------------------------------------------------------------------------
def load_memory_gene_set(method, timepoint='merge', top_n=500, merge_mode='union',
                         memgene_dir=MEMGENE_DIR, method_files=METHOD_FILES,
                         restrict_to=None, biased_only=True):
    """
    Build the memory-gene set for one method (see method_files).

    Files are pre-ranked (best gene first) with a 'timepoint' column (T1/T2).

    timepoint  : 'T1' | 'T2' | 'merge'
        Single timepoint, or combine both.
    top_n      : int or None
        Keep the top-N ranked genes (None = keep all).
    merge_mode : 'union' | 'intersection'
        Only used when timepoint == 'merge'.
    memgene_dir, method_files :
        Directory and {name: filename} map of the ranked method CSVs. Default to
        the module-level MEMGENE_DIR / METHOD_FILES; pass your own to override.
    restrict_to : iterable of gene symbols or None
        If given, keep only these genes (e.g. the measured genes, gene_summary.index)
        BEFORE ranking/top_n, so "top_n" counts genes that actually appear in the
        data. Pass this to get matched set sizes across methods.
    biased_only : bool
        For SCG files (the only ones with an 'is_biased' column) keep ONLY the
        called (is_biased==True) memory genes -- NEVER the top-ranked list. If SCG
        flagged nothing (is_biased empty), this returns an empty set and SCG drops
        out downstream. This is a firm team rule; leave it True. Other methods have
        no is_biased column and are unaffected (top-N ranked as before).

    For 'merge' the two timepoints are pooled into a single ranked list FIRST
    (each gene keeps its best = smallest within-timepoint rank), and top_n is
    applied to that merged list.
    """
    df = pd.read_csv(os.path.join(memgene_dir, method_files[method])).copy()
    # human_symbol matches the marson gene symbols when present (SCG file); else 'gene'.
    gene_col = 'human_symbol' if 'human_symbol' in df.columns else 'gene'
    df = df.dropna(subset=[gene_col])
    if biased_only and 'is_biased' in df.columns:           # SCG: only the called (is_biased)
        df = df[df['is_biased'] == True]                    # memory genes, never top-ranked
    if restrict_to is not None:
        df = df[df[gene_col].isin(set(restrict_to))]         # keep only measured genes first
    df['rank'] = df.groupby('timepoint').cumcount()          # rank AFTER restriction

    if timepoint != 'merge':
        # Any non-'merge' value selects a single timepoint. This used to test
        # `timepoint in ('T1','T2')`, which SILENTLY returned the merge for datasets
        # whose timepoints are not T1/T2 -- e.g. LARRY's floats 2.0/4.0/6.0. Now an
        # unknown timepoint raises instead of quietly handing back the wrong set.
        avail = set(df['timepoint'].unique())
        if timepoint not in avail:
            raise ValueError(
                f"timepoint {timepoint!r} not found in {method_files[method]}; "
                f"available: {sorted(avail)} (or 'merge')")
        genes = list(dict.fromkeys(df.loc[df['timepoint'] == timepoint, gene_col]))  # best-first
        return genes[:top_n] if top_n else genes

    # ---- merge FIRST (best rank per gene across timepoints), THEN top_n ----
    best = df.groupby(gene_col)['rank'].min()
    if merge_mode == 'intersection':
        n_tp = df.groupby(gene_col)['timepoint'].nunique()
        best = best[n_tp >= 2]
    merged = best.sort_values(kind='stable').index.tolist()  # best-ranked gene first
    return merged[:top_n] if top_n else merged


def build_consensus_gene_sets(
    min_methods=2, timepoint='merge', top_n=None, merge_mode='union',
    memgene_dir=MEMGENE_DIR, method_files=METHOD_FILES, methods=None,
    universe=None, exactly=False, label_fmt=None, verbose=True,
):
    """
    Build 'consensus' gene sets by how many methods call each gene a memory gene.

    exactly : bool
        False (default) -> cumulative, NESTED tiers: genes called by >= k methods,
                           for k = n_methods .. min_methods.
        True            -> DISJOINT tiers: genes called by EXACTLY k methods,
                           for k = n_methods .. min_methods (no overlap between tiers).

    Returns (tiers, counts):
      tiers  : {label: [genes]} in descending k.
      counts : pd.Series of per-gene method-agreement count over `universe`
               (0 = called by no method). Use counts[counts == 0] as the true
               non-memory background.

    universe : iterable or None
        Gene index to count over (e.g. gene_summary.index = measured genes). If
        None, only genes called by >=1 method appear in counts.
    """
    methods = methods or list(method_files)
    if label_fmt is None:
        label_fmt = 'exactly {k}' if exactly else '>={k} methods'
    sets = {m: set(load_memory_gene_set(m, timepoint, top_n, merge_mode,
                                        memgene_dir, method_files, restrict_to=universe))
            for m in methods}

    idx = list(universe) if universe is not None else sorted(set().union(*sets.values()))
    counts = pd.Series(0, index=pd.Index(idx, name='gene'), dtype=int)
    for s in sets.values():
        counts.loc[counts.index.isin(s)] += 1

    n = len(methods)
    tiers = {}
    for k in range(n, max(min_methods, 1) - 1, -1):
        mask = (counts == k) if exactly else (counts >= k)
        genes = counts.index[mask].tolist()
        tiers[label_fmt.format(k=k)] = genes
        if verbose:
            print(f"{label_fmt.format(k=k)}: {len(genes)} genes")
    if verbose:
        print(f"non-memory (0 methods): {int((counts == 0).sum())} genes")
    return tiers, counts


def build_all_gene_sets(timepoint='merge', top_n=500, merge_mode='union',
                        memgene_dir=MEMGENE_DIR, method_files=METHOD_FILES,
                        methods=None, restrict_to=None, equalize=False, verbose=True):
    """
    Return {method: gene_list} for the requested methods under one config.

    restrict_to : iterable or None
        Keep only these genes (e.g. gene_summary.index) before top_n, so the
        reported set sizes are counts of genes present in the data.
    equalize : bool
        If True, trim every set to the smallest set's size (keeping best-ranked
        genes) so all methods have an identical n. Combine with restrict_to.
    """
    methods = methods or list(method_files)
    sets = {m: load_memory_gene_set(m, timepoint, top_n, merge_mode,
                                    memgene_dir, method_files, restrict_to)
            for m in methods}
    empty = [m for m, g in sets.items() if len(g) == 0]      # e.g. SCG with no is_biased genes
    if empty:
        if verbose:
            print(f"dropped (empty set): {empty}")
        sets = {m: g for m, g in sets.items() if m not in empty}
    if equalize and sets:
        k = min(len(g) for g in sets.values())
        sets = {m: g[:k] for m, g in sets.items()}
        if verbose:
            print(f"equalized all sets to n={k}")
    if verbose:
        for m, g in sets.items():
            print(f"{m:6s}: {len(g):4d} genes")
    return sets


def plot_gene_set_responsiveness(
    gene_summary, gene_sets, metric='mean_abs_lfc',
    background_label='Non-memory', alternative='two-sided',
    style='boxen', figsize=(14, 8), fontsize=20,
    match_background=False, random_seed=0, background_genes=None,
    title=None, save_path=None, show=True,
):
    """
    One figure: per-gene responsiveness of each gene set vs a shared background.
    Mann-Whitney (set vs background) per set. Each column is annotated with its n;
    sets also show p-vs-background. Returns a stats DataFrame.

    background_genes : iterable or None
        Genes to use as the background group. If None (default) the background is
        measured genes in NONE of the sets. Pass an explicit list (e.g. genes no
        method called) when the sets overlap or are nested (consensus tiers).
    match_background : bool
        If True, randomly subsample the background down to the largest set's size
        (reproducible via random_seed) so it has the same n as the sets.
    """
    from matplotlib.transforms import blended_transform_factory

    measured = gene_summary.index
    union_all = set().union(*[set(g) for g in gene_sets.values()])

    # per-set responsiveness values (measured genes only)
    set_vals = {name: gene_summary.loc[measured.isin(set(genes)), metric].dropna().values
                for name, genes in gene_sets.items()}

    # background = explicit gene list if given, else measured genes in NONE of the sets
    bg_mask = measured.isin(set(background_genes)) if background_genes is not None \
        else ~measured.isin(union_all)
    bg_vals = gene_summary.loc[bg_mask, metric].dropna().values
    bg_n_full = len(bg_vals)
    if match_background and set_vals:
        target = max(len(v) for v in set_vals.values())
        if target < len(bg_vals):
            rng = np.random.default_rng(random_seed)
            bg_vals = rng.choice(bg_vals, size=target, replace=False)
    print(f"{background_label}: {len(bg_vals)} genes"
          + (f" (subsampled from {bg_n_full})" if match_background else "")
          + f"  |  measured universe = {len(measured)}")

    rows, stats = [], []
    for name, vals in set_vals.items():
        rows += [(name, v) for v in vals]
        _, p = (mannwhitneyu(vals, bg_vals, alternative=alternative)
                if len(vals) and len(bg_vals) else (np.nan, np.nan))
        stats.append({'set': name, 'n': len(vals),
                      'mean': np.mean(vals) if len(vals) else np.nan,
                      'median': np.median(vals) if len(vals) else np.nan,
                      'mwu_p_vs_bg': p})
    rows += [(background_label, v) for v in bg_vals]
    stats.append({'set': background_label, 'n': len(bg_vals),
                  'mean': float(np.mean(bg_vals)) if len(bg_vals) else np.nan,
                  'median': float(np.median(bg_vals)) if len(bg_vals) else np.nan,
                  'mwu_p_vs_bg': np.nan})

    plot_df = pd.DataFrame(rows, columns=['Group', 'value'])
    stats_df = pd.DataFrame(stats)
    stat_by_set = {s['set']: s for s in stats}
    order = list(gene_sets) + [background_label]
    palette = {n: c for n, c in zip(gene_sets, sns.color_palette('tab10', len(gene_sets)))}
    palette[background_label] = 'lightgray'

    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0, inner='box')
    plotter(data=plot_df, x='Group', y='value', order=order, palette=palette, ax=ax, **kw)

    bg_median = float(np.median(bg_vals))
    ax.axhline(bg_median, color='gray', ls=':', lw=1.5)

    # n (all columns) + p-vs-background (sets only), pinned to the top of each column
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, name in enumerate(order):
        s = stat_by_set[name]
        txt = f"n={int(s['n'])}"
        if name in gene_sets:
            txt += f"\np={s['mwu_p_vs_bg']:.1e}"
        ax.text(i, 0.99, txt, transform=trans, ha='center', va='top', fontsize=fontsize - 6)

    ax.set_xlabel('')
    ax.set_ylabel('Mean |LFC| across KOs', fontsize=fontsize)
    ax.set_title(title or 'Memory-gene responsiveness vs background', fontsize=fontsize + 2)
    ax.tick_params(axis='both', labelsize=fontsize - 2)

    # wrap long group names onto multiple lines so adjacent x-labels don't overlap
    import textwrap
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['\n'.join(textwrap.wrap(str(n), 12)) for n in order])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Memory-gene responsiveness figure saved to: {save_path}")
    if show:
        plt.show()
    return stats_df


def compute_gene_set_ko_response(lfc_df, gene_sets, background_label='Non-SCG'):
    """
    Per-KO mean |LFC| over each gene set's genes (and over the background = genes
    in NONE of the sets). Rows = KOs, columns = set names + background. Self-KO
    entries are already NaN in lfc_df, so they drop out of each row mean.
    """
    abs_lfc = lfc_df.abs()
    measured = lfc_df.columns
    union_all = set().union(*[set(g) for g in gene_sets.values()])
    out = {}
    for name, genes in gene_sets.items():
        cols = measured[measured.isin(set(genes))]
        out[name] = abs_lfc[cols].mean(axis=1)               # per-KO mean over set genes
    bg_cols = measured[~measured.isin(union_all)]
    out[background_label] = abs_lfc[bg_cols].mean(axis=1)
    return pd.DataFrame(out)


def plot_gene_set_ko_response(
    lfc_df, gene_sets, background_label='Non-SCG', alternative='two-sided',
    style='boxen', figsize=(14, 8), fontsize=20,
    title=None, save_path=None, show=True,
):
    """
    Per-KO companion to plot_gene_set_responsiveness: one dot = one knockout, y =
    mean |LFC| over a method's genes in that KO. Each set is tested against the
    background with a PAIRED Wilcoxon signed-rank test (same KOs in both), and
    every column is annotated with its n (number of KOs).
    Returns a stats DataFrame.
    """
    from matplotlib.transforms import blended_transform_factory
    from scipy.stats import wilcoxon

    ko_df = compute_gene_set_ko_response(lfc_df, gene_sets, background_label)
    bg = ko_df[background_label]

    stats = []
    for name in gene_sets:
        mask = ko_df[name].notna() & bg.notna()
        p = wilcoxon(ko_df.loc[mask, name], bg[mask])[1] if mask.sum() else np.nan
        stats.append({'set': name, 'n_ko': int(ko_df[name].notna().sum()),
                      'mean': float(ko_df[name].mean()), 'median': float(ko_df[name].median()),
                      'wilcoxon_p_vs_bg': p})
    stats.append({'set': background_label, 'n_ko': int(bg.notna().sum()),
                  'mean': float(bg.mean()), 'median': float(bg.median()),
                  'wilcoxon_p_vs_bg': np.nan})
    stats_df = pd.DataFrame(stats)
    stat_by_set = {s['set']: s for s in stats}

    order = list(gene_sets) + [background_label]
    plot_df = ko_df.melt(var_name='Group', value_name='value')
    palette = {n: c for n, c in zip(gene_sets, sns.color_palette('tab10', len(gene_sets)))}
    palette[background_label] = 'lightgray'

    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0, inner='box')
    plotter(data=plot_df, x='Group', y='value', order=order, palette=palette, ax=ax, **kw)

    ax.axhline(float(bg.median()), color='gray', ls=':', lw=1.5)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, name in enumerate(order):
        s = stat_by_set[name]
        txt = f"n={int(s['n_ko'])}"
        if name in gene_sets:
            txt += f"\np={s['wilcoxon_p_vs_bg']:.1e}"
        ax.text(i, 0.99, txt, transform=trans, ha='center', va='top', fontsize=fontsize - 6)

    ax.set_xlabel('')
    ax.set_ylabel('Per-KO mean |LFC| over gene set', fontsize=fontsize)
    ax.set_title(title or 'Per-knockout response of memory-gene sets', fontsize=fontsize + 2)
    ax.tick_params(axis='both', labelsize=fontsize - 2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-KO memory-gene figure saved to: {save_path}")
    if show:
        plt.show()
    return stats_df


# ---------------------------------------------------------------------------
# Stratified SCG-vs-Non-SCG responsiveness: recompute the per-gene metric
# separately within each level of a condition column (donor / dataset / etc.),
# each against ITS OWN control cells, and check the pattern is consistent.
# ---------------------------------------------------------------------------
def compute_gene_summary_by_condition(
    adata, condition_col, control_label='control', perturbation_col='perturbation',
    large_change_threshold=1.0, min_condition_cells=500, verbose=True, **lfc_kwargs
):
    """
    Recompute per-gene responsiveness (mean |LFC| across KOs) SEPARATELY within
    each level of adata.obs[condition_col]. Each level is pseudobulked against
    ITS OWN control cells, so the baseline is condition-matched.

    Extra keyword args (min_cells_per_ko, eff_threshold, chunk_size, ...) are
    forwarded to compute_pseudobulk_lfc.

    Returns {condition_value: gene_summary DataFrame}. Levels with no control
    cells or fewer than min_condition_cells cells are skipped with a warning.
    """
    if condition_col not in adata.obs.columns:
        raise KeyError(f"'{condition_col}' not in adata.obs.")
    cond = adata.obs[condition_col].astype(str).values
    pert = adata.obs[perturbation_col].astype(str).values

    summaries = {}
    for v in sorted(pd.unique(cond)):
        mask = cond == v
        n = int(mask.sum())
        n_ctrl = int((pert[mask] == control_label).sum())
        if n < min_condition_cells:
            if verbose:
                print(f"skip {condition_col}={v}: only {n} cells")
            continue
        if n_ctrl == 0:
            if verbose:
                print(f"skip {condition_col}={v}: no control cells")
            continue
        if verbose:
            print(f"--- {condition_col}={v}: {n} cells ({n_ctrl} control) ---")
        lfc_v = compute_pseudobulk_lfc(
            adata[mask], control_label=control_label,
            perturbation_col=perturbation_col, **lfc_kwargs,
        )
        summaries[v] = summarize_gene_response(lfc_v, large_change_threshold)
        del lfc_v
    return summaries


def plot_scg_by_condition(
    summaries, scg_genes, metric='mean_abs_lfc', alternative='two-sided',
    scg_label='SCG', bg_label='Non-SCG', style='boxen',
    figsize=(16, 8), fontsize=18, title=None, save_path=None, show=True,
):
    """
    Grouped figure: SCG vs Non-SCG per-gene responsiveness WITHIN each condition,
    so you can see whether the SCG > Non-SCG shift is consistent across levels.
    Mann-Whitney (SCG vs Non-SCG) per condition is annotated on top.

    summaries : {condition_value: gene_summary}  (from compute_gene_summary_by_condition)
    Returns a stats DataFrame.
    """
    from matplotlib.transforms import blended_transform_factory

    scg_set = set(scg_genes)
    rows, stats = [], []
    for cond, gs in summaries.items():
        is_scg = gs.index.isin(scg_set)
        scg_v = gs.loc[is_scg, metric].dropna().values
        bg_v = gs.loc[~is_scg, metric].dropna().values
        rows += [(str(cond), scg_label, v) for v in scg_v]
        rows += [(str(cond), bg_label, v) for v in bg_v]
        _, p = (mannwhitneyu(scg_v, bg_v, alternative=alternative)
                if len(scg_v) and len(bg_v) else (np.nan, np.nan))
        stats.append({'condition': str(cond), 'n_scg': len(scg_v), 'n_bg': len(bg_v),
                      'scg_median': float(np.median(scg_v)) if len(scg_v) else np.nan,
                      'bg_median': float(np.median(bg_v)) if len(bg_v) else np.nan,
                      'mwu_p': p})

    plot_df = pd.DataFrame(rows, columns=['condition', 'Group', 'value'])
    stats_df = pd.DataFrame(stats)
    order = [str(c) for c in summaries]
    palette = {bg_label: 'steelblue', scg_label: 'crimson'}

    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0)
    plotter(data=plot_df, x='condition', y='value', hue='Group', order=order,
            hue_order=[bg_label, scg_label], palette=palette, ax=ax, **kw)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    p_by_cond = {s['condition']: s['mwu_p'] for s in stats}
    for i, c in enumerate(order):
        ax.text(i, 0.99, f"p={p_by_cond[c]:.1e}", transform=trans,
                ha='center', va='top', fontsize=fontsize - 6)

    ax.set_xlabel('')
    ax.set_ylabel('Mean |LFC| across KOs', fontsize=fontsize)
    ax.set_title(title or f'{scg_label} vs {bg_label} responsiveness by condition',
                 fontsize=fontsize + 2)
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.legend(title='', fontsize=fontsize - 4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"By-condition figure saved to: {save_path}")
    if show:
        plt.show()
    return stats_df


def plot_scg_vs_nonscg_by_expression(
    gene_summary, gene_expr, scg_genes, metric='mean_abs_lfc',
    n_bins=10, alternative='two-sided',
    scg_label='SCG', bg_label='Non-SCG', style='boxen',
    figsize=(16, 8), fontsize=22, annot_fontsize=None,
    title=None, save_path=None, show=True,
):
    """
    Baseline check for the expression confound. Bin ALL genes into n_bins equal-size
    quantile bins of baseline expression, and compare SCG vs Non-SCG responsiveness
    WITHIN each bin. If the SCG shift persists across bins, it is not just an artifact
    of SCGs being more highly expressed.

    gene_expr : pd.Series (or 1-col DataFrame) of per-gene baseline expression
                (e.g. mean control expression), indexed by gene symbol.
    Returns a per-bin stats DataFrame (bin edges, n, medians, Mann-Whitney p).
    """
    from matplotlib.transforms import blended_transform_factory

    if isinstance(gene_expr, pd.DataFrame):          # tolerate a 1-col DataFrame
        gene_expr = gene_expr.iloc[:, 0]

    # align responsiveness + expression on shared genes
    df = pd.DataFrame({'value': gene_summary[metric]}).join(
        gene_expr.rename('expr'), how='inner').dropna()
    df['is_scg'] = df.index.isin(set(scg_genes))

    # equal-sized expression bins; ranking first breaks ties (many zero-expression
    # genes) so bins stay balanced instead of collapsing on duplicate edges.
    ranks = df['expr'].rank(method='first')
    df['bin'] = pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int)

    rows, stats = [], []
    for b in range(1, n_bins + 1):
        sub = df[df['bin'] == b]
        scg_v = sub.loc[sub['is_scg'], 'value'].values
        bg_v = sub.loc[~sub['is_scg'], 'value'].values
        rows += [(b, scg_label, v) for v in scg_v]
        rows += [(b, bg_label, v) for v in bg_v]
        _, p = (mannwhitneyu(scg_v, bg_v, alternative=alternative)
                if len(scg_v) and len(bg_v) else (np.nan, np.nan))
        stats.append({'bin': b,
                      'expr_min': float(sub['expr'].min()), 'expr_max': float(sub['expr'].max()),
                      'n_scg': len(scg_v), 'n_bg': len(bg_v),
                      'scg_median': float(np.median(scg_v)) if len(scg_v) else np.nan,
                      'bg_median': float(np.median(bg_v)) if len(bg_v) else np.nan,
                      'mwu_p': p})

    plot_df = pd.DataFrame(rows, columns=['bin', 'Group', 'value'])
    stats_df = pd.DataFrame(stats)
    order = list(range(1, n_bins + 1))
    palette = {bg_label: 'steelblue', scg_label: 'crimson'}

    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0)
    plotter(data=plot_df, x='bin', y='value', hue='Group', order=order,
            hue_order=[bg_label, scg_label], palette=palette, ax=ax, **kw)

    # add headroom at the top so the annotations sit in whitespace, clear of both
    # the tallest boxes and the title
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    p_by_bin = {s['bin']: (s['mwu_p'], s['n_scg']) for s in stats}
    for i, b in enumerate(order):
        p, n_scg = p_by_bin[b]
        ax.text(i, 0.98, f"p={p:.1e}\nn(SCG)={n_scg}", transform=trans,
                ha='center', va='top',
                fontsize=annot_fontsize if annot_fontsize is not None else fontsize - 4)

    ax.set_xlabel('Baseline-expression bin (low → high)', fontsize=fontsize)
    ax.set_ylabel('Mean |LFC| across KOs', fontsize=fontsize)
    ax.set_title(title or f'{scg_label} vs {bg_label} responsiveness within expression bins',
                 fontsize=fontsize + 2)
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.legend(title='', fontsize=fontsize - 4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Expression-stratified figure saved to: {save_path}")
    if show:
        plt.show()
    return stats_df


def compute_scg_effect_by_condition(
    summaries, scg_genes, metric='mean_abs_lfc', alternative='two-sided',
    n_boot=500, random_seed=0,
):
    """
    Per-condition SCG vs Non-SCG effect size for the responsiveness metric, in a
    form comparable ACROSS conditions.

    For each condition (a {name: gene_summary} entry) returns:
      - mwu_p         : Mann-Whitney p (SCG vs Non-SCG)
      - auc           : P(SCG > Non-SCG) = U / (n_scg * n_bg)   (0.5 = no effect)
      - cliffs_delta  : 2*auc - 1  in [-1, 1]   (scale-free effect size)
      - ci_low/ci_high: bootstrap 95% CI for cliffs_delta
      - scg_median, bg_median, median_diff, n_scg, n_bg

    Cliff's delta / AUC are rank-based, so they are comparable across conditions
    even though each condition has a different overall responsiveness magnitude.
    """
    rng = np.random.default_rng(random_seed)
    scg_set = set(scg_genes)
    rows = []
    for cond, gs in summaries.items():
        is_scg = gs.index.isin(scg_set)
        scg_v = gs.loc[is_scg, metric].dropna().values
        bg_v = gs.loc[~is_scg, metric].dropna().values
        n1, n2 = len(scg_v), len(bg_v)
        if n1 == 0 or n2 == 0:
            continue
        U, p = mannwhitneyu(scg_v, bg_v, alternative=alternative)
        auc = U / (n1 * n2)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            ub = mannwhitneyu(rng.choice(scg_v, n1, replace=True),
                              rng.choice(bg_v, n2, replace=True),
                              alternative=alternative)[0]
            boots[b] = 2 * (ub / (n1 * n2)) - 1
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            'condition': str(cond), 'n_scg': n1, 'n_bg': n2,
            'scg_median': float(np.median(scg_v)), 'bg_median': float(np.median(bg_v)),
            'median_diff': float(np.median(scg_v) - np.median(bg_v)),
            'auc': float(auc), 'cliffs_delta': float(2 * auc - 1),
            'ci_low': float(lo), 'ci_high': float(hi), 'mwu_p': float(p),
        })
    return pd.DataFrame(rows)


def plot_scg_effect_forest(
    effect_df, value_col='cliffs_delta', ci_cols=('ci_low', 'ci_high'),
    ref=0.0, sort=True, sig_alpha=0.05,
    figsize=(10, 8), fontsize=18, title=None, save_path=None, show=True,
):
    """
    Forest plot of the per-condition SCG-vs-Non-SCG effect (one row per condition)
    with bootstrap CIs and a 'no effect' reference line. If every point sits on
    the same side of `ref` with CIs clear of it, the SCG signal is consistent.

    value_col : 'cliffs_delta' (ref=0), 'auc' (pass ref=0.5), or 'median_diff'.
    Points are crimson where mwu_p < sig_alpha, gray otherwise.
    """
    df = effect_df.copy()
    if sort:
        df = df.sort_values(value_col).reset_index(drop=True)
    y = np.arange(len(df))
    colors = ['crimson' if s else 'gray' for s in (df['mwu_p'] < sig_alpha)]

    # The bootstrap CI is stored on the cliffs_delta scale. Put it on the SAME
    # scale as value_col so the whiskers bracket the dots (auc = (delta+1)/2).
    lo_col, hi_col = ci_cols
    ci_lo, ci_hi = df[lo_col].values, df[hi_col].values
    if value_col == 'auc' and (lo_col, hi_col) == ('ci_low', 'ci_high'):
        ci_lo, ci_hi = (ci_lo + 1) / 2, (ci_hi + 1) / 2
    elif value_col == 'median_dif' and (lo_col, hi_col) == ('ci_low', 'ci_high'):
        ci_lo = ci_hi = None                       # delta CI does not apply to median_diff

    fig, ax = plt.subplots(figsize=figsize)
    if ci_lo is not None:
        ax.hlines(y, ci_lo, ci_hi, color=colors, lw=2.5, alpha=0.7)
    ax.scatter(df[value_col], y, color=colors, s=70, zorder=3)
    ax.axvline(ref, color='black', ls='--', lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(df['condition'], fontsize=fontsize - 2)
    ax.set_ylim(-0.5, len(df) - 0.5)

    xlabels = {'cliffs_delta': "Cliff's delta  (SCG vs Non-SCG,  >0 = SCG more responsive)",
               'auc': 'P(SCG > Non-SCG)', 'median_dif': 'Median |LFC| difference (SCG − Non-SCG)'}
    ax.set_xlabel(xlabels.get(value_col, value_col), fontsize=fontsize)
    ax.set_title(title or 'SCG responsiveness effect across conditions', fontsize=fontsize + 2)
    ax.tick_params(axis='x', labelsize=fontsize - 2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forest plot saved to: {save_path}")
    if show:
        plt.show()
    return df


def plot_gene_responsiveness_panels(
    results_by_set, metric_label='Mean |LFC| across KOs',
    scg_label='Memory gene', bg_label='Non-memory gene',
    style='boxen', ncols=None, figsize=None, fontsize=22,
    sharey=True, suptitle=None, save_path=None, show=True,
):
    """
    One big figure with the per-gene distribution panel (the left panel of
    plot_gene_responsiveness) for several gene sets side by side. Only the
    distribution is drawn — no bootstrap-null panel.

    results_by_set : dict {name: results}
        Each `results` comes from compare_gene_responsiveness (uses scg_values /
        non_scg_values / mann_whitney_p / n_scg). Panel titles are the dict keys.
    """
    names = list(results_by_set)
    n = len(names)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.0 * ncols, 5.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey, squeeze=False)
    order = [scg_label, bg_label]
    palette = {bg_label: 'steelblue', scg_label: 'crimson'}
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0, inner='box')

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        res = results_by_set[name]
        scg_vals, bg_vals = res['scg_values'], res['non_scg_values']
        plot_df = pd.DataFrame({
            'value': np.concatenate([scg_vals, bg_vals]),
            'Group': [scg_label] * len(scg_vals) + [bg_label] * len(bg_vals),
        })
        plotter(data=plot_df, x='Group', y='value', order=order,
                hue='Group', hue_order=order, palette=palette, legend=False, ax=ax, **kw)
        ax.set_xlabel('')
        ax.set_ylabel(metric_label if (i % ncols == 0) else '', fontsize=fontsize)
        ax.set_title(f'{name}\np={res["mann_whitney_p"]:.1e}, n={res["n_scg"]}',
                     fontsize=fontsize + 2)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        # wrap long group names so 'Non-memory gene' / 'Memory gene' don't overlap
        import textwrap
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(['\n'.join(textwrap.wrap(str(t), 12)) for t in order])

    for j in range(n, nrows * ncols):            # hide unused axes
        axes[j // ncols][j % ncols].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Panel figure saved to: {save_path}")
    if show:
        plt.show()
    return fig, axes


def plot_responsiveness_by_perturbation_context(
    lfc_df, scg_genes, metric='mean_abs_lfc', large_change_threshold=1.0,
    alternative='two-sided', match_n=False, random_seed=0,
    scg_label='SCG gene', bg_label='Non-SCG gene',
    ko_labels=('SCG KO', 'Non-SCG KO'), style='boxen',
    figsize=(12, 8), fontsize=20, title=None, save_path=None, show=True,
):
    """
    Full 2x2 view of the per-gene responsiveness plot. Perturbations (KO rows of
    lfc_df) are split by whether the knocked-out gene is an SCG; within each
    perturbation subset the per-gene mean |LFC| is RECOMPUTED, then SCG vs Non-SCG
    *readout* genes are compared.

        x-axis = perturbation context (SCG KO vs Non-SCG KO)
        hue    = readout gene class (SCG vs Non-SCG)
        y      = mean |LFC| over that perturbation subset, per gene

    match_n : if True, subsample the Non-SCG readout genes to the SCG count within
              each perturbation context (reproducible via random_seed).
    Returns a stats DataFrame (n + Mann-Whitney p per context).
    """
    from matplotlib.transforms import blended_transform_factory

    scg_set = set(scg_genes)
    is_scg_ko = lfc_df.index.isin(scg_set)
    summaries = {
        ko_labels[0]: summarize_gene_response(lfc_df.loc[is_scg_ko], large_change_threshold),
        ko_labels[1]: summarize_gene_response(lfc_df.loc[~is_scg_ko], large_change_threshold),
    }
    print(f"{ko_labels[0]}: {int(is_scg_ko.sum())} perturbations   "
          f"{ko_labels[1]}: {int((~is_scg_ko).sum())} perturbations")

    rng = np.random.default_rng(random_seed)
    rows, stats = [], []
    for ctx, summary in summaries.items():
        is_scg_gene = summary.index.isin(scg_set)
        scg_v = summary.loc[is_scg_gene, metric].dropna().values
        bg_v = summary.loc[~is_scg_gene, metric].dropna().values
        if match_n and len(bg_v) > len(scg_v):
            bg_v = rng.choice(bg_v, len(scg_v), replace=False)
        rows += [(ctx, scg_label, v) for v in scg_v]
        rows += [(ctx, bg_label, v) for v in bg_v]
        _, p = mannwhitneyu(scg_v, bg_v, alternative=alternative)
        stats.append({'ko_context': ctx, 'n_scg': len(scg_v), 'n_bg': len(bg_v),
                      'scg_median': float(np.median(scg_v)), 'bg_median': float(np.median(bg_v)),
                      'mwu_p': float(p)})

    plot_df = pd.DataFrame(rows, columns=['ko_context', 'Group', 'value'])
    stats_df = pd.DataFrame(stats)
    order = list(ko_labels)
    palette = {bg_label: 'steelblue', scg_label: 'crimson'}

    fig, ax = plt.subplots(figsize=figsize)
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0)
    plotter(data=plot_df, x='ko_context', y='value', hue='Group', order=order,
            hue_order=[bg_label, scg_label], palette=palette, ax=ax, **kw)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    p_by = {s['ko_context']: s['mwu_p'] for s in stats}
    for i, ctx in enumerate(order):
        ax.text(i, 0.99, f"p={p_by[ctx]:.1e}", transform=trans,
                ha='center', va='top', fontsize=fontsize - 4)

    ax.set_xlabel('Perturbation context (gene knocked out)', fontsize=fontsize)
    ax.set_ylabel('Mean |LFC| over these perturbations', fontsize=fontsize)
    ax.set_title(title or 'Readout response (SCG vs Non-SCG), split by perturbation type',
                 fontsize=fontsize + 2)
    ax.tick_params(axis='both', labelsize=fontsize - 2)
    ax.legend(title='Readout gene', fontsize=fontsize - 4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2x2 responsiveness figure saved to: {save_path}")
    if show:
        plt.show()
    return stats_df


def compare_gene_responsiveness_expr_matched(
    gene_summary, gene_expr, scg_genes, metric='mean_abs_lfc',
    n_bins=10, n_iter=20, alternative='two-sided', random_seed=0,
):
    """
    Like compare_gene_responsiveness, but the comparison (Non-memory) group is
    EXPRESSION-MATCHED to the memory genes and averaged over several random draws.

    For each of n_iter iterations: within each of n_bins baseline-expression bins,
    sample as many non-memory genes as there are memory genes in that bin -> the
    sampled background has the same expression distribution as the memory set. The
    returned 'non_scg_values' pool all iterations (a stable box), and 'mann_whitney_p'
    is the MEDIAN Mann-Whitney p across iterations (each computed on matched n).

    Returns a results dict with the keys plot_gene_responsiveness_panels needs
    ('scg_values', 'non_scg_values', 'mann_whitney_p', 'n_scg', ...).
    """
    if isinstance(gene_expr, pd.DataFrame):
        gene_expr = gene_expr.iloc[:, 0]

    df = pd.DataFrame({'value': gene_summary[metric]}).join(
        gene_expr.rename('expr'), how='inner').dropna()
    df['is_scg'] = df.index.isin(set(scg_genes))
    df['bin'] = pd.qcut(df['expr'].rank(method='first'), n_bins, labels=False)

    scg_df = df[df['is_scg']]
    scg_vals = scg_df['value'].values
    scg_counts = scg_df.groupby('bin').size()                 # memory genes per bin
    bg_by_bin = {b: g['value'].values for b, g in df[~df['is_scg']].groupby('bin')}

    rng = np.random.default_rng(random_seed)
    draws, per_iter_p = [], []
    for _ in range(n_iter):
        draw = []
        for b, need in scg_counts.items():
            pool = bg_by_bin.get(b, np.array([]))
            if need == 0 or len(pool) == 0:
                continue
            draw.append(rng.choice(pool, size=int(need), replace=need > len(pool)))
        draw = np.concatenate(draw) if draw else np.array([])
        draws.append(np.sort(draw))                          # sort for order-statistic averaging
        if len(draw):
            per_iter_p.append(mannwhitneyu(scg_vals, draw, alternative=alternative)[1])

    # Average the sorted draws across iterations -> a stable Non-memory group with
    # the SAME n as the matched draw (~n_scg), not n_iter x that. Falls back to
    # pooling if draw sizes ever differ (a bin with memory but no non-memory genes).
    if draws and len({len(d) for d in draws}) == 1:
        non_scg_vals = np.vstack(draws).mean(axis=0)
    else:
        non_scg_vals = np.concatenate(draws) if draws else np.array([])

    return {
        'metric': metric, 'alternative': alternative, 'n_iter': n_iter,
        'scg_values': scg_vals, 'non_scg_values': non_scg_vals,
        'n_scg': int(len(scg_vals)), 'n_non_scg': int(len(non_scg_vals)),
        'scg_mean': float(scg_vals.mean()) if len(scg_vals) else np.nan,
        'non_scg_mean': float(non_scg_vals.mean()) if len(non_scg_vals) else np.nan,
        'mann_whitney_p': float(np.median(per_iter_p)) if per_iter_p else np.nan,
        'per_iter_p': per_iter_p,
    }


def plot_perturbation_context_by_readout(
    lfc_df, scg_genes, metric='mean_abs_lfc', large_change_threshold=1.0,
    metric_label='Mean |LFC| over perturbations',
    scg_label='SCG gene', bg_label='Non-SCG gene',
    ko_labels=('SCG KO', 'Non-SCG KO'), style='boxen',
    figsize=(14, 7), fontsize=20, sharey=True,
    suptitle=None, save_path=None, show=True,
):
    """
    Two panels, one per readout-gene class, each shown ACROSS perturbation contexts:

        Panel 1 (SCG readout)     : responsiveness over SCG-KO vs Non-SCG-KO perturbations
        Panel 2 (Non-SCG readout) : responsiveness over SCG-KO vs Non-SCG-KO perturbations

    Perturbations (KO rows of lfc_df) are split by whether the knocked-out gene is an
    SCG; the per-gene mean |LFC| is recomputed within each subset. Because each readout
    gene appears in BOTH contexts, the two boxes per panel are PAIRED -> Wilcoxon
    signed-rank p per panel. Returns a stats DataFrame.
    """
    from scipy.stats import wilcoxon

    scg_set = set(scg_genes)
    is_scg_ko = lfc_df.index.isin(scg_set)
    both = pd.DataFrame({
        ko_labels[0]: summarize_gene_response(lfc_df.loc[is_scg_ko], large_change_threshold)[metric],
        ko_labels[1]: summarize_gene_response(lfc_df.loc[~is_scg_ko], large_change_threshold)[metric],
    })
    both['is_scg'] = both.index.isin(scg_set)
    print(f"{ko_labels[0]}: {int(is_scg_ko.sum())} perturbations   "
          f"{ko_labels[1]}: {int((~is_scg_ko).sum())} perturbations")

    panels = [(scg_label, both[both['is_scg']]), (bg_label, both[~both['is_scg']])]
    palette = {ko_labels[0]: 'crimson', ko_labels[1]: 'steelblue'}
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    kw = {} if style == 'boxen' else dict(cut=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey)
    stats = []
    for ax, (readout_name, sub) in zip(axes, panels):
        paired = sub[list(ko_labels)].dropna()               # genes measured in both contexts
        long = paired.melt(var_name='context', value_name='value')
        plotter(data=long, x='context', y='value', order=list(ko_labels),
                hue='context', hue_order=list(ko_labels), palette=palette,
                legend=False, ax=ax, **kw)
        p = wilcoxon(paired[ko_labels[0]], paired[ko_labels[1]])[1] if len(paired) else np.nan
        stats.append({'readout': readout_name, 'n_genes': int(len(paired)),
                      f'median_{ko_labels[0]}': float(paired[ko_labels[0]].median()),
                      f'median_{ko_labels[1]}': float(paired[ko_labels[1]].median()),
                      'wilcoxon_p': float(p)})
        ax.set_xlabel('')
        ax.set_ylabel(metric_label, fontsize=fontsize)
        ax.set_title(f'{readout_name} readout\nWilcoxon p={p:.1e}, n={len(paired)}',
                     fontsize=fontsize + 1)
        ax.tick_params(axis='both', labelsize=fontsize - 2)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Perturbation-context-by-readout figure saved to: {save_path}")
    if show:
        plt.show()
    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# Directionality (sign coherence) of a gene set per perturbation.
#
# Everything above measures |LFC| (magnitude): "do memory genes respond MORE?"
# This asks the orthogonal question the sign carries: given each knockout, do
# the set's genes move COHERENTLY in one direction (all up / all down = a
# coordinated program shift) or scatter (some up, some down = noisy sensitivity)?
# ---------------------------------------------------------------------------
def compute_direction_by_ko(
    lfc_df, gene_set, background_pool=None, match_n=True,
    n_iter=50, random_seed=0,
):
    """
    Per-KO signed-direction summary over `gene_set` (signed LFC, NOT |LFC|).

    Rows = KOs. Columns:
      n_genes         : set genes measured in that KO (self-KO NaNs drop out)
      mean_signed_lfc : net directional push (~0 balanced, large |.| = one-sided)
      mean_abs_lfc    : magnitude over the same genes (ties back to responsiveness)
      frac_up         : share of set genes with LFC > 0
      coherence       : |mean signed LFC| / mean|LFC|  in [0,1]
                        0 = perfectly balanced up/down, 1 = every gene same way
      sign_p          : two-sided binomial test that up/down != 50/50 for this KO

    If match_n, also adds bg_frac_up / bg_coherence: the same per-KO metrics for a
    random same-size set drawn from `background_pool` (genes NOT in `gene_set` by
    default), averaged over `n_iter` draws -> a paired chance baseline per KO.
    """
    from scipy.stats import binomtest

    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    genes = [g for g in gene_set if g in measured]
    sub = lfc_df[genes]

    n_obs = sub.notna().sum(axis=1)
    n_up  = (sub > 0).sum(axis=1)
    mean_signed = sub.mean(axis=1, skipna=True)
    mean_abs    = sub.abs().mean(axis=1, skipna=True)

    per_ko = pd.DataFrame({
        'n_genes':         n_obs,
        'mean_signed_lfc': mean_signed,
        'mean_abs_lfc':    mean_abs,
        'frac_up':         n_up / n_obs.replace(0, np.nan),
        'coherence':       mean_signed.abs() / mean_abs.replace(0, np.nan),
    })
    per_ko['sign_p'] = [
        binomtest(int(u), int(n), 0.5).pvalue if n > 0 else np.nan
        for u, n in zip(n_up, n_obs)
    ]

    if match_n:
        if background_pool is None:
            in_set = set(genes)
            background_pool = [g for g in measured if g not in in_set]
        pool = np.array(background_pool)
        k = len(genes)
        bg_frac = np.full((len(per_ko), n_iter), np.nan)
        bg_coh  = np.full((len(per_ko), n_iter), np.nan)
        for j in range(n_iter):
            samp = rng.choice(pool, size=k, replace=False)
            bsub = lfc_df[list(samp)]
            bn = bsub.notna().sum(axis=1).replace(0, np.nan)
            bms = bsub.mean(axis=1, skipna=True)
            bg_frac[:, j] = ((bsub > 0).sum(axis=1) / bn).values
            bg_coh[:, j]  = (bms.abs() / bsub.abs().mean(axis=1, skipna=True)).values
        per_ko['bg_frac_up']   = np.nanmean(bg_frac, axis=1)
        per_ko['bg_coherence'] = np.nanmean(bg_coh, axis=1)

    return per_ko


def plot_consensus_direction_by_ko(
    per_ko, set_label='Consensus memory', bg_label='Matched non-memory',
    top_n=25, fontsize=18, figsize=(21, 7), title=None,
    save_path=None, show=True,
):
    """
    Three-panel view of directionality per perturbation (from compute_direction_by_ko):
      A: distribution of per-KO frac_up (set vs matched background). Mass pushed
         away from 0.5 = KOs drive the set coherently one way.
      B: distribution of per-KO coherence (set vs background), paired Wilcoxon.
      C: the top_n KOs by |net direction|, diverging bars colored up (red) / down
         (blue) -> which perturbations move the program, and which way.
    Returns a small summary dict.
    """
    from scipy.stats import wilcoxon

    has_bg = 'bg_coherence' in per_ko.columns
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- A: directional balance -------------------------------------------
    ax = axes[0]
    ax.hist(per_ko['frac_up'].dropna(), bins=30, density=True,
            color='tab:red', alpha=0.6, label=set_label)
    if has_bg:
        ax.hist(per_ko['bg_frac_up'].dropna(), bins=30, density=True,
                color='gray', alpha=0.5, label=bg_label)
    ax.axvline(0.5, color='k', ls=':', lw=1.5)
    ax.set_xlabel('Fraction of set genes up, per KO', fontsize=fontsize)
    ax.set_ylabel('density', fontsize=fontsize)
    ax.set_title('Directional balance per perturbation', fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 5)

    # --- B: coherence ------------------------------------------------------
    ax = axes[1]
    ax.hist(per_ko['coherence'].dropna(), bins=30, density=True,
            color='tab:red', alpha=0.6, label=set_label)
    p = np.nan
    if has_bg:
        ax.hist(per_ko['bg_coherence'].dropna(), bins=30, density=True,
                color='gray', alpha=0.5, label=bg_label)
        m = per_ko['coherence'].notna() & per_ko['bg_coherence'].notna()
        p = wilcoxon(per_ko.loc[m, 'coherence'], per_ko.loc[m, 'bg_coherence'])[1] if m.sum() else np.nan
    ax.set_xlabel('Coherence  |mean| / mean|LFC|', fontsize=fontsize)
    ax.set_ylabel('density', fontsize=fontsize)
    ax.set_title('Directional coherence per perturbation'
                 + (f'\npaired Wilcoxon p={p:.1e}' if has_bg else ''),
                 fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 5)

    # --- C: top KOs by net direction --------------------------------------
    ax = axes[2]
    top = (per_ko.reindex(per_ko['mean_signed_lfc'].abs()
                          .sort_values(ascending=False).index)
           .head(top_n)
           .sort_values('mean_signed_lfc'))
    colors = ['tab:blue' if v < 0 else 'tab:red' for v in top['mean_signed_lfc']]
    ax.barh(range(len(top)), top['mean_signed_lfc'].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=max(fontsize - 8, 6))
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Mean signed LFC over set', fontsize=fontsize)
    ax.set_title(f'Top {top_n} perturbations by |net direction|', fontsize=fontsize + 1)

    for ax in axes[:2]:
        ax.tick_params(axis='both', labelsize=fontsize - 4)
    axes[2].tick_params(axis='x', labelsize=fontsize - 4)

    fig.suptitle(title or 'Consensus memory-gene directionality per perturbation',
                 fontsize=fontsize + 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Directionality figure saved to: {save_path}")
    if show:
        plt.show()

    summary = {
        'n_ko': int(per_ko['mean_signed_lfc'].notna().sum()),
        'median_frac_up': float(per_ko['frac_up'].median()),
        'median_coherence': float(per_ko['coherence'].median()),
        'frac_ko_up_biased': float((per_ko['mean_signed_lfc'] > 0).mean()),
        'frac_ko_sig_directional': float((per_ko['sign_p'] < 0.05).mean()),
    }
    if has_bg:
        summary['median_bg_coherence'] = float(per_ko['bg_coherence'].median())
        summary['coherence_vs_bg_p'] = float(p)
    return summary


# ---------------------------------------------------------------------------
# Signed-LFC twin of plot_gene_set_responsiveness: instead of |LFC| magnitude,
# show the RAW (signed) LFC distribution per gene set as a ridgeline. A set whose
# ridge sits left of 0 is driven DOWN by perturbations; right of 0 = UP. This is
# the direction view the |LFC| plots throw away.
# ---------------------------------------------------------------------------
def plot_gene_set_lfc_ridge(
    lfc_df, gene_sets, background_label='Non-memory', background_genes=None,
    agg='pool', clip=(-3, 3), max_points=40000, grid=400, overlap=1.15,
    fontsize=18, figsize=(11, 9), title=None, save_path=None, show=True,
    random_seed=0,
):
    """
    Ridgeline of signed LFC, one row per gene set (+ background at the bottom).

    agg : how to reduce lfc_df (KOs x genes) to a value list per set
        'pool'      -> every (KO, gene) signed LFC entry  (full raw distribution)
        'gene_mean' -> per-gene mean signed LFC across KOs (per-gene direction;
                       the direct signed analog of the mean_abs_lfc plot)
        'ko_mean'   -> per-KO mean signed LFC over the set (per-perturbation push)

    Each ridge is annotated with its median and % of values below 0 (% down).
    Returns a stats DataFrame.
    """
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(random_seed)
    measured = lfc_df.columns
    union_all = set().union(*[set(g) for g in gene_sets.values()])

    groups = {name: measured[measured.isin(set(genes))] for name, genes in gene_sets.items()}
    bg_cols = (measured[measured.isin(set(background_genes))] if background_genes is not None
               else measured[~measured.isin(union_all)])
    groups[background_label] = bg_cols

    def values_for(cols):
        block = lfc_df[cols].values
        if agg == 'pool':
            v = block.ravel()
        elif agg == 'gene_mean':
            v = np.nanmean(block, axis=0)
        elif agg == 'ko_mean':
            v = np.nanmean(block, axis=1)
        else:
            raise ValueError(f"agg must be pool|gene_mean|ko_mean, got {agg!r}")
        return v[np.isfinite(v)]

    order = list(gene_sets) + [background_label]
    palette = {n: c for n, c in zip(gene_sets, sns.color_palette('tab10', len(gene_sets)))}
    palette[background_label] = 'lightgray'

    xs = np.linspace(clip[0], clip[1], grid)
    fig, ax = plt.subplots(figsize=figsize)
    stats = []
    # draw bottom-up so the first set in `order` ends up on top
    for i, name in enumerate(reversed(order)):
        v = values_for(groups[name])
        med = float(np.median(v))
        frac_down = float((v < 0).mean())
        stats.append({'set': name, 'n_values': int(v.size),
                      'median_lfc': med, 'frac_down': frac_down})

        vc = np.clip(v, *clip)
        if vc.size > max_points:
            vc = rng.choice(vc, max_points, replace=False)
        dens = gaussian_kde(vc)(xs)
        dens = dens / dens.max() * overlap
        base = float(i)
        color = palette.get(name, 'gray')
        ax.fill_between(xs, base, base + dens, color=color, alpha=0.75,
                        lw=1.2, edgecolor='black', zorder=i + 2)
        ax.plot([med, med], [base, base + np.interp(med, xs, dens)],
                color='black', lw=1.5, zorder=i + 2)
        ax.text(clip[0] + 0.03 * (clip[1] - clip[0]), base + 0.04,
                f'{name}   median={med:+.2f},  {frac_down*100:.0f}% down',
                fontsize=fontsize - 5, va='bottom', zorder=100)

    ax.axvline(0, color='red', ls='--', lw=1.8, zorder=1)
    ax.set_yticks([])
    ax.set_xlim(clip)
    ax.set_ylim(-0.1, len(order) - 1 + overlap + 0.2)
    ax.set_xlabel('Signed LFC (raw)', fontsize=fontsize)
    ax.set_title(title or 'Signed-LFC direction by gene set', fontsize=fontsize + 2)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    for s in ('left', 'right', 'top'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Signed-LFC ridge figure saved to: {save_path}")
    if show:
        plt.show()
    return pd.DataFrame(stats).set_index('set')


def plot_responsiveness_matched(
    gene_summary, scg_genes, background_genes, gene_expr, metric='mean_abs_lfc',
    ylabel='Mean |LFC| across KOs', n_bins=10, n_iter=20, n_perm=200,
    mem_name='SCG gene', bg_name='non-SCG gene',
    random_seed=0, style='boxen', fontsize=15, figsize=(7, 6),
    title=None, save_path=None, show=True,
):
    """
    SCG vs EXPRESSION-MATCHED non-SCG responsiveness (upgraded replacement for
    plot_gene_responsiveness). Non-SCG genes are binned by baseline expression
    (gene_expr) and matched to the SCG set. Significance is a permutation test:
    the SCG median responsiveness vs the medians of `n_perm` expression-matched
    random sets -> reports effect size (ratio), permutation z-score + p, and the
    intuitive 'beats N/n_perm' count. The plotted non-SCG distribution is the mean of
    the first n_iter matched draws' sorted values (equal n, stable).

    metric : column of gene_summary used as per-gene responsiveness (mean_abs_lfc).
    Returns a stats dict.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(random_seed)
    resp = gene_summary[metric].dropna()
    measured = resp.index
    expr = gene_expr.reindex(measured).dropna()

    mem = [g for g in scg_genes if g in resp.index and g in expr.index]
    pool = set(g for g in background_genes if g in resp.index and g not in set(mem))
    ebins = pd.Series(pd.qcut(expr, n_bins, labels=False, duplicates='drop'), index=expr.index)
    per_bin = ebins.loc[mem].value_counts()
    binpool = {b: ebins.index[(ebins == b) & (ebins.index.isin(pool))].to_numpy() for b in per_bin.index}

    n_perm = max(n_perm, n_iter)
    sorted_draws, null_medians = [], []
    for _ in range(n_perm):
        samp = []
        for b, need in per_bin.items():
            samp.extend(rng.choice(binpool[b], size=min(int(need), len(binpool[b])), replace=False))
        v = np.sort(resp[samp].values)
        sorted_draws.append(v); null_medians.append(np.median(v))
    null_medians = np.array(null_medians)

    L = min(len(d) for d in sorted_draws[:n_iter])
    bg_vals = np.mean([d[:L] for d in sorted_draws[:n_iter]], axis=0)

    mem_vals = resp[mem].values
    mem_med = float(np.median(mem_vals))
    n_beaten = int((null_medians < mem_med).sum())
    emp_p = float((np.sum(null_medians >= mem_med) + 1) / (n_perm + 1))
    null_mu, null_sd = float(null_medians.mean()), float(null_medians.std(ddof=1))
    z = (mem_med - null_mu) / null_sd if null_sd > 0 else np.nan
    z_p = float(norm.sf(z))
    ratio = mem_med / null_mu

    bg_label = f'{bg_name}\n(expr-matched)'
    mem_label = f'{mem_name}'
    plot_df = pd.DataFrame({
        metric: np.concatenate([bg_vals, mem_vals]),
        'Group': [bg_label] * len(bg_vals) + [mem_label] * len(mem_vals),
    })
    order = [scg_label, bg_label]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.9, figsize[1]))
    plotter = sns.boxenplot if style == 'boxen' else sns.violinplot
    plotter(data=plot_df, x='Group', y=metric, order=order,
            palette=['steelblue', 'crimson'], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=fontsize + 2)
    ax.set_title(f'n({mem_name})={len(mem_vals)}, n({bg_name})={len(bg_vals)}',
                 fontsize=fontsize - 2)
    ax.tick_params(labelsize=fontsize - 3)

    # right: expression-matched permutation null (replaces the old bootstrap panel)
    ax2.hist(null_medians, bins=30, color='lightgray', edgecolor='none')
    ax2.axvline(null_mu, color='steelblue', ls=':', lw=2,
                label=f'matched null median = {null_mu:.3f}')
    ax2.axvline(mem_med, color='crimson', ls='--', lw=2.5,
                label=f'{mem_name} median = {mem_med:.3f}')
    ax2.set_xlabel(f'median responsiveness of {n_perm} expr-matched sets', fontsize=fontsize - 2)
    ax2.set_ylabel('frequency', fontsize=fontsize - 2)
    ax2.set_title(f'{mem_name} beats {n_beaten}/{n_perm} matched sets\n'
                  f'z = {z:.1f},  p = {z_p:.1e}', fontsize=fontsize - 1)
    ax2.legend(fontsize=fontsize - 4)
    ax2.tick_params(labelsize=fontsize - 3)

    top = title or 'SCG vs expression-matched non-SCG responsiveness'
    fig.suptitle(f'{top}  —  {ratio:.2f}× more responsive', fontsize=fontsize + 1, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matched responsiveness figure saved to: {save_path}")
    if show:
        plt.show()
    return {'memory_median': mem_med, 'matched_median': null_mu, 'ratio': float(ratio),
            'z_score': float(z), 'z_p': z_p, 'empirical_p': emp_p,
            'beats_n_of_perm': f'{n_beaten}/{n_perm}', 'n_perm': n_perm, 'n': len(mem_vals)}


# ---------------------------------------------------------------------------
# Weinstock 2024 bulk perturb-seq: load, donor-corrected LFC, and the
# Cliff's-delta SCG-effect comparison (Marson vs pooled vs donor-corrected).
# ---------------------------------------------------------------------------
def load_weinstock_adata(counts_path, normalize_to=1e6, map_symbols=True):
    """
    Load the Weinstock GSE271788 featureCounts table into an AnnData (samples x genes,
    CPM-normalised, gene symbols). Sets obs['perturbation'] (AAVS1* -> 'control') and
    obs['donor'] (from the 'Donor_N_' prefix).
    """
    import scanpy as sc
    import anndata as ad

    df = pd.read_csv(counts_path, sep='\t', comment='#', index_col=0)
    counts = df.drop(columns=['Chr', 'Start', 'End', 'Strand', 'Length'])
    counts.columns = (counts.columns.str.replace(r'^.*/', '', regex=True)
                                    .str.replace(r'\.dedup\.bam$', '', regex=True))
    counts.index = counts.index.str.replace(r'\.\d+$', '', regex=True)
    target = (pd.Series(counts.columns).str.replace(r'^Donor_\d+_', '', regex=True)
              .str.replace(r'^AAVS1.*', 'control', regex=True))

    adata = ad.AnnData(
        X=counts.T.values.astype(np.float32),
        obs=pd.DataFrame({'perturbation': target.values}, index=counts.columns),
        var=pd.DataFrame(index=counts.index),
    )
    sc.pp.normalize_total(adata, target_sum=normalize_to)

    if map_symbols:
        import mygene
        res = mygene.MyGeneInfo().querymany(
            adata.var_names.tolist(), scopes='ensembl.gene',
            fields='symbol', species='human', as_dataframe=True)
        sym = res['symbol'].dropna()
        sym = sym[~sym.index.duplicated()]
        adata = adata[:, adata.var_names.isin(sym.index)].copy()
        adata.var_names = sym.reindex(adata.var_names).astype(str).values
        adata.var_names_make_unique()

    adata.obs['donor'] = (pd.Series(adata.obs_names, index=adata.obs_names)
                          .str.extract(r'^(Donor_\d+)_')[0])
    return adata


def compute_donor_corrected_lfc(adata, control_label='control', eps=1.0):
    """
    Donor-corrected LFC for Weinstock: compute each KO's LFC vs its OWN-donor control,
    then average per KO across donors (removes donor batch). Self-KO entries masked NaN.
    Returns (lfc_donor_corrected [KO x gene], per_donor_lfc dict).
    """
    X = pd.DataFrame(np.asarray(adata.X), index=adata.obs_names, columns=adata.var_names)
    pert = adata.obs['perturbation'].values
    donor = adata.obs['donor'].values

    per_donor = {}
    for dnr in pd.unique(donor):
        dm = donor == dnr
        ctrl = X.loc[dm & (pert == control_label)].mean(0)
        kos = {g: np.log2((X.loc[dm & (pert == g)].mean(0) + eps) / (ctrl + eps))
               for g in pd.unique(pert[dm]) if g != control_label}
        per_donor[dnr] = pd.DataFrame(kos).T

    genes = sorted(set.intersection(*[set(d.columns) for d in per_donor.values()]))
    kos = sorted(set().union(*[set(d.index) for d in per_donor.values()]))
    mat = pd.DataFrame(index=kos, columns=genes, dtype=float)
    for ko in kos:
        vecs = [d.loc[ko, genes].values for d in per_donor.values() if ko in d.index]
        mat.loc[ko] = np.nanmean(np.vstack(vecs), axis=0)
    for g in set(mat.index) & set(mat.columns):
        mat.loc[g, g] = np.nan
    return mat, per_donor


def cliffs_delta_ci(lfc, kos, readout, scg_set, metric='mean_abs_lfc', n_boot=500, seed=0):
    """
    Cliff's delta of SCG vs non-SCG responsiveness on `lfc`, restricted to `kos` x
    `readout`, with a bootstrap 95% CI. delta = 2*AUC - 1 (>0 = SCGs more responsive).
    Returns (delta, ci_lo, ci_hi).
    """
    from scipy.stats import mannwhitneyu
    sub = lfc.loc[lfc.index.isin(set(kos)), lfc.columns.isin(set(readout))]
    gs = summarize_gene_response(sub)
    a = gs.loc[gs.index.isin(scg_set), metric].dropna().values
    b = gs.loc[~gs.index.isin(scg_set), metric].dropna().values
    d = 2 * mannwhitneyu(a, b)[0] / (len(a) * len(b)) - 1
    rng = np.random.default_rng(seed)
    boots = [2 * mannwhitneyu(rng.choice(a, len(a), True), rng.choice(b, len(b), True))[0]
             / (len(a) * len(b)) - 1 for _ in range(n_boot)]
    return float(d), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def plot_cliffs_delta_comparison(
    lfc_by_label, kos, readout, scg_set, metric='mean_abs_lfc', colors=None,
    n_boot=500, fontsize=14, figsize=(8, 6), title=None, save_path=None, show=True,
):
    """
    Bar plot of Cliff's delta (SCG vs non-SCG responsiveness) for several LFC matrices
    on the SAME matched genes/perturbations -- e.g. Marson vs Weinstock-pooled vs
    Weinstock-donor-corrected. Error bars = bootstrap 95% CI. Returns a stats DataFrame.
    """
    labels = list(lfc_by_label)
    res = {lab: cliffs_delta_ci(lfc, kos, readout, scg_set, metric, n_boot)
           for lab, lfc in lfc_by_label.items()}
    d = [res[l][0] for l in labels]
    lo = [res[l][1] for l in labels]
    hi = [res[l][2] for l in labels]
    if colors is None:
        colors = (['#2b8cbe', '#bbbbbb', '#d6336c'] * len(labels))[:len(labels)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(labels)), d, color=colors, width=0.6)
    ax.errorbar(range(len(labels)), d, yerr=[np.subtract(d, lo), np.subtract(hi, d)],
                fmt='none', ecolor='black', capsize=6, lw=2)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=fontsize - 1, rotation=15, ha='right')
    ax.set_ylabel("Cliff's δ (SCG vs Non-SCG)", fontsize=fontsize)
    ax.set_title(title or 'SCG effect replicates once donor batch is corrected',
                 fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize - 2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cliff's-delta comparison saved to: {save_path}")
    if show:
        plt.show()
    return pd.DataFrame(res, index=['delta', 'ci_lo', 'ci_hi']).T


# ---------------------------------------------------------------------------
# Cross-dataset memory-gene responsiveness: compare each dataset's memory genes
# vs its 0-method expression-matched non-memory background, across all
# perturb-seq datasets (Marson/Weinstock/brain/melanoma) on one figure.
# ---------------------------------------------------------------------------
METHOD_ORDER  = ['CORAL', 'GEMLI', 'MOLD', 'POWERSEEK', 'SCG']
METHOD_COLORS = {'CORAL': '#d62728', 'GEMLI': '#ff7f0e', 'MOLD': '#2ca02c',
                 'POWERSEEK': '#1f77b4', 'SCG': '#9467bd'}


def cliffs_delta(a, b):
    """Cliff's delta = 2*P(a>b) - 1, in [-1, 1]. >0 => a tends to exceed b."""
    a, b = np.asarray(a), np.asarray(b)
    return 2 * mannwhitneyu(a, b, alternative='two-sided')[0] / (len(a) * len(b)) - 1


def cliffs_ci(a, b, n_boot=500, seed=0):
    """Bootstrap 95% CI for cliffs_delta(a, b)."""
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b)
    return np.percentile([cliffs_delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                          for _ in range(n_boot)], [2.5, 97.5])


def pct_ci(a, b, n_boot=500, seed=0):
    """Bootstrap 95% CI for the % median difference (median(a)/median(b) - 1)*100."""
    rng = np.random.default_rng(seed); a, b = np.asarray(a), np.asarray(b)
    return np.percentile([(np.median(rng.choice(a, len(a), True)) /
                           np.median(rng.choice(b, len(b), True)) - 1) * 100
                          for _ in range(n_boot)], [2.5, 97.5])


def cross_dataset_responsiveness(datasets, top_n=500, n_bins=30, n_iter=20,
                                 timepoint='merge', merge_mode='union', methods=None):
    """
    One row per (dataset, method): the responsiveness effect of that method's
    top-N memory genes vs the dataset's 0-method (called by no method)
    expression-matched non-memory background.

    datasets : list of dicts, each with keys
        name   : label for the dataset
        gs     : path to the per-gene responsiveness CSV (gene_summary_cpm)
        expr   : path to the control-expression CSV (mean_ctrl_expr in col 0)
        memdir : memory-gene directory for this dataset's lineage system
        files  : {method_name: filename} map inside memdir
      optional:
        case_insensitive : bool, default False. Set True when the memory-gene files
            and the perturb dataset live in DIFFERENT symbol cases -- e.g. the LARRY
            files are UPPERCASE while mouse gene_summary rows are title-case (Elane).
            Matching is then done on upper(), and the returned genes are translated
            back into the dataset's own symbols. Without it `restrict_to` matches
            nothing and the dataset silently contributes no rows.
        timepoint, merge_mode : per-dataset overrides of the global arguments, for
            when one dataset's timepoint labels differ (LARRY uses 2.0/4.0/6.0).

    timepoint : 'merge' or a specific timepoint label present in the files
        ('T1'/'T2' for T-lineage, 2.0/4.0/6.0 for LARRY, ...). 'merge' pools the
        timepoints, each gene keeping its BEST within-timepoint rank -- it does NOT
        require the gene to be called at every timepoint.
    merge_mode : 'union' | 'intersection'. Only used when timepoint == 'merge';
        'intersection' keeps only genes called at >= 2 timepoints.

    NOTE both the memory sets AND the background depend on top_n: the background is
    the set of measured genes called by ZERO methods at that top_n, so raising top_n
    shrinks and purifies the background as well as growing the memory sets.

    Returns a DataFrame with columns:
        dataset, method, n, cliffs_d, d_lo, d_hi, pct, pct_lo, pct_hi

    Thin wrapper over cross_dataset_topn_sweep() at a single cutoff (with CIs on).
    """
    cols = ['dataset', 'method', 'n', 'cliffs_d', 'd_lo', 'd_hi',
            'pct', 'pct_lo', 'pct_hi']
    S = cross_dataset_topn_sweep(datasets, top_ns=[top_n], n_bins=n_bins, n_iter=n_iter,
                                 timepoint=timepoint, merge_mode=merge_mode,
                                 methods=methods, ci=True)
    return S[cols] if len(S) else pd.DataFrame(columns=cols)


def cross_dataset_topn_sweep(datasets, top_ns=(100, 250, 500, 1000, 2000),
                             n_bins=30, n_iter=20, timepoint='merge',
                             merge_mode='union', methods=None, ci=False,
                             bg_top_n='max'):
    """
    cross_dataset_responsiveness() swept over several top-N cutoffs -> the long
    table plot_topn_heatmap() draws.

    Each dataset's two CSVs are read ONCE and reused across cutoffs; only the
    consensus-set build and the matched comparison repeat per cutoff. Bootstrap
    CIs are OFF by default -- they dominate the runtime (2 x 500 resamples per
    cell) and a heatmap has nowhere to show them.

    bg_top_n : 'max' (default) | int | 'per_cutoff'
        Which cutoff DEFINES the 0-method background (genes called by no method).

        'max' / int -> the background is built ONCE, at that single cutoff, and
            reused for every column. This is what a cutoff-on-the-x-axis figure
            needs: only the memory set may move along the axis, so a column-to-
            column change is unambiguously a property of the memory set.
        'per_cutoff' -> legacy behaviour: rebuild the background at each cutoff.
            Then a method's cell moves even when its OWN set is frozen, because
            the OTHER methods' sets grow and eat into the shared background --
            e.g. MOLD saturates by top-500 yet its top-1000 and top-2000 cells
            still differ. Do not use this under a cutoff axis.

        A fixed reference must be >= max(top_ns), else a gene could be in a
        method's top-N *and* in the background at the same cutoff; ints below
        that raise. 'max' picks max(top_ns), which also makes the single-cutoff
        wrapper cross_dataset_responsiveness() identical to 'per_cutoff'.

    ci : bool -- also compute d_lo/d_hi/pct_lo/pct_hi.

    Returns a long DataFrame:
        dataset, method, top_n, n, cliffs_d, pct [, d_lo, d_hi, pct_lo, pct_hi]
    with .attrs['bg_top_n'] recording the background definition (plot_topn_heatmap
    reads it to caption the figure).
    """
    top_ns = list(top_ns)
    if bg_top_n == 'max':
        bg_ref = max(top_ns)
    elif bg_top_n == 'per_cutoff':
        bg_ref = None
    else:
        bg_ref = int(bg_top_n)
        if bg_ref < max(top_ns):
            raise ValueError(f'bg_top_n={bg_ref} < max(top_ns)={max(top_ns)}: the '
                             'background would overlap the memory sets it is '
                             'compared against.')
    rows = []
    for ds in datasets:
        gs = pd.read_csv(ds['gs'], index_col=0)
        expr = pd.read_csv(ds['expr'], index_col=0).iloc[:, 0]
        tp = ds.get('timepoint', timepoint)
        mm = ds.get('merge_mode', merge_mode)

        # Symbol bridge: memory-file space -> dataset space (identity unless asked).
        if ds.get('case_insensitive'):
            up2sym = {}
            for g in gs.index.astype(str):
                up2sym.setdefault(g.upper(), g)
            universe = list(up2sym)                       # keys are in memory-file space
            to_data = lambda gl: [up2sym[x] for x in gl if x in up2sym]
        else:
            universe = gs.index
            to_data = lambda gl: [g for g in gl if g in gs.index]

        def _background(at):
            _, counts = build_consensus_gene_sets(
                min_methods=3, timepoint=tp, top_n=at, merge_mode=mm,
                memgene_dir=ds['memdir'], method_files=ds['files'],
                universe=universe, verbose=False)
            return to_data(counts.index[counts == 0].tolist())

        fixed_bg = None if bg_ref is None else _background(bg_ref)
        for top_n in top_ns:
            bg = fixed_bg if fixed_bg is not None else _background(top_n)
            bgset = set(bg)
            for method in (methods or list(ds['files'])):
                if method not in ds['files']:
                    continue
                mem = [g for g in to_data(load_memory_gene_set(method, tp, top_n, mm,
                                                               ds['memdir'], ds['files'],
                                                               restrict_to=universe))
                       if g not in bgset]
                if len(mem) < 10:
                    continue
                frame = gs.loc[list(dict.fromkeys(mem + bg))]
                m = compare_gene_responsiveness_expr_matched(
                    frame, expr, mem, metric='mean_abs_lfc', n_bins=n_bins, n_iter=n_iter)
                a, b = m['scg_values'], m['non_scg_values']
                med_a, med_b = np.median(a), np.median(b)  # not `mm`: that is merge_mode
                row = dict(dataset=ds['name'], method=method, top_n=top_n, n=len(mem),
                           cliffs_d=cliffs_delta(a, b), pct=(med_a / med_b - 1) * 100)
                if ci:
                    row['d_lo'], row['d_hi'] = cliffs_ci(a, b)
                    row['pct_lo'], row['pct_hi'] = pct_ci(a, b)
                rows.append(row)
            print(f"done: {ds['name']} (top{top_n})", flush=True)
    out = pd.DataFrame(rows)
    out.attrs['bg_top_n'] = 'per_cutoff' if bg_ref is None else bg_ref
    return out


def plot_cross_dataset_responsiveness(M, datasets, metric='pct', title=None,
                                      methods=METHOD_ORDER, colors=METHOD_COLORS,
                                      FS=16, save_path=None, show=True):
    """
    Per-dataset bars (median across methods) + per-method dots with 95% CI,
    from a cross_dataset_responsiveness() table.

    metric :
        'pct'     -> % median difference (0 = no effect)
        'winprob' -> P(memory out-responds non-memory) = (cliffs_d+1)/2 (0.5 = no effect)
        'pctile'  -> win probability as a percentile (50 = no effect)
    """
    d = M.copy()
    if metric == 'pct':
        d['val'], d['lo'], d['hi'], base = d['pct'], d['pct_lo'], d['pct_hi'], 0
        ylab = '% difference in KO responsiveness\n(memory vs matched non-memory · 0 = no effect)'
    elif metric == 'winprob':
        for c, s in [('val', 'cliffs_d'), ('lo', 'd_lo'), ('hi', 'd_hi')]:
            d[c] = (d[s] + 1) / 2
        base, ylab = 0.5, 'P(memory gene out-responds\nmatched non-memory) · 0.5 = no effect'
    elif metric == 'pctile':
        for c, s in [('val', 'cliffs_d'), ('lo', 'd_lo'), ('hi', 'd_hi')]:
            d[c] = (d[s] + 1) / 2 * 100
        base, ylab = 50, '% of ordinary genes a memory gene\nout-responds · 50 = no effect'
    else:
        raise ValueError("metric must be 'pct', 'winprob', or 'pctile'")

    order = [ds['name'] for ds in datasets if ds['name'] in set(d.dataset)]
    bar = d.groupby('dataset')['val'].median().reindex(order)
    n = len(methods)

    fig, ax = plt.subplots(figsize=(1.9 * len(order) + 3, 7))
    x = np.arange(len(order))
    ax.bar(x, bar.values - base, bottom=base, width=0.62,
           color='#b0b0b0', alpha=0.28, edgecolor='#808080', linewidth=2, zorder=1)
    for xi, name in enumerate(order):
        for _, r in d[d.dataset == name].iterrows():
            slot = (methods.index(r.method) - (n - 1) / 2) * 0.15
            ax.errorbar(xi + slot, r.val, yerr=[[r.val - r.lo], [r.hi - r.val]], fmt='o', ms=14,
                        color=colors[r.method], mec='w', mew=1.2,
                        ecolor=colors[r.method], elinewidth=1.8, capsize=4, zorder=3)
    ax.axhline(base, color='k', lw=1.5, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=FS, rotation=18, ha='right')
    ax.tick_params(axis='y', labelsize=FS - 1)
    ax.set_ylabel(ylab, fontsize=FS)
    ax.set_title(title or 'Difference in knockout responsiveness: memory vs non-memory genes',
                 fontsize=FS)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m],
                          markeredgecolor='w', markersize=15, label=m) for m in methods]
    ax.legend(handles=handles, fontsize=FS - 4, ncol=1, loc='upper right',
              framealpha=0.95, edgecolor='0.7', handletextpad=0.4, labelspacing=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-dataset responsiveness saved to: {save_path}")
    if show:
        plt.show()
    return fig, ax


def plot_topn_heatmap(S, datasets=None, methods=None, top_ns=None, metric='pct',
                      group='method', vmax=None, annot=True, FS=22,
                      cell=(1.05, 1.30), pad=(5.5, 2.4), title=None, note=None,
                      fs_ticks=None, fs_rows=None, fs_header=None, fs_xlabel=None,
                      save_path=None, show=True):
    """
    Heatmap of a cross_dataset_topn_sweep() table: one ROW per (perturb-seq
    dataset x lineage-barcode source), one COLUMN per (method, top-N cutoff),
    color = the effect size.

    group : 'method' -> columns blocked by method, cutoffs inside each block
                        (read one method's threshold sensitivity left-to-right)
            'top_n'  -> columns blocked by cutoff, methods inside each block
                        (compare methods head-to-head at a fixed cutoff)
    metric: 'pct'     -> % median difference, 0 = no effect (diverging at 0)
            'pctile'  -> % of ordinary genes a memory gene out-responds, 50 = none
            'winprob' -> P(memory out-responds non-memory), 0.5 = none
    vmax  : half-range of the color scale (default: 98th pct of |value - base|,
            so one outlier cell cannot flatten the rest of the map).

    Grey cells = that method has no set for that dataset (e.g. LARRY is SCG-only)
    or it fell under the 10-gene floor. Returns (fig, ax).
    """
    d = S.copy()
    if metric == 'pct':
        d['val'], base = d['pct'], 0.0
        clab = '% difference in KO responsiveness\n(memory vs matched non-memory · 0 = no effect)'
    elif metric == 'pctile':
        d['val'], base = (d['cliffs_d'] + 1) / 2 * 100, 50.0
        clab = '% of ordinary genes a memory gene\nout-responds · 50 = no effect'
    elif metric == 'winprob':
        d['val'], base = (d['cliffs_d'] + 1) / 2, 0.5
        clab = 'P(memory out-responds\nmatched non-memory) · 0.5 = no effect'
    else:
        raise ValueError("metric must be 'pct', 'pctile', or 'winprob'")

    top_ns = list(top_ns) if top_ns is not None else sorted(d['top_n'].unique())
    seen = set(d['method'])
    methods = [m for m in (methods or METHOD_ORDER + ['SCG-2-8']) if m in seen]
    rows = ([ds['name'] for ds in datasets if ds['name'] in set(d.dataset)]
            if datasets is not None else list(dict.fromkeys(d['dataset'])))

    cols = ([(m, t) for m in methods for t in top_ns] if group == 'method' else
            [(m, t) for t in top_ns for m in methods])
    blocks = [c[0] if group == 'method' else c[1] for c in cols]   # outer label
    inner = [c[1] if group == 'method' else c[0] for c in cols]

    A = (d.pivot_table(index='dataset', columns=['method', 'top_n'], values='val')
           .reindex(index=rows, columns=pd.MultiIndex.from_tuples(cols)).values)
    if vmax is None:
        finite = np.abs(A[np.isfinite(A)] - base)
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, 1e-9)

    # Diverging: two hues with a NEUTRAL GREY midpoint -- a white midpoint would
    # make "no effect" cells vanish into the figure background.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'div_grey', ['#2166ac', '#7fb0d3', '#eeeeee', '#e08b7d', '#b2182b']
    ).with_extremes(bad='#d9d9d9')

    # pad = inches reserved for the row labels + colour bar (width) and for the
    # column labels (height); at publication font sizes the default 5.5 in of
    # margin alone exceeds a 183 mm page
    fig, ax = plt.subplots(figsize=(cell[0] * len(cols) + pad[0],
                                    cell[1] * len(rows) + pad[1]))
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, aspect='auto',
                   vmin=base - vmax, vmax=base + vmax)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f'{v/1000:g}k' if isinstance(v, (int, np.integer)) and v >= 1000
                        else str(v) for v in inner], fontsize=fs_ticks or FS - 4)
    ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels(rows, fontsize=fs_rows or FS - 2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Hairline separators between every cell, heavier between method blocks.
    for i in range(1, len(rows)):
        ax.axhline(i - 0.5, color='w', lw=1.4)
    edges = [i for i in range(1, len(cols)) if blocks[i] != blocks[i - 1]]
    for i in range(1, len(cols)):
        ax.axvline(i - 0.5, color='w', lw=3.5 if i in edges else 1.4)

    # Block headers sit above the axis (data x, axes y).
    for b in dict.fromkeys(blocks):
        idx = [i for i, v in enumerate(blocks) if v == b]
        ax.text(np.mean(idx), 1.012, str(b), transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=fs_header or FS + 1,
                        fontweight='bold')
    # Say which background the column axis is measured against -- with a fixed
    # reference the axis is unambiguous, and the caption should prove it.
    if note is None:
        b = S.attrs.get('bg_top_n')
        note = ('background: genes called by NO method at top-%s, held fixed '
                'across all columns' % b if isinstance(b, (int, np.integer)) else
                'background: genes called by no method, rebuilt at each cutoff '
                '(moves with the x-axis)' if b == 'per_cutoff' else '')
    xlab = 'top-N cutof' if group == 'method' else 'method'
    ax.set_xlabel(xlab + (f'\n{note}' if note else ''), fontsize=fs_xlabel or FS - 2)

    if annot:
        for r in range(len(rows)):
            for c in range(len(cols)):
                v = A[r, c]
                if not np.isfinite(v):
                    continue
                txt = f'{v:.0f}' if abs(v) >= 10 else f'{v:.1f}'
                ax.text(c, r, txt, ha='center', va='center', fontsize=FS - 5,
                        color='w' if abs(v - base) > 0.62 * vmax else '#222222')

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015,
                      extend='both' if np.isfinite(A).any() else 'neither')
    cb.set_label(clab, fontsize=FS - 3); cb.ax.tick_params(labelsize=FS - 4)
    cb.outline.set_visible(False)
    ax.set_title(title or 'Memory-gene KO responsiveness across top-N cutoffs',
                 fontsize=FS + 3, pad=40)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Top-N heatmap saved to: {save_path}')
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Per-dataset responsiveness benchmark shared by the melanoma + mold notebooks:
# consensus/method groups -> median-|LFC| table -> line plot / method panels /
# UpSet. Each notebook supplies only its own config (paths, method files).
# ---------------------------------------------------------------------------
def median_ci(v, n_boot=500, seed=0):
    """Median of v plus its bootstrap 95% CI -> (median, lo, hi)."""
    rng = np.random.default_rng(seed); v = np.asarray(v)
    return np.median(v), *np.percentile([np.median(rng.choice(v, len(v), True))
                                         for _ in range(n_boot)], [2.5, 97.5])


def _bh(p):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = p[o] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[o[i]] = min(q[o[i]], q[o[i + 1]])
    return np.minimum(q, 1)


def build_benchmark_groups(memgene_dir, method_files, universe, top_n=500,
                           min_methods=3, merge_mode='union', methods=None,
                           exactly=False, include_union=True, strict_background=False):
    """
    Consensus tiers (+ optional UNION(>=1)) + each method's own top-N set, plus
    the non-memory background, counted over `universe` (measured genes).
    Returns (groups, background):
        groups     : {label: [genes]}  (tiers, ['UNION (>=1)'], then each method)
        background : [genes] treated as non-memory

    exactly : bool
        False -> nested '>=k methods' tiers (default).
        True  -> disjoint 'exactly k methods' tiers. Combine with min_methods=1
                 to show every agreement level.
    include_union : bool
        Add the 'UNION (>=1)' (any-method) group. Turn off in exact mode where a
        '>=' group is out of place.
    strict_background : bool
        False (default) -> background = genes in NO method's top-N set (rank-based;
                 can still contain a gene a loose method calls memory below its top-N).
        True  -> background = genes in NO method's FULL called set (top_n=None), so
                 it never contains any method's memory gene. Cleaner contrast, but
                 note loose callers (e.g. CORAL's weak Omega^2 floor) shrink it a lot.
    """
    methods = methods or list(method_files)
    # load each method's set first; drop any that come back empty (e.g. SCG with no
    # is_biased genes) so they don't appear in the plot or skew consensus counts.
    method_sets = {m: load_memory_gene_set(m, 'merge', top_n, merge_mode, memgene_dir,
                                           method_files, restrict_to=universe)
                   for m in methods}
    empty = [m for m, g in method_sets.items() if len(g) == 0]
    if empty:
        print(f"dropped (empty set): {empty}")
    methods = [m for m in methods if m not in empty]
    tiers, counts = build_consensus_gene_sets(
        min_methods=min_methods, timepoint='merge', top_n=top_n, merge_mode=merge_mode,
        memgene_dir=memgene_dir, method_files=method_files, methods=methods,
        universe=universe, exactly=exactly, verbose=False)
    if strict_background:                      # background = in NO method's FULL called set
        full_union = set()
        for mm in methods:
            full_union |= set(load_memory_gene_set(mm, 'merge', None, merge_mode,
                                                   memgene_dir, method_files, restrict_to=universe))
        background = [g for g in universe if g not in full_union]
    else:                                      # background = in no method's top-N set
        background = counts.index[counts == 0].tolist()
    groups = dict(tiers)
    if include_union:
        groups['UNION (>=1)'] = counts.index[counts >= 1].tolist()
    for m in methods:
        groups[m] = method_sets[m]
    return groups, background


def group_responsiveness_table(groups, background, gene_summary, control_expr,
                               n_bins=30, n_iter=20, min_genes=10):
    """
    For each group, compare its (non-background) genes' responsiveness to the
    shared 0-method `background`, expression-matched. Returns a DataFrame with
    median responsiveness (+CI) for memory and background, the ratio, Cliff's
    delta (+CI), one-sided MWU p, BH q, and a significance label. Feed to
    plot_group_responsiveness.
    """
    bg_set = set(background)
    rows = []
    for label, mem in groups.items():
        mem = [g for g in mem if g in gene_summary.index and g not in bg_set]
        if len(mem) < min_genes:
            print(f"SKIP {label}: {len(mem)} genes"); continue
        frame = gene_summary.loc[list(dict.fromkeys(mem + background))]
        m = compare_gene_responsiveness_expr_matched(
            frame, control_expr, mem, metric='mean_abs_lfc', n_bins=n_bins, n_iter=n_iter)
        a, b = m['scg_values'], m['non_scg_values']
        s_m, s_lo, s_hi = median_ci(a); b_m, b_lo, b_hi = median_ci(b)
        d_lo, d_hi = cliffs_ci(a, b)
        rows.append(dict(group=label, n=len(mem), scg=s_m, scg_lo=s_lo, scg_hi=s_hi,
                         bg=b_m, bg_lo=b_lo, bg_hi=b_hi, ratio=s_m / b_m,
                         cliffs_d=cliffs_delta(a, b), d_lo=d_lo, d_hi=d_hi,
                         p_greater=mannwhitneyu(a, b, alternative='greater')[1]))
    R = pd.DataFrame(rows)
    R['q'] = _bh(R['p_greater'])
    R['sig'] = R['q'].map(lambda q: '***' if q < 1e-3 else '**' if q < 1e-2
                          else '*' if q < 5e-2 else 'n.s.')
    return R


def plot_group_responsiveness(R, which='all', methods=METHOD_ORDER, title=None,
                              ylabel='median |LFC| (CPM)', FS=14, save_path=None, show=True):
    """
    Median responsiveness of memory (red) vs expr-matched non-memory (blue) per
    group, with 95% CIs and BH stars, from group_responsiveness_table.
    Points only (no connecting lines).

    which : 'all' | 'consensus' | 'methods'
        'consensus' -> only the cross-method overlap groups (>=k / exactly-k tiers
                       and 'UNION (>=1)'); 'methods' -> only the individual methods.
        Split is by label (consensus groups contain '>=', 'exactly', or 'UNION'),
        so any method name -- including 'SCG-2to8' -- lands in 'methods'.
    methods : kept for backward compatibility; no longer used for the split.
    """
    d = R.copy()
    is_consensus = d['group'].astype(str).str.contains(r'>=|exactly|UNION', regex=True)
    if which == 'consensus':
        d = d[is_consensus]
    elif which == 'methods':
        d = d[~is_consensus]
    d = d.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(1.6 * len(d) + 2, 6.5))
    x = np.arange(len(d))
    ax.errorbar(x - 0.08, d.scg, yerr=[d.scg - d.scg_lo, d.scg_hi - d.scg], fmt='o',
                color='crimson', capsize=5, markersize=9, label='memory gene set')
    ax.errorbar(x + 0.08, d.bg, yerr=[d.bg - d.bg_lo, d.bg_hi - d.bg], fmt='s',
                color='steelblue', capsize=5, markersize=9, label='expr-matched non-memory')
    for xi, r in enumerate(d.itertuples()):
        ax.annotate(r.sig, (xi - 0.08, r.scg_hi), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=FS + 2, fontweight='bold',
                    color='black' if r.sig != 'n.s.' else 'grey')
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\nn={n}" for g, n in zip(d.group, d.n)], fontsize=FS - 2)
    ax.tick_params(axis='y', labelsize=FS - 1)
    ax.set_ylabel(ylabel, fontsize=FS + 1)
    ax.set_title(title or 'Responsiveness vs expression-matched background', fontsize=FS + 3)
    ax.legend(fontsize=FS)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_method_panels(gene_summary, gene_expr, memgene_dir, method_files,
                       top_n=500, merge_mode='union', n_bins=5, n_iter=100,
                       methods=None, suptitle=None, save_path=None, show=True, fontsize=22):
    """
    Expression-matched responsiveness panels, one per method (memory vs
    expr-matched non-memory), via compare_gene_responsiveness_expr_matched +
    plot_gene_responsiveness_panels.
    """
    methods = methods or list(method_files)
    results = {}
    for method in methods:
        mem = load_memory_gene_set(method, 'merge', top_n, merge_mode,
                                   memgene_dir, method_files, restrict_to=gene_summary.index)
        if len(mem) < 10:                            # e.g. SCG with no is_biased genes -> skip
            print(f"skip {method}: {len(mem)} genes")
            continue
        results[method] = compare_gene_responsiveness_expr_matched(
            gene_summary, gene_expr, scg_genes=mem, n_bins=n_bins, n_iter=n_iter)
    return plot_gene_responsiveness_panels(
        results, scg_label='Memory gene', bg_label='Expr-matched non-memory',
        style='boxen', fontsize=fontsize, suptitle=suptitle, save_path=save_path, show=show)


def plot_memory_upset(memgene_dir, method_files, top_n=500, merge_mode='union',
                      methods=None, min_subset_size=5, title=None, save_path=None,
                      show=True, FS=22):
    """
    UpSet plot of the overlap between each method's top-N memory-gene set.
    """
    from upsetplot import from_contents, UpSet
    methods = methods or list(method_files)
    contents = {m: set(load_memory_gene_set(m, 'merge', top_n, merge_mode,
                                            memgene_dir, method_files))
                for m in methods}
    contents = {m: s for m, s in contents.items() if s}      # drop empty (e.g. SCG no is_biased)
    for m, s in contents.items():
        print(f"{m:11s} {len(s)} genes")
    with plt.rc_context({'font.size': FS, 'axes.labelsize': FS + 2,
                         'xtick.labelsize': FS - 2, 'ytick.labelsize': FS - 2}):
        fig = plt.figure(figsize=(20, 12))
        axd = UpSet(from_contents(contents), subset_size='count', sort_by='cardinality',
                    show_counts=True, min_subset_size=min_subset_size,
                    element_size=None).plot(fig=fig)
        for txt in axd['intersections'].texts:
            txt.set_fontsize(FS - 2)
        if title:
            plt.suptitle(title, y=1.02, fontsize=FS + 4)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
    return fig


def plot_memory_jaccard(memgene_dir, method_files, top_n=500, merge_mode='union',
                        methods=None, title=None, save_path=None, show=True, FS=13):
    """Heatmap of pairwise Jaccard overlap between each method's top-N memory set."""
    methods = methods or list(method_files)
    sets = {m: set(load_memory_gene_set(m, 'merge', top_n, merge_mode, memgene_dir, method_files))
            for m in methods}
    J = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            u = len(sets[a] | sets[b])
            J.loc[a, b] = (len(sets[a] & sets[b]) / u) if u else 0.0
    fig, ax = plt.subplots(figsize=(1.2 * len(methods) + 2, 1.0 * len(methods) + 1.5))
    sns.heatmap(J.astype(float).values, annot=True, fmt='.2', cmap='magma', vmin=0, vmax=1,
                xticklabels=methods, yticklabels=methods, ax=ax, annot_kws={'fontsize': FS})
    ax.set_title(title or f'Jaccard overlap, top {top_n}', fontsize=FS + 2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return J
