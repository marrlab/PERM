"""ClonePair memory-gene calling on the Harmange melanoma dataset (WM989).

A twin of findingSCGs.py adapted to this dataset's object. The statistics in
run_heritability_stats() are identical to the original; only the standardizer and
the defaults differ, and the defaults reproduce the published list:

  * grouping   -- cells are grouped by obs['batch'], the E and M sorted arms, so
                  the sister pairs and the shuffled "random" pairs both stay
                  inside one arm. Clones do not span arms, so no sister pair is
                  lost, and cross-arm random pairs, which would inflate the random
                  LFC, are avoided. Override with --group_col.
  * clone ids  -- obs carries `clone_id` and `clone_size`; used as-is.
  * gene names -- var.index already holds human gene symbols; they are copied to
                  var['human_symbol'] for output.
  * is_heritable -- not used for the call; is_biased is the ClonePair call.

Clone sizes: the paper uses every within-clone pair from clones of 2 to 8 cells
(--min_clone_size 2 --max_clone_size 8, the defaults). Above two cells the same
cell appears in several pairs, so the pairs are not independent and per-pair
t-test p-values are anti-conservative. --max_clone_size 2 restricts the call to
true two-cell sister pairs.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.sparse import issparse
from scipy.stats import distributions
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def standardize_melanoma(adata, group_col='batch', clone_col='clone_id', symbol_col=None):
    """Bring the melanoma AnnData in line with what run_heritability_stats expects.

    The object already has clone_id / clone_size and human gene symbols on var.index, so
    this mostly validates and copies symbols into var['human_symbol'].
    """
    adata = adata.copy()

    # gene symbols already sit on var.index; expose them as human_symbol.
    if symbol_col is not None and symbol_col in adata.var.columns:
        adata.var.index = pd.Index(adata.var[symbol_col].astype(str), name='gene')
    adata.var['human_symbol'] = adata.var.index.astype(str).values
    adata.var_names_make_unique()

    # clone ids: the object already has clone_id; only remap if a different col is named.
    if clone_col != 'clone_id' and clone_col in adata.obs.columns and 'clone_id' not in adata.obs.columns:
        adata.obs['clone_id'] = adata.obs[clone_col].astype(str)
    adata.obs['clone_id'] = adata.obs['clone_id'].astype(str)

    if 'clone_size' not in adata.obs.columns:
        adata.obs['clone_size'] = adata.obs['clone_id'].map(adata.obs['clone_id'].value_counts())

    if group_col not in adata.obs.columns:
        raise KeyError(f"group column {group_col!r} not in obs; available: {list(adata.obs.columns)}")

    groups = adata.obs[group_col].unique().tolist()
    return adata, groups, group_col


def _densify(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _within_clone_pairs(clone_positions, min_size=2, max_size=2):
    """Every C(n,2) pair within each clone whose size is in [min_size, max_size].

    max_size=2 reproduces the original definition exactly: a 2-cell clone yields
    exactly one pair, so this is backward-compatible.

    CAVEAT (pseudo-replication): for clones larger than 2, cells are REUSED across
    pairs -- an 8-cell clone contributes C(8,2)=28 pairs from only 8 cells. The
    t-test downstream treats those 28 as independent observations, which they are
    not. That inflates the effective n and makes the p-values anti-conservative
    (too small). The inflation grows with max_size: at max_size=4 it is ~1.5x,
    at max_size=8 ~3.5x. Kept as-is to match the collaborator's implementation.
    """
    a, b = [], []
    for pos in clone_positions:
        if not (min_size <= len(pos) <= max_size):
            continue
        ii, jj = np.triu_indices(len(pos), k=1)
        a.append(pos[ii]); b.append(pos[jj])
    if not a:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(a), np.concatenate(b)


def _random_cross_clone_pairs(labels, n_pairs, rng, cross_clone=True):
    """n_pairs random pairs; rejects self-pairs (and same-clone pairs if cross_clone).

    Samples WITH replacement across pairs -- necessary because the all-within-clone
    pair count can exceed n_cells/2, which makes disjoint sampling impossible.
    """
    n = len(labels)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    bad = (i == j)
    if cross_clone:
        bad |= (labels[i] == labels[j])
    while bad.any():
        i[bad] = rng.integers(0, n, size=bad.sum())
        j[bad] = rng.integers(0, n, size=bad.sum())
        bad = (i == j)
        if cross_clone:
            bad |= (labels[i] == labels[j])
    return i, j


def run_heritability_stats(adata, her_ensg_set, timepoints, time_col, n_iterations=100,
                           alpha_threshold=0.05, freq_threshold=0.80,
                           min_clone_size=2, max_clone_size=8, cross_clone=True,
                           min_gene_frac=0.05, seed=0):
    """Sister-concordant gene detection (identical to findingSCGs_bandler.py).

    min_clone_size/max_clone_size select which clones contribute pairs. The
    original definition is (2, 2) -- only true 2-cell clones, one pair each. Widen
    to (2, 8) to take ALL within-clone pairs; see _within_clone_pairs for the
    pseudo-replication caveat that introduces.

    Cells are restricted to the selected clones BEFORE the gene filter and before
    drawing the random pairs, so the null is drawn from the same small-clone pool
    as the sisters.
    """
    all_results = {}
    rng = np.random.default_rng(seed)

    for tp in timepoints:
        tp_mask = (adata.obs[time_col] == tp).values
        if np.sum(tp_mask) < 6: continue

        X_tp_full = _densify(adata[tp_mask].X)
        obs_tp = adata.obs[tp_mask]

        # --- restrict to clones of size min..max (both sisters AND the null pool) ---
        labels_all = obs_tp['clone_id'].to_numpy()
        counts = pd.Series(labels_all).value_counts()
        keep_clones = counts[(counts >= min_clone_size) & (counts <= max_clone_size)].index
        cell_mask = np.isin(labels_all, keep_clones)
        if cell_mask.sum() < 6: continue
        X_small = X_tp_full[cell_mask]
        labels = labels_all[cell_mask]

        gene_mask = (X_small > 0).mean(axis=0) >= min_gene_frac
        X_tp = X_small[:, gene_mask].astype(np.float32)
        var_tp = adata.var[gene_mask]
        n_cells, n_genes = X_tp.shape

        clone_positions = [np.where(labels == c)[0] for c in pd.unique(labels)]
        s1, s2 = _within_clone_pairs(clone_positions, min_clone_size, max_clone_size)
        if len(s1) < 3: continue

        sister_lfcs = np.abs(X_tp[s1] - X_tp[s2])
        n_pairs = len(s1)
        mean_sister_lfc = sister_lfcs.mean(axis=0)
        s1_var = np.var(sister_lfcs, axis=0, ddof=1)
        s1_n = n_pairs

        sig_matrix = np.zeros((n_iterations, n_genes), dtype=bool)
        random_lfc_sum = np.zeros(n_genes)

        print(f"   -> Group {tp}: {n_genes} genes, {len(keep_clones)} clones of size "
              f"{min_clone_size}-{max_clone_size} ({n_cells} cells) -> {n_pairs} within-clone pairs.")
        for i in tqdm(range(n_iterations), leave=False):
            r1, r2 = _random_cross_clone_pairs(labels, n_pairs, rng, cross_clone)
            rand_lfcs_raw = np.abs(X_tp[r1] - X_tp[r2])
            random_lfc_sum += rand_lfcs_raw.mean(axis=0)

            m1, m2 = mean_sister_lfc, rand_lfcs_raw.mean(axis=0)
            v2 = np.var(rand_lfcs_raw, axis=0, ddof=1)
            se = np.sqrt(s1_var/s1_n + v2/n_pairs)
            t_stat = (m1 - m2) / se
            df = 2 * n_pairs - 2
            p_vals = np.nan_to_num(distributions.t.cdf(t_stat, df), nan=1.0)
            reject, _, _, _ = multipletests(p_vals, alpha=alpha_threshold, method='fdr_bh')
            sig_matrix[i, :] = reject

        sig_freq = sig_matrix.mean(axis=0)            # continuous concordance: fraction of iterations significant
        mean_random_lfc = random_lfc_sum / n_iterations
        # effect size (pro's metric): sisters' LFC relative to random. < 1 for
        # concordant genes (sisters more similar than random); smaller = stronger.
        with np.errstate(divide='ignore', invalid='ignore'):
            lfc_ratio = mean_sister_lfc / mean_random_lfc
        results_df = pd.DataFrame({
            'sister_lfc': mean_sister_lfc,
            'random_lfc': mean_random_lfc,
            'lfc_ratio': lfc_ratio,
            'concordance_freq': sig_freq,
            'is_biased': sig_freq >= freq_threshold,
            # symbol-indexed object vs ENSG memory list: no annotation applied
            'is_heritable': np.zeros(n_genes, dtype=bool),
            'human_symbol': var_tp['human_symbol'].values if 'human_symbol' in var_tp.columns else var_tp.index.values
        }, index=var_tp.index)
        # rank: strongest effect first (lfc_ratio asc = sisters << random), ties
        # broken by most reliably-significant (concordance_freq desc)
        results_df = results_df.sort_values(['lfc_ratio', 'concordance_freq'], ascending=[True, False])

        all_results[tp] = results_df

    return all_results


def plot_heritability_diagonal(results_dict, dataset_name, plots_dir, freq_threshold=0.80):
    """Scatter of sister vs random LFC, one panel per group. No memory-gene layer."""
    n_panels = len(results_dict)
    if n_panels == 0: return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False, sharey=True)
    for i, (tp, df_tp) in enumerate(results_dict.items()):
        ax = axes[0, i]

        ax.scatter(df_tp.loc[~df_tp['is_biased'], 'sister_lfc'],
                   df_tp.loc[~df_tp['is_biased'], 'random_lfc'],
                   s=5, color='black', alpha=0.1, linewidth=0, label='Neutral')
        ax.scatter(df_tp.loc[df_tp['is_biased'], 'sister_lfc'],
                   df_tp.loc[df_tp['is_biased'], 'random_lfc'],
                   s=35, marker='^', facecolors='none', edgecolors='black', linewidth=0.7,
                   label=f'SCG (freq > {freq_threshold:.2f})')

        max_val = max(df_tp['random_lfc'].max(), df_tp['sister_lfc'].max())
        ax.plot([0, max_val], [0, max_val], color='grey', linestyle='--', alpha=0.4)
        ax.set_title(f"{dataset_name} - {tp}")
        if i == 0: ax.set_ylabel('Mean LFC (Random)')
        ax.set_xlabel('Mean LFC (Sisters)')
        ax.legend(loc='lower right', markerscale=1.5)

    plt.tight_layout()
    fig.savefig(plots_dir / f"{dataset_name}_diagonal_scatters_thr{int(freq_threshold*100)}.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='ClonePair memory-gene calling on the Harmange melanoma dataset '
                    '(WM989, E and M sorted arms). The defaults reproduce the published list.')
    parser.add_argument('--data', type=str,
                        default=f'{P.DATA}/datasets/Melanoma_Harmange_processed.h5ad',
                        help='Path to the processed Harmange h5ad')
    parser.add_argument('--name', type=str, default='Melanoma_Harmange',
                        help='Dataset name used in outputs; the ranked list goes to '
                             '<benchmark_dir>/<name>/')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--benchmark_dir', type=str, default=None,
                        help='Where the ranked lists go (default: $PERM_MEMGENES_ROOT/'
                             'python_implements, which is what the figures read)')
    parser.add_argument('--tag', type=str, default='clones2-8',
                        help="Filename tag appended to every output, e.g. "
                             "scg_memory_genes_clones2-8_E_M.csv")
    parser.add_argument('--group_col', type=str, default='batch',
                        help="obs column to group by, stands in for timepoint "
                             "(default: batch = the E and M sorted arms)")
    parser.add_argument('--freq_threshold', type=float, default=0.80,
                        help='Frequency threshold for biased (SCG) gene classification')
    parser.add_argument('--min_clone_size', type=int, default=2,
                        help='Smallest clone size contributing pairs (default 2)')
    parser.add_argument('--max_clone_size', type=int, default=8,
                        help='Largest clone size contributing pairs (default 8: every '
                             'within-clone pair from clones of 2-8 cells, the definition used '
                             'throughout the paper). See _within_clone_pairs for why cells '
                             'are reused across pairs above 2.')
    parser.add_argument('--min_gene_frac', type=float, default=0.05,
                        help='Keep genes detected in >= this fraction of the selected cells (default 0.05)')
    parser.add_argument('--no_cross_clone', action='store_true',
                        help='Allow random pairs to fall within one clone (original behaviour). '
                             'By default random pairs are forced cross-clone.')
    parser.add_argument('--n_iterations', type=int, default=100, help='Shuffling iterations')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for the shuffling')
    args = parser.parse_args()

    np.random.seed(args.seed)

    name = args.name
    tag = f"_{args.tag}" if args.tag else ""   # distinguishes runs by FILENAME within one dir (no extra dirs)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_dir is None:
        benchmark_dir = Path(P.MEMGENES)        # python_implements: what the figures read
    else:
        benchmark_dir = Path(args.benchmark_dir)

    print(f">>> Analyzing Dataset: {name}  (clone sizes {args.min_clone_size}-{args.max_clone_size})")
    ad = sc.read_h5ad(args.data)
    ad, groups, group_col = standardize_melanoma(ad, group_col=args.group_col)
    print(f"    {ad.n_obs} cells x {ad.n_vars} genes; grouping by {group_col!r} -> {groups}")

    res_dict = run_heritability_stats(ad, set(), groups, group_col,
                                      n_iterations=args.n_iterations,
                                      freq_threshold=args.freq_threshold,
                                      min_clone_size=args.min_clone_size,
                                      max_clone_size=args.max_clone_size,
                                      cross_clone=not args.no_cross_clone,
                                      min_gene_frac=args.min_gene_frac,
                                      seed=args.seed)

    if not res_dict:
        print("No group had enough sister pairs. Nothing written.")
        return

    plot_heritability_diagonal(res_dict, name + tag, plots_dir, freq_threshold=args.freq_threshold)

    # Ranked gene list per group, next to the other methods' outputs so top-N is
    # directly comparable across methods. EVERY tested gene in rank order (lfc_ratio
    # asc, then concordance_freq desc) -- is_biased kept as a column, not a filter.
    scg_frames = []
    for tp, df in res_dict.items():
        scg = df.copy()                           # already rank-sorted (lfc_ratio, concordance_freq)
        scg.insert(0, 'timepoint', tp)            # keep this name: analysis.ipynb keys off 'timepoint'
        scg_frames.append(scg)
    if scg_frames:
        ds_dir = benchmark_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        tp_suffix = '_'.join(str(tp) for tp in res_dict.keys())
        fpath = ds_dir / f'scg_memory_genes{tag}_{tp_suffix}.csv'
        pd.concat(scg_frames).to_csv(fpath, index_label='gene')
        n_biased = sum(int(f['is_biased'].sum()) for f in scg_frames)
        print(f"   Saved {sum(len(f) for f in scg_frames)} ranked genes "
              f"({n_biased} pass is_biased) -> {fpath}")

    print("\nBuilding Master DataFrame...")
    bias_summary = {tp: set(df.index[df['is_biased']]) for tp, df in res_dict.items()}
    all_biased_genes = sorted(set().union(*bias_summary.values())) if bias_summary else []
    master_df = pd.DataFrame(index=list(all_biased_genes))

    tp_cols = []
    for tp, gene_set in bias_summary.items():
        col_name = f"{name}_{tp}"
        master_df[col_name] = master_df.index.isin(gene_set)
        tp_cols.append(col_name)
    if tp_cols:
        master_df[f"{name}_Any"] = master_df[tp_cols].any(axis=1)

    suffix = f"_{int(args.freq_threshold*100)}"
    master_df.to_csv(out_dir / f'master_heritability_results{tag}{suffix}.csv')
    print(f"Saved {len(master_df)} SCGs -> master_heritability_results{tag}{suffix}.csv")

    # continuous concordance frequency (pre-threshold), ALL tested genes
    conc_cols = [df['concordance_freq'].rename(f"{name}_{tp}") for tp, df in res_dict.items()]
    if conc_cols:
        ds_block = pd.concat(conc_cols, axis=1)
        if len(conc_cols) > 1:
            ds_block[f"{name}_max"] = ds_block.max(axis=1)   # parallels boolean _Any
        ds_block.to_csv(out_dir / f'concordance_frequency{tag}{suffix}.csv')
        print(f"Saved continuous concordance scores -> concordance_frequency{tag}{suffix}.csv")

    for tp, df in res_dict.items():
        print(f"   {name} {tp}: {int(df['is_biased'].sum())} SCGs of {len(df)} genes tested")

    print(f"\nAnalysis complete. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
