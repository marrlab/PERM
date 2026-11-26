"""ClonePair memory-gene calling on the Bandler embryonic forebrain dataset
(GSE188528, Bandler et al. 2021). The defaults reproduce the published list.

A twin of findingSCGs.py adapted to this dataset's object. The statistics in
run_heritability_stats() are the same as the original; what differs:

  * grouping   -- obs['batch'] is the labelling age, E13 or E14. Both groups are
                  harvested at the same age, so this is a labelling-age contrast,
                  not a time course. Override with --group_col.
  * batch      -- each group pools four 10x libraries and clones never span a
                  library, so the shuffled "random" pairs are drawn within one
                  library (--batch_col library). A pooled null compares cells
                  across a boundary the sisters never cross; measured on the full
                  dataset it inflated the random LFC by up to 5.5%, i.e. 31-40% of
                  the apparent effect.
  * cells      -- the published run uses the 10,170-cell matched core (harvested
                  at P7, one strain, whole forebrain, STICR barcoding), so that the
                  labelling age is the only difference between the two groups.
  * clone ids  -- obs column is `cloneID`, not `clone_id`.
  * gene names -- the mouse symbols are taken from var['var_names'].
  * is_heritable -- not used for the call; is_biased is the ClonePair call.

Clone sizes: the paper uses every within-clone pair from clones of 2 to 8 cells
(the defaults). Above two cells the same cell appears in several pairs, so the
pairs are not independent and per-pair t-test p-values are anti-conservative.
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


def standardize_brain(adata, group_col='batch', clone_col='cloneID', symbol_col='var_names'):
    """Bring the brain AnnData in line with what run_heritability_stats expects."""
    adata = adata.copy()

    # gene symbols: var.index is a positional range, real symbols sit in var['var_names']
    if symbol_col in adata.var.columns:
        adata.var.index = pd.Index(adata.var[symbol_col].astype(str), name='gene')
        adata.var['human_symbol'] = adata.var.index.values
    adata.var_names_make_unique()

    # clone ids: `cloneID` here vs `clone_id` in the other datasets
    if clone_col in adata.obs.columns and 'clone_id' not in adata.obs.columns:
        adata.obs['clone_id'] = adata.obs[clone_col].astype(str)

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


def _random_within_batch_pairs(labels, batches, sister_batches, rng, cross_clone=True):
    """Random pairs that never cross a batch boundary.

    Sister pairs are always within one batch (a clone cannot span libraries), so
    a null drawn across the whole group compares cells that differ by batch in a
    way the sisters never do. That gap is read as concordance. Measured on the
    Bandler forebrain data, pooling libraries inside an age group inflated the
    random LFC by up to 5.5%, which is 31-40% of the apparent effect at E12/E14.

    Pairs are allocated per batch in the same proportion as the sister pairs, so
    the null keeps the sisters' batch composition as well as their pair count.
    """
    i_all, j_all = [], []
    for b, k in zip(*np.unique(sister_batches, return_counts=True)):
        idx = np.where(batches == b)[0]
        # cross_clone rejection cannot terminate if the batch offers no second
        # clone to pair against; skip rather than spin.
        if len(idx) < 2 or (cross_clone and len(np.unique(labels[idx])) < 2):
            print(f"      ! batch {b!r}: too few clones for a cross-clone null, "
                  f"{k} random pairs skipped")
            continue
        li, lj = _random_cross_clone_pairs(labels[idx], int(k), rng, cross_clone)
        i_all.append(idx[li]); j_all.append(idx[lj])
    if not i_all:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(i_all), np.concatenate(j_all)


def run_heritability_stats(adata, her_ensg_set, timepoints, time_col, n_iterations=100,
                           alpha_threshold=0.05, freq_threshold=0.80,
                           min_clone_size=2, max_clone_size=8, cross_clone=True,
                           min_gene_frac=0.05, seed=0, batch_col=None):
    """Sister-concordant gene detection.

    min_clone_size/max_clone_size select which clones contribute pairs. The
    original definition is (2, 2) -- only true 2-cell clones, one pair each. The
    collaborator's revision widens this to (2, 8) [her committed call uses (2, 4)]
    and takes ALL within-clone pairs; see _within_clone_pairs for the
    pseudo-replication caveat that introduces.

    Cells are restricted to the selected clones BEFORE the gene filter and before
    drawing the random pairs, so the null is drawn from the same small-clone pool
    as the sisters (this matches the collaborator's code, which calls
    find_concordant_genes on the already-subset `adata_small`).
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

        # batch_col=None reproduces the original pooled null exactly.
        if batch_col is not None:
            batches = obs_tp[batch_col].astype(str).to_numpy()[cell_mask]
            sister_batches = batches[s1]
            assert (sister_batches == batches[s2]).all(), \
                f"{batch_col!r} does not contain whole clones; a sister pair spans two batches"
        else:
            batches = sister_batches = None

        sister_lfcs = np.abs(X_tp[s1] - X_tp[s2])
        n_pairs = len(s1)
        mean_sister_lfc = sister_lfcs.mean(axis=0)
        s1_var = np.var(sister_lfcs, axis=0, ddof=1)
        s1_n = n_pairs

        sig_matrix = np.zeros((n_iterations, n_genes), dtype=bool)
        random_lfc_sum = np.zeros(n_genes)

        null_desc = f"within {batch_col}" if batch_col is not None else "pooled"
        print(f"   -> Group {tp}: {n_genes} genes, {len(keep_clones)} clones of size "
              f"{min_clone_size}-{max_clone_size} ({n_cells} cells) -> {n_pairs} within-clone pairs "
              f"({null_desc} null).")
        for i in tqdm(range(n_iterations), leave=False):
            if batch_col is not None:
                r1, r2 = _random_within_batch_pairs(labels, batches, sister_batches,
                                                    rng, cross_clone)
            else:
                r1, r2 = _random_cross_clone_pairs(labels, n_pairs, rng, cross_clone)
            rand_lfcs_raw = np.abs(X_tp[r1] - X_tp[r2])
            random_lfc_sum += rand_lfcs_raw.mean(axis=0)

            # n_rand == n_pairs unless a batch was skipped above, so this is the
            # same arithmetic as before whenever batch_col is None.
            n_rand = len(r1)
            m1, m2 = mean_sister_lfc, rand_lfcs_raw.mean(axis=0)
            v2 = np.var(rand_lfcs_raw, axis=0, ddof=1)
            se = np.sqrt(s1_var/s1_n + v2/n_rand)
            t_stat = (m1 - m2) / se
            df = n_pairs + n_rand - 2
            p_vals = np.nan_to_num(distributions.t.cdf(t_stat, df), nan=1.0)
            reject, _, _, _ = multipletests(p_vals, alpha=alpha_threshold, method='fdr_bh')
            sig_matrix[i, :] = reject

        sig_freq = sig_matrix.mean(axis=0)            # continuous concordance: fraction of iterations significant
        mean_random_lfc = random_lfc_sum / n_iterations
        # effect size (prof's metric): sisters' LFC relative to random. < 1 for
        # concordant genes (sisters more similar than random); smaller = stronger.
        with np.errstate(divide='ignore', invalid='ignore'):
            lfc_ratio = mean_sister_lfc / mean_random_lfc
        results_df = pd.DataFrame({
            'sister_lfc': mean_sister_lfc,
            'random_lfc': mean_random_lfc,
            'lfc_ratio': lfc_ratio,
            'concordance_freq': sig_freq,
            'is_biased': sig_freq >= freq_threshold,
            # mouse dataset vs human memory list, no ortholog mapping applied
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
        description='ClonePair memory-gene calling on the Bandler forebrain dataset. '
                    'The defaults reproduce the published list.')
    parser.add_argument('--data', type=str,
                        default=f'{P.DATA}/bandler_brain/bandler_brain.h5ad',
                        help='Path to bandler_brain.h5ad (the 10,170-cell matched core)')
    parser.add_argument('--name', type=str, default='BandlerBrain', help='Dataset name used in outputs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--benchmark_dir', type=str, default=None,
                        help='Where the ranked lists go (default: $PERM_MEMGENES_ROOT/'
                             'python_implements, which is what the figures read)')
    parser.add_argument('--tag', type=str, default='clones2-8',
                        help="Filename tag appended to every output, e.g. "
                             "scg_memory_genes_clones2-8_E13_E14.csv")
    parser.add_argument('--group_col', type=str, default='batch',
                        help="obs column to group by (default: batch = labelling age, E13 / E14)")
    parser.add_argument('--batch_col', type=str, default='library',
                        help="obs column holding the batch (default: library). The shuffled "
                             "random pairs are drawn WITHIN one batch and allocated to match "
                             "the sister pairs' batch composition, so a batch difference is not "
                             "scored as concordance. Pass an empty string for a pooled null.")
    parser.add_argument('--freq_threshold', type=float, default=0.80,
                        help='Frequency threshold for the is_biased call (default 0.80)')
    parser.add_argument('--min_clone_size', type=int, default=2,
                        help='Smallest clone size contributing pairs (default 2)')
    parser.add_argument('--max_clone_size', type=int, default=8,
                        help='Largest clone size contributing pairs (default 8: every '
                             'within-clone pair from clones of 2-8 cells, the definition used '
                             'throughout the paper). See _within_clone_pairs for why cells are '
                             'reused across pairs above 2.')
    parser.add_argument('--min_gene_frac', type=float, default=0.05,
                        help='Keep genes detected in >= this fraction of the selected cells '
                             '(default 0.05)')
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
    args.batch_col = args.batch_col or None     # '' -> pooled null

    print(f">>> Analyzing Dataset: {name}")
    ad = sc.read_h5ad(args.data)
    ad, groups, group_col = standardize_brain(ad, group_col=args.group_col)
    print(f"    {ad.n_obs} cells x {ad.n_vars} genes; grouping by {group_col!r} -> {groups}")

    # NOTE: the original findingSCGs.py never forwards freq_threshold here, so its
    # is_biased is always cut at 0.80 regardless of --freq_threshold. Fixed here.
    res_dict = run_heritability_stats(ad, set(), groups, group_col,
                                      n_iterations=args.n_iterations,
                                      freq_threshold=args.freq_threshold,
                                      min_clone_size=args.min_clone_size,
                                      max_clone_size=args.max_clone_size,
                                      cross_clone=not args.no_cross_clone,
                                      min_gene_frac=args.min_gene_frac,
                                      seed=args.seed,
                                      batch_col=args.batch_col)

    if not res_dict:
        print("No group had enough sister pairs. Nothing written.")
        return

    plot_heritability_diagonal(res_dict, name + tag, plots_dir, freq_threshold=args.freq_threshold)

    # Ranked gene list per group, next to the other methods' outputs so top-N is
    # directly comparable across methods.
    #
    # We write EVERY tested gene in rank order (lfc_ratio asc, then
    # concordance_freq desc) -- NOT only the is_biased ones. The ranking is
    # threshold-independent and stays informative even when no gene clears
    # freq_threshold, which happens on a small dataset: 90/24 sister pairs is
    # under the ~94 needed for even the strongest gene, so is_biased is empty
    # while lfc_ratio still orders the genes sensibly. `is_biased` is kept as a
    # column, so nothing is claimed about these genes being memory genes --
    # filter on it to recover the old behaviour.
    #
    # This also matters mechanically: analysis.ipynb sets TOP_N =
    # min(rows per method per timepoint), so a 0-row SCG file would silently
    # collapse the whole cross-method comparison to top-0.
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
    master_df.to_csv(out_dir / f'master_heritability_results{suffix}.csv')
    print(f"Saved {len(master_df)} SCGs -> master_heritability_results{suffix}.csv")

    # continuous concordance frequency (pre-threshold), ALL tested genes
    conc_cols = [df['concordance_freq'].rename(f"{name}_{tp}") for tp, df in res_dict.items()]
    if conc_cols:
        ds_block = pd.concat(conc_cols, axis=1)
        if len(conc_cols) > 1:
            ds_block[f"{name}_max"] = ds_block.max(axis=1)   # parallels boolean _Any
        ds_block.to_csv(out_dir / f'concordance_frequency{suffix}.csv')
        print(f"Saved continuous concordance scores -> concordance_frequency{suffix}.csv")

    for tp, df in res_dict.items():
        print(f"   {name} {tp}: {int(df['is_biased'].sum())} SCGs of {len(df)} genes tested")

    print(f"\nAnalysis complete. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
