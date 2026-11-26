# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
from perm_paths import MEMGENES  # noqa: E402
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import argparse
from pathlib import Path
from scipy.sparse import issparse
from scipy.stats import distributions
from statsmodels.stats.multitest import multipletests
from upsetplot import from_contents, plot
from tqdm import tqdm


# standardize timepoint columns
def classify_and_standardize(adata):
    ensg_column = 'gene_ids'
    time_col = next((col for col in ['batch', 'timepoint', 'Time point'] if col in adata.obs.columns), 'batch')
    if time_col not in adata.obs.columns:
        adata.obs['batch'] = ' '
        time_col = 'batch'
    timepoints = adata.obs[time_col].unique().tolist()
    return adata, timepoints, time_col, ensg_column

# identifying sister-concordant genes
def _within_clone_pairs(clone_ids, clone_sizes, min_clone_size, max_clone_size):
    """Positional within-clone pairs (all C(n,2)) for cells whose clone_size is in
    [min_clone_size, max_clone_size]. With [2,2] this is exactly one pair per
    true size-2 clone; the paper's [2,8] adds every within-clone pair. Returns (idx1, idx2) int arrays into the timepoint slice.
    NOTE: clones >2 give C(n,2) pseudo-replicated pairs, so the t-test below is
    anti-conservative -- effective n is the clone count, not the pair count."""
    in_win = (np.asarray(clone_sizes) >= min_clone_size) & (np.asarray(clone_sizes) <= max_clone_size)
    pos = np.where(in_win)[0]
    if pos.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    cids = np.asarray(clone_ids)[pos]
    order = np.argsort(cids, kind='stable')
    pos, cids = pos[order], cids[order]
    idx1, idx2, start, n = [], [], 0, len(cids)
    for i in range(1, n + 1):
        if i == n or cids[i] != cids[start]:
            grp = pos[start:i]
            if grp.size >= 2:
                a, b = np.triu_indices(grp.size, k=1)
                idx1.append(grp[a]); idx2.append(grp[b])
            start = i
    if not idx1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(idx1), np.concatenate(idx2)


def run_heritability_stats(adata, her_ensg_set, timepoints, time_col, n_iterations=100, alpha_threshold=0.05, freq_threshold=0.80, min_clone_size=2, max_clone_size=2):
    all_results = {}
    
    for tp in timepoints:
        tp_mask = (adata.obs[time_col] == tp).values
        n_obs_tp = np.sum(tp_mask)
        if n_obs_tp < 6: continue

        # Slice data for this TP and densify only this slice
        X_tp_full = adata[tp_mask].X
        if issparse(X_tp_full):
            X_tp_full = X_tp_full.toarray()
        
        gene_mask = (X_tp_full > 0).sum(axis=0) >= (n_obs_tp * 0.05)
        X_tp = X_tp_full[:, gene_mask].astype(np.float32)
        var_tp = adata.var[gene_mask]
        n_cells, n_genes = X_tp.shape
        
        adata_tp_obs = adata.obs[tp_mask]
        # within-clone pairs over the [min,max] clone-size window (default [2,2] =
        # original one-pair-per-size-2-clone sister definition; widen to 2-8 etc.)
        p1, p2 = _within_clone_pairs(adata_tp_obs['clone_id'].values,
                                     adata_tp_obs['clone_size'].values,
                                     min_clone_size, max_clone_size)
        
        if len(p1) < 3: continue

        n_pairs = len(p1)
        sister_lfcs = np.abs(X_tp[p1] - X_tp[p2])
        mean_sister_lfc = sister_lfcs.mean(axis=0)
        s1_var = np.var(sister_lfcs, axis=0, ddof=1)
        s1_n = n_pairs

        sig_matrix = np.zeros((n_iterations, n_genes), dtype=bool)
        random_lfc_sum = np.zeros(n_genes)

        print(f"   -> Timepoint {tp}: {n_genes} genes, {n_pairs} within-clone pairs "
              f"(clone_size {min_clone_size}-{max_clone_size}).")
        for i in tqdm(range(n_iterations), leave=False):
            if 2 * n_pairs <= n_cells:
                shuffled = np.random.choice(n_cells, size=n_pairs * 2, replace=False)
                idx1, idx2 = shuffled[:n_pairs], shuffled[n_pairs:]
            else:
                # widened windows can make n_pairs exceed n_cells/2 (pseudo-replication);
                # draw the null WITH replacement so it still has n_pairs pairs
                idx1 = np.random.choice(n_cells, size=n_pairs, replace=True)
                idx2 = np.random.choice(n_cells, size=n_pairs, replace=True)
                coincide = idx1 == idx2
                while coincide.any():
                    idx2[coincide] = np.random.choice(n_cells, size=int(coincide.sum()), replace=True)
                    coincide = idx1 == idx2
            rand_lfcs_raw = np.abs(X_tp[idx1] - X_tp[idx2])
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
            # Legacy annotation. Guarded on BOTH the set and the column: with no
            # --memory_path the set is empty, and a rebuilt object may not carry
            # 'gene_ids' at all (LARRY_processed.h5ad now stores the Ensembl id as
            # 'human_ensembl_id'). Falling back to all-False matches what
            # findingSCGs_{bandler,melanoma}.py already hardcode, and affects no
            # SCG statistic.
            'is_heritable': (var_tp['gene_ids'].astype(str).isin(her_ensg_set).values
                             if her_ensg_set and 'gene_ids' in var_tp.columns
                             else np.zeros(len(var_tp), dtype=bool)),
            'human_symbol': var_tp['human_symbol'].values if 'human_symbol' in var_tp.columns else var_tp.index.values
        }, index=var_tp.index)
        # rank: strongest effect first (lfc_ratio asc = sisters << random), ties
        # broken by most reliably-significant (concordance_freq desc)
        results_df = results_df.sort_values(['lfc_ratio', 'concordance_freq'], ascending=[True, False])

        all_results[tp] = results_df
        
    return all_results

# generate and save scatter plots for sister vs. random LFC
def plot_heritability_diagonal(results_dict, dataset_name, plots_dir, freq_threshold=0.80, tag=''):
    """Generate and save scatter plots for sister vs random LFC."""
    n_panels = len(results_dict)
    if n_panels == 0: return
    
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False, sharey=True)
    for i, (tp, df_tp) in enumerate(results_dict.items()):
        ax = axes[0, i]
        
        # 1. Neutral
        ax.scatter(df_tp.loc[~df_tp['is_biased'] & ~df_tp['is_heritable'], 'sister_lfc'], 
                   df_tp.loc[~df_tp['is_biased'] & ~df_tp['is_heritable'], 'random_lfc'], 
                   s=5, color='black', alpha=0.1, linewidth=0, label='Neutral')
        # 2. Neutral Memory
        ax.scatter(df_tp.loc[~df_tp['is_biased'] & df_tp['is_heritable'], 'sister_lfc'], 
                   df_tp.loc[~df_tp['is_biased'] & df_tp['is_heritable'], 'random_lfc'], 
                   s=10, color='#d62728', alpha=0.4, linewidth=0, label='Neutral (Memory)')

        label_scg = f'SCG (freq > {freq_threshold:.2f})'

        # 3. SCG
        ax.scatter(df_tp.loc[df_tp['is_biased'] & ~df_tp['is_heritable'], 'sister_lfc'], 
                   df_tp.loc[df_tp['is_biased'] & ~df_tp['is_heritable'], 'random_lfc'], 
                   s=35, marker='^', facecolors='none', edgecolors='black', linewidth=0.7, label=label_scg)
        # 4. SCG Memory
        ax.scatter(df_tp.loc[df_tp['is_biased'] & df_tp['is_heritable'], 'sister_lfc'], 
                   df_tp.loc[df_tp['is_biased'] & df_tp['is_heritable'], 'random_lfc'], 
                   s=80, marker='^', color='#d62728', edgecolors='black', label='SCG (Memory)')

        max_val = max(df_tp['random_lfc'].max(), df_tp['sister_lfc'].max())
        ax.plot([0, max_val], [0, max_val], color='grey', linestyle='--', alpha=0.4)
        ax.set_title(f"{dataset_name} - TP {tp}")
        if i == 0: ax.set_ylabel('Mean LFC (Random)')
        ax.set_xlabel('Mean LFC (Sisters)')
    
    plt.tight_layout()
    tag_part = f'{tag}_' if tag else ''
    fig.savefig(plots_dir / f"{dataset_name}_{tag_part}diagonal_scatters_thr{int(freq_threshold*100)}.png", dpi=300)
    plt.close(fig)
    

def main(adata_dict=None, memory_path=None, output_dir=None, freq_threshold=0.80, benchmark_dir=None, min_clone_size=2, max_clone_size=8, tag='clones2-8', seed=0):
    if adata_dict is None:
        parser = argparse.ArgumentParser(description='Run Heritability SCG Analysis')
        parser.add_argument('--memory_path', type=str, default=None,
                            help="OPTIONAL legacy annotation: CSV of 'potential heritable genes' "
                                 "used only to fill the is_heritable column and to emit "
                                 "memory_heritability_results_<NN>.csv. It does NOT affect any SCG "
                                 "call -- sister_lfc / random_lfc / lfc_ratio / concordance_freq / "
                                 "is_biased are all computed without it. Omit it (the default) and "
                                 "is_heritable is False throughout, exactly as findingSCGs_bandler.py "
                                 "and _melanoma.py already do.")
        parser.add_argument('--output_dir', type=str, default='heritability_results', help='Output directory')
        parser.add_argument('--data', nargs=2, action='append', metavar=('NAME', 'PATH'), help='Dataset name and path')
        parser.add_argument('--freq_threshold', type=float, default=0.80, help='Frequency threshold for biased gene classification')
        parser.add_argument('--min_clone_size', type=int, default=2, help='Smallest clone size contributing pairs (default 2).')
        parser.add_argument('--max_clone_size', type=int, default=8, help='Largest clone size contributing pairs (default 8: every within-clone pair from clones of 2-8 cells, the definition used throughout the paper). Above 2 the pairs share cells, so the t-test is anti-conservative (effective n = clone count). 2 = true two-cell sister pairs only.')
        parser.add_argument('--tag', type=str, default='clones2-8', help="Label inserted into output filenames, e.g. scg_memory_genes_clones2-8_T1_T2.csv.")
        parser.add_argument('--seed', type=int, default=0, help='RNG seed for the random pairing (as findingSCGs_bandler/_melanoma.py). The published T-lineage and LARRY lists use seed 0.')
        parser.add_argument('--benchmark_dir', type=str, default=None, help='Where the ranked SCG lists go (<dir>/<NAME>/); default perm_paths.MEMGENES.')
        args = parser.parse_args()
        
        mem_path = Path(args.memory_path) if args.memory_path else None
        out_dir = Path(args.output_dir)
        datasets_to_process = {name: path for name, path in args.data}
        freq_threshold = args.freq_threshold
        min_clone_size = args.min_clone_size
        max_clone_size = args.max_clone_size
        tag = args.tag
        seed = args.seed
        benchmark_dir = args.benchmark_dir
    else:
        # Notebook mode
        mem_path = Path(memory_path) if memory_path else None
        out_dir = Path(output_dir)
        datasets_to_process = adata_dict

    # the random pairing in run_heritability_stats draws from the global numpy RNG
    np.random.seed(seed)
    print(f"Random pairing seed: {seed}")

    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # where to drop the ranked SCG lists so they sit next to the other methods'
    # outputs (perm_paths.MEMGENES/<dataset>/); overridable via benchmark_dir
    if benchmark_dir is None:
        benchmark_dir = Path(MEMGENES)
    else:
        benchmark_dir = Path(benchmark_dir)

    # Legacy annotation only. An empty set leaves is_heritable False everywhere,
    # which is already the hardcoded behaviour of the three sibling scripts.
    if mem_path is not None:
        print(f"Loading Memory Gene List from {mem_path}...")
        her_ensg_set = set(pd.read_csv(mem_path)['gene_id'].dropna().astype(str).tolist())
    else:
        print("No --memory_path given: is_heritable will be False for all genes "
              "(this does not affect any SCG call).")
        her_ensg_set = set()
    
    dataset_bias_summary = {}
    concordance_summary = {}        # NEW: per-dataset continuous concordance frequencies
    core_symbol_mapping = {}

    for name, data_input in datasets_to_process.items():
        print(f"\n>>> Analyzing Dataset: {name}")
        
        if isinstance(data_input, (str, Path)):
            ad = sc.read_h5ad(data_input)
        else:
            ad = data_input
        
        ad, tps, time_col, _ = classify_and_standardize(ad)
        res_dict = run_heritability_stats(ad, her_ensg_set, tps, time_col,
                                          min_clone_size=min_clone_size, max_clone_size=max_clone_size)
        
        if res_dict:
            plot_heritability_diagonal(res_dict, name, plots_dir, freq_threshold=freq_threshold, tag=tag)
            dataset_bias_summary[name] = {tp: set(df.index[df['is_biased']]) for tp, df in res_dict.items()}
            concordance_summary[name] = {tp: df['concordance_freq'] for tp, df in res_dict.items()}
            for tp, df in res_dict.items():
                biased_genes = df[df['is_biased']]
                for ensg, sym in zip(biased_genes.index, biased_genes['human_symbol']):
                    core_symbol_mapping[ensg] = sym

            # additionally save the ranked SCG list per dataset/timepoint next to
            # the other methods, so top-N is directly comparable across methods
            scg_frames = []
            for tp, df in res_dict.items():
                scg = df[df['is_biased']].copy()          # already rank-sorted (lfc_ratio, concordance_freq)
                scg.insert(0, 'timepoint', tp)
                scg_frames.append(scg)
            if scg_frames:
                ds_dir = benchmark_dir / name
                ds_dir.mkdir(parents=True, exist_ok=True)
                tp_suffix = '_'.join(str(tp) for tp in res_dict.keys())
                tag_part = f'{tag}_' if tag else ''
                fpath = ds_dir / f'scg_memory_genes_{tag_part}{tp_suffix}.csv'
                pd.concat(scg_frames).to_csv(fpath, index_label='gene')
                print(f"   Saved {sum(len(f) for f in scg_frames)} ranked SCGs -> {fpath}")
        
        if isinstance(data_input, (str, Path)):
            del ad
            gc.collect()

    print("\nBuilding Master DataFrames...")
    all_biased_genes = set()
    for tps in dataset_bias_summary.values():
        for genes in tps.values():
            all_biased_genes.update(genes)
            
    master_df = pd.DataFrame(index=list(all_biased_genes))

    for ds_name, tps in dataset_bias_summary.items():
        tp_cols = []
        for tp, gene_set in tps.items():
            col_name = f"{ds_name}_{tp}"
            master_df[col_name] = master_df.index.isin(gene_set)
            tp_cols.append(col_name)
        if tp_cols:
            master_df[f"{ds_name}_Any"] = master_df[tp_cols].any(axis=1)

    suffix = f"_{tag}_{int(freq_threshold*100)}" if tag else f"_{int(freq_threshold*100)}"
    master_df.to_csv(out_dir / f'master_heritability_results{suffix}.csv')

    # Only meaningful when a heritable-gene list was supplied; without one this
    # would be an empty file, so it is skipped rather than written empty.
    if her_ensg_set:
        memory_df = master_df[master_df.index.isin(her_ensg_set)].copy()
        any_cols = [c for c in master_df.columns if c.endswith('_Any')]
        memory_df['n_datasets_shared'] = memory_df[any_cols].sum(axis=1)
        memory_df.to_csv(out_dir / f'memory_heritability_results{suffix}.csv')

    # --- NEW: continuous concordance frequency (pre-threshold), ALL tested genes ---
    conc_cols = []
    for ds_name, tp_series in concordance_summary.items():
        tp_cols = [s.rename(f"{ds_name}_{tp}") for tp, s in tp_series.items()]
        conc_cols.extend(tp_cols)
        if tp_cols:
            ds_block = pd.concat(tp_cols, axis=1)
            conc_cols.append(ds_block.max(axis=1).rename(f"{ds_name}_max"))   # parallels boolean _Any
    if conc_cols:
        concordance_df = pd.concat(conc_cols, axis=1)
        concordance_df.to_csv(out_dir / f'concordance_frequency{suffix}.csv')
        print(f"Saved continuous concordance scores -> concordance_frequency{suffix}.csv")

    # Without --memory_path, memory_df and any_cols were never defined and the run
    # crashed here, AFTER the SCG lists were saved. Everything below is the legacy
    # cross-dataset summary and needs the heritable-gene list.
    if not her_ensg_set:
        print(f"\nAnalysis complete. Results saved to: {out_dir}")
        return

    print("Generating UpSet Plot...")
    upset_data = from_contents({c.replace('_Any', ''): set(master_df.index[master_df[c]]) for c in any_cols})
    fig_upset = plt.figure(figsize=(10, 6))
    plot(upset_data, fig=fig_upset, show_counts=True)
    plt.suptitle("General SCG Overlap Across Datasets")
    fig_upset.savefig(plots_dir / f"upset_scg_overlap{('_'+tag) if tag else ''}.png", dpi=300)
    plt.close(fig_upset)

    num_datasets = len(datasets_to_process)
    core_genes = memory_df[memory_df['n_datasets_shared'] == num_datasets].index.tolist()
    
    if core_genes:
        print(f"Running enrichment for {len(core_genes)} core memory genes...")
        core_symbols = [core_symbol_mapping.get(g, g) for g in core_genes]
        try:
            # imported lazily: gseapy is only present in the aih twin_cell_env, and
            # the stats functions above are imported from envs that lack it
            import gseapy as gp
            enr = gp.enrichr(gene_list=core_symbols, gene_sets=['MSigDB_Hallmark_2020'], organism='Human')
            res_h = enr.results.sort_values('P-value')
            top_h = res_h[res_h['P-value'] < 0.05].head(10).copy()
            
            if not top_h.empty:
                top_h['count'] = top_h['Overlap'].apply(lambda x: int(x.split('/')[0]))
                labels = top_h['Term'].str.replace('HALLMARK_', '').str.replace('_', ' ').str.title()
                fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
                ax_pie.pie(top_h['count'], labels=labels, autopct='%1.1f%%', colors=plt.cm.Set3(np.linspace(0,1,len(labels))), startangle=140)
                ax_pie.set_title(f"Top Hallmark Pathways\n(Shared across {num_datasets} datasets)")
                plt.tight_layout()
                fig_pie.savefig(plots_dir / 'enrichment_pie_chart.png', dpi=300)
                plt.close(fig_pie)
        except Exception as e:
            print(f"Enrichment failed: {e}")

    print(f"\nAnalysis complete. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()