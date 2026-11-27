"""
marson_KO.py - knockout analysis on the Marson perturb-seq dataset.

Shared perturbation-response code: pseudobulks and log fold changes, per-gene
response summaries, memory-gene set loading and consensus, expression-matched
comparison, GMT enrichment, and effect-size helpers.

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


# ---------------------------------------------------------------------------
# GMT gene-set enrichment
# ---------------------------------------------------------------------------

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


def compare_gene_responsiveness_expr_matched(
    gene_summary, gene_expr, scg_genes, metric='mean_abs_lfc',
    n_bins=10, n_iter=20, alternative='two-sided', random_seed=0,
):
    """
    Memory genes vs non-memory genes, with the non-memory group
    EXPRESSION-MATCHED to the memory genes and averaged over several random draws.

    For each of n_iter iterations: within each of n_bins baseline-expression bins,
    sample as many non-memory genes as there are memory genes in that bin -> the
    sampled background has the same expression distribution as the memory set. The
    returned 'non_scg_values' pool all iterations (a stable box), and 'mann_whitney_p'
    is the MEDIAN Mann-Whitney p across iterations (each computed on matched n).

    Returns a results dict
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


# ---------------------------------------------------------------------------
# Cross-dataset memory-gene responsiveness: compare each dataset's memory genes
# vs its 0-method expression-matched non-memory background, across all
# perturb-seq datasets (Marson/Weinstock/brain/melanoma) on one figure.
# ---------------------------------------------------------------------------
METHOD_ORDER = ['CORAL', 'GEMLI', 'MOLD', 'POWERSEEK', 'SCG']


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

