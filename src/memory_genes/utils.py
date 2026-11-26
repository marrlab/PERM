"""
GEMLI-style memory-gene benchmark.

Idea in one paragraph
---------------------
"Memory genes" are genes whose expression is more similar *within* a lineage
(a clone of sister cells) than you'd expect if lineage labels were random. We
score each gene by the squared coefficient of variation, CV^2 = (sd / mean)^2,
of its per-lineage expression centres. Because CV^2 naturally depends on the
mean (low-expressed genes look noisier), we detrend the CV^2-vs-mean cloud with
a loess fit and keep the residual as the score. Finally we build a NULL by
shuffling the cell -> lineage assignment many times and re-scoring: a gene is
"heritable" only if its real score beats the shuffled scores (small p-value).

Typical usage from the notebook
-------------------------------
    import scanpy as sc
    from utils import build_input_dict, memory_gene_calling

    adata = sc.read_h5ad(T_LINEAGE_PATH)
    input_dict  = build_input_dict(adata, timepoint="T1")   # T-lineage: "T1"/"T2"
    output_dict = memory_gene_calling(input_dict, n_perm=20, random_state=0)

    memory_genes = output_dict["memory_genes"]              # DataFrame: var, p
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
from skmisc.loess import loess as skloess
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Core scoring helpers
# --------------------------------------------------------------------------- #
def cv_sq(data_matrix: np.ndarray) -> np.ndarray:
    """(sd/mean)^2 per row, ignoring NaNs. data_matrix: genes x lineages."""
    sd = np.nanstd(data_matrix, axis=1, ddof=1)
    mean = np.nanmean(data_matrix, axis=1)
    # 0/0 for unexpressed genes -> NaN; filtered out downstream, exactly as in
    # GEMLI's R code. Silence the benign divide/invalid warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sd / mean) ** 2


def loess_residuals(x, y, span=0.75):
    """Fit y ~ loess(x) (degree-2 local regression) and return y - fitted(x)."""
    fit = skloess(x, y, span=span, degree=2, surface="direct")
    fit.fit()
    return y - fit.outputs.fitted_values


def markers_by_cvsq_of_lineage_means(data_matrix: pd.DataFrame,
                                     lineage_dict: pd.Series,
                                     valid_lineage_sizes=range(2, 6),
                                     use_median: bool = True) -> pd.Series:
    """
    Score genes by the detrended CV^2 of their per-lineage expression centres.

    Parameters
    ----------
    data_matrix : DataFrame, genes (rows) x cells (columns)
    lineage_dict : Series, index = cell ID, values = lineage ID
    valid_lineage_sizes : iterable of ints
        Only keep lineages whose cell count is in this set (default 2..5).
    use_median : bool
        Per-lineage centre is the median (True) or mean (False) across its cells.

    Returns
    -------
    Series indexed by gene, sorted high -> low, of loess residuals (the score).
    """
    common_cells = data_matrix.columns.intersection(lineage_dict.index)
    lineage_dict_filt = lineage_dict.loc[common_cells]

    sizes = lineage_dict_filt.value_counts()
    valid_lineages = sizes[sizes.isin(list(valid_lineage_sizes))].index
    valid_lineage_dict = lineage_dict_filt[lineage_dict_filt.isin(valid_lineages)]

    unique_lineages = pd.unique(valid_lineage_dict)
    lineage_center = pd.DataFrame(index=data_matrix.index, columns=unique_lineages, dtype=float)

    for lineage in unique_lineages:
        cells = valid_lineage_dict.index[valid_lineage_dict == lineage]
        sub = data_matrix[cells]
        lineage_center[lineage] = (sub.median(axis=1, skipna=True) if use_median
                                   else sub.mean(axis=1, skipna=True))

    x = lineage_center.mean(axis=1).values
    y = cv_sq(lineage_center.values)
    # log2(0) -> -inf for zero-mean / zero-cv^2 genes; filtered out just below.
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log2(x)
        y = np.log2(y)

    keep = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y) | (x == 0))
    genes_kept = lineage_center.index[keep]
    x_kept, y_kept = x[keep], y[keep]

    # GEMLI regresses cv^2 ON mean and keeps the cv^2 residual, in R:
    #   loess(y ~ x)  with  y = log2(cv^2) (response), x = log2(mean) (predictor)
    # loess_residuals(x, y) fits y ~ x, so pass predictor first, response second:
    residuals = loess_residuals(x_kept, y_kept)

    out = pd.Series(residuals, index=genes_kept)
    return out.sort_values(ascending=False)


def memory_gene_calling(input_dict: dict,
                        valid_lineage_sizes=range(2, 6),
                        use_median: bool = True,
                        n_perm: int = 20,
                        random_state=None) -> dict:
    """
    Call memory genes with a permutation null.

    Expects input_dict['gex'] (genes x cells DataFrame) and
    input_dict['barcodes'] (Series: cell ID -> lineage ID). Adds
    input_dict['memory_genes'], a DataFrame with columns:
        'var' : observed detrended CV^2 score
        'p'   : fraction of shuffled runs whose score >= observed (empirical p)
    sorted by p (ascending) then var (descending).
    """
    rng = np.random.default_rng(random_state)
    data_matrix = input_dict['gex']
    lineage_dict = input_dict['barcodes']

    lineage_center_variation = markers_by_cvsq_of_lineage_means(
        data_matrix, lineage_dict, valid_lineage_sizes, use_median)

    control = pd.DataFrame(index=data_matrix.index, columns=range(n_perm), dtype=float)
    for i in tqdm(range(n_perm), desc="permutation null"):
        shuffled_ids = rng.permutation(lineage_dict.index.values)
        # values stay in original order; cell-ID labels get reshuffled onto them
        lineage_dict_sampled = pd.Series(lineage_dict.values, index=shuffled_ids)
        tmp = markers_by_cvsq_of_lineage_means(
            data_matrix, lineage_dict_sampled, valid_lineage_sizes, use_median)
        control.loc[tmp.index, i] = tmp.values

    shared_genes = lineage_center_variation.index.intersection(control.index)
    pvals = (control.loc[shared_genes]
             .gt(lineage_center_variation.loc[shared_genes], axis=0)
             .sum(axis=1, skipna=True) / n_perm)

    marker_table = pd.DataFrame({
        'var': lineage_center_variation.loc[shared_genes],
        'p': pvals.loc[shared_genes]
    }).sort_values(by=['p', 'var'], ascending=[True, False])

    input_dict['memory_genes'] = marker_table
    return input_dict


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def build_input_dict(adata,
                     timepoint,
                     timepoint_col: str = "batch",
                     lineage_col: str = "clone_id") -> dict:
    """
    Assemble the {'gex', 'barcodes'} input for `memory_gene_calling`.

    The method must run on library-size-normalised but NOT log-transformed data.
    The processed AnnData stores log1p(normalised) counts in `adata.X`, so we
    undo the log with expm1 here.

    Parameters
    ----------
    adata : AnnData
        Cells x genes, with `adata.X` = log1p(library-size-normalised counts).
    timepoint : hashable
        Which timepoint to keep, matched against `adata.obs[timepoint_col]`.
        For the T-lineage dataset this is "T1" or "T2".
    timepoint_col : str
        obs column holding the timepoint. T-lineage: "batch" (LARRY: "Time point").
    lineage_col : str
        obs column holding the lineage / clone label (default "clone_id").

    Returns
    -------
    dict with:
        'gex'      : genes x cells DataFrame of de-logged normalised expression
        'barcodes' : Series (index = cell ID, value = lineage ID)
    """
    # undo log1p -> library-size-normalised (NOT log-transformed) expression.
    # Same line you had, just for any adata:
    #     adata.uns["normalized"] = np.expm1(adata.X)
    adata.uns["normalized"] = np.expm1(adata.X)

    mask = (adata.obs[timepoint_col] == timepoint).values
    normalized = adata.uns["normalized"]
    dense = normalized.toarray() if hasattr(normalized, "toarray") else np.asarray(normalized)

    gex = pd.DataFrame(dense[mask].T,
                       index=adata.var_names,
                       columns=adata.obs_names[mask])
    barcodes = adata.obs[lineage_col][mask]

    return {"gex": gex, "barcodes": barcodes}


# =========================================================================== #
# CORAL  (Lin et al., Cell Systems 2026 — R pkg YongjieLin1998/CORAL)
# =========================================================================== #
# Memory / "heritable" genes are those whose expression is more consistent
# WITHIN a clone than between clones, quantified by the ANOVA effect size
# omega^2 across clones:
#
#     omega^2 = (SSb - (k-1)*MSw) / (SSt + MSw),   MSw = SSw / (N - k)
#
# with SSt = total sum of squares per gene, SSw = within-clone SS, SSb = SSt-SSw,
# k = number of clones, N = number of cells. Negative omega^2 is clamped to 0.
# A permutation null (shuffle clone labels, recompute omega^2) sets the cutoff at
# Q3 + 1.5*IQR of the pooled null values; genes above it (and with mean > 0.05)
# are memory genes.
#
# NOTE ON INPUT: CORAL reads Seurat's "data" layer = LOG-normalised expression,
# i.e. log1p(CP10k). That is exactly adata.X here, so CORAL uses adata.X DIRECTLY
# (no expm1 — the opposite of the GEMLI path above).


def _within_clone_ssw(expr: np.ndarray, inv: np.ndarray, k: int) -> np.ndarray:
    """
    Per-gene within-clone residual sum of squares, i.e. R's
        rowSums((sc_expression - lineage_mean_expression)^2)

    expr : genes x cells array.  inv : length-N clone index per cell.  k : #clones.
    """
    N = expr.shape[1]
    onehot = np.zeros((N, k), dtype=expr.dtype)
    onehot[np.arange(N), inv] = 1.0
    counts = onehot.sum(axis=0)                    # cells per clone
    clone_means = (expr @ onehot) / counts         # genes x clones (per-clone mean)
    fitted = clone_means[:, inv]                    # genes x cells (each cell -> clone mean)
    return ((expr - fitted) ** 2).sum(axis=1)


def _omega_square(SSt: np.ndarray, SSw: np.ndarray,
                  df_b: int, df_w_safe: int) -> np.ndarray:
    """omega^2 = (SSb - df_b*MSw)/(SSt + MSw), with den==0 -> 1 and neg -> 0 (as in CORAL)."""
    SSb = SSt - SSw
    MSw = SSw / df_w_safe
    num = SSb - df_b * MSw
    den = SSt + MSw
    den = np.where(den == 0, 1.0, den)
    omega = num / den
    return np.where(omega < 0, 0.0, omega)


def coral_memory_gene_calling(input_dict: dict,
                              min_gene_mean: float = 0.05,
                              permutation_repeats: int = 5,
                              clone_size_cutoff: int = 2,
                              random_state=None) -> dict:
    """
    CORAL memory-gene calling via omega^2 ANOVA effect size + permutation cutoff.

    Expects input_dict['expr'] (genes x cells DataFrame of LOG-normalised
    expression) and input_dict['barcodes'] (Series: cell ID -> clone ID).

    Returns input_dict with three keys added:
        'omega_square' : DataFrame (genes with Mean > min_gene_mean), columns
                         Mean, SSt, SSw, SSb, Omega_square, sorted by Omega_square.
        'threshold'    : Q3 + 1.5*IQR of the permutation-null omega^2 values.
        'memory_genes' : subset of 'omega_square' with Omega_square > threshold.
    """
    rng = np.random.default_rng(random_state)
    expr_df = input_dict["expr"]
    barcodes = input_dict["barcodes"]

    # align cells present in both, then keep clones with >= clone_size_cutoff cells
    common = expr_df.columns.intersection(barcodes.index)
    expr_df = expr_df[common]
    barcodes = barcodes.loc[common]

    sizes = barcodes.value_counts()
    keep_clones = sizes.index[sizes >= clone_size_cutoff]
    keep = barcodes.isin(keep_clones).values
    expr_df = expr_df.loc[:, keep]
    barcodes = barcodes[keep]

    genes = expr_df.index
    expr = expr_df.to_numpy(dtype=np.float64)          # genes x cells (log-normalised)
    clones, inv = np.unique(barcodes.to_numpy(), return_inverse=True)   # cell -> clone index
    k = len(clones)                                    # lineage_number_actual
    N = expr.shape[1]                                   # cell_total

    df_b = k - 1                                        # k - 1
    df_w = N - k                                        # N - k
    df_w_safe = 1 if df_w == 0 else df_w

    # ---- real per-gene statistics ----
    global_mean = expr.mean(axis=1)
    SSt = ((expr - global_mean[:, None]) ** 2).sum(axis=1)
    SSw = _within_clone_ssw(expr, inv, k)
    omega = _omega_square(SSt, SSw, df_b, df_w_safe)

    SS = pd.DataFrame({"Mean": global_mean, "SSt": SSt, "SSw": SSw,
                       "SSb": SSt - SSw, "Omega_square": omega}, index=genes)

    # ---- permutation null: pooled over ALL genes x repeats (NOT mean-filtered) ----
    fake = []
    for _ in range(permutation_repeats):
        inv_fake = inv[rng.permutation(N)]             # shuffle clone labels across cells
        SSw_fake = _within_clone_ssw(expr, inv_fake, k)
        fake.append(_omega_square(SSt, SSw_fake, df_b, df_w_safe))
    omega_fake = np.concatenate(fake)

    q1, q3 = np.nanpercentile(omega_fake, [25, 75])
    threshold = float(1.5 * (q3 - q1) + q3)

    # ---- select: mean > min_gene_mean AND omega^2 > threshold (both strict) ----
    omega_table = SS[SS["Mean"] > min_gene_mean].sort_values("Omega_square", ascending=False)
    memory_genes = omega_table[omega_table["Omega_square"] > threshold]

    input_dict["omega_square"] = omega_table
    input_dict["threshold"] = threshold
    input_dict["memory_genes"] = memory_genes
    return input_dict


def build_coral_input(adata,
                      timepoint,
                      timepoint_col: str = "batch",
                      lineage_col: str = "clone_id") -> dict:
    """
    Assemble the {'expr', 'barcodes'} input for `coral_memory_gene_calling`.

    CORAL uses LOG-normalised expression (Seurat "data" layer = log1p(CP10k)),
    which is exactly `adata.X` — so, unlike the GEMLI loader, we do NOT expm1.

    Returns
    -------
    dict with:
        'expr'     : genes x cells DataFrame of LOG-normalised expression
        'barcodes' : Series (index = cell ID, value = clone ID)
    """
    mask = (adata.obs[timepoint_col] == timepoint).values
    X = adata.X[mask]
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    expr = pd.DataFrame(dense.T,
                        index=adata.var_names,
                        columns=adata.obs_names[mask])
    barcodes = adata.obs[lineage_col][mask]

    return {"expr": expr, "barcodes": barcodes}


# =========================================================================== #
# Mold et al.  (Cell Systems 2024 — github MartyWeissman/ClonalOmics)
# =========================================================================== #
# "Clonal" (heritable / memory) genes via a per-gene one-way ANOVA F of
# LOG-normalised expression grouped by clone. This mirrors their ANOVA(gf, grouper)
# verbatim: intraclonal SS = sum_c var(ddof=1)_c * (n_c - 1) = SSw;
# interclonal SS = sum_c n_c * (clone_mean - grand_mean)^2 = SSb; then
#
#     F = (SSb / SSw) * (N - k) / (k - 1),   p = P(F_{k-1, N-k} > F)
#
# Genes are ranked by p; the top_n (their value: 500) are the clonal genes. One
# clone-label shuffle gives the "decoy" set. EXACTLY as in their code:
#   - NO clone-size filter (all clones, incl. singletons, enter the ANOVA);
#   - gene filter + normalisation live in build_mold_input: TPM-normalise to 1e6,
#     keep genes with TPM > 10 in > 5% of cells, then the ANOVA input is log1p(TPM).
# Output columns match theirs: 'total variance', 'interclonal variance',
# 'intraclonal variance', 'ANOVA F', 'p'.


def _anova_table(expr: np.ndarray, inv: np.ndarray, k: int, N: int, genes,
                 tie_break: bool = False) -> pd.DataFrame:
    """One-way ANOVA per gene, matching Mold's ANOVA(gf, grouper) exactly (k clones, N cells).

    tie_break=False (DEFAULT, the authors' code): sort by p alone, exactly as
    ClonalOmics' `ANOVA_df.sort_values('p')`. Keep this default -- it is what
    original_implementations/Mold/validation/verify_mold.py compares against,
    and that comparison currently passes with identical set AND identical order
    on all 12 groups.

    tie_break=True: sort by (p asc, ANOVA F desc). This is NOT a different
    ranking -- at fixed k and N, p = f.sf(F, k-1, N-k) is strictly decreasing
    in F, so p-ascending IS F-descending. It only matters where p UNDERFLOWS
    to exactly 0.0 (float64 bottoms out near 5e-324) and the p column loses the
    ability to order genes whose F still differs a lot. Melanoma_Harmange is the
    dataset where that bites: ~17.8 cells/clone (450 clones, 8000 cells) makes
    the df multiplier (N-k)/(k-1) = 16.8 against 1.4-1.8 elsewhere, so F is ~10x
    larger and ALL 500 saved genes underflow in the 'all' grouping.
    """
    import scipy.stats as ss

    onehot = np.zeros((N, k), dtype=expr.dtype)
    onehot[np.arange(N), inv] = 1.0
    counts = onehot.sum(axis=0)                        # cells per clone (n_c)
    grand_mean = expr.mean(axis=1)                     # per-gene grand mean
    clone_means = (expr @ onehot) / counts             # genes x clones
    fitted = clone_means[:, inv]                        # genes x cells (each cell -> its clone mean)

    total_ss = ((expr - grand_mean[:, None]) ** 2).sum(axis=1)                   # 'total variance'
    intra_ss = ((expr - fitted) ** 2).sum(axis=1)                               # 'intraclonal variance' (SSw)
    inter_ss = ((clone_means - grand_mean[:, None]) ** 2 * counts).sum(axis=1)  # 'interclonal variance' (SSb)

    with np.errstate(divide="ignore", invalid="ignore"):
        F = (inter_ss / intra_ss) * (N - k) / (k - 1)
    p = ss.f.sf(F, k - 1, N - k)

    out = pd.DataFrame({
        "total variance": total_ss,
        "interclonal variance": inter_ss,
        "intraclonal variance": intra_ss,
        "ANOVA F": F,
        "p": p,
    }, index=genes)
    if tie_break:                                   # p asc, then F desc (same ranking,
        return out.sort_values(["p", "ANOVA F"],     # resolved where p underflowed to 0)
                               ascending=[True, False], kind="stable")
    return out.sort_values("p")                     # the authors' line, verbatim


def mold_memory_gene_calling(input_dict: dict,
                             top_n: int = 500,
                             compute_decoy: bool = True,
                             random_state=None,
                             tie_break: bool = False) -> dict:
    """
    Mold et al. clonal ("memory") gene calling via per-gene one-way ANOVA F, exactly
    as in their ClonalOmics code: no clone-size filter, no gene filter here.

    Expects input_dict['expr'] (genes x cells, LOG-normalised) and
    input_dict['barcodes'] (cell -> clone). Returns input_dict with:
        'anova'        : all genes, columns 'total variance', 'interclonal variance',
                         'intraclonal variance', 'ANOVA F', 'p', sorted by p.
        'memory_genes' : top_n genes by p (their clonal genes).
        'decoy_genes'  : top_n genes from one clone-label shuffle (if compute_decoy).
    tie_break : bool
        Passed to _anova_table; see there. False = the authors' code verbatim.
        The benchmark runner passes True so that top_n is not an arbitrary cut
        through a block of underflowed (p == 0.0) genes.
    """
    rng = np.random.default_rng(random_state)
    expr_df = input_dict["expr"]
    barcodes = input_dict["barcodes"]

    common = expr_df.columns.intersection(barcodes.index)
    expr_df = expr_df[common]
    barcodes = barcodes.loc[common]

    genes = expr_df.index
    expr = expr_df.to_numpy(dtype=np.float64)          # genes x cells (log-normalised)
    clones, inv = np.unique(barcodes.to_numpy(), return_inverse=True)
    k = len(clones)                                    # ALL clones, incl. singletons
    N = expr.shape[1]

    input_dict["anova"] = _anova_table(expr, inv, k, N, genes, tie_break=tie_break)
    input_dict["memory_genes"] = input_dict["anova"].head(top_n)

    if compute_decoy:
        inv_shuff = inv[rng.permutation(N)]            # one clone-label shuffle
        input_dict["decoy_genes"] = _anova_table(expr, inv_shuff, k, N, genes,
                                                 tie_break=tie_break).head(top_n)

    return input_dict


def build_mold_input(adata,
                     timepoint,
                     timepoint_col: str = "batch",
                     lineage_col: str = "clone_id",
                     target_sum: float = 1e6,
                     min_tpm: float = 10,
                     min_frac: float = 0.05) -> dict:
    """
    Mold et al. input {'expr', 'barcodes'}, reproducing their preprocessing exactly:
      1. TPM-normalise each cell to target_sum=1e6. adata.X is log1p(CP10k), so
         expm1 recovers CP10k; we re-normalise per cell to target_sum -> TPM.
      2. Gene filter (on LINEAR TPM): keep genes with TPM > min_tpm in > min_frac
         of cells, i.e. (tpm > 10).sum(cells) > 0.05 * n_cells.
      3. log1p -> the ANOVA runs on log1p(TPM).
    """
    mask = (adata.obs[timepoint_col] == timepoint).values
    sub = adata[mask]
    dense = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)

    # undo log1p, then TPM_norm(target_sum) per cell -> linear TPM (cells x genes)
    lin = np.expm1(dense)
    tpm = lin / lin.sum(axis=1, keepdims=True) * target_sum

    # Mold gene filter on linear TPM: TPM > min_tpm in > min_frac of cells
    keep = (tpm > min_tpm).sum(axis=0) > (min_frac * tpm.shape[0])
    tpm = tpm[:, keep]
    genes = sub.var_names[keep]

    expr = pd.DataFrame(np.log1p(tpm).T, index=genes, columns=sub.obs_names)   # log1p(TPM), genes x cells
    barcodes = sub.obs[lineage_col]
    return {"expr": expr, "barcodes": barcodes}


# =========================================================================== #
# Power-Seek  (Ghosh, Chakrabarti & Raju, bioRxiv 2025.01.15.633183 — official
#              code RajuLab/CellMem-Power-Seek, PowerSeekAlgorithmOnData.ipynb)
# =========================================================================== #
# LINEAGE-FREE. z-score each gene, build the cell covariance X X^T / ng, and for
# each gene measure how much removing it shrinks the top-n eigenvalues:
#     delta_g = sum_{j=1..n} (lambda_j - lambda_j^{-g}).
# n = #eigenvalues in the power-law regime, chosen as the FIRST PEAK of the
# R^2-vs-count curve (log-log fit of the top eigenvalues). Memory genes = the
# top_n genes ranked by delta_g. Reproduces their code exactly (not the paper prose).


def _powerseek_eigs(X_E: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of X X^T / ng, using the smaller dimension for speed
    (nonzero eigenvalues of X X^T and X^T X are equal). X_E: cells x genes."""
    nc, ng = X_E.shape
    cv = (X_E @ X_E.T) / ng if nc <= ng else (X_E.T @ X_E) / ng
    return np.linalg.eigvalsh(cv)[::-1]


def _powerseek_threshold(ev: np.ndarray, ng: int, nc: int,
                         initial_pt: int = 20, step: int = 2) -> int:
    """n = #eigenvalues in the power-law regime = FIRST PEAK of the R^2-vs-count
    curve (log-log linear fit of the top-i eigenvalues), as in the Power-Seek code."""
    from scipy import stats
    ranks = np.arange(1, len(ev) + 1)
    r2, counts = [], []
    for i in range(initial_pt, min(ng, nc), step):
        res = stats.linregress(np.log(ranks[:i]), np.log(ev[:i]))
        r2.append(res.rvalue ** 2)
        counts.append(i)
    slope = np.diff(np.asarray(r2))
    sign_change = np.where((slope[:-1] > 0) & (slope[1:] < 0))[0] + 1   # first + -> - turn
    return counts[sign_change[0]]


def powerseek_memory_gene_calling(input_dict: dict,
                                  top_n: int = 500,
                                  initial_pt: int = 20,
                                  step: int = 2) -> dict:
    """
    Power-Seek memory-gene calling. Expects input_dict['expr'] = genes x cells
    DataFrame from build_powerseek_input. Adds:
        'delta'         : Series gene -> delta_g, sorted descending.
        'n_eigenvalues' : power-law regime size (first-peak-of-R^2).
        'memory_genes'  : top_n genes by delta_g (their ranking).
    """
    expr_df = input_dict["expr"]
    genes = expr_df.index.to_numpy()
    z = expr_df.to_numpy(dtype=np.float64).T               # cells x genes (float64, as their CSV read)
    with np.errstate(divide="ignore", invalid="ignore"):
        X_E = np.float32((z - z.mean(axis=0)) / z.std(axis=0))   # z-score per gene (ddof=0) -> float32
    nc, ng = X_E.shape

    ev = _powerseek_eigs(X_E)
    n = _powerseek_threshold(ev, ng, nc, initial_pt, step)

    delta = np.zeros(ng)
    for i in tqdm(range(ng), desc="Power-Seek (leave-one-gene-out)"):
        X1 = np.delete(X_E, i, axis=1)                      # remove gene i
        cv1 = (X1 @ X1.T) / (ng - 1) if nc <= ng else (X1.T @ X1) / (ng - 1)
        ev1 = np.linalg.eigvalsh(cv1)[::-1]
        delta[i] = np.sum(ev[:n] - ev1[:n])                 # sum of top-n eigenvalue changes

    order = delta.argsort()[::-1]                           # descending delta, exactly as in the source
    ranked = genes[order]
    input_dict["delta"] = pd.Series(delta[order], index=ranked)
    input_dict["n_eigenvalues"] = int(n)
    input_dict["memory_genes"] = pd.DataFrame({"delta": delta[order][:top_n]}, index=ranked[:top_n])
    return input_dict


def _powerseek_outlier_count(gene_expr: np.ndarray, lower: int = 9, upper: int = 91) -> int:
    """#outliers in a gene's nonzero log-expression by the 1.5*IQR whisker rule on the
    (lower, upper) percentiles -- Power-Seek's Outlier_detection."""
    nz = gene_expr[gene_expr > 0]
    if len(nz) == 0:
        return 0
    logv = np.log(nz)
    q1, q5 = np.percentile(logv, lower), np.percentile(logv, upper)
    iqr = q5 - q1
    lw, uw = q1 - 1.5 * iqr, q5 + 1.5 * iqr
    return int(((logv < lw) | (logv > uw)).sum())


def build_powerseek_input(adata,
                          timepoint,
                          timepoint_col: str = "batch",
                          min_cells: int = 200,
                          n_top_cv: int = 3000) -> dict:
    """
    Reproduce Power-Seek's data prep (PowerSeekDataPrepV2):
      1. total-count normalise each cell (counts / per-cell total). adata.X is
         log1p(CP10k); expm1 -> CP10k, then divide by the per-cell sum -> the same
         fractions their code makes from raw counts (this h5ad has no raw counts).
      2. keep genes with > min_cells nonzero cells.
      3. drop genes with any log-expression outlier (1.5*IQR on the 9th/91st pct).
      4. keep the top n_top_cv genes by coefficient of variation.
    n_top_cv=3000 matches their committed HighVariable3000 matrix (which generated
    the repo's memory-gene lists); their DataPrepV2 code line writes 5000.
    Returns {'expr': genes x cells DataFrame} for powerseek_memory_gene_calling.
    """
    mask = (adata.obs[timepoint_col] == timepoint).values
    sub = adata[mask]
    dense = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    df = pd.DataFrame(np.expm1(dense).T, index=sub.var_names, columns=sub.obs_names)  # genes x cells

    df = df[(df != 0).sum(axis=1) > 0]                              # drop all-zero genes
    df = df.divide(df.sum(axis=0).astype(np.float64), axis=1).astype("float32")   # per-cell total-count normalise
    df = df[(df != 0).sum(axis=1) > min_cells]                     # genes in > min_cells cells
    keep = df.apply(lambda r: _powerseek_outlier_count(r.to_numpy()) == 0, axis=1)
    df = df[keep]                                                  # drop outlier genes
    cv = df.std(axis=1) / df.mean(axis=1)                          # coefficient of variation
    df = df.iloc[np.argsort(-cv.to_numpy())].head(n_top_cv)        # top-CV genes

    return {"expr": df}


# =========================================================================== #
# Output layout
# =========================================================================== #
def dataset_save_dir(adata_path: str, base_dir: str = None) -> str:
    """
    Per-dataset output directory for predicted memory genes, named after the
    dataset file. E.g. ".../datasets/T-lineage_processed.h5ad" -> "<base_dir>/T-lineage".
    Creates the directory if needed and returns its path.

    base_dir defaults to "python_implements" next to this utils.py.

    NOTE, 2026-08-19: this used to default to "data". The tree was renamed when the
    paper moved to the authors' original implementations:

        data/                -> results from the AUTHORS' ORIGINAL code (the paper's)
        python_implements/   -> results from our Python ports (the old data/)

    Downstream consumers keep reading "data/" and therefore now read the original
    implementations without any path edits -- that was the point of the rename. The
    Python writers were repointed here so they cannot contaminate data/ with a mix
    of the two. NEVER let both live in one directory.
    """
    dataset = os.path.basename(adata_path)
    for suffix in (".h5ad", "_processed", "_singlets"):
        dataset = dataset.replace(suffix, "")
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_implements")
    out = os.path.join(base_dir, dataset)
    os.makedirs(out, exist_ok=True)
    return out


# =========================================================================== #
# Generic dataset loading  (dataset-agnostic entry point for every method)
# =========================================================================== #
# Every build_*_input above shares the same (adata, timepoint, timepoint_col,
# lineage_col) signature, but each dataset spells things differently:
#
#   dataset      group axis            lineage col   gene symbols
#   T-lineage    batch (T1/T2)         clone_id      var.index
#   LARRY        batch (2.0/4.0/6.0)   clone_id      var.index
#   Mold brain   brain (brain1/2)      cloneID       var['var_names'] (index is 0,1,2,..)
#
# load_lineage_dataset() reconciles those so a notebook works on any of them:
#
#     adata, groups, group_col = load_lineage_dataset(PATH)
#     for g in groups:
#         inp = build_coral_input(adata, timepoint=g, timepoint_col=group_col)
#
# On the group axis: for T-lineage/LARRY the groups are TIMEPOINTS, for the Mold
# brain they are ANIMALS (age is a single value, P15). Either way each group is
# scored independently, which is also what you want for the Mold brain -- clones
# never span brains, so pooling would let between-clone variance absorb the
# between-brain batch effect (see scg_analysis/run_mold.sh for how badly that
# misfires on the SCG test).

GROUP_COL_CANDIDATES = ("batch", "timepoint", "Time point", "brain")
LINEAGE_COL_CANDIDATES = ("clone_id", "cloneID", "clone", "lineage_id", "lineage")
SYMBOL_COL_CANDIDATES = ("var_names", "gene_symbol", "human_symbol")


def resolve_group_col(adata, group_col: str = None,
                      candidates=GROUP_COL_CANDIDATES) -> str:
    """
    Pick the obs column to split the analysis on (timepoint, brain, ...).

    Preference order: an explicit `group_col`, then the first of `candidates`
    present in obs with >= 2 distinct values, then the first present at all
    (single group == pooled). If none exist, a constant '_group' column is added
    so callers always have something to loop over.
    """
    if group_col is not None:
        if group_col not in adata.obs.columns:
            raise KeyError(f"group_col {group_col!r} not in obs; "
                           f"available: {list(adata.obs.columns)}")
        return group_col

    present = [c for c in candidates if c in adata.obs.columns]
    informative = [c for c in present if adata.obs[c].nunique(dropna=False) >= 2]
    if informative:
        return informative[0]
    if present:
        return present[0]

    adata.obs["_group"] = "all"
    return "_group"


def resolve_lineage_col(adata, lineage_col: str = None,
                        candidates=LINEAGE_COL_CANDIDATES) -> str:
    """Find the obs column holding the clone / lineage label."""
    if lineage_col is not None:
        if lineage_col not in adata.obs.columns:
            raise KeyError(f"lineage_col {lineage_col!r} not in obs; "
                           f"available: {list(adata.obs.columns)}")
        return lineage_col
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"no lineage column found; tried {list(candidates)}. "
                   f"Pass lineage_col= explicitly.")


# Labels that mean "this cell has NO clone", not "this cell belongs to clone X".
# LARRY_processed.h5ad uses -1 for cells where no barcode was detected: 48,038 of
# 72,946 cells (65.9%). Left in, every method treats them as ONE clone of ~17-23k
# cells per timepoint and the result is meaningless -- CORAL's t=2.0 run was 90%
# this one fake clone. Found 2026-08-20; all prior LARRY results are void.
#
# "0" is deliberately NOT here: T-lineage has a genuine 2-cell clone numbered 0.
# A sentinel is dataset convention, not a universal, so keep this list explicit and
# short rather than guessing at anything that "looks unassigned".
UNASSIGNED_CLONE_LABELS = frozenset({"-1", "nan", "NaN", "None", "NA", "", "unassigned"})


def standardize_lineage_adata(adata, group_col: str = None, lineage_col: str = None,
                              symbol_col: str = None, copy: bool = True, verbose: bool = True,
                              drop_unassigned: bool = True):
    """
    Normalise a lineage-tracing AnnData so the build_*_input functions can read it.

    Ensures: real gene symbols on var.index, a `clone_id` obs column, a
    `clone_size` obs column, and a resolved group column.

    Returns
    -------
    (adata, groups, group_col) -- `groups` is the list of values to loop over.
    """
    if copy:
        adata = adata.copy()

    # -- gene symbols -------------------------------------------------------
    # Some datasets (Mold brain) carry a positional var.index ('0','1',...) with
    # the real symbols parked in a var column.
    idx = adata.var.index.astype(str)
    if symbol_col is None:
        symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in adata.var.columns), None)
    if symbol_col is not None and idx.str.fullmatch(r"\d+").all():
        adata.var.index = pd.Index(adata.var[symbol_col].astype(str), name="gene")
        if verbose:
            print(f"  var.index was positional -> using var[{symbol_col!r}] as gene symbols")
    adata.var_names_make_unique()

    # -- clone id / size ----------------------------------------------------
    src = resolve_lineage_col(adata, lineage_col)
    if src != "clone_id":
        adata.obs["clone_id"] = adata.obs[src].astype(str)
        if verbose:
            print(f"  lineage column {src!r} -> copied to 'clone_id'")
    else:
        adata.obs["clone_id"] = adata.obs["clone_id"].astype(str)

    # Drop unassigned cells BEFORE clone_size is computed, or clone_size keeps
    # counting the unassigned pool as though it were a clone (which is exactly how
    # the LARRY h5ad arrived: its clone_size column stamps every -1 cell with the
    # size of the whole pool).
    if drop_unassigned:
        unassigned = adata.obs["clone_id"].isin(UNASSIGNED_CLONE_LABELS).to_numpy()
        n = int(unassigned.sum())
        if n:
            labels = sorted(set(adata.obs["clone_id"][unassigned]))
            adata = adata[~unassigned].copy()
            if verbose:
                print(f"  DROPPED {n} cells with no clone assignment "
                      f"(labels {labels}) -- {100*n/(n+adata.n_obs):.1f}% of the dataset")

    # -- group axis ---------------------------------------------------------
    group_col = resolve_group_col(adata, group_col)

    # clone_size is recomputed rather than trusted -- the incoming column can have
    # been built on a different cell set (LARRY's counted unassigned cells as a
    # clone). It MUST be counted WITHIN the group, not pooled across groups:
    # verified 2026-08-20 that LARRY's own column is per-(clone,timepoint) at
    # 100% and pooled at only 16%, and that 66% of its clones span >1 timepoint.
    # An earlier version of this fix pooled them, which silently redefined the
    # column -- findingSCGs.py reads clone_size for its [2,2] sister window, so a
    # pooled count would have changed which cells SCG calls sisters.
    adata.obs["clone_size"] = adata.obs.groupby(
        [group_col, "clone_id"], observed=True)["clone_id"].transform("size")
    groups = adata.obs[group_col].unique().tolist()
    if verbose:
        print(f"  grouping by {group_col!r} -> {groups}")
        for g in groups:
            o = adata.obs[adata.obs[group_col] == g]
            sizes = o["clone_id"].value_counts()
            print(f"    {g}: {len(o)} cells, {len(sizes)} clones, "
                  f"{(sizes >= 2).sum()} with >=2 cells")

    return adata, groups, group_col


def load_lineage_dataset(path: str, group_col: str = None, lineage_col: str = None,
                         symbol_col: str = None, verbose: bool = True):
    """
    Read an h5ad and standardize it in one call.

        adata, groups, group_col = load_lineage_dataset(MOLD_BRAIN_PATH)

    See standardize_lineage_adata for what gets normalised.
    """
    if verbose:
        print(f"Loading {os.path.basename(path)}")
    adata = sc.read_h5ad(path)
    return standardize_lineage_adata(adata, group_col=group_col, lineage_col=lineage_col,
                                     symbol_col=symbol_col, copy=False, verbose=verbose)
