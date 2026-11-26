"""
shi_KO.py - raw pseudobulk + CPM-normalised LFC for the whole-brain CRISPR atlas
(perturbai/wholebrain_crispr_atlas), the mold-brain perturb-seq knockout dataset.

Why this exists
---------------
The previous mold pseudobulk (/scratch/.../pseudobulk.npz) summed *per-cell
normalised* counts, so it can't be re-normalised (the depth is already gone and
the 0.01 pseudocount sat on the normalised scale). This rebuilds the pseudobulk
from RAW counts, then applies CPM + log2 LFC exactly like marson_KO, so the two
datasets are directly comparable.

Pipeline (mirrors marson_KO's split of the expensive vs cheap step):
  1. build_shi_pseudobulk(shards, target_genes)  -> RAW sums+counts per group
       (the only step that touches the 14 shards; cache it once with save_shi_pseudobulk)
  2. lfc_from_shi_pseudobulk(pb, ...)             -> per-cell-type CPM LFC vs Non_target
  3. summarize_gene_response(lfc_df)               -> reuse marson_KO's summariser

Grouping: one pseudobulk row per (gene_target x predicted_group) pair, over
single-guide (num_guides == 1) cells that pass QC, keeping pairs with >= min_cells.
LFC is computed within cell type: each (KO, cell type) vs Non_target of the SAME
cell type. Self-KO and Safe_target rows are handled downstream.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

SHARD_NAMES = [f"WB8588_{b}_{n}" for b, n in
               [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
                (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)]]
HF_REPO = "perturbai/wholebrain_crispr_atlas"


def download_shards(shard_names=SHARD_NAMES, dest_dir=None, repo=HF_REPO):
    """
    Download the shard h5ads to a persistent dir (run on a login node with
    internet). Returns the list of local paths. Skips files already present.
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for name in shard_names:
        local = os.path.join(dest_dir, f"{name}.h5ad")
        if os.path.exists(local):
            print(f"  have {name}", flush=True)
        else:
            print(f"  downloading {name} ...", flush=True)
            hf_hub_download(repo, f"h5ads/{name}.h5ad", repo_type="dataset",
                            local_dir=dest_dir)
            # hf_hub_download nests under h5ads/; normalise to a flat path
            nested = os.path.join(dest_dir, "h5ads", f"{name}.h5ad")
            if os.path.exists(nested) and not os.path.exists(local):
                local = nested
        paths.append(local)
    return paths


def build_shi_pseudobulk(
    shard_paths,
    target_genes,
    control_label='Non_target',
    gene_target_col='gene_target',
    celltype_col='predicted_group',
    num_guides_col='num_guides',
    qc_col='passes_qc',
    min_cells=50,
    chunk_size=50_000,
    verbose=True,
):
    """
    Sum RAW counts per (gene_target x predicted_group) group across all shards.

    target_genes : the exact gene order to emit (e.g. the old npz's 'genes', so
        the rebuild is gene-for-gene comparable). Genes absent from a shard are
        summed as 0 for that shard.

    Two passes: (1) obs-only, to fix the kept group set (>= min_cells over the
    filtered cells); (2) chunked sparse sum of raw X into those groups. Memory is
    one shard-chunk at a time plus a (n_groups x n_genes) float64 accumulator.

    Returns
    -------
    pb : DataFrame [groups x genes] mean RAW counts per cell (index = MultiIndex
         (gene_target, predicted_group)).
    counts : Series [groups] cells per group.
    """
    import anndata as ad

    target_genes = pd.Index(pd.unique(np.asarray(target_genes, dtype=str)))
    n_genes = len(target_genes)
    tgt_pos = {g: i for i, g in enumerate(target_genes)}

    def _cell_frame(obs):
        keep = (obs[num_guides_col].values == 1) & obs[qc_col].values.astype(bool)
        ko = obs[gene_target_col].astype(str).values
        ct = obs[celltype_col].astype(str).values
        return keep, ko, ct

    # ---- pass 1: enumerate groups over filtered cells ----
    if verbose:
        print(f"Pass 1/2: counting cells per group over {len(shard_paths)} shards...", flush=True)
    group_cells = {}
    for p in shard_paths:
        a = ad.read_h5ad(p, backed='r')
        keep, ko, ct = _cell_frame(a.obs)
        for k, c in zip(ko[keep], ct[keep]):
            group_cells[(k, c)] = group_cells.get((k, c), 0) + 1
        if verbose:
            print(f"  {os.path.basename(p)}: {int(keep.sum()):,} kept cells", flush=True)
        a.file.close()

    kept = [g for g, n in group_cells.items() if n >= min_cells]
    kept.sort()
    group_index = {g: i for i, g in enumerate(kept)}
    n_groups = len(kept)
    if verbose:
        print(f"Groups with >= {min_cells} cells: {n_groups:,} "
              f"(of {len(group_cells):,} seen).", flush=True)

    # ---- pass 2: sum raw X into the kept groups ----
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    if verbose:
        print("Pass 2/2: summing raw counts...", flush=True)

    for p in shard_paths:
        a = ad.read_h5ad(p, backed='r')
        keep, ko, ct = _cell_frame(a.obs)
        # cell -> global group row (-1 if its group was dropped or cell filtered)
        row = np.full(a.n_obs, -1, dtype=np.int64)
        for i in np.where(keep)[0]:
            row[i] = group_index.get((ko[i], ct[i]), -1)
        # gene reindex: shard column -> target position (skip genes not in target)
        shard_genes = a.var_names.astype(str).values
        src_cols, dst_cols = [], []
        for j, g in enumerate(shard_genes):
            t = tgt_pos.get(g)
            if t is not None:
                src_cols.append(j); dst_cols.append(t)
        src_cols = np.asarray(src_cols); dst_cols = np.asarray(dst_cols)

        n_cells = a.n_obs
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            r = row[start:end]
            local = np.where(r >= 0)[0]
            if local.size == 0:
                continue
            Xc = a.X[start:end]
            if sp.issparse(Xc):
                Xc = Xc.tocsr()
            Xc = Xc[local][:, src_cols]              # (n_used_cells x n_shared_genes)
            grp = r[local]
            ind = sp.csr_matrix(
                (np.ones(local.size, dtype=np.float64),
                 (grp, np.arange(local.size))),
                shape=(n_groups, local.size))
            contrib = ind @ Xc
            contrib = contrib.toarray() if sp.issparse(contrib) else np.asarray(contrib)
            sums[:, dst_cols] += contrib
            np.add.at(counts, grp, 1)
        if verbose:
            print(f"  summed {os.path.basename(p)} "
                  f"(running total {int(counts.sum()):,} cells)", flush=True)
        a.file.close()

    idx = pd.MultiIndex.from_tuples(kept, names=['gene_target', 'predicted_group'])
    pb = pd.DataFrame(sums / np.maximum(counts, 1)[:, None], index=idx, columns=target_genes)
    counts = pd.Series(counts, index=idx, name='n_cells')
    if verbose:
        print(f"Raw pseudobulk: {pb.shape[0]} groups x {pb.shape[1]} genes.", flush=True)
    return pb, counts


def aggregate_lfc_to_per_ko(lfc_df, counts=None, weight='cells'):
    """
    Collapse a (KO x cell type) LFC matrix to one row per KO, so the output is
    ~n_KO x genes -- parallel to marson's per-KO LFC.

    The within-cell-type LFC (which controls cell-type composition) is preserved;
    this only averages it across cell types. NaN entries (masked self-KO) are
    skipped per gene, and the weights are renormalised over the non-NaN cell types.

    weight : 'cells' -> weight each cell type by its cell count. This matches
                        marson, which pools all of a KO's cells so cell types
                        contribute in proportion to their size.
             'equal' -> each cell type contributes equally.
    """
    kos = lfc_df.index.get_level_values(0).values.astype(str)
    uniq = pd.unique(kos)
    ko_pos = {k: i for i, k in enumerate(uniq)}
    row_ko = np.array([ko_pos[k] for k in kos])

    if weight == 'cells':
        if counts is None:
            raise ValueError("weight='cells' needs the per-group cell counts.")
        w = counts.reindex(lfc_df.index).values.astype(np.float32)
    elif weight == 'equal':
        w = np.ones(len(kos), dtype=np.float32)
    else:
        raise ValueError("weight must be 'cells' or 'equal'.")

    X = lfc_df.values.astype(np.float32)
    mask = ~np.isnan(X)
    wcol = w[:, None]
    G = sp.csr_matrix((np.ones(len(kos), dtype=np.float32),
                       (row_ko, np.arange(len(kos)))), shape=(len(uniq), len(kos)))
    num = G @ (np.where(mask, X, np.float32(0.0)) * wcol)          # weighted sum over cell types
    den = G @ (mask.astype(np.float32) * wcol)                     # sum of weights actually used
    with np.errstate(invalid='ignore', divide='ignore'):
        per_ko = np.asarray(num) / np.asarray(den)
    per_ko[np.asarray(den) == 0] = np.nan
    out = pd.DataFrame(per_ko, index=pd.Index(uniq, name='gene_target'),
                       columns=lfc_df.columns)
    print(f"Aggregated to per-KO: {out.shape[0]} KOs x {out.shape[1]} genes (weight={weight}).")
    return out


def save_shi_pseudobulk(pb, counts, path):
    """Cache the raw pseudobulk (mean counts per group) + cell counts to .npz."""
    np.savez_compressed(
        path,
        values=pb.values.astype(np.float32),
        kos=pb.index.get_level_values(0).values.astype(str),
        cts=pb.index.get_level_values(1).values.astype(str),
        genes=pb.columns.values.astype(str),
        counts=counts.values.astype(np.int64),
    )
    print(f"Saved raw mold pseudobulk to {path} "
          f"({pb.shape[0]} groups x {pb.shape[1]} genes).")


def load_shi_pseudobulk(path):
    """Reload a raw pseudobulk cached by save_shi_pseudobulk. Returns (pb, counts)."""
    z = np.load(path, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays([z['kos'], z['cts']],
                                    names=['gene_target', 'predicted_group'])
    pb = pd.DataFrame(z['values'], index=idx, columns=z['genes'])
    counts = pd.Series(z['counts'], index=idx, name='n_cells')
    return pb, counts


def lfc_from_shi_pseudobulk(
    pb,
    counts=None,
    control_label='Non_target',
    min_cells=50,
    normalize='group',
    cpm_target=1e6,
    pseudocount=1.0,
    mask_self_ko=True,
    safe_prefix='Safe_target',
):
    """
    Per-(KO, cell type) CPM LFC vs the Non_target control of the SAME cell type.

    Mirrors marson_KO.lfc_from_pseudobulk, but the control is matched within
    predicted_group instead of being a single global control. Rows without a
    same-cell-type control, and control rows themselves, are dropped.

    normalize : 'group' -> scale each group's summed profile to CPM before the
        log2 ratio (removes per-group depth); None -> raw means (reproduces the
        old behaviour). On CPM a pseudocount ~1 (= 1 count per million) is a real
        detection floor.

    Returns
    -------
    lfc_df : DataFrame [(KO, cell type) x genes] of log2 fold-change.
             'is_safe' is available via index level 0 startswith(safe_prefix).
    """
    genes = pb.columns.astype(str)
    mat = pb.values.astype(np.float32)   # float32: halves peak memory on the big matrix
    kos = pb.index.get_level_values(0).values.astype(str)
    cts = pb.index.get_level_values(1).values.astype(str)

    if normalize == 'group':
        lib = mat.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        mat = mat / lib * cpm_target
        print(f"normalize='group': per-group CPM (target {cpm_target:g}); "
              f"max = {mat.max():.1f}; pseudocount = {pseudocount:g}.")
    elif normalize not in (None, False):
        raise ValueError(f"normalize must be 'group' or None, got {normalize!r}")

    # control row per cell type
    ctrl_row = {ct: i for i, (k, ct) in enumerate(zip(kos, cts)) if k == control_label}
    if not ctrl_row:
        raise ValueError(f"No control rows with gene_target == '{control_label}'.")

    keep = np.array([
        i for i in range(len(kos))
        if kos[i] != control_label and cts[i] in ctrl_row
        and (counts is None or counts.iloc[i] >= min_cells)
    ])
    ref = np.array([ctrl_row[cts[i]] for i in keep])

    lfc = np.log2((mat[keep] + pseudocount) / (mat[ref] + pseudocount))
    idx = pd.MultiIndex.from_arrays([kos[keep], cts[keep]],
                                    names=['gene_target', 'predicted_group'])
    lfc_df = pd.DataFrame(lfc, index=idx, columns=genes)

    if mask_self_ko:
        gene_pos = {g: j for j, g in enumerate(genes)}
        n = 0
        for r, ko in enumerate(lfc_df.index.get_level_values(0)):
            j = gene_pos.get(ko)
            if j is not None:
                lfc_df.values[r, j] = np.nan; n += 1
        print(f"Masked self-KO LFC for {n} (KO, cell type) rows.")

    n_safe = int(pd.Series(lfc_df.index.get_level_values(0)).str.startswith(safe_prefix).sum())
    print(f"Returned LFC: {lfc_df.shape[0]} (KO, cell type) rows x {lfc_df.shape[1]} genes "
          f"({n_safe} are {safe_prefix}* rows).")
    return lfc_df
