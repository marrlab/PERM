#!/usr/bin/env python
"""
prepare_velten.py - step 1 of the Velten HSPC Perturb-seq prep: turn the GEO
GSE320250 Parse/Split-seq matrices into ONE raw-count h5ad that the shared
pseudobulk step can consume unchanged.

Dataset: Bowness et al. 2026 (bioRxiv 10.64898/2026.08.27.747033), CRISPRi
Perturb-seq, 520 target genes + 4 non-targeting control groups, primary mouse
HSPC expansion cultures, Parse Evercode WT v3 + CRISPR Detect.

Why we build our own h5ad instead of downloading theirs
------------------------------------------------------
GSE320250 does ship *_DGE_filtered_anndata.h5ad (8.5G rep1 / 1.9G rep2). They are
unusable here, confirmed by reading their headers remotely:
  * X is scaled float64 (uns['log1p'] present, only layer is 'norm10k', no .raw)
    -> there are NO raw counts to pseudobulk, and this pipeline pseudobulks raw.
  * obs is bc_wells/sample/species/gene_count/tscp_count/mread_count/leiden
    -> NO guide, NO target, NO control label. Nothing to group by.
The .mtx route has both. The DGE and guide matrices share an IDENTICAL bc_wells
row order (asserted below), so the counts<->perturbation join is positional.

Design decisions
----------------
* BATCH = REPLICATE, not sublibrary.
  rep1 = 5 adult male mice; rep2 = 3 female + 2 male, different animals, an
  independently initiated culture. Replicate is the real biological batch and it
  is SEX-CONFOUNDED (rep1 is all male). LFC must be computed within replicate --
  pooling raw cells across replicates would push Xist/Y-linked genes into every
  "effect". Sublibraries are aliquots of one barcoded pool split before
  sequencing (technical; median-UMI spread is only 1.51x in rep1, 1.13x in rep2),
  so they are recorded in obs but are NOT the grouping: grouping on them would
  cut median cells/target from 89 to ~11 for no benefit.

  NOTE this is the OPPOSITE of the GSE213511/Mali situation. There the library was
  batch-split so batch was confounded WITH TARGET IDENTITY and within-batch was
  mandatory. Here the library is pooled -- 515/522 targets appear in all three
  rep2 sublibraries and every sublibrary carries its own controls -- so
  within-replicate is about the sex/culture difference, not about target
  confounding.

* SINGLE-TARGET, not single-guide, is the default cell filter.
  High-MOI screen: 37% (rep1) / 46% (rep2) of assigned cells carry several
  guides, and a cell knocked down in two different genes cannot be attributed to
  either. But a cell carrying two guides that BOTH target Cebpa is still one
  clean perturbation, so filtering on distinct TARGETS rather than on guide count
  recovers ~5% more cells (rep1 62,841 -> 65,950; median 84 -> 89 cells/target).
  The authors used single-guide; --guide-filter single_guide reproduces them.

* GUIDE UMI THRESHOLD default 2, matching the authors' "transcript threshold of 2".

* CONTROLS. Four non-targeting groups -- NO_SITE / ONE_INTERGENIC (50 guides each)
  and their _5X counterparts (10 each), 120 control guides total. All four are
  relabelled to CONTROL ('NTC') for the shared pipeline; the original group is
  kept in obs['control_group'] so they stay separable.

  *** Target_Gene in guides.csv carries TRAILING WHITESPACE on two of them: the
  literal values are 'NO_SITE ' and 'NO_SITE_5X '. `== 'NO_SITE'` therefore
  matches ZERO guides, silently, with no error. Without the .strip() below you
  lose 3,549 of 5,695 rep1 control cells (62%) AND gain two junk perturbation
  targets named 'NO_SITE '. The strip is load-bearing. ***

* '5X' IS NOT A BATCH. 20 genes (and the control groups) were ordered at 5-fold
  library representation, so they simply have ~5x the cells. The authors' DE
  tables split Rep1_5X from Rep1_Rem for that reason. Nothing to correct for.

* SYMBOL FIXES -- self-KO masking downstream is by var_name, so a target whose
  symbol is absent from the annotation is silently never masked:
    - 'Sars' -> 'Sars1'. The guide library uses the old symbol; mm39 calls it
      Sars1 (ENSMUSG00000068739). It is the ONLY one of the 520 targets missing
      from the annotation -- the same failure mode as Gltscr1/Bicra in the HSC
      script.
    - 'Ddit3' has two gene_ids. Canonical ENSMUSG00000025408 precedes
      ENSMUSG00000116429, so var_names_make_unique() leaves it holding the clean
      symbol. Asserted, not assumed.

Output (in $PERM_DATA/velten_hspc/):
  velten_perturb.h5ad   raw counts in X; obs.perturbation / batch / sublib

Run:  python prepare_velten.py
      python prepare_velten.py --guide-filter single_guide   # authors' rule
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import argparse
import gzip
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

D = f"{P.DATA}/velten_hspc"
P = os.path.join(D, "GSE320250_CRISPRi500_HSC_WT")
GUIDES_CSV = os.path.join(D, "GSE320250_CRISPRi500_guides.csv.gz")
OUT = os.path.join(D, "velten_perturb.h5ad")

CONTROL = "NTC"
CONTROL_PREFIXES = ("NO_SITE", "ONE_INTERGENIC")
TARGET_ALIASES = {"Sars": "Sars1"}
CANONICAL = {"Ddit3": "ENSMUSG00000025408"}
CHUNK = 20_000_000


def guide_to_target():
    """Guide-matrix column order -> target gene symbol (whitespace stripped)."""
    g = pd.read_csv(GUIDES_CSV)
    g.columns = [c.lstrip("#") for c in g.columns]
    raw = g.Target_Gene.astype(str)
    g["Target_Gene"] = raw.str.strip()
    n_ws = int((raw != g.Target_Gene).sum())
    if n_ws:
        bad = sorted(set(raw[raw != g.Target_Gene]))
        print(f"  stripped whitespace from {n_ws} guide rows: {bad}")
    ag = pd.read_csv(P + "_guide_RNAs_filtered_all_guides.csv.gz")
    tgt = pd.Series(ag.gene_name.values).map(
        dict(zip(g.Guide_Name, g.Target_Gene))).values
    if pd.isna(tgt).any():
        raise ValueError(f"{int(pd.isna(tgt).sum())} guide columns have no target")
    return tgt


def make_unique(index):
    """anndata's var_names_make_unique semantics, applied BEFORE concat.

    ad.concat reindexes var and dies on duplicate labels, so the gene index must
    already be unique when the per-replicate objects are built -- doing it after
    concat is too late. First occurrence keeps the bare symbol (which is what the
    self-KO mask needs), later ones get -1, -2, ...
    """
    idx = pd.Index(index).astype(str)
    if idx.is_unique:
        return idx
    seen, out = {}, []
    for name in idx:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return pd.Index(out)


def mtx_header(path):
    """(n_rows, n_cols, nnz) from a MatrixMarket header, without reading the body."""
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("%"):
                r, c, n = line.split()
                return int(r), int(c), int(n)
    raise ValueError(f"no dims line in {path}")


def read_mtx_small(path):
    """Whole-file read, for the guide matrix only (<=200k entries)."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=3, header=None,
                     names=["r", "c", "v"], dtype=np.int64)
    return df.r.values - 1, df.c.values - 1, df.v.values


def assign(rep, tgt, min_guide_umi, mode):
    """Per-cell perturbation call. Returns (keep_idx, labels DataFrame)."""
    n_cells, n_guides, _ = mtx_header(
        f"{P}_{rep}_guide_RNAs_filtered_count_matrix.mtx.gz")
    r, c, v = read_mtx_small(f"{P}_{rep}_guide_RNAs_filtered_count_matrix.mtx.gz")
    n_raw = len(r)
    keep = v >= min_guide_umi
    r, c, v = r[keep], c[keep], v[keep]
    print(f"  guide UMI >= {min_guide_umi}: {len(r):,}/{n_raw:,} guide calls kept")

    df = pd.DataFrame({"cell": r, "target": tgt[c]})
    gb = df.groupby("cell", sort=True)
    n_g = gb.size()
    n_t = gb.target.nunique()
    first = gb.target.first()

    if mode == "single_target":
        sel = n_t.index[n_t.values == 1]
    elif mode == "single_guide":
        sel = n_g.index[n_g.values == 1]
    elif mode == "all":
        sel = n_t.index
    else:
        raise ValueError(mode)
    sel = np.asarray(sel, dtype=np.int64)

    lab = pd.DataFrame({
        "target": first.loc[sel].values,
        "n_guides": n_g.loc[sel].values.astype(np.int32),
        "n_targets": n_t.loc[sel].values.astype(np.int32),
    })
    print(f"  cells with >=1 guide: {len(n_t):,}/{n_cells:,} | "
          f"filter '{mode}' keeps {len(sel):,}")
    return n_cells, sel, lab


def stream_dge(rep, keep_idx, n_cells, n_genes, chunk=CHUNK):
    """Raw-count CSR for keep_idx only, streamed so the full matrix never lands."""
    path = f"{P}_{rep}_DGE_filtered_count_matrix.mtx.gz"
    hr, hc, nnz = mtx_header(path)
    if hr != n_cells or hc != n_genes:
        raise ValueError(f"{rep} DGE is {hr}x{hc}, expected {n_cells}x{n_genes}")
    keep_mask = np.zeros(n_cells, dtype=bool)
    keep_mask[keep_idx] = True
    remap = np.full(n_cells, -1, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))

    rr, cc, vv = [], [], []
    seen = 0
    for ch in pd.read_csv(path, sep=r"\s+", skiprows=3, header=None,
                          names=["r", "c", "v"], dtype=np.int32, chunksize=chunk):
        seen += len(ch)
        r = ch.r.values.astype(np.int64) - 1
        m = keep_mask[r]
        if not m.any():
            continue
        rr.append(remap[r[m]].astype(np.int32))
        cc.append((ch.c.values[m] - 1).astype(np.int32))
        vv.append(ch.v.values[m].astype(np.int32))
        print(f"    {seen:,}/{nnz:,} entries", flush=True)
    if seen != nnz:
        raise ValueError(f"{rep}: read {seen:,} entries, header declared {nnz:,}")
    X = sp.coo_matrix(
        (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
        shape=(len(keep_idx), n_genes), dtype=np.int32).tocsr()
    print(f"  X {X.shape[0]:,} x {X.shape[1]:,}, {X.nnz:,} nonzero")
    return X


def build_replicate(rep, tgt, genes, args):
    print(f"\n===== {rep}")
    md = pd.read_csv(f"{P}_{rep}_DGE_filtered_cell_metadata.csv.gz")
    gmd = pd.read_csv(f"{P}_{rep}_guide_RNAs_filtered_cell_metadata.csv.gz")
    if not (md.bc_wells.values == gmd.bc_wells.values).all():
        raise ValueError(f"{rep}: DGE and guide bc_wells order differ -- "
                         "the positional join would be wrong")
    print("  bc_wells order identical across DGE and guide matrices: OK")

    n_cells, keep_idx, lab = assign(rep, tgt, args.min_guide_umi, args.guide_filter)
    if len(md) != n_cells:
        raise ValueError(f"{rep}: metadata has {len(md)} rows, matrix has {n_cells}")

    obs = md.iloc[keep_idx].reset_index(drop=True)
    obs["target"] = lab.target.values
    obs["n_guides"] = lab.n_guides.values
    obs["n_targets"] = lab.n_targets.values
    is_ctrl = obs.target.str.startswith(CONTROL_PREFIXES).values
    obs["control_group"] = np.where(is_ctrl, obs.target, "")
    obs["perturbation"] = np.where(is_ctrl, CONTROL,
                                   obs.target.replace(TARGET_ALIASES))
    obs["replicate"] = rep
    obs["batch"] = rep
    obs["sublib"] = obs.bc_wells.str.extract(r"__(s\d+)").iloc[:, 0].values
    obs["rep_sublib"] = obs.replicate + "|" + obs.sublib
    obs.index = obs.replicate + ":" + obs.bc_wells
    print(f"  controls -> {CONTROL}: {int(is_ctrl.sum()):,} cells "
          f"({obs.control_group[is_ctrl].value_counts().to_dict()})")

    X = stream_dge(rep, keep_idx, n_cells, len(genes))
    return ad.AnnData(X=X, obs=obs, var=genes.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", nargs="+", default=["rep1", "rep2"])
    ap.add_argument("--guide-filter", default="single_target",
                    choices=["single_target", "single_guide", "all"],
                    help="single_target (default) keeps cells hitting exactly one "
                         "gene; single_guide reproduces the authors' rule.")
    ap.add_argument("--min-guide-umi", type=int, default=2,
                    help="Guide transcript threshold (authors used 2).")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    genes = pd.read_csv(P + "_DGE_filtered_all_genes.csv.gz")
    genes = genes.rename(columns={"gene_name": "symbol"}).set_index("symbol")
    n_dup = int(genes.index.duplicated().sum())
    genes.index = make_unique(genes.index)
    genes.index.name = None
    print(f"annotation: {len(genes):,} genes ({n_dup} duplicate symbols de-duplicated)")

    tgt = guide_to_target()
    parts = [build_replicate(r, tgt, genes, args) for r in args.replicates]
    adata = parts[0] if len(parts) == 1 else ad.concat(parts, axis=0, merge="same")
    adata.var = genes.copy()
    assert adata.var_names.is_unique, "var_names must already be unique here"

    # self-KO masking is by var_name: the canonical row must keep the bare symbol
    for sym, gid in CANONICAL.items():
        if sym in adata.var_names and adata.var.loc[sym, "gene_id"] != gid:
            raise ValueError(f"{sym} resolved to {adata.var.loc[sym, 'gene_id']}, "
                             f"expected canonical {gid}")
    missing = sorted(set(adata.obs.perturbation) - {CONTROL} - set(adata.var_names))
    if missing:
        raise ValueError("targets absent from var_names (self-KO mask would "
                         f"silently miss them): {missing}")
    print(f"\nall {adata.obs.perturbation.nunique() - 1} targets present in var_names")

    adata.obs["perturbation"] = adata.obs.perturbation.astype("category")
    adata.obs["batch"] = adata.obs.batch.astype("category")
    vc = adata.obs.perturbation.value_counts()
    print(f"\n{adata.n_obs:,} cells x {adata.n_vars:,} genes | "
          f"{len(vc)} perturbations incl. {CONTROL}")
    print(f"{CONTROL} cells: {int(vc.get(CONTROL, 0)):,}")
    tv = vc.drop(CONTROL, errors="ignore")
    print(f"cells/target: median {int(tv.median())} min {int(tv.min())} "
          f"max {int(tv.max())}")
    print("cells per batch:")
    print(pd.crosstab(adata.obs.batch, adata.obs.perturbation == CONTROL)
          .rename(columns={False: "perturbed", True: CONTROL}).to_string())

    adata.write_h5ad(args.out, compression="gzip")
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
