#!/usr/bin/env python
"""
prepare_screen.py - one pipeline for all five main Perturb-seq screens.

    prepare_screen.py --screen zhu
    prepare_screen.py --screen xu --arm DMSO
    prepare_screen.py --screen shi --min-ctrl-expr 0.1

Replaces prepare_shi.py, prepare_melanoma.py, prepare_frangieh.py,
prepare_velten_pseudobulk.py and the build cells of prepare_lfc_pseudobulk.ipynb.
Weinstock is deliberately NOT here: it is a bulk-style count matrix needing
within-donor LFC, and it keeps its own prepare_weinstock.py.

The shared method
-----------------
Every screen runs the same six steps, which is why one script can serve them all:

  1. load            -> pseudobulk of MEAN RAW COUNTS PER CELL + cells per group
  2. LFC             -> CPM per group, log2((CPM_KO + 1) / (CPM_ctrl + 1)),
                        pseudocount 1, groups below --min-cells dropped,
                        self-KO masked
  3. summarise       -> per-gene |LFC| across knockouts
  4. control expr    -> mean raw counts per cell in the control cells
  5. detection floor -> keep genes with mean_ctrl_expr above --min-ctrl-expr
                        (the survival grid is always printed, so a floor can be
                        chosen and the screen re-run)
  6. write           -> <prefix>_lfc_cpm.csv, <prefix>_gene_summary_cpm.csv,
                        <prefix>_gene_ctrl_expr.csv

Steps 2-6 are written once, in _run_pooled / _run_nested. Everything that differs
between screens lives in the SCREENS registry and in that screen's loader.

Where the screens genuinely differ
----------------------------------
LFC mode:
  pooled  one control group per screen (or per treatment arm). Zhu, Frangieh, Xu.
  nested  the pseudobulk is (KO x sub-group); LFC is computed against the control
          of the SAME sub-group and then aggregated to one row per KO. This
          controls composition. Shi nests by cell type, Bowness by replicate.

Why each nests:
  Shi      the screen spans cell types, and a pooled LFC would read a shift in
           cell-type composition as a transcriptional effect.
  Bowness  rep1 is 5 adult male mice, rep2 is 3 female + 2 male from an
           independently initiated culture, so replicate is sex-confounded.
           Comparing each target to the NTCs of its own replicate cancels that.
           The library is pooled, so here the pooled LFC is still a legitimate
           secondary readout and is written alongside -- unlike a batch-split
           library, where it would simply be wrong.

Treatment arms: Frangieh and Xu were profiled under drug, so each arm is prepared
separately and the arm is part of the output filename. The other three have none.

The detection floor: Frangieh and Xu apply it at prep time (0.1), which is why
their published tables are already filtered. Zhu, Shi and Bowness write every
gene and the floor is applied downstream at analysis time. Those defaults are
kept exactly, because changing one would change published numbers.

Caching: the pseudobulk is the only expensive step, so it is cached as .npz and
reloaded unless --rebuild is given.

Use --out-dir to write somewhere temporary, which is how the outputs were checked
against the per-screen scripts this replaces.
"""
import argparse
import os
import sys

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

sys.path.insert(0, P.LIB)          # shared perturb-seq libraries
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import scipy.sparse as sp                                            # noqa: E402

from marson_KO import (compute_pseudobulk_matrix, save_pseudobulk,       # noqa: E402
                       load_pseudobulk, lfc_from_pseudobulk,
                       summarize_gene_response)
from shi_KO import (load_shi_pseudobulk, lfc_from_shi_pseudobulk,         # noqa: E402
                    aggregate_lfc_to_per_ko)

GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # the printed floor-survival grid


# ============================================================== the registry ==
# Each entry is one screen. `loader` returns (pb, sizes) for a pooled screen or
# (pb, counts) with a 2-level index for a nested one.
SCREENS = {
    "zhu": dict(
        label="Zhu (T cells)", prefix="marson", mode="pooled",
        out=f"{P.DATA}/marson",
        h5ad=f"{P.RAW}/marson_combined_with_controls.h5ad",
        cache=f"{P.DATA}/marson/marson_pseudobulk.npz",
        control="control", target_col="perturbation",
        arm_col=None, arm_in_name=False,
        min_cells=20, min_ctrl_expr=None, pseudocount=1.0,
        loader="marson",
    ),
    "frangieh": dict(
        label="Frangieh (melanoma)", prefix="frangieh", mode="pooled",
        out=f"{P.DATA}/melanoma/perturbseq",
        h5ad=f"{P.DATA}/melanoma/perturbseq/FrangiehIzar2021_RNA.h5ad",
        cache=None,                       # cheap enough to rebuild per arm
        control="control",                # lowercase in obs['perturbation']
        target_col="perturbation",
        arm_col="perturbation_2",         # scPerturb stores the treatment arm here
        arms=["Control"],                 # add "IFNy" / "Co-culture" for the others
        arm_in_name=True,
        min_cells=20, min_ctrl_expr=0.1, pseudocount=1.0,
        loader="h5ad_groupmean",
    ),
    "xu": dict(
        label="Xu (melanoma)", prefix="melanoma", mode="pooled",
        out=f"{P.DATA}/melanoma/perturbseq",
        h5ad=f"{P.DATA}/melanoma/perturbseq/melanoma_perturb.h5ad",
        cache=None,
        control="NO-TARGET", target_col="gene_target",
        arm_col="condition", arms=["DMSO", "PLX"], arm_in_name=True,
        min_cells=20, min_ctrl_expr=0.1, pseudocount=1.0,
        loader="xu",                      # .mtx + obs + genes -> h5ad, Ensembl->symbol
    ),
    "shi": dict(
        label="Shi (mouse brain)", prefix="mold", mode="nested",
        out=f"{P.DATA}/Mold_brain",
        cache=f"{P.DATA}/Mold_brain/mold_pseudobulk_raw.npz",
        control="Non_target", nest_by="cell type",
        safe_prefix="Safe_target",        # Shi carries safe-target control guides
        arm_col=None, arm_in_name=False,
        min_cells=20, min_ctrl_expr=0.1, pseudocount=1.0,
        raw_pseudocount=0.01,             # extra raw-scale summary, Shi only
        loader="shi_npz",
    ),
    "bowness": dict(
        label="Bowness (HSPC)", prefix="velten", mode="nested",
        out=f"{P.DATA}/velten_hspc",
        h5ad=f"{P.DATA}/velten_hspc/velten_perturb.h5ad",
        cache=f"{P.DATA}/velten_hspc/velten_pseudobulk_raw.npz",
        control="NTC", nest_by="replicate", group_by=["batch"],
        safe_prefix="__no_safe_targets__",   # no safe-target guides in this screen
        arm_col=None, arm_in_name=False,
        min_cells=20, min_ctrl_expr=0.1, pseudocount=1.0,
        # Bowness alone writes the UNFILTERED baseline: the published
        # velten_gene_ctrl_expr.csv has all 57,186 genes while its LFC has the
        # 8,356 above the floor. Harmless (the figures apply their own floor), but
        # reproduced here so the table matches byte for byte.
        floor_filters_ctrl_expr=False,
        loader="velten_h5ad",
        extras=("pooled", "by_group", "ontarget_kd"),
    ),
}


# ================================================================= loaders ==
def _groupmean(X, labels, var_names):
    """Mean raw counts per cell, per label. Returns (pb, sizes)."""
    groups = pd.Categorical(labels)
    ind = sp.csr_matrix((np.ones(len(groups)), (groups.codes, np.arange(len(groups)))),
                        shape=(len(groups.categories), len(groups)))
    sums = ind @ X
    sums = np.asarray(sums.todense()) if sp.issparse(sums) else np.asarray(sums)
    ncell = np.asarray(ind.sum(1)).ravel()
    pb = pd.DataFrame(sums / ncell[:, None], index=groups.categories, columns=var_names)
    return pb, pd.Series(ncell, index=groups.categories, name="n_cells")


def _load_marson(cfg, args, arm):
    """Zhu: one pseudobulk over all cells, cached. Donors and timepoints POOLED."""
    if cfg["cache"] and os.path.exists(cfg["cache"]) and not args.rebuild:
        pb, sizes = load_pseudobulk(cfg["cache"])
        print(f"loaded cached pseudobulk: {pb.shape}", flush=True)
        return pb, sizes
    import scanpy as sc
    print(f"reading {cfg['h5ad']} ...", flush=True)
    adata = sc.read_h5ad(cfg["h5ad"])
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    pb, sizes = compute_pseudobulk_matrix(
        adata, control_label=cfg["control"], perturbation_col=cfg["target_col"],
        eff_threshold=None, chunk_size=args.chunk_size,
    )
    if cfg["cache"]:
        os.makedirs(os.path.dirname(cfg["cache"]), exist_ok=True)
        save_pseudobulk(pb, sizes, cfg["cache"])
    return pb, sizes


def _load_h5ad_groupmean(cfg, args, arm):
    """Frangieh: subset the treatment arm, then group-mean by target."""
    import scanpy as sc
    print(f"loading {cfg['h5ad']} ...", flush=True)
    adata = sc.read_h5ad(cfg["h5ad"])
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    assert adata.var_names.is_unique, \
        "duplicate gene symbols in var; collapse before pseudobulk"
    print(f"atlas {adata.shape}  {cfg['arm_col']}: "
          f"{dict(adata.obs[cfg['arm_col']].value_counts())}", flush=True)
    sub = adata[adata.obs[cfg["arm_col"]] == arm]
    return _groupmean(sub.X, sub.obs[cfg["target_col"]].astype(str).values,
                      adata.var_names)


def _xu_symbol_map(cfg, ens_ids, build=False):
    """Ensembl (versioned) -> HGNC symbol via mygene, cached beside the data."""
    path = f"{cfg['out']}/melanoma_ensembl2symbol.csv"
    if os.path.exists(path) and not build:
        return pd.read_csv(path)
    print("querying mygene for the Ensembl->symbol map (needs internet; run with "
          "--map-only on a login node if this is a compute node) ...", flush=True)
    import mygene
    ens_nov = [e.split(".")[0] for e in ens_ids]
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list(dict.fromkeys(ens_nov)), scopes="ensembl.gene",
                       fields="symbol", species="human", as_dataframe=True)
    res = res[~res.index.duplicated(keep="first")]
    sym = res["symbol"] if "symbol" in res.columns else pd.Series(dtype=object)
    out = pd.DataFrame({"ensembl": ens_ids, "ensembl_noversion": ens_nov})
    out["gene_symbol"] = out["ensembl_noversion"].map(sym)
    out.to_csv(path, index=False)
    print(f"symbol map: {out['gene_symbol'].notna().sum()}/{len(out)} mapped "
          f"-> {path}", flush=True)
    return out


def _collapse_to_symbol(pb_ens, symbol_by_ens):
    """Sum Ensembl columns sharing a gene_symbol; drop unmapped genes."""
    sym = symbol_by_ens.reindex(pb_ens.columns).values
    valid = pd.notna(sym)
    uniq, inv = np.unique(sym[valid].astype(str), return_inverse=True)
    onehot = sp.csr_matrix((np.ones(inv.size), (np.arange(inv.size), inv)),
                           shape=(inv.size, len(uniq)))
    return pd.DataFrame(pb_ens.values[:, valid] @ onehot,
                        index=pb_ens.index, columns=uniq)


def _load_xu(cfg, args, arm):
    """Xu: build the atlas from .mtx the first time, then group-mean + collapse."""
    import anndata as ad
    import gc
    D = cfg["out"]
    genes = pd.read_csv(f"{D}/melanoma_genes.tsv", header=None)[0].values
    smap = _xu_symbol_map(cfg, genes)
    symbol_by_ens = pd.Series(smap["gene_symbol"].values, index=smap["ensembl"].values)

    if os.path.exists(cfg["h5ad"]) and not args.rebuild:
        print(f"loading {cfg['h5ad']} ...", flush=True)
        adata = ad.read_h5ad(cfg["h5ad"])
    else:
        import scipy.io
        print("building the atlas from melanoma_RNA_counts.mtx.gz ...", flush=True)
        X = scipy.io.mmread(f"{D}/melanoma_RNA_counts.mtx.gz").T.tocsr()  # cells x genes
        obs = pd.read_csv(f"{D}/melanoma_obs.csv").set_index("cell")
        assert X.shape[0] == len(obs) and X.shape[1] == len(genes), "shape mismatch"
        var = pd.DataFrame({"ensembl": genes,
                            "gene_symbol": symbol_by_ens.reindex(genes).values},
                           index=genes)
        adata = ad.AnnData(X=X, obs=obs, var=var)
        del X
        gc.collect()
        adata.write(cfg["h5ad"])
        print(f"wrote atlas {cfg['h5ad']} {adata.shape}", flush=True)

    print(f"atlas {adata.shape}  {cfg['arm_col']}: "
          f"{dict(adata.obs[cfg['arm_col']].value_counts())}", flush=True)
    # NOT .astype(str): obs['gene_target'] is stored in the h5ad as a categorical
    # whose category order is not plain alphabetical (KCTD5 before KCTD10, RNF7
    # before RNF146, SUPT7L before SUPT20H). pd.Categorical preserves that order,
    # and it is the row order of the published tables. Casting to str re-sorts
    # them ASCII and silently permutes three pairs of rows.
    sub = adata[adata.obs[cfg["arm_col"]] == arm]
    pb_ens, sizes = _groupmean(sub.X, sub.obs[cfg["target_col"]].values,
                               adata.var_names)
    pb = _collapse_to_symbol(pb_ens, symbol_by_ens)
    print(f"[{arm}] collapsed {pb_ens.shape[1]} Ensembl ids -> {pb.shape[1]} symbols",
          flush=True)
    return pb, sizes


def _load_shi_npz(cfg, args, arm):
    """Shi: a (KO x cell type) raw pseudobulk, pre-built by build_shi_pseudobulk.py."""
    pb, counts = load_shi_pseudobulk(cfg["cache"])
    n_ctrl = int((pb.index.get_level_values(0) == cfg["control"]).sum())
    print(f"raw pseudobulk: {pb.shape} | control rows: {n_ctrl}", flush=True)
    return pb, counts


def _build_nested_pseudobulk(adata, group_cols, target_col, chunk_size=20_000):
    """Mean raw counts per (target x group) group. Nothing filtered or normalised."""
    pert = adata.obs[target_col].astype(str).values
    grp = adata.obs[group_cols[0]].astype(str).values
    for c in group_cols[1:]:
        grp = np.char.add(np.char.add(grp, "|"), adata.obs[c].astype(str).values)

    keys = np.array([f"{p}\t{g}" for p, g in zip(pert, grp)])
    cats, codes = np.unique(keys, return_inverse=True)
    n_groups, n_genes = len(cats), adata.n_vars
    counts_arr = np.bincount(codes, minlength=n_groups)

    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    print(f"summing raw counts into {n_groups} (target x group) groups "
          f"x {n_genes} genes ...", flush=True)
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)
        c = codes[start:end]
        ind = sp.csr_matrix((np.ones(end - start, dtype=np.float64),
                             (c, np.arange(end - start))),
                            shape=(n_groups, end - start))
        contrib = ind @ adata.X[start:end]
        sums += contrib.toarray() if sp.issparse(contrib) else np.asarray(contrib)

    idx = pd.MultiIndex.from_tuples([tuple(k.split("\t")) for k in cats],
                                    names=["perturbation", "group"])
    pb = pd.DataFrame(sums / np.maximum(counts_arr, 1)[:, None],
                      index=idx, columns=adata.var_names.astype(str))
    counts = pd.Series(counts_arr, index=idx, name="n_cells")
    print(f"raw pseudobulk: {pb.shape[0]} groups x {pb.shape[1]} genes, "
          f"{int(counts.sum()):,} cells.", flush=True)
    return pb, counts


def _save_nested(pb, counts, path, group_by):
    """group_by is stored so a cache built under one grouping is never reused
    silently under another."""
    np.savez_compressed(
        path,
        values=pb.values.astype(np.float32),
        perts=pb.index.get_level_values(0).values.astype(str),
        groups=pb.index.get_level_values(1).values.astype(str),
        genes=pb.columns.values.astype(str),
        counts=counts.values.astype(np.int64),
        group_by=np.array(group_by, dtype=str))
    print(f"cached raw pseudobulk ({'+'.join(group_by)}) -> {path}", flush=True)


def _load_nested(path):
    """Returns (pb, counts, group_by); group_by is None for caches predating it."""
    z = np.load(path, allow_pickle=False)
    idx = pd.MultiIndex.from_arrays([z["perts"], z["groups"]],
                                    names=["perturbation", "group"])
    gb = [str(x) for x in z["group_by"]] if "group_by" in z.files else None
    return (pd.DataFrame(z["values"], index=idx, columns=z["genes"]),
            pd.Series(z["counts"], index=idx, name="n_cells"), gb)


def _load_velten_h5ad(cfg, args, arm):
    """Bowness: a (target x replicate) raw pseudobulk, cached."""
    group_by = args.group_by or cfg["group_by"]
    if cfg["cache"] and os.path.exists(cfg["cache"]) and not args.rebuild:
        pb, counts, cached_gb = _load_nested(cfg["cache"])
        if cached_gb == list(group_by):
            print(f"loading cached pseudobulk ({'+'.join(cached_gb)}) {cfg['cache']}",
                  flush=True)
            return pb, counts
        print(f"cache at {cfg['cache']} was built with group_by="
              f"{cached_gb if cached_gb else 'UNKNOWN'}, need {list(group_by)} "
              f"-> rebuilding", flush=True)

    import anndata as ad
    adata = ad.read_h5ad(cfg["h5ad"])
    print(f"loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes", flush=True)
    for c in group_by:
        if c not in adata.obs.columns:
            raise KeyError(f"--group-by column '{c}' not in obs")
    pb, counts = _build_nested_pseudobulk(adata, list(group_by), "perturbation")
    del adata
    if cfg["cache"]:
        _save_nested(pb, counts, cfg["cache"], list(group_by))
    return pb, counts


LOADERS = {"marson": _load_marson, "h5ad_groupmean": _load_h5ad_groupmean,
           "xu": _load_xu, "shi_npz": _load_shi_npz, "velten_h5ad": _load_velten_h5ad}


# =============================================== steps 3-6, shared by all ==
def _floor(control_expr, min_ctrl_expr):
    """Print the survival grid; return the genes to keep, or None for all."""
    print("\ngene survival by control-expression floor (raw counts/cell):", flush=True)
    for thr in GRID:
        print(f"  > {thr:<4}: {int((control_expr > thr).sum()):6d} / "
              f"{len(control_expr)} genes", flush=True)
    if min_ctrl_expr is None:
        print("no floor applied; keeping all genes. Pick a value from the grid "
              "above and re-run with --min-ctrl-expr to cut genes.", flush=True)
        return None
    keep = control_expr.index[control_expr > min_ctrl_expr]
    print(f"applying floor: keeping {len(keep)} genes "
          f"(mean_ctrl_expr > {min_ctrl_expr})", flush=True)
    return keep


def _write(cfg, args, arm, lfc, gs, control_expr, extra=None):
    out, pre = args.out_dir or cfg["out"], cfg["prefix"]
    os.makedirs(out, exist_ok=True)
    suf = f"_{arm}" if cfg["arm_in_name"] and arm else ""
    lfc.to_csv(f"{out}/{pre}_lfc_cpm{suf}.csv")
    gs.to_csv(f"{out}/{pre}_gene_summary_cpm{suf}.csv")
    control_expr.to_csv(f"{out}/{pre}_gene_ctrl_expr{suf}.csv")
    written = [f"{pre}_lfc_cpm{suf}.csv", f"{pre}_gene_summary_cpm{suf}.csv",
               f"{pre}_gene_ctrl_expr{suf}.csv"]
    for name, obj in (extra or {}).items():
        obj.to_csv(f"{out}/{pre}_{name}.csv")
        written.append(f"{pre}_{name}.csv")
    print(f"\nFINAL: {lfc.shape[0]} KOs x {lfc.shape[1]} genes")
    print("wrote " + ", ".join(written) + f" -> {out}")


def _run_pooled(cfg, args, arm):
    """One control group per screen (or per arm): Zhu, Frangieh, Xu."""
    pb, sizes = LOADERS[cfg["loader"]](cfg, args, arm)
    ctl = cfg["control"]
    if ctl not in pb.index:
        raise ValueError(f"no {ctl!r} group to serve as control "
                         f"(have: {list(pb.index[:5])} ...)")
    tag = f"[{arm}] " if arm else ""
    print(f"{tag}{len(sizes)} groups "
          f"({int((sizes >= args.min_cells).sum())} >= {args.min_cells} cells), "
          f"{pb.shape[1]} genes, control n={int(sizes[ctl])}", flush=True)

    lfc = lfc_from_pseudobulk(pb, group_sizes=sizes, min_cells_per_ko=args.min_cells,
                              control_label=ctl, normalize="group",
                              pseudocount=cfg["pseudocount"], mask_self_ko=True)
    gs = summarize_gene_response(lfc)

    control_expr = pb.loc[ctl].copy()
    control_expr.name = "mean_ctrl_expr"

    keep = _floor(control_expr, args.min_ctrl_expr)
    if keep is not None:
        lfc = lfc.loc[:, lfc.columns.intersection(keep)]
        gs = gs.loc[gs.index.intersection(keep)]
        control_expr = control_expr.loc[control_expr.index.intersection(keep)]
    _write(cfg, args, arm, lfc, gs, control_expr)


def _weighted_control_expr(pb, counts, control):
    """Mean raw control counts per gene, CELL-COUNT weighted across sub-groups.

    Weighted rather than a flat row mean so the baseline asks the same question as
    the aggregation, which is also cell-count weighted: control populations can
    differ hundredfold in size, and a flat mean would let the smallest count as
    much as the largest.
    """
    rows = pb.index.get_level_values(0) == control
    cw = counts[rows].values.astype(np.float64)
    return pd.Series((pb[rows].values * cw[:, None]).sum(0) / cw.sum(),
                     index=pb.columns, name="mean_ctrl_expr")


def _run_nested(cfg, args, arm):
    """LFC within each sub-group, then aggregated per KO: Shi, Bowness."""
    pb, counts = LOADERS[cfg["loader"]](cfg, args, arm)
    ctl, extras = cfg["control"], cfg.get("extras", ())
    print(f"\ngroups: {pb.shape[0]} | nested by {cfg['nest_by']}", flush=True)

    have_ctrl = set(pb.index[pb.index.get_level_values(0) == ctl].get_level_values(1))
    orphan = sorted({g for p, g in pb.index if p != ctl and g not in have_ctrl})
    if orphan:
        print(f"WARNING groups with no matching control (dropped): {orphan}", flush=True)

    kw = dict(control_label=ctl, min_cells=args.min_cells, normalize="group",
              pseudocount=cfg["pseudocount"], safe_prefix=cfg["safe_prefix"])

    print(f"\n=== within-{cfg['nest_by']} LFC (the readout) ===", flush=True)
    lfc_g = lfc_from_shi_pseudobulk(pb, counts, **kw)
    lfc = aggregate_lfc_to_per_ko(lfc_g, counts, weight=args.weight)

    # Shi's safe-target guides are controls, not knockouts: they must not enter the
    # per-gene summary, or the responsiveness floor would include control-vs-control.
    is_safe = pd.Series(lfc.index.get_level_values(0)).str.startswith(
        cfg["safe_prefix"]).values
    gs = summarize_gene_response(lfc[~is_safe])

    extra = {}

    if "ontarget_kd" in extras:
        # CRISPRi: each target's effect on its OWN transcript must be negative, and
        # that needs the UNMASKED self-LFC. The cheapest end-to-end check that the
        # guide -> target -> pseudobulk -> LFC chain is wired up correctly.
        print("\n=== on-target knockdown QC ===", flush=True)
        un = aggregate_lfc_to_per_ko(
            lfc_from_shi_pseudobulk(pb, counts, mask_self_ko=False, **kw),
            counts, weight=args.weight)
        rows = [{"target": ko, "self_lfc": float(un.loc[ko, ko])}
                for ko in un.index if ko != ctl and ko in un.columns]
        kd = pd.DataFrame(rows).set_index("target").sort_values("self_lfc")
        if len(kd):
            neg = float((kd.self_lfc < 0).mean())
            print(f"{len(kd)} targets measured on their own transcript | "
                  f"median self-LFC = {kd.self_lfc.median():.3f} | "
                  f"{100 * neg:.0f}% negative")
            if neg < 0.5:
                print("  *** WARNING: fewer than half the targets show negative "
                      "self-LFC. Suspect the guide->target mapping or the LFC. ***")
        extra["ontarget_kd"] = kd

    lfc_pool = None
    if "pooled" in extras:
        print("\n=== pooled LFC (secondary readout) ===", flush=True)
        w = counts.values.astype(np.float64)[:, None]
        num = pd.DataFrame(pb.values * w, index=pb.index, columns=pb.columns) \
            .groupby(level=0, observed=True).sum()
        den = counts.groupby(level=0, observed=True).sum()
        lfc_pool = lfc_from_pseudobulk(num.div(den, axis=0), den,
                                       min_cells_per_ko=args.min_cells,
                                       control_label=ctl, normalize="group",
                                       pseudocount=cfg["pseudocount"])
        sk, sg = lfc.index.intersection(lfc_pool.index), \
            lfc.columns.intersection(lfc_pool.columns)
        print(f"mean |LFC| within-{cfg['nest_by']} = {lfc.abs().mean().mean():.4f} | "
              f"pooled = {lfc_pool.loc[sk, sg].abs().mean().mean():.4f}")
        print("  (the library is pooled, so these should agree closely -- a large "
              "gap means a bug, not a confound)")

    if cfg.get("raw_pseudocount") is not None:
        # Shi only: the raw-scale summary that shows CPM removed the pseudocount
        # artefact. Kept because the published comparison table uses it.
        print(f"\n=== raw-scale LFC (pseudocount {cfg['raw_pseudocount']}) ===",
              flush=True)
        raw_kw = dict(kw, normalize=None, pseudocount=cfg["raw_pseudocount"])
        lfc_raw = aggregate_lfc_to_per_ko(
            lfc_from_shi_pseudobulk(pb, counts, **raw_kw), counts, weight=args.weight)
        extra["gene_summary_raw"] = summarize_gene_response(lfc_raw[~is_safe])
        del lfc_raw

    control_expr = _weighted_control_expr(pb, counts, ctl)
    keep = _floor(control_expr, args.min_ctrl_expr)
    if keep is not None:
        lfc = lfc[lfc.columns.intersection(keep)]
        gs = summarize_gene_response(lfc[~is_safe])
        if cfg.get("floor_filters_ctrl_expr", True):
            control_expr = control_expr.loc[control_expr.index.intersection(keep)]
        if lfc_pool is not None:
            lfc_pool = lfc_pool[lfc_pool.columns.intersection(keep)]
        if "gene_summary_raw" in extra:
            extra["gene_summary_raw"] = extra["gene_summary_raw"].loc[
                extra["gene_summary_raw"].index.intersection(keep)]

    if "by_group" in extras:
        tag = "_".join(args.group_by or cfg["group_by"])
        extra[f"lfc_cpm_by_{tag}"] = lfc_g
    if lfc_pool is not None:
        extra["lfc_cpm_pooled"] = lfc_pool

    dropped = sorted(set(pb.index.get_level_values(0)) - {ctl} - set(lfc.index))
    print(f"KOs dropped by min_cells={args.min_cells}: {dropped or 'none'}")
    # the LFC table keeps every row, safe-target controls included: they are only
    # excluded from the per-gene SUMMARY, where a control-vs-control comparison
    # would drag the responsiveness floor down. Matches prepare_shi.py.
    _write(cfg, args, arm, lfc, gs, control_expr, extra)


# ==================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", required=True, choices=sorted(SCREENS),
                    help="which screen to prepare")
    ap.add_argument("--arm", default=None,
                    help="one treatment arm (Frangieh, Xu). Default: every arm "
                         "in the registry.")
    ap.add_argument("--out-dir", default=None,
                    help="override the screen's output directory, e.g. to write "
                         "somewhere temporary for a comparison run")
    ap.add_argument("--min-cells", type=int, default=None,
                    help="per-group cell floor (default: the screen's own value, 20). "
                         "For Shi this must match the floor its cached .npz was built "
                         "with, or the result is silently re-filtered.")
    ap.add_argument("--min-ctrl-expr", type=float, default=None,
                    help="gene detection floor on mean control counts/cell. Default: "
                         "the screen's own value (0.1 for Frangieh and Xu, none for "
                         "Zhu, Shi and Bowness). The survival grid prints either way.")
    ap.add_argument("--weight", choices=["cells", "equal"], default="cells",
                    help="nested screens: per-KO aggregation weighting across "
                         "sub-groups. 'cells' is cell-count weighted, matching a "
                         "pooled screen.")
    ap.add_argument("--group-by", nargs="+", default=None,
                    help="nested screens built from an h5ad: the obs columns to nest "
                         "by (Bowness default: batch, i.e. the replicate)")
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute the pseudobulk even if the cache exists")
    ap.add_argument("--map-only", action="store_true",
                    help="Xu: build the Ensembl->symbol cache (needs internet) and exit")
    args = ap.parse_args()

    cfg = SCREENS[args.screen]
    # a screen's own defaults win unless the flag was given
    if args.min_cells is None:
        args.min_cells = cfg["min_cells"]
    if args.min_ctrl_expr is None:
        args.min_ctrl_expr = cfg["min_ctrl_expr"]

    if args.map_only:
        if args.screen != "xu":
            ap.error("--map-only applies to --screen xu only")
        genes = pd.read_csv(f"{cfg['out']}/melanoma_genes.tsv", header=None)[0].values
        _xu_symbol_map(cfg, genes, build=True)
        return

    arms = [args.arm] if args.arm else cfg.get("arms", [None])
    print(f"=== {cfg['label']}  [{cfg['mode']}]  arms: "
          f"{[a or '-' for a in arms]} ===\n", flush=True)
    run = _run_pooled if cfg["mode"] == "pooled" else _run_nested
    for arm in arms:
        run(cfg, args, arm)


if __name__ == "__main__":
    main()
