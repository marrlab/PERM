#!/usr/bin/env python
"""
KO axis: is perturbing a memory gene a bigger driver than perturbing a matched
non-memory gene?

The readout axis (figures 2a-2f) asks how much a gene MOVES when other genes are
knocked out. This asks the opposite: knock the memory gene out, and measure how
much the rest of the transcriptome moves. The two are not the same question and
the answer has not been the same.

    unit   : one KO (one row of <dataset>_lfc_cpm.csv)
    value  : median (and mean) |log2FC| over the readout genes above `floor`
    groups : memory KOs  = target is in that method's top-N memory set
             background  = target called by NO method at that top-N ("strict")

WHY MATCHING IS NOT OPTIONAL HERE
---------------------------------
Per-KO |LFC| is dominated by how many cells carried that guide: Spearman -0.63
against cells-per-KO in Marson. A 200-cell KO looks ~1.8x "stronger" than a
1,300-cell one from sampling noise alone. Memory-gene KOs have BOTH more cells and
much higher target expression, so the two confounds push the same way and either
one alone leaves an artefact. Everything below is matched on both, and the
UNMATCHED number is reported next to the matched one so the size of the confound
is visible rather than taken on trust.

`marson_KO.compare_ko_effect_groups` and `compare_top_scg_effect` match NEITHER
covariate. Do not use them for this question.

THREE FILTERS THAT ARE EASY TO CONFLATE
---------------------------------------
1. readout floor : which COLUMNS enter the per-KO summary (genes measured well
                   enough for a fold change to mean anything)
2. target floor  : which ROWS are eligible (the KO's own target must be measured,
                   otherwise "is this target a memory gene?" is unanswerable)
3. self-KO       : the target's own column must not count towards its own score.
                   Already NaN on the diagonal of the LFC tables; asserted below
                   rather than assumed.

Usage
-----
    python ko_axis_memory.py --dataset all --top-n 100
    python ko_axis_memory.py --dataset marson --top-n 100 250 500 --out results.csv
"""
from __future__ import annotations

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import argparse
import os
import sys

import numpy as np
import pandas as pd

MARSON_DIR = P.LIB
if MARSON_DIR not in sys.path:                      # shared libs must not move
    sys.path.insert(0, MARSON_DIR)

from marson_KO import load_memory_gene_set, cliffs_delta, cliffs_ci  # noqa: E402
from matching import matched_comparison                              # noqa: E402

DATA = P.DATA
BENCH = P.MEMGENES        # <PERM_MEMGENES_ROOT>/python_implements
SCG_VARIANT = "clones2-8_"        # figures.SCG_VARIANT_DEFAULT — keep in step

# The Mold METHOD is excluded from every figure (user, 2026-09-16); keep it out here
# too, or panel 2b would count memory genes over five methods while the rest of the
# figure uses four. Separate from the Mold BRAIN dataset exclusion.
METHOD_PREFIX = {"CORAL": "coral", "GEMLI": "gemli",
                 "Power-Seek": "powerseek", "SCG": "scg"}


def method_files(suffix, scg_variant=SCG_VARIANT):
    """{method: filename} for one lineage dataset's memory-gene tables."""
    out = {m: f"{p}_memory_genes_{suffix}.csv" for m, p in METHOD_PREFIX.items()}
    out["SCG"] = f"scg_memory_genes_{scg_variant}{suffix}.csv"
    return out


# Each screen needs a per-KO x gene LFC matrix AND per-KO cell counts. Screens
# without the cell counts are deliberately absent: without them the -0.63 depth
# confound cannot be matched and the numbers would not be trustworthy.
SCREENS = {
    "marson": dict(
        label="Zhu (T cells)", short="Zhu",
        lfc=f"{DATA}/marson/marson_lfc_cpm.csv",
        expr=f"{DATA}/marson/marson_gene_ctrl_expr.csv",
        npz=f"{DATA}/marson/marson_pseudobulk.npz",
        npz_keys=("index", "sizes"),      # (KO names, cells per KO)
        memdir=f"{BENCH}/T-lineage",
        suffix="T1_T2",
        memory_source="Michaels",
    ),
    "mold": dict(
        label="Shi (mouse brain)", short="Shi",
        lfc=f"{DATA}/Mold_brain/mold_lfc_cpm.csv",
        expr=f"{DATA}/Mold_brain/mold_gene_ctrl_expr.csv",
        npz=f"{DATA}/Mold_brain/mold_pseudobulk_raw.npz",
        npz_keys=("kos", "counts"),       # per (KO x cell type) — summed per KO
        # Shi's PERTURB-SEQ data lives in data/Mold_brain/, but its MEMORY GENES come
        # from Bandler: Mold brain is excluded from the figures (2026-09-11) and every
        # other panel pairs Shi with Bandler. Was MoldBrain / brain1_brain2 until
        # 2026-09-15 -- results computed before that used Mold's genes.
        memdir=f"{BENCH}/BandlerBrain",
        suffix="E13_E14",
        memory_source="Bandler",
    ),
    # No per-KO cell counts for these two, so control='matched' is unavailable —
    # control='random' is, and is what they are here for.
    "xu": dict(
        label="Xu (melanoma)", short="Xu",
        lfc=f"{DATA}/melanoma/perturbseq/melanoma_lfc_cpm_DMSO.csv",
        expr=f"{DATA}/melanoma/perturbseq/melanoma_gene_ctrl_expr_DMSO.csv",
        npz=None, npz_keys=None,
        memdir=f"{BENCH}/Melanoma_Harmange",
        suffix="E_M",
        memory_source="Harmange",
    ),
    "frangieh": dict(
        label="Frangieh (melanoma)", short="Frangieh",
        lfc=f"{DATA}/melanoma/perturbseq/frangieh_lfc_cpm_Control.csv",
        expr=f"{DATA}/melanoma/perturbseq/frangieh_gene_ctrl_expr_Control.csv",
        npz=None, npz_keys=None,
        memdir=f"{BENCH}/Melanoma_Harmange",
        suffix="E_M",
        memory_source="Harmange",
    ),
}


# ---------------------------------------------------------------- inputs --

def load_expr(path, floor):
    """Mean control expression per gene, and the set above `floor`."""
    s = pd.read_csv(path, index_col=0).squeeze("columns")
    s = s[~s.index.duplicated(keep="first")].astype(float)
    return s, set(s.index[s > floor])


def cells_per_ko(npz_path, keys):
    """Cells per KO, summed over whatever the rows are split by (e.g. cell type).

    Joined BY NAME downstream, never by position: Marson's cache carries 6,650
    targets while its LFC matrix has 6,646 rows, so a positional join would shift
    every count by an unpredictable amount.
    """
    z = np.load(npz_path, allow_pickle=True)
    name_key, count_key = keys
    names = pd.Index([str(x) for x in z[name_key]])
    counts = pd.Series(np.asarray(z[count_key], float), index=names)
    return counts.groupby(level=0).sum()


def load_abs_lfc(lfc_path, readout_genes, chunksize=400, verbose=True):
    """|log2FC| matrix, KOs x readout genes above the floor, as float32.

    Streams the CSV (these are 0.26-1.5 GB) but KEEPS the result: restricted to
    expressed readouts it is only ~90-370 MB, and holding it is what lets the same
    load answer "effect on all genes / on memory genes / on non-memory genes"
    without reading the file three times.

    NaN is preserved, which is what excludes each KO's own target from its score.
    """
    head = pd.read_csv(lfc_path, index_col=0, nrows=0)
    cols = [c for c in head.columns if c in readout_genes]
    if not cols:
        raise ValueError(f"no readout gene of {lfc_path} passed the floor")
    if verbose:
        print(f"  readout genes: {len(cols)} of {head.shape[1]} above the floor")
    parts, idx, seen_self = [], [], [0, 0]
    for chunk in pd.read_csv(lfc_path, index_col=0, chunksize=chunksize):
        sub = chunk[cols]
        for ko in sub.index:                      # self-KO must not score itself
            if ko in sub.columns:
                seen_self[0] += 1
                seen_self[1] += int(pd.isna(sub.at[ko, ko]))
        parts.append(np.abs(sub.to_numpy(dtype=np.float32)))
        idx.append(sub.index.to_numpy())
    M = pd.DataFrame(np.vstack(parts), index=pd.Index(np.concatenate(idx), name="ko"),
                     columns=cols)
    if seen_self[0]:
        frac = seen_self[1] / seen_self[0]
        if verbose:
            print(f"  self-KO masked: {seen_self[1]}/{seen_self[0]} ({frac:.1%})")
        if frac < 0.99:
            raise ValueError(
                f"{lfc_path}: only {frac:.1%} of self-KO entries are NaN — the "
                "target's own knockdown would count towards its own score")
    return M


def ko_stats(M, cols=None):
    """Median and mean |log2FC| per KO over `cols` (default: every column)."""
    a = M.to_numpy() if cols is None else M[list(cols)].to_numpy()
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(a, axis=1)
        mean = np.nanmean(a, axis=1)
    return pd.DataFrame(dict(median_abs_lfc=med, mean_abs_lfc=mean,
                             n_readout=np.isfinite(a).sum(axis=1)), index=M.index)


def _unused_per_ko_effect(lfc_path, readout_genes, chunksize=400, verbose=True):
    """Superseded by load_abs_lfc + ko_stats; kept only so the streaming variant
    is still on record if a matrix ever gets too big to hold."""
    head = pd.read_csv(lfc_path, index_col=0, nrows=0)
    cols = [c for c in head.columns if c in readout_genes]
    if not cols:
        raise ValueError(f"no readout gene of {lfc_path} passed the floor")
    if verbose:
        print(f"  readout genes: {len(cols)} of {head.shape[1]} above the floor")

    idx, med, mean, n_ok = [], [], [], []
    seen_self = [0, 0]                      # [checked, NaN]
    for chunk in pd.read_csv(lfc_path, index_col=0, chunksize=chunksize):
        sub = chunk[cols]
        for ko in sub.index:                # self-KO must not score itself
            if ko in sub.columns:
                seen_self[0] += 1
                seen_self[1] += int(pd.isna(sub.at[ko, ko]))
        a = np.abs(sub.to_numpy(dtype=float))
        with np.errstate(invalid="ignore"):
            med.append(np.nanmedian(a, axis=1))
            mean.append(np.nanmean(a, axis=1))
        n_ok.append(np.isfinite(a).sum(axis=1))
        idx.append(sub.index.to_numpy())

    out = pd.DataFrame(
        dict(median_abs_lfc=np.concatenate(med),
             mean_abs_lfc=np.concatenate(mean),
             n_readout=np.concatenate(n_ok)),
        index=pd.Index(np.concatenate(idx), name="ko"))
    if seen_self[0]:
        frac = seen_self[1] / seen_self[0]
        if verbose:
            print(f"  self-KO masked: {seen_self[1]}/{seen_self[0]} ({frac:.1%})")
        if frac < 0.99:
            raise ValueError(
                f"{lfc_path}: only {frac:.1%} of self-KO entries are NaN — the "
                "target's own knockdown would count towards its own score")
    return out


def memory_sets(memdir, files, top_n, universe):
    """{method: set of target genes it calls} plus the strict background."""
    sets, called = {}, set()
    for m, fn in files.items():
        if not os.path.exists(os.path.join(memdir, fn)):
            print(f"  [skip] {m}: {fn} not found")
            continue
        g = load_memory_gene_set(m, timepoint="merge", top_n=top_n,
                                 merge_mode="union", memgene_dir=memdir,
                                 method_files=files, restrict_to=universe)
        sets[m] = set(g)
        called |= set(g)
    return sets, called


# ---------------------------------------------------------------- compare --

def compare(eff, cov, mem, bg, metric, k, n_iter, seed):
    """Unmatched and matched comparison of memory KOs against background KOs."""
    mem = [g for g in mem if g in eff.index]
    bg = [g for g in bg if g in eff.index and g not in set(mem)]
    if len(mem) < 10 or len(bg) < len(mem):
        return None

    a = eff.loc[mem, metric].to_numpy(float)
    b = eff.loc[bg, metric].to_numpy(float)
    d_raw = cliffs_delta(a, b)
    lo_raw, hi_raw = cliffs_ci(a, b)

    r = matched_comparison(eff, cov, mem, bg, match_on=["expr", "cells"],
                           metric=metric, n_iter=n_iter, k=k, random_seed=seed)
    bal = r["balance"]
    return dict(
        n_memory=len(mem), n_background=len(bg),
        median_memory=float(np.median(a)), median_background_unmatched=float(np.median(b)),
        median_background_matched=float(np.median(r["non_scg_values"])),
        cliffs_d_unmatched=d_raw, d_lo_unmatched=lo_raw, d_hi_unmatched=hi_raw,
        cliffs_d_matched=r["cliffs_d"], d_lo_matched=r["d_lo"], d_hi_matched=r["d_hi"],
        p_matched=r["mann_whitney_p"],
        smd_expr_before=float(bal.loc["expr", "smd_before"]),
        smd_expr_after=float(bal.loc["expr", "smd_after"]),
        smd_cells_before=float(bal.loc["cells", "smd_before"]),
        smd_cells_after=float(bal.loc["cells", "smd_after"]),
    )


_EFF_CACHE = {}


def load_screen(key, floor=0.1, metric="median_abs_lfc", cache=True, verbose=True):
    """Per-KO effect + covariates for one screen. Streams the LFC matrix ONCE.

    Cached in-process, because the matrices are ~1.5 GB and a notebook will ask
    for several top-N cutoffs off the same table.
    """
    ck = (key, floor)
    if cache and ck in _EFF_CACHE:
        return _EFF_CACHE[ck]          # (covariate table, |LFC| matrix)
    cfg = SCREENS[key]
    if verbose:
        print(f"\n=== {cfg['label']} — memory genes from {cfg['memory_source']} ===")
    expr, expressed = load_expr(cfg["expr"], floor)
    M = load_abs_lfc(cfg["lfc"], expressed, verbose=verbose)
    eff = ko_stats(M)
    has_cells = cfg.get("npz") is not None
    cells = (cells_per_ko(cfg["npz"], cfg["npz_keys"]) if has_cells
             else pd.Series(dtype=float))

    # target floor: the KO's own target must itself be measured above the floor,
    # or "is this target a memory gene" cannot be answered
    eff = eff[eff.index.isin(expressed)]
    if has_cells:
        eff = eff[eff.index.isin(cells.index)]      # join by NAME
    eff["expr"] = expr.reindex(eff.index).to_numpy()
    eff["cells"] = (cells.reindex(eff.index).to_numpy() if has_cells else np.nan)
    eff = eff.dropna(subset=["expr", metric] + (["cells"] if has_cells else []))
    if verbose:
        print(f"  KO universe: {len(eff)} targets measured above {floor} CPM")
        if has_cells:
            rho = eff[metric].corr(eff["cells"], method="spearman")
            print(f"  depth confound: Spearman({metric}, cells per KO) = {rho:+.2f}")
        else:
            print("  no per-KO cell counts — control='matched' unavailable here")
    M = M.loc[eff.index]                 # same KOs, same order
    if cache:
        _EFF_CACHE[ck] = (eff, M)
    return eff, M


def ko_axis_results(key, top_n, floor=0.1, metric="median_abs_lfc", k=10,
                    n_iter=200, seed=0, eff=None, M=None, equalise=True,
                    readout="all", control="matched", bg_mode="strict",
                    tiers=None, gene_sets=None, exclude=None,
                    min_memory=10, verbose=False):
    """{method: result dict} — hands straight to figures.fig_responsiveness_violin().

    readout : which genes the per-KO score is taken over.
        'all'         every gene above the floor (the transcriptome-wide response)
        'memory'      only that method's own memory genes
        'non_memory'  only genes NO method called
      The KO's own target is excluded throughout (NaN on the diagonal), so a memory
      KO scored against memory readouts is never credited for its own knockdown.

    bg_mode : which KOs may serve as background.
        'strict'  targets called by NO method (cleanest, but on a small screen the
                  union of five methods can swallow nearly every perturbed gene)
        'own'     targets THIS method did not call — a larger pool, at the cost of
                  a background that may contain another method's memory genes
    tiers : instead of one entry per method, use CONSENSUS sets — e.g. (2, 3, 4)
        gives ">=2", ">=3", ">=4" methods agreeing. The background is always the
        strict one here (called by no method), since "this set's own complement"
        is not well defined for a consensus.
    gene_sets : {label: iterable of genes} to use INSTEAD of the method tables —
        e.g. a Power-Seek run on unperturbed control cells, or the cross-dataset
        consensus. The background is then everything outside the union of the
        supplied sets (plus `exclude`, for genes that should not serve as controls
        even though no supplied set names them).
    min_memory : skip a set with fewer than this many memory KOs.
    control : how the comparison KOs are chosen.
        'matched'  nearest neighbours on target expression AND cells per KO
        'random'   an equal number drawn at random from the non-memory KOs, with
                   NO matching. Simpler to describe, but per-KO |LFC| tracks cells
                   per KO at Spearman -0.63, so a difference here can be depth
                   rather than biology. Use it as a contrast to 'matched', not
                   instead of it.
    """
    cfg = SCREENS[key]
    if eff is None or M is None:
        eff, M = load_screen(key, floor, metric, verbose=verbose)
    if gene_sets is not None:
        sets = {k: set(v) & set(eff.index) for k, v in gene_sets.items()}
        called = set().union(*sets.values()) | set(exclude or ())
    else:
        sets, called = memory_sets(cfg["memdir"], method_files(cfg["suffix"]),
                                   top_n, eff.index)
    if tiers:
        from collections import Counter
        cnt = Counter(g for s_ in sets.values() for g in s_)
        sets = {f"\u2265{k}": {g for g, c in cnt.items() if c >= k} for k in tiers}
        bg_mode = "strict"        # a consensus set has no meaningful "own" complement
    rng = np.random.default_rng(seed)
    out = {}
    for m, genes in sets.items():
        mem = [g for g in sorted(genes) if g in eff.index]
        bg_pool = ([g for g in eff.index if g not in called] if bg_mode == "strict"
                   else [g for g in eff.index if g not in set(genes)])
        if len(mem) < min_memory or len(bg_pool) < len(mem):
            if verbose:
                print(f"  [skip] {m}: {len(mem)} memory KOs, "
                      f"{len(bg_pool)} background KOs")
            continue

        # readout columns can depend on the method, so the per-KO score is
        # recomputed here rather than taken from the cached all-gene table
        if readout == "memory":
            cols = [g for g in genes if g in M.columns]
        elif readout in ("non_memory", "non-memory"):
            cols = [g for g in M.columns if g not in called]
        else:
            cols = None
        if cols is not None and len(cols) < 20:
            print(f"  [skip] {m}: only {len(cols)} readout genes for readout={readout}")
            continue
        val = ko_stats(M, cols)[metric]
        tab = eff.assign(**{metric: val.reindex(eff.index).to_numpy()}).dropna(
            subset=[metric])
        mem = [g for g in mem if g in tab.index]
        bg = [g for g in bg_pool if g in tab.index]

        if control == "matched" and not np.isfinite(tab["cells"]).any():
            raise ValueError(
                f"{key}: no per-KO cell counts, so the depth confound cannot be "
                "matched. Use control='random' (and say so in the caption).")
        if control == "random":
            draws = [rng.choice(bg, size=len(mem), replace=False)
                     for _ in range(n_iter)]
            bgv = np.sort(np.vstack([tab.loc[list(d), metric].to_numpy()
                                     for d in draws]), axis=1).mean(0)
            from scipy.stats import mannwhitneyu
            memv = tab.loc[mem, metric].to_numpy()
            ps = [mannwhitneyu(memv, tab.loc[list(d), metric].to_numpy(),
                               alternative="two-sided")[1] for d in draws]
            d = cliffs_delta(memv, bgv)
            lo, hi = cliffs_ci(memv, bgv)
            r = dict(scg_values=memv, non_scg_values=bgv, n_scg=len(mem),
                     n_bg=len(bg), mann_whitney_p=float(np.median(ps)),
                     cliffs_d=d, d_lo=lo, d_hi=hi)
        else:
            r = matched_comparison(tab, tab[["expr", "cells"]], mem, bg,
                                   match_on=["expr", "cells"], metric=metric,
                                   n_iter=n_iter, k=k, random_seed=seed)
            if equalise:
                v = np.asarray(r["non_scg_values"], float)
                n = int(r["n_scg"])
                if n and v.size % n == 0:
                    r["non_scg_values"] = np.sort(v.reshape(-1, n), axis=1).mean(0)
        r["n_readout_genes"] = M.shape[1] if cols is None else len(cols)
        out[m] = r
    return out


def run_screen(key, top_ns, floor, metric, k, n_iter, seed, verbose=True):
    cfg = SCREENS[key]
    eff, _M = load_screen(key, floor, metric, verbose=verbose)
    files = method_files(cfg["suffix"])
    rows = []
    for top_n in top_ns:
        sets, called = memory_sets(cfg["memdir"], files, top_n, eff.index)
        bg = [g for g in eff.index if g not in called]      # strict background
        if verbose:
            print(f"  top_n={top_n}: strict background = {len(bg)} KOs")
        for m, genes in sets.items():
            res = compare(eff, eff[["expr", "cells"]], sorted(genes), bg,
                          metric, k, n_iter, seed)
            if res is None:
                print(f"    {m:11s} skipped (too few KOs)")
                continue
            rows.append(dict(screen=key, screen_label=cfg["label"],
                             memory_source=cfg["memory_source"], method=m,
                             top_n=top_n, metric=metric, **res))
            if verbose:
                print(f"    {m:11s} n={res['n_memory']:4d}  "
                      f"delta unmatched {res['cliffs_d_unmatched']:+.3f}  "
                      f"-> matched {res['cliffs_d_matched']:+.3f} "
                      f"[{res['d_lo_matched']:+.3f},{res['d_hi_matched']:+.3f}]  "
                      f"SMD expr {res['smd_expr_after']:+.3f} "
                      f"cells {res['smd_cells_after']:+.3f}")
    return pd.DataFrame(rows), eff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=["all", *SCREENS], nargs="+")
    ap.add_argument("--top-n", type=int, nargs="+", default=[100])
    ap.add_argument("--floor", type=float, default=0.1,
                    help="mean control CPM floor, for readouts AND targets")
    ap.add_argument("--metric", default="median_abs_lfc",
                    choices=["median_abs_lfc", "mean_abs_lfc"])
    ap.add_argument("-k", type=int, default=10, help="nearest neighbours to match in")
    ap.add_argument("--n-iter", type=int, default=200, help="matched draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(MARSON_DIR,
                                                  "ko_axis_memory_results.csv"))
    ap.add_argument("--per-ko-out", default=None,
                    help="optional CSV of the per-KO table (effect + covariates)")
    a = ap.parse_args()

    keys = list(SCREENS) if "all" in a.dataset else a.dataset
    out, per_ko = [], {}
    for key in keys:
        df, eff = run_screen(key, a.top_n, a.floor, a.metric, a.k, a.n_iter, a.seed)
        out.append(df)
        per_ko[key] = eff
    res = pd.concat(out, ignore_index=True)
    res.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(res)} rows)")

    if a.per_ko_out:
        pd.concat({k: v for k, v in per_ko.items()}, names=["screen"]).to_csv(a.per_ko_out)
        print(f"wrote {a.per_ko_out}")

    show = ["screen", "method", "top_n", "n_memory", "cliffs_d_unmatched",
            "cliffs_d_matched", "d_lo_matched", "d_hi_matched", "smd_cells_after"]
    print("\n" + res[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
