#!/usr/bin/env python
"""The consensus memory genes, as one long CSV, built exactly as the figures do.

This is the single definition of "consensus memory genes" in the project: it is
what Fig 1 and Fig 2 plot, what the GO enrichment runs on, and what the
supplementary gene-list tables contain.

WHY IT REUSES THE FIGURE CODE. figures.py does not read any gene-list file; it
rebuilds the consensus in memory from the per-method lists every time. So any
separate implementation here would be a second definition that could drift from
the one the paper reports. Instead this script calls the figure functions
themselves:

- files, suffix and SCG variant: figures._resolve_dir / _resolve_suffix /
  scg_variant_for, METHOD_PREFIX with METHOD_EXCLUDE already applied (no Mold)
- per-method gene list: marson_KO.load_memory_gene_set, the function behind
  figures.memory_set (Fig 2). A single group = that group's top N; 'merged' =
  each gene's best within-group rank, THEN the top N (timepoint="merge").
  SCG keeps only is_biased genes.

No depth cap to the shallowest method (Fig 2 has none either), so a method with
fewer called genes than top_n contributes all it has; `n_<method>` records it.
No restriction to a perturb-seq universe either: Fig 2 restricts to the genes
measured in Zhu before taking the top N, so its Michaels sets can differ slightly.

Output: $PERM_MEMGENES_ROOT/gene_lists/figure_consensus_top<N>.csv, one row per
(dataset, group, gene) with n_methods >= 2. go_enrichment.py reads the
group == "merged" rows; the per-group rows are the supplementary view.
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

import pandas as pd

sys.path.insert(0, P.FIGURES)
import figures as F                                    # noqa: E402

F._perturb_path()
from marson_KO import load_memory_gene_set             # noqa: E402

# figures.py key -> (paper name, paired perturb-seq screens)
PAPER = {"T-lineage": ("Michaels", "Zhu;Weinstock"),
         "Harmange":  ("Harmange", "Xu;Frangieh"),
         "LARRY":     ("Weinreb", "Bowness"),
         "Bandler":   ("Bandler", "Shi")}


def method_files(ds):
    """{method: filename} and the directory, resolved the way load_tables does."""
    folder = F.DATASETS.get(ds, {}).get("folder", ds)
    d = F._resolve_dir(folder, F.RESULTS_TREE, verbose=False)
    suf = F._resolve_suffix(d)
    v = F.scg_variant_for(ds, None)
    files = {m: f"{p}_memory_genes_{v if m == 'ClonePair' else ''}{suf}.csv"
             for m, p in F.METHOD_PREFIX.items()}
    missing = [f for f in files.values() if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise SystemExit(f"[{ds}] missing method files in {d}: {missing}")
    return d, files


def check_gene_column(d, files):
    """load_memory_gene_set reads `human_symbol` when a file has one. Fine only if
    it equals `gene`; otherwise one method would be in another namespace."""
    for m, f in files.items():
        df = pd.read_csv(os.path.join(d, f))
        if "human_symbol" in df.columns:
            if "is_biased" in df.columns:
                df = df[df["is_biased"] == True]           # noqa: E712
            bad = (df["human_symbol"].astype(str) != df["gene"].astype(str)).sum()
            if bad:
                raise SystemExit(f"[{d}] {f}: {bad} rows where human_symbol != gene")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--out-dir", default=os.path.join(P.MEMGENES_ROOT, "gene_lists"))
    args = ap.parse_args()

    methods = list(F.METHOD_PREFIX)
    rows = []
    for ds in F.active_datasets(None):
        d, files = method_files(ds)
        check_gene_column(d, files)
        tps = sorted(set.intersection(*(
            set(pd.read_csv(os.path.join(d, f), usecols=["timepoint"]).timepoint)
            for f in files.values())))
        for tp in tps + ["merge"]:
            lists = {m: load_memory_gene_set(m, timepoint=tp, top_n=args.top_n,
                                             memgene_dir=d, method_files=files)
                     for m in methods}
            pos = {m: {g: i + 1 for i, g in enumerate(lst)} for m, lst in lists.items()}
            genes = sorted({g for lst in lists.values() for g in lst})
            group = "merged" if tp == "merge" else str(tp)
            depth = {f"n_{m}": len(lists[m]) for m in methods}
            n_kept = 0
            for g in genes:
                called = [m for m in methods if g in pos[m]]
                if len(called) < 2:
                    continue
                n_kept += 1
                rows.append(dict(
                    dataset=PAPER[ds][0], paired_screens=PAPER[ds][1],
                    group=group, top_n=args.top_n, gene=g,
                    n_methods=len(called), methods=";".join(called),
                    consensus_ge2=True, consensus_ge3=len(called) >= 3,
                    consensus_all4=len(called) == len(methods),
                    **{f"rank_{m}": pos[m].get(g) for m in methods}, **depth))
            print(f"  {PAPER[ds][0]:9s} {group:7s} {depth}  >=2: {n_kept}")

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "group", "n_methods", "gene"], ascending=[True, True, False, True])
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"figure_consensus_top{args.top_n}.csv")
    out.to_csv(fp, index=False)
    print(f"\n{fp}: {len(out)} rows")
    print(out.groupby(["dataset", "group"])[["consensus_ge2", "consensus_ge3",
                                             "consensus_all4"]].sum().to_string())


if __name__ == "__main__":
    main()
