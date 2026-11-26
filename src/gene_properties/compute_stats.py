#!/usr/bin/env python
"""Are memory genes different from expression-matched non-memory genes in any of
34 gene features? One row per dataset x feature -> matched_with_ci.csv.

Every feature ever tested is in this one run, grouped into three families (CpG and
promoter, annotation and constraint, gene architecture), and q is Benjamini-Hochberg
over ALL features within a dataset, not within a family.

Memory genes and background come from feature_data.load(), i.e. from the figure
code: >= 2 of 4 methods at top 300 against the genes in none of the four
methods' top 300.

For each feature:
  smd_raw  naive standardised mean difference against the whole background
  est      the same after matching each memory gene to its 10 nearest background
           genes in log expression (expression predicts most of these features, so
           the unmatched difference is mostly an expression difference)
  lo, hi   bootstrap 95% CI of est
  pts      for the binary CpG-island call, the matched difference in percentage
           points, which is the readable unit ("N points fewer")
  p, z     permutation test with the memory label shuffled within 50 expression
           strata; q is Benjamini-Hochberg within each dataset

LOEUF, pLI and missense z come from gnomAD and exist for human genes only, so the
two mouse datasets (LARRY, Bandler) have no rows for them. The 879 TF-motif tests
are a separate table (different matching covariates).
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import robustness as RB                                               # noqa: E402
import feature_data as FD                                             # noqa: E402

# (column, label, kind, family). kind "pct" = a binary call, also reported in
# percentage points. Every feature that was ever tested is here, in one run, so the
# multiple-testing correction covers all of them.
CPG, ANN, ARCH = "CpG and promoter", "annotation and constraint", "gene architecture"
FEATS = [
    ("cgi_prom_500", "CpG island promoter (+/-500 bp)", "pct", CPG),
    ("cgi_prom_1000", "CpG island promoter (+/-1 kb)", "pct", CPG),
    ("cgi_prom_2000", "CpG island promoter (+/-2 kb)", "pct", CPG),
    ("cgi_obsExp", "island CpG obs/exp", "mean", CPG),
    ("cgi_perCpg", "island CpG percent", "mean", CPG),
    ("cgi_perGc", "island GC percent", "mean", CPG),
    ("log_cgi_dist", "distance to CpG island", "mean", CPG),
    ("prom500_cpg_oe", "promoter CpG obs/exp (+/-500 bp)", "mean", CPG),
    ("prom1000_cpg_oe", "promoter CpG obs/exp (+/-1 kb)", "mean", CPG),
    ("prom500_cpg_density", "promoter CpG density (+/-500 bp)", "mean", CPG),
    ("prom500_gc", "promoter GC (+/-500 bp)", "mean", CPG),
    ("prom1000_gc", "promoter GC (+/-1 kb)", "mean", CPG),
    ("log_pubs", "publications", "mean", ANN),
    ("log_go", "gene ontology terms", "mean", ANN),
    ("log_go_bp", "GO biological process terms", "mean", ANN),
    ("log_go_mf", "GO molecular function terms", "mean", ANN),
    ("log_go_cc", "GO cellular component terms", "mean", ANN),
    ("log_go_noniea", "GO terms, non-electronic", "mean", ANN),
    ("log_deg400", "protein partners (STRING score 400)", "mean", ANN),
    ("log_deg700", "protein partners (STRING score 700)", "mean", ANN),
    ("n_domains", "protein domains", "mean", ANN),
    ("LOEUF", "LOEUF", "mean", ANN),
    ("pLI", "pLI", "mean", ANN),
    ("mis_z", "missense z", "mean", ANN),
    ("log_len", "gene length (exonic)", "mean", ARCH),
    ("n_exons", "exon count", "mean", ARCH),
    ("n_introns", "intron count", "mean", ARCH),
    ("log_intron_len", "total intron length", "mean", ARCH),
    ("log_gene_span", "gene span", "mean", ARCH),
    ("log_tx_span", "transcript span", "mean", ARCH),
    ("log_isoforms", "isoform count", "mean", ARCH),
    ("cons_frac_gene", "conservation (gene body)", "mean", ARCH),
    ("cons_frac_promoter", "conservation (promoter)", "mean", ARCH),
    ("is_protein_coding", "protein coding", "pct", ARCH),
]
N = 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=FD.FEATURES)
    ap.add_argument("--out-dir", default=FD.OUT_DIR)
    a = ap.parse_args()

    M = FD.load(features_dir=a.features_dir)
    rows = []
    for d in FD.ORDER:
        s = M[M.dataset == d]
        for col, name, kind, family in FEATS:
            t = s.dropna(subset=["log_expr", col])
            fg, bg = t[t.set == "memory"], t[t.set == "background"]
            if len(fg) < 20 or len(bg) < 50:
                continue
            fv, fc = fg[col].to_numpy(float), fg[["log_expr"]].to_numpy(float)
            bv, bc = bg[col].to_numpy(float), bg[["log_expr"]].to_numpy(float)
            est = RB.matched_effect(fv, fc, bv, bc, kind="mean")
            lo, hi = RB.bootstrap_ci(fv, fc, bv, bc, kind="mean")
            pooled = np.sqrt((np.nanvar(fv, ddof=1) + np.nanvar(bv, ddof=1)) / 2)
            raw = (np.nanmean(fv) - np.nanmean(bv)) / pooled if pooled > 0 else np.nan
            pts = RB.matched_effect(fv, fc, bv, bc, kind="pct") if kind == "pct" else np.nan
            vals = np.concatenate([fv, bv])
            mem = np.concatenate([np.ones(len(fv), bool), np.zeros(len(bv), bool)])
            expr = np.concatenate([fc.ravel(), bc.ravel()])
            p, p_emp, z = RB.perm_test(vals, mem, expr, n=N)
            rows.append(dict(dataset=d, paper_name=FD.PAPER_NAME[d], family=family,
                             feature=col, label=name, kind=kind, smd_raw=raw, est=est, lo=lo, hi=hi,
                             pts=pts, p=p, p_emp=p_emp, z=z,
                             n_memory=len(fg), n_background=len(bg)))
            print(f"  {d:10s} {name:30s} raw {raw:+6.2f} -> {est:+6.2f} "
                  f"[{lo:+6.2f},{hi:+6.2f}] z={z:+6.1f} p={p:.1e}", flush=True)
    R = pd.DataFrame(rows)
    R["q"] = np.concatenate([RB.bh(g.p.values) for _, g in R.groupby("dataset", sort=False)])
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "matched_with_ci.csv")
    R.to_csv(out, index=False)
    print(f"\nwrote {out}  {R.shape}", flush=True)


if __name__ == "__main__":
    main()
