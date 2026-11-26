#!/usr/bin/env python
"""CpG depletion of memory genes against an expression-matched null.

observed = mean(memory) - mean(background) in RAW units, against a null of the same
statistic with the memory label shuffled WITHIN 50 expression strata, so every null
draw has the memory genes' expression profile. The null is therefore not centred on
zero: expression alone predicts CpG-island promoters, and the null centre is the part
of the difference expression explains.

Two measures:
  cpg_oe   promoter CpG observed/expected, +/-500 bp of the TSS (ratio, ~0.1-0.9)
  cgi_pct  % of genes whose promoter (+/-1 kb) overlaps a CpG island (percentage points)

Memory genes and background come from feature_data.load(): >= 2 of 4 methods at top
300, against the genes in none of the four methods' top 300.
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy import stats                                               # noqa: E402
import feature_data as FD                                             # noqa: E402

MEASURES = {"cpg_oe": ("prom500_cpg_oe", 1.0), "cgi_pct": ("cgi_raw", 100.0)}
N_PERM, N_STRATA = 2000, 50


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=FD.FEATURES)
    ap.add_argument("--out-dir", default=FD.OUT_DIR)
    a = ap.parse_args()

    M = FD.load(features_dir=a.features_dir)
    parts = []
    for sp in ("human", "mouse"):
        # promoter CpG o/e is already joined by load(); only the raw CGI call is re-read.
        # The CpG-island call is re-read RAW: load() turns a MISSING promoter
        # annotation into 0 = "no island", which would count unannotated genes as
        # island-less. Kept as NaN they drop out, as they do for cpg_oe.
        C = (pd.read_csv(f"{a.features_dir}/feature_tierC_promoter_{sp}.csv",
                         usecols=["gene_upper", "cgi_prom_1000"])
               .drop_duplicates("gene_upper").set_index("gene_upper"))
        C["cgi_raw"] = C.cgi_prom_1000.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        m = M[M.species == sp].set_index("gene_upper").drop(columns=["cgi_prom_1000"])
        parts.append(m.join(C[["cgi_raw"]], how="left").reset_index())
    M = pd.concat(parts, ignore_index=True)

    summ, nulls = [], []
    for key in FD.ORDER:
        label = FD.PAPER_NAME[key]
        s0 = M[(M.dataset == key) & M.set.isin(["memory", "background"])]
        for mname, (col, scale) in MEASURES.items():
            s = s0.dropna(subset=["log_expr", col])
            v = s[col].to_numpy(float) * scale
            mem = s.set.eq("memory").to_numpy()
            q = pd.qcut(s.log_expr, N_STRATA, labels=False, duplicates="drop").to_numpy()
            bins = [np.where(q == b)[0] for b in np.unique(q)]
            obs = v[mem].mean() - v[~mem].mean()
            rng = np.random.default_rng(0)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                p = mem.copy()
                for j in bins:
                    p[j] = rng.permutation(mem[j])
                null[i] = v[p].mean() - v[~p].mean()
            z = (obs - null.mean()) / null.std(ddof=1)
            pv = 2 * stats.norm.sf(abs(z))
            summ.append(dict(dataset=label, measure=mname, n_memory=int(mem.sum()),
                             n_nonmemory=int((~mem).sum()),
                             memory_mean=v[mem].mean(), nonmemory_mean=v[~mem].mean(),
                             observed_diff=obs, null_median=float(np.median(null)),
                             null_sd=float(null.std(ddof=1)),
                             null_2p5=float(np.percentile(null, 2.5)),
                             null_97p5=float(np.percentile(null, 97.5)),
                             z=z, p=pv, stars=stars(pv)))
            nulls.append(pd.DataFrame({"dataset": label, "measure": mname, "null_diff": null}))

    os.makedirs(a.out_dir, exist_ok=True)
    S, N = pd.DataFrame(summ), pd.concat(nulls, ignore_index=True)
    S.to_csv(f"{a.out_dir}/cpg_null_panel_summary.csv", index=False)
    N.to_csv(f"{a.out_dir}/cpg_null_panel_nulldraws.csv.gz", index=False, compression="gzip")
    pd.set_option("display.width", 200)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
