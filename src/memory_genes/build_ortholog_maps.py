#!/usr/bin/env python
"""Rebuild mouse2human_orthologs.csv for every mouse lineage dataset.

WHY THIS EXISTS. The previous maps came from mygene/HomoloGene, which is frozen and
incomplete: 34% of Weinreb's measured genes had no human partner, including obvious
1:1 orthologs (Hmga2, Ptma, Srgn, Tmsb4x, Cd52, Gzmb). Those genes silently dropped
out of every cross-dataset comparison.

SOURCES (all local, no API calls):
  gene_orthologs.gz        NCBI's curated ortholog pairs, by GeneID (human <-> mouse)
  Mus_musculus.gene_info   mouse GeneID -> symbol, plus synonyms for old var names
  Homo_sapiens.gene_info   human GeneID -> symbol, plus synonyms to CURRENT symbols

RULES
  1. NCBI pair first; a dataset gene not in NCBI falls back to the OLD map, so the
     ~900 Riken-style names HomoloGene resolved are not lost (source column says which).
  2. Mouse symbols are resolved through mouse synonyms, so a dataset using an older
     annotation still matches.
  3. Human symbols are written as the CURRENT NCBI symbol (CYR61 -> CCN1), because the
     human datasets are normalised the same way in figures.to_human().
  4. Every measured gene of the dataset gets a row; `human` is empty when unmapped.

Run:  python build_ortholog_maps.py            (add --dry-run to only report)
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import argparse
import os
from collections import defaultdict

import anndata as ad
import pandas as pd

RAW = f"{P.DATA}/gene_features/raw"
BENCH = os.path.dirname(os.path.abspath(__file__))
TREE = "python_implements"
# dataset folder -> (h5ad, var column holding the symbol or None for the index)
DATASETS = {
    "LARRY":       (f"{P.DATA}/datasets/LARRY_processed.h5ad", None),
    "BandlerBrain": (f"{P.DATA}/bandler_brain/bandler_brain.h5ad", None),
    "MoldBrain":   (f"{P.DATA}/Mold_brain/mold_brain.h5ad", "var_names"),
}


def load_tables():
    orth = pd.read_csv(f"{RAW}/gene_orthologs.gz", sep="\t")
    orth.columns = [c.lstrip("#") for c in orth.columns]
    hm = orth[(orth.tax_id == 9606) & (orth.relationship == "Ortholog")
              & (orth.Other_tax_id == 10090)]
    hi = pd.read_csv(f"{RAW}/Homo_sapiens.gene_info.gz", sep="\t",
                     usecols=["GeneID", "Symbol", "Synonyms"])
    mi = pd.read_csv(f"{RAW}/Mus_musculus.gene_info.gz", sep="\t",
                     usecols=["GeneID", "Symbol", "Synonyms"])
    h_sym, m_sym = dict(zip(hi.GeneID, hi.Symbol)), dict(zip(mi.GeneID, mi.Symbol))
    m2h = {}
    for hg, mg in zip(hm.GeneID, hm.Other_GeneID):
        m, h = m_sym.get(mg), h_sym.get(hg)
        if m and h:
            m2h.setdefault(m, h)
    m_syn = {}
    for s, syn in zip(mi.Symbol, mi.Synonyms):
        if isinstance(syn, str) and syn != "-":
            for a in syn.split("|"):
                m_syn.setdefault(a, s)
    h_cur = set(hi.Symbol)
    h_syn = defaultdict(set)
    for s, syn in zip(hi.Symbol, hi.Synonyms):
        if isinstance(syn, str) and syn != "-":
            for a in syn.split("|"):
                h_syn[a].add(s)
    def to_current(g):
        """old human symbol -> current one, only when the synonym is unambiguous"""
        if g in h_cur:
            return g
        hits = h_syn.get(g, set())
        return next(iter(hits)) if len(hits) == 1 else g
    return m2h, m_syn, to_current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m2h, m_syn, to_current = load_tables()
    print(f"NCBI mouse->human pairs: {len(m2h):,}")
    for folder, (h5ad, var_col) in DATASETS.items():
        out_path = f"{BENCH}/{TREE}/{folder}/mouse2human_orthologs.csv"
        A = ad.read_h5ad(h5ad, backed="r")
        genes = (A.var[var_col].astype(str).tolist() if var_col
                 else [str(g) for g in A.var_names])
        A.file.close()
        genes = list(dict.fromkeys(genes))
        old = {}
        if os.path.exists(out_path):
            o = pd.read_csv(out_path).dropna(subset=["human"])
            old = dict(zip(o["mouse"].astype(str), o["human"].astype(str)))
        rows, n_ncbi, n_old = [], 0, 0
        for g in genes:
            h = m2h.get(g) or m2h.get(m_syn.get(g, ""), None)
            src = "NCBI"
            if not h and g in old:
                h, src = old[g], "homologene"
            if not h:
                rows.append(dict(mouse=g, human="", source=""))
                continue
            rows.append(dict(mouse=g, human=to_current(h), source=src))
            n_ncbi += src == "NCBI"
            n_old += src == "homologene"
        df = pd.DataFrame(rows)
        n_map = int((df.human != "").sum())
        print(f"  {folder:13s} {n_map:6,}/{len(genes):6,} mapped ({100*n_map/len(genes):.0f}%)"
              f"  [NCBI {n_ncbi:,}, homologene fallback {n_old:,}]"
              + ("   (dry run, not written)" if args.dry_run else ""))
        if not args.dry_run:
            df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
