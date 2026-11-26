#!/usr/bin/env python
"""Raw-count pseudobulk per perturbation label for the double-perturbation screens.

One row per label (control, every single, every double), summed RAW counts over all
cells carrying it -- the cached input for double_lib.compute_lfc (CPM + pc 1, the
standard pipeline). Controls are POOLED across batches: in ProPer-seq every label is
spread evenly over the 12 batches (chi2 label x batch p = 0.34, checked 2026-10-14),
and Norman's 8 gemgroups each carry ~1.3-1.6k controls.

Labels are canonicalised to 'control', 'A' or 'A|B' with A < B alphabetically, so the
two datasets share one pair vocabulary (ProPer-seq writes 'A|B', Norman 'A_B').

Outputs, per dataset, in $PERM_DATA/double_perturbs/<ds>/:
  <ds>_pseudobulk_raw.npz      labels, genes, counts (labels x genes), sizes (cells/label)
  <ds>_gene_ctrl_expr.csv      mean control counts/cell (the 0.1 floor is on THIS scale)
  <ds>_labels.csv              label, n_targets, targets, n_cells

X must be raw counts; the reader is chunked and handles CSR (ProPer-seq) and CSC
(Norman, gene-major) without loading the full matrix.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import os, sys, time, argparse
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp

OUT = f"{P.DATA}/double_perturbs"
D = P.RAW
DATASETS = {
    "properseq": dict(h5ad=f"{D}/properseq/data/proper_filtered.h5ad",
                      col="perturbation", control="negative", sep="|"),
    "norman":    dict(h5ad=f"{D}/norman/NormanWeissman2019_filtered.h5ad",
                      col="perturbation", control="control", sep="_"),
    # Replogle 2020 direct-capture CRISPRi, K562. Labels come from gene_A/gene_B, NOT
    # `perturbation`: a '_2' suffix is a second sgRNA for the SAME gene (FDPS_2 + CTRL is
    # an FDPS single, although the file says n_genes_targeted == 2), and 'CTRL' is the
    # non-targeting slot. Guide variants are merged by gene, so FDPS / FDPS_2 cells pool
    # and the four FDPS x HUS1 guide combinations become one pair.
    "gse146194": dict(h5ad=f"{D}/GSE146194_exp6.h5ad", gene_cols=("gene_A", "gene_B"),
                      control_token="CTRL", guide_suffix="_2"),
}


def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


def obs_labels(f, col):
    node = f["obs"][col]
    if isinstance(node, h5py.Group):
        cats = np.array([_dec(c) for c in node["categories"][:]], dtype=object)
        return cats[node["codes"][:]]
    return np.array([_dec(v) for v in node[:]], dtype=object)


def canonical(label, control, sep):
    if label == control:
        return "control"
    return "|".join(sorted(label.split(sep)))


def pseudobulk(path, codes, n_labels, chunk=5000):
    """labels x genes summed counts; codes = label index per cell."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        enc = _dec(X.attrs.get("encoding-type", ""))
        n_cells, n_genes = (int(v) for v in X.attrs["shape"])
        vg = f["var"]
        genes = np.array([_dec(g) for g in vg[_dec(vg.attrs.get("_index", "_index"))][:]])
        ind = sp.csr_matrix((np.ones(n_cells), (codes, np.arange(n_cells))), shape=(n_labels, n_cells))
        indptr = X["indptr"][:]
        out = np.zeros((n_labels, n_genes))
        if "csr" in enc:
            for s in range(0, n_cells, chunk):
                e = min(s + chunk, n_cells); a, b = int(indptr[s]), int(indptr[e])
                blk = sp.csr_matrix((X["data"][a:b].astype(np.float64), X["indices"][a:b],
                                     indptr[s:e + 1] - a), shape=(e - s, n_genes))
                out += np.asarray((ind[:, s:e] @ blk).todense())
                print(f"    cells {e:,}/{n_cells:,}", flush=True)
        elif "csc" in enc:
            for s in range(0, n_genes, chunk):
                e = min(s + chunk, n_genes); a, b = int(indptr[s]), int(indptr[e])
                blk = sp.csc_matrix((X["data"][a:b].astype(np.float64), X["indices"][a:b],
                                     indptr[s:e + 1] - a), shape=(n_cells, e - s))
                out[:, s:e] = np.asarray((ind @ blk).todense())
                print(f"    genes {e:,}/{n_genes:,}", flush=True)
        else:
            raise SystemExit(f"unsupported X encoding '{enc}'")
    return out, genes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]
    out_dir = f"{OUT}/{a.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    t = time.time()
    with h5py.File(cfg["h5ad"], "r") as f:
        if "gene_cols" in cfg:
            ga, gb = (obs_labels(f, c) for c in cfg["gene_cols"])
        else:
            raw = obs_labels(f, cfg["col"])
    if "gene_cols" in cfg:
        sfx, ctl = cfg["guide_suffix"], cfg["control_token"]
        strip = lambda g: g[: -len(sfx)] if g.endswith(sfx) else g
        lab = np.array(["|".join(sorted({strip(a), strip(b)} - {ctl})) or "control"
                        for a, b in zip(ga, gb)], dtype=object)
    else:
        lab = np.array([canonical(x, cfg["control"], cfg["sep"]) for x in raw], dtype=object)
    labels, codes = np.unique(lab, return_inverse=True)
    sizes = np.bincount(codes, minlength=len(labels))
    print(f"{a.dataset}: {len(lab):,} cells, {len(labels)} labels "
          f"({int(sizes[labels == 'control'][0]):,} control cells)", flush=True)

    counts, genes = pseudobulk(cfg["h5ad"], codes, len(labels))
    if not np.array_equal(counts, np.round(counts)):
        raise SystemExit("pseudobulk is not integer -- X is not raw counts")
    print(f"  pseudobulk {counts.shape} ({time.time() - t:.0f}s)", flush=True)

    np.savez_compressed(f"{out_dir}/{a.dataset}_pseudobulk_raw.npz",
                        labels=labels.astype(str), genes=genes.astype(str), counts=counts, sizes=sizes)
    ci = int(np.where(labels == "control")[0][0])
    pd.Series(counts[ci] / sizes[ci], index=genes, name="ctrl_counts_per_cell").to_csv(
        f"{out_dir}/{a.dataset}_gene_ctrl_expr.csv", index_label="gene")
    pd.DataFrame({"label": labels,
                  "n_targets": [0 if l == "control" else l.count("|") + 1 for l in labels],
                  "targets": ["" if l == "control" else l for l in labels],
                  "n_cells": sizes}).to_csv(f"{out_dir}/{a.dataset}_labels.csv", index=False)
    print(f"  saved -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
