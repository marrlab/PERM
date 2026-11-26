#!/usr/bin/env python3
"""
write_lfc.py - save the per-perturbation x gene LFC matrices for Suppl. Fig 8.

Why this exists
---------------
responsiveness.ipynb computes these matrices but never keeps them. Its
gene_summary() walks the perturbations in chunks of 1,000, accumulates
sum(|LFC|) and n_obs into two per-gene vectors, and discards each chunk, "so an
18k x 38k dataset never needs its full LFC matrix in memory". Only the per-gene
margin reaches disk. That is all Suppl. Fig 8 plots, but it leaves no matrix to
hand out or re-analyse, so this script re-derives them from the cached
pseudobulks and writes them out.

It is a RE-DERIVATION, not a re-run: the pseudobulks are the inputs
responsiveness.ipynb already used, and the LFC call here is the same one with
the same arguments, so the numbers are identical to what the panel reports.
Nothing upstream is touched and no screen is reprocessed.

Faithfulness to the panel
-------------------------
Two things are reproduced exactly from responsiveness.ipynb cell 1:

* canonical_labels() - guide variants are merged by summing counts before the
  ratio, and the double-perturbation datasets get their 'control' / 'A' / 'A|B'
  labels rebuilt the same way. Replogle 2020 needs its source h5ad for this,
  since its perturbation ids are opaque and resolve only through gene_A/gene_B.
* target masking - lfc_from_pseudobulk(mask_self_ko=True) blanks the diagonal,
  and then EVERY target of a multi-target label is blanked too ('A|B' masks both
  A and B). Without that second step a CRISPRa target, pushed up enormously,
  would read as maximally responsive. Masked entries are NaN in the output.

Perturbations with fewer than MIN_CELLS cells are dropped, as in the panel.

Detection floor
---------------
Columns are restricted to genes with mean control expression > --floor (default
0.1 raw counts per control cell), which is what every other LFC matrix in the
project ships with: prepare_frangieh.py and prepare_melanoma.py hardcode
MIN_CTRL_EXPR = 0.1, and velten / mold / mali / hsc were all built with
--min-ctrl-expr 0.1 (verified against their *_gene_ctrl_expr.csv: 8,356 of
57,186 for velten, 8,919 of 32,287 for hsc, and so on). marson_lfc_cpm.csv is
the one unfloored exception and is documented as such.

ORDER MATTERS and follows those builders exactly: the LFC is computed over the
FULL gene set, so per-group CPM is normalised against the whole library, and only
then are columns subset to the surviving genes. Flooring the pseudobulk first
would change every library size and so every LFC. --floor 0 keeps all genes.

The floor is applied PER DATASET, matching powerseek_responsiveness(), which
filters each Suppl. Fig 8 row on its own ctrl_expr. Note that prepare_mali.py
instead pools control expression across ALPHA/BETA/GAMMA so its three batches
share one gene set; here they are separate datasets with separate pseudobulks
and so get separate floors (8,654 / 8,503 / 8,460 genes).

Output
------
One npz per dataset, float32, at --out-dir:

    <name>_lfc_cpm_<tag>.npz
        values     float32 (n_perturbations, n_genes), NaN where masked
        perts      unicode (n_perturbations,)  row labels, canonicalised
        genes      unicode (n_genes,)          column labels, post-floor
        ctrl_expr  float64 (n_genes,)  mean control counts/cell, the floored
                   quantity; the companion to *_gene_ctrl_expr.csv elsewhere
        n_cells    int64   (n_perturbations,)  cells behind each row
        kind       unicode (n_perturbations,)  'single' or 'double'
        meta       unicode scalar, JSON: dataset, tag, normalize, pseudocount,
                   cpm_target, min_cells, floor, genes before/after the floor,
                   masked_entries, source pseudobulk

float32 halves the footprint and is far finer than the precision of a
pseudobulk log-ratio. The three double-perturbation datasets get ONE matrix each
holding both perturbation types; split on `kind` to recover the panel's two rows
(the two gene_summary CSVs are those same rows summarised separately).

Usage
-----
    python write_lfc.py --all                      # all 15, sequentially
    python write_lfc.py --index $SLURM_ARRAY_TASK_ID   # one, for the array job
    python write_lfc.py --name XAtlas2025_HCT116
    python write_lfc.py --list                     # print the 15 and exit

Existing outputs are skipped unless --force. Run it on a compute node: the two
X-Atlas screens are 17.9k x 38.6k and peak around 20 GB because
lfc_from_pseudobulk works in float64.
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
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
sys.path.insert(0, P.LIB)

from marson_KO import load_pseudobulk, lfc_from_pseudobulk  # noqa: E402

# Identical to responsiveness.ipynb cell 0. Changing any of these makes the
# output stop matching Suppl. Fig 8, so they are constants, not flags.
PB_DIR = f"{P.SCRATCH}/power-seek_run/pseudobulk"
PB_TAG = "koqc-all_ctrlqc-same"
OUT_DIR = f"{P.SCRATCH}/power-seek_run/lfc"
MIN_CELLS = 20
# Detection floor on mean control counts/cell. 0.1 is what every other LFC matrix
# in the project uses, and what powerseek_responsiveness() applies at read time.
FLOOR = 0.1
PSEUDOCOUNT = 1.0
CPM_TARGET = 1e6
NORMALIZE = "group"
# Suppl. Fig 8 excludes the Marson rows: they are Figure 2. Same prefix test the
# notebook and figures.powerseek_responsiveness() both use.
EXCLUDE = ("Marson",)


def s8_datasets() -> pd.DataFrame:
    """The 15 datasets Suppl. Fig 8 draws, in panel order."""
    DS = pd.read_csv(f"{RUN}/datasets.tsv", sep="\t", dtype=str, keep_default_na=False)
    DS = DS[(DS.run_powerseek == "y") & (DS.repro_from == "-")]
    if EXCLUDE:
        DS = DS[~DS.name.str.startswith(tuple(EXCLUDE))]
    return DS.reset_index(drop=True)


def canonical_labels(row, raw):
    """Verbatim from responsiveness.ipynb cell 1.

    Double-perturbation labels as in build_double_pseudobulk.py: 'control', 'A',
    'A|B' (sorted). Replogle 2020 (GSE146194) is rebuilt from gene_A/gene_B:
    '_2' is a second guide for the SAME gene (merged), 'CTRL' is its
    non-targeting slot. Single-perturbation datasets: labels unchanged.
    """
    if row.design != "double":
        return {l: l for l in raw}
    if row["name"].startswith("Replogle2020"):
        import h5py

        import qc_filter_controls as q

        with h5py.File(os.path.expandvars(row.path), "r") as f:
            t = pd.DataFrame(
                {c: q.read_col(f, "obs", c).astype(str)
                 for c in ("perturbation", "gene_A", "gene_B")}
            ).drop_duplicates()
        m = {"control": "control"}
        for p, g in t.groupby("perturbation"):
            toks = sorted({x.replace("_2", "") for x in g.iloc[0][["gene_A", "gene_B"]]
                           if x not in ("CTRL", "", "nan")})
            m[p] = "|".join(toks) if toks else "control"
        return {l: m[l] for l in raw}
    sep = "|" if any("|" in l for l in raw) else "_"
    return {l: "control" if l == "control"
            else "|".join(sorted(l.split(sep))) for l in raw}


def build(row, out_dir=OUT_DIR, floor=FLOOR, force=False, verbose=True):
    """Write one dataset's LFC matrix. Returns the path, or None if skipped."""
    name = row["name"]
    npz = f"{PB_DIR}/{name}_pseudobulk_{PB_TAG}.npz"
    out = f"{out_dir}/{name}_lfc_cpm_{PB_TAG}.npz"
    if not os.path.exists(npz):
        print(f"  {name}: no pseudobulk at {npz} -- SKIPPED", flush=True)
        return None
    if os.path.exists(out) and not force:
        print(f"  {name}: {os.path.basename(out)} exists (--force to redo)", flush=True)
        return out

    pb, sizes = load_pseudobulk(npz)
    # merge guide variants by summing counts, exactly as gene_summary() does:
    # weight each group by its cell count, sum, then divide by the summed count
    lab = pb.index.map(canonical_labels(row, pb.index))
    if not (lab == pb.index).all():
        pb, sizes = pb.mul(sizes, axis=0).groupby(lab).sum(), sizes.groupby(lab).sum()
        pb = pb.div(sizes, axis=0)

    kos = [g for g in pb.index if g != "control" and sizes[g] >= MIN_CELLS]
    if not kos:
        print(f"  {name}: no perturbation clears {MIN_CELLS} cells -- SKIPPED", flush=True)
        return None

    sub = ["control"] + kos
    with contextlib.redirect_stdout(io.StringIO()):          # it prints per call
        lfc = lfc_from_pseudobulk(
            pb.loc[sub], sizes.loc[sub], min_cells_per_ko=MIN_CELLS,
            control_label="control", normalize=NORMALIZE,
            cpm_target=CPM_TARGET, pseudocount=PSEUDOCOUNT)

    # mask EVERY target of a label ('A|B' masks both A and B). The self-KO
    # diagonal is already NaN from mask_self_ko=True; this adds the partners.
    a = lfc.values
    col = {g: j for j, g in enumerate(lfc.columns)}
    for r_, l in enumerate(lfc.index):
        for t in str(l).split("|"):
            j = col.get(t)
            if j is not None:
                a[r_, j] = np.nan

    # detection floor, AFTER the LFC so per-group CPM used the whole library --
    # the order prepare_frangieh.py and prepare_mali.py both use. ctrl_expr is the
    # control row of the pseudobulk, i.e. mean RAW counts per control cell, which
    # lfc_from_pseudobulk leaves untouched (it copies before normalising).
    ctrl_expr = pb.loc["control"].astype(np.float64)
    n_genes_pre = lfc.shape[1]
    if floor and floor > 0:
        keep = ctrl_expr.index[ctrl_expr > floor]
        cols = lfc.columns.intersection(keep)
        if len(cols) == 0:
            print(f"  {name}: no gene clears floor {floor} -- SKIPPED", flush=True)
            return None
        a = a[:, lfc.columns.get_indexer(cols)]
        lfc = pd.DataFrame(a, index=lfc.index, columns=cols)
        ctrl_expr = ctrl_expr.reindex(cols)
    else:
        ctrl_expr = ctrl_expr.reindex(lfc.columns)
    n_masked = int(np.isnan(a).sum())

    perts = lfc.index.to_numpy(dtype=str)
    kind = np.array(["double" if "|" in p else "single" for p in perts], dtype=str)
    meta = dict(dataset=name, tag=PB_TAG, design=row.design, normalize=NORMALIZE,
                pseudocount=PSEUDOCOUNT, cpm_target=CPM_TARGET,
                min_cells_per_perturbation=MIN_CELLS, masked_entries=n_masked,
                ctrl_expr_floor=float(floor or 0.0),
                genes_before_floor=int(n_genes_pre), genes_after_floor=int(lfc.shape[1]),
                source_pseudobulk=npz, units="log2 fold change vs control, per-group CPM",
                floor_note=("columns restricted to mean control counts/cell > floor, "
                            "applied AFTER the LFC so CPM used the full library"),
                note="NaN = target masked (self and every partner of 'A|B')")

    os.makedirs(out_dir, exist_ok=True)
    tmp = out + ".tmp.npz"            # write-then-rename: a killed job leaves no
    np.savez(tmp,                     # half-written npz that looks complete
             values=a.astype(np.float32),
             perts=perts,
             genes=lfc.columns.to_numpy(dtype=str),
             ctrl_expr=ctrl_expr.to_numpy(dtype=np.float64),
             n_cells=sizes.reindex(lfc.index).to_numpy(dtype=np.int64),
             kind=kind,
             meta=np.array(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, out)

    if verbose:
        n_s, n_d = int((kind == "single").sum()), int((kind == "double").sum())
        print(f"  {name}: {lfc.shape[0]} x {lfc.shape[1]} "
              f"({n_s} single, {n_d} double)  genes {n_genes_pre} -> {lfc.shape[1]} "
              f"at floor {floor}  {n_masked} masked  "
              f"{os.path.getsize(out)/1e6:.0f} MB -> {os.path.basename(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every Suppl. Fig 8 dataset")
    g.add_argument("--index", type=int, help="one dataset by 0-based panel position")
    g.add_argument("--name", help="one dataset by name, as in datasets.tsv")
    g.add_argument("--list", action="store_true", help="print the datasets and exit")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--floor", type=float, default=FLOOR,
                    help="detection floor on mean control counts/cell "
                         f"(default {FLOOR}, as every other LFC matrix; 0 = keep all genes)")
    ap.add_argument("--force", action="store_true", help="overwrite existing npz")
    args = ap.parse_args()

    DS = s8_datasets()
    if args.list:
        for i, r in DS.iterrows():
            print(f"{i:3d}  {r['name']:38s} {r.design}")
        return
    if args.index is not None:
        if not 0 <= args.index < len(DS):
            print(f"--index must be 0..{len(DS)-1} ({len(DS)} datasets)", file=sys.stderr)
            sys.exit(2)
        DS = DS.iloc[[args.index]]
    elif args.name:
        DS = DS[DS.name == args.name]
        if DS.empty:
            print(f"{args.name!r} is not a Suppl. Fig 8 dataset; --list to see them",
                  file=sys.stderr)
            sys.exit(2)

    print(f"writing {len(DS)} LFC matrix/matrices to {args.out_dir}", flush=True)
    wrote = [build(r, out_dir=args.out_dir, floor=args.floor, force=args.force)
             for _, r in DS.iterrows()]
    ok = [p for p in wrote if p]
    total = sum(os.path.getsize(p) for p in ok)
    print(f"\ndone: {len(ok)}/{len(DS)} written, {total/1e9:.2f} GB", flush=True)
    if len(ok) != len(DS):
        sys.exit(1)


if __name__ == "__main__":
    main()
