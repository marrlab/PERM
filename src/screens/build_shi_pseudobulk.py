"""
build_shi_pseudobulk.py - RAW pseudobulk for the whole-brain CRISPR atlas.

Reads the 14 local shard h5ads, sums raw counts per (gene_target x predicted_group)
group, and caches the result to a persistent .npz. The CPM LFC is then built cheaply
in prepare_mold.ipynb from this npz (never needs the shards again).

Gene list + group set are inherited from the OLD normalised npz when present, so
the rebuild is gene-for-gene / group-for-group comparable; otherwise falls back to
the intersection of genes across shards.

Run order:
  1. (login node, has internet)   python build_shi_pseudobulk.py --download-only
  2. (compute node, ~64 GB)       python build_shi_pseudobulk.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../..")))  # repo root
from perm_paths import DATA, LIB, SCRATCH  # noqa: E402
import argparse
import os
import sys

import numpy as np

# shared perturb-seq libs live outside this build dir
MARSON_DIR = LIB
sys.path.insert(0, MARSON_DIR)
from shi_KO import (SHARD_NAMES, download_shards, build_shi_pseudobulk,
                     save_shi_pseudobulk)

SHARD_DIR = SCRATCH + '/mold_shards'                         # scratch: re-downloadable, OK to purge
OLD_NPZ   = SCRATCH + '/out/pseudobulk.npz'                   # for gene/group list (may be purged)
GENE_CSV  = DATA + '/Mold_brain/mold_target_genes.csv'   # persistent copy of the gene list
RAW_NPZ   = DATA + '/Mold_brain/mold_pseudobulk_raw.npz' # persistent (the irreplaceable output)


def resolve_target_genes(shard_paths):
    """Prefer the old npz's gene order; else a saved CSV; else intersect shards."""
    if os.path.exists(OLD_NPZ):
        genes = np.load(OLD_NPZ, allow_pickle=True)['genes'].astype(str)
        os.makedirs(os.path.dirname(GENE_CSV), exist_ok=True)
        import pandas as pd
        pd.Series(genes, name='gene').to_csv(GENE_CSV, index=False)
        print(f"target genes from OLD npz: {len(genes)} (saved to {GENE_CSV})", flush=True)
        return genes
    if os.path.exists(GENE_CSV):
        import pandas as pd
        genes = pd.read_csv(GENE_CSV)['gene'].astype(str).values
        print(f"target genes from saved CSV: {len(genes)}", flush=True)
        return genes
    import anndata as ad
    common = None
    for p in shard_paths:
        a = ad.read_h5ad(p, backed='r')
        g = set(a.var_names.astype(str)); a.file.close()
        common = g if common is None else (common & g)
    genes = np.array(sorted(common))
    print(f"target genes from shard intersection: {len(genes)}", flush=True)
    return genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download-only', action='store_true',
                    help='Download shards then exit (run on a login node).')
    ap.add_argument('--shard-dir', default=SHARD_DIR)
    ap.add_argument('--out', default=RAW_NPZ)
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--chunk-size', type=int, default=50_000)
    args = ap.parse_args()

    shard_paths = [os.path.join(args.shard_dir, f"{n}.h5ad") for n in SHARD_NAMES]
    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        print(f"{len(missing)} shards missing; downloading to {args.shard_dir} ...", flush=True)
        shard_paths = download_shards(SHARD_NAMES, dest_dir=args.shard_dir)
    if args.download_only:
        print("download-only: done."); return

    genes = resolve_target_genes(shard_paths)
    pb, counts = build_shi_pseudobulk(
        shard_paths, target_genes=genes,
        min_cells=args.min_cells, chunk_size=args.chunk_size, verbose=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_shi_pseudobulk(pb, counts, args.out)


if __name__ == '__main__':
    main()
