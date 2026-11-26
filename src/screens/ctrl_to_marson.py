"""Merge Marson-derived control cells into the full Marson Perturb-seq dataset.

Memory-safe approach for ~4.86M Marson cells:
  1. Open control h5ad in backed mode, filter to rows whose `dataset` starts
     with "Marson", materialize that subset only (~204k cells).
  2. Sparsify the control X and cast to Marson's int16 dtype so both files have
     compatible storage.
  3. Reorder control vars to match Marson's var ordering (defensive; they're
     already identical at inspection time).
  4. Write the aligned control to a temporary h5ad.
  5. Use anndata.experimental.concat_on_disk to stream the merge — Marson is
     never loaded into memory. obsm['X_pca'] from Marson is preserved (NaN for
     control rows); Marson's varm/uns are kept via merge="first".

Output: $PERM_RAW/marson_combined_with_controls.h5ad
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../..")))  # repo root
from perm_paths import RAW  # noqa: E402
import gc
import os
import resource
import sys
import time

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp


MARSON_PATH = RAW + "/Marson_with_PCA.h5ad"
CTRL_PATH = RAW + "/ctrl.h5ad"
OUTPUT_PATH = RAW + "/marson_combined_with_controls.h5ad"
TMP_CTRL_PATH = RAW + "/_tmp_ctrl_marson_aligned.h5ad"


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} | peak {peak_gb():.1f} GB] {msg}", flush=True)


def main():
    if os.path.exists(OUTPUT_PATH):
        log(f"ERROR: {OUTPUT_PATH} already exists. Move/delete it before re-running.")
        sys.exit(1)

    # ---- Step 1: peek at Marson (backed, no .X load) -----------------------
    log(f"Opening Marson in backed mode: {MARSON_PATH}")
    marson_backed = sc.read_h5ad(MARSON_PATH, backed="r")
    log(f"Marson: n_obs={marson_backed.n_obs:,}, n_vars={marson_backed.n_vars:,}")

    marson_var_names = marson_backed.var_names.copy()
    marson_X_dtype = marson_backed.X.dtype
    log(f"Marson X dtype: {marson_X_dtype}, sparse: {sp.issparse(marson_backed.X)}")
    marson_backed.file.close()
    del marson_backed
    gc.collect()

    # ---- Step 2: open control in backed mode and filter --------------------
    log(f"Opening control in backed mode: {CTRL_PATH}")
    ctrl_backed = sc.read_h5ad(CTRL_PATH, backed="r")
    log(f"Control: n_obs={ctrl_backed.n_obs:,}, n_vars={ctrl_backed.n_vars:,}")

    if "dataset" in ctrl_backed.obs.columns:
        mask = ctrl_backed.obs["dataset"].astype(str).str.startswith("Marson").to_numpy()
    else:
        log("WARNING: no 'dataset' column on control; falling back to obs_names prefix.")
        mask = ctrl_backed.obs_names.str.startswith("Marson").to_numpy()

    n_keep = int(mask.sum())
    log(f"Marson-derived control rows: {n_keep:,} / {ctrl_backed.n_obs:,}")
    if n_keep == 0:
        log("ERROR: zero Marson control cells found. Aborting.")
        sys.exit(1)

    log("Materializing filtered control into memory...")
    ctrl = ctrl_backed[mask].to_memory()
    ctrl_backed.file.close()
    del ctrl_backed
    gc.collect()
    log(f"Filtered control AnnData: shape={ctrl.shape}, X type={type(ctrl.X).__name__}")

    # ---- Step 3: sparsify + dtype-match to Marson --------------------------
    if not sp.issparse(ctrl.X):
        log("Converting control X (dense) to CSR sparse...")
        ctrl.X = sp.csr_matrix(ctrl.X)
        gc.collect()
    elif not sp.isspmatrix_csr(ctrl.X):
        log(f"Converting control X ({type(ctrl.X).__name__}) to CSR...")
        ctrl.X = ctrl.X.tocsr()
        gc.collect()

    if ctrl.X.dtype != marson_X_dtype:
        log(f"Casting control X {ctrl.X.dtype} -> {marson_X_dtype} to match Marson...")
        ctrl.X = ctrl.X.astype(marson_X_dtype)
        gc.collect()

    nnz = ctrl.X.nnz
    log(f"Control sparse CSR: nnz={nnz:,}, density={nnz / (ctrl.shape[0] * ctrl.shape[1]):.3%}")

    # ---- Step 4: align vars to Marson's order ------------------------------
    if not ctrl.var_names.equals(marson_var_names):
        common = marson_var_names.intersection(ctrl.var_names)
        log(f"Reordering / subsetting control vars to match Marson ({len(common):,} shared genes).")
        if len(common) == 0:
            log("ERROR: no shared genes between Marson and control. Aborting.")
            sys.exit(1)
        # Reorder to Marson's gene order, restricted to shared genes
        ordered = marson_var_names[marson_var_names.isin(ctrl.var_names)]
        ctrl = ctrl[:, ordered].copy()
        gc.collect()
    else:
        log("Control vars already match Marson order — no reordering needed.")

    # ---- Step 5: tag controls and write temp file --------------------------
    # We deliberately do NOT try to merge var/varm/uns through concat_on_disk:
    # `merge="first"` triggers an anndata bug in `_write_alt_mapping` where
    # bool columns become pandas Series with NaN fill and the writer crashes.
    # Instead we let concat_on_disk write an empty var (just var_names) and
    # then copy Marson's /var, /varm, /uns into the output via h5py below.
    ctrl.obs["is_control"] = True

    log(f"Writing aligned control to temp file: {TMP_CTRL_PATH}")
    if os.path.exists(TMP_CTRL_PATH):
        os.remove(TMP_CTRL_PATH)
    ctrl.write_h5ad(TMP_CTRL_PATH, compression="gzip")
    del ctrl
    gc.collect()
    log("Temp control written.")

    # ---- Step 6: on-disk concat (Marson is never loaded) -------------------
    log("Running anndata.experimental.concat_on_disk ...")
    log(f"  inputs: [{MARSON_PATH}, {TMP_CTRL_PATH}]")
    log(f"  output: {OUTPUT_PATH}")
    t0 = time.time()
    ad.experimental.concat_on_disk(
        in_files=[MARSON_PATH, TMP_CTRL_PATH],
        out_file=OUTPUT_PATH,
        axis=0,
        join="inner",          # intersect on vars (already identical)
        # merge / uns_merge left at default (None). Setting them triggers an
        # anndata bug in _write_alt_mapping; we patch /var, /varm, /uns in via
        # h5py after the concat finishes.
        label="source",
        keys=["marson", "control"],
        index_unique=None,
    )
    log(f"concat_on_disk finished in {(time.time() - t0) / 60:.1f} min.")

    # ---- Step 6b: patch /var, /varm, /uns into output from Marson ----------
    # concat_on_disk (with default merge=None) writes an empty /var and no
    # /varm or /uns. Since the two files have identical genes, we can copy
    # Marson's metadata groups directly — this is just metadata, fast.
    log("Patching /var, /varm, /uns into output from Marson source file...")
    with h5py.File(MARSON_PATH, "r") as src, h5py.File(OUTPUT_PATH, "r+") as dst:
        for group_name in ("var", "varm", "uns"):
            if group_name not in src:
                log(f"  Marson has no /{group_name}; skipping.")
                continue
            if group_name in dst:
                del dst[group_name]
            src.copy(group_name, dst, name=group_name)
            log(f"  Copied /{group_name} from Marson into output.")

    # ---- Step 7: cleanup ---------------------------------------------------
    try:
        os.remove(TMP_CTRL_PATH)
        log(f"Removed temp file {TMP_CTRL_PATH}")
    except OSError as e:
        log(f"Could not remove temp file: {e}")

    # ---- Step 8: sanity check ---------------------------------------------
    log("Verifying output ...")
    out_backed = sc.read_h5ad(OUTPUT_PATH, backed="r")
    log(f"Combined: n_obs={out_backed.n_obs:,}, n_vars={out_backed.n_vars:,}")
    if "source" in out_backed.obs.columns:
        log("Source counts:\n" + str(out_backed.obs["source"].value_counts()))
    out_backed.file.close()
    log("Done.")


if __name__ == "__main__":
    main()
