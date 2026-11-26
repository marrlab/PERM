#!/usr/bin/env python
"""Step 2 task list: one row per Power-Seek run -> tasks.tsv (row i = array task i).

Per dataset: the full run plus two disjoint halves (split-half reproducibility);
the Marson repro gets the full run only (it exists to be compared with the old run).

Cost: each of the ~3,000 leave-one-gene-out steps multiplies the cells x genes matrix,
so run time grows LINEARLY with cells (Frangieh Control, 15,361 cells, 8 threads:
36 min = ~2.3 min per 1k cells). Tier from that:
  short  : estimated <= 180 min at 16 threads            -> partition short (4 h max)
  normal : bigger (XAtlas / Mali full runs, ~140-200k)    -> partition normal
"""
import os, json, glob
import pandas as pd

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

# datasets.tsv is read from, and tasks.tsv written to, this directory (they are
# configuration and ship with the code); the QC thresholds come from the results
# that qc_filter_controls.py wrote to the data tree.
RUN_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(P.MEMGENES_ROOT, "power-seek_run", "results")
MIN_PER_1K_CELLS_16T = 1.4      # ~2.3 at 8 threads, assumed ~1.7x faster at 16
SHORT_LIMIT_MIN = 180

ds = pd.read_csv(os.path.join(RUN_DIR, "datasets.tsv"), sep="\t", dtype=str, keep_default_na=False)
rows = []
# Rows keep their datasets.tsv order, so task numbers never shift when rows are added at
# the end. run_powerseek == n -> tier 'dropped': numbered but never run (the 10 unused
# Marson groups; Wong/Mali pooled, superseded per group).
for _, d in ds.iterrows():
    name = d["name"]
    js = os.path.join(RESULTS, "qc", f"{name}_thresholds.json")
    n = json.load(open(js))["out_shape"][0]
    for half in ([0] if name.endswith("_repro") else [0, 1, 2]):
        n_used = n if half == 0 else n // 2
        est = MIN_PER_1K_CELLS_16T * n_used / 1000
        tier = "short" if est <= SHORT_LIMIT_MIN else "normal"
        rows.append(dict(name=name, half=half, n_cells=n_used, est_min_16t=round(est),
                         tier=tier if d.get("run_powerseek", "y") != "n" else "dropped"))
t = pd.DataFrame(rows)
t.insert(0, "task", range(len(t)))
t.to_csv(os.path.join(RUN_DIR, "tasks.tsv"), sep="\t", index=False)
print(t.to_string(index=False))
print(f"\n{len(t)} tasks: {dict(t.tier.value_counts())}")
