"""Every filesystem root PERM reads from or writes to, in one place.

Nothing here points at a personal directory. Each root defaults to a location
inside the repository and can be overridden with the environment variable named
beside it, so the code runs unchanged on another machine:

    export PERM_DATA=/path/to/perm-data

paths.local.sh.example lists every variable: copy it to paths.local.sh,
fill in your own roots and source it before running anything.
None of these directories ship with the repository. See the README for the
Zenodo archive and the per-dataset accessions.
"""
import os

_env = os.environ.get

# ---- code in this repository ------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "src")
LIB = os.path.join(SRC, "lib")                      # marson_KO, matching, ...
FIGURES = os.path.join(REPO, "figures")
MEMORY_GENES = os.path.join(SRC, "memory_genes")
CLONEPAIR = os.path.join(MEMORY_GENES, "clonepair")

# ---- data, none of which is in the repository -------------------------------
DATA = _env("PERM_DATA", os.path.join(REPO, "data"))
RAW = _env("PERM_RAW", os.path.join(DATA, "raw"))
PROJECT_DATA = _env("PERM_PROJECT_DATA", os.path.join(DATA, "reference"))

# memory-gene csvs. The root holds two trees:
#   python_implements/<dataset>/  our Python ports -- what every figure reads
#   data/<dataset>/               the authors' original implementations
MEMGENES_ROOT = _env("PERM_MEMGENES_ROOT", os.path.join(DATA, "memory_genes"))
MEMGENES = os.path.join(MEMGENES_ROOT, "python_implements")
MEMGENES_AUTHORS = os.path.join(MEMGENES_ROOT, "data")

# ---- outputs ----------------------------------------------------------------
OUTPUTS = _env("PERM_OUTPUTS", os.path.join(REPO, "outputs"))
PLOTS = _env("PERM_PLOTS", os.path.join(REPO, "plots"))
SCRATCH = _env("PERM_SCRATCH", os.path.join(DATA, "scratch"))
