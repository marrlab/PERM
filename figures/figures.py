"""
figures.py — panel builders for the publication figures.

One function per panel, everything that could plausibly change exposed as a
keyword argument, so a notebook cell is a single call. Nothing here reads
notebook globals: every function takes the dataset(s) it needs.

Which function draws which panel is listed in the repository README.

Conventions
-----------
* every figure function takes `save=` (None | True | path | list of paths) and
  `show=`, and RETURNS the matplotlib Figure plus the numbers behind it, so the
  same call serves both the figure and the results text.
* `top_n=None` everywhere means "the deepest cut every method can actually
  supply" — the shallowest method/timepoint sets it. A larger request is capped
  and the capping is printed, never silent.
* h5ad reads and the per-dataset mean-expression sweep are cached in-process
  (`clear_cache()` to drop them) — otherwise re-running a cell re-reads GBs.
"""

from __future__ import annotations

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P

import os
import re
import glob
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — edit here, or override per call
# ═══════════════════════════════════════════════════════════════════════════

BENCH = P.MEMGENES_ROOT
RESULTS_TREE = "python_implements"          # flip to "data" for the authors' originals
PANELS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")
# THE cutoff, set ONCE per notebook with use_top_n(). It is read from the
# environment rather than hardcoded because `%autoreload 2` re-executes this module
# and would otherwise reset a value assigned from the notebook (the long-standing
# "F.TOP_N went back to 100" trap). Every output then lands in panels/top<N>/, so two
# cutoffs can never overwrite each other's figures.
TOP_N_ENV = "MEMGENE_TOP_N"
_TOP_N_DEFAULT = 100
OUTDIR = os.path.join(PANELS_BASE,
                      f"top{int(os.environ.get(TOP_N_ENV, _TOP_N_DEFAULT))}")
# Every saved figure is ALSO written here as vector art, whatever raster format the
# call asked for — journals want editable vectors, and remembering per call does not
# scale. Set ALSO_SVG = False to switch it off.
SVGDIR = os.path.join(OUTDIR, "svg")


def use_top_n(n, base=None, make=True, clear=True, verbose=True):
    """Set THE cutoff for a whole notebook, once, at the top.

    Does three things that have to stay in step: TOP_N (what the panels compute),
    OUTDIR/SVGDIR (panels/top<N>/, so two cutoffs cannot overwrite each other), and
    the environment variable the module re-reads on `%autoreload`, which is what
    stops the setting from quietly reverting mid-session. Defaults of every
    `top_n=` parameter are retargeted too, so a call that omits the argument
    follows the notebook's setting rather than the value frozen at import.

    Returns the new OUTDIR.
    """
    global TOP_N, OUTDIR, SVGDIR, PANELS_BASE
    n, old = int(n), TOP_N
    os.environ[TOP_N_ENV] = str(n)
    TOP_N = n
    PANELS_BASE = base or PANELS_BASE
    OUTDIR = os.path.join(PANELS_BASE, f"top{n}")
    SVGDIR = os.path.join(OUTDIR, "svg")
    if make:
        os.makedirs(SVGDIR, exist_ok=True)
    _retarget_top_n(old, n)
    if clear:
        clear_cache()
    if verbose:
        print(f"top N = {n}  ->  {OUTDIR}")
    return OUTDIR


def _retarget_top_n(old, new):
    """Rebind every `top_n`-style default that still carries the previous cutoff."""
    import inspect
    names = ("top_n", "ref_top_n", "mark_top_n", "cons_top_n", "bg_top_n")
    for fn in list(globals().values()):
        if not inspect.isfunction(fn) or fn.__defaults__ is None:
            continue
        try:
            args = inspect.getfullargspec(fn)
        except TypeError:
            continue
        pos = args.args[-len(fn.__defaults__):]
        new_defaults = tuple(new if (a in names and d == old) else d
                             for a, d in zip(pos, fn.__defaults__))
        if new_defaults != fn.__defaults__:
            fn.__defaults__ = new_defaults


def _check_outdir(outdir):
    """Warn when a save would land somewhere that does not match the cutoff in
    force: the one way the 'set it once' rule can be broken is assigning F.OUTDIR
    (or F.TOP_N) by hand afterwards."""
    want = f"top{TOP_N}"
    if os.path.basename(os.path.normpath(str(outdir))) != want:
        warnings.warn(
            f"saving into {outdir!r} while TOP_N = {TOP_N}: the folder does not "
            f"match the cutoff. Set the cutoff with F.use_top_n({TOP_N}) and do "
            "not assign F.OUTDIR directly.", stacklevel=3)
ALSO_SVG = True

# label -> where its results and its gene universe live.
#   folder   : subfolder under BENCH/<tree>
#   h5ad     : defines the gene universe (and mean expression for the matched null)
#   species  : 'human' | 'mouse'  (mouse gets ortholog-mapped for cross-dataset work)
#   var_col  : var column holding the symbol; None = the var index already does
#   ortho    : that dataset's mouse->human cache (mouse only)
# LARRY: all five methods were re-run on the fixed h5ad 2026-09-08, so it is no
# longer excluded from the cross-dataset work (xd_datasets). Added to DATASETS
# 2026-09-09, once it had its OWN mouse2human_orthologs.csv — built by the same
# mygene/homologene route as BandlerBrain's, over the full h5ad
# var rather than the called genes, so all three mouse datasets are mapped from
# one source and the shared universe is not shrunk by a partial map.
DATASETS = {
    "T-lineage":  dict(folder="T-lineage",         species="human", var_col=None,
                       h5ad=f"{P.DATA}/datasets/T-lineage_processed.h5ad"),
    "Harmange":   dict(folder="Melanoma_Harmange", species="human", var_col=None,
                       h5ad=f"{P.DATA}/datasets/Melanoma_Harmange_processed.h5ad"),
    "LARRY":      dict(folder="LARRY",             species="mouse", var_col=None,
                       h5ad=f"{P.DATA}/datasets/LARRY_processed.h5ad",
                       ortho=f"{BENCH}/{RESULTS_TREE}/LARRY/mouse2human_orthologs.csv"),
    "Bandler":    dict(folder="BandlerBrain",      species="mouse", var_col=None,
                       h5ad=f"{P.DATA}/bandler_brain/bandler_brain.h5ad",
                       ortho=f"{BENCH}/{RESULTS_TREE}/BandlerBrain/mouse2human_orthologs.csv"),
}
ORTHO_FALLBACK = f"{BENCH}/{RESULTS_TREE}/BandlerBrain/mouse2human_orthologs.csv"

# ── house style ────────────────────────────────────────────────────────────
# ONE canonical order for methods and datasets, reused by every panel, legend and
# table. Change it here and every figure follows.
# NOTE the scripts outside figures.py keep their own method lists: build_gene_lists.py,
# build_consensus.py, marson_KO.py, ko_axis_memory.py.
METHOD_EXCLUDE = frozenset()
METHOD_ORDER_ALL = ["CORAL", "GEMLI", "Power-Seek", "ClonePair"]
METHOD_ORDER = [m for m in METHOD_ORDER_ALL if m not in METHOD_EXCLUDE]
# Ordered species-first (human, then mouse) so it matches the Fig 1a sunburst,
# whose species ring requires the mouse datasets to be adjacent.
DATASET_ORDER = ["T-lineage", "Harmange", "LARRY", "Bandler"]
MIN_FONTSIZE = 7.0          # nothing a reader must read goes below this

# Datasets are cited by FIRST AUTHOR. Only T-lineage changes (Michaels); the other
# three folder keys are already author names. Keys stay as they are so file lookup
# is untouched — this map is display only.
DATASET_LABELS = {"T-lineage": "Michaels", "Harmange": "Harmange",
                  "Bandler": "Bandler", "LARRY": "Weinreb"}
# What was actually profiled. NOTE Michaels is 33% myeloid progenitors, 32%
# MPP/HSC and only 17% progenitor T cells — "Progenitor T cells" names the
# intended lineage, not the majority of the cells.
# HSPC rather than "Haematopoietic progenitors": it is what xd_datasets already
# calls the Weinreb-Bowness row, so figures 1 and 2 now name it the same way.
DATASET_CELLTYPE = {"T-lineage": "Progenitor T cells", "Harmange": "Melanoma",
                    "Bandler": "Brain", "LARRY": "HSPC"}
# What the FIGURES call each dataset when LABEL_MODE == "celltype": the short form
# a reader can hold in their head, rather than the precise one in DATASET_CELLTYPE
# (whose "Progenitor T cells" and "HSPC" are the careful names, kept for Methods).
DATASET_CELLTYPE_SHORT = {"T-lineage": "T cells", "Harmange": "Melanoma",
                          "Bandler": "Brain", "LARRY": "HSPCs"}
# "author"   -> Michaels, Harmange, Weinreb, Bandler   (the default, as before)
# "celltype" -> T cells, Melanoma, HSPCs, Brain
# Set it ONCE per notebook with use_label_mode(), next to use_top_n(). Like TOP_N it
# lives in the ENVIRONMENT, because `%autoreload 2` re-executes this module and would
# otherwise silently put a plain `F.LABEL_MODE = "celltype"` back to "author"
# mid-session. dlabel()/slabel() are the only places a name is resolved, so nothing
# else has to know.
LABEL_MODE_ENV = "MEMGENE_LABEL_MODE"
LABEL_MODE = os.environ.get(LABEL_MODE_ENV, "author")


def use_label_mode(mode, verbose=True):
    """Set how datasets are named for a whole notebook: "author" or "celltype"."""
    global LABEL_MODE
    mode = str(mode).lower()
    if mode not in ("author", "celltype"):
        raise ValueError(f"mode must be 'author' or 'celltype', got {mode!r}")
    os.environ[LABEL_MODE_ENV] = mode
    LABEL_MODE = mode
    if verbose:
        print(f"labels = {mode}  (e.g. {dlabel('T-lineage')})")
    return mode

# ── which datasets the figures actually use ────────────────────────────────
# A panel may be restricted to a subset by adding labels here; an EXPLICIT
# datasets=/dataset= argument is NEVER filtered. Defaults resolve at CALL time,
# so the flag is live for the rest of a session once set in a notebook cell.
DATASET_EXCLUDE = frozenset()


def active_datasets(datasets=None, pool=None, include=()):
    """The dataset labels a panel should default to, in DATASET_ORDER.

    datasets : passed explicitly it WINS, untouched beyond ordering -- naming a
               dataset is never overruled by DATASET_EXCLUDE.
    pool     : what to choose from; default DATASET_ORDER. Pass a registry's keys
               where the panel's universe is not the canonical five (CHAR_DATASETS
               also holds iPSC, which has never been in the default list).
    include  : one label or several to keep even though they are excluded.
    """
    if datasets is not None:
        return in_order(list(datasets), DATASET_ORDER)
    if isinstance(include, str):
        include = (include,)
    pool = list(DATASET_ORDER) if pool is None else list(pool)
    keep = [d for d in pool if d not in DATASET_EXCLUDE or d in set(include)]
    return in_order(keep, DATASET_ORDER)


def active_subset(mapping, include=()):
    """`mapping` ({label: config}) with the excluded datasets dropped, in
    DATASET_ORDER -- for the panels that need the configs, not just the names."""
    return {k: mapping[k] for k in active_datasets(pool=list(mapping),
                                                   include=include)}


# ONE colour per dataset. Figure 1a's sunburst and figure 2a's are read side by
# side, so the same dataset MUST be the same colour in both; a map rather than a
# list because dropping a dataset then removes its entry instead of shifting
# every dataset after it onto its neighbour's colour. Human first (purples),
# then mouse (blues), matching DATASET_ORDER.
# Palette chosen 2026-09-14. Set HERE, not only in a notebook cell: %autoreload 2
# re-imports this file whenever it changes, which silently resets any
# F.DATASET_COLORS.update(...) made in a notebook back to these values.
DATASET_COLORS = {"T-lineage": "#9d4edd", "Harmange": "#e0aaff",
                  "LARRY": "#5aedc9", "Bandler": "#48bfe3"}
SPECIES_PALETTE = {"human": "#72369d", "mouse": "#64dfdf"}


# Group display names for figure 1b. DEFAULT: almost empty, on purpose -- every
# dataset keeps its OWN group names, because the groups mean different things in
# each one and a uniform "T1/T2/T3" would assert an equivalence that does not exist:
#   Michaels  T1 / T2        timepoints of one experiment  (native names, kept)
#   Harmange  E / M          STARTING POPULATIONS: E = FACS-enriched for primed
#                            cells, M = unsorted (primed fraction .62-.84 vs
#                            .06-.09). Not timepoints, and not replicates of each
#                            other -- Harmange's replicate axis is the SEPARATE
#                            `rep` column (r1/r2), fully crossed with startID.
#   Weinreb   T1 / T2 / T3   culture days 2/4/6 of ONE expansion -- a real
#                            timepoint series, so it is named like Michaels.
#   Bandler   E13 / E14      STICR INJECTION age, harvested P7 in both. Different
#                            embryos per age (the 8 libraries are perfectly NESTED
#                            in label_age, 4 and 4), so there is no replicate axis
#                            at all -- nothing analogous to Harmange's r1/r2.
#   Mold      brain1/brain2  two different ANIMALS.
# Group names are SPELLED OUT, never abbreviated (house rule 5). The authors'
# own codes are kept only in the data columns; everything a reader sees goes
# through glabel(). Harmange's E/M is the authors' Rmd convention
# ("determine start condition (M = mixed  E = Enriched for primed)",
# 10X1_r1_r2_Analysis_unorm_sctrans.Rmd:114): E = wells seeded with cells
# FACS-enriched for primed cells, M = wells seeded with the mixed, unsorted
# population. It is a STARTING POPULATION, not a timepoint.
GROUP_DISPLAY = {
    # Weinreb's groups are the culture days 2/4/6; they are a genuine timepoint
    # series, so they take the same naming as Michaels' T1/T2 (user, 2026-09-14).
    # Capitalised: these strings are drawn as axis/row labels, where a label starts
    # a line of its own (user, 2026-10-27). Prose in the legends keeps lower case.
    "LARRY": {"2.0": "Timepoint 1", "4.0": "Timepoint 2", "6.0": "Timepoint 3"},
    "T-lineage": {"T1": "Timepoint 1", "T2": "Timepoint 2"},
    "Harmange": {"E": "Enriched", "M": "Mixed"},
    "Bandler": {"E13": "Embryonic day 13", "E14": "Embryonic day 14"},
}

# Datasets whose groups are combined into ONE bar instead of drawn separately.
# "intersect" = a gene counts for a method only if that method called it in EVERY
# group. For Harmange that is "called in both the enriched and the mixed starting
# population", a strictly harder test than either arm alone -- so its bars are not
# on the same footing as a per-timepoint bar elsewhere, and the caption has to say
# so. Set GROUP_COMBINE = {} (or pass combine="none") to draw E and M separately.
# DEFAULT: empty -- every dataset's groups are drawn separately. Harmange's
# intersection is a DIFFERENT quantity from every other bar in figure 1b (each of
# those is "one group, five methods, top-N, genes called by >=k"), so putting it on
# the same shared y axis invites a comparison that is not like for like. It also
# collapses Harmange's >=4/>=5 from 51/6 to 5/0, driven by GEMLI surviving both arms
# with only 16 genes and Power-Seek with 53 -- method disagreement, not absence of
# memory genes. Keep it for a supplementary panel or a sentence in the results.
#     GROUP_COMBINE = {"Harmange": "intersect"}     # <- to switch it on globally
# or per call: fig_agreement(..., combine="intersect")
GROUP_COMBINE = {}
COMBINE_SEP = " \u2229 "        # the label joiner, e.g. "E \u2229 M"


def dkey(name):
    """Internal dataset key for `name`, which may be the display (author) label."""
    if name in DATASETS or name in LINEAGE_DATASETS:
        return name
    back = {v: k for k, v in DATASET_LABELS.items()}
    return back.get(name, name)


def glabel(dataset, group):
    """Display name for one of `dataset`'s groups: T1/T2/T3 where the groups are
    timepoints, the raw value otherwise. Unknown datasets/groups pass through."""
    return GROUP_DISPLAY.get(dkey(dataset), {}).get(str(group), str(group))


# ── perturb-seq screens: what each one profiled, for LABEL_MODE == "celltype" ──
# Two screens share T cells and two share melanoma, which is the point of the
# pairing, so slabel() adds the author in brackets only where a cell type repeats.
PERTURB_SHORT = {"marson": "Zhu", "weinstock": "Weinstock", "xu": "Xu",
                 "frangieh": "Frangieh", "mold": "Shi", "bowness": "Bowness"}
PERTURB_CELLTYPE = {"Zhu": "T cells", "Weinstock": "T cells",
                    "Xu": "Melanoma", "Frangieh": "Melanoma",
                    "Shi": "Brain", "Bowness": "HSPCs"}


def slabel(screen, among=None, mode=None, fmt="{ct} ({author})"):
    """Display name for a perturb-seq screen.

    LABEL_MODE == "author"   : the short author name, as before.
    LABEL_MODE == "celltype" : the cell type, with the author in brackets ONLY when
                               another screen in `among` profiled the same one, e.g.
                               'T cells (Marson)' but plain 'Brain'.
    `screen` may be the internal key ('mold') or the short name ('Shi').
    `among` defaults to all six screens; pass the subset a panel actually draws.
    """
    key = PERTURB_SHORT.get(str(screen).lower(), screen)
    if (LABEL_MODE if mode is None else mode) != "celltype":
        return key
    ct = PERTURB_CELLTYPE.get(key)
    if not ct:
        return key
    pool = [PERTURB_SHORT.get(str(a).lower(), a) for a in (among or PERTURB_CELLTYPE)]
    shared = sum(1 for a in pool if PERTURB_CELLTYPE.get(a) == ct) > 1
    return fmt.format(ct=ct, author=key) if shared else ct


def ctrl_label(title, among=None):
    """Display label for a CTRL_SPEC title such as 'Marson (Progenitor T cells)'.

    The titles are KEYS (CTRL_GEMLI_FILES is keyed by them), so they never change;
    this is only what the figure prints.
    """
    author = str(title).split(" (")[0]
    if LABEL_MODE != "celltype":
        # the title is a KEY (CTRL_GEMLI_FILES is keyed by it), so rename only what
        # is printed
        return title.replace(author, PERTURB_SHORT.get(author.lower(), author), 1)
    return slabel(author, among=among)      # default: all six screens, as figure 2


def alabel(name):
    """The AUTHOR label, whatever LABEL_MODE says.

    For the one panel that names datasets by author on purpose (figure 1a's
    sunburst, whose next ring in already carries the tissue), so flipping the
    notebook to cell types leaves it alone.
    """
    return DATASET_LABELS.get(dkey(name), name)


def dlabel(name, celltype=False, author=False):
    """Display name for a dataset.

    LABEL_MODE == "author"   : first author, optionally with the cell type in
                               brackets when `celltype` is set.
    LABEL_MODE == "celltype" : the cell type alone, and `celltype` is ignored,
                               since "T cells (T cells)" helps nobody.
    Two datasets share a cell type (Mold and Bandler are both brain), so in
    celltype mode a figure that shows both must disambiguate them itself.
    """
    key = dkey(name)
    if LABEL_MODE == "celltype":
        short = DATASET_CELLTYPE_SHORT.get(key) or DATASET_CELLTYPE.get(key)
        if short:
            # `author` adds the first author in brackets, for panels where WHICH
            # lineage dataset the genes came from is part of the row's identity
            return (f"{short} ({DATASET_LABELS.get(key, key)})" if author else short)
    lab = DATASET_LABELS.get(name, name)
    ct = DATASET_CELLTYPE.get(name)
    return f"{lab} ({ct})" if celltype and ct else lab
SHOW_TITLES = False         # titles live in the caption, never in the figure


# superscript digits, so a power of ten stays TEXT in the SVG: mathtext is
# always exported as glyph outlines, which no vector editor can retype
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

METRIC_LABELS = {
    "mean_abs_lfc": "mean absolute log₂ fold change",
    "median_abs_lfc": "median absolute log₂ fold change",
    "max_abs_lfc":  "maximum absolute log fold change",
    "mean_lfc":     "mean log fold change",
    "median_abs_lfc": "median absolute log fold change",
    "n_sig":        "number of significant perturbations",
}


def metric_label(metric):
    """Axis label for a metric key, written out — house rule 5, no abbreviations."""
    return METRIC_LABELS.get(metric, str(metric).replace("_", " "))


def fsize(x):
    """Clamp a derived font size to the house minimum."""
    return max(float(x), MIN_FONTSIZE)


def in_order(names, order):
    """`names` sorted into `order`, with anything unlisted appended."""
    names = list(names)
    known = [n for n in order if n in names]
    return known + [n for n in names if n not in set(known)]


# method display name -> csv prefix, in METHOD_ORDER
# NOTE the display name is ClonePair (2026-10-26); the files on disk stay scg_*.csv
METHOD_PREFIX_ALL = {"CORAL": "coral", "GEMLI": "gemli",
                     "Power-Seek": "powerseek", "ClonePair": "scg"}
METHOD_PREFIX = {m: p for m, p in METHOD_PREFIX_ALL.items() if m not in METHOD_EXCLUDE}

# ---- palette -------------------------------------------------------------
C_OBS = "#B02318"          # observed / primary
C_OBS2 = "#E08A5F"         # second timepoint
C_NULL = "#7F8C8D"         # random baseline (expression-matched); was #9aa2ab until 2026-10-26
C_NULL2 = "#c9ced4"        # second (uniform) baseline
# A third shade, APPENDED so every two-timepoint panel keeps exactly the colours it
# had. Weinreb is the only dataset with three groups, and _agreement_panel cycles
# with j % len(colors) — with two entries its day 2 and day 6 bars came out the
# same red and could not be told apart. A muted brown rather than a darker red:
# it has to separate from BOTH the strong red and the salmon.
C_OBS3 = "#A9764F"
C_TPS = [C_OBS, C_OBS2, C_OBS3]

# ---- publication rcParams (applied inside an rc_context, never globally) --
# Text in an SVG must stay TEXT, not outlines: Illustrator and Inkscape can then
# restyle or re-flow a label, and the downstream layout program recognises it as
# text at all. Set on the global rcParams as well as in PUB_RC, because several
# notebook cells call fig.savefig() directly and so never see PUB_RC.
plt.rcParams["svg.fonttype"] = "none"
# ONE font family, not a stack. With a list, matplotlib writes
#   font-family: 'Helvetica', 'Arial', 'DejaVu Sans', sans-serif
# and Inkscape's Text-and-Font dialog cannot resolve a list: the text is real text,
# but the font field comes up empty and edits do not apply. A single name gives
#   font-family: 'Arial'
# which both Inkscape and Illustrator edit normally.
plt.rcParams["font.family"] = "Helvetica"
# Helvetica is not installed on this machine (only DejaVu, Cantarell, STIX), so
# matplotlib lays the text out with its fallback and writes 'Helvetica' into the
# SVG. That is what we want for the journal, and the vector editor renders the real
# font. The findfont message it logs on every figure is noise, so quiet it.
import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

PUB_RC = {
    "svg.fonttype": "none",
    "font.family": "Helvetica",     # a single family: see the note above
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "ytick.minor.size": 1.3, "ytick.minor.width": .5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,      # keep text as text in Illustrator
    "savefig.bbox": "tight",
}

_CACHE: dict = {}          # ("tables"|"universe"|"expr"|"ortho", ...) -> value


def clear_cache(what: str | None = None) -> None:
    """Drop the in-process caches. `what` in {'tables','universe','expr','ortho'}."""
    if what is None:
        _CACHE.clear()
    else:
        for k in [k for k in _CACHE if k[0] == what]:
            del _CACHE[k]


# ═══════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_dir(folder: str, tree: str = RESULTS_TREE, verbose: bool = True) -> str:
    """python_implements/LARRY holds only mold; the full set is in LARRY_backup_prehuman."""
    d = os.path.join(BENCH, tree, folder)
    if len(glob.glob(f"{d}/*_memory_genes_*.csv")) >= 2:
        return d
    alt = f"{d}_backup_prehuman"
    if len(glob.glob(f"{alt}/*_memory_genes_*.csv")) >= 2:
        if verbose:
            print(f"  [{folder}] incomplete under {tree}/ -> reading {os.path.basename(alt)}")
        return alt
    return d


def _resolve_suffix(d: str, suffix: str | None = None) -> str:
    """The timepoint suffix differs per dataset (T1_T2 / A_B / E_M / brain1_brain2 /
    E13_E14 / LARRY's 2.0_6.0_4.0). Some folders hold BOTH a per-timepoint and a
    pooled '_all' run -> take the one with the most timepoints, so glob order can
    never silently decide which suffix wins."""
    if suffix is not None:
        return suffix
    sufs = sorted({os.path.basename(f).split("_memory_genes_", 1)[1][:-4]
                   for f in glob.glob(f"{d}/*_memory_genes_*.csv") if "clones2-8" not in f})
    if not sufs:
        raise FileNotFoundError(f"no *_memory_genes_*.csv in {d}")
    return max(sufs, key=lambda s: (len(s.split("_")), s))


def _scg_called_only(df, method, dataset="", verbose=True):
    """SCG: keep only the genes it CALLED -- `is_biased == True`.

    `is_biased` IS the memory-gene call. `is_heritable` is a dead legacy column and
    must never be read (findingSCGs_melanoma.py writes it all-zeros by construction).

    This has to happen at load time because the SCG csvs are inconsistent about it:
    findingSCGs.py writes `df[df['is_biased']]`, already filtered, while
    findingSCGs_melanoma.py keeps "is_biased as a column, not a filter" and writes
    every gene TESTED. On clones2-8: T-lineage (5,211) and LARRY (3,065) are 100%
    biased, but Harmange is 7,709/26,253, Bandler 3,697/18,193 and Mold 549/31,234.

    The uncalled rows are NOT parked at the end -- they interleave through the rank
    order, so a top-N of an unfiltered file is mostly genes SCG never called. Before
    this filter, Mold brain1's top 300 held only 63 called genes. Added 2026-09-10;
    build_consensus.filter_called is the same rule on the consensus side.
    """
    if method != "ClonePair" or "is_biased" not in df.columns:
        return df
    keep = df["is_biased"].astype(bool)
    n = int(len(df) - keep.sum())
    if n and verbose:
        print(f"  [{dataset}] SCG: dropped {n} of {len(df)} rows with is_biased == "
              f"False -> {int(keep.sum())} called genes")
    return df[keep].reset_index(drop=True)


def load_tables(dataset: str, methods=None, tree: str = RESULTS_TREE,
                scg_variant: str = "", suffix: str | None = None,
                folder: str | None = None, verbose: bool = True, cache: bool = True):
    """Per-method result tables for one dataset.

    dataset     : key of DATASETS, or any folder name if `folder` is given.
    methods     : subset/order of METHOD_PREFIX keys. None = all available.
    scg_variant : "" = canonical clone_size==2; "clones2-8_" = the widened variant.
                  Only ever applied to SCG — the other methods have no variant.
    suffix      : force a suffix (e.g. "all" for Harmange's pooled run).

    Returns (tables: dict[str, DataFrame], timepoints: list, suffix: str).
    `timepoints` are the values every loaded method shares, sorted (plain sort is
    chronological for all of T1/T2, A/B, E13/E14 and LARRY's numeric 2.0/4.0/6.0).
    """
    key = ("tables", dataset, tree, scg_variant, suffix, folder, tuple(methods or ()))
    if cache and key in _CACHE:
        return _CACHE[key]

    folder = folder or DATASETS.get(dataset, {}).get("folder", dataset)
    d = _resolve_dir(folder, tree, verbose)
    suf = _resolve_suffix(d, suffix)

    want = list(methods) if methods else list(METHOD_PREFIX)
    tabs, missing = {}, []
    for name in want:
        pre = METHOD_PREFIX[name]
        v = scg_variant if name == "ClonePair" else ""
        p = f"{d}/{pre}_memory_genes_{v}{suf}.csv"
        if os.path.exists(p):
            tabs[name] = _scg_called_only(pd.read_csv(p), name, dataset, verbose)
        else:
            missing.append(name)
    if not tabs:
        raise FileNotFoundError(f"[{dataset}] no method csv matched suffix '{suf}' in {d}")
    if missing and verbose:
        print(f"  [{dataset}] skipping (no csv): {', '.join(missing)}")

    tps = sorted(set.intersection(*(set(t["timepoint"].unique()) for t in tabs.values())))
    assert tps, f"[{dataset}] methods share no timepoint value — check timepoint dtypes"
    out = (tabs, tps, suf)
    if cache:
        _CACHE[key] = out
    return out


# Which SCG run every dataset uses unless a caller says otherwise. The widened
# clone_size 2-8 variant is the project default as of 2026-09-09 (team decision);
# "" is the older canonical clone_size==2 run, kept on disk and reachable with an
# explicit scg_variant="".
#
# Why the widened one: SCG is the only method whose output size is data-driven
# (Mold and Power-Seek write a fixed top-500, CORAL and GEMLI write thousands), so
# it alone sets the per-timepoint depth cap wherever clones are small. Canonical
# SCG calls 39 genes at Weinreb day 2 and 329 at Michaels T2; 2-8 lifts those to
# 145 and 2,260, which takes Michaels off the cap entirely and leaves Weinreb day 2
# as the only shallow arm anywhere. XD_METHODS_CONSENSUS already preferred it for
# the same reason.
SCG_VARIANT_DEFAULT = "clones2-8_"
SCG_VARIANT_BY_DATASET = {}          # per-dataset exceptions to the default


def scg_variant_for(dataset, scg_variant=None) -> str:
    """Resolve the SCG run to use for `dataset`.

    scg_variant : None  -> SCG_VARIANT_BY_DATASET, else SCG_VARIANT_DEFAULT
                  str   -> that variant for every dataset (explicit override;
                           "" forces the canonical clone_size==2 run)
                  dict  -> per dataset, keyed by internal name or author label
    """
    if isinstance(scg_variant, str):
        return scg_variant
    m = SCG_VARIANT_BY_DATASET if scg_variant is None else scg_variant
    m = {dkey(k): v for k, v in m.items()}
    return m.get(dkey(dataset), SCG_VARIANT_DEFAULT)


def depth_cap(tabs: dict, tps) -> int:
    """Deepest per-method cut that EVERY method can supply at EVERY timepoint."""
    return min(int((t["timepoint"] == tp).sum()) for t in tabs.values() for tp in tps)


def pooled_depth_cap(tabs: dict, tps) -> int:
    """Deepest cut every method can supply from its POOLED list.

    depth_cap() asks what each method has at EVERY timepoint separately, which is
    what a per-timepoint comparison needs. When the timepoints are pooled into one
    union list the binding constraint is the union's length instead, and that is
    usually far larger — SCG gives 329 genes at Michaels T2 but 1,387 pooled.
    """
    return min(len(pooled_genes(t, tps, 10 ** 9)) for t in tabs.values())


def top_genes(df: pd.DataFrame, timepoint, n: int) -> list:
    """The file's own row order is the ranking (best first) — never re-sorted here."""
    return list(dict.fromkeys(df.loc[df["timepoint"] == timepoint, "gene"]))[:n]


def pooled_genes(df: pd.DataFrame, tps, n: int) -> list:
    """One list per method across timepoints: interleave by within-timepoint rank,
    keep each gene's best rank, then truncate to n. Length stays n, so the same
    null applies as for a single timepoint."""
    d = df[df["timepoint"].isin(tps)].copy()
    d["_r"] = d.groupby("timepoint").cumcount()      # 0 = best inside its timepoint
    return list(dict.fromkeys(d.sort_values("_r", kind="stable")["gene"]))[:n]


# ---- gene universe / orthologs -------------------------------------------

def _read_var(dataset: str):
    """Symbols from the dataset's h5ad var (backed read, closed immediately)."""
    key = ("universe", dataset)
    if key in _CACHE:
        return _CACHE[key]
    import anndata as ad
    cfg = DATASETS[dataset]
    A = ad.read_h5ad(cfg["h5ad"], backed="r")
    ids = (A.var[cfg["var_col"]].astype(str).values if cfg["var_col"]
           else np.array(list(map(str, A.var_names))))
    A.file.close()
    _CACHE[key] = ids
    return ids


def universe_size(dataset: str) -> int:
    """|transcriptome| for the uniform null — the number of genes actually measured."""
    return len(_read_var(dataset))


def ortho_map(dataset: str, verbose: bool = True) -> dict:
    """That dataset's mouse->human ortholog map (falls back to BandlerBrain's).

    Built by mem_genes_benchmark/build_ortholog_maps.py from NCBI's curated
    gene_orthologs table, with the old mygene/HomoloGene map as a fallback for genes
    NCBI does not list (~900 Riken-style names in Weinreb). Human symbols in the file
    are already CURRENT NCBI symbols; to_human() normalises the human datasets the
    same way.

    The cache key carries the csv's mtime, so REBUILDING the ortholog file
    invalidates it. This is not defensive coding: the old intra_species notebook
    reported 230 shared genes because its `_OCACHE` still held the pre-rebuild
    BandlerBrain map hours after the csv had been rewritten on disk — a bigger map
    means a bigger shared universe, which moves every number in panel c.
    """
    p = DATASETS[dataset].get("ortho") or ORTHO_FALLBACK
    if not os.path.exists(p):
        if verbose:
            print(f"  [{dataset}] no own ortholog cache -> falling back to BandlerBrain's")
        p = ORTHO_FALLBACK
    key = ("ortho", p, os.path.getmtime(p))
    if key in _CACHE:
        return _CACHE[key]
    o = pd.read_csv(p).dropna(subset=["human"])
    m = dict(zip(o["mouse"].astype(str), o["human"].astype(str)))
    if verbose:
        import datetime as _dt
        stamp = _dt.datetime.fromtimestamp(key[2]).strftime("%Y-%m-%d %H:%M")
        print(f"  [{dataset}] ortholog map: {len(m)} mouse -> human "
              f"({os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}, built {stamp})")
    _CACHE[key] = m
    return m


HUMAN_GENE_INFO = f"{P.DATA}/gene_features/raw/Homo_sapiens.gene_info.gz"


def human_alias():
    """(current human symbols, unambiguous synonym -> current symbol), from NCBI."""
    if "human_alias" in _CACHE:
        return _CACHE["human_alias"]
    from collections import defaultdict
    d = pd.read_csv(HUMAN_GENE_INFO, sep="\t", usecols=["Symbol", "Synonyms"])
    cur = set(d.Symbol)
    syn = defaultdict(set)
    for sym, ss in zip(d.Symbol, d.Synonyms):
        if isinstance(ss, str) and ss != "-":
            for a in ss.split("|"):
                syn[a].add(sym)
    out = (cur, {a: next(iter(v)) for a, v in syn.items() if len(v) == 1})
    _CACHE["human_alias"] = out
    return out


def human_symbol(g):
    """Current NCBI symbol for a human gene (AARS -> AARS1, CYR61 -> CCN1).

    The human datasets were annotated with older GENCODE releases while the mouse
    ortholog maps emit CURRENT symbols, so without this the same gene counts twice and
    drops out of the shared universe. Only UNAMBIGUOUS synonyms are renamed; a symbol
    that is a synonym of several genes is left alone.
    """
    cur, syn = human_alias()
    g = str(g)
    return g if g in cur else syn.get(g, g)


def to_human(genes, dataset: str, verbose: bool = True) -> set:
    """Human symbols, normalised to the CURRENT NCBI symbol. Mouse datasets go through
    that dataset's mouse2human_orthologs.csv first (NCBI curated pairs, see
    mem_genes_benchmark/build_ortholog_maps.py). NOTE the h5ad 'human_symbol' columns
    are passthrough copies of the mouse symbol (Sox2 -> Sox2), never a mapping — so
    they are deliberately not used."""
    if DATASETS[dataset]["species"] == "human":
        return {human_symbol(g) for g in genes}
    m = ortho_map(dataset, verbose)
    return {human_symbol(m[str(g)]) for g in genes if str(g) in m}


# ═══════════════════════════════════════════════════════════════════════════
#  Small shared plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

# "tight" trims/extends every saved file to its drawn content, so two panels built at
# the same figsize come out at different sizes. None keeps the EXACT figsize: set it
# when panels are tiled on a page grid, so they butt together without rescaling (and
# rescaling is what makes font sizes drift between panels).
SAVE_BBOX = "tight"


def save_fig(fig, save, default_name: str, dpi: int = 400, outdir: str = None,
             formats=("png",), verbose: bool = True, also_svg=None, svgdir=None):
    """save=True -> <outdir>/<default_name>.<fmt> for every fmt; save=str/list -> those
    paths verbatim (their own extensions win).

    A vector copy of every figure also lands in `svgdir` (default SVGDIR), named
    after the raster file's stem. Pass also_svg=False to skip it for one call.
    """
    if not save:
        return []
    # None cannot be passed through: PUB_RC sets savefig.bbox="tight" and matplotlib
    # falls back to the rc value when bbox_inches is None. The figure's own bounds are
    # what "exact size" actually means.
    bbox = SAVE_BBOX if SAVE_BBOX is not None else fig.bbox_inches
    # read at call time: use_top_n() moves OUTDIR, and a default bound at import
    # would keep writing into the previous cutoff's folder
    outdir = OUTDIR if outdir is None else outdir
    _check_outdir(outdir)
    paths = ([os.path.join(outdir, f"{default_name}.{f}") for f in formats]
             if save is True else ([save] if isinstance(save, str) else list(save)))
    for p in paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches=bbox)
        if verbose:
            print("saved", p)

    if ALSO_SVG if also_svg is None else also_svg:
        sd = svgdir or SVGDIR
        os.makedirs(sd, exist_ok=True)
        seen = {os.path.abspath(p) for p in paths}
        for p in list(paths):
            if p.lower().endswith(".svg"):
                continue                      # already vector; do not duplicate it
            q = os.path.join(sd, os.path.splitext(os.path.basename(p))[0] + ".svg")
            if os.path.abspath(q) in seen:
                continue
            fig.savefig(q, bbox_inches=bbox)
            seen.add(os.path.abspath(q))
            paths.append(q)
            if verbose:
                print("saved", q)
    return paths


ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]


def panel_letters_for(spec, n):
    """'roman' -> i, ii, iii...; 'alpha' -> a, b, c...; a list is passed through."""
    if spec is None or spec is False:
        return None
    if spec is True or spec == "roman":
        return ROMAN[:n]
    if spec == "alpha":
        return [chr(ord("a") + i) for i in range(n)]
    return list(spec)[:n]


def _fig_bboxes(fig, axes):
    """Each axes' DRAWN box in figure coords. Not get_position(): an imshow with
    equal aspect shrinks its box inside that rect, and titles/letters are placed
    against the drawn box, which is why they end up on different lines."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    out = []
    for a in axes:
        bb = a.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        out.append((x0, y0, x1, y1))
    return out


def align_titles(fig, axes, dy=.012):
    """Put every panel title on one baseline.

    The titles are MOVED off the axes and redrawn as figure text. Repositioning
    ax.title in axes coords does not survive saving: an equal-aspect imshow
    re-runs apply_aspect() on the next draw, its box changes, and the title (being
    in axes coords) moves with it. Figure coords are immune to that.
    """
    axes = [a for a in axes if a is not None and a.get_title()]
    if not axes:
        return None
    boxes = _fig_bboxes(fig, axes)
    ytarget = max(b[3] for b in boxes) + dy
    for a, (x0, y0, x1, y1) in zip(axes, boxes):
        t = a.title
        fig.text((x0 + x1) / 2, ytarget, t.get_text(), ha="center", va="bottom",
                 fontsize=t.get_fontsize(), fontweight=t.get_fontweight(),
                 color=t.get_color())
        a.set_title("")
    return ytarget


def align_panel_letters(fig, axes, letters, fontsize=12, dx=.012, dy=None,
                        weight="bold", above_titles=True, y=None):
    """Place panel letters on ONE baseline, in FIGURE coordinates.

    Axes-relative placement (panel_letter) puts each letter at its own axes' top,
    so panels with different drawn boxes end up with the letters at visibly
    different heights. Anchored to the highest drawn top instead.
    """
    axes = [a for a in axes if a is not None]
    boxes = _fig_bboxes(fig, axes)
    if y is None:                     # y given = share the titles' baseline
        y = max(b[3] for b in boxes)
        if above_titles:
            y += .048 + (fontsize / 800.)
        y += (dy if dy is not None else .010)
    return [fig.text(b[0] - dx, y, l, fontsize=fontsize, fontweight=weight,
                     va="bottom", ha="left") for b, l in zip(boxes, letters)]


def panel_letter(ax, letter: str, x: float = -.16, y: float = 1.10,
                 fontsize: float = 9, weight: str = "bold"):
    """Nature-style bold panel letter in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight=weight, va="top", ha="left")


def _fmt(v: float) -> str:
    return "0" if v == 0 else (f"{v:.2g}" if v < 10 else f"{v:.0f}")


def zscore(observed: float, null: np.ndarray):
    """(observed - null mean) / null sd.

    Returns +/-inf when the null has ZERO spread — which happens for real at the
    deep thresholds (every one of 1000 draws gives 0 genes shared by >=4 datasets).
    That is not a huge z, it is an undefined one: report the permutation p there
    instead, and never print a z that a reader would take as a measured effect size.
    """
    sd = float(np.std(null, ddof=1))
    if sd == 0:
        return np.inf if observed > np.mean(null) else (-np.inf if observed < np.mean(null) else np.nan)
    return (observed - float(np.mean(null))) / sd


Z_DECIMALS = 1          # decimals on every printed/plotted z


def fold_label(observed, null, cap=1000, compact=False):
    """Enrichment over the null, formatted for a figure.

    When the null mean is exactly 0 the ratio is undefined, but it is still
    BOUNDED: 0 events in n draws puts a 95% upper bound of 3/n on the rate (the
    rule of three), so the enrichment is at least observed * n / 3. That bound is
    what gets reported, which is why ">1000x" appears where the null never once
    produced a gene rather than a blank.

    Above `cap` the ratio is not meaningfully estimated from n draws either, so it
    is shown as a bound there too.

    compact : render a capped bound as ">10^3x" rather than ">1 000x". Four glyphs
              instead of eight, which is what lets the label survive a 45 mm panel.
    """
    null = np.asarray(null, float)
    m = float(null.mean())
    bounded = m == 0
    if bounded:
        m = 3.0 / max(len(null), 1)          # rule of three, 95% upper bound
    if observed == 0:
        return "—"                            # nothing observed: no ratio to give
    f = observed / m
    if f >= cap:
        if compact:
            e = int(round(np.log10(cap)))
            return (f">10{str(e).translate(_SUPERSCRIPT)}×" if 10 ** e == cap
            else f">{cap:,.0f}×")
        return f">{cap:,.0f}×".replace(",", " ")
    return f"{'>' if bounded else ''}" + (f"{f:.0f}×" if f >= 10 else f"{f:.1f}×")


def _fmt_z(z: float, decimals: int = None) -> str:
    if not np.isfinite(z):
        return "z = n/a" if np.isnan(z) else "z = ∞"      # null sd == 0
    return f"z = {z:.{Z_DECIMALS if decimals is None else decimals}f}"


def _annot_metric(observed: float, null: np.ndarray, metric: str = "z") -> str:
    """What goes under the tick. Default is the plain z = (x - mu) / sigma against
    the permutation null. Caveat worth keeping in mind when writing the legend:
    only the >=2 null is actually normal (skew 0.09, Shapiro p = 0.12). At >=4 it
    takes just the values 0/1/2 (91% zeros), so sigma = 0.31 genes and z should be
    read as a standardised distance, not converted to a normal-tail p-value — the
    permutation p in stats[] is the test."""
    m = float(np.mean(null))
    fold = observed / m if m > 0 else np.inf
    fold_s = ("fold > " + f"{observed / (1.0 / len(null)):.0f}" if not np.isfinite(fold)
              else f"{fold:.1f}×" if fold < 100 else f"{fold:.0f}×")
    if metric == "fold":
        return fold_s
    if metric == "z":
        return _fmt_z(zscore(observed, null))
    if metric in ("z+fold", "both"):
        return f"{fold_s}\n{_fmt_z(zscore(observed, null))}"
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL B — method agreement vs a random baseline
# ═══════════════════════════════════════════════════════════════════════════

def _agreement_stats(label, thresholds, n_perm, seed, top_n, by_timepoint,
                     methods, tree, scg_variant, verbose, combine=None):
    var = scg_variant_for(label, scg_variant)
    if var and verbose:
        print(f"  [{label}] SCG variant '{var.rstrip('_')}' "
              "(see SCG_VARIANT_DEFAULT)")
    tabs, tps, suf = load_tables(label, methods=methods, tree=tree,
                                 scg_variant=var, verbose=verbose)
    cap = depth_cap(tabs, tps)
    U = universe_size(label)
    groups = list(tps) if by_timepoint else ["pooled"]

    # Depth is PER METHOD, per group (2026-10-27). Each method contributes its own
    # top_n, or everything it called if it called fewer. The old rule cut EVERY
    # method to the shallowest one, so Weinreb day 2 (SCG 2-8 calls 143) also cut
    # CORAL/GEMLI/Power-Seek to 143 -- the one bar in the figure built from a
    # different definition than its neighbours, and than Fig 2's gene sets
    # (build_consensus_gene_sets never capped). `caps` keeps the shallowest method
    # per group, for reporting only. top_n=None still means the deepest cut every
    # method can supply, applied equally.
    depth = ({g: [int((t["timepoint"] == g).sum()) for t in tabs.values()]
              for g in groups} if by_timepoint
             else {"pooled": [len(pooled_genes(t, tps, 10 ** 9)) for t in tabs.values()]})
    caps = {g: min(d) for g, d in depth.items()}
    Km = {g: (tuple(caps[g] for _ in d) if top_n is None
              else tuple(min(int(top_n), x) for x in d)) for g, d in depth.items()}
    K = {g: max(v) for g, v in Km.items()}               # nominal depth, for labels
    Kkey = {g: tuple(sorted(v)) for g, v in Km.items()}  # what the null depends on
    if verbose and top_n is not None:
        short = [g for g in groups if top_n > caps[g]]
        if short:
            print(f"  [{label}] top_n={top_n}: "
                  + ", ".join(f"{g} " + "/".join(f"{m} {n}" for m, n in zip(tabs, Km[g])
                                                if n < top_n)
                              for g in short)
                  + " -> those methods contribute all they called")

    # null: every method keeps its OWN list length but draws it at random from the
    # transcriptome. It depends only on (U, the per-method lengths), so it is drawn
    # once per distinct length profile and shared by every group with that profile.
    rng = np.random.default_rng(seed)
    byK = {}
    for key in sorted(set(Kkey.values())):
        nl = {k: np.empty(n_perm, int) for k in thresholds}
        for i in range(n_perm):
            h = Counter()
            for kv in key:
                h.update(rng.choice(U, size=kv, replace=False))
            v = np.fromiter(h.values(), int)
            for k in thresholds:
                nl[k][i] = (v >= k).sum()
        byK[key] = nl

    obs, members = {}, {}
    for g in groups:
        lists = ([top_genes(t, g, n) for t, n in zip(tabs.values(), Km[g])] if by_timepoint
                 else [pooled_genes(t, tps, n) for t, n in zip(tabs.values(), Km[g])])
        c = Counter(x for l in lists for x in l)
        v = np.fromiter(c.values(), int)
        obs[g] = {k: int((v >= k).sum()) for k in thresholds}
        members[g] = {k: sorted(x for x, n in c.items() if n >= k) for k in thresholds}

    # ---- optional: collapse the groups into ONE intersected bar --------------
    # A gene counts for a method only if that method called it in EVERY group. The
    # null has to be simulated rather than reused: after intersecting, a method's
    # effective list is far shorter than K and its length is itself random, so the
    # "draw K per method" null no longer describes this bar. Here the WHOLE pipeline
    # is nulled -- draw K genes per method PER GROUP, intersect, then count -- which
    # is the same question the observed bar answers.
    mode = GROUP_COMBINE.get(dkey(label)) if combine is None else combine
    if mode in (None, "none", False, "") or not by_timepoint or len(groups) < 2:
        mode = None
    combine_n = None
    if mode == "intersect":
        gname = COMBINE_SEP.join(glabel(label, g) for g in groups)
        per_method = [set.intersection(*(set(top_genes(t, g, Km[g][j])) for g in groups))
                      for j, t in enumerate(tabs.values())]
        combine_n = {m: len(s) for m, s in zip(tabs, per_method)}
        c = Counter(x for s in per_method for x in s)
        v = np.fromiter(c.values(), int)
        obs = {gname: {k: int((v >= k).sum()) for k in thresholds}}
        members = {gname: {k: sorted(x for x, n in c.items() if n >= k)
                           for k in thresholds}}
        rng2 = np.random.default_rng(seed)
        nl = {k: np.empty(n_perm, int) for k in thresholds}
        Kg = [K[g] for g in groups]
        for i in range(n_perm):
            h = Counter()
            for j in range(len(tabs)):
                s = set.intersection(*(set(rng2.choice(U, size=Km[g][j], replace=False))
                                       for g in groups))
                h.update(s)
            vv = np.fromiter(h.values(), int) if h else np.zeros(0, int)
            for k in thresholds:
                nl[k][i] = int((vv >= k).sum())
        if verbose:
            print(f"  [{label}] groups intersected -> '{gname}'; genes per method "
                  + ", ".join(f"{m} {n}" for m, n in combine_n.items()))
        Ks = sorted(set(K.values()))
        km = tuple(min(Km[g][j] for g in groups) for j in range(len(tabs)))
        return dict(label=label, groups=[gname], tps=tps,
                    K={gname: min(Kg)}, Km={gname: km},
                    Kkey={gname: ("intersect",) + tuple(sorted(km))},
                    caps={gname: min(caps.values())},
                    cap=cap, U=U, Kmax=max(Ks),
                    Klab=(str(Ks[0]) if len(Ks) == 1 else f"{Ks[0]}\u2013{Ks[-1]}"),
                    suffix=suf, n_methods=len(tabs), methods=list(tabs), obs=obs,
                    null={gname: nl}, members=members, combine="intersect",
                    combine_n=combine_n, combine_groups=list(groups))

    Ks = sorted(set(K.values()))
    return dict(label=label, groups=groups, tps=tps, K=K, Km=Km, Kkey=Kkey,
                caps=caps, cap=cap, U=U,
                Kmax=max(Ks), Klab=(str(Ks[0]) if len(Ks) == 1
                                    else f"{Ks[0]}\u2013{Ks[-1]}"),
                suffix=suf, n_methods=len(tabs), methods=list(tabs), obs=obs,
                null={g: byK[Kkey[g]] for g in groups}, members=members,
                combine=None, combine_n=None, combine_groups=None)


def _kdepth(key):
    """Sort key for a null's depth: an int K, or a per-method length profile."""
    v = [x for x in (key if isinstance(key, tuple) else (key,)) if isinstance(x, int)]
    return (max(v), sum(v))


def _kfmt(key):
    """'300', or '143–300' when the methods ran to different depths."""
    v = [x for x in (key if isinstance(key, tuple) else (key,)) if isinstance(x, int)]
    return str(v[0]) if min(v) == max(v) else f"{min(v)}–{max(v)}"


def _agreement_panel(ax, S, thresholds, *, logy=True, ymax=None, floor=3e-3,
                     colors=None, null_color=C_NULL, bar_width=.8,
                     hatches=None, hatch_color="white", hatch_edge_lw=0.0,
                     annot_color=None, legend_handlelength=1.0,
                     legend_handleheight=None,
                     annotate="all", annot_fontsize=None, ci=(2.5, 97.5),
                     legend=True, legend_loc="upper right", legend_fontsize=None,
                     null_label="Random", obs_label="Overlap", ylim=None,
                     null_bars="max",
                     enrich="fold", enrich_fontsize=None, enrich_cap=1000,
                     enrich_compact=False, show_thresholds=True,
                     threshold_below=False, yticklabels=True,
                     separators=True, sep_color="0.75", sep_lw=.6,
                     fs=11, headroom=1.34):
    """A dataset's groups (timepoints, or one pooled bar) plus ONE null bar per
    threshold. The null only depends on K, |universe| and the method count, so it
    is identical across a dataset's timepoints — drawing it once is not a
    simplification, it is the same number.

    annotate : "all" (default — counts on the observed bars AND the null's expected
               value, which is otherwise unreadable: at >=3 the null bar is a few
               pixels tall and at >=4 it is invisible), "observed", or None.
    enrich   : "fold" | "z" | "z+fold" | None — printed under each ≥k tick, so the
               reader gets the effect size without a second panel.
    """
    colors = colors or C_TPS
    gs = S["groups"]

    def klab(k):
        """'>=k', or just 'k' when k IS every method (there is no deeper level)"""
        return f"{k}" if k >= S["n_methods"] else f"≥{k}"
    # one null bar per DISTINCT depth: the null depends only on (K, |universe|,
    # n_methods), so groups that ran equally deep share a bar. They normally all do;
    # a dataset with one shallow arm (Weinreb day 2) is the exception and gets two.
    # nulls are told apart by the per-method length PROFILE (S["Kkey"]), not the
    # nominal K: since depth went per-method, Weinreb day 2 is nominally K=300 like
    # days 4 and 6 but its null is drawn with SCG at 143.
    NK = S.get("Kkey", S["K"])
    Ks = list(dict.fromkeys(NK[g] for g in gs))
    # null_bars: 'max' (default) draws ONE bar, at the depth most of the panel runs
    # at; 'each' draws one per distinct K; an int picks a specific K. With 'max' a
    # group that ran shallower (Weinreb day 2, K=145) is plotted beside a null that
    # is NOT its own — say so in the caption. The enrichment printed under each tick
    # is unaffected either way: it always pairs the mean observed with the mean null
    # over the same groups, each at its own K.
    # "group" pairs every observed bar with its OWN null bar, side by side, instead
    # of one shared null at the end. The paired nulls are usually the SAME number
    # (the null depends only on K, |universe| and the method count, so groups that
    # ran equally deep share it) -- the pairing is for reading, not new information,
    # and the caption should say so rather than imply three independent controls.
    if null_bars == "group":
        slots = [t for g in gs for t in (("obs", g), ("null", g))]
        drawK = Ks
    else:
        if null_bars == "each":
            drawK = Ks
        elif null_bars == "max":
            drawK = [max(Ks, key=_kdepth)]
        else:
            drawK = [next(Kv for Kv in Ks if _kdepth(Kv)[0] == int(null_bars))]
        slots = ([("obs", g) for g in gs]
                 + [("null", next(g for g in gs if NK[g] == Kv)) for Kv in drawK])
    nbar = len(slots)
    W = bar_width / nbar
    x = np.arange(len(thresholds))
    off = (np.arange(nbar) - (nbar - 1) / 2) * W
    base = floor if logy else 0.0
    afs = annot_fontsize or fs - 2
    efs = enrich_fontsize or fs - 2

    jn = -1
    for slot, (kind, g) in enumerate(slots):
        if kind == "obs":
            j = gs.index(g)
            o = np.array([S["obs"][g][k] for k in thresholds], float)
            col = colors[j % len(colors)] if len(gs) > 1 else colors[0]
            lab = obs_label if g == "pooled" else glabel(S["label"], g)
            # hatch instead of hue: a dataset's groups are the SAME measurement, so
            # one colour plus a texture per group says that, and the distinction
            # survives greyscale. A hatch takes its colour from the EDGE, so
            # edgecolor is the hatch colour while the outline stays at zero width.
            hat = hatches[j % len(hatches)] if hatches else None
            # matplotlib draws the hatch AND the outline in edgecolor, so a non-zero
            # hatch_edge_lw gives the bar a border in the hatch colour. That border is
            # what makes a pale fill readable: without it a lightly hatched white bar
            # has no edge at all.
            ax.bar(x + off[slot], o - base, W, bottom=base, color=col, zorder=3,
                   label=lab, hatch=hat,
                   edgecolor=(hatch_color if hat else "none"),
                   linewidth=(hatch_edge_lw if hat else 0))
            if annotate:
                # the count label defaults to the BAR colour, which is invisible once
                # the fill is white; annot_color overrides it
                acol = annot_color or col
                for xi, val in zip(x, o):
                    ax.annotate(f"{val:.0f}", (xi + off[slot], val),
                                textcoords="offset points", xytext=(0, 2.5),
                                ha="center", fontsize=afs, color=acol)
            continue
        jn += 1
        Kv = _kfmt(NK[g])
        nl = S["null"][g]
        m = np.array([nl[k].mean() for k in thresholds])
        hi = np.array([np.percentile(nl[k], ci[1]) for k in thresholds])
        # mask only on a LOG axis, where a 0 cannot be placed. On a linear axis keep
        # every bar: a zero-height bar draws nothing but keeps the BarContainer
        # non-empty, and an EMPTY container makes matplotlib colour the legend swatch
        # default-blue instead of the null grey (seen when a null is 0 at every
        # threshold, e.g. any intersected group).
        drawn = (m > floor) if logy else np.ones(len(m), bool)
        xn = x + off[slot]
        # name the depth only when more than one DISTINCT depth is drawn. Under
        # null_bars="group" the bars repeat one number next to each observed bar, so
        # they share a single legend entry and only the first one carries it --
        # three identical "Random" keys would read as three different controls.
        if null_bars == "group":
            nlab = null_label if jn == 0 else None
            shade = null_color
        else:
            nlab = null_label if len(drawK) == 1 else f"{null_label} (K={Kv})"
            shade = null_color if jn == 0 else _lighten(null_color, .35 * jn)
        ax.bar(xn[drawn], m[drawn] - base, W, bottom=base, color=shade,
               zorder=3, label=nlab)
        ax.errorbar(xn[drawn], m[drawn],
                    yerr=[np.zeros(int(drawn.sum())), np.maximum(hi - m, 0)[drawn]],
                    fmt="none", ecolor="0.35", capsize=2, lw=.9, zorder=4)
        if annotate == "all":
            for xi, val, h in zip(xn, m, hi):
                lab = ("0" if val == 0 else f"{val:.2g}" if val < 1 else
                       f"{val:.1f}" if val < 10 else f"{val:.0f}")
                ytop = ((floor * 8 if val <= floor else max(val, h)) if logy
                        else max(val, h))
                ax.annotate(lab, (xi, ytop), textcoords="offset points",
                            xytext=(0, 2.5), ha="center", fontsize=afs, color="0.35",
                            clip_on=False)

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(*(ylim or (floor, 3000)))
    else:
        ax.set_ylim(*(ylim or (0, (ymax or 1) * headroom)))
    ax.set_xlim(-.6, len(thresholds) - .4)
    ax.set_xticks(x)
    # dashed lines BETWEEN the threshold groups (>=2 | >=3 | ...), behind the bars
    if separators:
        for xs in np.arange(len(thresholds) - 1) + .5:
            ax.axvline(xs, color=sep_color, lw=sep_lw, ls=(0, (3, 2)), zorder=1)

    # enrichment under the tick: the point of the panel is "how far above chance",
    # and a bar chart alone cannot show a 500-fold ratio on a linear axis.
    if enrich:
        pooled = [np.mean([S["obs"][g][k] for g in gs]) for k in thresholds]
        # pair the mean observed with the mean null OVER THE SAME GROUPS. When every
        # group ran at one depth this is the single null repeated, so the printed
        # enrichment is byte-for-byte what it was before per-group depth existed.
        ncat = {k: np.concatenate([S["null"][g][k] for g in gs]) for k in thresholds}
        lab = []
        for k, o in zip(thresholds, pooled):
            parts = []
            if "fold" in enrich:
                parts.append(fold_label(o, ncat[k], enrich_cap,
                                        compact=enrich_compact))
            if "z" in enrich:
                parts.append(_fmt_z(zscore(o, ncat[k])))
            thr = klab(k) if show_thresholds else ""
            block = "\n".join(parts)
            lab.append("\n".join(x for x in ((block, thr) if threshold_below
                                             else (thr, block)) if x))
        ax.set_xticklabels(lab, fontsize=efs)
        for t in ax.get_xticklabels():
            t.set_linespacing(1.35)
    else:
        ax.set_xticklabels([klab(k) if show_thresholds else ""
                            for k in thresholds], fontsize=fsize(fs - 1))
    ax.tick_params(axis="y", labelsize=fs - 1)
    if not yticklabels:                      # keep the tick marks, drop the numbers
        ax.set_yticklabels([])

    # legend per panel: group names differ per dataset (T1/T2/T3, E ∩ M,
    # brain1/brain2), so one shared legend would mislabel every panel but the first.
    # Handles are REORDERED so the groups come first and the null last: under
    # null_bars="group" the bars interleave, and matplotlib would otherwise print
    # "T1, Random, T2, Random, T3" in draw order.
    if legend:
        h, l = ax.get_legend_handles_labels()
        isnull = [s.startswith(null_label) for s in l]
        order = ([i for i, n in enumerate(isnull) if not n]
                 + [i for i, n in enumerate(isnull) if n])
        # a hatched swatch needs a BIGGER key than a plain colour one: at the
        # default 1.0 x 0.7 a cross-hatch shows one or two crossings and reads as
        # noise, so both dimensions are adjustable
        _hh = {} if legend_handleheight is None else dict(handleheight=legend_handleheight)
        leg = ax.legend([h[i] for i in order], [l[i] for i in order],
                        frameon=separators, fontsize=legend_fontsize or fs - 1,
                        loc=legend_loc, handlelength=legend_handlelength,
                        handletextpad=.5, borderaxespad=.2, labelspacing=.3, **_hh)
        if separators:
            # plain white box, no border: hides the dashed separators behind the text
            leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1)
            leg.set_zorder(6)


def fig_agreement(dataset="all", *, thresholds=None, top_n=300,
                  by_timepoint=True, n_perm=2000, seed=0, logy=False,
                  methods=None, tree=RESULTS_TREE, scg_variant=None,
                  combine=None,
                  ncol=None, panel_w=4.3, panel_h=3.6, sharey=True,
                  colors=None, null_color=C_NULL, floor=3e-3, bar_width=.8,
                  hatches=None, hatch_color="white", hatch_lw=.7,
                  hatch_edge_lw=0.0, annot_color=None,
                  legend_handlelength=1.0, legend_handleheight=None,
                  null_bars="max",
                  annotate="all", annot_fontsize=None, legend=True,
                  legend_loc="upper right", ylim=None, yticks=None, fs=11,
                  enrich="fold", enrich_fontsize=None, enrich_cap=1000,
                  enrich_compact=False, threshold_ticks="all",
                  threshold_below=False, yticklabels=True,
                  xlabel="methods calling the gene", ylabel="number of genes",
                  title_fmt="{label}", label_celltype=True,
                  suptitle="", title_weight="normal",
                  panel_letters="roman", rc=None, axes=None, margins=None,
                  separators=True,
                  save=None, dpi=300, show=True, verbose=True):
    """(b) Observed method agreement vs a random baseline, per consensus threshold.

    dataset      : one label, a list, or "all" -> one panel per dataset.
    thresholds   : consensus levels (a gene called by ≥k methods). None (default) =
                   2 .. number of methods, so it follows METHOD_EXCLUDE. The deepest
                   level is labelled "4" rather than "≥4": with four methods it means
                   ALL of them, and "≥" would suggest a fifth could exist.
    top_n        : genes per method. A method that called fewer contributes all it
                   called (per method, per group; NOT cut to the shallowest method,
                   since 2026-10-27). None = the deepest cut every method can supply.
    by_timepoint : True  -> one bar per timepoint against the shared null bar.
                   False -> timepoints merged per method by best rank and
                            re-truncated to top_n, giving a single "Overlap" bar.
    logy         : log y-axis keeps the (tiny) null visible at ≥4/≥5; linear makes
                   bar LENGTH proportional to the count. Both are honest, pick one.
    enrich       : "fold" (default) | "z" | "z+fold" | None. Printed under each ≥k
                   tick. A linear bar chart cannot show a 500-fold ratio, and the
                   null bar at ≥4/≥5 is invisible next to the observed one, so the
                   number is the only way that information reaches the reader.
    combine      : None (default) honours GROUP_COMBINE, which collapses Harmange's
                   E and M into ONE intersected bar — a gene counts for a method only
                   if that method called it in BOTH arms. Pass "none" to draw every
                   dataset's groups separately, or "intersect" to force it on. Note
                   E and M are starting populations (enriched vs mixed), NOT replicates
                   of each other, so the intersection means "called in both starting
                   populations", a strictly harder test than any single-group bar
                   elsewhere in the figure. Its null is simulated through the same
                   intersection, so observed-vs-null is still like for like, but the
                   COUNTS are not comparable to the other panels' — say so in the
                   caption, and consider sharey=False.
    null_bars    : how many random-baseline bars to draw. "group" gives every group
                   its own null bar, drawn immediately beside it — the readable form,
                   but be honest in the caption: the null depends only on (K, universe,
                   method count), so groups that ran equally deep get the SAME number
                   repeated, not independent controls. "max" (default) draws ONE, at
                   the depth most of the panel uses; "each" draws one per distinct K;
                   an int picks a K. Nulls are told apart by each method's list
                   length, so Weinreb day 2 (SCG supplies 143, the others 300) has its
                   own null; under "max" it sits beside one that is not its own. The enrichment under each tick always uses each
                   group's own null, whichever you pick.
    threshold_ticks : where the "at least k methods" tick labels appear — 'all'
                   (default), 'bottom' (only the bottom row, when panels share an
                   x axis) or 'none'. The enrichment line under the ticks is
                   unaffected and stays on every panel.
    threshold_below : put the enrichment line ABOVE the "at least k" label rather
                   than below it, so the enrichment sits nearest the bars.
    yticklabels  : False keeps the tick marks but drops the numbers, for when each
                   bar already carries its count. Pass ylabel=None as well, or the
                   axis is named but unreadable.
    yticks       : explicit y tick positions, e.g. (0, 200, 400). Use it to thin the
                   axis out when the panel is printed small. Keep them evenly
                   spaced — uneven ticks on a linear axis misread as a broken axis.
    fs           : base font size; everything else is derived from it.
    axes         : a pre-made axes array to draw into (for a composed figure).
    panel_letters: "roman" (default -> i, ii, iii, iv), "alpha", an explicit list,
                   or None.
    label_celltype : panel labels read "Author (cell type)" — set False for the
                   bare author name.
    rc           : extra rcParams merged over PUB_RC for this call only.
    separators   : dashed vertical lines between the threshold groups (>=2 | >=3 | ...).
                   True by default; False removes them.
    margins      : None (default) fits the layout with tight_layout, so the panel boxes
                   shrink or grow around each figure's own labels. A dict of
                   subplots_adjust arguments (left, right, bottom, top, wspace, hspace)
                   fixes them instead: two figures with the same figsize, grid and
                   margins get IDENTICAL panel boxes, so panels built with the same
                   settings sit side by side at the same size.

    Returns (fig, tidy DataFrame with observed / null_mean / null_p975 / p).
    """
    # in_order, not the dict's own order: the panels, the legend and every table
    # must agree on ONE dataset order, and that order is DATASET_ORDER.
    labels = in_order((active_datasets() if dataset == "all"
                       else [dataset] if isinstance(dataset, str) else list(dataset)),
                      DATASET_ORDER)
    thresholds = (list(thresholds) if thresholds is not None
                  else list(range(2, len(methods or METHOD_FILES) + 1)))
    S = {lab: _agreement_stats(lab, thresholds, n_perm, seed, top_n, by_timepoint,
                               methods, tree, scg_variant, verbose, combine)
         for lab in labels}
    if verbose:
        for lab in labels:
            s = S[lab]
            print(f"{lab:11s} K={s['Klab']:>7s} universe={s['U']:6d} "
                  f"methods={s['n_methods']}  "
                  + "  ".join(f"{g}(K={s['K'][g]}): "
                              + "/".join(str(s['obs'][g][k]) for k in thresholds)
                              for g in s["groups"]))

    n = len(labels)
    ncol = ncol or (1 if n == 1 else 2)
    nrow = int(np.ceil(n / ncol))
    ymax = max(S[l]["obs"][g][k] for l in labels for g in S[l]["groups"] for k in thresholds)

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fs + 1, "xtick.labelsize": fsize(fs - 1),
             "ytick.labelsize": fsize(fs - 1), "legend.fontsize": fsize(fs - 1),
             "axes.linewidth": .9, "xtick.major.width": .9, "ytick.major.width": .9,
             "ytick.minor.width": .6, "hatch.linewidth": hatch_lw, **(rc or {})}
    with plt.rc_context(style):
        if axes is None:
            fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                                     sharey=sharey, squeeze=False)
        else:
            axes = np.atleast_2d(axes)
            fig = axes.flat[0].get_figure()
        _pl = panel_letters_for(panel_letters, len(labels))
        for i, lab in enumerate(labels):
            ax = axes[i // ncol][i % ncol]
            _agreement_panel(ax, S[lab], thresholds, logy=logy, ymax=ymax, floor=floor,
                             colors=colors, null_color=null_color, bar_width=bar_width,
                             hatches=hatches, hatch_color=hatch_color,
                             hatch_edge_lw=hatch_edge_lw, annot_color=annot_color,
                             legend_handlelength=legend_handlelength,
                             legend_handleheight=legend_handleheight,
                             null_bars=null_bars,
                             annotate=annotate, annot_fontsize=annot_fontsize,
                             legend=legend, legend_loc=legend_loc, ylim=ylim,
                             enrich=enrich, enrich_fontsize=enrich_fontsize,
                             enrich_cap=enrich_cap,
                             enrich_compact=enrich_compact,
                             show_thresholds=(threshold_ticks == "all"
                                              or (threshold_ticks == "bottom"
                                                  and i // ncol == nrow - 1)),
                             threshold_below=threshold_below,
                             yticklabels=yticklabels, separators=separators, fs=fs)
            if title_fmt:
                ax.set_title(title_fmt.format(label=dlabel(lab, label_celltype),
                                              K=S[lab]["Klab"], cap=S[lab]["cap"]),
                             fontsize=fs + 1, fontweight=title_weight)
            if _pl:
                panel_letter(ax, _pl[i], fontsize=fs + 2)
            if i // ncol == nrow - 1 and xlabel:
                ax.set_xlabel(xlabel)
            if yticks is not None:
                ax.set_yticks(list(yticks))
            if i % ncol == 0 and ylabel:
                ax.set_ylabel(ylabel)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        if suptitle:
            fig.suptitle(suptitle, fontsize=fs + 1.5)
        if margins:
            fig.subplots_adjust(**margins)
        else:
            fig.tight_layout()

        tag = "all" if dataset == "all" else "_".join(labels).replace(" ", "-")
        save_fig(fig, save,
                 f"agreement_vs_null_{tag}_top{max(S[l]['Kmax'] for l in labels)}"
                 f"_{'bytp' if by_timepoint else 'pooled'}_{'log' if logy else 'lin'}",
                 dpi=dpi, verbose=verbose)
        if show:
            plt.show()

    tidy = pd.DataFrame([
        dict(dataset=lab, group=g, group_label=glabel(lab, g),
             combined=S[lab]["combine"],
             threshold=k, K=S[lab]["K"][g],
             K_methods=" / ".join(f"{m} {n}" for m, n in zip(S[lab]["methods"],
                                                             S[lab]["Km"][g])),
             cap=S[lab]["caps"][g], universe=S[lab]["U"],
             n_methods=S[lab]["n_methods"],
             observed=S[lab]["obs"][g][k],
             null_mean=S[lab]["null"][g][k].mean(),
             null_sd=float(np.std(S[lab]["null"][g][k], ddof=1)),
             z=zscore(S[lab]["obs"][g][k], S[lab]["null"][g][k]),
             fold=(S[lab]["obs"][g][k] / S[lab]["null"][g][k].mean()
                   if S[lab]["null"][g][k].mean() > 0 else np.inf),
             null_p975=np.percentile(S[lab]["null"][g][k], 97.5),
             null_max=S[lab]["null"][g][k].max(),
             p=((S[lab]["null"][g][k] >= S[lab]["obs"][g][k]).sum() + 1) / (n_perm + 1))
        for lab in labels for g in S[lab]["groups"] for k in thresholds])
    tidy.attrs["stats"] = S           # members / raw nulls, if you need the gene lists
    return fig, tidy


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL C — shared across datasets + pairwise overlap
# ═══════════════════════════════════════════════════════════════════════════

def cross_dataset_sets(datasets=None, *, min_methods=2, top_n=None,
                       tree=RESULTS_TREE, scg_variant=None, methods=None,
                       pooled_depth=False,
                       restrict_to_shared_universe=True, verbose=True):
    """Per-dataset consensus memory genes, mapped to human symbols and restricted
    to the genes measured in EVERY dataset — so 'not shared' can never just mean
    'not tested'.

    Returns (sets: dict[label, set], universe: set, info: DataFrame).
    """
    labels = active_datasets(datasets)
    raw, unis, rows = {}, {}, []
    for lab in labels:
        tabs, tps, suf = load_tables(lab, methods=methods, tree=tree,
                                     scg_variant=scg_variant_for(lab, scg_variant),
                                     verbose=verbose)
        # the sets below are POOLED across timepoints, so the pooled cap is the
        # honest limit; the per-timepoint cap would throw away genes every method
        # actually supplies
        cap = pooled_depth_cap(tabs, tps) if pooled_depth else depth_cap(tabs, tps)
        # PER-METHOD depth (2026-10-27): each method gives its own top_n, or all it
        # called if fewer -- pooled_genes truncates per method. It used to cut every
        # method to `cap` (the shallowest), which bound in Suppl. Fig 1d at top 1000
        # (Power-Seek pools 859-991). `cap` is kept for reporting only. The null
        # (_cross_null) redraws each dataset's ACTUAL set size, so needs no change.
        K = cap if top_n is None else int(top_n)
        if top_n is not None and top_n > cap and verbose:
            print(f"  [{lab}] top_n={top_n}: shallowest method has {cap} "
                  "-> it contributes all it called")
        cnt = Counter(g for t in tabs.values() for g in pooled_genes(t, tps, K))
        cons = [g for g, v in cnt.items() if v >= min_methods]
        raw[lab] = (to_human(cons, lab, verbose), len(cons), K, cap)
        unis[lab] = to_human(_read_var(lab), lab, verbose)

    universe = set.intersection(*unis.values()) if restrict_to_shared_universe \
        else set.union(*unis.values())
    sets = {lab: (raw[lab][0] & universe) for lab in labels}
    # unrestricted sets and each dataset's own universe, for PAIRWISE comparisons,
    # where the fair gene space is what the two datasets being compared both measure
    raw_sets = {lab: raw[lab][0] for lab in labels}
    for lab in labels:
        s, n_raw, K, cap = raw[lab]
        rows.append(dict(dataset=lab, K=K, cap=cap, n_consensus=n_raw,
                         n_human=len(s), n_in_universe=len(sets[lab])))
        if verbose:
            print(f"  {lab:11s} K={K}/{cap}  consensus(>={min_methods})={n_raw:4d} "
                  f"-> human {len(s):4d} -> in shared universe {len(sets[lab]):4d}")
    if verbose:
        print(f"shared universe: {len(universe)} genes measured in all {len(unis)} datasets")
    info = pd.DataFrame(rows)
    info.attrs["raw_sets"], info.attrs["universes"] = raw_sets, unis
    return sets, universe, info


def _mean_expr(label, universe_sorted):
    """Mean log-expression per gene, in human symbols, aligned to the shared
    universe. Chunked so a backed h5ad never lands in memory whole."""
    key = ("expr", label, len(universe_sorted), hash(tuple(universe_sorted[:50])))
    if key in _CACHE:
        return _CACHE[key]
    import anndata as ad
    import scipy.sparse as sp
    cfg = DATASETS[label]
    A = ad.read_h5ad(cfg["h5ad"], backed="r")
    n, s = A.n_obs, np.zeros(A.n_vars)
    for X, _, _ in A.chunked_X(4000):
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        s += np.asarray(X.sum(0)).ravel()
    ids = (A.var[cfg["var_col"]].astype(str).values if cfg["var_col"]
           else np.array(list(map(str, A.var_names))))
    A.file.close()
    df = pd.DataFrame({"id": ids, "m": s / n})
    if cfg["species"] == "mouse":
        mp = ortho_map(label, verbose=False)
        df["id"] = df["id"].map(lambda g: mp.get(g))
    df = df.dropna(subset=["id"])
    # SAME symbol normalisation as to_human(), or a universe entry (AARS1) cannot find
    # this dataset's older symbol (AARS) and its expression comes back NaN -- which the
    # expression-matched null then cannot bin
    df["id"] = df["id"].map(human_symbol)
    out = df.groupby("id")["m"].max().reindex(universe_sorted)
    _CACHE[key] = out
    return out


def _cross_null(sets, universe_sorted, thresholds, n_perm, matched=True,
                n_bins=20, seed=0):
    """matched=True redraws each dataset's set inside ITS OWN expression bins, so
    the null keeps the fact that memory callers favour well-expressed genes — and
    that highly expressed genes are the SAME genes in every tissue. That is the
    defensible baseline; the uniform one overstates the signal."""
    rng = np.random.default_rng(seed)
    U = len(universe_sorted)
    pos = {g: i for i, g in enumerate(universe_sorted)}
    out = {k: np.empty(n_perm, int) for k in thresholds}
    if matched:
        E = {l: _mean_expr(l, universe_sorted) for l in sets}
        bins = {l: pd.qcut(E[l].rank(method="first"), n_bins, labels=False).values for l in sets}
        memb = {l: [np.where(bins[l] == b)[0] for b in range(n_bins)] for l in sets}
        prof = {l: np.bincount(bins[l][[pos[g] for g in sets[l]]], minlength=n_bins)
                for l in sets}
    for i in range(n_perm):
        h = Counter()
        for l in sets:
            if matched:
                pick = np.concatenate([rng.choice(memb[l][b], c, replace=c > len(memb[l][b]))
                                       for b, c in enumerate(prof[l]) if c])
            else:
                pick = rng.choice(U, size=len(sets[l]), replace=False)
            h.update(pick.tolist())
        v = np.fromiter(h.values(), int)
        for k in thresholds:
            out[k][i] = (v >= k).sum()
    return out


def fig_cross_dataset(datasets=None, *, min_methods=2, top_n=None,
                      pooled_depth=False,
                      thresholds=(2, 3, 4), n_perm=1000, seed=0,
                      null_mode="matched", n_bins=20, logy=True,
                      tree=RESULTS_TREE, scg_variant=None, methods=None,
                      fs=11, figsize=(8.8, 3.9), width_ratios=(1, .98), wspace=.34,
                      obs_color=C_OBS, null_colors=(C_NULL, C_NULL2), floor=5e-3,
                      bar_width=.8, annotate=True, annot_fontsize=None,
                      null_annot_fmt=None,
                      cmap="Reds", show_diagonal=True, matrix_metric="overlap_coef",
                      chance_line=True, chance_label="chance", chance_color="0.25",
                      chance_side="left",
                      show_z=True, z_where="xtick", z_metric="fold",
                      z_fontsize=None, z_color="0.25",
                      panel_letters="roman", letter_dy=0., matrix_annot="both",
                      title_weight="normal",
                      titles=None,        # house style: no titles inside figures
                      xlabel="datasets sharing the gene", ylabel="number of genes",
                      obs_label="Observed", null_label_short=True,
                      cbar_label=None, cbar_ticks=None,
                      threshold_below=False, yticklabels=True,
                      show_core_caption=False, core_max=40,
                      legend_loc="upper right",
                      panel="both", pair_universe=True,
                      rc=None, axes=None, save=None, dpi=400, show=True, verbose=True):
    """(c) Cross-dataset memory-gene overlap: counts vs null (left) and the
    pairwise matrix (right).

    min_methods    : a gene is that dataset's memory gene if ≥ this many methods call it.
    pooled_depth   : cap top_n against each method's POOLED list rather than its
                     shallowest single timepoint. Lets all four datasets run at the
                     same K (500 is reachable everywhere); the default keeps the
                     stricter per-timepoint cap.
    null_mode      : 'matched' (expression-matched — the defensible one), 'uniform',
                     or 'both' (draws two null bars, for the supplement).
    logy           : log y-axis. False is linear — bar lengths become proportional
                     to the counts, but the null at ≥3/≥4 (4 and 0.1 genes) shrinks
                     to nothing.
    fs             : base font size; every other size is derived from it.
    matrix_metric  : 'overlap_coe' |A∩B|/min(|A|,|B|), 'jaccard', 'count', or
                     'fold' = observed shared genes / the hypergeometric expectation
                     |A||B|/N inside that pair's own shared universe, so 1 is chance.
    null_label_short : legend reads "Random" rather than "Random
                     (expression-matched)". The null IS expression-matched — say so
                     in the caption, since the panel no longer does.
    show_core_caption : OFF by default — a list of gene symbols under the figure is
                     caption or supplementary-table material, not a panel element.
    show_z         : annotate the enrichment against the FIRST null (matched,
                     unless null_mode='uniform').
    z_metric       : 'z'      (observed - null mean) / null sd  (DEFAULT)
                     'fold'   observed / null mean
                     'z+fold' both, stacked.
    pair_universe  : True (default) — each cell of (ii) is computed over the genes
                     measured in THOSE TWO datasets, not in all of them: a gene one
                     of the other datasets never measured is still a fair test of
                     this pair. The diagonal then shows each dataset's full set.
                     False = the old behaviour, every cell over the all-dataset
                     universe (panel (i) always uses that one — it counts genes
                     across all datasets at once).
    matrix_annot   : what each cell of (ii) prints — 'both' (default), 'n' (just the
                     shared-gene count, letting colour carry the coefficient),
                     'coef', or None.
    letter_dy      : shift the panel letters in FIGURE coordinates; negative moves
                     them down, which reclaims the headroom left by having no titles.
    cbar_ticks     : explicit colourbar ticks, e.g. (0, .1, .2). Values outside the
                     colour range are simply not drawn.
    threshold_below: in (i), put the enrichment ABOVE the "at least k" label.
    yticklabels    : False keeps (i)'s tick marks but drops the numbers.
    z_where        : 'xtick' (second line under each ≥k tick) | 'bar' (above the
                     observed bar) | None.
    panel          : "both" (default) | "bars" (only the shared-across-datasets
                     counts vs null) | "matrix" (only the pairwise overlap matrix).
                     A single panel gets its own figure of `figsize`, so pass a
                     size that suits one panel, and its own filename (the save stem
                     gains "_bars" / "_matrix"), so the two halves can be saved
                     side by side for different figures.
    axes           : (ax_left, ax_right) to draw into a composed figure; pass a
                     single ax to draw only that panel.

    Returns (fig, dict(sets=, matrix=DataFrame, core=[genes], stats=DataFrame)).
    """
    sets, universe, info = cross_dataset_sets(
        datasets, min_methods=min_methods, top_n=top_n, tree=tree,
        pooled_depth=pooled_depth,
        scg_variant=scg_variant, methods=methods, verbose=verbose)
    uni = sorted(universe)
    labs = list(sets)
    n = len(labs)
    thresholds = list(thresholds)

    cnt = Counter(g for s in sets.values() for g in s)
    obs = np.array([sum(v >= k for v in cnt.values()) for k in thresholds], float)
    core = sorted(g for g, v in cnt.items() if v >= max(thresholds))

    nulls = []
    if null_mode in ("matched", "both"):
        nulls.append(("Random" if null_label_short else
                      "Random (expression-matched)", "expression_matched",
                      _cross_null(sets, uni, thresholds, n_perm, True, n_bins, seed),
                      null_colors[0]))
    if null_mode in ("uniform", "both"):
        nulls.append(("Uniform null" if null_label_short else "Random (uniform)",
                      "uniform",
                      _cross_null(sets, uni, thresholds, n_perm, False, n_bins, seed),
                      null_colors[1 % len(null_colors)]))

    afs = annot_fontsize if annot_fontsize is not None else fs - 2
    zfs = z_fontsize if z_fontsize is not None else fs - 2
    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fs + 1, "xtick.labelsize": fsize(fs - 1),
             "ytick.labelsize": fsize(fs - 1), "legend.fontsize": fsize(fs - 1),
             "axes.linewidth": .9, "xtick.major.width": .9, "ytick.major.width": .9,
             "ytick.minor.width": .6, **(rc or {})}
    with plt.rc_context(style):
        if panel not in ("both", "bars", "matrix"):
            raise ValueError(f"panel must be 'both', 'bars' or 'matrix', got {panel!r}")
        if axes is None and panel != "both":
            fig, _ax = plt.subplots(1, 1, figsize=figsize)
            axa, axb = (_ax, None) if panel == "bars" else (None, _ax)
        elif axes is None:
            fig, (axa, axb) = plt.subplots(
                1, 2, figsize=figsize,
                gridspec_kw=dict(width_ratios=list(width_ratios), wspace=wspace))
        else:
            axa, axb = (axes if isinstance(axes, (tuple, list, np.ndarray)) and len(axes) == 2
                        else (axes, None))
            # axes=(None, ax) draws only the matrix into someone else's figure
            fig = (axa if axa is not None else axb).get_figure()

        def _klab(k):
            """'>=k', or just 'k' when k IS every dataset (no deeper level exists)"""
            return f"{k}" if k >= n else f"≥{k}"

        # ---- a : observed vs null ----
        if axa is not None:
            nb = len(nulls) + 1
            W = bar_width / nb
            x = np.arange(len(thresholds))
            off = (np.arange(nb) - (nb - 1) / 2) * W
            FL = floor if logy else 0.0
            axa.bar(x + off[0], obs - FL, W, bottom=FL, color=obs_color, zorder=3,
                    label=obs_label)
            if annotate:
                for xi, val in zip(x, obs):
                    axa.annotate(f"{val:.0f}", (xi + off[0], val), textcoords="offset points",
                                 xytext=(0, 3), ha="center", fontsize=afs,
                                 color=obs_color)
            for j, (nm, _key, nl, col) in enumerate(nulls, start=1):
                m = np.array([nl[k].mean() for k in thresholds])
                hi = np.array([np.percentile(nl[k], 97.5) for k in thresholds])
                d = (m > FL) if logy else (m > 0)
                axa.bar((x + off[j])[d], m[d] - FL, W, bottom=FL, color=col, zorder=3, label=nm)
                axa.errorbar((x + off[j])[d], m[d],
                             yerr=[np.zeros(d.sum()), np.maximum(hi - m, 0)[d]],
                             fmt="none", ecolor="0.35", capsize=1.8, lw=.7, zorder=4)
                if annotate:
                    for xi, val, h in zip(x, m, hi):
                        # on a linear axis every near-zero null sits on the baseline, so
                        # adjacent nulls' labels overprint -> stagger them by null index
                        dy = 2.5 if logy else 2.5 + 7 * (j - 1)
                        # default _fmt keeps 2 significant figures below 10 (0.023,
                        # 4.9). null_annot_fmt="{:.0f}" rounds the null labels to
                        # whole genes instead; note that collapses every expectation
                        # under 0.5 to "0", so the fold-enrichment stops being
                        # readable off the panel and has to live in the caption.
                        _lab = (_fmt(val) if null_annot_fmt is None
                                else null_annot_fmt.format(val))
                        axa.annotate(_lab, (xi + off[j], max(val, h, FL)),
                                     textcoords="offset points", xytext=(0, dy),
                                     ha="center", fontsize=afs, color="0.35",
                                     clip_on=False)
            if logy:
                axa.set_yscale("log"); axa.set_ylim(FL, obs.max() * 14)
            else:
                axa.set_ylim(0, obs.max() * 1.22)
            axa.set_xlim(-.6, len(thresholds) - .4)
            # z against the FIRST null (matched unless null_mode='uniform')
            zs = ([_annot_metric(obs[j], nulls[0][2][k], z_metric)
                   for j, k in enumerate(thresholds)] if (show_z and nulls) else [])
            axa.set_xticks(x)
            if zs and z_where == "xtick":
                axa.set_xticklabels([f"{z}\n{_klab(k)}" if threshold_below
                                     else f"{_klab(k)}\n{z}"
                                     for k, z in zip(thresholds, zs)],
                                    fontsize=zfs + 1)
            else:
                axa.set_xticklabels([_klab(k) for k in thresholds])
            if zs and z_where == "bar":
                for xi, val, z in zip(x, obs, zs):
                    axa.annotate(z, (xi + off[0], val), textcoords="offset points",
                                 xytext=(0, 13), ha="center", fontsize=zfs,
                                 color=z_color, clip_on=False)
            axa.set_xlabel(xlabel); axa.set_ylabel(ylabel)
            if not yticklabels:
                axa.set_yticklabels([])
                axa.set_yticklabels([], minor=True)
            axa.legend(loc=legend_loc, handlelength=1.0, handletextpad=.5, labelspacing=.3)
            if titles and titles[0]:
                axa.set_title(titles[0], pad=7, fontweight=title_weight)

        # ---- b : pairwise matrix ----
        M = np.full((n, n), np.nan); C = np.zeros((n, n), int)
        U2 = np.zeros((n, n), int)
        # expectation under random sets of the SAME sizes drawn from the pair's own
        # shared universe: |A||B|/N, the hypergeometric mean. 'fold' divides by it, so
        # 1 is chance and the colour answers "is this more than chance" by itself.
        E = np.full((n, n), np.nan); P = np.full((n, n), np.nan)
        from scipy.stats import hypergeom as _hg
        raw_sets = info.attrs.get("raw_sets")
        unis = info.attrs.get("universes")
        use_pair = bool(pair_universe and raw_sets and unis)
        for i, j in combinations(range(n), 2):
            if use_pair:
                u = unis[labs[i]] & unis[labs[j]]
                a, b = raw_sets[labs[i]] & u, raw_sets[labs[j]] & u
                U2[j, i] = len(u)
            else:
                a, b = sets[labs[i]], sets[labs[j]]
                U2[j, i] = len(universe)
            inter = len(a & b)
            _N = U2[j, i]
            _exp = len(a) * len(b) / _N if _N else np.nan
            E[j, i] = _exp
            P[j, i] = (_hg.sf(inter - 1, _N, len(a), len(b)) if _N else np.nan)
            val = (inter / min(len(a), len(b)) if matrix_metric == "overlap_coef" else
                   inter / len(a | b) if matrix_metric == "jaccard" else
                   (inter / _exp if _exp else np.nan) if matrix_metric == "fold" else inter)
            M[j, i] = val; C[j, i] = inter
        if axb is not None:
            for s in ("top", "right", "left", "bottom"):
                axb.spines[s].set_visible(False)
            cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
            vmax = np.nanmax(M) * 1.05
            im = axb.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=vmax)
            for i in range(n):
                for j in range(i):
                    if matrix_metric == "count":
                        txt = f"{C[i,j]}"
                    elif matrix_annot == "n":
                        txt = f"n={C[i,j]}"
                    elif matrix_annot == "coef":
                        txt = (f"{M[i,j]:.1f}×" if matrix_metric == "fold"
                               else f"{M[i,j]:.2f}")
                    elif matrix_annot:
                        txt = f"{M[i,j]:.2f}\nn={C[i,j]}"
                    else:
                        txt = ""
                    axb.text(j, i, txt, ha="center", va="center", fontsize=afs,
                             color="white" if M[i, j] > np.nanmax(M) * .6 else "black")
                # all datasets stay on BOTH axes: a 3x3 triangle reads as if one were missing
                if show_diagonal:
                    _nd = len(raw_sets[labs[i]]) if use_pair else len(sets[labs[i]])
                    axb.text(i, i, f"{_nd}", ha="center", va="center",
                             fontsize=afs, color="0.45", style="italic")
            _dl = [dlabel(l) for l in labs]
            axb.set_xticks(range(n)); axb.set_xticklabels(_dl, rotation=30, ha="right")
            axb.set_yticks(range(n)); axb.set_yticklabels(_dl)
            axb.set_xlim(-.5, n - .5); axb.set_ylim(n - .5, -.5)
            axb.tick_params(length=0)
            cb = fig.colorbar(im, ax=axb, fraction=.04, pad=.04)
            cb.set_label(cbar_label if cbar_label is not None else
                         {"overlap_coe": "overlap coefficient",
                          "jaccard": "Jaccard index",
                          "fold": "× more shared than chance",
                          "count": "shared genes"}[matrix_metric], fontsize=fsize(fs - 1))
            if cbar_ticks is not None:
                cb.set_ticks(list(cbar_ticks))
            if matrix_metric == "fold" and chance_line:
                # the reference the reader needs: where on this scale "no better than
                # chance" sits. Drawn on the bar, not left to the caption.
                cb.ax.axhline(1, color=chance_color, lw=1.0, zorder=5)
                # the word goes in the TICK LANE, not beside the bar: to its right
                # sits the colour bar's own label and to its left the matrix, and a
                # floating annotation collides with one or the other
                if cbar_ticks is not None and chance_label:
                    cb.set_ticklabels([f"{t:g} = {chance_label}" if t == 1 else f"{t:g}"
                                       for t in cbar_ticks])
            cb.ax.tick_params(labelsize=fsize(fs - 2))
            if titles and len(titles) > 1 and titles[1]:
                axb.set_title(titles[1], pad=7, fontweight=title_weight)

        if axes is None and panel != "both":
            fig.tight_layout()      # one panel alone: nothing else fits its labels in
        # letters share the titles' baseline: "i  Shared across datasets" reads as
        # one header line rather than two stacked ones
        _ytitle = align_titles(fig, (axa, axb))
        _pl = panel_letters_for(panel_letters, 2)
        if _pl and (axa is None or axb is None):
            _pl = None          # one panel alone takes no letter
        if _pl:
            # align_titles returns None when the panels carry no titles, in which
            # case the letters set their own baseline and letter_dy is a nudge on it
            if _ytitle is None:
                align_panel_letters(fig, (axa, axb), _pl, fontsize=fs + 2,
                                    dy=.010 + letter_dy)
            else:
                align_panel_letters(fig, (axa, axb), _pl, fontsize=fs + 2,
                                    y=_ytitle + letter_dy)
        if show_core_caption and core:
            shown = ", ".join(core[:core_max]) + (" …" if len(core) > core_max else "")
            fig.text(.5, -.09, f"shared by all {n}: " + shown,
                     ha="center", fontsize=fs - 2, color="0.25", style="italic")

        save_fig(fig, save,
                 f"cross_dataset_m{min_methods}_top{top_n or 'cap'}_{null_mode}"
                 f"_{'log' if logy else 'lin'}"
                 + ("" if panel == "both" else f"_{panel}"), dpi=dpi, verbose=verbose)
        if show:
            plt.show()

    rows = []
    for j, k in enumerate(thresholds):
        r = dict(threshold=k, observed=int(obs[j]))
        line = f"  >={k} datasets: observed={obs[j]:.0f}"
        for nm, key, nl, _ in nulls:
            p = ((nl[k] >= obs[j]).sum() + 1) / (n_perm + 1)
            z = zscore(obs[j], nl[k])
            r[f"null_{key}"] = nl[k].mean()
            r[f"sd_{key}"] = float(np.std(nl[k], ddof=1))
            r[f"z_{key}"] = z
            r[f"fold_{key}"] = obs[j] / nl[k].mean() if nl[k].mean() > 0 else np.inf
            r[f"p_{key}"] = p
            line += (f"   {key}: {nl[k].mean():.2f}"
                     f"±{np.std(nl[k], ddof=1):.2f} ({_fmt_z(z)}, p={p:.3g})")
        rows.append(r)
        if verbose:
            print(line)
    if verbose:
        print("  core:", ", ".join(core))
    # one tidy row per pair, so the caption can quote the chance comparison without
    # re-deriving it: observed, the hypergeometric expectation in that pair's own
    # universe, their ratio, and the one-sided p
    _pairs = [dict(a=dlabel(labs[i]), b=dlabel(labs[j]), universe=int(U2[j, i]),
                   observed=int(C[j, i]), expected=E[j, i],
                   fold=(C[j, i] / E[j, i] if E[j, i] else np.nan), p=P[j, i])
              for i, j in combinations(range(n), 2)]
    return fig, dict(sets=sets, universe=universe, per_dataset=info,
                     matrix=pd.DataFrame(M, index=labs, columns=labs),
                     counts=pd.DataFrame(C, index=labs, columns=labs),
                     pair_universe=pd.DataFrame(U2, index=labs, columns=labs),
                     pair_stats=pd.DataFrame(_pairs),
                     core=core, stats=pd.DataFrame(rows))


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — memory-gene responsiveness in Marson perturb-seq
#
#  Ported from perturb-seq/marson_MGs_benchmark.ipynb (cells 8, 9/10, 19-22)
#  and perturb-seq/marson_direction.ipynb (cell 29).
# ═══════════════════════════════════════════════════════════════════════════

PERTURB_DIR = P.LIB
MARSON = dict(
    lfc=f"{P.DATA}/marson/marson_lfc_cpm.csv",              # KO x gene, SIGNED, 1.5 GB
    gene_summary=f"{P.DATA}/marson/marson_gene_summary_cpm.csv",
    expr=f"{P.DATA}/marson/marson_gene_ctrl_expr.csv",
    covariates=f"{P.DATA}/marson/marson_gene_covariates.csv",
    state_expr={s: f"{P.DATA}/marson/marson_gene_ctrl_expr_{s}.csv"
                for s in ("Rest", "Stim8hr", "Stim48hr")},
)
MEMGENE_DIR = f"{BENCH}/{RESULTS_TREE}/T-lineage"
METHOD_FILES = {m: f"{p}_memory_genes_T1_T2.csv"
                for m, p in METHOD_PREFIX.items()}          # METHOD_ORDER
# SCG = the clones 2-8 run, like EVERY other panel (SCG_VARIANT_DEFAULT). Until
# 2026-09-15 this map pointed at the canonical clone_size==2 file, so every
# Marson-based panel (fig 2 d/e/f, robustness, delta-vs-top-N, matching grid,
# leave-out, direction consistency) silently used a different SCG than figure 1.
METHOD_FILES["ClonePair"] = f"scg_memory_genes_{SCG_VARIANT_DEFAULT}T1_T2.csv"

# Per-gene exonic length, for the length-matched rung.
# These are the SAME tables the decay analysis regresses out (decay_lib.CONFOUNDS
# = log_expr, exonic_len, n_introns), built by
# prepare_and_preprocess/tlineage/extract_gene_architecture.py off a MANE-preferred
# representative transcript. Reusing them rather than building a second length
# resource keeps one definition of "gene length" across the whole project.
# SPECIES ASYMMETRY, carried over from that script and belonging in the methods:
# MANE_Select is human-only, so every mouse gene falls through to the most-exons
# rule and is_mane is 0 throughout the mouse table.
GENE_ARCH = {"human": f"{P.DATA}/tLineage_geo/gene_architecture.csv",
             "mouse": f"{P.DATA}/gene_architecture_mouse_vM36.csv"}

# ---- ONE cutoff for the whole figure -------------------------------------
# 100 is well below every method's ceiling at both timepoints, so no method is
# maxed out. Measured ceilings (fig_ranking_depth / ranking_depth_table, and the
# numbers move if the result tables are rebuilt): Mold and Power-Seek emit a
# FIXED 500 genes per group by construction; CORAL, GEMLI and SCG emit whatever
# passes their own rule, so SCG ranges from 329 (Michaels T2 — the binding
# per-group cap of the whole benchmark) to 13,669 (Harmange E). Pooled across
# groups the caps are 688-780. Cut deeper than the smallest of these and the
# shallow methods silently stop changing, so the methods are no longer being
# compared at the same depth.
# NOTE an earlier version of this comment quoted 381/421, 397/418 and 281; those
# predate the current result tables and were wrong by the time they were read.
TOP_N = int(os.environ.get(TOP_N_ENV, _TOP_N_DEFAULT))
MERGE_MODE = "union"          # only used when timepoint == 'merge'
METRIC = "mean_abs_lfc"       # 'median_abs_lfc' for the robust version
# The floor is NOT uniform in the source notebook and that is deliberate here:
# marson_MGs_benchmark cells 8/10 (panels a,b) ran before the floor cell existed, so
# they use the full gene set; the matching cells 15-22 (panel c) set APPLY_FLOOR=True.
# Pass floor=F.EXPR_FLOOR to every call if you want one rule across the whole figure.
EXPR_FLOOR = 0.1              # drop genes below this mean control expression

C_METHOD = {"CORAL": "#d62728", "isCORAL": "#d62728", "GEMLI": "#ff7f0e",
            "POWERSEEK": "#1f77b4", "Power-Seek": "#1f77b4",
            "SCG": "#9467bd", "SCG-2-8": "#9467bd", "ClonePair": "#9467bd"}
# Background = the SAME grey as figure 1's expression-matched null (C_NULL), so a
# reader who learns "grey = matched background" in figure 1 keeps it in figure 2.
# Was "#0072CE" (blue) until 2026-09-16.
# ONE red for memory genes everywhere (user, 2026-10-14): C_MEM is kept as a
# name because every panel signature defaults to it, but it IS C_OBS, so a
# panel that passes F.C_OBS and one that takes the default match exactly.
C_MEM, C_BG = C_OBS, C_NULL


def _perturb_path():
    if PERTURB_DIR not in __import__("sys").path:
        __import__("sys").path.insert(0, PERTURB_DIR)


def load_marson(floor=EXPR_FLOOR, verbose=True, cache=True):
    """gene_summary + per-gene mean control expression, both floored to the same
    index. `floor` drops genes below that mean control CPM (set None to keep all).

    Returns (gene_summary, gene_expr). lfc_df is NOT loaded — it is 1.5 GB and only
    fig_leaveout_kos() needs it (see load_marson_lfc()).
    """
    key = ("marson", floor)
    if cache and key in _CACHE:
        return _CACHE[key]
    gs = pd.read_csv(MARSON["gene_summary"], index_col=0)
    ge = pd.read_csv(MARSON["expr"], index_col=0).iloc[:, 0]
    if floor is not None:
        keep = ge.index[ge > floor]
        gs = gs.loc[gs.index.intersection(keep)]
        ge = ge.loc[gs.index]
    if verbose:
        print(f"marson: gene_summary {gs.shape}, gene_expr {ge.shape} "
              f"(floor={'off' if floor is None else floor})")
    out = (gs, ge)
    if cache:
        _CACHE[key] = out
    return out


def load_marson_lfc(verbose=True, cache=True):
    """The signed KO x gene LFC matrix. ~1.5 GB / ~30 s — cached in-process."""
    if cache and ("marson_lfc",) in _CACHE:
        return _CACHE[("marson_lfc",)]
    if verbose:
        print("reading marson_lfc_cpm.csv (~1.5 GB, ~30 s) ...")
    lfc = pd.read_csv(MARSON["lfc"], index_col=0)
    if verbose:
        print(f"lfc_df: {lfc.shape}")
    if cache:
        _CACHE[("marson_lfc",)] = lfc
    return lfc


def memory_set(method, timepoint="merge", top_n=TOP_N, restrict_to=None,
               merge_mode=MERGE_MODE, memgene_dir=None, method_files=None):
    """One method's top-N memory genes, restricted to the measured universe."""
    _perturb_path()
    from marson_KO import load_memory_gene_set
    return load_memory_gene_set(
        method, timepoint=timepoint, top_n=top_n, merge_mode=merge_mode,
        memgene_dir=memgene_dir or MEMGENE_DIR,
        method_files=method_files or METHOD_FILES, restrict_to=restrict_to)


# ---------------------------------------------------------------- panel a --

def expression_balance(gene_summary, gene_expr, mem_genes, n_bins=30, n_iter=100,
                       seed=0, metric=None):
    """Did the expression matching actually work?

    Returns the standardised mean difference in log10 expression between the memory
    genes and their matched control, plus the fraction of bins that had to be
    sampled WITH replacement (an exhausted bin means the control is reusing genes).

    |SMD| < 0.10 is the conventional 'balanced' bar. Report this: 'expression-matched'
    is a claim about the achieved control, not about the intent of the code. Note the
    SMD denominator is the pooled SD, so running WITHOUT an expression floor flatters
    it — the near-zero genes inflate the SD and hide real imbalance.
    """
    metric = metric or METRIC
    df = pd.DataFrame({"value": gene_summary[metric]}).join(
        gene_expr.rename("expr"), how="inner").dropna()
    df["is_mem"] = df.index.isin(set(mem_genes))
    df["bin"] = pd.qcut(df["expr"].rank(method="first"), n_bins, labels=False)
    m = df[df.is_mem]
    if not len(m):
        return dict(smd_expr=np.nan, oversampled_bins=np.nan, n_mem=0)
    ea = np.log10(m["expr"].values + 1e-3)
    counts = m.groupby("bin").size()
    poolE = {b: g["expr"].values for b, g in df[~df.is_mem].groupby("bin")}
    rng = np.random.default_rng(seed)
    es, exhaust = [], 0
    for _ in range(n_iter):
        for b, n in counts.items():
            if b not in poolE or not n:
                continue
            rep = int(n) > len(poolE[b])
            exhaust += rep
            es.append(rng.choice(poolE[b], int(n), replace=rep))
    eb = np.log10(np.concatenate(es) + 1e-3)
    smd = (ea.mean() - eb.mean()) / np.sqrt((ea.var(ddof=1) + eb.var(ddof=1)) / 2)
    return dict(smd_expr=float(smd), oversampled_bins=exhaust / n_iter, n_mem=len(m))



# ---------------------------------------------------------------- panel b --
# reconstructed from the call sites in marson_MGs_benchmark.ipynb cell 9

# y -> (axis label template, reference line)
_Y = {
    "delta": ("Δ {m}  (memory − matched control)", 0.0),
    "ratio": ("{m} ratio  (memory / matched control)", 1.0),
    "pct":   ("{m}: % higher than matched control", 0.0),
    "clif": ("Cliff's δ  (memory vs matched control)", 0.0),
    "win":   ("P(memory > matched control)  (%)", 50.0),
}


# ---------------------------------------------------------------- panel c --

def load_gene_lengths(species="human", col="exonic_len", cache=True):
    """symbol -> gene length, from the shared gene-architecture table.

    `col` is any column of that table: 'exonic_len' (default — mature length over
    the representative transcript, the featureCounts/TPM denominator and the one
    decay_lib already controls for), 'gene_span', 'total_intron_len', 'n_introns'.

    480 human and 312 mouse symbols appear on more than one row (readthroughs,
    scaffold copies). Deduplicated MANE-first then longest, so the surviving row is
    the same representative transcript the architecture table itself preferred.

    The index is UPPERCASED, so a case-sensitive lookup cannot silently return
    nothing for one dataset while working for the others. Callers should upper()
    their symbols before indexing. (This warning used to say LARRY's csvs were
    UPPERCASE while the other mouse datasets were title-case. That stopped being
    true on 2026-09-08, when the LARRY rerun renamed its `human_symbol` column to
    `human_ortholog`; all three mouse datasets now emit MOUSE title-case symbols.)
    """
    key = ("genelen", species, col)
    if cache and key in _CACHE:
        return _CACHE[key]
    A = pd.read_csv(GENE_ARCH[species],
                    usecols=["gene_name", "is_mane", col])
    A = (A.dropna(subset=["gene_name", col])
          .sort_values(["is_mane", col], ascending=False)
          .drop_duplicates("gene_name", keep="first"))
    out = pd.Series(A[col].values,
                    index=A["gene_name"].astype(str).str.upper()).astype(float)
    out = out[~out.index.duplicated()]
    if cache:
        _CACHE[key] = out
    return out


def gene_log_length(genes, species="human", col="exonic_len"):
    """log10 gene length aligned to `genes`, NaN where the symbol is unknown."""
    L = load_gene_lengths(species, col)
    idx = pd.Index([str(g).upper() for g in genes])
    return pd.Series(np.log10(L.reindex(idx).values), index=list(genes))


def load_marson_covariates(gene_summary=None, gene_expr=None, verbose=True, cache=True):
    """Per-gene covariates for matching, with `expr` and `state_restriction` added.

    state_restriction = log2( max(Rest,Stim8,Stim48) / min(...) ), i.e. how
    state-specific a gene is. This is Marson's analogue of the lineage-restriction
    score that took the HSC effect from +0.535 to ~0. Marson is one cell type, so
    it is a weaker version of that axis — which is a point in Marson's favour, not
    a gap.
    """
    if cache and ("marson_cov",) in _CACHE and gene_summary is None:
        return _CACHE[("marson_cov",)]
    if gene_summary is None or gene_expr is None:
        gene_summary, gene_expr = load_marson(EXPR_FLOOR, verbose)
    cov = pd.read_csv(MARSON["covariates"], index_col=0)
    cov["expr"] = gene_expr.reindex(cov.index)
    S = pd.DataFrame({s: pd.read_csv(p, index_col=0).iloc[:, 0]
                      for s, p in MARSON["state_expr"].items()})
    cov["state_restriction"] = np.log2((S.max(axis=1) + 0.01) / (S.min(axis=1) + 0.01))
    # log10 exonic length. Marson is human. 14 of 5,734 expressed genes are missing
    # here — pre-v47 aliases GENCODE has since renamed (C12orf4, SPATA5, ...) — and
    # they drop out of the length rung ONLY, because matched_comparison dropna()s
    # the covariates it was asked to match on. Every other rung keeps its full n.
    cov["log_len"] = gene_log_length(cov.index, "human").values
    cov = cov.reindex(gene_summary.index).dropna(subset=["expr"])
    if verbose:
        print(f"covariates: {cov.shape}")
        n_len = int(cov["log_len"].notna().sum())
        if n_len < len(cov):
            print(f"  log_len: {n_len}/{len(cov)} genes have a length "
                  f"({len(cov) - n_len} unmatched symbols)")
        # a covariate is only worth matching on if expression does NOT already
        # capture it. Near 0 = a genuinely new axis; near +/-1 = re-matching expr.
        for c in ("detection_rate", "disp_resid", "nb_phi", "state_restriction",
                  "log_len"):
            if c in cov:
                r = cov[["expr", c]].dropna().corr(method="spearman").iloc[0, 1]
                print(f"  Spearman(expr, {c:<18s}) = {r:+.3f}")
    if cache:
        _CACHE[("marson_cov",)] = cov
    return cov


# the rungs the figure shows, in order
MATCH_PANELS = {                       # one covariate each, tested independently
    "expression":        ["expr"],
    "detection":         ["detection_rate"],
    "state restriction": ["state_restriction"],
    "gene length":       ["log_len"],
}
# RECOMMENDED: each covariate tested on its own (matching is then EXACT, SMD 0.000)
# plus one joint rung. The joint rung is the same comparison as the cumulative
# ladder's last step, so nothing is lost, and there is no non-monotone ladder shape
# to explain — the cumulative middle-to-last rise is the matcher sacrificing the
# expression match under three constraints (SMD 0.006 -> 0.039), not a real effect.
# Gene length added as a fifth rung 2026-09-09. It earns a marginal rung rather
# than only a place in the joint one because it is very nearly orthogonal to
# everything already here — in Marson, Spearman(expr, log_len) = -0.106 and
# Spearman(detection_rate, log_len) = -0.062 — so it is a new axis by the same
# criterion load_marson_covariates prints, not a re-match of expression.
# Direction, worth keeping attached to the panel: the memory sets are SHORTER than
# the universe, not longer, so the length rung tests a SHORT-gene bias. That bias
# is a property of 3'-biased data — it is absent in the one full-length dataset.
MATCH_PANELS_JOINT = {
    "expression":        ["expr"],
    "detection":         ["detection_rate"],
    "state specificity": ["state_restriction"],
    "gene length":       ["log_len"],
    "all four":          ["expr", "detection_rate", "state_restriction", "log_len"],
}
# CUMULATIVE ladder: each rung controls for one MORE thing. Easier to explain than
# three independent rungs ("does it survive when we also control for X?"), and it
# answers the joint question rather than three separate marginal ones.
MATCH_PANELS_CUMULATIVE = {
    "expression":          ["expr"],
    "+ detection":         ["expr", "detection_rate"],
    "+ state specificity": ["expr", "detection_rate", "state_restriction"],
    "+ gene length":       ["expr", "detection_rate", "state_restriction", "log_len"],
}


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values, NaNs passed through."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    v = p[ok]
    n = len(v)
    if not n:
        return q
    order = np.argsort(v)
    adj = np.empty(n)
    adj[order] = v[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    q[ok] = np.minimum(adj, 1.0)
    return q


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


# ---------------------------------------------------------------- panel d --

def fig_leaveout_kos(memset="ClonePair", min_methods=5, timepoint="merge", top_n=TOP_N,
                     metric="abs", merge_mode=MERGE_MODE, method_files=None,
                     memgene_dir=None, step=25, n_bins=50, n_bg=2000, seed=0,
                     floor=EXPR_FLOOR, lfc_df=None, fontsize=20, figsize=(11, 6.5),
                     title=None, save=None, show=True, verbose=True):
    """(d) Memory genes stay more responsive after removing the strongest
    perturbations.

    Per KO, impact = mean |LFC| over the memory set; the most-impactful KOs are
    removed progressively. The memory line staying above the
    expression-matched non-memory line means the higher responsiveness is not
    carried by a handful of big perturbations.

    memset      : a method name (default 'SCG', matching panel c), or None -> the
                  CONSENSUS set (genes called by >= min_methods of the 5).
                  WARNING: the source cell used memset=None, min_methods=5 at
                  top_n=500, which gives 88 genes. At the figure's top_n=100 that
                  same rule yields only 10 genes and the effect INVERTS (memory
                  falls below background at every step). Use a single method, or
                  min_methods=2 (116 genes), at this cutoff — or pass top_n=500.
    metric      : 'abs'    mean |LFC| + a non-memory reference line (this figure)
                  'signed' mean signed LFC (the down-lean version, no reference)
    lfc_df      : pass a preloaded matrix to skip the 1.5 GB read.

    Returns (fig, DataFrame with ko_removed / mean_abs / bg_mean_abs).
    """
    _perturb_path()
    from marson_direction import consensus_and_background, plot_leaveout_kos

    gs, ge = load_marson(floor, verbose)
    method_files = method_files or METHOD_FILES
    consensus, nonmem, counts = consensus_and_background(
        timepoint=timepoint, top_n=top_n, merge_mode=merge_mode,
        memgene_dir=memgene_dir or MEMGENE_DIR, method_files=method_files,
        universe=gs.index)
    nonmem = never_called(gs.index, top_n, timepoint=timepoint, merge_mode=merge_mode,
                          memgene_dir=memgene_dir, method_files=method_files)
    if memset is None:
        mem = counts.index[counts >= min_methods].tolist()
        lab = f">={min_methods} methods"
    else:
        mem = memory_set(memset, timepoint, top_n, restrict_to=gs.index,
                         merge_mode=merge_mode, memgene_dir=memgene_dir,
                         method_files=method_files)
        lab = memset
    if verbose:
        for k in range(len(method_files), 0, -1):
            print(f"  >= {k} methods: {int((counts >= k).sum())} genes")
        print(f"using {lab} -> {len(mem)} memory genes, {len(nonmem)} non-memory")
    if len(mem) < 30:
        warnings.warn(
            f"[fig_leaveout_kos] only {len(mem)} memory genes ({lab} @ top_n={top_n}). "
            "The per-KO mean over so few genes is dominated by noise and the curve "
            "can sit BELOW the background — that is the sample size, not a result. "
            "Use a single method, min_methods=2, or a deeper top_n.", stacklevel=2)

    lfc = lfc_df if lfc_df is not None else load_marson_lfc(verbose)
    tn = "all" if top_n is None else str(top_n)
    out = plot_leaveout_kos(
        lfc, mem, metric=metric, background_genes=nonmem, gene_expr=ge,
        n_bins=n_bins, n_bg=n_bg, step=step, random_seed=seed, fontsize=fontsize,
        figsize=figsize,
        title=title or ("Memory genes stay more responsive after removing "
                        "the strongest perturbations"),
        save_path=(os.path.join(OUTDIR, f"fig2d_leaveout_kos_{metric}_top{tn}.png")
                   if save is True else (save or None)),
        show=show)
    return plt.gcf(), out


# ═══════════════════════════════════════════════════════════════════════════
#  Robustness / reviewer-defence panel
# ═══════════════════════════════════════════════════════════════════════════

RESPONSE_METRICS = {
    "mean_abs_lfc":   "mean |LFC|",
    "median_abs_lfc": "median |LFC|",
    "std_lfc":        "SD of LFC",
    "frac_large":     "fraction |LFC| > 1",
    "max_abs_lfc":    "max |LFC|",
    "n_obs":          "negative control",   # the number of perturbations a gene was
                                            # measured in: must show nothing
}
NEGATIVE_CONTROL = "n_obs"


def _matched_effect(gs, ge, mem, metric, n_bins, n_iter, n_boot=300, seed=0, bg=None):
    """delta + median ratio for one (metric, gene set), with bootstrap CIs."""
    _perturb_path()
    from marson_KO import compare_gene_responsiveness_expr_matched, cliffs_delta
    if bg is None:
        raise ValueError("_matched_effect needs bg= (never_called at the panel's top_n)")
    fgs, fge = _with_background(gs, ge, mem, bg)
    r = compare_gene_responsiveness_expr_matched(fgs, fge, mem, metric=metric,
                                                 n_bins=n_bins, n_iter=n_iter)
    a, b = np.asarray(r["scg_values"], float), np.asarray(r["non_scg_values"], float)
    rng = np.random.default_rng(seed)
    ds, rs = [], []
    for _ in range(n_boot):
        aa, bb = rng.choice(a, len(a), True), rng.choice(b, len(b), True)
        ds.append(cliffs_delta(aa, bb))
        rs.append(np.median(aa) / np.median(bb) if np.median(bb) > 0 else np.nan)
    return dict(metric=metric, delta=cliffs_delta(a, b),
                d_lo=np.percentile(ds, 2.5), d_hi=np.percentile(ds, 97.5),
                ratio=np.median(a) / np.median(b) if np.median(b) > 0 else np.nan,
                r_lo=np.nanpercentile(rs, 2.5), r_hi=np.nanpercentile(rs, 97.5),
                p=r["mann_whitney_p"], n_mem=r["n_scg"])


def robustness_tables(method="ClonePair", timepoint="merge", top_n=TOP_N,
                      floors=(None, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                      metrics=None, ref_floor=EXPR_FLOOR, n_bins=50, n_iter=100,
                      n_boot=300, seed=0, verbose=True, min_methods=None):
    """The two reviewer-defence tables.

    min_methods : use the CONSENSUS of >= this many of the five methods instead of one
        method's set (the gene set of figures 2e/2f). `method` is then ignored.

    floor_sensitivity : is the effect an artefact of the expression cutoff?
    metric_robustness : is it an artefact of how responsiveness is defined?
                        Includes n_obs as a NEGATIVE CONTROL — it must come out null.
    """
    def _mem(gs_index):
        """the gene set under test: one method, or the >= min_methods consensus"""
        if not min_methods:
            return memory_set(method, timepoint, top_n, restrict_to=gs_index)
        _perturb_path()
        from marson_KO import build_consensus_gene_sets
        _, cnt = build_consensus_gene_sets(
            min_methods=1, timepoint=timepoint, top_n=top_n, merge_mode=MERGE_MODE,
            memgene_dir=MEMGENE_DIR, method_files=METHOD_FILES,
            universe=gs_index, verbose=False)
        return cnt.index[cnt >= int(min_methods)].tolist()

    rows_f = []
    for fl in floors:
        gs, ge = load_marson(fl, verbose=False)
        mem = _mem(gs.index)
        e = _matched_effect(gs, ge, mem, "mean_abs_lfc", n_bins, n_iter, n_boot, seed,
                            bg=never_called(gs.index, top_n, timepoint=timepoint))
        rows_f.append(dict(floor=fl, universe=len(gs), **e))
    F1 = pd.DataFrame(rows_f)

    gs, ge = load_marson(ref_floor, verbose=False)
    mem = _mem(gs.index)
    bg_ref = never_called(gs.index, top_n, timepoint=timepoint)
    rows_m = []
    for m in (metrics or RESPONSE_METRICS):
        if m not in gs.columns:
            continue
        rows_m.append(dict(label=RESPONSE_METRICS.get(m, m),
                           is_control=(m == NEGATIVE_CONTROL),
                           **_matched_effect(gs, ge, mem, m, n_bins, n_iter, n_boot, seed,
                                             bg=bg_ref)))
    F2 = pd.DataFrame(rows_m)
    if verbose:
        print("floor sensitivity:"); print(F1.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        print("\nmetric robustness:"); print(F2.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return F1, F2


def fig_robustness_forest(method="ClonePair", timepoint="merge", top_n=TOP_N,
                          floors=(None, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                          ref_floor=EXPR_FLOOR, n_bins=50, n_iter=100, n_boot=300,
                          seed=0, tables=None, figsize=(4.0, 4.4), fs=7.5,
                          c_main=C_OBS, c_ref=C_OBS, c_ctrl=C_NULL,
                          group_labels=("Expression cutoff (CPM)", "Responsiveness metric"),
                          ref_suffix="", ref_metric=METRIC,
                          xlabel="Cliff's δ  (memory vs matched control, 95% CI)",
                          title=None, annotate_n=True, xlim=None,
                          min_methods=None,
                          save=None, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 3b) Both robustness results as ONE forest plot.

    Rows are grouped: the expression-cutoff sweep, then the responsiveness-metric
    sweep, on a shared bounded x-axis. Cliff's delta is used rather than the ratio
    because the ratio is not comparable across metrics (frac_large is 5.02x by
    construction) -- the ratios belong in the accompanying table.

    The cutoff and the metric actually used are bold (set ref_suffix="  ←used" to
    also label the cutoff), and the negative control (n_obs) is drawn
    in grey: its CI straddling zero is the point of including it.

    Returns (fig, tidy DataFrame of every row drawn).
    """
    F1, F2 = tables if tables is not None else robustness_tables(
        method, timepoint, top_n, floors, None, ref_floor, n_bins, n_iter,
        n_boot, seed, verbose=False, min_methods=min_methods)

    rows = []
    for _, r in F1.iterrows():
        is_ref = (r["floor"] == ref_floor)
        rows.append(dict(group=group_labels[0],
                         label=("none" if r["floor"] is None or pd.isna(r["floor"])
                                else f"{r['floor']:g}") + (ref_suffix if is_ref else ""),
                         delta=r["delta"], lo=r["d_lo"], hi=r["d_hi"],
                         ratio=r["ratio"], p=r["p"], n=int(r["universe"]),
                         kind="re" if is_ref else "main"))
    for _, r in F2.iterrows():
        rows.append(dict(group=group_labels[1], label=r["label"], delta=r["delta"],
                         lo=r["d_lo"], hi=r["d_hi"], ratio=r["ratio"], p=r["p"],
                         n=int(r["n_mem"]),
                         kind=("ctrl" if r["is_control"] else
                               "re" if r["metric"] == ref_metric else "main")))
    D = pd.DataFrame(rows)

    # y positions: a blank slot between the two groups
    ys, lab, gap = [], [], 0
    for i, r in D.iterrows():
        if i and D.loc[i, "group"] != D.loc[i - 1, "group"]:
            gap += 1.6
        ys.append(i + gap)
    D["y"] = ys
    ymax = max(ys)

    with plt.rc_context({**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fsize(fs - .5), "ytick.labelsize": fsize(fs - .5)}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(0, color="0.7", lw=.7, ls=(0, (3, 2)), zorder=1)
        for r in D.itertuples():
            c = {"main": c_main, "re": c_ref, "ctrl": c_ctrl}[r.kind]
            ax.errorbar(r.delta, ymax - r.y, xerr=[[r.delta - r.lo], [r.hi - r.delta]],
                        fmt="none", ecolor=c, lw=1.0, capsize=1.8, zorder=3)
            ax.scatter(r.delta, ymax - r.y, s=26 if r.kind == "ref" else 15, c=c,
                       zorder=4, edgecolor="white" if r.kind == "re" else "none",
                       linewidth=.5)
        ax.set_yticks(ymax - D["y"])
        ax.set_yticklabels(D["label"])
        for t, k in zip(ax.get_yticklabels(), D["kind"]):
            if k == "ctrl":
                t.set_color("0.45"); t.set_style("italic")
            elif k == "ref":
                t.set_fontweight("bold")
        # group headers, in the left margin
        for g, sub in D.groupby("group", sort=False):
            ax.text(-.02, ymax - sub["y"].min() + 1.0, g, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=fs - .5, fontweight="bold",
                    color="0.25")
        if annotate_n:
            xr = xlim or (min(-.15, D.lo.min() - .05), max(.78, D.hi.max() * 1.06))
            for r in D.itertuples():
                ax.text(xr[1], ymax - r.y, f"{r.ratio:.2f}×", va="center", ha="right",
                        fontsize=fs - 1.5,
                        color="0.45" if r.kind == "ctrl" else "0.3")
            ax.text(xr[1], ymax + 1.1, "ratio", va="center", ha="right",
                    fontsize=fs - 1.5, color="0.35", fontweight="bold")
        ax.set_xlim(*(xlim or (min(-.15, D.lo.min() - .05), max(.78, D.hi.max() * 1.06))))
        ax.set_ylim(-1.2, ymax + 1.8)
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if title is None:
            tn = "all" if top_n is None else top_n
            gene_set = (f"genes called by ≥{int(min_methods)} of "
                        f"{len(METHOD_FILES)} methods" if min_methods else f"{method} memory genes")
            title = f"{gene_set}, top {tn} ({timepoint})"
        if title:
            ax.set_title(title, fontsize=fs + .5, pad=8)
        save_fig(fig, save, f"robustness_forest_{method}_top{top_n}_{timepoint}",
                 dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, D.drop(columns="y")


# ═══════════════════════════════════════════════════════════════════════════
#  Cross-dataset responsiveness sweep  (from perturb-seq/cross_method_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════

# The source notebook read BENCH/data (authors' originals). Do NOT: CORAL's files
# there are in GENOME order, not ranked by omega^2, so load_memory_gene_set(top_n=N)
# returns an arbitrary N of its ~4,400 called genes and the CORAL row of the heatmap
# is noise (Marson: +16% unranked vs +47% ranked; the two top-100s share 1 gene).
# The ports are sorted. _xd_prune() checks this and warns for any file it can rank.
XD_MEMBASE = f"{BENCH}/{RESULTS_TREE}"        # python_implements: the ranked tree
XD_TOPNS = [100, 250, 500, 1000, 2000]
# The canonical clone_size==2 SCG is NOT listed any more (2026-09-15): only the
# clones 2-8 run is used anywhere. It used to be computed and then dropped from the
# heatmap -- but it still counted as "called" when the background was built.
# uppercase spellings of the same methods; METHOD_EXCLUDE is applied here too
XD_NAME = {"CORAL": "CORAL", "GEMLI": "GEMLI",
           "POWERSEEK": "Power-Seek", "SCG": "ClonePair", "SCG-2-8": "ClonePair"}
XD_METHODS6 = [m for m in ["CORAL", "GEMLI", "POWERSEEK", "SCG-2-8"]
               if XD_NAME[m] not in METHOD_EXCLUDE]
# For the CONSENSUS axis, SCG and SCG-2-8 must never both be counted: they are one
# method at two clone-size settings, so a gene they both call would score 2. The
# widened variant is the one kept, for the same reason the heatmap keeps it -- the
# canonical clone_size==2 SCG calls 0 genes per sorted arm on Harmange.
XD_METHODS_CONSENSUS = [m for m in ["CORAL", "GEMLI", "POWERSEEK", "SCG-2-8"]
                        if XD_NAME[m] not in METHOD_EXCLUDE]


def _xd_files(suffix):
    out = {"CORAL": f"coral_memory_genes_{suffix}.csv",
           "GEMLI": f"gemli_memory_genes_{suffix}.csv",
           "POWERSEEK": f"powerseek_memory_genes_{suffix}.csv",
           "SCG-2-8": f"scg_memory_genes_clones2-8_{suffix}.csv"}   # the ONLY SCG used
    return {m: f for m, f in out.items() if XD_NAME[m] not in METHOD_EXCLUDE}


# candidate ranking columns per method. A file counts as ranked if ANY of them is
# monotone in the direction that puts the best gene first (p-like columns ascend).
_RANK_COL = {"coral": ["Omega_square", "SSb"], "mold": ["ANOVA F", "p", "F"],
             "powerseek": ["delta", "p"], "gemli": ["var", "p"],
             # is_biased is SCG's memory-gene call. is_heritable is a DEAD legacy
             # column (findingSCGs_melanoma.py writes it all-zeros) -- never read it.
             "scg": ["is_biased", "concordance_freq", "score"]}
_ASCENDING = {"p", "padj", "pval", "fdr", "q"}


def _xd_check_ranked(path, method, verbose=True):
    """Is this results file actually RANKED? load_memory_gene_set takes the first N
    rows, so an unranked file silently turns 'top N' into 'an arbitrary N'.

    Returns True/False, or None when no known ranking column is present.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "timepoint" not in df.columns:
        return None
    sub = df[df["timepoint"] == sorted(df["timepoint"].unique())[0]]
    checked = []
    for col in _RANK_COL.get(method.split("-")[0].lower(), []):
        if col not in sub.columns or not pd.api.types.is_numeric_dtype(sub[col]):
            continue
        v = sub[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        d = np.diff(v)
        ok = bool(np.all(d >= -1e-12)) if col.lower() in _ASCENDING else \
            bool(np.all(d <= 1e-12))
        if ok:
            return True                      # ranked by this column
        checked.append(col)
    if checked and verbose:
        warnings.warn(
            f"[{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}] "
            f"not sorted by any of {checked} — a top-N cut on this file selects an "
            "ARBITRARY N, not the top N.", stacklevel=3)
        return False
    return None


def _xd_prune(datasets, verbose=True, min_methods=4):
    """Drop method files that do not exist on disk.

    Not defensive padding: a dataset can be missing a method's csv, and
    build_consensus_gene_sets
    reads EVERY file in the dict when it builds the 0-method background, so a missing
    one raises before any row is produced. Pruning also means that dataset's
    background is defined by the methods it actually has -- worth stating, since a
    5-method background is a slightly larger set than a 6-method one.
    """
    for ds in datasets:
        have = {m: f for m, f in ds["files"].items()
                if os.path.exists(os.path.join(ds["memdir"], f))}
        missing = [m for m in ds["files"] if m not in have]
        if missing and verbose:
            print(f"  [{ds['name'].splitlines()[-1]} @ {os.path.basename(ds['memdir'])}] "
                  f"no csv for {', '.join(missing)} -> background built from "
                  f"{len(have)} methods")
        for m, f in have.items():
            _xd_check_ranked(os.path.join(ds["memdir"], f), m, verbose)
        ds["files"] = have
    keep = [d for d in datasets if len(d["files"]) >= min_methods]
    for d in datasets:
        if d not in keep and verbose:
            print(f"  DROPPED {d['name'].splitlines()[-1]}: only {len(d['files'])} "
                  f"method(s) < min_methods={min_methods}. Its 0-method background "
                  "would be built from too few methods to compare with the others.")
    return keep


def _xd_pair(memset, screen, system, sep=" \u2013 "):
    """Row label for a lineage x perturb-seq pairing.

    Memory-gene dataset first, then the screen it was tested in, then the shared
    biological system. Both names are first authors, so the label reads the same
    way as every dataset name in figure 1.
    """
    return f"{memset}{sep}{screen}" + f"\n({system})"


def xd_datasets(membase=XD_MEMBASE, prune=True, verbose=True, min_methods=4):
    """The perturb-seq x lineage-source pairs, exactly as cross_method_analysis.ipynb
    defines them, but listed in DATASET_ORDER (the memory-gene dataset is the row's
    identity, and every figure and table has to put the datasets in the same order).

    NO dataset here is case_insensitive: since the 2026-09-08 LARRY rerun every
    memory-gene csv carries native symbols (mouse title-case for the three mouse
    datasets). The bridge stays for the case where that stops being true; see the
    `human_symbol` -> `human_ortholog` note on the Weinreb row below."""
    D = P.DATA
    ds = [
        dict(name=_xd_pair("Michaels", "Zhu", "Progenitor T Cells"),
             gs=f"{D}/marson/marson_gene_summary_cpm.csv",
             expr=f"{D}/marson/marson_gene_ctrl_expr.csv",
             memdir=f"{membase}/T-lineage", files=_xd_files("T1_T2")),
        dict(name=_xd_pair("Michaels", "Weinstock", "Progenitor T Cells"),
             gs=f"{D}/weinstock_gene_summary_cpm.csv",
             expr=f"{D}/weinstock_gene_ctrl_expr.csv",
             memdir=f"{membase}/T-lineage", files=_xd_files("T1_T2")),
        dict(name=_xd_pair("Harmange", "Xu", "melanoma"),
             gs=f"{D}/melanoma/perturbseq/melanoma_gene_summary_cpm_DMSO.csv",
             expr=f"{D}/melanoma/perturbseq/melanoma_gene_ctrl_expr_DMSO.csv",
             memdir=f"{membase}/Melanoma_Harmange", files=_xd_files("E_M")),
        dict(name=_xd_pair("Harmange", "Frangieh", "melanoma"),
             gs=f"{D}/melanoma/perturbseq/frangieh_gene_summary_cpm_Control.csv",
             expr=f"{D}/melanoma/perturbseq/frangieh_gene_ctrl_expr_Control.csv",
             memdir=f"{membase}/Melanoma_Harmange", files=_xd_files("E_M")),
        # Bowness (in vitro mouse HSPC CRISPRi) replaced Lara-Astiaso (GSE213511,
        # in vivo transplant) as LARRY's partner 2026-09-08: LARRY is an in vitro
        # LSK expansion culture, so Bowness matches its cell type AND its culture
        # context.
        # No case_insensitive: everything here is in MOUSE symbols. LARRY's SCG
        # csvs used to carry real orthologs in a 'human_symbol' column, which
        # load_memory_gene_set prefers over 'gene' -- so SCG alone came back
        # UPPERCASE (DLK1) while the other four returned mouse symbols (Alox5)
        # and the SCG cell went blank. Every other dataset's human_symbol is a
        # passthrough copy of the mouse symbol (0 rows differ), LARRY's differed
        # in 3,061/3,065. Renamed to 'human_ortholog' 2026-09-08 so the loader
        # falls back to 'gene'; the orthologs are still in the file.
        dict(name=_xd_pair("Weinreb", "Bowness", "HSPC"),
             gs=f"{D}/velten_hspc/velten_gene_summary_cpm.csv",
             expr=f"{D}/velten_hspc/velten_gene_ctrl_expr.csv",
             memdir=f"{membase}/LARRY", files=_xd_files("2.0_6.0_4.0")),
        dict(name=_xd_pair("Bandler", "Shi", "mouse brain"),
             gs=f"{D}/Mold_brain/mold_gene_summary_cpm.csv",
             expr=f"{D}/Mold_brain/mold_gene_ctrl_expr.csv",
             memdir=f"{membase}/BandlerBrain", files=_xd_files("E13_E14")),
    ]
    # A lineage dataset the figures exclude takes its PAIRS out of the sweep too,
    # or figure 2 and the supplement disagree about which datasets exist. Keyed on
    # memdir -- where the MEMORY genes came from -- and not on the screen: the
    # Bandler row reads the Mold_brain gene summary because Shi's screen lives in
    # that folder, and it has to survive Mold's removal.
    keep = {DATASETS[d]["folder"] for d in active_datasets() if d in DATASETS}
    ds = [d for d in ds if os.path.basename(d["memdir"]) in keep]
    return _xd_prune(ds, verbose, min_methods) if prune else ds


def _xd_symbol_bridge(ds, gs):
    """memory-file symbol space -> dataset symbol space (identity unless asked)."""
    if ds.get("case_insensitive"):
        up2sym = {}
        for g in gs.index.astype(str):
            up2sym.setdefault(g.upper(), g)
        return list(up2sym), (lambda gl: [up2sym[x] for x in gl if x in up2sym])
    return gs.index, (lambda gl: [g for g in gl if g in gs.index])


# Methods whose csv is a full SCORE TABLE rather than a selection: every tested
# gene appears with a statistic. CORAL lists 88-93% of the measured universe on the
# melanoma data, so counting it as "called" leaves a background of only ~5-10% —
# and those leftovers are genes CORAL could not score, i.e. a QC residue, which
# reads as NEGATIVE responsiveness. Exclude such methods from the background
# definition; they are still scored as memory sets.
BG_EXCLUDE_DEFAULT = ()          # with a bounded top-N background, CORAL cannot swamp it
# ONE background rule for every panel (decided 2026-09-15): genes that NONE of the
# five methods places in its top N, where N is the SAME cutoff the panel's memory genes
# use (top 300 memory genes -> background = none in any top 300). A fixed deeper
# background (500 / 1000) was tried and dropped: it removed so many highly expressed
# genes that expression matching failed (|SMD| up to 0.25 in Frangieh and 2d Mold).
BG_TOP_N = None          # None = follow the panel's own cutoff


def xd_background(ds, universe, to_data, tp, mm, mode="never_any", top_n=None,
                  method=None, files=None, bg_exclude=BG_EXCLUDE_DEFAULT,
                  min_frac=0.20, verbose=True):
    """The non-memory comparison group.

    mode='top_n'      (default, with top_n=BG_TOP_N=2000) a gene outside EVERY
        method's top-2000. Bounded, so a method whose csv is a full score table
        cannot swallow the universe, and it is fixed while the cutoff axis moves.
    mode='never_any'  a gene no method EVER called, at any depth: the
        union of every method's FULL ranked list is removed. Independent of the
        top-N axis, so a column-to-column change can only come from the memory set.
    mode='never_own'  per-method: genes that THIS method never called (other
        methods' calls stay in the background).
    mode='top_n'      legacy: genes called by no method within its top-`top_n`.
        Under a cutoff axis this moves with the axis — avoid.

    bg_exclude : methods that do NOT get a say in what counts as "called" (see
        BG_EXCLUDE_DEFAULT). min_frac warns if the background still ends up under
        that fraction of the universe.
    """
    _perturb_path()
    from marson_KO import build_consensus_gene_sets, load_memory_gene_set
    files = files if files is not None else ds["files"]
    if mode != "never_own" and bg_exclude:
        kept = {m: f for m, f in files.items() if m not in set(bg_exclude)}
        if kept:
            files = kept
    if mode == "never_own":
        called = set(load_memory_gene_set(method, tp, None, mm, ds["memdir"],
                                          files, restrict_to=universe))
        return to_data([g for g in universe if g not in called])
    at = None if mode == "never_any" else top_n
    _, counts = build_consensus_gene_sets(
        min_methods=1, timepoint=tp, top_n=at, merge_mode=mm,
        memgene_dir=ds["memdir"], method_files=files, universe=universe,
        verbose=False)
    bg = to_data(counts.index[counts == 0].tolist())
    if verbose and len(universe) and len(bg) / len(universe) < min_frac:
        warnings.warn(
            f"[{ds['name'].splitlines()[-1]}] background is only {len(bg)}/"
            f"{len(universe)} genes ({100*len(bg)/len(universe):.0f}%): the methods "
            f"{list(files)} between them call nearly everything measured, so what is "
            "left is a QC residue rather than a comparison group.", stacklevel=2)
    return bg


def xd_sweep(datasets=None, top_ns=None, methods=None, n_bins=50, n_iter=100,
             timepoint="merge", merge_mode="union", ci=False, bg_mode="top_n",
             bg_top_n=None, bg_exclude=BG_EXCLUDE_DEFAULT,
             metric="mean_abs_lfc", min_genes=10, floor=EXPR_FLOOR,
             cache=True, verbose=True):
    """Per-method responsiveness across top-N cutoffs, for every dataset.

    Self-contained (does not call marson_KO.cross_dataset_topn_sweep) so the
    background definition is ours to set — see xd_background(). Default
    bg_mode='never_any': a non-memory gene is one NO method ever called, at any
    depth, which keeps the comparison group fixed while the cutoff axis moves.

    Returns a long frame: dataset, method, top_n, n, n_bg, cliffs_d, pct, p.
    """
    _perturb_path()
    from marson_KO import (load_memory_gene_set,
                           compare_gene_responsiveness_expr_matched,
                           cliffs_delta, cliffs_ci, pct_ci)
    datasets = datasets if datasets is not None else xd_datasets()
    top_ns = list(top_ns if top_ns is not None else XD_TOPNS)
    methods = list(methods or XD_METHODS6)
    key = ("xd_sweep_v2", tuple(d["name"] for d in datasets), tuple(top_ns),
           tuple(methods), n_bins, n_iter, timepoint, merge_mode, ci, bg_mode,
           bg_top_n, tuple(bg_exclude or ()), metric, floor)
    if cache and key in _CACHE:
        return _CACHE[key]

    rows = []
    for ds in datasets:
        gs = pd.read_csv(ds["gs"], index_col=0)
        expr = pd.read_csv(ds["expr"], index_col=0).iloc[:, 0]
        # ONE expression rule across the whole figure (2026-09-16): drop genes below
        # `floor` mean control CPM BEFORE anything else, so each method contributes the
        # top N among EXPRESSED genes and the background is drawn from the same pool --
        # identical to load_marson(floor=...) + floor_when="universe" in 2d. Without it
        # this panel disagreed with 2d on the same screen (Marson SCG 51% vs 69%).
        if floor is not None:
            keep = expr.index[expr > floor]
            gs = gs.loc[gs.index.intersection(keep)]
            expr = expr.loc[expr.index.intersection(gs.index)]
        tp = ds.get("timepoint", timepoint)
        mm = ds.get("merge_mode", merge_mode)
        universe, to_data = _xd_symbol_bridge(ds, gs)
        if bg_mode == "top_n" and bg_top_n is not None and bg_top_n < max(top_ns):
            raise ValueError(
                f"bg_top_n={bg_top_n} < max(top_ns)={max(top_ns)}: a gene could be "
                "in a method's top-N and in the background at the same time.")
        # background = genes no method calls within the SAME top N as the memory set
        # (bg_top_n=None, the default since 2026-09-15); pass bg_top_n to fix one depth
        fixed_bg = (None if (bg_mode == "never_own" or bg_top_n is None) else
                    xd_background(ds, universe, to_data, tp, mm, bg_mode, bg_top_n,
                                  bg_exclude=bg_exclude, verbose=verbose))
        for top_n in top_ns:
            shared_bg = (None if bg_mode == "never_own" else fixed_bg if fixed_bg is not None
                         else xd_background(ds, universe, to_data, tp, mm, bg_mode, top_n,
                                            bg_exclude=bg_exclude, verbose=verbose))
            for method in methods:
                if method not in ds["files"]:
                    continue
                bg = (shared_bg if shared_bg is not None else
                      xd_background(ds, universe, to_data, tp, mm, "never_own",
                                    method=method))
                bgset = set(bg)
                mem = [g for g in to_data(load_memory_gene_set(
                    method, tp, top_n, mm, ds["memdir"], ds["files"],
                    restrict_to=universe)) if g not in bgset]
                if len(mem) < min_genes or len(bg) < min_genes:
                    continue
                frame = gs.loc[list(dict.fromkeys(mem + bg))]
                m = compare_gene_responsiveness_expr_matched(
                    frame, expr, mem, metric=metric, n_bins=n_bins, n_iter=n_iter)
                a, b = m["scg_values"], m["non_scg_values"]
                bal = expression_balance(frame, expr, mem, n_bins=n_bins, n_iter=n_iter,
                                         metric=metric)
                row = dict(dataset=ds["name"], method=method, top_n=top_n,
                           n=len(mem), n_bg=len(bg), smd_expr=bal["smd_expr"],
                           p=m["mann_whitney_p"],
                           cliffs_d=cliffs_delta(a, b),
                           pct=(np.median(a) / np.median(b) - 1) * 100)
                if ci:
                    row["d_lo"], row["d_hi"] = cliffs_ci(a, b)
                    row["pct_lo"], row["pct_hi"] = pct_ci(a, b)
                rows.append(row)
            if verbose:
                print(f"done: {ds['name'].splitlines()[-1]} (top{top_n})", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["bg_mode"] = bg_mode
    if verbose and len(out):
        print(f"xd_sweep: {len(out)} rows, background='{bg_mode}' "
              f"(median {int(out['n_bg'].median())} genes)")
    if cache:
        _CACHE[key] = out
    return out


def xd_consensus_sweep(datasets=None, tiers=(2, 3, 4, 5), top_n=500,
                       n_bins=50, n_iter=100, timepoint="merge", merge_mode="union",
                       methods=None, ci=False, bg_mode="top_n", bg_top_n=None,
                       bg_exclude=BG_EXCLUDE_DEFAULT, min_genes=10,
                       floor=EXPR_FLOOR, cache=True, verbose=True):
    """Same comparison as xd_sweep, but the columns are CONSENSUS TIERS instead of
    individual methods: a gene is 'memory' if >= k of the methods call it at `top_n`.

    The background (bg_mode='never_any') is every gene NO method ever called, at
    any depth — not merely outside the top-N. It is built once per dataset and
    shared by every tier, so moving along the tier axis only changes the memory
    set, never the comparison group.

    Returns a long frame shaped like xd_sweep's (dataset, method, top_n, n,
    cliffs_d, pct), where `method` holds '>=2' ... '>=5' so the same plotting code
    works. `top_n` is constant, which is the point.
    """
    _perturb_path()
    from marson_KO import (build_consensus_gene_sets,
                           compare_gene_responsiveness_expr_matched,
                           cliffs_delta, cliffs_ci, pct_ci)
    datasets = datasets if datasets is not None else xd_datasets()
    tiers = list(tiers)
    key = ("xd_cons", tuple(d["name"] for d in datasets), tuple(tiers), top_n,
           n_bins, n_iter, timepoint, merge_mode, ci, bg_mode,
           tuple(bg_exclude or ()), floor)
    if cache and key in _CACHE:
        return _CACHE[key]

    rows = []
    for ds in datasets:
        gs = pd.read_csv(ds["gs"], index_col=0)
        expr = pd.read_csv(ds["expr"], index_col=0).iloc[:, 0]
        # ONE expression rule across the whole figure (2026-09-16): drop genes below
        # `floor` mean control CPM BEFORE anything else, so each method contributes the
        # top N among EXPRESSED genes and the background is drawn from the same pool --
        # identical to load_marson(floor=...) + floor_when="universe" in 2d. Without it
        # this panel disagreed with 2d on the same screen (Marson SCG 51% vs 69%).
        if floor is not None:
            keep = expr.index[expr > floor]
            gs = gs.loc[gs.index.intersection(keep)]
            expr = expr.loc[expr.index.intersection(gs.index)]
        tp = ds.get("timepoint", timepoint)
        mm = ds.get("merge_mode", merge_mode)
        universe, to_data = _xd_symbol_bridge(ds, gs)

        want = methods if methods is not None else XD_METHODS_CONSENSUS
        files = {m: f for m, f in ds["files"].items() if m in want}
        if {"SCG", "SCG-2-8"} <= set(files):
            raise ValueError("consensus tiers would count SCG twice: pass methods= "
                             "with only one of 'SCG' / 'SCG-2-8' "
                             "(default XD_METHODS_CONSENSUS keeps SCG-2-8)")
        # one consensus build per dataset: counts gives every tier at once
        _, counts = build_consensus_gene_sets(
            min_methods=1, timepoint=tp, top_n=top_n, merge_mode=mm,
            memgene_dir=ds["memdir"], method_files=files, universe=universe,
            verbose=False)
        # background is NOT the complement of this top-N call — it is every gene
        # no method ever called, at any depth (see xd_background)
        # background = genes no method calls within the SAME top N as the consensus
        bg = xd_background(ds, universe, to_data, tp, mm, bg_mode,
                           (bg_top_n or top_n) if bg_mode == "top_n" else top_n,
                           files=files, bg_exclude=bg_exclude, verbose=verbose)
        bgset = set(bg)
        n_meth = len(files)
        for k in tiers:
            if k > n_meth:
                continue
            mem = [g for g in to_data(counts.index[counts >= k].tolist())
                   if g not in bgset]
            if len(mem) < min_genes:
                if verbose:
                    print(f"  skip {ds['name'].splitlines()[-1]} >={k}: only {len(mem)} genes")
                continue
            frame = gs.loc[list(dict.fromkeys(mem + bg))]
            m = compare_gene_responsiveness_expr_matched(
                frame, expr, mem, metric="mean_abs_lfc", n_bins=n_bins, n_iter=n_iter)
            a, b = m["scg_values"], m["non_scg_values"]
            bal = expression_balance(frame, expr, mem, n_bins=n_bins, n_iter=n_iter,
                                     metric="mean_abs_lfc")
            row = dict(dataset=ds["name"], method=f">={k}", top_n=top_n, n=len(mem),
                       n_bg=len(bg), smd_expr=bal["smd_expr"], cliffs_d=cliffs_delta(a, b),
                       pct=(np.median(a) / np.median(b) - 1) * 100)
            if ci:
                row["d_lo"], row["d_hi"] = cliffs_ci(a, b)
                row["pct_lo"], row["pct_hi"] = pct_ci(a, b)
            rows.append(row)
        if verbose:
            print(f"done: {ds['name'].splitlines()[-1]} (consensus @top{top_n})", flush=True)
    out = pd.DataFrame(rows)
    out.attrs["bg_mode"] = bg_mode
    out.attrs["axis"] = "consensus"
    if cache:
        _CACHE[key] = out
    return out


def fig_cross_dataset_sweep(kind="full", S=None, datasets=None, top_ns=None,
                            top_n=TOP_N, cons_top_n=500, tiers=(2, 3, 4, 5), metric="pct",
                            methods=None, group=None, drop_canonical_scg=True,
                            n_bins=50, n_iter=100, timepoint="merge",
                            merge_mode="union", vmax=None, annot=True, FS=22,
                            cell=(1.05, 1.30), pad=(5.5, 2.4), title=None, note=None,
                            block_headers=None, xtick_rotation=None, title_pad=None,
                            show_n=True, n_fmt="n={:.0f}", n_fs=None,
                            cbar_fs=None, cbar_width=2.6, cbar_len=1.0,
                            one_sided="auto", cmap=None, annot_fmt=None,
                            cbar_zero_label=None, cbar_below_label=None,
                            cbar_below_gap_pt=12, cbar_label_color="0.35",
                            cbar_label=None, cbar_top_label=None, cbar_ticks=None,
                            cbar_text_rotation=0,
                            fs_ticks=None, fs_rows=None, fs_header=None, fs_xlabel=None,
                            annot_fs=None,
                            bg_mode="top_n", bg_top_n=None, floor=EXPR_FLOOR,
                            save=None, dpi=300, show=True, verbose=True):
    """(Suppl. Fig 4b) Cross-dataset responsiveness heatmap, in three variants.

    kind = "full"      rows = perturb x lineage dataset, columns = method x cutoff
                       (the original figure; `top_ns` sets the cutoffs)
           "cutoff"    a SINGLE cutoff (`top_n`) -> columns are just the methods,
                       so the datasets can be compared method-by-method with no
                       threshold axis to read past
           "consensus" columns are consensus tiers >=2..>=5 at one fixed `top_n`,
                       i.e. "does agreement between methods buy responsiveness?"

    drop_canonical_scg : as in the source notebook, the HEATMAP drops the canonical
        clone_size==2 SCG column and shows the widened SCG-2-8 in its place labelled
        'SCG'. On Harmange the canonical setting calls 0 genes per sorted arm (51
        sister pairs vs 1,300 in FM06), so that column would be empty for the
        melanoma rows. Cells are REUSED across pairs in the 2-8 set, so call it the
        widened variant in any write-up.

    block_headers  : the bold label above each column block. Default None = drawn
        only for kind='full', where the blocks (methods) carry information. For
        'cutof'/'consensus' there is a single block whose label just repeats the
        cutoff already in the title, so it is dropped.
    xtick_rotation : default None = 35 deg for kind='cutoff' (method names are long
        enough to collide), 0 otherwise.

    Returns (fig, ax, S) — S is the long table behind the figure.
    """
    _perturb_path()
    from marson_KO import plot_topn_heatmap, METHOD_ORDER
    datasets = datasets if datasets is not None else xd_datasets()

    if S is None:
        if kind == "consensus":
            S = xd_consensus_sweep(datasets, tiers=tiers, top_n=cons_top_n, floor=floor,
                                   n_bins=n_bins,
                                   n_iter=n_iter, timepoint=timepoint,
                                   merge_mode=merge_mode, methods=methods,
                                   bg_mode=bg_mode, bg_top_n=bg_top_n,
                                   verbose=verbose)
        else:
            tns = [top_n] if kind == "cutoff" else list(top_ns or XD_TOPNS)
            S = xd_sweep(datasets, top_ns=tns, methods=methods, floor=floor, n_bins=n_bins,
                         n_iter=n_iter, timepoint=timepoint, merge_mode=merge_mode,
                         bg_mode=bg_mode, bg_top_n=bg_top_n, verbose=verbose)

    d = S.copy()
    if kind == "consensus":
        order = [f">={k}" for k in tiers if f">={k}" in set(d["method"])]
        group = group or "top_n"          # one block, methods(=tiers) inside it
        note = note if note is not None else ""
        title = title or ("" if not SHOW_TITLES else
                          "Method agreement vs KO responsiveness "
                          f"(top {cons_top_n} per method)")
    else:
        if drop_canonical_scg and "SCG-2-8" in set(d["method"]):
            d = d[d["method"] != "SCG"].copy()
            d["method"] = d["method"].astype(str).replace({"SCG-2-8": "ClonePair"})
        order = [m for m in (methods or METHOD_ORDER) if m in set(d["method"])]
        if kind == "cutoff":
            group = group or "top_n"      # single block -> columns are the methods
            title = title or ("" if not SHOW_TITLES else
                          f"Memory-gene KO responsiveness at top {top_n}")
        else:
            group = group or "method"     # blocked by method, cutoffs inside
    if show_n:
        annot = False        # drawn below instead, so the gene count can go with it
    if block_headers is None:
        block_headers = (kind == "full")
    if xtick_rotation is None:
        xtick_rotation = 35 if kind == "cutoff" else 0

    # draw without saving/showing, so the post-processing below lands in the file too
    fig, ax = plot_topn_heatmap(
        d, datasets, methods=order,
        top_ns=(tiers and None) if kind == "consensus" else None,
        metric=metric, group=group, vmax=vmax,
        # when we reformat the cells ourselves, let the helper draw nothing, or the
        # two sets of numbers land on top of each other
        annot=(annot and not (show_n or annot_fmt)), FS=FS, cell=cell, pad=pad,
        fs_ticks=fs_ticks, fs_rows=fs_rows, fs_header=fs_header, fs_xlabel=fs_xlabel,
        title=title if title else " ", note=note if note is not None else "",
        save_path=None, show=False)

    # value + n in each cell. plot_topn_heatmap only writes the value, so the
    # annotation is redrawn here with the gene count underneath it.
    if cmap is not None and ax.images:
        ax.images[0].set_cmap(cmap)
    if show_n or annot_fmt:
        from marson_KO import METHOD_ORDER as _MO
        _tns = ([cons_top_n] if kind == "consensus" else
                [top_n] if kind == "cutoff" else list(top_ns or XD_TOPNS))
        _cols = ([(m, t) for m in order for t in _tns] if group == "method"
                 else [(m, t) for t in _tns for m in order])
        _rows = [d["name"] for d in datasets if d["name"] in set(d2 for d2 in S["dataset"])]
        _v = d.pivot_table(index="dataset", columns=["method", "top_n"],
                           values={"pct": "pct", "pctile": "cliffs_d",
                                   "winprob": "cliffs_d"}[metric] if metric != "pct"
                                  else "pct")
        _n = d.pivot_table(index="dataset", columns=["method", "top_n"], values="n")
        _v = _v.reindex(index=_rows, columns=pd.MultiIndex.from_tuples(_cols))
        _n = _n.reindex(index=_rows, columns=pd.MultiIndex.from_tuples(_cols))
        # the switch to white text must follow the COLOUR scale, not the data
        # maximum: the scale is capped at the 98th percentile, so a cell well below
        # max(|value|) can still be dark enough to swallow dark text. At this point
        # the image still carries the DIVERGING limits; the one-sided rescale below
        # has not run yet, so anticipate it rather than read the current clim.
        _lo, _hi = (ax.images[0].get_clim() if ax.images
                    else (0.0, np.nanmax(np.abs(_v.values)) or 1.0))
        _base = (_lo + _hi) / 2
        _neg = bool(np.nanmin(_v.values) < _base)
        _one = (one_sided is True) or (one_sided == "auto" and not _neg)
        _cut = (_base + .60 * (_hi - _base)) if _one else (_lo + .60 * (_hi - _lo))
        for r_ in range(_v.shape[0]):
            for c_ in range(_v.shape[1]):
                val, nn = _v.values[r_, c_], _n.values[r_, c_]
                if not np.isfinite(val):
                    continue
                txt = (annot_fmt.format(val) if annot_fmt else
                       f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}")
                if show_n and np.isfinite(nn):
                    txt += "\n" + n_fmt.format(nn)
                ax.text(c_, r_, txt, ha="center", va="center",
                        fontsize=annot_fs or n_fs or FS - 6,
                        color="w" if val > _cut else "#222222")

    if not block_headers:
        # the headers are the only texts placed in xaxis_transform at y=1.012
        for t in list(ax.texts):
            if abs(t.get_position()[1] - 1.012) < 1e-9 and t.get_transform() is not ax.transData:
                t.remove()
        ax.title.set_position((.5, 1.0))
        ax.set_title(ax.get_title(), fontsize=FS + 3,
                     pad=title_pad if title_pad is not None else 14)
    elif title_pad is not None:
        ax.set_title(ax.get_title(), fontsize=FS + 3, pad=title_pad)

    if xtick_rotation:
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right",
                 rotation_mode="anchor")

    # One-sided colour bar. Memory genes respond MORE in every cell of this figure,
    # so the blue half of the diverging bar is dead space that makes the reader
    # search for negatives that are not there. Redrawing the bar from `base` up with
    # the UPPER HALF of the same map over HALF the range is the identical mapping,
    # so every cell keeps the exact colour it had -- only the legend changes.
    # "auto" falls back to the diverging bar the moment a negative value appears.
    im = ax.images[0] if ax.images else None
    if im is not None and one_sided:
        lo, hi = im.get_clim()
        base_ = (lo + hi) / 2
        arr = np.ma.masked_invalid(np.asarray(im.get_array(), float))
        has_neg = bool(arr.count()) and float(arr.min()) < base_
        if has_neg and one_sided == "auto":
            pass                                  # keep the diverging bar
        else:
            if has_neg:
                warnings.warn("one_sided=True but the heatmap has values below "
                              f"{base_:g} (min {float(arr.min()):.1f}); they are "
                              "drawn at the bar's bottom end.", stacklevel=2)
            from matplotlib.colors import ListedColormap
            old = im.get_cmap()
            cm = ListedColormap(old(np.linspace(.5, 1, 256)))
            try:
                cm.set_bad(old.get_bad())
            except AttributeError:                # matplotlib < 3.8
                cm.set_bad("#d9d9d9")
            im.set_cmap(cm)
            im.set_clim(base_, hi)
            # rebuild the bar: extend='both' was baked in by plot_topn_heatmap
            olds = [a for a in fig.axes if a is not ax]
            lab = olds[-1].get_ylabel() if olds else ""
            lab_fs = olds[-1].yaxis.label.get_fontsize() if olds else FS - 3
            tick_fs = (olds[-1].get_yticklabels()[0].get_fontsize()
                       if olds and olds[-1].get_yticklabels() else FS - 4)
            for a in olds:
                a.remove()
            cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015, extend="max")
            cb.set_label(lab, fontsize=lab_fs)
            cb.ax.tick_params(labelsize=tick_fs)
            cb.outline.set_visible(False)

    fig.tight_layout()

    # colour scale: plot_topn_heatmap builds it at fraction=0.028, which is a hair
    # at this figure size. Resized AFTER tight_layout, or the layout pass undoes it.
    cbs = [a for a in fig.axes if a is not ax]
    if cbs and (cbar_width != 1 or cbar_len != 1 or cbar_fs):
        cax = cbs[-1]
        p = cax.get_position()
        cax.set_position([p.x0, p.y0 + p.height * (1 - cbar_len) / 2,
                          p.width * cbar_width, p.height * cbar_len])
        if cbar_fs:
            cax.tick_params(labelsize=cbar_fs)
            if cax.yaxis.label is not None:
                cax.yaxis.label.set_fontsize(cbar_fs + 1)
    # labels beside the bar's ZERO end, as in fig_cross_dataset_combo: what 0 means,
    # and what a negative value would mean. Placed after the resize so they follow
    # the bar's final position.
    if cbs and (cbar_label is not None or cbar_ticks is not None):
        _cax = [a for a in fig.axes if a is not ax][-1]
        if cbar_label is not None:
            _cax.set_ylabel(cbar_label)
        if cbar_ticks is not None:
            # an empty sequence leaves the bar as a pure gradient with its arrow
            _cax.set_yticks(list(cbar_ticks))
    if (cbar_zero_label or cbar_below_label or cbar_top_label) and cbs:
        cax = [a for a in fig.axes if a is not ax][-1]
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        xs = ([t.get_window_extent(rend).x1 for t in cax.get_yticklabels()]
              or [cax.get_window_extent(rend).x1])
        x_txt = (max(xs) / fig.dpi + 6 / 72) / fig.get_figwidth()
        pos = cax.get_position()
        _fs = cbar_fs or FS - 4
        # rotated: the labels run UP the side of the bar instead of out to the
        # right, which is what makes a full-page-width heatmap affordable
        _rot = float(cbar_text_rotation or 0)
        _kw = (dict(rotation=_rot, rotation_mode="anchor") if _rot else {})
        if cbar_top_label:
            fig.text(x_txt, pos.y1, cbar_top_label,
                     ha="right" if _rot else "left", va="top",
                     fontsize=_fs, color=cbar_label_color, linespacing=1.05, **_kw)
        if cbar_zero_label:
            fig.text(x_txt, pos.y0, cbar_zero_label,
                     ha="left", va="top" if _rot else "center",
                     fontsize=_fs, color=cbar_label_color, **_kw)
        if cbar_below_label:
            _gap = (cbar_below_gap_pt / 72) / fig.get_figheight()
            fig.text(x_txt, pos.y0 - _gap, cbar_below_label, ha="left", va="top",
                     fontsize=_fs, color=cbar_label_color, linespacing=1.15)

    save_fig(fig, save, f"cross_dataset_{kind}_{metric}", dpi=dpi, verbose=verbose)
    if show:
        plt.show()
    return fig, ax, S


# ═══════════════════════════════════════════════════════════════════════════
#  Study-design sunburst: which lineage-barcode dataset is tested in which
#  perturb-seq screen, and with what perturbation modality.
# ═══════════════════════════════════════════════════════════════════════════

# One row per LEAF = one (lineage dataset x perturb-seq dataset) pairing, which is
# exactly one row of the cross-dataset figures. Columns are the rings, inner->outer.
# Rows must be grouped by hierarchy (siblings adjacent); a repeated label under a
# DIFFERENT parent correctly becomes its own wedge — Shi appears twice because it
# is paired with both Mold and Bandler.
# NOT "T cell": T-lineage's own cell_type column is 33% Myeloid Progenitors,
# 32% MPP/HSC, 17% Progenitor T Cells, 13% Mast, 5% Erythroid. It is multi-lineage
# haematopoiesis; only the SCREENS (Marson, Weinstock) are pure T cells. Calling
# the ring "T cell" would assert something the data contradicts — and would hide
# the mismatch that explains why 56% of its memory genes are silent in Marson.
DESIGN_ROWS = [
    #  system            lineage-barcode   perturb-seq   modality
    ("Haematopoiesis",   "T-lineage",      "Zhu",        "CRISPRi"),
    ("Haematopoiesis",   "T-lineage",      "Weinstock",  "CRISPR-KO"),
    ("Melanoma",         "Harmange",       "Xu",         "CRISPRi"),
    ("Melanoma",         "Harmange",       "Frangieh",   "CRISPR-KO"),
    ("Brain",            "Bandler",        "Shi",        "CRISPR-KO"),
]
# Kept short on purpose: these sit in the reserved gap, and the gap's ARC LENGTH
# shrinks toward the centre (r * theta), so a long word on an inner ring spills
# over the neighbouring wedges no matter how wide the gap is.
DESIGN_RING_LABELS = ("system", "lineage", "perturb-seq", "modality")
DESIGN_COLORS = ["#B02318", "#3B6EA5", "#2E7D5B", "#9467bd", "#E08A5F"]

# Library sizes AS PUBLISHED, kept only so perturb_screen_counts() can print the
# two side by side. The figure does NOT draw these: Shi and Frangieh report 1,947
# and 248 targets but 1,943 and 235 survive the 20-cell threshold, so a ring built
# from this dict would mix library sizes with analysed counts (2026-10-26).
PERTURB_N_PUBLISHED = {"Zhu": 6646, "Weinstock": 84, "Shi": 1947, "Xu": 142,
                       "Frangieh": 248, "Bowness": 520}

# The modality ring is coloured by MODALITY, not by the system family: the point of
# that ring is that Marson/Xu share a perturbation type across different tissues.
# Two shades of one hue keep them readable as a single category.
MODALITY_COLORS = {"CRISPRi": "#BCBDDC", "CRISPR-KO": "#756BB1"}
DESIGN_LEVEL_COLORS = {3: MODALITY_COLORS}

# Which levels form the grouping key for each ring. By default a wedge is defined
# by its FULL path, so Shi — one screen, used against both Mold and Bandler — would
# be drawn twice. Keying the perturb-seq ring on (system, screen) and the modality
# ring on (system, screen, modality) merges those two into a single wedge spanning
# both lineage datasets, without merging Frangieh's CRISPR-KO into Shi's.
DESIGN_KEY_LEVELS = {2: (0, 2), 3: (0, 2, 3)}


def _lighten(color, f):
    """Blend toward white by fraction f (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _curved_text(ax, r, ang_deg, text, fontsize, color, weight="normal",
                 zorder=4, rend=None, max_span=None):
    """Set `text` along the arc of radius r, centred on ang_deg.

    matplotlib has no curved text, so each glyph is placed and rotated
    individually. Widths are measured with the renderer rather than estimated,
    otherwise proportional fonts drift badly over a long label.
    """
    fig = ax.get_figure()
    rend = rend or fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def _measure(fs_):
        ws = []
        for ch in text:
            t = ax.text(0, 0, ch, fontsize=fs_, fontweight=weight)
            bb = t.get_window_extent(rend).transformed(inv)
            ws.append(max(bb.width, 1e-6))
            t.remove()
        return ws, np.degrees(sum(ws) / max(r, 1e-9))

    widths, span = _measure(fontsize)
    # A label wider than its wedge used to just run over the neighbours. Shrink it
    # to fit instead, but never below the house minimum — if it still does not fit
    # at that size the wedge genuinely needs a leader line.
    if max_span and span > max_span:
        fontsize = fsize(fontsize * max_span / span)
        widths, span = _measure(fontsize)
    # unflipped glyph rotation is (a - 90), which is upside down exactly when the
    # angle is in the LOWER half of the circle — not the upper one.
    # The interval is half-open at 0 and CLOSED at 180 on purpose. A label centred
    # exactly on 0 degrees is the one ambiguous case: both branches put it exactly
    # vertical, but ">180" sent it reading downward while every neighbour a few
    # degrees either side read upward, so the one label came out reversed. Treating
    # 0 as flipped matches what 359 does, and 180 keeps the branch it already had.
    a360 = ang_deg % 360
    flip = not (0 < a360 <= 180)
    sgn = -1 if not flip else 1
    start = ang_deg - sgn * span / 2
    out, acc = [], 0.0
    for ch, wch in zip(text, widths):
        dth = np.degrees(wch / max(r, 1e-9))
        a = start + sgn * (acc + dth / 2)
        rot = a - 90 if not flip else a + 90
        out.append(ax.text(r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a)), ch,
                           ha="center", va="center", rotation=rot,
                           rotation_mode="anchor", fontsize=fontsize,
                           fontweight=weight, color=color, zorder=zorder))
        acc += dth
    return out


def _upright(rot):
    """Normalise a text rotation into [-90, 90] so it is never upside down."""
    rot = (rot + 180) % 360 - 180
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    return rot


def _sunburst_nodes(rows, weights=None, start_angle=90, clockwise=True, gap_deg=0.0,
                    key_levels=None):
    """Rows of equal-depth tuples -> one node per wedge, with angular extents.

    Siblings are grouped by CONSECUTIVE identical prefixes, so the row order is
    the drawing order: sort rows by hierarchy before calling.
    """
    n = len(rows)
    w = list(weights) if weights is not None else [1.0] * n
    edges = np.concatenate([[0], np.cumsum(w)]) / sum(w) * (360.0 - gap_deg)
    sign = -1 if clockwise else 1
    ang = lambda a: start_angle + sign * a
    nodes = []
    key_levels = key_levels or {}
    for lev in range(len(rows[0])):
        kl = key_levels.get(lev, tuple(range(lev + 1)))
        keyf = lambda r: tuple(r[c] for c in kl)
        i = 0
        while i < n:
            key = keyf(rows[i])
            j = i
            while j < n and keyf(rows[j]) == key:
                j += 1
            a1, a2 = ang(edges[i]), ang(edges[j])
            nodes.append(dict(level=lev, label=rows[i][lev], key=key,
                              root=rows[i][0], row=rows[i],
                              t1=min(a1, a2), t2=max(a1, a2),
                              mid=(a1 + a2) / 2, span=abs(a2 - a1),
                              first_leaf=i, n_leaves=j - i))
            i = j
    return nodes


# ---- five-ring version: the lineage side and the perturb side have DIFFERENT
# cell systems, and that mismatch is a result, not a detail (56% of T-lineage
# memory genes are silent in Marson's pure T cells). One shared "system" ring
# would hide it. VERIFY the cell-type strings against your methods section.
DESIGN_ROWS_5 = [
    # lineage cell system        lineage      screen        screen cell system      modality
    ("Human haematopoiesis",  "T-lineage", "Zhu",       "Primary human T cells", "CRISPRi"),
    ("Human haematopoiesis",  "T-lineage", "Weinstock", "Primary human T cells", "CRISPR-KO"),
    ("Melanoma",              "Harmange",  "Xu",        "A375 melanoma",         "CRISPRi"),
    ("Melanoma",              "Harmange",  "Frangieh",  "Melanoma",              "CRISPR-KO"),
    ("Mouse cortex",          "Bandler",   "Shi",       "Mouse cortex",          "CRISPR-KO"),
]
DESIGN_RING_LABELS_5 = ("lineage\nsystem", "lineage\ndataset", "screen",
                        "screen\nsystem", "modality")
# merge the duplicated Shi wedge and everything downstream of it
DESIGN_KEY_LEVELS_5 = {2: (0, 2), 3: (0, 2, 3), 4: (0, 2, 3, 4)}

# ---- publication layout: 3 rings of SHORT labels, detail in a side legend ----
# Long cell-system names cannot fit horizontally inside a ring, and pushing them
# outside produces leader lines across the whole figure. So the rings carry only
# short names and the legend carries the cell systems on both sides — which also
# puts the lineage/screen mismatch (haematopoiesis vs pure T cells) in one place
# where a reader can actually compare them.
DESIGN_ROWS_PUB = [
    ("T-lineage", "Zhu",       "CRISPRi"),
    ("T-lineage", "Weinstock", "CRISPR-KO"),
    ("Harmange",  "Xu",        "CRISPRi"),
    ("Harmange",  "Frangieh",  "CRISPR-KO"),
    ("Bandler",   "Shi",       "CRISPR-KO"),
]
DESIGN_KEY_LEVELS_PUB = {1: (1,), 2: (1, 2)}      # one Shi wedge, one modality wedge
# lineage dataset -> (colour family key, lineage cell system, screen cell system)
DESIGN_SYSTEMS = {
    "T-lineage": ("Human haematopoiesis", "primary human T cells"),
    "Harmange":  ("Melanoma (scMemorySeq)", "A375 / patient melanoma"),
    "Bandler":   ("Mouse cortex", "mouse cortex"),
}


def fig_design_sunburst(rows=DESIGN_ROWS, weights=None, ring_labels=DESIGN_RING_LABELS,
                        colors=None, r0=.38, ring_w=.22, gap=.012, start_angle=None,
                        clockwise=True, fs=11, label_orient="auto",
                        label_gap_deg=36, level_colors=None, key_levels=None,
                        outside_labels=(), label_offset=None, curved_labels=False,
                        edge_labels=None, edge_pad=.09, edge_fs=None,
                        edge_rotate="radial", edge_at_ray=True, lim=None,
                        count_bars="hub",
                        ray_span=.30, ray_min_frac=.10,
                        edge_label_color="0.25",
                        family_of=None, family_level=0, label_colors=None,
                        counts=None, count_level=2, count_in_label=False,
                        count_at_bar=False, count_fmt="{:,}", count_fs=None,
                        hub_bars=True, hub_r0=.10, hub_pad=.10,
                        hub_scale="log", hub_min_frac=.16, hub_bar_frac=.74,
                        hub_axis=True, hub_ticks=(100, 1000, 10000),
                        outside_pad=.10, outside_min_dy=.16,
                        side_legend=None, legend_x=1.02,
                        legend_fs=None, ring_note=None,
                        min_span_inside=14, lighten_step=.17, edge_color="white",
                        edge_lw=1.6, center_text=None, center_fs=None,
                        ring_label_fs=None, ring_label_color="0.30",
                        ring_label_weight="normal", ring_label_stroke="white",
                        ring_label_stroke_lw=3, ring_label_bg=None,
                        ring_label_bg_alpha=.85, ring_label_bg_pad=.35,
                        ring_label_bg_round=.35,
                        ring_label_angle=None, leader_color="0.45",
                        title=None, figsize=(7.2, 7.2), save=None, dpi=400,
                        show=True, verbose=True):
    """Nested-ring (sunburst) summary of the study design.

    rows        : list of equal-length tuples, inner ring first. One row per leaf.
    weights     : per-leaf angular weight (default: every leaf equal).
    curved_labels : set each wedge label along its arc, one glyph at a time, as in
                  a classic sunburst. Only sensible with label_orient='tangential'.
    edge_labels : {leaf label: text} drawn just OUTSIDE the rim at that wedge's
                  angle — for counts that would not fit inside the wedge.
                  Multi-line text is fine.
    count_bars : where the `counts` bars go. 'hub' (default) draws them inside the
                  ring, 'extend' lengthens each leaf wedge itself so the ring and
                  the bar are one continuous shape, 'rays' radiates a separate,
                  narrower bar out of the rim, one per group wedge —
                  each ray sits on the wedge it belongs to, so the reader can see
                  which quantity is which without a key. ray_span sets the longest
                  ray's length in data units, ray_min_frac the shortest one's share.
    lim : axis half-width. The default leaves room for the worst case, which wastes
                  ~15% of the panel; measure the drawn content and pass a tighter
                  number when the panel size actually matters.
    edge_at_ray : with count_bars='rays', put each count block at the tip of its
                  OWN ray instead of on one common outer circle. Closer to what it
                  describes; set False if you prefer the labels aligned.
    edge_rotate : 'radial' (default) sets those counts along the spoke, which is
                  compact; 'horizontal' keeps them upright and easier to read but
                  needs more room at the sides.
    label_orient: 'auto' (default — horizontal where the wedge has room, otherwise
                  along the arc, and only outside with a leader line as a last
                  resort), 'horizontal' (leader lines for anything that does not
                  fit), 'tangential' (text follows the arc; radial text runs
                  along the radius and a name like "Weinstock" is longer than the
                  ring is wide, so it spills into the neighbouring rings),
                  'radial', or 'horizontal'.
    min_span_inside : wedges narrower than this many degrees get an outside label
                  with a leader line instead of text inside the wedge.
    counts      : {row tuple: n} (unambiguous; use it whenever two parents share a
                  group name) or {label: n} keyed on the ring at `count_level` (default the
                  perturb-seq ring). Drives the hub bar heights. The values are NOT
                  printed by default — set count_at_bar (tip of each bar) or
                  count_in_label (second line under the name) to show them.
    hub_scale   : 'log' (default) or 'linear'. The screens span 84 to 6,646
                  perturbations, a 79-fold range, so a linear hub makes every bar
                  but Marson's invisible.
    level_colors: {level: {label: color}} to colour a ring by its own categories
                  instead of the system family — used for the modality ring.
    family_level: which element of a row defines the colour family. 0 (default) is
                  the innermost ring; 1 gives every dataset its own colour, which
                  is what you want when the innermost ring is species and you are
                  encoding species as HUE and dataset as SHADE.
    ring_labels : short name per ring, written along each ring. None to omit.
    ring_label_bg : facecolor of a rounded box behind each ring name, e.g. "0.35".
                  Use it when the ring colours vary too much for one text colour
                  to stay legible; ring_label_bg_alpha / _pad / _round tune it.
    label_gap_deg : degrees of blank arc reserved for the ring names, centred on
                  ring_label_angle. Without it the names print on top of whichever
                  wedges happen to sit at that angle. 0 = full circle.
    outside_labels : labels that ALWAYS go outside on a leader line, whatever
                  label_orient says — for the one or two wedges too narrow to hold
                  their text.
    label_offset : {key: degrees} to slide a wedge's text along its own arc. The
                  key may be the node's full path tuple (e.g. ("Bandler","Brain")),
                  a (level, label) pair, or a bare label — most specific wins, so a
                  label shared by two wedges can be moved independently.
                  Was: {label: degrees} to slide a wedge's text along its own arc,
                  away from its midpoint. Use it to clear the ring names, which sit
                  at a fixed angle and would otherwise collide with whatever wedge
                  labels happen to be there. Positive = counter-clockwise.

    Returns (fig, ax, nodes) — nodes carries every wedge's label and angular span.
    """
    from matplotlib.patches import Wedge

    rla = ring_label_angle if ring_label_angle is not None else -90.0
    if label_gap_deg and start_angle is None:
        start_angle = rla + 360.0 - label_gap_deg / 2.0
    elif start_angle is None:
        start_angle = 90.0
    # measured from the data (20-cell threshold applied), never the published
    # library size -- see PERTURB_N_PUBLISHED
    counts = perturb_screen_counts(verbose=False) if counts is None else counts
    key_levels = DESIGN_KEY_LEVELS if key_levels is None else key_levels
    nodes = _sunburst_nodes(rows, weights, start_angle, clockwise, label_gap_deg,
                            key_levels)
    nlev = len(rows[0])
    colors = list(colors or DESIGN_COLORS)
    level_colors = DESIGN_LEVEL_COLORS if level_colors is None else level_colors
    label_colors = MODALITY_COLORS if label_colors is None else label_colors
    # Colour FAMILY: by default the innermost ring, but family_of lets several
    # inner wedges share one family (Mold and Bandler are both mouse cortex, so
    # without it Bandler would get its own unrelated colour).
    fam = ((lambda r: family_of.get(r[family_level], r[family_level]))
           if family_of else (lambda r: r[family_level]))
    roots = list(dict.fromkeys(fam(r) for r in rows))
    base = {k: colors[i % len(colors)] for i, k in enumerate(roots)}

    def _facecolor(nd):
        # by LABEL first, so a modality keeps its colour whatever ring it lands on
        if nd["label"] in (label_colors or {}):
            return label_colors[nd["label"]]
        override = (level_colors or {}).get(nd["level"], {})
        if nd["label"] in override:
            return override[nd["label"]]
        return _lighten(base[fam(nd["row"])], lighten_step * nd["level"])

    with plt.rc_context({**PUB_RC, "font.size": fs}):
        fig, ax = plt.subplots(figsize=figsize)
        placed = []
        for nd in nodes:
            lev = nd["level"]
            ri = r0 + lev * ring_w
            ro = ri + ring_w - gap
            c = _facecolor(nd)
            ax.add_patch(Wedge((0, 0), ro, nd["t1"], nd["t2"], width=ro - ri,
                               facecolor=c, edgecolor=edge_color, lw=edge_lw,
                               zorder=2))
            # ---- label: horizontal, inside if it fits, else outside ----
            rmid = (ri + ro) / 2
            _lo = label_offset or {}
            # two wedges can share a label ("Brain" under both Mold and Bandler),
            # so the full key wins, then (level, label), then the bare label
            off = _lo.get(tuple(nd["key"]), _lo.get((nd["level"], nd["label"]),
                                                    _lo.get(nd["label"], 0.)))
            off = float(off)
            # keep the nudged label inside its own wedge
            off = float(np.clip(off, -nd["span"] / 2 * .8, nd["span"] / 2 * .8))
            a = np.deg2rad(nd["mid"] + off)
            import matplotlib.colors as _mc
            dark = sum(_mc.to_rgb(c)) / 3 < .62      # luminance of the ACTUAL fill
            txt = nd["label"]
            if counts and lev == count_level and nd["label"] in counts and count_in_label:
                txt = f"{nd['label']}\n{count_fmt.format(counts[nd['label']])}"
            placed.append(dict(nd=nd, txt=txt, ri=ri, ro=ro, rmid=rmid, a=a,
                               ang=nd["mid"] + off, dark=dark))

        # ---- resolve label placement now that the axes exist ----
        lim = lim or r0 + nlev * ring_w + max(outside_pad + .34, .40) \
              + (.30 if edge_labels else 0.) \
              + (ray_span if (counts and hub_bars
                              and count_bars in ("rays", "extend")) else 0.)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()

        def _fits(t, p):
            """Does the HORIZONTAL label fit inside its wedge?

            The wedge's two free directions are radial and tangential, and which of
            those is 'horizontal' depends on where the wedge sits: at the top of the
            circle a wide label needs tangential room, at the right it needs RADIAL
            room (only one ring width). Project the text box onto both axes instead
            of assuming.
            """
            bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
            th = p["a"]
            c, sn = abs(np.cos(th)), abs(np.sin(th))
            radial_need = bb.width / 2 * c + bb.height / 2 * sn
            tang_need = bb.width / 2 * sn + bb.height / 2 * c
            chord = 2 * p["rmid"] * np.sin(np.deg2rad(min(p["nd"]["span"], 180)) / 2)
            return (radial_need <= (p["ro"] - p["ri"]) / 2 * .92
                    and tang_need <= chord / 2 * .92)

        # A leaf whose name is pushed outside carries its counts with it, on the
        # same leader line — otherwise the two annotations land on the same spot.
        outside, edge_done = [], set()

        def _with_edge(p):
            lb = p["nd"]["label"]
            if edge_labels and p["nd"]["level"] == nlev - 1 and lb in edge_labels:
                p["txt"] = p["txt"] + "\n" + edge_labels[lb]
                edge_done.add(lb)
            return p

        for p in placed:
            t = ax.text(p["rmid"] * np.cos(p["a"]), p["rmid"] * np.sin(p["a"]),
                        p["txt"], ha="center", va="center", fontsize=fs,
                        linespacing=1.2, rotation_mode="anchor",
                        color="white" if p["dark"] else "0.12", zorder=4)
            if p["nd"]["label"] in set(outside_labels):
                t.remove()
                outside.append(_with_edge(p))
                continue
            if label_orient in ("horizontal", "auto") and not _fits(t, p):
                if label_orient == "auto":
                    # horizontal did not fit -> follow the arc, which is what the
                    # wedge actually has room for. Only if THAT fails does the
                    # label go outside; leader lines are the last resort, not the
                    # default, or the figure turns into spaghetti.
                    t.set_rotation(_upright(nd_mid := p["nd"]["mid"] - 90))
                    if _fits(t, p):
                        continue
                t.remove()
                outside.append(_with_edge(p))
            elif label_orient == "tangential":
                if curved_labels:
                    t.remove()
                    _curved_text(ax, p["rmid"], p["ang"], p["txt"], fs,
                                 "white" if p["dark"] else "0.12", rend=rend,
                                 max_span=p["nd"]["span"] * .92)
                    continue
                t.set_rotation(_upright(p["ang"] - 90))
            elif label_orient == "radial":
                t.set_rotation(_upright(p["ang"]))

        # outside labels: just beyond the outer ring, at the wedge's OWN angle, so
        # the leader is short. Only if two land on top of each other are they
        # pushed apart along y.
        placed_y = []
        for p in sorted(outside, key=lambda q: -np.sin(q["a"])):
            a = p["a"]
            rl = p["ro"] + outside_pad + (ray_span if (
                counts and hub_bars and count_bars in ("rays", "extend")) else 0.)
            x, y = rl * np.cos(a), rl * np.sin(a)
            for yy in placed_y:                       # nudge off a taken slot
                if abs(y - yy) < outside_min_dy:
                    y = yy - outside_min_dy if y <= yy else yy + outside_min_dy
            placed_y.append(y)
            side = 1 if np.cos(a) >= 0 else -1
            ax.annotate(p["txt"], xy=(p["ro"] * np.cos(a), p["ro"] * np.sin(a)),
                        xytext=(x + side * .04, y),
                        ha="left" if side > 0 else "right", va="center",
                        fontsize=fs, color="0.12", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=leader_color,
                                        lw=.8, shrinkA=0, shrinkB=2,
                                        connectionstyle="arc3,rad=0"))

        # ---- counts as rays radiating out of the rim ("sun") ----
        r_rim = r0 + nlev * ring_w
        ray_out, ray_end = 0.0, {}
        if counts and hub_bars and count_bars in ("rays", "extend"):
            leaves = [nd for nd in nodes if nd["level"] == nlev - 1]
            # full row first, then the label: keyed on the label alone, datasets whose
            # groups share a name (Michaels and Weinreb T1/T2) take each other's counts
            vals = [counts.get(tuple(rows[nd["first_leaf"]]),
                               counts.get(rows[nd["first_leaf"]][count_level], np.nan))
                    for nd in leaves]
            good = [v for v in vals if np.isfinite(v) and v > 0]
            if good:
                # 'linear' is the honest choice for rays: length is then simply
                # proportional to the count, which is what a reader assumes.
                if hub_scale == "log":
                    lo, hi = np.log10(min(good)), np.log10(max(good))
                    nrm = lambda v: (np.log10(v) - lo) / max(hi - lo, 1e-9)
                else:
                    lo, hi = 0.0, max(good)
                    nrm = lambda v: (v - lo) / max(hi - lo, 1e-9)
                for nd, v in zip(leaves, vals):
                    if not (np.isfinite(v) and v > 0):
                        continue
                    ln = (ray_min_frac + (1 - ray_min_frac) * nrm(v)) * ray_span
                    if count_bars == "extend":
                        # continue the group's own wedge outward: no gap, no new
                        # shape, so the reader sees ONE wedge whose length is the
                        # cell count.
                        a0, a1, r_lo = nd["t1"], nd["t2"], r_rim - gap
                    else:
                        wdt = nd["span"] * hub_bar_frac
                        a0, a1, r_lo = (nd["mid"] - wdt / 2,
                                        nd["mid"] + wdt / 2, r_rim)
                    ray_end[nd["label"]] = r_lo + ln
                    ax.add_patch(Wedge((0, 0), r_lo + ln, a0, a1,
                                       width=ln,
                                       facecolor=_lighten(base[fam(nd["row"])],
                                                          lighten_step * nd["level"]),
                                       edgecolor="white", lw=1.0, zorder=3))
                ray_out = ray_span

        # ---- counts just outside the rim, one per leaf ----
        if edge_labels:
            # each block sits at the tip of ITS OWN ray, so the number is next to
            # the bar it describes rather than on a common outer circle.
            r_edge0 = r_rim + ray_out + edge_pad
            for nd in nodes:
                if (nd["level"] != nlev - 1 or nd["label"] not in edge_labels
                        or nd["label"] in edge_done):
                    continue
                aa = np.deg2rad(nd["mid"])
                r_edge = (ray_end[nd["label"]] + edge_pad
                          if edge_at_ray and nd["label"] in ray_end else r_edge0)
                cx, cy = np.cos(aa), np.sin(aa)
                if edge_rotate == "horizontal":
                    # upright text, anchored on the side it sits on so it always
                    # grows away from the wedge.
                    rot = 0
                    ha = "left" if cx > .15 else ("right" if cx < -.15 else "center")
                    va = "center" if abs(cy) < .3 else ("bottom" if cy > 0 else "top")
                else:
                    # anchor at the rim and read OUTWARD, so nothing reaches back
                    # into the wedge and over the group name.
                    rot = _upright(nd["mid"])
                    flip = abs((rot - nd["mid"] + 180) % 360 - 180) > 90
                    ha, va = ("right" if flip else "left"), "center"
                ax.text(r_edge * cx, r_edge * cy,
                        edge_labels[nd["label"]], ha=ha, va=va,
                        rotation=rot, rotation_mode="anchor",
                        fontsize=fsize(edge_fs or fs - 2), color=edge_label_color,
                        linespacing=1.25, zorder=5)

        # ---- hub: one radial bar per leaf, height = perturbations screened ----
        if counts and hub_bars and count_bars == "hub":
            leaves = [nd for nd in nodes if nd["level"] == len(rows[0]) - 1]
            # full row first, then the label: keyed on the label alone, datasets whose
            # groups share a name (Michaels and Weinreb T1/T2) take each other's counts
            vals = [counts.get(tuple(rows[nd["first_leaf"]]),
                               counts.get(rows[nd["first_leaf"]][count_level], np.nan))
                    for nd in leaves]
            good = [v for v in vals if np.isfinite(v) and v > 0]
            if good:
                if hub_scale == "log":
                    lo, hi = np.log10(min(good)), np.log10(max(good))
                    norm = lambda v: (np.log10(v) - lo) / max(hi - lo, 1e-9)
                else:
                    lo, hi = 0.0, max(good)
                    norm = lambda v: (v - lo) / max(hi - lo, 1e-9)
                rspan = (r0 - gap) - hub_r0 - hub_pad
                for nd, v in zip(leaves, vals):
                    if not (np.isfinite(v) and v > 0):
                        continue
                    h = hub_r0 + (hub_min_frac + (1 - hub_min_frac) * norm(v)) * rspan
                    w = nd["span"] * hub_bar_frac
                    ax.add_patch(Wedge((0, 0), h, nd["mid"] - w / 2, nd["mid"] + w / 2,
                                       width=h - hub_r0,
                                       facecolor=base[fam(nd["row"])],
                                       edgecolor="white", lw=1.0, zorder=3))
                    if count_at_bar:
                        aa = np.deg2rad(nd["mid"])
                        rr = h + .035
                        ax.text(rr * np.cos(aa), rr * np.sin(aa),
                                count_fmt.format(int(v)), ha="center", va="center",
                                rotation=_upright(nd["mid"] - 90),
                                rotation_mode="anchor",
                                fontsize=count_fs or fs - 2, color="0.25", zorder=4)

        # ---- ring names, written along the bottom of each ring ----
        if ring_labels:
            rla = ring_label_angle if ring_label_angle is not None else -90
            arad = np.deg2rad(rla)
            import matplotlib.patheffects as pe
            for lev, name in enumerate(ring_labels[:nlev]):
                rmid = r0 + lev * ring_w + (ring_w - gap) / 2
                rot = rla - 90
                if 90 < rla % 360 <= 270:
                    rot += 180
                t = ax.text(rmid * np.cos(arad), rmid * np.sin(arad), name,
                            ha="center", va="center", rotation=rot,
                            rotation_mode="anchor",
                            fontsize=ring_label_fs or fs - 1,
                            style="italic" if ring_label_weight == "normal" else "normal",
                            fontweight=ring_label_weight,
                            color=ring_label_color, zorder=6,
                            bbox=(dict(boxstyle=f"round,pad={ring_label_bg_pad},"
                                                f"rounding_size={ring_label_bg_round}",
                                       facecolor=ring_label_bg, edgecolor="none",
                                       alpha=ring_label_bg_alpha)
                                  if ring_label_bg else None))
                # the ring name sits ON the wedges; a stroke keeps it legible over
                # any fill colour (set ring_label_stroke=None for flat white text)
                if ring_label_stroke:
                    t.set_path_effects([pe.withStroke(
                        linewidth=ring_label_stroke_lw, foreground=ring_label_stroke)])
        if center_text:
            ax.text(0, 0, center_text, ha="center", va="center",
                    fontsize=center_fs or fs + 1, zorder=5)

        # ---- side legend: colour family -> lineage system / screen system ----
        if side_legend:
            from matplotlib.patches import Rectangle
            lfs = legend_fs or fs - 1
            y = .96
            for fam, entries in side_legend.items():
                ax.add_patch(Rectangle((legend_x, y - .035), .05, .05,
                                       transform=ax.transAxes, clip_on=False,
                                       facecolor=base.get(fam, "0.6"), edgecolor="none"))
                ax.text(legend_x + .075, y - .010, entries[0], transform=ax.transAxes,
                        fontsize=lfs, va="center", ha="left", clip_on=False)
                ax.text(legend_x + .075, y - .062, "→ " + entries[1],
                        transform=ax.transAxes, fontsize=lfs - 1, va="center",
                        ha="left", color="0.40", clip_on=False)
                y -= .135
        if ring_note:
            ax.text(.5, -.02, ring_note, transform=ax.transAxes, ha="center",
                    va="top", fontsize=(legend_fs or fs - 1), color="0.35",
                    style="italic", clip_on=False)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=fs + 2, pad=10)
        save_fig(fig, save, "design_sunburst", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, ax, nodes


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 design sunburst: figure 1's hierarchy plus the perturb-seq screen
#  each lineage dataset is tested against.
# ═══════════════════════════════════════════════════════════════════════════

# species, tissue, lineage dataset, the perturb-seq screen it is tested against.
# Shi appears twice because it is ONE screen run against both mouse datasets;
# PERTURB_KEY_LEVELS merges the two wedges back into one and the weights below
# halve it, so the screen is counted once.
# Rows in DATASET_ORDER, and the tissue ring worded as figure 1a's: the two
# figures share a colour per dataset, so they must also share its position. The
# tissue ring is worded more specifically here than in figure 1a ("Progenitor T
# Cells" and "HSPC" rather than "Blood" twice) — figure 1a's ring has to fit inside
# a wedge, this one has room, and naming the two haematopoietic systems apart is
# what shows they are not the same cells.
PERTURB_DESIGN_ROWS = [
    ("human", "Progenitor T Cells", "Michaels", "Zhu"),
    ("human", "Progenitor T Cells", "Michaels", "Weinstock"),
    ("human", "Melanoma",           "Harmange", "Xu"),
    ("human", "Melanoma",           "Harmange", "Frangieh"),
    ("mouse", "HSPC",               "Weinreb",  "Bowness"),
    ("mouse", "Brain",              "Bandler",  "Shi"),
]
PERTURB_DESIGN_RING_LABELS = ("species", "tissue", "datasets", "perturb-seq")
PERTURB_KEY_LEVELS = {3: (3,)}
# Per-label nudges, merged over the offsets computed from min_arc, in degrees
# (positive = counter-clockwise). The 0-degree boundary that used to need one is
# handled in _curved_text now; these exist because the default spin slides the ring
# names into the Mold wedge, so mouse / Brain / Mold / Shi have to step aside from
# "species" / "tissue" / "datasets" / "perturb-seq". They mirror figure 1a's
# {"mouse": -50, "Mold": -22, "b1": -18}, Shi being figure 2's outer-ring wedge
# where figure 1a has b1. Each is clipped to 40% of its own wedge.
# With Mold excluded the ring-name spoke lands in BANDLER's wedge, not Mold's, so
# it is Bandler and its Shi wedge that have to step aside. Put Mold back
# (DATASET_EXCLUDE = frozenset()) and the old tuning is
#     {"mouse": -50, "Brain": -30, "Mold": -22, "Shi": -18}
# which you pass as label_offset= for that run.
# "mouse" is +50, not the -50 it was with five datasets: the ring-name spoke now
# sits PAST the middle of the mouse arc (110 of 180 degrees) instead of before it
# (92 of 216), so the label has to step the other way to clear "species".
PERTURB_LABEL_OFFSET = {"mouse": 50, "Brain": -30, "Bandler": -22, "Shi": -18}
# figure 1's dataset colours, in figure 1's order, so the two figures read alike.
# Derived from DATASET_COLORS so the two sunbursts cannot drift apart; the live
# default is built per call from the ROWS, so excluding a dataset removes its
# colour instead of shifting its neighbours onto it.
PERTURB_DESIGN_COLORS = list(DATASET_COLORS.values())
PERTURB_SPECIES_COLORS = {"human": "#8E6FBF", "mouse": "#4A8FCB"}

_PERT_COUNTS = {}


def perturb_screen_counts(screens=None, verbose=True, cache=True):
    """Perturbations per perturb-seq screen, from the cross-dataset gene summaries.

    Read out of the SAME files xd_datasets() feeds the responsiveness sweep, so the
    figure cannot drift from the analysis: `n_obs` is the number of perturbations a
    gene was measured in, and its maximum over genes is the number of perturbations
    that entered the LFC table.

    These are NOT identical to the published library sizes (Shi 1,943 vs 1,947,
    Frangieh 235 vs 248; the rest agree) — verbose=True prints both.
    """
    screens = set(screens or [r[3] for r in PERTURB_DESIGN_ROWS])
    if cache and _PERT_COUNTS and screens <= set(_PERT_COUNTS):
        return {s: _PERT_COUNTS[s] for s in screens}
    out = {}
    for ds in xd_datasets(prune=False, verbose=False):
        _, screen, _ = _xd_pair_parts(ds["name"])
        if screen in screens:
            out[screen] = int(pd.read_csv(ds["gs"], usecols=["n_obs"])["n_obs"].max())
    missing = screens - set(out)
    if missing:
        raise KeyError(f"no gene summary in xd_datasets() for {sorted(missing)}")
    if cache:
        _PERT_COUNTS.update(out)
    if verbose:
        print("perturbations per screen:", out)
        print("published library size  :",
              {k: v for k, v in PERTURB_N_PUBLISHED.items() if k in out})
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Memory-gene characterisation: expression, detection rate, protein abundance
#  vs the never-called background, per dataset and per consensus tier.
# ═══════════════════════════════════════════════════════════════════════════

PAXDB_CSV = f"{P.DATA}/paxdb/paxdb_human_symbol.csv"
PAXDB_URL = "https://pax-db.org/downloads/5.0/datasets/9606/9606-WHOLE_ORGANISM-integrated.txt"

# label -> h5ad + memory-gene folder. `files` is an explicit per-method map because
# ipsc writes the SAME four timepoints in two different orders (coral/gemli/mold as
# d63_d60_d32_d46, powerseek/scg as d32_d46_d60_d63); a single resolved suffix
# silently drops half the methods.
def _mg(prefixes, suffix):
    return {m: f"{p}_memory_genes_{suffix}.csv" for m, p in prefixes.items()}


def _scg_files(dataset, files, scg_variant=None):
    """`files` with the SCG entry pointing at that dataset's chosen SCG run.

    The variant sits between "scg_memory_genes_" and the timepoint suffix, so the
    rewrite is a single insertion. A dataset without SCG is returned untouched, and
    so is one whose variant file is missing — falling back to the canonical run and
    saying so beats reading nothing at all.
    """
    var = scg_variant_for(dataset, scg_variant)
    # The method was renamed SCG -> ClonePair. Testing only "SCG" made this return
    # `files` untouched, so characterisation_table silently read the clone_size==2
    # lists (Harmange: 0 is_biased genes -> ClonePair dropped out). Accept either.
    key = next((k for k in ("ClonePair", "SCG") if k in files), None)
    if not var or key is None:
        return files
    new = files[key].replace("scg_memory_genes_", f"scg_memory_genes_{var}", 1)
    folder = CHAR_DATASETS.get(dataset, {}).get("folder", dataset)
    if not os.path.exists(os.path.join(BENCH, RESULTS_TREE, folder, new)):
        warnings.warn(f"[{dataset}] no {new} -> falling back to {files[key]}",
                      stacklevel=2)
        return files
    return {**files, key: new}


CHAR_DATASETS = {
    "T-lineage": dict(h5ad=f"{P.DATA}/datasets/T-lineage_processed.h5ad",
                      folder="T-lineage", species="human", var_col=None,
                      files=_mg(METHOD_PREFIX, "T1_T2")),
    "Harmange":  dict(h5ad=f"{P.DATA}/datasets/Melanoma_Harmange_processed.h5ad",
                      folder="Melanoma_Harmange", species="human", var_col=None,
                      files=_mg(METHOD_PREFIX, "E_M")),
    "LARRY":     dict(h5ad=f"{P.DATA}/datasets/LARRY_processed.h5ad",
                      folder="LARRY", species="mouse", var_col=None,
                      files=_mg(METHOD_PREFIX, "2.0_6.0_4.0")),
    # the two brain lineage datasets are kept SEPARATE: they are different
    # labelling experiments (Mold = brain1/brain2, Bandler = E13/E14 STICR) and
    # only share the Shi perturb-seq screen.
    "Bandler":   dict(h5ad=f"{P.DATA}/bandler_brain/bandler_brain.h5ad",
                      folder="BandlerBrain", species="mouse", var_col=None,
                      files=_mg(METHOD_PREFIX, "E13_E14")),
    # iPSC/organoid is defined but NOT in the default list — it is not one of the
    # perturb-seq-relevant datasets. Pass datasets=[..., "iPSC"] to include it.
    # Its five methods write the same four timepoints in two different orders, so
    # the file map has to be explicit or SCG and Power-Seek are silently dropped.
    "iPSC":      dict(h5ad=f"{P.DATA}/iPSC/itracer_organoid.h5ad",
                      folder="ipsc", species="human", var_col=None,
                      files={**_mg({k: v for k, v in METHOD_PREFIX.items()
                                    if k in ("CORAL", "GEMLI")}, "d63_d60_d32_d46"),
                             **_mg({k: v for k, v in METHOD_PREFIX.items()
                                    if k in ("SCG", "Power-Seek")}, "d32_d46_d60_d63")}),
}
# Was a frozen list. Resolved per call by active_datasets() now, so flipping
# DATASET_EXCLUDE in a notebook cell reaches this panel too; kept as a name only
# because a cell may still print it.
CHAR_DEFAULT = list(DATASET_ORDER)      # informational -- see active_datasets()
BG_MATCHED = "background (matched)"      # never shortened to "bg"

CHAR_METRICS = {
    "expression": ("log₁₀ mean expression", "log10 mean CP10k across cells"),
    "detection":  ("detection rate", "fraction of cells with non-zero counts"),
    "protein":    ("log₁₀ protein abundance", "PaxDb integrated human, ppm"),
    "length":     ("log₁₀ gene length", "exonic length of the representative transcript, bp"),
}


def load_paxdb(path=PAXDB_CSV, rebuild=False, verbose=True):
    """Per-gene protein abundance (ppm) from PaxDb's integrated human dataset.

    ~19k proteins at 99% proteome coverage. Ensembl protein ids are mapped to
    symbols with mygene once and cached; isoforms collapse by max. Human only —
    mouse datasets are ortholog-mapped before the join.
    """
    key = ("paxdb", path)
    if key in _CACHE and not rebuild:
        return _CACHE[key]
    if os.path.exists(path) and not rebuild:
        out = pd.read_csv(path, index_col=0)["abundance"]
    else:
        import requests, mygene
        from io import StringIO
        if verbose:
            print("downloading PaxDb (needs network) ...")
        txt = requests.get(PAXDB_URL, timeout=300).text
        raw = pd.read_csv(StringIO(txt), sep="\t", comment="#", header=None,
                          names=["ensp", "abundance"])
        raw["ensp"] = raw["ensp"].str.split(".").str[-1]
        res = mygene.MyGeneInfo().querymany(raw["ensp"].tolist(),
                                            scopes="ensembl.protein", fields="symbol",
                                            species="human", verbose=False)
        m = {}
        for r in res:
            if r.get("symbol") and not r.get("notfound"):
                m.setdefault(r["query"], r["symbol"])
        raw["gene"] = raw["ensp"].map(m)
        ok = raw.dropna(subset=["gene"])
        out = ok.groupby(ok["gene"].str.upper())["abundance"].max()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_csv(path, header=["abundance"])
        if verbose:
            print(f"PaxDb: mapped {len(ok)}/{len(raw)} proteins -> {len(out)} symbols")
    if verbose:
        print(f"PaxDb: {len(out)} gene symbols")
    _CACHE[key] = out
    return out


def gene_stats(label, cfg=None, chunk=4000, verbose=True):
    """Mean expression (CP10k) and detection rate per gene, from the h5ad."""
    key = ("genestats", label)
    if key in _CACHE:
        return _CACHE[key]
    import anndata as ad
    import scipy.sparse as sp
    cfg = cfg or CHAR_DATASETS[label]
    A = ad.read_h5ad(cfg["h5ad"], backed="r")
    n = A.n_obs
    tot = np.zeros(A.n_vars)
    det = np.zeros(A.n_vars)
    for X, _, _ in A.chunked_X(chunk):
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        tot += np.asarray(X.expm1().sum(0)).ravel()      # log1p(CP10k) -> CP10k
        det += np.asarray((X > 0).sum(0)).ravel()
    ids = (A.var[cfg["var_col"]].astype(str).values if cfg.get("var_col")
           else np.array(list(map(str, A.var_names))))
    A.file.close()
    df = pd.DataFrame({"gene": ids, "expression": tot / n, "detection": det / n})
    df = df.groupby("gene").max()
    if verbose:
        print(f"  [{label}] {len(df)} genes, {n} cells")
    _CACHE[key] = df
    return df


def characterisation_table(datasets=None, tiers=(2, 3, 4), top_n=TOP_N,
                           timepoint="merge", merge_mode=MERGE_MODE,
                           bg_label="background", bg_match=None, bg_ref_tier=None,
                           bg_bins=30, seed=0, scg_variant=None, verbose=True):
    """Long table: one row per (dataset, group, gene) with all three metrics.

    group is `bg_label` for genes NO method called, else '>=k' for the deepest tier
    the gene reaches — tiers are NESTED, so a >=4 gene is also in >=2. Keeping them
    nested is the point: it shows whether agreement buys anything.

    bg_match : None  the background is EVERY never-called gene (11k-22k). Unequal n
                     is not a problem for Mann-Whitney and gives the most precise
                     estimate of the background distribution.
               'n'   subsample the background to the reference tier's size. Makes
                     the boxes visually comparable; changes nothing statistically.
               'expression'  subsample to the same size AND the same expression
                     distribution as the reference tier. This is the one that
                     matters: detection rate correlates with expression at rho~0.97,
                     so an unmatched background makes rows ii and iii partly restate
                     row i. Matched, they answer "is the protein/detection
                     difference MORE than the expression difference already implies".
    bg_ref_tier : which tier the background is matched to (default: the deepest).
    """
    _perturb_path()
    from marson_KO import build_consensus_gene_sets
    datasets = active_datasets(datasets)
    pax = load_paxdb(verbose=verbose)
    rows = []
    for lab in datasets:
        cfg = CHAR_DATASETS[lab]
        # CHAR_DATASETS['files'] is built by _mg(), which spells the CANONICAL scg
        # filename. Left alone, this panel would read clone_size==2 while panels b
        # and c read 2-8 — the same figure disagreeing with itself about what a
        # memory gene is. Resolve it here, at call time, so it tracks
        # SCG_VARIANT_DEFAULT rather than freezing whatever it was at import.
        cfg = dict(cfg, files=_scg_files(lab, cfg["files"], scg_variant))
        st = gene_stats(lab, cfg, verbose=verbose)
        # protein abundance, ortholog-mapped for mouse
        idx = st.index.astype(str)
        if cfg["species"] == "mouse":
            mp = ortho_map(lab, verbose=False)      # label matches a DATASETS key
            hum = pd.Series([mp.get(g) for g in idx], index=idx)
        else:
            hum = pd.Series(idx.str.upper(), index=idx)
        st = st.assign(protein=hum.map(pax).values)
        # gene length, from the shared architecture table. NOT ortholog-mapped like
        # protein: we hold a table per species, so mouse genes are looked up in the
        # mouse one directly rather than routed through a human symbol.
        gl = load_gene_lengths(cfg["species"], "exonic_len")
        st = st.assign(length=gl.reindex(idx.str.upper()).values)

        _, counts = build_consensus_gene_sets(
            min_methods=1, timepoint=timepoint, top_n=top_n, merge_mode=merge_mode,
            memgene_dir=f"{BENCH}/{RESULTS_TREE}/{cfg['folder']}",
            method_files=cfg["files"], universe=st.index, verbose=False)
        counts = counts.reindex(st.index).fillna(0)
        for k in tiers:
            g = st.index[counts >= k]
            rows.append(st.loc[g].assign(dataset=lab, group=f"≥{k}", n_methods=k))
        bg_all = pd.Index(never_called(
            st.index, top_n, timepoint=timepoint, merge_mode=merge_mode,
            memgene_dir=f"{BENCH}/{RESULTS_TREE}/{cfg['folder']}",
            method_files=cfg["files"]))
        rows.append(st.loc[bg_all].assign(dataset=lab, group=bg_label, n_methods=0))
        bg_idx = bg_all
        if bg_match:
            ref_k = bg_ref_tier or max(tiers)
            ref = st.loc[st.index[counts >= ref_k]]
            rng = np.random.default_rng(seed)
            if bg_match == "expression" and len(ref) and len(bg_idx):
                pool = st.loc[bg_idx]
                edges = np.quantile(np.log10(pool["expression"] + 1e-6),
                                    np.linspace(0, 1, bg_bins + 1))
                edges[0], edges[-1] = -np.inf, np.inf
                pb = np.digitize(np.log10(pool["expression"] + 1e-6), edges[1:-1])
                rb = np.digitize(np.log10(ref["expression"] + 1e-6), edges[1:-1])
                pick = []
                for b, need in pd.Series(rb).value_counts().items():
                    cand = pool.index[pb == b]
                    if len(cand):
                        pick.append(rng.choice(cand, int(need),
                                               replace=need > len(cand)))
                bg_idx = pd.Index(np.concatenate(pick)) if pick else bg_idx
            elif bg_match == "n" and len(ref) < len(bg_idx):
                bg_idx = pd.Index(rng.choice(bg_idx, len(ref), replace=False))
            rows.append(st.loc[bg_idx].assign(dataset=lab, group=BG_MATCHED,
                                              n_methods=0))
        if verbose:
            print(f"  [{lab}] " + "  ".join(f">={k}: {(counts >= k).sum()}" for k in tiers)
                  + f"   background: {len(bg_all)}"
                  + (f" (+{len(bg_idx)} {bg_match}-matched to "
                     f">={bg_ref_tier or max(tiers)})" if bg_match else "")
                  + f"   PaxDb: {st['protein'].notna().sum()}")
    out = pd.concat(rows).reset_index().rename(columns={"index": "gene"})
    out.attrs.update(bg_match=bg_match, bg_ref_tier=bg_ref_tier or max(tiers),
                     top_n=top_n, tiers=tuple(tiers))
    return out


def _panel_matched_bg(pool_expr, ref_expr, k=10, seed=0):
    """1:1 nearest-neighbour match of `pool` to `ref` on log expression.

    Returns positional indices into pool, EXACTLY len(ref) of them (given enough
    pool), each used once. Nearest-neighbour rather than the quantile bins
    characterisation_table uses: on these datasets the bins leave a mean |SMD| of
    0.14 in log expression and up to 0.63, k-NN leaves 0.03. The pool is never the
    binding constraint here -- every expression decile has 9-57x more never-called
    genes than a tier needs -- so the bins were simply too coarse.
    """
    from scipy.spatial import cKDTree
    pool_expr = np.asarray(pool_expr, float)
    ref_expr = np.asarray(ref_expr, float)
    if len(pool_expr) < len(ref_expr) or not len(ref_expr):
        return None
    rng = np.random.default_rng(seed)
    tree = cKDTree(pool_expr.reshape(-1, 1))
    kq = min(k, len(pool_expr))
    nn = np.atleast_2d(tree.query(ref_expr.reshape(-1, 1), k=kq)[1])
    # hardest-to-match first, so they are not left with the leftovers
    iso = np.abs(pool_expr[nn[:, -1]] - ref_expr)
    used, out = set(), []
    for i in np.argsort(-iso):
        cand = [c for c in nn[i] if c not in used]
        kk = kq
        while not cand and kk < len(pool_expr):
            kk = min(kk * 5, len(pool_expr))
            cand = [c for c in np.atleast_1d(
                tree.query(ref_expr[i].reshape(1, 1), k=kk)[1]).ravel() if c not in used]
        if not cand:
            cand = [c for c in range(len(pool_expr)) if c not in used]
        p = int(rng.choice(cand))
        used.add(p)
        out.append(p)
    return np.array(out, int)


def never_called(universe, depth, timepoint="merge", merge_mode=MERGE_MODE,
                 memgene_dir=None, method_files=None):
    """THE background: genes in `universe` that no method places in its top `depth`.
    Pass the panel's OWN memory-gene cutoff as depth (the one rule for every figure);
    depth=None means each method's full ranked list."""
    _perturb_path()
    from marson_KO import build_consensus_gene_sets
    files = method_files or METHOD_FILES
    key = ("never_called", depth, timepoint, merge_mode, memgene_dir or MEMGENE_DIR,
           tuple(sorted(files.items())), hash(tuple(universe)))
    if key in _CACHE:
        return list(_CACHE[key])
    _, cnt = build_consensus_gene_sets(
        min_methods=1, timepoint=timepoint, top_n=depth, merge_mode=merge_mode,
        memgene_dir=memgene_dir or MEMGENE_DIR, method_files=files,
        universe=list(universe), verbose=False)
    out = cnt.index[cnt == 0].tolist()
    _CACHE[key] = out
    return list(out)


def _with_background(gs, ge, mem, bg):
    """gene summary + expression restricted to memory genes and the background, so a
    matched draw can ONLY come from the background."""
    keep = list(dict.fromkeys([g for g in mem if g in gs.index] +
                              [g for g in bg if g in gs.index]))
    return gs.loc[keep], (ge.reindex(keep) if ge is not None else None)


def responsiveness_results(top_n=TOP_N, timepoint="merge", methods=None,
                           metric=METRIC, n_bins=50, n_iter=100, floor=EXPR_FLOOR,
                           merge_mode=MERGE_MODE, floor_when="universe",
                           matcher="bins", k=10, seed=0, background="strict",
                           report_balance=True, verbose=True):
    """{method: results dict} for memory vs expression-matched non-memory.

    floor_when : where the expression floor is applied relative to the top-N cut.
        'universe' (default) floor first, then take the top N of what is left. Every
            method contributes exactly N genes, but they are the top N AMONG
            EXPRESSED genes — reaching rank 233-337 in the original ranking, and
            to different depths per method, so delta is not comparable ACROSS
            methods.
        'after'    take the true top N, then keep those above the floor. The rank
            cut is identical for every method (a real top N), at the cost of an
            unequal and much smaller n (26-51 of 100 in Marson) because most memory
            genes are below 0.1 CPM here.
        None / 'none'  no floor at all.

    matcher : 'bins' (default) stratifies the expressed universe into n_bins
        expression bins and averages n_iter sorted draws. 'knn' uses the SAME
        k-nearest-neighbour matcher as matching_rung_results(), so the per-method
        panel and the covariate rungs become one comparison computed one way —
        with matching arguments, the expression rung then reproduces this panel's
        value for that method exactly.

    Used by fig_responsiveness_violin (one split violin per method).
    """
    _perturb_path()
    from marson_KO import compare_gene_responsiveness_expr_matched
    after = (floor_when == "after" and floor is not None)
    # rank_gs: the universe the top-N is taken FROM.  cmp_gs: the universe the
    # comparison runs IN.  They differ only for floor_when='after'.
    rank_gs, _ = load_marson(None if after else floor, verbose)
    cmp_gs, cmp_ge = load_marson(floor, verbose=False) if after else (rank_gs, None)
    if not after:
        cmp_gs, cmp_ge = load_marson(floor if floor_when == "universe" else None,
                                     verbose=False)
    gs, ge = cmp_gs, cmp_ge
    # ONE background definition for every panel (2026-09-15): genes called by NONE of
    # the five methods within their top N. It used to be built only for matcher="knn";
    # the default "bins" matcher compared against every gene outside THIS method's set,
    # which can include genes the other four methods call.
    strict = never_called(gs.index, top_n, timepoint=timepoint, merge_mode=merge_mode)
    if matcher == "knn":
        from matching import matched_comparison
        cov = load_marson_covariates(gs, ge, verbose)
    res = {}
    for m in (methods or METHOD_FILES):
        genes = memory_set(m, timepoint, top_n, restrict_to=rank_gs.index,
                           merge_mode=merge_mode)
        if after:
            n_before = len(genes)
            genes = [g for g in genes if g in cmp_gs.index]
            if verbose:
                print(f"  {m:11s} true top-{top_n}: {n_before} -> {len(genes)} "
                      f"above {floor} CPM")
        if matcher == "knn":
            genes = [g for g in genes if g not in set(strict)]
            pool = (strict if background == "strict"
                    else [g for g in gs.index if g not in set(genes)])
            r = matched_comparison(gs, cov, genes, pool, match_on=["expr"],
                                   metric=metric, n_iter=n_iter, k=k,
                                   random_seed=seed)
            v = np.asarray(r["non_scg_values"], float)
            n = int(r["n_scg"])
            if n and v.size % n == 0:        # one equal-size control, as for bins
                r["non_scg_values"] = np.sort(v.reshape(-1, n), axis=1).mean(0)
        else:
            # restrict the frame to memory genes + the background pool, so the matched
            # draws can only come from genes no method called
            keep = (list(dict.fromkeys([g for g in genes if g in gs.index] + strict))
                    if background == "strict" else list(gs.index))
            fgs, fge = gs.loc[keep], ge.reindex(keep)
            r = compare_gene_responsiveness_expr_matched(
                fgs, fge, scg_genes=genes, metric=metric, n_bins=n_bins, n_iter=n_iter)
        if report_balance:
            bal_gs, bal_ge = (fgs, fge) if matcher != "knn" else (gs, ge)
            r.update(expression_balance(bal_gs, bal_ge, genes, n_bins=n_bins,
                                        n_iter=n_iter, metric=metric))
        res[m] = r
        if verbose:
            b = r.get("smd_expr", np.nan)
            print(f"  {m:11s} n={r['n_scg']:4d}  p={r['mann_whitney_p']:.2e}"
                  + ("" if not np.isfinite(b) else f"  SMD(expr)={b:+.3f}"
                     + ("  <-- NOT balanced" if abs(b) > .10 else "")))
    return res


def _fmt_pct(v):
    """'+12%' / '-4%' / '0%'. Rounds FIRST, so -0.3 prints as '0%', never '-0%' or '+0%'."""
    if v is None or not np.isfinite(v):
        return ""
    r = int(round(float(v)))
    return "0%" if r == 0 else f"{r:+d}%"


def fig_responsiveness_violin(res=None, top_n=TOP_N, timepoint="merge", methods=None,
                              metric=METRIC, n_bins=50, n_iter=100, floor=EXPR_FLOOR,
                              merge_mode=MERGE_MODE,
                              mem_label="Memory", bg_label="Expression-matched",
                              mem_color=C_MEM, bg_color=C_BG, mem_colors=None,
                              inner="quart",
                              fill=True, outline_lw=None,
                              median_inside=False, median_line_lw=None,
                              median_weight="bold",
                              median_pad=3, pct_where="above", pct_dx=6,
                              pct_y=None, pct_margin=.45,
                              cut=0, bw_adjust=.9, width=.9, linewidth=.8,
                              log_y=False, ylim=None, ylim_pct=99.0,
                              annot="delta+stars", annot_ci=False, fdr=True, q=None,
                              annot_fs=None, show_medians=True,
                              median_fmt="{:.2f}", median_fs=None,
                              fs=12, figsize=None, ylabel=None, xlabel=None,
                              title=None,
                              legend_loc=None, legend_out=True, ax=None,
                              legend=True, xtick_rotation=0, save=None, dpi=400,
                              show=True, verbose=True):
    """(a) One SPLIT violin per method: left half = memory genes, right half = the
    equal-size expression-matched control.

    A split violin puts the two distributions on a shared baseline, so the shape
    difference is read directly rather than across two boxes. The halves have the
    same n by construction, so equal violin areas are meaningful.

    inner : 'quart' (quartile lines), 'box', 'stick', or None.
    show_medians : print each half's median beside its violin, coloured to match.
    mem_colors : {violin label: colour} -- give each violin's MEMORY half its own
               colour (e.g. the dataset colours of the sunbursts); labels missing from
               the dict fall back to mem_color. The background half keeps bg_color.
    xtick_rotation : degrees to tilt the x tick labels (e.g. 45 for long rung names).
               Labels are right-aligned and anchored at the tick, so the END of each
               label sits under its violin rather than its middle.
    legend_out : place the legend outside the axes (default). Inside it lands on
            the tallest violins.
    log_y : the metric is a positive, right-skewed |LFC|; log10 spreads the bulk
            out, at the cost of dropping exact zeros (reported when it happens).
    Returns (fig, ax, stats).
    """
    import seaborn as sns
    from scipy.stats import mannwhitneyu
    _perturb_path()
    from marson_KO import cliffs_delta, cliffs_ci

    if res is None:
        res = responsiveness_results(top_n, timepoint, methods, metric, n_bins,
                                     n_iter, floor, merge_mode, verbose=verbose)
    order = list(res)
    rows, stats = [], []
    for m in order:
        a = np.asarray(res[m]["scg_values"], float)
        b = np.asarray(res[m]["non_scg_values"], float)
        rows.append(pd.DataFrame({"method": m, "group": mem_label, "value": a}))
        rows.append(pd.DataFrame({"method": m, "group": bg_label, "value": b}))
        stats.append(dict(method=m, n_mem=len(a), n_bg=len(b),
                          median_mem=np.median(a), median_bg=np.median(b),
                          ratio=np.median(a) / np.median(b) if np.median(b) else np.nan,
                          pct=(np.median(a) / np.median(b) - 1) * 100
                              if np.median(b) else np.nan,
                          cliffs_d=cliffs_delta(a, b),
                          p=mannwhitneyu(a, b).pvalue,
                          smd_expr=res[m].get("smd_expr", np.nan)))
        if annot_ci:
            lo, hi = cliffs_ci(a, b, n_boot=200)
            stats[-1].update(d_lo=lo, d_hi=hi)
    D = pd.concat(rows, ignore_index=True)
    S = pd.DataFrame(stats)
    S["q"] = bh_fdr(S["p"].values) if fdr else S["p"]
    if q is not None:
        # precomputed q-values keyed by gene-set label: lets a caller split one BH family
        # across several axes without silently shrinking the family to each axis
        S["q"] = S["method"].map(q).values

    if log_y:
        n0 = int((D["value"] <= 0).sum())
        if n0 and verbose:
            print(f"  log_y: dropping {n0} non-positive values "
                  f"({100*n0/len(D):.1f}% of points)")
        D = D[D["value"] > 0].copy()
        D["value"] = np.log10(D["value"])

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fs + 1, "xtick.labelsize": fs, "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
    with plt.rc_context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (1.35 * len(order) + 2.2, 4.2))
            own_fig = True
        else:
            fig, own_fig = ax.get_figure(), False
        sns.violinplot(data=D, x="method", y="value", hue="group",
                       hue_order=[mem_label, bg_label], order=order, split=True,
                       gap=.04, inner=None if median_inside else inner,
                       cut=cut, bw_adjust=bw_adjust,
                       width=width, linewidth=linewidth, density_norm="width",
                       palette={mem_label: mem_color, bg_label: bg_color},
                       saturation=1, ax=ax)

        # Unfilled violins: the shape is carried by its outline, so the median line
        # and its number can sit INSIDE without fighting a solid fill.
        from matplotlib.collections import PolyCollection
        bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
        poly, body_key = {}, {}
        for c in bodies:
            v = c.get_paths()[0].vertices
            if len(v) < 3:
                continue
            i_ = int(np.clip(round(np.mean([v[:, 0].min(), v[:, 0].max()])),
                             0, len(S) - 1))
            key = (i_, "mem" if v[:, 0].max() <= i_ + 1e-6 else "bg")
            poly[key] = v
            body_key[id(c)] = key

        def _mem_col(i):
            return (mem_colors or {}).get(order[i], mem_color)

        # per-violin memory colour, applied before the outline step copies face -> edge
        if mem_colors:
            for c in bodies:
                key = body_key.get(id(c))
                if key and key[1] == "mem":
                    c.set_facecolor(_mem_col(key[0]))

        def _x_at(v, y, side):
            """Where the drawn outline sits at height y — 'min' or 'max' edge."""
            tol = np.ptp(v[:, 1]) * .02 + 1e-12
            xs = v[np.abs(v[:, 1] - y) < tol, 0]
            if xs.size == 0:
                return None
            return float(xs.min()) if side == "min" else float(xs.max())

        if not fill:
            for c in bodies:
                fc = c.get_facecolor()
                edge = fc[0] if len(fc) else (0, 0, 0, 1)
                c.set_facecolor("none")
                c.set_edgecolor(edge)
                c.set_linewidth(outline_lw or linewidth * 2.0)
        if ylim is None and not log_y:
            ylim = (0, np.percentile(D["value"], ylim_pct))
        if ylim:
            ax.set_ylim(*ylim)
        # median line + its value, drawn at the median INSIDE each half violin.
        # The x extent is read off the drawn polygon, so it tracks the actual shape
        # rather than assuming a nominal half-width.
        if median_inside and poly:
            mcolor = {mem_label: mem_color, bg_label: bg_color}
            for (i, half), verts in poly.items():
                r = S.iloc[i]
                left = half == "mem"
                med = r.median_mem if left else r.median_bg
                if log_y:
                    # the stats (and so the medians) are on the RAW scale, the axis is
                    # log10 -- without this the median line lands at the raw value
                    if not (np.isfinite(med) and med > 0):
                        continue
                    med = float(np.log10(med))
                lab = mem_label if left else bg_label
                x0 = _x_at(verts, med, "min")
                x1 = _x_at(verts, med, "max")
                if x0 is None or x1 is None or x1 - x0 <= 0:
                    continue
                lc = _mem_col(i) if left else mcolor[lab]
                ax.plot([x0, x1], [med, med], color=lc,
                        lw=median_line_lw or linewidth * 2.2,
                        solid_capstyle="butt", zorder=5, clip_on=True)
                ax.annotate(median_fmt.format(med), ((x0 + x1) / 2, med),
                            xytext=(0, median_pad), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=median_fs or fsize(fs - 2),
                            color=lc, fontweight=median_weight, zorder=6)

        # `or` would swallow an explicit "" and restore the default — that is how a
        # grid ends up with the same axis label repeated under every panel
        ax.set_xlabel("" if xlabel is None else xlabel)
        ax.set_ylabel((("log₁₀ " if log_y else "") + metric_label(metric))
                      if ylabel is None else ylabel)
        # Header block above each violin pair, bottom -> top:
        #   stars, "median_mem, median_bg", [delta], [pct]
        # The two medians are drawn as separate texts so each keeps its group's
        # colour while reading as one comma-separated line.
        afs_ = annot_fs or fs - 1
        # the median pair is the widest header row ("0.20, 0.13"); at fs+1 it runs
        # into the neighbouring method's pair at five violins
        mfs = median_fs or fs - 2
        line_h = max(afs_, mfs) * 1.28
        layers = []
        if annot and "stars" in annot:
            layers.append("stars")
        if show_medians:
            layers.append("medians")
        if annot and "delta" in annot:
            layers.append("delta")
        if annot and "pct" in annot and pct_where == "above":
            layers.append("pct")
        for k, kind in enumerate(layers):
            dy = 4 + k * line_h
            for i, r in enumerate(S.itertuples()):
                if kind == "medians":
                    ax.annotate(median_fmt.format(r.median_mem) + ",", (i, 1.0),
                                xycoords=("data", "axes fraction"),
                                xytext=(-3, dy), textcoords="offset points",
                                ha="right", va="bottom", fontsize=mfs,
                                color=mem_color, fontweight="bold", clip_on=False)
                    ax.annotate(median_fmt.format(r.median_bg), (i, 1.0),
                                xycoords=("data", "axes fraction"),
                                xytext=(3, dy), textcoords="offset points",
                                ha="left", va="bottom", fontsize=mfs,
                                color=bg_color, fontweight="bold", clip_on=False)
                    continue
                if kind == "stars":
                    txt, col, size = _stars(r.q), "0.25", afs_
                elif kind == "delta":
                    txt = f"δ {r.cliffs_d:+.2f}"
                    if annot_ci:
                        txt += f" [{r.d_lo:+.2f}, {r.d_hi:+.2f}]"
                    col, size = "0.25", afs_
                else:                                    # pct
                    txt, col, size = _fmt_pct(r.pct), "0.15", afs_ + 1
                ax.annotate(txt, (i, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, dy), textcoords="offset points",
                            ha="center", va="bottom", fontsize=size, color=col,
                            clip_on=False)
        # percentages beside each pair rather than stacked in the header
        if annot and "pct" in annot and pct_where == "left":
            ax.set_xlim(-.5 - pct_margin, len(S) - .5)
            y0, y1 = ax.get_ylim()
            yy = (float(pct_y) if isinstance(pct_y, (int, float))
                  else y0 + .55 * (y1 - y0))
            for i, r in enumerate(S.itertuples()):
                v = poly.get((i, "mem"))
                # hug the outline where the violin is narrow, so the label sits in
                # the empty space beside the neck instead of over the shape
                xl = _x_at(v, yy, "min") if v is not None else None
                ax.annotate(_fmt_pct(r.pct),
                            (i - .02 if xl is None else xl, yy),
                            xytext=(-pct_dx, 0), textcoords="offset points",
                            ha="right", va="center",
                            fontsize=(annot_fs or fs - 1) + 1, color="0.15",
                            clip_on=False)

        if not legend:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        elif legend_out:
            ax.legend(title=None, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      frameon=False, borderaxespad=0)
        else:
            ax.legend(title=None, loc=legend_loc or "upper right", frameon=False)
        if title:
            ax.set_title(title, fontsize=fs + 1, pad=fs + 10)
        ax.spines[["top", "right"]].set_visible(False)
        if xtick_rotation:
            plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right",
                     rotation_mode="anchor")
        if own_fig:
            fig.tight_layout()
            tn = "all" if top_n is None else top_n
            save_fig(fig, save, f"fig2a_responsiveness_violin_top{tn}_{timepoint}",
                     dpi=dpi, verbose=verbose)
            if show:
                plt.show()
    if verbose:
        print(S.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return fig, ax, S


# ═══════════════════════════════════════════════════════════════════════════
#  Donor-level replication: delta computed WITHIN each donor, then aggregated
#  with the donor as the unit of replication.
# ═══════════════════════════════════════════════════════════════════════════

MARSON_DONORS = ["D1", "D2", "D3", "D4"]     # the values in the data, do not rename


def donor_label(g):
    """Display name for a donor key: 'D3' -> 'Donor 3' (house rule 5)."""
    m = re.fullmatch(r"[Dd](\d+)", str(g))
    return f"Donor {m.group(1)}" if m else str(g)
# Purple ramp (ColorBrewer 4-class Purples). Deliberately NOT red/blue: those two
# already mean "memory gene" and "expression-matched control" in the violin panel,
# and reusing them for donors would make one colour carry two meanings.
# Qualitative and distinct, with NO red and NO blue: those two already mean
# "memory gene" and "expression-matched control" in the violin panel.
DONOR_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]   # teal, orange, purple, magenta
MARSON_TIMEPOINTS = ["Rest", "Stim8hr", "Stim48hr"]


def marson_group(group):
    """Per-group (donor, timepoint, or 'D1_Rest') gene summary + control expression.

    These were rebuilt per group by perturb-seq/per_group_KO.py; the pooled
    marson_lfc_cpm.csv mixes 4 donors x 3 timepoints, so a pooled statistic cannot
    say whether an effect holds within a donor.
    """
    key = ("marson_group", group)
    if key in _CACHE:
        return _CACHE[key]
    D = f"{P.DATA}/marson"
    gs = pd.read_csv(f"{D}/marson_gene_summary_cpm_{group}.csv", index_col=0)
    ge = pd.read_csv(f"{D}/marson_gene_ctrl_expr_{group}.csv", index_col=0).iloc[:, 0]
    _CACHE[key] = (gs, ge)
    return gs, ge


def donor_responsiveness(groups=None, methods=None, top_n=TOP_N, timepoint="merge",
                         metric=METRIC, n_bins=50, n_iter=100, floor=EXPR_FLOOR,
                         merge_mode=MERGE_MODE, verbose=True):
    """Memory vs expression-matched control, computed SEPARATELY inside each donor.

    Returns a long table: one row per (group, method) with delta, % higher median
    and the gene-level p. Each row is a complete, self-contained comparison — no
    donor is pooled with another at any point.
    """
    _perturb_path()
    from marson_KO import compare_gene_responsiveness_expr_matched, cliffs_delta
    groups = list(groups or MARSON_DONORS)
    rows = []
    for grp in groups:
        gs, ge = marson_group(grp)
        if floor is not None:
            keep = ge.index[ge > floor]
            gs = gs.loc[gs.index.intersection(keep)]
            ge = ge.loc[gs.index]
        bg = never_called(gs.index, top_n, timepoint=timepoint, merge_mode=merge_mode)
        for m in (methods or METHOD_FILES):
            genes = memory_set(m, timepoint, top_n, restrict_to=gs.index,
                               merge_mode=merge_mode)
            fgs, fge = _with_background(gs, ge, genes, bg)
            r = compare_gene_responsiveness_expr_matched(
                fgs, fge, genes, metric=metric, n_bins=n_bins, n_iter=n_iter)
            a = np.asarray(r["scg_values"]); b = np.asarray(r["non_scg_values"])
            rows.append(dict(group=grp, method=m, n=len(a), n_universe=len(gs),
                             median_mem=np.median(a), median_bg=np.median(b),
                             pct=(np.median(a) / np.median(b) - 1) * 100,
                             cliffs_d=cliffs_delta(a, b), p=r["mann_whitney_p"]))
        if verbose:
            print(f"  {grp}: {len(gs)} genes above the floor", flush=True)
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].values)
    return out


def fig_donor_replication(table=None, groups=None, methods=None, value="median",
                          top_n=TOP_N, timepoint="merge", metric=METRIC,
                          n_bins=50, n_iter=100, floor=EXPR_FLOOR,
                          donor_palette=None, palette_range=(0., .80),
                          pair_lw=.9, pair_alpha=.55,
                          mem_color=C_MEM, bg_color=C_BG, offset=.17,
                          dot_color="0.55", dot_size=26, jitter=.13,
                          mean_color="0.25", mean_size=8, ci=.95,
                          ci_color="0.45", legend_loc="upper left", legend_ncol=1,
                          legend_frame=True, legend_edge="0.8", legend_lw=.8,
                          legend_fs=None,
                          legend_bbox=(.05, 1.0), annot_weight="normal",
                          annot=True, annot_fmt=None, order_by="canonical",
                          group_labels=None,
                          show_mean=True, orient="h", tick_rotation=None,
                          ref=None, ref_label="no difference",
                          fs=12, figsize=None, xlabel=None, title=None,
                          note=None, save=None, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 3c) Donor as the unit of replication, drawn as a standard forest plot.

    One line per method: the four donor estimates (grey dots) and the across-donor
    mean with a t-based CI. The interval reflects BIOLOGICAL variability between
    donors, not the number of genes tested — which is what a gene-level p of 1e-12
    cannot tell you.

    value : 'median' (default) the RAW medians on the same mean |LFC| axis as the
                violin panel: per method, each donor contributes a memory point and
                a control point joined by a line, so both the absolute level and
                the within-donor pairing are visible.
            'win'      P(memory > matched control) = (delta+1)/2, 0.5 = no effect
            'cliffs_d' | 'pct'
    Returns (fig, ax, per-donor table, per-method summary).
    """
    from scipy import stats as sps
    if table is None:
        table = donor_responsiveness(groups, methods, top_n, timepoint, metric,
                                     n_bins, n_iter, floor, verbose=verbose)
    table = table.copy()
    table["win"] = (table["cliffs_d"] + 1) / 2
    grps = list(dict.fromkeys(table["group"]))
    if donor_palette is None:
        dcol = {g: DONOR_COLORS[i % len(DONOR_COLORS)] for i, g in enumerate(grps)}
    elif isinstance(donor_palette, dict):
        dcol = donor_palette
    elif isinstance(donor_palette, (list, tuple)):
        dcol = {g: donor_palette[i % len(donor_palette)] for i, g in enumerate(grps)}
    else:
        cmap = plt.get_cmap(donor_palette)
        lo, hi = palette_range
        dcol = {g: cmap(i % cmap.N) if cmap.N <= 20
                else cmap(lo + (hi - lo) * i / max(len(grps) - 1, 1))
                for i, g in enumerate(grps)}

    if value == "median":
        from scipy import stats as sps
        S = (table.groupby("method", sort=False)
                  .agg(n_groups=("group", "size"),
                       mem=("median_mem", "mean"), bg=("median_bg", "mean"))
                  .reset_index())
        S["mean"] = S["mem"]
        if order_by == "mean":
            S = S.sort_values("mem").reset_index(drop=True)
        style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                 "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fs,
                 "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=figsize or (6.4, .70 * len(S) + 2.0))
            for y, r in enumerate(S.itertuples()):
                sub = table[table.method == r.method]
                for g in grps:                       # one line per donor: the pair
                    d = sub[sub.group == g]
                    if not len(d):
                        continue
                    mm, bb = d["median_mem"].iloc[0], d["median_bg"].iloc[0]
                    ax.plot([bb, mm], [y - offset, y + offset], color=ci_color,
                            lw=pair_lw, alpha=pair_alpha, zorder=2)
                    ax.scatter([mm], [y + offset], s=dot_size + 12, color=dcol[g],
                               zorder=3, edgecolor="white", linewidth=.6,
                               label=g if y == 0 else None)
                    ax.scatter([bb], [y - offset], s=dot_size + 12, color=dcol[g],
                               zorder=3, edgecolor="white", linewidth=.6,
                               marker="s")
                if show_mean:
                    ax.plot([r.mem], [y + offset], "o", ms=mean_size + 1,
                            color=mem_color, markeredgecolor="white",
                            markeredgewidth=1.1, zorder=4)
                    ax.plot([r.bg], [y - offset], "s", ms=mean_size,
                            color=bg_color, markeredgecolor="white",
                            markeredgewidth=1.1, zorder=4)
                if annot:
                    ax.annotate(f"{(r.mem / r.bg - 1) * 100:+.0f}%",
                                (max(r.mem, r.bg), y), xytext=(9, 0),
                                textcoords="offset points", va="center", ha="left",
                                fontsize=fs - 1, color="0.2",
                                fontweight=annot_weight)
            ax.set_yticks(range(len(S)))
            ax.set_yticklabels(S["method"])
            ax.set_ylim(-.75, len(S) - .25)
            ax.set_xlabel(xlabel or "median mean |LFC|  (CPM)")
            xr = ax.get_xlim()
            ax.set_xlim(xr[0], xr[1] + .16 * (xr[1] - xr[0]))
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            from matplotlib.lines import Line2D
            h = [Line2D([], [], marker="o", ls="", color=mem_color, ms=7,
                        label="Memory"),
                 Line2D([], [], marker="s", ls="", color=bg_color, ms=6,
                        label="Expression-matched")]
            h += [Line2D([], [], marker="o", ls="", color=dcol[g], ms=6, label=g)
                  for g in grps]
            leg = ax.legend(handles=h, loc=legend_loc, frameon=legend_frame,
                            ncol=legend_ncol, handletextpad=.4, labelspacing=.35,
                            fontsize=fs - 2, borderaxespad=.4)
            if legend_frame and leg is not None:
                leg.get_frame().set_edgecolor(legend_edge)
                leg.get_frame().set_linewidth(legend_lw)
                leg.get_frame().set_facecolor("white")
            if note is None:
                note = ("circle = memory genes, square = matched control; "
                        f"one pair per donor (n = {len(grps)}); "
                        "large markers = mean across donors")
            if note:
                fig.text(.5, -.005, note, ha="center", va="top", fontsize=fs - 2,
                         color="0.35")
            if title:
                ax.set_title(title, fontsize=fs + 1, pad=8)
            fig.tight_layout()
            save_fig(fig, save, "fig2_donor_replication_median", dpi=dpi,
                     verbose=verbose)
            if show:
                plt.show()
        if verbose:
            print(S.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        return fig, ax, table, S
    ref = ({"win": .5, "cliffs_d": 0., "pct": 0.}[value] if ref is None else ref)
    annot_fmt = annot_fmt or ({"win": "{:.2f}", "cliffs_d": "{:+.2f}",
                               "pct": "{:+.0f}%"}[value])
    rows = []
    for m, d in table.groupby("method", sort=False):
        v = d[value].to_numpy(float)
        n = len(v)
        mean, sd = v.mean(), (v.std(ddof=1) if n > 1 else np.nan)
        half = (sps.t.ppf(.5 + ci / 2, n - 1) * sd / np.sqrt(n)
                if n > 1 and np.isfinite(sd) else np.nan)
        t, p = (sps.ttest_1samp(v, ref) if n > 1 else (np.nan, np.nan))
        rows.append(dict(method=m, n_groups=n, mean=mean, sd=sd, lo=mean - half,
                         hi=mean + half, t=t, p_donor=p, min=v.min(), max=v.max()))
    S = pd.DataFrame(rows)
    if order_by == "mean":
        S = S.sort_values("mean").reset_index(drop=True)
    elif order_by in ("canonical", "house", "method"):
        # house rule 1: one method order, every figure. Reversed because row 0 is
        # drawn at the BOTTOM, so this reads CORAL -> SCG top to bottom.
        S = (S.set_index("method")
               .loc[in_order(S["method"], METHOD_ORDER)[::-1]]
               .reset_index())

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fs,
             "axes.linewidth": .9}
    rng = np.random.default_rng(0)
    with plt.rc_context(style):
        vert = orient in ("v", "vertical")
        # vertical = methods along x, effect on y (house rule 2: the same quantity
        # is on y in 2a and 2c, so it belongs on y here too)
        if vert:
            order = S["method"].tolist()[::-1]        # row 0 draws at the bottom
            S = S.set_index("method").loc[order].reset_index()
        fig, ax = plt.subplots(figsize=figsize or (
            (.9 * len(S) + 1.8, 4.6) if vert else (5.8, .58 * len(S) + 1.8)))
        (ax.axhline if vert else ax.axvline)(ref, color="0.65", lw=1.0,
                                             ls=(0, (4, 3)), zorder=1)
        for y, r in enumerate(S.itertuples()):
            # donor points and the summary share ONE line, so each method reads as
            # a single row rather than two stacked marks
            sub = table[table.method == r.method]
            for k, g in enumerate(grps):
                vv = sub.loc[sub.group == g, value]
                if len(vv):
                    off = (y + (k - (len(grps) - 1) / 2)
                           * (2 * jitter / max(len(grps) - 1, 1)))
                    ax.scatter(*((off, vv.iloc[0]) if vert else (vv.iloc[0], off)),
                               s=dot_size + 10, color=dcol[g], zorder=2,
                               edgecolor="white", linewidth=.5,
                               label=((group_labels or {}).get(g, donor_label(g))
                                      if y == 0 else None))
            ax.plot(*(([y, y], [r.lo, r.hi]) if vert else ([r.lo, r.hi], [y, y])),
                    color=ci_color, lw=2.0, solid_capstyle="butt", zorder=3)
            if show_mean:
                ax.plot(*(([y], [r.mean]) if vert else ([r.mean], [y])), "o",
                        ms=mean_size, color=mean_color,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4)
            if annot:
                ax.annotate(annot_fmt.format(r.mean),
                            (y, r.hi) if vert else (r.hi, y),
                            xytext=(0, 7) if vert else (7, 0),
                            textcoords="offset points",
                            va="bottom" if vert else "center",
                            ha="center" if vert else "left",
                            fontsize=fs - 1, color=mean_color,
                            fontweight=annot_weight)
        vlab = xlabel or {
            "win": "P(memory gene more responsive than matched control)",
            "cliffs_d": "Cliff's δ vs expression-matched control",
            "pct": "% higher median responsiveness"}[value]
        if vert:
            ax.set_xticks(range(len(S)))
            ax.set_xticklabels(S["method"],
                               rotation=30 if tick_rotation is None else tick_rotation,
                               ha="right", rotation_mode="anchor")
            ax.set_xlim(-.65, len(S) - .35)
            ax.set_ylabel(vlab)
            yr = ax.get_ylim()
            ax.set_ylim(yr[0], yr[1] + .16 * (yr[1] - yr[0]))
            if ref_label:
                ax.annotate(ref_label, (len(S) - .45, ref), xytext=(0, 4),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=fs - 2, color="0.45", style="italic")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_yticks(range(len(S)))
            ax.set_yticklabels(S["method"],
                               rotation=0 if tick_rotation is None else tick_rotation)
            ax.set_ylim(-.65, len(S) - .35)
            ax.set_xlabel(vlab)
            xr = ax.get_xlim()
            ax.set_xlim(xr[0], xr[1] + .16 * (xr[1] - xr[0]))
            if ref_label:
                ax.annotate(ref_label, (ref, len(S) - .45), xytext=(4, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=fs - 2, color="0.45", style="italic")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        # nudged off the axes' left edge: the reference line sits at x=0, i.e. at
        # the very left, and a plain 'upper left' legend prints on top of it
        leg = ax.legend(loc=legend_loc, frameon=legend_frame, ncol=legend_ncol,
                        bbox_to_anchor=legend_bbox, handletextpad=.4,
                        labelspacing=.35, fontsize=fsize(legend_fs or fs - 2),
                        borderaxespad=.4)
        if legend_frame and leg is not None:
            leg.get_frame().set_edgecolor(legend_edge)
            leg.get_frame().set_linewidth(legend_lw)
            leg.get_frame().set_facecolor("white")
        if note is None:
            note = (f"each point = one donor (n = {S['n_groups'].iloc[0]}); "
                    f"bar = mean ± {int(ci*100)}% CI across donors")
        if note:
            fig.text(.5, -.005, note, ha="center", va="top", fontsize=fs - 2,
                     color="0.35")
        if title:
            ax.set_title(title, fontsize=fs + 1, pad=8)
        fig.tight_layout()
        save_fig(fig, save, f"fig2_donor_replication_{value}", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    if verbose:
        print(S.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return fig, ax, table, S


def matching_rung_results(memset="ClonePair", min_methods=None,
                          timepoint="T1", top_n=TOP_N, panels=None,
                          metric=METRIC, merge_mode=MERGE_MODE, method_files=None,
                          memgene_dir=None, n_iter=20, k=10, seed=0,
                          floor=EXPR_FLOOR, background="strict", equalise=False,
                          expr_bins=False, n_bins=50, expr_n_iter=100,
                          verbose=True):
    """{rung label: matched_comparison(...)} — the data behind the matching-rung panels.

    Returned dicts carry 'scg_values'/'non_scg_values', so they can be handed
    straight to fig_responsiveness_violin() to draw the rungs as split violins
    in the same style as the per-method panel.

    min_methods : use the CONSENSUS of >= this many of the five methods instead of
        one method's set. Note the consensus shrinks fast: at top_n=100 all five
        agree on only 9 genes in the expressed universe.
    background : 'strict' (default) = genes called by 0 of the 5 methods, so a
        "non-memory" gene is never a memory gene that merely missed one cutoff.
        'non_memory' = every gene outside THIS method's own set, which is what
        responsiveness_results() matches against — use it when the rungs have to be
        directly comparable with the per-method panel.
    equalise : average the sorted draws into one equal-size control (n = n_memory)
        instead of pooling all n_iter draws. Again, matches the per-method panel.
    expr_bins : run any rung that matches on expression ALONE through the same
        quantile-bin matcher the per-method panel uses, so that rung reproduces
        that panel's bar exactly. Bins cannot do the multi-covariate rungs — k bins
        per covariate gives k**p cells, which empty out — so those stay k-NN.
    """
    _perturb_path()
    from marson_KO import build_consensus_gene_sets
    from matching import matched_comparison

    gs, ge = load_marson(floor, verbose)
    cov = load_marson_covariates(gs, ge, verbose)
    method_files = method_files or METHOD_FILES
    _, counts = build_consensus_gene_sets(
        min_methods=1, timepoint=timepoint, top_n=top_n, merge_mode=merge_mode,
        memgene_dir=memgene_dir or MEMGENE_DIR, method_files=method_files,
        universe=gs.index, verbose=False)
    strict = never_called(gs.index, top_n, timepoint=timepoint, merge_mode=merge_mode,
                          memgene_dir=memgene_dir, method_files=method_files)
    # memset=a method name, or min_methods=k for the consensus of >= k of the five
    mem = (counts.index[counts >= int(min_methods)].tolist() if min_methods
           else memory_set(memset, timepoint, top_n, restrict_to=gs.index,
                           merge_mode=merge_mode, memgene_dir=memgene_dir,
                           method_files=method_files))
    mem = [g for g in mem if g in gs.index and g not in set(strict)]
    # 'strict' excludes any gene another method called; 'non_memory' is every gene
    # outside THIS method's set, which is the pool the per-method panel matches in
    bg_pool = (strict if background == "strict"
               else [g for g in gs.index if g not in set(mem)])
    if verbose:
        print(f"{memset} @ {timepoint} top{top_n}: {len(mem)} memory genes "
              f"vs {len(bg_pool)} {background} background")
    mem_raw = list(mem) if min_methods else memory_set(
        memset, timepoint, top_n, restrict_to=gs.index, merge_mode=merge_mode,
        memgene_dir=memgene_dir, method_files=method_files)
    res = {}
    for lab, cols in (panels or MATCH_PANELS).items():
        if expr_bins and list(cols) == ["expr"]:
            from marson_KO import compare_gene_responsiveness_expr_matched
            # same background as the other rungs (bg_pool), not "every other gene"
            keep = list(dict.fromkeys([g for g in mem_raw if g in gs.index] + list(bg_pool)))
            res[lab] = compare_gene_responsiveness_expr_matched(
                gs.loc[keep], ge.reindex(keep), scg_genes=mem_raw, metric=metric,
                n_bins=n_bins, n_iter=expr_n_iter)
        else:
            res[lab] = matched_comparison(gs, cov, mem, bg_pool, match_on=cols,
                                          metric=metric, n_iter=n_iter, k=k,
                                          random_seed=seed)
    if equalise:
        # collapse the pooled draws into ONE equal-size control by averaging the
        # sorted draws — the aggregation the per-method panel uses, so the two
        # panels' control halves are the same kind of object
        for r in res.values():
            v = np.asarray(r["non_scg_values"], float)
            n = int(r["n_scg"])
            if n and v.size % n == 0:
                r["non_scg_values"] = np.sort(v.reshape(-1, n), axis=1).mean(0)
    if verbose:
        for lab, r in res.items():
            if "balance" not in r:
                print(f"  {lab:24s} (quantile bins) p={r['mann_whitney_p']:.2e}")
                continue
            print(f"  {lab:24s} d={r['cliffs_d']:+.3f} "
                  f"[{r['d_lo']:+.3f},{r['d_hi']:+.3f}] p={r['mann_whitney_p']:.2e} "
                  f"worstSMD={r['balance'].smd_after.abs().max():.3f}")
    return res


# ---------------------------------------------------------- length bias --

# Sequencing chemistry per lineage dataset. This is the axis the panel is FOR:
# in a 3'-biased protocol one UMI is one molecule whatever the transcript's
# length, while in a full-length protocol reads scale with length, so length buys
# a gene measurement precision it has not earned. Anything that ranks genes by how
# reproducible they are across sisters can pick that up as if it were biology.
DATASET_CHEMISTRY = {
    "T-lineage": "10x (3'-biased)", "Harmange": "10x (3'-biased)",
    "LARRY": "10x (3'-biased)", "Bandler": "10x (3'-biased)",
}


def fig_leaveout_grid(methods=None, lfc_df=None, top_n=TOP_N, timepoint="merge",
                      step=25, min_methods=None, floor=EXPR_FLOOR,
                      ncol=3, panel_w=3.5, panel_h=2.8,
                      x="n", xticks=None, yticks=None, ylim=None, fs=10,
                      xlabel=None, ylabel=None,
                      annot_ratio=True, annot_dy=(0, 12), sharey=True,
                      panel_letters=None, legend_panel=0, out=None,
                      save=None, dpi=400, show=True, verbose=True, **kw):
    """(Suppl. Fig 4d) One leave-out-KO curve per method, on a shared grid.

    The per-method version of fig_leaveout_curve, so the "not driven by a few
    perturbations" check can be shown for every caller rather than one. Pass a
    preloaded `lfc_df` — otherwise the 1.5 GB matrix is read once per method.

    xlabel / ylabel : axis text for the outer panels only — inner panels are always
        blank. None keeps fig_leaveout_curve's defaults.
    out : {method: DataFrame} from a previous run, to redraw without recomputing.
    Returns (fig, axes, {method: DataFrame}).
    """
    meths = in_order(methods or list(METHOD_FILES), METHOD_ORDER)
    if lfc_df is None:
        lfc_df = load_marson_lfc(verbose=verbose)
    data = dict(out or {})
    for m in meths:
        if m not in data:
            data[m] = leaveout_curve_data(memset=m, top_n=top_n, step=step,
                                          timepoint=timepoint, lfc_df=lfc_df,
                                          min_methods=min_methods or 5,
                                          floor=floor, verbose=verbose)
    nrow = int(np.ceil(len(meths) / ncol))
    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                                 squeeze=False, sharey=sharey)
        for i, m in enumerate(meths):
            ax = axes[i // ncol][i % ncol]
            fig_leaveout_curve(out=data[m], x=x, ax=ax, fs=fs, ylim=ylim,
                               xticks=xticks, yticks=yticks, annot_ratio=annot_ratio,
                               annot_dy=annot_dy, legend=(i == legend_panel),
                               xlabel="" if i // ncol < nrow - 1 else xlabel,
                               ylabel="" if i % ncol else ylabel,
                               title=m, save=None, show=False, verbose=False, **kw)
            ax.set_title(m, fontsize=fs + 1, pad=6)
        for j in range(len(meths), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        if panel_letters:
            for ax, l in zip(fig.axes, panel_letters_for(panel_letters, len(meths))):
                panel_letter(ax, l, fontsize=fs + 2)
        fig.tight_layout()
        save_fig(fig, save, f"leaveout_grid_top{top_n}_{x}", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, axes, data


def fig_cross_dataset_lines(S=None, datasets=None, tiers=(2, 3, 4), cons_top_n=500,
                            floor=EXPR_FLOOR,
                            metric="pct", n_bins=50, n_iter=100, timepoint="merge",
                            merge_mode=MERGE_MODE, bg_mode="top_n",
                            bg_top_n=None, methods=None,
                            colors=None, cmap="tab10", lw=2.0, ms=7,
                            show_n=True, n_fs=None, annot_last=True,
                            summary=None, summary_stat="median",
                            summary_color="#111111", summary_lw=3.6,
                            summary_ms=9, summary_label=None, line_alpha=1.0,
                            strip_mathtext=False,
                            ref=0., ref_label=None, fs=13, figsize=(6.4, 4.6),
                            xlabel="methods calling the gene",
                            ylabel=None, title=None, legend_loc="best",
                            legend_out=False, legend=True, save=None, dpi=400, show=True,
                            verbose=True):
    """(Suppl. Fig 4c) Consensus sweep as LINES: x = agreement tier, one line per dataset.

    The heatmap shows the same numbers, but a line makes the TREND across tiers
    readable per dataset — whether requiring more methods to agree buys a larger
    effect — which is the question the tier axis exists to answer.

    summary : draw one heavy line for the summary_stat ('median' or 'mean') across
        datasets. Pair it with line_alpha (~.35) so the per-dataset lines recede to
        context and the summary carries the message.
    show_n : print the gene count beside each point (the tiers are nested and
        shrink fast: ~600 genes at >=2 down to ~100 at >=4, so a rising line can
        be a real trend or just a smaller, more selective set).
    Returns (fig, ax, S).
    """
    if S is None:
        S = xd_consensus_sweep(datasets, tiers=tiers, top_n=cons_top_n, floor=floor,
                               n_bins=n_bins, n_iter=n_iter, timepoint=timepoint,
                               merge_mode=merge_mode, methods=methods,
                               bg_mode=bg_mode, bg_top_n=bg_top_n, verbose=verbose)
    col = {"pct": "pct", "cliffs_d": "cliffs_d",
           "winprob": "cliffs_d", "pctile": "cliffs_d"}[metric]
    d = S.copy()
    if metric == "winprob":
        d["val"] = (d["cliffs_d"] + 1) / 2
    elif metric == "pctile":
        d["val"] = (d["cliffs_d"] + 1) / 2 * 100
    else:
        d["val"] = d[col]
    # the sweep writes ASCII ">=k"; the unicode form is only for the tick labels
    order = [f">={k}" for k in tiers]
    d = d[d["method"].isin(order)]
    if d.empty:
        raise ValueError(f"no rows for tiers {order}; table has "
                         f"{sorted(set(S['method']))}")
    labs = list(dict.fromkeys(d["dataset"]))
    cm = plt.get_cmap(cmap)
    colors = colors or {l: cm(i % cm.N) for i, l in enumerate(labs)}
    x = np.arange(len(order))

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "xtick.labelsize": fs, "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 3), "axes.linewidth": .9}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=figsize)
        if ref is not None:
            ax.axhline(ref, color="0.7", lw=.9, ls=(0, (3, 2)), zorder=1)
        for lab in labs:
            sub = d[d.dataset == lab].set_index("method").reindex(order)
            # keep the mathtext: the dataset names carry \mathbf{} on the first
            # line, which is what makes the perturb-seq dataset bold in the heatmap
            lab_txt = (str(lab).replace("$\\mathbf{", "").replace("}$", "")
                               .replace("\\ ", " ") if strip_mathtext else str(lab))
            ax.plot(x, sub["val"], "-o", color=colors[lab], lw=lw, ms=ms,
                    alpha=line_alpha, zorder=3, label=lab_txt)
            if show_n:
                for xi, (v, nn) in enumerate(zip(sub["val"], sub["n"])):
                    if np.isfinite(v) and np.isfinite(nn):
                        ax.annotate(f"{nn:.0f}", (xi, v), xytext=(0, 7),
                                    textcoords="offset points", ha="center",
                                    va="bottom", fontsize=fsize(n_fs or fs - 5),
                                    color=colors[lab])
        # one heavy line across the datasets: the per-dataset lines are the
        # evidence, this is the summary a reader is meant to take away
        if summary:
            M = np.vstack([d[d.dataset == l].set_index("method")
                            .reindex(order)["val"].to_numpy(float) for l in labs])
            agg = (np.nanmedian if summary_stat == "median" else np.nanmean)(M, axis=0)
            ax.plot(x, agg, "-o", color=summary_color, lw=summary_lw,
                    ms=summary_ms, zorder=5, solid_capstyle="round",
                    markeredgecolor="white", markeredgewidth=1.0,
                    label=summary_label or f"{summary_stat} across datasets")

        ax.set_xticks(x)
        ax.set_xticklabels([o.replace(">=", "≥") for o in order])
        ax.set_xlim(-.3, len(order) - .7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or {"pct": "% higher median responsiveness",
                                 "cliffs_d": "Cliff's δ",
                                 "winprob": "P(memory > matched)",
                                 "pctile": "percentile"}[metric])
        ax.spines[["top", "right"]].set_visible(False)
        # legend=False: no per-panel legend, for a figure with one shared legend
        if legend and legend_out:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        elif legend:
            ax.legend(loc=legend_loc, frameon=False, labelspacing=.3)
        if title:
            ax.set_title(title, fontsize=fs + 1, pad=8)
        fig.tight_layout()
        save_fig(fig, save, f"cross_dataset_consensus_lines_{metric}", dpi=dpi,
                 verbose=verbose)
        if show:
            plt.show()
    return fig, ax, S


def _xd_pair_parts(label):
    """'$\mathbf{Michaels}$ – $\mathbf{Marson}$\n(Progenitor T Cells)' ->
    ('Michaels', 'Marson', 'Progenitor T Cells')."""
    names = re.findall(r"\\mathbf\{([^}]*)\}", str(label))
    if not names:                       # plain label: 'Michaels – Zhu\n(system)'
        names = [t.strip() for t in re.split(r"\s[–-]\s",
                                             str(label).split("\n")[0].strip())]
    sysm = re.search(r"\(([^)]*)\)", str(label))
    lineage = names[0] if names else str(label)
    screen = names[1] if len(names) > 1 else ""
    return lineage, screen, (sysm.group(1) if sysm else "")


def fig_cross_dataset_combo(S_lines, S_heat, tiers=(2, 3, 4), metric="pct",
                            methods=None, colors=None, dash_second=True,
                            line_color=None, row_labels=None,
                            lw=.7, ms=2, summary=True, summary_stat="mean",
                            summary_color="black", summary_lw=2.2, summary_ms=3.5,
                            summary_text=True, summary_values=False,
                            summary_value_fmt="{:.0f}%",
                            summary_value_pos=("above", "below", "above"),
                            legend_lw=None,
                            ref=0., cmap="RdBu_r", vmax=None, annot=True,
                            annot_fmt="{:.0f}", annot_stars=False,
                            fs=7, figsize=(4.7, 2.2),
                            width_ratios=None, margins=None,
                            panel_order="lines", lines_y_side=None, lines=True,
                            label_box=False, label_chip=True,
                            box_pad=.28, box_lw=1.1,
                            box_round=.25,
                            chip_pt=9, label_pad_pt=15, label_side="between",
                            label_gap_pt=3, label_air_pt=2, w_pad=.25, lines_ratio=1.45,
                            heat_w_in=1.05, gap_lines_in=.10, gap_heat_in=.03,
                            cbar=True, cbar_labels=("more\nresponsive", "less\nresponsive"),
                            cbar_ticks=(0,), cbar_room_in=.74, one_sided=True,
                            cbar_zero_label=None, cbar_zero_color="0.35",
                            cbar_zero_linespacing=1.05,
                            cbar_below_label=None, cbar_below_gap_pt=12,
                            cbar_text_rotation=0, cbar_rotate="all",
                            xlabel_lines="methods calling the gene",
                            xlabel_heat="method", ylabel=None,
                            save=None, dpi=400, show=True):
    """Cross-dataset responsiveness as ONE figure: consensus lines (left) and the
    per-method heatmap (right), on one shared height.

    The two tables have the same rows (perturb-seq screen x lineage dataset), so
    the heatmap's row labels double as the legend for the lines: each label carries
    a short sample of its line. Lines take the LINEAGE dataset's colour
    (DATASET_COLORS, the palette of figures 1a and 2a); a lineage dataset with two
    screens draws its second screen dashed (dash_second=True).

    line_color : one colour for every per-row line (e.g. "0.82"), with the markers
              keeping the lineage dataset's colour. Default None = coloured lines.
    annot_stars : print BH-adjusted significance above each cell value. Needs a 'q'
              column on S_heat, or a 'p' column, which is then BH-corrected across
              every cell the heatmap draws.

    S_lines : xd_consensus_sweep() table (method column = '>=k').
    S_heat  : xd_sweep() table at one top_n (fig_cross_dataset_sweep(kind='cutoff')
              returns it). The canonical SCG column is replaced by SCG-2-8, as the
              standalone heatmap does.
    label_chip : False draws the row names alone, with no coloured line sample and
              no frame (label_box must also be False). Use it when the lines are
              keyed some other way.
    label_side : "between" (default) gives the row labels their own column between
              the two plots: a line sample on the left, the name CENTRED in the rest
              of the column, rows locked to the heatmap's. "right" puts them as
              tick labels on the far right of the heatmap.
    width_ratios : column widths. Default (lines_ratio, .8, 1) for "between" (lines,
              labels, heatmap -- the label column is then resized to its content)
              and (lines_ratio, 1.25) for "right". lines_ratio > 1 makes the lines
              plot wider than the heatmap.
    label_gap_pt / label_air_pt : gap between a line sample and its name, and spare
              room around the widest name. Lower = tighter.
    heat_w_in / gap_lines_in / gap_heat_in : with label_side="between" the three
              parts are packed explicitly (tight_layout gives every gap the same
              width, set by the heatmap's rotated tick labels): the heatmap is
              heat_w_in wide, the label column exactly fits the names, the gaps are
              as given, and the lines plot takes everything left over.
    one_sided : True (default) -- colour scale from 0 (white) to the maximum (red),
              using the red half of `cmap`, with only the upper arrow and label.
              Right when every value is >= 0; a negative value would be drawn white,
              so the function warns if it finds one. False = the diverging scale
              centred on 0, with both arrows and both labels.
    cbar : a slim colour bar right beside the heatmap, at its full height, with arrow
              ends and cbar_labels written beside its red (top) and blue (bottom) arrow
              tips, so the direction reads without a caption. Only the 0
              tick by default -- the cells already print the values. cbar_room_in is
              the width (inches) kept free for it on the right.
    summary_stat : "mean" (default) or "median" across the screen pairs, drawn as one
              heavy black line. summary_text writes its name beside the last point
              (True = the statistic's name, or pass your own string, False = none).
    Cells print the value (annot); the colour bar only carries the direction. At
    colourbar costs a third of the panel width and repeats the printed numbers.
    Returns (fig, (ax_lines, ax_heat), matrix DataFrame).
    """
    from matplotlib.lines import Line2D
    col = {"pct": "pct", "cliffs_d": "cliffs_d"}[metric]
    summary_color = summary_color or "black"
    rows = list(dict.fromkeys(S_heat["dataset"]))

    # ---- colours / line styles per row, keyed on the lineage dataset ----
    # the dash marks the SECOND SCREEN of a lineage dataset, not the second row:
    # with one row per (pair, group) the row count would dash a screen's own second
    # group and then everything after it
    order_seen, style = {}, {}
    for r in rows:
        lin, scr, _ = _xd_pair_parts(r)
        seq = order_seen.setdefault(lin, [])
        if scr not in seq:
            seq.append(scr)
        c = (colors or {}).get(r) or DATASET_COLORS.get(dkey(lin), "0.5")
        style[r] = dict(color=c,
                        ls="--" if (dash_second and seq.index(scr)) else "-")

    # ---- heatmap matrix: rows x methods ----
    H = S_heat.copy()
    if "SCG-2-8" in set(H["method"]):
        H = H[H["method"] != "SCG"].copy()
        H["method"] = H["method"].replace({"SCG-2-8": "ClonePair"})
    H["method"] = H["method"].replace({"POWERSEEK": "Power-Seek"})
    order = [m for m in (methods or METHOD_ORDER) if m in set(H["method"])]
    M = (H.pivot_table(index="dataset", columns="method", values=col, aggfunc="first")
          .reindex(index=rows, columns=order))
    vmax = vmax or float(np.nanmax(np.abs(M.values)))

    style_rc = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                "xtick.labelsize": fs, "ytick.labelsize": fs}
    with plt.rc_context(style_rc):
        between = label_side != "right"
        if not lines:
            # heatmap only -- the tier lines are drawn as their own figure. Two
            # columns when the row names keep their own column, one when they are
            # the heatmap's own tick labels. axl stays None and every lines-only
            # step below is skipped.
            label_chip = False       # a line sample would key lines that are gone
            wr = list(width_ratios or ((.8, 1) if between else (1,)))
            axs = np.atleast_1d(plt.subplots(
                1, len(wr), figsize=figsize,
                gridspec_kw=dict(width_ratios=wr))[1])
            fig = axs[0].get_figure()
            axl = None
            if between:
                axlab, axh = (axs if panel_order != "heat" else axs[::-1])
            else:
                axlab, axh = None, axs[0]
        elif between:
            wr = list(width_ratios or (lines_ratio, .8, 1))
            axs = plt.subplots(1, len(wr), figsize=figsize,
                               gridspec_kw=dict(width_ratios=wr))[1]
            fig = axs[0].get_figure()
            # panel_order="heat" puts the heatmap on the LEFT and the lines on the
            # right; the label column stays between them either way, and the colour
            # bar and the legend chips follow the heatmap (see below).
            axl, axlab, axh = (axs if panel_order != "heat" else axs[::-1])
        else:
            wr = list(width_ratios or (lines_ratio, 1.25))
            axs = plt.subplots(1, len(wr), figsize=figsize,
                               gridspec_kw=dict(width_ratios=wr))[1]
            fig = axs[0].get_figure()
            axl, axh = axs
            axlab = None
        # keep room on the right for the colour bar (and, with label_side="right",
        # the colour bar sits past the row labels, so give it the same room)
        _room = (cbar_room_in / figsize[0]) if cbar else 0
        rect = ([0, 0, 1 - _room, 1] if panel_order != "heat"
                else [_room, 0, 1, 1])        # the bar now needs room on the LEFT
        if lines:
            # ---- lines ----
            L = S_lines[S_lines["method"].isin([f">={k}" for k in tiers])]
            x = np.arange(len(tiers))
            # which side the lines plot's y axis ends up on: anything annotated at the
            # last point has to know, or it lands on the axis
            _ys = lines_y_side or ("right" if panel_order == "heat" else "left")
            if ref is not None:
                axl.axhline(ref, color="0.75", lw=.7, ls=(0, (3, 2)), zorder=1)
            vals = []
            for r in rows:
                v = (L[L.dataset == r].set_index("method")
                     .reindex([f">={k}" for k in tiers])[col].to_numpy(float))
                vals.append(v)
                # line_color: one neutral colour for every line, the MARKERS keeping the
                # lineage dataset's colour. The lines then stop competing with the black
                # summary and the colour key moves to the points.
                st = (dict(style[r]) if line_color is None else
                      dict(ls=style[r]["ls"], color=line_color,
                           markerfacecolor=style[r]["color"],
                           markeredgecolor=style[r]["color"]))
                axl.plot(x, v, marker="o", lw=lw, ms=ms, zorder=3, **st)
            if summary:
                agg = (np.nanmedian if summary_stat == "median" else np.nanmean)(
                    np.vstack(vals), axis=0)
                axl.plot(x, agg, "-o", color=summary_color, lw=summary_lw, ms=summary_ms,
                         zorder=5, markeredgecolor="white", markeredgewidth=.8)
                if summary_values:
                    # one value per tier on the summary line only. The per-dataset lines
                    # cross each other, so labelling them all is unreadable; the
                    # above/below sequence keeps a label off the line it belongs to.
                    _pos = list(summary_value_pos)
                    _pos += ["above"] * (len(x) - len(_pos))
                    for _xi, _v, _p in zip(x, agg, _pos):
                        _dy, _va = (5, "bottom") if str(_p).startswith("a") else (-5, "top")
                        axl.annotate(summary_value_fmt.format(_v), (_xi, _v),
                                     xytext=(0, _dy), textcoords="offset points",
                                     ha="center", va=_va, fontsize=fsize(fs - 1),
                                     color=summary_color, fontweight="bold", zorder=6)
                if summary_text:
                    lab = summary_stat if summary_text is True else str(summary_text)
                    # with the y axis on the right there is no room beside the last
                    # point, so the name stacks ABOVE its value instead of beside it
                    axl.annotate(lab, (x[-1], agg[-1]),
                                 xytext=((4, 0) if _ys == "left" else (0, 15)),
                                 textcoords="offset points",
                                 ha="left" if _ys == "left" else "center",
                                 va="center" if _ys == "left" else "bottom",
                                 fontsize=fs, color=summary_color, fontweight="bold",
                                 zorder=6, clip_on=False)
            axl.set_xticks(x)
            # "4" not ">=4" when the tier IS every method (no deeper level exists)
            n_meth = len(METHOD_FILES)
            axl.set_xticklabels([f"{k}" if k >= n_meth else f"≥{k}" for k in tiers])
            axl.set_xlim(-.3, len(tiers) - .7)
            axl.set_xlabel(xlabel_lines)
            axl.set_ylabel(ylabel or {"pct": "% higher median responsiveness",
                                      "cliffs_d": "Cliff's δ"}[metric])
            # with the lines on the RIGHT, their y axis belongs on the figure's outer
            # edge: on the inner side it lands between the row labels and the plot
            if _ys == "right":
                axl.yaxis.set_ticks_position("right")
                axl.yaxis.set_label_position("right")
                axl.spines[["top", "left"]].set_visible(False)
                axl.spines["right"].set_visible(True)
            else:
                axl.spines[["top", "right"]].set_visible(False)

        # ---- heatmap ----
        if one_sided:
            if np.nanmin(M.values) < 0:
                warnings.warn("one_sided=True but the heatmap has negative values "
                              f"(min {np.nanmin(M.values):.1f}); they are drawn as 0. "
                              "Pass one_sided=False for a diverging scale.", stacklevel=2)
            from matplotlib.colors import ListedColormap
            base_cmap = plt.get_cmap(cmap)
            cmap_draw = ListedColormap(base_cmap(np.linspace(.5, 1, 256)))
            vmin_draw = 0.0
        else:
            cmap_draw, vmin_draw = cmap, -vmax
        im = axh.imshow(M.values, cmap=cmap_draw, vmin=vmin_draw, vmax=vmax, aspect="auto")
        if annot:
            Q = None
            if annot_stars:
                _pc = "q" if "q" in H.columns else "p" if "p" in H.columns else None
                if _pc is None:
                    warnings.warn("annot_stars=True but the heatmap table has no 'p' "
                                  "or 'q' column; stars are skipped.", stacklevel=2)
                else:
                    Q = (H.pivot_table(index="dataset", columns="method", values=_pc,
                                       aggfunc="first")
                          .reindex(index=rows, columns=order).values.astype(float))
                    if _pc == "p":        # ONE BH correction over every cell drawn
                        flat = Q.ravel().copy()
                        ok = np.isfinite(flat)
                        flat[ok] = bh_fdr(flat[ok])
                        Q = flat.reshape(Q.shape)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M.values[i, j]
                    if np.isfinite(v):
                        txt = annot_fmt.format(v)
                        if Q is not None and np.isfinite(Q[i, j]):
                            txt = f"{_stars(Q[i, j])}\n{txt}"
                        axh.text(j, i, txt, ha="center", va="center",
                                 fontsize=fsize(fs - 1), linespacing=1.0,
                                 color="white" if abs(v) > .6 * vmax else "0.15")
        axh.set_xticks(range(len(order)))
        axh.set_xticklabels(order, rotation=35, ha="right", rotation_mode="anchor")
        axh.set_xlabel(xlabel_heat)
        if not between:
            axh.yaxis.tick_right()
        axh.set_yticks(range(len(rows)))
        labs = [f"{a} – {b}" if b else a for a, b, _ in map(_xd_pair_parts, rows)]
        # the default label is rebuilt from the mathbf parts of the row name, so
        # any extra tag in that string (an experimental group, say) is discarded.
        # row_labels says what the rows are called: a dict keyed by the row value,
        # or a list in row order.
        if row_labels is not None:
            labs = ([row_labels.get(r, l) for r, l in zip(rows, labs)]
                    if isinstance(row_labels, dict) else list(row_labels))
        if between:
            axh.set_yticklabels([])
            axh.tick_params(axis="y", length=0)
            # the label column: same rows as the heatmap, drawn as plain text
            axlab.set_ylim(axh.get_ylim())
            axlab.set_xlim(0, 1)
            axlab.axis("off")
        else:
            axh.set_yticklabels(labs)
            axh.tick_params(axis="y", length=0, pad=label_pad_pt)
        axh.tick_params(axis="x", length=0)
        for sp in axh.spines.values():
            sp.set_visible(False)

        if margins:
            fig.subplots_adjust(**margins)
        else:
            fig.tight_layout(w_pad=w_pad, rect=rect)
        if between and width_ratios is None:
            # pack lines | labels | heatmap explicitly: the label column fits the
            # names exactly, the heatmap has a fixed width, the gaps are small and
            # the lines plot gets whatever width is left
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            W = fig.get_figwidth()
            probe = [fig.text(0, 0, t, fontsize=fs) for t in labs]
            w_txt = max(p.get_window_extent(rend).width for p in probe) / fig.dpi
            for p in probe:
                p.remove()
            lab_w = w_txt + (chip_pt + label_gap_pt + label_air_pt) / 72
            if label_box:
                # no chip beside the name; the frame around it needs its padding
                lab_w = w_txt + (2 * box_pad * fs + label_air_pt + 4) / 72
            elif not label_chip:
                # plain names, nothing drawn beside them
                lab_w = w_txt + (label_air_pt + 4) / 72
            # the span to fill is the outer edges of the row, whichever panel is
            # where: with panel_order="heat" the lines axes is the RIGHT one, and
            # reading left from it gives a negative width
            _pos = [a.get_position() for a in (axl, axlab, axh) if a is not None]
            left = min(p_.x0 for p_ in _pos) * W
            right = max(p_.x1 for p_ in _pos) * W
            lines_w = (right - left) - lab_w - heat_w_in - gap_lines_in - gap_heat_in
            x = left
            if not lines:
                # heatmap and row names only: pack them against the heatmap's side
                # of the row and let the spare width fall on the other side
                seq = (((axh, heat_w_in, gap_heat_in), (axlab, lab_w, 0.))
                       if panel_order == "heat" else
                       ((axlab, lab_w, gap_heat_in), (axh, heat_w_in, 0.)))
                if panel_order != "heat":
                    x = right - (lab_w + gap_heat_in + heat_w_in)
            else:
                seq = (((axl, lines_w, gap_lines_in), (axlab, lab_w, gap_heat_in),
                        (axh, heat_w_in, 0.)) if panel_order != "heat" else
                       ((axh, heat_w_in, gap_heat_in), (axlab, lab_w, gap_lines_in),
                        (axl, lines_w, 0.)))
            for ax_, w_, gap_ in seq:
                p_ = ax_.get_position()
                ax_.set_position([x / W, p_.y0, w_ / W, p_.height])
                x += w_ + gap_
            fig.canvas.draw()
        # align only when the heatmap HAS a label: its baseline sits below the rotated
        # method names, and aligning to an empty label still drags the lines plot's
        # x label down to that level
        if xlabel_heat and lines:
            fig.align_xlabels([axl, axh])

        # ---- colour bar: direction of the heatmap scale ----
        if cbar:
            fig.canvas.draw()
            W = fig.get_figwidth()
            ph = axh.get_position()
            if not between:        # past the row labels on the right
                rend = fig.canvas.get_renderer()
                x_start = max(t.get_window_extent(rend).x1 for t in axh.get_yticklabels()
                              ) / fig.dpi / W + (4 / 72) / W
            else:
                x_start = ph.x1 + .05 / W
            wbar = .07 / W
            if panel_order == "heat":         # bar on the far side, left of the heatmap
                x_start = ph.x0 - .05 / W - wbar
            # arrows add extendfrac of the bar length at each end: shrink the bar so
            # the arrow tips, not the bar ends, line up with the heatmap's edges
            ef = .06
            n_arrows = 1 if one_sided else 2
            hbar = ph.height / (1 + n_arrows * ef)
            y_bar = ph.y0 if one_sided else ph.y0 + ph.height * ef / (1 + 2 * ef)
            axcb = fig.add_axes([x_start, y_bar, wbar, hbar])
            cb = fig.colorbar(im, cax=axcb, extend="max" if one_sided else "both",
                              extendfrac=ef)
            cb.set_ticks(list(cbar_ticks or []))
            if panel_order == "heat":
                # the bar sits left of the heatmap, so its numbers go left too,
                # beside the words rather than wedged against the first column
                cb.ax.yaxis.set_ticks_position("left")
            cb.ax.tick_params(labelsize=fsize(fs - 1), length=2, width=.6, pad=1.5)
            cb.outline.set_linewidth(.6)
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            if panel_order == "heat":     # text to the LEFT of the bar
                x_txt = (min([t.get_window_extent(rend).x0
                              for t in cb.ax.get_yticklabels()]
                             + [axcb.get_window_extent(rend).x0])
                         / fig.dpi / W) - (3 / 72) / W
                _txt_ha = "right"
            else:
                x_txt = (max([t.get_window_extent(rend).x1
                              for t in cb.ax.get_yticklabels()]
                             + [axcb.get_window_extent(rend).x1])
                         / fig.dpi / W) + (3 / 72) / W
                _txt_ha = "left"
            cmap_ = plt.get_cmap(cmap)
            label_specs = [(cbar_labels[0], ph.y1, "top", cmap_(.88), "top")]
            if not one_sided and len(cbar_labels) > 1:
                label_specs.append((cbar_labels[1], ph.y0, "bottom", cmap_(.12),
                                    "bottom"))
            # On a one-sided bar the bottom end is the palest colour, so a label
            # there must NOT take its colour from the map or it prints white on
            # white. It also carries a different message: 0 is the neutral point,
            # not the minimum of the data.
            # "0 = ..." sits LEVEL with the 0 tick (va='center'), so the two zeros
            # line up. Anything further is a separate block below the bar, which
            # lands in the band the heatmap's column labels occupy.
            if cbar_zero_label:
                label_specs.append((cbar_zero_label, ph.y0, "center",
                                    cbar_zero_color, "zero"))
            if cbar_below_label:
                _gap = (cbar_below_gap_pt / 72) / fig.get_figheight()
                label_specs.append((cbar_below_label, ph.y0 - _gap, "top",
                                    cbar_zero_color, "below"))
            for lab, yy, va, cc, kind in label_specs:
                _rot_this = (cbar_text_rotation
                             if cbar_rotate == "all" or kind in cbar_rotate else 0)
                if _rot_this:
                    # running the words along the bar costs height instead of width:
                    # the top label hangs down from the bar's top, the others run up
                    # and down from the zero end
                    va = {"top": "top", "center": "bottom", "bottom": "top"}[va]
                    # keep the block on its own side of the bar: centring a
                    # rotated label on the anchor puts half of it over the arrow
                    _ha, _rot = _txt_ha, _rot_this
                else:
                    _ha, _rot = _txt_ha, 0
                if lab:
                    # the zero label gets its own line spacing: its two lines say
                    # different things (what 0 is, what a negative would mean) and
                    # read better separated than the two halves of "more/responsive"
                    _ls = (cbar_zero_linespacing
                           if lab in (cbar_zero_label, cbar_below_label) else 1.05)
                    fig.text(x_txt, yy, lab, ha=_ha, va=va, rotation=_rot,
                             linespacing=_ls, fontsize=fsize(fs - 1), color=cc)

        # ---- line samples in front of each row label: the legend for the lines ----
        fig.canvas.draw()
        if between:
            # line sample at the left edge of the column, name centred in the rest
            w_in = axlab.get_position().width * fig.get_figwidth()
            _chip = (chip_pt / 72) / w_in
            # the sample sits on the side the lines plot is on
            c0, c1 = ((0.0, _chip) if panel_order != "heat" else (1.0 - _chip, 1.0))
            gap = (label_gap_pt / 72) / w_in
            for i, (r, lab) in enumerate(zip(rows, labs)):
                if not label_chip and not label_box:
                    # names only: the lines carry their key somewhere else (their
                    # markers, say), so nothing coloured is drawn in this column
                    axlab.text(.5, i, lab, ha="center", va="center", fontsize=fs)
                    continue
                if label_box:
                    # the name itself carries the line's colour and dash pattern, as
                    # a frame around it, so the legend needs no separate sample
                    axlab.text(.5, i, lab, ha="center", va="center", fontsize=fs,
                               bbox=dict(boxstyle=f"round,pad={box_pad}",
                                         facecolor="none", edgecolor=style[r]["color"],
                                         linestyle=style[r]["ls"],
                                         linewidth=(box_lw if legend_lw is None
                                                    else legend_lw),
                                         joinstyle="round"))
                    continue
                axlab.add_line(Line2D([c0, c1], [i, i], clip_on=False,
                                      lw=(lw + .3 if legend_lw is None else legend_lw),
                                      **style[r]))
                # the name is centred in whatever the chip leaves free, on the
                # other side of it
                _cx = ((c1 + gap + 1) / 2 if panel_order != "heat"
                       else (c0 - gap) / 2)
                axlab.text(_cx, i, lab, ha="center", va="center", fontsize=fs)
        else:
            w_in = axh.get_position().width * fig.get_figwidth()
            x0, x1 = 1 + (2 / 72) / w_in, 1 + (chip_pt / 72) / w_in
            tr = mtransforms.blended_transform_factory(axh.transAxes, axh.transData)
            for i, r in enumerate(rows):
                axh.add_line(Line2D([x0, x1], [i, i], transform=tr, clip_on=False,
                                    lw=(lw + .3 if legend_lw is None else legend_lw),
                                    **style[r]))

        save_fig(fig, save, "fig2_cross_dataset_combo", dpi=dpi)
        if show:
            plt.show()
    return fig, (axl, axh), M


def leaveout_curve_data(memset="ClonePair", min_methods=5, timepoint="merge", top_n=TOP_N,
                        merge_mode=MERGE_MODE, method_files=None, memgene_dir=None,
                        step=25, n_bins=50, n_bg=2000, seed=0, floor=EXPR_FLOOR,
                        lfc_df=None, fast=True, verbose=True):
    """The leave-out-KO curves, without drawing anything.

    Reuses marson_direction.plot_leaveout_kos for the computation (including its
    expression-matched background) and throws its figure away, so the numbers stay
    identical to the original analysis while the drawing is ours.
    """
    if not fast:
        fig, out = fig_leaveout_kos(memset=memset, min_methods=min_methods,
                                   timepoint=timepoint, top_n=top_n, metric="abs",
                                   merge_mode=merge_mode, method_files=method_files,
                                   memgene_dir=memgene_dir, step=step, n_bins=n_bins,
                                   n_bg=n_bg, seed=seed, floor=floor, lfc_df=lfc_df,
                                   save=None, show=False, verbose=verbose)
        plt.close(fig)
        return out

    # ---- same curve, computed with suffix sums ------------------------------
    # plot_leaveout_kos recomputes the mean over the remaining perturbations at every
    # step, so its cost is (points x perturbations x genes) and step=25 is what keeps
    # it affordable. The perturbations are removed in ONE fixed order, so every point
    # is a suffix mean: sum the rows once, cumulate from the end, divide by the running
    # count. That is exact (not an approximation) and makes step=1 cheap.
    _perturb_path()
    from marson_direction import consensus_and_background, _match_expression
    gs, ge = load_marson(floor, verbose)
    mf = method_files or METHOD_FILES
    _, _, counts = consensus_and_background(
        timepoint=timepoint, top_n=top_n, merge_mode=merge_mode,
        memgene_dir=memgene_dir or MEMGENE_DIR, method_files=mf,
        universe=gs.index, verbose=False)
    nonmem = never_called(gs.index, top_n, timepoint=timepoint, merge_mode=merge_mode,
                          memgene_dir=memgene_dir, method_files=method_files)
    mem = (counts.index[counts >= int(min_methods)].tolist() if memset is None
           else memory_set(memset, timepoint, top_n, restrict_to=gs.index,
                           merge_mode=merge_mode, memgene_dir=memgene_dir,
                           method_files=method_files))
    if lfc_df is None:
        lfc_df = load_marson_lfc(verbose)
    measured = lfc_df.columns
    mem = [g for g in mem if g in measured]
    rng = np.random.default_rng(seed)
    pool = _match_expression(mem, [g for g in nonmem if g in measured], ge, measured,
                             n_bins=n_bins, rng=rng)

    sub = lfc_df[mem]
    ranked = sub.abs().mean(axis=1).sort_values(ascending=False).index   # strongest first
    ks = list(range(0, len(sub) - 50, step))

    def suffix_mean(frame):
        """mean |LFC| over rows ranked[k:], for every k, in one pass"""
        v = np.abs(frame.loc[ranked].to_numpy(float))
        ok = np.isfinite(v)
        tot = np.where(ok, v, 0.0).sum(axis=1)
        cnt = ok.sum(axis=1).astype(float)
        # cumulative from the END: element k is the total over rows k..n-1
        suf_t = np.concatenate([np.cumsum(tot[::-1])[::-1], [0.0]])
        suf_c = np.concatenate([np.cumsum(cnt[::-1])[::-1], [0.0]])
        return np.array([suf_t[k] / suf_c[k] if suf_c[k] else np.nan for k in ks])

    out = pd.DataFrame({"ko_removed": ks, "mean_abs": suffix_mean(sub),
                        "bg_mean_abs": suffix_mean(lfc_df[pool])})
    if verbose:
        print(f"leave-out curve: {len(mem)} memory genes, {len(pool)} matched background "
              f"genes, {len(ks)} points (step {step})")
    return out


def fig_leaveout_curve(out=None, x="frac", mem_color=C_MEM, bg_color=C_BG,
                       fill_alpha=.13, lw=2.2, fs=13, figsize=(6.4, 4.4),
                       mem_label="Memory genes", bg_label="Expression-matched",
                       annot_ratio=True, annot_fs=None, ratio_fmt="{:.2f}×",
                       annot_dy=(0, 0), xticks=None, yticks=None,
                       xlabel=None, ylabel=None, title=None, legend_loc="upper right",
                       legend_frame=False, ylim=None, ax=None, legend=True,
                       save=None, dpi=400, show=True, verbose=True, **kw):
    """Leave-out-KO curves, drawn in the figure-2 style.

    Progressively removes the most-impactful perturbations and re-measures mean
    |LFC| over the remaining ones. The memory curve staying ABOVE the matched
    control means the higher responsiveness is not carried by a few strong KOs.

    x : 'frac' (% of perturbations removed, comparable across datasets) or 'n'.
    The band between the curves is shaded, so the quantity that matters — the gap —
    is a visible area rather than something the reader has to subtract by eye.
    """
    if out is None:
        out = leaveout_curve_data(verbose=verbose, **kw)
    k = out["ko_removed"].to_numpy(float)
    a = out["mean_abs"].to_numpy(float)
    b = out["bg_mean_abs"].to_numpy(float)
    xv = 100 * k / (k.max() + (k[1] - k[0] if len(k) > 1 else 1)) if x == "frac" else k

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
    with plt.rc_context(style):
        own = ax is None
        fig, ax = (plt.subplots(figsize=figsize) if own
                   else (ax.get_figure(), ax))
        ax.fill_between(xv, b, a, where=a >= b, color=mem_color, alpha=fill_alpha,
                        lw=0, zorder=1)
        ax.plot(xv, a, color=mem_color, lw=lw, zorder=3, label=mem_label)
        ax.plot(xv, b, color=bg_color, lw=lw, zorder=3, label=bg_label)
        if annot_ratio:
            for k, (i, ha, dx) in enumerate(((0, "left", 4),
                                             (len(xv) - 1, "right", -4))):
                if np.isfinite(a[i]) and np.isfinite(b[i]) and b[i] > 0:
                    ax.annotate(ratio_fmt.format(a[i] / b[i]),
                                (xv[i], (a[i] + b[i]) / 2),
                                xytext=(dx, annot_dy[k]),
                                textcoords="offset points", ha=ha, va="center",
                                fontsize=fsize(annot_fs or fs - 2), color="0.25")
        # `or` would swallow an explicit "" and restore the default, which is how a
        # grid ends up with the same axis label under every panel
        ax.set_xlabel(("percentage of perturbations removed" if x == "frac"
                       else "number of most-impactful perturbations removed")
                      if xlabel is None else xlabel)
        # identical wording to 2a/2c — same quantity, same axis, same label
        ax.set_ylabel(metric_label(METRIC) if ylabel is None else ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(xv.min(), xv.max())
        if xticks is not None:
            ax.set_xticks(list(xticks))
        if yticks is not None:
            ax.set_yticks(list(yticks))
        ax.spines[["top", "right"]].set_visible(False)
        if legend:
            ax.legend(loc=legend_loc, frameon=legend_frame)
        if title:
            ax.set_title(title, fontsize=fs + 1, pad=8)
        fig.tight_layout()
        save_fig(fig, save, "fig2_leaveout_curve", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    if verbose:
        print(f"ratio memory/control: start {a[0]/b[0]:.2f}x  end {a[-1]/b[-1]:.2f}x  "
              f"| memory above control at {100*np.mean(a > b):.0f}% of steps")
    return fig, ax, out


def fig_matching_grid(methods=None, panels=None, memset=None, timepoint="merge",
                      top_n=TOP_N, metric=METRIC, merge_mode=MERGE_MODE,
                      method_files=None, memgene_dir=None, n_iter=2000, k=10,
                      seed=0, floor=EXPR_FLOOR, res=None,
                      background="strict", equalise=True, expr_bins=True,
                      n_bins=50, expr_n_iter=100,
                      annot="pct+stars", show_medians=False, ylim=None, ylim_pct=99,
                      fill=False, median_inside=True, pct_where="left",
                      xlabel="gene property the background was matched on",
                      ylabel=None,
                      fs=12, panel_w=7.6, panel_h=3.0, sharey=True, ncol=1,
                      method_label="auto", legend=True, legend_out=None,
                      title_pad=6, title_fs=None, title_weight="bold",
                      xlabel_fs=None, ylabel_fs=None,
                      row_label_x=-.075, suptitle=None, save=None, dpi=400,
                      show=True, verbose=True):
    """(Suppl. Fig 2c) The covariate matching rungs for EVERY method.

    Styling and matcher default to the main figure's panel b — unfilled violins,
    the median drawn inside each half, percentages beside the necks, and the
    expression rung run through the same quantile-bin matcher as panel a.

    One row per method, each row a set of split violins over the matching rungs.
    Shows that the main figure's choice of method is not doing the work — every
    method survives the same ladder.

    Returns (fig, axes, {method: {rung: result}}, tidy stats).
    """
    panels = panels or MATCH_PANELS_CUMULATIVE
    methods = list(methods or METHOD_FILES)
    if res is None:
        res = {}
        for m in methods:
            if verbose:
                print(f"  {m} ...", flush=True)
            res[m] = matching_rung_results(
                memset=m, timepoint=timepoint, top_n=top_n, panels=panels,
                metric=metric, merge_mode=merge_mode, method_files=method_files,
                memgene_dir=memgene_dir, n_iter=n_iter, k=k, seed=seed,
                floor=floor, background=background, equalise=equalise,
                expr_bins=expr_bins, n_bins=n_bins, expr_n_iter=expr_n_iter,
                verbose=False)
    methods = [m for m in methods if m in res]

    stats = []
    with plt.rc_context({**PUB_RC, "font.size": fs}):
        # one column = one row per method (the original layout); ncol > 1 tiles the
        # methods into a grid, which only works if the method name moves from the
        # rotated side label into a title (a side label would sit on its left
        # neighbour) and the legend comes back inside the first panel.
        _nc = int(ncol or 1)
        _nr = int(np.ceil(len(methods) / _nc))
        fig, _grid = plt.subplots(_nr, _nc, sharey=sharey, squeeze=False,
                                  figsize=(panel_w * _nc, panel_h * _nr))
        axes = _grid.ravel()[:len(methods)]
        for _ax in _grid.ravel()[len(methods):]:
            _ax.set_visible(False)
        _side = method_label == "side" or (method_label == "auto" and _nc == 1)
        _lout = (_nc == 1) if legend_out is None else legend_out
        for i, (m, ax) in enumerate(zip(methods, axes)):
            _last_row = i + _nc >= len(methods)
            _, _, S = fig_responsiveness_violin(
                res=res[m], bg_label="Matched background", annot=annot,
                show_medians=show_medians, ylim=ylim, ylim_pct=ylim_pct,
                fill=fill, median_inside=median_inside, pct_where=pct_where,
                xlabel=xlabel if _last_row else "",
                fs=fs, ax=ax, legend=(legend and i == 0), legend_out=_lout,
                show=False, verbose=False)
            S.insert(0, "memory_set", m)
            stats.append(S)
            ax.set_ylabel("")                 # one shared label via supylabel
            if _side:
                ax.text(row_label_x, .5, m, transform=ax.transAxes, ha="right",
                        va="center", fontsize=title_fs or fs + 2,
                        fontweight=title_weight, rotation=90)
            else:
                ax.set_title(m, fontsize=title_fs or fs + 1,
                             fontweight=title_weight, pad=title_pad)
            if xlabel_fs:
                ax.xaxis.label.set_size(xlabel_fs)
            if not _last_row:
                ax.set_xticklabels([])
        fig.supylabel(metric_label(metric) if ylabel is None else ylabel,
                      fontsize=ylabel_fs or fs + 1, x=.015)
        if suptitle:
            fig.suptitle(suptitle, fontsize=fs + 2)
        fig.tight_layout()
        save_fig(fig, save, f"figS_matching_grid_top{top_n}", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    S = pd.concat(stats, ignore_index=True)
    S["q"] = bh_fdr(S["p"].values)
    if verbose:
        print(S.pivot(index="memory_set", columns="method", values="pct")
               .round(1).to_string())
    return fig, axes, res, S


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1a — the lineage-barcode datasets themselves (no perturb-seq)
# ═══════════════════════════════════════════════════════════════════════════

# The column that defines the paired groups every method compares across, and the
# clone id. Harmange's `batch` is a single value 'all'; its arms live in `startID`.
LINEAGE_DATASETS = {
    "T-lineage": dict(h5ad=DATASETS["T-lineage"]["h5ad"], species="human",
                      system="haematopoiesis", clone="clone_id", group="batch"),
    "Harmange":  dict(h5ad=DATASETS["Harmange"]["h5ad"], species="human",
                      system="melanoma", clone="clone_id", group="startID"),
    # Weinreb's groups are the three culture DAYS (obs 'Time point', float
    # 2.0/4.0/6.0 — the same values that name its result csvs). Clone size is
    # per-day like everywhere else: 66% of LARRY clones span timepoints, so
    # collapsing a clone across days would invent one large clone out of three
    # small ones the methods never compare jointly.
    "LARRY":     dict(h5ad=DATASETS["LARRY"]["h5ad"], species="mouse",
                      system="HSPC", clone="clone_id", group="Time point"),
    "Bandler":   dict(h5ad=DATASETS["Bandler"]["h5ad"], species="mouse",
                      system="brain", clone="cloneID", group="batch"),
}
SPECIES_COLORS = {"human": "#B02318", "mouse": "#3B6EA5"}
CLONE_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, np.inf, "≥10")]


def lineage_design_rows(datasets=None, systems=None, group_labels=None,
                        weight="clones_ge2", hub="cells_in_ge2", min_clone=2,
                        share="global", verbose=True):
    """Rows, weights and hub values for a lineage-dataset sunburst, measured from
    the h5ads rather than typed in.

    rows    : (species, system, dataset, group) per leaf — one leaf per experimental
              group, since that is the unit the methods actually compare across.
    weight / hub : which measured quantity drives the wedge width and the hub bar.
        'cells'         all cells
        'clones'        all clones
        'clones_ge2'    clones with >= min_clone cells  (default width)
        'cells_in_ge2'  cells sitting in those clones   (default hub)
        'mean_clone'    mean cells per clone
      Singletons carry no within-clone information for any of these methods, so the
      >=2 variants are the ones that reflect what a dataset actually contributes.
    share   : how the weights are normalised into wedge widths.
        'global'   raw `weight`, so a wedge's width is its share of ALL datasets.
        'dataset'  every dataset gets an EQUAL slice of the circle and `weight`
                   only splits that slice between its own groups. Use this once one
                   dataset dwarfs the rest — Weinreb has 4,687 clones of >=2 cells
                   against Mold's 318, so under 'global' it takes 55% of the circle,
                   squeezes Michaels and Harmange into 15% between them and the
                   figure stops being readable as a design diagram. The panel is a
                   map of what was tested, not a bar chart of dataset size; the
                   actual counts are on the rim, where they can be read exactly.
                   NOTE this also detaches the inner rings from any total: under
                   'dataset' the species ring is 2/5 human and 3/5 mouse because
                   there are 2 and 3 datasets, not because of any cell count.
    """
    import anndata as ad
    datasets = datasets or active_subset(LINEAGE_DATASETS)
    datasets = {dkey(k): v for k, v in datasets.items()}
    # `systems` / `group_labels` may be keyed by the display (author) name.
    systems = ({dkey(k): v for k, v in systems.items()} if systems else
               {k: v["system"].capitalize() for k, v in datasets.items()})
    group_labels = {(dkey(k[0]), k[1]): v
                    for k, v in (group_labels or {}).items()}
    rows, weights, counts, info = [], [], {}, []
    for lab, cfg in datasets.items():
        A = ad.read_h5ad(cfg["h5ad"], backed="r")
        obs = A.obs
        A.file.close()
        cl = obs[cfg["clone"]].astype(str)
        gp = (obs[cfg["group"]].astype(str) if cfg.get("group") in obs.columns
              else pd.Series("all", index=obs.index))
        keep = ~cl.isin({"nan", "None", "", "-1"})
        for g in sorted(gp[keep].unique()):
            m = keep & (gp == g)
            sz = cl[m].value_counts()
            big = sz[sz >= min_clone]
            key = group_labels.get((lab, g), g)
            vals = dict(cells=int(m.sum()), clones=int(len(sz)),
                        clones_ge2=int(len(big)), cells_in_ge2=int(big.sum()),
                        mean_clone=float(sz.mean()))
            rows.append((cfg["species"].capitalize(), systems[lab], lab, key))
            weights.append(vals[weight])
            counts[key] = float(vals[hub])
            info.append(dict(dataset=lab, group=key, median_clone=float(sz.median()),
                             **vals))
    I = pd.DataFrame(info)
    if share == "dataset":
        # rescale each dataset's group weights to sum to 1/n_datasets
        tot = {}
        for r, x in zip(rows, weights):
            tot[r[2]] = tot.get(r[2], 0.0) + x
        n = len(tot)
        weights = [x / tot[r[2]] / n for r, x in zip(rows, weights)]
    elif share != "global":
        raise ValueError(f"share must be 'global' or 'dataset', got {share!r}")
    if verbose:
        print(I.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        if share == "dataset":
            print(f"  share='dataset': each of the {len(set(r[2] for r in rows))} "
                  "datasets takes an equal slice; widths are within-dataset shares")
    return rows, weights, counts, I


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary — SCG threshold sensitivity, and per-gene direction
# ═══════════════════════════════════════════════════════════════════════════
# (a) The sensitivity check behind the canonical SCG call. `concordance_freq` is
#     continuous and computed WITHOUT reference to the 0.80 cut, so the whole
#     0->1 sweep falls out of a SINGLE statistics pass — that pass is
#     scg_analysis/sweep_thresholds_SCGs.py, which wrote the three tables read
#     here. Nothing is recomputed; this only re-draws them in house style.
# (b) Of the genes several methods agree are memory genes, does each one move the
#     SAME way under every knockout? The statistic comes from
#     marson_direction.gene_direction_consistency — imported, not reimplemented.

SCG_SWEEP_DIR = f"{P.OUTPUTS}/scg_threshold_sweep"
SCG_CANON = 0.80          # the cut findingSCGs.py actually uses — do not move casually
SCG_CUTS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
C_DIR_CMAP = "PRGn"       # diverging: purple = down, green = up


def load_scg_sweep(dataset="T-lineage", sweep_dir=None, cache=True, verbose=True):
    """The three threshold-sweep tables for one dataset stem.

    Returns dict(dataset, full, counts, overlap, timepoints, seeds):
      full    every tested gene x seed x timepoint, with the continuous score
      counts  seed x threshold x timepoint -> n_scg
      overlap seed x threshold -> T1/T2 intersection and Jaccard
    """
    sweep_dir = sweep_dir or SCG_SWEEP_DIR
    key = ("scg_sweep", dataset, sweep_dir)
    if cache and key in _CACHE:
        return _CACHE[key]
    sw = {"dataset": dataset}
    for name, stem in (("full", "scg_concordance_full"),
                       ("counts", "scg_threshold_counts"),
                       ("overlap", "scg_threshold_overlap")):
        sw[name] = pd.read_csv(os.path.join(sweep_dir, f"{stem}_{dataset}.csv"))
    sw["timepoints"] = sorted(sw["counts"]["timepoint"].unique())
    sw["seeds"] = sorted(sw["full"]["seed"].unique())
    if verbose:
        print(f"{dataset} sweep: {len(sw['full']):,} gene-rows, "
              f"seeds {sw['seeds']}, timepoints {sw['timepoints']}, "
              f"thresholds {sw['counts'].threshold.min():.2f}–"
              f"{sw['counts'].threshold.max():.2f}")
    if cache:
        _CACHE[key] = sw
    return sw


def scg_threshold_table(sw=None, cuts=SCG_CUTS, dataset="T-lineage", verbose=True):
    """Counts (mean / min / max over seeds) and T1–T2 overlap at the conventional
    cuts — the numbers behind panel (a), as a table for the supplement."""
    if sw is None:
        sw = load_scg_sweep(dataset, verbose=verbose)
    c, o = sw["counts"], sw["overlap"]
    piv = (c[c.threshold.isin(cuts)]
           .groupby(["threshold", "timepoint"])["n_scg"]
           .agg(["mean", "min", "max"]).round(1).unstack("timepoint"))
    piv.columns = [f"{stat}_{tp}" for stat, tp in piv.columns]
    ovl = (o[o.threshold.isin(cuts)]
           .groupby("threshold")[["n_intersect", "jaccard"]].mean().round(3))
    return piv.join(ovl).reset_index()


def fig_scg_threshold(sw=None, dataset="T-lineage", canon=SCG_CANON, colors=None,
                      seed=None, logy=False, bin_width=.05, xlim=(-.03, 1.06),
                      annotate=True, canon_label=True, cuts=SCG_CUTS,
                      fs=13, figsize=(11.0, 4.4), wspace=.28, lw=2.0,
                      linestyles=None,
                      panel_letters="roman", titles=None, legend=True, legend_title="timepoint",
                      xlabels=None, ylabels=None, margins=None,
                      save=None, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 1a, 1b) How the sister-concordant gene call depends on the threshold.

    i   genes called vs the concordance-frequency threshold, one line per
        timepoint, band = min–max over the reshuffle seeds. The count at the
        canonical cut is annotated, so the figure states what the pipeline used
        rather than leaving it to the caption.
    ii  where the continuous score actually sits. A near-empty middle with mass at
        both ends means the cut lands in a valley — the call is not knife-edge.

    Returns (fig, (ax_i, ax_ii), table) with `table` from scg_threshold_table().
    """
    if sw is None:
        sw = load_scg_sweep(dataset, verbose=verbose)
    counts, full = sw["counts"], sw["full"]
    tps, seeds = sw["timepoints"], sw["seeds"]
    sd = seeds[0] if seed is None else seed
    colors = colors or {tp: (C_TPS + [f"C{i}" for i in range(len(tps))])[i]
                        for i, tp in enumerate(tps)}
    if linestyles is None:
        lss = {tp: "-" for tp in tps}
    elif isinstance(linestyles, dict):
        lss = {tp: linestyles.get(tp, "-") for tp in tps}
    else:
        lss = {tp: list(linestyles)[k % len(linestyles)] for k, tp in enumerate(tps)}
    if titles is None:
        titles = (("Genes called vs threshold", "Distribution of the score")
                  if SHOW_TITLES else ())
    xlabels = xlabels or ("concordance-frequency threshold", "concordance frequency")
    ylabels = ylabels or ("sister-concordant genes called",
                          "genes (log scale)")

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fsize(fs + 1),
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
    with plt.rc_context(style):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw=dict(wspace=wspace))

        # ---- i: how many survive each threshold ----
        for tp in tps:
            g = counts[counts.timepoint == tp].groupby("threshold")["n_scg"]
            m, lo, hi = g.mean(), g.min(), g.max()
            axa.plot(m.index, m.values, color=colors[tp], lw=lw, ls=lss[tp],
                     label=tp, zorder=3)
            axa.fill_between(m.index, lo.values, hi.values, color=colors[tp],
                             alpha=.18, lw=0, zorder=2)
            near = m.index[np.isclose(m.index.to_numpy(float), canon)]
            if annotate and len(near):
                axa.annotate(f"{m.loc[near[0]]:.0f}", (canon, m.loc[near[0]]),
                             textcoords="offset points", xytext=(6, 6),
                             color=colors[tp], fontweight="bold",
                             fontsize=fsize(fs - 2), zorder=4)
        axa.axvline(canon, color="0.45", ls="--", lw=1.0, zorder=1)
        if canon_label:
            # blended transform: x in DATA (pinned to the line), y in AXES coords —
            # ax.get_ylim()[1] is read before the final autoscale, so a data-coord
            # y lands above the axes and gets clipped by bbox_inches="tight"
            from matplotlib.transforms import blended_transform_factory
            axa.text(canon - .012, .995, f"canonical {canon:.2f}", color="0.45",
                     va="top", ha="right", fontsize=fsize(fs - 3),
                     transform=blended_transform_factory(axa.transData,
                                                         axa.transAxes))
        if logy:
            axa.set_yscale("log")
        axa.set_xlim(0, 1.0)
        axa.set_xlabel(xlabels[0])
        axa.set_ylabel(ylabels[0])

        # ---- ii: where the continuous score sits ----
        # concordance_freq only takes values k/100 and 0.05-spaced edges land
        # exactly on many of them; numpy then splits those genes by whichever side
        # floating point puts the edge on. Offset the edges by half the underlying
        # grid so no value sits on one — the last bin then cleanly isolates the
        # freq == 1.00 spike.
        edges = np.arange(-.005, 1.0 + bin_width - .004, bin_width)
        for tp in tps:
            v = full[(full.seed == sd) & (full.timepoint == tp)]["concordance_freq"]
            axb.hist(v, bins=edges, histtype="step", lw=lw, ls=lss[tp],
                     color=colors[tp], label=tp, zorder=3)
        axb.axvline(canon, color="0.45", ls="--", lw=1.0, zorder=1)
        axb.set_yscale("log")
        axb.set_xlim(*xlim)
        axb.set_xlabel(xlabels[1])
        axb.set_ylabel(ylabels[1])

        for ax in (axa, axb):
            ax.spines[["top", "right"]].set_visible(False)
            if legend:
                ax.legend(title=legend_title, frameon=False,
                          title_fontsize=fsize(fs - 1))
        if titles:
            for ax, t in zip((axa, axb), titles):
                ax.set_title(t, pad=8)
        # lay out BEFORE the letters, leaving headroom for them: they are figure text
        # above the titles, and nothing else keeps them on the canvas
        if margins:
            fig.subplots_adjust(**margins)
        else:
            fig.tight_layout(rect=(0, 0, 1, .90 if panel_letters else 1))
        _pl = panel_letters_for(panel_letters, 2)
        if _pl:
            align_titles(fig, (axa, axb))
            align_panel_letters(fig, (axa, axb), _pl, fontsize=fs + 2,
                                y=None, above_titles=True)
        save_fig(fig, save, f"figS_scg_threshold_{dataset}", dpi=dpi, verbose=verbose)
        if show:
            plt.show()

    table = scg_threshold_table(sw, cuts=cuts, verbose=False)
    if verbose:
        print(table.to_string(index=False))
    return fig, (axa, axb), table


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary — matching resolution, per-cell replication, ordered leave-out
# ═══════════════════════════════════════════════════════════════════════════
# (a) How fine does the expression matching have to be before the gap stops
#     moving? The memory median is fixed; only the matched background moves, so
#     the two curves converging as bins are added IS the sensitivity result.
#     Drawn for both LFC definitions, because the raw one is the pre-CPM version.
# (b) The same comparison recomputed inside each donor x stimulation cell —
#     matched within the cell, so nothing is pooled at any point.
# (d) The leave-out curve with the genes removed in order of their INDIVIDUAL
#     contribution, biggest first: the fastest possible way to destroy the
#     effect. Where it crosses the non-memory reference says how many genes the
#     result rests on.

MARSON_RAW_SUMMARY = f"{P.DATA}/marson/marson_gene_summary_real.csv"
BIN_GRID = (5, 10, 20, 30, 50, 75, 100)
# per-cell tables written by prepare_and_preprocess/marson/marson_per_cell.py
MARSON_CELL = P.DATA + "/marson/marson_gene_summary_cpm_{cell}.csv"
MARSON_CELL_EXPR = P.DATA + "/marson/marson_gene_ctrl_expr_{cell}.csv"


def _matching_supply_lineage(top_ns, min_methods=2, n_bins=50, cache=True,
                             verbose=True):
    """matching_supply_table for the FOUR LINEAGE datasets (figure 1d's comparison).

    Expression here is each lineage dataset's own mean CP10k, and the matcher is
    figure 1d's: 1:1 nearest neighbour on log expression (_panel_matched_bg), not the
    perturb panels' repeated binned draw. Same columns, so the same figure code works.
    """
    key = ("lineage_supply", tuple(top_ns), min_methods, n_bins)
    if cache and key in _CACHE:
        return _CACHE[key]
    rows, lab = [], f"\u2265{min_methods}"
    for top_n in top_ns:
        char = characterisation_table(top_n=top_n, tiers=(min_methods,), verbose=False)
        for ds in active_datasets():
            d = char[(char["dataset"] == ds) & (char["expression"] > 0)]
            mem = d.loc[d["group"] == lab, "expression"].to_numpy(float)
            bg = d.loc[d["group"] == "background", "expression"].to_numpy(float)
            if len(mem) < 2 or len(bg) < 2:
                continue
            f = pd.DataFrame({"expr": np.concatenate([mem, bg]),
                              "is_mem": np.r_[np.ones(len(mem), bool),
                                              np.zeros(len(bg), bool)]})
            f["bin"] = pd.qcut(f["expr"].rank(method="first"), n_bins, labels=False)
            g = f.groupby("bin")
            need = g.is_mem.sum()
            per = pd.DataFrame({"expr": g.expr.median(), "need": need,
                                "avail": g.size() - need})
            idx = _panel_matched_bg(np.log10(pd.Series(bg)), np.log10(pd.Series(mem)),
                                    k=10, seed=0)
            dr, lm = np.log10(bg[idx]), np.log10(mem)
            smd = (dr.mean() - lm.mean()) / np.sqrt((dr.var() + lm.var()) / 2)
            for b, r in per.iterrows():
                rows.append(dict(dataset=dlabel(ds), source="lineage",
                                 top_n=top_n, bin=int(b),
                                 expr=float(r.expr), need=int(r.need),
                                 avail=int(r.avail),
                                 ratio=(r.avail / r.need) if r.need else np.nan,
                                 n_mem=len(mem), n_bg=len(bg), smd_expr=float(smd),
                                 mem_expr=mem, bg_expr=bg))
        if verbose:
            print(f"done: lineage datasets (top {top_n})", flush=True)
    out = pd.DataFrame(rows)
    if cache:
        _CACHE[key] = out
    return out


def matching_supply_table(datasets=None, top_ns=(100, 300, 500), min_methods=2,
                          source="perturb", n_bins=50, n_iter=100, timepoint="merge",
                          merge_mode="union", methods=None, bg_mode="top_n",
                          bg_top_n=None, bg_exclude=BG_EXCLUDE_DEFAULT,
                          floor=EXPR_FLOOR, cache=True, verbose=True):
    """Is the expression-matched background actually SUPPLIABLE, bin by bin?

    The matcher (marson_KO.compare_gene_responsiveness_expr_matched) cuts the
    memory+background set into `n_bins` equal-count expression bins and samples, per
    bin, as many background genes as there are memory genes. If a bin holds fewer
    background genes than memory genes it falls back to `replace=True` and reuses
    genes -- silently. This table is the audit: per (dataset, top_n, bin) it reports
    how many memory genes NEED matching and how many background genes are AVAILABLE.

    ratio < 1 is the failure mode; the achieved |SMD| in log expression rides along
    so one table answers both "are the genes similar" and "are there enough of them".

    Built exactly like xd_consensus_sweep (same universe, floor, consensus tier and
    background rule), so it describes the matching the figures actually perform.

    source : 'perturb' -- the six perturb-seq screen x lineage pairs, in each
        screen's own control expression (the matching behind figures 2d-2h).
        'lineage' -- the four lineage datasets in their OWN expression, which is what
        figure 1d compares; use this when the question is about figure 1d.

    Returns a long frame: dataset, top_n, bin, expr, need, avail, ratio, n_mem,
    n_bg, smd_expr, plus `mem_expr`/`bg_expr` objects carrying the per-gene values
    so a cumulative view can be drawn without recomputing.
    """
    if source == "lineage":
        return _matching_supply_lineage(top_ns, min_methods, n_bins, cache, verbose)
    _perturb_path()
    from marson_KO import build_consensus_gene_sets
    datasets = datasets if datasets is not None else xd_datasets()
    top_ns = list(top_ns)
    key = ("xd_supply", tuple(d["name"] for d in datasets), tuple(top_ns), min_methods,
           n_bins, n_iter, timepoint, merge_mode, bg_mode, bg_top_n,
           tuple(bg_exclude or ()), floor)
    if cache and key in _CACHE:
        return _CACHE[key]

    rows = []
    for ds in datasets:
        gs = pd.read_csv(ds["gs"], index_col=0)
        expr = pd.read_csv(ds["expr"], index_col=0).iloc[:, 0]
        if floor is not None:                      # the one expression rule
            keep = expr.index[expr > floor]
            gs = gs.loc[gs.index.intersection(keep)]
            expr = expr.loc[expr.index.intersection(gs.index)]
        tp = ds.get("timepoint", timepoint)
        mm = ds.get("merge_mode", merge_mode)
        universe, to_data = _xd_symbol_bridge(ds, gs)
        want = methods if methods is not None else XD_METHODS_CONSENSUS
        files = {m: f for m, f in ds["files"].items() if m in want}
        for top_n in top_ns:
            _, counts = build_consensus_gene_sets(
                min_methods=1, timepoint=tp, top_n=top_n, merge_mode=mm,
                memgene_dir=ds["memdir"], method_files=files, universe=universe,
                verbose=False)
            bg = xd_background(ds, universe, to_data, tp, mm, bg_mode,
                               (bg_top_n or top_n) if bg_mode == "top_n" else top_n,
                               files=files, bg_exclude=bg_exclude, verbose=False)
            bgset = set(bg)
            mem = [g for g in to_data(counts.index[counts >= min_methods].tolist())
                   if g not in bgset]
            if len(mem) < 2 or len(bg) < 2:
                continue
            d = pd.DataFrame({"expr": expr.reindex(list(mem) + list(bg)).values,
                              "is_mem": [True] * len(mem) + [False] * len(bg)}).dropna()
            # the matcher's own binning: equal-count bins of memory+background, ranked
            # first so ties cannot pile into one bin
            d["bin"] = pd.qcut(d["expr"].rank(method="first"), n_bins, labels=False)
            g = d.groupby("bin")
            need = g.is_mem.sum()
            per = pd.DataFrame({"expr": g.expr.median(), "need": need,
                                "avail": g.size() - need})
            frame = gs.loc[list(dict.fromkeys(mem + bg))]
            bal = expression_balance(frame, expr, mem, n_bins=n_bins, n_iter=n_iter,
                                     metric=METRIC)
            mem_e = d.loc[d.is_mem, "expr"].to_numpy(float)
            bg_e = d.loc[~d.is_mem, "expr"].to_numpy(float)
            for b, r in per.iterrows():
                rows.append(dict(dataset=ds["name"], source="perturb",
                                 top_n=top_n, bin=int(b),
                                 mem_expr=mem_e, bg_expr=bg_e,
                                 expr=float(r.expr), need=int(r.need),
                                 avail=int(r.avail),
                                 ratio=(r.avail / r.need) if r.need else np.nan,
                                 n_mem=len(mem), n_bg=len(bg),
                                 smd_expr=bal["smd_expr"]))
        if verbose:
            print(f"done: {ds['name'].splitlines()[-1]}", flush=True)
    out = pd.DataFrame(rows)
    if cache:
        _CACHE[key] = out
    return out


def fig_matching_supply(table=None, datasets=None, top_ns=(100, 300, 500),
                        kind="counts", colors=None, mem_colors=None, hist_bins=30,
                        ncol=3, panel_w=2.0, panel_h=1.7,
                        fs=7, ref_top_n=300, annot_smd=True, xlabel=None, ylabel=None,
                        sharey=True, headroom=4.0, save=None, dpi=400, show=True,
                        verbose=True,
                        **kw):
    """(Suppl. Fig 3a) One panel per screen: is the matched background suppliable?

    The matcher sorts memory + candidate background genes by expression, cuts them
    into equal-count bins, and draws per bin as many background genes as there are
    memory genes. Too few background genes in a bin and it reuses them silently.

    kind='counts' (default) -- per expression bin, background genes AVAILABLE (grey)
        against memory genes NEEDING a match (red). Grey above red everywhere is the
        claim: every memory gene can take a distinct background gene of its own
        expression. Drawn at `ref_top_n`. This is the readable one.
    kind='ratio'  -- available / needed, one line per cutoff, with a dashed line at 1
        (below it the sampler reuses genes). Denser, and shows every cutoff at once,
        but the y axis is an abstraction rather than a gene count.

    Both annotate each panel with the worst bin and the achieved |SMD| at ref_top_n.
    """
    if table is None:
        table = matching_supply_table(datasets=datasets, top_ns=top_ns,
                                      verbose=verbose, **kw)
    tns = [t for t in top_ns if t in set(table["top_n"])]
    if colors is None:                      # light -> dark with depth, C_MEM in the middle
        cmap = plt.get_cmap("Reds")
        colors = {t: cmap(.35 + .5 * i / max(len(tns) - 1, 1)) for i, t in enumerate(tns)}
        if ref_top_n in colors:
            colors[ref_top_n] = C_MEM
    names = [n for n in dict.fromkeys(table["dataset"])]
    # memory bars take the DATASET colour where there is one (figure 1d's palette, so
    # the two figures can be read together); C_MEM otherwise
    _dcol = {dlabel(k): v for k, v in DATASET_COLORS.items()}

    def _mcol(nm):
        """Sunburst colour of the LINEAGE dataset the memory genes came from.

        Cross-dataset rows are named '<lineage> - <screen>', so the palette (keyed by
        lineage label) has to be looked up on the head of the name; a lineage-only
        table matches whole. Same convention as figure 2b: colour = where the memory
        genes were called, not which screen they were tested in.
        """
        # a single colour (string or RGBA tuple) applies to every panel: one red for
        # memory genes everywhere, the dataset palette only when a dict is passed
        if mem_colors is not None and not isinstance(mem_colors, dict):
            return mem_colors
        if mem_colors:
            return mem_colors.get(nm, C_MEM)
        t = dlabel_pair(nm)
        if t in _dcol:
            return _dcol[t]
        return _dcol.get(re.split(r"[\u2013\u2014-]", t)[0].strip(), C_MEM)

    # the shared legend is built from the FIRST panel, so when the memory bars differ
    # in colour between panels its swatch would claim one dataset's colour means
    # "memory" everywhere; say so only when they actually differ
    _mem_label = "memory genes needing a match"
    if len({_mcol(n) for n in names}) > 1:
        _mem_label += "\n(each dataset's colour)"

    nrow = int(np.ceil(len(names) / ncol))
    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(fs - 1), "axes.linewidth": .9}
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                                 squeeze=False, sharey=sharey)
        for i, name in enumerate(names):
            ax = axes[i // ncol][i % ncol]
            sub = table[table["dataset"] == name]
            ax.set_xscale("log"); ax.set_yscale("log")
            if kind == "hist":
                # plain overlapping histograms in real expression units. Counts differ
                # by ~70x (300 memory genes vs ~20k background), so the y axis is log;
                # on a linear one the memory bars are invisible.
                u = sub[sub["top_n"] == ref_top_n]
                me = np.asarray(u["mem_expr"].iloc[0], float)
                be = np.asarray(u["bg_expr"].iloc[0], float)
                lo, hi = min(me.min(), be.min()), max(me.max(), be.max())
                edges = np.logspace(np.log10(lo), np.log10(hi), hist_bins + 1)
                ax.hist(be, bins=edges, color=C_BG, lw=0, zorder=2,
                        label="background genes available")
                # the shared legend is built from the first panel, so its swatch would
                # claim one dataset's colour means "memory" in all of them
                ax.hist(me, bins=edges, color=_mcol(name), lw=0, zorder=3,
                        label=_mem_label)
                ax.set_xscale("log"); ax.set_yscale("log")
            elif kind == "cumulative":
                # genes AT OR ABOVE each expression: the tail question asked directly,
                # with no binning to hide behind. Grey above red at every x means that
                # wherever you cut, candidates outnumber the genes needing a match.
                u = sub[sub["top_n"] == ref_top_n]
                me = np.sort(np.asarray(u["mem_expr"].iloc[0], float))
                be = np.sort(np.asarray(u["bg_expr"].iloc[0], float))
                ax.step(be, np.arange(len(be), 0, -1), where="post", color=C_BG,
                        lw=1.2, zorder=3, label="background genes available")
                ax.step(me, np.arange(len(me), 0, -1), where="post", color=C_MEM,
                        lw=1.2, zorder=4, label="memory genes needing a match")
            elif kind == "counts":
                u = sub[sub["top_n"] == ref_top_n].sort_values("expr")
                ax.fill_between(u["expr"], .5, u["avail"], step="mid",
                                color=C_BG, alpha=.25, lw=0, zorder=1)
                ax.step(u["expr"], u["avail"], where="mid", color=C_BG, lw=1.1,
                        zorder=3, label="background genes available")
                # clipped so a bin with 0 memory genes stays on a log axis
                ax.step(u["expr"], u["need"].clip(lower=.5), where="mid",
                        color=C_MEM, lw=1.1, zorder=4,
                        label="memory genes needing a match")
            else:
                for t in tns:
                    u = sub[(sub["top_n"] == t) & (sub["need"] > 0)].sort_values("expr")
                    if len(u):
                        ax.plot(u["expr"], u["ratio"], color=colors[t], lw=1.0,
                                label=f"top {t}", zorder=3)
                ax.axhline(1, ls="--", lw=.8, color="0.3", zorder=2)
            ref = sub[(sub["top_n"] == ref_top_n) & (sub["need"] > 0)]
            if len(ref) and kind in ("cumulative", "hist"):
                me = np.asarray(ref["mem_expr"].iloc[0], float)
                be = np.asarray(ref["bg_expr"].iloc[0], float)
                p95 = np.quantile(me, .95)
                nm, nb = int((me > p95).sum()), int((be > p95).sum())
                bits = ["top 5% of memory genes:", f"{nb} candidates for {nm} genes"]
                if annot_smd:
                    bits.append(f"|SMD| {abs(ref['smd_expr'].iloc[0]):.3f}")
                # bars fill the bottom-left in 'hist'; the curves fill the top-left
                # in 'cumulative'
                _y, _va = ((.97, "top") if kind == "hist" else (.03, "bottom"))
                ax.text(.03, _y, "\n".join(bits), transform=ax.transAxes, va=_va,
                        ha="left", fontsize=fsize(fs - 2), color="0.25",
                        linespacing=1.15)
            elif len(ref):
                bits = [f"worst {ref['ratio'].min():.1f}x"]
                if annot_smd:
                    bits.append(f"|SMD| {abs(ref['smd_expr'].iloc[0]):.3f}")
                # top-left: the dashed 1x line lives at the bottom of every panel
                ax.text(.03, .97, "\n".join(bits), transform=ax.transAxes, va="top",
                        ha="left", fontsize=fsize(fs - 2), color="0.25",
                        linespacing=1.15)
            ax.set_title(dlabel_pair(name), fontsize=fs, pad=3)
            ax.spines[["top", "right"]].set_visible(False)
        for j in range(len(names), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        # headroom so the corner annotation and the legend clear the curves, which
        # run right up to the top of every panel
        # with sharey the axes share ONE ylim, so scaling each in turn compounds
        # (4x per panel -> 4096x over six). Scale once.
        if headroom and headroom != 1:
            axs = [axes[i // ncol][i % ncol] for i in range(len(names))]
            for ax in (axs[:1] if sharey else axs):
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi * headroom)
        _yl = {"counts": "genes in each expression bin",
               "cumulative": "genes at or above this expression",
               "hist": "number of genes",
               "ratio": "background genes available\nper memory gene needed"}
        fig.supylabel(ylabel or _yl.get(kind, ""), fontsize=fs)
        # x label and legend are placed AFTER tight_layout, relative to the axes.
        # fig.supxlabel is reserved space by tight_layout on TOP of any rect, which
        # left 0.85in of blank band under a 3.2in figure.
        fig.tight_layout(rect=(.02, .02, 1, 1))
        y0 = min(axes[i // ncol][i % ncol].get_position().y0 for i in range(len(names)))
        dy = .38 / fig.get_figheight()          # ~0.38in from the axes to the label
        # lineage datasets are measured in their own cells, not perturb-seq controls
        _src = (table["source"].iloc[0] if "source" in table.columns else "perturb")
        fig.text(.5, y0 - dy, xlabel or ("mean expression (CP10k)" if _src == "lineage"
                                         else "mean control expression (CPM)"),
                 ha="center", va="top", fontsize=fs)
        # shared legend: in 'counts' mode the grey band fills every panel, so an
        # in-axes legend lands on the data whatever corner it picks
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=fsize(fs - 1), ncol=len(l),
                   loc="upper center", bbox_to_anchor=(.5, y0 - dy - .06))
        save_fig(fig, save, "matching_supply", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, axes, table


def dlabel_pair(name):
    """Two-line cross-dataset row name -> one short title, mathtext stripped.

    In LABEL_MODE == "celltype" a lineage–screen pair collapses to the screen's
    own label: both halves profile the same cell type, so 'T cells – T cells
    (Marson)' would say it twice.
    """
    t = re.sub(r"\$|\\mathbf|\{|\}", "", str(name)).split("\n")[0].strip()
    if LABEL_MODE == "celltype":
        parts = [p.strip() for p in re.split(r"\s[–-]\s", t)]
        if len(parts) == 2 and parts[1] in PERTURB_CELLTYPE:
            return slabel(parts[1])
        if parts and parts[0] in DATASET_LABELS.values():
            return dlabel(parts[0])
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary — benchmark depth, cutoff sensitivity, ortholog mapping
# ═══════════════════════════════════════════════════════════════════════════
# The three panels behind Figure 1's parameter choices:
#   ranking depth   how many genes each method can even supply -> why TOP_N
#   depth sweep     agreement and cross-dataset overlap as the cutoff moves
#   orthologs       what the mouse->human step costs, gene by gene
# LARRY is deliberately absent from all three (see DATASETS).

def ranking_depth_table(datasets=None, methods=None, tree=RESULTS_TREE,
                        scg_variant=None, verbose=True):
    """How many genes each method actually ranks, per dataset and group.

    Two ceilings, and they are NOT the same kind of thing:
      * Power-Seek emits a FIXED number of genes (500) by construction —
        their output size is a parameter, not a discovery.
      * CORAL, GEMLI and SCG emit however many genes pass their own rule, so their
        depth varies with the dataset (SCG: 329 at Michaels T2, 13,669 at Harmange E).
    Any cutoff above the smallest of these is not comparing methods at equal depth,
    because the shallow ones have simply stopped changing.

    `scope` is 'group' for a single timepoint/group and 'pooled' for the union
    list — the pooled cap is the honest limit whenever the analysis pools.

    Returns the long table (dataset, method, group, scope, n_genes).
    """
    rows = []
    for ds in active_datasets(datasets):
        tabs, tps, suf = load_tables(ds, methods=methods, tree=tree,
                                     scg_variant=scg_variant_for(ds, scg_variant),
                                     verbose=False)
        for m, t in tabs.items():
            for tp in tps:
                rows.append(dict(dataset=ds, method=m, group=str(tp), scope="group",
                                 n_genes=int((t["timepoint"] == tp).sum())))
            rows.append(dict(dataset=ds, method=m, group="pooled", scope="pooled",
                             n_genes=len(pooled_genes(t, tps, 10 ** 9))))
        if verbose:
            g = [r["n_genes"] for r in rows if r["dataset"] == ds and r["scope"] == "group"]
            p = [r["n_genes"] for r in rows if r["dataset"] == ds and r["scope"] == "pooled"]
            print(f"  {dlabel(ds):11s} ({suf}) per-group cap {min(g)}, pooled cap {min(p)}")
    return pd.DataFrame(rows)


def fig_ranking_depth(table=None, datasets=None, cutoffs=((TOP_N, "top 100"), (300, "top 300")),
                      colors=None, hatches=None, hatch_color="white", hatch_lw=.7,
                      hatch_edge_lw=0.0, annot_color=None,
                      pooled_marker="D", pooled_color="0.35",
                      pooled_label="pooled", pooled_legend="all",
                      group_labels=None,
                      logy=True, annotate=True, bar_width=.38, headroom=45,
                      ncol=2, celltype=True, fs=14, panel_w=4.6, panel_h=3.6,
                      h_pad=3.2, legend_ncol=None, margins=None,
                      legend_fs=None, legend_handlelength=1.1,
                      legend_handleheight=None,
                      panel_letters="roman", titles=None,
                      ylabel=None, legend_loc="upper center",
                      save=None, dpi=400, show=True,
                      verbose=True, **kw):
    """(Suppl. Fig 1c) What each method can actually supply, and where the cutoffs sit.

    One panel per dataset on an `ncol`-wide grid, one bar per method and group,
    with the pooled list as an open marker. The cutoff lines are the point: every
    analysis cut has to lie below the shortest bar, or the shallow methods are
    silently maxed out and the comparison is no longer at equal depth.

    Returns (fig, axes, table).
    """
    if table is None:
        table = ranking_depth_table(datasets=datasets, verbose=verbose, **kw)
    dss = in_order(dict.fromkeys(table.dataset), DATASET_ORDER)
    methods = in_order(dict.fromkeys(table.method), METHOD_ORDER)
    cutoffs = [(c, f"top {c}") if not isinstance(c, (tuple, list)) else tuple(c)
               for c in (cutoffs or [])]
    n = len(dss)
    ncol = int(ncol or n)
    nrow = int(np.ceil(n / ncol))

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fsize(fs + 1),
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fsize(fs - 1),
             "legend.fontsize": fsize(legend_fs or fs - 3), "axes.linewidth": .9}
    with plt.rc_context(style):
        fig, grid = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, panel_h * nrow),
                                 sharey=True, squeeze=False)
        flat = list(grid.ravel())
        for ax in flat[n:]:
            ax.set_visible(False)
        axes = flat[:n]
        for i, (ax, ds) in enumerate(zip(axes, dss)):
            d = table[table.dataset == ds]
            groups = [g for g in dict.fromkeys(d[d.scope == "group"].group)]
            # colors: a {group: colour} map, or a LIST applied by position within the
            # dataset, which is what lets one spec cover datasets with different
            # group names (as figure 1c does)
            cols = ({g: colors[j % len(colors)] for j, g in enumerate(groups)}
                    if isinstance(colors, (list, tuple)) else
                    colors or {g: (C_TPS + [f"C{k}" for k in range(len(groups))])[j]
                               for j, g in enumerate(groups)})
            # group names: spelled out via glabel() unless `group_labels` overrides
            # them for this figure only (the long forms collide in a narrow panel).
            # Keys may be (dataset, group), group, or {dataset: {group: label}}.
            def _glab(g, ds=ds):
                if isinstance(group_labels, dict):
                    for k in ((ds, g), (dlabel(ds), g), g):
                        if k in group_labels:
                            return group_labels[k]
                    sub = group_labels.get(ds, group_labels.get(dlabel(ds)))
                    if isinstance(sub, dict) and g in sub:
                        return sub[g]
                return glabel(ds, g)
            x = np.arange(len(methods))
            # the method slot is fixed, so three groups get narrower bars than two —
            # a fixed width per bar spills into the neighbouring method
            bw = min(bar_width, .8 / max(len(groups), 1))
            off = (np.arange(len(groups)) - (len(groups) - 1) / 2) * bw
            for gi, g in enumerate(groups):
                v = [float(d[(d.method == m) & (d.group == g)].n_genes.iloc[0])
                     if len(d[(d.method == m) & (d.group == g)]) else np.nan
                     for m in methods]
                # one colour plus a texture per group, as in figure 1c: the groups of
                # a dataset are the same measurement, and the distinction survives
                # greyscale. A hatch takes its colour from the EDGE.
                _hat = hatches[gi % len(hatches)] if hatches else None
                ax.bar(x + off[gi], v, width=bw, color=cols[g], label=_glab(g),
                       hatch=_hat,
                       edgecolor=(hatch_color if _hat else "white"),
                       lw=(hatch_edge_lw if _hat else .5), zorder=3)
                if annotate:
                    for xi, vi in zip(x + off[gi], v):
                        if np.isfinite(vi):
                            ax.annotate(f"{vi:.0f}", (xi, vi), xytext=(0, 2),
                                        textcoords="offset points", ha="center",
                                        va="bottom", rotation=90,
                                        fontsize=fsize(fs - 4),
                                        color=annot_color or "0.25")
            pooled = [float(d[(d.method == m) & (d.scope == "pooled")].n_genes.iloc[0])
                      if len(d[(d.method == m) & (d.scope == "pooled")]) else np.nan
                      for m in methods]
            # "pooled" means the same thing in every panel, so repeating the key
            # four times only costs legend width: pooled_legend="first" keeps it
            # once, None drops it to the caption.
            _pool_key = (pooled_legend == "all"
                         or (pooled_legend == "first" and i == 0))
            ax.scatter(x, pooled, marker=pooled_marker, s=30, facecolor="none",
                       edgecolor=pooled_color, linewidth=1.1, zorder=4,
                       label=(pooled_label if _pool_key else "_nolegend_"))
            last_in_row = (i % ncol == ncol - 1) or (i == n - 1)
            for c, lab in cutoffs:
                ax.axhline(c, color="0.45", ls="--", lw=1.0, zorder=1)
                if last_in_row:
                    ax.annotate(lab, (len(methods) - .4, c), xytext=(3, 2),
                                textcoords="offset points", ha="left", va="bottom",
                                fontsize=fsize(fs - 3), color="0.45",
                                annotation_clip=False)
            if logy:
                ax.set_yscale("log")
            # the legend and the rotated counts share the space above the bars, so
            # the headroom is what keeps them off each other
            lo = min([c for c, _ in cutoffs] + [table.n_genes.min()]) * .45
            ax.set_ylim(lo, table.n_genes.max() * headroom)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_title(dlabel(ds, celltype=celltype) if titles is None
                         else titles[i], pad=8)
            ax.spines[["top", "right"]].set_visible(False)
            # one horizontal line at the top: it needs far less vertical room than
            # a stacked legend, which is what collides with the rotated counts
            # a hatched key needs a bigger swatch than a plain colour one
            _hh = ({} if legend_handleheight is None
                   else dict(handleheight=legend_handleheight))
            ax.legend(loc=legend_loc,
                      ncol=legend_ncol or len(groups) + int(bool(_pool_key)),
                      frameon=False,
                      handlelength=legend_handlelength, columnspacing=1.1,
                      labelspacing=.3, **_hh)
            if i % ncol == 0:
                ax.set_ylabel(ylabel or "genes ranked by the method")
        # tight_layout BEFORE the letters: it re-lays out the axes, and the titles
        # and letters are figure-coordinate text that would be left behind.
        # h_pad opens the gap between rows so the lower row's letters and titles do
        # not land on the upper row's rotated tick labels.
        # margins= fixes the panel boxes (subplots_adjust kwargs): two figures with the
        # same width, column count and margins get identical columns
        if margins:
            fig.subplots_adjust(**margins)
        else:
            fig.tight_layout(h_pad=h_pad)
        _pl = panel_letters_for(panel_letters, n)
        if _pl:
            # one baseline PER ROW — a single baseline across a grid would drag the
            # lower rows' titles up to the top row's
            for r in range(nrow):
                row = axes[r * ncol:(r + 1) * ncol]
                if not row:
                    continue
                align_titles(fig, row)
                align_panel_letters(fig, row, _pl[r * ncol:(r + 1) * ncol],
                                    fontsize=fs + 2, y=None, above_titles=True)
        save_fig(fig, save, "figS_ranking_depth", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    if verbose:
        piv = (table[table.scope == "group"]
               .pivot_table(index="method", columns=["dataset", "group"],
                            values="n_genes"))
        print(piv.to_string(float_format=lambda x: f"{x:.0f}"))
    return fig, axes, table


def ortholog_table(datasets=None, verbose=True):
    """What the mouse->human step costs, dataset by dataset.

    Columns: genes measured; genes with a human orthologue; DISTINCT human symbols
    they collapse to; how many human symbols receive more than one mouse gene
    (many-to-one collapse — the real loss here); how many mouse genes map to more
    than one human symbol (ambiguous, and silently resolved by dict order in
    ortho_map, so it must be reported as zero or handled); and how many survive
    into the universe shared by every dataset.

    Human datasets pass through unmapped and are included so the shared universe
    is attributable — it is an intersection, so a human dataset can be the binding
    constraint just as easily as the mapping is.
    """
    labels = active_datasets(datasets)
    unis, rows = {}, []
    for ds in labels:
        var = _read_var(ds)
        hu = to_human(var, ds, verbose=False)
        unis[ds] = hu
        r = dict(dataset=ds, species=DATASETS[ds]["species"], measured=len(var),
                 human_symbols=len(hu))
        if DATASETS[ds]["species"] == "mouse":
            p = DATASETS[ds].get("ortho") or ORTHO_FALLBACK
            o = pd.read_csv(p)
            om = o.dropna(subset=["human"])
            r.update(in_cache=len(o), mapped=len(om),
                     mapped_pct=100 * len(om) / max(len(o), 1),
                     many_to_one=int((om.groupby("human")["mouse"].nunique() > 1).sum()),
                     ambiguous=int((om.groupby("mouse")["human"].nunique() > 1).sum()))
        else:
            r.update(in_cache=np.nan, mapped=np.nan, mapped_pct=100.0,
                     many_to_one=0, ambiguous=0)
        rows.append(r)
    shared = set.intersection(*unis.values())
    T = pd.DataFrame(rows)
    T["in_shared_universe"] = [len(unis[d] & shared) for d in labels]
    T["kept_pct"] = 100 * T["in_shared_universe"] / T["measured"]
    T.attrs["shared_universe"] = len(shared)
    if verbose:
        print(T.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        print(f"shared universe (measured in all {len(labels)} datasets): {len(shared)} genes")
    return T


def fig_ortholog_mapping(T=None, datasets=None,
                         seg_colors=(C_OBS, "#7f9bb5", "#dfe3e8"),
                         seg_labels=("measured in every dataset",
                                     "human symbol, measured in only some",
                                     "no human orthologue"),
                         celltype=False, bar_h=.6, annot=True, min_frac=.08,
                         ref_line=True, fs=14, figsize=(9.0, 3.9), ax=None,
                         xlabel=None, save=None, dpi=400, show=True, verbose=True,
                         **kw):
    """(S) What the mouse->human step costs, and what the shared universe leaves.

    Each dataset's measured transcriptome, split into what survives to the shared
    gene space and what does not. The dark segment is IDENTICAL in every bar by
    construction — the shared universe is an intersection, so every dataset
    contains all of it — which is exactly the point: the dashed line is the gene
    space every cross-dataset number is computed over, and it is the same space
    for all four. A gene outside it was never testable, so 'not shared' can never
    quietly mean 'not measured'.

    For the mouse datasets the light segment is the orthology loss; the mapped
    percentage is printed with the total. The many-to-one collapse (several mouse
    genes onto one human symbol) is in the table, not the figure — it is a caveat
    about the mapping, not a quantity on this axis.

    Returns (fig, ax, T).
    """
    if T is None:
        T = ortholog_table(datasets=datasets, verbose=verbose, **kw)
    labels = list(T["dataset"])
    y = np.arange(len(labels))[::-1]
    shared = int(T.attrs.get("shared_universe", T["in_shared_universe"].max()))
    seg = np.column_stack([
        T["in_shared_universe"].to_numpy(float),
        (T["human_symbols"] - T["in_shared_universe"]).to_numpy(float),
        (T["measured"] - T["human_symbols"]).to_numpy(float)])

    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fsize(fs + 1),
             "xtick.labelsize": fsize(fs - 1), "ytick.labelsize": fs,
             "legend.fontsize": fsize(fs - 2), "axes.linewidth": .9}
    with plt.rc_context(style):
        own = ax is None
        fig, ax = (plt.subplots(figsize=figsize) if own else (ax.get_figure(), ax))
        xmax = float(T["measured"].max())
        left = np.zeros(len(labels))
        for j, (c, lab) in enumerate(zip(seg_colors, seg_labels)):
            ax.barh(y, seg[:, j], left=left, height=bar_h, color=c, label=lab,
                    edgecolor="white", linewidth=.7, zorder=3)
            if annot:
                for yi, (w, l) in enumerate(zip(seg[:, j], left)):
                    if w / xmax >= min_frac:
                        ax.text(l + w / 2, y[yi], f"{w:,.0f}".replace(",", " "),
                                ha="center", va="center",
                                fontsize=fsize(fs - 3), zorder=4,
                                color="white" if j == 0 else "0.25")
            left += seg[:, j]
        if ref_line:
            ax.axvline(shared, color="0.25", ls="--", lw=1.0, zorder=5)
            ax.annotate(f"shared universe   {shared:,}".replace(",", " "),
                        (shared, y.max() + bar_h / 2), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=fsize(fs - 2), color="0.25")
        if annot:
            for yi, r in zip(y, T.itertuples()):
                note = (f"{r.measured:,}".replace(",", " ") +
                        (f"   {100 * r.human_symbols / r.measured:.0f}% mapped"
                         if r.species == "mouse" else ""))
                ax.annotate(note, (r.measured, yi), xytext=(6, 0),
                            textcoords="offset points", va="center",
                            fontsize=fsize(fs - 2), color="0.35")
        ax.set_yticks(y)
        ax.set_yticklabels([dlabel(d, celltype=celltype) for d in labels])
        ax.set_xlim(0, xmax * 1.28)          # room for the totals at the bar ends
        ax.set_ylim(-.7, y.max() + .9)
        ax.set_xlabel(xlabel or "genes")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v/1000:g}k" if v else "0"))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3,
                  frameon=False, handlelength=1.1, columnspacing=1.6)
        if own:
            fig.tight_layout()
        save_fig(fig, save, "figS_ortholog_mapping", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, ax, T


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary — the SCG definition: call threshold and clone-size window
# ═══════════════════════════════════════════════════════════════════════════
# Two choices define an SCG call, and both are shown here rather than asserted:
#   * the concordance-frequency threshold (is_biased == concordance_freq >= 0.80,
#     verified exact in every result table), and
#   * the clone-size window: 2 (true sister pairs, one pair per clone) or 2–8 (all
#     within-clone pairs; the primary definition since 2026-09-02).
# The window figure shows INDEPENDENT CLONES next to within-clone PAIRS on purpose:
# 2–8 multiplies pairs far more than clones, the pairs are pseudo-replicated, and a
# reader has to be able to see that the effective sample is the clone count.

SCG_WINDOWS = {"2–8": "clones2-8_", "2": ""}          # display -> file infix, primary first
SCG_WINDOW_STYLE = {"2–8": dict(ls="-", lw=2.0), "2": dict(ls=(0, (4, 2)), lw=1.6)}
SCG_WINDOW_COLORS = {"2–8": C_OBS, "2": "#E6B8A2"}
GROUP_SHADES = [C_OBS, C_OBS2, "#D9A441"]             # 1st/2nd/3rd group of a dataset
SCG_THRESHOLDS = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2))


def fig_ortholog_and_overlap(ortho=None, figsize=(18.4, 4.2), width_ratios=(14.6, 3.4),
                             fs=14, margins=None, panel_letters="roman",
                             ortho_kw=None, cross_kw=None,
                             save=None, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 1e, 1f) One row: what the mouse->human step leaves (left) and how much each pair
    of datasets shares (right). Draws fig_ortholog_mapping and the matrix half of
    fig_cross_dataset into one figure, so the row is a single file at page width.

    ortho_kw / cross_kw are forwarded to the two panel functions (e.g. cross_kw=
    dict(n_perm=1000, pair_universe=True)). Returns (fig, (ax_left, ax_right),
    (ortho_table, cross_result)).
    """
    ortho_kw, cross_kw = dict(ortho_kw or {}), dict(cross_kw or {})
    # the matrix slot is sized to the heatmap's natural (square) width: the colorbar
    # attaches to the slot's right edge, so a wider slot strands it at the margin
    margins = margins or dict(left=.065, right=.945, bottom=.27, top=.86, wspace=.20)
    # tick sizes are fixed when an axes is CREATED, and these axes are created here,
    # not inside the panel functions — so the full font set has to be set here too
    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fsize(fs + 1), "xtick.labelsize": fsize(fs - 1),
             "ytick.labelsize": fs, "legend.fontsize": fsize(fs - 2),
             "axes.linewidth": .9}
    with plt.rc_context(style):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios), **margins)
        ax_o, ax_m = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        _, _, ortho = fig_ortholog_mapping(T=ortho, ax=ax_o, fs=fs, save=None,
                                           show=False, verbose=verbose, **ortho_kw)
        cross_kw = {**dict(min_methods=2, top_n=300, pooled_depth=True,
                           thresholds=(2, 3, 4), null_mode="matched", n_perm=1000,
                           seed=0, logy=False, matrix_metric="overlap_coef",
                           matrix_annot="n", cbar_ticks=(0, .1, .2)), **cross_kw}
        _, res = fig_cross_dataset(panel="matrix", axes=(None, ax_m), fs=fs,
                                   save=None, show=False, verbose=verbose, **cross_kw)
        # an equal-aspect heatmap cannot fill a wide slot; pin it to the left edge of
        # its slot so the leftover space sits at the page margin, not between panels
        ax_m.set_anchor("W")
        _pl = panel_letters_for(panel_letters, 2)
        if _pl:
            align_panel_letters(fig, (ax_o, ax_m), _pl, fontsize=fs + 2,
                                above_titles=False, dy=.03)
        save_fig(fig, save, "figS_ortholog_and_overlap", dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, (ax_o, ax_m), (ortho, res)


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary — memory genes from perturb-seq CONTROL cells vs lineage barcodes
# ═══════════════════════════════════════════════════════════════════════════
# Built for perturb-seq/perturb_control_mgs.ipynb "FINAL 3": per perturb-seq screen,
# each lineage-free method (Power-Seek, GEMLI) is scored twice on the SAME knockout
# readout — once on memory genes it found in that screen's control cells, once on
# memory genes it found in the cell-type-matched barcoded dataset — plus the genes
# both methods call on the control cells. The control-cell sets are what the figure
# is about, so they are the ones a reader's eye is sent to: shaded slot, bold label.

C_ROUTE_SHADE = "0.925"


def _route_of(key):
    """(group, route, lineage, n) parsed from a FINAL-3 gene-set label."""
    import re
    lin = re.search(r"\(([^)=]+)\)", key)
    n = re.search(r"n\s*=\s*(\d+)", key)
    if key.lower().startswith("shared"):
        return "both methods", "shared", None, int(n.group(1)) if n else None
    route = "controls" if "perturb-seq controls" in key else "barcode"
    return key.split("\n")[0], route, (lin.group(1) if lin else None), None


def fig_control_vs_barcode_violins(panels, ylim=(0, 1.0), fs=7, figsize=(7.09, 8.4),
                                   median_weight="bold", ytick_fs=None,
                                   pct_where="left", col_title_pad=None,
                                   panel_letters="roman", shade=None,
                                   col_order=("Power-Seek", "GEMLI", "both methods"),
                                   col_titles=None, h_pad=1.4, w_pad=.6, lead=.35,
                                   ylabel=None, legend=True,
                                   save=None, dpi=400, show=True, verbose=True):
    """(S) Knockout responsiveness of memory genes found WITHOUT lineage barcodes.

    panels : {screen title: {gene-set label: expression-matched result}} — the
             PANELS3 dict of FINAL 3, labels as written there.

    A grid: one ROW per perturb-seq screen, one COLUMN per method — Power-Seek and
    GEMLI each show their control-cell set next to their lineage-barcode set; the
    last column is the genes both methods call on the control cells. Control-cell
    sets are shaded and bold. Per violin, left half = the gene set, right half = its
    expression-matched non-memory background; "+x%" is the median difference.

    Stars stay BH-adjusted across ALL gene sets of a screen (five tests per row), as
    in the single-axis version: the q-values are computed per row first and passed
    to each subplot, so splitting a row into columns does not change the family.
    One y range everywhere. Returns (fig, axes[rows][cols], stats).
    """
    from scipy.stats import mannwhitneyu
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    screens = list(panels)
    cols = list(col_order)
    col_titles = col_titles or {"both methods": "both methods"}
    # slots per column follow the data (2, 2, 1), so every violin gets the same width
    per_col = {c: max(sum(_route_of(k)[0] == c for k in panels[t]) for t in screens)
               + lead for c in cols}     # + the left lead, so violin widths stay equal
    V = dict(annot="pct+stars", show_medians=False, fill=False, median_inside=True,
             pct_where=pct_where, ylim=ylim, legend=False, fs=fs, show=False,
             verbose=False, xlabel="", ylabel="", width=.74,
             median_weight=median_weight)
    style = {**PUB_RC, "font.size": fs, "axes.labelsize": fs,
             "axes.titlesize": fsize(fs + 1), "xtick.labelsize": fs,
             "ytick.labelsize": ytick_fs or fs, "legend.fontsize": fs}
    rows_out = []
    with plt.rc_context(style):
        fig, grid = plt.subplots(len(screens), len(cols), figsize=figsize, sharey=True,
                                 squeeze=False,
                                 gridspec_kw=dict(width_ratios=[per_col[c] for c in cols]))
        for r, title in enumerate(screens):
            res = panels[title]
            keys = list(res)
            p = {k: mannwhitneyu(np.asarray(res[k]["scg_values"], float),
                                 np.asarray(res[k]["non_scg_values"], float)).pvalue
                 for k in keys}
            qv = dict(zip(keys, bh_fdr(np.array([p[k] for k in keys]))))
            n_sets = {k: int(len(res[k]["scg_values"])) for k in keys}
            n_mode = max(set(n_sets.values()), key=list(n_sets.values()).count)
            for c, col in enumerate(cols):
                ax = grid[r][c]
                sub = [k for k in keys if _route_of(k)[0] == col]
                if not sub:
                    ax.set_visible(False)
                    continue
                _, _, S = fig_responsiveness_violin(res={k: res[k] for k in sub}, ax=ax,
                                                    q=qv, **V)
                labels, bold = [], []
                for k in sub:
                    _, route, lin, _ = _route_of(k)
                    lab = ("control cells" if route in ("controls", "shared")
                           else (f"barcode\n({lin})" if lin else "barcode"))
                    if n_sets[k] != n_mode:
                        lab += f"\nn = {n_sets[k]}"
                    labels.append(lab)
                    bold.append(route in ("controls", "shared"))
                x = np.arange(len(sub))
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                for t, b in zip(ax.get_xticklabels(), bold):
                    t.set_fontweight("bold" if b else "normal")
                # "+x%" is drawn to the LEFT of each violin; in a split layout the first
                # violin's label needs room inside its own axes, not in the neighbour's
                left = -.5 - lead
                for xi, b in zip(x, bold):
                    if b and shade:          # off by default: bold labels carry it
                        ax.axvspan(left if xi == 0 else xi - .5, xi + .5, color=shade,
                                   zorder=0, lw=0)
                ax.set_xlim(left, len(sub) - .5)
                ax.set_xlabel("")
                ax.spines[["top", "right"]].set_visible(False)
                if r == 0:
                    ax.set_title(col_titles.get(col, col),
                                 pad=col_title_pad if col_title_pad is not None
                                     else fs + 8,
                                 fontweight="bold")
                if c == 0:
                    # the row's screen is its y label; the metric is said once, below
                    ax.set_ylabel(ctrl_label(title), fontweight="bold", labelpad=4)
                else:
                    ax.tick_params(labelleft=False)
                S = S.assign(screen=title, column=col,
                             gene_set=[k.replace("\n", " ") for k in sub])
                rows_out.append(S)
        bottom = .05 if legend else .02
        fig.tight_layout(h_pad=h_pad, w_pad=w_pad, rect=(.03, bottom, 1, 1))
        fig.supylabel(ylabel or metric_label("mean_abs_lfc"), x=.005, fontsize=fs)
        if legend:
            h = [Line2D([], [], color=C_MEM, lw=1.2, label="memory genes"),
                 Line2D([], [], color=C_BG, lw=1.2,
                        label="expression-matched non-memory genes")]
            if shade:
                h.append(Patch(facecolor=shade, edgecolor="none",
                               label="found in perturb-seq control cells"))
            fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
                       bbox_to_anchor=(.5, 0.0), handlelength=1.6, columnspacing=1.6)
        _pl = panel_letters_for(panel_letters, len(screens))
        if _pl:
            for r, l in enumerate(_pl):
                panel_letter(grid[r][0], l, x=-.42, y=1.18, fontsize=fs + 2)
        save_fig(fig, save, "figS_control_vs_barcode_responsiveness", dpi=dpi,
                 verbose=verbose)
        if show:
            plt.show()
    stats = pd.concat(rows_out, ignore_index=True)
    if verbose:
        print(stats[["screen", "gene_set", "n_mem", "pct", "q"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return fig, grid, stats


# ═══════════════════════════════════════════════════════════════════════════
#  Supplementary panels that used to live inline in suppl_figures.ipynb
#  (moved here 2026-10-14). The bodies are those cells VERBATIM; only the
#  cell-level constants became keyword arguments, with the cells' own values as
#  defaults, so every panel renders exactly as it did from the notebook.
# ═══════════════════════════════════════════════════════════════════════════

# The per-group split used by Suppl. Fig 2 and Suppl. Fig 4: each timepoint / arm
# is treated as its own dataset. LARRY's timepoint column is FLOAT (2.0) while the
# others are strings -- passing "2.0" raises
# ValueError: timepoint '2.0' not found ... available: [2.0, 4.0, 6.0]
PERGROUP_GROUPS = {"T-lineage": ["T1", "T2"], "Melanoma_Harmange": ["E", "M"],
                   "LARRY": [2.0, 4.0, 6.0], "BandlerBrain": ["E13", "E14"]}
PERGROUP_SUFFIX = {"T-lineage": "T1_T2", "Melanoma_Harmange": "E_M",
                   "LARRY": "2.0_6.0_4.0", "BandlerBrain": "E13_E14"}
PERGROUP_LABEL = {"T-lineage": "Michaels", "Melanoma_Harmange": "Harmange",
                  "LARRY": "Weinreb", "BandlerBrain": "Bandler"}
PERGROUP_FOLDER = {v: k for k, v in PERGROUP_LABEL.items()}
# the benchmark FOLDER names are not the dataset keys GROUP_DISPLAY is keyed by, so
# glabel("Melanoma_Harmange", "E") would pass "E" straight through instead of
# spelling it out. This is the bridge.
PERGROUP_DKEY = {"T-lineage": "T-lineage", "Melanoma_Harmange": "Harmange",
                 "LARRY": "LARRY", "BandlerBrain": "Bandler"}


def folder_glabel(folder, group):
    """Spelled-out group name for a benchmark folder's group value."""
    return glabel(PERGROUP_DKEY.get(folder, folder), group)
PERGROUP_SCREEN_FOLDER = {"marson": "T-lineage", "weinstock": "T-lineage",
                          "xu": "Melanoma_Harmange", "frangieh": "Melanoma_Harmange",
                          "bowness": "LARRY", "mold": "BandlerBrain"}
# same definition Figure 2 uses: >= TIER of the 4 methods' top N
PERGROUP_TOP_N, PERGROUP_TIER = 300, 2
# LARRY's float groups, named as in Figure 1
PERGROUP_GLAB = {2.0: "T1", 4.0: "T2", 6.0: "T3"}

MEMSEQ_RESIDUALS = (f"{P.PROJECT_DATA}/"
                    "ground-truth/bulk_WM989_ExpSample_residuals.csv")
MELANOMA_PERTURB = f"{P.DATA}/melanoma/perturbseq"


def _ko_axis():
    """ko_axis_memory, imported from the perturb-seq tree (sys.path set first)."""
    _perturb_path()
    import ko_axis_memory as K
    return K


def _save_path(save, name):
    """save=True -> <OUTDIR>/<name>.png, read at CALL time so a notebook that sets
    F.OUTDIR after import still writes where it expects."""
    return os.path.join(OUTDIR, f"{name}.png") if save is True else save


# ---------------------------------------------------------------- Suppl. Fig 1d
def fig_topn_sharing(cuts=(100, 500, 1000), min_methods=2, thresholds=(2, 3, 4),
                     pooled_depth=True, null_mode="matched", n_perm=1000, seed=0,
                     logy=False, fs=10, panel_w=2.7, panel_h=3.2,
                     null_annot_fmt=None,
                     name="figS_cross_dataset_cutoffs", save=True, dpi=400,
                     show=True, verbose=True):
    """(Suppl. Fig 1d) Figure 1d repeated at several top-N cutoffs.

    Each panel is its own analysis: the consensus set, the shared-universe
    restriction and the expression-matched null are all rebuilt at that depth, so
    the panels are comparable in design but not nested in content.

    Returns (fig, axes, stats) with `stats` carrying a top_n column.
    """
    with plt.rc_context({**PUB_RC, "font.size": fs}):
        fig, axes = plt.subplots(1, len(cuts), figsize=(panel_w * len(cuts), panel_h))
        stats = {}
        for ax, n in zip(axes, cuts):
            _, res = fig_cross_dataset(
                panel="bars", axes=ax,          # draw into this panel, no figure of its own
                min_methods=min_methods, top_n=n, pooled_depth=pooled_depth,
                thresholds=thresholds, null_mode=null_mode, n_perm=n_perm, seed=seed,
                logy=logy, fs=fs, z_metric=None, threshold_below=True,
                yticklabels=False, null_annot_fmt=null_annot_fmt,
                ylabel="number of genes" if n == cuts[0] else "",
                show=False, verbose=False)
            ax.set_title(f"top {n} per method", fontsize=fs)
            stats[n] = res["stats"]
        for ax in axes[1:]:                     # one legend is enough
            if ax.get_legend():
                ax.get_legend().remove()
        fig.tight_layout()
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, axes, pd.concat(stats, names=["top_n", "i"]).reset_index(0)


# ---------------------------------------------------------------- Suppl. Fig 1g
def fig_characterisation_matched(char=None, top_n=None, tiers=(2,), metrics=None,
                                 k=10, seed=0, fs=10, figsize=(10.4, 2.7),
                                 margins=None, name="figS_characterisation_split",
                                 save=True, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 1g) detection, protein and length for the >=2-method genes
    against a 1:1 EXPRESSION-MATCHED background.

    The background is matched per metric, on the genes that actually carry it
    (protein is missing for some), and the stars come from ONE Benjamini-Hochberg
    correction across all 12 tests, not per panel.

    Returns (fig, axes, stats).
    """
    from scipy.stats import mannwhitneyu
    top_n = TOP_N if top_n is None else top_n
    if char is None:
        char = characterisation_table(top_n=top_n, tiers=tiers)   # same genes as Fig 1d
    if metrics is None:
        metrics = [  # (column, y label, log axis, y limits)
            ("detection", "fraction of cells with\nnon-zero counts", False, (0, 1)),
            ("protein",   "log₁₀ protein abundance (ppm)",      True,  None),
            ("length",    "log₁₀ gene length (bp)",             True,  None),
        ]
    margins = dict(left=.065, right=.995, bottom=.24, top=.86,
                   wspace=.30) if margins is None else margins

    res, rows = {}, []
    for met, _, _, _ in metrics:
        res[met] = {}
        for ds in active_datasets():
            # only genes that HAVE this metric (protein is missing for some)
            d = char[(char.dataset == ds) & (char.expression > 0) & char[met].notna()]
            ref, pool = d[d.group == "≥2"], d[d.group == "background"]
            # 1:1 nearest-neighbour match on log expression: exactly len(ref) background genes
            idx = _panel_matched_bg(np.log10(pool.expression), np.log10(ref.expression),
                                    k=k, seed=seed)
            bg = pool.iloc[idx]
            lab = dlabel(ds)
            a, b = ref[met].to_numpy(float), bg[met].to_numpy(float)
            res[met][lab] = dict(scg_values=a, non_scg_values=b)
            la, lb = np.log10(ref.expression), np.log10(bg.expression)
            rows.append(dict(metric=met, dataset=lab, n=len(a),
                             p=mannwhitneyu(a, b).pvalue,
                             median_mem=np.median(a), median_bg=np.median(b),
                             smd_log_expr=((la.mean() - lb.mean())
                                           / np.sqrt((la.var() + lb.var()) / 2))))

    stats = pd.DataFrame(rows)
    stats["q"] = bh_fdr(stats.p.values)          # ONE correction across all 12 tests

    with plt.rc_context({**PUB_RC, "font.size": fs}):
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        for ax, (met, ylab, logy, ylim) in zip(axes, metrics):
            q_row = dict(zip(stats[stats.metric == met].dataset,
                             stats[stats.metric == met].q))
            fig_responsiveness_violin(
                res=res[met], ax=ax, q=q_row, fdr=False,   # stars from the 12-test correction
                mem_label="≥2 methods", bg_label="expression-matched",
                bg_color=C_NULL,                           # memory halves red
                log_y=logy, ylim=ylim, annot="stars",
                show_medians=False, fill=False, median_inside=True, median_fmt="",
                xlabel="", ylabel=ylab, xtick_rotation=35,
                legend=False, fs=fs, show=False, verbose=False)
        # fixed margins: panels of identical size (tight_layout would make them unequal)
        fig.subplots_adjust(**margins)
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, axes, stats


# ----------------------------------------------------------- Suppl. Fig 2 setup
def pergroup_memory_sets(top_n=None, tier=PERGROUP_TIER, groups=None,
                         suffix=None, label=None, verbose=True):
    """(Suppl. Fig 2/4 setup) per-group consensus sets: no merging across timepoints.

    Returns ({unit: (memory set at >= tier, every gene any method called)}, summary)
    where unit is e.g. 'Michaels-T1'.
    """
    K = _ko_axis()
    from marson_KO import load_memory_gene_set
    top_n = TOP_N if top_n is None else top_n
    groups = PERGROUP_GROUPS if groups is None else groups
    suffix = PERGROUP_SUFFIX if suffix is None else suffix
    label = PERGROUP_LABEL if label is None else label
    units = [(folder, g, f"{label[folder]}-{g}")
             for folder, gs in groups.items() for g in gs]
    sets = {}
    for folder, g, unit in units:
        memdir, files = f"{K.BENCH}/{folder}", K.method_files(suffix[folder])
        n, called = Counter(), set()
        for m, fn in files.items():
            if not os.path.exists(os.path.join(memdir, fn)):
                continue
            genes = load_memory_gene_set(m, timepoint=g, top_n=top_n,
                                         merge_mode="union", memgene_dir=memdir,
                                         method_files=files)
            n.update(genes)
            called |= set(genes)
        sets[unit] = ({x for x, c in n.items() if c >= tier}, called)
    summary = pd.DataFrame([dict(unit=u, n_memory=len(sets[u][0]),
                                 n_called_by_any=len(sets[u][1]))
                            for _, _, u in units])
    if verbose:
        print(summary.to_string(index=False))
    return sets, summary


def _register_pergroup_screens(K):
    """The three screens Figure 2 adds by hand: Shi's memory genes from BANDLER
    Weinstock's donor-corrected bulk LFC, and Bowness."""
    K.SCREENS["mold"].update(memdir=f"{K.BENCH}/BandlerBrain", suffix="E13_E14",
                             memory_source="Bandler")
    K.SCREENS["weinstock"] = dict(
        label="Weinstock (T cells)", short="Weinstock",
        lfc=f"{P.DATA}/weinstock_lfc_cpm_donor_corrected.csv",
        expr=f"{P.DATA}/weinstock_gene_ctrl_expr.csv", npz=None, npz_keys=None,
        memdir=f"{K.BENCH}/T-lineage", suffix="T1_T2", memory_source="Michaels")
    K.SCREENS["bowness"] = dict(
        label="Bowness (HSPC)", short="Bowness",
        lfc=f"{P.DATA}/velten_hspc/velten_lfc_cpm.csv",
        expr=f"{P.DATA}/velten_hspc/velten_gene_ctrl_expr.csv",
        npz=None, npz_keys=None,
        memdir=f"{K.BENCH}/LARRY", suffix="2.0_6.0_4.0", memory_source="Weinreb")
    return K


# ---------------------------------------------------------------- Suppl. Fig 2b
def fig_marson_rungs_per_group(groups=None, per_rung=None, min_methods=2, top_n=None,
                               panels=None, n_iter=2000, k=10, seed=0,
                               floor=EXPR_FLOOR, background="strict", equalise=True,
                               expr_bins=True, n_bins=50, expr_n_iter=100,
                               rung_names=None, fs=7, panel_w=.62, panel_h=2.05,
                               name="supp2_e_matching_rungs_grid",
                               save=True, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 2b) Figure 2d repeated within each experimental group of Marson.

    MARSON-ONLY, and not extensible: state_restriction is built from Marson's Rest /
    Stim8hr / Stim48hr conditions, and expr + detection_rate come from Marson's own
    covariate table. No other screen has either, so the grid is Michaels T1 and T2.

    Returns (fig, axes, stats) with stats keyed by unit.
    """
    top_n = TOP_N if top_n is None else top_n
    groups = PERGROUP_GROUPS["T-lineage"] if groups is None else groups
    panels = MATCH_PANELS_JOINT if panels is None else panels
    rung_names = {"detection": "detection rate"} if rung_names is None else rung_names
    if per_rung is None:
        per_rung = {g: matching_rung_results(
                        min_methods=min_methods, timepoint=g, top_n=top_n,
                        panels=panels, n_iter=n_iter, k=k, seed=seed, floor=floor,
                        background=background, equalise=equalise,
                        expr_bins=expr_bins, n_bins=n_bins, expr_n_iter=expr_n_iter,
                        verbose=False)
                    for g in groups}
        per_rung = {g: {rung_names.get(kk, kk): v for kk, v in r.items()}
                    for g, r in per_rung.items()}

    with plt.rc_context({**PUB_RC, "font.size": fs}):
        fig, axes = plt.subplots(len(groups), 1, sharey=True, sharex=True,
                                 figsize=(panel_w * len(panels) + 1.6,
                                          panel_h * len(groups)))
        axes = np.atleast_1d(axes)
        stats = {}
        for ax, g in zip(axes, groups):
            _, _, st = fig_responsiveness_violin(
                res=per_rung[g], ax=ax, bg_label="Matched background",
                xlabel="", ylabel=(f"{dlabel('T-lineage', author=True)}\n"
                                   f"{glabel('T-lineage', g)}"
                                   if LABEL_MODE == "celltype" else f"Michaels-{g}"),
                annot="pct+stars", pct_where="left", show_medians=False, fill=False,
                median_inside=True, median_fmt="", ylim=(0, 1.0), legend=False,
                fs=fs, xtick_rotation=35, show=False, verbose=False)
            stats[(f"{dlabel('T-lineage', author=True)} - {glabel('T-lineage', g)}"
                   if LABEL_MODE == "celltype" else f"Michaels-{g}")] = st
        axes[-1].set_xlabel("background matched on", fontsize=fs)
        fig.supylabel("mean absolute log₂ fold change per gene", fontsize=fs, x=-0.01)
        fig.tight_layout(h_pad=.8)
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=False)
        if show:
            plt.show()
    return fig, axes, stats


# ---------------------------------------------------------- Suppl. Fig 4a, 4c
def fig_crossdataset_per_group(datasets=None, groups=None, tiers=(2, 3, 4), top_n=None,
                               fs=7, panel_h=0.30, fig_w=7.0, cmap=None, legend_lw=1.2,
                               return_tables=False,
                               name="supp2_gh_crossdataset_per_group",
                               save=True, dpi=400, show=True, verbose=True, **kw):
    """(Suppl. Fig 4a heatmap, 4c lines) The Figure 2 cross-dataset panels with every screen x
    lineage-GROUP as its own row.

    xd_sweep / xd_consensus_sweep take ONE timepoint for the whole call, but each
    dataset has its own group names, so this runs one pair at a time and
    concatenates. The row label has to be supplied separately: the panel rebuilds
    it from the \\mathbf parts of the name, so a group tag appended to that string
    is discarded.

    Returns (fig, (ax_consensus, ax_cutoff), matrix).
    """
    from matplotlib.colors import ListedColormap
    top_n = TOP_N if top_n is None else top_n
    datasets = xd_datasets(verbose=False) if datasets is None else datasets
    groups = PERGROUP_GROUPS if groups is None else groups

    con, cut, row_labels = [], [], {}
    for d in datasets:
        lin, scr, _ = _xd_pair_parts(d["name"])
        for g in groups[PERGROUP_FOLDER[lin]]:
            a = xd_consensus_sweep([d], tiers=tiers, top_n=top_n, timepoint=g,
                                   verbose=False)
            b = xd_sweep([d], top_ns=[top_n], timepoint=g, verbose=False)
            tag = f"{d['name']}  [{g}]"
            for fr in (a, b):
                fr["dataset"] = tag
            row_labels[tag] = (f"{slabel(scr)} - "
                               f"{folder_glabel(PERGROUP_FOLDER[lin], g)}"
                               if LABEL_MODE == "celltype" else
                               # the group spelled out, on its own line: "T1" and
                               # "E13" mean nothing to a reader of the supplement
                               f"{lin} – {scr}\n"
                               f"{folder_glabel(PERGROUP_FOLDER[lin], g)}")
            con.append(a)
            cut.append(b)

    S_con = pd.concat(con, ignore_index=True)
    S_cut = pd.concat(cut, ignore_index=True)
    if return_tables:
        # the two sweep tables and the row names, for drawing the heatmap and the
        # tier lines as separate figures (fig_cross_dataset_lines takes S_con)
        return S_con, S_cut, row_labels

    if cmap is None:
        _g = plt.get_cmap("Greys")
        cmap = ListedColormap(np.vstack([_g(np.zeros(256)), _g(np.linspace(0, 1, 256))]))

    # same layout as figure 2f+g, so the supplement reads as its per-group twin:
    # heatmap left, lines right, each name framed in its dataset's colour
    _kw = dict(panel_order="heat", label_box=True, gap_lines_in=.12,
               annot_fmt="{:.0f}%",
               cbar_zero_label="no difference",
               cbar_below_label="negative values:\nless response",
               cbar_below_gap_pt=12, cbar_zero_linespacing=1.15,
               summary_values=True, summary_value_pos=("above", "below", "above"),
               legend_lw=legend_lw, fs=fs, xlabel_heat="",
               figsize=(fig_w, panel_h * S_con.dataset.nunique() + 1.1))
    _kw.update(kw)
    fig, (ax_g, ax_h), M = fig_cross_dataset_combo(
        S_con, S_cut, tiers=tiers, metric="pct", cmap=cmap,
        row_labels=row_labels,                          # keeps the group in the label
        save=_save_path(save, name), dpi=dpi, show=show, **_kw)
    return fig, (ax_g, ax_h), M


# ---------------------------------------------------------------- Suppl. Fig 5a
def memoryseq_overlap_table(top_n=None, tiers=(2, 3, 4), n_perm=1000, n_bins=20,
                            seed=0, memseq_csv=MEMSEQ_RESIDUALS, verbose=True):
    """(Suppl. Fig 5a) Overlap between the lineage-barcoded Harmange memory genes
    and the bulk MemorySeq hits, against an expression-matched null drawn in BOTH
    sources' own expression.

    Returns (overlap frame, group order).
    """
    top_n = TOP_N if top_n is None else top_n
    # bulk MemorySeq (Shaffer et al., WM989): 227 is_hit genes; universe = genes
    # above the TPM floor
    memseq_ref = pd.read_csv(memseq_csv)
    bulk = memseq_ref[memseq_ref.above_floor].dropna(subset=["GeneSymbol"])
    bulk = bulk.groupby("GeneSymbol")["Mean_tpm"].max()          # 1 row per symbol
    shaffer_hits = set(memseq_ref.loc[memseq_ref.is_hit, "GeneSymbol"].dropna())

    # lineage-barcoded (Harmange et al.): genes in >=k of the methods' top N, PER
    # GROUP -- exactly the Fig 1b bars (enriched and mixed kept separate)
    tabs, groups, _ = load_tables("Harmange", scg_variant=scg_variant_for("Harmange"),
                                  verbose=False)
    per_group = {g: Counter(x for t in tabs.values() for x in top_genes(t, g, top_n))
                 for g in groups}
    harm_sets = {(g, k): {x for x, v in c.items() if v >= k}
                 for g, c in per_group.items() for k in tiers}

    # shared universe: measured in Harmange AND fitted above the floor in bulk
    universe = sorted(set(_mean_expr("Harmange", sorted(bulk.index)).dropna().index))
    pos = {g: i for i, g in enumerate(universe)}
    expr = {"Harmange": _mean_expr("Harmange", universe).values,
            "Shaffer": bulk.reindex(universe).values}
    bins = {s: pd.qcut(pd.Series(e).rank(method="first"), n_bins, labels=False).values
            for s, e in expr.items()}
    members = {s: [np.where(b == i)[0] for i in range(n_bins)] for s, b in bins.items()}

    def matched_draw(rng, source, genes):
        """random genes with the same expression-bin profile as `genes`, in that
        source's own expression"""
        prof = np.bincount(bins[source][[pos[g] for g in genes]], minlength=n_bins)
        return set(np.concatenate(
            [rng.choice(members[source][i], c, replace=c > len(members[source][i]))
             for i, c in enumerate(prof) if c]).tolist())

    shaffer_u = {g for g in shaffer_hits if g in pos}
    rng = np.random.default_rng(seed)
    rows = []
    for k in tiers:
        for g in groups:
            harm_u = {x for x in harm_sets[(g, k)] if x in pos}
            obs = len(harm_u & shaffer_u)
            nl = np.array([len(matched_draw(rng, "Harmange", harm_u)
                               & matched_draw(rng, "Shaffer", shaffer_u))
                           for _ in range(n_perm)])
            rows.append(dict(tier=k, group=g, group_label=glabel("Harmange", g),
                             n_harmange=len(harm_sets[(g, k)]),
                             n_harmange_in_universe=len(harm_u),
                             n_shaffer_in_universe=len(shaffer_u),
                             observed=obs, null_mean=nl.mean(),
                             null_hi=np.percentile(nl, 97.5),
                             fold=obs / nl.mean() if nl.mean() else np.nan,
                             p=((nl >= obs).sum() + 1) / (n_perm + 1)))
    table = pd.DataFrame(rows)
    if verbose:
        print(table.round(3).to_string(index=False))
    return table, list(groups)


def fig_memoryseq_overlap(table=None, groups=None, tiers=(2, 3, 4), fs=10,
                          figsize=(3.6, 2.5), hatch_lw=2.0,
                          xlabel=None,
                          ylabel="genes shared with\nbulk MemorySeq",
                          name="suppl5a_memoryseq_overlap",
                          save=True, dpi=400, show=True, verbose=True, **kw):
    """(Suppl. Fig 5a) The overlap bars, in the Figure 1c style: group 1 solid red,
    group 2 white with a red cross-hatch, each with its own grey random bar and
    dashed lines between tiers.

    Returns (fig, ax, table).
    """
    if table is None:
        table, groups = memoryseq_overlap_table(tiers=tiers, verbose=verbose, **kw)
    groups = list(dict.fromkeys(table.group)) if groups is None else groups
    if xlabel is None:      # names the dataset the methods were run on
        xlabel = (f"methods calling the gene in the {dlabel('Harmange').lower()} dataset"
                  if LABEL_MODE == "celltype" else "Harmange methods calling the gene")

    style = {groups[0]: dict(color=C_OBS, hatch=""),        # enriched: solid
             groups[1]: dict(color="white", hatch="xxx")}   # mixed: white, red cross-hatch
    with plt.rc_context({**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs, "ytick.labelsize": fs - 1,
                         "legend.fontsize": fs - 1, "hatch.linewidth": hatch_lw}):
        fig, ax = plt.subplots(figsize=figsize)              # Fig 1b panel size
        order = [(g, kind) for g in groups for kind in ("obs", "null")]
        nb, W = len(order), 0.95 / len(order)
        off = (np.arange(nb) - (nb - 1) / 2) * W
        x = np.arange(len(tiers))
        first_null = True
        for j, (g, kind) in enumerate(order):
            d = table[table.group == g].set_index("tier").loc[list(tiers)]
            if kind == "obs":
                st = style[g]
                # the hatch takes its colour from the edge, which also draws the outline
                ax.bar(x + off[j], d.observed, W, color=st["color"],
                       hatch=st["hatch"] or None,
                       edgecolor=C_OBS if st["hatch"] else "none",
                       linewidth=.8 if st["hatch"] else 0,
                       label=glabel("Harmange", g), zorder=3)
                for xi, o in zip(x, d.observed):
                    ax.annotate(f"{o:.0f}", (xi + off[j], o), textcoords="offset points",
                                xytext=(0, 2.5), ha="center", fontsize=fs - 2,
                                color=C_OBS)      # red: a white bar colour would vanish
            else:
                ax.bar(x + off[j], d.null_mean, W, color=C_NULL, zorder=3,
                       label="Random" if first_null else None)
                first_null = False
                # the whisker is the null's 97.5th percentile. In a sparse tier the
                # null is almost all zeros, so that percentile can sit BELOW the
                # null mean (e.g. mean 0.019, p97.5 = 0) and matplotlib rejects the
                # negative error. Clamp at 0: the bar top already states the mean.
                ax.errorbar(x + off[j], d.null_mean,
                            yerr=[np.zeros(len(x)),
                                  np.clip(d.null_hi - d.null_mean, 0, None)],
                            fmt="none", ecolor="0.35", capsize=2, lw=.9, zorder=4)
        for xs in x[:-1] + .5:
            ax.axvline(xs, color="0.75", lw=.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"≥{k}" for k in tiers])
        ax.set_xlim(-.6, len(tiers) - .4)
        ax.set_ylim(0, table.observed.max() * 1.34)
        ax.set_yticklabels([])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        h, l = ax.get_legend_handles_labels()
        idx = ([i for i, s in enumerate(l) if s != "Random"]
               + [i for i, s in enumerate(l) if s == "Random"])
        leg = ax.legend([h[i] for i in idx], [l[i] for i in idx], frameon=True,
                        loc="upper right", handlelength=2.0, handleheight=1.2,
                        handletextpad=.5, borderaxespad=.2, labelspacing=.3)
        leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1)
        leg.set_zorder(6)
        fig.tight_layout()

    save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
    if show:
        plt.show()
    return fig, ax, table


# ---------------------------------------------------------------- Suppl. Fig 5b
def fig_memoryseq_responsiveness(top_n=None, min_methods=2, screens=None, labels=None,
                                 memseq_csv=MEMSEQ_RESIDUALS, n_bins=50, n_iter=100,
                                 fs=14, figsize=(13.0, 5.6),
                                 title_fmt="{screen} perturb-seq screen", legend=True,
                                 name="suppl5b_memoryseq_responsiveness",
                                 save=True, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 5b) Responsiveness of the lineage-barcoded Harmange set
    (>= min_methods) against the bulk MemorySeq hits, in two melanoma screens.

    Returns (fig, axes, stats).
    """
    from matplotlib.lines import Line2D
    _perturb_path()
    from marson_KO import (build_consensus_gene_sets,
                           compare_gene_responsiveness_expr_matched)
    top_n = TOP_N if top_n is None else top_n

    # bulk MemorySeq (Shaffer et al., WM989): the 227 is_hit genes
    memseq_ref = pd.read_csv(memseq_csv)
    shaffer_genes = memseq_ref.loc[memseq_ref.is_hit, "GeneSymbol"].dropna().unique().tolist()

    # lineage-barcoded (Harmange et al.): >= min_methods of the methods, each
    # method's top N, enriched and mixed pooled per method (merge/union) -- the
    # xd_consensus_sweep definition
    harm = next(d for d in xd_datasets(verbose=False) if "Harmange" in d["memdir"])
    harm_files = {m: f for m, f in harm["files"].items() if m in XD_METHODS_CONSENSUS}

    if screens is None:
        screens = {
            "Xu": (f"{MELANOMA_PERTURB}/melanoma_gene_summary_cpm_DMSO.csv",
                   f"{MELANOMA_PERTURB}/melanoma_gene_ctrl_expr_DMSO.csv"),
            "Frangieh": (f"{MELANOMA_PERTURB}/frangieh_gene_summary_cpm_Control.csv",
                         f"{MELANOMA_PERTURB}/frangieh_gene_ctrl_expr_Control.csv"),
        }
    labels = {"Harmange": "Harmange\nlineage-barcoded",
              "Shaffer": "Shaffer\nbulk MemorySeq"} if labels is None else labels

    res_by_screen = {}
    for screen, (gs_path, expr_path) in screens.items():
        gs = pd.read_csv(gs_path, index_col=0)
        ge = pd.read_csv(expr_path, index_col=0).iloc[:, 0]
        tier_sets, _ = build_consensus_gene_sets(
            min_methods=min_methods, timepoint="merge", top_n=top_n, merge_mode="union",
            memgene_dir=harm["memdir"], method_files=harm_files, universe=gs.index,
            verbose=False)
        sets = {"Harmange": tier_sets[f">={min_methods} methods"],
                "Shaffer": [g for g in shaffer_genes if g in gs.index]}
        res = {}
        for nm, genes in sets.items():
            genes = [g for g in genes if g in ge.index]
            if verbose:
                print(f"{screen} | {nm}: {len(genes)} memory genes measured")
            # background = every other measured gene, expression-matched (same for both bars)
            r = compare_gene_responsiveness_expr_matched(gs, ge, scg_genes=genes,
                                                         metric=METRIC, n_bins=n_bins,
                                                         n_iter=n_iter)
            r.update(expression_balance(gs, ge, genes, n_bins=n_bins, n_iter=n_iter,
                                        metric=METRIC))
            res[labels[nm]] = r
        res_by_screen[screen] = res

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    stats = []
    for i, (ax, (screen, res)) in enumerate(zip(axes, res_by_screen.items())):
        _, _, S = fig_responsiveness_violin(
            res=res, ax=ax,
            mem_label="Memory genes", bg_label="Expression-matched",
            annot="pct+stars", show_medians=False, median_fmt="{:.2f}",
            fill=False, median_inside=True, pct_where="left",
            ylim=(0, 1.0), xlabel="", title=title_fmt.format(screen=screen),
            ylabel=None if i == 0 else "",
            fs=fs, legend=(legend and i == 1), legend_out=True, verbose=False,
        )
        ax.set_xlim(-0.8, 1.6)
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs - 1)
        ax.xaxis.label.set_size(fs + 1)
        ax.yaxis.label.set_size(fs + 1)
        stats.append(S.assign(screen=screen))
    stats = pd.concat(stats, ignore_index=True)

    fig.tight_layout()
    # thin divider between the two screens
    x = (axes[0].get_position().x1 + axes[1].get_position().x0) / 2
    fig.add_artist(Line2D([x, x], [0.08, 0.95], transform=fig.transFigure,
                          color="0.6", lw=1))

    save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
    if show:
        plt.show()
    return fig, axes, stats


# ---------------------------------------------------------------- Suppl. Fig 6a
# Memory genes found WITHOUT lineage barcodes. Moved out of
# perturb-seq/perturb_control_mgs.ipynb (FINAL 1 + FINAL 3) on 2026-10-14; the
# constants and the loop are those cells verbatim, so the panel is unchanged.
#   Lineage barcode : the method with real barcodes on the cell-type-matched
#                     lineage dataset (python_implements)
#   Perturb-seq controls : the same method run on that screen's control cells
#                     (GEMLI lineage-free = predicted clones, cutoff 50)
# GEMLI lists use the authors' memory-gene definition (p <= 0.05) in the authors'
# order, order(p, -var).
# CAVEATS: GEMLI Frangieh uses 17,050 of its 57,605 control cells (every other
# dataset uses all of them), and on control cells nothing checks per dataset that
# GEMLI's predicted clones are real.
_CTRL_M = f"{P.DATA}/marson"
_CTRL_V = f"{P.DATA}/velten_hspc"
_CTRL_P = f"{P.DATA}/melanoma/perturbseq"
_CTRL_S = f"{P.DATA}/Mold_brain"     # Shi's screen lives in the Mold_brain folder
CTRL_GEMLI_DIR = f"{P.DATA}/gemli_control/results"
_CTRL_TAIL = "_floor0.1_seed0_ss75_cut50.csv"
CTRL_MARSON_GROUPS = [f"{d}_{c}" for d in ("D1", "D2", "D3", "D4")
                      for c in ("Rest", "Stim8hr", "Stim48hr")]

# (title, gene summary, control expression, Power-Seek control lists,
#  (lineage dir, lineage file), lineage label, y limits)
# Naming follows figure 2 (DATASET_LABELS / DATASET_CELLTYPE): lineage datasets are
# named by first author -- T-lineage = Michaels, LARRY = Weinreb. NOTE the data dir
# is `velten_hspc`, but the figure label for GSE320250 is Bowness.
CTRL_SPEC = [
    ("Marson (Progenitor T cells)",
     f"{_CTRL_M}/marson_gene_summary_cpm.csv", f"{_CTRL_M}/marson_gene_ctrl_expr.csv",
     sorted(glob.glob(f"{_CTRL_M}/powerseek_control/"
                      "powerseek_control_delta_Marson2025_*_nall_cv3000_floor0.1.csv")),
     ("T-lineage", "powerseek_memory_genes_T1_T2.csv"), "Michaels", (0, 1.5)),
    # order: the canonical figure order (xd_datasets), so the rows match
    # Fig. 2 and Suppl. 4a -- T cells, melanoma, HSPCs, brain
    ("Xu (Melanoma)",
     f"{_CTRL_P}/melanoma_gene_summary_cpm_DMSO.csv",
     f"{_CTRL_P}/melanoma_gene_ctrl_expr_DMSO.csv",
     [f"{_CTRL_P}/powerseek_control/powerseek_control_delta_perturbfate_DMSO"
      "_nall_cv3000_floor0.1.csv"],
     ("Melanoma_Harmange", "powerseek_memory_genes_E_M.csv"), "Harmange", (0, 0.8)),
    ("Frangieh (Melanoma)",
     f"{_CTRL_P}/frangieh_gene_summary_cpm_Control.csv",
     f"{_CTRL_P}/frangieh_gene_ctrl_expr_Control.csv",
     [f"{_CTRL_P}/powerseek_control/powerseek_control_delta_frangieh_Control"
      "_nall_cv3000_floor0.1.csv"],
     ("Melanoma_Harmange", "powerseek_memory_genes_E_M.csv"), "Harmange", (0, 0.8)),
    ("Bowness (HSPC)",
     f"{_CTRL_V}/velten_gene_summary_cpm.csv", f"{_CTRL_V}/velten_gene_ctrl_expr.csv",
     [f"{_CTRL_V}/powerseek_control/powerseek_control_delta_velten_{r}"
      "_nall_cv3000_floor0.1.csv" for r in ("rep1", "rep2")],
     ("LARRY", "powerseek_memory_genes_2.0_6.0_4.0.csv"), "Weinreb", (0, 1.0)),
    # 46 groups = cell type x sex. Barcode partner is Bandler; "mold" in these
    # paths is this project's internal name for the Shi screen, not a dataset.
    ("Shi (Brain)",
     f"{_CTRL_S}/mold_gene_summary_cpm.csv", f"{_CTRL_S}/mold_gene_ctrl_expr.csv",
     sorted(glob.glob(f"{_CTRL_S}/powerseek_control/"
                      "powerseek_control_delta_shi_*_nall_cv3000_floor0.1.csv")),
     ("BandlerBrain", "powerseek_memory_genes_E13_E14.csv"), "Bandler", (0, 1.0)),
]

CTRL_BARCODE_FILES = {
    "GEMLI": {"T-lineage": "gemli_memory_genes_T1_T2.csv",
              "LARRY": "gemli_memory_genes_2.0_6.0_4.0.csv",
              "Melanoma_Harmange": "gemli_memory_genes_E_M.csv",
              "BandlerBrain": "gemli_memory_genes_E13_E14.csv"},
    "Power-Seek": {"T-lineage": "powerseek_memory_genes_T1_T2.csv",
                   "LARRY": "powerseek_memory_genes_2.0_6.0_4.0.csv",
                   "Melanoma_Harmange": "powerseek_memory_genes_E_M.csv",
                   "BandlerBrain": "powerseek_memory_genes_E13_E14.csv"},
    # isCORAL's barcode partner: CORAL is the same omega^2 on the REAL clones
    "CORAL": {"T-lineage": "coral_memory_genes_T1_T2.csv",
              "LARRY": "coral_memory_genes_2.0_6.0_4.0.csv",
              "Melanoma_Harmange": "coral_memory_genes_E_M.csv",
              "BandlerBrain": "coral_memory_genes_E13_E14.csv"},
}

CTRL_GEMLI_FILES = {
    "Marson (Progenitor T cells)": [f"{CTRL_GEMLI_DIR}/marson/py_pred_all_Marson2025_{g}{_CTRL_TAIL}"
                                    for g in CTRL_MARSON_GROUPS],
    "Bowness (HSPC)": [f"{CTRL_GEMLI_DIR}/velten/py_pred_all_{r}{_CTRL_TAIL}"
                       for r in ("rep1", "rep2")],
    "Frangieh (Melanoma)": [f"{CTRL_GEMLI_DIR}/frangieh/py_pred_all_Control_n17050{_CTRL_TAIL}"],
    "Xu (Melanoma)": [f"{CTRL_GEMLI_DIR}/perturbfate/py_pred_all_DMSO{_CTRL_TAIL}"],
    "Shi (Brain)": sorted(glob.glob(f"{CTRL_GEMLI_DIR}/shi/py_pred_all_*{_CTRL_TAIL}")),
}

# isCORAL (Lin & Lin 2026, bioRxiv 10.64898/2026.10.19.760045) on the SAME control cells
# GEMLI used (meta.csv md5 verified per group): one run per group, at most 4,000 cells,
# bin size N/50, package seed 20251009 -- original_implementations/isCORAL/
# run_iscoral_control.sbatch (job 5097157). Ranked by omega_square_infer; genes below
# mean 0.05 are unscored (omega 0) and dropped.
CTRL_ISCORAL_DIR = f"{P.DATA}/iscoral_control/results"
_ISC_TAIL = "_floor0.1_seed20251009_bin*.csv"


def _iscoral_files(ds, groups):
    return [f for g in groups
            for f in sorted(glob.glob(f"{CTRL_ISCORAL_DIR}/{ds}/iscoral_{g}{_ISC_TAIL}"))]


CTRL_ISCORAL_FILES = {
    "Marson (Progenitor T cells)": _iscoral_files("marson", [f"Marson2025_{g}"
                                                             for g in CTRL_MARSON_GROUPS]),
    "Bowness (HSPC)": _iscoral_files("velten", ["rep1", "rep2"]),
    "Frangieh (Melanoma)": _iscoral_files("frangieh", ["Control_n17050"]),
    "Xu (Melanoma)": _iscoral_files("perturbfate", ["DMSO"]),
    "Shi (Brain)": sorted(glob.glob(f"{CTRL_ISCORAL_DIR}/shi/iscoral_*{_ISC_TAIL}")),
}


def control_vs_barcode_panels(spec=None, top_n=None, n_bins=50, n_iter=100, floor=0.1,
                              metric="mean_abs_lfc", seed=0, bench=None,
                              barcode_files=None, gemli_files=None, iscoral_files=None,
                              verbose=True):
    """(Suppl. Fig 6a) The seven gene sets per screen, each scored on the SAME readout.

    Power-Seek, GEMLI and isCORAL on the screen's control cells, each next to the same
    method with lineage barcodes (isCORAL's barcode partner is CORAL, the same omega^2
    on real clones), plus the genes ALL THREE call on the control cells.
    All sets in a panel are scored on that screen's own knockout |LFC|, matched on
    the same control expression, with the floor-0.1 prep and the Figure-2 rank merge
    (best rank per gene across groups).

    Returns (panels, ylim, table) where `panels` is the {screen: {label: result}}
    dict fig_control_vs_barcode_violins() takes.
    """
    _perturb_path()
    from marson_KO import (compare_gene_responsiveness_expr_matched,
                           load_memory_gene_set)
    top_n = TOP_N if top_n is None else top_n
    spec = CTRL_SPEC if spec is None else spec
    bench = os.path.join(BENCH, RESULTS_TREE) if bench is None else bench
    barcode_files = CTRL_BARCODE_FILES if barcode_files is None else barcode_files
    gemli_files = CTRL_GEMLI_FILES if gemli_files is None else gemli_files
    iscoral_files = CTRL_ISCORAL_FILES if iscoral_files is None else iscoral_files

    def _readout(summary, expr):
        gs = pd.read_csv(summary, index_col=0)
        eg = pd.read_csv(expr, index_col=0).iloc[:, 0]
        return gs.loc[gs.index.intersection(eg[eg > floor].index)], eg

    def _merge_ranked(lists, restrict):
        """Figure 2's merge: restrict first, rank within each list, keep each gene's
        best rank."""
        best = {}
        for l in lists:
            for r, g in enumerate([x for x in l if x in restrict]):
                if g not in best or r < best[g]:
                    best[g] = r
        return [g for g, _ in sorted(best.items(), key=lambda kv: kv[1])]

    def _gemli_ranked(path):
        """Authors' memory genes (p <= 0.05) in the authors' order, order(p, -var)."""
        t = pd.read_csv(path)
        t = t[t["p"] <= 0.05].sort_values(["p", "var"], ascending=[True, False],
                                          kind="stable")
        return list(t["gene"])

    def _iscoral_ranked(path):
        """Scored genes by omega_square_infer, best first (the file is already sorted)."""
        t = pd.read_csv(path)
        t = t[t["scored"]].sort_values("omega_square_infer", ascending=False,
                                       kind="stable")
        return list(t["gene"])

    panels, ylims, rows = {}, {}, []
    for title, s, e, ps_files, (lin_dir, _), lin_label, ylim in spec:
        missing = [f for f in gemli_files[title] if not os.path.exists(f)]
        assert not missing, f"missing GEMLI control lists: {missing}"
        # isCORAL must cover the same groups as GEMLI, or its merged list would rest on
        # a subset. A screen whose runs are not all done is SKIPPED (loudly) rather than
        # drawn with a partial isCORAL column -- it appears once the runs finish.
        n_isc, n_gem = len(iscoral_files.get(title, [])), len(gemli_files[title])
        if n_isc != n_gem:
            print(f"!! SKIPPING {title}: isCORAL has {n_isc} of {n_gem} groups "
                  "(see original_implementations/isCORAL/iscoral_control_status.sh)")
            continue
        gs, eg = _readout(s, e)
        U = set(gs.index)

        def _barcode(method, lin_dir=lin_dir, gs=gs):
            return load_memory_gene_set(method, timepoint="merge", top_n=None,
                                        merge_mode="union",
                                        memgene_dir=f"{bench}/{lin_dir}",
                                        method_files={method: barcode_files[method][lin_dir]},
                                        restrict_to=gs.index)

        # Method-matched pairs, each in FINAL 1's order (controls, then barcode):
        # every method is compared with ITSELF on the two routes. The isCORAL column's
        # barcode violin is CORAL; its key starts with "isCORAL" so it lands in that column.
        ranked = {
            "Power-Seek\nperturb-seq controls":
                _merge_ranked([list(pd.read_csv(f, index_col=0).index) for f in ps_files], U),
            f"Power-Seek\nlineage barcode\n({lin_label})": _barcode("Power-Seek"),
            "GEMLI\nperturb-seq controls":
                _merge_ranked([_gemli_ranked(f) for f in gemli_files[title]], U),
            f"GEMLI\nlineage barcode\n({lin_label})": _barcode("GEMLI"),
            "isCORAL\nperturb-seq controls":
                _merge_ranked([_iscoral_ranked(f) for f in iscoral_files[title]], U),
            f"isCORAL\nlineage barcode\n({lin_label})": _barcode("CORAL"),
        }

        # 7th violin: genes ALL THREE methods call on the same control cells -- the
        # intersection of the three top-N lists above, within the readout universe. It
        # has fewer than top_n genes by construction (n is in the label). Still
        # expression-matched, so the shared lean toward highly expressed genes does
        # not inflate it.
        _top = lambda r: [g for g in r if g in U][:top_n]
        _others = [set(_top(ranked[f"{m}\nperturb-seq controls"]))
                   for m in ("Power-Seek", "isCORAL")]
        shared = [g for g in _top(ranked["GEMLI\nperturb-seq controls"])
                  if all(g in o for o in _others)]
        ranked["shared by all three methods\nperturb-seq controls\n"
               f"(n = {len(shared)})"] = shared

        panels[title] = {n: compare_gene_responsiveness_expr_matched(
                             gs, eg, [g for g in r if g in U][:top_n],
                             metric=metric, n_bins=n_bins, n_iter=n_iter,
                             random_seed=seed)
                         for n, r in ranked.items()}
        ylims[title] = ylim
        for n, res in panels[title].items():
            a, b = res["scg_values"], res["non_scg_values"]
            rows.append(dict(dataset=title, gene_set=n.replace("\n", " "), n=res["n_scg"],
                             pct=100 * (np.median(a) / np.median(b) - 1),
                             p=res["mann_whitney_p"]))
    table = pd.DataFrame(rows)
    if verbose:
        print(table.round({"pct": 1}).to_string(index=False))
    return panels, ylims, table


def fig_lineage_free_responsiveness(panels=None, table=None, ylim=(0, 1.0), fs=7,
                                    figsize=(7.09, 8.4),
                                    median_weight="bold", ytick_fs=None,
                                    pct_where="left", col_title_pad=None, legend=True,
                                    col_order=("Power-Seek", "GEMLI", "isCORAL",
                                               "both methods"),
                                    col_titles=None, panel_letters=None,
                                    name="suppl6a_control_vs_barcode_responsiveness",
                                    also_save=None, save=True, dpi=400, show=True,
                                    verbose=True, **kw):
    """(Suppl. Fig 6a) Knockout responsiveness of memory genes found WITHOUT lineage
    barcodes: one row per screen, Power-Seek | GEMLI | isCORAL | all three methods.

    `also_save` takes extra paths (the working copy next to the GEMLI control runs,
    say). Returns (fig, axes, stats, table, panels) -- pass `panels` back to redraw
    without recomputing.
    """
    # "both methods" stays the internal column key (_route_of returns it for every
    # "shared ..." label); only its title says three
    col_titles = col_titles or {"both methods": "all three methods"}
    if panels is None:
        panels, _, table = control_vs_barcode_panels(verbose=verbose, **kw)
    paths = ([_save_path(save, name)] if save else [])
    paths += list(also_save or [])
    fig, axes, stats = fig_control_vs_barcode_violins(
        panels, ylim=ylim, fs=fs, figsize=figsize, col_order=col_order,
        median_weight=median_weight, ytick_fs=ytick_fs, pct_where=pct_where,
        col_title_pad=col_title_pad, legend=legend,
        col_titles=col_titles, panel_letters=panel_letters,
        save=paths or None, dpi=dpi, show=show, verbose=verbose)
    return fig, axes, stats, table, panels


# ---------------------------------------------------------------- Suppl. Fig 6b
def _ctrl_route(gene_set):
    """'controls' | 'barcode' | 'shared' from a flattened FINAL-3 gene-set label."""
    g = str(gene_set).lower()
    if g.startswith("shared"):
        return "shared"
    return "controls" if "perturb-seq controls" in g else "barcode"


def fig_lineage_free_summary(stats, screens=None,
                             methods=("Power-Seek", "GEMLI", "isCORAL", "both methods"),
                             shared_label="all three methods",
                             colors=None, shared_color="0.25", alpha_q=.05,
                             annot_values=True, value_fmt="{:+.0f}%", ns_label="n.s.",
                             row_gap=.26, ms=5.5, fs=8, figsize=(5.4, 3.4),
                             annot_fs=None, legend_fs=None, tick_fs=None,
                             xlabel="% higher median responsiveness than matched control",
                             name="suppl6b_lineage_free_summary",
                             save=True, dpi=400, show=True, verbose=True):
    """(Suppl. Fig 6b) The Suppl. 6a grid summarised: every estimate on one axis.

    One row per perturb-seq screen, four sub-rows: Power-Seek, GEMLI, isCORAL (barcode
    partner: CORAL) and the genes all three call on the control cells. For each method
    a dumbbell joins its two routes,
    so the question the panel asks is read off the direction of the segment: filled
    marker = memory genes found on the screen's own CONTROL cells (no barcodes),
    open marker = the same method on the cell-type-matched BARCODED dataset. A
    segment pointing right means the barcode-free route gave the more responsive
    set. Estimates with q > `alpha_q` are labelled, since a long bar that is not
    significant should not read as a result.

    stats : the frame returned by fig_lineage_free_responsiveness / the violin grid
            (columns screen, column, gene_set, pct, q).
    Returns (fig, ax, frame) with `frame` the plotted rows.
    """
    from matplotlib.lines import Line2D
    d = stats.copy()
    d["route"] = [_ctrl_route(g) for g in d["gene_set"]]
    screens = list(dict.fromkeys(d["screen"])) if screens is None else list(screens)
    colors = ({m: C_METHOD.get(m, shared_color) for m in methods}
              if colors is None else colors)

    with plt.rc_context({**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs - 1, "ytick.labelsize": fs}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(0, color="0.6", lw=.9, ls=(0, (3, 2)), zorder=1)
        for yi, scr in enumerate(screens):
            if yi:                       # hairline between screens
                ax.axhline(yi - .5, color="0.9", lw=.6, zorder=0)
            for j, m in enumerate(methods):
                y = yi + (len(methods) - 1) / 2 * row_gap - j * row_gap
                sub = d[(d.screen == scr) & (d.column == m)]
                col = colors.get(m, shared_color)
                pts = {r.route: r for r in sub.itertuples()}
                if "controls" in pts and "barcode" in pts:
                    ax.plot([pts["barcode"].pct, pts["controls"].pct], [y, y],
                            color=col, lw=1.4, alpha=.55, zorder=2,
                            solid_capstyle="round")
                for route, r in pts.items():
                    filled = route != "barcode"          # barcode = open marker
                    mk = "D" if route == "shared" else "o"
                    ax.plot(r.pct, y, mk, ms=ms if mk == "o" else ms - 1.2,
                            color=col if filled else "white",
                            mec=col, mew=1.3, zorder=4)
                    bits = []
                    if annot_values:
                        bits.append(value_fmt.format(r.pct))
                    if ns_label and getattr(r, "q", 0) > alpha_q:
                        bits.append(ns_label)
                    if bits:
                        ax.annotate(" ".join(bits), (r.pct, y), textcoords="offset points",
                                    xytext=(0, 5 if filled else -9), ha="center",
                                    fontsize=annot_fs or fs - 2.5, color="0.3", zorder=5)
        ax.set_yticks(range(len(screens)))
        ax.set_yticklabels([ctrl_label(s) for s in screens])
        ax.set_ylim(len(screens) - .5, -.5)              # first screen at the top
        ax.set_xlabel(xlabel)
        if tick_fs:                       # axis numbers smaller than the rest
            ax.tick_params(axis="x", labelsize=tick_fs)
        ax.spines[["top", "right"]].set_visible(False)
        h = ([Line2D([], [], ls="", marker="o", ms=ms, color=colors.get(m, shared_color),
                     label=m) for m in methods if m != "both methods"]
             + [Line2D([], [], ls="", marker="D", ms=ms - 1.2,
                       color=colors.get("both methods", shared_color), label=shared_label),
                Line2D([], [], ls="", marker="o", ms=ms, color="0.35", mec="0.35",
                       label="perturb-seq controls"),
                Line2D([], [], ls="", marker="o", ms=ms, color="white", mec="0.35",
                       mew=1.3, label="lineage barcode")])
        ax.legend(handles=h, frameon=False, fontsize=legend_fs or fs - 1.5, ncol=3,
                  loc="upper center",
                  bbox_to_anchor=(.5, -.22), handletextpad=.4, columnspacing=1.0)
        fig.tight_layout()
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    frame = d[["screen", "column", "route", "gene_set", "pct", "q"]].reset_index(drop=True)
    if verbose:
        print(frame.round({"pct": 1}).to_string(index=False))
    return fig, ax, frame


# ────────────────────────────────────────────────── underpowered screens (Fig 2b)
def ko_background_values(eff, M, called, metric="mean_abs_lfc"):
    """Per-KO score for EVERY knockout no method called: a screen's full background.

    Fig 2b's usual grey half is an equal-sized random draw from these, averaged over
    many draws. That collapses to a point cloud when the memory set is one or five
    knockouts, so an underpowered screen shows the whole pool instead. It is not
    matched on anything, which is exactly why the panel must label it as untested.
    """
    K = _ko_axis()
    val = K.ko_stats(M, None)[metric]
    tab = eff.assign(**{metric: val.reindex(eff.index).to_numpy()}).dropna(subset=[metric])
    skip = set(called)
    return tab.loc[[g for g in tab.index if g not in skip], metric].to_numpy(float)


def mark_underpowered(ax, res, counts, min_n=10, dx=.20, jitter=.0, color=None,
                      point_size=16, point_lw=.6, point_alpha=1.0,
                      label="too few\nperturbations\n(n = {n})", label_y=1.0,
                      label_dy=4,
                      label_fs=None, label_color="0.35", fs=7, seed=0, verbose=True):
    """Screens with too few memory-gene knockouts to test, drawn honestly.

    The grey background half STAYS (so the panel still says where that screen's
    knockouts sit), the memory violin and everything derived from it are removed
    (a density and a percentage from five points are not estimates), and the few
    knockouts are shown as individual points beside the grey half, with the count
    written where the significance stars would be.

    ax     : the axes fig_responsiveness_violin() drew into
    res    : the same {label: result} dict it was given, in the same order
    counts : {label: number of memory-gene knockouts}
    Returns the labels it marked.
    """
    labels = list(res)
    weak = [i for i, k in enumerate(labels) if counts[k] < min_n]
    # match the significance stars exactly: they are drawn at `annot_fs or fs - 1`
    # with no house-minimum clamp, so deriving the size here would come out larger.
    # Read it off a star that is still on the axes, and fall back to the same rule.
    if label_fs is None:
        _stars_txt = {"n.s.", "*", "**", "***", "****"}
        _found = [t.get_fontsize() for t in ax.texts if t.get_text() in _stars_txt]
        label_fs = _found[0] if _found else fs - 1
    color = C_OBS if color is None else color
    rng = np.random.default_rng(seed)
    _anchor = lambda t: float(np.atleast_1d(getattr(t, "xy", t.get_position()))[0])
    for i in weak:
        # the LEFT half only: seaborn's split violin puts the memory half entirely at
        # x <= i, so the grey half (x >= i) survives untouched
        for col in list(ax.collections):
            paths = col.get_paths()
            if not paths:
                continue
            v = paths[0].vertices
            if len(v) < 3:
                continue
            mid = np.mean([v[:, 0].min(), v[:, 0].max()])
            if abs(mid - i) < .55 and v[:, 0].max() <= i + 1e-6:
                col.remove()
        # the memory half's median line, drawn across the left side only
        for ln in list(ax.lines):
            xd = np.asarray(ln.get_xdata(), float)
            if xd.size and (i - .5) <= xd.min() and xd.max() <= i + 1e-6:
                ln.remove()
        # percentage and stars: both are claims about a comparison that was not made
        for t in list(ax.texts):
            if abs(_anchor(t) - i) < .55:
                t.remove()
        y = np.asarray(res[labels[i]]["scg_values"], float)
        x = np.full(y.size, i - dx)
        if jitter:
            x = x + rng.uniform(-jitter, jitter, y.size)
        ax.scatter(x, y, s=point_size, facecolor=color, edgecolor="white",
                   linewidth=point_lw, alpha=point_alpha, zorder=6, clip_on=False)
        if label:
            # same anchor the significance stars use in fig_responsiveness_violin:
            # (i, 1.0) in (data, axes fraction) with a 4 pt offset, va="bottom", so
            # the label sits on the stars' line instead of inside the axes
            ax.annotate(label.format(n=int(counts[labels[i]])), (i, label_y),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, label_dy), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=label_fs, color=label_color,
                        style="italic", zorder=6, clip_on=False)
    if verbose and weak:
        print("underpowered:", ", ".join(f"{labels[i]} (n={counts[labels[i]]})"
                                         for i in weak))
    return [labels[i] for i in weak]


# ───────────────────────────────── Power-Seek on every perturb-seq dataset (S8)
# Ported from mem_genes_benchmark/power-seek_run/responsiveness.ipynb (cells 0-2
# and the summary cell) on 2026-10-16. The expensive half stays in that notebook:
# it writes one per-gene summary CSV per dataset, and this reads them.
PS_RUN = f"{P.MEMGENES_ROOT}/power-seek_run"
PS_PB_TAG = "koqc-all_ctrlqc-same"      # pseudobulk QC as in the main paper
PS_RESULTS = f"{PS_RUN}/results/powerseek"
PS_PB_DIR = f"{P.SCRATCH}/power-seek_run/pseudobulk"
PS_CACHE = f"{PS_RUN}/results/responsiveness"
PS_MIN_CELLS = 20
PS_DOUBLE_KINDS = ("single", "double")
# FIRST author of each dataset's publication, not the senior one: Mali 2026 is
# Nourreddine et al., Replogle 2025 is Nadig et al., X-Atlas is Huang et al.
PS_LABELS = {
    "ReplogleWeissman2022_K562_essential": "Replogle 2022\nK562 essential",
    "ReplogleWeissman2022_K562_gwps":      "Replogle 2022\nK562 genome-wide",
    "ReplogleWeissman2022_rpe1":           "Replogle 2022\nRPE1",
    "ReplogleWeissman2025_HepG2":          "Nadig 2025\nHepG2",
    "ReplogleWeissman2025_Jurkat":         "Nadig 2025\nJurkat",
    "XAtlas2025_HCT116":                   "Huang 2025\nHCT116",
    "XAtlas2025_HEK293T":                  "Huang 2025\nHEK293T",
    "Marson2025_D1_Rest":                  "Zhu 2026\ndonor 1, resting",
    "Marson2025_D4_Stim8hr":               "Zhu 2026\ndonor 4, stimulated 8 h",
    "Wong2026_Untreated":                  "Wong 2026\nuntreated",
    "Wong2026_Treated":                    "Wong 2026\ntreated",
    "Mali2026_ALPHA":                      "Nourreddine 2026\nbatch ALPHA",
    "Mali2026_BETA":                       "Nourreddine 2026\nbatch BETA",
    "Mali2026_GAMMA":                      "Nourreddine 2026\nbatch GAMMA",
    "Norman2019_K562_CRISPRa":             "Norman 2019\nK562",
    "ProPerSeq_K562_CRISPRa":              "Swinderman 2026\nK562",
    "Replogle2020_K562_CRISPRi":           "Replogle 2020\nK562",
}


def _ps_split_half(name, top_n):
    """Top-N overlap of the two disjoint halves (the reproducibility gate)."""
    h = [sorted(glob.glob(f"{PS_RESULTS}/powerseek_control_delta_{name}_n*h{k}"
                          "_cv3000_floor0.1.csv")) for k in (1, 2)]
    if not (h[0] and h[1]):
        return np.nan
    a, b = (pd.read_csv(x[0], index_col=0).index[:top_n] for x in h)
    return len(set(a) & set(b)) / top_n


def powerseek_responsiveness(datasets=None, top_n=None, floor=EXPR_FLOOR, n_bins=50,
                             n_iter=100, metric="mean_abs_lfc", exclude=("Marson",),
                             labels=None, verbose=True):
    """(Suppl. Fig 7) Power-Seek memory genes against an expression-matched
    background, in every perturb-seq dataset the control-cell run covers.

    Reads the per-gene summaries that responsiveness.ipynb cached (mean |LFC| per
    gene over that dataset's knockouts, plus control expression) and the Power-Seek
    ranking of the full control-cell run. A dataset whose cache is missing is
    skipped with a message rather than silently dropped.

    Returns (res, q, table): the per-dataset result dicts in display order, the
    Benjamini-Hochberg q values over ALL rows (one family, not one per panel), and
    a tidy table.
    """
    _perturb_path()
    from marson_KO import compare_gene_responsiveness_expr_matched
    from scipy.stats import mannwhitneyu
    top_n = TOP_N if top_n is None else top_n
    labels = PS_LABELS if labels is None else labels

    # the dataset manifest is configuration and ships with the code
    DS = pd.read_csv(os.path.join(P.SRC, "generalisation", "datasets.tsv"),
                     sep="\t", dtype=str, keep_default_na=False)
    DS = DS[(DS.run_powerseek == "y") & (DS.repro_from == "-")]
    if datasets is not None:
        DS = DS[DS.name.isin(list(datasets))]
    elif exclude:
        DS = DS[~DS.name.str.startswith(tuple(exclude))]

    res_s, res_d, section, missing = {}, {}, {}, []
    for name, design in zip(DS.name, DS.design):
        for kind in (PS_DOUBLE_KINDS if design == "double" else ("all",)):
            tag = "" if kind == "all" else f"_{kind}"
            gs_p = f"{PS_CACHE}/{name}_gene_summary_{PS_PB_TAG}{tag}.csv"
            ps_p = f"{PS_RESULTS}/powerseek_control_delta_{name}_nall_cv3000_floor0.1.csv"
            if not (os.path.exists(gs_p) and os.path.exists(ps_p)):
                missing.append(name)
                break
            gs = pd.read_csv(gs_p, index_col=0)
            gs = gs[gs.ctrl_expr > floor].dropna(subset=[metric])   # floor first: the universe
            ranked = pd.read_csv(ps_p, index_col=0).index.astype(str)
            mem = [g for g in ranked if g in gs.index][:top_n]
            r = compare_gene_responsiveness_expr_matched(
                gs, gs["ctrl_expr"], mem, metric=metric, n_bins=n_bins, n_iter=n_iter)
            r.update(expression_balance(gs, gs["ctrl_expr"], mem, n_bins=n_bins,
                                        n_iter=n_iter, metric=metric))
            r.update(n_kos=int(gs.n_obs.max()), n_genes=len(gs),
                     split_half=_ps_split_half(name, top_n))
            # how big the screen was, for the generalisability line: perturbations of
            # THIS kind, and the dataset's cells (counted once per dataset)
            _pb = f"{PS_PB_DIR}/{name}_pseudobulk_{PS_PB_TAG}.npz"
            # perturbations = what the analysis actually used for THIS kind, which
            # is n_obs.max() from the cached summary. Counting labels in the
            # pseudobulk index cannot do it: only ProPer-seq separates paired
            # targets with '|', Norman uses 'A_B' and Replogle 2020 opaque ids that
            # are resolved from gene_A/gene_B, so a '|' test reports 0 doubles.
            r["n_perturbations"] = r["n_kos"]
            if os.path.exists(_pb):
                _z = np.load(_pb, allow_pickle=True)
                _idx, _sz = _z["index"], _z["sizes"]
                _is_ko = _idx != "control"
                r.update(n_cells=int(_sz.sum()),
                         n_control_cells=int(_sz[~_is_ko].sum()), dataset_name=name)
            else:
                r.update(n_cells=np.nan, n_control_cells=np.nan, dataset_name=name)
            lab = labels.get(name, name) + (", double" if kind == "double" else "")
            (res_d if kind == "double" else res_s)[lab] = r
            section[lab] = "double" if kind == "double" else "single"
    res = {**res_s, **res_d}        # single-perturbation rows first, then the doubles
    q = dict(zip(res, bh_fdr(np.array(
        [mannwhitneyu(v["scg_values"], v["non_scg_values"]).pvalue for v in res.values()]))))
    table = pd.DataFrame([dict(
        dataset=k.replace("\n", ", "), section=section[k],
        dataset_name=v.get("dataset_name"), knockouts=v["n_kos"],
        perturbations=v.get("n_perturbations"), cells=v.get("n_cells"),
        control_cells=v.get("n_control_cells"), genes_above_floor=v["n_genes"],
        memory_genes=len(v["scg_values"]),
        median_memory=float(np.median(v["scg_values"])),
        median_background=float(np.median(v["non_scg_values"])),
        pct_higher=100 * (np.median(v["scg_values"]) / np.median(v["non_scg_values"]) - 1),
        smd_expression=v["smd_expr"], split_half_overlap=v["split_half"], q=q[k])
        for k, v in res.items()])
    if verbose:
        if missing:
            print("not ready:", ", ".join(dict.fromkeys(missing)))
        print(f"{len(res)} rows from {DS.name.nunique()} datasets")
    return res, q, table


def fig_powerseek_summary(res=None, q=None, table=None, section=None, fs=7,
                          panels="both", orient="h", annot_fmt="{stars}  {pct:+.0f}%",
                          annot_dx=8, pad_right=.22, col_w=.42, fig_h=3.4,
                          xtick_rotation=35, annot_rotation=90, annot_row=False,
                          sort_by=None, ms=22, lw=1.2, legend_loc=None,
                          fig_w=7.2, row_h=.32, mem_color=None, bg_color=None,
                          mem_label=None, bg_label="expression-matched non-memory genes",
                          ylabel=None,
                          name="suppl7_powerseek_responsiveness",
                          save=True, dpi=400, show=True, verbose=True, **kw):
    """(Suppl. Fig 7) One row per perturb-seq dataset: the two medians as a dumbbell
    (left) and Cliff's delta with a bootstrap interval (right).

    Pass `res`/`q`/`table` from powerseek_responsiveness() to redraw without
    recomputing. Returns (fig, (ax_medians, ax_delta), table).
    """
    _perturb_path()
    from marson_KO import cliffs_delta, cliffs_ci
    if res is None:
        res, q, table = powerseek_responsiveness(verbose=verbose, **kw)
    mem_color = C_MEM if mem_color is None else mem_color
    bg_color = C_BG if bg_color is None else bg_color
    if mem_label is None:       # the cutoff is the notebook's, not a literal
        _n = (int(table.memory_genes.max()) if table is not None
              and "memory_genes" in table else TOP_N)
        mem_label = f"memory genes (derived from Power-Seek, top {_n})"
    sec = (dict(zip(table.dataset, table.section)) if section is None else section)

    S = pd.DataFrame([dict(dataset=k.replace("\n", ", "),
                           median_memory=float(np.median(r["scg_values"])),
                           median_background=float(np.median(r["non_scg_values"])),
                           delta=cliffs_delta(r["scg_values"], r["non_scg_values"]),
                           ci=cliffs_ci(r["scg_values"], r["non_scg_values"], n_boot=1000),
                           q=q[k])
                      for k, r in res.items()])
    S[["ci_low", "ci_high"]] = pd.DataFrame(S.pop("ci").tolist(), index=S.index)
    S["pct_higher"] = 100 * (S.median_memory / S.median_background - 1)
    S["section"] = [sec.get(d, "single") for d in S.dataset]
    if sort_by:         # e.g. "pct_higher": strongest first, within each section
        S = (S.assign(_sec=(S.section == "double").astype(int))
              .sort_values(["_sec", sort_by], ascending=[True, False])
              .drop(columns="_sec").reset_index(drop=True))
    gap = (S.section == "double").cumsum().clip(upper=1).values * 0.8   # space before the doubles
    y = -(np.arange(len(S)) + gap)                                      # first dataset on top

    _both = panels != "medians"
    if orient.startswith("v"):
        if _both:
            raise ValueError("orient='v' draws the medians panel only; "
                             "pass panels='medians'")
        # two short lines: the horizontal form's label is taller than this axes
        _metric_label = (ylabel if ylabel else
                         "median of mean absolute\nlog₂ fold change per gene")
        x = np.arange(len(S)) + (S.section == "double").cumsum().clip(upper=1).values * 0.8
        with plt.rc_context({**PUB_RC, "font.size": fs}):
            fig, ax_m = plt.subplots(figsize=(col_w * len(S) + 1.6, fig_h))
            ax_m.vlines(x, S.median_background, S.median_memory, color="0.78", lw=lw,
                        zorder=1)
            ax_m.scatter(x, S.median_background, s=ms, color=bg_color, zorder=2,
                         label=bg_label)
            ax_m.scatter(x, S.median_memory, s=ms, color=mem_color, zorder=3,
                         label=mem_label)
            # the statistics go ABOVE the higher marker, rotated: 18 columns leave no
            # room for horizontal text
            _top = float(S[["median_memory", "median_background"]].to_numpy().max())
            for xi, (mm, mb, qv, pc) in zip(
                    x, S[["median_memory", "median_background", "q", "pct_higher"]].values):
                # annot_row: every label on ONE line near the top, so the eye reads
                # them as a row of numbers instead of chasing the markers
                _y, _dy = ((_top, annot_dx) if annot_row else (max(mm, mb), annot_dx))
                ax_m.annotate(annot_fmt.format(stars=_stars(qv), pct=pc),
                              (xi, _y), xytext=(0, _dy),
                              textcoords="offset points", ha="center", va="bottom",
                              rotation=annot_rotation, fontsize=fs - 1, color="0.15",
                              annotation_clip=False)
            ax_m.set_xticks(x, S.dataset, rotation=xtick_rotation, ha="right",
                            rotation_mode="anchor")
            ax_m.set_ylabel(_metric_label)
            # just enough room above the tallest marker for the row of numbers:
            # a taller axis only squeezes the red-grey gaps
            ax_m.set_ylim(0, float(S[["median_memory", "median_background"]]
                                   .to_numpy().max()) * (1 + pad_right))
            ax_m.set_xlim(-.8, x.max() + .8)
            ax_m.spines[["top", "right"]].set_visible(False)
            if (S.section == "double").any() and (S.section != "double").any():
                x_b = (x[S.section.values != "double"].max()
                       + x[S.section.values == "double"].min()) / 2
                ax_m.axvline(x_b, color="0.4", ls="--", lw=0.8)
                ax_m.text(x_b + .12, .99, "double perturbations",
                          transform=ax_m.get_xaxis_transform(), ha="left", va="top",
                          fontsize=fs, style="italic", color="0.3")
            if legend_loc:      # inside the axes, where the data leaves room
                ax_m.legend(loc=legend_loc, frameon=False, handletextpad=.4,
                            borderaxespad=.2, fontsize=fs)
            else:
                fig.legend(*ax_m.get_legend_handles_labels(), loc="upper center",
                           ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout(rect=(0, 0, 1, 1 if legend_loc else 0.94))
            save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
            if show:
                plt.show()
        return fig, (ax_m, None), S

    with plt.rc_context({**PUB_RC, "font.size": fs}):
        if _both:
            fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=(fig_w, row_h * len(S) + 1.1),
                                             sharey=True,
                                             gridspec_kw=dict(width_ratios=(1, 1.15)))
        else:
            # one panel: the dumbbells carry the statistics themselves
            fig, ax_m = plt.subplots(figsize=(fig_w, row_h * len(S) + 1.1))
            ax_d = None
        ax_m.hlines(y, S.median_background, S.median_memory, color="0.75", lw=1.2, zorder=1)
        ax_m.scatter(S.median_background, y, s=22, color=bg_color, zorder=2,
                     label=bg_label)
        ax_m.scatter(S.median_memory, y, s=22, color=mem_color, zorder=3,
                     label=mem_label)
        ax_m.set_xlabel("median of mean absolute log₂ fold change\n"
                        "per gene (across all knockouts)")
        ax_m.set_yticks(y, S.dataset)
        ax_m.set_xlim(left=0)
        if not _both:
            # stars and the median difference, past the rightmost of the two markers
            for yi, (mm, mb, qv, pc) in zip(
                    y, S[["median_memory", "median_background", "q", "pct_higher"]].values):
                ax_m.annotate(annot_fmt.format(stars=_stars(qv), pct=pc),
                              (max(mm, mb), yi), xytext=(annot_dx, 0),
                              textcoords="offset points", ha="left", va="center",
                              fontsize=fs - 1, color="0.15", annotation_clip=False)
            _x1 = float(S[["median_memory", "median_background"]].to_numpy().max())
            ax_m.set_xlim(0, _x1 * (1 + pad_right))     # room for the annotations
        fig.legend(*ax_m.get_legend_handles_labels(), loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        if ax_d is not None:
            ax_d.axvline(0, color="0.6", lw=0.8, ls="--", zorder=0)
            ax_d.errorbar(S.delta, y, xerr=[S.delta - S.ci_low, S.ci_high - S.delta],
                          fmt="o", ms=4.5, color=mem_color, ecolor=mem_color,
                          elinewidth=1, capsize=2)
            for yi, (d, hi, qv, pc) in zip(
                    y, S[["delta", "ci_high", "q", "pct_higher"]].values):
                ax_d.text(hi + 0.03, yi, f"{_stars(qv)}  {pc:+.0f}%", va="center",
                          fontsize=fs - 1)
            ax_d.set_xlabel("Cliff's delta, memory genes vs background\n"
                            "(95% bootstrap interval)")
            ax_d.set_xlim(min(-0.1, S.ci_low.min() - 0.05), 1.15)
            ax_d.tick_params(left=False)
        if (S.section == "double").any() and (S.section != "double").any():
            y_b = (y[S.section.values != "double"].min()
                   + y[S.section.values == "double"].max()) / 2
            for ax in (ax_m, ax_d):
                if ax is not None:
                    ax.axhline(y_b, color="0.4", ls="--", lw=0.8)
            ax_m.text(0.01, y_b - 0.12, "double perturbations",
                      transform=ax_m.get_yaxis_transform(), ha="left", va="top",
                      fontsize=fs, style="italic", color="0.3")
        fig.tight_layout(w_pad=4.0 if _both else 1.0, rect=(0, 0, 1, 0.96))
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, (ax_m, ax_d), S


# what each Power-Seek dataset was run in. Wong 2026 is left unnamed on purpose:
# fill it in rather than guess, since it decides how the panel groups.
PS_CELLTYPE = {
    "ReplogleWeissman2022_K562_essential": "K562",
    "ReplogleWeissman2022_K562_gwps": "K562",
    "Norman2019_K562_CRISPRa": "K562",
    "ProPerSeq_K562_CRISPRa": "K562",
    "Replogle2020_K562_CRISPRi": "K562",
    "ReplogleWeissman2022_rpe1": "RPE1",
    "ReplogleWeissman2025_HepG2": "HepG2",
    "ReplogleWeissman2025_Jurkat": "Jurkat",
    "XAtlas2025_HCT116": "HCT116",
    "XAtlas2025_HEK293T": "HEK293T",
    "Mali2026_ALPHA": "hiPSC", "Mali2026_BETA": "hiPSC", "Mali2026_GAMMA": "hiPSC",
    "Wong2026_Untreated": "Wong", "Wong2026_Treated": "Wong",
}


def fig_powerseek_compact(table=None, res=None, q=None, dot_color=None, ref_color="0.6",
                          jitter=.085, ms=26, seed=0, show_double=True, median=True,
                          summary=True, summary_fmt=("{n_ds} datasets\n"
                                                     "{n_pert:,} perturbations\n"
                                                     "{n_cells} cells"),
                          summary_xy=(.02, .98), summary_ha="left", summary_va="top",
                          xlim=(-.6, .6), med_x=.30, group_by=None, group_w=.62,
                          divider_color="0.8", xtick_rotation=35,
                          examples=False,
                          example_labels=("single-perturb", "double-perturb"),
                          example_offsets=((12, -6), (10, 10)),
                          example_color="0.25", example_lw=.6,
                          ylabel="% higher median responsiveness\nthan matched background",
                          xlabel="perturb-seq datasets", med_lw=1.6, fs=7,
                          figsize=(2.4, 2.2), name="fig2_powerseek_compact",
                          save=True, dpi=400, show=True, verbose=True, **kw):
    """(figure-sized) The 15-dataset Power-Seek result as ONE column of points.

    Suppl. Fig 7 names every dataset; this is the same numbers compressed to what a
    main figure can carry: one point per dataset (open = the paired-perturbation
    rows), the median across them, and how much data stands behind it.

    Pass `table` from powerseek_responsiveness() to avoid recomputing.
    """
    if table is None:
        res, q, table = powerseek_responsiveness(verbose=verbose, **kw)
    dot_color = C_OBS if dot_color is None else dot_color
    T = table if show_double else table[table.section != "double"]
    rng = np.random.default_rng(seed)
    single = (T.section != "double").to_numpy()
    _groups, _centres = None, None
    if group_by:
        # one slot per cell type, points jittered inside it, dashed dividers between
        g = (T.dataset_name.map(PS_CELLTYPE).fillna("other") if group_by == "celltype"
             else T[group_by])
        _groups = list(dict.fromkeys(g))
        _centres = {k: i for i, k in enumerate(_groups)}
        x = g.map(_centres).to_numpy(float) + rng.uniform(-jitter, jitter, len(T))
        xlim = (-.6, len(_groups) - .4)
    else:
        x = rng.uniform(-jitter, jitter, len(T))

    with plt.rc_context({**PUB_RC, "font.size": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs, "ytick.labelsize": fs}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color=ref_color, lw=.8, ls=(0, (3, 2)), zorder=1)
        ax.scatter(x[single], T.pct_higher[single], s=ms, color=dot_color,
                   edgecolor="white", linewidth=.5, zorder=3, label="one dataset")
        if (~single).any():
            ax.scatter(x[~single], T.pct_higher[~single], s=ms, facecolor="white",
                       edgecolor=dot_color, linewidth=1.1, zorder=3,
                       label="paired perturbations")
        if median:
            med = float(np.median(T.pct_higher))
            ax.hlines(med, -med_x, med_x, color="0.15", lw=med_lw, zorder=4)
            ax.annotate(f"median {med:.0f}%", (med_x, med), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=fs - 1, fontweight="bold", color="0.15",
                        annotation_clip=False)
        if summary:
            # the generalisability line: datasets, perturbations, cells. Cells are
            # counted ONCE per dataset, since a paired-perturbation row re-uses them.
            _u = table.drop_duplicates("dataset_name")
            n_cells = float(_u.cells.sum())
            txt = summary_fmt.format(
                n_ds=int(_u.dataset_name.nunique()),
                n_pert=int(table.perturbations.sum()),
                n_cells=(f"{n_cells / 1e6:.1f} million" if n_cells >= 1e6
                         else f"{n_cells:,.0f}"))
            ax.text(*summary_xy, txt, transform=ax.transAxes, ha=summary_ha,
                    va=summary_va, fontsize=fs - 1, color="0.3", linespacing=1.25)
        if examples and len(T):
            # name the two marker kinds where they are, instead of in a legend:
            # a leader line to one filled and one open point
            _p = T.pct_higher.to_numpy(float)
            _pick = [(np.flatnonzero(single)[np.argmax(_p[single])]
                      if single.any() else None),
                     (np.flatnonzero(~single)[np.argmax(_p[~single])]
                      if (~single).any() else None)]
            for _i, _lab, _off in zip(_pick, example_labels, example_offsets):
                if _i is None or not _lab:
                    continue
                ax.annotate(_lab, (x[_i], _p[_i]), xytext=_off,
                            textcoords="offset points",
                            ha="right" if _off[0] < 0 else "left", va="center",
                            fontsize=fs - 1, color=example_color,
                            arrowprops=dict(arrowstyle="-", lw=example_lw,
                                            color=example_color,
                                            shrinkA=1, shrinkB=4),
                            annotation_clip=False, zorder=6)
        if _groups:
            for i in range(len(_groups) - 1):
                ax.axvline(i + .5, color=divider_color, lw=.7, ls=(0, (3, 2)),
                           zorder=0)
            ax.set_xticks(range(len(_groups)), _groups, rotation=xtick_rotation,
                          ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticks([])
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, ax, T


# ──────────────────────────── strongest responders ─────────────────────────────
# The CONVERSE of figures 2c-g. Those ask whether memory genes respond more than
# matched controls — a shift in the middle of the distribution. This asks whether
# the genes that respond MOST STRONGLY are enriched for memory genes — the tail.
# The two can disagree: a broad shift moves the median without putting memory genes
# in the extreme tail, and a tail-driven effect would mean figure 2 rests on a few
# genes.
#
# mean_abs_lfc rises with expression and with how deeply the screen was sampled, so
# the null CANNOT be hypergeometric: the drawn sets reproduce the memory set's own
# expression composition (the same 50 equal-count bins as every other matched panel)
# and come from the same never-called background pool as figure 2.
TR_CUTS = (100, 300, 500, 1000)


def top_responder_enrichment(datasets=None, cuts=TR_CUTS, min_methods=2, top_n=None,
                             metric="mean_abs_lfc", n_bins=50, n_draws=1000,
                             rank_universe="all", floor=EXPR_FLOOR,
                             timepoint="merge", merge_mode="union", methods=None,
                             bg_mode="top_n", bg_exclude=BG_EXCLUDE_DEFAULT,
                             min_genes=10, seed=0, cache=True, verbose=True):
    """Are the strongest-responding genes enriched for consensus memory genes?

    Per screen: rank the expressed genes by `metric`, call the top `cut` the
    strongest responders, and ask how many memory genes (called by >= min_methods
    of the four methods at top_n) land there, against an expression-matched null.

    rank_universe : 'all'  rank every expressed gene (the plain reading of
                           "which genes respond most strongly").
                    'pool' rank only memory genes + the background pool, i.e. drop
                           the genes called by exactly one method.

    The null draws n_draws background sets with the memory set's per-bin counts,
    WITH replacement inside a bin (the bins hold a few memory genes against a pool
    one to two orders of magnitude larger, so collisions are rare; the figure-2
    sampler draws without replacement where the bin allows it, which is the only
    difference between them).

    Returns a long frame: one row per (screen, cut) with the observed count, the
    null mean and its 2.5/97.5 percentiles, the fold, an upper-tail empirical p,
    the share of the top-N that is memory genes, and the achieved |SMD| on log10
    expression so the matching can be checked rather than assumed.
    """
    _perturb_path()
    from marson_KO import build_consensus_gene_sets
    datasets = datasets if datasets is not None else xd_datasets()
    top_n = TOP_N if top_n is None else top_n
    key = ("top_resp", tuple(d["name"] for d in datasets), tuple(cuts), min_methods,
           top_n, metric, n_bins, n_draws, rank_universe, floor, timepoint,
           merge_mode, tuple(methods or ()), bg_mode, tuple(bg_exclude or ()), seed)
    if cache and key in _CACHE:
        return _CACHE[key]

    rows = []
    for ds in datasets:
        gs = pd.read_csv(ds["gs"], index_col=0)
        expr = pd.read_csv(ds["expr"], index_col=0).iloc[:, 0]
        # same expression rule as every other matched panel (2026-09-16): floor the
        # universe FIRST, so the ranking, the memory set and the background pool all
        # live in the same expressed universe
        if floor is not None:
            keep = expr.index[expr > floor]
            gs = gs.loc[gs.index.intersection(keep)]
            expr = expr.loc[expr.index.intersection(gs.index)]
        tp = ds.get("timepoint", timepoint)
        mm = ds.get("merge_mode", merge_mode)
        universe, to_data = _xd_symbol_bridge(ds, gs)

        want = methods if methods is not None else XD_METHODS_CONSENSUS
        files = {m: f for m, f in ds["files"].items() if m in want}
        _, counts = build_consensus_gene_sets(
            min_methods=1, timepoint=tp, top_n=top_n, merge_mode=mm,
            memgene_dir=ds["memdir"], method_files=files, universe=universe,
            verbose=False)
        bg = xd_background(ds, universe, to_data, tp, mm, bg_mode, top_n,
                           files=files, bg_exclude=bg_exclude, verbose=False)
        bgset = set(bg)
        mem = [g for g in to_data(counts.index[counts >= min_methods].tolist())
               if g not in bgset]
        if len(mem) < min_genes:
            if verbose:
                print(f"  skip {ds['name'].splitlines()[0]}: only {len(mem)} genes")
            continue

        # ---- the ranking: who responds most strongly in this screen ----
        rank = gs[metric].dropna().sort_values(ascending=False)
        if rank_universe == "pool":
            rank = rank.loc[rank.index.intersection(list(dict.fromkeys(mem + bg)))]
        rk = pd.Series(np.arange(1, len(rank) + 1), index=rank.index, dtype=float)

        # ---- the matched null, drawn on the SAME frame as figure 2's comparison ----
        frame = (pd.DataFrame({"value": gs[metric]})
                 .join(expr.rename("expr"), how="inner").dropna())
        frame = frame.loc[frame.index.intersection(list(dict.fromkeys(mem + bg)))]
        frame["is_mem"] = frame.index.isin(set(mem))
        frame["bin"] = pd.qcut(frame["expr"].rank(method="first"), n_bins, labels=False)
        rk_all = rk.reindex(frame.index).to_numpy(float)
        rk_all = np.where(np.isnan(rk_all), np.inf, rk_all)   # unranked: never a hit
        le_all = np.log10(frame["expr"].to_numpy(float))
        is_m, bn = frame["is_mem"].to_numpy(), frame["bin"].to_numpy()

        rng = np.random.default_rng(seed)
        blocks = []
        for b in np.unique(bn):
            need = int(((bn == b) & is_m).sum())
            pool = np.flatnonzero((bn == b) & ~is_m)
            if need == 0 or pool.size == 0:
                continue
            blocks.append(rng.choice(pool, size=(n_draws, need), replace=True))
        if not blocks:
            continue
        pos = np.concatenate(blocks, axis=1)
        draw_rk, draw_le = rk_all[pos], le_all[pos]
        mem_pos = np.flatnonzero(is_m)
        mem_rk, mem_le = rk_all[mem_pos], le_all[mem_pos]
        smd = float((mem_le.mean() - draw_le.mean())
                    / np.sqrt((mem_le.var() + draw_le.var()) / 2))

        lin, scr, _ = _xd_pair_parts(ds["name"])
        for c in cuts:
            k = int(round(c * len(rank))) if 0 < c < 1 else int(c)
            k = min(k, len(rank))
            obs = int((mem_rk <= k).sum())
            null = (draw_rk <= k).sum(1).astype(float)
            rows.append(dict(
                dataset=ds["name"], lineage=lin, screen=scr, cut=c, k=k,
                n_universe=len(rank), n_mem=int(len(mem_rk)), n_draw=pos.shape[1],
                n_bg=len(bg), obs=obs, obs_frac=obs / len(mem_rk),
                null_mean=null.mean(),
                null_lo=float(np.percentile(null, 2.5)),
                null_hi=float(np.percentile(null, 97.5)),
                fold=obs / null.mean() if null.mean() > 0 else np.nan,
                p=(1 + int((null >= obs).sum())) / (n_draws + 1),
                pct_of_top=100 * obs / k, smd_expr=smd))
        if verbose:
            f = [r["fold"] for r in rows if r["dataset"] == ds["name"]]
            print(f"  {scr:10s} n_mem={len(mem_rk):4d}  fold "
                  + " ".join(f"{x:.2f}" for x in f)
                  + f"   SMD(expr)={smd:+.3f}"
                  + ("  <-- NOT balanced" if abs(smd) > .10 else ""))
    out = pd.DataFrame(rows)
    if cache:
        _CACHE[key] = out
    return out


# =============================================================================
# Fig 1f -- molecular features: mRNA half-life, transcriptional bursting and
#           promoter CpG content, memory vs non-memory genes, per group
# Fig 1g -- expression autocorrelation across cell divisions (clone-size decay)
#
# Ported from the original analysis script (mg_features_autocorrelation.py). The statistics and the
# plotting are hers and unchanged. The INPUTS were changed to match the rest of the
# paper, and each change can move the figure:
#   cells     the processed h5ads in DATASETS (QC-filtered, unassigned cells
#             removed, log1p CP10k). The original read LSK_d2_d4_d6.h5ad for
#             Weinreb, where 65.9% of cells carry clone_id -1; the decay fit treated
#             those as ONE clone (7,061 cells at day 6), which lands in the largest
#             size bin. It also rebuilt Michaels from the raw 10x matrices without QC.
#   memory    per-group consensus built here from the current per-method lists:
#             each method cut to its own top N, memory = >= 2 methods, non-memory =
#             in no method's top N (the paper's per-group mode). The original read a
#             17 Sep table that predates the seed-0 ClonePair lists for Michaels and
#             Weinreb and the per-method depth rule.
#   bursting  mouse values (Boe et al.) carried to human genes through the NCBI
#             one-to-one orthologue maps, as Methods state. The original matched
#             the uppercased mouse symbol.
#   tables    half-life and bursting rows are de-duplicated by gene before joining;
#             duplicated names would otherwise enter the test twice.
# =============================================================================
MOLFEAT_AGARWAL = f"{P.DATA}/agarwal_half_life_table.xlsx"
MOLFEAT_BURSTING = f"{P.DATA}/boe_bursting_mmc3.xlsx"     # Boe et al., supplementary table mmc3
MOLFEAT_CPG = os.path.join(os.environ.get("PERM_FEATURES",
                                          os.path.join(P.OUTPUTS, "gene_properties")),
                           "memory_gene_cpg_per_gene.csv")
MOLFEAT_GROUP_COL = {"T-lineage": "batch", "Harmange": "startID",
                     "LARRY": "Time point", "Bandler": "batch"}
MOLFEAT_CLONE_COL = {"Bandler": "cloneID"}               # everyone else: clone_id
MOLFEAT_PAPER = {"T-lineage": "Michaels", "Harmange": "Harmange",
                 "LARRY": "Weinreb", "Bandler": "Bandler"}
MOLFEAT_GROUP_KEY = {"2.0": "T1", "4.0": "T2", "6.0": "T3"}   # Weinreb days -> timepoints
MOLFEAT_METRICS = ["half-life (PC1)", "kon_C57", "koff_C57", "promoter_cpg_obs_exp_500bp"]
MOLFEAT_LABELS = ["mRNA half-life", "Burst frequency", "Burst length", "Promoter CpG"]
MOLFEAT_DATASET_LABEL = {"Michaels": "T cells", "Harmange": "Melanoma",
                         "Weinreb": "HSPCs", "Bandler": "Brain"}
MOLFEAT_GROUP_LABEL = {"T1": "Timepoint 1", "T2": "Timepoint 2", "T3": "Timepoint 3",
                       "E": "Enriched", "M": "Mixed",
                       "E13": "Embryonic\nday 13", "E14": "Embryonic\nday 14"}


def pergroup_method_lists(ds, top_n=None):
    """{group: {method: [genes]}}: each method's list for one group, cut to its own
    top N, built exactly as build_consensus.py builds its per-group rows."""
    top_n = TOP_N if top_n is None else top_n
    _perturb_path()
    from marson_KO import load_memory_gene_set
    d = _resolve_dir(DATASETS[ds]["folder"], RESULTS_TREE, verbose=False)
    suf = _resolve_suffix(d)
    v = scg_variant_for(ds, None)
    files = {m: f"{p}_memory_genes_{v if m == 'ClonePair' else ''}{suf}.csv"
             for m, p in METHOD_PREFIX.items()}
    # native values (Weinreb's are floats); load_memory_gene_set matches on them
    tps = sorted(set.intersection(*(
        set(pd.read_csv(os.path.join(d, f), usecols=["timepoint"]).timepoint)
        for f in files.values())), key=str)
    return {str(tp): {m: list(load_memory_gene_set(m, timepoint=tp, top_n=top_n,
                                                   memgene_dir=d, method_files=files))
                      for m in files}
            for tp in tps}


def pergroup_memory(ds, top_n=None, min_methods=2):
    """{group: (memory genes, genes in any method's top N)} for one dataset."""
    out = {}
    for g, lists in pergroup_method_lists(ds, top_n).items():
        c = Counter(x for lst in lists.values() for x in set(lst))
        out[g] = ({x for x, n in c.items() if n >= min_methods}, set(c))
    return out


def molecular_feature_table(datasets=None, top_n=None, n_bins=10, max_cells=10_000,
                            seed=42, verbose=True):
    """One row per (dataset, group, gene) for memory and non-memory genes, with the
    expression bins and the four Fig 1f features joined on."""
    import anndata as ad
    import scipy.sparse as _sp
    datasets = active_datasets(datasets)
    hl = {sp: (pd.read_excel(MOLFEAT_AGARWAL, sheet_name=sp, header=2)
                 [["Gene name", "half-life (PC1)"]]
                 .dropna(subset=["Gene name"]).drop_duplicates("Gene name"))
          for sp in ("human", "mouse")}
    burst = (pd.read_excel(MOLFEAT_BURSTING, header=0)[["Gene", "kon_C57", "koff_C57"]]
               .rename(columns={"Gene": "gene_mouse"})
               .dropna(subset=["gene_mouse"]).drop_duplicates("gene_mouse"))
    # human -> mouse, inverted from the NCBI mouse -> human maps; one-to-one only
    m2h = {}
    for lab, cfg in DATASETS.items():
        if cfg.get("species") == "mouse":
            m2h.update(ortho_map(lab, verbose=False))
    inv = pd.Series(m2h, name="gene_human").rename_axis("gene_mouse").reset_index()
    inv = inv[~inv.gene_human.duplicated(keep=False)]
    h2m = dict(zip(inv.gene_human, inv.gene_mouse))
    cpg = pd.read_csv(MOLFEAT_CPG, usecols=["dataset", "gene_symbol_in_dataset",
                                            "promoter_cpg_obs_exp_500bp"])

    frames = []
    for lab in datasets:
        paper, sp = MOLFEAT_PAPER[lab], DATASETS[lab]["species"]
        A = ad.read_h5ad(DATASETS[lab]["h5ad"])
        grp = A.obs[MOLFEAT_GROUP_COL[lab]].astype(str).to_numpy()
        sets = pergroup_memory(lab, top_n)
        for g in sorted(set(grp)):
            if g not in sets:
                continue
            Ag = A[grp == g]
            det = np.asarray((Ag.X > 0).mean(axis=0)).ravel()
            keep = det > 0.01
            rng = np.random.default_rng(seed)
            idx = rng.choice(Ag.n_obs, size=min(max_cells, Ag.n_obs), replace=False)
            Xs = Ag.X[idx][:, keep]
            Xs = Xs.toarray() if _sp.issparse(Xs) else np.asarray(Xs)
            df = pd.DataFrame({"gene": np.asarray(Ag.var_names)[keep],
                               "median_expr": np.median(Xs, axis=0),
                               "mean_expr": Xs.mean(axis=0)})
            df["expr_bin_median"] = pd.qcut(df.median_expr, n_bins, labels=False,
                                            duplicates="drop")
            df["expr_bin_mean"] = pd.qcut(df.mean_expr, n_bins, labels=False,
                                          duplicates="drop")
            mem, called = sets[g]
            df["memory"] = df.gene.isin(mem)
            df["non-memory"] = ~df.gene.isin(called)
            df = df[df["memory"] | df["non-memory"]]
            df = df.merge(hl[sp], left_on="gene", right_on="Gene name", how="left")
            df["gene_mouse"] = (df.gene.map(lambda x: h2m.get(human_symbol(x)))
                                if sp == "human" else df.gene)
            df = df.merge(burst, on="gene_mouse", how="left")
            df = df.merge(cpg[cpg.dataset == paper]
                          .drop_duplicates("gene_symbol_in_dataset")
                          [["gene_symbol_in_dataset", "promoter_cpg_obs_exp_500bp"]],
                          left_on="gene", right_on="gene_symbol_in_dataset", how="left")
            df["dataset"], df["group"] = paper, MOLFEAT_GROUP_KEY.get(g, g)
            frames.append(df)
            if verbose:
                print(f"  {paper:9s} {MOLFEAT_GROUP_KEY.get(g, g):4s} memory "
                      f"{int(df.memory.sum()):4d}  non-memory {int(df['non-memory'].sum()):6d}"
                      f"  | with half-life {int(df['half-life (PC1)'].notna().sum()):6d}"
                      f"  bursting {int(df.kon_C57.notna().sum()):5d}"
                      f"  CpG {int(df.promoter_cpg_obs_exp_500bp.notna().sum()):6d}")
    return pd.concat(frames, ignore_index=True)


def expression_matched_permutation_test(summary, metric, expr_bin="expr_bin_mean",
                                        sister_col="memory", n_permutations=10_000,
                                        random_state=42, drop_unmatched_bins=False):
    """Difference in mean `metric` between memory and all remaining genes, against a
    null that redraws the memory label within expression bins with the memory set's
    own per-bin composition. The null therefore absorbs whatever difference the
    expression profile alone produces; the effect is observed_diff - null_center.
    drop_unmatched_bins discards bins without memory genes, which could never receive
    a permuted label. Two-sided p measures deviation from null_center.
    """
    rng = np.random.default_rng(random_state)
    df = summary.dropna(subset=[metric, expr_bin]).reset_index(drop=True)
    sister_mask = df[sister_col].fillna(False).astype(bool).to_numpy()
    values = df[metric].to_numpy(dtype=float)
    codes = pd.factorize(df[expr_bin])[0]

    groups, counts = [], []
    for c in range(codes.max() + 1):
        idx = np.flatnonzero(codes == c)
        k = int(sister_mask[idx].sum())
        if k == 0 and drop_unmatched_bins:
            continue
        groups.append(idx)
        counts.append(k)
    if drop_unmatched_bins:
        keep = np.sort(np.concatenate(groups))
        values, sister_mask = values[keep], sister_mask[keep]
        remap = np.full(len(df), -1)
        remap[keep] = np.arange(len(keep))
        groups = [remap[g] for g in groups]

    n_sister = int(sister_mask.sum())
    n_control = values.size - n_sister
    if n_sister == 0 or n_control == 0:
        raise ValueError(f"{metric}: need both groups non-empty "
                         f"(n_sister={n_sister}, n_control={n_control}).")
    obs_s, obs_c = values[sister_mask].mean(), values[~sister_mask].mean()
    observed_diff = obs_s - obs_c

    total = values.sum()
    null_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        s = 0.0
        for idx, k in zip(groups, counts):
            if k:
                s += values[rng.choice(idx, size=k, replace=False)].sum()
        null_diffs[i] = s / n_sister - (total - s) / n_control

    null_center = null_diffs.mean()
    return {
        "metric": metric, "n_sister": n_sister, "n_control": n_control,
        "n_bins": len(groups),
        "observed_sister_mean": obs_s, "observed_control_mean": obs_c,
        "observed_diff": observed_diff, "null_diffs": null_diffs,
        "null_center": null_center,
        "observed_diff_centered": observed_diff - null_center,
        "null_diffs_centered": null_diffs - null_center,
        "p_greater": (np.sum(null_diffs >= observed_diff) + 1) / (n_permutations + 1),
        "p_less": (np.sum(null_diffs <= observed_diff) + 1) / (n_permutations + 1),
        "p_two_sided": (np.sum(np.abs(null_diffs - null_center)
                               >= abs(observed_diff - null_center)) + 1)
                       / (n_permutations + 1),
    }


def molecular_feature_tests(table, metrics=MOLFEAT_METRICS, n_permutations=10_000,
                            random_state=42, drop_unmatched_bins=True):
    """{dataset: {group: {metric: result}}} plus a tidy stats table."""
    results, rows = {}, []
    for dataset in table.dataset.unique():
        df = table[table.dataset == dataset]
        results[dataset] = {}
        for group in df.group.unique():
            results[dataset][group] = {}
            for metric in metrics:
                r = expression_matched_permutation_test(
                    df[df.group == group], metric, sister_col="memory",
                    n_permutations=n_permutations, random_state=random_state,
                    drop_unmatched_bins=drop_unmatched_bins)
                results[dataset][group][metric] = r
                sd = np.asarray(r["null_diffs"]).std(ddof=1)
                rows.append(dict(dataset=dataset, group=group, metric=metric,
                                 label=MOLFEAT_LABELS[MOLFEAT_METRICS.index(metric)]
                                 if metric in MOLFEAT_METRICS else metric,
                                 n_memory=r["n_sister"], n_nonmemory=r["n_control"],
                                 memory_mean=r["observed_sister_mean"],
                                 nonmemory_mean=r["observed_control_mean"],
                                 observed_diff=r["observed_diff"],
                                 null_center=r["null_center"],
                                 z=(r["observed_diff"] - r["null_center"]) / sd if sd > 0 else np.nan,
                                 p_two_sided=r["p_two_sided"]))
    return results, pd.DataFrame(rows)


def fig_molecular_features(results=None, table=None, datasets=None, top_n=None,
                           n_permutations=10_000, cmap="bwr", vmax=None, ns_white=True,
                           alpha=0.05, col_width=0.5, row_height=0.42, left=0.42,
                           name="fig1f_molecular_features", save=True, dpi=400,
                           show=True, verbose=True):
    """(Fig 1f) Heatmap of z (memory - non-memory, against the expression-matched
    null) per group and feature; non-significant cells white, stars = empirical
    two-sided p. Returns (fig, ax, stats)."""
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm
    if results is None:
        if table is None:
            table = molecular_feature_table(datasets=datasets, top_n=top_n, verbose=verbose)
        results, stats = molecular_feature_tests(table, n_permutations=n_permutations)
    else:
        stats = None
    order = [MOLFEAT_PAPER[d] for d in active_datasets(datasets) if MOLFEAT_PAPER[d] in results]
    rows = [(d, g) for d in order for g in results[d]]
    metrics = MOLFEAT_METRICS
    Z = np.full((len(rows), len(metrics)), np.nan)
    Pv = np.full((len(rows), len(metrics)), np.nan)
    for i, (d, g) in enumerate(rows):
        for j, m in enumerate(metrics):
            r = results[d][g][m]
            sd = np.asarray(r["null_diffs"], float).std(ddof=1)
            if sd > 0:
                Z[i, j] = (r["observed_diff"] - np.mean(r["null_diffs"])) / sd
            Pv[i, j] = r["p_two_sided"]

    def sig_label(p):
        return ("∗∗∗" if p < 0.001 else "∗∗" if p < 0.01
                else "∗" if p < 0.05 else "")

    vmax = vmax if vmax is not None else np.nanmax(np.abs(Z))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cm = mpl.colormaps[cmap]
    colours = cm(norm(np.nan_to_num(Z)))
    if ns_white:
        colours[Pv >= alpha] = (1, 1, 1, 1)
    colours[np.isnan(Z)] = (0.9, 0.9, 0.9, 1)

    with plt.rc_context({"svg.fonttype": "none"}):
        n_rows, n_cols = Z.shape
        fig, ax = plt.subplots(figsize=(col_width * n_cols + 3.4, row_height * n_rows + 1.8))
        ax.imshow(colours, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(1, n_cols) - 0.5, minor=True)
        ax.set_yticks(np.arange(1, n_rows) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for i in range(n_rows):
            for j in range(n_cols):
                lab = sig_label(Pv[i, j])
                if not lab:
                    continue
                r_, g_, b_, _ = colours[i, j]
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax.text(j, i, lab, ha="center", va="center", fontsize=10,
                        color="white" if lum < 0.5 else "black")
        ax.set_xticks(np.arange(n_cols))
        ax.xaxis.tick_top()
        ax.set_xticklabels(MOLFEAT_LABELS, fontsize=10, rotation=45, ha="left",
                           rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([MOLFEAT_GROUP_LABEL.get(g, g) for _, g in rows], fontsize=10)
        ax.tick_params(axis="y", length=0)
        start = 0
        for d in order:
            if start > 0:
                ax.axhline(start - 0.5, color="black", linewidth=0.8)
            start += len(results[d])
        for s in ax.spines.values():
            s.set_visible(False)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03, extend="neither")
        cbar.set_label("Standardised difference\n(memory − non-memory, z vs null)",
                       fontsize=10)
        cbar.outline.set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(left=left)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w_pt = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels()) * 72 / fig.dpi
        start = 0
        for d in order:
            n = len(results[d])
            ax.annotate(MOLFEAT_DATASET_LABEL.get(d, d), xy=(0, start + (n - 1) / 2),
                        xycoords=ax.get_yaxis_transform(), xytext=(-(w_pt + 10), 0),
                        textcoords="offset points", ha="right", va="center",
                        rotation=90, fontsize=11, annotation_clip=False)
            start += n
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, ax, stats


def _icc(X, codes, K):
    """One-way random-effects ICC per gene, unequal clone sizes (n0 correction)."""
    import scipy.sparse as _sp
    N, G = X.shape
    if K < 2 or N - K < 1:
        return np.full(G, np.nan, np.float32)
    oh = _sp.csr_matrix((np.ones(N), (np.arange(N), codes)), shape=(N, K))
    n = np.asarray(oh.sum(0)).ravel()
    S = np.asarray(oh.T @ X)
    g = X.sum(0)
    sq = (X.astype(float) ** 2).sum(0)
    ssb = (S ** 2 / n[:, None]).sum(0) - g ** 2 / N
    ssw = np.clip(sq - g ** 2 / N - ssb, 0, None)
    msb, msw = ssb / (K - 1), ssw / (N - K)
    n0 = (N - (n ** 2).sum() / N) / (K - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
    icc[~np.isfinite(icc)] = np.nan
    return icc.astype(np.float32)


def compute_decay(adata, clone_key, cell_mask=None, edges=(2, 3, 5, 9, 17, 33, np.inf),
                  min_clones=8, floor=1e-3, icc_gate=0.05, min_bins=3):
    """ICC per gene within clone-size bins, and a per-gene decay slope.

    Returns (meta, icc_bins, bin_x, w): meta has gene, pooled icc, slope, tau_gen and
    `reliable` (ICC > icc_gate in the small-clone bins and measured in >= min_bins
    bins; slope/tau_gen are NaN otherwise). adata.X must be log-normalised.
    """
    import scipy.sparse as _sp
    X = adata.X if cell_mask is None else adata.X[cell_mask]
    clones = adata.obs[clone_key].astype(str).to_numpy()
    if cell_mask is not None:
        clones = clones[cell_mask]
    size = pd.Series(clones).map(pd.Series(clones).value_counts()).to_numpy()
    keep = size >= 2
    Xk = np.asarray(_sp.csr_matrix(X[keep]).todense(), np.float32)
    clones, size = clones[keep], size[keep]
    log2sz = np.log2(size)
    icc_all = _icc(Xk, pd.Categorical(clones, np.unique(clones)).codes,
                   np.unique(clones).size)
    xs, ws, cols = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (size >= lo) & (size < hi)
        uu = np.unique(clones[m])
        if uu.size < min_clones:
            continue
        cols.append(_icc(Xk[m], pd.Categorical(clones[m], uu).codes, uu.size))
        xs.append(log2sz[m].mean())
        ws.append(uu.size)
    icc_bins = np.vstack(cols).T
    bin_x, w = np.array(xs), np.array(ws, float)
    Y = np.log(np.clip(icc_bins.T, floor, None))
    xb = np.average(bin_x, weights=w)
    yb = np.average(Y, 0, weights=w)
    slope = (w * (bin_x - xb)) @ (Y - yb) / np.sum(w * (bin_x - xb) ** 2)
    with np.errstate(divide="ignore"):
        tau_gen = -1.0 / slope
    tau_gen[slope >= 0] = np.inf
    small = bin_x <= np.median(bin_x)
    signal = np.where(np.isfinite(icc_bins) & small, icc_bins, -np.inf).max(axis=1)
    n_ok = np.isfinite(icc_bins).sum(1)
    reliable = (signal > icc_gate) & (n_ok >= min_bins) & np.isfinite(slope)
    meta = pd.DataFrame({"gene": np.asarray(adata.var_names), "icc": icc_all,
                         "slope": slope, "tau_gen": tau_gen, "reliable": reliable})
    meta.loc[~reliable, ["slope", "tau_gen"]] = np.nan
    return meta, icc_bins, bin_x, w


def plot_decay_facets(datasets, floor=1e-3, bands="both", c_mem="#C0392B",
                      c_non="#7F8C8D", unit_in=0.5, pad_frac=0.10, panel_h=2.0,
                      wspace=0.15, left_in=1.25, right_in=0.20, top_in=0.45,
                      bottom_in=1.05):
    """Median and IQR of ICC per clone-size bin, memory vs non-memory, one facet per
    dataset; facet widths are proportional to their x-range."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    lims = [(np.asarray(d["bin_x"], float).min() - pad_frac,
             np.asarray(d["bin_x"], float).max() + pad_frac) for d in datasets]
    spans = np.array([hi - lo for lo, hi in lims])
    n = len(datasets)
    fig_w = spans.sum() * unit_in + wspace * unit_in * spans.mean() * (n - 1) + left_in + right_in
    fig_h = panel_h + top_in + bottom_in
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True,
                             gridspec_kw={"width_ratios": spans, "wspace": wspace})
    axes = np.atleast_1d(axes)
    for ax, d, (x0, x1) in zip(axes, datasets, lims):
        for res in d["results"]:
            Y = np.clip(np.asarray(res["icc_bins"], float), floor, None)
            x = np.asarray(res["bin_x"], float)
            meta = res["meta"]
            for idx, color, is_mem in (
                    (np.where(meta["reliable"] & res["memory"])[0], c_mem, True),
                    (np.where(meta["reliable"] & res["non-memory"])[0], c_non, False)):
                if idx.size == 0:
                    continue
                lo, med, hi = np.nanpercentile(Y[idx], [25, 50, 75], axis=0)
                if bands == "both" or (bands == "memory" and is_mem):
                    ax.fill_between(x, lo, hi, facecolor=color,
                                    alpha=0.18 if res["hatch"] is None else 0.10,
                                    hatch=res["hatch"],
                                    edgecolor=color if res["hatch"] else "none",
                                    linewidth=0, zorder=1)
                ax.plot(x, med, color=color, linestyle=res["linestyle"], lw=2.4,
                        zorder=3, solid_capstyle="round")
        if len(d["results"]) > 1:
            ax.legend(handles=[Line2D([], [], color="black", lw=2.4,
                                      linestyle=r["linestyle"], label=r["group"])
                               for r in d["results"]],
                      frameon=False, fontsize=11, loc="upper right")
        ax.set_yscale("log")
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        title = (f'{d["dataset"]} ({d["results"][0]["group"]})'
                 if len(d["results"]) == 1 else d["dataset"])
        ax.set_title(title, fontsize=15, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=13)
    axes[0].set_ylabel("Expression\nautocorrelation", fontsize=15)
    axes[0].set_ylim(5e-3, 1.5)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        bottom=bottom_in / fig_h, top=1 - top_in / fig_h, wspace=wspace)
    fig.supxlabel("Generation (estimated by clone size)", fontsize=15,
                  y=(bottom_in - 0.58) / fig_h)
    fig.legend(handles=[Line2D([], [], color=c_mem, lw=2.4, label="memory"),
                        Line2D([], [], color=c_non, lw=2.4, label="non-memory")],
               frameon=False, fontsize=13, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.04 / fig_h))
    return fig, axes


# Michaels and Weinreb: last time point only, the earlier groups are mostly 2-cell clones
AUTOCORR_SPECS = [("T-lineage", ["T2"]), ("Harmange", ["E", "M"]),
                  ("LARRY", ["6.0"]), ("Bandler", ["E13", "E14"])]


def fig_expression_autocorrelation(specs=AUTOCORR_SPECS, top_n=None, min_clones=15,
                                   name="fig1g_expression_autocorrelation", save=True,
                                   dpi=400, show=True, verbose=True):
    """(Fig 1g) ICC of memory vs non-memory genes across clone-size bins. Returns
    (fig, axes, data) with `data` the per-dataset results the plot was drawn from."""
    import anndata as ad
    data = []
    for lab, groups in specs:
        A = ad.read_h5ad(DATASETS[lab]["h5ad"])
        grp = A.obs[MOLFEAT_GROUP_COL[lab]].astype(str).to_numpy()
        sets = pergroup_memory(lab, top_n)
        results = []
        for g in groups:
            sub = A[grp == g]
            sub = sub[:, np.asarray(sub.X.sum(axis=0)).ravel() > 0].copy()
            meta, icc_bins, bin_x, _ = compute_decay(
                sub, MOLFEAT_CLONE_COL.get(lab, "clone_id"), min_clones=min_clones)
            mem, called = sets[g]
            gl = MOLFEAT_GROUP_KEY.get(g, g)
            results.append({"group": gl, "meta": meta, "icc_bins": icc_bins,
                            "bin_x": bin_x,
                            "memory": meta.gene.isin(mem).to_numpy(),
                            "non-memory": ~meta.gene.isin(called).to_numpy(),
                            "linestyle": "-" if not results else "--",
                            "hatch": None if not results else "///"})
            if verbose:
                print(f"  {MOLFEAT_PAPER[lab]:9s} {gl:4s} bins {len(bin_x)} | reliable "
                      f"memory {int((meta.reliable & results[-1]['memory']).sum())}, "
                      f"non-memory {int((meta.reliable & results[-1]['non-memory']).sum())}")
        data.append({"dataset": MOLFEAT_PAPER[lab], "results": results,
                     "bin_x": results[0]["bin_x"]})
    with plt.rc_context({"svg.fonttype": "none"}):
        fig, axes = plot_decay_facets(data)
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, axes, data


# =============================================================================
# Suppl. Fig 8: "box" figure -- structure of perturbation responses
#
#   a  empirical rank-one RMS factorisation, R_iq ~ G_i H_q (Bowness)
#   b  C, G_S/G_F, H_S/H_F with bootstrap intervals, every screen
#   c  first-mode fractions: non-memory LFC / memory LFC / total RMS
#   d  held-out RMS prediction
#   e  clustered perturbation-response correlations (Bowness)
#   f  spectral scaling example (largest memory - control slope difference)
#   g  paired spectral slopes across screens
#
# Ported from the original analysis notebook (FIG_final.ipynb); statistics, layout and styling unchanged.
# Panels a, b and e are COMPUTED here from the screens' LFC matrices. Panels c, d, f
# and g plot five tables computed separately; they are read from BOX_DIR
# ($PERM_DATA/box_figure, deposited with the data) and are not regenerated in this
# repository. Changed inputs: memory genes come from the paper's consensus table
# (figure_consensus_top300.csv, merged, >= 2 methods) instead of a copy, and the
# LFC matrices from their locations in this project.
# =============================================================================
BOX_DIR = os.environ.get("PERM_BOX", os.path.join(P.DATA, "box_figure"))
BOX_TABLES = {"points": "Original_Multigroup_Prediction_Points.csv",
              "curves": "all_spectral_curve_points.csv",
              "fits": "all_spectral_slopes.csv",
              "lfc_modes": "Panel_C_Memory_Matched_LFC_Fractions_Paired.csv",
              "rms_modes": "Panel_C_Mode_Fractions_All_Datasets.csv"}
BOX_HALFLIFE = "halflife_genes_real_units_provisional.csv"     # optional, in BOX_DIR
BOX_GHC_CACHE = ("C_G_H_Matched_Datasets.csv", "C_G_H_Matched_Bootstrap.csv")
BOX_POWERSEEK_TOP = os.path.join(P.SCRATCH, "power-seek_run", "lfc",
                                 "powerseek_memory_genes_top500.csv")
BOX_LFC = {
    "shi_mold": f"{P.DATA}/Mold_brain/mold_lfc_cpm.csv",
    "bowness_velten": f"{P.DATA}/velten_hspc/velten_lfc_cpm.csv",
    "frangieh_Control": f"{P.DATA}/melanoma/perturbseq/frangieh_lfc_cpm_Control.csv",
    "weinstock": f"{P.DATA}/weinstock_lfc_cpm_donor_corrected.csv",
    "xu_PLX": f"{P.DATA}/melanoma/perturbseq/melanoma_lfc_cpm_PLX.csv",
    "xu_DMSO": f"{P.DATA}/melanoma/perturbseq/melanoma_lfc_cpm_DMSO.csv",
    **{f"marson_{s}": f"{P.DATA}/marson/marson_lfc_cpm_{s}.npz"
       for s in ("Stim8hr", "Stim48hr", "Rest")},
}
BOX_GENERIC_LFC = os.path.join(P.SCRATCH, "power-seek_run", "lfc",
                               "{screen}_lfc_cpm_koqc-all_ctrlqc-same.npz")
# screen -> lineage dataset whose memory genes it is scored with
BOX_STUDY = {"shi_mold": "Bandler", "bowness_velten": "Weinreb",
             "frangieh_Control": "Harmange", "xu_PLX": "Harmange",
             "xu_DMSO": "Harmange", "weinstock": "Michaels",
             "marson_Stim8hr": "Michaels", "marson_Stim48hr": "Michaels",
             "marson_Rest": "Michaels"}
BOX_SEED, BOX_NBOOT, BOX_TOP, BOX_CENTERED_LFC, BOX_MIN_COMMON_M = 42, 300, 300, True, 5
BOX_MEM, BOX_CTL, BOX_CTL_BAND, BOX_GRAY = "#C62F46", "#2674AF", "#A8CBE5", "#777777"
BOX_RC = {"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17,
          "axes.titlesize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
          "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
          "axes.grid": False, "svg.fonttype": "none", "pdf.fonttype": 42}


def _box_norm(x):
    return str(x.decode() if isinstance(x, bytes) else x).strip().upper()


def _box_mean_sem(x):
    from scipy import stats as _st
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    return float(np.mean(x)), (float(_st.sem(x)) if len(x) > 1 else np.nan)


def _box_paired_test(x, y):
    from scipy import stats as _st
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return np.nan
    if np.allclose(x, y):
        return 1.0
    return float(_st.ttest_rel(x, y).pvalue)


def box_short_name(s):
    return (str(s).replace("ReplogleWeissman", "RW").replace("Replogle", "Rep")
            .replace("XAtlas", "XA").replace("Norman2019_", "Norman ")
            .replace("ProPerSeq_", "ProPerSeq ").replace("bowness_velten", "Bowness")
            .replace("frangieh_Control", "Frangieh").replace("shi_mold", "Shi")
            .replace("_", " "))


def box_tables(box_dir=None):
    """The five precomputed tables, from BOX_DIR."""
    box_dir = box_dir or BOX_DIR
    missing = [f for f in BOX_TABLES.values() if not os.path.exists(os.path.join(box_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"missing in {box_dir}: {missing}. These are deposited with the data "
            "(Zenodo); point PERM_BOX at the folder that holds them.")
    return {k: pd.read_csv(os.path.join(box_dir, f)) for k, f in BOX_TABLES.items()}


def box_load_lfc(screen):
    """(X, perturbations, genes) for one screen; |LFC| > 100 and self-targets -> NaN."""
    path = BOX_LFC.get(screen, BOX_GENERIC_LFC.format(screen=screen))
    if not os.path.exists(path):
        raise FileNotFoundError(f"{screen}: {path}")
    if path.endswith(".csv"):
        d = pd.read_csv(path, index_col=0)
        X = d.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        perts = np.array([_box_norm(x) for x in d.index])
        genes = np.array([_box_norm(x) for x in d.columns])
    else:
        with np.load(path, allow_pickle=True) as d:
            X = d["values"].astype(np.float32)
            perts = np.array([_box_norm(x) for x in d["perts" if "perts" in d else "index"]])
            genes = np.array([_box_norm(x) for x in d["genes" if "genes" in d else "columns"]])
    if X.shape == (len(genes), len(perts)) and len(genes) != len(perts):
        X = X.T
    assert X.shape == (len(perts), len(genes))
    ri, ci = ~pd.Index(perts).duplicated(), ~pd.Index(genes).duplicated()
    X, perts, genes = X[np.ix_(ri, ci)], perts[ri], genes[ci]
    X[(~np.isfinite(X)) | (np.abs(X) > 100)] = np.nan
    lookup = {g: j for j, g in enumerate(genes)}
    for i, g in enumerate(perts):
        if g in lookup:
            X[i, lookup[g]] = np.nan
    return X, perts, genes


def box_memory_sets(top=BOX_TOP):
    """{screen: memory genes}: the paper's merged >= 2 consensus for the main screens,
    the top-`top` Power-Seek control-cell genes for the generalisation screens."""
    cons = pd.read_csv(os.path.join(P.MEMGENES_ROOT, "gene_lists",
                                    f"figure_consensus_top{BOX_TOP}.csv"))
    cons = cons[(cons.group.astype(str).str.lower() == "merged") & cons.consensus_ge2]
    sets = {s: set(cons.loc[cons.dataset == v, "gene"].dropna().map(_box_norm))
            for s, v in BOX_STUDY.items()}
    if os.path.exists(BOX_POWERSEEK_TOP):
        b = pd.read_csv(BOX_POWERSEEK_TOP)
        cols = sorted([c for c in b if c.startswith("gene_")], key=lambda c: int(c.split("_")[-1]))
        sets.update({str(r.dataset): set(map(_box_norm, r[cols[:top]].dropna()))
                     for _, r in b.iterrows()})
    return sets


def _box_block_rms(X, r, c):
    z = X[np.ix_(r, c)]
    z = z[np.isfinite(z)]
    return np.sqrt(np.mean(z.astype(float) ** 2)) if len(z) else np.nan


def _box_ghc(X, sr, fr, sc, fc):
    ss, sf, fs, ff = [_box_block_rms(X, r, c)
                      for r, c in [(sr, sc), (sr, fc), (fr, sc), (fr, fc)]]
    if not np.all(np.isfinite([ss, sf, fs, ff])) or min(ss, sf, fs, ff) <= 0:
        return None
    return [ss * ff / (sf * fs), np.sqrt(ss * sf / (fs * ff)),
            np.sqrt(ss * fs / (sf * ff)), ss, sf, fs, ff]


def box_ghc(screens, nboot=BOX_NBOOT, seed=BOX_SEED, cache=True, recompute=False,
            box_dir=None, verbose=True):
    """Panel b: C, G_S/G_F and H_S/H_F per screen, median and 95% bootstrap interval.
    Memory genes split sources (perturbations) and targets (measured genes) into S
    (memory) and F (other). Cached in BOX_DIR, like the original."""
    box_dir = box_dir or BOX_DIR
    c1, c2 = (os.path.join(box_dir, f) for f in BOX_GHC_CACHE)
    if cache and not recompute and os.path.exists(c1) and os.path.exists(c2):
        if verbose:
            print("panel b: cached C/G/H statistics from", box_dir)
        return pd.read_csv(c1), pd.read_csv(c2), pd.DataFrame(columns=["screen", "reason"])
    sets = box_memory_sets()
    rows, boots, failed = [], [], []
    rng = np.random.default_rng(seed)
    for s in screens:
        try:
            X, perts, genes = box_load_lfc(s)
            M = list(sets[s])
            sr, fr = np.flatnonzero(np.isin(perts, M)), np.flatnonzero(~np.isin(perts, M))
            sc, fc = np.flatnonzero(np.isin(genes, M)), np.flatnonzero(~np.isin(genes, M))
            if verbose:
                print(f"  {s}: S sources={len(sr)}, F sources={len(fr)}, "
                      f"S targets={len(sc)}, F targets={len(fc)}", flush=True)
            if min(len(sr), len(fr), len(sc), len(fc)) == 0:
                raise ValueError("missing source/target class")
            ns, nt = min(len(sr), len(fr)), min(len(sc), len(fc))
            draws = []
            for _ in range(nboot):
                r = _box_ghc(X, rng.choice(sr, ns, replace=True), rng.choice(fr, ns, replace=True),
                             rng.choice(sc, nt, replace=True), rng.choice(fc, nt, replace=False))
                if r is not None:
                    draws.append(r)
            if len(draws) < 30:
                raise ValueError("insufficient bootstrap draws")
            z = np.asarray(draws)
            row = {"screen": s, "n_S_sources": len(sr), "n_F_sources": len(fr),
                   "n_S_targets": len(sc), "n_F_targets": len(fc)}
            for j, key in enumerate(["C", "G", "H", "L_SS", "L_SF", "L_FS", "L_FF"]):
                row[key] = np.median(z[:, j])
                row[key + "_lo"], row[key + "_hi"] = np.quantile(z[:, j], [.025, .975])
            rows.append(row)
            boots.append(pd.DataFrame({"screen": s, "bootstrap": np.arange(len(z)),
                                       "C": z[:, 0], "G": z[:, 1], "H": z[:, 2]}))
        except Exception as e:                                    # noqa: BLE001
            failed.append((s, str(e)))
            if verbose:
                print("  SKIPPED:", s, e)
    R = pd.DataFrame(rows)
    if R.empty:
        raise RuntimeError("no valid screens for panel b")
    B = pd.concat(boots, ignore_index=True)
    F = pd.DataFrame(failed, columns=["screen", "reason"])
    if cache:
        os.makedirs(box_dir, exist_ok=True)
        R.to_csv(c1, index=False)
        B.to_csv(c2, index=False)
    return R, B, F


def box_first_mode(T, centered=BOX_CENTERED_LFC):
    """Panel c: first-mode fractions per screen and the three paired tests."""
    lfc = T["lfc_modes"]
    lfc = lfc.loc[lfc.centered.astype(str).str.lower() == str(centered).lower()].copy()
    rms = T["rms_modes"][["screen", "fraction_first"]].rename(columns={"fraction_first": "rms_total"})
    d = (lfc.merge(rms, on="screen", how="inner")
            .dropna(subset=["control", "memory", "rms_total"])
            .drop_duplicates("screen").sort_values("screen").reset_index(drop=True))
    labels = ["Nonmemory LFC", "Memory LFC", "Total RMS"]
    v = d[["control", "memory", "rms_total"]].to_numpy(float)
    tests = pd.DataFrame([{"comparison": labels[i] + " vs " + labels[j],
                           "mean_difference": np.mean(v[:, i] - v[:, j]),
                           "p": _box_paired_test(v[:, i], v[:, j])}
                          for i, j in [(0, 1), (1, 2), (0, 2)]])
    return d, v, tests


def _box_fit_line(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or len(np.unique(x[good])) < 3:
        return np.nan, np.nan
    return np.polyfit(x[good], y[good], 1)


def box_spectral(T, min_common_m=BOX_MIN_COMMON_M):
    """Panels f and g: paired memory / matched-control spectral slopes per screen and
    the example screen (largest memory - control slope difference)."""
    rows = []
    for screen, z in T["fits"].groupby("screen", sort=False):
        m = pd.to_numeric(z.loc[z.type == "Memory", "slope"], errors="coerce").dropna().to_numpy(float)
        c = pd.to_numeric(z.loc[z.type == "Matched nonmemory", "slope"], errors="coerce").dropna().to_numpy(float)
        if len(m) != 1 or len(c) < 1:
            continue
        rows.append({"screen": screen, "memory_slope": float(m[0]),
                     "control_slope": float(np.median(c)),
                     "difference": float(m[0] - np.median(c))})
    paired = pd.DataFrame(rows).sort_values("screen")
    examples = []
    for s in paired.screen:
        z = T["curves"][T["curves"].screen == s].copy()
        z["m"] = pd.to_numeric(z["m"], errors="coerce")
        z["ratio"] = pd.to_numeric(z["ratio"], errors="coerce")
        z = z.dropna(subset=["m", "ratio"])
        mm = z[z.type == "Memory"].groupby("m", as_index=False).agg(memory=("ratio", "mean"))
        cc = (z[z.type == "Matched nonmemory"].groupby("m")["ratio"]
                .agg(control="median", lower=lambda v: np.quantile(v, .025),
                     upper=lambda v: np.quantile(v, .975)).reset_index())
        j = mm.merge(cc, on="m").sort_values("m")
        if len(j) < min_common_m:
            continue
        x = j.m.to_numpy(float)
        sm, bm = _box_fit_line(x, j.memory.to_numpy(float))
        sc, bc = _box_fit_line(x, j.control.to_numpy(float))
        if np.all(np.isfinite([sm, bm, sc, bc])):
            examples.append({"screen": s, "data": j, "sm": sm, "bm": bm,
                             "sc": sc, "bc": bc, "delta": sm - sc})
    if not examples:
        raise RuntimeError("no valid spectral scaling example")
    return paired, max(examples, key=lambda q: q["delta"])


def box_correlations(dataset="bowness_velten", max_perturbations=300, min_coverage=0.70):
    """Panel e: correlations of perturbation LFC profiles, average-linkage ordered."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    X, perts, genes = box_load_lfc(dataset)
    sd = np.nanstd(X, axis=0)
    keep = (np.mean(np.isfinite(X), axis=0) >= min_coverage) & np.isfinite(sd) & (sd > 1e-8)
    X = X[:, keep]
    med = np.nanmedian(X, axis=0)
    rr, cc = np.where(~np.isfinite(X))
    X[rr, cc] = med[cc]
    row_rms = np.sqrt(np.mean(X.astype(float) ** 2, axis=1))
    if len(perts) > max_perturbations:
        take = np.argsort(row_rms)[-max_perturbations:]
        X, perts = X[take], perts[take]
    X = X - X.mean(axis=1, keepdims=True)
    valid = X.std(axis=1) > 1e-12
    X, perts = X[valid], perts[valid]
    corr = np.clip(np.nan_to_num(np.corrcoef(X)), -1, 1)
    np.fill_diagonal(corr, 1)
    dist = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return corr[np.ix_(order, order)], perts[order]


def _box_prediction_metrics(z):
    x, y = z.observed.to_numpy(float), z.predicted.to_numpy(float)
    lx, ly = np.log10(x), np.log10(y)
    res = ly - lx
    denom = np.sum((lx - lx.mean()) ** 2)
    return {"n_points": len(z),
            "log10_R2": 1 - np.sum(res ** 2) / denom if denom > 0 else np.nan,
            "log10_Pearson_r": (np.corrcoef(lx, ly)[0, 1]
                                if len(lx) > 2 and np.std(lx) > 0 and np.std(ly) > 0 else np.nan),
            "log10_RMSE": np.sqrt(np.mean(res ** 2)),
            "median_absolute_log10_error": np.median(np.abs(res)),
            "median_predicted_over_observed": np.median(y / x)}


def box_prediction(T):
    """Panel d: held-out predicted vs observed RMS, overall and per screen."""
    q = T["points"].copy()
    q["observed"] = pd.to_numeric(q.observed, errors="coerce")
    q["predicted"] = pd.to_numeric(q.predicted, errors="coerce")
    q = q.loc[np.isfinite(q.observed) & np.isfinite(q.predicted)
              & (q.observed > 0) & (q.predicted > 0)].copy()
    return (q, _box_prediction_metrics(q),
            pd.DataFrame([{"screen": s, **_box_prediction_metrics(z)} for s, z in q.groupby("screen")]))


def box_rms_decomposition(dataset="bowness_velten", lineage="Weinreb",
                          max_perturbations=80, box_dir=None):
    """Panel a: RMS response of each perturbation in memory genes and five non-memory
    quintiles, and its rank-one fit R_iq ~ G_i H_q. Non-memory genes are ordered by
    RNA half-life when BOX_HALFLIFE is present (>= 50 genes), else by response SD."""
    from matplotlib.colors import LogNorm
    box_dir = box_dir or BOX_DIR
    X, perts, genes = box_load_lfc(dataset)
    cons = pd.read_csv(os.path.join(P.MEMGENES_ROOT, "gene_lists",
                                    f"figure_consensus_top{BOX_TOP}.csv"))
    mem = set(cons.loc[(cons.dataset == lineage) & (cons.group.astype(str).str.lower() == "merged")
                       & cons.consensus_ge2, "gene"].dropna().map(_box_norm))
    mem_idx = np.flatnonzero(np.isin(genes, list(mem)))
    if not len(mem_idx):
        raise RuntimeError(f"no {lineage} memory genes in {dataset}")
    non_idx = np.flatnonzero(~np.isin(genes, list(mem)))
    order_values, ordering = np.nanstd(X, axis=0), "response variability (fallback)"
    hlfile = os.path.join(box_dir, BOX_HALFLIFE)
    if os.path.exists(hlfile):
        hl = pd.read_csv(hlfile)
        gc = next((c for c in hl if "gene" in c.lower() or "symbol" in c.lower()), None)
        hc = next((c for c in hl if "half" in c.lower() and pd.api.types.is_numeric_dtype(hl[c])), None)
        if gc is not None and hc is not None:
            lk = dict(zip(hl[gc].map(_box_norm), pd.to_numeric(hl[hc], errors="coerce")))
            hv = np.array([lk.get(g, np.nan) for g in genes], float)
            ok = non_idx[np.isfinite(hv[non_idx]) & (hv[non_idx] > 0)]
            if len(ok) >= 50:
                non_idx, order_values, ordering = ok, hv, f"RNA half-life: {hc}"
    non_idx = non_idx[np.isfinite(order_values[non_idx])]
    non_idx = non_idx[np.argsort(order_values[non_idx])]
    groups = [mem_idx] + list(np.array_split(non_idx, 5))
    labels = ["Memory", "Q1", "Q2", "Q3", "Q4", "Q5"]
    Rq = np.column_stack([np.sqrt(np.nanmean(X[:, idx].astype(float) ** 2, axis=1)) for idx in groups])
    ok = np.all(np.isfinite(Rq) & (Rq > 0), axis=1)
    Rq, perts = Rq[ok], perts[ok]
    U, s, Vt = np.linalg.svd(Rq, full_matrices=False)
    G, H = U[:, 0] * s[0], Vt[0, :].copy()
    if H.mean() < 0:
        G, H = -G, -H
    sc = H.mean()
    H, G = H / sc, G * sc
    fit = np.outer(G, H)
    order = np.argsort(G)[::-1]
    if len(order) > max_perturbations:
        order = order[np.linspace(0, len(order) - 1, max_perturbations).astype(int)]
    show, show_fit = Rq[order], fit[order]
    pos = np.r_[show.ravel(), show_fit.ravel()]
    pos = pos[np.isfinite(pos) & (pos > 0)]
    return {"R": Rq, "fit": fit, "G": G, "H": H, "perts": perts, "labels": labels,
            "n_genes": list(map(len, groups)), "ordering": ordering,
            "f1": s[0] ** 2 / np.sum(s ** 2),
            "r2": 1 - np.sum((Rq - fit) ** 2) / np.sum((Rq - Rq.mean()) ** 2),
            "show": show, "show_fit": show_fit, "G_show": G[order],
            "norm": LogNorm(vmin=pos.min(), vmax=pos.max()), "dataset": dataset}


def box_compute(recompute_ghc=False, box_dir=None, verbose=True):
    """(Suppl. Fig 8) Everything the box figure needs, in one dict."""
    T = box_tables(box_dir)
    screens = T["points"].screen.dropna().astype(str).unique().tolist()
    R, B, Fl = box_ghc(screens, recompute=recompute_ghc, box_dir=box_dir, verbose=verbose)
    D, Dv, Dt = box_first_mode(T)
    paired, ex = box_spectral(T)
    corr, perts_e = box_correlations()
    q, g_all, g_by = box_prediction(T)
    A = box_rms_decomposition(box_dir=box_dir)
    order = list(dict.fromkeys(screens + list(T["curves"].screen.dropna().astype(str))))
    return {"tables": T, "screens": screens, "screen_order": order,
            "ghc": R, "ghc_boot": B, "ghc_failed": Fl,
            "first_mode": D, "first_mode_values": Dv, "first_mode_tests": Dt,
            "paired_slopes": paired, "example": ex,
            "corr": corr, "corr_perts": perts_e,
            "pred": q, "pred_all": g_all, "pred_by_screen": g_by, "A": A}


def _box_letter(ax, letter, x=-.11, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=30, fontweight="bold",
            va="bottom", ha="left", clip_on=False)


_BOX_A_REF = (.025, .56, .310, .40)          # panel a's area in the combined figure


def _box_draw_a(fig, A, region=_BOX_A_REF):
    """Panel a: observed R_iq -> G_i x H_q ~ reconstructed, drawn into `region`
    (figure coordinates); the original absolute positions are mapped into it."""
    from matplotlib.colors import LogNorm
    X0, Y0, W0, H0 = _BOX_A_REF
    x1, y1, w1, h1 = region

    def rect(x, y, w, h):
        return [x1 + (x - X0) / W0 * w1, y1 + (y - Y0) / H0 * h1, w / W0 * w1, h / H0 * h1]

    def pt(x, y):
        return x1 + (x - X0) / W0 * w1, y1 + (y - Y0) / H0 * h1

    cm = plt.get_cmap("coolwarm")
    a_obs, a_g = fig.add_axes(rect(.037, .66, .092, .220)), fig.add_axes(rect(.166, .66, .016, .220))
    a_h, a_fit = fig.add_axes(rect(.216, .755, .056, .025)), fig.add_axes(rect(.301, .66, .092, .220))
    a_obs.imshow(A["show"], cmap=cm, norm=A["norm"], aspect="auto", interpolation="nearest", rasterized=True)
    a_fit.imshow(A["show_fit"], cmap=cm, norm=A["norm"], aspect="auto", interpolation="nearest", rasterized=True)
    a_g.imshow(A["G_show"][:, None], cmap=cm, norm=LogNorm(vmin=A["G_show"].min(), vmax=A["G_show"].max()),
               aspect="auto", interpolation="nearest", rasterized=True)
    a_h.imshow(A["H"][None, :], cmap=cm, norm=LogNorm(vmin=A["H"].min(), vmax=A["H"].max()),
               aspect="auto", interpolation="nearest")
    for aa in (a_obs, a_g, a_h, a_fit):
        aa.set_yticks([])
        for sp in aa.spines.values():
            sp.set_visible(True)
            sp.set_color(".55")
            sp.set_linewidth(.8)
    a_obs.set_xticks(range(6))
    a_obs.set_xticklabels(A["labels"], fontsize=11, rotation=45, ha="right")
    a_obs.set_ylabel("Perturbations $i$", fontsize=14)
    for aa in (a_g, a_h, a_fit):
        aa.set_xticks([])
    a_obs.set_title(r"Observed $R_{iq}$", fontsize=15, pad=7)
    a_g.set_title(r"$G_i$", fontsize=15, pad=7)
    a_h.set_title(r"$H_q$", fontsize=15, pad=7)
    a_fit.set_title(r"$G_iH_q$", fontsize=15, pad=7)
    fig.text(*pt(.148, .772), r"$\Rightarrow$", ha="center", va="center", fontsize=26, color=".45")
    fig.text(*pt(.199, .772), r"$\times$", ha="center", va="center", fontsize=23)
    fig.text(*pt(.286, .772), r"$\approx$", ha="center", va="center", fontsize=23)
    fig.text(*pt(.185, .925), r"$R_{iq}\simeq G_iH_q$", ha="center", fontsize=22,
             bbox=dict(boxstyle="round,pad=.3", facecolor=".94", edgecolor="none"))
    fig.text(*pt(.244, .709), "Gene-set\nsusceptibility", ha="center", va="center", fontsize=16)
    fig.text(*pt(.174, .63), "perturbation\npotency", ha="center", fontsize=16)
    fig.text(*pt(.084, .60), "Target populations $q$", ha="center", fontsize=14)


def _box_draw_b(axes, R):
    keys, labels = ["C", "G", "H"], [r"$C$", r"$G_S/G_F$", r"$H_S/H_F$"]
    titles = ["Rank-one factorization", "Source potency", "Target susceptibility"]
    for j, (key, label, title, color) in enumerate(zip(keys, labels, titles,
                                                       [BOX_GRAY, BOX_CTL, BOX_MEM])):
        ax, v = axes[j], R[key].to_numpy(float)
        if len(v) >= 2 and np.ptp(v) > 0:
            vio = ax.violinplot([v], positions=[0], widths=.65, showmeans=False, showextrema=False)
            for body in vio["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.20)
        jit = np.random.default_rng(BOX_SEED + j).uniform(-.15, .15, len(R))
        for i, r in R.reset_index(drop=True).iterrows():
            ax.plot([jit[i]] * 2, [r[key + "_lo"], r[key + "_hi"]], color=color, lw=.85, alpha=.4)
            ax.scatter(jit[i], r[key], s=22, c=color, edgecolor="white", linewidth=.4, zorder=3)
        mean, sem = _box_mean_sem(v)
        ax.errorbar(0, mean, yerr=sem, fmt="D", ms=7, c="black", capsize=4, zorder=5)
        ax.axhline(1, color="black", ls="--", lw=.9)
        ax.set_xlim(-.40, .40)
        ax.set_ylim(.5, 2.5)
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.set_title(title, fontsize=15, pad=9)
        ax.set_yticks([.5, 1, 1.5, 2, 2.5])
        if j > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.text(.03, .97, f"n={len(v)}\n{mean:.3f} ± {sem:.3f} SEM",
                transform=ax.transAxes, va="top", fontsize=11)
    axes[0].set_ylabel("Ratio")


def _box_draw_c(ax, Dv):
    jit = np.random.default_rng(BOX_SEED).uniform(-.05, .05, len(Dv))
    for i, y in enumerate(Dv):
        xx = np.arange(3) + jit[i]
        ax.plot(xx, y, color=".70", lw=1.0, alpha=.65)
        ax.scatter(xx, y, color=".42", s=17, alpha=.8)
    ax.errorbar([0, 1, 2], Dv.mean(axis=0), yerr=Dv.std(axis=0, ddof=1) / np.sqrt(len(Dv)),
                fmt="D", ms=8, color="black", capsize=4, lw=1.5, zorder=5, label="Mean ± SEM")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Nonmemory\nLFC", "Memory\nLFC", "Total\nRMS"])
    ax.set_ylabel("First-mode fraction")
    ax.set_xlim(-.25, 2.25)
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, fontsize=12, loc="upper left")


def _box_draw_d(ax, q, screen_order):
    cm = plt.get_cmap("coolwarm")
    cols = {s: cm(i / max(len(screen_order) - 1, 1)) for i, s in enumerate(screen_order)}
    for s in screen_order:
        z = q[q.screen == s]
        if not z.empty:
            ax.scatter(z.observed, z.predicted, s=10, alpha=.35, color=cols[s],
                       edgecolors="none", rasterized=True, label=box_short_name(s))
    v = np.concatenate([q.observed.to_numpy(float), q.predicted.to_numpy(float)])
    lo, hi = max(v.min() * .8, 1e-10), v.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.3)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel("Observed target-group RMS")
    ax.set_ylabel("Held-out predicted RMS")
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left", markerscale=1.2,
              columnspacing=.5, handletextpad=.3)


def _box_draw_e(fig, ax, cax, corr, dataset="bowness_velten"):
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest",
                   aspect="auto", rasterized=True)
    ax.set_title(box_short_name(dataset), fontsize=16)
    ax.set_xlabel("Perturbation (clustered)")
    ax.set_ylabel("Perturbation (clustered)")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Correlation of perturbation LFC profiles", fontsize=14)
    cb.ax.tick_params(labelsize=12)


def _box_draw_f(ax, ex):
    from matplotlib.ticker import MaxNLocator
    z = ex["data"]
    x = z.m.to_numpy(float)
    ax.fill_between(x, z.lower.to_numpy(float), z.upper.to_numpy(float),
                    color=BOX_CTL_BAND, alpha=.35, linewidth=0)
    ax.scatter(x, z.memory.to_numpy(float), s=28, color=BOX_MEM, edgecolors="white",
               linewidths=.3, zorder=5, label="Memory")
    ax.scatter(x, z.control.to_numpy(float), s=26, color=BOX_CTL, marker="s",
               edgecolors="white", linewidths=.3, zorder=5, label="Matched nonmemory")
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, ex["sm"] * xx + ex["bm"], color=BOX_MEM, lw=2)
    ax.plot(xx, ex["sc"] * xx + ex["bc"], color=BOX_CTL, lw=2)
    ax.set_xlim(0, x.max() * 1.07)
    ax.set_xlabel(r"Number of target genes, $m$")
    ax.set_ylabel(r"$\lambda_1/\lambda_{\mathrm{bulk}}$")
    ax.set_title(box_short_name(ex["screen"]), fontsize=16)
    ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _box_draw_g(ax, paired):
    from matplotlib.ticker import MaxNLocator
    v = paired[["memory_slope", "control_slope"]].to_numpy(float)
    jit = np.random.default_rng(BOX_SEED + 1).uniform(-.04, .04, len(paired))
    for i, y in enumerate(v):
        xx = np.array([0., 1.]) + jit[i]
        ax.plot(xx, y, color=".70", lw=1.1, alpha=.7)
        ax.scatter(xx, y, color=".4", s=22, zorder=3)
    ax.errorbar([0, 1], v.mean(axis=0), yerr=v.std(axis=0, ddof=1) / np.sqrt(len(v)),
                fmt="D", ms=8, c="black", capsize=4, lw=1.5, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Memory", "Matched\nnonmemory"])
    ax.set_ylabel("Spectral scaling slope")
    ax.set_xlim(-.25, 1.25)
    ax.axhline(0, color="black", lw=.8)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def fig_box(data=None, name="figS_box_response_structure", save=True, dpi=600,
            show=True, verbose=True):
    """(Suppl. Fig 8) The complete box figure, panels a-g, in the original layout. Returns (fig, data)."""
    data = data if data is not None else box_compute(verbose=verbose)
    with plt.rc_context(BOX_RC):
        fig = plt.figure(figsize=(29, 13), facecolor="white")
        axA = fig.add_axes(list(_BOX_A_REF))
        axA.axis("off")
        _box_letter(axA, "a", x=-.035, y=.96)
        _box_draw_a(fig, data["A"])
        bl, br, bg = .045, .328, .013
        bw = (br - bl - 2 * bg) / 3
        axB = [fig.add_axes([bl + j * (bw + bg), .105, bw, .335]) for j in range(3)]
        _box_draw_b(axB, data["ghc"])
        _box_letter(axB[0], "b", x=-.38, y=1.07)
        axC = fig.add_axes([.405, .565, .245, .375])
        _box_draw_c(axC, data["first_mode_values"])
        _box_letter(axC, "c")
        axD = fig.add_axes([.405, .105, .245, .365])
        _box_draw_d(axD, data["pred"], data["screen_order"])
        _box_letter(axD, "d")
        axE, cax = fig.add_axes([.715, .575, .245, .365]), fig.add_axes([.968, .575, .009, .365])
        _box_draw_e(fig, axE, cax, data["corr"])
        _box_letter(axE, "e")
        axF = fig.add_axes([.715, .105, .115, .335])
        _box_draw_f(axF, data["example"])
        _box_letter(axF, "f", x=-.23)
        axG = fig.add_axes([.865, .105, .100, .335])
        _box_draw_g(axG, data["paired_slopes"])
        _box_letter(axG, "g", x=-.28)
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, data


_BOX_PANEL_SIZE = {"a": (9.0, 5.2), "b": (9.2, 4.6), "c": (7.1, 4.9), "d": (7.1, 4.7),
                   "e": (7.6, 4.7), "f": (3.6, 4.4), "g": (3.0, 4.4)}


def fig_box_panel(panel, data=None, name=None, save=True, dpi=600, show=True, verbose=True):
    """(Suppl. Fig 8a-g) One panel of the box figure on its own ("a" ... "g"). Returns (fig, data)."""
    panel = panel.lower()
    data = data if data is not None else box_compute(verbose=verbose)
    name = name or f"figS_box_panel_{panel}"
    with plt.rc_context(BOX_RC):
        fig = plt.figure(figsize=_BOX_PANEL_SIZE[panel], facecolor="white")
        if panel == "a":
            _box_draw_a(fig, data["A"], region=(.0, .0, 1.0, 1.0))
        elif panel == "b":
            axes = [fig.add_axes([.12 + j * .29, .14, .26, .70]) for j in range(3)]
            _box_draw_b(axes, data["ghc"])
        elif panel == "c":
            _box_draw_c(fig.add_axes([.14, .14, .82, .80]), data["first_mode_values"])
        elif panel == "d":
            _box_draw_d(fig.add_axes([.14, .14, .82, .80]), data["pred"], data["screen_order"])
        elif panel == "e":
            _box_draw_e(fig, fig.add_axes([.10, .12, .72, .78]),
                        fig.add_axes([.85, .12, .03, .78]), data["corr"])
        elif panel == "f":
            _box_draw_f(fig.add_axes([.24, .14, .70, .76]), data["example"])
        elif panel == "g":
            _box_draw_g(fig.add_axes([.30, .14, .64, .80]), data["paired_slopes"])
        else:
            raise ValueError(f"panel must be one of a-g, got {panel!r}")
        save_fig(fig, _save_path(save, name), name, dpi=dpi, verbose=verbose)
        if show:
            plt.show()
    return fig, data


def box_stats(data):
    """(Suppl. Fig 8) The manuscript numbers the original notebook printed, as tables."""
    from scipy import stats as _st
    R, Dv, paired, q, ex = (data["ghc"], data["first_mode_values"], data["paired_slopes"],
                            data["pred"], data["example"])
    ghc = []
    for key, desc in [("C", "rank-one factorization C"), ("G", "source potency G_S/G_F"),
                      ("H", "target susceptibility H_S/H_F")]:
        v = R[key].to_numpy(float)
        mu, se = _box_mean_sem(v)
        ghc.append({"statistic": desc, "n": len(v), "mean": mu, "sem": se,
                    "median": np.median(v),
                    "p_vs_1": _st.ttest_1samp(v, 1).pvalue if len(v) > 1 else np.nan})
    off = data["corr"][np.triu_indices(len(data["corr"]), k=1)]
    labels = ["Nonmemory LFC", "Memory LFC", "Total RMS"]
    fm = pd.DataFrame([{"set": n, "mean": Dv[:, j].mean(),
                        "sem": Dv[:, j].std(ddof=1) / np.sqrt(len(Dv)),
                        "median": np.median(Dv[:, j]), "q25": np.quantile(Dv[:, j], .25),
                        "q75": np.quantile(Dv[:, j], .75)} for j, n in enumerate(labels)])
    v = paired[["memory_slope", "control_slope"]].to_numpy(float)
    diff = v[:, 0] - v[:, 1]
    md, sd = _box_mean_sem(diff)
    pos = v[:, 1] > 0
    slopes = {"n_screens": len(paired),
              "memory_mean": v[:, 0].mean(), "memory_sem": v[:, 0].std(ddof=1) / np.sqrt(len(v)),
              "control_mean": v[:, 1].mean(), "control_sem": v[:, 1].std(ddof=1) / np.sqrt(len(v)),
              "ratio_of_means": v[:, 0].mean() / v[:, 1].mean(),
              "paired_t_p": _box_paired_test(v[:, 0], v[:, 1]),
              "mean_difference": md,
              "ci95_difference": _st.t.interval(.95, len(diff) - 1, loc=md, scale=sd),
              "median_within_screen_ratio": np.median(v[pos, 0] / v[pos, 1]) if pos.any() else np.nan,
              "example_screen": ex["screen"], "example_memory_slope": ex["sm"],
              "example_control_slope": ex["sc"], "example_difference": ex["delta"]}
    return {"ghc": pd.DataFrame(ghc),
            "correlations": {"screen": "bowness_velten", "n_perturbations": len(data["corr"]),
                             "median_offdiag": np.median(off), "mean_abs_offdiag": np.mean(np.abs(off))},
            "first_mode": fm, "first_mode_tests": data["first_mode_tests"],
            "spectral": slopes,
            "prediction_all": {"n_screens": q.screen.nunique(), **data["pred_all"]},
            "prediction_by_screen": data["pred_by_screen"],
            "A": {"screen": data["A"]["dataset"], "first_mode_fraction": data["A"]["f1"],
                  "R2": data["A"]["r2"], "ordering": data["A"]["ordering"]}}
