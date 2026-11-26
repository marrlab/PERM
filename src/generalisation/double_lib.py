"""Double-perturbation analysis: do memory genes behave differently in gene pairs?

Two framings, both on the raw-count pseudobulk from build_double_pseudobulk.py:

  3a  PAIR level (the driver question): pairs of a memory-gene target with a regular
      target vs pairs of two regular targets. Is their genetic interaction stronger,
      and does the memory gene dominate the double's profile?
  3b  READOUT level (extends Fig 2): across all pairs, are memory genes, as readouts,
      more NON-ADDITIVE than expression-matched background genes?

"Memory genes" = the top-N of a Power-Seek run on the dataset's own control cells
(perturb-seq/powerseek_control.py), N = 300 by default.

Interaction per pair A|B, over readout genes above the 0.1 control floor:
  residual = LFC(A|B) - [LFC(A) + LFC(B)]                 (additive expectation)
  LFC(A|B) ~ c_A LFC(A) + c_B LFC(B)                       (Norman et al. 2019 model)
Every label's own targets are masked (NaN) in its LFC row, so a pair's residual is
NaN at A and B.

Confounds to carry (see memory notes): per-condition |LFC| tracks cells per condition
(rho -0.63 in Marson), and residuals scale with the singles' effect size, so pair-level
comparisons are covariate-adjusted, and readout-level ones are expression-matched.
"""

# --- repo root on sys.path, so `import perm_paths` works from any depth ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != "/" and not _os.path.exists(_os.path.join(_r, "perm_paths.py")):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import perm_paths as P
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, P.LIB)
from marson_KO import cliffs_delta, cliffs_ci          # shared effect-size helpers

DATA = f"{P.DATA}/double_perturbs"
PS_DIR = f"{DATA}/powerseek_control"
EXPR_FLOOR = 0.1          # mean control counts/cell, the project-wide readout floor
MIN_CELLS = 20            # per label, as min_cells_per_ko in the single-KO pipeline
TOP_N = 300

PAIR_CLASSES = ("memory x regular", "regular x regular", "memory x memory")


# --------------------------------------------------------------------------- loading
def load(ds):
    """(counts DataFrame labels x genes, sizes Series, ctrl_expr Series) for one dataset."""
    z = np.load(f"{DATA}/{ds}/{ds}_pseudobulk_raw.npz", allow_pickle=False)
    counts = pd.DataFrame(z["counts"], index=z["labels"], columns=z["genes"])
    sizes = pd.Series(z["sizes"], index=z["labels"], name="n_cells")
    ctrl = pd.read_csv(f"{DATA}/{ds}/{ds}_gene_ctrl_expr.csv", index_col=0).iloc[:, 0]
    return counts, sizes, ctrl


def memory_genes(ds, top_n=TOP_N, run="nall", floor_tag="_floor0.1"):
    """Top-N Power-Seek genes from `ds`'s own control cells (files are sorted by delta)."""
    m = pd.read_csv(f"{PS_DIR}/powerseek_control_{ds}_{run}_cv3000{floor_tag}.csv", index_col=0)
    if not m["delta"].is_monotonic_decreasing:
        m = m.sort_values("delta", ascending=False)
    if top_n > len(m):
        raise ValueError(f"asked for top {top_n}, the file has {len(m)}")
    return list(m.index[:top_n])


def targets(label):
    return label.split("|")


# --------------------------------------------------------------------------- LFC
def compute_lfc(counts, sizes, min_cells=MIN_CELLS, pseudocount=1.0, cpm_target=1e6):
    """Per-label CPM, LFC = log2((CPM+pc)/(CPM_ctrl+pc)), own targets masked to NaN.

    Same normalisation as marson_KO.lfc_from_pseudobulk(normalize='group'); the
    masking differs only in that a double masks BOTH of its targets."""
    lib = counts.sum(axis=1).replace(0, 1.0)
    cpm = counts.div(lib, axis=0) * cpm_target
    lfc = np.log2((cpm + pseudocount) / (cpm.loc["control"] + pseudocount))
    keep = (lfc.index != "control") & (sizes.reindex(lfc.index) >= min_cells).to_numpy()
    dropped = sorted(set(lfc.index[~keep]) - {"control"})
    lfc = lfc.loc[keep].copy()
    col = {g: i for i, g in enumerate(lfc.columns)}
    for r, lab in enumerate(lfc.index):
        for g in targets(lab):
            if g in col:
                lfc.iat[r, col[g]] = np.nan
    if dropped:
        print(f"dropped {len(dropped)} labels with < {min_cells} cells: {dropped}")
    return lfc


def readout_genes(ctrl, floor=EXPR_FLOOR):
    return list(ctrl.index[ctrl > floor])


# --------------------------------------------------------------------------- pairs
def pair_metrics(lfc, sizes, readout):
    """One row per double whose two singles are present; plus the residual matrix.

    Columns: A, B, n_AB/n_A/n_B (cells), eff_A/eff_B/eff_AB (mean |LFC| over readout),
    mean_abs_resid, median_abs_resid, rel_resid (= sum|resid| / sum|LFC_A + LFC_B|),
    c_A, c_B (Norman model), magnitude (= sqrt(c_A^2 + c_B^2)), nonadd_fit
    (= 1 - R^2 of the Norman model), nonadd_add (= 1 - R^2 of plain addition)."""
    L = lfc.reindex(columns=readout)
    labels = set(L.index)
    rows, resid = [], {}
    for lab in L.index:
        if "|" not in lab:
            continue
        a, b = targets(lab)
        if a not in labels or b not in labels:
            continue
        y, xa, xb = L.loc[lab].to_numpy(), L.loc[a].to_numpy(), L.loc[b].to_numpy()
        ok = np.isfinite(y) & np.isfinite(xa) & np.isfinite(xb)
        y, xa, xb = y[ok], xa[ok], xb[ok]
        r = y - xa - xb
        X = np.column_stack([xa, xb])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss = float(np.sum(y ** 2))
        full = np.full(L.shape[1], np.nan); full[ok] = r
        resid[lab] = full
        rows.append(dict(
            pair=lab, A=a, B=b,
            n_AB=int(sizes[lab]), n_A=int(sizes[a]), n_B=int(sizes[b]),
            eff_A=float(np.mean(np.abs(xa))), eff_B=float(np.mean(np.abs(xb))),
            eff_AB=float(np.mean(np.abs(y))),
            mean_abs_resid=float(np.mean(np.abs(r))), median_abs_resid=float(np.median(np.abs(r))),
            rel_resid=float(np.sum(np.abs(r)) / max(np.sum(np.abs(xa + xb)), 1e-12)),
            c_A=float(c[0]), c_B=float(c[1]), magnitude=float(np.hypot(*c)),
            nonadd_fit=float(np.sum((y - X @ c) ** 2) / ss),
            nonadd_add=float(np.sum(r ** 2) / ss),
            n_readout=int(ok.sum()),
        ))
    pt = pd.DataFrame(rows).set_index("pair")
    pt["log_n_AB"] = np.log10(pt["n_AB"])
    pt["log_n_single_min"] = np.log10(pt[["n_A", "n_B"]].min(axis=1))
    pt["log_eff_singles"] = np.log10(pt["eff_A"] + pt["eff_B"])
    return pt, pd.DataFrame(resid, index=readout).T


def classify_pairs(pt, mem):
    """Add memory status; for memory x regular pairs, orient the columns as mem/reg."""
    mem = set(mem)
    pt = pt.copy()
    pt["mem_A"], pt["mem_B"] = pt["A"].isin(mem), pt["B"].isin(mem)
    k = pt["mem_A"].astype(int) + pt["mem_B"].astype(int)
    pt["pair_class"] = np.select([k == 1, k == 0, k == 2], PAIR_CLASSES, default="")
    mx = pt["pair_class"] == "memory x regular"
    a_is_mem = pt["mem_A"]
    for col in ("", "c_", "eff_", "n_"):
        src_a, src_b = (f"{col}A", f"{col}B") if col else ("A", "B")
        pt[f"{col}mem" if col else "mem_gene"] = np.where(mx, np.where(a_is_mem, pt[src_a], pt[src_b]), None)
        pt[f"{col}reg" if col else "reg_gene"] = np.where(mx, np.where(a_is_mem, pt[src_b], pt[src_a]), None)
    return pt


# --------------------------------------------------------------------------- 3a: pairs
def _ols_coef(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def compare_pair_classes(pt, metric, covars=("log_n_AB", "log_n_single_min", "log_eff_singles"),
                         n_perm=10000, n_boot=1000, seed=0):
    """memory x regular vs regular x regular on `metric`.

    Unadjusted: Cliff's delta (+ bootstrap CI) and Mann-Whitney p.
    Adjusted:   OLS metric ~ 1 + is_memory_pair + covariates (z-scored); the is_memory
                coefficient is in units of the metric, p by permuting the class label.
    Also returns the covariate correlation matrix -- collinear covariates make the
    adjusted coefficient meaningless (never match on collinear covariates)."""
    d = pt[pt["pair_class"].isin(PAIR_CLASSES[:2])].dropna(subset=[metric, *covars])
    g = (d["pair_class"] == "memory x regular").to_numpy()
    y = d[metric].to_numpy(float)
    Z = d[list(covars)].to_numpy(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    X = np.column_stack([np.ones(len(y)), g.astype(float), Z])
    beta = _ols_coef(y, X)[1]
    rng = np.random.default_rng(seed)
    perm = np.array([_ols_coef(y, np.column_stack([X[:, :1], rng.permutation(g).astype(float), Z]))[1]
                     for _ in range(n_perm)])
    lo, hi = cliffs_ci(y[g], y[~g], n_boot=n_boot, seed=seed)
    return dict(
        metric=metric, n_mem=int(g.sum()), n_reg=int((~g).sum()),
        median_mem=float(np.median(y[g])), median_reg=float(np.median(y[~g])),
        cliffs_delta=float(cliffs_delta(y[g], y[~g])), ci_low=float(lo), ci_high=float(hi),
        mwu_p=float(mannwhitneyu(y[g], y[~g], alternative="two-sided")[1]),
        adj_coef=float(beta), adj_perm_p=float((np.sum(np.abs(perm) >= abs(beta)) + 1) / (n_perm + 1)),
        covar_corr=d[list(covars)].corr(method="spearman"),
    )


def dominance(pt, n_perm=10000, seed=0):
    """Does the memory gene dominate the double MORE than its single effect predicts?

    d_coef = log2(|c_mem| / |c_reg|) from the Norman model; d_eff = log2(eff_mem / eff_reg).
    The expected d_coef given d_eff is fitted on the regular x regular pairs (both
    orientations, so the fit is symmetric); excess = d_coef - expected, tested against
    0 by sign-flip permutation. Returns (per-pair table, summary dict)."""
    rr = pt[pt["pair_class"] == "regular x regular"]
    dc = np.log2(np.abs(rr["c_A"]) / np.abs(rr["c_B"]))
    de = np.log2(rr["eff_A"] / rr["eff_B"])
    x, y = np.r_[de, -de], np.r_[dc, -dc]
    ok = np.isfinite(x) & np.isfinite(y)
    slope = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2))        # through the origin
    mx = pt[pt["pair_class"] == "memory x regular"].copy()
    mx["d_coe"] = np.log2(np.abs(mx["c_mem"].astype(float)) / np.abs(mx["c_reg"].astype(float)))
    mx["d_ef"] = np.log2(mx["eff_mem"].astype(float) / mx["eff_reg"].astype(float))
    mx["excess"] = mx["d_coe"] - slope * mx["d_eff"]
    e = mx["excess"].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(e * rng.choice([-1, 1], len(e))) for _ in range(n_perm)])
    return mx[["mem_gene", "reg_gene", "c_mem", "c_reg", "eff_mem", "eff_reg",
               "d_coe", "d_ef", "excess"]], dict(
        n=len(e), slope_reg_pairs=slope, mean_excess=float(e.mean()),
        median_excess=float(np.median(e)),
        perm_p=float((np.sum(np.abs(null) >= abs(e.mean())) + 1) / (n_perm + 1)))


def leave_one_gene_out(pt, metrics=("rel_resid", "nonadd_fit"), n_perm=2000, seed=0):
    """Re-run the pair tests dropping all pairs of one memory gene at a time.

    Pairs sharing a target are not independent (ProPer-seq: TGFBR2 is in 9 of 21
    memory x regular pairs), so a result that dies when one gene goes is that gene's."""
    mx = pt["pair_class"] == "memory x regular"
    rows = []
    for g in [None] + sorted(pt.loc[mx, "mem_gene"].unique()):
        sub = pt if g is None else pt[~(mx & (pt["mem_gene"] == g))]
        row = {"dropped": g or "(none)", "n_mem_pairs": int((sub["pair_class"] == "memory x regular").sum())}
        for m in metrics:
            r = compare_pair_classes(sub, m, n_perm=n_perm, seed=seed)
            row[f"{m}_cliffs_delta"] = r["cliffs_delta"]
            row[f"{m}_adj_coe"] = r["adj_coef"]
            row[f"{m}_adj_p"] = r["adj_perm_p"]
        _, d = dominance(sub, n_perm=n_perm, seed=seed)
        row["dominance_excess"], row["dominance_p"] = d["mean_excess"], d["perm_p"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("dropped")


# --------------------------------------------------------------------------- Q1: gating
def conditional_effects(lfc, pt, readout):
    """Gating / sensitisation: does perturbing one gene change what the OTHER one does?

    For each pair A|B and each direction X given Y:
        effect of X alone          = LFC(X)
        effect of X when Y is hit  = LFC(A|B) - LFC(Y)
    amplification = log2( mean|X given Y| / mean|X alone| )   (>0: Y sensitises X)
    fidelity      = Pearson r( X given Y, X alone )            (<1: Y re-routes X)
    over the readout genes. Pure additivity gives amplification 0, fidelity 1 (up to
    noise, which pushes amplification UP and fidelity DOWN -- hence the adjusted test
    and the regular x regular reference, which carries the same noise).
    Returns one row per (pair, direction) with given = the conditioning gene."""
    L = lfc.reindex(columns=readout)
    rows = []
    for lab, p in pt.iterrows():
        ab = L.loc[lab].to_numpy()
        for x, y in ((p["A"], p["B"]), (p["B"], p["A"])):
            xa, ya = L.loc[x].to_numpy(), L.loc[y].to_numpy()
            ok = np.isfinite(ab) & np.isfinite(xa) & np.isfinite(ya)
            cond = ab[ok] - ya[ok]
            rows.append(dict(pair=lab, effect_of=x, given=y,
                             amplification=float(np.log2(np.mean(np.abs(cond)) / np.mean(np.abs(xa[ok])))),
                             fidelity=float(np.corrcoef(cond, xa[ok])[0, 1]),
                             log_n_AB=p["log_n_AB"], log_n_given=float(np.log10(p["n_A"] if y == p["A"] else p["n_B"])),
                             log_eff_effect=float(np.log10(np.mean(np.abs(xa[ok]))))))
    return pd.DataFrame(rows)


def gating_test(ce, pt, mem, n_perm=10000, seed=0):
    """Compare 'regular gene's effect GIVEN a memory gene is perturbed' (memory x regular
    pairs, direction reg-given-mem) with the same quantity in regular x regular pairs
    (both directions averaged per pair, so every pair counts once). And the converse
    direction (memory gene's effect given the regular gene). Covariate-adjusted as in
    compare_pair_classes."""
    mem = set(mem)
    cls = pt["pair_class"]
    out = {}
    for name, want_given_mem in (("regular gene, given memory gene hit", True),
                                 ("memory gene, given regular gene hit", False)):
        mx = ce[ce["pair"].map(cls).eq("memory x regular") &
                (ce["given"].isin(mem) if want_given_mem else ce["effect_of"].isin(mem))]
        rr = ce[ce["pair"].map(cls).eq("regular x regular")].groupby("pair").mean(numeric_only=True)
        d = pd.concat([mx.set_index("pair").assign(pair_class="memory x regular"),
                       rr.assign(pair_class="regular x regular")])
        cov = ("log_n_AB", "log_n_given", "log_eff_effect")
        out[name] = {m: compare_pair_classes(d, m, covars=cov, n_perm=n_perm, seed=seed)
                     for m in ("amplification", "fidelity")}
        out[name]["_table"] = d
    return out


# --------------------------------------------------------------------------- Q3: program
def matched_background(ctrl, readout, mem, n_bins=50, n_draws=20, seed=0):
    """n_draws expression-matched background sets for the memory readouts (same draw
    rule as matched_compare); returns a list of gene lists."""
    d = pd.DataFrame({"expr": ctrl.reindex(readout)})
    d["is_mem"] = d.index.isin(set(mem))
    d["bin"] = pd.qcut(d["expr"].rank(method="first"), n_bins, labels=False)
    need = d[d["is_mem"]].groupby("bin").size()
    pools = {b: g.index.to_numpy() for b, g in d[~d["is_mem"]].groupby("bin")}
    rng = np.random.default_rng(seed)
    return [list(np.concatenate([rng.choice(pools[b], int(k), replace=k > len(pools[b]))
                                 for b, k in need.items() if b in pools])) for _ in range(n_draws)]


def program_effects(lfc, ctrl, readout, mem, n_bins=50, n_draws=20, seed=0):
    """Per perturbation (single or double): how much it moves the MEMORY PROGRAM.

    mem_resp    mean |LFC| over memory readouts (own targets excluded via the NaN mask)
    bg_resp     same over expression-matched background readouts (mean of n_draws sets)
    specificity log2(mem_resp / bg_resp): >0 = hits memory genes more than comparable genes
    mem_shift   mean SIGNED LFC over memory readouts (direction of the program shift)"""
    mem_r = [g for g in mem if g in set(readout)]
    L = lfc.reindex(columns=readout)
    bgs = matched_background(ctrl, readout, mem_r, n_bins=n_bins, n_draws=n_draws, seed=seed)
    pe = pd.DataFrame(index=L.index)
    pe["mem_resp"] = L[mem_r].abs().mean(axis=1)
    pe["bg_resp"] = np.mean([L[b].abs().mean(axis=1).to_numpy() for b in bgs], axis=0)
    pe["specificity"] = np.log2(pe["mem_resp"] / pe["bg_resp"])
    pe["mem_shift"] = L[mem_r].mean(axis=1)
    pe["bg_shift"] = np.mean([L[b].mean(axis=1).to_numpy() for b in bgs], axis=0)
    pe["is_double"] = pe.index.str.contains("|", regex=False)
    return pe


def program_test(pe, pt, mem, sizes, n_perm=10000, seed=0):
    """Q3: (a) do memory-gene perturbations hit the memory program more specifically than
    regular-gene perturbations (singles; and pairs by class)? (b) in memory x regular
    pairs, does the regular gene push the program shift further or cancel it?
    (a) is covariate-adjusted for cells per condition."""
    mem = set(mem)
    sg = pe[~pe["is_double"]].copy()
    sg["pair_class"] = np.where(sg.index.isin(mem), "memory x regular", "regular x regular")
    sg["log_n"] = np.log10(sizes.reindex(sg.index).to_numpy())
    db = pe[pe["is_double"]].join(pt[["pair_class", "log_n_AB", "mem_gene", "reg_gene"]], how="inner")
    res = {"singles: memory vs regular target":
               compare_pair_classes(sg, "specificity", covars=("log_n",), n_perm=n_perm, seed=seed),
           "doubles: memory x regular vs regular x regular":
               compare_pair_classes(db, "specificity", covars=("log_n_AB",), n_perm=n_perm, seed=seed)}
    mx = db[db["pair_class"] == "memory x regular"].copy()
    mx["shift_mem_alone"] = pe.loc[mx["mem_gene"], "mem_shift"].to_numpy()
    mx["shift_reg_alone"] = pe.loc[mx["reg_gene"], "mem_shift"].to_numpy()
    mx["shift_double"] = mx["mem_shift"]
    mx["shift_additive"] = mx["shift_mem_alone"] + mx["shift_reg_alone"]
    mx["reg_pushes_further"] = np.sign(mx["shift_reg_alone"]) == np.sign(mx["shift_mem_alone"])
    mx["double_beyond_mem_alone"] = (np.abs(mx["shift_double"]) > np.abs(mx["shift_mem_alone"]))
    return res, sg, db, mx[["mem_gene", "reg_gene", "shift_mem_alone", "shift_reg_alone",
                            "shift_additive", "shift_double", "reg_pushes_further",
                            "double_beyond_mem_alone"]]


# --------------------------------------------------------------------------- 3b: readouts
def readout_table(lfc, resid, ctrl, readout):
    """Per readout gene: control expression, responsiveness to singles and doubles,
    mean |residual| over pairs, and excess_resid = log mean|resid| residualised on
    log single responsiveness across genes (non-additivity beyond plain responsiveness)."""
    L = lfc.reindex(columns=readout)
    single = [l for l in L.index if "|" not in l]
    double = [l for l in L.index if "|" in l]
    rt = pd.DataFrame({
        "ctrl_expr": ctrl.reindex(readout),
        "single_resp": L.loc[single].abs().mean(),
        "double_resp": L.loc[double].abs().mean(),
        "mean_abs_resid": resid.abs().mean(),
        "n_pairs": resid.notna().sum(),
    })
    x, y = np.log10(rt["single_resp"]), np.log10(rt["mean_abs_resid"])
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1)
    rt["excess_resid"] = y - np.polyval(b, x)
    # per-gene relative residual: sum over pairs |resid| / |LFC_A + LFC_B| -- the fraction
    # of the additive expectation that addition gets wrong; no global fit involved.
    pairs = list(resid.index)
    expect = np.vstack([L.loc[a].to_numpy() + L.loc[b].to_numpy()
                        for a, b in (targets(p) for p in pairs)])
    num = np.nansum(np.abs(resid.loc[pairs].to_numpy()), axis=0)
    den = np.nansum(np.abs(np.where(np.isfinite(resid.loc[pairs].to_numpy()), expect, np.nan)), axis=0)
    rt["rel_resid"] = num / np.where(den > 0, den, np.nan)
    return rt


def matched_compare(rt, mem, metric, n_bins=50, n_iter=20, n_boot=500, seed=0):
    """Memory vs EXPRESSION-MATCHED background on one readout metric.

    Same draw as marson_KO.compare_gene_responsiveness_expr_matched (rank-qcut bins,
    per-bin draw of as many background genes as memory genes, n_iter draws), plus the
    achieved SMD on log10 expression, which must be reported (5 bins left SMD 0.29)."""
    d = rt[["ctrl_expr", metric]].dropna().copy()
    d["is_mem"] = d.index.isin(set(mem))
    d["bin"] = pd.qcut(d["ctrl_expr"].rank(method="first"), n_bins, labels=False)
    m = d[d["is_mem"]]
    need = m.groupby("bin").size()
    pools = {b: g for b, g in d[~d["is_mem"]].groupby("bin")}
    rng = np.random.default_rng(seed)
    deltas, ps, smds, draws = [], [], [], []
    le_m = np.log10(m["ctrl_expr"])
    for _ in range(n_iter):
        pick = pd.concat([pools[b].iloc[rng.choice(len(pools[b]), int(k), replace=k > len(pools[b]))]
                          for b, k in need.items() if b in pools])
        v = pick[metric].to_numpy()
        draws.append(np.sort(v))
        deltas.append(cliffs_delta(m[metric], v))
        ps.append(mannwhitneyu(m[metric], v)[1])
        le_b = np.log10(pick["ctrl_expr"])
        smds.append((le_m.mean() - le_b.mean()) / np.sqrt((le_m.var() + le_b.var()) / 2))
    bg = np.vstack(draws).mean(0) if len({len(x) for x in draws}) == 1 else np.concatenate(draws)
    lo, hi = cliffs_ci(m[metric].to_numpy(), bg, n_boot=n_boot, seed=seed)
    return dict(metric=metric, n_mem=len(m), n_bg_universe=int((~d["is_mem"]).sum()),
                median_mem=float(m[metric].median()), median_bg=float(np.median(bg)),
                pct_diff=float((m[metric].median() / np.median(bg) - 1) * 100),
                cliffs_delta=float(np.median(deltas)), ci_low=float(lo), ci_high=float(hi),
                mwu_p_median=float(np.median(ps)), achieved_smd=float(np.median(smds)),
                mem_values=m[metric].to_numpy(), bg_values=bg)


# --------------------------------------------------------------------------- one call
def analyse(ds, top_n=TOP_N, mem_source=None, floor=EXPR_FLOOR, min_cells=MIN_CELLS,
            pair_metrics_list=("mean_abs_resid", "rel_resid", "nonadd_fit", "magnitude"),
            readout_metrics=("single_resp", "double_resp", "mean_abs_resid", "rel_resid", "excess_resid"),
            n_bins=50, n_iter=20, n_perm=10000, seed=0):
    """Everything for one dataset. mem_source = which dataset's Power-Seek list defines
    'memory' (default: its own); top_n is the list cut (300 = Fig 2 convention)."""
    counts, sizes, ctrl = load(ds)
    mem = memory_genes(mem_source or ds, top_n=top_n)
    readout = readout_genes(ctrl, floor)
    lfc = compute_lfc(counts, sizes, min_cells=min_cells)
    pt, resid = pair_metrics(lfc, sizes, readout)
    pt = classify_pairs(pt, mem)
    print(f"{ds}: {len(readout)} readout genes > {floor}; {len(pt)} pairs "
          f"{pt['pair_class'].value_counts().to_dict()}; memory = {mem_source or ds} top-{top_n} "
          f"({len(set(mem) & set(readout))} of them in the readout)")
    rt = readout_table(lfc, resid, ctrl, readout)
    pair_res = pd.DataFrame([{k: v for k, v in compare_pair_classes(pt, m, n_perm=n_perm, seed=seed).items()
                              if k != "covar_corr"} for m in pair_metrics_list]).set_index("metric")
    covar_corr = compare_pair_classes(pt, pair_metrics_list[0], n_perm=1)["covar_corr"]
    dom_tab, dom = dominance(pt, n_perm=n_perm, seed=seed)
    logo = leave_one_gene_out(pt, n_perm=min(n_perm, 2000), seed=seed)
    ro = {m: matched_compare(rt, mem, m, n_bins=n_bins, n_iter=n_iter, seed=seed) for m in readout_metrics}
    ro_res = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_values")} for r in ro.values()]
                          ).set_index("metric")
    return dict(ds=ds, mem=mem, readout=readout, lfc=lfc, pairs=pt, resid=resid, readout_table=rt,
                pair_results=pair_res, covar_corr=covar_corr, dominance_table=dom_tab,
                dominance=dom, leave_one_gene_out=logo, readout_results=ro_res, readout_draws=ro)
