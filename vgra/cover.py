"""CoVeR: conformal view referral (Part B; rules to be frozen in protocol amendment v1.2 before any test use).

Policy for the fracture finding and one detector, per PA + LAT pair (study side):
  t                    = the Part A primary threshold: 95th percentile of the max(PA, LAT) score of the
                         valid negatives (95% specificity), so "always both views" is the Part A decision
  PA score > t         -> positive from the PA view alone. Exact: max(PA, LAT) >= PA, so both views agree
  PA score <= lo       -> negative from the PA view alone (the lateral view is not read)
  lo < PA score <= t   -> refer: read the lateral view; positive if max(PA, LAT) > t

Risk that lo is calibrated against (relative to "always both views", so every alpha is feasible):
  extra miss = a fracture-positive pair that both views catch (max > t) but the gate calls negative.
  Unit = the patient, because pairs of one patient are not exchangeable (follow-up studies, both wrists).
  Loss of a patient with at least one positive pair = share of their positive pairs that are extra misses;
  risk = mean loss over such patients. lo = REFER_ALL (-1) reads every lateral view below t: the gate is then
  exactly "always both views" and the risk is 0 for any data.

Calibration of lo on valid (largest point of a fixed, data-independent grid that passes; the risk is
non-decreasing in lo, so the search runs from small to large lo and stops at the first failure):
  naive  empirical risk <= alpha                        (no guarantee; reference only)
  crc    conformal risk control, loss bound B = 1:  N/(N+1) * risk + 1/(N+1) <= alpha
         -> expected extra-miss risk on new patients <= alpha (Angelopoulos et al., ICLR 2024)
  ltt    learn-then-test, Hoeffding-Bentkus p-value <= delta, fixed-sequence testing
         -> risk <= alpha with probability >= 1 - delta over the calibration set
            (Bates et al., J ACM 2021; Angelopoulos et al., Ann Appl Stat 2025)
Grouping "age": lo is calibrated separately in every age band of protocol 4.4 (Mondrian / group-conditional);
the unit is then (patient, age band). A band without a calibrated lo reads every lateral view (REFER_ALL).

Technical note: t is a quantile of the calibration negatives, and some patients have positive and negative
pairs, so calibration and new patients are exchangeable only conditionally on t; the design study
(repeated patient half-splits of valid) checks the guarantees empirically with exactly this structure.

Everything here runs on CPU from the study tables of vgra.study (fracture rows); nothing here sees images.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binom

from vgra import study as st

FINDING = "fracture"
ALPHAS = (0.01, 0.02)
DELTA = 0.1
METHODS = ("naive", "crc", "ltt")
GROUPINGS = ("all", "age")
REFER_ALL = -1.0
GRID = np.concatenate([[REFER_ALL, 0.0], np.logspace(-3, 0, 3001)])
# Primary configuration: PROPOSAL, frozen only by protocol amendment v1.2 (docs/PROTOCOL.md).
PRIMARY = {"method": "ltt", "alpha": 0.02, "grouping": "all"}
SUBGROUP_COLS = ("age_band", "torus", "initial_exam", "cast")


# ----------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------

def decide(pa, lat, t: float, lo) -> tuple[np.ndarray, np.ndarray]:
    """Gate decision and referral flags. lo: scalar or one value per row."""
    pa, lat = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), pa.shape)
    auto_pos = pa > t
    refer = ~auto_pos & (pa > lo)
    decision = auto_pos | (refer & (np.maximum(pa, lat) > t))
    return decision, refer


def both_views(pa, lat, t: float) -> np.ndarray:
    """Part A primary decision (always both views, max rule)."""
    return np.maximum(np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64)) > t


def rescued(pa, lat, y, t: float) -> np.ndarray:
    """Positive pairs that both views catch but the PA view alone (at t) does not."""
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    return y & (pa <= t) & (lat > t)


def oracle_reads(pa, lat, t: float) -> float:
    """Smallest share of lateral reads that reproduces 'always both views' exactly (needs the labels' outcome:
    the lateral view changes the decision). Reference only."""
    pa, lat = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return float(((pa <= t) & (lat > t)).mean()) if pa.size else float("nan")


# ----------------------------------------------------------------------------------------------
# risk and calibration
# ----------------------------------------------------------------------------------------------

def units_of(patient_id) -> np.ndarray:
    return np.asarray(patient_id).astype(str)


def unit_loss(extra, y, units) -> tuple[np.ndarray, np.ndarray]:
    """Per unit with at least one positive pair: (units, share of its positive pairs that are extra misses)."""
    y = np.asarray(y).astype(bool)
    extra = np.asarray(extra).astype(bool) & y
    u, inv = np.unique(np.asarray(units)[y], return_inverse=True)
    if u.size == 0:
        return u, np.zeros(0)
    n_pos = np.bincount(inv)
    n_extra = np.bincount(inv, weights=extra[y].astype(float))
    return u, n_extra / n_pos


def patient_risk(extra, y, units) -> float:
    _, loss = unit_loss(extra, y, units)
    return float(loss.mean()) if loss.size else float("nan")


def risk_curve(pa, lat, y, t: float, units, grid=GRID) -> tuple[np.ndarray, int]:
    """Empirical extra-miss risk at every grid value of lo, and N = units with a positive pair."""
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    units = np.asarray(units)
    u, inv = np.unique(units[y], return_inverse=True)
    n_units = int(u.size)
    grid = np.asarray(grid, dtype=np.float64)
    if n_units == 0:
        return np.zeros(grid.size), 0
    weight = 1.0 / np.bincount(inv)[inv]                  # per positive pair: 1 / positives of its unit
    resc = rescued(pa, lat, y, t)[y]
    s_pa = pa[y][resc]
    order = np.argsort(s_pa, kind="stable")
    s_pa, cw = s_pa[order], np.cumsum(weight[resc][order])
    k = np.searchsorted(s_pa, grid, side="right")         # rescued pairs with PA <= lo
    risk = np.where(k > 0, cw[np.maximum(k - 1, 0)] if cw.size else 0.0, 0.0) / n_units
    return risk, n_units


def hb_pvalue(risk, n: int, alpha: float) -> np.ndarray:
    """Hoeffding-Bentkus p-value for H0: risk > alpha, losses in [0, 1] (Bates et al. 2021)."""
    risk = np.asarray(risk, dtype=np.float64)
    if n <= 0:
        return np.ones_like(risk)
    a = np.minimum(risk, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = np.where(a > 0, a * np.log(a / alpha), 0.0) + (1 - a) * np.log((1 - a) / (1 - alpha))
    hoeffding = np.exp(-n * h1)
    bentkus = math.e * binom.cdf(np.ceil(n * risk - 1e-9), n, alpha)
    return np.minimum(1.0, np.minimum(hoeffding, bentkus))


def passes(risk, n: int, alpha: float, method: str, delta: float = DELTA) -> np.ndarray:
    risk = np.asarray(risk, dtype=np.float64)
    if method == "naive":
        return risk <= alpha
    if method == "crc":
        return (n / (n + 1)) * risk + 1.0 / (n + 1) <= alpha + 1e-12
    if method == "ltt":
        return hb_pvalue(risk, n, alpha) <= delta
    raise ValueError(f"unknown method {method!r}")


def calibrate(pa, lat, y, t: float, patient_id, alpha: float, method: str = "ltt", delta: float = DELTA,
              groups=None, grid=GRID) -> dict:
    """lo per group ({'all': lo} without groups). Fixed-sequence search from small to large lo."""
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    pid = np.asarray(patient_id)
    grid = np.asarray(grid, dtype=np.float64)
    keys = ["all"] if groups is None else sorted(set(np.asarray(groups).tolist()))
    out = {}
    for g in keys:
        m = np.ones(pa.size, dtype=bool) if groups is None else (np.asarray(groups) == g)
        risk, n = risk_curve(pa[m], lat[m], y[m], t, units_of(pid[m]), grid)
        ok = passes(risk, n, alpha, method, delta) & (grid <= t)
        ok[grid == REFER_ALL] = True                      # risk is exactly 0 there
        first_fail = int(np.argmin(ok)) if not ok.all() else ok.size
        out[g] = float(grid[max(first_fail - 1, 0)])
    return out


def lo_per_row(lo_map: dict, groups=None, n: int | None = None) -> np.ndarray:
    if groups is None:
        if "all" not in lo_map:
            raise ValueError("lo_map has no 'all' entry; pass the groups")
        return np.full(int(n), lo_map["all"], dtype=np.float64)
    return np.array([lo_map.get(g, REFER_ALL) for g in np.asarray(groups).tolist()], dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# metrics and baselines
# ----------------------------------------------------------------------------------------------

def summarize(pa, lat, y, t: float, lo_rows, patient_id, age=None) -> dict:
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    dec, refer = decide(pa, lat, t, lo_rows)
    both = both_views(pa, lat, t)
    extra = y & both & ~dec
    units = units_of(patient_id)
    r = {"n_pairs": int(y.size), "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "n_refer": int(refer.sum()),
         "refer_rate": float(refer.mean()), "auto_pos_rate": float((pa > t).mean()),
         "sens": float(dec[y].mean()), "spec": float((~dec[~y]).mean()),
         "sens_both": float(both[y].mean()), "spec_both": float((~both[~y]).mean()),
         "extra_pair": float(extra[y].mean()), "extra_patient": patient_risk(extra, y, units),
         "oracle_reads": oracle_reads(pa, lat, t)}
    if age is not None:
        worst, worst_band = -1.0, ""
        age = np.asarray(age)
        for band in sorted(set(age.tolist())):
            m = age == band
            if (y & m).any():
                v = patient_risk(extra[m], y[m], units[m])
                if v > worst:
                    worst, worst_band = v, band
        r["extra_worst_age"], r["worst_age_band"] = worst, worst_band
    return r


def same_budget_random(pa, lat, y, t: float, n_refer: int) -> dict:
    """Expected metrics if n_refer lateral views are read for random pairs among those with PA <= t."""
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    cand = pa <= t
    m = int(cand.sum())
    q = min(int(n_refer), m) / m if m else 0.0
    both = both_views(pa, lat, t)
    auto = pa > t
    tp = (y & auto).sum() + q * (y & cand & both).sum()
    fp = (~y & auto).sum() + q * (~y & cand & both).sum()
    extra = rescued(pa, lat, y, t)
    return {"sens": float(tp / y.sum()) if y.any() else float("nan"),
            "spec": float(1 - fp / (~y).sum()) if (~y).any() else float("nan"),
            "extra_pair": float((1 - q) * extra[y].mean()) if y.any() else float("nan")}


def same_budget_ranked(rank, pa, lat, y, t: float, n_refer: int) -> dict:
    """Metrics if the n_refer pairs with the highest rank score (among PA <= t) are referred."""
    pa, lat, y = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64), np.asarray(y).astype(bool)
    cand = np.flatnonzero(pa <= t)
    order = cand[np.argsort(-np.asarray(rank, dtype=np.float64)[cand], kind="stable")]
    refer = np.zeros(pa.size, dtype=bool)
    refer[order[:int(n_refer)]] = True
    dec = (pa > t) | (refer & both_views(pa, lat, t))
    extra = y & both_views(pa, lat, t) & ~dec
    return {"sens": float(dec[y].mean()) if y.any() else float("nan"),
            "spec": float((~dec[~y]).mean()) if (~y).any() else float("nan"),
            "extra_pair": float(extra[y].mean()) if y.any() else float("nan")}


# ----------------------------------------------------------------------------------------------
# tables of one run
# ----------------------------------------------------------------------------------------------

def fracture_rows(tab: pd.DataFrame) -> pd.DataFrame:
    t = tab[tab["finding"] == FINDING] if "finding" in tab else tab
    return t.reset_index(drop=True)


def ensemble_table(tabs: dict) -> pd.DataFrame:
    """Mean PA and LAT scores over runs evaluated on the same pairs in the same order."""
    runs = sorted(tabs)
    base = fracture_rows(tabs[runs[0]]).copy()
    for r in runs[1:]:
        if not np.array_equal(fracture_rows(tabs[r])["pair_id"].values, base["pair_id"].values):
            raise ValueError("runs are not evaluated on the same pairs in the same order")
    base["pa"] = np.mean([fracture_rows(tabs[r])["pa"].values for r in runs], axis=0)
    base["lat"] = np.mean([fracture_rows(tabs[r])["lat"].values for r in runs], axis=0)
    return base


def disagreement(tabs: dict) -> np.ndarray:
    """Per pair: standard deviation of the PA score over runs (rank score of the uncertainty baseline)."""
    runs = sorted(tabs)
    return np.std([fracture_rows(tabs[r])["pa"].values for r in runs], axis=0)


# ----------------------------------------------------------------------------------------------
# design study on valid: repeated patient half-splits
# ----------------------------------------------------------------------------------------------

def design_study(tab: pd.DataFrame, alphas=ALPHAS, methods=METHODS, groupings=GROUPINGS, delta: float = DELTA,
                 n_splits: int = 300, seed: int = 0, rank_scores: dict | None = None) -> pd.DataFrame:
    """Calibrate t and lo on half of the valid patients, evaluate on the other half; repeat."""
    tab = fracture_rows(tab)
    pa, lat = tab["pa"].values.astype(float), tab["lat"].values.astype(float)
    y, pid = tab["y"].values.astype(bool), tab["patient_id"].values
    age = tab["age_band"].values if "age_band" in tab else np.full(len(tab), "all")
    uniq = np.unique(pid)
    rng = np.random.default_rng(seed)
    rows = []
    for split in range(n_splits):
        cal = np.isin(pid, rng.choice(uniq, uniq.size // 2, replace=False))
        ev = ~cal
        t = st.threshold_at(np.maximum(pa, lat)[cal & ~y])
        t_pa = st.threshold_at(pa[cal & ~y])
        pa_only = pa[ev] > t_pa
        ye = y[ev]
        base = {"split": split, "t": t, "sens_pa": float(pa_only[ye].mean()), "spec_pa": float((~pa_only[~ye]).mean())}
        for grouping in groupings:
            g_cal = None if grouping == "all" else age[cal]
            g_ev = None if grouping == "all" else age[ev]
            for method in methods:
                for alpha in alphas:
                    lo = calibrate(pa[cal], lat[cal], y[cal], t, pid[cal], alpha, method, delta, g_cal)
                    lor = lo_per_row(lo, g_ev, int(ev.sum()))
                    r = summarize(pa[ev], lat[ev], ye, t, lor, pid[ev], age[ev])
                    rnd = same_budget_random(pa[ev], lat[ev], ye, t, r["n_refer"])
                    r.update({f"random_{k}": v for k, v in rnd.items()})
                    for name, score in (rank_scores or {}).items():
                        rk = same_budget_ranked(np.asarray(score)[ev], pa[ev], lat[ev], ye, t, r["n_refer"])
                        r.update({f"{name}_{k}": v for k, v in rk.items()})
                    r.update(base)
                    r.update(grouping=grouping, method=method, alpha=alpha, lo=lo.get("all", float("nan")),
                             held=bool(r["extra_patient"] <= alpha), held_all_ages=bool(r["extra_worst_age"] <= alpha))
                    rows.append(r)
    return pd.DataFrame(rows)


def design_summary(design: pd.DataFrame) -> pd.DataFrame:
    keys = ["run", "grouping", "method", "alpha"] if "run" in design else ["grouping", "method", "alpha"]
    num = [c for c in design.columns if c not in keys + ["split", "worst_age_band"] and design[c].dtype != object]
    g = design.groupby(keys, sort=False)
    out = g[num].mean().reset_index()
    out["refer_p05"] = g["refer_rate"].quantile(0.05).values
    out["refer_p95"] = g["refer_rate"].quantile(0.95).values
    out["extra_patient_p90"] = g["extra_patient"].quantile(0.90).values
    out["splits"] = g.size().values
    return out


def risk_referral_curve(tab: pd.DataFrame, points: int = 200) -> pd.DataFrame:
    """Full valid (descriptive): for a thinned set of lo values, share of lateral reads and the risks."""
    tab = fracture_rows(tab)
    pa, lat = tab["pa"].values.astype(float), tab["lat"].values.astype(float)
    y, pid = tab["y"].values.astype(bool), tab["patient_id"].values
    t = st.threshold_at(np.maximum(pa, lat)[~y])
    grid = GRID[GRID <= t]
    keep = np.unique(np.linspace(0, grid.size - 1, min(points, grid.size)).astype(int))
    rows = []
    for lo in grid[keep]:
        r = summarize(pa, lat, y, t, lo, pid)
        rows.append({"lo": lo, "t": t, **{k: r[k] for k in ("refer_rate", "sens", "spec", "extra_pair",
                                                          "extra_patient", "sens_both", "spec_both")}})
    return pd.DataFrame(rows)


def who_is_referred(tab: pd.DataFrame, lo_map: dict, grouping: str, t: float | None = None) -> pd.DataFrame:
    """Share of pairs referred and the extra-miss risk per subgroup, on the table the policy is applied to.
    t defaults to the 95%-specificity threshold of the table itself (valid, descriptive use)."""
    tab = fracture_rows(tab)
    pa, lat = tab["pa"].values.astype(float), tab["lat"].values.astype(float)
    y, pid = tab["y"].values.astype(bool), tab["patient_id"].values
    if t is None:
        t = st.threshold_at(np.maximum(pa, lat)[~y])
    groups = tab["age_band"].values if grouping == "age" else None
    dec, refer = decide(pa, lat, t, lo_per_row(lo_map, groups, len(tab)))
    extra = y & both_views(pa, lat, t) & ~dec
    rows = []
    masks = {("all", "all"): np.ones(len(tab), dtype=bool)}
    for col in SUBGROUP_COLS:
        if col in tab:
            for level in sorted(set(tab[col].tolist())):
                masks[(col, str(level))] = tab[col].values == level
    for (col, level), m in masks.items():
        rows.append({"subgroup": col, "level": level, "n_pairs": int(m.sum()), "n_pos": int((y & m).sum()),
                     "refer_rate": float(refer[m].mean()) if m.any() else float("nan"),
                     "extra_pair": float(extra[m & y].mean()) if (m & y).any() else float("nan"),
                     "extra_patient": patient_risk(extra[m], y[m], units_of(pid[m])) if (m & y).any() else float("nan")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# confirmatory use on test (only after amendment v1.2): calibrate on all of valid, apply to test
# ----------------------------------------------------------------------------------------------

def _patient_aggregates(pid, cols: dict) -> tuple[np.ndarray, dict]:
    uniq, inv = np.unique(np.asarray(pid), return_inverse=True)
    return uniq, {k: np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=uniq.size) for k, v in cols.items()}


def confirm(tab_v: pd.DataFrame, tab_e: pd.DataFrame, alpha: float, method: str, grouping: str,
            delta: float = DELTA, n_boot: int = 2000, seed: int = 0, rank_e: dict | None = None) -> dict:
    """One configuration: lo from all of valid, metrics on the evaluation split with patient-bootstrap CIs."""
    v, e = fracture_rows(tab_v), fracture_rows(tab_e)
    pa_v, lat_v, y_v = v["pa"].values.astype(float), v["lat"].values.astype(float), v["y"].values.astype(bool)
    t = st.threshold_at(np.maximum(pa_v, lat_v)[~y_v])            # = Part A primary threshold
    t_pa = st.threshold_at(pa_v[~y_v])
    g_v = v["age_band"].values if grouping == "age" else None
    g_e = e["age_band"].values if grouping == "age" else None
    lo = calibrate(pa_v, lat_v, y_v, t, v["patient_id"].values, alpha, method, delta, g_v)
    pa, lat, y, pid = e["pa"].values.astype(float), e["lat"].values.astype(float), e["y"].values.astype(bool), e["patient_id"].values
    lor = lo_per_row(lo, g_e, len(e))
    dec, refer = decide(pa, lat, t, lor)
    both = both_views(pa, lat, t)
    pa_only = pa > t_pa
    extra = y & both & ~dec
    point = summarize(pa, lat, y, t, lor, pid, e["age_band"].values if "age_band" in e else None)
    point.update(sens_pa=float(pa_only[y].mean()), spec_pa=float((~pa_only[~y]).mean()))
    rnd = same_budget_random(pa, lat, y, t, point["n_refer"])
    point.update({f"random_{k}": val for k, val in rnd.items()})
    for name, score in (rank_e or {}).items():
        rk = same_budget_ranked(score, pa, lat, y, t, point["n_refer"])
        point.update({f"{name}_{k}": val for k, val in rk.items()})
    for name, res in (("mcnemar_pos", st.mcnemar(dec[y], both[y])),       # c = extra misses, b = 0 by construction
                      ("mcnemar_neg", st.mcnemar(~dec[~y], ~both[~y]))):  # b = false alarms avoided
        point.update({f"{name}_b": res["b"], f"{name}_c": res["c"], f"{name}_p": res["p"]})

    # whole-patient bootstrap on per-patient sums (same resampling scheme as vgra.study.PatientBootstrap)
    cand = pa <= t
    _, ag = _patient_aggregates(pid, {
        "pos": y, "neg": ~y, "pairs": np.ones(len(e)), "refer": refer, "cand": cand,
        "tp": dec & y, "fp": dec & ~y, "tp_both": both & y, "fp_both": both & ~y,
        "tp_auto": (pa > t) & y, "fp_auto": (pa > t) & ~y, "resc_pos": cand & both & y, "resc_neg": cand & both & ~y,
        "extra": extra})
    has_pos = ag["pos"] > 0
    frac = np.where(has_pos, ag["extra"] / np.maximum(ag["pos"], 1), 0.0)
    rng = np.random.default_rng(seed)
    n_u = frac.size
    boots = {k: [] for k in ("refer_rate", "sens", "spec", "extra_pair", "extra_patient", "sens_diff_both",
                             "spec_diff_both", "sens_diff_random")}
    for _ in range(int(n_boot)):
        c = np.bincount(rng.integers(0, n_u, n_u), minlength=n_u).astype(float)
        s = {k: float((c * v).sum()) for k, v in ag.items()}
        q = s["refer"] / s["cand"] if s["cand"] else 0.0
        sens = s["tp"] / s["pos"]
        boots["refer_rate"].append(s["refer"] / s["pairs"])
        boots["sens"].append(sens)
        boots["spec"].append(1 - s["fp"] / s["neg"])
        boots["extra_pair"].append(s["extra"] / s["pos"])
        boots["extra_patient"].append(float((c * frac)[has_pos].sum() / c[has_pos].sum()))
        boots["sens_diff_both"].append(sens - s["tp_both"] / s["pos"])
        boots["spec_diff_both"].append((1 - s["fp"] / s["neg"]) - (1 - s["fp_both"] / s["neg"]))
        boots["sens_diff_random"].append(sens - (s["tp_auto"] + q * s["resc_pos"]) / s["pos"])
    for k, vals in boots.items():
        point[f"{k}_lo"], point[f"{k}_hi"] = st._ci(vals)
    point.update(method=method, alpha=alpha, grouping=grouping, delta=delta, t=t, t_pa=t_pa,
                 lo=lo.get("all", float("nan")), lo_map=json.dumps(lo, sort_keys=True),
                 held=bool(point["extra_patient"] <= alpha))
    return point
