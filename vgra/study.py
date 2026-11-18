"""Part A, patient (study) level analysis (I2; docs/PROTOCOL.md section 4).

For every PA + LAT pair and finding:
  - image score   = highest detector confidence of the class in the image after NMS
                    (= own-image maximum if it exceeds the NMS confidence 0.001, else 0)
  - study label   = the finding is annotated in the PA or the LAT image
  - study scores  = PA only, LAT only, and the two fused by max (primary rule), mean, noisy-OR and
                    logistic stacking on [logit PA, logit LAT] fitted on valid
  - threshold     = 95th percentile of the negative-study scores on valid (95% specificity),
                    applied unchanged to the evaluation split; a study is called positive if its
                    score is above the threshold
Statistics: sensitivity / specificity / missed per 100 positive studies, AUC, patient-bootstrap 95% CIs
(whole patients resampled), McNemar on the paired decisions of the positive studies (exact if fewer than
25 discordant pairs), DeLong for paired AUCs, Holm within families, subgroups, and the pooled bootstrap
over seeds.

Everything here runs on CPU from the prediction tables of scripts/10_predict.py; nothing here sees images.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm, rankdata

from vgra import premise as pm

SETTINGS = ("pa", "lat", "max", "mean", "noisyor", "stack")
SINGLE = ("pa", "lat")
RULES = ("max", "mean", "noisyor", "stack")
PRIMARY_RULE = "max"
PRIMARY_FINDING = "fracture"
FINDINGS = ("fracture", "periostealreaction", "pronatorsign", "softtissue", "metal")  # protocol 1
SPECIFICITY = 0.95
NMS_CONF = 0.001
MCNEMAR_EXACT_BELOW = 25
AGE_BANDS = ((0.0, 6.0, "<6"), (6.0, 10.0, "6-<10"), (10.0, 14.0, "10-<14"), (14.0, math.inf, ">=14"))
TORUS_CODE = "M/2.1"
# Contrasts (new, reference). The primary test is (max, pa) for fracture.
CONTRASTS = (("max", "pa"), ("max", "lat"), ("lat", "pa"), ("mean", "max"), ("noisyor", "max"), ("stack", "max"))


# ----------------------------------------------------------------------------------------------
# study table
# ----------------------------------------------------------------------------------------------

def image_score(own_max, conf: float = NMS_CONF) -> np.ndarray:
    """Highest confidence of a class after NMS: the own-image maximum if it passes the NMS confidence
    (Ultralytics keeps candidates with score > conf, and the top candidate of a class always survives NMS)."""
    own_max = np.asarray(own_max, dtype=np.float64)
    return np.where(own_max > conf, own_max, 0.0)


def age_band(age) -> str:
    if age is None or not np.isfinite(age):
        return "unknown"
    for lo, hi, name in AGE_BANDS:
        if lo <= age < hi:
            return name
    return "unknown"


def attach_metadata(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add subgroup columns to the pair list (protocol 4.4).

    meta: dataset.csv rows (filestem, patient_id, study_number, age, initial_exam, cast, ao_classification).
      age       = age on the PA image (both images of a pair come from the same study)
      torus     = any image of the same study (patient, study number) has an AO code containing M/2.1
      initial   = initial_exam is 1 on either image of the pair
      cast      = cast is 1 on either image of the pair
    """
    m = meta.set_index("filestem")
    out = pairs.copy()
    out["age"] = m["age"].reindex(out["ap_image_id"]).astype(float).values
    out["age_band"] = [age_band(a) for a in out["age"]]
    ao = meta["ao_classification"].fillna("").astype(str).str.contains(TORUS_CODE, regex=False)
    torus_study = set(zip(meta.loc[ao, "patient_id"].astype(int), meta.loc[ao, "study_number"].astype(int)))
    out["torus"] = [int((int(p), int(s)) in torus_study) for p, s in zip(out["patient_id"], out["study_number"])]
    for col in ("initial_exam", "cast"):
        a = m[col].reindex(out["ap_image_id"]).fillna(0).astype(float).values
        b = m[col].reindex(out["lat_image_id"]).fillna(0).astype(float).values
        out[col] = (np.maximum(a, b) >= 1).astype(int)
    return out


def study_table(imgs: pd.DataFrame, pairs: pd.DataFrame, classes: list, findings=FINDINGS,
                conf: float = NMS_CONF) -> pd.DataFrame:
    """One row per pair and finding: pa, lat (image scores), y_pa, y_lat, y (study label) + pair columns.

    imgs: per-image, per-class table of vgra.infer.predict (image_id, cls, own_max, present).
    """
    key = imgs.set_index(["image_id", "cls"])
    need = set(pairs["ap_image_id"]) | set(pairs["lat_image_id"])
    missing = need - set(imgs["image_id"])
    if missing:
        raise ValueError(f"{len(missing)} paired images have no predictions, e.g. {sorted(missing)[:3]}")
    keep = [c for c in pairs.columns if c not in ("ap_image_id", "lat_image_id") and not c.startswith("state_")]
    rows = []
    for f in findings:
        c = classes.index(f)
        pa = key.loc[list(zip(pairs["ap_image_id"], [c] * len(pairs)))]
        lat = key.loc[list(zip(pairs["lat_image_id"], [c] * len(pairs)))]
        t = pairs[keep].copy()
        t["finding"] = f
        t["pa"] = image_score(pa["own_max"].values, conf)
        t["lat"] = image_score(lat["own_max"].values, conf)
        t["y_pa"] = (pa["present"].values > 0).astype(int)
        t["y_lat"] = (lat["present"].values > 0).astype(int)
        t["y"] = np.maximum(t["y_pa"], t["y_lat"])
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# fusion and operating point
# ----------------------------------------------------------------------------------------------

def stack_features(pa, lat) -> np.ndarray:
    return np.stack([pm.logit(pa), pm.logit(lat)], 1)


def fit_stack(pa, lat, y) -> dict:
    return pm.fit_logistic(stack_features(pa, lat), np.asarray(y))


def fuse(pa, lat, setting: str, stack_model: dict | None = None) -> np.ndarray:
    pa, lat = np.asarray(pa, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if setting == "pa":
        return pa
    if setting == "lat":
        return lat
    if setting == "max":
        return np.maximum(pa, lat)
    if setting == "mean":
        return (pa + lat) / 2
    if setting == "noisyor":
        return 1 - (1 - pa) * (1 - lat)
    if setting == "stack":
        if stack_model is None:
            raise ValueError("stacking needs a model fitted on valid")
        return pm.predict_logistic(stack_model, stack_features(pa, lat))
    raise ValueError(f"unknown setting {setting!r}")


def threshold_at(neg_scores, specificity: float = SPECIFICITY) -> float:
    """Threshold with the given specificity on the negative scores (decision: score > threshold)."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.size == 0:
        return float("nan")
    return float(np.quantile(neg, specificity))


def rates(y, decision) -> dict:
    y, d = np.asarray(y).astype(bool), np.asarray(decision).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    sens = float(d[y].mean()) if n_pos else float("nan")
    spec = float((~d[~y]).mean()) if n_neg else float("nan")
    return {"n_pos": n_pos, "n_neg": n_neg, "tp": int((d & y).sum()), "fn": int((~d & y).sum()),
            "fp": int((d & ~y).sum()), "sens": sens, "spec": spec,
            "missed_per_100": 100 * (1 - sens) if n_pos else float("nan")}


# ----------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------

def mcnemar(dec_new, dec_ref, exact_below: int = MCNEMAR_EXACT_BELOW) -> dict:
    """Paired test of two decisions on the same studies (use the positive studies for sensitivity).

    b = new right, reference wrong; c = reference right, new wrong. Exact two-sided binomial test if
    b + c < exact_below, else the chi-square test with continuity correction.
    """
    a, r = np.asarray(dec_new).astype(bool), np.asarray(dec_ref).astype(bool)
    b, c = int((a & ~r).sum()), int((~a & r).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "method": "none"}
    if n < exact_below:
        return {"b": b, "c": c, "p": float(binomtest(b, n, 0.5).pvalue), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "p": float(chi2.sf(stat, 1)), "method": "chi2_cc"}


def delong(y, s_new, s_ref) -> dict:
    """DeLong test for two correlated AUCs (fast algorithm with midranks, Sun and Xu 2014)."""
    y = np.asarray(y).astype(bool)
    s = np.stack([np.asarray(s_new, dtype=np.float64), np.asarray(s_ref, dtype=np.float64)])
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return {"auc_new": float("nan"), "auc_ref": float("nan"), "diff": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    pos, neg = s[:, y], s[:, ~y]
    v01, v10, aucs = np.empty((2, m)), np.empty((2, n)), np.empty(2)
    for k in range(2):
        tx, ty = rankdata(pos[k]), rankdata(neg[k])
        tz = rankdata(np.concatenate([pos[k], neg[k]]))
        aucs[k] = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
        v01[k] = (tz[:m] - tx) / n
        v10[k] = 1 - (tz[m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 1e-18:
        return {"auc_new": float(aucs[0]), "auc_ref": float(aucs[1]), "diff": diff, "lo": diff, "hi": diff,
                "p": 1.0 if abs(diff) < 1e-15 else 0.0}
    se = math.sqrt(var)
    return {"auc_new": float(aucs[0]), "auc_ref": float(aucs[1]), "diff": diff,
            "lo": diff - 1.959964 * se, "hi": diff + 1.959964 * se, "p": float(2 * norm.sf(abs(diff) / se))}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    if q.size == 0:
        return out
    order = np.argsort(q)
    adj = np.minimum(1.0, (q.size - np.arange(q.size)) * q[order])
    adj = np.maximum.accumulate(adj)
    res = np.empty_like(q)
    res[order] = adj
    out[ok] = res
    return out


# ----------------------------------------------------------------------------------------------
# bootstrap over patients
# ----------------------------------------------------------------------------------------------

class PatientBootstrap:
    """Draws of row indices in which whole patients are resampled with replacement."""

    def __init__(self, groups, n: int = 2000, seed: int = 0):
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        uniq, start = np.unique(groups[order], return_index=True)
        self.members = np.split(order, start[1:])
        self.n_groups = len(uniq)
        self.n = int(n)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            pick = rng.integers(0, self.n_groups, self.n_groups)
            yield np.concatenate([self.members[i] for i in pick])


def _ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _sens(y, d, idx) -> float:
    yy = y[idx]
    return float(d[idx][yy].mean()) if yy.any() else float("nan")


def _spec(y, d, idx) -> float:
    yy = ~y[idx]
    return float((~d[idx][yy]).mean()) if yy.any() else float("nan")


# ----------------------------------------------------------------------------------------------
# one detector run
# ----------------------------------------------------------------------------------------------

def scores_for(fit_tab: pd.DataFrame, eval_tab: pd.DataFrame, same_split: bool, folds: int = 5,
               seed: int = 0) -> tuple[dict, dict, dict]:
    """Scores of every setting on the fitting split (valid) and the evaluation split, and thresholds.

    Stacking is fitted on valid. If the evaluation split is valid itself (development), the stacking
    scores are out-of-fold (patient-grouped cross-fitting) so they are not optimistic.
    """
    pa_v, lat_v, y_v = fit_tab["pa"].values, fit_tab["lat"].values, fit_tab["y"].values
    pa_e, lat_e = eval_tab["pa"].values, eval_tab["lat"].values
    s_fit, s_eval, thr = {}, {}, {}
    for s in SETTINGS:
        if s == "stack":
            if y_v.min() == y_v.max():
                s_fit[s] = s_eval[s] = None
                continue
            if same_split:
                oof = pm.crossfit(stack_features(pa_v, lat_v), y_v, fit_tab["patient_id"].values, folds, seed)
                s_fit[s] = s_eval[s] = oof
            else:
                model = fit_stack(pa_v, lat_v, y_v)
                s_fit[s] = fuse(pa_v, lat_v, s, model)
                s_eval[s] = fuse(pa_e, lat_e, s, model)
        else:
            s_fit[s] = fuse(pa_v, lat_v, s)
            s_eval[s] = fuse(pa_e, lat_e, s)
        thr[s] = threshold_at(s_fit[s][y_v == 0])
    return s_fit, s_eval, thr


def evaluate_finding(fit_tab: pd.DataFrame, eval_tab: pd.DataFrame, same_split: bool, n_boot: int = 2000,
                     seed: int = 0) -> tuple[list, list, dict]:
    """Metrics per setting and contrasts for one run and one finding.

    Returns (setting rows, contrast rows, decisions {setting: bool array on eval_tab})."""
    _, s_eval, thr = scores_for(fit_tab, eval_tab, same_split, seed=seed)
    y = eval_tab["y"].values.astype(bool)
    settings = [s for s in SETTINGS if s_eval.get(s) is not None]
    dec = {s: s_eval[s] > thr[s] for s in settings}

    rows = []
    boot_sens = {s: [] for s in settings}
    boot_spec = {s: [] for s in settings}
    for idx in PatientBootstrap(eval_tab["patient_id"].values, n_boot, seed):
        for s in settings:
            boot_sens[s].append(_sens(y, dec[s], idx))
            boot_spec[s].append(_spec(y, dec[s], idx))
    for s in settings:
        r = rates(y, dec[s])
        r.update(setting=s, threshold=thr[s], auc=pm.auc(y, s_eval[s]))
        r["sens_lo"], r["sens_hi"] = _ci(boot_sens[s])
        r["spec_lo"], r["spec_hi"] = _ci(boot_spec[s])
        rows.append(r)

    contrasts = []
    for new, ref in CONTRASTS:
        if new not in dec or ref not in dec:
            continue
        lo, hi = _ci(np.asarray(boot_sens[new]) - np.asarray(boot_sens[ref]))
        slo, shi = _ci(np.asarray(boot_spec[new]) - np.asarray(boot_spec[ref]))
        mc = mcnemar(dec[new][y], dec[ref][y])  # positives: detected
        ms = mcnemar(~dec[new][~y], ~dec[ref][~y])  # negatives: correctly not flagged
        dl = delong(y, s_eval[new], s_eval[ref])
        rn, rr = rates(y, dec[new]), rates(y, dec[ref])
        contrasts.append({
            "contrast": f"{new}_vs_{ref}", "new": new, "ref": ref,
            "sens_new": rn["sens"], "sens_ref": rr["sens"], "sens_diff": rn["sens"] - rr["sens"],
            "sens_diff_lo": lo, "sens_diff_hi": hi, "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
            "mcnemar_p": mc["p"], "mcnemar_method": mc["method"],
            "spec_new": rn["spec"], "spec_ref": rr["spec"], "spec_diff": rn["spec"] - rr["spec"],
            "spec_diff_lo": slo, "spec_diff_hi": shi, "spec_mcnemar_b": ms["b"], "spec_mcnemar_c": ms["c"],
            "spec_mcnemar_p": ms["p"], "spec_mcnemar_method": ms["method"],
            "auc_new": dl["auc_new"], "auc_ref": dl["auc_ref"],
            "auc_diff": dl["diff"], "auc_diff_lo": dl["lo"], "auc_diff_hi": dl["hi"], "delong_p": dl["p"]})
    return rows, contrasts, dec


def evaluate_run(fit_tab: pd.DataFrame, eval_tab: pd.DataFrame, run: str, same_split: bool,
                 n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All findings of one run. Adds Holm-adjusted p-values within families:
    family = (contrast, test) over the findings, leaving out the primary test (fracture, max vs pa)."""
    set_rows, con_rows, decisions = [], [], {}
    for f in fit_tab["finding"].unique():
        a = fit_tab[fit_tab["finding"] == f].reset_index(drop=True)
        b = eval_tab[eval_tab["finding"] == f].reset_index(drop=True)
        rows, cons, dec = evaluate_finding(a, b, same_split, n_boot, seed)
        decisions[f] = dec
        set_rows += [{"run": run, "finding": f, **r} for r in rows]
        con_rows += [{"run": run, "finding": f, **c} for c in cons]
    settings = pd.DataFrame(set_rows)
    con = pd.DataFrame(con_rows)
    if len(con):
        con["primary"] = (con["finding"] == PRIMARY_FINDING) & (con["contrast"] == f"{PRIMARY_RULE}_vs_pa")
        for test in ("mcnemar_p", "spec_mcnemar_p", "delong_p"):
            adj = np.full(len(con), np.nan)
            for _, idx in con[~con["primary"]].groupby("contrast").groups.items():
                adj[np.asarray(list(idx))] = holm(con.loc[idx, test].values)
            prim = con["primary"].values
            adj[prim] = con.loc[prim, test].values  # the primary test is not adjusted
            con[test.replace("_p", "_p_holm")] = adj
    return settings, con, decisions


# ----------------------------------------------------------------------------------------------
# subgroups and seeds
# ----------------------------------------------------------------------------------------------

def subgroup_masks(tab: pd.DataFrame) -> dict:
    """Protocol 4.4: torus, age bands, initial exam, cast (exploratory)."""
    masks = {("all", "all"): np.ones(len(tab), dtype=bool)}
    if "torus" in tab:
        masks[("torus", "yes")] = tab["torus"].values == 1
        masks[("torus", "no")] = tab["torus"].values == 0
    if "age_band" in tab:
        for _, _, name in AGE_BANDS:
            masks[("age", name)] = tab["age_band"].values == name
    for col in ("initial_exam", "cast"):
        if col in tab:
            masks[(col, "yes")] = tab[col].values == 1
            masks[(col, "no")] = tab[col].values == 0
    return masks


def subgroups(eval_tab: pd.DataFrame, decisions: dict, run: str, n_boot: int = 2000, seed: int = 0,
              compare=(("max", "pa"), ("max", "lat"))) -> pd.DataFrame:
    """Sensitivity (and specificity where the subgroup has negatives) per subgroup, with the
    patient-bootstrap CI of the fused-minus-single difference. Thresholds stay those of the whole valid set."""
    rows = []
    for f, dec in decisions.items():
        tab = eval_tab[eval_tab["finding"] == f].reset_index(drop=True)
        y = tab["y"].values.astype(bool)
        for (name, level), mask in subgroup_masks(tab).items():
            sub = np.flatnonzero(mask)
            if sub.size == 0:
                continue
            boots = {(n, r): [] for n, r in compare}
            for idx in PatientBootstrap(tab["patient_id"].values[sub], n_boot, seed):
                ii = sub[idx]
                for n, r in compare:
                    boots[(n, r)].append(_sens(y, dec[n], ii) - _sens(y, dec[r], ii))
            row = {"run": run, "finding": f, "subgroup": name, "level": level,
                   "n_pos": int(y[sub].sum()), "n_neg": int((~y[sub]).sum())}
            for s in ("pa", "lat", "max"):
                rr = rates(y[sub], dec[s][sub])
                row[f"sens_{s}"], row[f"spec_{s}"] = rr["sens"], rr["spec"]
            for n, r in compare:
                row[f"diff_{n}_vs_{r}"] = row[f"sens_{n}"] - row[f"sens_{r}"]
                row[f"diff_{n}_vs_{r}_lo"], row[f"diff_{n}_vs_{r}_hi"] = _ci(boots[(n, r)])
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_seed_bootstrap(eval_tabs: dict, decisions: dict, finding: str, new: str, ref: str,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean over seeds of the sensitivity difference new - ref, with a patient bootstrap that applies the
    same resampled patients to every seed (protocol 7: pooled bootstrap). Seeds must share the same pairs."""
    runs = sorted(eval_tabs)
    tabs = {r: eval_tabs[r][eval_tabs[r]["finding"] == finding].reset_index(drop=True) for r in runs}
    ref_ids = tabs[runs[0]]["pair_id"].values
    for r in runs[1:]:
        if not np.array_equal(tabs[r]["pair_id"].values, ref_ids):
            raise ValueError("seeds are not evaluated on the same pairs in the same order")
    y = tabs[runs[0]]["y"].values.astype(bool)
    point = [_sens(y, decisions[r][finding][new], np.arange(len(y))) - _sens(y, decisions[r][finding][ref], np.arange(len(y)))
             for r in runs]
    draws = []
    for idx in PatientBootstrap(tabs[runs[0]]["patient_id"].values, n_boot, seed):
        draws.append(np.mean([_sens(y, decisions[r][finding][new], idx) - _sens(y, decisions[r][finding][ref], idx)
                              for r in runs]))
    lo, hi = _ci(draws)
    return {"finding": finding, "contrast": f"{new}_vs_{ref}", "seeds": len(runs), "runs": " ".join(runs),
            "diff_mean": float(np.mean(point)), "diff_sd": float(np.std(point, ddof=1)) if len(runs) > 1 else float("nan"),
            "pooled_lo": lo, "pooled_hi": hi}
