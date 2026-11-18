"""Gate P decision (docs/PROTOCOL.md section 5).

For each class: two logistic models predict "is this detection a true positive at IoU 0.5?"
  base = [logit conf, LAT indicator, other indicator, logit own-image max of the class]
  new  = base + [logit p (presence), LAT indicator x logit p]
fitted with patient-grouped cross-fitting (out-of-fold). Each model's probabilities rescore the
detections; the rescored AP50 of new vs base is the quantity of interest, with a 95% interval from
resampling whole patients. PASS if >= 2 of the 3 target classes have an interval above 0 and the
mean rescored AP50 over all classes with labels is not lower for new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vgra import premise as pm

TARGETS = ("pronatorsign", "softtissue", "periostealreaction")
MIN_TP = 10


def design(conf, view, own_max, presence=None) -> np.ndarray:
    view = np.asarray(view)
    lat = (view == 1).astype(float)
    other = (view == 2).astype(float)
    cols = [pm.logit(conf), lat, other, pm.logit(own_max)]
    if presence is not None:
        lp = pm.logit(presence)
        cols += [lp, lat * lp]
    return np.stack(cols, 1)


def ap50(tp50, scores, n_targets) -> float:
    return pm.ap_of(np.asarray(tp50, bool)[:, None], np.asarray(scores, float), int(n_targets))[0]


def bootstrap_ap_delta(tp50, s_base, s_new, groups, boxes_per_group: dict, n: int = 500,
                       seed: int = 0) -> tuple[float, float]:
    """95% interval of AP50(s_new) - AP50(s_base) when whole patients are resampled.
    boxes_per_group: patient -> number of labelled boxes of the class (patients without
    detections count too, through their labelled boxes)."""
    tp50, s_base, s_new, groups = map(np.asarray, (tp50, s_base, s_new, groups))
    patients = np.array(sorted(boxes_per_group))
    boxes = np.array([boxes_per_group[p] for p in patients])
    rows_of = {p: np.flatnonzero(groups == p) for p in patients}
    empty = np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(n)):
        pick = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([rows_of.get(patients[i], empty) for i in pick])
        n_t = int(boxes[pick].sum())
        if n_t == 0 or idx.size == 0:
            continue
        deltas.append(ap50(tp50[idx], s_new[idx], n_t) - ap50(tp50[idx], s_base[idx], n_t))
    if len(deltas) < max(20, n // 5):
        return float("nan"), float("nan")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(dets: pd.DataFrame, imgs: pd.DataFrame, classes: list, folds: int = 5, n_boot: int = 500,
             targets=TARGETS, seed: int = 0) -> pd.DataFrame:
    """One row per class: AP50 of detector / base / new, gain, interval (targets only), image AUCs."""
    rows = []
    for c, name in enumerate(classes):
        d = dets[dets["cls"] == c].reset_index(drop=True)
        im = imgs[imgs["cls"] == c]
        n_t = int(im["n_boxes"].sum())
        row = {"cls": name, "target": name in targets, "n_targets": n_t, "n_det": len(d),
               "n_tp": int(d["tp50"].sum())}
        if n_t == 0:
            rows.append({**row, "note": "no labels"})
            continue
        y, g = d["tp50"].to_numpy().astype(int), d["patient_id"].to_numpy()
        row["ap50_conf"] = ap50(y, d["conf"], n_t)
        if row["n_tp"] < MIN_TP or len(d) - row["n_tp"] < MIN_TP:
            row.update(ap50_base=row["ap50_conf"], ap50_new=row["ap50_conf"], gain=0.0,
                       note=f"fewer than {MIN_TP} true or false positives: detector confidence kept")
        else:
            p_base = pm.crossfit(design(d["conf"], d["view"], d["own_max"]), y, g, folds, seed)
            p_new = pm.crossfit(design(d["conf"], d["view"], d["own_max"], d["extra"]), y, g, folds, seed)
            row.update(ap50_base=ap50(y, p_base, n_t), ap50_new=ap50(y, p_new, n_t),
                       auc_det_base=pm.auc(y, p_base), auc_det_new=pm.auc(y, p_new), note="")
            row["gain"] = row["ap50_new"] - row["ap50_base"]
            if name in targets:
                per_patient = im.groupby("patient_id")["n_boxes"].sum().to_dict()
                for p in np.unique(g):
                    per_patient.setdefault(p, 0)
                row["gain_lo"], row["gain_hi"] = bootstrap_ap_delta(y, p_base, p_new, g, per_patient, n_boot, seed)
        # image level (reported only): does presence add to the own-image max?
        yi, gi = im["present"].to_numpy().astype(int), im["patient_id"].to_numpy()
        if 10 <= yi.sum() <= len(yi) - 10:
            vi = im["view"].to_numpy()
            pi_b = pm.crossfit(design(im["own_max"], vi, im["own_max"])[:, :3], yi, gi, folds, seed)
            pi_n = pm.crossfit(np.c_[design(im["own_max"], vi, im["own_max"])[:, :3], pm.logit(im["extra"])], yi, gi,
                               folds, seed)
            row.update(auc_img_own=pm.auc(yi, im["own_max"]), auc_img_presence=pm.auc(yi, im["extra"]),
                       auc_img_base=pm.auc(yi, pi_b), auc_img_new=pm.auc(yi, pi_n))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, targets=TARGETS, min_classes: int = 2) -> tuple[bool, list, float, float]:
    """(PASS?, target classes with a clear gain, mean AP50 base, mean AP50 new)."""
    good = [r.cls for r in table.itertuples() if r.cls in targets and getattr(r, "gain_lo", np.nan) > 0]
    labelled = table[table["n_targets"] > 0]
    mean_base, mean_new = float(labelled["ap50_base"].mean()), float(labelled["ap50_new"].mean())
    return (len(good) >= min_classes and mean_new >= mean_base), good, mean_base, mean_new
