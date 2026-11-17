"""Premise test (STEP-04, Gate 2): does the partner view carry class evidence that the detector's
own score does not?

Two levels, both on validation pairs and for every class:

detection level  every detection of class c after NMS (conf >= 0.001, as in Ultralytics validation)
                 target: true positive at IoU 0.5
                 base model:    logit(own conf) + view
                 partner model: base + logit(partner image's max score of class c) + view x partner
image level      every image and class c
                 target: class c annotated in the image
                 base model:    logit(own max score of class c) + view
                 partner model: base + logit(partner's max score of class c) + view x partner

Models are small logistic regressions, fitted with patient-grouped cross-fitting (out-of-fold
predictions), so no image is scored by a model that saw its own patient. The gain is
AUC(partner) - AUC(base), with a 95% patient-bootstrap interval. The base model contains the view,
so the gain measures the partner, not the projection. Detection-level rescoring with the
out-of-fold probabilities also gives AP50 / AP50-95 per class (a first look at control B3).

Everything here is NumPy (no scikit-learn needed).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties count half). NaN if only one class is present."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _design(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), (x - mean) / std])


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> dict:
    """Logistic regression with an intercept (Newton / IRLS, small ridge), standardised features."""
    x = np.asarray(x, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y, dtype=np.float64)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-12] = 1.0
    a = _design(x, mean, std)
    w = np.zeros(a.shape[1])
    ridge = l2 * np.eye(a.shape[1])
    ridge[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
        g = a.T @ (p - y) + ridge @ w
        h = (a * (p * (1 - p))[:, None]).T @ a + ridge + 1e-9 * np.eye(a.shape[1])
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"w": w, "mean": mean, "std": std}


def predict_logistic(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    return 1 / (1 + np.exp(-np.clip(_design(x, model["mean"], model["std"]) @ model["w"], -30, 30)))


def group_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number per row; all rows of one group (patient) share a fold."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq[rng.permutation(len(uniq))], np.arange(len(uniq)) % k))
    return np.array([fold_of[g] for g in groups])


def crossfit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities from patient-grouped k-fold logistic regression."""
    y = np.asarray(y).astype(float)
    folds = group_folds(groups, k, seed)
    out = np.full(len(y), np.nan)
    for f in range(k):
        test = folds == f
        train = ~test
        if not test.any():
            continue
        if y[train].min() == y[train].max():  # one class only in the training folds
            out[test] = y[train].mean()
            continue
        out[test] = predict_logistic(fit_logistic(x[train], y[train]), x[test])
    return out


def bootstrap_delta(y: np.ndarray, s_base: np.ndarray, s_new: np.ndarray, groups: np.ndarray,
                    n: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% interval of AUC(s_new) - AUC(s_base) when whole patients are resampled."""
    y, s_base, s_new, groups = map(np.asarray, (y, s_base, s_new, groups))
    order = np.argsort(groups, kind="stable")
    uniq, start = np.unique(groups[order], return_index=True)
    members = np.split(order, start[1:])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(n)):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        d = auc(y[idx], s_new[idx]) - auc(y[idx], s_base[idx])
        if np.isfinite(d):
            deltas.append(d)
    if len(deltas) < max(20, n // 5):
        return float("nan"), float("nan")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def features(own: np.ndarray, view: np.ndarray, partner: np.ndarray | None = None) -> np.ndarray:
    """[logit(own), view] or [logit(own), view, logit(partner), view * logit(partner)]."""
    cols = [logit(own), np.asarray(view, dtype=np.float64)]
    if partner is not None:
        lp = logit(partner)
        cols += [lp, cols[1] * lp]
    return np.stack(cols, 1)


def compare(y, own, view, partner, groups, k: int = 5, n_boot: int = 500, seed: int = 0, min_pos: int = 10) -> dict:
    """Base vs partner model for one class at one level. Returns AUCs, the gain, its interval and the
    out-of-fold probabilities of both models."""
    y = np.asarray(y).astype(int)
    out = {"n": int(y.size), "n_pos": int(y.sum()), "auc_own": auc(y, own)}
    if out["n_pos"] < min_pos or y.size - out["n_pos"] < min_pos:
        out.update(auc_base=float("nan"), auc_partner=float("nan"), gain=float("nan"), gain_lo=float("nan"),
                   gain_hi=float("nan"), p_base=None, p_partner=None, note="too few examples")
        return out
    p_base = crossfit(features(own, view), y, groups, k, seed)
    p_part = crossfit(features(own, view, partner), y, groups, k, seed)
    lo, hi = bootstrap_delta(y, p_base, p_part, groups, n_boot, seed)
    out.update(auc_base=auc(y, p_base), auc_partner=auc(y, p_part), gain_lo=lo, gain_hi=hi, p_base=p_base,
               p_partner=p_part, note="")
    out["gain"] = out["auc_partner"] - out["auc_base"]
    return out


class _Matcher:
    """Ultralytics' own prediction-to-label matching (BaseValidator.match_predictions)."""

    def __init__(self):
        import torch

        self.iouv = torch.linspace(0.5, 0.95, 10)

    def __call__(self, pred_cls, true_cls, iou):
        from ultralytics.engine.validator import BaseValidator

        return BaseValidator.match_predictions(self, pred_cls, true_cls, iou)


def match(pred_xyxy, pred_cls, gt_xyxy, gt_cls) -> np.ndarray:
    """(D, 10) true-positive table at IoU 0.50:0.95, the same matching as Ultralytics validation."""
    import torch
    from ultralytics.utils.metrics import box_iou

    pred_xyxy, gt_xyxy = torch.as_tensor(pred_xyxy, dtype=torch.float32), torch.as_tensor(gt_xyxy, dtype=torch.float32)
    pred_cls, gt_cls = torch.as_tensor(pred_cls), torch.as_tensor(gt_cls)
    if len(pred_cls) == 0:
        return np.zeros((0, 10), dtype=bool)
    if len(gt_cls) == 0:
        return np.zeros((len(pred_cls), 10), dtype=bool)
    return _Matcher()(pred_cls, gt_cls, box_iou(gt_xyxy, pred_xyxy)).cpu().numpy()


def ap_of(tp: np.ndarray, conf: np.ndarray, n_targets: int) -> tuple[float, float]:
    """AP50 and AP50-95 of one class from its detections (tp table (D, 10), confidences) and the
    number of labelled boxes, with Ultralytics' ap_per_class."""
    from ultralytics.utils.metrics import ap_per_class

    if n_targets == 0:
        return float("nan"), float("nan")
    if len(conf) == 0:
        return 0.0, 0.0
    out = ap_per_class(tp, np.asarray(conf), np.zeros(len(conf)), np.zeros(n_targets))
    ap = out[5]
    return float(ap[0, 0]), float(ap[0].mean())
