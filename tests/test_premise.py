"""STEP-04 tests: AUC, logistic cross-fitting, patient bootstrap, matching and AP."""

import numpy as np
import pytest

from vgra import premise as pm

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert pm.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert pm.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert pm.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(pm.auc(np.ones(4), np.arange(4)))


def test_logistic_fit_recovers_direction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2))
    y = (x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=2000) > 0).astype(int)
    m = pm.fit_logistic(x, y)
    assert m["w"][1] > 0 and m["w"][2] < 0
    assert pm.auc(y, pm.predict_logistic(m, x)) > 0.95


def test_group_folds_keep_patients_together():
    groups = np.repeat(np.arange(50), 4)
    folds = pm.group_folds(groups, k=5)
    for g in range(50):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == set(range(5))


def synthetic(informative: bool, n_pat: int = 400, seed: int = 0):
    """Each patient has two views; the class is in both or neither. The own score is noisy;
    the partner score is either a second noisy look at the truth or pure noise."""
    rng = np.random.default_rng(seed)
    truth = rng.random(n_pat) < 0.3
    groups = np.repeat(np.arange(n_pat), 2)
    y = np.repeat(truth, 2).astype(int)
    view = np.tile([0, 1], n_pat)
    own = 1 / (1 + np.exp(-(1.2 * (2 * y - 1) + rng.normal(size=2 * n_pat))))
    partner_logit = 1.2 * (2 * y - 1) + rng.normal(size=2 * n_pat) if informative else rng.normal(size=2 * n_pat)
    partner = 1 / (1 + np.exp(-partner_logit))
    return y, own, view, partner, groups


def test_compare_finds_informative_partner():
    r = pm.compare(*synthetic(True), n_boot=200)
    assert r["gain"] > 0.01 and r["gain_lo"] > 0  # own AUC is already 0.96; partner lifts it to 0.99


def test_compare_finds_no_gain_from_noise():
    r = pm.compare(*synthetic(False), n_boot=200)
    assert abs(r["gain"]) < 0.02 and r["gain_lo"] < 0  # noise never passes the gate (needs lower bound > 0)


def test_compare_too_few_positives():
    y, own, view, partner, groups = synthetic(True)
    y = np.zeros_like(y)
    y[:3] = 1
    r = pm.compare(y, own, view, partner, groups)
    assert r["note"] == "too few examples" and np.isnan(r["gain"])


def test_match_uses_ultralytics_rules():
    gt = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=float)
    gt_cls = np.array([3, 3])
    pred = np.array([[0, 0, 100, 100], [0, 0, 100, 100], [200, 200, 260, 300], [500, 500, 600, 600]], dtype=float)
    pred_cls = np.array([3, 3, 3, 3])
    tp = pm.match(pred, pred_cls, gt, gt_cls)
    assert tp.shape == (4, 10)
    assert tp[0].all() and not tp[1].any()  # one label matches one prediction only
    assert tp[2, 0] and not tp[2, -1]  # IoU 0.6: true at 0.5, false at 0.95
    assert not tp[3].any()
    assert pm.match(pred, np.array([5, 5, 5, 5]), gt, gt_cls).sum() == 0  # wrong class


def test_ap_of_perfect_and_empty():
    tp = np.ones((3, 10), dtype=bool)
    ap50, ap = pm.ap_of(tp, np.array([0.9, 0.8, 0.7]), 3)
    assert ap50 > 0.99 and ap > 0.99
    assert pm.ap_of(np.zeros((0, 10), dtype=bool), np.array([]), 2) == (0.0, 0.0)
    assert all(np.isnan(v) for v in pm.ap_of(tp, np.array([0.9, 0.8, 0.7]), 0))
