"""I6 tests: CoVeR (vgra/cover.py) and scripts/13_cover.py."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgra import cover as cv
from vgra import study as st

REPO = Path(__file__).resolve().parents[1]
RUN = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace", "cwd": REPO,
       "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}}
CLASSES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
           "periostealreaction", "pronatorsign", "softtissue", "text"]


def population(n_pairs=600, seed=0, n_patients=None):
    """Fracture rows: visible on PA, LAT or both; scores follow visibility; several pairs per patient."""
    rng = np.random.default_rng(seed)
    n_patients = n_patients or n_pairs // 2
    pid = rng.integers(1, n_patients + 1, n_pairs)
    y = rng.random(n_pairs) < 0.65
    state = rng.choice(["10", "01", "11"], n_pairs, p=[0.1, 0.12, 0.78])
    vis_pa, vis_lat = y & (state != "01"), y & (state != "10")
    sig = lambda v: 1 / (1 + np.exp(-v))
    pa = sig(rng.normal(np.where(vis_pa, 2.5, -2.5), 1.3))
    lat = sig(rng.normal(np.where(vis_lat, 2.5, -2.5), 1.3))
    age = rng.uniform(1, 18, n_pairs)
    return pd.DataFrame({"pair_id": [f"{i:05d}" for i in range(n_pairs)], "patient_id": pid, "finding": "fracture",
                         "pa": pa, "lat": lat, "y": y.astype(int), "age": age,
                         "age_band": [st.age_band(a) for a in age], "torus": rng.integers(0, 2, n_pairs),
                         "initial_exam": rng.integers(0, 2, n_pairs), "cast": rng.integers(0, 2, n_pairs)})


# ---------- policy ---------------------------------------------------------------------------

def test_gate_positive_implies_both_and_pa_positive_is_free():
    tab = population()
    pa, lat, t = tab["pa"].values, tab["lat"].values, 0.7
    both = cv.both_views(pa, lat, t)
    for lo in (cv.REFER_ALL, 0.0, 0.05, 0.3, t):
        dec, refer = cv.decide(pa, lat, t, lo)
        assert not (dec & ~both).any()                  # the gate never flags what both views do not
        assert dec[pa > t].all() and not refer[pa > t].any()


def test_refer_all_is_always_both_and_lo_t_is_pa_only():
    tab = population()
    pa, lat, t = tab["pa"].values, tab["lat"].values, 0.6
    dec, refer = cv.decide(pa, lat, t, cv.REFER_ALL)
    assert np.array_equal(dec, cv.both_views(pa, lat, t)) and np.array_equal(refer, pa <= t)
    dec, refer = cv.decide(pa, lat, t, t)
    assert np.array_equal(dec, pa > t) and not refer.any()


# ---------- risk and calibration -------------------------------------------------------------

def test_risk_curve_weights_patients_equally():
    # patient 1: two positive pairs, one rescued (PA 0.1); patient 2: one positive pair, rescued (PA 0.2)
    pa = np.array([0.1, 0.9, 0.2, 0.05])
    lat = np.array([0.9, 0.9, 0.9, 0.0])
    y = np.array([1, 1, 1, 0])
    pid = np.array([1, 1, 2, 3])
    risk, n = cv.risk_curve(pa, lat, y, 0.5, cv.units_of(pid), np.array([cv.REFER_ALL, 0.0, 0.15, 0.25, 0.5]))
    assert n == 2
    np.testing.assert_allclose(risk, [0.0, 0.0, 0.25, 0.75, 0.75])
    extra = cv.rescued(pa, lat, y, 0.5) & (pa <= 0.25)
    assert math.isclose(cv.patient_risk(extra, y, cv.units_of(pid)), 0.75)


def test_hb_pvalue_values():
    # risk 0: Hoeffding term exp(n * log(1 - alpha)); Bentkus term e * (1 - alpha)^n
    n, a = 96, 0.02
    p = cv.hb_pvalue(np.array([0.0, 0.01, a, 0.5]), n, a)
    assert math.isclose(p[0], min(math.exp(n * math.log(1 - a)), math.e * (1 - a) ** n), rel_tol=1e-9)
    assert p[0] < p[1] <= p[2] == 1.0 and p[3] == 1.0


def test_crc_needs_enough_patients():
    assert cv.passes(np.array([0.0]), 99, 0.01, "crc")[0]
    assert not cv.passes(np.array([0.0]), 98, 0.01, "crc")[0]


def test_calibrate_is_fixed_sequence_and_falls_back_to_refer_all():
    tab = population(seed=1)
    pa, lat, y, pid = tab["pa"].values, tab["lat"].values, tab["y"].values, tab["patient_id"].values
    t = st.threshold_at(np.maximum(pa, lat)[y == 0])
    for method in cv.METHODS:
        lo = cv.calibrate(pa, lat, y, t, pid, 0.02, method)["all"]
        assert lo == cv.REFER_ALL or lo <= t
        risk, n = cv.risk_curve(pa, lat, y, t, cv.units_of(pid), np.array([lo]))
        assert lo == cv.REFER_ALL or cv.passes(risk, n, 0.02, method)[0]
        nxt = cv.GRID[cv.GRID > lo]
        nxt = nxt[nxt <= t]
        if nxt.size:  # the next grid point fails (or the search would have gone on)
            r2, _ = cv.risk_curve(pa, lat, y, t, cv.units_of(pid), nxt[:1])
            assert not cv.passes(r2, n, 0.02, method)[0]
    # stricter methods read more lateral views: naive <= crc <= ltt in lateral reads (lo non-increasing)
    los = [cv.calibrate(pa, lat, y, t, pid, 0.02, m)["all"] for m in ("naive", "crc", "ltt")]
    assert los[0] >= los[1] >= los[2]
    # tiny calibration set: CRC cannot certify anything -> read every lateral view below t
    small = tab.iloc[:20]
    lo = cv.calibrate(small["pa"], small["lat"], small["y"], t, small["patient_id"], 0.01, "crc")["all"]
    assert lo == cv.REFER_ALL


def test_mondrian_lo_per_group_and_unseen_group_reads_all():
    tab = population(n_pairs=1200, seed=2)
    pa, lat, y, pid, g = (tab[c].values for c in ("pa", "lat", "y", "patient_id", "age_band"))
    t = st.threshold_at(np.maximum(pa, lat)[y == 0])
    lo = cv.calibrate(pa, lat, y, t, pid, 0.05, "crc", groups=g)
    assert set(lo) == set(g)
    rows = cv.lo_per_row(lo, np.array(["<6", "unknown"]))
    assert rows[0] == lo["<6"] and rows[1] == cv.REFER_ALL
    with pytest.raises(ValueError):
        cv.lo_per_row(lo, None, 3)


def test_ltt_and_crc_hold_in_repeated_splits():
    """Validity check on a synthetic population: LTT holds in >= ~90% of splits, CRC holds on average."""
    pop = population(n_pairs=4000, seed=3, n_patients=2400)
    d = cv.design_study(pop, alphas=(0.02,), methods=("crc", "ltt"), groupings=("all",), n_splits=60, seed=0)
    ltt = d[d["method"] == "ltt"]
    crc = d[d["method"] == "crc"]
    assert ltt["held"].mean() >= 0.85
    assert crc["extra_patient"].mean() <= 0.02 + 0.004
    assert (ltt["refer_rate"].values >= crc["refer_rate"].values - 1e-12).all()


# ---------- baselines and study tables -------------------------------------------------------

def test_random_same_budget_matches_monte_carlo():
    tab = population(seed=4)
    pa, lat, y = tab["pa"].values, tab["lat"].values, tab["y"].values.astype(bool)
    t = 0.6
    closed = cv.same_budget_random(pa, lat, y, t, 40)
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(pa <= t)
    sims = []
    for _ in range(3000):
        refer = np.zeros(pa.size, dtype=bool)
        refer[rng.choice(cand, 40, replace=False)] = True
        dec = (pa > t) | (refer & cv.both_views(pa, lat, t))
        sims.append(dec[y].mean())
    assert abs(np.mean(sims) - closed["sens"]) < 0.002


def test_ranked_by_pa_reproduces_the_gate():
    tab = population(seed=5)
    pa, lat, y = tab["pa"].values, tab["lat"].values, tab["y"].values.astype(bool)
    t, lo = 0.6, 0.2
    dec, refer = cv.decide(pa, lat, t, lo)
    rk = cv.same_budget_ranked(pa, pa, lat, y, t, int(refer.sum()))
    assert math.isclose(rk["sens"], dec[y].mean())


def test_ensemble_and_disagreement_need_same_pairs():
    a, b = population(seed=6), population(seed=7)
    ens = cv.ensemble_table({"a": a, "b": b})
    np.testing.assert_allclose(ens["pa"], (a["pa"] + b["pa"]) / 2)
    assert cv.disagreement({"a": a, "b": b}).shape == (len(a),)
    with pytest.raises(ValueError):
        cv.ensemble_table({"a": a, "b": b.iloc[::-1]})


def test_design_summary_and_curve_and_who():
    tab = population(n_pairs=800, seed=8)
    d = cv.design_study(tab, n_splits=4, seed=1, rank_scores={"uncert": np.random.default_rng(0).random(len(tab))})
    assert len(d) == 4 * len(cv.GROUPINGS) * len(cv.METHODS) * len(cv.ALPHAS)
    assert {"held", "held_all_ages", "random_sens", "uncert_sens", "extra_worst_age", "sens_pa"} <= set(d.columns)
    s = cv.design_summary(d)
    assert len(s) == len(cv.GROUPINGS) * len(cv.METHODS) * len(cv.ALPHAS) and (s["splits"] == 4).all()
    c = cv.risk_referral_curve(tab)
    assert c["refer_rate"].is_monotonic_decreasing and c["extra_patient"].is_monotonic_increasing
    w = cv.who_is_referred(tab, {"all": 0.1}, "all")
    assert {"age_band", "torus", "initial_exam", "cast"} <= set(w["subgroup"])


def test_parta_check_matches_the_study_module():
    v, e = population(n_pairs=700, seed=13), population(n_pairs=400, seed=14)
    settings, _, _ = st.evaluate_run(v, e, "R_s0", same_split=False, n_boot=5)
    ok = cv.parta_check(v, e, settings, "R_s0")
    assert ok["ok"] and math.isclose(ok["sens"], ok["parta_sens"])
    bad = settings.copy()
    bad.loc[(bad["setting"] == "max"), "sens"] += 0.001
    assert not cv.parta_check(v, e, bad, "R_s0")["ok"]
    assert not cv.parta_check(v, e, settings, "other_run")["ok"]
    # same split (valid development table)
    settings_v, _, _ = st.evaluate_run(v, v, "R_s0", same_split=True, n_boot=5)
    assert cv.parta_check(v, v, settings_v, "R_s0")["ok"]


def test_confirm_uses_valid_threshold_and_gives_cis():
    v, e = population(n_pairs=900, seed=9), population(n_pairs=500, seed=10)
    r = cv.confirm(v, e, 0.02, "ltt", "all", n_boot=60, seed=0)
    t = st.threshold_at(np.maximum(v["pa"], v["lat"])[v["y"] == 0])
    assert math.isclose(r["t"], t) and r["mcnemar_pos_b"] == 0
    for k in ("refer_rate", "sens", "spec", "extra_patient", "sens_diff_both", "sens_diff_random"):
        assert r[f"{k}_lo"] <= r[f"{k}_hi"]
    assert r["sens_diff_both_hi"] <= 1e-12          # the gate can only lose sensitivity vs both views
    assert json.loads(r["lo_map"]).keys() == {"all"}
    r = cv.confirm(v, e, 0.05, "crc", "age", n_boot=20, seed=0)
    assert set(json.loads(r["lo_map"])) == set(v["age_band"])


# ---------- script ---------------------------------------------------------------------------

def write_predictions(pred: Path, runs=("B0_s0", "B0_s1")):
    pred.mkdir(parents=True, exist_ok=True)
    for split, seed in (("valid", 11), ("test", 12)):
        tab = population(n_pairs=500, seed=seed, n_patients=300)
        pairs = tab[["pair_id", "patient_id"]].copy()
        pairs["study_number"], pairs["laterality"], pairs["split"] = 1, "L", split
        pairs["ap_image_id"] = [f"a{i}" for i in range(len(tab))]
        pairs["lat_image_id"] = [f"l{i}" for i in range(len(tab))]
        for col in ("initial_exam", "cast", "torus", "age", "age_band"):
            pairs[col] = tab[col]
        pairs.to_csv(pred / f"pairs_{split}.csv", index=False)
        for k, run in enumerate(runs):
            rows = []
            noise = np.random.default_rng(100 + k).normal(0, 0.02, len(tab))
            for i in range(len(tab)):
                for c in range(len(CLASSES)):
                    for side, s in (("a", tab["pa"][i]), ("l", tab["lat"][i])):
                        own = float(np.clip(s + noise[i], 0, 1)) if CLASSES[c] == "fracture" else 0.0
                        present = int(tab["y"][i]) if CLASSES[c] == "fracture" else 0
                        rows.append({"image_id": f"{side}{i}", "patient_id": tab["patient_id"][i],
                                     "view": 0 if side == "a" else 1, "cls": c, "own_max": own,
                                     "present": present, "n_boxes": present})
            (pred / run).mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(pred / run / f"{split}_images.csv.gz", index=False)


def test_cover_script_valid_and_test(tmp_path):
    pred = tmp_path / "predictions"
    write_predictions(pred)
    script = str(REPO / "scripts" / "13_cover.py")
    r = subprocess.run([sys.executable, script, "--pred", str(pred), "--runs", "B0_s0", "B0_s1", "--eval", "valid",
                        "--ensemble", "--splits", "3"], **RUN)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "DESIGN STUDY" in r.stdout and "COVER DONE" in r.stdout
    out = pred / "cover_valid"
    for name in ("design_splits.csv.gz", "design_summary.csv", "curves.csv", "who.csv", "summary.txt", "info.json"):
        assert (out / name).is_file(), name
    assert set(pd.read_csv(out / "design_summary.csv")["run"]) == {"B0_s0", "B0_s1", "ENS"}

    r = subprocess.run([sys.executable, script, "--pred", str(pred), "--runs", "B0_s0", "--eval", "test"], **RUN)
    assert r.returncode != 0 and "opened once" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, script, "--pred", str(pred), "--runs", "B0_s0", "--eval", "test",
                        "--open-test"], **RUN)
    assert r.returncode != 0 and "--check-parta" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, script, "--pred", str(pred), "--runs", "B0_s0", "--ens-runs", "B9_s0",
                        "--eval", "valid"], **RUN)
    assert r.returncode != 0 and "subset" in (r.stdout + r.stderr)

    # Part A settings of the synthetic test split, as scripts/12_study_level.py would write them
    tabs = {}
    for split in ("valid", "test"):
        pairs = pd.read_csv(pred / f"pairs_{split}.csv", dtype={"pair_id": str, "laterality": str})
        for run in ("B0_s0", "B0_s1"):
            imgs = pd.read_csv(pred / run / f"{split}_images.csv.gz")
            tabs[(run, split)] = st.study_table(imgs, pairs, CLASSES, findings=("fracture",))
    settings = pd.concat([st.evaluate_run(tabs[(r, "valid")], tabs[(r, "test")], r, False, 5)[0]
                          for r in ("B0_s0", "B0_s1")], ignore_index=True)
    settings.to_csv(tmp_path / "settings.csv", index=False)
    bad = settings.copy()
    bad.loc[bad["setting"] == "max", "spec"] -= 0.01
    bad.to_csv(tmp_path / "bad.csv", index=False)
    common = ["--pred", str(pred), "--runs", "B0_s0", "B0_s1", "--ens-runs", "B0_s0", "B0_s1", "--ensemble",
              "--eval", "test", "--open-test", "--boot", "30"]
    r = subprocess.run([sys.executable, script, *common, "--check-parta", str(tmp_path / "bad.csv")], **RUN)
    assert r.returncode != 0 and "CONSISTENCY CHECK FAILED" in (r.stdout + r.stderr)
    assert not (pred / "cover_test" / "results.csv").exists()
    r = subprocess.run([sys.executable, script, *common, "--check-parta", str(tmp_path / "settings.csv")], **RUN)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "PART A CONSISTENCY CHECK" in r.stdout and "PASS" in r.stdout and "PRIMARY limit held in" in r.stdout
    res = pd.read_csv(pred / "cover_test" / "results.csv")
    assert set(res["run"]) == {"B0_s0", "B0_s1", "ENS"} and res["primary"].sum() == 3
    rec = json.loads((pred / "cover_test" / "COVER_TEST_RUN.json").read_text())
    assert rec["ens_runs"] == ["B0_s0", "B0_s1"] and rec["check_parta"].endswith("settings.csv")
