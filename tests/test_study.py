"""I1/I2 tests: patient-level analysis (vgra/study.py), its script, and the test-split guard of 10_predict."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vgra import premise as pm
from vgra import study as st

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
           "periostealreaction", "pronatorsign", "softtissue", "text"]


# ---------- synthetic data -------------------------------------------------------------------

def synthetic(n_pairs=400, seed=0, lat_signal=1.5, n_patients=None):
    """Pairs whose fracture is visible on PA, on LAT or on both; scores follow visibility."""
    rng = np.random.default_rng(seed)
    n_patients = n_patients or n_pairs // 2
    pid = rng.integers(1, n_patients + 1, n_pairs)
    pairs = pd.DataFrame({"pair_id": [f"{i:05d}" for i in range(n_pairs)], "patient_id": pid,
                          "study_number": 1, "laterality": "L", "split": "valid",
                          "ap_image_id": [f"a{i}" for i in range(n_pairs)],
                          "lat_image_id": [f"l{i}" for i in range(n_pairs)],
                          "initial_exam": rng.integers(0, 2, n_pairs), "cast": rng.integers(0, 2, n_pairs),
                          "torus": rng.integers(0, 2, n_pairs), "age": rng.uniform(1, 18, n_pairs)})
    pairs["age_band"] = [st.age_band(a) for a in pairs["age"]]
    pos = rng.random(n_pairs) < 0.6
    state = rng.choice(["10", "01", "11"], n_pairs, p=[0.15, 0.15, 0.7])
    y_pa = pos & (state != "01")
    y_lat = pos & (state != "10")
    rows = []
    for i in range(n_pairs):
        for c in range(len(CLASSES)):
            for side, y in (("a", y_pa[i]), ("l", y_lat[i])):
                present = bool(y) if CLASSES[c] == "fracture" else bool(rng.random() < 0.2)
                z = rng.normal(lat_signal if present else -1.5, 1.0)
                rows.append({"image_id": f"{side}{i}", "patient_id": pid[i], "view": 0 if side == "a" else 1,
                             "cls": c, "own_max": 1 / (1 + np.exp(-z)), "present": int(present),
                             "n_boxes": int(present)})
    return pairs, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    pairs, imgs = synthetic()
    return pairs, imgs, st.study_table(imgs, pairs, CLASSES)


# ---------- building blocks ------------------------------------------------------------------

def test_image_score_follows_nms_confidence():
    assert list(st.image_score([0.0005, 0.001, 0.0011, 0.9])) == [0.0, 0.0, 0.0011, 0.9]


def test_age_band():
    assert [st.age_band(a) for a in (0.5, 5.99, 6.0, 9.9, 10.0, 13.99, 14.0, 19.0, float("nan"))] == \
        ["<6", "<6", "6-<10", "6-<10", "10-<14", "10-<14", ">=14", ">=14", "unknown"]


def test_attach_metadata_torus_is_study_level():
    meta = pd.DataFrame({
        "filestem": ["p1a", "p1l", "p1o", "p2a", "p2l", "p3a", "p3l"],
        "patient_id": [1, 1, 1, 2, 2, 3, 3], "study_number": [1, 1, 1, 1, 1, 1, 1],
        "age": [5.0, 5.0, 5.0, 12.0, 12.0, 15.0, 15.0],
        "initial_exam": [1, None, None, None, None, 0, 1], "cast": [None, None, None, 1, None, None, None],
        "ao_classification": [None, None, "23r-M/2.1", "23-M/3.1", "", "23u-M/2.1; 23r-E/1.1", None]})
    pairs = pd.DataFrame({"pair_id": ["x", "y", "z"], "patient_id": [1, 2, 3], "study_number": [1, 1, 1],
                          "ap_image_id": ["p1a", "p2a", "p3a"], "lat_image_id": ["p1l", "p2l", "p3l"]})
    out = st.attach_metadata(pairs, meta)
    assert list(out["torus"]) == [1, 0, 1]  # patient 1: the code sits on a third image of the study
    assert list(out["age_band"]) == ["<6", "10-<14", ">=14"]
    assert list(out["initial_exam"]) == [1, 0, 1] and list(out["cast"]) == [0, 1, 0]


def test_study_table_labels_and_scores(data):
    pairs, imgs, tab = data
    assert set(tab["finding"]) == set(st.FINDINGS) and len(tab) == len(pairs) * len(st.FINDINGS)
    t = tab[tab["finding"] == "fracture"].set_index("pair_id")
    key = imgs.set_index(["image_id", "cls"])
    i = pairs.index[5]
    a, l = pairs.at[i, "ap_image_id"], pairs.at[i, "lat_image_id"]
    row = t.loc[pairs.at[i, "pair_id"]]
    assert row["pa"] == pytest.approx(st.image_score([key.at[(a, 3), "own_max"]])[0])
    assert row["y"] == max(key.at[(a, 3), "present"], key.at[(l, 3), "present"])


def test_study_table_refuses_missing_images(data):
    pairs, imgs, _ = data
    with pytest.raises(ValueError):
        st.study_table(imgs[imgs["image_id"] != pairs["lat_image_id"].iloc[0]], pairs, CLASSES)


def test_fusion_rules():
    pa, lat = np.array([0.2, 0.9]), np.array([0.5, 0.1])
    assert list(st.fuse(pa, lat, "max")) == [0.5, 0.9]
    assert st.fuse(pa, lat, "mean") == pytest.approx([0.35, 0.5])
    assert st.fuse(pa, lat, "noisyor") == pytest.approx([1 - 0.8 * 0.5, 1 - 0.1 * 0.9])
    with pytest.raises(ValueError):
        st.fuse(pa, lat, "stack")
    with pytest.raises(ValueError):
        st.fuse(pa, lat, "median")


def test_threshold_gives_the_specificity():
    neg = np.random.default_rng(0).random(10_000)
    thr = st.threshold_at(neg)
    assert np.mean(neg <= thr) == pytest.approx(0.95, abs=0.001)


def test_rates():
    r = st.rates([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
    assert (r["tp"], r["fn"], r["fp"]) == (2, 1, 1)
    assert r["sens"] == pytest.approx(2 / 3) and r["spec"] == 0.5 and r["missed_per_100"] == pytest.approx(100 / 3)


def test_mcnemar_exact_and_chi2():
    from scipy.stats import binomtest, chi2

    a = np.array([1] * 10 + [0] * 2 + [1] * 5)
    b = np.array([0] * 10 + [1] * 2 + [1] * 5)
    r = st.mcnemar(a, b)
    assert (r["b"], r["c"], r["method"]) == (10, 2, "exact")
    assert r["p"] == pytest.approx(binomtest(10, 12, 0.5).pvalue)
    a = np.array([1] * 30 + [0] * 10)
    b = np.array([0] * 30 + [1] * 10)
    r = st.mcnemar(a, b)
    assert r["method"] == "chi2_cc" and r["p"] == pytest.approx(chi2.sf((20 - 1) ** 2 / 40, 1))
    assert st.mcnemar([1, 0], [1, 0])["p"] == 1.0


def test_delong_matches_auc_and_bootstrap():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.random(n) < 0.4
    s1 = rng.normal(y * 1.2, 1.0)
    s2 = 0.6 * s1 + rng.normal(y * 0.5, 0.8)
    r = st.delong(y, s1, s2)
    assert r["auc_new"] == pytest.approx(pm.auc(y, s1)) and r["auc_ref"] == pytest.approx(pm.auc(y, s2))
    se = (r["hi"] - r["lo"]) / (2 * 1.959964)
    boot = []
    for _ in range(800):
        i = rng.integers(0, n, n)
        boot.append(pm.auc(y[i], s1[i]) - pm.auc(y[i], s2[i]))
    assert se == pytest.approx(np.std(boot), rel=0.15)
    assert st.delong(y, s1, s1)["p"] == 1.0


def test_holm():
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = st.holm([0.2, np.nan, 0.01])
    assert out[0] == pytest.approx(0.2) and np.isnan(out[1]) and out[2] == pytest.approx(0.02)


def test_patient_bootstrap_keeps_patients_whole():
    groups = np.array([1, 1, 2, 3, 3, 3])
    for idx in st.PatientBootstrap(groups, n=50, seed=1):
        counts = pd.Series(groups[idx]).value_counts()
        for g, k in counts.items():
            assert k % int((groups == g).sum()) == 0


# ---------- whole analysis -------------------------------------------------------------------

def test_evaluate_run_development(data):
    _, _, tab = data
    s, c, dec = st.evaluate_run(tab, tab, "B0_s0", same_split=True, n_boot=200)
    f = s[s["finding"] == "fracture"].set_index("setting")
    assert set(f.index) == set(st.SETTINGS)
    assert all(abs(f.loc[k, "spec"] - 0.95) < 0.03 for k in ("pa", "lat", "max"))  # in-sample threshold
    assert f.loc["max", "sens"] > f.loc["pa", "sens"]  # LAT-only fractures are found only with both views
    assert f.loc["max", "sens_lo"] <= f.loc["max", "sens"] <= f.loc["max", "sens_hi"]
    prim = c[c["primary"]]
    assert len(prim) == 1 and prim.iloc[0]["finding"] == "fracture" and prim.iloc[0]["contrast"] == "max_vs_pa"
    assert prim.iloc[0]["mcnemar_p_holm"] == prim.iloc[0]["mcnemar_p"]  # the primary test is not adjusted
    sec = c[(c["contrast"] == "max_vs_pa") & ~c["primary"]]
    assert np.allclose(sec["mcnemar_p_holm"].values, st.holm(sec["mcnemar_p"].values), equal_nan=True)
    assert {"spec_diff", "spec_mcnemar_p", "auc_diff", "delong_p"} <= set(c.columns)
    assert set(dec) == set(st.FINDINGS) and dec["fracture"]["max"].dtype == bool


def test_evaluate_run_confirmatory_uses_valid_thresholds():
    pv, iv = synthetic(seed=1)
    pt, it = synthetic(seed=2)
    tv, tt = st.study_table(iv, pv, CLASSES), st.study_table(it, pt, CLASSES)
    s, _, _ = st.evaluate_run(tv, tt, "B0_s0", same_split=False, n_boot=100)
    a = tv[tv["finding"] == "fracture"]
    thr_pa = st.threshold_at(a.loc[a["y"] == 0, "pa"].values)
    got = s[(s["finding"] == "fracture") & (s["setting"] == "pa")].iloc[0]
    assert got["threshold"] == pytest.approx(thr_pa)
    b = tt[tt["finding"] == "fracture"]
    assert got["sens"] == pytest.approx(np.mean(b.loc[b["y"] == 1, "pa"].values > thr_pa))


def test_subgroups_and_pooled_seeds(data):
    _, _, tab = data
    _, _, dec = st.evaluate_run(tab, tab, "B0_s0", same_split=True, n_boot=50)
    sg = st.subgroups(tab, dec, "B0_s0", n_boot=50)
    assert {"torus", "age", "initial_exam", "cast"} <= set(sg["subgroup"])
    allrow = sg[(sg["finding"] == "fracture") & (sg["subgroup"] == "all")].iloc[0]
    y = tab[tab["finding"] == "fracture"]["y"].values.astype(bool)
    assert allrow["sens_max"] == pytest.approx(dec["fracture"]["max"][y].mean())
    pooled = st.pooled_seed_bootstrap({"B0_s0": tab, "B0_s1": tab}, {"B0_s0": dec, "B0_s1": dec},
                                      "fracture", "max", "pa", n_boot=50)
    assert pooled["diff_sd"] == 0.0 and pooled["diff_mean"] == pytest.approx(allrow["diff_max_vs_pa"])
    with pytest.raises(ValueError):
        st.pooled_seed_bootstrap({"a": tab, "b": tab.iloc[::-1]}, {"a": dec, "b": dec}, "fracture", "max", "pa", 10)


# ---------- scripts --------------------------------------------------------------------------

def test_study_script_end_to_end(tmp_path):
    pred = tmp_path / "predictions"
    pred.mkdir()
    for split, seed in (("valid", 4), ("test", 5)):
        pairs, imgs = synthetic(n_pairs=200, seed=seed)
        pairs.to_csv(pred / f"pairs_{split}.csv", index=False)
        for run in ("B0_s0", "B0_s1"):
            (pred / run).mkdir(exist_ok=True)
            imgs.to_csv(pred / run / f"{split}_images.csv.gz", index=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "12_study_level.py"), "--pred", str(pred),
                        "--runs", "B0_s0", "B0_s1", "--eval", "test", "--boot", "50"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "CONFIRMATORY" in r.stdout and "STUDY-LEVEL DONE" in r.stdout
    out = pred / "study_test"
    for name in ("settings.csv", "contrasts.csv", "subgroups.csv", "seeds.csv", "summary.txt"):
        assert (out / name).is_file()
    seeds = pd.read_csv(out / "seeds.csv")
    assert seeds.loc[(seeds["finding"] == "fracture") & (seeds["contrast"] == "max_vs_pa"), "seeds"].iloc[0] == 2


def test_predict_refuses_test_without_flag():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "10_predict.py"), "--runs", "B0_s0", "--split", "test"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "opened once" in (r.stdout + r.stderr)


def test_predict_script_on_tiny_data(tmp_path):
    """I1 end to end: tiny dataset, index, an untrained detector checkpoint, predictions on valid."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    sys.path.insert(0, str(REPO / "tests"))
    from tiny_data import make_tiny_dataset, tiny_config
    from ultralytics.nn.tasks import DetectionModel

    from vgra import index as ix

    root = make_tiny_dataset(tmp_path / "GRAZ")
    index, _ = ix.build_index(tiny_config(root))
    index.to_csv(tmp_path / "index.csv", index=False)
    torch.manual_seed(0)
    torch.save({"model": DetectionModel("yolov8n.yaml", nc=9, verbose=False)}, tmp_path / "w.pt")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "10_predict.py"), "--runs", "T_s0", "--weights",
                        str(tmp_path / "w.pt"), "--split", "valid", "--root", str(root), "--index",
                        str(tmp_path / "index.csv"), "--imgsz", "64", "--device", "cpu", "--no-warm",
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    pred = tmp_path / "out" / "predictions"
    pairs = pd.read_csv(pred / "pairs_valid.csv")
    imgs = pd.read_csv(pred / "T_s0" / "valid_images.csv.gz")
    dets = pd.read_csv(pred / "T_s0" / "valid_detections.csv.gz")
    assert len(pairs) == 4 and set(pairs["torus"]) == {0, 1}
    assert len(imgs) == 8 * 9 and set(imgs["image_id"]) == set(pairs["ap_image_id"]) | set(pairs["lat_image_id"])
    assert {"tp50", "tp95"} <= set(dets.columns)
    assert '"smoke": true' in (pred / "T_s0" / "valid_info.json").read_text()
    st.study_table(imgs, pairs, CLASSES)  # the analysis accepts what the runner wrote
