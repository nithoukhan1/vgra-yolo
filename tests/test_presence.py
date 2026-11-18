"""Gate P tests: presence branch, training helpers, AP50-gain statistics, verdict, inference."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from tiny_data import CLASSES, make_tiny_dataset, tiny_config  # noqa: E402

from vgra import gate_p as gp  # noqa: E402
from vgra import presence as pz  # noqa: E402

CH = (16, 32, 64)


def feats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, c, s, s, generator=g) for c, s in zip(CH, (16, 8, 4))]


def scored(n_classes=9):
    """A branch whose scorer is not at its zero start (all logits are 0 before training)."""
    torch.manual_seed(0)
    b = pz.PresenceBranch(CH, n_classes)
    torch.nn.init.normal_(b.pres_w, std=0.5)
    return b.eval()


def test_scorer_starts_at_zero():
    z = pz.PresenceBranch(CH, 9).eval()(feats(2), torch.zeros(2, dtype=torch.long))
    assert torch.equal(z, torch.zeros(2, 9))  # every class starts at presence 0.5


def test_branch_shapes_and_projection():
    b = scored()
    f = feats(4)
    z_pa = b(f, torch.zeros(4, dtype=torch.long))
    z_lat = b(f, torch.ones(4, dtype=torch.long))
    assert z_pa.shape == (4, 9)
    assert not torch.allclose(z_pa, z_lat)  # the projection tag changes the answer
    assert b.attention(f, torch.zeros(4, dtype=torch.long)).shape == (4, 9, 16 * 16 + 8 * 8 + 4 * 4)


def test_branch_images_are_independent():
    b = scored()
    f = feats(3)
    v = torch.tensor([0, 1, 2])
    z1 = b(f, v)
    f2 = [x.clone() for x in f]
    for x in f2:
        x[2] += 5.0
    z2 = b(f2, v)
    assert torch.equal(z1[:2], z2[:2]) and not torch.equal(z1[2], z2[2])


def test_branch_learns_a_simple_presence_signal():
    torch.manual_seed(0)
    b = pz.PresenceBranch(CH, 2)
    opt = torch.optim.AdamW(b.parameters(), lr=3e-3)
    for _ in range(150):
        f = feats(16, seed=int(torch.randint(0, 10_000, (1,))))
        y = torch.zeros(16, 2)
        y[:8, 0] = 1
        f[0][:8, 0] += 3.0  # class 0 present = a bright channel in P3
        loss = pz.presence_loss(b(f, torch.zeros(16, dtype=torch.long)), y, torch.ones(2))
        opt.zero_grad()
        loss.backward()
        opt.step()
    f = feats(16, seed=99999)
    f[0][:8, 0] += 3.0
    p = torch.sigmoid(b.eval()(f, torch.zeros(16, dtype=torch.long)))
    p = p.detach()
    assert float(p[:8, 0].mean()) > 0.8 and float(p[8:, 0].mean()) < 0.2


def test_targets_weights_and_schedule():
    batch = {"cls": torch.tensor([[3.0], [6.0], [3.0]]), "batch_idx": torch.tensor([0.0, 1.0, 1.0])}
    y = pz.presence_targets(batch, 3, 9)
    assert y[0, 3] == 1 and y[1, 6] == 1 and y[1, 3] == 1 and y[2].sum() == 0
    rows = pd.DataFrame({f"has_{c}": [0] * 100 for c in CLASSES})
    rows.loc[:9, "has_fracture"] = 1  # 10 of 100 -> weight 9
    rows.loc[:0, "has_pronatorsign"] = 1  # 1 of 100 -> 99, capped at 50
    w = pz.positive_weights(rows, CLASSES)
    assert w[CLASSES.index("fracture")] == pytest.approx(9.0) and w[CLASSES.index("pronatorsign")] == 50.0
    assert w[CLASSES.index("text")] == 50.0  # no positives -> cap
    assert pz.lr_at(0, 1000, 1e-3) == pytest.approx(5e-6)
    assert pz.lr_at(199, 1000, 1e-3) == pytest.approx(1e-3)
    assert pz.lr_at(1000, 1000, 1e-3) == pytest.approx(0.0, abs=1e-12)
    assert pz.view_code("ap") == 0 and pz.view_code("lat") == 1 and pz.view_code("other") == 2 and pz.view_code("") == 2


def synthetic_class(informative: bool, n_pat: int = 300, seed: int = 0):
    """Detections of one class. Positive images: one true positive + false positives; negative
    images: false positives only. Presence is high in positive images if informative."""
    rng = np.random.default_rng(seed)
    det, img = [], []
    for p in range(n_pat):
        for view in (0, 1):
            pos = rng.random() < 0.3
            pres = (0.85 if pos else 0.15) + 0.1 * rng.normal() if informative else rng.random()
            pres = float(np.clip(pres, 0.01, 0.99))
            confs = list(rng.uniform(0.01, 0.6, 3))
            tps = [False] * 3
            if pos:
                confs.append(float(np.clip(0.45 + 0.2 * rng.normal(), 0.02, 0.99)))
                tps.append(True)
            own = max(confs)
            det += [(p, view, c, own, pres, t) for c, t in zip(confs, tps)]
            img.append((p, view, own, pres, int(pos), int(pos)))
    dets = pd.DataFrame(det, columns=["patient_id", "view", "conf", "own_max", "extra", "tp50"])
    imgs = pd.DataFrame(img, columns=["patient_id", "view", "own_max", "extra", "present", "n_boxes"])
    return dets, imgs


def test_ap_gain_interval_informative_vs_noise():
    for informative, expect in ((True, True), (False, False)):
        d, im = synthetic_class(informative)
        y, g = d["tp50"].to_numpy().astype(int), d["patient_id"].to_numpy()
        from vgra import premise as pm
        pb = pm.crossfit(gp.design(d["conf"], d["view"], d["own_max"]), y, g)
        pn = pm.crossfit(gp.design(d["conf"], d["view"], d["own_max"], d["extra"]), y, g)
        boxes = im.groupby("patient_id")["n_boxes"].sum().to_dict()
        lo, hi = gp.bootstrap_ap_delta(y, pb, pn, g, boxes, n=200)
        assert (lo > 0) is expect, (informative, lo, hi)


def build(informative_for):
    dets, imgs = [], []
    for c in range(len(CLASSES)):
        d, im = synthetic_class(CLASSES[c] in informative_for, n_pat=200, seed=c)
        dets.append(d.assign(cls=c))
        imgs.append(im.assign(cls=c))
    return pd.concat(dets, ignore_index=True), pd.concat(imgs, ignore_index=True)


def test_verdict_pass_and_fail():
    d, im = build({"pronatorsign", "softtissue"})
    table = gp.evaluate(d, im, CLASSES, n_boot=100)
    ok, good, mb, mn = gp.verdict(table)
    assert ok and set(good) >= {"pronatorsign", "softtissue"} and mn >= mb
    d, im = build({"pronatorsign"})  # one target class only -> FAIL by rule
    ok, good, _, _ = gp.verdict(gp.evaluate(d, im, CLASSES, n_boot=100))
    assert not ok and good == ["pronatorsign"]


def test_predict_on_tiny_data(tmp_path):
    from ultralytics.nn.tasks import DetectionModel

    from vgra import index as ix
    from vgra import infer

    root = make_tiny_dataset(tmp_path / "GRAZ")
    index, _ = ix.build_index(tiny_config(root))
    valid = list(index.loc[index["split"] == "valid", "path"])
    ds = infer.image_dataset(valid, CLASSES, 128, tmp_path / "lists" / "valid.txt")
    torch.manual_seed(0)
    model = DetectionModel("yolo11n.yaml", nc=len(CLASSES), verbose=False).eval()
    branch = pz.PresenceBranch((64, 128, 256), len(CLASSES)).eval()
    info = infer.image_info(index)
    dets, imgs, tp = infer.predict(model, infer.image_loader(ds, 4, workers=0), CLASSES, 128, info,
                                   extra=lambda f, v, b: torch.sigmoid(branch(f, v)))
    assert len(imgs) == len(valid) * len(CLASSES) and len(tp) == len(dets)
    assert set(imgs["view"]) <= {0, 1, 2} and imgs["extra"].between(0, 1).all()
    assert imgs.loc[imgs["cls"] == CLASSES.index("fracture"), "present"].all()  # every tiny image has a fracture
