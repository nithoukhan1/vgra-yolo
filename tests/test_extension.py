"""Extension detectors of protocol amendment v1.2, section 11.3: RD = RT-DETR-l, FR = Faster R-CNN R50-FPN v2.

Configs against the protocol; Faster R-CNN data, schedule, checkpoints, stop and resume; RT-DETR detections against
the Ultralytics RT-DETR validator; the premise test from I1 tables against its own inference; and the scripts end to
end on the tiny dataset. Nothing is downloaded (random weights)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from tiny_data import CLASSES, make_tiny_dataset, tiny_config  # noqa: E402

from vgra import frcnn, infer, utils  # noqa: E402
from vgra.data import path_key  # noqa: E402

RUN = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace", "cwd": REPO,
       "env": {**os.environ, "PYTHONIOENCODING": "utf-8"}}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from vgra import index as ix

    root = make_tiny_dataset(tmp_path_factory.mktemp("graz") / "GRAZ")
    index, _ = ix.build_index(tiny_config(root))
    path = root.parent / "index.csv"
    index.to_csv(path, index=False)
    return root, index, path


def split_rows(index, split):
    return index[(index["split"] == split) & index["is_base"] & index["in_metadata"]].sort_values("path")


# ---------- configs ---------------------------------------------------------------------------

def test_configs_follow_protocol_11_3():
    rd = utils.load_config(REPO / "configs" / "RD.yaml")
    t = rd["train"]
    assert rd["model"]["kind"] == "rtdetr" and rd["model"]["weights"] == "rtdetr-l.pt"
    assert (t["imgsz"], t["epochs"], t["patience"], t["batch"], t["optimizer"]) == (640, 72, 50, 8, "AdamW")
    assert (t["lr0"], t["weight_decay"], t["warmup_epochs"], t["warmup_bias_lr"]) == (1e-4, 1e-4, 3, 0.0)
    fr = utils.load_config(REPO / "configs" / "FR.yaml")
    t = fr["train"]
    assert fr["model"]["kind"] == "frcnn" and fr["model"]["weights"] == "COCO_V1"
    assert (t["imgsz"], t["epochs"], t["batch"], t["optimizer"], t["lr0"], t["momentum"], t["weight_decay"]) == \
        (1024, 24, 4, "SGD", 0.005, 0.9, 1e-4)
    assert (t["warmup_iters"], t["lr_steps"], t["lr_gamma"], t["hflip"], t["amp"]) == (1000, [16, 22], 0.1, 0.5, True)
    assert fr["frcnn"] == {"score_thresh": 0.001, "nms_iou": 0.5, "max_det": 300}
    for c in ("B0", "B1", "V8", "V8R"):
        assert utils.load_config(REPO / "configs" / f"{c}.yaml")["model"]["kind"] == "yolo"


# ---------- Faster R-CNN ----------------------------------------------------------------------

def test_frcnn_dataset_resize_boxes_and_flip(tiny):
    _, index, _ = tiny
    rows = split_rows(index, "valid")
    ds = frcnn.FrcnnDataset(list(rows["path"]), list(rows["label_path"]), imgsz=320)
    x, t, m = ds[0]
    assert tuple(x.shape) == (3, 320, 192) and m["hw0"] == (160, 96) and float(x.max()) <= 1.0  # longer side 320
    i = int((t["cls"] == 3).nonzero()[0])  # label "3 0.46875 0.5 0.3125 0.25" in a 192 x 320 image
    torch.testing.assert_close(t["boxes"][i], torch.tensor([60.0, 120.0, 120.0, 200.0]))
    assert torch.equal(t["labels"], t["cls"] + 1)  # 0 is background
    xf, tf, mf = frcnn.FrcnnDataset(list(rows["path"]), list(rows["label_path"]), 320, hflip=1.0)[0]
    assert mf["flipped"] and torch.equal(xf, x.flip(-1))
    torch.testing.assert_close(tf["boxes"][:, [0, 2]], 192 - t["boxes"][:, [2, 0]])
    many = frcnn.FrcnnDataset(list(rows["path"]) * 50, list(rows["label_path"]) * 50, 64, hflip=0.5, seed=3)
    draws = [many[k][2]["flipped"] for k in range(len(many))]
    assert 0.35 < np.mean(draws) < 0.65 and draws == [many[k][2]["flipped"] for k in range(len(many))]


def test_frcnn_lr_schedule_is_the_torchvision_reference():
    f = lambda e, it: frcnn.lr_factor(e, it, 1000, [16, 22], 0.1)  # noqa: E731
    assert math.isclose(f(0, 0), 0.001) and math.isclose(f(0, 500), 0.001 + 0.999 * 0.5) and f(0, 1000) == 1.0
    assert f(15, 10**6) == 1.0 and math.isclose(f(16, 10**6), 0.1) and math.isclose(f(22, 10**6), 0.01)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.001, total_iters=1000)
    for it in range(1000):
        assert math.isclose(opt.param_groups[0]["lr"], f(0, it), rel_tol=1e-9)
        opt.step()
        warm.step()


def test_frcnn_checkpoint_roundtrip_tables_and_metrics(tiny, tmp_path):
    _, index, _ = tiny
    torch.manual_seed(0)
    cfg = utils.load_config(REPO / "configs" / "FR.yaml")
    meta = {**frcnn.model_meta(cfg, 9), "imgsz": 128}
    model = frcnn.build_model(9, None, 128).eval()
    frcnn.save_checkpoint(tmp_path / "best.pt", model, meta, epoch=0)
    assert frcnn.is_checkpoint(tmp_path / "best.pt") and not frcnn.is_checkpoint(tmp_path / "nothing.pt")
    again, ck = frcnn.load_model(tmp_path / "best.pt")
    assert ck["imgsz"] == 128 and ck["n_classes"] == 9
    x = [torch.rand(3, 128, 80)]
    with torch.no_grad():
        assert torch.equal(model(x)[0]["scores"], again(x)[0]["scores"])
    rows = split_rows(index, "valid")
    ds = frcnn.FrcnnDataset(list(rows["path"]), list(rows["label_path"]), 128)
    dets, imgs, tp = frcnn.predict(again, frcnn.make_loader(ds, 4, False, 0), CLASSES, infer.image_info(index), "cpu")
    assert list(imgs.columns) == ["image_id", "patient_id", "view", "cls", "own_max", "present", "n_boxes"]
    assert list(dets.columns) == ["image_id", "patient_id", "view", "cls", "conf", "own_max", "tp50"]
    assert len(imgs) == 9 * len(rows) and tp.shape == (len(dets), 10)
    s = frcnn.summarise_tables(dets, imgs, tp, CLASSES)
    assert set(s) == {"map50", "map50_95", "precision", "recall", "classes_evaluated", "per_class"}
    assert s["classes_evaluated"] == 3  # fracture, pronator sign, text are labelled in the tiny data


def test_frcnn_train_stops_for_time_and_resumes(tiny, tmp_path):
    _, index, _ = tiny
    from vgra import lists, runs

    cfg = utils.load_config(REPO / "configs" / "FR.yaml")
    cfg["model"]["weights"] = None
    cfg["train"].update(epochs=2, warmup_iters=2, lr_steps=[1], imgsz=96)
    label_of = dict(zip(index["path"], index["label_path"]))
    train = lists.train_image_paths(index, "orig")[:6]
    valid = split_rows(index, "valid")
    data = (train, [label_of[p] for p in train], list(valid["path"]), list(valid["label_path"]), infer.image_info(index))

    class StopNow:
        def should_stop(self, seconds):
            return True

        def elapsed(self):
            return 0.0

    run = tmp_path / "FR_s0"
    m1 = frcnn.train_run(cfg, run, *data, seed=0, device="cpu", batch=2, budget=StopNow(), workers=0, log=lambda *a: None)
    assert m1["stopped_for_time"] and not m1["finished"] and m1["epochs_total"] == 1
    assert (run / "weights" / "resume.pt").is_file() and (run / "weights" / "best.pt").is_file()
    assert not runs.is_finished(run)
    with pytest.raises(SystemExit):  # a resumed run keeps its batch
        frcnn.train_run(cfg, run, *data, seed=0, device="cpu", batch=1, resume=str(run / "weights" / "resume.pt"),
                        workers=0, log=lambda *a: None)
    m2 = frcnn.train_run(cfg, run, *data, seed=0, device="cpu", resume=str(run / "weights" / "resume.pt"),
                         workers=0, log=lambda *a: None)  # no --batch: the checkpoint's batch
    assert m2["finished"] and m2["epochs_total"] == 2 and runs.is_finished(run) and m2["batch"] == 2
    assert not (run / "weights" / "resume.pt").exists()
    res = pd.read_csv(run / "results.csv")
    assert list(res["epoch"]) == [1, 2]
    assert math.isclose(res["lr"].iloc[1], 0.005 * 2 / 4 * 0.1)  # batch 2 halves lr; x 0.1 from epoch index 1
    assert m2["valid"]["classes_evaluated"] == 3


def test_frcnn_resumes_a_run_trained_from_pretrained_weights(tiny, tmp_path, monkeypatch):
    """2026-11-24, notebook D version 2: FR_s0 (trained from COCO weights) could not be resumed. torchvision freezes
    conv1, bn1 and layer1 only when pretrained weights are given; the model rebuilt without them to resume had every
    layer trainable (209 tensors instead of 176) and the optimizer state did not load."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    _, index, _ = tiny
    from vgra import lists

    fake_coco = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91).state_dict()
    monkeypatch.setattr(FasterRCNN_ResNet50_FPN_V2_Weights, "get_state_dict", lambda self, *a, **k: fake_coco)
    trainable = lambda m: [n for n, p in m.named_parameters() if p.requires_grad]  # noqa: E731
    coco, rebuilt = frcnn.build_model(9, "COCO_V1", 96), frcnn.build_model(9, None, 96)
    assert trainable(coco) == trainable(rebuilt) and len(trainable(coco)) == 176
    assert not [n for n in trainable(coco) if n.startswith(("backbone.body.conv1", "backbone.body.bn1",
                                                            "backbone.body.layer1"))]
    cfg = utils.load_config(REPO / "configs" / "FR.yaml")  # weights COCO_V1, as on Kaggle
    cfg["train"].update(epochs=2, warmup_iters=2, lr_steps=[1], imgsz=96)
    label_of = dict(zip(index["path"], index["label_path"]))
    train = lists.train_image_paths(index, "orig")[:4]
    valid = split_rows(index, "valid")
    data = (train, [label_of[p] for p in train], list(valid["path"]), list(valid["label_path"]), infer.image_info(index))

    class StopNow:
        def should_stop(self, seconds):
            return True

        def elapsed(self):
            return 0.0

    run = tmp_path / "FR_s0"
    m1 = frcnn.train_run(cfg, run, *data, seed=0, device="cpu", batch=2, budget=StopNow(), workers=0, log=lambda *a: None)
    assert m1["stopped_for_time"] and m1["epochs_total"] == 1
    m2 = frcnn.train_run(cfg, run, *data, seed=0, device="cpu", resume=str(run / "weights" / "resume.pt"), workers=0,
                         log=lambda *a: None)
    assert m2["finished"] and m2["epochs_total"] == 2


def test_frcnn_skips_non_finite_batches_and_stops_when_diverged(tiny, tmp_path, monkeypatch):
    _, index, _ = tiny
    from vgra import lists

    cfg = utils.load_config(REPO / "configs" / "FR.yaml")
    cfg["model"]["weights"] = None
    cfg["train"].update(epochs=1, warmup_iters=2, imgsz=96)
    label_of = dict(zip(index["path"], index["label_path"]))
    train = lists.train_image_paths(index, "orig")[:6]
    valid = split_rows(index, "valid")
    data = (train, [label_of[p] for p in train], list(valid["path"]), list(valid["label_path"]), infer.image_info(index))
    real, n = frcnn.build_model, {"calls": 0}

    def nan_on_first_batch(*a, **k):
        m = real(*a, **k)
        fwd = m.forward

        def forward(images, targets=None):
            out = fwd(images, targets)
            if targets is not None:
                n["calls"] += 1
                if n["calls"] == 1:
                    out = {key: v * float("nan") for key, v in out.items()}
            return out

        m.forward = forward
        return m

    monkeypatch.setattr(frcnn, "build_model", nan_on_first_batch)
    m = frcnn.train_run(cfg, tmp_path / "a", *data, seed=0, device="cpu", batch=2, workers=0, log=lambda *a: None)
    res = pd.read_csv(tmp_path / "a" / "results.csv")
    assert m["finished"] and list(res["skipped"]) == [1] and n["calls"] == 3  # 6 images, batch 2
    monkeypatch.setattr(frcnn, "MAX_SKIPPED", 0)
    n["calls"] = 0
    with pytest.raises(FloatingPointError, match="diverged"):
        frcnn.train_run(cfg, tmp_path / "b", *data, seed=0, device="cpu", batch=2, workers=0, log=lambda *a: None)


# ---------- RT-DETR ---------------------------------------------------------------------------

class RtdetrStub(torch.nn.Module):
    """Returns RT-DETR-shaped outputs (N, 300, 4 + 9) that depend only on the input; about 40% of the queries score
    below 0.001 for every class, scattered over the query order (as background queries of a trained model can)."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        g = torch.Generator().manual_seed(int(x.sum().item() * 1000) % (2 ** 31))
        boxes = torch.rand(x.shape[0], 300, 4, generator=g) * 0.4 + 0.3
        scores = torch.rand(x.shape[0], 300, 9, generator=g) ** 6
        low = torch.rand(x.shape[0], 300, 1, generator=g) < 0.4
        scores = torch.where(low, scores * 0.0009, scores)
        return torch.cat([boxes, scores], -1), None


def test_rtdetr_validator_mask_fix():
    """The 8.4.7 validator masks sorted predictions with the unsorted mask; RDValidator does not."""
    from ultralytics.models.rtdetr.val import RTDETRValidator

    from vgra.rtdetr import RDValidator

    y = torch.zeros(1, 3, 4 + 9)
    y[0, :, :4] = 0.5
    y[0, :, 4] = torch.tensor([0.0005, 0.9, 0.5])
    stock = RTDETRValidator(args=dict(imgsz=640, conf=0.001)).postprocess([y.clone(), None])[0]["conf"]
    fixed = RDValidator(args=dict(imgsz=640, conf=0.001)).postprocess([y.clone(), None])[0]["conf"]
    torch.testing.assert_close(stock, torch.tensor([0.5, 0.0005]))  # the defect: 0.9 dropped, 0.0005 kept
    torch.testing.assert_close(fixed, torch.tensor([0.9, 0.5]))


def test_rtdetr_detections_equal_the_validator(tiny, tmp_path):
    from ultralytics.models.rtdetr.val import RTDETRValidator

    from vgra.rtdetr import RDValidator

    _, index, _ = tiny
    rows = split_rows(index, "valid")
    ds = infer.rtdetr_dataset(list(rows["path"]), CLASSES, 160, tmp_path / "valid.txt")
    info = infer.image_info(index)
    stub = RtdetrStub()
    dets, imgs, tp = infer.predict_rtdetr(stub, infer.image_loader(ds, 4, workers=0), CLASSES, 160, info,
                                         conf=0.001, device=torch.device("cpu"))
    v = RDValidator(save_dir=tmp_path / "val", args=dict(imgsz=160, conf=0.001))
    stock = RTDETRValidator(save_dir=tmp_path / "val0", args=dict(imgsz=160, conf=0.001))
    n_images, stock_differs = 0, 0
    for batch in infer.image_loader(ds, 4, workers=0):
        x = batch["img"].float() / 255
        y = stub(x)[0]
        out = v.postprocess([y.clone(), None])
        out0 = stock.postprocess([y.clone(), None])
        for j, f in enumerate(batch["im_file"]):
            image_id = info[path_key(f)][0]
            mine = dets[dets["image_id"] == image_id]
            assert len(mine) == len(out[j]["conf"]) > 0
            np.testing.assert_allclose(mine["conf"].to_numpy(), out[j]["conf"].numpy(), atol=1e-6)
            assert (mine["cls"].to_numpy() == out[j]["cls"].numpy().astype(int)).all()
            assert float(out[j]["conf"].min()) > 0.001
            own = imgs[imgs["image_id"] == image_id].sort_values("cls")["own_max"].to_numpy()
            np.testing.assert_allclose(own, y[j, :, 4:].amax(0).numpy(), atol=1e-6)
            n_images += 1
            stock_differs += int(float(out0[j]["conf"].min()) <= 0.001)  # stock keeps sub-threshold rows
    assert n_images == len(rows) and stock_differs == n_images


# ---------- premise test from the I1 tables ---------------------------------------------------

def test_premise_from_i1_tables_equals_its_own_inference(tiny, tmp_path):
    from ultralytics.nn.tasks import DetectionModel

    root, _, index_csv = tiny
    torch.manual_seed(1)
    w = tmp_path / "runs" / "T_s0" / "weights"
    w.mkdir(parents=True)
    torch.save({"model": DetectionModel("yolov8n.yaml", nc=9, verbose=False)}, w / "best.pt")
    common = ["--root", str(root), "--index", str(index_csv), "--imgsz", "64", "--device", "cpu"]
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "10_predict.py"), "--runs", "T_s0", "--weights",
                        str(w / "best.pt"), "--split", "valid", "--no-warm", "--out", str(tmp_path / "i1"), *common], **RUN)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_premise_test.py"), "--weights", str(w / "best.pt"),
                        "--split", "valid", "--boot", "20", "--pairs", "3", "--out", str(tmp_path / "a"), *common], **RUN)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_premise_test.py"), "--runs", "T_s0", "--split",
                        "valid", "--boot", "20", "--pairs", "3", "--from-pred", str(tmp_path / "i1" / "predictions"),
                        "--out", str(tmp_path / "b"), *common], **RUN)
    assert r.returncode == 0 and "inputs from the I1 tables" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    for name in ("premise.csv", "images_T_s0.csv.gz", "detections_T_s0.csv.gz"):
        a = pd.read_csv(tmp_path / "a" / "premise" / name, dtype={"pair_id": str})
        b = pd.read_csv(tmp_path / "b" / "premise" / name, dtype={"pair_id": str})
        assert a.equals(b), name


# ---------- the scripts end to end ------------------------------------------------------------

@pytest.mark.parametrize("cfg_name,imgsz,weights,kind", [("FR", 128, "none", "frcnn"),
                                                          ("RD", 160, "rtdetr-l.yaml", "rtdetr")])
def test_smoke_training_then_i1_i3_i4(tiny, tmp_path, cfg_name, imgsz, weights, kind):
    root, _, index_csv = tiny
    common = ["--root", str(root), "--index", str(index_csv), "--device", "cpu"]
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "03_train_detector.py"), "--config",
                        str(REPO / "configs" / f"{cfg_name}.yaml"), "--seed", "0", "--smoke", "--fraction", "0.5",
                        "--imgsz", str(imgsz), "--weights", weights, "--batch", "2", "--no-warm",
                        "--out", str(tmp_path / "runs"), *common], **RUN)
    assert r.returncode == 0 and "RUN STATUS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    run_dir = tmp_path / "runs" / f"{cfg_name}_s0_smoke{imgsz}"
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["kind"] == kind and m["batch"] == 2 and m["epochs_total"] == 1
    best = run_dir / "weights" / "best.pt"
    assert best.is_file()

    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "10_predict.py"), "--runs", f"{cfg_name}_s0",
                        "--weights", str(best), "--split", "valid", "--imgsz", str(imgsz), "--no-warm",
                        "--out", str(out), *common], **RUN)
    assert r.returncode == 0 and "PREDICTIONS DONE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    pred = out / "predictions"
    info = json.loads((pred / f"{cfg_name}_s0" / "valid_info.json").read_text())
    assert info["kind"] == kind and info["smoke"] and info["imgsz"] == imgsz
    dets = pd.read_csv(pred / f"{cfg_name}_s0" / "valid_detections.csv.gz")
    assert {"tp50", "tp95", "conf", "own_max"} <= set(dets.columns)

    r = subprocess.run([sys.executable, str(REPO / "scripts" / "11_val_detectors.py"), "--runs", f"{cfg_name}_s0",
                        "--weights", str(best), "--split", "valid", "--imgsz", str(imgsz), "--out", str(out), *common],
                       **RUN)
    assert r.returncode == 0 and "DETECTOR VALIDATION DONE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    rec = json.loads((out / "detectors_valid" / f"{cfg_name}_s0.json").read_text())
    assert rec["kind"] == kind and rec["parameters"] > 1_000_000

    r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_premise_test.py"), "--runs", f"{cfg_name}_s0",
                        "--split", "valid", "--boot", "20", "--from-pred", str(pred), "--out", str(out)], **RUN)
    assert r.returncode == 0 and "GATE 2:" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
