"""FR: Faster R-CNN ResNet-50 FPN v2 (torchvision), extension detector (protocol amendment v1.2, section 11.3).

Same data as the YOLO detectors: the training list of `vgra/lists.py` (original images only), the YOLO label files
(class cx cy w h, normalised) and the valid / test images of the index. Recipe (`configs/FR.yaml`): COCO_V1 weights
with a new box predictor (9 classes + background); images resized so that the longer side is 1024 (aspect ratio
kept, no letterbox; cv2 INTER_LINEAR as Ultralytics); horizontal flip 0.5; SGD momentum 0.9, lr 0.005 at batch 4
(scaled with the batch), weight decay 1e-4; linear warm-up over 1,000 iterations from 0.001 x lr; lr x 0.1 at
epochs 16 and 22; 24 epochs; mixed precision. A batch with a non-finite loss is skipped and counted (results.csv
column `skipped`); more than MAX_SKIPPED in one epoch stops the run (the torchvision reference stops at the first,
which would block an unattended run on the same batch in every later version). Valid after every epoch; the best epoch has the highest Ultralytics
fitness 0.1 mAP50 + 0.9 mAP50-95, computed with Ultralytics matching and `ap_per_class` (`vgra/premise.py`), so the
numbers mean the same as for the YOLO detectors. Inference: score > 0.001, NMS IoU 0.5 per class, at most 300
detections per image.

Run folders follow the layout `vgra/runs.py` expects (results.csv, weights/best.pt, last.pt, resume.pt,
metrics.json, run_info.json), so --auto-resume, the time budget and the evaluation scripts treat FR like any run.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

KIND = "frcnn"
LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
RESULTS_COLS = ["epoch", *[f"train/{k}" for k in LOSS_KEYS], "train/loss", "metrics/precision(B)",
                "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness", "lr", "skipped", "train_s",
                "val_s"]
MAX_SKIPPED = 10  # batches with a non-finite loss are skipped; more than this in one epoch stops the run


# ----------------------------------------------------------------------------------------------
# model and checkpoints
# ----------------------------------------------------------------------------------------------

def build_model(n_classes: int, weights: str | None = "COCO_V1", imgsz: int = 1024, score_thresh: float = 0.001,
                nms_iou: float = 0.5, max_det: int = 300):
    """torchvision Faster R-CNN R50-FPN v2. `weights` = a FasterRCNN_ResNet50_FPN_V2_Weights name or None
    (random weights, used to rebuild a trained checkpoint and in tests; nothing is downloaded then)."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    w = getattr(FasterRCNN_ResNet50_FPN_V2_Weights, weights) if weights else None
    model = fasterrcnn_resnet50_fpn_v2(weights=w, weights_backbone=None, min_size=int(imgsz), max_size=int(imgsz),
                                       box_score_thresh=float(score_thresh), box_nms_thresh=float(nms_iou),
                                       box_detections_per_img=int(max_det))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(n_classes) + 1)
    return model


def model_meta(cfg: dict, n_classes: int) -> dict:
    f = cfg.get("frcnn", {})
    return {"kind": KIND, "n_classes": int(n_classes), "imgsz": int(cfg["train"]["imgsz"]),
            "score_thresh": float(f.get("score_thresh", 0.001)), "nms_iou": float(f.get("nms_iou", 0.5)),
            "max_det": int(f.get("max_det", 300))}


def save_checkpoint(path: str | Path, model, meta: dict, **extra) -> None:
    """Write atomically (a crash never leaves half a file)."""
    path = Path(path)
    ckpt = {**meta, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **extra}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def read_checkpoint(path: str | Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def is_checkpoint(path: str | Path) -> bool:
    try:
        ck = read_checkpoint(path)
    except Exception:
        return False
    return isinstance(ck, dict) and ck.get("kind") == KIND


def load_model(path: str | Path, device="cpu"):
    """Rebuild a trained FR model from best.pt / last.pt (no download). Returns (model in eval mode, checkpoint)."""
    ck = read_checkpoint(path)
    if not (isinstance(ck, dict) and ck.get("kind") == KIND):
        raise ValueError(f"{path} is not a Faster R-CNN checkpoint of this repo")
    model = build_model(ck["n_classes"], None, ck["imgsz"], ck["score_thresh"], ck["nms_iou"], ck["max_det"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def read_yolo_boxes(label_path: str | Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boxes in pixels (xyxy, clipped to the image) and class ids from a YOLO label file; degenerate boxes dropped."""
    try:
        text = Path(label_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    boxes, cls = [], []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(int(float(p[0])))
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    c = torch.tensor(cls, dtype=torch.int64)
    if len(b):
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        keep = ((b[:, 2] - b[:, 0]) > 0.01) & ((b[:, 3] - b[:, 1]) > 0.01)
        b, c = b[keep], c[keep]
    return b, c


class FrcnnDataset(Dataset):
    """One image per item: RGB float [0, 1] resized so that the longer side is `imgsz` (aspect ratio kept),
    target boxes in pixels of the resized image, labels = class + 1 (0 is background). Horizontal flip with
    probability `hflip`, drawn from (seed, epoch, item), so a resumed run sees the same flips."""

    def __init__(self, paths: list, label_paths: list, imgsz: int = 1024, hflip: float = 0.0, seed: int = 0):
        if len(paths) != len(label_paths):
            raise ValueError("paths and label_paths differ in length")
        self.paths, self.label_paths = list(paths), list(label_paths)
        self.imgsz, self.hflip, self.seed, self.epoch = int(imgsz), float(hflip), int(seed), 0

    def __len__(self) -> int:
        return len(self.paths)

    def load_image(self, i: int):
        import cv2

        im = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # BGR, 8 bit, as Ultralytics reads it
        if im is None:
            raise FileNotFoundError(f"image not found: {self.paths[i]}")
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(im[:, :, ::-1]), (h0, w0)

    def __getitem__(self, i: int):
        im, hw0 = self.load_image(i)
        h, w = im.shape[:2]
        boxes, cls = read_yolo_boxes(self.label_paths[i], w, h)
        flipped = False
        if self.hflip > 0 and np.random.default_rng([self.seed, self.epoch, i]).random() < self.hflip:
            im = np.ascontiguousarray(im[:, ::-1])
            fb = boxes.clone()
            fb[:, 0], fb[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
            boxes, flipped = fb, True
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        target = {"boxes": boxes, "labels": cls + 1, "cls": cls}
        return x, target, {"path": self.paths[i], "hw0": hw0, "hw": (h, w), "flipped": flipped}


def collate(batch: list):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def make_loader(ds: FrcnnDataset, batch: int, shuffle: bool, workers: int = 2, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch), shuffle=bool(shuffle), num_workers=int(workers), collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)


# ----------------------------------------------------------------------------------------------
# inference and metrics
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, classes: list, info: dict, device=None, amp: bool = True):
    """Same outputs as vgra.infer.predict: (dets, imgs, tp). Boxes and labels in pixels of the resized image;
    own_max = highest score of the class among the image's detections (0 without one)."""
    from vgra import premise as pm
    from vgra.data import path_key

    device = torch.device(device) if device is not None else next(model.parameters()).device
    nc = len(classes)
    det_rows, img_rows, tps = [], [], []
    model.eval()
    for xs, targets, metas in loader:
        xs = [x.to(device, non_blocking=True) for x in xs]
        with torch.autocast(device.type, dtype=torch.float16, enabled=bool(amp) and device.type == "cuda"):
            outs = model(xs)
        for t, m, o in zip(targets, metas, outs):
            image_id, patient, view = info[path_key(m["path"])]
            scores = o["scores"].float().cpu()
            order = torch.argsort(scores, descending=True)
            scores = scores[order]
            boxes = o["boxes"].float().cpu()[order]
            labels = (o["labels"].cpu()[order] - 1).long()
            gt_cls = t["cls"].long()
            tp = pm.match(boxes, labels, t["boxes"], gt_cls)
            tps.append(tp)
            maxs = np.zeros(nc)
            for c, s in zip(labels.tolist(), scores.tolist()):
                maxs[c] = max(maxs[c], s)
            n_boxes = np.bincount(gt_cls.numpy(), minlength=nc)
            for c in range(nc):
                img_rows.append([image_id, patient, view, c, float(maxs[c]), int(n_boxes[c] > 0), int(n_boxes[c])])
            for r in range(len(scores)):
                c = int(labels[r])
                det_rows.append([image_id, patient, view, c, float(scores[r]), float(maxs[c]), bool(tp[r, 0])])
    dets = pd.DataFrame(det_rows, columns=["image_id", "patient_id", "view", "cls", "conf", "own_max", "tp50"])
    imgs = pd.DataFrame(img_rows, columns=["image_id", "patient_id", "view", "cls", "own_max", "present", "n_boxes"])
    tp_all = np.concatenate(tps) if tps else np.zeros((0, 10), dtype=bool)
    return dets, imgs, tp_all


def summarise_tables(dets: pd.DataFrame, imgs: pd.DataFrame, tp: np.ndarray, classes: list) -> dict:
    """mAP, precision, recall and per-class numbers exactly as the Ultralytics validator computes them
    (`ap_per_class` over all detections), in the format of scripts/03_train_detector.summarise."""
    from vgra import premise as pm

    target_cls = np.repeat(imgs["cls"].to_numpy(), imgs["n_boxes"].to_numpy())
    return pm.summarise_detections(tp, dets["conf"].to_numpy(), dets["cls"].to_numpy(), target_cls, classes)


def fitness(summary: dict) -> float:
    return 0.1 * float(summary["map50"]) + 0.9 * float(summary["map50_95"])


# ----------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------

def lr_factor(epoch: int, it: int, warmup_iters: int, steps, gamma: float) -> float:
    """torchvision reference schedule: linear warm-up from 0.001 over `warmup_iters` iterations, then x gamma at
    every epoch in `steps` (0-based epochs: with steps [16, 22] the 17th and 23rd epochs use the lower rates)."""
    warm = 1.0 if it >= warmup_iters else 0.001 + (1 - 0.001) * it / max(int(warmup_iters), 1)
    return warm * float(gamma) ** sum(int(epoch) >= int(s) for s in steps)


def _append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULTS_COLS})


def train_run(cfg: dict, run_dir: str | Path, train_paths: list, train_labels: list, valid_paths: list,
              valid_labels: list, valid_info: dict, seed: int, device: str, batch: int | None = None,
              smoke: bool = False, fraction: float = 0.1, resume: str | None = None, budget=None,
              workers: int = 4, log=print, n_train_full: int | None = None) -> dict:
    """Train (or continue) one FR run and return the metrics dict written to metrics.json."""
    from vgra import utils

    t_cfg = cfg["train"]
    classes = list(cfg["data"]["classes"])
    run_dir = Path(run_dir)
    wdir = run_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    dev = torch.device(device if (device == "cpu" or str(device).startswith("cuda")) else f"cuda:{device}")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)  # nothing of this run lands on the other GPU (pinned-memory thread, AMP)
    use_amp = bool(t_cfg.get("amp", True)) and dev.type == "cuda"
    cfg_batch = int(t_cfg["batch"])
    resume_ck = read_checkpoint(resume) if resume else None
    if resume_ck is not None:  # a resumed run keeps its batch (as Ultralytics does)
        if batch and int(batch) != int(resume_ck["batch"]):
            raise SystemExit(f"resume: the run used batch {resume_ck['batch']}, not {batch}")
        batch = int(resume_ck["batch"])
    batch = int(batch or cfg_batch)
    lr = float(t_cfg["lr0"]) * batch / cfg_batch  # linear scaling with the batch (protocol 11.3)
    epochs = 1 if smoke else int(t_cfg["epochs"])
    steps, gamma = list(t_cfg.get("lr_steps", [16, 22])), float(t_cfg.get("lr_gamma", 0.1))
    warmup_iters = int(t_cfg.get("warmup_iters", 1000))
    if smoke:
        n = round(len(train_paths) * float(fraction))
        train_paths, train_labels = train_paths[:n], train_labels[:n]

    utils.set_seed(seed)
    meta = model_meta(cfg, len(classes))
    model = build_model(len(classes), None if resume_ck else cfg["model"].get("weights"), meta["imgsz"],
                        meta["score_thresh"], meta["nms_iou"], meta["max_det"]).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=float(t_cfg.get("momentum", 0.9)),
                                weight_decay=float(t_cfg.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, it_global, best_fit, best_epoch, best_summary = 0, 0, -1.0, -1, None
    if resume_ck:
        model.load_state_dict(resume_ck["state_dict"])
        optimizer.load_state_dict(resume_ck["optimizer"])
        if resume_ck.get("scaler"):
            scaler.load_state_dict(resume_ck["scaler"])
        start_epoch, it_global = int(resume_ck["epoch"]) + 1, int(resume_ck["it"])
        best_fit, best_epoch = float(resume_ck["best_fitness"]), int(resume_ck["best_epoch"])
        best_summary = resume_ck.get("best_summary")
        log(f"resume: continue after epoch {start_epoch} (iteration {it_global}); best epoch so far "
            f"{best_epoch + 1} (fitness {best_fit:.4f})")

    train_ds = FrcnnDataset(train_paths, train_labels, meta["imgsz"], float(t_cfg.get("hflip", 0.5)), seed)
    valid_ds = FrcnnDataset(valid_paths, valid_labels, meta["imgsz"], 0.0, seed)
    valid_loader = make_loader(valid_ds, max(batch, 1), shuffle=False, workers=workers, seed=seed)
    log(f"Faster R-CNN R50-FPN v2 | {count_parameters(model):,} parameters | train {len(train_ds)} images | "
        f"valid {len(valid_ds)} | batch {batch} | lr {lr:g} | epochs {epochs} | amp {use_amp} | device {dev}")

    clock = {"train": [], "val": []}
    stopped_for_time = False
    t_session = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        train_ds.epoch = epoch
        loader = make_loader(train_ds, batch, shuffle=True, workers=workers, seed=seed * 1000 + epoch)
        model.train()
        sums = dict.fromkeys(LOSS_KEYS, 0.0)
        n_it = skipped = 0
        t0 = time.time()
        for xs, targets, _ in loader:
            f = lr_factor(epoch, it_global, warmup_iters, steps, gamma)
            for g in optimizer.param_groups:
                g["lr"] = lr * f
            xs = [x.to(dev, non_blocking=True) for x in xs]
            tg = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in targets]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                losses = model(xs, tg)
                loss = sum(losses.values())
            if not torch.isfinite(loss):  # skipped and counted; more than MAX_SKIPPED in one epoch = diverged
                skipped += 1
                log(f"non-finite loss at epoch {epoch + 1}, iteration {it_global}: batch skipped "
                    f"{ {k: float(v.detach()) for k, v in losses.items()} }")
                if skipped > MAX_SKIPPED:
                    raise FloatingPointError(f"more than {MAX_SKIPPED} non-finite losses in epoch {epoch + 1}: "
                                             f"training diverged")
                optimizer.zero_grad(set_to_none=True)
                it_global += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in LOSS_KEYS:
                sums[k] += float(losses[k].detach())
            n_it += 1
            it_global += 1
        train_s = time.time() - t0
        clock["train"].append(train_s)

        t1 = time.time()
        dets, imgs, tp = predict(model, valid_loader, classes, valid_info, dev, amp=use_amp)
        summary = summarise_tables(dets, imgs, tp, classes)
        val_s = time.time() - t1
        clock["val"].append(val_s)
        fit = fitness(summary)
        row = {"epoch": epoch + 1, **{f"train/{k}": round(sums[k] / max(n_it, 1), 5) for k in LOSS_KEYS},
               "train/loss": round(sum(sums.values()) / max(n_it, 1), 5), "metrics/precision(B)": summary["precision"],
               "metrics/recall(B)": summary["recall"], "metrics/mAP50(B)": summary["map50"],
               "metrics/mAP50-95(B)": summary["map50_95"], "fitness": round(fit, 5),
               "lr": optimizer.param_groups[0]["lr"], "skipped": skipped, "train_s": round(train_s, 1),
               "val_s": round(val_s, 1)}
        _append_row(results_csv, row)
        if fit > best_fit:
            best_fit, best_epoch, best_summary = fit, epoch, summary
            save_checkpoint(wdir / "best.pt", model, meta, epoch=epoch, fitness=fit, summary=summary,
                            commit=utils.git_commit())
        save_checkpoint(wdir / "last.pt", model, meta, epoch=epoch, fitness=fit, summary=summary,
                        commit=utils.git_commit())
        save_checkpoint(wdir / "resume.pt", model, meta, epoch=epoch, it=it_global, optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), best_fitness=best_fit, best_epoch=best_epoch,
                        best_summary=best_summary, batch=batch, commit=utils.git_commit())
        log(f"epoch {epoch + 1}/{epochs} | loss {row['train/loss']:.4f} | valid mAP50 {summary['map50']:.4f} "
            f"mAP50-95 {summary['map50_95']:.4f} | fitness {fit:.4f} (best epoch {best_epoch + 1}) | "
            f"train {train_s / 60:.1f} min, valid {val_s / 60:.1f} min")
        if budget is not None and epoch < epochs - 1 and budget.should_stop(train_s + val_s):
            stopped_for_time = True
            log(f"time budget: stopping after epoch {epoch + 1} ({budget.elapsed() / 3600:.2f} h used). "
                f"Continue in the next version with --auto-resume.")
            break

    finished = (not stopped_for_time) and epoch >= epochs - 1
    if finished and not smoke:
        (wdir / "resume.pt").unlink(missing_ok=True)
    epochs_total = max(len(results_csv.read_text(encoding="utf-8").strip().splitlines()) - 1, 0) \
        if results_csv.exists() else 0
    metrics = {"run_name": run_dir.name, "exp_id": cfg["exp_id"], "kind": KIND, "seed": int(seed),
               "commit": utils.git_commit(), "finished": finished, "stopped_for_time": stopped_for_time,
               "imgsz": meta["imgsz"], "train_set": t_cfg.get("train_set", "orig"), "batch": batch, "lr0": lr,
               "epochs_this_session": len(clock["train"]), "epochs_total": epochs_total,
               "best_epoch": best_epoch + 1, "best_fitness": round(best_fit, 5),
               "total_seconds": round(time.time() - t_session),
               "train_seconds_per_epoch": round(sum(clock["train"]) / max(len(clock["train"]), 1), 1),
               "parameters": count_parameters(model), "weights": "best.pt", "valid": best_summary}
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if smoke:
        per_img = sum(clock["train"]) / max(len(train_ds), 1)
        mem = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0
        log("\n=== timing (smoke run) ===")
        log(f"trained on {len(train_ds)} images in {sum(clock['train']):.0f} s = {per_img:.3f} s per image "
            f"(batch {batch})")
        if n_train_full:
            log(f"estimated full epoch on {n_train_full} images: {per_img * n_train_full / 60:.1f} min training "
                f"+ {(clock['val'][0] if clock['val'] else float('nan')) / 60:.1f} min validation")
        log(f"peak GPU memory: {mem:.1f} GB")
    return metrics
