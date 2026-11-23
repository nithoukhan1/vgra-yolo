"""Reusable single-image inference (S1): detections, per-image class scores, true-positive tables.

Works on any list of images (any split; paired or unpaired). YOLO settings follow Ultralytics
validation: square letterbox, NMS with conf 0.001, IoU 0.7, max 300 boxes, multi-label.
RT-DETR (`predict_rtdetr`) follows the Ultralytics RT-DETR validator (`vgra.rtdetr.RDValidator`): image stretched
to a square (no letterbox), no NMS, one class per query (its highest score), conf > 0.001; the image score of a class
is its highest score over the 300 queries. Faster R-CNN inference is in vgra/frcnn.py and returns the same tables.
An optional `extra(feats, batch)` callback returns extra per-image scores (N, nc), for example the
presence branch's probabilities; they are stored next to the detector's own scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vgra import premise as pm
from vgra.data import pair_hyp, path_key
from vgra.presence import view_code


def image_dataset(paths: list, classes: list, imgsz: int, list_file: str | Path, augment: bool = False,
                  hyp_overrides: dict | None = None):
    """Ultralytics YOLODataset over an explicit image list (mosaic and mixing always off)."""
    from ultralytics.data.dataset import YOLODataset

    list_file = Path(list_file)
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
    data = {"names": dict(enumerate(classes)), "nc": len(classes), "channels": 3}
    return YOLODataset(img_path=str(list_file), imgsz=int(imgsz), augment=bool(augment),
                       hyp=pair_hyp(imgsz, hyp_overrides), rect=False, stride=32, pad=0.0,
                       prefix=f"{list_file.stem}: ", task="detect", data=data)


def rtdetr_dataset(paths: list, classes: list, imgsz: int, list_file: str | Path):
    """Ultralytics RTDETRDataset over an explicit image list, exactly as the RT-DETR validator builds it
    (no augmentation, no rect: every image stretched to imgsz x imgsz)."""
    from ultralytics.models.rtdetr.val import RTDETRDataset

    list_file = Path(list_file)
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
    data = {"names": dict(enumerate(classes)), "nc": len(classes), "channels": 3}
    return RTDETRDataset(img_path=str(list_file), imgsz=int(imgsz), augment=False, hyp=pair_hyp(imgsz), rect=False,
                         prefix=f"{list_file.stem}: ", data=data)


def image_loader(ds, batch: int = 16, shuffle: bool = False, workers: int = 2, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch), shuffle=bool(shuffle), num_workers=int(workers),
                      collate_fn=ds.collate_fn, pin_memory=torch.cuda.is_available(), generator=g,
                      drop_last=False)


def image_info(index: pd.DataFrame) -> dict:
    """path key -> (image_id, patient_id, view code) for every image in the index."""
    return {path_key(p): (i, int(pid) if pid == pid else -1, view_code(v))
            for p, i, pid, v in zip(index["path"], index["image_id"], index["patient_id"], index["view"])}


@torch.no_grad()
def predict(model, loader, classes: list, imgsz: int, info: dict, conf: float = 0.001, iou: float = 0.7,
            max_det: int = 300, extra=None, device=None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run a detector over a loader of single images.

    Returns
      dets  one row per detection: image_id, patient_id, view, cls, conf, own_max, [extra_max], tp50
      imgs  one row per image and class: image_id, patient_id, view, cls, own_max, [extra], present, n_boxes
      tp    (len(dets), 10) true-positive table at IoU 0.50:0.95 (Ultralytics matching)
    """
    from ultralytics.utils import nms
    from ultralytics.utils.ops import xywh2xyxy

    device = device or next(model.parameters()).device
    half = next(model.parameters()).dtype == torch.float16
    nc = len(classes)
    det_rows, img_rows, tps = [], [], []
    for batch in loader:
        x = batch["img"].to(device).float() / 255
        y, preds = model(x.half() if half else x)
        y = y.float()
        maxs = y[:, 4:, :].amax(-1).cpu().numpy()  # (N, nc) own-image highest score per class
        ext = None
        if extra is not None:
            views = torch.tensor([info[path_key(f)][2] for f in batch["im_file"]], device=device)
            ext = extra(preds["feats"], views, batch).float().cpu().numpy()
        out = nms.non_max_suppression(y, conf, iou, multi_label=True, max_det=max_det)
        for j, f in enumerate(batch["im_file"]):
            image_id, patient, view = info[path_key(f)]
            sel = batch["batch_idx"] == j
            gt_cls = batch["cls"][sel].view(-1).long()
            gt_xyxy = xywh2xyxy(batch["bboxes"][sel]) * imgsz
            d = out[j].cpu()
            tp = pm.match(d[:, :4], d[:, 5].long(), gt_xyxy, gt_cls)
            tps.append(tp)
            n_boxes = np.bincount(gt_cls.numpy(), minlength=nc)
            for c in range(nc):
                row = [image_id, patient, view, c, float(maxs[j, c])]
                row += [float(ext[j, c])] if ext is not None else []
                img_rows.append(row + [int(n_boxes[c] > 0), int(n_boxes[c])])
            for r in range(d.shape[0]):
                c = int(d[r, 5])
                row = [image_id, patient, view, c, float(d[r, 4]), float(maxs[j, c])]
                row += [float(ext[j, c])] if ext is not None else []
                det_rows.append(row + [bool(tp[r, 0])])
    ex = ["extra"] if extra is not None else []
    dets = pd.DataFrame(det_rows, columns=["image_id", "patient_id", "view", "cls", "conf", "own_max", *ex, "tp50"])
    imgs = pd.DataFrame(img_rows, columns=["image_id", "patient_id", "view", "cls", "own_max", *ex, "present", "n_boxes"])
    tp_all = np.concatenate(tps) if tps else np.zeros((0, 10), dtype=bool)
    return dets, imgs, tp_all


@torch.no_grad()
def predict_rtdetr(model, loader, classes: list, imgsz: int, info: dict, conf: float = 0.001,
                   device=None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """RT-DETR over a loader of `rtdetr_dataset`. Same outputs as `predict`. Detections as the Ultralytics RT-DETR
    validator makes them (per query: highest class score and its class; boxes x imgsz; conf > `conf`);
    own_max of a class = its highest score over all queries of the image."""
    from ultralytics.utils.ops import xywh2xyxy

    device = device or next(model.parameters()).device
    half = next(model.parameters()).dtype == torch.float16
    nc = len(classes)
    det_rows, img_rows, tps = [], [], []
    for batch in loader:
        x = batch["img"].to(device).float() / 255
        out = model(x.half() if half else x)
        y = (out[0] if isinstance(out, (list, tuple)) else out).float()  # (N, 300, 4 + nc)
        boxes, scores = y.split((4, nc), dim=-1)
        maxs = scores.amax(1).cpu().numpy()  # (N, nc) highest score of each class in each image
        for j, f in enumerate(batch["im_file"]):
            image_id, patient, view = info[path_key(f)]
            score, cls = scores[j].max(-1)
            order = score.argsort(descending=True)
            keep = order[score[order] > conf]
            d = torch.cat([xywh2xyxy(boxes[j] * imgsz)[keep], score[keep, None], cls[keep, None].float()], -1).cpu()
            sel = batch["batch_idx"] == j
            gt_cls = batch["cls"][sel].view(-1).long()
            gt_xyxy = xywh2xyxy(batch["bboxes"][sel]) * imgsz
            tp = pm.match(d[:, :4], d[:, 5].long(), gt_xyxy, gt_cls)
            tps.append(tp)
            n_boxes = np.bincount(gt_cls.numpy(), minlength=nc)
            for c in range(nc):
                img_rows.append([image_id, patient, view, c, float(maxs[j, c]), int(n_boxes[c] > 0), int(n_boxes[c])])
            for r in range(d.shape[0]):
                c = int(d[r, 5])
                det_rows.append([image_id, patient, view, c, float(d[r, 4]), float(maxs[j, c]), bool(tp[r, 0])])
    dets = pd.DataFrame(det_rows, columns=["image_id", "patient_id", "view", "cls", "conf", "own_max", "tp50"])
    imgs = pd.DataFrame(img_rows, columns=["image_id", "patient_id", "view", "cls", "own_max", "present", "n_boxes"])
    tp_all = np.concatenate(tps) if tps else np.zeros((0, 10), dtype=bool)
    return dets, imgs, tp_all
