"""STEP-03: train a single-view detector (B0 = YOLO11s, B1 = YOLO11s + SCC, V8 = YOLOv8s,
V8R = YOLOv8s + ResCBAM of Ju et al.; extension detectors of protocol amendment v1.2, section 11.3:
RD = RT-DETR-l with the Ultralytics RT-DETR trainer (vgra/rtdetr.py: validator mask fix), FR = Faster R-CNN R50-FPN v2
with vgra/frcnn.py).
The detector family comes from model.kind in the config (yolo | rtdetr | frcnn); every option below works for all.

Check an extension detector (no training):
    python scripts/03_train_detector.py --config configs/RD.yaml --check
    python scripts/03_train_detector.py --config configs/FR.yaml --check

Pre-specified batch fallback (protocol 11.3: only if the configured batch runs out of memory in the first epoch):
    python scripts/03_train_detector.py --config configs/RD.yaml --seed 0 --device 0 --auto-resume --batch 4

Check the ResCBAM model (no training):
    python scripts/03_train_detector.py --config configs/V8R.yaml --check

Check the SCC swap (no training):
    python scripts/03_train_detector.py --config configs/B1.yaml --check

Short timing run (1 epoch on a fraction of the training images):
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --smoke --imgsz 1024

The same, with the images held in memory (measures training speed without image loading):
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --smoke --imgsz 1024 --cache ram

Full run:
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0

Background training on Kaggle (one command for every version of the notebook):
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --device 0 --auto-resume
  Version 1 starts the run. It stops cleanly before the time budget (train.max_hours) and keeps
  weights/resume.pt. Later versions, with the previous output attached as input, find the run
  folder under /kaggle/input, copy it and continue. A finished run is not trained again.

Continue a run by hand:
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --resume <path to resume.pt>

One run uses one GPU (--device). Outputs go to <out_dir>/<run name>/.
Before training, every image and label the run needs is read once in parallel (skip with --no-warm),
because Kaggle input files are slow on their first read in a session.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

T_START = time.time()  # the time budget counts from here

REPO_ROOT = Path(__file__).resolve().parents[1]

# train arguments that are passed from the config to Ultralytics as they are
PASS_THROUGH = ("epochs", "batch", "patience", "optimizer", "lr0", "cos_lr", "workers", "close_mosaic",
                "warmup_epochs", "amp", "cache", "weight_decay", "warmup_bias_lr")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--imgsz", type=int, default=None, help="override train.imgsz")
    ap.add_argument("--check", action="store_true", help="check the SCC swap or the ResCBAM model and exit")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a fraction of the training images")
    ap.add_argument("--fraction", type=float, default=0.1, help="share of training images in a smoke run")
    ap.add_argument("--resume", default=None, help="path to resume.pt (or last.pt) of the run to continue")
    ap.add_argument("--auto-resume", action="store_true",
                    help="continue the run found under --resume-search, or start it if none is found")
    ap.add_argument("--resume-search", default="/kaggle/input", help="where earlier outputs are attached")
    ap.add_argument("--max-hours", type=float, default=None, help="time budget; default train.max_hours")
    ap.add_argument("--reserve-min", type=float, default=None, help="minutes kept for the final evaluation")
    ap.add_argument("--test", action="store_true", help="also evaluate on the test split (final runs only)")
    ap.add_argument("--cache", choices=["ram"], default=None, help="keep resized images in memory (Ultralytics cache)")
    ap.add_argument("--no-warm", action="store_true", help="skip reading the files into memory before training")
    ap.add_argument("--warm-threads", type=int, default=32)
    ap.add_argument("--index", default=None, help="path to index.csv")
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--weights", default=None, help="override model.weights")
    ap.add_argument("--batch", type=int, default=None,
                    help="override train.batch (protocol 11.3 fallback after an out-of-memory first epoch)")
    return ap.parse_args()


def check_scc(weights: str, n_classes: int) -> bool:
    """A model with fresh SCC adapters must give exactly the outputs of the stock model."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from vgra.model import count_parameters
    from vgra.scc import add_scc, scc_strengths

    stock = YOLO(weights).model if str(weights).endswith(".pt") else DetectionModel(weights, nc=n_classes, verbose=False)
    stock = stock.float().eval()
    ours = add_scc(copy.deepcopy(stock)).eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        a, b = stock(x), ours(x)
        a = a[0] if isinstance(a, (list, tuple)) else a
        b = b[0] if isinstance(b, (list, tuple)) else b
        buf = io.BytesIO()
        torch.save({"model": ours}, buf)
        buf.seek(0)
        again = torch.load(buf, weights_only=False)["model"](x)
        again = again[0] if isinstance(again, (list, tuple)) else again
        clone = copy.deepcopy(ours)(x)
        clone = clone[0] if isinstance(clone, (list, tuple)) else clone

    n0, n1 = count_parameters(stock), count_parameters(ours)
    checks = [
        ("layers 2 and 4 are C3K2_SCC", all(type(ours.model[i]).__name__ == "C3K2_SCC" for i in (2, 4)), ""),
        ("outputs equal the stock model", bool(torch.equal(a, b)), f"max abs diff {float((a - b).abs().max()):.3g}"),
        ("model saves and reloads", bool(torch.equal(again, b)), ""),
        ("model deep-copies", bool(torch.equal(clone, b)), ""),
        ("adapters start switched off", all(v == 0.0 for v in scc_strengths(ours).values()), str(scc_strengths(ours))),
    ]
    print(f"weights: {weights}")
    print(f"parameters: stock {n0:,} | with SCC {n1:,} | added {n1 - n0:,} ({100 * (n1 - n0) / n0:.2f}%)")
    ok_all = True
    for name, ok, detail in checks:
        ok_all = ok_all and ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print("SCC CHECK: PASS" if ok_all else "SCC CHECK: FAIL")
    return ok_all


def check_rescbam(weights: str, n_classes: int) -> bool:
    """The ResCBAM model has Ju et al.'s layout, gets every shareable pretrained tensor, and saves cleanly."""
    import torch
    from ultralytics import YOLO

    from vgra import rescbam
    from vgra.model import count_parameters

    stock = YOLO(weights).model
    ours = rescbam.build(n_classes)
    r = rescbam.transfer_stock_weights(ours, stock)
    ours.eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        a = ours(x)
        a = a[0] if isinstance(a, (list, tuple)) else a
        buf = io.BytesIO()
        torch.save({"model": ours}, buf)
        buf.seek(0)
        again = torch.load(buf, weights_only=False)["model"](x)
        again = again[0] if isinstance(again, (list, tuple)) else again
        clone = copy.deepcopy(ours)(x)
        clone = clone[0] if isinstance(clone, (list, tuple)) else clone
    # only the class layers of the head differ (80 COCO classes vs ours): stock layer 22 = our layer 26
    shared_ok = all(k.startswith("model.22.cv3.") for k in r["skipped"]) and \
        all(k.startswith("model.26.cv3.") for k in r["missing_outside_blocks"])
    n0, n1 = count_parameters(stock), count_parameters(ours)
    checks = [
        ("ResCBAM blocks at layers 13, 17, 21, 25", rescbam.block_indices(ours) == [13, 17, 21, 25], ""),
        ("detect reads 17, 21, 25; P4 skip reads 12", ours.model[26].f == [17, 21, 25] and ours.model[19].f == [-1, 12], ""),
        ("pretrained tensors copied", shared_ok,
         f"{r['transferred']}/{r['stock_tensors']}, not copied: {len(r['skipped'])} class-layer tensors"),
        ("model saves and reloads", bool(torch.equal(again, a)), ""),
        ("model deep-copies", bool(torch.equal(clone, a)), ""),
    ]
    print(f"weights: {weights}")
    print(f"parameters: stock YOLOv8s {n0:,} (80 classes) | YOLOv8s + ResCBAM {n1:,} ({n_classes} classes)")
    ok_all = True
    for name, ok, detail in checks:
        ok_all = ok_all and ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print("RESCBAM CHECK: PASS" if ok_all else "RESCBAM CHECK: FAIL")
    return ok_all


def check_rtdetr(weights: str, n_classes: int) -> bool:
    """RT-DETR-l: the COCO checkpoint loads, a 9-class model takes every tensor but the class layers, and the
    evaluation output is (images, 300 queries, 4 boxes + classes)."""
    import torch
    from ultralytics.nn.tasks import RTDETRDetectionModel, load_checkpoint

    from vgra.model import count_parameters

    stock, _ = load_checkpoint(weights, device="cpu")
    ours = RTDETRDetectionModel(stock.yaml, nc=n_classes, verbose=False)
    csd, osd = stock.float().state_dict(), ours.state_dict()
    shared = [k for k in osd if k in csd and csd[k].shape == osd[k].shape]
    not_shared = [k for k in osd if k not in shared]
    class_layers = all(("score_head" in k) or ("denoising_class_embed" in k) for k in not_shared)
    ours.load(stock, verbose=False)
    copied = all(torch.equal(ours.state_dict()[k], csd[k]) for k in shared)
    ours.eval()
    with torch.no_grad():
        y = ours(torch.rand(1, 3, 640, 640))[0]
    n0, n1 = count_parameters(stock), count_parameters(ours)
    checks = [("COCO checkpoint loads", n0 > 0, f"{n0:,} parameters (80 classes)"),
              ("pretrained tensors copied", copied and class_layers,
               f"{len(shared)}/{len(osd)}, not copied: {len(not_shared)} class-layer tensors"),
              ("evaluation output (1, 300, 4 + classes)", tuple(y.shape) == (1, 300, 4 + n_classes), str(tuple(y.shape)))]
    print(f"weights: {weights} | RT-DETR-l with {n_classes} classes: {n1:,} parameters")
    ok_all = True
    for name, ok, detail in checks:
        ok_all = ok_all and ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print("RTDETR CHECK: PASS" if ok_all else "RTDETR CHECK: FAIL")
    return ok_all


def check_frcnn(cfg: dict, n_classes: int, weights: str | None) -> bool:
    """Faster R-CNN: the model builds with a 10-way box predictor, trains one step and predicts on a dummy image."""
    import torch

    from vgra import frcnn

    meta = frcnn.model_meta(cfg, n_classes)
    model = frcnn.build_model(n_classes, weights, 256, meta["score_thresh"], meta["nms_iou"], meta["max_det"])
    x = [torch.rand(3, 256, 200)]
    t = [{"boxes": torch.tensor([[20.0, 30.0, 120.0, 140.0]]), "labels": torch.tensor([4])}]
    model.train()
    losses = model(x, t)
    model.eval()
    with torch.no_grad():
        out = model(x)[0]
    checks = [("box predictor = classes + background", model.roi_heads.box_predictor.cls_score.out_features == n_classes + 1,
               str(model.roi_heads.box_predictor.cls_score.out_features)),
              ("training step gives the 4 losses", set(losses) == set(frcnn.LOSS_KEYS), ", ".join(sorted(losses))),
              ("prediction keys", {"boxes", "labels", "scores"} <= set(out), f"{len(out['scores'])} detections"),
              ("inference settings", (model.roi_heads.score_thresh, model.roi_heads.nms_thresh,
                                      model.roi_heads.detections_per_img) == (meta["score_thresh"], meta["nms_iou"],
                                                                              meta["max_det"]), "")]
    print(f"weights: {weights or 'random (no download)'} | Faster R-CNN R50-FPN v2: {frcnn.count_parameters(model):,} "
          f"parameters")
    ok_all = True
    for name, ok, detail in checks:
        ok_all = ok_all and ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    print("FRCNN CHECK: PASS" if ok_all else "FRCNN CHECK: FAIL")
    return ok_all


def summarise(m) -> dict:
    """Overall and per-class numbers from an Ultralytics detection metrics object."""
    box, names = m.box, m.names
    per_class = {}
    for j, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {"ap50": round(float(box.ap50[j]), 4), "ap50_95": round(float(box.ap[j]), 4),
                                    "precision": round(float(box.p[j]), 4), "recall": round(float(box.r[j]), 4)}
    return {"map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "classes_evaluated": len(per_class), "per_class": per_class}


def evaluate(weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: str, save_dir: Path,
             kind: str = "yolo") -> dict:
    """Validate saved Ultralytics weights on one split (YOLO or RT-DETR validator)."""
    from ultralytics import RTDETR, YOLO

    extra = {}
    if kind == "rtdetr":
        from vgra.rtdetr import RDValidator

        extra["validator"] = RDValidator  # the 8.4.7 validator with its confidence mask aligned (vgra/rtdetr.py)
    model = RTDETR(str(weights)) if kind == "rtdetr" else YOLO(str(weights))
    return summarise(model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False,
        project=str(save_dir), name=f"eval_{split}", exist_ok=True, **extra))


def train_frcnn(args, cfg: dict, index, run_dir: Path, name: str, train_set: str, n_train_full: int, imgsz) -> int:
    """FR (Faster R-CNN): same lists, run folder, time budget and resume rules as the Ultralytics detectors."""
    from vgra import frcnn, infer, lists, utils
    from vgra.runs import TimeBudget

    if int(imgsz) != int(cfg["train"]["imgsz"]):
        cfg["train"]["imgsz"] = int(imgsz)
    if args.weights:  # e.g. COCO_V1, or none = random weights (tests; nothing is downloaded)
        cfg["model"]["weights"] = None if args.weights.lower() == "none" else args.weights
    info_path = utils.write_run_info(run_dir, cfg, args.seed)
    print(f"run folder: {run_dir} | run info: {info_path.name}")
    train_cfg = cfg["train"]
    max_hours = args.max_hours if args.max_hours is not None else (None if args.smoke else train_cfg.get("max_hours"))
    reserve = args.reserve_min if args.reserve_min is not None else train_cfg.get("reserve_minutes", 20)
    budget = TimeBudget(max_hours, reserve, start=T_START) if max_hours else None
    if budget:
        print(f"time budget: {max_hours} h from the start of this script; the run stops cleanly after the "
              f"last epoch that fits (reserve {reserve} min) and can be continued with --auto-resume")
    label_of = dict(zip(index["path"], index["label_path"]))
    train_paths = lists.train_image_paths(index, train_set)
    valid = index[(index["split"] == "valid") & index["is_base"] & index["in_metadata"]].sort_values("path")
    metrics = frcnn.train_run(
        cfg, run_dir, train_paths, [label_of[p] for p in train_paths], list(valid["path"]), list(valid["label_path"]),
        infer.image_info(index), int(args.seed), str(args.device), args.batch, bool(args.smoke), float(args.fraction),
        args.resume, budget, int(train_cfg.get("workers", 4)), n_train_full=n_train_full)
    if args.test and metrics["finished"]:
        import torch

        test = index[(index["split"] == "test") & index["is_base"] & index["in_metadata"]].sort_values("path")
        dev = torch.device("cpu" if str(args.device) == "cpu" else f"cuda:{args.device}")
        model, _ = frcnn.load_model(run_dir / "weights" / "best.pt", dev)
        ds = frcnn.FrcnnDataset(list(test["path"]), list(test["label_path"]), int(train_cfg["imgsz"]))
        dets, imgs, tp = frcnn.predict(model, frcnn.make_loader(ds, metrics["batch"], False, 2), cfg["data"]["classes"],
                                       infer.image_info(index), dev)
        metrics["test"] = frcnn.summarise_tables(dets, imgs, tp, cfg["data"]["classes"])
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    v = metrics.get("valid") or {"map50": float("nan"), "map50_95": float("nan"), "classes_evaluated": 0, "per_class": {}}
    print(f"\n=== {name} ===")
    print(f"valid mAP50 {v['map50']:.4f} | mAP50-95 {v['map50_95']:.4f} | classes evaluated {v['classes_evaluated']} "
          f"| best epoch {metrics['best_epoch']}")
    for c, r in v["per_class"].items():
        print(f"  {c:<20} AP50 {r['ap50']:.4f}  AP50-95 {r['ap50_95']:.4f}")
    print(f"wrote {run_dir / 'metrics.json'}")
    print(f"RUN STATUS: {'FINISHED' if metrics['finished'] else 'STOPPED FOR TIME'} after {metrics['epochs_total']} "
          f"epochs (this session {metrics['epochs_this_session']})")
    return 0


def main() -> int:
    args = parse_args()
    from vgra import index as ix
    from vgra import lists, pairs, utils

    cfg = utils.load_config(args.config)
    data, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    if args.root:
        data["root"] = args.root
    out_root = Path(args.out or cfg["out_dir"])
    weights = args.weights or model_cfg["weights"]
    use_scc = bool(model_cfg["scc"])
    use_rescbam = bool(model_cfg.get("rescbam", False))
    kind = model_cfg.get("kind", "yolo")
    if kind not in ("yolo", "rtdetr", "frcnn"):
        raise SystemExit(f"model.kind must be yolo, rtdetr or frcnn, not {kind!r}")
    print(f"repo commit: {utils.git_commit()} | config: {cfg['exp_id']} | kind: {kind} | scc: {use_scc} | "
          f"rescbam: {use_rescbam}")

    if args.check:
        if kind == "rtdetr":
            return 0 if check_rtdetr(weights, len(data["classes"])) else 1
        if kind == "frcnn":
            w = None if (args.weights or "none").lower() == "none" else args.weights
            return 0 if check_frcnn(cfg, len(data["classes"]), w) else 1
        if use_rescbam:
            return 0 if check_rescbam(weights, len(data["classes"])) else 1
        if not use_scc:
            print("this config has scc: false, so there is no swap to check.\nSCC CHECK: PASS")
            return 0
        return 0 if check_scc(weights, len(data["classes"])) else 1

    imgsz = args.imgsz or train_cfg.get("imgsz")
    if not imgsz:
        raise SystemExit("image size is not set: pass --imgsz or set train.imgsz in the config")
    name = utils.run_name(cfg, args.seed) + (f"_smoke{imgsz}" if args.smoke else "")

    if args.auto_resume and not args.resume and not args.smoke:
        from vgra import runs

        prev = runs.find_previous_run(name, args.resume_search)
        if prev is None:
            print(f"auto-resume: no earlier run called {name} under {args.resume_search}; starting a new run")
        else:
            dst = runs.restore_run(prev, out_root / name)
            print(f"auto-resume: found {prev} ({runs.results_rows(dst)} epochs logged), copied to {dst}")
            if runs.is_finished(dst):
                v = runs.read_metrics(dst).get("valid", {})
                print(f"auto-resume: {name} has already finished (valid mAP50 {v.get('map50')}, "
                      f"mAP50-95 {v.get('map50_95')}). Nothing to train.")
                return 0
            ckpt = dst / "weights" / "resume.pt"
            if not ckpt.is_file():
                raise SystemExit(f"auto-resume: {dst} has no weights/resume.pt to continue from")
            args.resume = str(ckpt)
    train_set = train_cfg.get("train_set", "orig")
    index = pairs.read_index(ix.locate_index(data, args.index))
    data_yaml = lists.write_data_files(index, data, out_root / "lists", train_set)
    n_train_full = len(lists.train_image_paths(index, train_set))
    print(f"data file: {data_yaml} | train set: {train_set} ({n_train_full} images) | image size: {imgsz}")

    if not args.no_warm:
        import psutil

        frac = float(args.fraction) if args.smoke else 1.0
        files = lists.files_to_warm(index, train_set, frac, include_test=args.test)
        w = lists.warm_files(files, threads=args.warm_threads)
        mem = psutil.virtual_memory()
        print(f"read {w['files']:,} files ({w['bytes'] / 1e9:.1f} GB) in {w['seconds'] / 60:.1f} min "
              f"= {w['bytes'] / 1e6 / max(w['seconds'], 1e-9):.0f} MB/s, {w['files'] / max(w['seconds'], 1e-9):.0f} files/s, "
              f"errors {w['errors']} | RAM total {mem.total / 1e9:.1f} GB, available {mem.available / 1e9:.1f} GB")

    if kind == "frcnn":
        return train_frcnn(args, cfg, index, out_root / name, name, train_set, n_train_full, imgsz)

    from vgra.model import count_parameters, make_trainer
    from vgra.scc import scc_strengths

    run_dir = out_root / name
    if args.resume:
        overrides = {"resume": args.resume, "device": args.device}
    else:
        overrides = {k: train_cfg[k] for k in PASS_THROUGH if train_cfg.get(k) is not None}
        overrides.update(model=weights, data=str(data_yaml), imgsz=int(imgsz), seed=int(args.seed),
                         project=str(out_root), name=name, exist_ok=True, device=args.device, plots=not args.smoke)
        if args.cache:
            overrides["cache"] = args.cache
        if args.batch:
            overrides["batch"] = int(args.batch)
        if args.smoke:
            overrides.update(epochs=1, fraction=float(args.fraction), patience=0)

    if kind == "rtdetr":
        from vgra.rtdetr import RDTrainer  # stock RT-DETR trainer; validator mask fix only (vgra/rtdetr.py)

        trainer = RDTrainer(overrides=overrides)
    else:
        trainer = make_trainer(use_scc, rescbam=use_rescbam)(overrides=overrides)
    run_dir = Path(trainer.save_dir)
    info_path = utils.write_run_info(run_dir, cfg, args.seed)
    print(f"run folder: {run_dir} | run info: {info_path.name}")

    clock = {"train": [], "val": []}
    strength_csv = run_dir / "scc_strength.csv"

    from vgra.runs import TimeBudget

    max_hours = args.max_hours if args.max_hours is not None else (None if args.smoke else train_cfg.get("max_hours"))
    reserve = args.reserve_min if args.reserve_min is not None else train_cfg.get("reserve_minutes", 20)
    budget = TimeBudget(max_hours, reserve, start=T_START) if max_hours else None
    if budget:
        print(f"time budget: {max_hours} h from the start of this script; the run stops cleanly after the "
              f"last epoch that fits (reserve {reserve} min) and can be continued with --auto-resume")

    def on_model_save(t):
        # last.pt is stripped of its optimizer when training ends; keep a full copy to continue from
        shutil.copy2(t.last, t.wdir / "resume.pt")

    def on_train_start(t):
        clock["setup"] = time.time() - clock["t0"]
        clock["mark"] = time.time()

    def on_train_epoch_start(t):
        clock["t_train"] = time.time()

    def on_train_epoch_end(t):
        clock["train"].append(time.time() - clock["t_train"])

    def on_val_start(v):
        clock["t_val"] = time.time()

    def on_val_end(v):
        if "t_val" in clock:
            clock["val"].append(time.time() - clock["t_val"])

    def on_fit_epoch_end(t):
        now = time.time()
        if budget and not t.stop:
            epoch_s = now - clock.get("mark", now)
            clock["mark"] = now
            if budget.should_stop(epoch_s, now):
                t.stop = True
                clock["time_stop"] = True
                print(f"\ntime budget: stopping after epoch {t.epoch + 1} "
                      f"({budget.elapsed(now) / 3600:.2f} h used, last epoch {epoch_s / 60:.1f} min). "
                      f"Continue in the next version with --auto-resume.")
        model = t.ema.ema if getattr(t, "ema", None) else t.model
        s = scc_strengths(model)
        if s and t.epoch not in clock.setdefault("logged", set()):  # Ultralytics fires this twice at the end
            clock["logged"].add(t.epoch)
            new = not strength_csv.exists()
            with open(strength_csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["epoch"] + list(s))
                w.writerow([t.epoch + 1] + [round(v, 6) for v in s.values()])

    for event, fn in (("on_train_start", on_train_start), ("on_model_save", on_model_save),
                      ("on_train_epoch_start", on_train_epoch_start), ("on_train_epoch_end", on_train_epoch_end),
                      ("on_val_start", on_val_start), ("on_val_end", on_val_end), ("on_fit_epoch_end", on_fit_epoch_end)):
        trainer.add_callback(event, fn)

    t0 = clock["t0"] = time.time()
    trainer.train()
    total_s = time.time() - t0

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    final = best if best.exists() else last
    batch = int(trainer.args.batch)
    device = str(args.device)
    results_csv = run_dir / "results.csv"
    epochs_total = max(len(results_csv.read_text().strip().splitlines()) - 1, 0) if results_csv.exists() else 0
    finished = not clock.get("time_stop", False)
    if finished and not args.smoke:
        (run_dir / "weights" / "resume.pt").unlink(missing_ok=True)  # not needed once the run is complete
    metrics = {
        "run_name": name, "exp_id": cfg["exp_id"], "kind": kind, "seed": int(args.seed), "commit": utils.git_commit(),
        "finished": finished, "stopped_for_time": not finished, "max_hours": max_hours,
        "scc": use_scc, "rescbam": use_rescbam, "imgsz": int(trainer.args.imgsz), "train_set": train_set,
        "epochs_this_session": len(clock["train"]), "epochs_total": epochs_total, "total_seconds": round(total_s),
        "setup_seconds": round(clock.get("setup", 0.0)), "cache": str(trainer.args.cache),
        "train_seconds_per_epoch": round(sum(clock["train"]) / max(len(clock["train"]), 1), 1),
        "parameters": count_parameters(trainer.model), "batch": batch,
        "scc_strength": {k: round(v, 6) for k, v in scc_strengths(
            trainer.ema.ema if getattr(trainer, "ema", None) else trainer.model).items()},
        "weights": final.name,
    }
    # Ultralytics has just validated best.pt at the end of training: reuse those numbers
    # (re-validating would re-scan the whole validation folder, which is slow on Kaggle).
    final_metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    if best.exists() and final_metrics is not None and len(getattr(final_metrics.box, "ap_class_index", [])):
        metrics["valid"] = summarise(final_metrics)
    else:
        metrics["valid"] = evaluate(final, data_yaml, "val", int(trainer.args.imgsz), batch, device, run_dir, kind)
    if args.test:
        metrics["test"] = evaluate(final, data_yaml, "test", int(trainer.args.imgsz), batch, device, run_dir, kind)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    v = metrics["valid"]
    print(f"\n=== {name} ===")
    print(f"valid mAP50 {v['map50']:.4f} | mAP50-95 {v['map50_95']:.4f} | classes evaluated {v['classes_evaluated']}")
    for c, r in v["per_class"].items():
        print(f"  {c:<20} AP50 {r['ap50']:.4f}  AP50-95 {r['ap50_95']:.4f}")
    if metrics["scc_strength"]:
        print("SCC strength tanh(alpha):", {k: f"{s:+.6f}" for k, s in metrics["scc_strength"].items()})
    print(f"wrote {run_dir / 'metrics.json'}")
    print(f"RUN STATUS: {'FINISHED' if finished else 'STOPPED FOR TIME'} after {epochs_total} epochs "
          f"(this session {len(clock['train'])})")

    if args.smoke:
        import torch

        n_seen = len(trainer.train_loader.dataset)
        per_img = sum(clock["train"]) / max(n_seen, 1)
        val_s = clock["val"][0] if clock["val"] else float("nan")
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print("\n=== timing (smoke run) ===")
        n_it = -(-n_seen // batch)
        print(f"setup before the first epoch (model, label scan, image cache): {clock.get('setup', 0.0) / 60:.1f} min")
        print(f"trained on {n_seen} images in {sum(clock['train']):.0f} s = {per_img:.3f} s per image, "
              f"{sum(clock['train']) / max(n_it, 1):.2f} s per batch of {batch} | image cache: {trainer.args.cache}")
        print(f"estimated full epoch on {n_train_full} images: {per_img * n_train_full / 60:.1f} min training "
              f"+ {val_s / 60:.1f} min validation")
        print(f"peak GPU memory: {mem:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
