"""I1: run trained detectors on every image of one split and save their answers (no statistics here).

Part A (docs/PROTOCOL.md section 4) needs, for each detector, the per-image class scores on valid
(thresholds, fusion fitting) and on test (final numbers). This script produces them; the patient-level
analysis (scripts/12_study_level.py) then runs on CPU from these files.

Run on Kaggle (GPU; images, vgra-index and the datasets holding the trained runs attached):
    python scripts/10_predict.py --runs B0_s0 B1_s0 --split valid
The test split is refused unless --open-test is given (protocol section 2: opened once, after every
model and choice is fixed). Opening it writes TEST_OPENED.json with the time, commit and runs:
    python scripts/10_predict.py --runs B0_s0 B0_s1 B0_s2 B1_s0 V8_s0 V8_s1 --split test --open-test

Settings follow Ultralytics validation and the protocol: square letterbox at the training image size,
NMS conf 0.001, IoU 0.7, max 300 boxes, multi-label (vgra/infer.py). The detector family and image size of a run
come from its config (configs/<run prefix>.yaml, model.kind): RT-DETR as the Ultralytics RT-DETR validator
(vgra/infer.py predict_rtdetr), Faster R-CNN as vgra/frcnn.py (protocol amendment v1.2, section 11.3).

Writes to <out_dir>/predictions/:
    pairs_<split>.csv                 the PA + LAT pairs of the split with subgroup columns
                                      (age, age band, torus, initial exam, cast)
    <run>/<split>_images.csv.gz       one row per image and class: own-image max score, label present, boxes
    <run>/<split>_detections.csv.gz   one row per detection after NMS, with tp at IoU 0.50 ... 0.95
    <run>/<split>_info.json           weights, commit, settings, per-class AP50 / AP50-95, timing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IOUS = [f"tp{int(round(100 * t))}" for t in [0.5 + 0.05 * i for i in range(10)]]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--runs", nargs="+", required=True, help="run names searched under --search, e.g. B0_s0")
    ap.add_argument("--weights", nargs="+", default=None, help="explicit .pt files instead (same order as --runs)")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--open-test", action="store_true", help="required to run on the test split")
    ap.add_argument("--imgsz", type=int, default=None, help="default: train.imgsz of the config")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def weights_for(run: str, search: str) -> Path:
    from vgra.runs import find_previous_run

    folder = find_previous_run(run, search)
    if folder is None:
        raise FileNotFoundError(f"no run folder {run!r} under {search}; attach the dataset that holds it")
    best = folder / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"{folder} has no weights/best.pt (is the run finished?)")
    return best


def run_config(run: str) -> dict:
    """Config of a run from its name (<exp_id>_s<seed>), e.g. RD_s0 -> configs/RD.yaml."""
    from vgra import utils

    exp_id = run.rsplit("_s", 1)[0]
    path = REPO_ROOT / "configs" / f"{exp_id}.yaml"
    if not path.is_file():
        return utils.load_config(REPO_ROOT / "configs" / "base.yaml")  # unknown prefix (tests): base settings
    return utils.load_config(path)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.open_test:
        raise SystemExit("the test split is opened once, at the end (protocol section 2); add --open-test then")
    if args.weights and len(args.weights) != len(args.runs):
        raise SystemExit("--weights needs one file per name in --runs")

    import numpy as np
    import pandas as pd
    import torch
    from ultralytics.nn.tasks import load_checkpoint

    import vgra.rescbam  # noqa: F401  (needed to load V8R checkpoints)
    import vgra.scc  # noqa: F401  (needed to load SCC checkpoints)
    from vgra import index as ix
    from vgra import infer, lists, pairs as pr, premise as pm, study as st, utils

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    classes = list(data["classes"])
    run_cfgs = {run: run_config(run) for run in args.runs}
    kinds = {run: c["model"].get("kind", "yolo") for run, c in run_cfgs.items()}
    sizes = {run: int(args.imgsz or c["train"]["imgsz"]) for run, c in run_cfgs.items()}
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out or cfg["out_dir"]) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.limit) or any(sizes[r] != int(run_cfgs[r]["train"]["imgsz"]) for r in args.runs)
    commit = utils.git_commit()
    print(f"repo commit: {commit} | split: {args.split} | device {dev} | "
          + ", ".join(f"{r}: {kinds[r]} {sizes[r]}" for r in args.runs)
          + (" | SMOKE TEST (not for the paper)" if smoke else ""))

    if args.split == "test":
        record = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
                  "runs": args.runs, "smoke": smoke, "kinds": kinds}
        (out / "TEST_OPENED.json").write_text(json.dumps(record, indent=2))
        print(f"TEST SPLIT OPENED: {record}")

    # images, pairs and subgroup columns
    index = pr.read_index(ix.locate_index(data, args.index))
    info = infer.image_info(index)
    rows = index[(index["split"] == args.split) & index["is_base"] & index["in_metadata"]].sort_values("image_id")
    if args.limit:
        rows = rows.head(args.limit)
    paths = list(rows["path"])
    meta = pd.read_csv(Path(data["root"]) / data["meta_csv"], encoding="utf-8-sig")
    pairs = pr.build_pairs(index, classes)
    pairs = pairs[(pairs["split"] == args.split) & pairs["ap_image_id"].isin(rows["image_id"])
                  & pairs["lat_image_id"].isin(rows["image_id"])]
    pairs = st.attach_metadata(pairs.reset_index(drop=True), meta)
    pairs.to_csv(out / f"pairs_{args.split}.csv", index=False)
    print(f"{len(paths)} {args.split} images | {len(pairs)} PA + LAT pairs | "
          f"torus {int(pairs['torus'].sum())}, initial exam {int(pairs['initial_exam'].sum())}, "
          f"cast {int(pairs['cast'].sum())}")
    if not args.no_warm:
        w = lists.warm_files(list(rows["path"]) + list(rows["label_path"]))
        print(f"read {w['files']:,} files ({w['bytes'] / 1e9:.1f} GB) in {w['seconds'] / 60:.1f} min")
    datasets = {}

    def dataset(kind: str, size: int):
        """One dataset per detector family and image size, built when first needed."""
        if (kind, size) not in datasets:
            if kind == "frcnn":
                from vgra import frcnn

                datasets[(kind, size)] = frcnn.FrcnnDataset(paths, list(rows["label_path"]), size)
            elif kind == "rtdetr":
                datasets[(kind, size)] = infer.rtdetr_dataset(paths, classes, size,
                                                              out / "lists" / f"{args.split}_rtdetr{size}.txt")
            else:
                datasets[(kind, size)] = infer.image_dataset(paths, classes, size, out / "lists" / f"{args.split}.txt",
                                                             augment=False)
        return datasets[(kind, size)]

    for k, run in enumerate(args.runs):
        t0 = time.time()
        kind, imgsz = kinds[run], sizes[run]
        weights = Path(args.weights[k]) if args.weights else weights_for(run, args.search)
        ds = dataset(kind, imgsz)
        if kind == "frcnn":
            from vgra import frcnn

            model, _ = frcnn.load_model(weights, dev)
            loader = frcnn.make_loader(ds, args.batch, shuffle=False, workers=args.workers)
            dets, imgs, tp = frcnn.predict(model, loader, classes, info, dev, amp=dev.type == "cuda")
        else:
            model, _ = load_checkpoint(str(weights), device="cpu")
            model = model.to(dev).eval()
            model = model.half() if dev.type == "cuda" else model.float()
            loader = infer.image_loader(ds, args.batch, workers=args.workers)
            if kind == "rtdetr":
                dets, imgs, tp = infer.predict_rtdetr(model, loader, classes, imgsz, info, args.conf, device=dev)
            else:
                dets, imgs, tp = infer.predict(model, loader, classes, imgsz, info, args.conf, args.iou, args.max_det,
                                               device=dev)
        for j, name in enumerate(IOUS):
            dets[name] = tp[:, j] if len(tp) else []
        folder = out / run
        folder.mkdir(parents=True, exist_ok=True)
        imgs.to_csv(folder / f"{args.split}_images.csv.gz", index=False)
        dets.to_csv(folder / f"{args.split}_detections.csv.gz", index=False)

        per_class = {}
        for c, name in enumerate(classes):
            sel = (dets["cls"] == c).values
            n_t = int(imgs.loc[imgs["cls"] == c, "n_boxes"].sum())
            if n_t:
                a50, a = pm.ap_of(tp[sel], dets.loc[sel, "conf"].values, n_t)
                per_class[name] = {"boxes": n_t, "ap50": round(a50, 4), "ap50_95": round(a, 4)}
        done = {k2: v for k2, v in per_class.items()}
        m50 = float(np.mean([v["ap50"] for v in done.values()])) if done else float("nan")
        m = float(np.mean([v["ap50_95"] for v in done.values()])) if done else float("nan")
        run_info = {"run": run, "kind": kind, "weights": str(weights), "split": args.split, "commit": commit,
                    "smoke": smoke, "imgsz": imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                    "images": len(paths), "pairs": len(pairs), "detections": len(dets),
                    "map50": round(m50, 4), "map50_95": round(m, 4), "per_class": per_class,
                    "seconds": round(time.time() - t0, 1)}
        (folder / f"{args.split}_info.json").write_text(json.dumps(run_info, indent=2))
        print(f"{run}: {len(dets):,} detections | mAP50 {m50:.4f} | mAP50-95 {m:.4f} "
              f"({len(done)} classes with labels) | {time.time() - t0:.0f} s | {weights}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"PREDICTIONS DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
