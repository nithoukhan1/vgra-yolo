"""STEP-04: premise test (Gate 2). Inference only.

Does the partner view carry class evidence that the detector's own score does not?
Runs trained single-view detectors on the validation pairs and, per class, compares a small
model that uses the detector's own score (+ view) with one that also sees the partner image's
score of the same class. See `vgra/premise.py` for the method.

Run on Kaggle (GPU; dataset + index + the dataset or output that holds the trained runs attached):
    python scripts/04_premise_test.py --config configs/base.yaml --runs B1_s0 B0_s0

Writes to <out_dir>/premise/:
    premise.csv              one row per run, level and class (AUCs, gain, 95% interval, AP before/after)
    detections_<run>.csv.gz  every detection with its own and partner scores (for later analysis)
    images_<run>.csv.gz      every image and class with own / partner max scores and the label
Ends with 'GATE 2: PASS' or 'GATE 2: FAIL' for the first run (B1 by default).

Gate 2 rule (fixed before any result was seen, docs/DECISIONS.md): PASS if at least 2 gateable
classes have a detection-level gain whose 95% interval lies above 0 and whose partner rescoring
raises AP50 over the view-only rescoring.

I3, RQ1 confirmatory (docs/PROTOCOL.md 4.3 item 5): the same test and rule on the test pairs, for every
detector, with 2,000 bootstrap resamples (protocol section 7). Needs --open-test:
    python scripts/04_premise_test.py --runs B0_s0 B0_s1 B0_s2 B1_s0 V8_s0 V8_s1 --split test --open-test
Writes to <out_dir>/premise_test/ and ends with 'RQ1 (test): ...' per run instead of a gate decision.

From the I1 tables instead of running the detectors (any detector family; protocol amendment v1.2, 11.3):
    python scripts/04_premise_test.py --runs RD_s0 FR_s0 --split test --open-test --from-pred <predictions>
The per-image class scores, detections and true-positive flags of scripts/10_predict.py are the same numbers this
script computes itself (2026-11-23: the Kaggle dry-run outputs of B0_s0, B1_s0 and V8R_s0 on 64 validation pairs were
reproduced exactly, tables and results), so both routes give the same analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_CLASSES = 2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--runs", nargs="+", default=["B1_s0"], help="run names searched under --search")
    ap.add_argument("--weights", nargs="+", default=None, help="explicit .pt files instead of --runs (same order)")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--split", default="valid", choices=["valid", "test"],
                    help="valid = Gate 2 (decided there); test = RQ1 confirmatory, needs --open-test")
    ap.add_argument("--open-test", action="store_true", help="required to run on the test split")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=8, help="pairs per batch")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=None, help="default 500 on valid (Gate 2), 2000 on test (protocol 7)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (smoke test)")
    ap.add_argument("--from-pred", default=None,
                    help="predictions folder of scripts/10_predict.py: build the inputs from its tables")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def weights_for(run: str, search: str) -> Path:
    from vgra.runs import find_previous_run

    folder = find_previous_run(run, search)
    if folder is None:
        raise FileNotFoundError(f"no run folder {run!r} under {search}")
    for name in ("best.pt", "last.pt"):
        if (folder / "weights" / name).is_file():
            return folder / "weights" / name
    raise FileNotFoundError(f"no best.pt or last.pt in {folder / 'weights'}")


def infer(model, loader, pairs_df, classes, imgsz, dev, args, run):
    """Detections and per-image max scores for every pair image."""
    import numpy as np
    import pandas as pd
    import torch
    from ultralytics.utils import nms
    from ultralytics.utils.ops import xywh2xyxy

    from vgra import premise as pm

    patient = dict(zip(pairs_df["pair_id"], pairs_df["patient_id"]))
    nc = len(classes)
    det_rows, img_rows, tp_tables = [], [], []
    half = dev.type == "cuda"
    for batch in loader:
        x = batch["img"].to(dev).float() / 255
        with torch.no_grad():
            y = model(x.half() if half else x)[0].float()  # (N, 4 + nc, anchors), scores after sigmoid
        maxs = y[:, 4:, :].amax(-1).cpu().numpy()  # (N, nc) highest score of each class in each image
        out = nms.non_max_suppression(y, args.conf, args.iou, multi_label=True, max_det=args.max_det)
        n = batch["n_pairs"]
        for j in range(2 * n):
            i_pair = j % n
            pid = batch["pair_id"][i_pair]
            view = 0 if j < n else 1
            k = j + n if j < n else j - n  # partner image
            sel = batch["batch_idx"] == j
            gt_cls = batch["cls"][sel].view(-1).long()
            gt_xyxy = xywh2xyxy(batch["bboxes"][sel]) * imgsz
            d = out[j].cpu()
            tp = pm.match(d[:, :4], d[:, 5].long(), gt_xyxy, gt_cls)
            tp_tables.append(tp)
            present = np.zeros(nc, dtype=int)
            present[gt_cls.numpy()] = 1
            n_boxes = np.bincount(gt_cls.numpy(), minlength=nc)
            for c in range(nc):
                img_rows.append((run, pid, patient[pid], view, c, float(maxs[j, c]), float(maxs[k, c]),
                                 int(present[c]), int(n_boxes[c])))
            for r in range(d.shape[0]):
                c = int(d[r, 5])
                det_rows.append((run, pid, patient[pid], view, c, float(d[r, 4]), float(maxs[j, c]),
                                 float(maxs[k, c]), bool(tp[r, 0])))
    dets = pd.DataFrame(det_rows, columns=["run", "pair_id", "patient_id", "view", "cls", "conf", "own_max",
                                           "partner_max", "tp50"])
    imgs = pd.DataFrame(img_rows, columns=["run", "pair_id", "patient_id", "view", "cls", "own_max",
                                           "partner_max", "present", "n_boxes"])
    tp_all = np.concatenate(tp_tables) if tp_tables else np.zeros((0, 10), dtype=bool)
    return dets, imgs, tp_all


IOUS = [f"tp{int(round(100 * t))}" for t in [0.5 + 0.05 * i for i in range(10)]]


def tables_from_predictions(pred: Path, run: str, split: str, limit: int = 0, pairs_per_batch: int = 8):
    """The inputs of `analyse` from the I1 tables: every image of a PA + LAT pair with its own and its partner's
    highest score per class, its detections and their true-positive flags at IoU 0.50:0.95. Rows come in the order
    `infer` produces them (pairs by pair_id in batches of `pairs_per_batch`: the PA images of a batch, then their LAT
    partners; detections of an image in NMS order), because ties in the AP and bootstrap code depend on row order."""
    import pandas as pd

    pairs = pd.read_csv(pred / f"pairs_{split}.csv", dtype={"pair_id": str, "laterality": str}).sort_values("pair_id")
    if limit:
        pairs = pairs.head(limit)
    pairs = pairs.reset_index(drop=True)
    pairs["chunk"], pairs["slot"] = pairs.index // int(pairs_per_batch), pairs.index % int(pairs_per_batch)
    side = pd.concat([pairs.assign(image_id=pairs["ap_image_id"], partner_id=pairs["lat_image_id"], view=0),
                      pairs.assign(image_id=pairs["lat_image_id"], partner_id=pairs["ap_image_id"], view=1)])
    side = side[["image_id", "partner_id", "pair_id", "patient_id", "view", "chunk", "slot"]]
    # exact float parsing: the default parser can change the last digit, which re-orders tied scores
    imgs = pd.read_csv(pred / run / f"{split}_images.csv.gz", float_precision="round_trip")
    dets = pd.read_csv(pred / run / f"{split}_detections.csv.gz", float_precision="round_trip")
    dets["_row"] = range(len(dets))
    own = imgs.set_index(["image_id", "cls"])["own_max"]
    im = side.merge(imgs[["image_id", "cls", "own_max", "present", "n_boxes"]], on="image_id", how="inner")
    if len(im) != 2 * len(pairs) * imgs["cls"].nunique():
        raise ValueError(f"{run}: the I1 tables do not cover every image of the {split} pairs")
    im["partner_max"] = own.reindex(pd.MultiIndex.from_arrays([im["partner_id"], im["cls"]])).to_numpy()
    im = im.sort_values(["chunk", "view", "slot", "cls"]).reset_index(drop=True)
    de = side.merge(dets, on="image_id", how="inner", suffixes=("", "_det"))
    de["partner_max"] = own.reindex(pd.MultiIndex.from_arrays([de["partner_id"], de["cls"]])).to_numpy()
    de = de.sort_values(["chunk", "view", "slot", "_row"]).reset_index(drop=True)
    tp_all = de[IOUS].to_numpy(dtype=bool)
    de["run"], im["run"] = run, run
    dets_out = de[["run", "pair_id", "patient_id", "view", "cls", "conf", "own_max", "partner_max", "tp50"]].copy()
    imgs_out = im[["run", "pair_id", "patient_id", "view", "cls", "own_max", "partner_max", "present", "n_boxes"]].copy()
    return dets_out, imgs_out, tp_all, len(pairs)


def analyse(dets, imgs, tp_all, classes, gate_idx, args):
    """One row per level and class."""
    import numpy as np

    from vgra import premise as pm

    rows = []
    for c, name in enumerate(classes):
        sel = (dets["cls"] == c).to_numpy()
        d = dets[sel]
        n_targets = int(imgs.loc[imgs["cls"] == c, "n_boxes"].sum())
        r = pm.compare(d["tp50"].to_numpy(), d["conf"].to_numpy(), d["view"].to_numpy(), d["partner_max"].to_numpy(),
                       d["patient_id"].to_numpy(), args.folds, args.boot)
        tp = tp_all[sel]
        ap = {"ap50_own": np.nan, "ap_own": np.nan, "ap50_base": np.nan, "ap_base": np.nan,
              "ap50_partner": np.nan, "ap_partner": np.nan}
        ap["ap50_own"], ap["ap_own"] = pm.ap_of(tp, d["conf"].to_numpy(), n_targets)
        if r["p_base"] is not None:
            ap["ap50_base"], ap["ap_base"] = pm.ap_of(tp, r["p_base"], n_targets)
            ap["ap50_partner"], ap["ap_partner"] = pm.ap_of(tp, r["p_partner"], n_targets)
        rows.append({"level": "detection", "cls": name, "gateable": c in gate_idx, "n_targets": n_targets,
                     **{k: v for k, v in r.items() if not k.startswith("p_")}, **ap})

        g = imgs[imgs["cls"] == c]
        r = pm.compare(g["present"].to_numpy(), g["own_max"].to_numpy(), g["view"].to_numpy(),
                       g["partner_max"].to_numpy(), g["patient_id"].to_numpy(), args.folds, args.boot)
        rows.append({"level": "image", "cls": name, "gateable": c in gate_idx, "n_targets": n_targets,
                     **{k: v for k, v in r.items() if not k.startswith("p_")}})
    return rows


def gate(rows) -> tuple[bool, list]:
    """Gate 2: gateable classes whose detection-level gain interval is above 0 and whose partner
    rescoring beats the view-only rescoring in AP50."""
    good = [r["cls"] for r in rows if r["level"] == "detection" and r["gateable"]
            and r["gain_lo"] == r["gain_lo"] and r["gain_lo"] > 0 and r["ap50_partner"] > r["ap50_base"]]
    return len(good) >= MIN_CLASSES, good


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.open_test:
        raise SystemExit("the test split is opened once, at the end (protocol section 2); add --open-test then")
    if args.boot is None:
        args.boot = 2000 if args.split == "test" else 500
    import json
    from datetime import datetime, timezone

    import pandas as pd
    import torch
    from ultralytics.nn.tasks import load_checkpoint

    import vgra.rescbam  # noqa: F401  (ResBlockCBAM must be importable to load V8R)
    import vgra.scc  # noqa: F401  (C3K2_SCC must be importable to load B1)
    from vgra import data as vd
    from vgra import index as ix
    from vgra import modules as vm
    from vgra import pairs as pr
    from vgra import utils

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    classes = list(data["classes"])
    gate_idx = vm.gate_indices(cfg)
    imgsz = int(args.imgsz or cfg["train"]["imgsz"])
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out or cfg["out_dir"]) / ("premise" if args.split == "valid" else "premise_test")
    out.mkdir(parents=True, exist_ok=True)
    print(f"repo commit: {utils.git_commit()} | split {args.split} | image size {imgsz} | device {dev} | "
          f"bootstrap {args.boot}")
    if args.split == "test":
        record = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "commit": utils.git_commit(),
                  "script": "04_premise_test.py" + (f" --from-pred {args.from_pred}" if args.from_pred else ""),
                  "runs": [Path(w).parent.parent.name for w in args.weights] if args.weights else args.runs,
                  "smoke": bool(args.limit) or imgsz != int(cfg["train"]["imgsz"])}
        (out / "TEST_OPENED.json").write_text(json.dumps(record, indent=2))
        print(f"TEST SPLIT OPENED: {record}")

    runs = args.runs if not args.weights else [Path(w).parent.parent.name for w in args.weights]
    if args.from_pred:
        pred = Path(args.from_pred)
        print(f"inputs from the I1 tables in {pred} (no detector is run here)")
    else:
        index = pr.read_index(ix.locate_index(data, args.index))
        pairs = pr.build_pairs(index, classes)
        if args.limit:
            keep = pairs[pairs["split"] == args.split].sort_values("pair_id").head(args.limit)["pair_id"]
            pairs = pairs[pairs["pair_id"].isin(keep)]
        ds = vd.PairDataset(pairs, index, args.split, classes, imgsz, augment=False, list_dir=out / "lists")
        loader = vd.build_pair_loader(ds, args.pairs, shuffle=False, workers=args.workers)
        print(f"{len(ds)} {args.split} pairs ({2 * len(ds)} images)")

    all_rows, verdict = [], None
    for i, run in enumerate(runs):
        t0 = time.time()
        if args.from_pred:
            w = pred / run
            dets, imgs, tp_all, n_pairs = tables_from_predictions(pred, run, args.split, args.limit, args.pairs)
            print(f"\n{run}: I1 tables {w} | {n_pairs} pairs | {len(dets):,} detections")
            model = None
        else:
            w = Path(args.weights[i]) if args.weights else weights_for(run, args.search)
            model, _ = load_checkpoint(str(w), device="cpu")
            model = model.to(dev).eval()
            model = model.half() if dev.type == "cuda" else model.float()
            dets, imgs, tp_all = infer(model, loader, ds.pairs, classes, imgsz, dev, args, run)
            print(f"\n{run}: {w} | {len(dets):,} detections in {time.time() - t0:.0f} s")
        dets.to_csv(out / f"detections_{run}.csv.gz", index=False)
        imgs.to_csv(out / f"images_{run}.csv.gz", index=False)
        rows = analyse(dets, imgs, tp_all, classes, gate_idx, args)
        for r in rows:
            r["run"] = run
        all_rows += rows

        table = pd.DataFrame(rows)
        show = table[table["level"] == "detection"][["cls", "gateable", "n_targets", "n", "n_pos", "auc_base",
                                                      "auc_partner", "gain", "gain_lo", "gain_hi", "ap50_own",
                                                      "ap50_base", "ap50_partner"]]
        print("detection level (target: true positive at IoU 0.5)")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        show = table[table["level"] == "image"][["cls", "gateable", "n", "n_pos", "auc_own", "auc_base",
                                                  "auc_partner", "gain", "gain_lo", "gain_hi"]]
        print("image level (target: class annotated in the image)")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        det = table[(table["level"] == "detection") & (table["n_targets"] > 0)]
        print(f"[sanity] mean AP50 over {len(det)} classes with labels: {det['ap50_own'].mean():.3f} "
              f"(validation pairs only, square letterbox; should be close to the run's validation mAP50)")
        ok, good = gate(rows)
        print(f"{run}: gateable classes with a clear partner gain: {', '.join(good) or 'none'}")
        if args.split == "test":
            print(f"RQ1 (test), {run}: partner gain in {len(good)} gateable classes (Gate 2 rule needs {MIN_CLASSES}) "
                  f"-> {'partner HELPS box detection' if ok else 'no partner gain (as expected from Gate 2)'}")
        if verdict is None:
            verdict = (run, ok, good)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    cols = ["run", "level", "cls", "gateable", "n_targets", "n", "n_pos", "auc_own", "auc_base", "auc_partner",
            "gain", "gain_lo", "gain_hi", "ap50_own", "ap50_base", "ap50_partner", "ap_own", "ap_base", "ap_partner",
            "note"]
    pd.DataFrame(all_rows).reindex(columns=cols).to_csv(out / "premise.csv", index=False)
    print(f"\n[info] tables: {out}")
    run, ok, good = verdict
    if args.split == "valid":
        print(f"GATE 2: {'PASS' if ok else 'FAIL'} ({run}: {len(good)} gateable classes with a clear gain; "
              f"rule needs {MIN_CLASSES})")
    else:
        print("RQ1 CONFIRMATORY DONE (test pairs): see the per-run lines above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
