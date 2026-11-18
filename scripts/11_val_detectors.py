"""I4: standard Ultralytics validation of every detector on one split (detector table T2).

Protocol section 8: detector metrics on the full test split with the standard Ultralytics validator
(the same split as Ju et al.), so that the numbers are comparable with the literature table.

Run on Kaggle (GPU; images, vgra-index and the datasets holding the trained runs attached):
    python scripts/11_val_detectors.py --runs B0_s0 B1_s0 --split valid
    python scripts/11_val_detectors.py --runs B0_s0 B0_s1 B0_s2 B1_s0 V8_s0 V8_s1 V8R_s0 V8R_s1 --split test --open-test

Ultralytics validation settings: image size of the config (1024), batch 16, conf 0.001, IoU 0.7, max 300
boxes, rectangular batches (Ultralytics default for validation).

Writes to <out_dir>/detectors_<split>/:
    detectors.csv     one row per run: parameters, mAP50, mAP50-95, precision, recall
    per_class.csv     one row per run and class: AP50, AP50-95, precision, recall
    <run>.json        the same for one run, with weights path and commit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", default=None, help="explicit .pt files instead (same order as --runs)")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--open-test", action="store_true", help="required to run on the test split")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.open_test:
        raise SystemExit("the test split is opened once, at the end (protocol section 2); add --open-test then")
    if args.weights and len(args.weights) != len(args.runs):
        raise SystemExit("--weights needs one file per name in --runs")

    import pandas as pd
    from ultralytics import YOLO

    import vgra.rescbam  # noqa: F401  (needed to load V8R checkpoints)
    import vgra.scc  # noqa: F401  (needed to load SCC checkpoints)
    from vgra import index as ix
    from vgra import lists, pairs as pr, utils
    from vgra.model import count_parameters

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    train = __import__("03_train_detector")  # summarise() is shared with the training script
    predict = __import__("10_predict")  # weights_for() is shared with I1

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    imgsz = int(args.imgsz or cfg["train"]["imgsz"])
    out = Path(args.out or cfg["out_dir"]) / f"detectors_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    commit = utils.git_commit()
    smoke = imgsz != int(cfg["train"]["imgsz"])
    print(f"repo commit: {commit} | split {args.split} | image size {imgsz}" + (" | SMOKE TEST" if smoke else ""))
    if args.split == "test":
        record = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
                  "runs": args.runs, "script": "11_val_detectors.py", "smoke": smoke}
        (out / "TEST_OPENED.json").write_text(json.dumps(record, indent=2))
        print(f"TEST SPLIT OPENED: {record}")

    index = pr.read_index(ix.locate_index(data, args.index))
    data_yaml = lists.write_data_files(index, data, out / "lists", "orig")
    split = "val" if args.split == "valid" else "test"

    overall, per_class = [], []
    for k, run in enumerate(args.runs):
        weights = Path(args.weights[k]) if args.weights else predict.weights_for(run, args.search)
        model = YOLO(str(weights))
        n_params = count_parameters(model.model)
        m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=args.batch, device=args.device,
                      plots=False, verbose=False, project=str(out), name=f"val_{run}", exist_ok=True)
        s = train.summarise(m)
        rec = {"run": run, "detector": run.rsplit("_s", 1)[0], "split": args.split, "weights": str(weights),
               "commit": commit, "smoke": smoke, "imgsz": imgsz, "parameters": n_params, **s}
        (out / f"{run}.json").write_text(json.dumps(rec, indent=2))
        overall.append({k2: rec[k2] for k2 in ("run", "detector", "parameters", "map50", "map50_95", "precision",
                                               "recall", "classes_evaluated")})
        for c, r in s["per_class"].items():
            per_class.append({"run": run, "detector": rec["detector"], "class": c, **r})
        print(f"{run}: mAP50 {s['map50']:.4f} | mAP50-95 {s['map50_95']:.4f} | {n_params:,} parameters | {weights}")

    overall = pd.DataFrame(overall)
    overall.to_csv(out / "detectors.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    if len(overall) > 1:
        agg = overall.groupby("detector")[["map50", "map50_95"]].agg(["mean", "std", "count"])
        print("\nmean and SD over seeds:")
        print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"DETECTOR VALIDATION DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
