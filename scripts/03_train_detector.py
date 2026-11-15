"""STEP-03: train a single-view detector (B0 = YOLO11s, B1 = YOLO11s + SCC).

Check the SCC swap (no training):
    python scripts/03_train_detector.py --config configs/B1.yaml --check

Short timing run (1 epoch on a fraction of the training images):
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --smoke --imgsz 640

Full run:
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0

Continue a run that was cut off:
    python scripts/03_train_detector.py --config configs/B1.yaml --seed 0 --resume <path to last.pt>

One run uses one GPU (--device). Outputs go to <out_dir>/<run name>/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# train arguments that are passed from the config to Ultralytics as they are
PASS_THROUGH = ("epochs", "batch", "patience", "optimizer", "lr0", "cos_lr", "workers", "close_mosaic",
                "warmup_epochs", "amp")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--imgsz", type=int, default=None, help="override train.imgsz")
    ap.add_argument("--check", action="store_true", help="check the SCC swap and exit")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a fraction of the training images")
    ap.add_argument("--fraction", type=float, default=0.1, help="share of training images in a smoke run")
    ap.add_argument("--resume", default=None, help="path to last.pt of the run to continue")
    ap.add_argument("--test", action="store_true", help="also evaluate on the test split (final runs only)")
    ap.add_argument("--index", default=None, help="path to index.csv")
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--weights", default=None, help="override model.weights")
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


def evaluate(weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: str, save_dir: Path) -> dict:
    """Validate saved weights on one split."""
    from ultralytics import YOLO

    return summarise(YOLO(str(weights)).val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False,
        project=str(save_dir), name=f"eval_{split}", exist_ok=True))


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
    print(f"repo commit: {utils.git_commit()} | config: {cfg['exp_id']} | scc: {use_scc}")

    if args.check:
        if not use_scc:
            print("this config has scc: false, so there is no swap to check.\nSCC CHECK: PASS")
            return 0
        return 0 if check_scc(weights, len(data["classes"])) else 1

    imgsz = args.imgsz or train_cfg.get("imgsz")
    if not imgsz:
        raise SystemExit("image size is not set: pass --imgsz or set train.imgsz in the config")
    train_set = train_cfg.get("train_set", "orig")
    index = pairs.read_index(ix.locate_index(data, args.index))
    data_yaml = lists.write_data_files(index, data, out_root / "lists", train_set)
    n_train_full = len(lists.train_image_paths(index, train_set))
    print(f"data file: {data_yaml} | train set: {train_set} ({n_train_full} images) | image size: {imgsz}")

    from vgra.model import count_parameters, make_trainer
    from vgra.scc import scc_strengths

    name = utils.run_name(cfg, args.seed) + (f"_smoke{imgsz}" if args.smoke else "")
    run_dir = out_root / name
    if args.resume:
        overrides = {"resume": args.resume, "device": args.device}
    else:
        overrides = {k: train_cfg[k] for k in PASS_THROUGH if train_cfg.get(k) is not None}
        overrides.update(model=weights, data=str(data_yaml), imgsz=int(imgsz), seed=int(args.seed),
                         project=str(out_root), name=name, exist_ok=True, device=args.device, plots=not args.smoke)
        if args.smoke:
            overrides.update(epochs=1, fraction=float(args.fraction), patience=0)

    trainer = make_trainer(use_scc)(overrides=overrides)
    run_dir = Path(trainer.save_dir)
    info_path = utils.write_run_info(run_dir, cfg, args.seed)
    print(f"run folder: {run_dir} | run info: {info_path.name}")

    clock = {"train": [], "val": []}
    strength_csv = run_dir / "scc_strength.csv"

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

    for event, fn in (("on_train_epoch_start", on_train_epoch_start), ("on_train_epoch_end", on_train_epoch_end),
                      ("on_val_start", on_val_start), ("on_val_end", on_val_end), ("on_fit_epoch_end", on_fit_epoch_end)):
        trainer.add_callback(event, fn)

    t0 = time.time()
    trainer.train()
    total_s = time.time() - t0

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    final = best if best.exists() else last
    batch = int(trainer.args.batch)
    device = str(args.device)
    results_csv = run_dir / "results.csv"
    epochs_total = max(len(results_csv.read_text().strip().splitlines()) - 1, 0) if results_csv.exists() else 0
    metrics = {
        "run_name": name, "exp_id": cfg["exp_id"], "seed": int(args.seed), "commit": utils.git_commit(),
        "scc": use_scc, "imgsz": int(trainer.args.imgsz), "train_set": train_set,
        "epochs_this_session": len(clock["train"]), "epochs_total": epochs_total, "total_seconds": round(total_s),
        "train_seconds_per_epoch": round(sum(clock["train"]) / max(len(clock["train"]), 1), 1),
        "parameters": count_parameters(trainer.model),
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
        metrics["valid"] = evaluate(final, data_yaml, "val", int(trainer.args.imgsz), batch, device, run_dir)
    if args.test:
        metrics["test"] = evaluate(final, data_yaml, "test", int(trainer.args.imgsz), batch, device, run_dir)
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

    if args.smoke:
        import torch

        n_seen = len(trainer.train_loader.dataset)
        per_img = sum(clock["train"]) / max(n_seen, 1)
        val_s = clock["val"][0] if clock["val"] else float("nan")
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print("\n=== timing (smoke run) ===")
        print(f"trained on {n_seen} images in {sum(clock['train']):.0f} s = {per_img:.3f} s per image (batch {batch})")
        print(f"estimated full epoch on {n_train_full} images: {per_img * n_train_full / 60:.1f} min training "
              f"+ {val_s / 60:.1f} min validation")
        print(f"peak GPU memory: {mem:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
