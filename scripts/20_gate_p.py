"""Gate P (docs/PROTOCOL.md section 5): does a projection-aware presence branch add information
beyond the detector itself? Decided on valid only.

Run on Kaggle (GPU; images, vgra-index and the dataset holding runs/B0_s0 attached):
    python scripts/20_gate_p.py --config configs/base.yaml
Smoke run (tiny data, CPU):
    python scripts/20_gate_p.py --weights <best.pt> --root <tiny> --index <index.csv> --imgsz 128 --epochs 1 --device cpu --boot 50

Steps
1. Load the frozen B0_s0 detector (best.pt).
2. Train the presence branch on the train-split originals with the fixed settings (5 epochs).
3. Run detector + branch on all valid images; compare base vs new rescoring per class.
4. Print the table and 'GATE P: PASS' or 'GATE P: FAIL'.
Outputs in <out_dir>/gate_p/: gate_p.csv, detections.csv.gz, images.csv.gz, presence_branch.pt, run_info.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed by the protocol (section 5); change only for smoke tests, never for the real run.
PROTOCOL = {"epochs": 5, "batch": 16, "imgsz": 1024, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 200,
            "pos_weight_cap": 50.0, "d": 64, "seed": 0, "folds": 5, "boot": 500}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--run", default="B0_s0", help="detector run searched under --search")
    ap.add_argument("--weights", default=None, help="explicit detector .pt instead of --run")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--epochs", type=int, default=PROTOCOL["epochs"])
    ap.add_argument("--imgsz", type=int, default=PROTOCOL["imgsz"])
    ap.add_argument("--batch", type=int, default=PROTOCOL["batch"])
    ap.add_argument("--boot", type=int, default=PROTOCOL["boot"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training images (smoke only)")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def find_detector(args) -> Path:
    from vgra.runs import find_previous_run

    if args.weights:
        return Path(args.weights)
    folder = find_previous_run(args.run, args.search)
    if folder is None or not (folder / "weights" / "best.pt").is_file():
        raise FileNotFoundError(f"no runs/{args.run}/weights/best.pt under {args.search}; attach the detector dataset")
    return folder / "weights" / "best.pt"


def main() -> int:
    args = parse_args()
    import pandas as pd
    import torch
    from ultralytics.nn.tasks import load_checkpoint

    import vgra.scc  # noqa: F401  (needed to load SCC checkpoints)
    from vgra import gate_p as gp
    from vgra import index as ix
    from vgra import infer, lists, pairs as pr, presence as pz, utils
    from vgra.data import path_key
    from vgra.head import head_input_channels

    deviations = {k: getattr(args, k) for k in ("epochs", "imgsz", "batch", "boot") if getattr(args, k) != PROTOCOL[k]}
    if args.fraction != 1.0:
        deviations["fraction"] = args.fraction
    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    classes = list(data["classes"])
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out or cfg["out_dir"]) / "gate_p"
    out.mkdir(parents=True, exist_ok=True)
    utils.set_seed(PROTOCOL["seed"])
    t_start = time.time()
    print(f"repo commit: {utils.git_commit()} | device {dev} | protocol settings "
          f"{'kept' if not deviations else 'CHANGED (smoke test only): ' + str(deviations)}")

    # 1. frozen detector
    weights = find_detector(args)
    model, _ = load_checkpoint(str(weights), device="cpu")
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.half() if dev.type == "cuda" else model.float()
    channels = head_input_channels(model, imgsz=256)
    print(f"detector: {weights} | Detect inputs {channels}")

    # 2. data
    index = pr.read_index(ix.locate_index(data, args.index))
    info = infer.image_info(index)
    train_paths = lists.train_image_paths(index, "orig")
    if args.fraction < 1.0:
        train_paths = train_paths[: max(8, int(len(train_paths) * args.fraction))]
    valid_rows = index[(index["split"] == "valid") & index["is_base"]] if "is_base" in index else index[index["split"] == "valid"]
    valid_paths = list(valid_rows["path"])
    train_rows = index[index["path"].map(path_key).isin({path_key(p) for p in train_paths})]
    if not args.no_warm:
        w = lists.warm_files([p for pair in zip(index.loc[index["path"].isin(train_paths + valid_paths), "path"],
                                               index.loc[index["path"].isin(train_paths + valid_paths), "label_path"])
                              for p in pair])
        print(f"read {w['files']:,} files ({w['bytes'] / 1e9:.1f} GB) in {w['seconds'] / 60:.1f} min")
    train_ds = infer.image_dataset(train_paths, classes, args.imgsz, out / "lists" / "train.txt", augment=True)
    loader = infer.image_loader(train_ds, args.batch, shuffle=True, workers=args.workers, seed=PROTOCOL["seed"])
    pos_weight = pz.positive_weights(train_rows, classes, PROTOCOL["pos_weight_cap"])
    print(f"train {len(train_ds)} images | valid {len(valid_paths)} images | positive weights "
          + ", ".join(f"{c} {w:.1f}" for c, w in zip(classes, pos_weight.tolist())))

    # 3. train the presence branch
    branch = pz.PresenceBranch(channels, len(classes), PROTOCOL["d"]).to(dev)
    opt = torch.optim.AdamW(branch.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"])
    total = args.epochs * len(loader)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, history = 0, []
    for epoch in range(args.epochs):
        branch.train()
        t0, running = time.time(), []
        for batch in loader:
            for group in opt.param_groups:
                group["lr"] = pz.lr_at(step, total, PROTOCOL["lr"], PROTOCOL["warmup"])
            x = batch["img"].to(dev).float() / 255
            with torch.no_grad():
                _, preds = model(x.half() if dev.type == "cuda" else x)
            views = torch.tensor([info[path_key(f)][2] for f in batch["im_file"]], device=dev)
            targets = pz.presence_targets(batch, x.shape[0], len(classes))
            with torch.autocast(dev.type, enabled=use_amp):
                logits = branch([f.float() for f in preds["feats"]], views)
            loss = pz.presence_loss(logits, targets, pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(float(loss.detach()))
            step += 1
        history.append({"epoch": epoch + 1, "loss": sum(running) / len(running), "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}/{args.epochs}: presence loss {history[-1]['loss']:.4f} "
              f"({history[-1]['minutes']:.1f} min)")
    torch.save({"state_dict": branch.state_dict(), "channels": channels, "classes": classes,
                "protocol": PROTOCOL, "deviations": deviations}, out / "presence_branch.pt")

    # 4. evaluate on valid
    branch.eval()
    valid_ds = infer.image_dataset(valid_paths, classes, args.imgsz, out / "lists" / "valid.txt", augment=False)

    def presence(feats, views, batch):
        with torch.autocast(dev.type, enabled=use_amp):
            return torch.sigmoid(branch([f.float() for f in feats], views))

    t0 = time.time()
    dets, imgs, _ = infer.predict(model, infer.image_loader(valid_ds, args.batch, workers=args.workers), classes,
                                  args.imgsz, info, extra=presence, device=dev)
    print(f"valid inference: {len(dets):,} detections in {time.time() - t0:.0f} s")
    dets.to_csv(out / "detections.csv.gz", index=False)
    imgs.to_csv(out / "images.csv.gz", index=False)

    table = gp.evaluate(dets, imgs, classes, PROTOCOL["folds"], args.boot)
    table.to_csv(out / "gate_p.csv", index=False)
    show = [c for c in ["cls", "target", "n_targets", "n_tp", "ap50_conf", "ap50_base", "ap50_new", "gain",
                        "gain_lo", "gain_hi", "auc_img_own", "auc_img_presence", "auc_img_base", "auc_img_new"]
            if c in table]
    pd.set_option("display.width", 250)
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ok, good, mb, mn = gp.verdict(table)
    info_out = {"commit": utils.git_commit(), "detector": str(weights), "protocol": PROTOCOL,
                "deviations": deviations, "history": history, "minutes": (time.time() - t_start) / 60,
                "verdict": "PASS" if ok else "FAIL", "classes_with_gain": good,
                "mean_ap50_base": mb, "mean_ap50_new": mn}
    (out / "run_info.json").write_text(json.dumps(info_out, indent=2))
    print(f"target classes with a gain interval above 0: {', '.join(good) or 'none'} (rule needs 2) | "
          f"mean AP50 over labelled classes: base {mb:.4f}, new {mn:.4f}")
    if deviations:
        print("NOTE: protocol settings were changed; this run cannot decide Gate P.")
    print(f"[info] outputs: {out}")
    print(f"GATE P: {'PASS' if ok else 'FAIL'}" + (" (smoke test, not valid)" if deviations else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
