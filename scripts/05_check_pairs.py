"""STEP-05: check the paired loader.

Run (CPU is enough):
    python scripts/05_check_pairs.py --config configs/base.yaml

Checks
1. Every pair: the AP and LAT files belong to the same patient, study and wrist side,
   with projection 1 (AP) and 2 (LAT).
2. Twenty augmented training batches and two validation batches: layout [AP..., LAT...],
   image i and image B + i are one pair, every box class also exists in the original label.
3. How often augmentation removes a class from an image (it changes the pair's state).
Saves image grids (AP | LAT per row, boxes drawn) to <out_dir>/pair_check/ for a visual check.
Ends with 'PAIR CHECK: PASS' or 'PAIR CHECK: FAIL'.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLORS = [(0, 165, 255), (255, 0, 255), (128, 128, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 0), (255, 0, 0), (0, 255, 255), (160, 160, 160)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--batches", type=int, default=20, help="training batches to check")
    ap.add_argument("--save", type=int, default=4, help="training batches to save as images")
    ap.add_argument("--batch-pairs", type=int, default=None, help="pairs per batch; default train.pair_batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def describe(im_file: str, prefixes: dict) -> dict | None:
    """Patient, study, side and projection from an image path."""
    from vgra.index import parse_filestem, split_variant

    _, stem = split_variant(Path(im_file).stem, prefixes)
    return parse_filestem(stem)


def save_grid(batch: dict, classes: list, path: Path, cell: int = 256) -> None:
    """One row per pair: AP image | LAT image, boxes and class names drawn."""
    import cv2
    import numpy as np

    n = batch["n_pairs"]
    rows = []
    for i in range(n):
        cells = []
        for j, view in ((i, "AP"), (n + i, "LAT")):
            img = batch["img"][j].permute(1, 2, 0).numpy()[:, :, ::-1].copy()  # RGB tensor -> BGR image
            h, w = img.shape[:2]
            sel = batch["batch_idx"] == j
            for c, (x, y, bw, bh) in zip(batch["cls"][sel].view(-1).tolist(), batch["bboxes"][sel].tolist()):
                p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
                p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
                color = COLORS[int(c) % len(COLORS)]
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(img, classes[int(c)], (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            img = cv2.resize(img, (cell, cell))
            cv2.putText(img, f"{batch['pair_id'][i]} {view}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cells.append(img)
        rows.append(np.hstack(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> int:
    args = parse_args()
    from vgra import data as vd
    from vgra import index as ix
    from vgra import lists, pairs as pr, utils

    cfg = utils.load_config(args.config)
    data, train_cfg = cfg["data"], cfg["train"]
    if args.root:
        data["root"] = args.root
    classes = list(data["classes"])
    prefixes = data.get("variant_prefixes") or {}
    imgsz = int(args.imgsz or train_cfg["imgsz"])
    batch_pairs = int(args.batch_pairs or train_cfg.get("pair_batch", 8))
    out = Path(args.out or cfg["out_dir"]) / "pair_check"
    print(f"repo commit: {utils.git_commit()} | image size {imgsz} | {batch_pairs} pairs per batch")

    index = pr.read_index(ix.locate_index(data, args.index))
    pairs = pr.build_pairs(index, classes)
    print(f"pairs: {len(pairs)} ({', '.join(f'{s} {n}' for s, n in pairs['split'].value_counts().items())})")

    if not args.no_warm:
        use = pairs[pairs["split"].isin(["train", "valid"])]
        ids = set(use["ap_image_id"]) | set(use["lat_image_id"])
        rows = index[index["image_id"].isin(ids)]
        w = lists.warm_files([p for pr_ in zip(rows["path"], rows["label_path"]) for p in pr_])
        print(f"read {w['files']:,} files ({w['bytes'] / 1e9:.1f} GB) in {w['seconds'] / 60:.1f} min")

    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))

    t0 = time.time()
    train = vd.PairDataset(pairs, index, "train", classes, imgsz, augment=True, list_dir=out / "lists")
    valid = vd.PairDataset(pairs, index, "valid", classes, imgsz, augment=False, list_dir=out / "lists")
    print(f"datasets built in {time.time() - t0:.0f} s | train {len(train)} pairs, valid {len(valid)} pairs")

    # 1. every pair, from the file names
    bad = []
    for ds in (train, valid):
        for i in range(len(ds)):
            a = describe(ds.images.im_files[ds.ap_index[i]], prefixes)
            b = describe(ds.images.im_files[ds.lat_index[i]], prefixes)
            same = a and b and all(a[k] == b[k] for k in ("patient_id", "study_number", "laterality"))
            if not (same and a["projection"] == 1 and b["projection"] == 2):
                bad.append(ds.pairs.at[i, "pair_id"])
    check("every pair is one AP and one LAT image of the same study-side", not bad,
          f"{len(train) + len(valid)} pairs, bad: {len(bad)} {bad[:3]}")

    # 2. batches
    labels_of = index.set_index("path")
    has_cols = [f"has_{c}" for c in classes]
    layout_bad = class_bad = changed = pair_classes = 0
    seconds = []
    loaders = [("train", vd.build_pair_loader(train, batch_pairs, shuffle=True, workers=args.workers, seed=0),
                args.batches), ("valid", vd.build_pair_loader(valid, batch_pairs, shuffle=False, workers=args.workers), 2)]
    for split, loader, n_batches in loaders:
        t_prev = time.time()
        for k, batch in enumerate(loader):
            if k >= n_batches:
                break
            seconds.append(time.time() - t_prev) if split == "train" and k > 0 else None
            n = batch["n_pairs"]
            if tuple(batch["img"].shape) != (2 * n, 3, imgsz, imgsz) or (batch["batch_idx"].numel() and int(batch["batch_idx"].max()) >= 2 * n):
                layout_bad += 1
            has = vd.class_sets(batch, len(classes))
            states = vd.batch_states(batch, len(classes))
            for i in range(n):
                a = describe(batch["im_file"][i], prefixes)
                b = describe(batch["im_file"][n + i], prefixes)
                if not (a and b and a["projection"] == 1 and b["projection"] == 2
                        and a["patient_id"] == b["patient_id"] and batch["pair_id"][i].startswith(f"{a['patient_id']:04d}_")):
                    layout_bad += 1
                row = train.pairs if split == "train" else valid.pairs
                original = row.set_index("pair_id").loc[batch["pair_id"][i]]
                for j in (i, n + i):
                    allowed = labels_of.loc[batch["im_file"][j], has_cols].to_numpy().astype(bool)
                    if (has[j].numpy().astype(bool) & ~allowed).any():
                        class_bad += 1
                for c, name in enumerate(classes):
                    if original[f"state_{name}"] != "00":
                        pair_classes += 1
                        changed += states[i][c] != original[f"state_{name}"]
            if k < args.save:
                save_grid(batch, classes, out / f"{split}_batch_{k:02d}.jpg")
            t_prev = time.time()

    check("batch layout: image i and image B + i are the AP and LAT of pair i", layout_bad == 0, f"problems: {layout_bad}")
    check("every box class exists in the original label", class_bad == 0, f"images with a new class: {class_bad}")
    print(f"[info] augmentation changed {changed} of {pair_classes} non-empty pair states "
          f"({100 * changed / max(pair_classes, 1):.1f}%); states are recomputed from batch labels in training")
    if seconds:
        print(f"[info] loading: {sum(seconds) / len(seconds):.2f} s per batch of {batch_pairs} pairs "
              f"({args.workers} workers)")
    print(f"[info] image grids: {out}")
    ok = all(results)
    print("PAIR CHECK: PASS" if ok else "PAIR CHECK: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
