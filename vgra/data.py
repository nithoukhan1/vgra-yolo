"""Paired AP/LAT loader (STEP-05).

A batch of B pairs holds 2B images in a fixed layout:

    [ap_1, ..., ap_B, lat_1, ..., lat_B]

so image i and image B + i are the two views of pair i. Images are loaded, letterboxed and
augmented by an Ultralytics YOLODataset, independently for each view (no geometric link
between the two views is assumed). Mosaic, mixup, cutmix and copy-paste are off, so every
training image stays one whole radiograph and keeps its partner.

Only pairs are used here. Unpaired images are handled single-view (VGRA off).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

PAIR_OFF = {"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}  # one radiograph per image


def path_key(path) -> str:
    """One spelling for a file path, so paths from index.csv match the paths Ultralytics returns.
    Ultralytics turns '/' into the OS separator (backslash on Windows); case is ignored on Windows."""
    text = str(path).replace("\\", "/")
    return os.path.normcase(os.path.normpath(text)).replace("\\", "/")


def pair_hyp(imgsz: int, overrides: dict | None = None):
    """Ultralytics augmentation settings for paired training: defaults, with mixing off."""
    cfg = {"imgsz": int(imgsz), "rect": False, "cache": False, **PAIR_OFF, **(overrides or {})}
    for key, value in PAIR_OFF.items():  # mixing must stay off whatever the overrides say
        cfg[key] = value
    return get_cfg(DEFAULT_CFG, cfg)


class PairDataset(Dataset):
    """The AP/LAT pairs of one split. Item i is the two views of pair i."""

    def __init__(self, pairs: pd.DataFrame, index: pd.DataFrame, split: str, classes: list, imgsz: int,
                 augment: bool, list_dir: str | Path, hyp_overrides: dict | None = None):
        rows = pairs[pairs["split"] == split].sort_values("pair_id").reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"no pairs in split {split!r}")
        path_of = dict(zip(index["image_id"], index["path"]))
        ap_paths = [path_of[i] for i in rows["ap_image_id"]]
        lat_paths = [path_of[i] for i in rows["lat_image_id"]]

        # Ultralytics reads an image list from a text file (one path per line)
        list_dir = Path(list_dir)
        list_dir.mkdir(parents=True, exist_ok=True)
        list_file = list_dir / f"pairs_{split}.txt"
        list_file.write_text("\n".join(sorted(ap_paths + lat_paths)) + "\n", encoding="utf-8")

        data = {"names": dict(enumerate(classes)), "nc": len(classes), "channels": 3}
        self.images = YOLODataset(
            img_path=str(list_file), imgsz=int(imgsz), augment=bool(augment), hyp=pair_hyp(imgsz, hyp_overrides),
            rect=False, stride=32, pad=0.0, prefix=f"pairs {split}: ", task="detect", data=data,
        )
        position = {path_key(p): i for i, p in enumerate(self.images.im_files)}
        missing = [p for p in ap_paths + lat_paths if path_key(p) not in position]
        if missing:
            raise RuntimeError(f"{len(missing)} pair images not loaded by Ultralytics, e.g. {missing[:2]}")

        self.split = split
        self.augment = bool(augment)
        self.pairs = rows
        self.ap_index = [position[path_key(p)] for p in ap_paths]
        self.lat_index = [position[path_key(p)] for p in lat_paths]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        return {
            "ap": self.images[self.ap_index[i]],
            "lat": self.images[self.lat_index[i]],
            "pair_id": self.pairs.at[i, "pair_id"],
        }


def collate_pairs(items: list) -> dict:
    """Batch B pairs as 2B images: all AP images first, then their LAT partners in the same order."""
    batch = YOLODataset.collate_fn([it["ap"] for it in items] + [it["lat"] for it in items])
    n = len(items)
    batch["n_pairs"] = n
    batch["pair_id"] = [it["pair_id"] for it in items]
    batch["view"] = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])  # 0 ap, 1 lat
    batch["has_partner"] = torch.ones(2 * n, dtype=torch.bool)
    return batch


def build_pair_loader(dataset: PairDataset, batch_pairs: int, shuffle: bool, workers: int = 4,
                      seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_pairs), shuffle=bool(shuffle), num_workers=int(workers),
                      collate_fn=collate_pairs, pin_memory=torch.cuda.is_available(), drop_last=False,
                      generator=generator, persistent_workers=False)


def class_sets(batch: dict, n_classes: int) -> torch.Tensor:
    """(2B, n_classes) 0/1 table: which classes have at least one box in each image of the batch."""
    n_img = batch["img"].shape[0]
    has = torch.zeros(n_img, n_classes, dtype=torch.long)
    if batch["cls"].numel():
        has[batch["batch_idx"].long(), batch["cls"].view(-1).long()] = 1
    return has


def batch_states(batch: dict, n_classes: int) -> list:
    """Visibility state per pair and class, AP digit first ('00', '10', '01', '11'), from the batch labels."""
    has = class_sets(batch, n_classes)
    n = batch["n_pairs"]
    return [[f"{int(has[i, c])}{int(has[n + i, c])}" for c in range(n_classes)] for i in range(n)]
