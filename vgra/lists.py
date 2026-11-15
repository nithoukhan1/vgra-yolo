"""Training lists and the Ultralytics data file (STEP-03).

The Kaggle dataset is read-only and its training folder mixes originals (orig_*) with
offline copies (aug_*). Instead of copying images, we write a text file with the image
paths we want and point Ultralytics at it. Labels are found from the image path
(.../images/... -> .../labels/...).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

TRAIN_SETS = ("orig", "orig+aug")


def train_image_paths(index: pd.DataFrame, train_set: str = "orig") -> list:
    """Image paths of the training split. 'orig' = originals only; 'orig+aug' = all files."""
    if train_set not in TRAIN_SETS:
        raise ValueError(f"train_set must be one of {TRAIN_SETS}, got {train_set!r}")
    rows = index[(index["split"] == "train") & index["in_metadata"]]
    if train_set == "orig":
        rows = rows[rows["is_base"]]
    return sorted(rows["path"].tolist())


def write_data_files(index: pd.DataFrame, data_cfg: dict, out_dir: str | Path, train_set: str = "orig") -> Path:
    """Write <out_dir>/train_<set>.txt and <out_dir>/data_<set>.yaml; return the yaml path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = train_set.replace("+", "_")

    paths = train_image_paths(index, train_set)
    if not paths:
        raise ValueError("no training images selected")
    list_path = out_dir / f"train_{tag}.txt"
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")

    images = Path(data_cfg["root"]) / data_cfg["data_dir"] / "images"
    data = {
        "train": list_path.as_posix(),
        "val": (images / data_cfg["folders"]["valid"]).as_posix(),
        "test": (images / data_cfg["folders"]["test"]).as_posix(),
        "names": {i: name for i, name in enumerate(data_cfg["classes"])},
    }
    yaml_path = out_dir / f"data_{tag}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path
