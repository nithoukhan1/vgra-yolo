"""Training lists and the Ultralytics data file (STEP-03).

The Kaggle dataset is read-only and its training folder mixes originals (orig_*) with
offline copies (aug_*). Instead of copying images, we write a text file with the image
paths we want and point Ultralytics at it. Labels are found from the image path
(.../images/... -> .../labels/...).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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


def files_to_warm(index: pd.DataFrame, train_set: str = "orig", fraction: float = 1.0,
                  include_test: bool = False) -> list:
    """Image and label files a training run will read: the training list (first `fraction`,
    as Ultralytics takes it), the validation split and, if asked, the test split."""
    train = train_image_paths(index, train_set)
    if fraction < 1:
        train = train[: round(len(train) * fraction)]
    label_of = dict(zip(index["path"], index["label_path"]))
    files = [p for path in train for p in (path, label_of[path])]
    splits = ["valid"] + (["test"] if include_test else [])
    rows = index[index["split"].isin(splits) & index["is_base"] & index["in_metadata"]]
    files += [p for pair in zip(rows["path"], rows["label_path"]) for p in pair]
    return files


def warm_files(paths: list, threads: int = 32, chunk: int = 1 << 20) -> dict:
    """Read every file once, in parallel, so that later reads come from memory.

    On Kaggle the input datasets are slow on the first read of each file in a session;
    after one read the operating system keeps the file in memory.
    """

    def read(path):
        n = 0
        try:
            with open(path, "rb") as f:
                while block := f.read(chunk):
                    n += len(block)
        except OSError:
            return 0, 1
        return n, 0

    t0 = time.time()
    total = errors = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for n, err in pool.map(read, paths):
            total += n
            errors += err
    return {"files": len(paths), "bytes": total, "seconds": time.time() - t0, "errors": errors}


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
