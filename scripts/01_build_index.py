"""STEP-01: build the image index and print EDA tables 1-4.

Run:  python scripts/01_build_index.py --config configs/base.yaml
Writes index.csv and the EDA tables, then ends with 'INDEX CHECK: PASS' or 'FAIL' (exit code 1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from vgra import index as ix
from vgra import utils

REPO_ROOT = Path(__file__).resolve().parents[1]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--root", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="override the folder for index.csv and eda/")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    index_csv = Path(data["index_csv"])
    eda_dir = Path(data["eda_dir"])
    if args.out:
        index_csv = Path(args.out) / "index.csv"
        eda_dir = Path(args.out) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"repo commit: {utils.git_commit()}")
    print(f"data root  : {data['root']}")
    t0 = time.time()
    index, meta = ix.build_index(cfg)
    print(f"indexed {len(index)} image files in {time.time() - t0:.0f} s")

    index.to_csv(index_csv, index=False)
    print(f"wrote {index_csv}")

    tables = {
        "eda1_counts": ix.eda_counts(index, meta),
        "eda2_class_freq": ix.eda_class_freq(index, list(data["classes"])),
        "eda3_views": ix.eda_views(index),
    }
    per_size, combos = ix.eda_studies(index)
    tables["eda4_images_per_study_side"] = per_size
    tables["eda4_view_combinations"] = combos

    titles = {
        "eda1_counts": "EDA-1 counts",
        "eda2_class_freq": "EDA-2 class frequency (base images)",
        "eda3_views": "EDA-3 views",
        "eda4_images_per_study_side": "EDA-4a images per study-side",
        "eda4_view_combinations": "EDA-4b view combinations per study-side",
    }
    for name, table in tables.items():
        table.to_csv(eda_dir / f"{name}.csv", index=False)
        show(titles[name], table)
    print(f"\nwrote {len(tables)} tables to {eda_dir}")

    print("\n=== checks ===")
    checks = ix.check_index(index, meta, data.get("expect"))
    ok_all = True
    for name, ok, detail in checks:
        ok_all = ok_all and ok
        print(f"[{'ok' if ok else 'FAIL'}] {name} ({detail})")
    print("INDEX CHECK: PASS" if ok_all else "INDEX CHECK: FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
