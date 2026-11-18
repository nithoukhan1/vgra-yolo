"""A tiny dataset in the Kaggle layout, with real small PNG images, for pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
           "periostealreaction", "pronatorsign", "softtissue", "text"]
FOLDERS = {"train": "train_aug", "valid": "valid", "test": "test"}


def make_tiny_dataset(root: str | Path, n_train: int = 12, n_valid: int = 4, n_test: int = 2) -> Path:
    """Patients with one AP and one LAT image each. AP images get a fracture box, LAT images a
    fracture and a pronator sign box, so the pronator sign state is '01' in every pair.
    Patient 1 also has a third (oblique) image, so its study-side is not a pair.
    Even patients have a torus fracture code (AO 23r-M/2.1), odd patients 23r-M/3.1."""
    import cv2

    root = Path(root)
    rng = np.random.default_rng(0)
    rows = []
    splits = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    for pid, split in enumerate(splits, start=1):
        projections = (1, 2, 3) if pid == 1 else (1, 2)
        for proj in projections:
            stem = f"{pid:04d}_{1000 + pid * 10 + proj}_01_WRI-L{proj}_M010"
            rows.append({"filestem": stem, "patient_id": pid, "study_number": 1, "timehash": 1000 + pid * 10 + proj,
                         "gender": "M", "age": 10.0, "laterality": "L", "projection": proj, "initial_exam": 1.0,
                         "ao_classification": "23r-M/2.1" if pid % 2 == 0 else "23r-M/3.1", "cast": None, "fracture_visible": 1.0, "metal": None, "split": split})
    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.drop(columns="split").to_csv(root / "dataset.csv", index=False)
    for split, folder in FOLDERS.items():
        part = df[df["split"] == split]
        part.drop(columns="split").to_csv(root / f"{split}_data.csv", index=False)
        img_dir = root / "data" / "images" / folder
        lbl_dir = root / "data" / "labels" / folder
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in part.itertuples():
            im = rng.integers(0, 60, (160, 96, 3), dtype=np.uint8)
            im[60:100, 30:60] = 220
            text = "3 0.46875 0.5 0.3125 0.25\n8 0.1 0.1 0.1 0.1\n"
            if r.projection == 2:
                im[110:130, 10:40] = 160
                text += "6 0.2604 0.75 0.3125 0.125\n"
            names = [f"orig_{r.filestem}", f"aug_{r.filestem}"] if split == "train" else [r.filestem]
            for name in names:
                cv2.imwrite(str(img_dir / f"{name}.png"), im)
                (lbl_dir / f"{name}.txt").write_text(text)
    return root


def tiny_config(root: str | Path) -> dict:
    return {"data": {
        "root": str(root), "meta_csv": "dataset.csv", "data_dir": "data",
        "split_csvs": {s: f"{s}_data.csv" for s in FOLDERS},
        "folders": FOLDERS,
        "variant_prefixes": {"orig_": "orig", "aug_": "aug"},
        "views": {1: "ap", 2: "lat"},
        "classes": CLASSES,
    }}
