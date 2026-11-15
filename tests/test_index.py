"""STEP-01 tests: the index is built correctly on a tiny made-up dataset."""

import pandas as pd
import pytest

from vgra import index as ix

CLASSES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
           "periostealreaction", "pronatorsign", "softtissue", "text"]

# filestem, patient, study, side, projection, split, label text
ROWS = [
    ("0001_100_01_WRI-L1_M014", 1, 1, "L", 1, "train", "3 0.5 0.5 0.1 0.1\n8 0.1 0.1 0.1 0.1\n"),
    ("0001_140_01_WRI-L2_M014", 1, 1, "L", 2, "train", "3 0.5 0.5 0.1 0.1\n3 0.6 0.6 0.1 0.1\n"),
    ("0002_200_01_WRI-R1_F012", 2, 1, "R", 1, "train", ""),
    ("0003_300_01_WRI-R1_O016", 3, 1, "R", 1, "valid", "8 0.1 0.1 0.1 0.1\n"),
    ("0003_340_01_WRI-R2_O016", 3, 1, "R", 2, "valid", "6 0.4 0.4 0.1 0.1\n"),
    ("0004_400_02_WRI-L3_M009", 4, 2, "L", 3, "test", "4 0.4 0.4 0.1 0.1\n"),
]
FOLDERS = {"train": "train_aug", "valid": "valid", "test": "test"}


def make_dataset(root, rows=ROWS):
    meta = pd.DataFrame([{
        "filestem": r[0], "patient_id": r[1], "study_number": r[2], "laterality": r[3],
        "projection": r[4], "gender": "M", "age": 10.0, "initial_exam": 1.0, "cast": None,
        "fracture_visible": 1.0 if "3 " in r[6] else None, "metal": None, "timehash": 1,
    } for r in rows])
    meta.to_csv(root / "dataset.csv", index=False)
    for split in FOLDERS:
        meta[[r[5] == split for r in rows]].to_csv(root / f"{split}_data.csv", index=False)
    for stem, *_, split, text in rows:
        img = root / "data" / "images" / FOLDERS[split]
        lbl = root / "data" / "labels" / FOLDERS[split]
        img.mkdir(parents=True, exist_ok=True)
        lbl.mkdir(parents=True, exist_ok=True)
        names = [f"orig_{stem}", f"aug_{stem}"] if split == "train" else [stem]
        for name in names:
            (img / f"{name}.png").write_bytes(b"")
            (lbl / f"{name}.txt").write_text(text)


def make_cfg(root):
    return {"data": {
        "root": str(root), "meta_csv": "dataset.csv", "data_dir": "data",
        "split_csvs": {s: f"{s}_data.csv" for s in FOLDERS},
        "folders": FOLDERS,
        "variant_prefixes": {"orig_": "orig", "aug_": "aug"},
        "views": {1: "ap", 2: "lat"},
        "classes": CLASSES,
        "expect": {"images": 6, "train": 3, "valid": 2, "test": 1},
    }}


@pytest.fixture
def built(tmp_path):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    index, meta = ix.build_index(cfg)
    return cfg, index, meta


def test_parse_filestem():
    assert ix.parse_filestem("0001_1297860395_01_WRI-L1_M014") == {
        "patient_id": 1, "study_number": 1, "laterality": "L", "projection": 1}
    assert ix.parse_filestem("0412_1230151360_01_WRI-R1_O016")["projection"] == 1  # sex code O
    assert ix.parse_filestem("not_a_name") is None


def test_split_variant():
    prefixes = {"orig_": "orig", "aug_": "aug"}
    assert ix.split_variant("aug_0001_x", prefixes) == ("aug", "0001_x")
    assert ix.split_variant("orig_0001_x", prefixes) == ("orig", "0001_x")
    assert ix.split_variant("0001_x", prefixes) == ("orig", "0001_x")


def test_index_rows_and_base(built):
    _, index, _ = built
    assert len(index) == 9  # 3 train images x 2 variants + 2 valid + 1 test
    assert index["is_base"].sum() == 6
    assert set(index.loc[index["is_base"], "variant"]) == {"orig"}
    assert index["in_metadata"].all() and index["label_found"].all()


def test_index_columns(built):
    _, index, _ = built
    row = index.set_index("image_id").loc["orig_0001_140_01_WRI-L2_M014"]
    assert row["split"] == "train" and row["view"] == "lat"
    assert row["study_side_id"] == "0001_01_L"
    assert row["n_fracture"] == 2 and row["has_fracture"] == 1 and row["n_boxes"] == 2
    assert row["fracture_visible"] == 1 and row["cast"] == 0
    other = index.set_index("image_id").loc["0004_400_02_WRI-L3_M009"]
    assert other["view"] == "other" and other["has_metal"] == 1


def test_checks_pass(built):
    cfg, index, meta = built
    checks = ix.check_index(index, meta, cfg["data"]["expect"])
    assert all(ok for _, ok, _ in checks), [c for c in checks if not c[1]]


def test_checks_catch_problems(tmp_path):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    # a missing label, an image without metadata, and a copy whose label differs
    (tmp_path / "data/labels/valid/0003_300_01_WRI-R1_O016.txt").unlink()
    (tmp_path / "data/images/test/9999_1_01_WRI-L1_M001.png").write_bytes(b"")
    (tmp_path / "data/labels/train_aug/aug_0002_200_01_WRI-R1_F012.txt").write_text("3 0.5 0.5 0.1 0.1\n")
    index, meta = ix.build_index(cfg)
    failed = {name for name, ok, _ in ix.check_index(index, meta, cfg["data"]["expect"]) if not ok}
    assert failed == {"every image has a metadata row", "every image has a label file",
                      "copies share the original's labels"}


def test_patient_in_two_splits_is_caught(tmp_path):
    rows = list(ROWS)
    rows[1] = rows[1][:5] + ("valid",) + rows[1][6:]  # patient 1 now in train and valid
    make_dataset(tmp_path, rows)
    index, meta = ix.build_index(make_cfg(tmp_path))
    failed = {name for name, ok, _ in ix.check_index(index, meta) if not ok}
    assert "no patient in two splits" in failed


def test_eda_tables(built):
    cfg, index, meta = built
    counts = ix.eda_counts(index, meta).set_index("item")["value"]
    assert counts["base images (one per filestem)"] == 6
    assert counts["train: images with no boxes"] == 1
    freq = ix.eda_class_freq(index, CLASSES).set_index("class")
    assert freq.loc["fracture", "all_boxes"] == 3 and freq.loc["fracture", "all_images"] == 2
    views = ix.eda_views(index)
    assert views["all"].sum() == 6
    per_size, combos = ix.eda_studies(index)
    assert combos.set_index("views_in_study_side").loc["1 ap + 1 lat", "study_sides"] == 2
    assert per_size["all"].sum() == 4  # four study-sides
