"""STEP-03 tests: training lists and the Ultralytics data file."""

import pandas as pd
import pytest
import yaml

from vgra import lists
from vgra.index import locate_index

DATA_CFG = {
    "root": "/data/GRAZ", "data_dir": "data",
    "folders": {"train": "train_aug", "valid": "valid", "test": "test"},
    "classes": ["boneanomaly", "fracture"],
}


def make_index():
    rows = [
        ("orig_a", "train", True, "/data/GRAZ/data/images/train_aug/orig_a.png"),
        ("aug_a", "train", False, "/data/GRAZ/data/images/train_aug/aug_a.png"),
        ("orig_b", "train", True, "/data/GRAZ/data/images/train_aug/orig_b.png"),
        ("aug_b", "train", False, "/data/GRAZ/data/images/train_aug/aug_b.png"),
        ("c", "valid", True, "/data/GRAZ/data/images/valid/c.png"),
    ]
    df = pd.DataFrame(rows, columns=["image_id", "split", "is_base", "path"])
    df["in_metadata"] = True
    return df


def test_train_image_paths():
    index = make_index()
    orig = lists.train_image_paths(index, "orig")
    assert [p.rsplit("/", 1)[1] for p in orig] == ["orig_a.png", "orig_b.png"]
    assert len(lists.train_image_paths(index, "orig+aug")) == 4
    with pytest.raises(ValueError):
        lists.train_image_paths(index, "everything")


def test_write_data_files(tmp_path):
    path = lists.write_data_files(make_index(), DATA_CFG, tmp_path, "orig")
    data = yaml.safe_load(path.read_text())
    assert path.name == "data_orig.yaml"
    assert data["val"] == "/data/GRAZ/data/images/valid" and data["test"].endswith("/images/test")
    assert data["names"] == {0: "boneanomaly", 1: "fracture"}
    listed = (tmp_path / "train_orig.txt").read_text().split()
    assert listed == lists.train_image_paths(make_index(), "orig")
    assert lists.write_data_files(make_index(), DATA_CFG, tmp_path, "orig+aug").name == "data_orig_aug.yaml"


def test_locate_index(tmp_path):
    f = tmp_path / "index.csv"
    f.write_text("x\n")
    assert locate_index({"index_csv": str(tmp_path / "missing.csv"), "index_input": str(f)}) == f
    assert locate_index({"index_csv": "nope"}, override=str(f)) == f
    with pytest.raises(FileNotFoundError):
        locate_index({"index_csv": "nope", "index_input": "nope2"})


def test_files_to_warm():
    index = make_index()
    index["label_path"] = index["path"].str.replace("/images/", "/labels/").str.replace(".png", ".txt")
    files = lists.files_to_warm(index, "orig")
    assert files[:2] == ["/data/GRAZ/data/images/train_aug/orig_a.png", "/data/GRAZ/data/labels/train_aug/orig_a.txt"]
    assert len(files) == 6  # 2 originals + 1 validation image, each with its label
    assert len(lists.files_to_warm(index, "orig", fraction=0.5)) == 4
    assert len(lists.files_to_warm(index, "orig+aug")) == 10


def test_warm_files(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 3000)
    b.write_bytes(b"y" * 5)
    out = lists.warm_files([str(a), str(b), str(tmp_path / "missing")], threads=2, chunk=1024)
    assert out["files"] == 3 and out["bytes"] == 3005 and out["errors"] == 1
