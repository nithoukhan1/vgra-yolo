"""STEP-05 tests: the paired loader on a tiny dataset with real images."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

from tiny_data import CLASSES, make_tiny_dataset, tiny_config  # noqa: E402

from vgra import data as vd  # noqa: E402
from vgra import index as ix  # noqa: E402
from vgra import pairs as pr  # noqa: E402

FRACTURE, PRONATOR = CLASSES.index("fracture"), CLASSES.index("pronatorsign")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = make_tiny_dataset(tmp_path_factory.mktemp("tiny") / "GRAZ")
    index, _ = ix.build_index(tiny_config(root))
    pairs = pr.build_pairs(index, CLASSES)
    return root, index, pairs


def make(tiny, split, augment, tmp_path):
    _, index, pairs = tiny
    return vd.PairDataset(pairs, index, split, CLASSES, imgsz=128, augment=augment, list_dir=tmp_path)


def test_pairs_exclude_the_three_image_study(tiny):
    _, _, pairs = tiny
    assert len(pairs) == 17  # 18 patients, patient 1 has three images
    assert "0001_01_L" not in set(pairs["pair_id"])


def test_items_are_the_two_views_of_one_pair(tiny, tmp_path):
    ds = make(tiny, "valid", augment=False, tmp_path=tmp_path)
    assert len(ds) == 4
    for i in range(len(ds)):
        item = ds[i]
        ap, lat = item["ap"]["im_file"], item["lat"]["im_file"]
        assert "WRI-L1" in ap and "WRI-L2" in lat
        assert ap.rsplit("/", 1)[1][:4] == lat.rsplit("/", 1)[1][:4] == item["pair_id"][:4]


def test_training_list_uses_originals_only(tiny, tmp_path):
    ds = make(tiny, "train", augment=True, tmp_path=tmp_path)
    assert len(ds) == 11 and all("/orig_" in f for f in ds.images.im_files)


def test_collate_layout_and_states(tiny, tmp_path):
    ds = make(tiny, "valid", augment=False, tmp_path=tmp_path)
    batch = vd.collate_pairs([ds[0], ds[1], ds[2]])
    assert batch["img"].shape == (6, 3, 128, 128) and batch["n_pairs"] == 3
    assert batch["view"].tolist() == [0, 0, 0, 1, 1, 1]
    for i in range(3):
        assert "WRI-L1" in batch["im_file"][i] and "WRI-L2" in batch["im_file"][3 + i]
    states = vd.batch_states(batch, len(CLASSES))
    assert all(s[FRACTURE] == "11" and s[PRONATOR] == "01" for s in states)


def test_mixing_stays_off():
    hyp = vd.pair_hyp(128, {"mosaic": 1.0, "fliplr": 0.0})
    assert hyp.mosaic == 0.0 and hyp.mixup == 0.0 and hyp.fliplr == 0.0


def test_loader(tiny, tmp_path):
    ds = make(tiny, "train", augment=True, tmp_path=tmp_path)
    loader = vd.build_pair_loader(ds, batch_pairs=4, shuffle=True, workers=0, seed=0)
    batches = list(loader)
    assert [b["n_pairs"] for b in batches] == [4, 4, 3]
    b = batches[0]
    assert b["img"].shape == (8, 3, 128, 128)
    assert int(b["batch_idx"].max()) < 8
