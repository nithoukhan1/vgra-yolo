"""STEP-02 tests: pairs, states and Gate 1 on a small hand-made index."""

import pandas as pd

from vgra import pairs as pr

CLASSES = ["fracture", "pronatorsign"]


def make_index():
    # image_id, study_side_id, patient, view, split, has_fracture, has_pronatorsign, is_base, initial, cast
    rows = [
        ("a1", "0001_01_L", 1, "ap", "train", 1, 0, True, 1, 0),
        ("a2", "0001_01_L", 1, "lat", "train", 1, 1, True, 1, 0),   # pair: fracture 11, pronator 01
        ("b1", "0002_01_R", 2, "ap", "train", 1, 0, True, 0, 1),
        ("b2", "0002_01_R", 2, "lat", "train", 0, 0, True, 0, 1),   # pair: fracture 10, cast
        ("c1", "0003_01_L", 3, "ap", "train", 0, 0, True, 1, 0),    # ap only: no pair
        ("d1", "0004_01_R", 4, "ap", "valid", 0, 0, True, 1, 0),
        ("d2", "0004_01_R", 4, "lat", "valid", 1, 0, True, 1, 0),   # pair: fracture 01
        ("e1", "0005_01_L", 5, "ap", "test", 0, 0, True, 1, 0),
        ("e2", "0005_01_L", 5, "lat", "test", 0, 0, True, 1, 0),
        ("e3", "0005_01_L", 5, "other", "test", 0, 0, True, 1, 0),  # three images: no pair
        ("f1", "0006_01_R", 6, "ap", "test", 0, 0, True, 1, 0),
        ("f2", "0006_01_R", 6, "ap", "test", 0, 0, True, 1, 0),     # two ap: no pair
        ("aug_a1", "0001_01_L", 1, "ap", "train", 1, 0, False, 1, 0),  # copy: ignored
    ]
    df = pd.DataFrame(rows, columns=["image_id", "study_side_id", "patient_id", "view", "split",
                                     "has_fracture", "has_pronatorsign", "is_base", "initial_exam", "cast"])
    df["in_metadata"] = True
    df["study_number"] = 1
    df["laterality"] = df["study_side_id"].str[-1]
    return df


def test_build_pairs():
    pairs = pr.build_pairs(make_index(), CLASSES)
    assert pairs["pair_id"].tolist() == ["0001_01_L", "0002_01_R", "0004_01_R"]
    first = pairs.iloc[0]
    assert (first["ap_image_id"], first["lat_image_id"]) == ("a1", "a2")
    assert first["state_fracture"] == "11" and first["state_pronatorsign"] == "01"
    assert pairs.set_index("pair_id").loc["0002_01_R", "state_fracture"] == "10"
    assert pairs.set_index("pair_id").loc["0002_01_R", "cast"] == 1
    assert pairs.set_index("pair_id").loc["0004_01_R", "split"] == "valid"


def test_states_survive_csv(tmp_path):
    pairs = pr.build_pairs(make_index(), CLASSES)
    path = tmp_path / "pairs.csv"
    pairs.to_csv(path, index=False)
    back = pr.read_pairs(path)
    assert back["state_pronatorsign"].tolist() == ["01", "00", "00"]  # not 1, 0, 0


def test_eda_pairing():
    index = make_index()
    table = pr.eda_pairing(index, pr.build_pairs(index, CLASSES)).set_index("split")
    assert table.loc["all", "pairs"] == 3 and table.loc["all", "study_sides"] == 6
    assert table.loc["all", "images"] == 12 and table.loc["all", "images_in_pairs"] == 6
    assert table.loc["train", "unpaired_images"] == 1


def test_eda_states_and_lift():
    pairs = pr.build_pairs(make_index(), CLASSES)
    s = pr.eda_states(pairs, CLASSES).set_index("class")
    assert s.loc["fracture", ["n00", "n10", "n01", "n11"]].tolist() == [0, 1, 1, 1]
    assert s.loc["fracture", "one_view_only"] == 2
    assert s.loc["fracture", "pct_one_view_of_present"] == 66.7
    assert pr.eda_states(pairs, CLASSES, "train").set_index("class").loc["fracture", "present"] == 2
    lift = pr.eda_partner_lift(pairs, CLASSES).set_index("class")
    assert lift.loc["fracture", "p_ap_given_lat"] == 0.5  # lat has it in 2 pairs, ap in 1 of them
    assert lift.loc["pronatorsign", "p_lat"] == round(1 / 3, 4)


def test_class_by_view():
    table = pr.eda_class_by_view(make_index(), CLASSES).set_index("class")
    assert table.loc["fracture", "ap_images"] == 2 and table.loc["fracture", "lat_images"] == 2
    assert table.loc["pronatorsign", "lat_share_of_class"] == 100.0


def test_gate1():
    index = make_index()
    pairs = pr.build_pairs(index, CLASSES)
    easy = {"min_pair_image_share": 0.4, "min_one_view_pairs_train": 1, "min_one_view_share": 0.1, "min_classes": 2}
    table, lines, ok = pr.gate1(index, pairs, CLASSES, easy)
    assert ok and table["has_mass"].all() and len(lines) == 2
    hard = dict(easy, min_pair_image_share=0.9)
    assert pr.gate1(index, pairs, CLASSES, hard)[2] is False
