"""AP/LAT pairs and the pair audit (STEP-02).

A pair is one `ap` image and one `lat` image of the same study-side
(patient + study + wrist side), and nothing else in that study-side.
Only base images (originals) are used.

For each pair and class the state is two digits, AP first:
00 neither view, 10 AP only, 01 LAT only, 11 both views.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STATES = ("00", "10", "01", "11")
SPLITS = ("train", "valid", "test")


def read_index(path: str | Path) -> pd.DataFrame:
    """Read index.csv with the text columns kept as text."""
    return pd.read_csv(
        path,
        dtype={"laterality": str, "study_side_id": str, "label_hash": str, "gender": str},
        low_memory=False,
    )


def read_pairs(path: str | Path) -> pd.DataFrame:
    """Read pairs.csv. States must stay text ('01' must not become 1)."""
    head = pd.read_csv(path, nrows=0)
    text = {c: str for c in head.columns if c.startswith("state_") or c in ("pair_id", "laterality")}
    return pd.read_csv(path, dtype=text)


def base_images(index: pd.DataFrame) -> pd.DataFrame:
    return index[index["is_base"] & index["in_metadata"]]


def build_pairs(index: pd.DataFrame, classes: list) -> pd.DataFrame:
    """One row per pair: exactly one ap and one lat image in a study-side."""
    base = base_images(index)
    size = base.groupby("study_side_id")["image_id"].transform("size")
    two = base[size == 2]
    n_views = two.groupby("study_side_id")["view"].transform("nunique")
    two = two[(n_views == 2) & two["view"].isin(["ap", "lat"])]
    # keep study-sides where both remaining rows are present (one ap and one lat)
    two = two[two.groupby("study_side_id")["image_id"].transform("size") == 2]

    ap = two[two["view"] == "ap"].set_index("study_side_id")
    lat = two[two["view"] == "lat"].set_index("study_side_id").reindex(ap.index)

    pairs = pd.DataFrame({
        "pair_id": ap.index,
        "patient_id": ap["patient_id"].astype(int).values,
        "study_number": ap["study_number"].astype(int).values,
        "laterality": ap["laterality"].values,
        "split": ap["split"].values,
        "ap_image_id": ap["image_id"].values,
        "lat_image_id": lat["image_id"].values,
        "initial_exam": ap[["initial_exam"]].join(lat[["initial_exam"]], rsuffix="_lat").max(axis=1).astype(int).values,
        "cast": ap[["cast"]].join(lat[["cast"]], rsuffix="_lat").max(axis=1).astype(int).values,
    })
    for c in classes:
        pairs[f"state_{c}"] = (ap[f"has_{c}"].astype(int).astype(str).values
                               + lat[f"has_{c}"].astype(int).astype(str).values)
    return pairs.sort_values("pair_id").reset_index(drop=True)


# ---------------------------------------------------------------- EDA tables

def eda_pairing(index: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """EDA-5: how many study-sides and images end up in a pair."""
    base = base_images(index)
    rows = []
    for split in SPLITS + ("all",):
        b = base if split == "all" else base[base["split"] == split]
        p = pairs if split == "all" else pairs[pairs["split"] == split]
        sides = b["study_side_id"].nunique()
        rows.append({
            "split": split,
            "study_sides": sides,
            "pairs": len(p),
            "pct_study_sides_paired": round(100 * len(p) / max(sides, 1), 2),
            "images": len(b),
            "images_in_pairs": 2 * len(p),
            "pct_images_in_pairs": round(100 * 2 * len(p) / max(len(b), 1), 2),
            "unpaired_images": len(b) - 2 * len(p),
        })
    return pd.DataFrame(rows)


def eda_class_by_view(index: pd.DataFrame, classes: list) -> pd.DataFrame:
    """EDA-6: in which view each class is annotated (all base images)."""
    base = base_images(index)
    n_view = base["view"].value_counts()
    rows = []
    for c in classes:
        has = base[base[f"has_{c}"] == 1]["view"].value_counts()
        ap, lat, other = (int(has.get(v, 0)) for v in ("ap", "lat", "other"))
        rows.append({
            "class": c,
            "ap_images": ap,
            "lat_images": lat,
            "other_images": other,
            "pct_of_ap_images": round(100 * ap / max(int(n_view.get("ap", 0)), 1), 2),
            "pct_of_lat_images": round(100 * lat / max(int(n_view.get("lat", 0)), 1), 2),
            "lat_share_of_class": round(100 * lat / max(ap + lat, 1), 1),
        })
    return pd.DataFrame(rows)


def eda_states(pairs: pd.DataFrame, classes: list, split: str = "all") -> pd.DataFrame:
    """EDA-7: four-state counts per class over the pairs of one split (or all)."""
    p = pairs if split == "all" else pairs[pairs["split"] == split]
    rows = []
    for c in classes:
        n = p[f"state_{c}"].value_counts()
        n00, n10, n01, n11 = (int(n.get(s, 0)) for s in STATES)
        present = n10 + n01 + n11
        one = n10 + n01
        rows.append({
            "split": split, "class": c, "n00": n00, "n10": n10, "n01": n01, "n11": n11,
            "present": present,
            "one_view_only": one,
            "pct_one_view_of_present": round(100 * one / present, 1) if present else 0.0,
            "pct_one_view_of_pairs": round(100 * one / max(len(p), 1), 2),
        })
    return pd.DataFrame(rows)


def eda_partner_lift(pairs: pd.DataFrame, classes: list) -> pd.DataFrame:
    """EDA-7b: how much the partner's label says about the target's label.

    p_ap is the share of pairs with the class in the AP image.
    p_ap_given_lat is that share among pairs where the LAT image has the class.
    A large gap between the two means the partner view is informative.
    """
    rows = []
    for c in classes:
        s = pairs[f"state_{c}"]
        ap = s.str[0] == "1"
        lat = s.str[1] == "1"
        both = int((ap & lat).sum())
        rows.append({
            "class": c,
            "p_ap": round(float(ap.mean()), 4),
            "p_ap_given_lat": round(both / int(lat.sum()), 4) if lat.any() else None,
            "p_ap_given_no_lat": round(int((ap & ~lat).sum()) / int((~lat).sum()), 4) if (~lat).any() else None,
            "p_lat": round(float(lat.mean()), 4),
            "p_lat_given_ap": round(both / int(ap.sum()), 4) if ap.any() else None,
            "p_lat_given_no_ap": round(int((lat & ~ap).sum()) / int((~ap).sum()), 4) if (~ap).any() else None,
        })
    return pd.DataFrame(rows)


def eda_exam_type(pairs: pd.DataFrame, focus_class: str = "fracture") -> pd.DataFrame:
    """EDA-8: pairs by exam type, with the focus class's one-view-only share."""
    label = {(1, 0): "initial, no cast", (0, 0): "follow-up, no cast",
             (0, 1): "follow-up, cast", (1, 1): "initial, cast"}
    rows = []
    for (initial, cast), g in pairs.groupby(["initial_exam", "cast"]):
        s = g[f"state_{focus_class}"]
        present = int((s != "00").sum())
        one = int(s.isin(["10", "01"]).sum())
        row = {"exam_type": label.get((int(initial), int(cast)), f"{initial},{cast}"), "pairs": len(g)}
        for split in SPLITS:
            row[split] = int((g["split"] == split).sum())
        row[f"{focus_class}_present"] = present
        row[f"{focus_class}_one_view_only"] = one
        row["pct_one_view_of_present"] = round(100 * one / present, 1) if present else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pairs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Gate 1

def gate1(index: pd.DataFrame, pairs: pd.DataFrame, classes: list, rule: dict) -> tuple[pd.DataFrame, list, bool]:
    """Gate 1: are there enough pairs, and enough classes with one-view-only cases?

    A class 'has one-view-only mass' when, in the training pairs, it has at least
    `min_one_view_pairs_train` one-view-only pairs and they are at least
    `min_one_view_share` of the pairs where the class is present.
    Returns (per-class table, summary lines, passed).
    """
    train = eda_states(pairs, classes, "train")
    train["enough_pairs"] = train["one_view_only"] >= int(rule["min_one_view_pairs_train"])
    train["enough_share"] = train["pct_one_view_of_present"] >= 100 * float(rule["min_one_view_share"])
    train["has_mass"] = train["enough_pairs"] & train["enough_share"]
    table = train[["class", "present", "one_view_only", "pct_one_view_of_present",
                   "enough_pairs", "enough_share", "has_mass"]]

    share = 2 * len(pairs) / max(len(base_images(index)), 1)
    ok_share = share >= float(rule["min_pair_image_share"])
    with_mass = table.loc[table["has_mass"], "class"].tolist()
    ok_classes = len(with_mass) >= int(rule["min_classes"])
    lines = [
        f"[{'ok' if ok_share else 'FAIL'}] images in pairs: {100 * share:.1f}% (need {100 * float(rule['min_pair_image_share']):.0f}%)",
        f"[{'ok' if ok_classes else 'FAIL'}] classes with one-view-only mass: {len(with_mass)} {with_mass} (need {int(rule['min_classes'])})",
    ]
    return table, lines, bool(ok_share and ok_classes)
