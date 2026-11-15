"""Image index (STEP-01).

Builds one table with one row per image file on disk, joined with the dataset
metadata (dataset.csv) and the split files (train/valid/test CSVs).

Terms
- filestem : image name in dataset.csv, e.g. 0001_1297860395_01_WRI-L1_M014
- image_id : file name on disk without extension (may carry a prefix, e.g. orig_ or aug_)
- variant  : "orig" (original image) or "aug" (brightness/contrast copy made offline)
- is_base  : True for exactly one row per filestem (the original). EDA and pairs use base rows.
- study_side_id : patient + study + wrist side; the unit inside which AP/LAT pairs are formed
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

FILESTEM_RE = re.compile(
    r"^(?P<patient>\d+)_(?P<timehash>\d+)_(?P<study>\d+)_WRI-"
    r"(?P<side>[LR])(?P<projection>\d)_(?P<sex>[A-Z])(?P<age>\d+)$"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
META_COLUMNS = [
    "filestem", "patient_id", "study_number", "laterality", "projection",
    "gender", "age", "initial_exam", "cast", "fracture_visible", "metal",
]
FLAG_COLUMNS = ["initial_exam", "cast", "fracture_visible", "metal"]


# ---------------------------------------------------------------- small helpers

def parse_filestem(stem: str) -> dict | None:
    """Split a filestem into its parts, or None if it does not follow the pattern."""
    m = FILESTEM_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "patient_id": int(d["patient"]),
        "study_number": int(d["study"]),
        "laterality": d["side"],
        "projection": int(d["projection"]),
    }


def split_variant(image_id: str, prefixes: dict) -> tuple[str, str]:
    """Return (variant, filestem) for a file name. No known prefix means an original."""
    for prefix, variant in prefixes.items():
        if image_id.startswith(prefix):
            return str(variant), image_id[len(prefix):]
    return "orig", image_id


def read_label(path: str | os.PathLike, n_classes: int) -> tuple[bool, list, str]:
    """Read one YOLO label file.

    Returns (found, boxes per class, short hash of the text).
    A missing file gives (False, zeros, ""). An empty file is a valid "no boxes" label.
    """
    counts = [0] * n_classes
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return False, counts, ""
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if not 0 <= cls < n_classes:
            raise ValueError(f"class id {cls} out of range in {path}")
        counts[cls] += 1
    digest = hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]
    return True, counts, digest


def locate_index(data_cfg: dict, override: str | None = None) -> Path:
    """Find index.csv: an explicit path, then data.index_csv, then data.index_input."""
    candidates = [override, data_cfg.get("index_csv"), data_cfg.get("index_input")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "index.csv not found. Looked at: " + ", ".join(str(c) for c in candidates if c)
        + ". Attach the Kaggle dataset that holds index.csv, or pass --index <path>."
    )


# ---------------------------------------------------------------- loading

def load_metadata(data_cfg: dict) -> pd.DataFrame:
    """dataset.csv plus a `split` column taken from the three split CSV files."""
    root = Path(data_cfg["root"])
    meta = pd.read_csv(root / data_cfg["meta_csv"], encoding="utf-8-sig")
    missing = [c for c in META_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"{data_cfg['meta_csv']} is missing columns: {missing}")
    if meta["filestem"].duplicated().any():
        raise ValueError("duplicate filestem in metadata")

    split_of: dict = {}
    for split, fname in data_cfg["split_csvs"].items():
        part = pd.read_csv(root / fname, encoding="utf-8-sig", usecols=["filestem"])
        for stem in part["filestem"]:
            if stem in split_of:
                raise ValueError(f"{stem} appears in two split files ({split_of[stem]} and {split})")
            split_of[stem] = split

    meta = meta[META_COLUMNS].copy()
    meta["split"] = meta["filestem"].map(split_of)
    return meta


def scan_images(data_cfg: dict) -> pd.DataFrame:
    """List every image file in the split folders."""
    root = Path(data_cfg["root"]) / data_cfg["data_dir"]
    prefixes = data_cfg.get("variant_prefixes") or {}
    rows = []
    for split, folder in data_cfg["folders"].items():
        img_dir = root / "images" / folder
        lbl_dir = root / "labels" / folder
        with os.scandir(img_dir) as it:
            for entry in it:
                stem, ext = os.path.splitext(entry.name)
                if not entry.is_file() or ext.lower() not in IMAGE_EXTS:
                    continue
                variant, filestem = split_variant(stem, prefixes)
                rows.append({
                    "image_id": stem,
                    "filestem": filestem,
                    "variant": variant,
                    "folder": folder,
                    "folder_split": split,
                    "path": (img_dir / entry.name).as_posix(),
                    "label_path": (lbl_dir / (stem + ".txt")).as_posix(),
                })
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


# ---------------------------------------------------------------- index

def build_index(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (index, metadata). The index has one row per image file on disk."""
    data_cfg = cfg["data"]
    classes = list(data_cfg["classes"])
    views = {int(k): str(v) for k, v in (data_cfg.get("views") or {}).items()}

    meta = load_metadata(data_cfg)
    index = scan_images(data_cfg).merge(meta, on="filestem", how="left", indicator=True)
    index["in_metadata"] = index.pop("_merge").eq("both")

    # labels
    found, digests, counts = [], [], []
    for path in index["label_path"]:
        ok, c, h = read_label(path, len(classes))
        found.append(ok)
        counts.append(c)
        digests.append(h)
    index["label_found"] = found
    index["label_hash"] = digests
    count_df = pd.DataFrame(counts, columns=[f"n_{c}" for c in classes], index=index.index)
    index = pd.concat([index, count_df], axis=1)
    index["n_boxes"] = count_df.sum(axis=1)
    for c in classes:
        index[f"has_{c}"] = (index[f"n_{c}"] > 0).astype(int)

    # derived columns
    for col in FLAG_COLUMNS:
        index[col] = index[col].fillna(0).astype(int)
    index["view"] = index["projection"].map(views).fillna("other")
    index.loc[~index["in_metadata"], "view"] = ""
    known = index["in_metadata"]
    index["study_side_id"] = ""
    index.loc[known, "study_side_id"] = (
        index.loc[known, "patient_id"].astype(int).map("{:04d}".format)
        + "_" + index.loc[known, "study_number"].astype(int).map("{:02d}".format)
        + "_" + index.loc[known, "laterality"].astype(str)
    )

    # one base row per filestem: the original if present, else the first variant
    order = index.assign(_not_orig=index["variant"].ne("orig")).sort_values(["filestem", "_not_orig", "image_id"])
    index["is_base"] = False
    index.loc[order.drop_duplicates("filestem").index, "is_base"] = True

    front = [
        "image_id", "filestem", "variant", "is_base", "split", "folder", "folder_split",
        "patient_id", "study_number", "laterality", "study_side_id", "projection", "view",
        "gender", "age", "initial_exam", "cast", "fracture_visible", "metal",
        "in_metadata", "label_found", "label_hash", "n_boxes",
    ]
    rest = [c for c in index.columns if c not in front and c not in ("path", "label_path")]
    return index[front + rest + ["path", "label_path"]], meta


# ---------------------------------------------------------------- checks

def check_index(index: pd.DataFrame, meta: pd.DataFrame, expect: dict | None = None) -> list:
    """Pass/fail checks. Returns a list of (name, ok, detail)."""
    out = []
    base = index[index["is_base"] & index["in_metadata"]]

    missing = sorted(set(meta["filestem"]) - set(base["filestem"]))
    out.append(("every metadata row has an image", not missing, f"missing: {len(missing)} {missing[:3]}"))

    orphans = index.loc[~index["in_metadata"], "image_id"].tolist()
    out.append(("every image has a metadata row", not orphans, f"without metadata: {len(orphans)} {orphans[:3]}"))

    unassigned = int(meta["split"].isna().sum())
    out.append(("every metadata row has a split", unassigned == 0, f"unassigned: {unassigned}"))

    known = index[index["in_metadata"]]
    wrong = known[known["split"] != known["folder_split"]]
    out.append(("folder agrees with split file", wrong.empty, f"disagreements: {len(wrong)} {wrong['image_id'].head(3).tolist()}"))

    per_patient = meta.dropna(subset=["split"]).groupby("patient_id")["split"].nunique()
    leaked = int((per_patient > 1).sum())
    out.append(("no patient in two splits", leaked == 0, f"patients in more than one split: {leaked}"))

    no_label = int((~index["label_found"]).sum())
    out.append(("every image has a label file", no_label == 0, f"label files missing: {no_label}"))

    hashes = known.groupby("filestem")["label_hash"].nunique()
    differ = int((hashes > 1).sum())
    out.append(("copies share the original's labels", differ == 0, f"filestems whose variants differ: {differ}"))

    parsed = base["filestem"].map(parse_filestem)
    bad_name = int(parsed.isna().sum())
    disagree = 0
    for key in ("patient_id", "study_number", "laterality", "projection"):
        got = parsed.map(lambda d, k=key: d[k] if d else None)
        ok_rows = got.notna()
        want = base.loc[ok_rows, key]
        if key == "laterality":
            disagree += int((got[ok_rows].astype(str) != want.astype(str)).sum())
        else:  # numeric fields: compare as integers (the column may be float after a merge)
            disagree += int((got[ok_rows].astype(int) != want.astype(int)).sum())
    out.append(("file names agree with metadata", bad_name == 0 and disagree == 0,
                f"names off-pattern: {bad_name}, fields disagreeing: {disagree}"))

    if expect:
        got = {"images": len(base)}
        got.update(base.groupby("split").size().to_dict())
        diffs = {k: (got.get(k, 0), v) for k, v in expect.items() if int(got.get(k, 0)) != int(v)}
        out.append(("totals match the expected numbers", not diffs, f"(found, expected) where different: {diffs}"))
    return out


# ---------------------------------------------------------------- EDA tables (base rows only)

def _base(index: pd.DataFrame) -> pd.DataFrame:
    return index[index["is_base"] & index["in_metadata"]]


def eda_counts(index: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """EDA-1: what is on disk and how it lines up with the metadata."""
    base = _base(index)
    rows = [
        ("metadata rows", len(meta)),
        ("patients", meta["patient_id"].nunique()),
        ("image files on disk", len(index)),
        ("base images (one per filestem)", len(base)),
    ]
    for (folder, variant), n in index.groupby(["folder", "variant"]).size().items():
        rows.append((f"files in {folder} / variant {variant}", n))
    for split in ("train", "valid", "test"):
        part = base[base["split"] == split]
        rows.append((f"{split}: images", len(part)))
        rows.append((f"{split}: patients", part["patient_id"].nunique()))
        rows.append((f"{split}: images with no boxes", int((part["n_boxes"] == 0).sum())))
    rows.append(("boxes in base images", int(base["n_boxes"].sum())))
    return pd.DataFrame(rows, columns=["item", "value"])


def eda_class_freq(index: pd.DataFrame, classes: list) -> pd.DataFrame:
    """EDA-2: boxes and images per class, per split."""
    base = _base(index)
    rows = []
    for c in classes:
        row = {"class": c}
        for split in ("train", "valid", "test"):
            part = base[base["split"] == split]
            row[f"{split}_boxes"] = int(part[f"n_{c}"].sum())
            row[f"{split}_images"] = int(part[f"has_{c}"].sum())
        row["all_boxes"] = int(base[f"n_{c}"].sum())
        row["all_images"] = int(base[f"has_{c}"].sum())
        row["pct_images"] = round(100 * row["all_images"] / max(len(base), 1), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def eda_views(index: pd.DataFrame) -> pd.DataFrame:
    """EDA-3: images per projection code and view, per split."""
    base = _base(index)
    table = base.groupby(["projection", "view", "split"]).size().unstack("split", fill_value=0)
    table = table.reindex(columns=[c for c in ("train", "valid", "test") if c in table.columns])
    table["all"] = table.sum(axis=1)
    return table.reset_index()


def eda_studies(index: pd.DataFrame, top: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EDA-4: images per study-side, and which view combinations occur."""
    base = _base(index)
    groups = base.groupby("study_side_id")
    size = groups.size().rename("images")
    split = groups["split"].first()
    per_size = pd.crosstab(size, split).reindex(columns=["train", "valid", "test"], fill_value=0)
    per_size["all"] = per_size.sum(axis=1)
    per_size = per_size.reset_index().rename(columns={"images": "images_per_study_side"})

    def combo(views: pd.Series) -> str:
        counts = views.value_counts()
        return " + ".join(f"{int(counts[v])} {v}" for v in ("ap", "lat", "other") if v in counts)

    combos = groups["view"].apply(combo).value_counts()
    combo_df = combos.head(top).rename_axis("views_in_study_side").reset_index(name="study_sides")
    if len(combos) > top:
        extra = pd.DataFrame([{"views_in_study_side": "(other combinations)", "study_sides": int(combos.iloc[top:].sum())}])
        combo_df = pd.concat([combo_df, extra], ignore_index=True)
    combo_df["pct"] = (100 * combo_df["study_sides"] / combos.sum()).round(2)
    return per_size, combo_df
