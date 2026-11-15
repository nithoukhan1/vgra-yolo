"""STEP-02: build AP/LAT pairs from index.csv and print EDA tables 5-8 and Gate 1.

Run:  python scripts/02_pair_audit.py --config configs/base.yaml
Needs index.csv from STEP-01. It is looked for, in this order, at:
  --index, data.index_csv, data.index_input
Writes pairs.csv and the EDA tables, then prints 'GATE 1: PASS' or 'GATE 1: FAIL'.
The thresholds are starting values from the config; the final decision is ours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vgra import pairs as pr
from vgra import utils

REPO_ROOT = Path(__file__).resolve().parents[1]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))


def find_index(args_index: str | None, data: dict) -> Path:
    candidates = [args_index, data.get("index_csv"), data.get("index_input")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise SystemExit(
        "index.csv not found. Looked at: "
        + ", ".join(str(c) for c in candidates if c)
        + "\nRun scripts/01_build_index.py first, or attach the Kaggle dataset that holds index.csv, "
          "or pass --index <path>."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--index", default=None, help="path to index.csv")
    ap.add_argument("--out", default=None, help="override the folder for pairs.csv and eda/")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    classes = list(data["classes"])
    pairs_csv = Path(data["pairs_csv"])
    eda_dir = Path(data["eda_dir"])
    if args.out:
        pairs_csv = Path(args.out) / "pairs.csv"
        eda_dir = Path(args.out) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)

    index_path = find_index(args.index, data)
    print(f"repo commit: {utils.git_commit()}")
    print(f"index      : {index_path}")
    index = pr.read_index(index_path)
    pairs = pr.build_pairs(index, classes)
    pairs.to_csv(pairs_csv, index=False)
    print(f"wrote {pairs_csv} ({len(pairs)} pairs)")

    states = pd.concat([pr.eda_states(pairs, classes, s) for s in ("all",) + pr.SPLITS], ignore_index=True)
    gate_table, gate_lines, gate_ok = pr.gate1(index, pairs, classes, data["gate1"])
    # cross-class table: leave out text (present everywhere) and foreignbody (4 pairs)
    cc_classes = [c for c in classes if c not in ("text", "foreignbody")]
    cc_lat, cc_counts_ap = pr.eda_cross_class(pairs, cc_classes, target_view="lat")
    cc_ap, cc_counts_lat = pr.eda_cross_class(pairs, cc_classes, target_view="ap")
    tables = {
        "eda5_pairing": ("EDA-5 pairing", pr.eda_pairing(index, pairs)),
        "eda6_class_by_view": ("EDA-6 class presence by view (all base images)", pr.eda_class_by_view(index, classes)),
        "eda7_states": ("EDA-7 four-state counts per class (all pairs, then per split)", states),
        "eda7b_partner_lift": ("EDA-7b what the partner's label says about the target's", pr.eda_partner_lift(pairs, classes)),
        "eda7c_partner_counts": ("EDA-7c pairs where the AP image has the class", cc_counts_ap),
        "eda7c_cross_class_target_lat": (
            "EDA-7c target = LAT image: P(LAT has row class | AP partner has column class)", cc_lat),
        "eda7c_partner_counts_lat": ("EDA-7c pairs where the LAT image has the class", cc_counts_lat),
        "eda7c_cross_class_target_ap": (
            "EDA-7c target = AP image: P(AP has row class | LAT partner has column class)", cc_ap),
        "eda8_exam_type": ("EDA-8 pairs by exam type", pr.eda_exam_type(pairs)),
        "gate1_classes": ("Gate 1 per class (training pairs)", gate_table),
    }
    for name, (title, table) in tables.items():
        table.to_csv(eda_dir / f"{name}.csv", index=False)
        show(title, table)
    print(f"\nwrote {len(tables)} tables to {eda_dir}")

    print("\n=== Gate 1 (starting thresholds from the config) ===")
    for line in gate_lines:
        print(line)
    print("GATE 1: PASS" if gate_ok else "GATE 1: FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
