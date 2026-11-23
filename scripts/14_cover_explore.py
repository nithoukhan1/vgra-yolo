"""EXPLORATORY (post hoc, after the CoVeR test run of 2026-11-23): why did validation under-estimate the test risk?

Compares the valid and test splits on the fracture-positive pairs, from the prediction tables already produced:
  label composition  share annotated on PA only (10), LAT only (01), both (11); torus; first visit; torus at first
                     visit; share of patients with a single positive pair (they weigh most in the patient-unit risk)
  model-based        per detector (threshold t from valid, as in Part A): share of fracture pairs that only the
                     lateral view catches (PA <= t < LAT), and the part of them the PA view practically misses
                     (PA score <= --blind, default 0.005, the order of CoVeR's lo)
Nothing here changes any frozen analysis; every number is labelled exploratory.

    python scripts/14_cover_explore.py --pred <test-session runs/predictions> \
        --runs B0_s0 B0_s1 B0_s2 V8_s0 V8_s1 V8R_s0 V8R_s1 --open-test --out results/cover/test_exploratory
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--blind", type=float, default=0.005, help="PA score at or below which the PA view 'misses'")
    ap.add_argument("--open-test", action="store_true", help="required: reads the (already opened) test split")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.open_test:
        raise SystemExit("this exploratory analysis reads the test split; add --open-test")
    import numpy as np
    import pandas as pd

    from vgra import study as st
    from vgra import utils

    classes = list(utils.load_config(args.config)["data"]["classes"])
    pred, out = Path(args.pred), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = {s: pd.read_csv(pred / f"pairs_{s}.csv", dtype={"pair_id": str, "laterality": str}) for s in ("valid", "test")}
    tabs = {(r, s): st.study_table(pd.read_csv(pred / r / f"{s}_images.csv.gz"), pairs[s], classes, findings=("fracture",))
            for r in args.runs for s in ("valid", "test")}

    rows = []
    for s in ("valid", "test"):
        t0 = tabs[(args.runs[0], s)]
        pos = t0[t0["y"] == 1]
        state = (pos["y_pa"].astype(str) + pos["y_lat"].astype(str)).value_counts(normalize=True)
        per_patient = pos.groupby("patient_id").size()
        rows.append({"split": s, "measure": "fracture-positive pairs", "value": len(pos)})
        rows.append({"split": s, "measure": "patients with a positive pair", "value": pos["patient_id"].nunique()})
        for code, name in (("10", "annotated on PA only"), ("01", "annotated on LAT only"), ("11", "annotated on both")):
            rows.append({"split": s, "measure": name, "value": float(state.get(code, 0.0))})
        for col, name in (("torus", "torus"), ("initial_exam", "first visit")):
            if col in pos:
                rows.append({"split": s, "measure": name, "value": float(pos[col].mean())})
        if {"torus", "initial_exam"} <= set(pos.columns):
            rows.append({"split": s, "measure": "torus at first visit",
                         "value": float(((pos["torus"] == 1) & (pos["initial_exam"] == 1)).mean())})
        rows.append({"split": s, "measure": "patients with one positive pair", "value": float((per_patient == 1).mean())})
        for r in args.runs:
            tv, te = tabs[(r, "valid")], tabs[(r, s)]
            t = st.threshold_at(np.maximum(tv["pa"], tv["lat"])[tv["y"] == 0])
            p = te[te["y"] == 1]
            lateral_only = (p["pa"] <= t) & (p["lat"] > t)
            rows.append({"split": s, "measure": f"{r}: only the lateral view catches", "value": float(lateral_only.mean())})
            rows.append({"split": s, "measure": f"{r}: ... and PA score <= {args.blind}",
                         "value": float((lateral_only & (p["pa"] <= args.blind)).mean())})
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="measure", columns="split", values="value", sort=False).reset_index()
    for label in ("only the lateral view catches", f"PA score <= {args.blind}"):
        sub = df[df["measure"].str.endswith(label)]
        mean = sub.groupby("split")["value"].mean()
        wide = pd.concat([wide, pd.DataFrame([{"measure": f"MEAN over runs: {label}", **mean.to_dict()}])],
                         ignore_index=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"EXPLORATORY (post hoc) | repo commit {utils.git_commit()} | runs {' '.join(args.runs)} | blind <= {args.blind}")
        pd.set_option("display.width", 200)
        print(wide.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    text = buf.getvalue()
    print(text)
    wide.to_csv(out / "composition.csv", index=False)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(f"EXPLORATORY COMPOSITION DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
