"""I6: CoVeR (conformal view referral) for the fracture finding. CPU only.

Reads the prediction tables of scripts/10_predict.py (the same inputs as scripts/12_study_level.py).

    python scripts/13_cover.py --pred <predictions> --runs B0_s0 B0_s1 B0_s2 B1_s0 V8_s0 V8_s1 V8R_s0 V8R_s1 \
        --eval valid --ensemble
    python scripts/13_cover.py --pred <predictions> --runs <frozen runs> --eval test --open-test --ensemble

--eval valid  = DESIGN STUDY (exploratory): repeated patient half-splits of valid (calibrate on one half,
                evaluate on the other) for every run x grouping x method x alpha; the risk / lateral-read curve on
                all of valid; who is referred under the primary configuration. Not for claims about test.
--eval test   = CONFIRMATORY USE, only after protocol amendment v1.2 freezes the rules: lo calibrated on all
                of valid, applied once to test, patient-bootstrap CIs. Refused without --open-test; writes
                COVER_TEST_RUN.json (time, commit, runs, configuration).

Writes to <out>/cover_<eval>/:
  valid: design_splits.csv.gz, design_summary.csv, curves.csv, who.csv, summary.txt, info.json
  test:  results.csv (every configuration; primary flagged), who.csv, summary.txt, COVER_TEST_RUN.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--pred", required=True, help="folder written by scripts/10_predict.py (predictions/)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", required=True, choices=["valid", "test"])
    ap.add_argument("--open-test", action="store_true", help="required for --eval test (after amendment v1.2)")
    ap.add_argument("--ensemble", action="store_true", help="also evaluate ENS = mean score of the given runs")
    ap.add_argument("--alphas", nargs="+", type=float, default=None, help="default: vgra.cover.ALPHAS")
    ap.add_argument("--delta", type=float, default=None, help="default: vgra.cover.DELTA")
    ap.add_argument("--splits", type=int, default=300, help="design study: patient half-splits of valid")
    ap.add_argument("--boot", type=int, default=2000, help="test: patient-bootstrap resamples (protocol: 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the --pred folder")
    return ap.parse_args()


def show(df, cols, title):
    import pandas as pd

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"\n=== {title} ===")
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}") if len(df) else "(none)")


def main() -> int:
    args = parse_args()
    if args.eval == "test" and not args.open_test:
        raise SystemExit("CoVeR on the test split is opened once, after protocol amendment v1.2 freezes its rules; "
                         "add --open-test then")
    import numpy as np
    import pandas as pd

    from vgra import cover as cv
    from vgra import study as st
    from vgra import utils

    alphas = tuple(args.alphas) if args.alphas else cv.ALPHAS
    delta = cv.DELTA if args.delta is None else args.delta
    cfg = utils.load_config(args.config)
    classes = list(cfg["data"]["classes"])
    pred = Path(args.pred)
    out = Path(args.out or pred) / f"cover_{args.eval}"
    out.mkdir(parents=True, exist_ok=True)
    commit = utils.git_commit()

    pairs_v = pd.read_csv(pred / "pairs_valid.csv", dtype={"pair_id": str, "laterality": str})
    tabs_v, tabs_e = {}, {}
    for run in args.runs:
        tabs_v[run] = cv.fracture_rows(st.study_table(pd.read_csv(pred / run / "valid_images.csv.gz"), pairs_v,
                                                      classes, findings=(cv.FINDING,)))
    if args.eval == "test":
        pairs_e = pd.read_csv(pred / "pairs_test.csv", dtype={"pair_id": str, "laterality": str})
        for run in args.runs:
            tabs_e[run] = cv.fracture_rows(st.study_table(pd.read_csv(pred / run / "test_images.csv.gz"), pairs_e,
                                                          classes, findings=(cv.FINDING,)))
    runs = list(args.runs)
    unc_v = cv.disagreement(tabs_v) if len(runs) >= 2 else None
    unc_e = cv.disagreement(tabs_e) if (len(runs) >= 2 and tabs_e) else None
    if args.ensemble and len(runs) >= 2:
        tabs_v["ENS"] = cv.ensemble_table({r: tabs_v[r] for r in runs})
        if tabs_e:
            tabs_e["ENS"] = cv.ensemble_table({r: tabs_e[r] for r in runs})
        runs.append("ENS")

    buf = io.StringIO()
    with redirect_stdout(buf):
        prim = cv.PRIMARY
        print(f"repo commit: {commit} | eval: {args.eval} | runs: {' '.join(runs)} | alphas {alphas} | delta {delta}"
              f" | primary (PROPOSAL until amendment v1.2): {prim}")
        if args.eval == "valid":
            print(f"DESIGN STUDY (exploratory): {args.splits} patient half-splits of valid; not for claims about test")
            design, curves, who = [], [], []
            for run in runs:
                ranks = {"uncert": unc_v} if (unc_v is not None and run != "ENS") else {}
                d = cv.design_study(tabs_v[run], alphas, cv.METHODS, cv.GROUPINGS, delta, args.splits, args.seed, ranks)
                d.insert(0, "run", run)
                design.append(d)
                c = cv.risk_referral_curve(tabs_v[run])
                c.insert(0, "run", run)
                curves.append(c)
                tv = tabs_v[run]
                t = st.threshold_at(np.maximum(tv["pa"], tv["lat"])[tv["y"] == 0])
                groups = tv["age_band"].values if prim["grouping"] == "age" else None
                lo = cv.calibrate(tv["pa"], tv["lat"], tv["y"], t, tv["patient_id"], prim["alpha"], prim["method"],
                                  delta, groups)
                w = cv.who_is_referred(tv, lo, prim["grouping"], t)
                w.insert(0, "run", run)
                who.append(w)
            design = pd.concat(design, ignore_index=True)
            summary = cv.design_summary(design)
            curves = pd.concat(curves, ignore_index=True)
            who = pd.concat(who, ignore_index=True)
            cols = ["run", "grouping", "method", "alpha", "refer_rate", "refer_p05", "refer_p95", "sens", "sens_both",
                    "sens_pa", "spec", "spec_both", "extra_patient", "extra_patient_p90", "held", "extra_worst_age",
                    "held_all_ages", "random_sens", "uncert_sens", "oracle_reads"]
            show(summary[(summary["method"] == prim["method"]) & (summary["alpha"] == prim["alpha"])
                         & (summary["grouping"] == prim["grouping"])], cols, "PRIMARY configuration (proposal), per run")
            show(summary[summary["run"] == ("ENS" if "ENS" in runs else runs[0])], cols,
                 f"All configurations, {'ENS' if 'ENS' in runs else runs[0]}")
            show(summary.groupby(["grouping", "method", "alpha"], sort=False)[
                     ["refer_rate", "sens", "sens_both", "extra_patient", "held", "held_all_ages", "random_sens"]]
                 .mean().reset_index(), ["grouping", "method", "alpha", "refer_rate", "sens", "sens_both",
                                         "extra_patient", "held", "held_all_ages", "random_sens"],
                 "All configurations, mean over runs")
            show(who[who["run"] == ("ENS" if "ENS" in runs else runs[0])],
                 ["run", "subgroup", "level", "n_pairs", "n_pos", "refer_rate", "extra_pair", "extra_patient"],
                 "Who is referred (primary configuration calibrated and applied on all of valid; descriptive)")
            design.to_csv(out / "design_splits.csv.gz", index=False)
            summary.to_csv(out / "design_summary.csv", index=False)
            curves.to_csv(out / "curves.csv", index=False)
            who.to_csv(out / "who.csv", index=False)
            (out / "info.json").write_text(json.dumps({"commit": commit, "runs": runs, "splits": args.splits,
                                                       "alphas": alphas, "delta": delta, "primary_proposal": prim,
                                                       "seed": args.seed}, indent=2))
        else:
            print("CONFIRMATORY USE: lo calibrated on all of valid, applied once to test")
            if args.boot != 2000:
                print("NOTE: bootstrap count differs from the protocol (2000); not for the paper")
            rows, who = [], []
            for run in runs:
                ranks = {"uncert": unc_e} if (unc_e is not None and run != "ENS") else None
                for grouping in cv.GROUPINGS:
                    for method in cv.METHODS:
                        for alpha in alphas:
                            r = cv.confirm(tabs_v[run], tabs_e[run], alpha, method, grouping, delta, args.boot,
                                           args.seed, ranks)
                            r["primary"] = (method == prim["method"] and alpha == prim["alpha"]
                                            and grouping == prim["grouping"])
                            rows.append({"run": run, **r})
                            if r["primary"]:
                                w = cv.who_is_referred(tabs_e[run], json.loads(r["lo_map"]), grouping, r["t"])
                                w.insert(0, "run", run)
                                who.append(w)
            res = pd.DataFrame(rows)
            who = pd.concat(who, ignore_index=True) if who else pd.DataFrame()
            show(res[res["primary"]], ["run", "lo", "refer_rate", "refer_rate_lo", "refer_rate_hi", "sens", "sens_lo",
                                       "sens_hi", "sens_both", "sens_pa", "spec", "spec_both", "extra_patient",
                                       "extra_patient_lo", "extra_patient_hi", "held", "sens_diff_random_lo",
                                       "sens_diff_random_hi", "mcnemar_pos_c", "mcnemar_neg_b"],
                 "PRIMARY configuration on test")
            show(res, ["run", "grouping", "method", "alpha", "refer_rate", "sens", "spec", "extra_patient", "held"],
                 "All configurations on test")
            show(who, ["run", "subgroup", "level", "n_pairs", "n_pos", "refer_rate", "extra_pair", "extra_patient"],
                 "Who is referred on test (primary configuration)")
            res.to_csv(out / "results.csv", index=False)
            who.to_csv(out / "who.csv", index=False)
            record = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
                      "runs": runs, "alphas": alphas, "delta": delta, "primary": prim, "boot": args.boot}
            (out / "COVER_TEST_RUN.json").write_text(json.dumps(record, indent=2))
            print(f"COVER TEST RUN RECORDED: {record}")

    text = buf.getvalue()
    print(text)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(f"COVER DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
