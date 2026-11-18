"""I2: Part A patient (study) level analysis (docs/PROTOCOL.md section 4). CPU only.

Reads the prediction tables of scripts/10_predict.py and answers, per detector run and finding:
PA only vs LAT only vs both projections combined (max = primary rule; mean, noisy-OR, stacking),
at the 95%-specificity threshold fixed on valid.

    python scripts/12_study_level.py --pred <folder with pairs_valid.csv and <run>/valid_images.csv.gz> \
        --runs B0_s0 B1_s0 --eval valid
    python scripts/12_study_level.py --pred <predictions> --runs B0_s0 B0_s1 B0_s2 B1_s0 V8_s0 V8_s1 V8R_s0 V8R_s1 --eval test

--eval valid  = development: thresholds and numbers on the same split (stacking out-of-fold). Not for claims.
--eval test   = confirmatory: thresholds and stacking from valid, numbers on test (needs the test predictions).

Writes to <out>/study_<eval>/:
    settings.csv     run x finding x setting: threshold, sensitivity [95% CI], specificity [CI], missed per 100, AUC
    contrasts.csv    run x finding x contrast: sensitivity and specificity differences [CI] with McNemar,
                     AUC difference (DeLong), Holm
    subgroups.csv    run x finding x subgroup: sensitivity PA / LAT / both, difference [CI]
    seeds.csv        detector x finding x contrast: mean +- SD over seeds and the pooled-bootstrap CI
    summary.txt      the printed tables
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FAMILY = "B0"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--pred", required=True, help="folder written by scripts/10_predict.py (predictions/)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", required=True, choices=["valid", "test"])
    ap.add_argument("--boot", type=int, default=2000, help="patient-bootstrap resamples (protocol: 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the --pred folder")
    return ap.parse_args()


def family(run: str) -> str:
    return run.rsplit("_s", 1)[0]


def show(df, cols, title):
    import pandas as pd

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"\n=== {title} ===")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}") if len(df) else "(none)")


def main() -> int:
    args = parse_args()
    import pandas as pd

    from vgra import study as st
    from vgra import utils

    cfg = utils.load_config(args.config)
    classes = list(cfg["data"]["classes"])
    pred = Path(args.pred)
    out = Path(args.out or pred) / f"study_{args.eval}"
    out.mkdir(parents=True, exist_ok=True)
    same = args.eval == "valid"

    pairs_v = pd.read_csv(pred / "pairs_valid.csv", dtype={"pair_id": str, "laterality": str})
    pairs_e = pairs_v if same else pd.read_csv(pred / f"pairs_{args.eval}.csv", dtype={"pair_id": str, "laterality": str})

    buf = io.StringIO()
    settings, contrasts, subs, eval_tabs, decisions, infos = [], [], [], {}, {}, {}
    with redirect_stdout(buf):
        print(f"repo commit: {utils.git_commit()} | eval split: {args.eval} "
              f"({'DEVELOPMENT: thresholds and numbers on the same split' if same else 'CONFIRMATORY: thresholds from valid'})"
              f" | bootstrap {args.boot} | pairs valid {len(pairs_v)}, {args.eval} {len(pairs_e)}")
        if args.boot != 2000:
            print("NOTE: bootstrap count differs from the protocol (2000); not for the paper")
        for run in args.runs:
            imgs_v = pd.read_csv(pred / run / "valid_images.csv.gz")
            imgs_e = imgs_v if same else pd.read_csv(pred / run / f"{args.eval}_images.csv.gz")
            for split in {"valid", args.eval}:
                p = pred / run / f"{split}_info.json"
                if p.is_file():
                    infos[(run, split)] = json.loads(p.read_text())
                    if infos[(run, split)].get("smoke"):
                        print(f"NOTE: {run} {split} predictions are a SMOKE TEST; not for the paper")
            tab_v = st.study_table(imgs_v, pairs_v, classes)
            tab_e = tab_v if same else st.study_table(imgs_e, pairs_e, classes)
            s, c, dec = st.evaluate_run(tab_v, tab_e, run, same, args.boot, args.seed)
            settings.append(s)
            contrasts.append(c)
            subs.append(st.subgroups(tab_e, dec, run, args.boot, args.seed))
            eval_tabs[run], decisions[run] = tab_e, dec
        settings = pd.concat(settings, ignore_index=True)
        contrasts = pd.concat(contrasts, ignore_index=True)
        subs = pd.concat(subs, ignore_index=True)

        seeds = []
        fams = {}
        for r in args.runs:
            fams.setdefault(family(r), []).append(r)
        for fam, runs in fams.items():
            for f in st.FINDINGS:
                for new, ref in (("max", "pa"), ("max", "lat")):
                    if len(runs) >= 2:
                        res = st.pooled_seed_bootstrap({r: eval_tabs[r] for r in runs},
                                                       {r: decisions[r] for r in runs}, f, new, ref, args.boot, args.seed)
                    else:
                        row = contrasts[(contrasts["run"] == runs[0]) & (contrasts["finding"] == f)
                                        & (contrasts["contrast"] == f"{new}_vs_{ref}")].iloc[0]
                        res = {"finding": f, "contrast": f"{new}_vs_{ref}", "seeds": 1, "runs": runs[0],
                               "diff_mean": row["sens_diff"], "diff_sd": float("nan"),
                               "pooled_lo": row["sens_diff_lo"], "pooled_hi": row["sens_diff_hi"]}
                    seeds.append({"detector": fam, **res})
        seeds = pd.DataFrame(seeds)

        prim = contrasts[contrasts["primary"]].copy()
        prim["detector"] = prim["run"].map(family)
        show(prim, ["run", "sens_ref", "sens_new", "sens_diff", "sens_diff_lo", "sens_diff_hi", "mcnemar_b",
                    "mcnemar_c", "mcnemar_p", "mcnemar_method", "spec_ref", "spec_new", "spec_diff_lo", "spec_diff_hi",
                    "spec_mcnemar_p", "auc_ref", "auc_new", "delong_p"],
             "PRIMARY: fracture, sensitivity at 95% specificity, both projections (max) vs PA only")
        show(seeds[(seeds["finding"] == st.PRIMARY_FINDING) & (seeds["contrast"] == "max_vs_pa")],
             ["detector", "seeds", "diff_mean", "diff_sd", "pooled_lo", "pooled_hi"],
             "PRIMARY, over seeds (mean +- SD of the difference; pooled patient bootstrap 95% CI)")
        main_set = settings[settings["setting"].isin(["pa", "lat", "max"])]
        show(main_set, ["run", "finding", "setting", "n_pos", "n_neg", "sens", "sens_lo", "sens_hi", "spec",
                        "missed_per_100", "auc"], "Findings: PA only / LAT only / both (max)")
        show(contrasts[contrasts["contrast"].isin(["max_vs_pa", "max_vs_lat"])],
             ["run", "finding", "contrast", "sens_diff", "sens_diff_lo", "sens_diff_hi", "mcnemar_p", "mcnemar_p_holm",
              "auc_diff", "delong_p", "delong_p_holm"], "Contrasts (Holm within each contrast over the secondary findings)")
        show(settings[settings["finding"] == st.PRIMARY_FINDING],
             ["run", "setting", "sens", "spec", "missed_per_100", "auc"], "Fusion rules, fracture (RQ4)")
        show(subs[subs["finding"] == st.PRIMARY_FINDING],
             ["run", "subgroup", "level", "n_pos", "n_neg", "sens_pa", "sens_lat", "sens_max", "diff_max_vs_pa",
              "diff_max_vs_pa_lo", "diff_max_vs_pa_hi"], "Subgroups, fracture (exploratory)")

    text = buf.getvalue()
    print(text)
    settings.to_csv(out / "settings.csv", index=False)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    subs.to_csv(out / "subgroups.csv", index=False)
    seeds.to_csv(out / "seeds.csv", index=False)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(f"STUDY-LEVEL DONE: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
