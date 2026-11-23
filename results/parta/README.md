# Part A results (aggregate tables only)

`test/` = the one-time confirmatory test session (2026-11-21, tag `parta-test-v1` = 26ede6c; test split opened at
06:19:09 UTC by I1, 06:25:07 by I3, 06:34:20 by I4; see the TEST_OPENED.json files). `valid_dev/` = the full dry run
on the validation split with the same code and the same 8 frozen detectors (development numbers: thresholds and results
on the same split, not for claims).

| Folder | Content | Script |
|---|---|---|
| `test/study/` | patient-level analysis: `summary.txt` (printed tables), `settings.csv` (sensitivity, specificity, missed per 100, AUC per run, finding and setting), `contrasts.csv` (differences with CIs, McNemar, DeLong, Holm), `subgroups.csv`, `seeds.csv` (mean over seeds, pooled bootstrap) | `scripts/12_study_level.py` (I2) |
| `test/detectors/` | Ultralytics validation on test: `detectors.csv`, `per_class.csv`, one json per run | `scripts/11_val_detectors.py` (I4) |
| `test/premise/` | RQ1 confirmatory premise test on the test pairs (`premise.csv`) | `scripts/04_premise_test.py` (I3) |
| `test/predictions_info/` | run information of the prediction runner per run and split (commit, weights, settings, counts) | `scripts/10_predict.py` (I1) |
| `valid_dev/study/`, `valid_dev/detectors/` | the same tables on the validation split (dry run 2026-11-21) | same |

The per-image prediction tables (`*_images.csv.gz`, `*_detections.csv.gz`) and the pair lists stay outside git (they
carry patient identifiers of the public dataset and are several MB); they are in the Kaggle outputs. Detector weights:
SHA-256 in `docs/final_weights.json`. Kaggle account names in paths are replaced by `<owner>`.
