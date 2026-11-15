# Decisions

One dated line per decision, with the reason. Newest at the bottom.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-15 | New clean implementation in this repo; earlier code is reference only | Clear naming and reproducible experiments |
| 2026-11-15 | Ultralytics installed from pip, pinned to 8.4.7, never edited; custom code only in `vgra` | Keeps our changes separate from library code |
| 2026-11-15 | Backbone add-on is named SCC (Self-Calibrated Convolution); block class `C3K2_SCC` at backbone layers 2 and 4 | Avoids confusion with the 2023 reconstruction convolution |
| 2026-11-15 | Detection head class is named `VGRADetect`; it holds CQP, VSH, GRA | One clear place for the cross-view method |
| 2026-11-15 | One YAML config per experiment; one experiment per GPU; every run writes `run_info.json` | Each result is tied to a commit and a config |
| 2026-11-15 | Repo is public; local pushes use SSH; Kaggle clones over HTTPS without a token | Simplest workflow |
| 2026-11-15 | Order: STEP-00 setup, then EDA (STEP-01, STEP-02) before any model code | The pair audit decides whether the idea has data to work with |
| 2026-11-15 | Split = patient-level split of Ju et al., read from `train_data.csv`, `valid_data.csv`, `test_data.csv`; their `split.py` is never re-run | The dataset has no official split; this one is public and patient-level; the script has no random seed, so only the files define it |
| 2026-11-15 | In `train_aug`, `orig_*` files are originals and `aug_*` files are offline brightness/contrast copies with identical labels; EDA and pairs use originals only | A copy is the same radiograph, not a new image or pair |
| 2026-11-15 | Pair unit is the study-side: patient + study + wrist side | Some studies contain both wrists |
