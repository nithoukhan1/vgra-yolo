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
