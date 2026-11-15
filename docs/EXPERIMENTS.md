# Experiment log

One row per run, including failed runs (say why in the note). Never delete a row.

| Run name | Commit | Date | Split | mAP50 | mAP50-95 | Status | Note |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Rules:
- A run counts only if its folder has `run_info.json`.
- Run name = `<expID>_s<seed>`, e.g. `V1_s0`.
- Per-class numbers live in `results/<run name>/metrics.json`.
