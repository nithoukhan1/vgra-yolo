# Experiment log

One row per run, including failed runs (say why in the note). Never delete a row.

| Run name | Commit | Date | Split | mAP50 | mAP50-95 | Status | Note |
|---|---|---|---|---|---|---|---|
| B1_s0_smoke640 | 2495910 | 2026-11-16 | valid | 0.2577 | 0.1663 | timing test | 1 epoch on 10% of train (1,420 images). 1.1 it/s; est. 13.2 min train + 2.5 min val per full epoch; peak GPU 4.3 GB. First run of the session: label scan took about 10 min (cold disk). Harmless dataloader-thread error at exit |
| B1_s0_smoke1024 | 2495910 | 2026-11-16 | valid | 0.2434 | 0.1648 | timing test | Same as above at 1024. 1.1 s/it; est. 15.6 min train + 2.1 min val per full epoch; peak GPU 10.8 GB. Files already cached, scan fast |

Timing-test mAP after one epoch on 10% of the data says nothing about the final accuracy of either image size.

Rules:
- A run counts only if its folder has `run_info.json`.
- Run name = `<expID>_s<seed>`, e.g. `V1_s0`.
- Per-class numbers live in `results/<run name>/metrics.json`.
