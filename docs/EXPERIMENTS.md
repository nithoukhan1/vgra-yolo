# Experiment log

One row per run, including failed runs (say why in the note). Never delete a row.

| Run name | Commit | Date | Split | mAP50 | mAP50-95 | Status | Note |
|---|---|---|---|---|---|---|---|
| B1_s0_smoke640 | 2495910 | 2026-11-16 | valid | 0.2577 | 0.1663 | timing test | 1 epoch on 10% of train (1,420 images). 1.1 it/s; est. 13.2 min train + 2.5 min val per full epoch; peak GPU 4.3 GB. First run of the session: label scan took about 10 min (cold disk). Harmless dataloader-thread error at exit |
| B1_s0_smoke1024 | 2495910 | 2026-11-16 | valid | 0.2434 | 0.1648 | timing test | Same as above at 1024. 1.1 s/it; est. 15.6 min train + 2.1 min val per full epoch; peak GPU 10.8 GB. Files already cached, scan fast |
| B1_s0_smoke1024 | 4f6e686 | 2026-11-16 | valid | 0.2434 | 0.1648 | timing test | New session with file warm-up: read 4.3 GB in 1.5 min (47 MB/s); setup 0.6 min; 0.96 s per batch; est. 14.3 + 2.0 min per epoch |
| B1_s0_smoke1024 | 4f6e686 | 2026-11-16 | valid | 0.2719 | 0.1753 | timing test | Same with cache=ram: 0.96 s per batch (no gain), so image decoding is not the limit. mAP differs because of a different preprocessing path |
| B0_s0_smoke1024 | 4f6e686 | 2026-11-16 | valid | 0.1801 | 0.1173 | timing test | Parallel test, GPU 0, running together with B1: 1.11 s per batch; est. 16.5 + 2.4 min per epoch; peak 10.0 GB |
| B1_s0_smoke1024 | 4f6e686 | 2026-11-16 | valid | 0.2434 | 0.1648 | timing test | Parallel test, GPU 1, running together with B0: 1.17 s per batch; est. 17.4 + 3.0 min per epoch; peak 10.8 GB |
| B0_s0 | 60fc83f | 2026-11-16 | valid | - | - | failed (version 1) | Crashed at start: `yolo26n.pt` / `yolo11s.pt` was downloaded by both runs at the same moment and B0 read a broken file (`PytorchStreamReader failed reading file ... archive is corrupted`). No checkpoint; restart from scratch in version 2. Fix: notebook downloads the weights once before launching |
| B1_s0 | 60fc83f | 2026-11-16 | valid | 0.628 | 0.4046 | stopped for time (version 1) | 50 of 100 epochs in one 11 h version, running alone on its GPU after B0 crashed (about 13 min per epoch with val). Resume in version 2 |

Timing-test mAP after one epoch on 10% of the data says nothing about the final accuracy of either image size.

Rules:
- A run counts only if its folder has `run_info.json`.
- Run name = `<expID>_s<seed>`, e.g. `V1_s0`.
- Per-class numbers live in `results/<run name>/metrics.json`.
| B1_s0 | 60fc83f | 2026-11-17 | valid | 0.628 | 0.4046 | stopped for time (version 2) | Resumed at epoch 51, reached 91; best epoch still 48 (mild overfitting after about epoch 50; resume verified continuous) |
| B0_s0 | 60fc83f | 2026-11-17 | valid | 0.627 | 0.4013 | stopped for time (version 2) | Restarted from scratch (download race fixed), epochs 1-49, best epoch 43 |
| B1_s0 | 60fc83f | 2026-11-17 | valid | 0.628 | 0.4046 | FINISHED (version 3) | 100 epochs; best epoch 48 |
| B0_s0 | 60fc83f | 2026-11-18 | valid | 0.627 | 0.4013 | FINISHED (version 4) | 99 epochs in version 3, epoch 100 in version 4; best epoch 43 |
| premise B1_s0, B0_s0 | f7a7eb4 | 2026-11-18 | valid pairs | 0.635 / 0.618 | - | GATE 2 FAIL | Premise test (STEP-04): 0 of 6 gateable classes with a partner gain (rule needs 2). Mean AP50 on validation pairs (square letterbox) shown. Follow-up analysis and study-level result: claude/VGRA_gate2_report.md (project docs) |
| gate_p B0_s0 | a40d1a1 | 2026-11-18 | valid (all 4,094 images) | - | - | GATE P FAIL | Presence branch on frozen B0_s0, protocol settings (5 epochs, loss 0.638 -> 0.237, 33 min). Rescored AP50 gain [95% CI]: pronatorsign +0.007 [-0.009, +0.023], softtissue -0.007 [-0.019, +0.004], periostealreaction -0.000 [-0.004, +0.003]; mean AP50 base 0.5934 vs new 0.5935. Branch image AUC below the detector's own max (pronator 0.985 vs 0.986, soft tissue 0.907 vs 0.925, periosteal 0.974 vs 0.982). Views correct (2,030 PA, 2,047 LAT, 17 other); branch learned projection logic (pronator p 0.015 on PA, 0.905 on LAT positives) |
| study valid B0_s0, B1_s0 | (I2 patch) | 2026-11-18 | valid pairs (1,884) | - | - | DEVELOPMENT | Patient level, fracture, sensitivity at 95% specificity PA / LAT / both (max): B0 0.941 / 0.955 / 0.974 (missed per 100: 5.9 / 4.5 / 2.6), B1 0.929 / 0.962 / 0.977; McNemar max vs PA p < 0.001 for both. Subgroups (B0): torus PA 0.917 vs both 0.980; initial exam PA 0.856 vs both 0.951. Hypothesis-generating only (thresholds and numbers on the same split) |
| final detectors (valid) | 60fc83f / aeaa4c2 / 81a91c8 | 2026-11-21 | valid (4,094 images) | 100 each | best epoch 43-56 | FINISHED | Ultralytics validation of best.pt, mAP50 / mAP50-95: B0_s0 0.627 / 0.401, B0_s1 0.629 / 0.410, B0_s2 0.632 / 0.406 (B0 mean 0.629 +- 0.003 / 0.406 +- 0.004); B1_s0 0.628 / 0.405; V8_s0 0.615 / 0.404, V8_s1 0.632 / 0.404 (mean 0.624 +- 0.012); V8R_s0 0.633 / 0.408, V8R_s1 0.620 / 0.404 (mean 0.627 +- 0.009). Fracture AP50 0.946-0.951 for every run. Weights frozen with SHA-256 in docs/final_weights.json |
| dry run kaggle_eval (valid) | 8fd2f34 | 2026-11-19 | valid | - | - | PASS | I1, I3 (64 pairs), I4, I2 on Kaggle with B0_s0 + B1_s0; I4 reproduces the training metrics; I2 reproduces the local development numbers exactly (B0 fracture PA 0.9413 / LAT 0.9549 / max 0.9744) |
