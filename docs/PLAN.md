# Plan and status

Tick a step only after its pass check holds on Kaggle. Then merge to `main` and tag `stepNN`.

| Done | Step | What | Where | Pass check |
|---|---|---|---|---|
| [x] | STEP-00 | Repo skeleton, pinned requirements, launcher notebook | Local + Kaggle | Kaggle clones, installs; `SETUP CHECK: PASS`; tests pass |
| [ ] | STEP-01 | EDA part 1: image index and dataset checks | Kaggle CPU | Image and split totals match; no patient in two splits |
| [ ] | STEP-02 | EDA part 2: pair audit (**Gate 1**) | Kaggle CPU | Enough pairs and one-view-only cases; gateable class list fixed |
| [ ] | STEP-03 | SCC and detectors B0, B1 | Kaggle GPU | B1 with fresh adapters equals pretrained YOLO11s; saves and reloads |
| [ ] | STEP-04 | Premise test (**Gate 2**), inference only | Kaggle GPU | Partner class scores help separate true/false detections |
| [ ] | STEP-05 | Pair loader | Local + Kaggle | 20 batches inspected; pairs and labels intact |
| [ ] | STEP-06 | CQP, VSH, GRA | Local CPU | Shapes correct; correction exactly 0 at start and without partner |
| [ ] | STEP-07 | `VGRADetect` head | Kaggle GPU | Predictions identical to B1 with zero correction; saves, reloads, copies |
| [ ] | STEP-08 | Oracle run (**Gate 3**) | Kaggle GPU | Beats B1 beyond seed noise on weak classes |
| [ ] | STEP-09 | Loss and trainer | Kaggle GPU | 1-epoch smoke run; frozen detector unchanged |
| [ ] | STEP-10 | Validator, three modes | Kaggle GPU | Single-view mode reproduces B1 |
| [ ] | STEP-11 | First full run `V1_s0` | Kaggle GPU | Stable; compared with B1 and oracle |
| [ ] | STEP-12 | All experiments | Kaggle GPU | Every run logged in `EXPERIMENTS.md` |

Stop and review after STEP-02, STEP-04 and STEP-08. If a gate fails, stop or redesign before writing more code.

## Step routine

1. Create branch `step-NN-name`.
2. Add the code and tests; run `python -m pytest -q` locally.
3. Commit (`STEP-NN: ...`) and push.
4. In Kaggle, set `REF` to the branch name and run the launcher.
5. If the pass check holds: merge to `main`, tag `stepNN`, tick the box above.
