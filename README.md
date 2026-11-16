# vgra-yolo

VGRA-YOLO: paired AP/LAT detection of pediatric wrist abnormalities on GRAZPEDWRI-DX.

**Model = YOLO11s + SCC + VGRADetect**

- **SCC**: self-calibrated convolution (SCNet, CVPR 2020) adapter in the two early
  backbone C3k2 blocks. Block class: `C3K2_SCC`.
- **VGRADetect**: detection head = stock Detect head + CQP (class-query pooling),
  VSH (visibility state head), GRA (gated residual assistance).

**Status:** see `docs/PLAN.md`. Data report: `docs/EDA.md`. Run log: `docs/EXPERIMENTS.md`.

## Quick start on Kaggle

1. In Kaggle: File > Import Notebook > `notebooks/kaggle_run.ipynb`.
2. Notebook settings: Internet **on**.
3. Run the cells in order: clone, install, check.

The check cell must end with `SETUP CHECK: PASS` and all tests passing.

## Background training on Kaggle (long runs)

Kaggle "Save Version" runs stop after 12 hours, so one training runs as several versions of the
same notebook. Use `notebooks/kaggle_train.ipynb`, one notebook per Kaggle account:

1. Set `REF` to the fixed code tag (e.g. `detectors-v1`) and `RUNS` to at most two runs, one per GPU.
2. Version 1: Save Version (Save & Run All), GPU T4 x2, Internet on, inputs: image dataset + `vgra-index`.
3. Each run stops cleanly when the next epoch would not fit in `train.max_hours` (11 h) and keeps
   `weights/resume.pt`. The summary cell prints `STOPPED FOR TIME` or `FINISHED`.
4. Version 2, 3, ...: attach the previous version's output as input, change nothing, Save Version.
   `--auto-resume` finds each run folder under `/kaggle/input`, copies it and continues;
   `results.csv` continues in the same file and `run_info.json` records every session.
5. When both runs print `FINISHED`, copy `runs/<run>/metrics.json` into `results/<run>/` and log the
   run in `docs/EXPERIMENTS.md`. Weights go to the Kaggle Dataset `vgra-weights`.

Never change `REF` in the middle of a run: every version must run the same code.

## Local (VS Code)

```
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

## Rules

- Ultralytics is installed from pip at the pinned version. Its source is never edited.
- All custom code lives in the `vgra` package.
- One branch per step (`step-NN-name`), merged to `main` after its pass check, then tagged `stepNN`.
- One config file per experiment in `configs/`; it lists only keys that differ from `base.yaml`.
- Every run writes `run_info.json` (commit, config, seed, versions). A run without it does not count.
- Never in git: dataset, images, weights, tokens.

## Layout

```
configs/     base.yaml + one file per experiment
vgra/        the package (code is added step by step)
scripts/     numbered entry points
notebooks/   kaggle_run.ipynb (checks, short tests), kaggle_train.ipynb (background training)
tests/       CPU unit tests
results/     small result files (metrics.json per run)
docs/        NAMING, PLAN, EXPERIMENTS, DECISIONS
```
