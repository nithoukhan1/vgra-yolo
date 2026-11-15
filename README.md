# vgra-yolo

VGRA-YOLO: paired AP/LAT detection of pediatric wrist abnormalities on GRAZPEDWRI-DX.

**Model = YOLO11s + SCC + VGRADetect**

- **SCC**: self-calibrated convolution (SCNet, CVPR 2020) adapter in the two early
  backbone C3k2 blocks. Block class: `C3K2_SCC`.
- **VGRADetect**: detection head = stock Detect head + CQP (class-query pooling),
  VSH (visibility state head), GRA (gated residual assistance).

**Status: STEP-00 (repo skeleton). No model code yet.** See `docs/PLAN.md`.

## Quick start on Kaggle

1. In Kaggle: File > Import Notebook > `notebooks/kaggle_run.ipynb`.
2. Notebook settings: Internet **on**.
3. Run the cells in order: clone, install, check.

The check cell must end with `SETUP CHECK: PASS` and all tests passing.

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
notebooks/   kaggle_run.ipynb, the only notebook
tests/       CPU unit tests
results/     small result files (metrics.json per run)
docs/        NAMING, PLAN, EXPERIMENTS, DECISIONS
```
