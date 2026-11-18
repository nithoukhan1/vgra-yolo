# Guide for co-authors: running a training notebook on your own Kaggle account

Thank you for helping. You run the notebook yourself on your own account (Kaggle allows one person
per account; please do not share your login). Nothing in the code needs editing.

## What you need
- A Kaggle account with phone verification (needed for GPUs) and some free GPU quota
  (each version uses about 11-12 hours of the 30-hour weekly quota).
- Access to two datasets (the project owner shares them with you or makes them public):
  - `nithoukhan/grazpedwri-dx-aug` (the images)
  - `nithoukhan1/vgra-index` (the image index)
- The notebook `notebooks/kaggle_train.ipynb` from https://github.com/nithoukhan1/vgra-yolo
  (Kaggle: Create -> New Notebook -> File -> Import Notebook -> GitHub, or the file sent to you).

## Version 1
1. Add the two datasets as inputs (right panel -> Add Input).
2. Settings: Accelerator **GPU T4 x2**, Internet **on**.
3. In the first code cell, check `REF` and `RUNS`:
   - Notebook A: `REF = "detectors-v2"`, `RUNS = [("configs/B0.yaml", 1, 0), ("configs/B0.yaml", 2, 1)]`
   - Notebook B: `REF = "detectors-v2"`, `RUNS = [("configs/V8.yaml", 0, 0), ("configs/V8.yaml", 1, 1)]`
4. **Save Version -> Save & Run All (Commit)**. It runs in the background for about 11-12 hours; you can
   close the browser.
5. When it ends, open the version's log and send the project owner the output of the last cell
   (lines like `B0_s1: STOPPED FOR TIME ... epochs 31`).

## Versions 2, 3, ...
1. Add as input the **output of this same notebook** (Add Input -> Your Work -> this notebook).
   If the input is already there, make sure it points to the newest version.
2. Change nothing else. **Save Version -> Save & Run All (Commit)** again.
3. After about 10 minutes you can open the running log: it should say
   `auto-resume: found ... (N epochs logged)` for each run.
4. Repeat until the last cell shows `FINISHED` for both runs (usually 3 versions).

## At the end
1. Notebook page -> Output -> **New Dataset** (for example `vgra-b0-s1-s2` or `vgra-v8-s0-s1`).
2. Share the dataset with the project owner (dataset Settings -> Sharing -> add the owner's username),
   or make it public.
3. Tell the project owner the dataset name.

## Please do not
- change `REF`, `RUNS` or any code between versions;
- remove the "weights" cell (it prevents two runs from corrupting a shared download);
- run the same notebook twice at the same time.
