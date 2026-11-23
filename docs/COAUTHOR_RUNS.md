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
   - Notebook C: `REF = "detectors-v3"`, `RUNS = [("configs/V8R.yaml", 0, 0), ("configs/V8R.yaml", 1, 1)]`
     (YOLOv8s + ResCBAM; output dataset name for example `vgra-v8r-s0-s1`)
   - Notebook D: `REF = "detectors-v4"`, `RUNS = [("configs/RD.yaml", 0, 0), ("configs/FR.yaml", 0, 1)]`
     (RT-DETR-l and Faster R-CNN). Notebook D starts with a short smoke version: see the section below.
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

## Notebook D (RT-DETR and Faster R-CNN): a short smoke version first
1. Smoke version: the settings as in the notebook (`SMOKE = True`, `BATCH = {}`). Each run trains 1 epoch on 10%
   of the training images (well under an hour in total). The last cell prints, for each run, either
   `SMOKE RUN DONE | batch ...` with the timing lines and the peak GPU memory, or `OUT OF MEMORY`.
2. Only if a run printed `OUT OF MEMORY`: add the fallback that the message names (`BATCH = {"RD": 4}`,
   `BATCH = {"FR": 2}` or both) and run the smoke version again. This is the only batch change the protocol allows.
3. Full runs: set `SMOKE = False` (keep `BATCH` as it is) and Save Version. The smoke output is not needed as input.
   From here on everything is as in "Versions 2, 3, ..." above.
4. Send the project owner the last-cell lines of every version (smoke included). Both runs must show `FINISHED`
   by 2026-12-14; RT-DETR runs 72 epochs and Faster R-CNN 24, so expect about 3 versions.

## At the end
1. Notebook page -> Output -> **New Dataset** (for example `vgra-b0-s1-s2`, `vgra-v8-s0-s1`, `vgra-v8r-s0-s1` or
   `vgra-rd-fr-s0`).
2. Share the dataset with the project owner (dataset Settings -> Sharing -> add the owner's username),
   or make it public.
3. Tell the project owner the dataset name.

## If the second cell says `STOP: the code is not at ...`
The code tag named in `REF` is not on GitHub yet (or is misspelled). Nothing has run and almost no quota is used.
Tell the project owner; once the tag is pushed, Save Version again with nothing changed.

## Please do not
- change `REF`, `RUNS` or any code between versions (notebook D: `SMOKE` changes once, after the smoke version, and
  `BATCH` only as in step 2 of its section);
- remove the "weights" cell (it prevents two runs from corrupting a shared download);
- run the same notebook twice at the same time.
