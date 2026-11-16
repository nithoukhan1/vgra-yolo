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
| 2026-11-15 | `index.csv` is kept as the Kaggle Dataset `vgra-index` and outside git; small EDA tables go in `results/eda/` | Building the index takes about 36 minutes on Kaggle |
| 2026-11-16 | Gate 1 passed: 91.4% of images are in a pair; see `docs/EDA.md` | Measured on the real labels |
| 2026-11-16 | View coding fixed: projection 1 = ap, 2 = lat | 563 of 566 pronator sign images are projection 2 |
| 2026-11-16 | SCC lives in `vgra/scc.py` (`SCC`, `SCCAdapter`, `C3K2_SCC`); `add_scc` swaps backbone layers 2 and 4 of a stock YOLO11 model and copies the native weights | Ultralytics stays unedited; a model with fresh adapters gives exactly the stock outputs (checked) |
| 2026-11-16 | Detectors train on the originals only, through a list of image paths written from `index.csv` | The Kaggle dataset is read-only and mixes originals with offline copies |
| 2026-11-16 | Detector training settings: 100 epochs, patience 50, batch 16, SGD with lr0 0.01, other Ultralytics defaults; the same for every model | Epochs and patience follow the defaults of Ju et al.; one fixed recipe keeps comparisons fair |
| 2026-11-16 | The adapter's `alpha` stays in the weight-decay group, as in the original design; tanh(alpha) is logged every epoch in `scc_strength.csv` | Shows whether the adapters switch on; revisit only if they stay near zero |
| 2026-11-16 | Candidate classes for the gate: periostealreaction, fracture, boneanomaly, metal; softtissue weak; pronatorsign not usable with a same-class gate. Final list is set at the spec freeze after the oracle run | The AP image never shows a pronator sign; see `docs/EDA.md` section 7 |
| 2026-11-16 | Image size 1024 and batch 16 for every model (confirmed) | Used by Ju et al. and later YOLO work on this split; fits a T4 (10.8 GB) |
| 2026-11-16 | 100 epochs on the 14,204 originals (888 iterations per epoch = 88,800 in total), patience 50, SGD lr0 0.01 | Literature trains 100 epochs on the doubled set (177,600 iterations). Earlier YOLO11s runs on this split peaked on validation at 46,000-75,000 iterations (epoch 26-42 on the doubled set), so 88,800 iterations with the learning rate annealed at the end covers the peak at half the cost. Check: if B0's validation mAP is still clearly rising over the last 10 epochs, revisit |
| 2026-11-16 | File warm-up on by default; no RAM image cache | Warm-up removes the slow first read of each session; RAM cache gave no speed-up (decoding is not the limit) |
| 2026-11-16 | Long runs: Kaggle Save Version, one notebook per account, two runs per notebook (one per GPU), 11-hour budget per version, `--auto-resume` from the previous version's output, code fixed by a tag | Versions stop at 12 hours. Two runs at once are 15-20% slower each but do twice the work per session |
| 2026-11-16 | Pair B0 and B1 of the same seed in one notebook | Both get the same session conditions, so their comparison is fair |
| 2026-11-16 | Doubled-set check run postponed; run only if the budget or reviewers ask for it | Cost; the offline copies are one fixed brightness/contrast change |
| 2026-11-16 | Pair loader in `vgra/data.py`: a batch of B pairs is 2B images laid out [AP_1..AP_B, LAT_1..LAT_B]; `pair_batch` 8 (16 images, same as the detector batch) | Image i and image B + i are always partners, so VGRA can pair features by index with no lookup |
| 2026-11-16 | Paired training: mosaic, mixup, cutmix and copy-paste are forced off; all other Ultralytics augmentations stay at defaults and are applied to each view independently | Mixing would put pieces of other patients into an image and break the partner relation. AP and LAT are different projections, so a shared geometric transform has no meaning |
| 2026-11-16 | Visibility states are recomputed from the batch labels after augmentation, not read from `pairs.csv` | Crops can remove a box (7.5% of non-empty pair states changed in the tiny test); the target must match what the image shows |
| 2026-11-16 | VGRA trains on pairs only (train 6,475 pairs, originals); unpaired images are evaluated single-view with VGRA off | Unpaired study-sides have no partner; this is the same as the "no partner" path |
| 2026-11-16 | Paired evaluation uses square letterbox (no rectangular batches); the single-view comparison in STEP-10 must use the same setting | Pairs mix image shapes; the comparison is fair only with one validation setting |
| 2026-11-16 | VGRA modules in `vgra/modules.py`: `ClassQueryPooling` (CQP), `VisibilityStateHead` (VSH), `GatedResidual` (GRA), wrapper `VGRA`; built from the config by `build_vgra` | One file, names as in the plan |
| 2026-11-16 | GRA acts on the class scores in the layout of Ultralytics 8.4.7 `Detect.forward_head`, (N, nc, anchors of P3+P4+P5); `VGRADetect` (STEP-07) only overrides `forward_head` and replaces `scores` | Boxes, DFL, loss and decoding stay stock |
| 2026-11-16 | Pairing inside a batch is given by `partner` (index of the other view, -1 if none) and `view` (0 AP, 1 LAT); any batch mix of pairs and single images works | The same code serves paired training, unpaired images and the mismatched-partner control |
| 2026-11-16 | Partner dropout uses a `use_partner` mask: the correction is switched off for that image, but VSH still learns from the pair | Dropout is meant to train the single-view fallback, not to waste state labels |
| 2026-11-16 | Gate variants: signed g_ap = q11 - q01, g_lat = q11 - q10 (default); unsigned g = q11 (A1b); none g = 1 (A1a); oracle = signed gate from the true states (A1d) | Fixes the meaning of the A1 ablation before any run |
| 2026-11-16 | Presence head: per-class linear scorer on the CQP descriptor, zero-initialised; state index = has_ap + 2 * has_lat (00, 10, 01, 11) | Simple and matches the batch labels |
| 2026-11-16 | Descriptor 'scores' (A2c) is left for the ablation step | Needs the partner's class scores, which only exist inside `VGRADetect` |
| 2026-11-16 | Measured: VGRA adds 138,128 parameters (1.44% of YOLO11s + SCC); CPU at 1024 with 8 pairs: VGRA forward + backward 1.0 s vs detector forward 13.8 s | `scripts/06_check_modules.py` |
| 2026-11-17 | The training notebook downloads `yolo11s.pt` and `yolo26n.pt` (AMP check) once before launching parallel runs; training code and tag `detectors-v1` unchanged | Ultralytics writes downloads straight to the final file; two runs downloading at once corrupted the file and crashed B0_s0 version 1 |
| 2026-11-16 | `VGRADetect` (in `vgra/head.py`) is a subclass of Ultralytics' `Detect` built from a trained Detect head (`from_detect`, same layers and weights); it overrides only `forward_head` and replaces `scores` | Everything that checks `isinstance(m, Detect)` (loss, stride, fuse, export) keeps working; boxes untouched |
| 2026-11-16 | The pairing of a batch is passed with `vgra_context(model, partner, view, true_states, use_partner)`; without a context the head is the stock Detect head | The stock validator gives single-view results with no extra code; the same model serves paired and single-view use |
| 2026-11-16 | The batch context and the last aux output are never copied or saved (`__getstate__`); the aux output stays readable after the context block until the next context is set | EMA copies and checkpoints stay clean; the loss can read the VSH and presence outputs after the forward pass |
| 2026-11-16 | Stage 2 helpers: `freeze_detector` (only VGRA trainable) and `detector_eval` (BatchNorm of the frozen detector in eval mode) | With BatchNorm frozen, VGRA with zero correction gives exactly B1's outputs in training mode too (checked) |
| 2026-11-16 | Single GPU per run (no DDP) for VGRA training | The batch context is set on the module; one run per GPU is already our setup |
