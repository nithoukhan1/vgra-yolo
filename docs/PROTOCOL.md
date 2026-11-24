# Analysis protocol (FROZEN v1.0, approved by the project owner on 2026-11-18; amendment v1.1 on 2026-11-18; amendment v1.2 on 2026-11-23)

Status: FROZEN. Tag `protocol-v1` marks v1.0. Amendment v1.1 (owner approval 2026-11-18, before the test
split was opened) adds one robustness detector, V8R; see section 3, 4.3 item 6 and section 10. Amendment v1.2
(owner approval 2026-11-23, AFTER the Part A test session, tag `protocol-v1.2`) adds Part B = CoVeR, two
extension detectors from other families (RT-DETR-l, Faster R-CNN) and one reporting change (B1 in the
supplement); see section 11. It changes no Part A endpoint, threshold or result. The test split may be opened only after all
models and choices are fixed (section 2). Any later change is logged in section 10 with date and
reason (deviations are allowed, but always reported).

Owner decisions at freezing (2026-11-18): Part A primary measure = sensitivity at 95% specificity;
Gate P = at least 2 of 3 target classes; PAC non-inferiority margin = 0.005 mAP50; subgroups = the
four listed in 4.4.

Purpose: fix every endpoint, threshold, model-selection rule, subgroup and statistical test
**before** results on the test split are seen, so that the paper's claims cannot be shaped by
the results (pre-registration).

---

## 1. Data
- Dataset: GRAZPEDWRI-DX (20,327 images, 6,091 patients). Split: Ju et al. patient-level split
  (train 14,204 / valid 4,094 / test 2,029 images). Training uses original images only (`orig_*`).
- Projections: 1 = posteroanterior (PA), 2 = lateral (LAT), 3 = other. Papers write PA / LAT.
- Pair (study-side): one PA and one LAT image of the same patient, study and side
  (`vgra/pairs.py`). Pairs: train 6,475 / valid 1,884 / test 929. Study-sides without exactly one
  PA and one LAT are reported and excluded from paired analyses; they are included in image-level
  detector metrics.
- Findings analysed: fracture, periostealreaction, pronatorsign, softtissue, metal (primary set);
  boneanomaly, bonelesion, foreignbody reported descriptively (too few cases); text excluded
  from clinical claims.
- Metadata (from `dataset.csv`): age, initial_exam, cast, ao_classification.

## 2. Use of the splits
| Split | Allowed use |
|---|---|
| train | model training only |
| valid | model selection (best epoch), Gate P, all hyper-parameter and design choices, operating thresholds, fusion-rule fitting |
| test | opened once, after this protocol is FROZEN and all models and choices are fixed; final numbers only |
Nothing measured on test may change any model, threshold or choice. If a bug is found after
opening test, it is fixed, the whole test evaluation is rerun, and the event is reported (section 10).

## 3. Models
| ID | Model | Seeds | Selection |
|---|---|---|---|
| B0 | YOLO11s (Ultralytics 8.4.7) | 0, 1, 2 | best epoch on valid (Ultralytics fitness) |
| B1 | YOLO11s + SCC | 0 | same |
| V8 | YOLOv8s | 0, 1 | same |
| V8R | YOLOv8s + ResCBAM (Ju et al., YOLOv8-AM), amendment v1.1 | 0, 1 | same |
| +PAC | PAC on each detector above (Part B, only if Gate P passes) | as detector | fixed number of epochs set on valid (section 6) |
| RD | RT-DETR-l (Ultralytics 8.4.7, transformer), amendment v1.2 | 0 | best epoch on valid (Ultralytics fitness); recipe in 11.3 |
| FR | Faster R-CNN R50-FPN v2 (torchvision, two-stage), amendment v1.2 | 0 | best epoch on valid (0.1 mAP50 + 0.9 mAP50-95); recipe in 11.3 |
Common recipe (B0, B1, V8, V8R): imgsz 1024, batch 16, 100 epochs, SGD lr0 0.01, patience 50, Ultralytics defaults.
RD and FR follow their own standard recipes (section 11.3). Reporting (amendment v1.2): B1 appears in the supplement only (11.2).
V8R: Ju et al.'s ResBlock_CBAM after each of the four neck C2f layers, layout copied from their YAML
(`configs/models/yolov8s-rescbam.yaml`, `vgra/rescbam.py`); pretrained YOLOv8s weights in every shared
layer, the four blocks randomly initialised; same recipe and seeds as V8, so V8R vs V8 isolates the module.
V8R deadline (fixed in advance): if V8R_s0 and V8R_s1 have not both FINISHED within 7 days after the last
of B0_s1, B0_s2, V8_s0, V8_s1 finishes, the test session goes ahead without V8R and V8R is not reported
on test.

## 4. Part A: paired-projection analysis

### 4.1 Questions
- RQ1 (box level): does the partner projection add information to detection inside an image?
- RQ2 (patient level): does combining PA and LAT improve the study decision over one projection?
- RQ3: which findings depend on which projection?
- RQ4: is a learned two-view fusion better than a fixed rule?
- RQ5: do the answers hold in subgroups?

### 4.2 Definitions
- Image score of finding c: highest detector confidence of class c in the image (after NMS with
  conf 0.001, IoU 0.7, max 300 boxes, square letterbox 1024).
- Study reference standard for finding c: positive if c is annotated in the PA or the LAT image.
- Study scores: PA-only = PA image score; LAT-only = LAT image score; fused = combination by
  - max(PA, LAT) (**primary rule**),
  - mean, noisy-OR 1 - (1 - PA)(1 - LAT),
  - logistic stacking on [logit PA, logit LAT] fitted on valid, applied to test.
- Operating point: for each model, setting and finding, the threshold that gives 95% specificity
  on valid (95th percentile of negative-study scores); applied unchanged to test.

### 4.3 Endpoints
- **Primary (Part A):** on test pairs, fracture, model B0 (mean over seeds 0-2, each seed tested):
  sensitivity at the valid-fixed 95%-specificity threshold, fused (max) vs PA-only.
  Test: McNemar on paired study decisions (per seed); reported with patient-bootstrap 95% CI of the
  difference.
- Secondary:
  1. Same comparison vs LAT-only; AUC fused vs PA-only and vs LAT-only (DeLong).
  2. Same endpoints for periostealreaction, pronatorsign, softtissue, metal.
  3. Achieved specificity on test; missed findings per 100 positive studies.
  4. Fusion rules compared (AUC, sensitivity at the fixed operating point).
  5. RQ1 confirmatory: the STEP-04 premise test rerun on test pairs with the Gate 2 rule
     (detection-level gain with 95% CI above 0 in >= 2 of the gateable classes AND partner
     rescoring AP50 > view-only rescoring). Expected outcome: no gain.
  6. Every endpoint repeated for B1_s0, V8 seeds 0-1 and V8R seeds 0-1 (robustness across detectors).
  7. Module comparison (descriptive, amendment v1.1): V8R vs V8 per seed on the test detector metrics
     (mAP50, mAP50-95, per-class AP50) and on the primary endpoint; the V8R - V8 difference is shown
     next to the seed-to-seed difference of V8 (and of B0). No claim of superiority is made from two seeds.

### 4.4 Subgroups (RQ5), defined in advance
| Subgroup | Definition |
|---|---|
| Torus / buckle fracture | any image of the study has an AO code containing `M/2.1` |
| Age bands | < 6, 6 - < 10, 10 - < 14, >= 14 years (age of the study) |
| Exam type | initial_exam = 1 vs not |
| Cast | cast = 1 on any image of the study vs not |
Subgroup results are reported as estimates with 95% CI; no subgroup claim is made without a CI
that excludes no effect, and subgroup tests are labelled exploratory.

## 5. Part B gate: Gate P (decided on valid only)
All settings below were fixed before Gate P was run.
- Detector: B0_s0 `best.pt`, frozen (eval mode, no gradients).
- Presence branch: class-query pooling over the detector's P3/P4/P5 neck outputs (width 64, one
  learned query per class for all 9 classes, one cross-attention layer + feed-forward), a
  projection embedding (PA / LAT / other) added to the queries, and a per-class linear presence
  scorer (`vgra/presence.py`).
- Training: train split originals (14,204 images), image size 1024, batch 16, Ultralytics default
  augmentation with mosaic, mixup, cutmix and copy-paste off; presence target of class c = at least
  one box of c in the augmented image; loss = binary cross-entropy with per-class positive weight
  min(negatives / positives, 50) counted on the train split; AdamW, learning rate 1e-3, weight
  decay 1e-4, 200 warm-up iterations then cosine decay to 0; 5 epochs; seed 0; the weights after
  the last epoch are used (no selection on valid).
- Evaluation: all valid images (paired and unpaired), square letterbox 1024, detector NMS as
  Ultralytics validation (conf 0.001, IoU 0.7, max 300, multi-label). Patient-grouped 5-fold
  cross-fitting and 500 patient bootstraps (the STEP-04 machinery):
  - detection level, target = true positive at IoU 0.5:
    base [logit conf, LAT indicator, other indicator, logit own-image max of c]
    vs new = base + [logit p_c, LAT indicator x logit p_c];
  - rescored AP50 per class from the out-of-fold probabilities of both models; classes with fewer
    than 10 true positives keep the detector confidence in both models;
  - image level (reported, not part of the rule): AUC of own-image max vs + presence.
- Target classes: pronatorsign, softtissue, periostealreaction.
- **PASS** if at least 2 of the 3 target classes have a rescored AP50 gain (new - base) whose 95%
  patient-bootstrap CI lies above 0, AND the mean rescored AP50 over all classes with labels is not
  lower for new than for base. Otherwise **FAIL**: Part B is dropped and Part A is published alone.

## 6. Part B: PAC evaluation (only if Gate P passes)
- PAC: calibrated class logit l' = l + alpha_c * log sigmoid(z_c), alpha_c = cap * tanh(theta_c),
  theta_c = 0 at start (identical to the detector), cap 2.0; detector frozen; presence + detection
  loss; mosaic off; number of epochs, loss variant and lambda chosen on valid and then fixed.
- **Primary (Part B):** on test images, mean AP50 over the three target classes, B0 + PAC vs B0
  (mean over seeds 0-2; paired patient bootstrap of the difference per seed and pooled).
- **Non-inferiority:** overall mAP50 on test must not drop by more than 0.005 (B0 + PAC vs B0).
- Secondary: per-class AP50 and AP50-95; mAP50-95; false positives per image at the recall of the
  baseline at its default threshold (0.25); image-level AUC; Part A patient-level endpoints
  recomputed with PAC; the same for B1_s0 and V8_s0.
- Controls (same test protocol): separate image classifier gating the detector; joint fine-tuning
  with a presence loss but no calibration; per-class score recalibration fitted on valid.
- Ablations: run and selected on valid only; test numbers of ablations are reported, never used
  for selection.

## 7. Statistics
- Confidence intervals: patient-level bootstrap, 2,000 resamples (whole patients resampled).
- Paired AUC comparison: DeLong; paired sensitivity / specificity: McNemar (exact if < 25 discordant).
- Multiple findings: Holm correction within each family of secondary endpoints.
- Seeds: every metric per seed, plus mean +- SD over seeds; primary tests per seed and on pooled
  bootstrap.
- Significance level 0.05 (two-sided). Effect sizes always reported with CIs.

## 8. Reporting
- CLAIM checklist (Checklist for Artificial Intelligence in Medical Imaging).
- Detector metrics on the full test split with the standard Ultralytics validator (same split as
  Ju et al.); results of other papers listed separately because their splits and protocols differ.
- Code tag and commit for every number; all runs listed in `docs/EXPERIMENTS.md`.

## 9. What is already known (valid only, before this protocol)
Gate 2 (STEP-04) and its exploratory follow-up, and the study-level pilot (fracture misses
PA-only 7.1%, LAT-only 3.8%, fused 2.3%), were computed on valid. These are hypothesis-generating;
the confirmatory numbers are the test-split numbers under this protocol.

## 10. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-24 | FR RESUME FIX (code tag `detectors-v5`), during notebook D. Full version 2 could not resume FR_s0: it stopped before training, at `optimizer.load_state_dict` (parameter-group size mismatch). Cause: with pretrained weights, torchvision trains Faster R-CNN with conv1, bn1 and layer1 frozen (`trainable_backbone_layers` 3, its default and the reference recipe's setting, which 11.3 left implicit), but the model rebuilt without weights to resume had every layer trainable (209 trainable tensors instead of 176). `vgra/frcnn.py` now freezes the same layers explicitly whenever the model is built. Epochs 1-15 (version 1) trained with exactly these layers frozen; the fixed code rebuilds the same 176 tensors in the same order, so the epoch-15 optimizer state loads onto the matching tensors (checked against the detectors-v4 model). Nothing was trained in the failed session. RD's code is unchanged in detectors-v5, so RD finishes its last epochs at that tag. Training notebook: the summary now reports a non-zero exit code as CRASHED (for FR it had printed the status restored from version 1). New regression test (train from pretrained weights, stop for time, resume) fails on detectors-v4 | Bug fix during training; recipe unchanged (11.3's torchvision reference recipe, whose frozen layers are now stated) |
| 2026-11-23 | EXTENSION DETECTOR CODE (amendment v1.2, section 11.3), written and tested before any RD or FR training; code tag `detectors-v4`. `model.kind` in the configs (yolo / rtdetr / frcnn); `configs/RD.yaml` and `configs/FR.yaml` hold the 11.3 values. RD: the Ultralytics 8.4.7 RT-DETR trainer and validator (COCO `rtdetr-l.pt`: 926 of 941 tensors copied, the other 15 are class layers), with one library defect corrected in subclasses (`vgra/rtdetr.py`; Ultralytics itself stays unedited): the 8.4.7 validator sorts the 300 query predictions by score but applies the confidence mask of the unsorted order, so when a query at or below 0.001 comes before a higher one a good detection is dropped and a sub-threshold one kept; the subclass applies the mask in the sorted order (what the code intends; later Ultralytics releases rewrote the method), for the fitness of every epoch, the final evaluation of best.pt and T2 (shown in the repo tests). Wording correction to 11.3: the Ultralytics RT-DETR pipeline resizes every image to 640 x 640 by stretching, without letterbox padding, in training and validation (11.3 wrote "square letterbox"); the library behaviour is kept, because 11.3 fixes the library, and this was settled before any RD training. RD image score of a class = its highest score over the 300 queries (the counterpart of the YOLO pre-NMS maximum of 4.2); detections = one class per query with score > 0.001, as the corrected validator (checked equal in the repo tests). FR (`vgra/frcnn.py`, the torchvision reference recipe of 11.3): (a) image score of a class = its highest score among the image's final detections (score > 0.001, NMS IoU 0.5, at most 300; NMS always keeps the highest box of a class); (b) a batch with a non-finite loss is skipped and counted (`results.csv` column `skipped`), more than 10 in one epoch stops the run (the torchvision reference stops at the first one, which would hit the same batch again in every later Kaggle version); (c) a resumed run keeps its batch; (d) validation after every epoch with the repo AP code, equal to the Ultralytics validator (tests). I1, T2 and I3 handle all three kinds; for RD and FR, I3 reads the I1 tables (`scripts/04_premise_test.py --from-pred`) instead of running the detector a second time: identical inputs and results (the Kaggle dry-run tables of B0_s0, B1_s0 and V8R_s0 reproduced exactly; tests). Kaggle notebook D (`notebooks/kaggle_train.ipynb`): one smoke version (1 epoch on 10% of train) whose only use is the batch fallback, then the full runs with --auto-resume | 11.3: the implementation is logged here before training; the library defect was found while testing and is corrected before any RD result exists |
| 2026-11-23 | CoVeR TEST RUN executed once with the section 11.4 command at 08:47:35 UTC (COVER_TEST_RUN.json), code = tag `protocol-v1.2` (50c904f, tree 6b36ac0): the record shows commit baada13 because the package was imported from a local checkout whose tree is identical (6b36ac0) to the tag. Part A consistency check PASS for all 8 runs. No step failed, nothing was rerun, no deviation. PRIMARY (LTT, alpha 0.02, all ages): lateral reads 20-23%; extra-miss risk 1.6-3.6% (all 95% CIs include 2%); limit held in 1 of 7 main-text detectors (V8_s0). Main secondary (per age, LTT alpha 0.02): limit held in 7 of 7 (0.3-1.5%), lateral reads 29-37%. Outputs: `results/cover/test/`. Exploratory follow-up (post hoc, `scripts/14_cover_explore.py`): the test split has more fractures the PA view practically misses | Pre-specified test run of amendment v1.2 |
| 2026-11-23 | AMENDMENT v1.2 frozen (owner approval 2026-11-23; tag `protocol-v1.2`), after the Part A test session: section 11 = (a) Part B CoVeR with its primary configuration, secondary configurations, baselines, outcomes, Part A consistency check and the one command that applies it to test; (b) extension detectors RD (RT-DETR-l) and FR (Faster R-CNN R50-FPN v2) with recipes, selection, deadline and evaluation; (c) B1 reported in the supplement only. CoVeR had been designed on valid only (row below); nothing from this section had touched the test split | Owner decisions of 2026-11-23; fixed before any new use of the test split |
| 2026-11-23 | POST HOC (after the test session), validation split only: Part B = CoVeR, conformal view referral (`vgra/cover.py`, `scripts/13_cover.py`, I6). Design study on valid (8 frozen detectors + their mean, 300 patient half-splits); the test split was not used. Any use of CoVeR on test requires amendment v1.2 (rules, primary configuration, baselines, outputs) frozen and tagged first; the script refuses the test split without --open-test and records the run | New research question raised by the Part A result; kept separate from the confirmatory analysis |
| 2026-11-21 | TEST SESSION executed once with `notebooks/kaggle_eval.ipynb` (OPEN_TEST = True) at tag `parta-test-v1` (26ede6c): all 8 best.pt matched `docs/final_weights.json`; test split opened 06:19:09 UTC (I1), 06:25:07 (I3), 06:34:20 (I4); no step failed and nothing was rerun; no deviation from the protocol. Outputs: `results/parta/test/`. Every analysis added after this date is post hoc and is reported as exploratory | Confirmatory results fixed |
| 2026-11-21 | Detector weights frozen before the test split is opened: all 8 runs FINISHED (100 epochs each; V8R within the deadline of section 3); SHA-256 of every best.pt in `docs/final_weights.json`; the test session uses one dataset `vgra-final-detectors` (runs/<name>/ with best.pt, metrics.json, results.csv, args.yaml, run_info.json) and the evaluation notebook refuses to open the test split if any best.pt differs from the frozen value. B0_s0 and B1_s0 were trained at commit 60fc83f, the others at aeaa4c2 (detectors-v2) and 81a91c8 (detectors-v3); training code and recipe are identical across these commits (later commits add tools only) | The evaluated models are provably the ones fixed before any test result |
| 2026-11-19 | Part A tools I3 + I4 + evaluation notebook, before the test split is opened: `scripts/04_premise_test.py --split test --open-test` (RQ1 confirmatory, Gate 2 rule, every detector, 2,000 bootstrap resamples per section 7; Gate 2 on valid kept 500 as frozen); `scripts/11_val_detectors.py` (standard Ultralytics validation, T2); `notebooks/kaggle_eval.ipynb` runs I1, I3, I4, I2 in one session, with a dry-run mode on valid and a check that every run is FINISHED; each script refuses the test split without `--open-test` and writes TEST_OPENED.json when it opens it | One session opens the test split; the dry run proves the pipeline first |
| 2026-11-18 | Part A tools I1 + I2, written and tested on valid before the test split is opened (implementation of 4.2, 4.3 and 7, no change of endpoint): `scripts/10_predict.py` (all images of a split, protocol NMS; the test split is refused without `--open-test`, which writes TEST_OPENED.json); `vgra/study.py` + `scripts/12_study_level.py`. Fixed details: (a) image score = own-image maximum if > 0.001 else 0, which equals the top box after NMS (checked identical on all 36,846 valid image-class rows of B0_s0); (b) a study is positive if its score is above the valid threshold = 95th percentile (numpy linear) of the negative-study scores; (c) stacking = logistic regression on [logit PA, logit LAT], fitted on all valid pairs and applied to test (on valid itself, development numbers use patient-grouped 5-fold out-of-fold scores); (d) every contrast also reports the specificity difference with McNemar on the negative studies (4.3 item 3: specificity achieved on test can differ from 95%); (e) Holm families = one contrast and one test over the findings, the primary test (fracture, max vs PA) unadjusted; (f) pooled seed bootstrap = the same resampled patients for every seed, mean of the per-seed differences; (g) subgroups use the whole-valid thresholds; torus = AO code containing M/2.1 on any image of the same study (patient, study number); age = age on the PA image; initial exam and cast = 1 on either image; (h) contrasts computed: max vs PA (primary for fracture), max vs LAT, LAT vs PA (RQ3), mean / noisy-OR / stacking vs max (RQ4) | Every analysis choice is in code and tested before any test result exists |
| 2026-11-18 | Amendment v1.1 (owner approval, before the test split was opened): V8R = YOLOv8s + ResCBAM of Ju et al. (best attention module published on this split), seeds 0 and 1, same recipe as V8, as an extra robustness detector (4.3 item 6) and a descriptive module comparison (4.3 item 7); 7-day deadline rule in section 3. Code tag `detectors-v3` (adds V8R only; B0, B1 and V8 code unchanged) | Answers "would a published attention detector change the conclusion?" and "is a published module gain larger than seed noise?"; decided before any test result |
| 2026-11-18 | v1.0 frozen; before freezing, section 5 was made explicit (Gate P training and evaluation settings) | Pre-registration completeness; no Gate P result existed |

## 11. Amendment v1.2 (owner approval 2026-11-23, after the Part A test session)

### 11.1 Status
Written after the Part A test session of 2026-11-21. It adds new, pre-specified analyses and changes no Part A
endpoint, threshold or result; Part A stays the confirmatory analysis of the paper. Tag `protocol-v1.2` marks the
commit that contains this section and the code it names; nothing in this section may use the test split before that
tag is on GitHub. In the paper, everything here is reported as pre-specified analyses added after the Part A test
session.

### 11.2 Reporting change
Main text: B0 (3 seeds), V8 (2), V8R (2; Ju et al.'s published model) and the extension detectors RD and FR (11.3).
B1 (YOLO11s + SCC, an adapter from our earlier design) is reported in full in the supplementary material, not in the
main text, so that the main text shows published architectures only. B1 was pre-specified, its test results are
already public (`results/parta/test/`) and agree with the other detectors (fracture both vs PA +0.074, p < 0.001);
nothing is removed from the record.

### 11.3 Extension detectors (other detector families)
Purpose: do the Part A findings and CoVeR hold beyond one-stage YOLO detectors?

| | RD: RT-DETR-l (transformer) | FR: Faster R-CNN R50-FPN v2 (two-stage) |
|---|---|---|
| Library, weights | Ultralytics 8.4.7, `rtdetr-l.pt` (COCO) | torchvision `fasterrcnn_resnet50_fpn_v2`, COCO_V1 weights; new box predictor, 9 classes + background |
| Input | imgsz 640 (square letterbox) | longer side 1024 (min_size = max_size = 1024), no letterbox |
| Schedule | 72 epochs, patience 50 | 24 epochs; learning rate x 0.1 at epochs 16 and 22; linear warm-up 1,000 iterations |
| Batch | 8 (4 if 8 runs out of memory in the first epoch) | 4 (2 if 4 runs out of memory in the first epoch), learning rate scaled linearly |
| Optimiser | AdamW, lr0 1e-4, weight decay 1e-4; warm-up 3 epochs rising from 0 for every parameter group (`warmup_bias_lr` 0: the Ultralytics default 0.1 is 1,000x lr0); linear decay to lr0 x 0.01 (Ultralytics default) | SGD momentum 0.9, lr 0.005 at batch 4, weight decay 1e-4; mixed precision |
| Augmentation | Ultralytics defaults (as for the YOLO detectors) | horizontal flip p = 0.5 (torchvision standard) |
| Selection | best epoch on valid, Ultralytics fitness | valid after every epoch; best epoch by 0.1 mAP50 + 0.9 mAP50-95, computed with the repo AP code (Ultralytics matching and `ap_per_class`, `vgra/premise.py`) |
| Inference | end-to-end (no NMS), at most 300 queries, score > 0.001 | score > 0.001, NMS IoU 0.5 per class, at most 300 detections |

Common: seed 0; the training images, classes and splits of section 1 (original training images only).
RD input 640: RT-DETR-l needs 275 GFLOPs at 1024 (5x YOLO11s); every endpoint compares views within one detector, so the
input size does not bias it. The image score of a class is its highest confidence in the image (as in 4.2).
Implementation (`configs/RD.yaml`, `configs/FR.yaml`, training, inference and validation code, a Kaggle notebook)
follows this section and is logged in section 10; a smoke run (1 epoch on 10% of train) may only decide the batch
fallback above. Deadline: each detector must finish all epochs within 21 days of the `protocol-v1.2` tag; an
unfinished detector is reported as not completed and is never evaluated on test.
Freezing and evaluation: SHA-256 of each best weights file added to `docs/final_weights.json`; a dry run on valid; one
evaluation session that opens the test split once for RD and FR only (I1, I3, I4, each writes TEST_OPENED.json); then
I2 (section 4: primary endpoint per detector, secondary endpoints, RQ1 with the Gate 2 rule, subgroups) and CoVeR
(11.4) on CPU. Detector metrics (T2) use the Ultralytics validator for RD and the repo AP code for FR.

### 11.4 Part B: CoVeR (conformal view referral), fracture
Code: `vgra/cover.py` and `scripts/13_cover.py` at tag `protocol-v1.2`.
Policy per PA + LAT pair and detector: t = the Part A primary threshold of the detector (4.2); PA score > t -> positive
from PA alone; PA score <= lo -> negative from PA alone (lateral not read); otherwise read the lateral and decide
max(PA, LAT) > t. Risk = extra misses relative to always-both at t (fracture-positive pairs that always-both catches
and the gate misses); unit = patient (loss = share of the patient's positive pairs that are extra misses; risk = mean
over patients with a positive pair); lo from a fixed grid by fixed-sequence search (code docstring).

PRIMARY configuration: learn-then-test with the Hoeffding-Bentkus p-value, alpha = 0.02, delta = 0.10, one lo for all
ages, calibrated on all valid pairs of the detector, applied once to test. In words: at most 2 extra missed fractures
per 100 fracture patients compared with always reading both views, with 90% confidence.

Detectors: each main-text detector separately (B0_s0, B0_s1, B0_s2, V8_s0, V8_s1, V8R_s0, V8R_s1; RD_s0 and FR_s0
after 11.3); ENS = mean PA and mean LAT score of the 7 main-text Part A detectors (secondary); B1_s0 in the supplement.

Outcomes per detector on test (whole-patient bootstrap 95% CIs, 2,000 resamples, seed 0):
1. share of pairs whose lateral view is read;
2. extra-miss risk and whether it is <= 0.02 ("limit held"); the number of the 7 main-text Part A detectors in which
   the limit held (each is expected to hold with probability >= 0.9; no further claim);
3. sensitivity and specificity of CoVeR, of always-both and of PA only (PA at its own valid 95%-specificity threshold);
4. sensitivity of CoVeR minus random referral of the same number of lateral reads (expected value), with CI;
5. McNemar CoVeR vs always-both on the positive pairs (extra misses) and on the negative pairs (false alarms
   avoided), descriptive.
Secondary configurations: per age band (4.4 bands) with LTT alpha 0.02 (main secondary: every band protected
separately); LTT alpha 0.01 (all ages and per band); CRC alpha 0.01 and 0.02; naive alpha 0.01 and 0.02 (reference
without correction). Baselines at the same number of lateral reads: random referral and ranking by detector
disagreement (SD of the PA score over the 7 main-text Part A detectors); oracle reads as reference.
Subgroups (exploratory, 4.4): share of lateral reads and extra-miss risk per age band, torus, initial exam, cast.
Consistency check: before any CoVeR number is computed, the always-both threshold, sensitivity, specificity and counts
of every run must equal the Part A table (`results/parta/test/study/settings.csv`) exactly; the script stops otherwise.
The valid design study (`results/cover/valid_design/`) is hypothesis-generating only.

Command (run once, after the tag; predictions = `runs/predictions/` of the Part A test session):

    python scripts/13_cover.py --pred <test-session runs/predictions> \
        --runs B0_s0 B0_s1 B0_s2 V8_s0 V8_s1 V8R_s0 V8R_s1 B1_s0 \
        --ens-runs B0_s0 B0_s1 B0_s2 V8_s0 V8_s1 V8R_s0 V8R_s1 --ensemble \
        --eval test --open-test --check-parta results/parta/test/study/settings.csv

Outputs (aggregate tables only, committed under `results/cover/test/`): `results.csv` (every run and configuration,
primary flagged), `who.csv`, `summary.txt`, `COVER_TEST_RUN.json`. RD_s0 and FR_s0 get the same command after their own
Part A tables exist (11.3), with `--check-parta` pointing to those tables.
