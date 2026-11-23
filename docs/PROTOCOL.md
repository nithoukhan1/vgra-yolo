# Analysis protocol (FROZEN v1.0, approved by the project owner on 2026-11-18; amendment v1.1 on 2026-11-18)

Status: FROZEN. Tag `protocol-v1` marks v1.0. Amendment v1.1 (owner approval 2026-11-18, before the test
split was opened) adds one robustness detector, V8R; see section 3, 4.3 item 6 and section 10. The test split may be opened only after all
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
Common recipe: imgsz 1024, batch 16, 100 epochs, SGD lr0 0.01, patience 50, Ultralytics defaults.
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
| 2026-11-21 | TEST SESSION executed once with `notebooks/kaggle_eval.ipynb` (OPEN_TEST = True) at tag `parta-test-v1` (26ede6c): all 8 best.pt matched `docs/final_weights.json`; test split opened 06:19:09 UTC (I1), 06:25:07 (I3), 06:34:20 (I4); no step failed and nothing was rerun; no deviation from the protocol. Outputs: `results/parta/test/`. Every analysis added after this date is post hoc and is reported as exploratory | Confirmatory results fixed |
| 2026-11-21 | Detector weights frozen before the test split is opened: all 8 runs FINISHED (100 epochs each; V8R within the deadline of section 3); SHA-256 of every best.pt in `docs/final_weights.json`; the test session uses one dataset `vgra-final-detectors` (runs/<name>/ with best.pt, metrics.json, results.csv, args.yaml, run_info.json) and the evaluation notebook refuses to open the test split if any best.pt differs from the frozen value. B0_s0 and B1_s0 were trained at commit 60fc83f, the others at aeaa4c2 (detectors-v2) and 81a91c8 (detectors-v3); training code and recipe are identical across these commits (later commits add tools only) | The evaluated models are provably the ones fixed before any test result |
| 2026-11-19 | Part A tools I3 + I4 + evaluation notebook, before the test split is opened: `scripts/04_premise_test.py --split test --open-test` (RQ1 confirmatory, Gate 2 rule, every detector, 2,000 bootstrap resamples per section 7; Gate 2 on valid kept 500 as frozen); `scripts/11_val_detectors.py` (standard Ultralytics validation, T2); `notebooks/kaggle_eval.ipynb` runs I1, I3, I4, I2 in one session, with a dry-run mode on valid and a check that every run is FINISHED; each script refuses the test split without `--open-test` and writes TEST_OPENED.json when it opens it | One session opens the test split; the dry run proves the pipeline first |
| 2026-11-18 | Part A tools I1 + I2, written and tested on valid before the test split is opened (implementation of 4.2, 4.3 and 7, no change of endpoint): `scripts/10_predict.py` (all images of a split, protocol NMS; the test split is refused without `--open-test`, which writes TEST_OPENED.json); `vgra/study.py` + `scripts/12_study_level.py`. Fixed details: (a) image score = own-image maximum if > 0.001 else 0, which equals the top box after NMS (checked identical on all 36,846 valid image-class rows of B0_s0); (b) a study is positive if its score is above the valid threshold = 95th percentile (numpy linear) of the negative-study scores; (c) stacking = logistic regression on [logit PA, logit LAT], fitted on all valid pairs and applied to test (on valid itself, development numbers use patient-grouped 5-fold out-of-fold scores); (d) every contrast also reports the specificity difference with McNemar on the negative studies (4.3 item 3: specificity achieved on test can differ from 95%); (e) Holm families = one contrast and one test over the findings, the primary test (fracture, max vs PA) unadjusted; (f) pooled seed bootstrap = the same resampled patients for every seed, mean of the per-seed differences; (g) subgroups use the whole-valid thresholds; torus = AO code containing M/2.1 on any image of the same study (patient, study number); age = age on the PA image; initial exam and cast = 1 on either image; (h) contrasts computed: max vs PA (primary for fracture), max vs LAT, LAT vs PA (RQ3), mean / noisy-OR / stacking vs max (RQ4) | Every analysis choice is in code and tested before any test result exists |
| 2026-11-18 | Amendment v1.1 (owner approval, before the test split was opened): V8R = YOLOv8s + ResCBAM of Ju et al. (best attention module published on this split), seeds 0 and 1, same recipe as V8, as an extra robustness detector (4.3 item 6) and a descriptive module comparison (4.3 item 7); 7-day deadline rule in section 3. Code tag `detectors-v3` (adds V8R only; B0, B1 and V8 code unchanged) | Answers "would a published attention detector change the conclusion?" and "is a published module gain larger than seed noise?"; decided before any test result |
| 2026-11-18 | v1.0 frozen; before freezing, section 5 was made explicit (Gate P training and evaluation settings) | Pre-registration completeness; no Gate P result existed |
