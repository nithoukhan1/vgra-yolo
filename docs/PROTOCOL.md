# Analysis protocol (DRAFT v0.1, 2026-11-18)

Status: DRAFT for review. When the project owner approves it, the status becomes FROZEN, the
commit is tagged `protocol-v1`, and only then may the test split be opened. Any later change is
logged in section 10 with date and reason (deviations are allowed, but always reported).

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
| +PAC | PAC on each detector above (Part B, only if Gate P passes) | as detector | fixed number of epochs set on valid (section 6) |
Common recipe: imgsz 1024, batch 16, 100 epochs, SGD lr0 0.01, patience 50, Ultralytics defaults.

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
  6. Every endpoint repeated for B1_s0 and V8 seeds 0-1 (robustness across detectors).

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
- Model: presence branch (class-query pooling over P3/P4/P5 + projection embedding + per-class
  presence scorer) trained on the frozen B0_s0 detector, train split originals, mosaic off.
  Training length and loss are fixed in the Gate P script before it is run on valid.
- Evaluation on valid pairs and images with the STEP-04 machinery (patient-grouped 5-fold
  cross-fitting, 500 patient bootstraps):
  - detection level, target = true positive at IoU 0.5:
    base [logit conf, view, logit own image max of c] vs base + [logit p_c, view x logit p_c];
  - rescored AP50 from the out-of-fold probabilities of both models.
- Target classes: pronatorsign, softtissue, periostealreaction.
- **PASS** if at least 2 of the 3 target classes have a detection-level rescored AP50 gain whose
  95% patient-bootstrap CI lies above 0, AND the mean AP50 over all classes with labels is not
  lower than the base. Otherwise **FAIL**: Part B is dropped and Part A is published alone.

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
| | | |
