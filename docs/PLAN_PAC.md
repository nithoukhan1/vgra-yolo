# Plan: combined paper (Part A analysis + Part B PAC), phases and steps

Full design: project doc `claude/PAC_PLAN.md`. Rules: `docs/PROTOCOL.md`. Decisions: `docs/DECISIONS.md`.
Tick a step only when its pass check is met. `[ ]` open, `[~]` running, `[x]` done, `[-]` dropped.

## Phase 0: setup and protocol (local, no GPU)
- [x] P0.1 Apply the phase-0 patch: V8 config, notebook v2, protocol draft, this plan. Merge, tag `detectors-v2`.
- [x] P0.2 Review `docs/PROTOCOL.md`; owner approves; status FROZEN; tag `protocol-v1` (2026-11-18).
      Pass check: tag exists; test split not yet opened.

## Phase 1: background detector training (co-author accounts, GPU)
- [~] P1.1 Notebook A (account owner runs it): B0 seeds 1 and 2. About 3 versions.
- [~] P1.2 Notebook B (account owner runs it): V8 seeds 0 and 1 (one per GPU). About 3 versions.
- [ ] P1.3 Guide for account owners: `docs/COAUTHOR_RUNS.md`. Each finished output saved as a dataset and shared with the project owner.
      Pass check: every run FINISHED; rows in `docs/EXPERIMENTS.md`.

## Phase 2: Part A tools (local code + short Kaggle checks)
- [ ] P2.1 S1 inference module: per-image class max scores, detections, TP tables; any split; paired and unpaired.
      Pass check: AP equals the Ultralytics validator on the same images (as in STEP-04).
- [ ] P2.2 S2 patient-level analysis: fusion rules, operating points, bootstrap, DeLong, McNemar, Holm, subgroups.
      Pass check: unit tests (known AUC / DeLong / McNemar values); valid numbers reproduce the pilot.

## Phase 3: Gate P (valid only, decision D1)
- [ ] P3.1 S3 presence branch + trainer (frozen detector, mosaic off, projection embedding) + evaluation script.
      Pass check: tests; identity checks; smoke run.
- [ ] P3.2 Run Gate P on B0_s0 (1-2 GPU-h).
- [ ] P3.3 D1: PASS -> Phases 4-5; FAIL -> skip to Phase 6 with Part A only.

## Phase 4: PAC (only if D1 = PASS)
- [ ] P4.1 S4 `PresenceDetect` head. Pass check: identical to the detector at start; checkpoint / EMA round trip.
- [ ] P4.2 S5 stage-2 trainer (resumable). Pass check: smoke run; detector weights unchanged (hash).
- [ ] P4.3 Choose epochs / loss / lambda on valid (B0_s0); fix them in the protocol deviations log if needed.
- [ ] P4.4 Main runs: PAC on B0_s0, s1, s2, B1_s0, V8_s0, V8_s1.

## Phase 5: controls and ablations (only if D1 = PASS)
- [ ] P5.1 Controls: separate image classifier gate; joint aux-loss only; per-class recalibration.
- [ ] P5.2 Ablations (B0_s0, one change each).

## Phase 6: one-time test evaluation
- [ ] P6.1 All models and choices fixed; protocol FROZEN.
- [ ] P6.2 Run every endpoint of `docs/PROTOCOL.md` on test once; save all tables.

## Phase 7: paper
- [ ] P7.1 Figures and tables script.
- [ ] P7.2 Manuscript section by section; CLAIM checklist.
- [ ] P7.3 Journal choice (JCR quartile checked), submission.

## Compute (session-hours, approximate)
| Block | Hours | Where |
|---|---|---|
| B0_s1 + B0_s2 | ~44 | co-author account A |
| V8_s0 + V8_s1 | ~44 | co-author account B |
| Gate P | ~2 | own account |
| PAC main + ablations + controls | ~35 | own + co-author accounts |
| Test evaluation | ~2 | own account |
