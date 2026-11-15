# EDA report and Gate 1 decision

Source: `scripts/01_build_index.py` (commit `5859c67`) and `scripts/02_pair_audit.py` (commit `3f1778c`),
run on Kaggle on 2026-11-15. All numbers use the original images only (one per file stem).
The tables are in `results/eda/`.

## 1. Dataset and split

| Item | Value |
|---|---|
| Images / patients | 20,327 / 6,091 |
| Split (Ju et al., patient-level) | train 14,204 · valid 4,094 · test 2,029 images |
| Patients per split | 4,263 · 1,218 · 610; no patient in two splits |
| Boxes | 47,443; 15 images have no box |
| Training folder | 14,204 originals (`orig_*`) + 14,204 offline copies (`aug_*`) with identical labels |

Index checks (9 of 9 passed): every image has metadata and a label file, folders agree with the split
files, copies share the original's labels, file names agree with the metadata, totals match.

## 2. Classes

| Class | Images | Boxes | Train | Valid | Test images |
|---|---|---|---|---|---|
| fracture | 13,550 | 18,090 | 9,485 | 2,755 | 1,310 |
| periostealreaction | 2,235 | 3,453 | 1,557 | 458 | 220 |
| metal | 707 | 818 | 488 | 147 | 72 |
| pronatorsign | 566 | 567 | 407 | 104 | 55 |
| softtissue | 439 | 464 | 299 | 84 | 56 |
| boneanomaly | 192 | 276 | 121 | 38 | 33 |
| bonelesion | 42 | 45 | 26 | 8 | 8 |
| foreignbody | 8 | 8 | 8 | 0 | 0 |
| text | 20,274 | 23,722 | 14,170 | 4,085 | 2,019 |

foreignbody has no validation or test image and cannot be evaluated. bonelesion has 8 test images.

## 3. Views and pairs

| Item | Value |
|---|---|
| Views | ap 10,086 · lat 10,148 · other (projection 3) 93 |
| View coding | Confirmed: 563 of 566 pronator sign images are projection 2, so 2 = lateral, 1 = AP/PA |
| Study-sides (patient + study + wrist side) | 10,842 |
| Pairs (exactly one ap + one lat) | 9,288 = 85.67% of study-sides |
| Images in pairs | 18,576 = 91.39% of images; 1,751 images stay unpaired |
| Pairs per split | train 6,475 · valid 1,884 · test 929 |
| Pairs by exam type | initial, no cast 5,003 · follow-up, cast 2,606 · follow-up, no cast 1,677 · other 2 |

Unpaired study-sides: 744 lat only, 620 ap only, the rest have duplicates or a projection-3 image.

## 4. Class presence by view (all images)

| Class | ap images | lat images | Share in lat |
|---|---|---|---|
| pronatorsign | 3 | 563 | 99.5% |
| softtissue | 72 | 366 | 83.6% |
| fracture | 6,553 | 6,924 | 51.4% |
| metal | 351 | 353 | 50.1% |
| periostealreaction | 1,154 | 1,072 | 48.2% |
| boneanomaly | 122 | 68 | 35.8% |
| bonelesion | 28 | 14 | 33.3% |

## 5. Visibility states over the 9,288 pairs

State = two digits, AP first: 00 neither, 10 AP only, 01 LAT only, 11 both.

| Class | 00 | 10 | 01 | 11 | Present | One-view-only | Share of present |
|---|---|---|---|---|---|---|---|
| fracture | 2,773 | 137 | 521 | 5,857 | 6,515 | 658 | 10.1% |
| periostealreaction | 7,983 | 304 | 232 | 769 | 1,305 | 536 | 41.1% |
| pronatorsign | 8,767 | 0 | 520 | 1 | 521 | 520 | 99.8% |
| softtissue | 8,907 | 40 | 317 | 24 | 381 | 357 | 93.7% |
| metal | 8,964 | 1 | 4 | 319 | 324 | 5 | 1.5% |
| boneanomaly | 9,166 | 65 | 5 | 52 | 122 | 70 | 57.4% |
| bonelesion | 9,263 | 12 | 0 | 13 | 25 | 12 | 48.0% |
| foreignbody | 9,284 | 0 | 0 | 4 | 4 | 0 | 0.0% |

Test pairs with the class present: fracture 634, periostealreaction 132, pronatorsign 51, softtissue 47,
metal 30, boneanomaly 22, bonelesion 3.

Fracture one-view-only share by exam type: initial exams 20.2%; follow-up with cast 4.6%; follow-up
without cast 4.2%.

## 6. What the partner's label says about the target's label

Same class in both views. "given" = the partner has the class; "otherwise" = it does not.

| Class | P(ap has it) given lat / otherwise | P(lat has it) given ap / otherwise |
|---|---|---|
| fracture | 0.918 / 0.047 | 0.977 / 0.158 |
| periostealreaction | 0.768 / 0.037 | 0.717 / 0.028 |
| metal | 0.988 / 0.000 | 0.997 / 0.000 |
| boneanomaly | 0.912 / 0.007 | 0.444 / 0.001 |
| softtissue | 0.070 / 0.005 | 0.375 / 0.034 |
| pronatorsign | 0.002 / 0.000 | (ap has it in 1 pair) / 0.056 |

Other classes in the partner (cross-class), main effects only:

| Target | Base rate | Partner finding | Probability |
|---|---|---|---|
| fracture in lat | 0.687 | periostealreaction in ap | 0.996 |
| fracture in ap | 0.645 | periostealreaction in lat | 0.995 |
| fracture in ap | 0.645 | pronatorsign in lat | 0.827 |
| pronatorsign in lat | 0.056 | fracture in ap | 0.072 |
| pronatorsign in lat | 0.056 | periostealreaction in ap | 0.004 |
| pronatorsign in lat | 0.056 | metal in ap | 0.006 |
| pronatorsign in lat | 0.056 | softtissue in ap (64 pairs) | 0.406 |
| softtissue in lat | 0.037 | fracture in ap | 0.037 |
| softtissue in lat | 0.037 | softtissue in ap (64 pairs) | 0.375 |
| periostealreaction in ap | 0.116 | pronatorsign in lat | 0.008 |

## 7. Gate 1 decision

Gate 1 asks whether the data can support cross-view assistance at all.

| Condition | Result |
|---|---|
| At least 70% of images in a pair | 91.4%: met |
| At least 3 classes with one-view-only cases (100 training pairs and 10% of present) | 4 classes: met |

**Decision: Gate 1 passed (2026-11-16).**

Reading of the result, class by class, for a gate that uses the same class in the partner view:

| Class | Verdict | Reason |
|---|---|---|
| periostealreaction | Strong candidate | Strong partner signal; 41% one-view-only in both directions |
| fracture | Candidate | Strong partner signal; 10% one-view-only (20% in initial exams); already detected well |
| boneanomaly | Candidate, low support | Strong partner signal for the AP view; 22 test pairs |
| metal | Agreement only | Views agree in 98.5% of pairs; already detected well |
| softtissue | Weak | Mostly a lateral finding; the AP partner shows it in 64 pairs |
| pronatorsign | Not usable | The AP image never shows it, so the partner never carries the same class |
| bonelesion, foreignbody, text | Excluded | Too few pairs, not evaluable, or not a finding |

Consequences:

1. The one-view-only count was a poor proxy for pronatorsign and softtissue: they are lateral findings by
   nature, not findings that happen to be visible in one view.
2. Cross-class evidence is mostly exclusion. Healing-stage findings in the partner (periosteal reaction,
   metal) make an acute sign (pronator sign) very unlikely. A fracture in the AP view raises the chance of a
   pronator sign in the lateral view only slightly (0.072 against a base of 0.056).
3. Label statistics show what the partner could tell the detector. Whether the detector actually makes
   errors that this information fixes is measured by the premise test (STEP-04, Gate 2) and the oracle run
   (STEP-08, Gate 3).
