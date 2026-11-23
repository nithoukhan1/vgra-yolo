# Part B results: CoVeR (conformal view referral), aggregate tables only

`valid_design/` = DESIGN STUDY on the validation split (2026-11-23, exploratory, after the Part A test session; the
test split was not used). `scripts/13_cover.py --eval valid --ensemble --splits 300` on the predictions of the
8-detector dry run (2026-11-21): for every run (and ENS = mean score of the 8), 300 patient-level half-splits of valid
(threshold t and lo calibrated on one half, evaluated on the other) for every grouping (all ages / per age band),
method (naive, CRC, LTT) and alpha (0.01, 0.02), delta 0.1.

| File | Content |
|---|---|
| `summary.txt` | printed tables: primary configuration (proposal) per run, all configurations, who is referred |
| `design_summary.csv` | run x grouping x method x alpha: means over the 300 splits (lateral reads, sensitivity, specificity, extra-miss risk, share of splits where the guarantee held, worst age group, same-budget baselines: random and detector disagreement, oracle reads), 5th-95th percentile of lateral reads, 90th percentile of the extra-miss risk |
| `curves.csv` | all of valid (descriptive): lateral reads vs sensitivity / specificity / extra-miss risk over lo |
| `who.csv` | all of valid (descriptive): primary configuration calibrated and applied on valid; lateral reads and extra misses per subgroup |
| `info.json` | commit, runs, settings |

The per-split table (`design_splits.csv.gz`, 32,400 rows) stays outside git; it is reproducible with the command above
from the dry-run predictions. Rules for any use on test: protocol amendment v1.2, section 11.4 (frozen
2026-11-23, tag `protocol-v1.2`); the test results will go to `test/`.

`test/` = the one CoVeR test run (amendment v1.2, section 11.4; 2026-11-23 08:47:35 UTC; code = tag `protocol-v1.2`,
tree 6b36ac0; `COVER_TEST_RUN.json` shows commit baada13, a local commit with the identical tree). Part A consistency
check PASS.

| File | Content |
|---|---|
| `test/summary.txt` | printed tables: consistency check, primary configuration per run, all configurations, who is referred |
| `test/results.csv` | run x grouping x method x alpha: lateral reads, sensitivity, specificity, extra-miss risk (patient and pair), limit held, same-budget baselines, McNemar counts, patient-bootstrap 95% CIs (2,000), lo |
| `test/who.csv` | primary configuration: lateral reads and extra misses per subgroup (age band, torus, first visit, cast) |
| `test/COVER_TEST_RUN.json` | time, commit, runs, configuration |
| `test_exploratory/` | POST HOC (`scripts/14_cover_explore.py`): valid vs test composition of the fracture-positive pairs |
