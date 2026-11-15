# Naming

One scheme, used in code, configs, docs and the paper.

## Model and parts

| Name | Full name | What it is | Code |
|---|---|---|---|
| VGRA-YOLO | | Full model: YOLO11s + SCC + VGRADetect | `vgra.model.build_model()` |
| SCC | Self-Calibrated Convolution | Adapter in the two early backbone C3k2 blocks | `vgra/scc.py` |
| `C3K2_SCC` | | C3k2 block with the SCC adapter; backbone layers 2 and 4 | `vgra/scc.py` |
| VGRA | Visibility-State-Gated Cross-View Residual Assistance | Cross-view method = CQP + VSH + GRA | `vgra/modules.py` |
| `VGRADetect` | | Detection head: stock Detect head + CQP, VSH, GRA | `vgra/head.py` |
| CQP | Class-Query Pooling | One descriptor per class per view | `ClassQueryPooling` |
| VSH | Visibility State Head | Four visibility states per class | `VisibilityStateHead` |
| GRA | Gated Residual Assistance | Gated, capped correction to class logits | `GatedResidual` |

SCC classes: `SCC` (the operation), `SCCAdapter` (zero-initialised residual wrapper), `C3K2_SCC` (the block).

In the paper, write SCC out as "self-calibrated convolution (SCNet, CVPR 2020)" at first use.

## Data and training terms

| Name | Meaning |
|---|---|
| `ap`, `lat` | The two views |
| partner | The other view of the same study |
| pair | One `ap` image + one `lat` image of the same patient and study |
| state | Two digits per class, AP first: `00` neither, `10` AP only, `01` LAT only, `11` both |
| `L_det` | Standard YOLO detection loss |
| `L_vis` | Visibility loss (four-state) |
| `L_pres` | Presence loss (per view, per class) |
| partner dropout | Training rule: hide the partner with probability `p_drop` |

## Project terms

| Thing | Rule | Example |
|---|---|---|
| Step | `STEP-00` ... `STEP-12` | |
| Branch / tag | `step-NN-name` / `stepNN` | `step-06-modules` / `step06` |
| Commit message | starts with the step | `STEP-06: add GatedResidual` |
| Experiment ID | B baseline, V full model, A ablation, C control | `A2a` |
| Run name | `<expID>_s<seed>` | `V1_s0` |
| Config | `configs/<expID>.yaml` | `configs/V1.yaml` |
| Run folder | `runs/<run name>/` | `runs/V1_s0/` |
| Files | lowercase snake_case | `pair_audit.py` |

## Names that must not be used

SCConv-EARLY, ESC, `SCConv`, `SelfCalibConv`, `C3k2_TPSC`, `C3k2ESC`, companion.
