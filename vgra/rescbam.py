"""ResCBAM: the residual CBAM block of Ju et al. (YOLOv8-AM), in YOLOv8s, for the V8R detector.

Source of the design (copied faithfully, not re-designed):
  Ju, Chien, Chiang. YOLOv8-AM / YOLOv8-ResCBAM (ICONIP 2024; IEEE Access 2025),
  https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8
  - block `ResBlock_CBAM` in `ultralytics/nn/modules/conv.py` (their Ultralytics 8.0.147 fork),
  - layout `ultralytics/cfg/models/v8/yolov8_ResBlock_CBAM.yaml`: one block after each of the four
    C2f layers of the neck; the detection head reads the four block outputs except that the P4
    skip connection of the bottom-up path reads the C2f output before the block (as in their file).
  Their CBAM is the Ultralytics CBAM (channel attention = 1x1 conv + sigmoid on the pooled feature,
  then 7x7 spatial attention), so the Ultralytics class is used directly.

Ultralytics itself is never edited. `register()` makes the name `ResBlock_CBAM` known to the
Ultralytics model parser at run time, so the YAML file `configs/models/yolov8s-rescbam.yaml` can be
built. Pretrained YOLOv8s (COCO) weights are copied into every layer that also exists in stock
YOLOv8s (`transfer_stock_weights`); the four ResCBAM blocks start from random initialisation.
This matches the V8 detector (stock YOLOv8s from the same weights), so V8R vs V8 isolates the module.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import CBAM

YAML_PATH = Path(__file__).resolve().parents[1] / "configs" / "models" / "yolov8s-rescbam.yaml"
NAME = "ResBlock_CBAM"  # the name used in Ju et al.'s YAML file and in ours


class ResBlockCBAM(nn.Module):
    """Residual block with CBAM (Ju et al.): 1x1 -> 3x3 -> 1x1 bottleneck (BatchNorm, LeakyReLU 0.1),
    CBAM on the result, identity shortcut, ReLU. Input and output have `c1` channels.

    Same arguments and same computation as their `ResBlock_CBAM(in_places, places)` with stride 1,
    no downsampling and expansion 1 (the only setting their YAML uses).
    """

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c2 != c1:
            raise ValueError(f"ResBlockCBAM keeps the channel count (identity shortcut): got {c1} -> {c2}")
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.cbam = CBAM(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cbam(self.bottleneck(x))
        return self.relu(out + x)


def register() -> None:
    """Make `ResBlock_CBAM` in a YAML file resolve to ResBlockCBAM (run-time only, files untouched)."""
    import ultralytics.nn.tasks as tasks

    setattr(tasks, NAME, ResBlockCBAM)


def build(nc: int, verbose: bool = False):
    """YOLOv8s + ResCBAM DetectionModel with `nc` classes, randomly initialised."""
    from ultralytics.nn.tasks import DetectionModel

    register()
    return DetectionModel(str(YAML_PATH), nc=nc, verbose=verbose)


def is_rescbam(model: nn.Module) -> bool:
    return any(isinstance(m, ResBlockCBAM) for m in model.modules())


def block_indices(model: nn.Module) -> list[int]:
    """Layer indices of the ResCBAM blocks in an Ultralytics DetectionModel."""
    return [i for i, m in enumerate(model.model) if isinstance(m, ResBlockCBAM)]


def stock_layer_map(model: nn.Module) -> dict[int, int]:
    """Stock YOLOv8s layer index -> layer index in `model` (the ResCBAM blocks are skipped)."""
    ours = [i for i, m in enumerate(model.model) if not isinstance(m, ResBlockCBAM)]
    return {j: i for j, i in enumerate(ours)}


def transfer_stock_weights(model: nn.Module, stock) -> dict:
    """Copy the weights of a stock YOLOv8s model (module, checkpoint dict or state dict) into `model`.

    Layers are matched by position after skipping the ResCBAM blocks, and every tensor must match in
    name and shape. The Detect head keeps its own class layers when the class count differs (as
    Ultralytics does). Returns counts for the log.
    """
    if isinstance(stock, dict) and "model" in stock:
        stock = stock.get("ema") or stock["model"]
    sd = stock.float().state_dict() if isinstance(stock, nn.Module) else stock
    mapping = stock_layer_map(model)
    target = model.state_dict()
    new, skipped = {}, []
    for k, v in sd.items():
        parts = k.split(".")
        if parts[0] != "model" or not parts[1].isdigit():
            skipped.append(k)
            continue
        j = int(parts[1])
        if j not in mapping:
            skipped.append(k)
            continue
        k2 = ".".join(["model", str(mapping[j]), *parts[2:]])
        if k2 in target and target[k2].shape == v.shape:
            new[k2] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(new, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys after transfer: {unexpected[:5]}")
    blocks = set(block_indices(model))
    not_from_stock = [k for k in missing if int(k.split(".")[1]) not in blocks]
    return {"transferred": len(new), "stock_tensors": len(sd), "skipped": skipped,
            "missing_outside_blocks": not_from_stock}
