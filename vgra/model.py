"""Model building for single-view detectors (STEP-03).

B0 = stock YOLO11s. B1 = YOLO11s + SCC. V8 = stock YOLOv8s. V8R = YOLOv8s + ResCBAM (Ju et al.).
Ultralytics itself is never edited: for SCC a stock model is built first and `add_scc` swaps the two
early backbone blocks; for ResCBAM the model is built from `configs/models/yolov8s-rescbam.yaml` and the
pretrained YOLOv8s weights are copied into every layer it shares with stock YOLOv8s (`vgra/rescbam.py`).
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from vgra.scc import SCC_LAYERS, add_scc


def make_trainer(scc: bool, scc_layers: tuple = SCC_LAYERS, rescbam: bool = False) -> type:
    """Return a DetectionTrainer class that builds the model with or without SCC, or as YOLOv8s + ResCBAM.

    The flag is a class attribute because Ultralytics rejects unknown training arguments.
    Single-GPU training only (one experiment per GPU). SCC and ResCBAM are separate detectors.
    """
    if scc and rescbam:
        raise ValueError("SCC (YOLO11s) and ResCBAM (YOLOv8s) are separate detectors; set only one")

    class DetectorTrainer(DetectionTrainer):
        use_scc = bool(scc)
        use_rescbam = bool(rescbam)
        layers = tuple(scc_layers)

        def get_model(self, cfg=None, weights=None, verbose=True):
            if self.use_rescbam:
                return _rescbam_model(self.data["nc"], weights, verbose and RANK == -1)
            model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
            if self.use_scc:
                before = count_parameters(model)
                add_scc(model, self.layers)  # before loading, so native weights land in the new blocks
                LOGGER.info(f"SCC added at backbone layers {list(self.layers)}: "
                            f"+{count_parameters(model) - before:,} parameters")
            if weights:
                model.load(weights)
            return model

    return DetectorTrainer


def _rescbam_model(nc: int, weights, verbose: bool):
    """YOLOv8s + ResCBAM. `weights` is stock YOLOv8s (new run: shared layers copied) or a V8R model
    (resumed run: everything loaded)."""
    from vgra import rescbam

    model = rescbam.build(nc, verbose=verbose)
    if weights is not None:
        if rescbam.is_rescbam(weights):
            model.load(weights)
        else:
            r = rescbam.transfer_stock_weights(model, weights)
            LOGGER.info(f"ResCBAM: copied {r['transferred']}/{r['stock_tensors']} pretrained YOLOv8s tensors "
                        f"(not copied: {len(r['skipped'])}, the class layers of the 80-class head); "
                        f"ResCBAM blocks at layers {rescbam.block_indices(model)} start from random init")
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
