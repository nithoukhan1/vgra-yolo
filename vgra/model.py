"""Model building for single-view detectors (STEP-03).

B0 = stock YOLO11s. B1 = YOLO11s + SCC. Ultralytics itself is never edited:
a stock model is built first and `add_scc` swaps the two early backbone blocks.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from vgra.scc import SCC_LAYERS, add_scc


def make_trainer(scc: bool, scc_layers: tuple = SCC_LAYERS) -> type:
    """Return a DetectionTrainer class that builds the model with or without SCC.

    The flag is a class attribute because Ultralytics rejects unknown training arguments.
    Single-GPU training only (one experiment per GPU).
    """

    class DetectorTrainer(DetectionTrainer):
        use_scc = bool(scc)
        layers = tuple(scc_layers)

        def get_model(self, cfg=None, weights=None, verbose=True):
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


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
