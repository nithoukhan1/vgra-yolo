"""VGRADetect (STEP-07): the stock Detect head plus VGRA.

VGRADetect is a subclass of Ultralytics' Detect. It keeps every Detect layer and weight and only
overrides `forward_head`: after the stock head computes boxes and class scores, VGRA replaces the
scores with the corrected scores. Boxes, DFL, the loss, decoding and NMS are untouched.

The pairing of the current batch is set from outside with `vgra_context(model, partner, view, ...)`:

    with vgra_context(model, partner, view, use_partner=keep):
        preds = model(batch["img"])

Without a context (or with VGRA switched off) the head is exactly the stock Detect head, so the
stock Ultralytics validator gives single-view results. The context is never saved in a checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from ultralytics.nn.modules.head import Detect

from vgra.modules import VGRA, build_vgra

_RUNTIME = ("_ctx", "last_aux")  # per-batch state, never copied or saved


class VGRADetect(Detect):
    """Detect head with a VGRA module acting on the class scores."""

    @classmethod
    def from_detect(cls, detect: Detect, vgra: VGRA) -> "VGRADetect":
        """Turn a built (and possibly trained) Detect head into a VGRADetect, reusing its layers."""
        if type(detect) is not Detect:
            raise TypeError(f"expected a plain Detect head, got {type(detect).__name__}")
        if getattr(detect, "end2end", False):
            raise ValueError("end-to-end (one-to-one) heads are not supported")
        head = cls.__new__(cls)
        head.__dict__ = detect.__dict__.copy()
        head._modules = detect._modules.copy()
        head._parameters = detect._parameters.copy()
        head._buffers = detect._buffers.copy()
        head.add_module("vgra", vgra)
        head.vgra_on = True
        head._ctx = None
        head.last_aux = None
        head.type = f"{cls.__module__}.{cls.__name__}"
        head.np = sum(p.numel() for p in head.parameters())
        return head

    def set_context(self, partner: torch.Tensor | None, view: torch.Tensor | None = None,
                    true_states: torch.Tensor | None = None, use_partner: torch.Tensor | None = None) -> None:
        """Pairing for the next forward pass(es). None clears it; the last aux output is kept for the loss
        until a new context is set."""
        if partner is None:
            self._ctx = None
            return
        self._ctx = {"partner": partner, "view": view, "true_states": true_states, "use_partner": use_partner}
        self.last_aux = None

    def forward_head(self, x: list, box_head: nn.Module = None, cls_head: nn.Module = None) -> dict:
        preds = super().forward_head(x, box_head, cls_head)
        ctx = getattr(self, "_ctx", None)
        if not preds or not getattr(self, "vgra_on", True) or ctx is None:
            return preds
        n = preds["scores"].shape[0]
        dev = preds["scores"].device
        args = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in ctx.items()}
        if args["partner"].shape[0] != n:
            raise ValueError(f"VGRA context is for {args['partner'].shape[0]} images, batch has {n}")
        scores, aux = self.vgra(x, preds["scores"], args["partner"], args["view"],
                                true_states=args["true_states"], use_partner=args["use_partner"])
        preds["scores"] = scores
        self.last_aux = aux
        return preds

    def __getstate__(self):  # pickling (checkpoints) and deepcopy (EMA) skip per-batch state
        state = self.__dict__.copy()
        for key in _RUNTIME:
            state[key] = None
        return state


def vgra_head(model: nn.Module) -> VGRADetect:
    head = model.model[-1]
    if not isinstance(head, VGRADetect):
        raise TypeError(f"model has no VGRADetect head (last layer is {type(head).__name__})")
    return head


def add_vgra(model: nn.Module, cfg: dict, gate: str | None = None, imgsz: int = 256) -> nn.Module:
    """Replace the Detect head of a DetectionModel (in place) by a VGRADetect with fresh VGRA modules.

    The channel widths of P3/P4/P5 are read from a dry run, so the call works for any model scale.
    With fresh modules the model gives exactly the same outputs as before.
    """
    head = model.model[-1]
    channels = head_input_channels(model, imgsz)
    vgra = build_vgra(cfg, channels, gate=gate)
    p = next(model.parameters())
    new = VGRADetect.from_detect(head, vgra.to(device=p.device, dtype=p.dtype))
    model.model[-1] = new
    return model


@torch.no_grad()
def head_input_channels(model: nn.Module, imgsz: int = 256) -> tuple:
    """Channel widths of the tensors that enter the Detect head (P3, P4, P5)."""
    head = model.model[-1]
    seen = {}

    def grab(module, inputs):  # a pre-hook must return None, or it replaces the inputs
        seen["c"] = tuple(t.shape[1] for t in inputs[0])

    hook = head.register_forward_pre_hook(grab)
    was_training = model.training
    p = next(model.parameters())
    try:
        model.eval()
        model(torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype))
    finally:
        hook.remove()
        model.train(was_training)
    return seen["c"]


@contextmanager
def vgra_context(model: nn.Module, partner: torch.Tensor, view: torch.Tensor, true_states: torch.Tensor | None = None,
                 use_partner: torch.Tensor | None = None):
    """Set the pairing of the next forward pass(es); always cleared on exit."""
    head = vgra_head(model)
    head.set_context(partner, view, true_states=true_states, use_partner=use_partner)
    try:
        yield head
    finally:
        head.set_context(None)


def vgra_parameters(model: nn.Module) -> list:
    """Parameters of the VGRA modules only (stage 2 trains these with the detector frozen)."""
    return list(vgra_head(model).vgra.parameters())


def freeze_detector(model: nn.Module) -> int:
    """Stage 2: freeze everything except VGRA. Returns the number of trainable parameters.
    BatchNorm layers must also be kept in eval mode by the trainer (see `detector_eval`)."""
    keep = {id(p) for p in vgra_parameters(model)}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detector_eval(model: nn.Module) -> None:
    """Put every BatchNorm of the frozen detector in eval mode (running statistics fixed)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
