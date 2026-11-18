"""Presence branch (Gate P and PAC): does finding c appear anywhere in this image?

The branch reads the detector's P3/P4/P5 neck outputs, so it adds no backbone. It reuses the
tested class-query pooling (CQP, `vgra.modules.ClassQueryPooling`): one learned query per class
attends over all locations of the three levels. A projection embedding (PA / LAT / other) is added
to the queries, because some findings (pronator sign, soft-tissue swelling) are read on one
projection only. A per-class linear scorer turns each class descriptor into a presence logit.

Settings for Gate P are fixed in docs/PROTOCOL.md section 5.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vgra.modules import ClassQueryPooling

VIEWS = {"ap": 0, "lat": 1}  # anything else (other projections, unknown) -> 2
N_VIEWS = 3


def view_code(view: str) -> int:
    return VIEWS.get(str(view), 2)


class PresenceBranch(ClassQueryPooling):
    """CQP with projection-conditioned queries. forward(feats, view) -> presence logits (N, K)."""

    def __init__(self, channels: tuple, n_classes: int, d: int = 64, ffn: int = 2, n_views: int = N_VIEWS):
        super().__init__(channels, n_classes, d, ffn)
        self.view_embed = nn.Parameter(torch.zeros(n_views, self.d))
        nn.init.normal_(self.view_embed, std=0.02)

    def descriptors(self, feats: list, view: torch.Tensor) -> torch.Tensor:
        t = self.tokens(feats)  # (N, T, d)
        queries = self.query[None] + self.view_embed[view.long()][:, None, :]  # (N, K, d)
        attn = torch.einsum("nkd,ntd->nkt", self.q(queries), self.k(t)) / math.sqrt(self.d)
        h = torch.einsum("nkt,ntd->nkd", attn.softmax(-1), self.v(t))
        h = self.norm1(h + queries)
        return self.norm2(h + self.ffn(h))

    def forward(self, feats: list, view: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.presence(self.descriptors(feats, view))

    def attention(self, feats: list, view: torch.Tensor) -> torch.Tensor:
        """(N, K, T) attention of each class query over all tokens (for attention maps)."""
        t = self.tokens(feats)
        queries = self.query[None] + self.view_embed[view.long()][:, None, :]
        return (torch.einsum("nkd,ntd->nkt", self.q(queries), self.k(t)) / math.sqrt(self.d)).softmax(-1)


def presence_targets(batch: dict, n_images: int, n_classes: int) -> torch.Tensor:
    """(N, K) 0/1: class has at least one box in the (augmented) image."""
    y = torch.zeros(n_images, n_classes)
    if batch["cls"].numel():
        y[batch["batch_idx"].long(), batch["cls"].view(-1).long()] = 1.0
    return y


def positive_weights(index_rows, classes: list, cap: float = 50.0) -> torch.Tensor:
    """Per-class BCE positive weight min(negatives / positives, cap), counted on the given images."""
    n = len(index_rows)
    pos = np.array([float(index_rows[f"has_{c}"].sum()) for c in classes])
    w = np.where(pos > 0, (n - pos) / np.maximum(pos, 1.0), cap)
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32)


def presence_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.to(logits.device),
                                              pos_weight=pos_weight.to(logits.device))


def lr_at(step: int, total: int, base_lr: float, warmup: int = 200) -> float:
    """Linear warm-up, then cosine decay to 0 (protocol section 5)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
