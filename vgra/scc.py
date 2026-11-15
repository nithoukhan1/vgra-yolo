"""SCC: self-calibrated convolution in the early backbone (STEP-03).

SCC is the self-calibrated convolution of SCNet (CVPR 2020). It is added to the two
early backbone C3k2 blocks of YOLO11 (layers 2 and 4) as a residual adapter:

    Y = X + tanh(alpha) * SCC(X),   alpha starts at 0

so a block with fresh adapters gives exactly the output of the native block, and
pretrained YOLO11 weights keep their behaviour at the start of training.

Classes
- SCC         : the operation
- SCCAdapter  : zero-initialised residual wrapper around SCC
- C3K2_SCC    : native C3k2 block with one adapter after each inner block
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.utils.torch_utils import initialize_weights

SCC_LAYERS = (2, 4)  # backbone layers of YOLO11 that hold the early C3k2 blocks (P2/4 and P3/8)


class SCC(nn.Module):
    """Self-calibrated convolution: features are rescaled by a map built from a pooled copy of themselves."""

    def __init__(self, channels: int, pooling_r: int = 4, groups: int = 1):
        super().__init__()
        if pooling_r < 1 or groups < 1:
            raise ValueError("pooling_r and groups must be at least 1")
        if channels % groups:
            raise ValueError("channels must be divisible by groups")

        def conv_bn() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(channels),
            )

        self.k2 = nn.Sequential(nn.AvgPool2d(pooling_r, pooling_r), conv_bn())  # context branch
        self.k3 = conv_bn()
        self.k4 = conv_bn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = F.interpolate(self.k2(x), size=x.shape[-2:], mode="nearest")
        calibration = torch.sigmoid(x + context)
        return self.k4(self.k3(x) * calibration)


class SCCAdapter(nn.Module):
    """Y = X + tanh(alpha) * SCC(X). With alpha = 0 the adapter returns X unchanged."""

    def __init__(self, channels: int, pooling_r: int = 4, groups: int = 1):
        super().__init__()
        self.scc = SCC(channels, pooling_r=pooling_r, groups=groups)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.alpha) * self.scc(x)

    def strength(self) -> float:
        """tanh(alpha): 0 means the adapter is switched off, values near +-1 mean fully on."""
        return float(torch.tanh(self.alpha.detach())) + 0.0  # "+ 0.0" turns -0.0 into 0.0


class C3K2_SCC(C3k2):
    """YOLO11 C3k2 block with an SCC adapter after each native inner block."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, attn: bool = False,
                 g: int = 1, shortcut: bool = True, *, pooling_r: int = 4, scc_groups: int = 1):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        self.scc_adapters = nn.ModuleList(
            SCCAdapter(self.c, pooling_r=pooling_r, groups=scc_groups) for _ in range(n)
        )

    def _fuse(self, y: list) -> torch.Tensor:
        for block, adapter in zip(self.m, self.scc_adapters):
            y.append(adapter(block(y[-1])))
        return self.cv2(torch.cat(y, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fuse(list(self.cv1(x).chunk(2, 1)))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        return self._fuse([y[0], y[1]])

    @classmethod
    def from_native(cls, block: C3k2, pooling_r: int = 4, scc_groups: int = 1) -> "C3K2_SCC":
        """Build a C3K2_SCC with the same shape and weights as an existing native C3k2 block."""
        if type(block) is not C3k2:
            raise TypeError(f"expected a native C3k2 block, got {type(block).__name__}")
        first = block.m[0]
        attn = isinstance(first, nn.Sequential)
        c3k = isinstance(first, C3k)
        bottleneck = first[0] if attn else (first.m[0] if c3k else first)
        c2 = block.cv2.conv.out_channels
        new = cls(
            c1=block.cv1.conv.in_channels,
            c2=c2,
            n=len(block.m),
            c3k=c3k,
            e=block.c / c2,
            attn=attn,
            g=bottleneck.cv2.conv.groups,
            shortcut=bool(bottleneck.add),
            pooling_r=pooling_r,
            scc_groups=scc_groups,
        )
        if new.c != block.c:
            raise RuntimeError(f"hidden width changed: {new.c} != {block.c}")
        initialize_weights(new)  # same BatchNorm eps/momentum as the rest of the Ultralytics model
        missing, unexpected = new.load_state_dict(block.state_dict(), strict=False)
        if unexpected or any(not k.startswith("scc_adapters.") for k in missing):
            raise RuntimeError(f"weight transfer failed: missing={missing}, unexpected={unexpected}")
        ref = next(block.parameters())
        new.to(device=ref.device, dtype=ref.dtype)
        new.train(block.training)
        return new


def add_scc(model: nn.Module, layers: tuple = SCC_LAYERS, pooling_r: int = 4, scc_groups: int = 1) -> nn.Module:
    """Replace the native C3k2 blocks at `layers` of an Ultralytics DetectionModel by C3K2_SCC.

    The native weights are copied, so the model's outputs do not change.
    """
    seq = model.model
    for i in layers:
        old = seq[i]
        if isinstance(old, C3K2_SCC):
            continue  # already converted
        new = C3K2_SCC.from_native(old, pooling_r=pooling_r, scc_groups=scc_groups)
        new.i, new.f = old.i, old.f  # layer index and input index used by the Ultralytics forward pass
        new.type = f"{C3K2_SCC.__module__}.{C3K2_SCC.__name__}"
        new.np = sum(p.numel() for p in new.parameters())
        seq[i] = new
    return model


def scc_strengths(model: nn.Module) -> dict:
    """tanh(alpha) of every adapter, keyed by module name. Empty if the model has no SCC."""
    return {name: m.strength() for name, m in model.named_modules() if isinstance(m, SCCAdapter)}
