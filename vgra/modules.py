"""VGRA modules (STEP-06): CQP, VSH and GRA.

VGRA = Visibility-State-Gated Cross-View Residual Assistance.

Shapes (N images in the batch, K gateable classes, d descriptor width, r rank, A anchors):
    feats   list of 3 tensors (N, C_l, H_l, W_l)  the P3/P4/P5 inputs of the Detect head
    scores  (N, nc, A)                             class logits of all anchors, levels flattened
                                                   (Ultralytics 8.4.7 Detect.forward_head layout)
    partner (N,) long                              index of the other view of the same study-side,
                                                   -1 when the image has no partner (or it was dropped)
    view    (N,) long                              0 = AP, 1 = LAT
    use_partner (N,) bool, optional                False switches the correction off for that image
                                                   while VSH still sees the pair (partner dropout)

Visibility states per pair and class, AP digit first:
    0 = '00' (neither view), 1 = '10' (AP only), 2 = '01' (LAT only), 3 = '11' (both)
so state = has_ap + 2 * has_lat.

Guarantees (tested):
    * at initialisation the corrected scores equal the input scores exactly (theta = 0);
    * an image without a partner gets exactly zero correction;
    * only the gateable class channels change; boxes are never touched;
    * nothing mixes information across images other than an image and its partner.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

STATES = ("00", "10", "01", "11")
GATES = ("signed", "unsigned", "none", "oracle")


def paired_layout(n_pairs: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """partner and view for the loader layout [AP_1..AP_B, LAT_1..LAT_B]."""
    b = int(n_pairs)
    ar = torch.arange(b, device=device)
    partner = torch.cat([ar + b, ar])
    view = torch.cat([torch.zeros(b, dtype=torch.long, device=device), torch.ones(b, dtype=torch.long, device=device)])
    return partner, view


def pair_rows(partner: torch.Tensor, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (ap, lat) of every complete pair in the batch, ordered by the AP image."""
    ap = torch.nonzero((view == 0) & (partner >= 0), as_tuple=False).view(-1)
    lat = partner[ap]
    if (view[lat] != 1).any() or (partner[lat] != ap).any():
        raise ValueError("partner/view are inconsistent: an AP image must point to a LAT image and back")
    return ap, lat


def state_targets(has: torch.Tensor, ap: torch.Tensor, lat: torch.Tensor, gate_idx) -> torch.Tensor:
    """(P, K) state index per pair and gateable class from an (N, nc) 0/1 class-presence table."""
    h = has[:, list(gate_idx)].long()
    return h[ap] + 2 * h[lat]


class ClassQueryPooling(nn.Module):
    """CQP: one learnable query per gateable class attends over the tokens of P3/P4/P5.

    Output h (N, K, d): one descriptor per image and class. A per-class linear scorer on h gives
    presence logits (N, K), trained by L_pres on all images (paired or not).
    """

    def __init__(self, channels: tuple, n_classes: int, d: int = 64, ffn: int = 2):
        super().__init__()
        self.d = int(d)
        self.proj = nn.ModuleList(nn.Conv2d(c, self.d, 1) for c in channels)
        self.level = nn.Parameter(torch.zeros(len(channels), self.d))
        self.query = nn.Parameter(torch.randn(n_classes, self.d) * 0.02)
        self.q = nn.Linear(self.d, self.d)
        self.k = nn.Linear(self.d, self.d)
        self.v = nn.Linear(self.d, self.d)
        self.norm1 = nn.LayerNorm(self.d)
        self.ffn = nn.Sequential(nn.Linear(self.d, ffn * self.d), nn.SiLU(), nn.Linear(ffn * self.d, self.d))
        self.norm2 = nn.LayerNorm(self.d)
        self.pres_w = nn.Parameter(torch.zeros(n_classes, self.d))
        self.pres_b = nn.Parameter(torch.zeros(n_classes))
        nn.init.normal_(self.level, std=0.02)

    def tokens(self, feats: list) -> torch.Tensor:
        out = []
        for i, (x, proj) in enumerate(zip(feats, self.proj)):
            t = proj(x).flatten(2).transpose(1, 2)  # (N, H*W, d)
            out.append(t + self.level[i])
        return torch.cat(out, 1)

    def forward(self, feats: list) -> torch.Tensor:
        t = self.tokens(feats)  # (N, T, d)
        q = self.q(self.query)  # (K, d)
        attn = torch.einsum("kd,ntd->nkt", q, self.k(t)) / math.sqrt(self.d)
        h = torch.einsum("nkt,ntd->nkd", attn.softmax(-1), self.v(t))
        h = self.norm1(h + self.query)
        return self.norm2(h + self.ffn(h))

    def presence(self, h: torch.Tensor) -> torch.Tensor:
        return (h * self.pres_w).sum(-1) + self.pres_b


class PooledDescriptor(nn.Module):
    """Ablation A2a: global average pooling of P3/P4/P5, mapped to one descriptor per class."""

    def __init__(self, channels: tuple, n_classes: int, d: int = 64):
        super().__init__()
        self.d, self.k = int(d), int(n_classes)
        self.proj = nn.ModuleList(nn.Conv2d(c, self.d, 1) for c in channels)
        self.out = nn.Linear(len(channels) * self.d, self.k * self.d)
        self.norm = nn.LayerNorm(self.d)
        self.pres_w = nn.Parameter(torch.zeros(n_classes, self.d))
        self.pres_b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, feats: list) -> torch.Tensor:
        g = torch.cat([p(x).mean((2, 3)) for x, p in zip(feats, self.proj)], 1)
        return self.norm(self.out(g).view(-1, self.k, self.d))

    presence = ClassQueryPooling.presence


class VisibilityStateHead(nn.Module):
    """VSH: per pair and class, z = [h_ap, h_lat, |h_ap - h_lat|, h_ap * h_lat] -> MLP -> 4 state logits.

    shared: one MLP for all classes plus a learned class embedding added to z (default).
    per_class: a separate MLP per class (ablation A3a).
    """

    def __init__(self, n_classes: int, d: int = 64, hidden: int = 128, mode: str = "shared"):
        super().__init__()
        if mode not in ("shared", "per_class"):
            raise ValueError(f"vsh must be shared or per_class, got {mode!r}")
        self.mode, self.k, z = mode, int(n_classes), 4 * int(d)
        if mode == "shared":
            self.embed = nn.Parameter(torch.zeros(n_classes, z))
            self.mlp = nn.Sequential(nn.Linear(z, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        else:
            self.w1 = nn.Parameter(torch.randn(n_classes, z, hidden) / math.sqrt(z))
            self.b1 = nn.Parameter(torch.zeros(n_classes, hidden))
            self.w2 = nn.Parameter(torch.randn(n_classes, hidden, 4) / math.sqrt(hidden))
            self.b2 = nn.Parameter(torch.zeros(n_classes, 4))

    def forward(self, h_ap: torch.Tensor, h_lat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h_ap, h_lat, (h_ap - h_lat).abs(), h_ap * h_lat], -1)  # (P, K, 4d)
        if self.mode == "shared":
            return self.mlp(z + self.embed)
        x = F.silu(torch.einsum("pkz,kzh->pkh", z, self.w1) + self.b1)
        return torch.einsum("pkh,kho->pko", x, self.w2) + self.b2


def gates_from_states(q: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate per pair and class for the AP image and for the LAT image, from state probabilities
    q (P, K, 4) ordered 00, 10, 01, 11 (for the oracle, q is the one-hot true state).

    signed:   g_ap = q11 - q01, g_lat = q11 - q10  (boost when both views show the class,
              push down when only the partner shows it, zero when the partner lacks it)
    unsigned: g_ap = g_lat = q11                   (agreement boost only; ablation A1b)
    none:     g = 1                                (ungated; ablation A1a)
    oracle:   signed gate from the true states     (upper bound; ablation A1d, Gate 3)
    """
    if mode in ("signed", "oracle"):
        return q[..., 3] - q[..., 2], q[..., 3] - q[..., 1]
    if mode == "unsigned":
        return q[..., 3], q[..., 3]
    if mode == "none":
        one = torch.ones_like(q[..., 0])
        return one, one
    raise ValueError(f"gate must be one of {GATES}, got {mode!r}")


class GatedResidual(nn.Module):
    """GRA: bounded correction to the class logits of the gateable classes.

    m(x)  = sigmoid(<U_l f(x), V h_partner,c> / sqrt(r))         local match, in (0, 1)
    delta = cap * tanh(theta_c) * gate_c * m(x)                  |delta| < cap; theta_c = 0 at start
    """

    def __init__(self, channels: tuple, n_classes: int, d: int = 64, rank: int = 16, cap: float = 2.0):
        super().__init__()
        self.r, self.cap = int(rank), float(cap)
        self.u = nn.ModuleList(nn.Conv2d(c, self.r, 1, bias=False) for c in channels)
        self.v = nn.Linear(d, self.r, bias=False)
        self.theta = nn.Parameter(torch.zeros(n_classes))

    def match(self, feats: list, h_partner: torch.Tensor) -> torch.Tensor:
        """(N, K, A) local match of every anchor with the partner's class descriptors."""
        vh = self.v(h_partner)  # (N, K, r)
        out = [torch.einsum("nrhw,nkr->nkhw", u(x), vh).flatten(2) for x, u in zip(feats, self.u)]
        return torch.sigmoid(torch.cat(out, -1) / math.sqrt(self.r))

    def forward(self, feats: list, h_partner: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """(N, K, A) correction; gate (N, K) is already zero for images without a partner."""
        scale = self.cap * torch.tanh(self.theta)  # (K,)
        return scale[None, :, None] * gate[..., None] * self.match(feats, h_partner)


class VGRA(nn.Module):
    """CQP + VSH + GRA on top of the Detect head outputs. Boxes are not touched."""

    def __init__(self, channels: tuple, nc: int, gate_idx: list, d: int = 64, descriptor: str = "cqp",
                 vsh: str = "shared", gate: str = "signed", gate_grad: str = "detached", rank: int = 16,
                 cap: float = 2.0):
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        if gate_grad not in ("detached", "attached"):
            raise ValueError(f"gate_grad must be detached or attached, got {gate_grad!r}")
        gate_idx = [int(i) for i in gate_idx]
        if not gate_idx or len(set(gate_idx)) != len(gate_idx) or min(gate_idx) < 0 or max(gate_idx) >= nc:
            raise ValueError(f"bad gateable class indices {gate_idx} for nc={nc}")
        self.nc, self.gate_mode, self.detach_gate = int(nc), gate, gate_grad == "detached"
        self.register_buffer("gate_idx", torch.tensor(gate_idx, dtype=torch.long), persistent=False)
        k = len(gate_idx)
        if descriptor == "cqp":
            self.desc = ClassQueryPooling(channels, k, d)
        elif descriptor == "pooled":
            self.desc = PooledDescriptor(channels, k, d)
        else:
            raise NotImplementedError(f"descriptor {descriptor!r} (partner class scores, ablation A2c) is not built yet")
        self.vsh = VisibilityStateHead(k, d, mode=vsh)
        self.gra = GatedResidual(channels, k, d, rank, cap)

    def forward(self, feats: list, scores: torch.Tensor, partner: torch.Tensor, view: torch.Tensor,
                true_states: torch.Tensor | None = None,
                use_partner: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Corrected scores (same shape as scores) and an aux dict for the losses and logging.

        aux: h (N, K, d), pres_logits (N, K), pairs (ap, lat), state_logits (P, K, 4),
             gate (N, K), delta (N, K, A).
        """
        n = scores.shape[0]
        if partner.shape != (n,) or view.shape != (n,):
            raise ValueError("partner and view must have one entry per image")
        h = self.desc(feats)
        aux = {"h": h, "pres_logits": self.desc.presence(h)}
        ap, lat = pair_rows(partner, view)
        aux["pairs"] = (ap, lat)
        aux["state_logits"] = self.vsh(h[ap], h[lat])
        k = len(self.gate_idx)

        if self.gate_mode == "oracle":
            if true_states is None:
                raise ValueError("gate 'oracle' needs true_states (P, K)")
            q = F.one_hot(true_states.long(), 4).to(scores.dtype)
        else:
            q = aux["state_logits"].softmax(-1)
            if self.detach_gate:
                q = q.detach()
        g_ap, g_lat = gates_from_states(q, self.gate_mode)
        gate = scores.new_zeros(n, k)  # images without a partner keep gate 0
        gate = gate.index_put((ap,), g_ap.to(gate.dtype)).index_put((lat,), g_lat.to(gate.dtype))
        if use_partner is not None:
            gate = gate * use_partner.to(gate.dtype)[:, None]
        aux["gate"] = gate

        safe = partner.clamp(min=0)  # unpaired rows point at themselves; their gate is 0
        delta = self.gra(feats, h[safe], gate)
        aux["delta"] = delta
        full = scores.new_zeros(scores.shape).index_add(1, self.gate_idx, delta.to(scores.dtype))
        return scores + full, aux


def count_parameters(module: nn.Module) -> dict:
    parts = {name: sum(p.numel() for p in child.parameters()) for name, child in module.named_children()}
    parts["total"] = sum(p.numel() for p in module.parameters())
    return parts


def gate_indices(cfg: dict) -> list:
    """Indices of the gateable classes (config data.gateable) in data.classes."""
    classes, gateable = list(cfg["data"]["classes"]), list(cfg["data"]["gateable"])
    unknown = [c for c in gateable if c not in classes]
    if unknown:
        raise ValueError(f"gateable classes not in data.classes: {unknown}")
    return [classes.index(c) for c in gateable]


def build_vgra(cfg: dict, channels: tuple, gate: str | None = None) -> VGRA:
    """VGRA from the config (model.* keys). gate overrides model.gate (e.g. 'oracle')."""
    m = cfg["model"]
    return VGRA(tuple(int(c) for c in channels), len(cfg["data"]["classes"]), gate_indices(cfg),
                d=int(m["d_desc"]), descriptor=m["descriptor"], vsh=m["vsh"], gate=gate or m["gate"],
                gate_grad=m["gate_grad"], rank=int(m["rank"]), cap=float(m["cap"]))
