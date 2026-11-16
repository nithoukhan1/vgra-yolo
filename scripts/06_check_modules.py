"""STEP-06: check CQP, VSH and GRA on real YOLO11s + SCC features (CPU is enough).

Run:
    python scripts/06_check_modules.py --config configs/base.yaml
    python scripts/06_check_modules.py --imgsz 640 --pairs 2      # quicker

Builds YOLO11s + SCC from the model YAML (random weights, no download), runs one batch of
pairs in the loader layout [AP..., LAT...], takes the Detect head inputs and class scores,
and checks that VGRA
1. keeps the scores unchanged at initialisation,
2. keeps them unchanged for images without a partner,
3. changes only the gateable class channels, never by more than the cap,
4. trains: a backward pass reaches CQP, VSH and GRA while the detector stays frozen.
Reports parameters and time. Ends with 'MODULE CHECK: PASS' or 'MODULE CHECK: FAIL'.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--imgsz", type=int, default=None, help="default train.imgsz")
    ap.add_argument("--pairs", type=int, default=None, help="pairs per batch; default train.pair_batch")
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.tasks import DetectionModel

    from vgra import modules as vm
    from vgra import utils
    from vgra.scc import add_scc

    cfg = utils.load_config(args.config)
    imgsz = int(args.imgsz or cfg["train"]["imgsz"])
    b = int(args.pairs or cfg["train"].get("pair_batch", 8))
    classes = list(cfg["data"]["classes"])
    gate_idx = vm.gate_indices(cfg)
    dev = torch.device(args.device)
    print(f"repo commit: {utils.git_commit()} | image size {imgsz} | {b} pairs ({2 * b} images) | device {dev}")
    print(f"gateable: {', '.join(classes[i] for i in gate_idx)}")

    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(bool(ok))
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))

    torch.manual_seed(0)
    det = DetectionModel("yolo11s.yaml", nc=len(classes), verbose=False)
    if cfg["model"].get("scc", True):
        add_scc(det)
    det = det.to(dev).eval()
    for p in det.parameters():
        p.requires_grad_(False)
    n_det = sum(p.numel() for p in det.parameters())

    x = torch.rand(2 * b, 3, imgsz, imgsz, device=dev)
    t0 = time.time()
    with torch.no_grad():
        _, preds = det(x)
    t_det = time.time() - t0
    feats, scores = preds["feats"], preds["scores"]
    channels = tuple(f.shape[1] for f in feats)
    anchors = sum(f.shape[2] * f.shape[3] for f in feats)
    print(f"Detect inputs: {[tuple(f.shape[1:]) for f in feats]} | scores {tuple(scores.shape)}")
    check("scores layout is (N, nc, anchors of all levels)", tuple(scores.shape) == (2 * b, len(classes), anchors))

    vgra = vm.build_vgra(cfg, channels).to(dev)
    parts = vm.count_parameters(vgra)
    print(f"VGRA parameters: {parts['total']:,} ({100 * parts['total'] / n_det:.2f}% of the detector, "
          f"{n_det:,}) | CQP {parts['desc']:,}, VSH {parts['vsh']:,}, GRA {parts['gra']:,}")
    partner, view = vm.paired_layout(b, dev)

    with torch.no_grad():
        out, aux = vgra(feats, scores, partner, view)
    check("no change at initialisation", torch.equal(out, scores))

    with torch.no_grad():
        vgra.gra.theta.fill_(1.0)
        out, _ = vgra(feats, scores, torch.full_like(partner, -1), view)
    check("no change without a partner", torch.equal(out, scores))

    with torch.no_grad():
        vgra.gra.theta.fill_(50.0)  # largest possible correction
        out, aux = vgra(feats, scores, partner, view)
    diff = (out - scores).abs()
    other = [c for c in range(len(classes)) if c not in gate_idx]
    check("only gateable classes change", float(diff[:, other].max()) == 0.0)
    cap = float(cfg["model"]["cap"])
    check(f"correction stays below the cap ({cap})", float(diff.max()) <= cap, f"max {float(diff.max()):.3f}")

    # one training step of stage 2: detector frozen, VGRA learns
    with torch.no_grad():
        vgra.gra.theta.fill_(0.0)
    vgra.train()
    opt = torch.optim.SGD(vgra.parameters(), lr=0.01)
    t0 = time.time()
    out, aux = vgra(feats, scores, partner, view)
    ap, lat = aux["pairs"]
    has = (torch.rand(2 * b, len(classes), device=dev) < 0.3).long()
    states = vm.state_targets(has, ap, lat, gate_idx)
    loss = (F.cross_entropy(aux["state_logits"].reshape(-1, 4), states.reshape(-1))
            + F.binary_cross_entropy_with_logits(aux["pres_logits"], has[:, gate_idx].float())
            + out.sigmoid().mean())
    loss.backward()
    t_vgra = time.time() - t0
    reached = {name: any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
               for name, part in (("CQP", vgra.desc), ("VSH", vgra.vsh), ("GRA", vgra.gra))}
    check("gradients reach CQP, VSH and GRA", all(reached.values()), str(reached))
    check("detector stays frozen", all(p.grad is None for p in det.parameters()))
    opt.step()
    check("theta leaves 0 after one step (correction switches on)", float(vgra.gra.theta.detach().abs().max()) > 0)

    print(f"[info] time: detector forward {t_det:.1f} s, VGRA forward + backward {t_vgra:.1f} s ({dev})")
    ok = all(results)
    print("MODULE CHECK: PASS" if ok else "MODULE CHECK: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
