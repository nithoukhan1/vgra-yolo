"""STEP-07: check VGRADetect on trained B1 weights and real pairs (Kaggle GPU; also runs on CPU).

Run on Kaggle (GPU, dataset + index + an output that holds runs/B1_s0 attached):
    python scripts/07_check_head.py --config configs/base.yaml --weights auto
Run locally without weights (random YOLO11s + SCC, random images):
    python scripts/07_check_head.py --weights none --imgsz 320 --pairs 2 --device cpu

Checks
1. With fresh VGRA modules, VGRADetect gives exactly B1's outputs (paired context and no context),
   in eval mode and in training mode (detector BatchNorm frozen).
2. With the correction switched on, boxes never change and only gateable class scores change.
3. Checkpoint round trip through Ultralytics' own loader; EMA copy works.
4. One stage-2 step (detector frozen, AMP on GPU): stock detection loss on the corrected scores
   + state loss + presence loss; gradients reach VGRA only. Reports peak GPU memory and time.
Ends with 'HEAD CHECK: PASS' or 'HEAD CHECK: FAIL'.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--weights", default="auto",
                    help="path to a B1 .pt, 'auto' (find runs/<run>/weights under --search), or 'none' (random)")
    ap.add_argument("--run", default="B1_s0", help="run folder looked for with --weights auto")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--batches", type=int, default=3, help="validation pair batches to compare")
    ap.add_argument("--pairs", type=int, default=None, help="pairs per batch; default train.pair_batch")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help="default cuda:0 if available, else cpu")
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def find_weights(arg: str, run: str, search: str) -> Path | None:
    from vgra.runs import find_previous_run

    if arg == "none":
        return None
    if arg != "auto":
        return Path(arg)
    folder = find_previous_run(run, search)
    if folder is None:
        raise FileNotFoundError(f"no run folder {run!r} under {search}; attach the training output or pass --weights")
    for name in ("best.pt", "last.pt", "resume.pt"):
        if (folder / "weights" / name).is_file():
            return folder / "weights" / name
    raise FileNotFoundError(f"no weights in {folder / 'weights'}")


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import ModelEMA

    from vgra import head as vh
    from vgra import modules as vm
    from vgra import utils
    from vgra.scc import add_scc

    cfg = utils.load_config(args.config)
    data = cfg["data"]
    if args.root:
        data["root"] = args.root
    classes = list(data["classes"])
    gate_idx = vm.gate_indices(cfg)
    imgsz = int(args.imgsz or cfg["train"]["imgsz"])
    b = int(args.pairs or cfg["train"].get("pair_batch", 8))
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    weights = find_weights(args.weights, args.run, args.search)
    print(f"repo commit: {utils.git_commit()} | image size {imgsz} | {b} pairs | device {dev}")
    print(f"weights: {weights or 'none (random YOLO11s + SCC)'}")

    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(bool(ok))
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))

    # B1 and the VGRA model built from it
    if weights is None:
        torch.manual_seed(0)
        b1 = DetectionModel("yolo11s.yaml", nc=len(classes), verbose=False)
        add_scc(b1)
    else:
        b1, _ = load_checkpoint(str(weights), device="cpu")
    b1 = b1.float().to(dev).eval()
    for p in b1.parameters():
        p.requires_grad_(False)
    check("B1 has the right number of classes", b1.model[-1].nc == len(classes), f"nc {b1.model[-1].nc}")
    model = vh.add_vgra(copy.deepcopy(b1), cfg).eval()
    head = vh.vgra_head(model)
    n_v = sum(p.numel() for p in head.vgra.parameters())
    n_b = sum(p.numel() for p in b1.parameters())
    print(f"parameters: B1 {n_b:,} | VGRA {n_v:,} (+{100 * n_v / n_b:.2f}%)")

    # batches: real validation pairs, or random images without weights/data
    batches = []
    if weights is not None or args.index or args.root:
        from vgra import data as vd
        from vgra import index as ix
        from vgra import pairs as pr

        index = pr.read_index(ix.locate_index(data, args.index))
        pairs = pr.build_pairs(index, classes)
        out = Path(args.out or cfg["out_dir"]) / "head_check"
        valid = vd.PairDataset(pairs, index, "valid", classes, imgsz, augment=False, list_dir=out / "lists")
        loader = vd.build_pair_loader(valid, b, shuffle=False, workers=2)
        for k, batch in enumerate(loader):
            if k >= args.batches:
                break
            batches.append(batch)
    else:
        for k in range(args.batches):
            g = torch.Generator().manual_seed(k)
            batches.append({"img": (torch.rand(2 * b, 3, imgsz, imgsz, generator=g) * 255).to(torch.uint8),
                            "n_pairs": b, "cls": torch.zeros(0, 1), "bboxes": torch.zeros(0, 4),
                            "batch_idx": torch.zeros(0)})

    def prep(batch):
        return batch["img"].to(dev).float() / 255

    # 1. identical to B1 at the start
    worst = 0.0
    t_b1 = t_v = 0.0
    with torch.no_grad():
        for batch in batches:
            x, n = prep(batch), batch["n_pairs"]
            t0 = time.time()
            y0, p0 = b1(x)
            t1 = time.time()
            with vh.vgra_context(model, *vm.paired_layout(n, dev)):
                y1, p1 = model(x)
            t2 = time.time()
            y2, _ = model(x)
            t_b1, t_v = t_b1 + t1 - t0, t_v + t2 - t1
            worst = max(worst, float((y0 - y1).abs().max()), float((y0 - y2).abs().max()),
                        float((p0["scores"] - p1["scores"]).abs().max()))
    check(f"eval: identical to B1 on {len(batches)} batches, paired and single-view", worst == 0.0,
          f"max difference {worst:.2e}")

    b1t, mt = copy.deepcopy(b1).train(), copy.deepcopy(model).train()
    vh.detector_eval(b1t)
    vh.detector_eval(mt)
    with torch.no_grad():
        x, n = prep(batches[0]), batches[0]["n_pairs"]
        q0 = b1t(x)
        with vh.vgra_context(mt, *vm.paired_layout(n, dev)):
            q1 = mt(x)
    d = max(float((q0["scores"] - q1["scores"]).abs().max()), float((q0["boxes"] - q1["boxes"]).abs().max()))
    check("training mode: identical to B1", d == 0.0, f"max difference {d:.2e}")
    del b1t, mt

    # 2. correction on: boxes fixed, only gateable scores move
    with torch.no_grad():
        head.vgra.gra.theta.fill_(50.0)
        x, n = prep(batches[0]), batches[0]["n_pairs"]
        _, p0 = b1(x)
        oracle = torch.full((n, len(gate_idx)), 3, device=dev)  # what a perfectly confident 'both views' gate gives
        head.vgra.gate_mode = "oracle"
        with vh.vgra_context(model, *vm.paired_layout(n, dev), true_states=oracle):
            _, p1 = model(x)
        head.vgra.gate_mode = cfg["model"]["gate"]
        head.vgra.gra.theta.fill_(0.0)
    diff = (p1["scores"] - p0["scores"]).abs()
    other = [c for c in range(len(classes)) if c not in gate_idx]
    check("correction on: boxes unchanged", torch.equal(p0["boxes"], p1["boxes"]))
    check("correction on: only gateable classes change, within the cap",
          float(diff[:, other].max()) == 0.0 and float(diff.max()) <= float(cfg["model"]["cap"]),
          f"max change {float(diff.max()):.3f}")

    # 3. checkpoint round trip with Ultralytics' loader, and an EMA copy
    with torch.no_grad():
        head.vgra.gra.theta.fill_(0.5)
        x, n = prep(batches[0]), batches[0]["n_pairs"]
        with vh.vgra_context(model, *vm.paired_layout(n, dev)):
            y_ref, _ = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vgra_check.pt"
        torch.save({"model": copy.deepcopy(model).half(), "train_args": {}}, path)
        loaded, _ = load_checkpoint(str(path), device=dev)
    loaded = loaded.float().eval()
    with torch.no_grad(), vh.vgra_context(loaded, *vm.paired_layout(n, dev)):
        y_new, _ = loaded(x)
    rel = float((y_ref - y_new).abs().max()) / max(float(y_ref.abs().max()), 1e-9)
    check("checkpoint saved and reloaded with Ultralytics' loader", isinstance(loaded.model[-1], vh.VGRADetect)
          and rel < 1e-2, f"relative difference after half precision {rel:.1e}")
    ema = ModelEMA(model)
    check("EMA copy keeps VGRADetect, without the batch context", isinstance(ema.ema.model[-1], vh.VGRADetect)
          and ema.ema.model[-1]._ctx is None)
    with torch.no_grad():
        head.vgra.gra.theta.fill_(0.0)

    # 4. one stage-2 training step on real labels
    model.args = get_cfg(DEFAULT_CFG)
    trainable = vh.freeze_detector(model)
    model.train()
    vh.detector_eval(model)
    opt = torch.optim.AdamW(vh.vgra_parameters(model), lr=1e-3)
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batches[0].items()}
    batch["img"] = prep(batches[0])
    n = batch["n_pairs"]
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        with vh.vgra_context(model, *vm.paired_layout(n, dev)):
            preds = model(batch["img"])
        aux = head.last_aux
        det_loss, _ = model.loss(batch, preds)
        has = torch.zeros(2 * n, len(classes), dtype=torch.long, device=dev)
        if batch["cls"].numel():
            has[batch["batch_idx"].long(), batch["cls"].view(-1).long()] = 1
        ap, lat = aux["pairs"]
        l_vis = F.cross_entropy(aux["state_logits"].float().reshape(-1, 4), vm.state_targets(has, ap, lat, gate_idx).reshape(-1))
        l_pres = F.binary_cross_entropy_with_logits(aux["pres_logits"].float(), has[:, gate_idx].float())
        loss = det_loss.sum() + float(cfg["loss"]["w_vis"]) * l_vis + float(cfg["loss"]["w_pres"]) * l_pres
    loss.backward()
    opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_step = time.time() - t0
    grads = {name: any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in part.parameters())
             for name, part in (("CQP", head.vgra.desc), ("VSH", head.vgra.vsh), ("GRA", head.vgra.gra))}
    frozen = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    check("stage-2 step: stock detection loss runs on the corrected scores", bool(torch.isfinite(loss.detach())),
          f"loss {float(loss.detach()):.3f}, L_vis {float(l_vis.detach()):.3f}, L_pres {float(l_pres.detach()):.3f}")
    check("stage-2 step: gradients reach CQP, VSH, GRA; detector frozen", all(grads.values()) and frozen,
          f"{grads}, trainable {trainable:,}")

    mem = f", peak GPU memory {torch.cuda.max_memory_allocated(dev) / 1e9:.2f} GB" if dev.type == "cuda" else ""
    print(f"[info] inference per batch of {2 * b} images: B1 {1000 * t_b1 / len(batches):.0f} ms, "
          f"VGRA model {1000 * t_v / len(batches):.0f} ms")
    print(f"[info] one stage-2 step: {t_step:.2f} s{mem}")
    ok = all(results)
    print("HEAD CHECK: PASS" if ok else "HEAD CHECK: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
