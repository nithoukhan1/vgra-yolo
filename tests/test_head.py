"""STEP-07 tests: VGRADetect inside a real YOLO11s + SCC model (random weights, CPU, small images)."""

import copy
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import ModelEMA  # noqa: E402

from vgra import head as vh  # noqa: E402
from vgra import modules as vm  # noqa: E402
from vgra.scc import add_scc  # noqa: E402

CLASSES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
           "periostealreaction", "pronatorsign", "softtissue", "text"]
CFG = {
    "data": {"classes": CLASSES, "gateable": ["fracture", "periostealreaction", "pronatorsign",
                                              "softtissue", "metal", "boneanomaly"]},
    "model": {"d_desc": 64, "descriptor": "cqp", "vsh": "shared", "gate": "signed",
              "gate_grad": "detached", "rank": 16, "cap": 2.0},
}
GATE = vm.gate_indices(CFG)
IMG = 128


@pytest.fixture(scope="module")
def b1():
    torch.manual_seed(0)
    m = DetectionModel("yolo11s.yaml", nc=len(CLASSES), verbose=False)
    add_scc(m)
    return m.eval()


@pytest.fixture()
def pair(b1):
    v = copy.deepcopy(b1)
    vh.add_vgra(v, CFG)
    return b1, v.eval()


def images(n, seed=1):
    return torch.rand(n, 3, IMG, IMG, generator=torch.Generator().manual_seed(seed))


def run(model, x, n_pairs=None, **kw):
    if n_pairs is None:
        return model(x)
    with vh.vgra_context(model, *vm.paired_layout(n_pairs), **kw):
        return model(x)


def switch_on(model, value=1.0):
    with torch.no_grad():
        vh.vgra_head(model).vgra.gra.theta.fill_(value)


def test_head_is_a_detect_subclass_with_same_layers(pair):
    b1, v = pair
    head = v.model[-1]
    assert isinstance(head, vh.VGRADetect) and isinstance(head, Detect)
    assert head.type == "vgra.head.VGRADetect"
    old = dict(b1.model[-1].named_parameters())
    for name, p in head.named_parameters():
        if not name.startswith("vgra."):
            assert torch.equal(p, old[name]), name


def test_identical_to_b1_at_start_eval(pair):
    b1, v = pair
    x = images(4)
    with torch.no_grad():
        y0, p0 = run(b1, x)
        y1, p1 = run(v, x, n_pairs=2)
        y2, _ = run(v, x)  # no context: plain Detect
    assert torch.equal(y0, y1) and torch.equal(y0, y2)
    assert torch.equal(p0["scores"], p1["scores"]) and torch.equal(p0["boxes"], p1["boxes"])


def test_identical_to_b1_at_start_train_mode(pair):
    b1, v = pair
    x = images(4)
    b1t, vt = copy.deepcopy(b1).train(), copy.deepcopy(v).train()
    vh.detector_eval(b1t)
    vh.detector_eval(vt)
    with torch.no_grad():
        p0 = run(b1t, x)
        p1 = run(vt, x, n_pairs=2)
    assert torch.equal(p0["scores"], p1["scores"]) and torch.equal(p0["boxes"], p1["boxes"])


def test_correction_changes_only_gateable_scores_and_never_boxes(pair):
    b1, v = pair
    switch_on(v, 50.0)
    x = images(4)
    with torch.no_grad():
        y0, p0 = run(b1, x)
        y1, p1 = run(v, x, n_pairs=2, true_states=torch.full((2, len(GATE)), 3))
    assert torch.equal(p0["boxes"], p1["boxes"])
    diff = (p1["scores"] - p0["scores"]).abs()
    other = [c for c in range(len(CLASSES)) if c not in GATE]
    assert float(diff[:, other].max()) == 0.0
    head = vh.vgra_head(v)
    assert head.last_aux is not None and head.last_aux["delta"].shape[1] == len(GATE)


def test_oracle_gate_through_the_head(b1):
    v = copy.deepcopy(b1)
    vh.add_vgra(v, CFG, gate="oracle")
    v.eval()
    switch_on(v, 1.0)
    x = images(2)
    states = torch.zeros(1, len(GATE), dtype=torch.long)
    states[0, GATE.index(3)] = 3  # fracture in both views
    with torch.no_grad():
        _, p0 = b1(x)
        _, p1 = run(v, x, n_pairs=1, true_states=states)
    diff = (p1["scores"] - p0["scores"]).abs()
    assert float(diff[:, 3].min()) > 0  # fracture boosted in both images
    assert float(diff[:, [c for c in GATE if c != 3]].max()) == 0.0  # state 00: gate 0


def test_context_is_cleared_and_checked(pair):
    _, v = pair
    switch_on(v)
    x = images(4)
    with torch.no_grad():
        run(v, x, n_pairs=2)
        assert vh.vgra_head(v)._ctx is None  # cleared after the block
        with pytest.raises(ValueError):
            run(v, x, n_pairs=3)  # context for 6 images, batch has 4


def test_switch_off(pair):
    b1, v = pair
    switch_on(v)
    vh.vgra_head(v).vgra_on = False
    x = images(2)
    with torch.no_grad():
        y0, _ = b1(x)
        y1, _ = run(v, x, n_pairs=1)
    assert torch.equal(y0, y1)


def test_deepcopy_ema_and_checkpoint_roundtrip(pair):
    _, v = pair
    switch_on(v, 0.7)
    x = images(2)
    with torch.no_grad():
        y_ref, _ = run(v, x, n_pairs=1)
        with vh.vgra_context(v, *vm.paired_layout(1)):
            c = copy.deepcopy(v)  # copied while a context is set
    assert vh.vgra_head(c)._ctx is None and vh.vgra_head(c).last_aux is None
    ema = ModelEMA(v)
    assert isinstance(ema.ema.model[-1], vh.VGRADetect)
    buf = io.BytesIO()
    torch.save({"model": copy.deepcopy(v).half()}, buf)  # what Ultralytics saves
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)["model"].float().eval()
    with torch.no_grad():
        y_new, _ = run(loaded, x, n_pairs=1)
    assert isinstance(loaded.model[-1], vh.VGRADetect)
    assert torch.allclose(y_ref, y_new, atol=5e-2)  # weights went through half precision


def test_fuse_keeps_vgra_and_outputs(pair):
    _, v = pair
    switch_on(v, 0.5)
    x = images(2)
    with torch.no_grad():
        y0, _ = run(v, x, n_pairs=1)
        f = copy.deepcopy(v).fuse(verbose=False)
        y1, _ = run(f, x, n_pairs=1)
    assert isinstance(f.model[-1], vh.VGRADetect)
    assert torch.allclose(y0, y1, atol=1e-4)


def test_stage2_freeze(pair):
    _, v = pair
    n = vh.freeze_detector(v)
    assert n == sum(p.numel() for p in vh.vgra_head(v).vgra.parameters())
    v.train()
    vh.detector_eval(v)
    switch_on(v, 0.0)
    x = images(4)
    preds = run(v, x, n_pairs=2)
    aux = vh.vgra_head(v).last_aux
    loss = preds["scores"].sigmoid().mean() + aux["state_logits"].logsumexp(-1).mean() + aux["pres_logits"].mean()
    loss.backward()
    assert all(p.grad is None for p in v.parameters() if not p.requires_grad)
    assert any(p.grad is not None for p in vh.vgra_parameters(v))
    bns = [m for m in v.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)


def test_add_vgra_rejects_other_heads(pair):
    _, v = pair
    with pytest.raises(TypeError):
        vh.add_vgra(v, CFG)  # already a VGRADetect
