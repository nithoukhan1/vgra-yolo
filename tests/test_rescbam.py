"""V8R tests: ResCBAM block (Ju et al.), YOLOv8s + ResCBAM layout, pretrained-weight transfer.
Skipped where torch or Ultralytics is not installed."""

import copy
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from vgra import rescbam  # noqa: E402
from vgra.model import count_parameters, make_trainer  # noqa: E402
from vgra.rescbam import ResBlockCBAM  # noqa: E402


def first_output(y):
    return y[0] if isinstance(y, (list, tuple)) else y


def block_params(c: int) -> int:
    """Parameters of one block: three bias-free convs (c*c, 9*c*c, c*c), three BatchNorms (2c each),
    CBAM channel fc (c*c + c) and spatial 7x7 conv (2*49)."""
    return 11 * c * c + 6 * c + c * c + c + 98


@pytest.fixture(scope="module")
def stock():
    torch.manual_seed(0)
    return DetectionModel("yolov8s.yaml", nc=9, verbose=False).eval()


@pytest.fixture(scope="module")
def ours(stock):
    torch.manual_seed(1)
    model = rescbam.build(9)
    report = rescbam.transfer_stock_weights(model, stock)
    return model.eval(), report


def test_block_computation_matches_ju_et_al():
    torch.manual_seed(0)
    block = ResBlockCBAM(32).eval()
    x = torch.randn(2, 32, 12, 10)
    with torch.no_grad():
        out = block.bottleneck(x)  # 1x1 -> 3x3 -> 1x1 with BatchNorm and LeakyReLU(0.1)
        ca = out * torch.sigmoid(block.cbam.channel_attention.fc(out.mean((2, 3), keepdim=True)))
        sa_in = torch.cat([ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], 1)
        cbam = ca * torch.sigmoid(block.cbam.spatial_attention.cv1(sa_in))
        expected = torch.relu(cbam + x)  # identity shortcut, then ReLU
        assert torch.allclose(block(x.clone()), expected, atol=1e-6)
    leaky = [m for m in block.bottleneck if isinstance(m, torch.nn.LeakyReLU)]
    assert len(leaky) == 2 and all(m.negative_slope == 0.1 for m in leaky)
    assert count_parameters(block) == block_params(32)


def test_block_keeps_channels():
    with pytest.raises(ValueError):
        ResBlockCBAM(32, 64)
    assert ResBlockCBAM(16)(torch.rand(1, 16, 8, 8)).shape == (1, 16, 8, 8)


def test_parameter_count_matches_ju_et_al_table():
    # Their table: YOLOv8 (l scale) 43.61M -> YOLOv8 + ResCBAM 53.87M, i.e. +10.26M.
    # At the l scale the four blocks have 512, 256, 512, 512 channels: 10.24M, which agrees with the
    # table within 0.02M (their two numbers are rounded separately and come from their own fork).
    added_l = sum(block_params(c) for c in (512, 256, 512, 512))
    assert round(added_l / 1e6, 2) == 10.24 and abs(added_l / 1e6 - 10.26) < 0.03


def test_layout(ours, stock):
    model, _ = ours
    assert rescbam.block_indices(model) == [13, 17, 21, 25]
    assert model.model[26].f == [17, 21, 25]  # Detect reads the block outputs
    assert model.model[19].f == [-1, 12]  # P4 skip reads the C2f before the block (Ju et al.'s YAML)
    assert model.model[23].f == [-1, 9]
    assert [model.model[i].bottleneck[0].in_channels for i in (13, 17, 21, 25)] == [256, 128, 256, 512]
    added = count_parameters(model) - count_parameters(stock)
    assert added == sum(block_params(c) for c in (256, 128, 256, 512))
    assert count_parameters(model) == 16_062_739


def test_transfer_copies_every_stock_tensor(ours, stock):
    model, report = ours
    assert report["transferred"] == report["stock_tensors"] and not report["skipped"]
    assert not report["missing_outside_blocks"]
    mapping = rescbam.stock_layer_map(model)
    assert mapping[12] == 12 and mapping[13] == 14 and mapping[22] == 26
    sd_s, sd_o = stock.state_dict(), model.state_dict()
    for k, v in sd_s.items():
        parts = k.split(".")
        k2 = ".".join(["model", str(mapping[int(parts[1])]), *parts[2:]])
        assert torch.equal(sd_o[k2], v), k


def test_transfer_skips_only_class_layers_when_classes_differ():
    torch.manual_seed(0)
    coco = DetectionModel("yolov8s.yaml", nc=80, verbose=False)
    model = rescbam.build(9)
    r = rescbam.transfer_stock_weights(model, coco)
    assert r["skipped"] and all(k.startswith("model.22.cv3.") for k in r["skipped"])
    assert all(k.startswith("model.26.cv3.") for k in r["missing_outside_blocks"])


def test_backbone_matches_stock(ours, stock):
    model, _ = ours
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        a, b = x, x
        for i in range(10):  # backbone layers 0-9 are identical, so their outputs are too
            a, b = stock.model[i](a), model.model[i](b)
        assert torch.equal(a, b)


def test_model_pickles_and_copies(ours):
    model, _ = ours
    x = torch.rand(1, 3, 128, 128)
    buf = io.BytesIO()
    torch.save({"model": model}, buf)
    buf.seek(0)
    again = torch.load(buf, weights_only=False)["model"]
    with torch.no_grad():
        ref = first_output(model(x))
        assert torch.equal(first_output(again(x)), ref)
        assert torch.equal(first_output(copy.deepcopy(model)(x)), ref)


def test_trainer_builds_rescbam_and_rejects_mixing(stock):
    with pytest.raises(ValueError):
        make_trainer(True, rescbam=True)
    trainer_cls = make_trainer(False, rescbam=True)
    fake = trainer_cls.__new__(trainer_cls)
    fake.data = {"nc": 9, "channels": 3}
    built = fake.get_model(weights=stock, verbose=False)
    assert rescbam.block_indices(built) == [13, 17, 21, 25]
    resumed = fake.get_model(weights=built, verbose=False)  # a V8R checkpoint loads as a whole
    for k, v in built.state_dict().items():
        assert torch.equal(resumed.state_dict()[k], v), k


def test_v8r_config():
    from vgra.utils import load_config

    cfg = load_config("configs/V8R.yaml")
    assert cfg["exp_id"] == "V8R" and cfg["model"]["rescbam"] is True and cfg["model"]["scc"] is False
    assert cfg["model"]["weights"] == "yolov8s.pt"
    v8 = load_config("configs/V8.yaml")
    assert {k: v for k, v in cfg["train"].items()} == v8["train"]  # same recipe as V8
