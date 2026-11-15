"""STEP-03 tests: SCC blocks and the model swap. Skipped where torch or Ultralytics is not installed."""

import copy
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.nn.modules.block import C3k2  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from vgra.model import count_parameters, make_trainer  # noqa: E402
from vgra.scc import C3K2_SCC, SCC, SCCAdapter, add_scc, scc_strengths  # noqa: E402


def first_output(y):
    return y[0] if isinstance(y, (list, tuple)) else y


@pytest.fixture(scope="module")
def stock():
    torch.manual_seed(0)
    model = DetectionModel("yolo11s.yaml", nc=9, verbose=False)
    model.train()
    with torch.no_grad():  # move the BatchNorm statistics away from their defaults
        model(torch.rand(2, 3, 128, 128))
    return model.eval()


def test_scc_keeps_shape():
    x = torch.rand(2, 32, 40, 24)
    assert SCC(32)(x).shape == x.shape


def test_adapter_is_identity_at_start():
    x = torch.rand(2, 32, 16, 16)
    adapter = SCCAdapter(32).eval()
    assert torch.equal(adapter(x), x) and adapter.strength() == 0.0
    with torch.no_grad():
        adapter.alpha.fill_(0.5)
    assert not torch.equal(adapter(x), x)


def test_add_scc_swaps_only_layers_2_and_4(stock):
    ours = add_scc(copy.deepcopy(stock))
    swapped = [i for i, m in enumerate(ours.model) if isinstance(m, C3K2_SCC)]
    assert swapped == [2, 4]
    assert type(stock.model[2]) is C3k2  # the original model is untouched
    assert ours.model[2].i == 2 and ours.model[4].f == stock.model[4].f
    assert count_parameters(ours) - count_parameters(stock) == 138_818
    add_scc(ours)  # calling it twice changes nothing
    assert [i for i, m in enumerate(ours.model) if isinstance(m, C3K2_SCC)] == [2, 4]


def test_outputs_equal_stock_model(stock):
    ours = add_scc(copy.deepcopy(stock)).eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        assert torch.equal(first_output(stock(x)), first_output(ours(x)))


def test_model_pickles_and_copies(stock):
    ours = add_scc(copy.deepcopy(stock)).eval()
    x = torch.rand(1, 3, 128, 128)
    buf = io.BytesIO()
    torch.save({"model": ours}, buf)
    buf.seek(0)
    again = torch.load(buf, weights_only=False)["model"]
    with torch.no_grad():
        ref = first_output(ours(x))
        assert torch.equal(first_output(again(x)), ref)
        assert torch.equal(first_output(copy.deepcopy(ours)(x)), ref)


def test_strengths_and_gradient(stock):
    ours = add_scc(copy.deepcopy(stock)).train()
    assert scc_strengths(ours) == {"model.2.scc_adapters.0": 0.0, "model.4.scc_adapters.0": 0.0}
    out = ours.model[2](torch.rand(2, ours.model[2].cv1.conv.in_channels, 32, 32))
    out.sum().backward()
    assert ours.model[2].scc_adapters[0].alpha.grad is not None  # alpha can move away from 0
    assert scc_strengths(stock) == {}


def test_from_native_rejects_other_blocks(stock):
    with pytest.raises(TypeError):
        C3K2_SCC.from_native(stock.model[0])


def test_make_trainer_flags():
    assert make_trainer(True).use_scc is True
    assert make_trainer(False).use_scc is False
