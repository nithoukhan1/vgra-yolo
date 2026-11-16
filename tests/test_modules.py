"""STEP-06 tests: CQP, VSH, GRA on CPU with random features."""

import pytest

torch = pytest.importorskip("torch")

from vgra import modules as vm  # noqa: E402

NC = 9
GATE = [3, 5, 6, 7, 4, 0]  # fracture, periostealreaction, pronatorsign, softtissue, metal, boneanomaly
CH = (16, 32, 64)
SIZES = ((16, 16), (8, 8), (4, 4))
A = sum(h * w for h, w in SIZES)


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = [torch.randn(n, c, h, w, generator=g) for c, (h, w) in zip(CH, SIZES)]
    scores = torch.randn(n, NC, A, generator=g)
    return feats, scores


def model(**kw):
    torch.manual_seed(0)
    return vm.VGRA(CH, NC, GATE, **kw).eval()


def switch_on(m, value=1.0):
    with torch.no_grad():
        m.gra.theta.fill_(value)
    return m


def test_shapes():
    m, b = model(), 3
    feats, scores = inputs(2 * b)
    partner, view = vm.paired_layout(b)
    out, aux = m(feats, scores, partner, view)
    k = len(GATE)
    assert out.shape == scores.shape
    assert aux["h"].shape == (2 * b, k, 64) and aux["pres_logits"].shape == (2 * b, k)
    assert aux["state_logits"].shape == (b, k, 4) and aux["gate"].shape == (2 * b, k)
    assert aux["delta"].shape == (2 * b, k, A)
    assert torch.allclose(aux["state_logits"].softmax(-1).sum(-1), torch.ones(b, k))


def test_paired_layout():
    partner, view = vm.paired_layout(3)
    assert partner.tolist() == [3, 4, 5, 0, 1, 2] and view.tolist() == [0, 0, 0, 1, 1, 1]
    ap, lat = vm.pair_rows(partner, view)
    assert ap.tolist() == [0, 1, 2] and lat.tolist() == [3, 4, 5]


def test_zero_correction_at_start():
    m, b = model(), 4
    feats, scores = inputs(2 * b)
    out, aux = m(feats, scores, *vm.paired_layout(b))
    assert torch.equal(out, scores) and float(aux["delta"].detach().abs().max()) == 0.0


def test_zero_correction_without_partner():
    m = switch_on(model())
    feats, scores = inputs(4)
    partner = torch.full((4,), -1, dtype=torch.long)
    out, aux = m(feats, scores, partner, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(out, scores) and aux["state_logits"].shape[0] == 0


def test_partner_dropout_switches_correction_off_but_keeps_pair():
    m, b = switch_on(model(gate="none")), 2
    feats, scores = inputs(2 * b)
    use = torch.tensor([False, True, False, True])  # pair 0 dropped, pair 1 kept
    out, aux = m(feats, scores, *vm.paired_layout(b), use_partner=use)
    assert torch.equal(out[[0, 2]], scores[[0, 2]])
    assert not torch.equal(out[[1, 3]], scores[[1, 3]])
    assert aux["state_logits"].shape[0] == 2  # VSH still sees both pairs


def test_mixed_batch_paired_and_unpaired():
    m = switch_on(model(gate="none"))
    feats, scores = inputs(5)
    partner = torch.tensor([2, -1, 0, -1, -1])
    view = torch.tensor([0, 0, 1, 1, 0])
    out, aux = m(feats, scores, partner, view)
    assert torch.equal(out[[1, 3, 4]], scores[[1, 3, 4]])
    assert not torch.equal(out[[0, 2]], scores[[0, 2]])
    assert aux["pairs"][0].tolist() == [0] and aux["pairs"][1].tolist() == [2]


def test_only_gateable_channels_change_and_cap_holds():
    cap = 2.0
    m, b = switch_on(model(gate="none", cap=cap), 50.0), 3  # tanh(50) = 1: largest possible scale
    feats, scores = inputs(2 * b)
    out, _ = m(feats, scores, *vm.paired_layout(b))
    diff = out - scores
    other = [c for c in range(NC) if c not in GATE]
    assert float(diff[:, other].abs().max()) == 0.0
    assert float(diff[:, GATE].abs().max()) <= cap and float(diff[:, GATE].abs().min()) > 0


def test_gate_formulas():
    q = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])  # 00, 10, 01, 11
    g_ap, g_lat = vm.gates_from_states(q, "signed")
    assert torch.allclose(g_ap, torch.tensor([[0.4 - 0.3]])) and torch.allclose(g_lat, torch.tensor([[0.4 - 0.2]]))
    one_hot = torch.eye(4)[None]  # true states 00, 10, 01, 11 for one pair and four classes
    g_ap, g_lat = vm.gates_from_states(one_hot, "oracle")
    assert g_ap.tolist() == [[0, 0, -1, 1]]  # AP is pushed down only when the LAT alone shows the class
    assert g_lat.tolist() == [[0, -1, 0, 1]]
    assert vm.gates_from_states(q, "unsigned")[0].item() == pytest.approx(0.4)
    with pytest.raises(ValueError):
        vm.gates_from_states(q, "bad")


def test_oracle_gate_uses_true_states():
    m, b = switch_on(model(gate="oracle")), 2
    feats, scores = inputs(2 * b)
    states = torch.zeros(b, len(GATE), dtype=torch.long)
    states[0, 0] = 3  # pair 0, fracture in both views
    states[1, 0] = 2  # pair 1, fracture on LAT only
    _, aux = m(feats, scores, *vm.paired_layout(b), true_states=states)
    g = aux["gate"][:, 0].tolist()
    assert g == [1.0, -1.0, 1.0, 0.0]  # AP_1, AP_2, LAT_1, LAT_2
    with pytest.raises(ValueError):
        m(feats, scores, *vm.paired_layout(b))


def test_detached_gate_blocks_gradient_to_vsh():
    for grad, expect in (("detached", False), ("attached", True)):
        m, b = switch_on(model(gate_grad=grad)).train(), 2
        feats, scores = inputs(2 * b)
        _, aux = m(feats, scores, *vm.paired_layout(b))
        aux["delta"].sum().backward()
        got = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.vsh.parameters())
        assert got is expect, grad


def test_no_leak_between_unrelated_images():
    m, b = switch_on(model()), 3
    feats, scores = inputs(2 * b)
    out1, _ = m(feats, scores, *vm.paired_layout(b))
    feats2 = [f.clone() for f in feats]
    for f in feats2:  # change pair 2 (images 2 and 5) only
        f[[2, 5]] += 5.0
    out2, _ = m(feats2, scores, *vm.paired_layout(b))
    assert torch.equal(out1[[0, 1, 3, 4]], out2[[0, 1, 3, 4]])
    assert not torch.equal(out1[[2, 5]], out2[[2, 5]])


def test_partner_matters():
    """Changing only the LAT image changes the AP correction (cross-view path is live)."""
    m, b = switch_on(model(gate="none")), 1
    feats, scores = inputs(2)
    out1, _ = m(feats, scores, *vm.paired_layout(b))
    feats2 = [f.clone() for f in feats]
    for f in feats2:
        f[1] += 3.0
    out2, _ = m(feats2, scores, *vm.paired_layout(b))
    assert not torch.equal(out1[0], out2[0])


def test_mismatched_partner_control():
    """Control C1: pairing images across patients runs and changes the result."""
    m, b = switch_on(model(gate="none")), 2
    feats, scores = inputs(2 * b)
    partner, view = vm.paired_layout(b)
    out1, _ = m(feats, scores, partner, view)
    wrong = torch.tensor([3, 2, 1, 0])  # AP_1 <-> LAT_2, AP_2 <-> LAT_1
    out2, _ = m(feats, scores, wrong, view)
    assert not torch.equal(out1, out2)


def test_state_targets():
    has = torch.zeros(4, NC, dtype=torch.long)
    has[0, 3] = 1  # AP_1 fracture
    has[2, 3] = 1  # LAT_1 fracture  -> pair 1 '11'
    has[3, 6] = 1  # LAT_2 pronator  -> pair 2 '01'
    has[1, 4] = 1  # AP_2 metal      -> pair 2 '10'
    ap, lat = vm.pair_rows(*vm.paired_layout(2))
    t = vm.state_targets(has, ap, lat, GATE)
    assert t[0, GATE.index(3)] == 3 and t[1, GATE.index(6)] == 2 and t[1, GATE.index(4)] == 1
    assert int(t.sum()) == 6


def test_variants_build_and_run():
    for kw in ({"descriptor": "pooled"}, {"vsh": "per_class"}, {"gate": "unsigned"}):
        m, b = switch_on(model(**kw)), 2
        feats, scores = inputs(2 * b)
        out, _ = m(feats, scores, *vm.paired_layout(b))
        assert out.shape == scores.shape and torch.isfinite(out).all(), kw
    with pytest.raises(NotImplementedError):
        model(descriptor="scores")
    with pytest.raises(ValueError):
        model(gate="sometimes")


def test_bad_pairing_is_rejected():
    m = model()
    feats, scores = inputs(2)
    with pytest.raises(ValueError):
        m(feats, scores, torch.tensor([1, 0]), torch.tensor([0, 0]))  # AP paired with AP
