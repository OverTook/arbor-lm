import pytest
import torch
import torch.nn.functional as F

from arbor.layers.hashmem import HashTable, ngram_rows
from arbor.layers.mixer import Mixer, gla_reference, window_attention
from arbor.layers.treeffn import TreeFFN
from arbor.models.lm import LM, make_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")


def test_window_attention_matches_full_masked():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 150, 16) for _ in range(3))
    i, j = torch.arange(150)[:, None], torch.arange(150)[None]
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i) & (j > i - 64))
    torch.testing.assert_close(window_attention(q, k, v, 64), ref, atol=1e-5, rtol=1e-5)


@cuda
def test_gla_chunk_matches_recurrent():
    from fla.ops.gla import chunk_gla
    torch.manual_seed(0)
    B, T, H, K = 2, 200, 3, 32
    q, k, v = (torch.randn(B, T, H, K, device="cuda") for _ in range(3))
    g = F.logsigmoid(torch.randn(B, T, H, K, device="cuda")) / 16
    o, _ = chunk_gla(q, k, v, g, scale=K ** -0.5)
    ref = gla_reference(q, k, v, g, K ** -0.5)
    assert ((o - ref).norm() / ref.norm()).item() < 5e-3


@cuda
def test_mixer_chunk_mode_equals_recurrent_mode():
    torch.manual_seed(0)
    m = Mixer(128, 4, 3).cuda()
    x = torch.randn(2, 130, 128, device="cuda")
    a = m(x)
    m.use_reference = True
    torch.testing.assert_close(a, m(x), atol=2e-3, rtol=2e-3)


def test_tree_beam_equals_masked_dense():
    torch.manual_seed(0)
    f = TreeFFN(64, 128)
    with torch.no_grad():
        f.router.weight.mul_(10)
    x = torch.randn(300, 64)
    torch.testing.assert_close(f(x), f.forward_dense_masked(x), atol=1e-5, rtol=1e-5)


def test_tree_selects_exactly_beam_leaves_and_differs_from_plain_topk_sometimes():
    torch.manual_seed(1)
    f = TreeFFN(64, 128)
    with torch.no_grad():
        f.router.weight.mul_(10)
    x = torch.randn(2000, 64)
    from arbor.layers.treeffn import beam_select
    cum = f.leaf_logprob(x)
    sel = beam_select([cum[:, :: 16 >> (l + 1), l] for l in range(4)], 4)
    assert (sel.sum(1) == 4).all()
    plain = torch.zeros_like(sel).scatter_(1, cum[:, :, -1].topk(4, 1).indices, True)
    assert (sel != plain).any()


def test_hash_rows_in_range_and_deterministic():
    ids = torch.randint(0, 32000, (2, 50))
    r = ngram_rows(ids, 10)
    assert r.shape == (2, 50, 4)
    for s in range(4):
        assert ((r[..., s] >> 10) == s).all()
    assert torch.equal(r, ngram_rows(ids, 10))


def test_hash_shared_by_two_layers_and_adagrad_updates_rows():
    torch.manual_seed(0)
    c = make_config("S", "A3", d=64, heads=4, gla_heads=3, ffn=128, layers=6, hash_log_rows=8, vocab=100)
    m = LM(c)
    injects = [b.inject for b in m.blocks if b.inject is not None]
    assert len(injects) == 2 and isinstance(m.hash, HashTable)
    ids = torch.randint(0, 100, (2, 32))
    m.hash.table.normal_(0, 0.1)
    before = m.hash.table.clone()
    _, loss, _ = m(ids, ids)
    loss.backward()
    rows = m.hash._idx
    assert m.hash.rows.grad is not None
    m.hash.step()
    changed = (m.hash.table != before).any(1)
    assert changed[rows].all() and changed.sum() == len(rows)


def test_swiglu_weighted_grad_matches_autograd():
    from arbor.layers.treeffn import SwiGLUWeighted
    torch.manual_seed(0)
    g, u = torch.randn(3, 5, 7, dtype=torch.double, requires_grad=True), torch.randn(3, 5, 7, dtype=torch.double, requires_grad=True)
    w = torch.rand(3, 5, 1, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(SwiGLUWeighted.apply, (g, u, w))


def test_chunked_ce_equals_full_ce():
    from arbor.models.lm import chunked_ce
    torch.manual_seed(0)
    h = torch.randn(5000, 32, requires_grad=True)
    w = torch.randn(100, 32, requires_grad=True)
    t = torch.randint(0, 100, (5000,))
    a = chunked_ce(h, w, t, chunk=1024)
    a.backward()
    ga = (h.grad.clone(), w.grad.clone())
    h.grad = w.grad = None
    b = F.cross_entropy(h @ w.T, t)
    b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(ga[0], h.grad)
    torch.testing.assert_close(ga[1], w.grad)


def test_tree_grads_match_masked_dense():
    torch.manual_seed(0)
    f = TreeFFN(32, 64)
    with torch.no_grad():
        f.router.weight.mul_(10)
    x = torch.randn(200, 32, requires_grad=True)
    f(x).pow(2).sum().backward()
    g1 = [x.grad.clone()] + [p.grad.clone() for p in f.parameters()]
    x.grad = None
    f.zero_grad()
    f.forward_dense_masked(x).pow(2).sum().backward()
    g2 = [x.grad] + [p.grad for p in f.parameters()]
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@cuda
def test_gdn_chunk_matches_recurrent():
    from arbor.layers.mixer import gdn_reference
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    torch.manual_seed(0)
    B, T, H, K = 2, 150, 3, 32
    q, k, v = (torch.randn(B, T, H, K, device="cuda") for _ in range(3))
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda")) / 16
    beta = torch.rand(B, T, H, device="cuda")
    o = chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5, use_qk_l2norm_in_kernel=True)[0]
    ref = gdn_reference(q, k, v, g, beta, K ** -0.5)
    assert ((o - ref).norm() / ref.norm()).item() < 5e-3


def test_shared_tree_active_width_and_dense_match():
    torch.manual_seed(0)
    f = TreeFFN(32, 64, shared=True)
    assert f.beam == 3 and f.w_gate.shape[-1] == 16 and f.shared is not None
    with torch.no_grad():
        f.router.weight.mul_(10)
    x = torch.randn(100, 32)
    torch.testing.assert_close(f(x), f.forward_dense_masked(x), atol=1e-5, rtol=1e-5)
    m = LM(make_config("S", "A3s", d=64, heads=4, gla_heads=3, ffn=128, layers=2, hash=False, vocab=100))
    m0 = LM(make_config("S", "A0", d=64, heads=4, ffn=128, layers=2, vocab=100))
    assert abs(m.active_params() / m0.active_params() - 1) < 0.1


@cuda
def test_hash_table_on_gpu_matches_cpu_update():
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (2, 40), device="cuda")
    tabs = [HashTable(log_rows=8, device=dev) for dev in ("cpu", "cuda")]
    for t in tabs:
        t.table.copy_(torch.randn(t.table.shape, generator=torch.Generator().manual_seed(1)).to(t.table.device))
    w = torch.randn(4 * 256, device="cuda")
    for t in tabs:
        for _ in range(3):
            (t.gather(ids) @ w).pow(2).sum().backward()
            t.step()
    torch.testing.assert_close(tabs[0].table, tabs[1].table.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tabs[0].state, tabs[1].state.cpu(), atol=1e-5, rtol=1e-5)
