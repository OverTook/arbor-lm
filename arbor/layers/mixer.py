import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = chunk_gated_delta_rule = None


def rope(x, base=10000.0):
    T, D = x.shape[2], x.shape[3]
    inv = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def gla_reference(q, k, v, g, scale):
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1], dtype=torch.float32)
    out = []
    for t in range(T):
        S = S * g[:, t].float().exp().unsqueeze(-1) + k[:, t].float().unsqueeze(-1) * v[:, t].float().unsqueeze(-2)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t].float() * scale, S))
    return torch.stack(out, 1).to(q.dtype)


def gdn_reference(q, k, v, g, beta, scale):
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1])
    out = []
    for t in range(T):
        S = S * g[:, t].float().exp()[..., None, None]
        kt, vt, bt = k[:, t], v[:, t].float(), beta[:, t].float()[..., None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, bt * (vt - pred))
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, 1).to(v.dtype)


def window_attention(q, k, v, w):
    B, H, T, D = q.shape
    pad = (-T) % w
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    n = q.shape[2] // w
    qb = q.view(B, H, n, w, D)
    kb, vb = (torch.cat([F.pad(x.view(B, H, n, w, D), (0, 0, 0, 0, 1, 0))[:, :, :-1], x.view(B, H, n, w, D)], 3)
              for x in (k, v))
    i = torch.arange(w, device=q.device).view(w, 1) + w
    j = torch.arange(2 * w, device=q.device).view(1, 2 * w)
    mask = (j <= i) & (j > i - w)
    blk0 = torch.ones(n, 1, 1, dtype=torch.bool, device=q.device)
    blk0[0] = False
    mask = mask & ((j >= w) | blk0)
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask)
    return o.reshape(B, H, n * w, D)[:, :, :T]


class Mixer(nn.Module):

    def __init__(self, d, n_heads, n_gla_heads, window=64, gate_rank=16, rnn="gla"):
        super().__init__()
        self.h, self.hg, self.dh, self.window, self.rnn = n_heads, n_gla_heads, d // n_heads, window, rnn
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if rnn == "gla":
            self.gate = nn.Sequential(nn.Linear(d, gate_rank, bias=False),
                                      nn.Linear(gate_rank, n_gla_heads * self.dh, bias=True))
        else:
            self.gate = nn.Linear(d, 2 * n_gla_heads, bias=True)
        self.gla_norm = nn.RMSNorm(self.dh)
        self.use_reference = False

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        hg = self.hg
        qg, kg, vg = q[:, :, :hg], k[:, :, :hg], v[:, :, :hg]
        scale = self.dh ** -0.5
        ref = self.use_reference or chunk_gla is None or not x.is_cuda
        if self.rnn == "gla":
            g = F.logsigmoid(self.gate(x).view(B, T, hg, self.dh).float()) / 16
            og = gla_reference(qg, kg, vg, g, scale) if ref else chunk_gla(qg, kg, vg, g.to(qg.dtype), scale=scale)[0]
        else:
            a, b = self.gate(x).float().view(B, T, 2, hg).unbind(2)
            g, beta = F.logsigmoid(a) / 16, torch.sigmoid(b)
            if ref:
                og = gdn_reference(qg, kg, vg, g, beta, scale)
            else:
                og = chunk_gated_delta_rule(qg, kg, vg, g, beta.to(qg.dtype), scale=scale,
                                            use_qk_l2norm_in_kernel=True)[0]
        og = self.gla_norm(og)
        qw, kw, vw = (t[:, :, hg:].transpose(1, 2) for t in (q, k, v))
        ow = window_attention(rope(qw), rope(kw), vw, self.window).transpose(1, 2)
        return self.o(torch.cat([og, ow], 2).reshape(B, T, -1))
