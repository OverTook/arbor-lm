import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUWeighted(torch.autograd.Function):

    @staticmethod
    def forward(ctx, g, u, w):
        ctx.save_for_backward(g, u, w)
        return F.silu(g) * u * w

    @staticmethod
    def backward(ctx, dy):
        g, u, w = ctx.saved_tensors
        sg = torch.sigmoid(g.float())
        silu = g.float() * sg
        dyw = dy.float() * w.float()
        du = (dyw * silu).to(u.dtype)
        dg = (dyw * u.float() * sg * (1 + g.float() * (1 - sg))).to(g.dtype)
        dw = (dy.float() * silu * u.float()).sum(-1, keepdim=True).to(w.dtype)
        return dg, du, dw


def _pad_row(t):
    return torch.cat([t, t.new_zeros(1, t.shape[1])])


class Dispatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, src, pos):
        ctx.save_for_backward(pos)
        return _pad_row(x).index_select(0, src)

    @staticmethod
    def backward(ctx, dxp):
        (pos,) = ctx.saved_tensors
        N, beam = pos.shape
        return dxp.index_select(0, pos.flatten()).view(N, beam, -1).sum(1), None, None


class Combine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, src, pos):
        ctx.save_for_backward(src)
        N, beam = pos.shape
        return y.index_select(0, pos.flatten()).view(N, beam, -1).sum(1)

    @staticmethod
    def backward(ctx, dout):
        (src,) = ctx.saved_tensors
        return _pad_row(dout).index_select(0, src), None, None


def leaf_paths(depth):
    n = 2 ** depth
    nodes = torch.zeros(n, depth, dtype=torch.long)
    signs = torch.zeros(n, depth)
    for leaf in range(n):
        node = 0
        for lvl in range(depth):
            right = (leaf >> (depth - 1 - lvl)) & 1
            nodes[leaf, lvl], signs[leaf, lvl] = node, -1.0 if right else 1.0
            node = 2 * node + 1 + right
    return nodes, signs


def beam_select(cum, beam):
    alive = torch.ones_like(cum[0], dtype=torch.bool)
    for lvl, lp in enumerate(cum):
        if lvl:
            alive = alive.repeat_interleave(2, 1)
        lp = lp.masked_fill(~alive, float("-inf"))
        if alive.shape[1] > beam:
            keep = lp.topk(beam, 1).indices
            alive = torch.zeros_like(alive).scatter_(1, keep, True)
    return alive


class SharedBlock(nn.Module):

    def __init__(self, d, f):
        super().__init__()
        self.gate, self.up = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TreeFFN(nn.Module):

    def __init__(self, d, ffn_width, depth=4, beam=4, shared=False):
        super().__init__()
        if shared:
            beam = 3
        self.n, self.beam, self.depth = 2 ** depth, beam, depth
        leaf = ffn_width // 4
        self.shared = SharedBlock(d, leaf) if shared else None
        self.w_gate = nn.Parameter(torch.randn(self.n, d, leaf) * d ** -0.5)
        self.w_up = nn.Parameter(torch.randn(self.n, d, leaf) * d ** -0.5)
        self.w_down = nn.Parameter(torch.randn(self.n, leaf, d) * leaf ** -0.5)
        self.router = nn.Linear(d, self.n - 1, bias=False)
        nodes, signs = leaf_paths(depth)
        self.register_buffer("nodes", nodes, persistent=False)
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("temp", torch.tensor(1.0))
        self.aux_loss = None

    def leaf_logprob(self, x):
        z = self.router(x).float() / self.temp
        zs = z.index_select(1, self.nodes.flatten()).view(-1, self.n, self.depth)
        steps = F.logsigmoid(zs * self.signs).unbind(-1)
        acc, cum = steps[0], [steps[0]]
        for s in steps[1:]:
            acc = acc + s
            cum.append(acc)
        return torch.stack(cum, -1)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        cum = self.leaf_logprob(x)
        per_lvl = [cum[:, :: self.n >> (l + 1), l] for l in range(self.depth)]
        sel = beam_select(per_lvl, self.beam)
        lp = cum[:, :, -1]
        top_lp, top_idx = lp.masked_fill(~sel, float("-inf")).topk(self.beam, 1)
        w = top_lp.softmax(1).to(x.dtype)

        probs = lp.exp()
        frac = torch.bincount(top_idx.flatten(), minlength=self.n).float() / top_idx.numel()
        self.aux_loss = self.n * (frac * probs.mean(0)).sum()

        out = self._grouped(x, top_idx, w)
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(shape)

    def _grouped(self, x, top_idx, w):
        N, d = x.shape
        xc = x.to(torch.get_autocast_dtype(x.device.type)) if torch.is_autocast_enabled(x.device.type) else x
        flat_idx = top_idx.flatten()
        order = flat_idx.argsort(stable=True)
        leaf_sorted = flat_idx[order]
        counts = torch.bincount(flat_idx, minlength=self.n)
        start = counts.cumsum(0) - counts
        slot = torch.arange(len(order), device=x.device) - start[leaf_sorted]
        cap = int(counts.max())
        flat_slot = leaf_sorted * cap + slot
        src = torch.full((self.n * cap,), N, device=x.device, dtype=torch.long)
        src[flat_slot] = order // self.beam
        pos = torch.empty_like(flat_slot)
        pos[order] = flat_slot
        pos = pos.view(N, self.beam)
        wpad = torch.zeros(self.n * cap, device=x.device, dtype=xc.dtype)
        wpad[flat_slot] = w.flatten()[order].to(xc.dtype)
        dt = xc.dtype
        xp = Dispatch.apply(xc, src, pos).view(self.n, cap, d)
        h = SwiGLUWeighted.apply(torch.bmm(xp, self.w_gate.to(dt)), torch.bmm(xp, self.w_up.to(dt)),
                                 wpad.view(self.n, cap, 1))
        y = torch.bmm(h, self.w_down.to(dt))
        return Combine.apply(y.view(-1, d), src, pos).to(x.dtype)

    def forward_dense_masked(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        cum = self.leaf_logprob(x)
        per_lvl = [cum[:, :: self.n >> (l + 1), l] for l in range(self.depth)]
        sel = beam_select(per_lvl, self.beam)
        wts = cum[:, :, -1].masked_fill(~sel, float("-inf")).softmax(1).to(x.dtype)
        h = F.silu(torch.einsum("nd,edf->nef", x, self.w_gate)) * torch.einsum("nd,edf->nef", x, self.w_up)
        y = torch.einsum("nef,efd->ned", h, self.w_down)
        out = (y * wts[..., None]).sum(1)
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(shape)
