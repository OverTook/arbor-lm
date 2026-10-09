import torch
import torch.nn as nn

N_SUB = 4
_MULT = [0x9E3779B97F4A7C15 - 2**64, 0xC2B2AE3D27D4EB4F - 2**64, 0x165667B19E3779F9, 0x27D4EB2F165667C5]


def ngram_rows(ids, log_rows):
    p1 = torch.nn.functional.pad(ids, (1, 0), value=-1)[:, :-1]
    p2 = torch.nn.functional.pad(ids, (2, 0), value=-1)[:, :-2]
    grams = [(ids, p1, None), (ids, p1, None), (ids, p1, p2), (ids, p1, p2)]
    rows = []
    for s, (a, b, c) in enumerate(grams):
        h = a * _MULT[s] + b
        if c is not None:
            h = h * _MULT[s] + c
        h = (h * _MULT[(s + 1) % 4]) >> (64 - log_rows)
        rows.append((h & ((1 << log_rows) - 1)) + (s << log_rows))
    return torch.stack(rows, -1)


class HashTable:

    def __init__(self, log_rows=21, dim=256, lr=1e-2, eps=1e-8, device="cpu"):
        self.log_rows, self.dim, self.lr, self.eps = log_rows, dim, lr, eps
        n = N_SUB << log_rows
        self.table = torch.empty(n, dim, device=device).zero_()
        self.state = torch.empty(n, device=device).zero_()
        self._buf = None
        self._idx = None
        self.rows = None

    def gather(self, ids):
        r = ngram_rows(ids, self.log_rows)
        uniq, inv = r.flatten().unique(return_inverse=True)
        if self.table.is_cuda:
            self._idx = uniq
            self.rows = self.table.index_select(0, uniq).requires_grad_(True)
            return self.rows[inv].view(*ids.shape, N_SUB * self.dim)
        idx = uniq.cpu()
        if self._buf is None or self._buf.shape[0] < len(idx):
            self._buf = torch.empty(int(len(idx) * 1.5), self.dim).pin_memory()
        buf = self._buf[: len(idx)]
        torch.index_select(self.table, 0, idx, out=buf)
        self.rows = buf.to(ids.device, non_blocking=True).requires_grad_(True)
        self._idx = idx
        return self.rows[inv].view(*ids.shape, N_SUB * self.dim)

    @torch.no_grad()
    def step(self):
        if self.rows is None or self.rows.grad is None:
            return
        g = self.rows.grad.float().to(self.table.device)
        idx = self._idx
        st = self.state.index_select(0, idx).add_(g.pow(2).mean(1))
        self.state.index_copy_(0, idx, st)
        self.table.index_add_(0, idx, g.div_(st.sqrt_().add_(self.eps)[:, None]), alpha=-self.lr)
        self.rows = None


class HashInject(nn.Module):

    def __init__(self, d, dim=256):
        super().__init__()
        self.proj = nn.Linear(N_SUB * dim, d, bias=False)
        self.gate = nn.Linear(d, 1, bias=True)

    def forward(self, h, emb):
        return h + torch.sigmoid(self.gate(h)) * self.proj(emb.to(h.dtype))
