from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from arbor.layers.hashmem import HashInject, HashTable
from arbor.layers.mixer import Mixer, rope
from arbor.layers.treeffn import TreeFFN


@dataclass
class Config:
    vocab: int = 32000
    d: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 1344
    mixer: str = "attn"
    tree: bool = False
    hash: bool = False
    gla_heads: int = 6
    rnn: str = "gla"
    shared: bool = False
    window: int = 64
    hash_layers: tuple = (1, 5)
    hash_log_rows: int = 21
    hash_device: str = "cpu"
    ckpt: bool = False


PRESETS = {
    "S": dict(d=512, layers=8, heads=8, ffn=1344, gla_heads=6),
    "M": dict(d=768, layers=12, heads=12, ffn=2048, gla_heads=9),
}
LADDER = {
    "A0": dict(mixer="attn"),
    "A1": dict(mixer="mixer"),
    "A2": dict(mixer="mixer", tree=True),
    "A3": dict(mixer="mixer", tree=True, hash=True),
    "A3s": dict(mixer="mixer", tree=True, hash=True, shared=True),
}


def make_config(size, variant, **kw):
    return Config(**{**PRESETS[size], **LADDER[variant], **kw})


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate, self.up = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config, inject: bool):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c.d), nn.RMSNorm(c.d)
        self.mix = Attention(c.d, c.heads) if c.mixer == "attn" else Mixer(c.d, c.heads, c.gla_heads, c.window, rnn=c.rnn)
        self.ffn = TreeFFN(c.d, c.ffn, shared=c.shared) if c.tree else SwiGLU(c.d, c.ffn)
        self.inject = HashInject(c.d) if inject else None

    def forward(self, x, emb):
        x = x + self.mix(self.n1(x))
        if self.inject is not None:
            x = self.inject(x, emb)
        return x + self.ffn(self.n2(x))


def _ce_sum(h, w, t):
    return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")


def chunked_ce(h, w, targets, chunk=2048):
    total = sum(checkpoint(_ce_sum, h[i:i + chunk], w, targets[i:i + chunk], use_reentrant=False)
                for i in range(0, h.shape[0], chunk))
    return total / h.shape[0]


class LM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c, c.hash and i in c.hash_layers) for i in range(c.layers))
        self.norm = nn.RMSNorm(c.d)
        self.hash = HashTable(c.hash_log_rows, device=c.hash_device) if c.hash else None

    def forward(self, ids, targets=None):
        x = self.emb(ids)
        emb = self.hash.gather(ids) if self.hash else None
        for b in self.blocks:
            x = checkpoint(b, x, emb, use_reentrant=False) if self.c.ckpt and self.training else b(x, emb)
        h = self.norm(x)
        if targets is None:
            return F.linear(h, self.emb.weight)
        loss = chunked_ce(h.flatten(0, 1), self.emb.weight, targets.reshape(-1))
        aux = sum(b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, TreeFFN))
        return None, loss, aux

    def active_params(self):
        n = 0
        for name, p in self.named_parameters():
            if name.startswith("blocks") and ".ffn.w_" in name:
                ffn = self.blocks[0].ffn
                n += p.numel() * ffn.beam // ffn.n
            else:
                n += p.numel()
        return n

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
