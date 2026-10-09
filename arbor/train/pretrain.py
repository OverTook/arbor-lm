import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from arbor.data.memmap import load_bin
from arbor.models.lm import LM, make_config
from arbor.layers.treeffn import TreeFFN


def set_gpu_cap(gpu_mem_gb=None):
    total = torch.cuda.get_device_properties(0).total_memory
    frac = 0.97 if gpu_mem_gb is None else min(0.97, gpu_mem_gb * 1e9 / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac


class Mix:

    def __init__(self, groups, weights, seq):
        self.seq = seq
        self.groups = [(g, [load_bin(p) for p in paths]) for g, paths in groups.items()]
        w = np.array([weights[g] for g, _ in self.groups], dtype=np.float64)
        self.w = w / w.sum()

    def batch(self, seed, step, n):
        rng = np.random.default_rng([seed, step])
        out = np.empty((n, self.seq + 1), dtype=np.int64)
        gi = rng.choice(len(self.groups), size=n, p=self.w)
        for i, g in enumerate(gi):
            arrs = self.groups[g][1]
            sizes = np.array([len(a) for a in arrs], dtype=np.float64)
            a = arrs[rng.choice(len(arrs), p=sizes / sizes.sum())]
            o = int(rng.integers(0, len(a) - self.seq - 1))
            out[i] = a[o:o + self.seq + 1]
        return torch.from_numpy(out)


@torch.no_grad()
def eval_bpb(model, val, seq, micro):
    model.eval()
    res = {}
    for g, (path, meta) in val.items():
        a = load_bin(path)
        n = (len(a) - 1) // seq
        nats, pred = 0.0, 0
        for i in range(0, n, micro):
            idx = np.arange(i, min(n, i + micro))
            x = torch.from_numpy(np.stack([a[j * seq:j * seq + seq + 1] for j in idx]).astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x[:, :-1], x[:, 1:])
            nats += loss.item() * x[:, 1:].numel()
            pred += x[:, 1:].numel()
        used_bytes = meta["bytes"] * (n * seq) / meta["tokens"]
        res[g] = nats / math.log(2) / used_bytes
    model.train()
    if "ko" in res and "en" in res:
        res["judge"] = 0.6 * res["ko"] + 0.4 * res["en"]
    return res


def lr_at(step, total, peak, warmup_frac=0.01, final_frac=0.1):
    w = max(1, int(total * warmup_frac))
    if step < w:
        return peak * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def _param_groups(m, wd):
    decay = [p for p in m.parameters() if p.ndim >= 2]
    no = [p for p in m.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def train(run_dir, size, variant, tokens, lr, seed, data, micro, tokens_per_step, seq=2048,
          cfg_overrides=None, compile=True, ckpt_minutes=20, evals=4, log=print, stop_after=None, gpu_mem_gb=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    torch.manual_seed(seed)
    set_gpu_cap(gpu_mem_gb)
    c = make_config(size, variant, **(cfg_overrides or {}))
    m = LM(c).cuda()
    opt = torch.optim.AdamW(_param_groups(m, 0.1), lr=lr, betas=(0.9, 0.95), fused=True)
    accum = max(1, tokens_per_step // (micro * seq))
    total = math.ceil(tokens / (accum * micro * seq))
    act = m.active_params()
    mix = Mix(data["groups"], data["weights"], seq)
    step = 0
    ck = run_dir / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cuda", weights_only=False)
        m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        if m.hash is not None:
            h = torch.load(run_dir / "ckpt_hash.pt", map_location="cpu", weights_only=False)
            m.hash.table.copy_(h["table"]); m.hash.state.copy_(h["state"])
            if h.get("step") != step:
                log(f"[train] 경고: 해시 표 step {h.get('step')} ≠ 모델 step {step}")
        log(f"[train] {run_dir.name}: step {step}에서 재개")
    if compile:
        for b in m.blocks:
            b.compile()
    hist = []
    if (run_dir / "evals.json").exists():
        hist = json.loads((run_dir / "evals.json").read_text())
    eval_at = {round(total * (k + 1) / evals) for k in range(evals)}
    trees = [b.ffn for b in m.blocks if isinstance(b.ffn, TreeFFN)]
    t0, last_ck, done_tok = time.time(), time.time(), 0
    logf = open(run_dir / "train.jsonl", "a")
    log(f"[train] {run_dir.name}: 활성 {act:,} 총 {m.total_params():,} 스텝 {total} × {accum * micro * seq:,}토큰, "
        f"예상 FLOP {6 * act * tokens:.2e}")
    while step < total:
        for p in opt.param_groups:
            p["lr"] = lr_at(step, total, lr)
        for t in trees:
            t.temp.fill_(1.0 - 0.5 * step / total)
        loss_sum = 0.0
        for a in range(accum):
            x = mix.batch(seed, step * accum + a, micro).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, aux = m(x[:, :-1], x[:, 1:])
            ((loss + 0.01 * aux) / accum).backward()
            if m.hash is not None:
                m.hash.step()
            loss_sum += loss.item() / accum
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        done_tok += accum * micro * seq
        el = time.time() - t0
        tps = done_tok / el
        rec = dict(step=step, loss=round(loss_sum, 4), lr=p["lr"], gnorm=round(gn, 3), tok_s=round(tps),
                   eta_h=round((total - step) * accum * micro * seq / tps / 3600, 2))
        logf.write(json.dumps(rec) + "\n")
        if step % 50 == 0 or step == total:
            logf.flush()
            log(f"[train] {run_dir.name}: {step}/{total} loss {loss_sum:.3f} {tps:,.0f} tok/s 남은 {rec['eta_h']}h")
        if not math.isfinite(loss_sum):
            raise RuntimeError(f"{run_dir.name}: 손실이 발산했다 (step {step})")
        if step in eval_at:
            r = eval_bpb(m, data["val"], seq, micro)
            hist.append(dict(step=step, tokens=step * accum * micro * seq, **r))
            (run_dir / "evals.json").write_text(json.dumps(hist, indent=1))
            log(f"[eval] {run_dir.name}: step {step} {r}")
        if time.time() - last_ck > ckpt_minutes * 60 and step < total:
            _save(run_dir, m, opt, step)
            last_ck = time.time()
        if stop_after is not None and step == stop_after:
            _save(run_dir, m, opt, step)
            return None
    logf.close()
    final = hist[-1]
    res = dict(name=run_dir.name, size=size, variant=variant, tokens=tokens, lr=lr, seed=seed,
               active_params=act, total_params=m.total_params(), train_flop=6 * act * tokens,
               tok_per_s=round(tps), hours=round((time.time() - t0) / 3600, 3), bpb=final,
               cfg=cfg_overrides or {})
    torch.save(m.state_dict(), run_dir / "model.pt")
    if m.hash is not None:
        torch.save({"table": m.hash.table.cpu(), "state": m.hash.state.cpu()}, run_dir / "hash.pt")
    for f in ("ckpt.pt", "ckpt_hash.pt"):
        (run_dir / f).unlink(missing_ok=True)
    (run_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def _save(run_dir, m, opt, step):
    if m.hash is not None:
        torch.save({"table": m.hash.table, "state": m.hash.state, "step": step}, run_dir / "ckpt_hash.tmp")
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step}, run_dir / "ckpt.tmp")
    if m.hash is not None:
        os.replace(run_dir / "ckpt_hash.tmp", run_dir / "ckpt_hash.pt")
    os.replace(run_dir / "ckpt.tmp", run_dir / "ckpt.pt")
