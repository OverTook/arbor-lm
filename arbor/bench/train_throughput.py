import argparse
import gc
import json
import os
import time

import torch

from arbor.models.lm import LM, make_config


def make_optimizer(params, kind, lr=3e-4):
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, fused=True)


def run(a, variant):
    torch.manual_seed(a.seed)
    from arbor.train.pretrain import set_gpu_cap
    set_gpu_cap(a.gpu_mem_gb)
    c = make_config(a.size, variant, ckpt=a.ckpt, hash_device=a.hash_device)
    m = LM(c).cuda()
    opt = make_optimizer(m.parameters(), a.opt)
    if a.compile:
        for b in m.blocks:
            b.compile()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(a.steps):
        ids = torch.randint(0, c.vocab, (a.micro, a.seq + 1), device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, aux = m(ids[:, :-1], ids[:, 1:])
        (loss + 0.01 * aux).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if m.hash:
            m.hash.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    warm = 5 if a.compile else 3
    dt = sum(times[warm:]) / len(times[warm:])
    tok_s = a.micro * a.seq / dt
    act = m.active_params()
    return dict(active_params=act, total_params=m.total_params(),
                tok_per_s=round(tok_s), eff_tflops=round(6 * act * tok_s / 1e12, 2),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                alloc_retries=torch.cuda.memory_stats()["num_alloc_retries"],
                loss=round(loss.item(), 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="S")
    p.add_argument("--variants", default="A0,A1,A2,A3")
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--steps", type=int, default=13)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--opt", default="adamw", choices=["adamw", "adamw8bit"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--gpu-mem-gb", type=float, help="GPU 메모리 사용 상한(GB)")
    p.add_argument("--hash-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.expanduser("~/arbor-data/bench/throughput.jsonl"))
    a = p.parse_args()
    for v in a.variants.split(","):
        head = dict(size=a.size, variant=v, micro=a.micro, seq=a.seq, ckpt=a.ckpt, opt=a.opt, compile=a.compile,
                    hash_device=a.hash_device, gpu_mem_gb=a.gpu_mem_gb)
        try:
            r = {**head, **run(a, v)}
        except torch.OutOfMemoryError:
            r = {**head, "oom": True}
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
