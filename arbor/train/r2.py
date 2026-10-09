import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from arbor.train.r1 import ROOT, Log, data_spec, gpu_hot_spells, stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    dcfg = yaml.safe_load(open(ROOT / cfg["data_config"], encoding="utf-8"))
    root = Path(os.path.expanduser(cfg["data_root"]))
    (root / "runs/r2").mkdir(parents=True, exist_ok=True)
    state, log = root / "state", Log(root / "logs" / "r2.log")
    if not (state / "tokenize.done").exists():
        sys.exit("데이터가 없습니다. R1(run_server.py 기본 설정)을 먼저 끝내야 합니다.")
    log(f"설정: {json.dumps(cfg, ensure_ascii=False)}")
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used",
                            "--format=csv,noheader", "-l", "60"], stdout=open(root / "logs/gpu_r2.csv", "a"))
    data = data_spec(root, dcfg["sources"], dcfg["mix"])
    best, rnn, mem, hdev = cfg["best_arbor"], cfg["rnn"], cfg["gpu_mem_gb"], cfg["hash_device"]

    from arbor.train.pretrain import train

    def ov(variant):
        return {} if variant == "A0" else {"rnn": rnn, "hash_device": hdev, **cfg.get("model_overrides", {})}

    def run(name, size, variant, tokens, lr, seed, micro, tps):
        d = root / "runs/r2" / name
        for attempt, mb in enumerate((micro, micro // 2)):
            try:
                return train(d, size=size, variant=variant, tokens=float(tokens), lr=lr, seed=seed, data=data,
                             micro=mb, tokens_per_step=tps, cfg_overrides=ov(variant), gpu_mem_gb=mem, log=log)
            except Exception as e:
                import torch
                torch.cuda.empty_cache()
                oom = isinstance(e, torch.OutOfMemoryError)
                log(f"[r2] {name} 실패 (micro {mb}): {type(e).__name__}: {str(e)[:300]}")
                if not oom or attempt == 1 or mb < 2:
                    return {"name": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                shutil.rmtree(d, ignore_errors=True)
                log(f"[r2] {name}: OOM → micro {mb // 2}로 처음부터 다시 (스텝당 토큰 동일)")

    def judge(r):
        return r.get("bpb", {}).get("judge") if isinstance(r, dict) else None

    @stage(state, "r2_bench", log)
    def _():
        out = {}
        for v in ("A0", best):
            r = subprocess.run([os.path.basename(sys.executable), "-m", "arbor.bench.train_throughput",
                                "--size", "M", "--variants", v, "--micro", str(cfg["m"]["micro"]), "--compile",
                                "--steps", "14", "--gpu-mem-gb", str(mem), "--hash-device", hdev,
                                "--out", str(root / "bench/throughput.jsonl")],
                               capture_output=True, text=True, executable=sys.executable, cwd=ROOT,
                               env={**os.environ,
                                    "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
                                    "PATH": os.pathsep.join(filter(None, [os.path.dirname(sys.executable), os.environ.get("PATH")]))})
            lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
            out[v] = json.loads(lines[-1]) if lines else {"error": r.stderr[-500:]}
            log(f"[bench] M-{v}: {out[v]}")
        return out

    s = cfg["s"]
    ext = cfg["lr_ext"]

    @stage(state, "r2_lr_ext", log)
    def _():
        r1 = json.loads((state / "lr_sweep.done").read_text())
        res, best_lr = {}, {}
        for v in ("A0", "A3"):
            pts = {lr: judge(r1.get(f"{v}_lr{lr}")) for lr in cfg["r1_lrs"]}
            for lr in ext[v]:
                r = run(f"lrx_{v}_{lr}", "S", v, ext["tokens"], lr, ext["seed"], s["micro"], s["tokens_per_step"])
                res[f"{v}_lr{lr}"] = r
                pts[lr] = judge(r)
            ok = {lr: b for lr, b in pts.items() if b is not None}
            best_lr[v] = min(ok, key=ok.get)
            edge = best_lr[v] in (min(ok), max(ok))
            log(f"[r2] S {v} 학습률: {sorted(ok.items())} → {best_lr[v]}{' (여전히 범위 끝)' if edge else ''}")
            res[f"{v}_points"] = {str(k): v2 for k, v2 in sorted(ok.items())}
            res[f"{v}_edge"] = edge
        res["best"] = best_lr
        return res

    lr_s = json.loads((state / "r2_lr_ext.done").read_text())["best"]
    lr_s = {k: float(v) for k, v in lr_s.items()}

    @stage(state, "r2_s_rerun", log)
    def _():
        res = {}
        for v, key in (("A0", "A0"), (best, "A3")):
            if abs(lr_s[key] - cfg["r1_best_lr"][key]) < 1e-12:
                log(f"[r2] S {v}: 학습률 변화 없음 ({lr_s[key]}) → R1 결과 그대로 사용")
                res[v] = "unchanged"
                continue
            for seed in s["seeds"]:
                res[f"{v}_s{seed}"] = run(f"s_{v}_lr{lr_s[key]}_s{seed}", "S", v, s["full_tokens"], lr_s[key], seed,
                                          s["micro"], s["tokens_per_step"])
        return res

    m = cfg["m"]

    @stage(state, "r2_m_sweep", log)
    def _():
        res, best_lr = {}, {}
        for v, key in (("A0", "A0"), (best, "A3")):
            pts = {}
            for k in m["sweep_mults"]:
                lr = round(lr_s[key] * k, 8)
                res[f"{v}_lr{lr}"] = run(f"msw_{v}_{lr}", "M", v, m["sweep_tokens"], lr, 0, m["micro"], m["tokens_per_step"])
                pts[lr] = judge(res[f"{v}_lr{lr}"])
            ok = {lr: b for lr, b in pts.items() if b is not None}
            best_lr[v] = min(ok, key=ok.get)
            res[f"{v}_edge"] = best_lr[v] in (min(ok), max(ok))
            log(f"[r2] M {v} 학습률: {sorted(ok.items())} → {best_lr[v]}")
        for d in (root / "runs/r2").glob("msw_*"):
            (d / "hash.pt").unlink(missing_ok=True)
        res["best"] = best_lr
        return res

    lr_m = {k: float(v) for k, v in json.loads((state / "r2_m_sweep.done").read_text())["best"].items()}
    full = {}
    for seed in (0, 1):
        for v in ("A0", best):
            full[f"{v}_s{seed}"] = run(f"m_{v}_s{seed}", "M", v, m["full_tokens"], lr_m[v], seed,
                                       m["micro"], m["tokens_per_step"])
        if seed == 0:
            a, b = judge(full["A0_s0"]), judge(full[f"{best}_s0"])
            if a is not None and b is not None and abs(a - b) >= m["seed_rule_bpb"]:
                log(f"[r2] M 시드 0 차이 {a - b:+.4f} (≥ {m['seed_rule_bpb']}) → 시드 1 생략")
                break
            log(f"[r2] M 시드 0 차이 {None if a is None or b is None else round(a - b, 4)} → 시드 1 추가")
    write_summary(root, state, cfg, best, lr_s, lr_m, full, log)
    mon.terminate()
    log("== R2 전체 끝")


def write_summary(root, state, cfg, best, lr_s, lr_m, full, log):
    def mean(v):
        xs = [r["bpb"]["judge"] for k, r in full.items() if k.startswith(v + "_s") and "bpb" in r]
        return sum(xs) / len(xs) if xs else None

    rows = ["| 실행 | 판정 bpb | ko | en | code | 학습률 | 학습 tok/s | 시간(h) | 학습 FLOP |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in full.items():
        if "bpb" not in r:
            rows.append(f"| {k} | 실패: {r.get('error', '')[:60]} |||||||")
            continue
        b = r["bpb"]
        rows.append(f"| {k} | {b['judge']:.4f} | {b['ko']:.4f} | {b['en']:.4f} | {b.get('code', float('nan')):.4f} | "
                    f"{r['lr']} | {r['tok_per_s']:,} | {r['hours']} | {r['train_flop']:.2e} |")
    a0, ar = mean("A0"), mean(best)
    rq = "계산 불가"
    if a0 is not None and ar is not None:
        f0 = next(r["train_flop"] for k, r in full.items() if k.startswith("A0_") and "bpb" in r)
        f1 = next(r["train_flop"] for k, r in full.items() if k.startswith(best + "_") and "bpb" in r)
        d = a0 - ar
        verdict = "성공" if d >= 0.03 else ("실패" if d < -0.03 else "부분 성공")
        rq = (f"A0-M {a0:.4f} − {best}-M {ar:.4f} = **{d:+.4f}** → {verdict} "
              f"(학습 FLOP 비 {f1 / f0:.3f}, 기준 ±10%)")
    lx = json.loads((state / "r2_lr_ext.done").read_text())
    ms = json.loads((state / "r2_m_sweep.done").read_text())
    md = ["# R2 결과 (자동 생성) `[실측]`", "",
          f"- 최선 Arbor: {best} (CPU tok/s 동률 규칙), 혼합층 {cfg['rnn']}, 해시 표 위치 {cfg['hash_device']}, GPU 상한 {cfg['gpu_mem_gb']}GB",
          f"- S 학습률 보강: A0 {lx['A0_points']} (범위 끝: {lx['A0_edge']}), A3 {lx['A3_points']} (범위 끝: {lx['A3_edge']})",
          f"- S 최선 학습률: {lr_s}",
          f"- M 학습률 탐색 최선: {lr_m} (범위 끝: A0 {ms['A0_edge']}, {best} {ms[f'{best}_edge']})", "",
          "## M 본 학습 (20억 토큰)", *rows, "",
          f"## R-Q (자동 계산, 최종 판정은 REPORT.md): {rq}", "",
          f"## GPU 83°C 초과 3분 이상 구간: {gpu_hot_spells(root / 'logs/gpu_r2.csv') or '없음'}", "",
          "## M 처리량 (r2_bench)", "```", json.dumps(json.loads((state / "r2_bench.done").read_text()),
                                                     ensure_ascii=False, indent=1), "```"]
    (root / "runs/r2/summary.md").write_text("\n".join(md), encoding="utf-8")
    log("[r2] 요약: runs/r2/summary.md")


if __name__ == "__main__":
    main()
