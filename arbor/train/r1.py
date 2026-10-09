import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def gpu_hot_spells(path, limit=83.0, minutes=3):
    spells, run = [], []
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        parts = [p.strip() for p in line.split(",")]
        try:
            hot = float(parts[1]) > limit
        except (IndexError, ValueError):
            continue
        run = run + [parts[0]] if hot else []
        if len(run) == minutes:
            spells.append(run[0])
    return spells


def stage(state, name, log):
    done = state / f"{name}.done"

    def deco(fn):
        if done.exists():
            log(f"== {name}: 이미 끝남, 건너뜀")
            return json.loads(done.read_text() or "null")
        log(f"== {name}: 시작")
        t = time.time()
        out = fn()
        done.write_text(json.dumps(out, ensure_ascii=False, default=str))
        log(f"== {name}: 끝 ({(time.time() - t) / 60:.1f}분)")
        return out
    return deco


def data_spec(root, srcs, mix):
    tokmeta = json.loads((root / "state/tokenize.done").read_text())
    return {"groups": {g: [str(root / "tok" / f"{n}.train.bin") for n, s in srcs.items() if s["group"] == g]
                       for g in mix},
            "weights": mix,
            "val": {g: (str(root / "tok" / f"val_{g}.bin"), tokmeta[f"val_{g}"]) for g in mix}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    cfg["shareable_dir"] = str((ROOT / os.path.expanduser(cfg["shareable_dir"])).resolve())
    root = Path(os.path.expanduser(cfg["data_root"]))
    for d in ("state", "logs", "raw", "clean", "tok", "runs/r1", "bench"):
        (root / d).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    state, log = root / "state", Log(root / "logs" / "r1.log")
    log(f"설정: {json.dumps(cfg, ensure_ascii=False)}")
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used",
                            "--format=csv,noheader", "-l", "60"], stdout=open(root / "logs/gpu.csv", "a"))
    srcs = cfg["sources"]

    @stage(state, "bench", log)
    def _():
        out = {}
        for v in ("A0", "A3"):
            r = subprocess.run([os.path.basename(sys.executable), "-m", "arbor.bench.train_throughput",
                                "--size", "M", "--variants", v,
                                "--micro", str(cfg["bench_micro"]), "--compile", "--steps", "14",
                                "--out", str(root / "bench/throughput.jsonl")], capture_output=True, text=True,
                               executable=sys.executable, cwd=ROOT,
                               env={**os.environ,
                                    "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
                                    "PATH": os.pathsep.join(filter(None, [os.path.dirname(sys.executable), os.environ.get("PATH")]))})
            lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
            out[v] = json.loads(lines[-1]) if lines else {"error": r.stderr[-500:]}
            log(f"[bench] M-{v}: {out[v]}")
        return out

    from arbor.data.fetch import fetch_source

    @stage(state, "fetch", log)
    def _():
        return {n: fetch_source(n, s, root / "raw", log) for n, s in srcs.items()}

    from arbor.data.clean import run_clean
    from arbor.data.leakage import collect_eval_ngrams

    @stage(state, "clean", log)
    def _():
        hashes, lstats = collect_eval_ngrams(cfg["benchmarks"], cfg["shareable_dir"], log)
        log(f"[leak] 평가 13-gram {len(hashes):,}개")
        split = {n: {"val": s["val"], "mock": s["mock"]} for n, s in srcs.items()}
        cstats = run_clean(list(srcs), root / "raw", root / "clean", hashes, split, cfg["workers"], log)
        return {"leak_sources": lstats, "clean": cstats}

    from arbor.data.bpe import train_bpe

    @stage(state, "bpe", log)
    def _():
        mix = cfg["mix"]
        fw = []
        for g, w in mix.items():
            members = [n for n, s in srcs.items() if s["group"] == g]
            sizes = [json.loads((root / "state/clean.done").read_text())["clean"][n]["bytes"]["train"] for n in members]
            fw += [(root / "clean" / f"{n}.train.jsonl", w * sz / sum(sizes)) for n, sz in zip(members, sizes)]
        tok = train_bpe(fw, float(cfg["bpe"]["sample_bytes"]), root / "tok/tokenizer.json", cfg["bpe"]["vocab"])
        return {"vocab": tok.get_vocab_size()}

    from tokenizers import Tokenizer
    from arbor.data.memmap import tokenize_file

    @stage(state, "tokenize", log)
    def _():
        tok = Tokenizer.from_file(str(root / "tok/tokenizer.json"))
        out = {}
        for n in srcs:
            out[n] = tokenize_file(tok, [root / "clean" / f"{n}.train.jsonl"], root / "tok" / f"{n}.train.bin")
            log(f"[tok] {n}.train: {out[n]}")
        for g in cfg["mix"]:
            members = [root / "clean" / f"{n}.val.jsonl" for n, s in srcs.items() if s["group"] == g]
            out[f"val_{g}"] = tokenize_file(tok, members, root / "tok" / f"val_{g}.bin")
            log(f"[tok] val_{g}: {out[f'val_{g}']}")
        for g, w in cfg["mix"].items():
            have = sum(out[n]["tokens"] for n, s in srcs.items() if s["group"] == g)
            for size, need in (("S", 6e8), ("M", 2e9)):
                log(f"[tok] {g}: 학습 토큰 {have:,}, {size} 필요 {w * need:,.0f} → 약 {w * need / have:.2f}회 반복")
        return out

    data = data_spec(root, srcs, cfg["mix"])
    from arbor.train.pretrain import train
    r1 = cfg["r1"]
    common = dict(data=data, micro=r1["micro"], tokens_per_step=r1["tokens_per_step"], log=log)

    def run(name, **kw):
        try:
            return train(root / "runs/r1" / name, **common, **kw)
        except Exception as e:
            log(f"[r1] {name} 실패: {type(e).__name__}: {e}")
            import torch
            torch.cuda.empty_cache()
            return {"name": name, "error": f"{type(e).__name__}: {e}"}

    p = r1["pilot"]

    @stage(state, "pilot", log)
    def _():
        res = {rnn: run(f"pilot_{p['variant']}_{rnn}", size=r1["size"], variant=p["variant"], tokens=float(p["tokens"]),
                        lr=p["lr"], seed=p["seed"], cfg_overrides={"rnn": rnn}) for rnn in p["rnns"]}
        ok = {k: v for k, v in res.items() if "bpb" in v}
        res["chosen"] = min(ok, key=lambda k: ok[k]["bpb"]["judge"]) if ok else p["rnns"][0]
        log(f"[r1] 혼합층 선택: {res['chosen']} ({ {k: v.get('bpb', {}).get('judge') for k, v in ok.items()} })")
        return res

    rnn = json.loads((state / "pilot.done").read_text())["chosen"]
    sw = r1["lr_sweep"]

    @stage(state, "lr_sweep", log)
    def _():
        res, best = {}, {}
        for v in sw["variants"]:
            for lr in sw["lrs"]:
                ov = {} if v == "A0" else {"rnn": rnn}
                res[f"{v}_lr{lr}"] = run(f"lr_{v}_{lr}", size=r1["size"], variant=v, tokens=float(sw["tokens"]),
                                         lr=lr, seed=sw["seed"], cfg_overrides=ov)
            ok = {lr: res[f"{v}_lr{lr}"]["bpb"]["judge"] for lr in sw["lrs"] if "bpb" in res[f"{v}_lr{lr}"]}
            best[v] = min(ok, key=ok.get) if ok else sw["lrs"][1]
            log(f"[r1] {v} 학습률 선택: {best[v]} ({ok})")
        res["best"] = best
        return res

    best_lr = json.loads((state / "lr_sweep.done").read_text())["best"]
    for d in Path(root / "runs/r1").glob("lr_*"):
        (d / "hash.pt").unlink(missing_ok=True)
    full = r1["full"]
    results = {}
    for seed in full["seeds"]:
        for v in full["variants"]:
            lr = best_lr["A0"] if v == "A0" else best_lr["A3"]
            ov = {} if v == "A0" else {"rnn": rnn}
            results[f"{v}_s{seed}"] = run(f"full_{v}_s{seed}", size=r1["size"], variant=v, tokens=float(full["tokens"]),
                                          lr=lr, seed=seed, cfg_overrides=ov)
    write_summary(root, state, results, rnn, best_lr, log)
    mon.terminate()
    log("== 전체 끝")


def write_summary(root, state, results, rnn, best_lr, log):
    rows = ["| 실행 | 판정 bpb (ko·en 60:40) | ko | en | code | 학습 tok/s | 시간(h) | 학습 FLOP |",
            "|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        if "bpb" not in r:
            rows.append(f"| {k} | 실패: {r.get('error', '')[:60]} | | | | | | |")
            continue
        b = r["bpb"]
        rows.append(f"| {k} | {b['judge']:.4f} | {b['ko']:.4f} | {b['en']:.4f} | {b.get('code', float('nan')):.4f} | "
                    f"{r['tok_per_s']:,} | {r['hours']} | {r['train_flop']:.2e} |")
    bench = json.loads((state / "bench.done").read_text())
    pilot = json.loads((state / "pilot.done").read_text())
    sweep = json.loads((state / "lr_sweep.done").read_text())
    pick = lambda d: {k: v.get("bpb", {}).get("judge") for k, v in d.items() if isinstance(v, dict) and "name" in v}
    md = ["# R1 결과 (자동 생성) `[실측]`", "", f"- 혼합층 파일럿 (판정 bpb): {pick(pilot)}",
          f"- 학습률 스윕 (판정 bpb): {pick(sweep)}", f"- 선택: 혼합층 {rnn}, 학습률 {best_lr}", "",
          "## S 사다리", *rows, "",
          f"## GPU 83°C 초과 3분 이상 구간: {gpu_hot_spells(root / 'logs/gpu.csv') or '없음'}", "",
          "## 이 서버의 M 처리량 (bench 단계)", "```", json.dumps(bench, ensure_ascii=False, indent=1), "```"]
    (root / "runs/r1/summary.md").write_text("\n".join(md), encoding="utf-8")
    (root / "runs/r1/summary.json").write_text(json.dumps(dict(rnn=rnn, lr=best_lr, results=results), indent=1))
    log("[r1] 요약: runs/r1/summary.md")


if __name__ == "__main__":
    main()
