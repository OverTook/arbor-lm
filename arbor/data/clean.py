import hashlib
import json
import multiprocessing as mp
import re
import zlib
from pathlib import Path

import numpy as np

WORD = re.compile(r"\w+")
NGRAM = 13
SHINGLE, NPERM, BANDS = 5, 128, 16
_P = (1 << 31) - 1
_rng = np.random.default_rng(12345)
_A = _rng.integers(1, _P, NPERM, dtype=np.uint64)
_B = _rng.integers(0, _P, NPERM, dtype=np.uint64)

_EVAL = None


def words(text):
    return WORD.findall(text.lower())


def ngram_hashes(ws, n=NGRAM):
    return {hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)}


def minhash(ws):
    sh = [" ".join(ws[i:i + SHINGLE]) for i in range(max(1, len(ws) - SHINGLE + 1))][:5000]
    h = np.fromiter((zlib.crc32(s.encode()) for s in sh), dtype=np.uint64, count=len(sh))
    return ((_A[:, None] * h[None, :] + _B[:, None]) % _P).min(1).astype(np.uint32)


def doc_key(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def _process(lines):
    keys, sigs, leak = [], [], []
    for line in lines:
        text = json.loads(line)["text"]
        ws = words(text)
        keys.append(doc_key(text))
        sigs.append(minhash(ws))
        leak.append(bool(_EVAL) and any(hash(tuple(ws[i:i + NGRAM])) in _EVAL for i in range(len(ws) - NGRAM + 1)))
    return np.array(keys, dtype=np.uint64), np.stack(sigs), np.array(leak)


def _chunks(path, size=2000):
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) == size:
                yield buf
                buf = []
    if buf:
        yield buf


def near_dup_mask(sigs):
    n = len(sigs)
    dup = np.zeros(n, dtype=bool)
    r = NPERM // BANDS
    for b in range(BANDS):
        band = np.ascontiguousarray(sigs[:, b * r:(b + 1) * r]).view(np.dtype((np.void, 4 * r))).ravel()
        _, first, inv = np.unique(band, return_index=True, return_inverse=True)
        dup |= first[inv] != np.arange(n)
    return dup


def run_clean(sources, raw_dir, out_dir, eval_hashes, split, workers, log=print):
    global _EVAL
    _EVAL = eval_hashes
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys, sigs, leak, owner = [], [], [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for si, name in enumerate(sources):
            for k, s, l in pool.imap(_process, _chunks(raw_dir / f"{name}.jsonl"), chunksize=1):
                keys.append(k); sigs.append(s); leak.append(l); owner.append(np.full(len(k), si))
            log(f"[clean] {name}: 서명 계산 끝")
    keys, sigs, leak, owner = map(np.concatenate, (keys, sigs, leak, owner))
    _, first = np.unique(keys, return_index=True)
    exact = np.ones(len(keys), dtype=bool)
    exact[first] = False
    near = near_dup_mask(sigs) & ~exact
    keep = ~(exact | near | leak)
    u = (keys >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    stats = {}
    offset = 0
    for si, name in enumerate(sources):
        m = owner == si
        idx = np.nonzero(m)[0]
        p_val, p_mock = split[name]["val"], split[name]["mock"]
        files = {s: open(out_dir / f"{name}.{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "mock")}
        nb = {"train": 0, "val": 0, "mock": 0}
        nd = {"train": 0, "val": 0, "mock": 0}
        with open(raw_dir / f"{name}.jsonl", encoding="utf-8") as f:
            for j, line in enumerate(f):
                g = offset + j
                if not keep[g]:
                    continue
                s = "val" if u[g] < p_val else "mock" if u[g] < p_val + p_mock else "train"
                files[s].write(line)
                nb[s] += len(json.loads(line)["text"].encode())
                nd[s] += 1
        for fh in files.values():
            fh.close()
        offset += len(idx)
        stats[name] = dict(docs_in=int(m.sum()), exact_dup=int((exact & m).sum()), near_dup=int((near & m).sum()),
                           leak=int((leak & m & ~exact & ~near).sum()), docs=nd, bytes=nb)
        log(f"[clean] {name}: {json.dumps(stats[name], ensure_ascii=False)}")
    (out_dir / "clean_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats
