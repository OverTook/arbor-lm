import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from arbor.data.clean import ngram_hashes, words


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _records(path):
    p = str(path)
    if p.endswith(".parquet"):
        yield from pq.read_table(p).to_pylist()
    elif p.endswith(".jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif p.endswith(".json"):
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else [data])
    elif p.endswith((".csv", ".tsv")):
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        yield from csv.DictReader(io.StringIO(text), delimiter="\t" if p.endswith(".tsv") else ",")


def benchmark_texts(spec):
    files = HfApi().list_repo_files(spec["repo"], repo_type="dataset")
    inc, exc = spec.get("include", []), spec.get("exclude", [])
    for f in files:
        if not f.endswith((".parquet", ".jsonl", ".json", ".csv", ".tsv")):
            continue
        if inc and not any(s in f for s in inc):
            continue
        if any(s in f for s in exc):
            continue
        for rec in _records(hf_hub_download(spec["repo"], f, repo_type="dataset")):
            yield " ".join(_strings(rec))


def pdf_text(path):
    from pypdf import PdfReader
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)


def collect_eval_ngrams(benchmarks, shareable_dir, log=print):
    hashes, stats = set(), {}
    for name, spec in benchmarks.items():
        before, n = len(hashes), 0
        try:
            for t in benchmark_texts(spec):
                hashes |= ngram_hashes(words(t))
                n += 1
            stats[name] = dict(records=n, new_ngrams=len(hashes) - before)
        except Exception as e:
            stats[name] = dict(error=f"{type(e).__name__}: {e}"[:300])
        log(f"[leak] {name}: {stats[name]}")
    for p in sorted(Path(shareable_dir).glob("*")):
        if p.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        before = len(hashes)
        text = pdf_text(p) if p.suffix.lower() == ".pdf" else p.read_text(encoding="utf-8")
        hashes |= ngram_hashes(words(text))
        stats[f"shareable:{p.name}"] = dict(chars=len(text), new_ngrams=len(hashes) - before)
        log(f"[leak] {p.name}: {stats[f'shareable:{p.name}']}")
    return hashes, stats
