import json
import os
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def list_shards(repo, prefix):
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def fetch_source(name, spec, out_dir, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, done = out_dir / f"{name}.jsonl", out_dir / f"{name}.done"
    if done.exists():
        log(f"[fetch] {name}: 이미 끝남")
        return json.loads(done.read_text())
    shards = list_shards(spec["repo"], spec["prefix"])
    if not shards:
        raise RuntimeError(f"{name}: {spec['repo']}/{spec['prefix']} 에 parquet 없음")
    target, field = float(spec["target_bytes"]), spec["field"]
    nbytes = ndocs = 0
    used = []
    with open(out, "w", encoding="utf-8") as f:
        for shard in shards:
            path = hf_hub_download(spec["repo"], shard, repo_type="dataset")
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                for text in pf.read_row_group(rg, columns=[field]).column(field).to_pylist():
                    if not text:
                        continue
                    text = unicodedata.normalize("NFC", text)
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    nbytes += len(text.encode("utf-8"))
                    ndocs += 1
                if nbytes >= target:
                    break
            used.append(shard)
            os.remove(os.path.realpath(path))
            log(f"[fetch] {name}: {shard} 끝, 누적 {nbytes / 1e9:.2f}GB / 목표 {target / 1e9:.2f}GB, 문서 {ndocs:,}")
            if nbytes >= target:
                break
    meta = dict(source=name, repo=spec["repo"], shards=used, docs=ndocs, bytes=nbytes,
                reached_target=nbytes >= target)
    done.write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]
