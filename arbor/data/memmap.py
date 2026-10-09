import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from arbor.data.bpe import EOS
from arbor.data.fetch import iter_jsonl


def tokenize_file(tok: Tokenizer, srcs, dst, batch=4096):
    dst = Path(dst)
    ntok = nbytes = ndocs = 0
    with open(dst, "wb") as f:
        buf = []

        def flush():
            nonlocal ntok
            for enc in tok.encode_batch(buf, add_special_tokens=False):
                ids = np.asarray(enc.ids + [EOS], dtype=np.uint16)
                ids.tofile(f)
                ntok += len(ids)

        for src in srcs:
            for t in iter_jsonl(src):
                buf.append(t)
                nbytes += len(t.encode())
                ndocs += 1
                if len(buf) == batch:
                    flush()
                    buf = []
        if buf:
            flush()
    meta = dict(tokens=ntok, bytes=nbytes, docs=ndocs)
    Path(str(dst) + ".meta.json").write_text(json.dumps(meta))
    return meta


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")
