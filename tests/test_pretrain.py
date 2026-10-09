import json

import numpy as np
import pytest
import torch

from arbor.train.pretrain import train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
TINY = dict(d=64, heads=4, gla_heads=3, ffn=128, layers=2, vocab=300, hash_log_rows=8)


def _data(tmp_path):
    rng = np.random.default_rng(0)
    out = {"groups": {}, "weights": {"ko": 0.6, "en": 0.4}, "val": {}}
    for g in ("ko", "en"):
        a = rng.integers(0, 300, 20000).astype(np.uint16)
        a.tofile(tmp_path / f"{g}.bin")
        v = rng.integers(0, 300, 3000).astype(np.uint16)
        v.tofile(tmp_path / f"val_{g}.bin")
        out["groups"][g] = [str(tmp_path / f"{g}.bin")]
        out["val"][g] = (str(tmp_path / f"val_{g}.bin"), {"tokens": 3000, "bytes": 9000})
    return out


@cuda
@pytest.mark.parametrize("variant", ["A0", "A3"])
def test_resume_after_kill_matches_uninterrupted(tmp_path, variant):
    data = _data(tmp_path)
    kw = dict(size="S", variant=variant, tokens=64 * 4 * 8, lr=2e-3, seed=0, data=data, micro=4,
              tokens_per_step=64 * 4, seq=64, cfg_overrides=TINY, compile=False, evals=1, log=lambda *_: None)
    torch.backends.cudnn.deterministic = True
    a = train(tmp_path / "a", **kw)
    assert train(tmp_path / "b", **kw, stop_after=3) is None
    assert (tmp_path / "b" / "ckpt.pt").exists()
    b = train(tmp_path / "b", **kw)
    assert abs(a["bpb"]["judge"] - b["bpb"]["judge"]) < 1e-3
    assert not (tmp_path / "b" / "ckpt.pt").exists()
    assert json.loads((tmp_path / "b" / "result.json").read_text())["tokens"] == kw["tokens"]
