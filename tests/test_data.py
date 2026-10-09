import json

import numpy as np

from arbor.data.bpe import EOS, SPECIAL, new_tokenizer
from arbor.data.clean import near_dup_mask, minhash, ngram_hashes, run_clean, words
from arbor.data.memmap import load_bin, tokenize_file


def _write(path, texts):
    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")


def test_bpe_splits_digits_individually_and_reserves_specials(tmp_path):
    from tokenizers import trainers, pre_tokenizers
    tok = new_tokenizer()
    tok.train_from_iterator(["가격은 32000000원 12345 abc"] * 50,
                            trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    assert len(SPECIAL) == 64 and tok.token_to_id("<|eos|>") == EOS
    ids = tok.encode("12345", add_special_tokens=False).ids
    assert len(ids) == 5 and tok.decode(ids) == "12345"


def test_tokenize_roundtrip_and_meta(tmp_path):
    from tokenizers import trainers, pre_tokenizers
    tok = new_tokenizer()
    tok.train_from_iterator(["안녕하세요 hello"] * 20, trainers.BpeTrainer(
        vocab_size=300, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    src = tmp_path / "a.jsonl"
    _write(src, ["안녕하세요", "hello 1"])
    meta = tokenize_file(tok, [src], tmp_path / "a.bin")
    a = load_bin(tmp_path / "a.bin")
    assert meta["docs"] == 2 and meta["bytes"] == len("안녕하세요".encode()) + len(b"hello 1")
    assert len(a) == meta["tokens"] and (a == EOS).sum() == 2
    first = a[: list(a).index(EOS)]
    assert tok.decode(first.tolist()) == "안녕하세요"


def test_near_dup_detects_small_edit_not_different_doc():
    base = " ".join(f"단어{i}" for i in range(300))
    sigs = np.stack([minhash(words(base)), minhash(words(base + " 끝")), minhash(words(base[::-1]))])
    dup = near_dup_mask(sigs)
    assert list(dup) == [False, True, False]


def test_clean_removes_leaks_dups_and_splits_deterministically(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    leak_src = "the quick brown fox jumps over the lazy dog near the river bank today ok"
    docs = [f"문서 {i} " + " ".join(f"w{i}_{j}" for j in range(40)) for i in range(200)]
    _write(raw / "a.jsonl", docs + [docs[3], "prefix " + leak_src + " suffix"])
    _write(raw / "b.jsonl", [docs[5]])
    ev = ngram_hashes(words(leak_src))
    split = {"a": {"val": 0.1, "mock": 0.1}, "b": {"val": 0.0, "mock": 0.0}}
    st = run_clean(["a", "b"], raw, out, ev, split, workers=2, log=lambda *_: None)
    assert st["a"]["exact_dup"] == 1 and st["a"]["leak"] == 1 and st["b"]["exact_dup"] == 1
    kept = sum(st["a"]["docs"].values())
    assert kept == 200 and 0 < st["a"]["docs"]["val"] < 50 and 0 < st["a"]["docs"]["mock"] < 50
    first = (out / "a.val.jsonl").read_text()
    run_clean(["a", "b"], raw, out, ev, split, workers=2, log=lambda *_: None)
    assert (out / "a.val.jsonl").read_text() == first
