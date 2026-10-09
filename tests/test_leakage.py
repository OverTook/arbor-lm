import json

from arbor.data.clean import NGRAM, ngram_hashes, run_clean, words


def test_ngram_is_13_words_and_case_punct_insensitive():
    assert NGRAM == 13
    a = ngram_hashes(words("One two three four five six seven eight nine ten eleven twelve thirteen"))
    b = ngram_hashes(words("one, two three four five six seven eight nine ten eleven twelve thirteen!"))
    assert len(a) == 1 and a == b
    assert not ngram_hashes(words("하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 열하나 열둘"))


def test_doc_with_eval_13gram_removed_12gram_kept(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    ev_text = " ".join(f"e{i}" for i in range(20))
    leak = "앞 " + " ".join(f"e{i}" for i in range(3, 16)) + " 뒤"
    near = " ".join(f"x{i}" for i in range(60)) + " 앞 " + " ".join(f"e{i}" for i in range(3, 15)) + " 뒤"
    with open(raw / "s.jsonl", "w", encoding="utf-8") as f:
        for t in (leak, near):
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    st = run_clean(["s"], raw, out, ngram_hashes(words(ev_text)), {"s": {"val": 0, "mock": 0}}, 1, lambda *_: None)
    assert st["s"]["leak"] == 1 and st["s"]["docs"]["train"] == 1
    assert "e14 뒤" in (out / "s.train.jsonl").read_text(encoding="utf-8")
