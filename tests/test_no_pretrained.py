import re
from pathlib import Path

FORBIDDEN = [r"from_pretrained\(", r"\bAutoModel", r"\bAutoTokenizer", r"tiktoken", r"sentencepiece",
             r"\bopenai\b", r"\banthropic\b", r"api\.openai\.com", r"gensim", r"fasttext"]


def test_no_pretrained_or_llm_api_in_code():
    hits = []
    for p in Path(__file__).resolve().parents[1].joinpath("arbor").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        hits += [f"{p.name}: {pat}" for pat in FORBIDDEN if re.search(pat, text)]
    assert not hits, hits


def test_tokenizer_is_trained_from_scratch():
    from arbor.data.bpe import new_tokenizer
    tok = new_tokenizer()
    assert tok.get_vocab_size() == 0
