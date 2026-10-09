import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from arbor.data.fetch import iter_jsonl

SPECIAL = ["<|eos|>", "<|pad|>", "<사용자>", "<모델>", "<근거>", "</근거>", "<기억>", "</기억>",
           "<결과>", "</결과>", "<web>", "</web>", "<fetch>", "</fetch>", "<run>", "</run>",
           "<calc>", "</calc>", "<doc>", "</doc>", "<ask>", "</ask>", "[메타]"]
SPECIAL += [f"<|reserved_{i}|>" for i in range(64 - len(SPECIAL))]
EOS = 0


def new_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def sample_texts(files_weights, total_bytes, seed):
    rng = random.Random(seed)
    for path, w in files_weights:
        budget, got = total_bytes * w, 0
        for t in iter_jsonl(path):
            if rng.random() < 0.5:
                continue
            yield t
            got += len(t.encode())
            if got >= budget:
                break


def train_bpe(files_weights, total_bytes, out_path, vocab=32000, seed=0):
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(sample_texts(files_weights, total_bytes, seed), trainer=trainer)
    assert tok.token_to_id("<|eos|>") == EOS
    tok.save(str(out_path))
    return tok
