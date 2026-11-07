#!/usr/bin/env python3
"""Generate a small BPE model in all three supported formats into
tests/fixtures/ so the loaders can be compared against each other:

  mini_encoder.json / mini_vocab.bpe   (GPT-2 format)
  mini_tokenizer.json                  (HF format)
  mini.tiktoken                        (tiktoken format)

One underlying model: 256 byte tokens (id == byte value), merges
a+b->ab(256), ab+c->abc(257), " "+x->" x"(258), t+h->th(259),
special <|eot|> = 260, non-special added token "hello" = 261."""
import base64
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "tests", "fixtures")

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()

def remap(data: bytes) -> str:
    return "".join(B2U[b] for b in data)

MERGES = [("a", "b"), ("ab", "c"), (remap(b" "), "x"), ("t", "h")]
EXTRA_IDS = {"ab": 256, "abc": 257, remap(b" ") + "x": 258, "th": 259}
SPECIAL = ("<|eot|>", 260)
ADDED_NONSPECIAL = ("hello", 261)

def write_gpt2():
    enc = {remap(bytes([b])): b for b in range(256)}
    enc.update(EXTRA_IDS)
    enc[SPECIAL[0]] = SPECIAL[1]
    with open(os.path.join(OUT, "mini_encoder.json"), "w") as f:
        json.dump(enc, f)
    with open(os.path.join(OUT, "mini_vocab.bpe"), "w") as f:
        f.write("#version: 0.2\n")
        for l, r in MERGES:
            f.write("%s %s\n" % (l, r))

def write_tokenizer_json():
    vocab = {remap(bytes([b])): b for b in range(256)}
    vocab.update(EXTRA_IDS)
    tj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": SPECIAL[1], "content": SPECIAL[0], "special": True},
            {"id": ADDED_NONSPECIAL[1], "content": ADDED_NONSPECIAL[0],
             "special": False},
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": True, "use_regex": True,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False,
            "vocab": vocab,
            "merges": [[l, r] for l, r in MERGES],
        },
    }
    with open(os.path.join(OUT, "mini_tokenizer.json"), "w") as f:
        json.dump(tj, f)

def write_tiktoken():

    with open(os.path.join(OUT, "mini.tiktoken"), "w") as f:
        for b in range(256):
            f.write(base64.b64encode(bytes([b])).decode() + "\n")
        for tok in ["ab", "abc", " x", "th"]:
            f.write(base64.b64encode(tok.encode("utf-8")).decode() + "\n")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write_gpt2()
    write_tokenizer_json()
    write_tiktoken()
    print("wrote mini model fixtures to", OUT)
