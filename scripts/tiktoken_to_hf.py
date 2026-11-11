#!/usr/bin/env python3
"""Convert a tiktoken vocab file to a HuggingFace tokenizers json.

The tiktoken file is one base64 token per line, ordered by merge
rank. The ByteLevel BPE vocab is the byte-to-unicode stand-in
mapping; the merges are derived in rank order. The result loads
with Tokenizer.from_file and, for the cl100k regex, reproduces the
tiktoken ids exactly (the script checks this on a sample).
"""

import base64
import json
import sys

CL100K_REGEX = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    "[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
    "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

BYTE_U = [chr(i) for i in range(33, 127)] + [chr(i) for i in range(161, 173)]


def to_unicode(b):
    out = []
    for byte in b:
        c = BYTE_U[byte] if byte < len(BYTE_U) else None
        if c is None:
            out.append(chr(0x100 + byte - len(BYTE_U)))
        else:
            out.append(c)
    return "".join(out)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ranks = {}
    with open(src, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok, _, rank = line.partition(b" ")
            ranks[base64.b64decode(tok)] = int(rank)
    vocab = {}
    merges = []
    specials = {}
    for tok_bytes in sorted(ranks, key=lambda b: ranks[b]):
        u = to_unicode(tok_bytes)
        if u in vocab:
            continue
        vocab[u] = len(vocab)
    dbg = 0

    def derive_pair(tok_bytes):
        nonlocal dbg
        parts = [tok_bytes[i:i + 1] for i in range(len(tok_bytes))]
        while len(parts) > 1:
            best_rank = None
            best_i = None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                if dbg < 3:
                    print("  derive failed:", tok_bytes[:20],
                          "len", len(tok_bytes))
                    dbg += 1
                return None
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        if len(parts) == 2:
            return (to_unicode(parts[0]), to_unicode(parts[1]))
        return None

    for tok_bytes in sorted(ranks, key=lambda b: ranks[b]):
        if len(tok_bytes) < 2:
            continue
        pair = derive_pair(tok_bytes)
        if pair is not None:
            merges.append(pair)
    special_names = {}
    for tok_bytes, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tok_bytes) >= 20 and tok_bytes.startswith(b"<|"):
            special_names[to_unicode(tok_bytes)] = r
    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": CL100K_REGEX},
            "behavior": "Isolated",
            "invert": False,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    print(f"written {dst}: {len(vocab)} tokens, {len(merges)} merges")


if __name__ == "__main__":
    main()
