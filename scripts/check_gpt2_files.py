#!/usr/bin/env python3
"""Data-level cross-check: convert GPT-2 encoder.json + vocab.bpe into
tiktoken format and verify the result matches the official
r50k_base.tiktoken (modulo the excluded <|endoftext|> special and the
optional rank suffix). Validates the byte<->unicode alphabet inverse.

usage: check_gpt2_files.py ENCODER_JSON VOCAB_BPE [OFFICIAL_TIKTOKEN]"""
import base64
import json
import sys

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
    return {b: chr(c) for b, c in zip(bs, cs)}

def main():
    encoder_path, merges_path = sys.argv[1], sys.argv[2]
    official_path = sys.argv[3] if len(sys.argv) > 3 else None

    b2u = bytes_to_unicode()
    u2b = {c: b for b, c in b2u.items()}
    unremap = lambda s: bytes(u2b[c] for c in s)

    enc = json.load(open(encoder_path))
    merges = [l.split() for l in open(merges_path).read().splitlines()
              if l and not l.startswith("#")]

    lines = [None] * len(enc)
    for tok, i in enc.items():
        lines[i] = base64.b64encode(unremap(tok)).decode()
    print("vocab entries:", len(lines))

    for l, r in merges:
        unremap(l)
        unremap(r)

    if official_path:
        official = [l.split(" ")[0]
                    for l in open(official_path).read().splitlines()]
        n = len(official)
        assert len(lines) in (n, n + 1), "vocab size mismatch"
        if lines[:n] == official:
            print("content identical to %s (%d mergeable tokens)" %
                  (official_path, n))
        else:
            for i, (a, b) in enumerate(zip(lines, official)):
                if a != b:
                    print("first diff at line", i, repr(a), repr(b))
                    break
            sys.exit(1)
    print("OK")

if __name__ == "__main__":
    main()
