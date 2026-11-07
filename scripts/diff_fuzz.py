#!/usr/bin/env python3
"""Differential fuzz: compare the C tokenizer against python tiktoken on
randomized inputs. usage: diff_fuzz.py c_binary model_arg scheme [n] [format]

format: tiktoken (default) -> model_arg is the tiktoken vocab file
        gpt2 | tokenizer_json -> model_arg is that model file"""
import os
import random
import subprocess
import sys
import tiktoken

BIN, MODEL, SCHEME = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
FORMAT = sys.argv[5] if len(sys.argv) > 5 else "tiktoken"

THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 0

rng = random.Random(42)

ALPHA = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PUNCT = "!@#$%^&*()[]{};:'\",.<>/?\\|`~-_=+"
WS = " \t\n\r\v\f"
UNI = "éüñßØ你好世界中文日本語한국어ПриветмирΓειάمرحباनमस्ते🚀🎉❤️👍été“curly”‘quote’—dash…€£¥"
CONTR = "'s 't 're 've 'm 'll 'd 'S 'T".split()
SPECIAL = "<|endoftext|>"

def sample():
    parts = []
    for _ in range(rng.randint(1, 25)):
        kind = rng.random()
        if kind < 0.35:
            parts.append("".join(rng.choice(ALPHA) for _ in range(rng.randint(1, 12))))
        elif kind < 0.5:
            parts.append(rng.choice(PUNCT) * rng.randint(1, 4))
        elif kind < 0.65:
            parts.append(rng.choice(WS) * rng.randint(1, 5))
        elif kind < 0.8:
            parts.append(rng.choice(UNI) * rng.randint(1, 3))
        elif kind < 0.9:
            parts.append(rng.choice(CONTR))
        else:
            parts.append(str(rng.randint(0, 10**9)))
    s = "".join(parts)

    if rng.random() < 0.15 and FORMAT == "tiktoken":
        s = s.replace("e", SPECIAL, 1)  # will be compared with encode_ordinary
    return s

enc_name = {"gpt2": "r50k_base", "cl100k": "cl100k_base",
            "o200k": "o200k_base"}[SCHEME]
enc = tiktoken.get_encoding(enc_name)

fails = 0
for i in range(N):
    s = sample()
    data = s.encode("utf-8", errors="surrogatepass")
    path = "/tmp/tsimd_diff_input.bin"
    with open(path, "wb") as f:
        f.write(data)
    out = subprocess.run(
        [BIN, "--format", FORMAT]
        + (["--threads", str(THREADS)] if THREADS else [])
        + ([MODEL] if FORMAT != "gpt2" else [MODEL, "/tmp/gpt2/vocab.bpe"])
        + [SCHEME, path], capture_output=True, text=True)
    got = [int(x) for x in out.stdout.split()]
    want = enc.encode_ordinary(s)
    if got != want:
        fails += 1
        print("MISMATCH on sample %d (len %d):" % (i, len(data)))
        print("  got : %s..." % got[:20])
        print("  want: %s..." % want[:20])
        if fails > 3:
            break
print("%s: %d samples, %d failures" % (SCHEME, N, fails))
sys.exit(1 if fails else 0)
