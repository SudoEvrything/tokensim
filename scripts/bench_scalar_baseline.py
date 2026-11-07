import sys
import time

import tiktoken
from tokenizers import Tokenizer

CORPUS = "testdata/owt16m.txt"
data = open(CORPUS, "rb").read()
text = data.decode("utf-8", errors="replace")
mb = len(data) / 1e6


def best_of(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def tiktoken_o200k():
    enc = tiktoken.get_encoding("o200k_base")
    return best_of(lambda: enc.encode_ordinary(text))


def tiktoken_cl100k():
    enc = tiktoken.get_encoding("cl100k_base")
    return best_of(lambda: enc.encode_ordinary(text))


def hf_deepseek():
    tok = Tokenizer.from_file(
        "models/reference/deepseek_tokenizer.json")
    return best_of(lambda: tok.encode(text, add_special_tokens=False))


for name, fn in [("tiktoken o200k", tiktoken_o200k),
                 ("tiktoken cl100k", tiktoken_cl100k),
                 ("hf deepseek", hf_deepseek)]:
    dt = fn()
    print(f"{name}: {dt:.2f}s  {mb / dt:.1f} MB/s single-thread")
