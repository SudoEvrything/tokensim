#!/usr/bin/env python3
"""The simple-call benchmark: every engine called exactly the way a
real program calls it - construct the tokenizer, encode the document,
nothing else. No custom drivers, no special-cased APIs.

Corpus: testdata/owt16m.txt (16 MB), one document, best of 3 reps,
timed from the encode call only. The id counts are printed so the
cross-engine correctness is visible next to the rates.

Engines (all default settings, as shipped):
  tiktoken   enc.encode_ordinary(text)          single-threaded
  hf         tok.encode(text)                   single-threaded (Rust)
  gigatoken  tk.encode(text)                    rayon threads
  tokensim   tp.encode(data)                    auto threads (SIMD/C)

usage: bench_simple_calls.py [corpus]
"""

import os
import sys
import time

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "testdata/owt16m.txt"
REPS = 3

data = open(CORPUS, "rb").read()
text = data.decode("utf-8", errors="replace")
mb = len(data) / 1e6

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "bindings", "python"))


def bench(name, construct, encode, count):
    tok = construct()
    ids = encode(tok)
    want = count(ids)
    cold = 1e18
    best = 1e18
    for r in range(REPS):
        t0 = time.perf_counter()
        got = encode(tok)
        dt = time.perf_counter() - t0
        if r == 0:
            cold = dt
        best = min(best, dt)
        assert count(got) == want, f"{name}: count mismatch"
    print(f"{name}: cold {cold:.2f}s = {mb / cold:.1f} MB/s | "
          f"warm best {best:.2f}s = {mb / best:.1f} MB/s  ({want} ids)")


def run_tiktoken():
    import tiktoken
    for model in ["o200k_base", "cl100k_base"]:
        bench(
            f"tiktoken {model}",
            lambda: tiktoken.get_encoding(model),
            lambda enc: enc.encode_ordinary(text),
            lambda r: len(r),
        )


def run_hf():
    from tokenizers import Tokenizer
    for name, path in [
        ("hf deepseek",
         "models/reference/deepseek_tokenizer.json"),
        ("hf qwen3", "models/reference/qwen3_tokenizer.json"),
    ]:
        bench(
            name,
            lambda: Tokenizer.from_file(path),
            lambda tok: tok.encode(text, add_special_tokens=False).ids,
            lambda r: len(r),
        )


def run_giga():
    from gigatoken import Tokenizer as Giga
    for model in ["o200k_base", "cl100k_base"]:
        bench(
            f"gigatoken {model}",
            lambda: Giga.from_tiktoken("models/" + model + ".tiktoken",
                                       model.replace("_base", "")),
            lambda tk: tk.encode(text),
            lambda r: len(r),
        )


def run_tokensim():
    import tokensim
    for model, fmt in [
        ("cl100k_base.tiktoken", None),
        ("o200k_base.tiktoken", None),
    ]:
        path = "models/" + model
        bench(
            f"tokensim {model}",
            lambda: tokensim.tokenizer(path),
            lambda tp: tp.encode(data),
            lambda r: len(r) // 4,
        )


if __name__ == "__main__":
    print(f"corpus: {CORPUS} ({mb:.1f} MB, one document, best of "
          f"{REPS})")
    run_tokensim()
    run_tiktoken()
    run_giga()
    run_hf()
