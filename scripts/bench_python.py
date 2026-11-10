#!/usr/bin/env python3
"""Python-wrapper throughput bench.

usage: bench_python.py KIND MODEL FORMAT SCHEME CORPUS [REPS]
  KIND: tsimd | tsimd_par | tsimd_into | sentencepiece | tiktoken
  FORMAT (the tsimd arms): tokenizer_json | sentencepiece | tiktoken
Prints one line per rep: MiB/s, count, then best (the last-rep rule
excluded; best over the warm reps).
  tsimd      = the sequential bytes-returning encode
  tsimd_par  = encode_threads(6) -> bytes
  tsimd_into = encode_threads_into(6) -> the caller's bytearray
"""
import sys
import time
import array
import os.path

def array_frombytes(raw):
    a = array.array("I")
    a.frombytes(raw)
    return a

def main():
    kind, model, fmt, scheme, path = sys.argv[1:6]
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    if kind == "sentencepiece":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model)
        enc = sp.encode
    elif kind == "tiktoken":
        import tiktoken
        tk = tiktoken.get_encoding(model)
        enc = tk.encode
    else:
        sys.path.insert(0, "bindings/python")
        import tsimd
        tok = tsimd.load(model, fmt, scheme if scheme != "auto" else None)
        if kind == "tsimd_par":
            def enc(b):
                return tok.encode_threads(b, threads)
        elif kind == "tsimd_into":
            bound = tok.encode_bound(os.path.getsize(path))
            if bound > (1 << 30):
                print("into skipped: bound %.1f GiB > 1 GiB cap; "
                      "the par protocol runs" % (bound / 2**30))
                kind = "tsimd_par"

                def enc(b):
                    return tok.encode_threads(b, threads)
            else:
                out = bytearray((bound + 3) & ~3)
                state = {"n": 0}

                def enc(b):
                    state["n"] = tok.encode_threads_into(b, out, threads)
                    return out
        else:
            def enc(b):
                return array_frombytes(tok.encode_bytes(b))
    with open(path, "rb") as f:
        data = f.read()
    mib = len(data) / 1048576.0
    best = 0.0
    for r in range(reps):
        t0 = time.perf_counter()
        ids = enc(data)
        dt = time.perf_counter() - t0
        count = state["n"] if kind == "tsimd_into" \
            else (len(ids) // 4 if kind == "tsimd_par" else len(ids))
        rate = mib / dt if dt > 0 else 0.0
        print("rep=%d %.1fMiB/s count=%d" % (r, rate, count))
        if r > 0 and rate > best:
            best = rate
    if reps > 1:
        print("best=%.1fMiB/s" % best)

if __name__ == "__main__":
    main()
