"""Python-binding arm of the wrapper ladder.

usage: bench_python_arm.py MODEL FORMAT SCHEME THREADS FILE [REPS] [API]
API is `reuse` (the encode_into path, single thread), `batch`
(encode_batch_raw, zero-copy ids) or `parallel` (encode_parallel).
"""
import sys
import time

import tsimd

def open_model(path, fmt, scheme):
    if hasattr(tsimd, "load_model"):
        return tsimd.load_model(path, fmt, scheme)
    if fmt == "tiktoken":
        return tsimd.load(path, scheme=scheme)
    return tsimd.load_tokenizer_json(path)

def main():
    model, fmt, scheme, threads, path = sys.argv[1:6]
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    api = sys.argv[7] if len(sys.argv) > 7 else "batch"
    tok = open_model(model, fmt, scheme)
    with open(path, "rb") as f:
        data = f.read()
    bound = tok.encode_bound(data) if hasattr(tok, "encode_bound") else len(data) + 16
    outbuf = bytearray(bound * 4)
    best = 0.0
    n = 0
    for r in range(reps):
        t0 = time.perf_counter()
        ids = None
        if api == "out":
            n = tok.encode_out(data, outbuf, threads=int(threads))
        elif api == "raw":
            ids = tok.encode_raw(data, threads=int(threads))
            n = len(ids) // 4
        elif api == "reuse":
            ids = tok.encode_reuse(data)
            n = len(ids)
        elif api == "parallel":
            ids = tok.encode_parallel(data, threads=int(threads))
            n = len(ids)
        else:
            ids, _offsets = tok.encode_batch_raw([data], threads=int(threads))
            n = len(ids) // 4
        el = time.perf_counter() - t0
        mibs = (len(data) / 1048576.0) / el
        best = max(best, mibs)
        print("REP python-%s %d %.1f %d" % (api, r, mibs, n), flush=True)
        del ids
    print("RESULT python-%s %s %s %s %.1f %d" % (api, scheme, path, threads, best, n))

if __name__ == "__main__":
    main()
