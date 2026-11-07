#!/usr/bin/env python3
"""gigatoken arm of the ladder: the INSTALLED wheel's production API.

usage: gt_drive2.py KIND PATH CORPUS MIB REPS DOC_MIB [SCHEME]

KIND = tiktoken | json | spm.  Loads the tokenizer once, cuts the slice
into doc_mib documents outside the timed region, encodes them reps times,
and prints one bench_encode-shaped line per rep plus an `IDS` line for
the exactness check.  The wheel is the id-exact instrument (verified
against our CLI); the local source build under /tmp/gigatoken_src is
NOT (its Python path drops 6.6% of the ids).
"""
import sys
import time

import awkward as ak
from gigatoken import BytesSource, Tokenizer

kind, path, corpus, mib, reps, doc_mib = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)
scheme = sys.argv[7] if len(sys.argv) > 7 else ""

if kind == "tiktoken":
    tok = Tokenizer.from_tiktoken(path, pretokenizer=scheme or None)
elif kind == "spm":
    tok = Tokenizer.from_sentencepiece(path)
else:
    tok = Tokenizer.from_json(open(path, "rb").read())

with open(corpus, "rb") as fh:
    raw = fh.read(mib * 1024 * 1024)
step = (doc_mib if doc_mib > 0 else mib) * 1024 * 1024
docs = [raw[i : i + step] for i in range(0, len(raw), step)]
total = sum(len(d) for d in docs) / 1048576.0
prev = None
for i in range(reps):
    t0 = time.perf_counter()
    out = tok.encode_batch(BytesSource(docs))
    dt = time.perf_counter() - t0
    n = int(ak.count(out))
    print(f"rep {i}  {dt:8.3f} s  {total / dt:9.1f} MiB/s  ({n} ids)", flush=True)
    if prev is not None and n != prev:
        print("ID COUNT MISMATCH across reps", flush=True)
        sys.exit(1)
    prev = n
print(f"IDS {prev}", flush=True)
