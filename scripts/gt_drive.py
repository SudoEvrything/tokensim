#!/usr/bin/env python3
"""gigatoken arm of the same-window ladder (see scripts/ladder_giga.py).

Loads the tokenizer ONCE, then encodes the same slice `reps` times and
prints one line per rep in the bench_encode format, so the two engines'
rows are directly comparable. Document shape matches ours (--batch 4):
the slice is cut into doc-mib pieces outside the timed region.

usage: gt_drive.py MODEL CORPUS MIB REPS DOC_MIB SCHEME
"""
import sys
import time

import awkward as ak
from gigatoken import BytesSource, Tokenizer

model, corpus, mib, reps, doc_mib, scheme = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    sys.argv[6] or None,
)
tok = Tokenizer.from_tiktoken(model, pretokenizer=scheme)
with open(corpus, "rb") as fh:
    raw = fh.read(mib * 1024 * 1024)
step = doc_mib * 1024 * 1024
docs = [raw[i : i + step] for i in range(0, len(raw), step)]
total = sum(len(d) for d in docs) / 1048576.0
for i in range(reps):
    t0 = time.perf_counter()
    out = tok.encode_batch(BytesSource(docs))
    dt = time.perf_counter() - t0
    print(f"rep {i}  {dt:8.3f} s  {total / dt:9.1f} MiB/s  ({int(ak.count(out))} ids)",
          flush=True)
