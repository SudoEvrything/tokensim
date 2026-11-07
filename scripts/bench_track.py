#!/usr/bin/env python3
"""Benchmark regression tracker.

Runs the benchmark binaries, parses throughput numbers, and compares
against a JSON baseline.

    python3 scripts/bench_track.py --save          # record a baseline
    python3 scripts/bench_track.py --check         # fail on regression
    python3 scripts/bench_track.py --check --threshold 0.20

Metrics (MiB/s unless noted):
  pretok.gpt2.simd / pretok.gpt2.scalar_ref  — bench_pretokenizer
  encode.cl100k.sequential                   — CLI single-shot on corpus
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
CORPUS = os.environ.get("TSIMD_BENCH_CORPUS", "/tmp/bench_nat.txt")
VOCAB = os.environ.get("TSIMD_BENCH_VOCAB", "/tmp/cl100k_base.tiktoken")
BASELINE = os.path.join(os.path.dirname(__file__), "bench_baseline.json")

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def collect():
    metrics = {}
    out = run([os.path.join(BUILD, "bench_pretokenizer"), "gpt2", "32"])
    for line in out.stdout.splitlines():
        m = re.search(r"(scalar-ref|simd)\s+([\d.]+) MiB/s", line)
        if m:
            metrics["pretok.gpt2." + m.group(1)] = float(m.group(2))

    if os.path.exists(VOCAB) and os.path.exists(CORPUS):
        best = 0.0
        for _ in range(2):
            with tempfile.NamedTemporaryFile() as tf:
                t0 = time.perf_counter()
                with open(os.devnull, "wb") as devnull:
                    r = subprocess.run(
                        [os.path.join(BUILD, "tsimd-tokenize"), VOCAB,
                         "cl100k", CORPUS], stdout=devnull,
                        stderr=subprocess.DEVNULL)
                dt = time.perf_counter() - t0
            if r.returncode == 0 and dt > 0:
                size = os.path.getsize(CORPUS)
                best = max(best, size / dt / (1024 * 1024))
        metrics["encode.cl100k.sequential"] = round(best, 2)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="allowed relative regression (default 0.15)")
    args = ap.parse_args()

    metrics = collect()
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.save:
        with open(BASELINE, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print("baseline saved to", os.path.normpath(BASELINE))
        return 0

    if not os.path.exists(BASELINE):
        print("no baseline; run with --save first", file=sys.stderr)
        return 2
    with open(BASELINE) as f:
        base = json.load(f)

    failed = False
    for key, got in metrics.items():
        want = base.get(key)
        if not want:
            continue
        rel = (want - got) / want
        status = "OK"
        if rel > args.threshold:
            status = "REGRESSION"
            failed = True
        print(f"  {key:32s} {got:8.1f} vs {want:8.1f}  {status}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
