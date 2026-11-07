#!/usr/bin/env python3
"""A/B throughput tester for TokenSIMD.

Runs two configurations against the same corpus in interleaved rounds
(A,B,A,B,...) so machine-load drift affects both sides equally, verifies
the token output is byte-identical between A and B (an A/B comparison is
only meaningful for correctness-neutral changes), and reports medians.

Variants combine CLI flags and environment variables:

    cache-on / cache-off     pretoken memo cache (TSIMD_DISABLE_PCACHE)
    simd-scalar/sse2/avx2    forced SIMD level (TSIMD_SIMD_FORCE)
    mmap / read              input stage (--no-mmap)
    threads=N                parallel encode (--threads N)

usage: ab_test.py --a cache-on --b cache-off [--corpus F] [--vocab V]
                  [--scheme cl100k] [--rounds 5] [--verify-only]
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
BIN = os.path.join(BUILD, "tsimd-tokenize")

VARIANTS = {
    "cache-on":    {"env": {}, "args": []},
    "cache-off":   {"env": {"TSIMD_DISABLE_PCACHE": "1"}, "args": []},
    "simd-scalar": {"env": {"TSIMD_SIMD_FORCE": "scalar"}, "args": []},
    "simd-sse2":   {"env": {"TSIMD_SIMD_FORCE": "sse2"}, "args": []},
    "simd-avx2":   {"env": {"TSIMD_SIMD_FORCE": "avx2"}, "args": []},
    "mmap":        {"env": {}, "args": []},
    "read":        {"env": {}, "args": ["--no-mmap"]},
}

def variant(name):
    if name.startswith("threads="):
        n = name.split("=", 1)[1]
        return {"env": {}, "args": ["--threads", n]}
    if name not in VARIANTS:
        known = ", ".join(sorted(VARIANTS) + ["threads=N"])
        raise SystemExit(f"unknown variant {name!r}; known: {known}")
    return VARIANTS[name]

def run_once(cmd, env, out=None):
    """One full CLI invocation; returns (seconds, returncode)."""
    if out is None:
        out_fh = open(os.devnull, "wb")
    else:
        out_fh = open(out, "wb")
    t0 = time.perf_counter()
    with out_fh:
        rc = subprocess.call(cmd, stdout=out_fh,
                             stderr=subprocess.DEVNULL,
                             env={**os.environ, **env})
    return time.perf_counter() - t0, rc

def build_cmd(variant, vocab, scheme, corpus):
    return [BIN] + variant["args"] + [vocab, scheme, corpus]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--corpus", default=os.environ.get("TSIMD_BENCH_CORPUS",
                                                       "/tmp/bench_nat.txt"))
    ap.add_argument("--vocab", default=os.environ.get("TSIMD_BENCH_VOCAB",
                                                      "/tmp/cl100k_base.tiktoken"))
    ap.add_argument("--scheme", default="cl100k")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--verify-only", action="store_true",
                    help="only verify A and B produce identical output")
    args = ap.parse_args()

    va, vb = variant(args.a), variant(args.b)
    cmd_a = build_cmd(va, args.vocab, args.scheme, args.corpus)
    cmd_b = build_cmd(vb, args.vocab, args.scheme, args.corpus)
    if not os.path.exists(BIN):
        raise SystemExit(f"build {BIN} first")
    if not os.path.exists(args.corpus):
        raise SystemExit(f"corpus not found: {args.corpus}")

    with tempfile.TemporaryDirectory() as td:
        fa, fb = os.path.join(td, "a.bin"), os.path.join(td, "b.bin")
        _, rca = run_once(cmd_a, va["env"], out=fa)
        _, rcb = run_once(cmd_b, vb["env"], out=fb)
        if rca or rcb:
            raise SystemExit(f"A/B invocation failed (rc {rca}/{rcb})")
        same = open(fa, "rb").read() == open(fb, "rb").read()
        if args.verify_only:
            print("outputs identical:", same)
            return 0 if same else 1
        if not same:
            raise SystemExit("A/B outputs DIFFER — not a valid A/B pair")

    times = {"a": [], "b": []}
    order = []
    for i in range(args.rounds):
        order.append("ab" if i % 2 == 0 else "ba")
        for side in order[-1]:
            cmd = cmd_a if side == "a" else cmd_b
            dt, rc = run_once(cmd, (va if side == "a" else vb)["env"])
            if rc != 0:
                raise SystemExit("invocation failed during timing")
            times[side].append(dt)

    size_mib = os.path.getsize(args.corpus) / (1024 * 1024)
    med_a = statistics.median(times["a"])
    med_b = statistics.median(times["b"])
    mib_a = size_mib / med_a
    mib_b = size_mib / med_b

    print(f"A = {args.a}: median {med_a:.3f}s  ({mib_a:.1f} MiB/s)  "
          f"runs {[f'{t:.3f}' for t in times['a']]}")
    print(f"B = {args.b}: median {med_b:.3f}s  ({mib_b:.1f} MiB/s)  "
          f"runs {[f'{t:.3f}' for t in times['b']]}")
    verdict = ("B faster" if med_b < med_a * 0.98 else
               "A faster" if med_a < med_b * 0.98 else "tie (±2%)")
    print(f"speedup B/A: {med_a / med_b:.2f}x   ({verdict})")
    print("round order:", " ".join(order))
    return 0

if __name__ == "__main__":
    sys.exit(main())
