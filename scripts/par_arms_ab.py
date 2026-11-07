#!/usr/bin/env python3
"""Run 101: paired A/B of the pool's job-assignment modes.

TSIMD_PAR_STATIC selects how the pool hands jobs to workers:

    0   the dynamic atomic queue (the pre-run-101 behavior)
    1   static per-participant ranges (a contiguous, weight-balanced
        slice of the job list)
    2   static ranges + steal-from-the-tail (the default)

Arms alternate round by round (0,1,2,0,1,2,...) inside ONE window, so
machine drift hits every arm equally; only the assignment changes. The
token output is byte-identical across arms (asserted by test_parallel),
so a difference is a pure throughput effect.

usage: par_arms_ab.py [--corpus F] [--mib N] [--reps R] [--rounds K]
                      [--threads T] [--arms 0,1,2] [--doc-mib D]
                      [--stats]
"""
import argparse
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bench_encode")

PINS = {
    1: "0",
    2: "0,2",
    4: "0,2,4,6",
    6: "0,2,4,6,8,10",
    10: "0,2,4,6,8,10,12,13,14,15",
    16: "0-15",
}

import re
REP_RE = re.compile(r"^rep\s+(\d+)\s+([\d.]+)\s*s\s+([\d.]+)\s*MiB/s", re.M)

def run(args, threads, arm, extra_env=None):
    env = dict(os.environ)
    env["TSIMD_PAR_STATIC"] = str(args.static_arm if args.sweep else arm)
    if extra_env:
        env.update(extra_env)
    cmd = [
        "taskset", "-c", PINS[threads],
        BIN, "--format", "tiktoken", "--scheme", args.scheme,
        "--reps", str(args.reps), "--mib", str(args.mib),
        "--batch", str(max(args.doc_mib, 1)), "--threads", str(threads),
        args.model, args.corpus,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
        raise SystemExit(f"failed: arm {arm}")
    return p

def parse(out):
    return [(int(i), float(v)) for i, _s, v in REP_RE.findall(out)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="testdata/owt.txt")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--arms", default="0,1,2")
    ap.add_argument("--sweep", default="",
                    help="K=V1,V2,V3 with --static-arm A: interleave the "
                         "values of K as the arms")
    ap.add_argument("--static-arm", type=int, default=2,
                    help="TSIMD_PAR_STATIC for --sweep mode")
    ap.add_argument("--model", default="models/o200k_base.tiktoken")
    ap.add_argument("--scheme", default="o200k")
    ap.add_argument("--doc-mib", type=int, default=4)
    ap.add_argument("--stats", action="store_true",
                    help="one round with TSIMD_PHASE_STATS (+per-worker)")
    args = ap.parse_args()
    sweep_env_key = None
    if args.sweep:
        key, _, vals = args.sweep.partition("=")
        arms = [v for v in vals.split(",") if v]
        sweep_env_key = key
    else:
        arms = [int(a) for a in args.arms.split(",")]
        sweep_env = None

    if args.stats:
        for arm in arms:
            extra = {sweep_env_key: arm} if sweep_env_key else {}
            extra["TSIMD_PHASE_STATS"] = "1"
            extra["TSIMD_PERWORKER"] = "1"
            p = run(args, args.threads, arm, extra)
            print(f"## arm {arm}")
            for line in p.stderr.splitlines():
                if line.startswith(("tsimd mix", "tsimd phases",
                                    "tsimd worker")):
                    print("  " + line)
        return

    print(f"# {args.corpus}  {args.mib} MiB  scheme={args.scheme}  "
          f"{args.threads}t  reps={args.reps} rounds={args.rounds}  "
          f"arms={arms}")
    rows = {a: [] for a in arms}
    for _ in range(args.rounds):
        for a in arms:
            extra = {sweep_env_key: a} if sweep_env_key else None
            rows[a].append(parse(run(args, args.threads, a,
                                     extra).stdout))
    reps = args.reps
    for a in arms:
        for r, sample in enumerate(rows[a]):
            cells = "  ".join(f"rep{i} {v:8.1f}" for i, v in sample)
            print(f"  arm{a} run{r}  {cells}")
    if len(arms) > 1:
        base = arms[0]
        for k in range(reps):
            vals = {a: [rows[a][r][k][1] for r in range(args.rounds)]
                    for a in arms}
            med = {a: statistics.median(v) for a, v in vals.items()}
            parts = "  ".join(f"arm{a} {med[a]:8.1f}" for a in arms)
            r = "  ".join(f"{med[a] / med[base]:.3f}x" for a in arms[1:])
            print(f"  rep{k}  {parts}   ratio-vs-arm{base}: {r}")

if __name__ == "__main__":
    main()
