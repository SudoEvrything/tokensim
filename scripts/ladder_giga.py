#!/usr/bin/env python3
"""TokenSIMD vs gigatoken ladder, both measured in ONE window.

Both engines load the tokenizer once per process, encode the same
`--mib` slice of the same corpus `--reps` times, and report per-rep
MiB/s (rep 0 = cold caches, rep k = warm). The engines alternate
process by process (ours, theirs, ours, ...) so machine drift — this
laptop drifts up to ~25% between windows — hits both arms.

Document shape is matched: ours runs `--batch D` (D-MiB docs) and the
gigatoken driver cuts the same slice into D-MiB docs. `--doc-mib 0`
gives gigatoken one whole document instead.

Pinning (the project convention): 1t = c0; 6t = c0,2,4,6,8,10;
10t = +c12-15; 16t = c0-15.

usage: ladder_giga.py [--corpus F] [--mib N] [--reps R] [--rounds K]
                      [--threads 1,6] [--model M] [--scheme o200k]
                      [--doc-mib D] [--engine ours|giga|both]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bench_encode")
DRIVER = os.path.join(ROOT, "scripts", "gt_drive.py")

PINS = {
    1: "0",
    6: "0,2,4,6,8,10",
    10: "0,2,4,6,8,10,12,13,14,15",
    16: "0-15",
}
REP_RE = re.compile(r"^rep\s+(\d+)\s+([\d.]+)\s*s\s+([\d.]+)\s*MiB/s", re.M)

def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=ROOT)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return p.stdout

def ours(args, threads):
    cmd = [
        "taskset", "-c", PINS[threads],
        BIN, "--format", "tiktoken", "--scheme", args.scheme,
        "--reps", str(args.reps), "--mib", str(args.mib),
        "--batch", str(max(args.doc_mib, 1)), "--threads", str(threads),
        args.model, args.corpus,
    ]
    return parse(run(cmd))

def giga(args, threads):
    cmd = [
        "taskset", "-c", PINS[threads],
        sys.executable, DRIVER, args.model, args.corpus,
        str(args.mib), str(args.reps), str(args.doc_mib), args.scheme,
    ]
    return parse(run(cmd, env={"RAYON_NUM_THREADS": str(threads)}))

def parse(out):
    return [(int(i), float(mbs)) for i, _s, mbs in REP_RE.findall(out)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="testdata/owt.txt")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--threads", default="1,6")
    ap.add_argument("--model", default="models/o200k_base.tiktoken")
    ap.add_argument("--scheme", default="o200k")
    ap.add_argument("--doc-mib", type=int, default=4)
    ap.add_argument("--engine", default="both", choices=["ours", "giga", "both"])
    args = ap.parse_args()

    arms = []
    if args.engine in ("ours", "both"):
        arms.append(("ours", ours))
    if args.engine in ("giga", "both"):
        arms.append(("giga", giga))

    print(f"# {args.corpus}  {args.mib} MiB  scheme={args.scheme} "
          f"doc={args.doc_mib} MiB  reps={args.reps} rounds={args.rounds}")
    for threads in (int(t) for t in args.threads.split(",")):
        rows = {name: [] for name, _ in arms}
        for _ in range(args.rounds):
            for name, fn in arms:            # ours then theirs, each round
                rows[name].append(fn(args, threads))
        print(f"## {threads}t  (taskset -c {PINS[threads]})")
        for name, _ in arms:
            for r, sample in enumerate(rows[name]):
                cells = "  ".join(f"rep{i} {v:8.1f}" for i, v in sample)
                print(f"  {name:4s} run{r}  {cells}")
        if len(arms) == 2:
            for k in range(args.reps):
                a = [rows["ours"][r][k][1] for r in range(args.rounds)]
                b = [rows["giga"][r][k][1] for r in range(args.rounds)]
                am, bm = sorted(a)[len(a) // 2], sorted(b)[len(b) // 2]
                print(f"  rep{k}  ratio ours/giga = {am / bm:.3f}x "
                      f"({am:.1f} vs {bm:.1f})")

if __name__ == "__main__":
    main()
