#!/usr/bin/env python3
"""Compare TokenSIMD against the reference implementations on a large
corpus — memory-frugally (the naive full-file tiktoken run materializes
~6 GB of Python ints and OOMs; this script streams in bounded chunks).

Measured:
  tiktoken o200k_base   (python; chunked single-core, ids counted, freed)
  HF tokenizers GPT-2   (Rust + rayon; encode_batch over chunk groups)
  gigatoken o200k       (Rust + worker pool; encode_batch parallel —
                         the library the benchmark claims are from)
  TokenSIMD o200k       (CLI, 1 / 8 / 16 threads)
  TokenSIMD gpt2        (CLI, 1 / 8 / 16 threads)

Parity is verified on a bounded slice (whole-file encode on both sides,
so the pretokenization is identical): TokenSIMD == tiktoken for o200k
and TokenSIMD == HF for GPT-2.

usage: python3 scripts/compare_reference.py [--corpus testdata/document.txt]
                                            [--slice-mib 32]
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
BIN = os.path.join(BUILD, "tsimd-tokenize")

TIKTOKEN_CHUNK = 8 * 1024 * 1024      # 8 MiB text per encode call
HF_GROUP, HF_CHUNK = 4, 4 * 1024 * 1024   # 4 x 4 MiB per rayon batch
GT_CHUNK = 4 * 1024 * 1024            # 4 MiB docs per gigatoken batch

def mib(n):
    return n / (1024 * 1024)

def time_call(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result

def run_tokensimd(vocab_args, corpus, threads):
    cmd = [BIN] + (["--threads", str(threads)] if threads > 1 else []) \
        + vocab_args + [corpus]
    devnull = open(os.devnull, "wb")
    t0 = time.perf_counter()
    with devnull:
        rc = subprocess.call(cmd, stdout=devnull, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(f"TokenSIMD failed: {' '.join(cmd)}")
    return dt

def iter_chunks(path, size):
    """Yield text chunks, decoupled from pretoken boundaries (fine for
    reference timing; parity is checked on a whole-encoded slice)."""
    with open(path, "rb") as f:
        while True:
            b = f.read(size)
            if not b:
                break
            yield b.decode("utf-8", errors="replace")

def tiktoken_o200k(path, n_bytes):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tokens = 0
    def work():
        nonlocal tokens
        for chunk in iter_chunks(path, TIKTOKEN_CHUNK):
            tokens += len(enc.encode_ordinary(chunk))
    dt, _ = time_call(work)
    return dt, tokens

def gigatoken_o200k(path, n_bytes):
    import gigatoken as gt
    tok = gt.Tokenizer.from_tiktoken(
        os.path.join(ROOT, "models/o200k_base.tiktoken"), "o200k")
    tokens = 0
    def work():
        nonlocal tokens
        group = []
        for chunk in iter_chunks(path, GT_CHUNK):
            group.append(chunk)
            if len(group) == 16:
                enc = tok.encode_batch(group, parallel=True)
                tokens += sum(len(e) for e in enc.ids) if hasattr(
                    enc, "ids") else sum(len(e) for e in enc)
                group = []
        if group:
            enc = tok.encode_batch(group, parallel=True)
            tokens += sum(len(e) for e in enc.ids) if hasattr(
                enc, "ids") else sum(len(e) for e in enc)
    dt, _ = time_call(work)
    return dt, tokens

def hf_gpt2(path, n_bytes):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    hf = Tokenizer(models.BPE.from_file(
        os.path.join(ROOT, "models/encoder.json"),
        os.path.join(ROOT, "models/vocab.bpe")))
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                use_regex=True)
    hf.decoder = decoders.ByteLevel()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokens = 0
    def work():
        nonlocal tokens
        group = []
        for chunk in iter_chunks(path, HF_CHUNK):
            group.append(chunk)
            if len(group) == HF_GROUP:
                encs = hf.encode_batch(group)
                tokens += sum(len(e.ids) for e in encs)
                group = []
        if group:
            encs = hf.encode_batch(group)
            tokens += sum(len(e.ids) for e in encs)
    dt, _ = time_call(work)
    return dt, tokens

def verify_parity(path, slice_bytes):
    """Whole-encode the first slice on both sides; compare id lists."""
    import tiktoken
    data = open(path, "rb").read(slice_bytes)
    open("/tmp/tsimd_cmp_slice.bin", "wb").write(data)

    enc = tiktoken.get_encoding("o200k_base")
    ref = list(enc.encode_ordinary(data.decode("utf-8", errors="replace")))
    out = subprocess.run(
        [BIN, os.path.join(ROOT, "models/o200k_base.tiktoken"), "o200k",
         "/tmp/tsimd_cmp_slice.bin"], capture_output=True, text=True)
    got = [int(x) for x in out.stdout.split()]
    o200k_ok = got == ref

    gt_ok = False
    try:
        import gigatoken as gt
        gtok = gt.Tokenizer.from_tiktoken(
            os.path.join(ROOT, "models/o200k_base.tiktoken"), "o200k")
        gt_ids = gtok.encode(data.decode("utf-8", errors="replace")).ids \
            if hasattr(gtok.encode(data.decode("utf-8", errors="replace")),
                       "ids") \
            else list(gtok.encode(data.decode("utf-8", errors="replace")))
        gt_ok = got == list(gt_ids)
    except Exception as e:
        print("  gigatoken parity check failed:", e)

    from tiktoken import get_encoding as _ge
    enc_cl = _ge("cl100k_base")
    ref_cl = list(enc_cl.encode_ordinary(data.decode("utf-8", errors="replace")))
    out_cl = subprocess.run(
        [BIN, os.path.join(ROOT, "models/cl100k_base.tiktoken"), "cl100k",
         "/tmp/tsimd_cmp_slice.bin"], capture_output=True, text=True)
    got_cl = [int(x) for x in out_cl.stdout.split()]
    cl100k_ok = got_cl == ref_cl

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    hf = Tokenizer(models.BPE.from_file(
        os.path.join(ROOT, "models/encoder.json"),
        os.path.join(ROOT, "models/vocab.bpe")))
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                use_regex=True)
    hf.decoder = decoders.ByteLevel()
    ref2 = hf.encode(data.decode("utf-8", errors="replace")).ids
    out2 = subprocess.run(
        [BIN, "--format", "gpt2",
         os.path.join(ROOT, "models/encoder.json"), "gpt2",
         "/tmp/tsimd_cmp_slice.bin"], capture_output=True, text=True)
    got2 = [int(x) for x in out2.stdout.split()]
    gpt2_ok = got2 == list(ref2)

    return o200k_ok, cl100k_ok, gpt2_ok, gt_ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT,
                                                     "testdata/document.txt"))
    ap.add_argument("--slice-mib", type=int, default=32,
                    help="parity-check slice size (whole-encoded)")
    args = ap.parse_args()
    n = os.path.getsize(args.corpus)

    print(f"corpus: {args.corpus} ({mib(n):.1f} MiB)")
    rows = []

    print("  TokenSIMD o200k  1 thread ...", flush=True)
    dt = run_tokensimd(
        [os.path.join(ROOT, "models/o200k_base.tiktoken"), "o200k"],
        args.corpus, 1)
    rows.append(("TokenSIMD o200k", "1", dt))

    for threads in (8, 16):
        print(f"  TokenSIMD o200k  {threads} threads ...", flush=True)
        dt = run_tokensimd(
            [os.path.join(ROOT, "models/o200k_base.tiktoken"), "o200k"],
            args.corpus, threads)
        rows.append(("TokenSIMD o200k", str(threads), dt))

    print("  TokenSIMD gpt2   1 thread ...", flush=True)
    dt = run_tokensimd(
        ["--format", "gpt2", os.path.join(ROOT, "models/encoder.json"),
         "gpt2"], args.corpus, 1)
    rows.append(("TokenSIMD gpt2", "1", dt))

    for threads in (8, 16):
        print(f"  TokenSIMD gpt2   {threads} threads ...", flush=True)
        dt = run_tokensimd(
            ["--format", "gpt2", os.path.join(ROOT, "models/encoder.json"),
             "gpt2"], args.corpus, threads)
        rows.append(("TokenSIMD gpt2", str(threads), dt))

    print("  tiktoken o200k_base (chunked 8 MiB, 1 core) ...", flush=True)
    dt, tk = tiktoken_o200k(args.corpus, n)
    rows.append(("tiktoken o200k_base", "1", dt))

    print("  HF tokenizers GPT-2 (rayon batches) ...", flush=True)
    dt, hf = hf_gpt2(args.corpus, n)
    rows.append(("HF tokenizers GPT-2", "rayon", dt))

    print("  gigatoken o200k (worker pool) ...", flush=True)
    dt, gt_n = gigatoken_o200k(args.corpus, n)
    rows.append(("gigatoken o200k", "pool", dt))

    ours8 = [r for r in rows if r[0] == "TokenSIMD o200k" and r[1] == "8"][0][2]
    ours1 = [r for r in rows if r[0] == "TokenSIMD o200k" and r[1] == "1"][0][2]
    ours16 = next((r[2] for r in rows
                   if r[0] == "TokenSIMD o200k" and r[1] == "16"), None)

    print()
    print(f"{'tokenizer':26s} {'mode':7s} {'seconds':>8s} {'MiB/s':>7s} "
          f"{'vs ours-8t':>10s}")
    for name, mode, dt in rows:
        tag = f"{dt / ours8:.1f}x slower" if dt > ours8 else "baseline"
        if name.startswith("TokenSIMD") and mode == "8":
            tag = "baseline"
        print(f"{name:26s} {mode:7s} {dt:8.2f} {mib(n) / dt:7.1f} {tag:>10s}")

    print(f"\nTokenSIMD thread speedup (o200k): 1t {ours1 / ours8:.2f}x -> 8t"
          + (f", {ours1 / ours16:.2f}x -> 16t" if ours16 else ""))
    gt_row = next((r for r in rows if r[0] == "gigatoken o200k"), None)
    if gt_row:
        print(f"gigatoken o200k vs TokenSIMD 16t: "
              f"{gt_row[2] / ours16:.2f}x slower" if ours16 else "")

    print(f"\nparity slice ({args.slice_mib} MiB, whole-encoded both sides):")
    slice_bytes = args.slice_mib * 1024 * 1024
    o200k_ok, cl100k_ok, gpt2_ok, gt_ok = verify_parity(
        args.corpus, min(slice_bytes, n))
    print(f"  TokenSIMD == tiktoken (o200k): {'MATCH' if o200k_ok else 'MISMATCH'}")
    print(f"  TokenSIMD == tiktoken (cl100k): {'MATCH' if cl100k_ok else 'MISMATCH'}")
    print(f"  TokenSIMD == HF (gpt2):        {'MATCH' if gpt2_ok else 'MISMATCH'}")
    print(f"  TokenSIMD == gigatoken (o200k): {'MATCH' if gt_ok else 'MISMATCH'}")
    return 0 if (o200k_ok and cl100k_ok and gpt2_ok and gt_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
