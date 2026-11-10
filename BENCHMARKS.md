# Benchmarks

All numbers below were measured on the development machine in one
session, with the standing protocol: warm = the last of three
rounds (the first round warms the caches), cold = the first pass
over an evicted corpus, and every comparison against another engine
runs in the same window - the same corpus, the same pinning, the
same thread counts. Machine: Intel Core i5-13450HX (10 cores, 16
hardware threads: 6 performance cores with hyperthreading, 4
efficiency cores), 20 MB L3, 16 GB RAM, Linux, AVX2 (no AVX-512 on
this CPU; the AVX-512 paths are verified functionally under the
Intel SDE emulator). Absolute numbers move with the silicon; the
ratios are the durable claim.

Reproduce any table from a fresh clone:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
bash scripts/run_ladder.sh                # the family ladder, warm protocol
bash scripts/bench_wrappers.sh both       # the C / Python / Rust wrapper arms
python3 scripts/ladder_giga.py --corpus testdata/owt.txt --mib 476 \
    --reps 2 --threads 1,6,10 --engine both   # the same-window gigatoken ladder
```

## Family ladder (MB/s, warm, 476 MB owt corpus)

| Family | 1 thread | 6 threads | 10 threads |
| --- | --- | --- | --- |
| o200k | 567.0 | 2705.5 | 3500.8 |
| cl100k | 566.9 | 2722.3 | 3545.5 |
| deepseek | 525.7 | 2366.7 | 2921.4 |
| qwen2.5 | 448.1 | 1924.6 | 2332.2 |
| nemotron | 525.3 | 2229.1 | 2510.8 |
| glm4 | 524.2 | 2063.3 | 2516.5 |
| gemma4 | 148.9 | 343.1 | 325.0 |
| gemma3-4b | 134.0 | 328.0 | 342.3 |
| smollm2 | 517.1 | 2054.4 | 2700.0 |
| mistral | 150.4 | 309.4 | 315.5 |
| kimi | 103.4 | 397.7 | 557.8 |
| xlmr / t5 | 32.1 / 26.2 | 43.6 / 33.6 | 34.2 / 44.9 |

Shapes: the byte-level families scale past 6 threads; the
metaspace families (gemma4, gemma3-4b, mistral) flatten once
their pieces get small; the unigram families (xlmr, t5) are a
dependency chain and peak early. Every row is byte-exact against
the family's reference; the ladder also checks the per-family
corpus hashes at each thread count (0 mismatches).

## Against the scalar reference engines

(single thread, warm, the same 16 MB natural-text corpus, the
same window):

| encoding | reference engine, scalar | tokensim, 1 thread | tokensim, 6 threads |
| --- | --- | --- | --- |
| o200k | tiktoken 32.1 MB/s | 571.1 MB/s (**17.8x**) | 2709.0 MB/s |
| cl100k | tiktoken 22.5 MB/s | 576.0 MB/s (**25.6x**) | 2783.4 MB/s |
| deepseek | official HF tokenizer 2.3 MB/s | 509.0 MB/s (**~220x**) | 2355.7 MB/s |

The deepseek reference's 2.3 MB/s is its real-world single-thread
behavior - its pre-tokenizer regex is famously expensive. Reproduce
with `scripts/bench_scalar_baseline.py`.

## Against gigatoken (the Rust reference engine)

The same-window ladder, both engines alternating process by
process, 476 MB of owt.txt, ours/theirs:

| Threads | cold (first pass) | warm (second pass) |
| --- | --- | --- |
| 1 | 0.90 | 0.87 |
| 6 | 0.99 | 1.12 |
| 10 | 1.04 | 1.45 |

The single-thread gap is the recorded honest frontier. Warm - the
state sustained training and indexing pipelines actually run in -
is ours ahead at every multi-thread point. An earlier window on
the 495 MB document corpus showed the same shape (warm 1.26-1.39x
at 6-10 threads).

## The wrappers: how a program calls it

The three consumers in `scripts/consumers/` tokenize the corpora
the way a real program does: the mmap input, 16 MiB training
documents, an inference-style prompt batch, and every document
decoded and byte-compared before exit. On the 476 MB owt corpus,
threads auto:

| wrapper | encode MB/s | decode MB/s | round trip MB/s |
| --- | --- | --- | --- |
| C | 1985.1 | 4090.3 | 1192.9 |
| Rust | 2161.7 | 4049.4 | 1276.0 |
| Python | 2145.5 | 3595.2 | 1043.0 |

The ids totals and the fold hashes are identical across the three
wrappers on every corpus (5 corpora, 15 of 15 round trips
byte-exact). The python decode gives back the bytes-object copy;
the C and rust hand back the engine's buffer with no extra copy.
Callers who keep both legs in their own buffers (the `*_into`
APIs) reach 4.1-4.4 GB/s decode in all three languages.

## What is NOT claimed

- No comparison against other projects' published numbers: the
  protocols differ, the machines differ, and a ratio measured in
  one window is the only comparison this project trusts.
- The AVX-512 paths are functionally verified under SDE; their
  throughput waits for a first real AVX-512 host.
- Single-thread cold on the big corpora is the honest gap against
  gigatoken (0.87-0.90x) and is reported as such.
