# Benchmarks

Numbers measured on the development machine: Intel i5-13450HX (Raptor
Lake, 10 cores / 16 threads, **AVX2 only, no AVX-512**), gcc, Release
(`-O3`), single thread unless stated. MiB/s are input bytes per second;
Mtok/s are output token ids per second.

> **Environment caveat (Wave 1 hardening):** the post-hardening tables
> below were measured while a system integrity-check process saturated
> one core; absolute numbers are depressed by roughly the same factor
> across scalar and SIMD paths (the simd/scalar ratio is unchanged at
> ~3.1x, so no algorithmic regression). Re-measure on a quiet machine
> before quoting absolute throughput.

Reproduce:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
build/bench_pretokenizer [gpt2|cl100k] [MiB]      # pretokenizer stage
TSIMD_STATS=1 build/tokenize_file --format tiktoken /tmp/r50k_base.tiktoken \
    gpt2 CORPUS > /dev/null                        # end-to-end + cache stats
```

Corpora: `natural` = 40 MB mixed system text (code, licenses, docs;
31k distinct short pretokens per 8 MB sample); `wordsoup` = 33 MB
synthetic word soup (~50 distinct words, best case for the pretoken
cache); `random` = 15 MB random alphanumeric words (worst case).

## Final campaign state (runs 183-249, 2026-10)

The numbers below are the campaign's closing tables, all measured on
this machine with the standing protocol: interleaved A/B arms under
`perf stat`, cold rounds via user-space `fadvise(DONTNEED)` eviction,
64 MiB of owt.txt unless stated. Ratios are ours/gigatoken (the Rust
crate), both arms run here.

### 6-thread warm (64 MiB, encode-into, ratio vs gigatoken)

| model | ratio | | model | ratio |
|---|---|---|---|---|
| o200k | 2.05 | | nemotron | 1.84 |
| cl100k | 1.86 | | glm4 | 1.41-1.64 |
| deepseek | 2.29 | | gemma4 | 1.52 |
| qwen2.5 | 1.43 | | mistral | 1.35-1.40 |

### 1-thread warm (ratio vs gigatoken)

o200k 0.96, cl100k 0.96, qwen 0.89, nemotron 0.91, glm4 0.85-0.89,
gemma4 1.85, mistral 1.60, deepseek 1.34-1.37.

### 6-thread full corpus (whole files; warm | cold)

| dataset | o200k | cl100k | deepseek | glm4 | gemma4 | mistral |
|---|---|---|---|---|---|---|
| owt (476 MiB) | 1.82/1.38 | 1.50/1.19 | 1.92/1.47 | 1.28/0.86 | 1.06/0.61 | 0.94/0.59 |
| document (495 MiB) | 1.69/1.35 | 1.39/1.16 | 1.97/1.57 | 1.16/1.01 | 1.73/0.63 | 1.40/0.62 |

t5 and xlmr: gigatoken has no unigram support (no comparison); ours
30-42 MiB/s full corpus, byte-exact.

### Model load time (per process start)

o200k 192 ms, cl100k 85 ms, deepseek 93 ms, gemma4 289 ms,
mistral 27 ms. Cold SPM rows (0.59-0.63) are at the merge engine's
measured floor, the gap to gigatoken's cache-less steady state is
product shape (first-occurrence cache fill), not a missing
optimization (docs/243-245).

### What was measured out (do not retry without new information)

PGO on the final tree (o200k -19% cold), slots load 2x (build-page
faults), cross grids in three shapes (asymmetric pair population),
the pair memo (twice), the seed-skip whole-check (0.07%), the
codepoint piece table (~1%), the branchless SPM scan (distribution
decides), the 128-byte window rewrite. The warm 1t rows sit at
parity (0.96) and near floor; 6-thread scaling is 88% of the
power-limited all-core clock ceiling.

### The wide-thread ladder (run 252; whole datasets, both engines, 3 reps)

Whole owt (476 MB) and whole document (495 MB), all ten models,
1/6/10/16 threads, ours vs gigatoken, cold = first rep, warm =
best of the last two, MiB/s. Pinning: 1t = cores 4,5 (both arms),
6t = cores 0-5, 10t/16t unpinned. The full 80-row table is in
docs/252. Headline ratios (ours/giga, warm):

| dataset | model | 1t | 6t | 10t | 16t |
|---|---|---|---|---|---|
| owt | o200k | 0.96 | 1.72 | 1.08 | 1.21 |
| owt | cl100k | 0.91 | 1.74 | 1.10 | 1.17 |
| owt | deepseek | 1.27 | 2.22 | 1.19 | 1.34 |
| owt | qwen2.5 | 0.83 | 1.48 | 1.02 | 1.01 |
| owt | nemotron | 0.96 | 1.55 | 0.89 | 1.09 |
| owt | glm4 | 0.91 | 1.39 | 0.81 | 0.97 |
| owt | gemma4 | 1.24 | 1.04 | 0.82 | 0.87 |
| owt | mistral | 1.13 | 0.94 | 0.76 | 0.73 |
| doc | o200k | 1.02 | 1.69 | 1.09 | 1.18 |
| doc | cl100k | 0.97 | 1.65 | 1.02 | 1.17 |
| doc | deepseek | 1.26 | 2.16 | 1.14 | 1.24 |
| doc | qwen2.5 | 0.88 | 1.50 | 0.90 | 1.03 |
| doc | nemotron | 0.85 | 1.09 | 0.66 | 0.70 |
| doc | glm4 | 0.81 | 1.17 | 0.68 | 0.72 |
| doc | gemma4 | 2.48 | 1.72 | 0.82 | 0.92 |
| doc | mistral | 2.17 | 1.40 | 0.74 | 0.82 |

Ours peaks at 3,838 MiB/s (cl100k owt 16t); the 10t/16t ratios
compress because gigatoken's batch API scales across the E-cores
and hyperthreads while our per-worker caches lose locality beyond
6 cores (the standing "wide rows swing" note applies: qwen owt
warm 2070 at 6t, 1540 at 10t, 2901 at 16t). t5/xlmr: gigatoken has
no unigram support; ours 32-46 MiB/s, byte-exact.

### The structured regime (run 258; the cache-hostile corpus)

testdata/structured_mix.bin, 481 MB (the C sources + the tokenizer
JSONs + the base64, rotated). All ten models, 1/6/10/16 threads,
ours vs gigatoken, whole file. This corpus INVERTS several plain-
text conclusions:

- 1t warm: ALL models below giga (0.45-0.85), the high-entropy
  pretokens miss the private caches and the fill path dominates.
- 6t warm: our biggest wins (o200k 2.43, nemotron 1.99, cl100k
  1.99, glm4 1.77) EXCEPT qwen's NFC+pool pathology (0.48).
- 16t: THE COLLAPSE for ours (o200k 690 MiB/s vs 1,572 at 10t;
  qwen 138 vs 813) while gigatoken stays flat (~500-600): 16
  private caches + plong tables against unshareable content.
- The E-cores consumed 42 percent of the 16t cycles (the
  straggler barrier); the fill-heavy regime is where the shared-
  cache design (gigatoken) wins at every thread count.

The top lever this exposes: the cache-hostile mode switch (the
fill-heavy detection + the per-worker plong disable / the shared
cache). The full table and the counters: docs/258.

## Wave 1 hardening (2026-09)

All audit findings fixed with regression tests: scalar-fallback error
propagation, multi-byte digit-phase carry, 4 GiB streaming spans,
exact-regex tokenizer.json detection (Qwen2-class files), pair-table
grow-at-3/4-load (hang on crafted vocabs), heap-based BPE merge loop
(O(n log n) for long pretokens, byte-identical output, differential
tested), single-pass added-token scan (O(N), was O(hits x N)),
HF added-token semantics, thread-safe SIMD dispatch, per-call
`max_threads` honoring, CLI hardening. Verified: 10/10 tests under
Release, ASan/UBSan, and TSan (with ASLR disabled for the TSan runtime);
differential fuzz vs tiktoken through `--threads 4/8`; 38 MB parallel
== sequential.

## Wave 2, decode API + o200k (2026-09)

Public `tsimd_tokenizer_decode` (round-trip tested, specials
emitted/skipped). New o200k scheme (GPT-4o): parity-exact vs tiktoken
on smoke cases, 80 randomized differential samples, and a 2 MB
real-world corpus (MATCH). Two genuine bugs found and fixed by corpus
testing: heap BPE ghost-node merges (stale entries merging removed
nodes, now marked, with a long-same-symbol-run regression test) and
o200k's `[\r\n/]*` other-run suffix (slash re-absorbed after
newlines). Llama-3 confirmed to ship the cl100k regex, covered by
exact-match detection.

## Waves 3-4, packaging, bindings, fuzzing, tracking (2026-09)

- Install targets verified by a downstream `find_package(tokensim)`
  consumer; pkg-config and the `tsimd` Python module ship with the lib.
- Python parity suite: 23/23 (id-exact vs tiktoken for all three
  schemes, parallel/batch equality, decode round-trips).
- Property fuzz harness (oracle==SIMD, round-trip, split invariance)
  runs in CI as a deterministic driver; libFuzzer driver behind
  `-DTSIMD_ENABLE_LIBFUZZER=ON` (clang).
- `scripts/bench_track.py` guards against throughput regressions.

## A/B experiments (interleaved, output-verified)

`scripts/ab_test.py` compares configurations in interleaved rounds and
refuses to report unless A and B produce byte-identical tokens.
Measured during the system-load caveat above (medians of 5 rounds):

| pair | corpus | B/A median | conclusion |
| --- | --- | --- | --- |
| cache-off vs cache-on | prose 39 MB | 0.78 | pretoken cache = 1.28x end-to-end |
| cache-off vs cache-on | code 38 MB | 0.63 | cache = 1.59x on repetitive text |
| simd-scalar vs simd-avx2 | prose 39 MB | 0.80 | SIMD pretokenizer = 1.25x |
| simd-sse2 vs simd-avx2 | prose 39 MB | 1.03 | tie: algebra dominates, not the classifier |
| read vs mmap | code 38 MB | 1.00 | tie (warm page cache) |
| threads=1 vs threads=8 | prose 39 MB | 0.64 | 1.56x |
| threads=8 vs threads=16 | prose 39 MB | 0.89 | 1.13x |

## Large-file comparison (2026-09-14, testdata/document.txt)

495.4 MiB real-world corpus (Simple English Wikipedia XML + enwik8).
TokenSIMD = CLI end-to-end (mmap read, encode, ids to /dev/null);
tiktoken = python API in 8 MiB chunks (its recommended usage, single
core); HF tokenizers = Rust GPT-2 with ByteLevel via `encode_batch`
(rayon, 4x4 MiB groups). Parity verified on a whole-encoded 32 MiB
slice: **TokenSIMD == tiktoken (o200k) MATCH**, **TokenSIMD == HF
(GPT-2) MATCH**.

| tokenizer | mode | seconds | MiB/s | vs TokenSIMD 8t |
| --- | --- | --- | --- | --- |
| TokenSIMD o200k | 8 threads | 4.48 | **110.5** | baseline |
| TokenSIMD o200k | 1 thread | 15.39 | 32.2 | 3.4x slower |
| TokenSIMD gpt2 | 8 threads | 3.26 | **151.8** | baseline (gpt2) |
| TokenSIMD gpt2 | 1 thread | 7.72 | 64.2 | 1.7x slower |
| tiktoken o200k_base | 1 core, chunked | 24.33 | 20.4 | **5.4x slower** |
| HF tokenizers GPT-2 | rayon batches | 83.74 | 5.9 | **18.7x slower** |

Readings:
- **5.4x vs tiktoken** (same o200k model math, ids verified identical
  on the parity slice) at end-to-end throughput; 1.6x per core
  single-threaded (32.2 vs 20.4 MiB/s).
- **vs Hugging Face**: 25.7x on GPT-2 (151.8 vs 5.9 MiB/s); even
  single-threaded TokenSIMD is 10.9x faster than HF's rayon-parallel
  path on this corpus. HF's number depends on batch sizing; the
  canonical ByteLevel pipeline is simply slow on markup-heavy XML.
- TokenSIMD thread speedup at this scale: 3.43x (8 workers, hybrid
  P/E-core laptop).
- Reproduce: `python3 scripts/compare_reference.py`
  (memory-frugal: chunked streaming, ~100-500 MB peak RSS, a naive
  full-file tiktoken run materializes ~6 GB of Python ints and OOMs).

## CPU-pinning results (2026-09-17)

Pinning the pool's spawned workers to the first hardware thread of
each physical core (SMT siblings last, TSIMD_PIN=0 opts out) fixed
both scaling targets. Engine ladder, 499 MB OpenWebText, o200k,
first pass: 1t 48.8, 6t 226.2 (4.63x), 10t 285.4, 16t **298.4 MB/s
(6.11x)**, before pinning, 6t was ~2.2x and 16t ~2.9-3.5x.
**Wiki 495 MB: 8t 181.6, 16t 223.2 MB/s, the hybrid-core imbalance
is fixed (16t now faster than 8t) and wiki 16-thread PARITY with
gigatoken (212-241 MB/s) is achieved.** OpenWebText remains
memory-bound at 16 threads (2.1-2.6x behind gigatoken's first pass).
Reproduce: the C driver `/tmp/bench_batch` or the module's
encode_batch_raw; details in docs/03_cpu_pinning.md.

## Dual-cursor A/B + gigatoken 0.10.0 (2026-09-17, clean; aide killed)

The dual-cursor A/B: one binary, `TSIMD_DISABLE_DUAL=1` forces the
single-cursor path. 119 x 4 MiB docs from OpenWebText (476.0 MiB),
3 warm passes per row, byte-exact checksums in all 90+ passes
(`dec2534ddc77499e`, 106,625,690 ids). Warm means (dual / single):

| threads | dual-cursor | single-cursor | gain |
| --- | --- | --- | --- |
| 1 | 46.6 | 47.4 | neutral (−1.6 to +3.3 across order pairs) |
| 2 | 81.2 | 83.4 | **−2.5%** (consistent, both orders) |
| 4 | 150.6 | 150.7 | neutral (was −9.3% before the churn fix) |
| 8 | 230.3 | 227.7 | +1.1% |
| 12 | 299.8 | 294.5 | +1.8% |
| 16 | 326.2 | 321.2 | +1.6% |

The dual-cursor's per-job output vectors (malloc/free + grow-from-zero
churn) cost 9.3% at 4 threads, fixed by hoisting them into persistent
worker state. The earlier +4%/+8% (1t/2t) was the cool-start order
artifact; with the order balanced the 1t/2t effect is neutral-to-small.
**The dual-cursor on this machine is NEUTRAL, not +15%** (the claim on
gigatoken's pages is not reproduced here).

gigatoken 0.10.0 on THIS machine, same corpus, same 4 MiB doc shape,
bytes input both engines (engine work differs <0.5%: gigatoken emits
special id 199999 for the corpus's 812 literal "cJSON>", the
OpenWebText document separator; TokenSIMD/tiktoken encode it as
normal text):

| engine | mode | MiB/s |
| --- | --- | --- |
| gigatoken 0.10.0 | 1t cold (unseen batch) | **312** |
| gigatoken 0.10.0 | 1t warm (memo-cache serve) | **529** |
| gigatoken 0.10.0 | 16t warm (rayon ladder) | **2428-2644** |
| gigatoken 0.10.0 | own CLI, full corpus | 1704 MB/s |
| TokenSIMD | 1t (encode_batch) | 46.6 |
| TokenSIMD | 16t (encode_batch) | 312-330 |

**TokenSIMD is 5-8x behind gigatoken 0.10.0 on this machine.** The
recorded "per-core parity" (48 vs 39-49 MB/s/thread) described an
older reference or a str-conversion-polluted methodology. The gap
lives in the BPE/merge + dispatch stage (~84% of e2e), not the scan
(~16%, ~300 MiB/s per core). Details in
docs/08_dual_cursor_ab_and_the_gigatoken_gap.md.

## Run 10, the tight loop, the getenv find, the caller pin (2026-09-17)

Three fixes landed, one reverted, all byte-exact (12/12 tests,
Release + ASan/UBSan):

1. **The tight loop** (the sequential-style single-pass fused path;
   no match-ahead batch arrays): 1t +41% over the run-7 batched
   structure. A measured CROSSOVER between 4t and 8t, the tight
   loop wins up to 7 threads, the batched structure wins from 8
   (the E cores need the match-ahead), the dispatcher picks by pool
   width.
2. **The leftover debug getenv** in tsimd_match_o200k (25.5% of all
   cycles, called per piece): removed. **The scan 111.9 -> 276.0
   MiB/s (+147%); the sequential e2e 77.4 -> 128.6 (+66%).**
3. **The caller pin** (the unpinned caller landed on the worker's
   sibling core): **+14-20% everywhere.**
4. REVERTED: the two-tier merge table (the BPE's lookup mix is
   mostly misses; the extra tier-1 probe costs more than the
   hot-hit gain, the BPE cold path 46.4 vs 52.0 MiB/s).

Cumulative engine ladder (owt.txt, 476 MiB, 4 MiB docs, warm passes,
byte-exact; ours + gigatoken back-to-back, same machine state):

| threads | pre-run | run 10 | gigatoken 0.10.0 | gap |
| --- | --- | --- | --- | --- |
| 1 | 45.4 | **127.8** | 547 | 4.3x |
| 2 | 81.2 | 126.8 | 970 | 7.6x |
| 4 | 150.6 | 158.4 | 1536 | 9.7x |
| 8 | 230.3 | **416.9** | 2295 | 5.5x |
| 12 | 299.8 | 423.6 | 2009 | 4.7x |
| 16 | 326.2 | **549.3** | 2458 | 4.5x |

vs gigatoken 0.10.0: **4.3-4.5x at the widths that matter** (was
7.5-8x pre-run; 11.4x before run 8's correction). OPEN: 2 threads do
not scale (the serve path + the pool's wait machinery at low
widths). Details in docs/10_fused_tight_cache_scan_bpe.md.

## Run 11, the 2-thread scaling (2026-09-17): memory-latency-bound

The pool caches were capped at 1<<18 (10 MB) below the 256 MiB
budget at low widths; the cap raised to 22 (the budget bounds , 
width 2/4/8/16 now 80/40/20/10 MB). Within-run A/B (owt.txt,
byte-exact): **2t +7.4%, 4t +4.5%** (120.6/153.1 against 112.2/146.5
with the old cap). The scaling stays flat (2t 120.9 against 1t
106.4) and the same on a 16 MiB corpus (1t 110.2, 2t 102.8) , 
the 2-thread edge is MEMORY-LATENCY-bound: the serve path's hash
probes double at 2t and contend for the L3/DRAM completion. The
next experiment: batch only the lookups at the memory-bound widths.
Details in docs/11_2thread_scaling_memory_latency.md.

## Large-file comparison vs references (2026-09-15, final; low load 1.3)
## Run 69, the model-zoo encode matrix (2026-09-21)

Warm-pass encode throughput for every implemented family: the new
`bench_encode` tool (load once, mmap slice, 3 warm passes,
CLOCK_MONOTONIC, id-count checksum guard). Single thread,
`taskset -c 0`, this machine. MiB/s are input bytes/s.

| encoding | owt 2 MiB | owt 20 MiB | document 2 MiB | document 20 MiB |
|---|---|---|---|---|
| cl100k (baseline) | 114.2 | 113.1 | 101.8 | 91.1 |
| o200k |, | 370.6 |, | 276.3 |
| gpt2 |, | 130.3 |, | 103.8 |
| spm-BPE mistral7b | 2.9 | 2.8 | 3.6 | 3.1 |
| unigram xlmr | 0.4 | 0.4 | 0.4 | 0.5 |
| unigram t5 |, | 2.0 |, | 1.3 |
| gemma4 | 2.8 | 2.3 | 3.0 | 2.7 |
| nemotron | 119.1 | 110.8 | 33.2 | 30.3 |

Readings:
- **Nemotron matches the cl100k-class speed**: 119.1 vs 114.2 MiB/s
  on owt 2 MiB (faster), 110.8 vs 113.1 on owt 20 MiB (the same) , 
  a byte-level BPE over the O200K-family pretokens, the same pipeline
  class as cl100k. On the markup-heavy document corpus it drops to
  30.3 MiB/s (punct-heavy pretokens, more cache misses) but stays
  2.9x above the spm families.
- **The score-based families are 40-300x slower** (2.8-3.6 spm-BPE,
  2.3-3.0 gemma4, 1.3-2.0 unigram t5, 0.4-0.5 unigram xlmr): they are
  whole-chain score-based algorithms (the sentencepiece Viterbi / the
  codepoint-BPE chain) with no byte-level SIMD pretokenizer and no
  span split. Correctness-first: every family is byte-exact vs its
  reference at the parity scale (docs/66, docs/69).
- **The xlmr load costs 18.1-18.5 s** (the 256k-piece vocab build;
  t5's 32k load is 0.3-0.5 s), a known load cost, not correctness.

Reproduce: `taskset -c 0 ./build/bench_encode --format F [--scheme S]
--reps 3 --mib N MODEL [MODEL2] CORPUS`. Details in
docs/69_nemotron_parity_and_the_encode_benchmarks.md.

## Run 70, the full-dataset ladder, all covered models (2026-09-21)

Warm-pass encode throughput on the FULL corpora (owt.txt 476 MiB,
document.txt 495 MiB), `bench_encode --reps 2`, taskset -c 0-15, this
machine. MiB/s are input bytes/s. The batched keyed serve (run 70) is
the production path for the walker schemes; the whole-chain schemes
(spm/unigram/gemma/qwen3-NFC) take the sequential path under the pool
(the run-70 guards: seq == parallel exactly).

### owt.txt (476 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 158.9 | 690.2 | 680.3 |
| o200k | 313.8 | 743.8 | 810.8 |
| gpt2 | 190.1 | 777.3 | 753.2 |
| gemma4 | 2.0 | 2.0 | 2.0 |
| nemotron | 151.0 | 417.4 | 410.0 |
| deepseek | 86.7 | 360.8 | 389.4 |
| qwen3 (NFC) | 2.5 | 2.5 | 2.5 |
| glm4 | 146.9 | 473.7 | 448.6 |
| smollm2 | 124.2 | 407.1 | 345.5 |
| spm-BPE mistral7b | 2.1 |, |, |
| unigram xlmr | 1.7 |, |, |
| unigram t5 | 3.3 |, |, |

### document.txt (495 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 123.7 | 633.4 | 640.4 |
| o200k | 198.0 | 662.9 | 705.7 |
| gpt2 | 148.8 | 676.7 | 671.1 |
| gemma4 | 2.4 | 2.4 | 2.4 |
| nemotron | 26.8 | 32.1 | 32.4 |
| deepseek | 39.4 | 67.3 | 69.3 |
| qwen3 (NFC) | 2.4 | 2.4 | 2.4 |
| glm4 | 100.3 | 332.8 | 337.7 |
| smollm2 | 84.9 | 310.5 | 345.4 |
| spm-BPE mistral7b | 2.7 |, |, |
| unigram xlmr | 2.3 |, |, |
| unigram t5 | 3.9 |, |, |

Readings:
- **The batched keyed serve (run 70)** lifted the walker schemes
  +43-54% single-thread (cl100k 128 -> 172 on the 20 MiB slice;
  159 on the full corpus) and 8t/16t scale 3.3-4.5x.
- **The whole-chain schemes are 40-300x slower** by design: the
  sentencepiece merge replay / the Viterbi with no byte-level SIMD
  pretokenizer. The unigram gains from run 70 hold at the full scale
  (xlmr 1.7-2.3, t5 3.3-3.9).
- **qwen3 (NFC)**: the sequential fallback under the pool (the run-70
  guard); the NFC normalize is 30 MiB/s in isolation, the 2.4-2.5
  e2e is the NFC + the whole-chain-bounded path; the NFC parallel
  machinery is the follow-up.
- **llama3.2/minimax**: gated upstream (the reference files are
  "Invalid username or password." stubs), not benchmarked.
- **gemma4 under the pool**: the sequential fallback (the run-70
  guard, the seam-split parallel path produced wrong ids and was
  removed). 1t/8t/16t all 1.9-2.7.

Reproduce: `taskset -c 0-15 ./build/bench_encode --format F [--scheme
S] --reps 2 --threads N MODEL [MODEL2] CORPUS`. Details in
docs/70_batched_serve_and_speed.md.

## Run 72, the two encode shapes settled (2026-09-21)

The interleaved ladder (owt.txt 476 MiB, taskset -c 0-15, best of 2
warm) comparing the two shapes for the same schemes:

| scheme | shape | 1t | 8t | 16t |
|---|---|---|---|---|
| o200k | batch (4 MiB docs) | 393.4 | 999.6 | 1178.7 |
| o200k | parallel (one buffer) | 345.8 | 846.6 | 916.7 |
| cl100k | batch | 176.1 | 978.3 | 943.8 |
| cl100k | parallel | 164.8 | 820.1 | 770.1 |
| gpt2 | batch | 212.5 | 1065.8 | 1042.8 |
| gpt2 | parallel | 197.9 | 889.1 | 855.9 |

SETTLED: **the batch shape is +15-19% faster at 8t/16t, consistently**.
The ~1000 MB/s figure is the batch shape. The one-buffer parallel's
16t <= 8t; the batch's 16t ~= 8t; the scaling ceiling is the chip.

## Run 72, the unigram segment memo, opt-in (2026-09-21)

The memo (TSIMD_ENABLE_UNIGRAM_MEMO=1): the input split at the ws-run
starts, each unit probed in the keyed cache. The production default
is the whole-chain path (exact, the memo's DPs flip f32 ties at
corpus scale, 3.67M diffs on owt 20 MiB, documented).

| encoding | whole-chain | memo (opt-in) | gain |
|---|---|---|---|
| unigram xlmr (owt 20 MiB) | 2.0 | **64.6** | 32x |
| unigram t5 (owt 20 MiB) | 3.3 | **62.6** | 19x |

The simdutf micro-bench (vendored, the phase-B evidence): convert_utf8_to_utf32
3.2 GB/s (13x our per-char loop), count_utf8 14-30 GB/s (60-120x),
validate_utf8 17 GB/s, all below the 10% e2e gate for the current
hot paths (the unigram utf8 share ~1.6%, the NFC may_change ~0.2%).
The vendor + the micro-bench stay as evidence.

## Run 73, the full-dataset ladder after the safe-split fix, the fingerprint filter and the direct ccc table (2026-09-24)

Warm-pass encode throughput on the FULL corpora (owt.txt 476 MiB,
document.txt 495 MiB), `bench_encode --reps 2`, taskset -c 0-15, this
machine. The whole-chain schemes (gemma4, spm-BPE mistral7b, unigram
xlmr/t5) take the sequential path at every thread count (the run-70
guards); their 8t/16t rows were run once and repeat the 1t number.
qwen3 takes the NFC-fallback sequential path (the run-70 fallback);
its rows were re-measured after the direct ccc table landed.

### owt.txt (476 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 158.9 | 742.7 | 709.7 |
| o200k | 335.7 | 808.0 | 846.8 |
| gpt2 | 188.3 | 835.8 | 794.4 |
| nemotron | 151.6 | 427.3 | 425.2 |
| deepseek | 90.3 | 386.7 | 404.9 |
| glm4 | 150.8 | 488.7 | 471.3 |
| smollm2 | 127.0 | 452.2 | 417.8 |
| qwen3 (NFC) | 36.0 | 35.9 | 36.3 |
| gemma4 | 1.9 | 1.9 | 1.8 |
| spm-BPE mistral7b | 2.2 |, |, |
| unigram xlmr | 3.7 |, |, |
| unigram t5 | 5.6 |, |, |

### document.txt (495 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 116.8 | 585.1 | 589.9 |
| o200k | 189.8 | 639.0 | 663.3 |
| gpt2 | 140.7 | 622.6 | 622.8 |
| nemotron | 26.7 | 31.9 | 32.0 |
| deepseek | 40.2 | 68.5 | 69.8 |
| glm4 | 100.5 | 341.8 | 343.0 |
| smollm2 | 84.7 | 336.6 | 365.7 |
| qwen3 (NFC) | 33.9 | 34.2 | 34.1 |
| gemma4 | 2.5 | 2.5 | 2.5 |
| spm-BPE mistral7b | 2.7 |, |, |
| unigram xlmr | 4.6 |, |, |
| unigram t5 | 6.8 |, |, |

Readings against the run-70 ladder (same machine, same protocol):
- **qwen3 14x its run-70 baseline** (2.5 -> 36.0 owt, 2.4 -> 33.9
  document): the safe-split fix (2.5 -> 22.9) then the direct ccc
  table (22.9 -> 36.0). The NFC normalize's per-char binary search
  was the remaining cost.
- **xlmr 2.2x** (1.7 -> 3.7) and **t5 1.7x** (3.3 -> 5.6): the
  fingerprint pre-filter. The interleaved A/B (the same binary, the
  filter off then on): xlmr 1.7 -> 3.1 on the 8 MiB slice, t5
  3.1 -> 5.1.
- **The BPE families unchanged** (the pair table untouched; the
  byte-hash probes are rare there): cl100k 158.9 = 158.9, o200k
  313.8 -> 335.7, gpt2 190.1 -> 188.3, nemotron 151.0 -> 151.6,
  deepseek 86.7 -> 90.3, glm4 146.9 -> 150.8, smollm2 124.2 -> 127.0
  on owt.
- The phase map (bench_phases, owt 8 MiB, 1t warm best): the
  whole-chain DP is 96-99% of the encode (xlmr dp 3.3 MiB/s, t5 3.4,
  gemma4 2.8; the normalizer 0.4-3.6%); the deepseek scalar walker is
  83% of its warm encode (110 of 92 MiB/s), nemotron's 60% (285 of
  170), cl100k's SIMD walk 73% (248 of 181), gpt2's 67.5%; o200k's
  fused serve loop encodes at 444 MiB/s total. The next levers: the
  bitcannon port (cl100k-class + deepseek grammars; one cl100k
  grammar with the digit_cap knob covers cl100k/qwen/llama3/glm),
  then the parallel-with-normalizer machinery for qwen3.

Reproduce: `taskset -c 0-15 ./build/bench_encode --format F [--scheme
S] --reps 2 --threads N MODEL [MODEL2] CORPUS` and
`./scripts/run_ladder_73.sh` (the ladder). The phase map:
`taskset -c 0 ./build/bench_phases --format F [--scheme S] MODEL
[MODEL2] CORPUS`. Details in
docs/73_safe_split_fix_fingerprint_filter_and_the_hf_study.md.

## Run 74, the batch-shape ladder after C3 and the guards (2026-09-24)

Warm-pass encode throughput on the FULL corpora, encode_batch over
4-MiB docs (the run-72 Phase A shape), `bench_encode --reps 2`,
taskset -c 0-15, this machine. The whole-chain schemes take the
per-doc sequential fallback inside encode_batch (the run-70/74
guards); their 8t/16t rows were run once and repeat the 1t number.
qwen3 takes the NFC fallback (the run-74 guard); the batch id counts
differ from the single-buffer's at the 4 MiB cuts (the run-26
documented seam semantics).

### owt.txt (476 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 150.8 | 741.4 | 779.0 |
| o200k | 351.9 | 790.6 | 873.7 |
| gpt2 | 188.4 | 838.3 | 864.4 |
| nemotron | 144.9 | 383.3 | 420.0 |
| deepseek | 81.7 | 372.8 | 399.7 |
| glm4 | 138.8 | 225.1 | 235.3 |
| smollm2 | 70.5 | 218.1 | 178.2 |
| qwen3 (NFC fallback) | 36.0 | 35.9 | 35.9 |
| gemma4 | 2.6 |, |, |
| spm-BPE mistral7b | 3.0 |, |, |
| unigram xlmr | 3.6 |, |, |
| unigram t5 | 5.5 |, |, |

### document.txt (495 MiB)

| encoding | 1t | 8t | 16t |
|---|---|---|---|
| cl100k | 130.1 | 737.6 | 733.9 |
| o200k | 221.1 | 779.0 | 854.2 |
| gpt2 | 155.1 | 809.9 | 824.3 |
| nemotron | 27.7 | 32.7 | 33.5 |
| deepseek | 41.8 | 73.6 | 74.4 |
| glm4 | 110.3 | 402.7 | 410.9 |
| smollm2 | 92.4 | 406.6 | 430.6 |
| qwen3 (NFC fallback) | 33.3 | 33.6 | 33.5 |
| gemma4 |, |, |, |
| spm-BPE mistral7b | 4.0 |, |, |
| unigram xlmr | 4.5 |, |, |
| unigram t5 | 6.7 |, |, |

Readings:
- **cl100k single-thread 158.9 -> 150.8 on the batch shape** (the
  fused serve's C3: the batch shape's scalar-fill crossover differs;
  the single-thread full on the PARALLEL shape measured 392.8, the
  C3 gain is on the one-buffer shape).
- **o200k 335.7 -> 351.9** and **gpt2 188.3 -> 188.4** on the batch
  shape (the same as run 72's batch numbers).
- **qwen3 = the NFC fallback** (the run-74 guard): the sequential
  speed at every thread count, the correct ids (the batch's rows
  before the guard silently skipped the normalization).
- **The cold gap vs gigatoken (this machine)**: their CLI bench on
  o200k owt = 1103.56 MB/s (all cores); ours = 873.7 (the batch 16t)
  = 0.79x, 790.6 (8t) = 0.72x. Run 26's cold ratio was 0.63x (ours
  761, theirs 1211).

Reproduce: `taskset -c 0-15 ./build/bench_encode --format F [--scheme
S] --reps 2 --batch 4 --threads N MODEL [MODEL2] CORPUS`.
Details in docs/74_c3_cl100k_fused_serve_memo_offset_batch_nfc_guard.md.

## Run 75, the fresh gigatoken window and the memo parity (2026-09-25)

**The gigatoken on THIS MACHINE, re-run (2026-09-24)**: their CLI
bench, o200k owt (499.12 MB, all 10 cores, this machine):
**1103.56 MB/s**. Ours, same config: 873.7 (the batch 16t) = 0.79x,
790.6 (8t) = 0.72x. The run-51/52 reports' better numbers are the
interleaved batch ladders measured 2026-09-21 on this machine (the
run-52 table: the warm t1 1.07x, t16 1.02x with the spill landed;
the cold 0.64-0.71x), the machine state spans the windows; the
interleaved rows are the comparison.

**The memo parity (run 75)**: the offset-passing memo vs the
sentencepiece reference on owt 20 MiB: ref 5052919 ids, ours
5052919, 2 diff positions at 2508909/2508910, the SAME positions
and ids as the whole-input DP's own documented run-66 residual (the
reference's own f32 tie coin-flips). The whole-chain parity: the
same 2 positions. The memo-default flip was attempted and reverted:
the non-ASCII-whitespace cases (the NBSP, the ideographic space)
diverge (the memo's unit split breaks on multi-byte whitespace), so
the memo stays opt-in (TSIMD_ENABLE_UNIGRAM_MEMO=1: xlmr 16.5x,
t5 9.6x) and the whole-chain stays the production default.

Reproduce: `TSIMD_ENABLE_UNIGRAM_MEMO=1 python3 scripts/spm_parity.py
models/reference/xlmr_spiece.model testdata/owt.txt --limit-mib 20`
and `/home/raising-reuel/.local/bin/gigatoken bench
models/o200k_base.tiktoken testdata/owt.txt`. Details in
docs/75_memo_exactness_investigation_and_gigatoken_rerun.md.

## Run 76, the unigram memo as the production default (2026-09-25)

The offset-passing memo is the production default
(TSIMD_DISABLE_UNIGRAM_MEMO=1 = the A/B fallback). Warm-pass encode
throughput on owt 8 MiB, 1t, this machine: **xlmr 38.1 MiB/s** (the
whole-chain's 3.1 = 12.3x), **t5 36.1 MiB/s** (the whole-chain's 5.6
= 6.5x). The correctness: test_unigram 26+26 byte-exact; the 20 MiB
owt parity: the counts match, 2 diff positions = the whole-input DP's
own run-66 residual; the t5's parity residual is pre-existing and
carried identically by the whole-chain.

Reproduce: `taskset -c 0 ./build/bench_phases --format sentencepiece
--scheme spm --reps 2 --mib 8 models/reference/xlmr_spiece.model
testdata/owt.txt`. Details in
docs/76_memo_default_and_the_encoding_levers.md.

## Run 77, the deepseek/nemotron mask fills, the parallel NFC, the encoding sweep (2026-09-26)

The deepseek and nemotron mask fills landed (the gates, the ctest, the
wiring, the parity EXACT both corpora); the parallel-with-normalizer
landed (the chunked NFC pass, the parallel fork/join, the compact
pass; the affinity-inheritance fix). Batch-shape ladder, both full
corpora (owt 499 MB, document 495 MB), encode_batch over 4-MiB docs,
taskset -c 0-15, this machine. Throughput MiB/s (rep 1 warm best
unless noted):

| encoding | owt 1t | owt 8t | owt 16t | doc 1t | doc 16t |
|---|---:|---:|---:|---:|---:|
| cl100k | 174.7 | 932.4 | 928.3 | 130.7 | 758.0 |
| o200k | 400.7 | 982.8 | 1128.5 | 226.0 | 878.2 |
| gpt2 | 213.4 | 1067.5 | 1057.8 | 160.6 | 863.8 |
| nemotron | 160.4 | 515.7 | 511.2 | 28.1 | 33.7 |
| deepseek | 96.1 | 447.3 | 448.5 | 42.8 | 74.6 |
| glm4 | 162.8 | 616.2 | 591.1 | 109.2 | 414.9 |
| smollm2 | 126.4 | 541.1 | 549.8 | 95.0 | 437.7 |
| qwen3 (batch = sequential fallback) | 34.4 | 34.5 | 34.3 | 32.2 | 32.3 |
| gemma4 (whole-chain) | 2.8 | 2.8 | 2.8 | 3.5 | 3.4 |
| spm-BPE mistral7b (whole-chain) | 3.1 | 3.1 | 3.1 | 3.9 | 3.9 |
| unigram xlmr (whole-chain) | 30.1 | 30.0 | 29.8 | 14.0 | 14.0 |
| unigram t5 (whole-chain) | 31.2 | 31.1 | 31.2 | 15.9 | 15.9 |

The parallel-NFC rows (encode_parallel, NOT the batch, the batch's
NFC guard keeps the per-doc sequential fallback): qwen3 owt **1t
34.3, 8t 171.9, 16t 222.9** (5.0x/6.5x); document **1t 32.3, 8t
154.8, 16t 187.8**. The NFC pass alone: 8t 266-278 MiB/s (5.8x over
the sequential 46-52).

Correctness this run: deepseek + nemotron + qwen3 id parity EXACT on
both corpora (4-MiB slices, tokenizers as the reference); the suite
34/34 green. The deepseek P0 (the newline fusion in the arm-b SPACE
branch) fixed, the pretokens now match the reference exactly.

Reproduce: `bash scripts/run_ladder_74_batch.sh` (the log in
.ladder77.log). Details in
docs/77_deepseek_nemotron_mask_fills_parallel_nfc_and_the_encoding_sweep.md.
## Run 78, the batch routing fix and the o200k parity sweep (2026-09-26)

The batch shape's per-job dispatch now routes every fused-serve scheme
(O200K, CL100K, DEEPSEEK, NEMO) to fused_o200k_serve, the mask fills
run in the batch shape too. Batch-shape ladder numbers (499 MB owt,
4-MiB docs, taskset -c 0-15, rep 1 warm best):

| encoding | 1t | 8t | 16t |
|---|---:|---:|---:|
| cl100k | 422.6 | 863.1 | 996.7 |
| o200k | 402.6 | 998.1 | 1179.7 |
| gpt2 | 215.6 | 1113.3 | 1080.6 |
| deepseek | 302.0 | 511.6 | 567.2 |
| nemotron | 274.7 | 531.7 | 511.8 |
| glm4 | 353.8 | 635.1 | 629.5 |
| smollm2 | 127.7 | 532.6 | 540.4 |
| qwen3 (batch, parallel NFC) | 34.7 | 110.1 | 113.2 |
| **gigatoken (their bench, all cores, 1 pass)** |, |, | **1955.6** |

16t beats 8t for cl100k, o200k, deepseek, smollm2; ties within rep
variance for gpt2/nemotron/glm4. The A/B at 16t (glm4): the default
split (P-cores mask fill, E-cores+siblings scalar fill) 606.5-622.3
MiB/s is optimal; TSIMD_MASK_FILL=1 476.1-529.7 and
TSIMD_SCALAR_FILL=1 520.3-530.7 both lose.

The qwen3 batch row now takes the parallel-with-normalizer (the
per-doc normalized copies feed the pool path; run 78): 3.2x on owt,
3.5x on document. The parallel-NFC rows (encode_parallel): owt 16t
222.9; document 16t 187.8.

The gains from the batch routing fix (1t warm, run 77 -> run 78):
cl100k 174.7 -> 422.6 (2.4x), deepseek 96.1 -> 302.0 (3.1x),
nemotron 160.4 -> 274.7 (1.7x), glm4 162.8 -> 353.8 (2.2x).

The gigatoken comparison (fresh, this machine, the machine idle,
2026-09-26): their CLI bench o200k owt 499.12 MB = **1955.55 MB/s**
(one pass, all cores); ours best of 3, batch 4-MiB, 16t: **o200k
1439.2** = 0.74x, gpt2 1356.5, cl100k 1181.5, o200k 8t 1009.3. The
run-75 comparison (a busier window): theirs 1103.56, ours 873.7 =
0.79x, the ratio is stable across the machine states; the gap is the
cold band's per-piece miss path (the run-51/52 list stands).

Correctness: test_batch_fused (the batch ids == the sequential for
every fused-serve scheme on a 1-MiB doc); the suite 35/35 green.

Reproduce: `bash scripts/run_ladder_74_batch.sh` (the log in
.ladder78.log); the gigatoken row:
`taskset -c 0-15 ~/.local/bin/gigatoken bench
models/o200k_base.tiktoken testdata/owt.txt`. Details in
docs/78_batch_routing_fix_and_the_o200k_parity_sweep.md.

## Run 80, the gpt2 mask fill, the one-ahead precompute, the branchless short-merge, and the three-engine matrix (2026-09-27)

The mask fills' per-window debug counters (plain globals) were the
fill's scaling killer; they are env-gated now. The gpt2 (r50k) mask
scanner landed (all-mask pool default), the one-ahead window precompute
landed in all five fills, and the short-merge min-scan is branchless.
All gates green (38/38); the ids byte-exact vs tiktoken (o200k and
gpt2). Measured on this machine (the i5-13450HX), owt 476 MiB,
bench_encode --batch 16, best of 3 reps (the same-window gigatoken
numbers come from its own library via benches/encode_local.rs in the
gigatoken tree; hf-tokenizers via tokbench):

| scheme / engine | 1t cold | 1t warm | 8t cold | 8t warm | 16t cold | 16t warm |
|---|---|---|---|---|---|---|
| o200k (ours) | 289.3 | 358.7 | 978.2 | 1380.1 | 1116.9 | 1315.6 |
| cl100k (ours) | 303.2 | 388.1 | 979.4 | 1407.4 | 1034.8 | 1153.6 |
| **gpt2 (ours, after this run)** | **337.5** | **439.5** | 1143.9 | **1677.1** | 1328.9 | **1729.8** |
| deepseek (ours) | 151.8 | 182.8 |, |, | 848.1 | 1012.5 |
| nemotron (ours) | 260.7 | 334.6 |, |, | 985.1 | 1002.5 |
| gigatoken o200k 1t | 373-414 | 619-645 |, |, |, |, |
| gigatoken o200k all-core | 1741-1922 | 2251-2953 |, |, |, |, |
| gigatoken r50k all-core |, | 2765 |, |, |, |, |
| hf-tokenizers (tokbench, 1t, verified) | 5.6 | 5.6 |, |, |, |, |

The machine was throttled for part of the session (the Chrome tabs);
the same-window pairs are the meaningful comparison and the ratios
range 0.45-0.71x warm / 0.64-0.79x cold depending on the window. The
user's run-51/52 observation (the warm parity, the cold gap) is
inverted against the CURRENT gigatoken: its warm machinery widened
the warm gap. The details, the failed experiments, and the honest gap
statement: docs/80_gpt2_mask_fill_one_ahead_branchless_and_the_three_engine_matrix.md.

## Run 81, the dual-cursor scalar fill, the all-mask pool default, the final matrix (2026-09-27)

The pool's scalar-fill split (the run-38-era E-core rule) was re-A/B'd
with the run-80/81 machinery and REMOVED: all five fused-serve schemes
prefer all-mask now (o200k 16t 1818.8 vs 1572.3, cl100k 1902.9 vs
1357.6, nemo 1758.2 vs 1573.0, deepseek 1756.3 vs 1177.1). The scalar
fill gained the dual-cursor ILP (gigatoken's step 8; gate
test_dual_cursor): +6.3% at 8t, up to +14% at 16t, ~0% at 1t. The
probe mask-select was A/B'd and reverted (3.7% regression at 1t).
Parity: o200k byte-identical to tiktoken (FNV-1a dceb720ad60e3efe).
ctest 39/39.

| scheme (ours) | 1t cold | 1t warm | 8t cold | 8t warm | 16t cold | 16t warm |
|---|---|---|---|---|---|---|
| o200k | 320.8 | 388.2 | 1167.2 | 1670.7 | 1370.5 | **1847.6** |
| cl100k | 314.7 | 382.2 | 1186.6 | 1708.8 | 1377.3 | **1844.6** |
| gpt2 | 370.9 | **472.5** | 1088.4 | 1867.2 | 1349.2 | **1959.6** |
| deepseek | 290.0 | 339.6 | 1028.9 | 1530.7 | 1295.2 | 1665.6 |
| nemo | 302.2 | 364.3 | 1138.5 | 1550.4 | 1299.7 | 1722.5 |
| gigatoken o200k 1t (same window) | 461 | 679-689 |, |, |, |, |
| gigatoken o200k all-core | 1957 | 2799-3568 |, |, |, |, |
| gigatoken r50k all-core |, | 3059-3502 |, |, |, |, |
| hf-tokenizers (tokbench, 1t, verified) | 5.6 | 5.6 |, |, |, |, |

Same-window ratios: warm 0.52-0.66x, cold 0.70x. The run-80 numbers in
the section above were measured on a throttled machine (the user's
Chrome tabs, reported mid-run); the levers' isolated deltas are the A/B
tables in docs/81_dual_cursor_all_mask_defaults_and_the_final_matrix.md.

## Run 82, the deepseek P/S in-mask classify, the fused-fill batching (2026-09-27)

The deepseek mask classify moved the \p{P}\p{S} chars into the masks
(the punct runs) after the gate drove the design from the broad
letters+marks form down to the provably-safe narrow one (three
gate-caught divergence classes documented in the report; the letters'
path needs an arm-a-run-aware boundary rule, the documented next
stage). The o200k fused emitter now batches the whole batch's windows
per call (~+2% fill-only). Parity: o200k byte-identical to tiktoken;
ctest 39/39.

| scheme (ours) | 1t cold | 1t warm | 16t cold | 16t warm |
|---|---|---|---|---|
| o200k | 322.3 | 383.4 | 1243.1 | 1732.1 |
| gpt2 | 365.3 | 467.6 | 1295.3 | **1909.6** |
| cl100k | 308.8 | 368.8 | 1263.8 | 1762.8 |
| deepseek | 248.5 | 290.9 | 1068.9 | 1491.9 |
| nemo | 289.8 | 337.2 | 1204.6 | 1666.5 |
| gigatoken o200k 1t (same window) | 457 | 632-650 |, |, |
| gigatoken o200k all-core | 1846 | 2530-2985 |, |, |
| gigatoken r50k all-core |, | 2694 |, |, |

Same-window ratios: o200k 16t warm 0.58x, gpt2 16t warm 0.71x, cold
0.67x.

## Run 83, the batched emitters everywhere, the batch-4 ladder standard (2026-09-27)

The emitter batching extended to all five fills (the gates green); the
deepseek letters in-mask was measured (the 4.07% walker rate caps the
win at ~1%) and skipped with the design notes recorded; the batch
granularity standard is now 4 MiB (+4.8% at 16t, three consistent
same-window pairs). Parity: o200k byte-identical to tiktoken; ctest
39/39.

| scheme (ours) | 16t cold | 16t warm | gigatoken all-core warm | ratio |
|---|---|---|---|---|
| o200k | 1293.6 | 1865.2 | 2630-2849 | 0.65x |
| gpt2 | 1294.7 | 1959.6 | 3011 (r50k) | 0.65x |
| cl100k | 1299.9 | 1812.2 |, |, |
| deepseek | 1062.4 | 1439.2 |, |, |
| nemo | 1240.9 | 1652.2 |, |, |
| o200k document | 947.8 | 1298.4 | 1959 | 0.66x |


## Run 85, the blocked whole-chain BPE, the seam property, and the 1/6/10/16t ladder (2026-09-29)

The whole-chain families (Gemma-4, sentencepiece-BPE) gained the
**blocked, provably exact BPE**: the chain is cut at metaspace seams no
merge can cross (a seam between a non-▁ char and a ▁ is merge-proof
when no merge pair (a, b) exists with a ending there and b starting
there, the vocab-level coarse property plus a per-seam pair search),
so blocked == whole-chain ids (the gate `tests/test_bpe_seam.c`) and
the blocks run in parallel across the pool. Before: no parallelism at
all for these schemes (the run-70 seam-split finding) and 3.0/3.3
MiB/s at 1t.

document.txt, 64 MiB, MiB/s (bench_encode --reps 3):

| scheme | 1t | 6t | 10t | 16t | reference (1t) |
|---|---|---|---|---|---|
| gemma4 | 5.0 | 24.3 | 27.0 | 30.6 | HF tokenizers 0.23.2: 1.2 |
| mistral7b spm-BPE | 6.1 | 27.0 | 31.5 | 35.0 | (ids == sentencepiece) |

Ids: gemma 1t and 16t byte-identical to the HF reference (1 MiB
slice), mistral identical to sentencepiece, the blocked path identical
to the whole chain (the gate, 12 fuzz texts x 3 targets x 2 models).

The o200k ladder, 476 MiB, rep 0 cold / rep 2 warm, MiB/s (ours vs
gigatoken, same window, taskset, ours `--batch 4`; *gigatoken's rep 0
excludes the input load, it read()s the corpus before the timed loop,
ours mmaps and faults in rep 0*):

owt.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 247.3 | 332.4 | 401 | 614 | 0.54x |
| 6t | 886.4 | 1290.0 | 1472 | 2263 | 0.57x |
| 10t | 1112.9 | 1618.6 | 1618 | 2945 | 0.55x |
| 16t | 1007.1 | 1705.6 | 1732 | 2960 | 0.58x |

document.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 163.3 | 188.4 | 245 | 398 | 0.47x |
| 6t | 738.6 | 906.5 | 1007 | 1639 | 0.55x |
| 10t | 858.7 | 1124.2 | 1147 | 1781 | 0.63x |
| 16t | 885.8 | 1220.9 | 1142 | 2111 | 0.58x |

The 1t cost split on a 16-MiB document slice (warm): our raw keyed
o200k fill 770 MiB/s vs the gigatoken FastO200k pretokenizer 936 MiB/s
(0.82x; ours also packs the 128-bit key and issues the L2 prefetch);
the full encode 300 vs 425 MiB/s. Per piece (4.07M pieces per 16 MiB):
ours 5.1 ns fill + 8.6 ns serve/BPE, gigatoken 4.2 + 5.5. The gap is
in the serve/BPE, not the fill.

ctest 42/42 (the new test_bpe_seam is the 42nd). Parity: o200k and
cl100k byte-identical to tiktoken (run 84), gemma/mistral byte-exact
vs their references (this run).

## Run 86, the cache upgrade, the cold/warm levers, and the ladder with levers (2026-09-30)

The measured warm lever: the adaptive sequential pretok-cache upgrade
(1M -> 4M entries on the first >= 64 MiB buffer; owt 1t warm 360.5 ->
425.8 MiB/s, +18%). The cold path gained MADV_WILLNEED + MAP_POPULATE
(unmeasurable in a warm page cache, kept). A larger pool cache budget
was AB'd and REVERTED (no gain: 1707.9 / 1683.9 / 1641.0 MiB/s at
512K/2M/4M entries per worker, 16t).

o200k, 476 MiB, rep 0 cold / rep 3 warm, MiB/s, same-window gigatoken:

owt.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 300.7 | 420.4 | 406 | 609 | 0.69x |
| 6t | 942.3 | 1431.1 | 1425 | 2519 | 0.57x |
| 10t | 1105.1 | 1779.4 | 1631 | 3048 | 0.58x |
| 16t | 1120.4 | 1747.3 | 1688 | 3000 | 0.58x |

document.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 209.6 | 314.7 | 248 | 406 | 0.78x |
| 6t | 721.3 | 1029.9 | 1015 | 1830 | 0.56x |
| 10t | 853.6 | 1207.5 | 1198 | 1922 | 0.63x |
| 16t | 867.6 | 1335.3 | 1173 | 1671 | 0.80x |

The slow families (document 64 MiB, run 85's blocked/parallel seam BPE):
gemma-4 5.0/24.3/27.0/30.6 and mistral7b spm-BPE 6.1/27.0/31.5/35.0
MiB/s at 1/6/10/16t; ids byte-exact vs the HF/sentencepiece goldens.
o200k 16-MiB slice: 3,590,319 ids position-exact vs tiktoken.

## Run 87, the miss-path analysis and the merge-probe prefetch (2026-09-30)

The all-miss probe said the short-merge loop is the miss path
(per-symbol charge 6.7 -> 23.5 ns from 4- to 12-byte spans: the
dependent probe -> rank_to_id random-load chain). The landed lever is
`tsimd_vocab_merge_prefetch`: the next probe's line is issued at the
rank-init and re-probe sites, all-miss 68.0 -> 60.7 ns/span (-11%),
~1% end-to-end.

o200k, 476 MiB, rep 0 cold / rep 3 warm, MiB/s, same-window gigatoken:

owt.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 297.7 | 418.3 | 408 | 609 | 0.69x |
| 6t | 912.4 | 1490.1 | 1445 | 2340 | 0.64x |
| 10t | 1093.8 | 1749.0 | 1631 | 3067 | 0.57x |
| 16t | 1128.6 | 1751.9 | 1629 | 3157 | 0.55x |

document.txt:

| threads | ours cold | ours warm | giga cold | giga warm | warm ratio |
|---|---|---|---|---|---|
| 1t | 210.9 | 310.1 | 249 | 394 | 0.79x |
| 6t | 683.4 | 992.0 | 1014 | 1794 | 0.55x |
| 10t | 839.1 | 1169.7 | 1161 | 2158 | 0.54x |
| 16t | 857.2 | 1303.1 | 1147 | 1769 | 0.74x |

Our 10t == 16t (1749 / 1752): no E-core gain, the curve is flat at
~0.55-0.69x, the remaining gap is per-thread per-piece work (fill
5.3 vs 4.2 ns/span; serve/BPE ~8 vs ~5.5). Parity not reached; the
named targets are the merge probe's dependent `rank_to_id` load, a
two-tier pretoken cache, and the emit path.

## Run 90, the merged-id probe, the whole-chain block target, and the per-piece gap (2026-09-30)

Two landed changes and one re-aimed target. **The merged-id probe**
(`tsimd_vocab_merge_prio_id` + the slot-parallel `mgr_ids` /
`mgr_dense_ids` arrays): the sentencepiece win site called
`tsimd_vocab_merge_id`, a pair-table WALK, once per merge because the
collapsed score ranks make the priority -> id inverse ambiguous
(mistral7b: 97 tied entries). The probe answers priority and id from the
same level in one pass. Isolated probe (spm replay only, no fill/cache/
pool, 1 MiB document.txt, best of 3): gemma-4 4.54/4.80 vs 4.56/4.64
MiB/s after/before, exact ids (316789), effect within the machine's
spread; recorded as a strict work reduction, not a win. End-to-end id
hashes over the 16-MiB document slice: qwen2.5, smollm2, glm4 and
gemma-4 all unchanged. Gate: `tests/test_merge_id_probe.c` (synthetic
tie + 200k random pairs per model against the pair-walk oracle; ctest
43/43).

**The whole-chain block target** was the measured win. The replay's
per-byte cost grows with the block (isolated: 7.68 / 6.65 / 6.17 / 4.72
/ 3.89 MiB/s at 32 / 128 / 256 / 1024 / 4096 KiB), and the run-85
default was 256 KiB, never swept. The default is now 16 KiB
(`TSIMD_SEAM_BLOCK_KB` overrides, `TSIMD_NO_SEAM_BLOCK=1` disables).
Same-window paired runs, document.txt, `--reps 3 --mib 64` (1t:
`--mib 16 --reps 2`), taskset per row:

gemma-4 MiB/s:

| threads | 16 KiB (new) | 256 KiB (old) | gain |
|---|---|---|---|
| 1t | 6.1 | 5.1 | +20% |
| 6t | 34.5 | 23.4 | +47% |
| 10t | 40.4 | 26.7 | +51% |
| 16t | 51.4 | 32.1 | +60% |

mistral7b spm-BPE MiB/s:

| threads | 16 KiB (new) | 256 KiB (old) | gain |
|---|---|---|---|
| 1t | 7.4 | 6.1 | +21% |
| 6t | 33.3 | 29.2 | +14% |
| 10t | 48.3 | 32.0 | +51% |
| 16t | 56.7 | 38.3 | +48% |

Exactness: blocked == whole on 12 fuzz texts x 4 targets x 2 models
(16384 added to `test_bpe_seam`); the 16-MiB document slice hashes
identically for {1t, 16t} x {16 KiB, 256 KiB} on both models (gemma-4
`1b2e78a0...`; mistral7b `7ed70cee...`, 6,168,146 tokens). o200k/cl100k
untouched (16-MiB owt slice hash stable, o200k 3,590,319 ids).

The measurement that re-aimed the run: the ladder's gap is per-piece
work, not footprint, a repetitive 64-MiB corpus (rep64) reproduces the
ratio (ours/theirs 0.74x at 1t, 0.56x at 16t) exactly as the real
corpus does; the scalar fill is 2.8x/2.1x/1.6x SLOWER than the mask
fill at 1t/6t/16t; and the 10t -> 16t plateau is shared by both engines
(ours 1740 -> 1635, gigatoken 2941 -> 2922 on rep64), so the E-core
lever has no measured upside here. Also falsified: run 89's
attribution of the gap to the merge chain's `rank_to_id` load, the
tiktoken loader never sets ranks, so `id_of_priority` is a register
move there.

## Run 91, the per-byte fill, the heap sift, the phase probes (2026-09-30)

Four landed changes, each paired same-window against the run-90 tree,
plus the measurement that re-aims the next lever.

**The run-91 paired ladder** (A = this tree, B = the run-90 tree, one
window, MiB/s):

o200k, owt 476 MiB, `--batch 4 --reps 3`:

| threads | A | B | delta |
|---|---|---|---|
| 1t | 375.3 | 373.0 | +0.6% |
| 6t | 1227.3 | 1164.8 | +5.4% |
| 10t | 1458.2 | 1354.2 | +7.7% |
| 16t | 1365.6 | 1288.7 | +6.0% |

o200k, document.txt 476 MiB: 1t 288.5 vs 287.8 (+0.2%), 6t 867.6 vs
797.1 (+8.8%), 16t 977.9 vs 924.8 (+5.7%).

The slow families, document.txt 64 MiB (1t on 16 MiB): gemma-4 7.9 vs
6.8 / 41.7 vs 35.5 / 65.8 vs 57.8 MiB/s at 1/6/16t (+16/+17/+14%);
mistral7b 9.3 vs 7.9 / 46.5 vs 39.5 / 73.4 vs 63.8 (+18/+18/+15%).

**The landed levers.** (1) The serve's hit counter moved out of the
probe into a batch-local register: 1t +1.35% (5/5 pairs), 16t +4.8%
(3/3). (2) The fused fill shape everywhere (the key pack inside the
walk): a four-variant A/B on rep64 gave C2 1450.4 vs A 1374.2 at 6t
(+5.5%, 4/4) and 1575.7 vs 1517.3 at 16t (+3.8%, 4/4); the per-piece L1
prefetch was A/B'd too and is neutral, so it stays. (3) The merge heap's
8-way select is branchless and its grandchild prefetch index is fixed
(`8 * best + 1`, it had been the 4-ary `4 * best + 1` since the heap
changed arity): gemma-4 1t 6.1 -> 6.9 (+13%), mistral7b 1t 7.4 -> 8.6
(+15%), 16t +9..31%.

**The measurement that re-aims the classify work.** The fill is bound by
BYTES, not pieces: two 64-MiB corpora at 3.78 and 2.0 B/piece give
4.34-4.47 and 4.45-4.55 fill cycles per BYTE, i.e. the fill is
~4.2 cyc/byte + 0.5 cyc/piece. The window classify + carries + mask
algebra is the fill (59-64% of the warm path per the new
`TSIMD_PHASE_STATS` probe), and the per-piece key pack/walk/emit is
negligible at natural density. The spm probe (`TSIMD_SPM_STATS`) gives
the whole-chain replay's split: merge loop 84-86%, heap build 10-11%,
init 4-5%.

**Methodology caveat.** The same binary read 1747-1758 MiB/s at 16t in
the 05:0x window and 1365.6 at 05:47 (same corpus and flags), the
package sits at 85 C under sustained benchmarking and the governor is
`powersave`. Absolute ladder numbers drift up to ~25% between windows;
only paired same-window deltas are evidence, and the gigatoken ladder
needs one single window to be meaningful.

Exactness: all id hashes unchanged from run 90, o200k `c8fdcf23…`,
cl100k `a8fbfce3…` (16-MiB owt slice), gemma-4 `1b2e78a0…`, mistral7b
`7ed70cee…` (6,168,146 tokens) at both 1t and 16t. New gate:
`tests/test_bpe_heap_oracle.c` (the naive-rescan oracle for the heap
path); ctest 44/44.

## Run 92, the phase-by-phase profile, the invalid-lead fix, the crc32 hash (2026-10-01)

The run profiled both engines in the same window with the same probe
shape (TSC cycles around the fill and around the probe+emit), which is
the first apples-to-apples phase comparison against gigatoken on this
machine. 256 MiB owt, o200k, 1t, warm:

| | fill | serve | total |
|---|---|---|---|
| TokenSIMD | 16.7-17.2 (58%) | 11.6-12.3 (42%) | 28.3-29.5 |
| gigatoken (their tree + an rdtsc probe in /tmp/gt_prof) | 12.2-12.4 (61%) | 7.7-7.9 (39%) | 19.7-20.3 |
| gap | 5.0 | 4.5 | 9.5 |

Their FastO200k pretokenizer alone (`cargo bench --bench pretok_local` in
their tree, our slice, 1t) is 53405989 pretokens at 1158 MB/s
(0.9 ns/byte), so their fill costs only ~0.2 ticks/byte more than their
bare span walk. Ours: the ablation inside `fused_put` puts the per-piece
pack+hash at 4.8 cyc/piece and the L2 prefetch at 0.3.

The baseline ladder (one window, warm, MiB/s, `taskset` per row):

| threads | ours | gigatoken | ratio |
|---|---|---|---|
| 1t owt | 445.2 | 653.0 | 0.68x |
| 6t owt | 1617.7 | 2641.3 | 0.61x |
| 1t document | 331.5 | 424.4 | 0.78x |
| 6t document | 1155.0 | 1560.9 | 0.68x |

**The landed changes.** (1) A P0 correctness fix: the o200k mask
classifier took the character length from the lead byte, so an invalid
or truncated multi-byte lead (`0xff`, `0xe6 0x20`) marked the next 1-3
window bytes as a bogus letter and suppressed the real boundaries after
them, the mask path merged "9" + "hb" into one span. The other four
mask schemes got this fix in run 74; o200k was missed. New gate: 15
invalid-lead patterns x 81 window offsets in `test_o200k_masks`. The
16-MiB id hashes are unchanged on the corpora (valid UTF-8 is not
affected). (2) The crc32c bucket hash is inline again: `bpe.c` carries
`-msse4.2` and the arm inlines to two crc32 uops (run 54 made the
multiply the default when the arm was an outlined call). Measured
paired: serve 11.3 vs 11.7-11.8, e2e 449.6-450.8 vs 442.3-443.9 MiB/s
at 1t, ids identical.

**The reverted experiments.** The BMI1/BMI2/LZCNT flags on the mask TUs:
no effect (fill 16.9-17.1 vs 17.0-17.2). The branchless serve
(gigatoken's `probe_emit_chunk` shape: unconditional 4-lane store,
branchless home-pair probe, one cold branch): 12.6-12.8 vs 11.6-11.8
cyc/piece in the serve, 423.9-427.9 vs 434.0-440.7 MiB/s e2e, 3/3 pairs
,  the route mix shows the fast path served 99.9% of pieces, so the four
unconditional value loads cost more than the branches they remove. The
branchless key pack with a hoisted 16-byte bound: a wash.

**The final paired ladder** (both trees in one window, 256 MiB owt,
o200k, `--reps 3`), A = the run-92 start tree, B = this tree:

| threads | A | B | delta |
|---|---|---|---|
| 1t | 442.4 / 441.9 | 454.5 / 449.0 | +2.2% |
| 6t | 1428.6 / 1534.8 | 1564.9 / 1581.4 | +1.7% / +3.0% |

Parity is not reached: 0.70x at 1t and 0.61x at 6t against the same-window
gigatoken numbers. The full 1t/6t/10t/16t ladder was not run, per the
instruction to run it only at parity.

Exactness: ctest 46/46; o200k `c8fdcf23…` and cl100k `a8fbfce3…`
(16-MiB owt slice, 1t and 6t, SAME through every change); qwen2.5
`95ac549c…`, gemma4 `924a8796…`, mistral7b `ff04be8f…` all SAME.

## Run 93, the per-piece pack and the serve ablations (2026-10-01)

**Landed: the fill's per-piece pack, fixed from the disassembly.** The
`objdump` of the fill TU showed the hash arm being re-resolved per
piece (`mov state,%eax; test; je <init>; cmp $1; je <out-of-line arm>`,
with a getenv call and a seq_cst store in the init arm forcing spills)
and the pack-mask table's address re-materialized per piece
(`lea masks(%rip)`). Both are gone: the arm is resolved once per fill
and threaded down through `mask_emit_fused`/`fused_put` as a parameter,
and the table base is pinned through an asm empty (gigatoken's
`black_box`). Applied to all five mask schemes.

Measured paired, one window, 256 MiB owt, o200k, 1t: median e2e +1.9%
(2/3 pairs), fill −0.3 to −0.8 cyc/piece (16.6/16.7/17.0 -> 16.5/16.2/
16.2). cl100k: +0.4%/+1.2%. The final 2-round 1t/6t ladder against the
same frozen tree is a wash inside this window's noise.

**The serve ablated (scratch tree, compile-time switches):**

| serve variant | fill | serve |
|---|---|---|
| full | 17.0 | 12.1 |
| no L1 prefetch | 16.5 | 11.4 |
| no probe (constant hit, store kept) | 16.7 | 4.8 |

The probe is 7.3 of the serve's 12.1 cyc/piece. A runtime sweep of the
prefetch distance in one window (D = 0/4/8/16/32/64) gives 12.2/12.2/
12.1/12.4/12.6/12.8, the prefetch is neutral and long distances hurt,
so the probe is NOT L1-latency-bound.

**Tried and reverted: the probe view** (gigatoken's `ProbeView`: base +
slot mask held in registers for the chunk). Both the struct and the
scalar-parameter forms made the serve 2.5x slower (11.4 -> 27.9-28.5
cyc/piece) with the route mix unchanged, the extra live values pushed
GCC into a spill storm (`addq $1,0x78(%rsp)` for the hit counter, cursor
and base reloads in the fast path). Reverted with the numbers recorded.

Exactness: ctest 46/46; o200k `c8fdcf23…` and cl100k `a8fbfce3…`
(16-MiB owt slice, 1t and 6t) unchanged.

## Run 94, the provable guard and the pre-folded bucket mask (2026-10-01)

Two provable micro-wins, both measured paired in one window (256 MiB
owt, o200k, 1t): (1) the fused emission's `bpos > pos` guard is provably
true (non-crossed: `need << 1` puts every bit at >= k+1 so
`bpos >= pos+1`; crossed: every bit is `>= scan > pos`) and is gone;
(2) every probe's address chain now starts with `(hash << 1) &
pair_mask` using a pre-folded `pair_mask = mask & ~1` carried in the
cache struct, the identity `((h & (mask>>1)) << 1) == ((h<<1) &
(mask & ~1))`, one op less on the head of the probe, the same value
gigatoken's ProbeView carries.

| round | A (reverted) | B | delta |
|---|---|---|---|
| 1 (throttled) | 343.5 | 357.2 | +4.0% |
| 2 | 427.3 | 439.9 | +2.9% |
| 3 | 422.3 | 434.3 | +2.8% |

3/3 pairs. Fill 17.3/17.5 -> 16.8/17.0, serve 11.9/12.0 -> 11.6/11.8.
cl100k 1t +0.3..+1.7%; 6t +2.4% in the un-throttled pair.

The first form of (2) dropped the `<< 1` and was a DIFFERENT bucket
function; `ctest` caught it immediately in `test_insert_slot` (the two
insert paths' tables diverged) and `test_tokenizer`
(`tsimd_pcache_lookup` returned NULL for seeded entries). Fixed; 46/46.

**Tried and reverted: the per-window mask precompute.** The one-ahead
stash only ever covered a fill's first window, so the gather was
extended to every window (gigatoken's per-batch precompute). A/B against
`TSIMD_NO_PRECOMPUTE=1`: 17.3-17.4 vs 17.2 fill, 424.5-427.9 vs
426.5-427.6 e2e, a wash. The reorder window already covers the mask
chain on this core.

Exactness: o200k `c8fdcf23…`, cl100k `a8fbfce3…` (16-MiB owt slice, 1t
and 6t), mistral7b `ff04be8f…`, gemma4 `924a8796…` all unchanged.

## Run 95, the cross-engine phase split (2026-10-01)

The gigatoken tree was restored (fresh clone; the /tmp copy had been
removed by the system cleaner) and instrumented to split their fill into
phase A (boundary harvest) and phase B (key pack + entry write), the
same split our probe reports. 256 MiB owt, o200k, 1t, TSC cyc/piece:

| engine | fill | = A | + B | serve | total |
|---|---|---|---|---|---|
| gigatoken (rep 1) | 12.0 | 7.6 | 4.1 | 7.3 | 19.3 |
| TokenSIMD, two-phase (their algorithm) | 26.0 | 13.5 | 8.4 | 11.4 | 37.4 |
| TokenSIMD, fused (production) | 17.1-17.5 |, |, | 11.8-12.1 | 29.0-29.6 |

Our implementation of their exact algorithm is 2.2x slower than theirs
(26 vs 12), which rules out algorithm selection: a faithful port lands
further from them than our own fused design does. Their phase B does
~16 uops/piece in 4.1 cycles (IPC ~4); ours does the same work in 8.4
(IPC ~2).

**Retried from the earlier reports' misses** (all landed, all measured):
the harvest's `*nb` pointer round-trip made a local (A 14.2 -> 13.5,
~5%); phase B's cursor, hash arm and pack table localized/pinned like
run 93 did for the fused path (B unchanged at 8.4); the per-piece
`mask_stats_on()` gate, a global load whose untaken arm holds a getenv
call, hoisted once per fill in all five mask schemes (unchanged); the
key written with one 16-byte store instead of two 8-byte stores
(gigatoken's entry shape). The two-phase remains 1.5x worse than the
fused path, so the fused shape stays the default. The BMI flags, the
branchless serve, the branchless pack and the ProbeView have recorded
negative measurements from runs 92-94 and were not retried.

**Machine-load caveat (binds every number above the final line):**
partway through the run a browser held ~0.83 of a core and the package
reached 87-93 C; interleaved same-window rounds then differed by up to
2x (426/425/394 MiB/s for one binary on one core) and the gigatoken arm
fell from 629-633 to 581-585 MB/s. Only the phase splits (taken before
the load appeared) are trusted; the micro-deltas after it are recorded
as inconclusive.

Final paired window (both engines, same conditions): **ours 424.7 /
414.0 MiB/s against gigatoken's 554 / 558 MiB/s, 0.77x / 0.74x at 1t
warm**, against 0.68x at the start of run 92. Exactness: ctest 46/46;
o200k `c8fdcf23…`, cl100k `a8fbfce3…`, mistral7b `ff04be8f…`,
gemma4 `924a8796…`, qwen2.5 `95ac549c…` unchanged.

## Run 96, the window mask priced, the flatten retried (2026-10-01)

The run-92 mask ablation changed one engine's piece density and could
not be subtracted; this run reused window 0's (usable, bad) for every
window in BOTH engines, so the density shifts identically and the mask
cost falls out. 256 MiB owt, o200k, 1t, TSC cyc/piece:

| engine | normal | masks free | mask cost |
|---|---|---|---|
| TokenSIMD | fill 17.9-18.2 | 12.5 | **5.4 cyc/piece = 69 cyc/window** |
| gigatoken | fill 11.9 (A 7.2 + B 3.9) | 11.0 (A 1.1 + B 9.2) | **6.1 cyc/piece = 78 cyc/window** |

**Our classify + carries + algebra is 12% cheaper than theirs.** The
gap is in the rest of the fill: ours 12.5 cyc/piece against their 5.0
(phase A remainder 1.1 + phase B 3.9).

Instruction counts of the AVX2 classify, from both sources: ours 68
vector ops and 18 movemasks in one inlined body; theirs 34 + 13 vector
ops and 18 movemasks, with `ascii_extra_avx2` a separate
`#[inline(never)]` call per window. The counts match the measurement.

**The flatten, retried in the production fused path** (its three earlier
attempts were in the two-phase harvest), both paths in one binary behind
`TSIMD_NO_FLATTEN`: flatten 27.5/27.9/28.0 vs per-bit 18.0/18.0/17.9
fill, **3/3 pairs, a 55% regression**. The dense-array round trip costs
more than the `tzcnt`→`blsr` chain it removes. Fourth measurement of the
same conclusion; the per-bit loop stays.

**Methodology failure, recorded:** the store-preserving no-pack ablation
read its flag through a per-piece cross-TU call, which cost ~11
cyc/piece by itself (the same build's normal arm read 29.4 against 18.0
without the probe). Its "pack = 2.5" result is meaningless, and run 92's
"pack = 4.8" is an upper bound for the same reason. Both ablations were
removed before the run ended; the tree is back to its run-95 state and
ctest is 46/46 with all id hashes unchanged (o200k `c8fdcf23…`, cl100k
`a8fbfce3…` at 1t and 6t).

## Run 97, the pack priced correctly (2026-10-01)

The run-96 pack ablation was invalidated by its own per-piece probe call;
this one carries the flag in a register (resolved once per fill) and
keeps the entry stores, so the store traffic is identical in both arms.
256 MiB owt, o200k, 1t, interleaved: normal fill 17.2/17.4/17.8 vs pack
removed 13.1/13.3/13.9, **the pack is 4.1 cyc/piece**. The fill now
decomposes completely:

| component | ours | theirs |
|---|---|---|
| classify + carries + algebra | 5.4 | 6.1 |
| key pack + hash + prefetch + routing | 4.1 | ~3 |
| pop loop + entry stores + control | **7.9** | ~2 |

**Hoisting the pack's 16-byte bound check to the window** (and dropping
`d`/`len` from the call, targeting the stack reload of the entry base
that objdump shows in the pop loop) measured 18.3/18.4/18.5 against the
per-piece form's 16.9/17.1/17.3, **3/3 pairs, ~7% worse**. Reverted.
That is the fifth emission restructuring to lose; the two that won were
the hash-arm hoist (run 93) and the pair_mask fold (run 94).

Final paired window on a quiet machine: **ours 456.2 / 459.3 MiB/s
against gigatoken's 612.3 MiB/s = 0.745x / 0.750x at 1t warm** (0.68x at
the start of run 92). ctest 46/46; o200k `c8fdcf23…` and cl100k
`a8fbfce3…` unchanged at 1t and 6t.

## Run 98, the flatten root cause: our jump table (2026-10-01)

The flatten lost for us four times (runs 45, 46, 55, 96) and wins for
gigatoken. The line-by-line read settles it: **our
`tsimd_o200k_flatten_bounds` compiles to a jump table, an indirect
branch per octet (8 per window), plus a conditional per octet, a bounds
check per octet and scalar u32 stores. Theirs is 8 unconditional 16-byte
vector stores of `BIT_POS[b] + base` with the SWAR exclusive prefix as
the offset and no branch at all.** The run-96 explanation ("the round
trip costs more than the tzcnt chain") was wrong.

Their body was transcribed to C (`tsimd_bitpos8[256][8]` +
`tsimd_flatten_bits16`), gated by a unit test first (60,000 masks
bit-identical to the reference), then measured in the fused emission
behind `TSIMD_FLAT=1`:

| arm | fill |
|---|---|
| their flatten + counted emission | 17.3 / 17.4 / 17.5 |
| per-bit pop loop (default) | 16.5 / 16.5 / 16.8 |

**+9.7 cyc/piece with our flatten becomes +0.8 with theirs.** The
residual 0.8 is the position buffer's traffic, which the fused shape pays
and their two-phase shape does not. The per-bit loop stays the default;
the vectorized flatten stays in the tree, tested, as the right body for a
two-phase harvest (which still uses the old jump-table version).

Also measured neutral: `fused_put` now takes the buffer end instead of
`(d, len)` (one fewer live register in the pop loop, whose objdump shows
the entry base reloaded from the stack every iteration). Kept.
ctest 46/46; o200k `c8fdcf23…`, cl100k `a8fbfce3…` unchanged at 1t and 6t.

## Run 99, the full-key sentinel and the key-first entry (2026-10-01)

Two candidates from run 97/98's list, both landed.

**The full-key emptiness sentinel.** An entry was empty iff `key0 == 0`,
which made a span with eight leading NUL bytes uncacheable and put a
`k0 == 0` special case in every fill and a `k0 != 0` test in the serve.
Empty is now the FULL 16-byte key; a long span is marked `k0 = 0, k1 = 1`
(impossible for a real short span, whose length tag is never 0); the
serve's route test becomes "has a length tag" (`k1 >> 56 != 0`). Gate
first (`test_insert_slot`: a NUL-led span must insert and probe back, it
was red on the old code), then ~12 sites. Fill 16.5-16.8 -> 16.3-16.4
cyc/piece (the pack's NUL branch and prefetch branch are gone).

The change exposed a real hazard: the walking insert's private path still
searched slots with `key0 == 0`, which under the new sentinel is true for
a NUL-led entry, it would have overwritten it. `test_insert_slot` caught
it (five table mismatches between the two insert paths); fixed to the
full-key test. The shared table's CAS claims a slot by writing key0, so
NUL-led spans stay BPE-only there (one-line guard; the shared table is an
opt-in mode).

**The key-first entry.** `tsimd_pcache_span_t` went from
`{ptr, k0, k1, meta}` to `{k0, k1, ptr, meta}` + `aligned(16)`: the serve
loads the key with ONE aligned 16-byte load instead of two 8-byte loads,
and the pack's key store is one aligned store. Measured 3/3: fill
16.3-16.4 -> 16.1-16.2, serve 11.1-11.3 -> 11.0-11.1, e2e 450.4-455.8 ->
**458.7-459.0 MiB/s**.

**Final paired ladder, one window, quiet machine:**

| threads | ours | gigatoken | ratio |
|---|---|---|---|
| 1t | 455.6 / 462.3 MiB/s | 612.3 / 608.5 MiB/s | 0.744x / 0.760x |
| 6t | 1641.5 / 1654.8 MiB/s | 2514 / 2446 MiB/s | 0.653x / 0.677x |

ctest 46/46; o200k `c8fdcf23…`, cl100k `a8fbfce3…` (1t and 6t),
mistral7b `ff04be8f…`, gemma4 `924a8796…` all unchanged.

## Run 100, the parallel gap and the unseeded worker caches (2026-10-01)

1t sits at 0.75x but 6t at 0.67x, so the run profiled the parallel path.
Per-worker counters showed every worker equally loaded and equally
missing: at 2t each worker missed 0.27% of its pieces against the
sequential cache's 0.03%, and at 6t the aggregate BPE rate was 20-40x the
1t rate while the serve cost 18.9 cyc/piece against 11.0.

**Landed: the pool workers' caches are seeded with the vocabulary.**
`tsimd_pcache_seed_vocab` was called only for the sequential cache; the
`worker_t.seeded` guard had existed since run 50 with nothing calling it.
Paired A/B at 6t, one binary (`TSIMD_NO_WSEED=1`): **seeded 1328.8 /
1284.4 MiB/s against unseeded 1257.4 / 1241.4, +5.7% / +3.5%, 2/2**,
with the cold pass's hit rate better too (98.1% vs 97.5%).

**Landed: the pool cache budget 256 -> 1024 MiB**, 6 workers now get
4M-slot tables (128 MB, the sequential path's size) instead of 2M slots.
Measured +2% at 6t in its window.

**Ruled out:** capacity. A 4x budget did not move the rep-1 miss rate,
which pointed at the seeding and at the doc-to-worker assignment.

**Still open:** the assignment is a dynamic atomic queue, so each worker's
cache converges toward the WHOLE corpus's unique set (1.90% -> 0.51%
misses over four reps) instead of its own share; a stable partition , 
gigatoken forks each worker with a cache sized to its share, would
converge in one pass. And the serve's 15-33 cyc/piece at 2-6t against
11.0 at 1t is memory contention over 6 x 128 MB tables, which these two
changes do not address.

Final ladder (one window, thermally loaded): 1t 0.748x / 0.762x,
6t 0.673x / 0.669x. ctest 46/46; o200k `c8fdcf23…`, cl100k `a8fbfce3…`
unchanged at 1t and 6t.

## Run 101, the static doc-to-worker partition (2026-10-02)

Run 100 left the doc-to-worker assignment as the open item: a dynamic
atomic queue meant every worker touched every part of the corpus, so each
worker cache converged toward the whole unique set (0.51% BPE misses at
6t after four reps against 0.03% sequential). This run gives every pool
participant a stable, contiguous, weight-balanced range of the job list,
with steal-from-the-tail so uneven cores still self-correct.
`TSIMD_PAR_STATIC` selects the arm: 0 = the dynamic queue, 1 = static,
2 = static + steal (default).

**Arms A/B at 6t, 256 MiB owt, o200k, one window, three rounds x four
reps, median MiB/s per rep index:**

| rep | arm 0 dynamic | arm 1 static | arm 2 static+steal | arm2 vs arm0 |
|---|---|---|---|---|
| 0 (cold) | 514.0 | 484.2 | 535.3 | +4.1% |
| 1 | 1250.9 | 1541.4 | 1660.8 | +32.8% |
| 2 | 1435.3 | 1545.9 | 1636.7 | +14.0% |
| 3 | 1496.9 | 1526.0 | 1694.9 | +13.2% |

**The mechanism, from `TSIMD_PHASE_STATS=1`** (the BPE rate is the
pretoken-cache miss rate; serve is cycles per piece): the static arms
reach the sequential rate in ONE pass, 1.90% (cold) -> 0.03% -> 0.03%
-> 0.03%, while the dynamic arm reads 1.90% -> 1.04% -> 0.74% -> 0.50%
after four. The 6t serve cost falls from 26.4-19.9 to 13.7-13.2 cycles
per piece, against 11.0 at 1t. Per-worker lines (six workers, warm rep):
99.88-99.90% short-hit and 0.023-0.030% BPE, a spread of 0.007 points , 
each cache now covers its own range.

**Steal matters.** Static without the steal (arm 1) is not a reliable
win, 0.973x-1.031x against the dynamic arm in an earlier window, because
one straggler stalls the generation. Arm 2 wins every warm rep of every
window, and at 10t (mixed P/E) it beats arm 1 by +12% (1.230x vs 1.093x),
at 16t by +12% (1.250x vs 1.113x).

**Ladder against gigatoken, this machine, one window** (256 MiB owt,
4-MiB docs, `taskset` per the project convention): 1t ours 429.4-436.6
against their 599.9-608.6 = 0.710-0.719x; 6t ours 1448.5-1706.6 against
their 1686.2-2287.3 = 0.746-0.752x (reps 2-3). Scaling from that window:
ours 432.4 -> 1705.9 = 3.95x on six cores, theirs 608.6 -> 2287.3 =
3.76x, our parallel scaling is now better than theirs, so the 6t-specific
gap is closed and the 6t ratio matches the 1t ratio. Run 100 read
0.673x / 0.669x at 6t in a different (thermally loaded) window; the
paired arms table above is this run's evidence.

**The cold pass gains little** (+4.1% at 6t, -3.7% at 16t): it is
dominated by pool creation, the vocabulary seed, and page faults.

Exactness: ctest 47/47 (the new `test_par_split` is the 47th); o200k
`c8fdcf23…` and cl100k `a8fbfce3…` unchanged at 1t and 6t; TSan clean on
`test_par_split` and `test_parallel`; the negative control (the owner's
atomic claim replaced by a plain load) made the probe report
`slot 16 ran 2 times`.

Still open: the 1t gap (0.71-0.72x), the fill and the assemble at 1t,
untouched by this run, and the ten-encoding cold/warm ladder.

## Run 102, the parallel pool cache init and the plong counter race (2026-10-02)

Runs 100/101 left the serve's parallel cost over six big tables, and the
static partition shrank each worker's working set to about 1/width, so
the 4M-slot tables looked oversized. The size sweep answered with a
different root cause, and that root cause is the landed change.

**The sweep** (`TSIMD_PCACHE_BITS` caps the pool tables; paired A/B, 6t,
256 MiB owt, o200k, 2 rounds x 4 reps, median MiB/s): 4M slots (128 MB)
592.5 / 1710.4 / 1741.5 / 1797.5, 2M 673.5 / 1681.2 / 1700.1 / 1817.5,
1M 720.5 / 1676.1 / 1677.0 / 1652.2, 512k 763.9 / 1549.0 / 1567.8 /
1585.5. Smaller tables buy the cold pass (+13.7% to +28.9%) and lose the
warm reps (-2% to -12%), so the cold gain was not capacity. A 4 MiB
input (encode ~4 ms) isolates it: rep 0 costs 0.223 s at 4M slots,
0.176 s at 2M, 0.147 s at 1M, the difference scales with the table
bytes at ~7.6 GB/s, a memset. The pool build was serially pre-faulting
and seeding every worker's table on the creating thread: ~0.22 s of the
cold rep's 0.43 s at 6 x 128 MB.

**Landed: the worker caches are built on the worker threads** , 
allocate, THP hint, pre-fault, and the vocabulary seed, each on its
owner thread before it parks. The pre-fault stays ahead of the first
generation (the mmap-lock reason it exists). A worker whose build fails
records `init_rc = -1` and still parks; pool_create checks every
participant after the barrier, so a NULL cache never reaches a drain.
`TSIMD_SERIAL_POOL_INIT=1` is the A/B knob, `TSIMD_POOL_INIT_FAIL=<idx>`
the fault injection for the failure-path test.

Paired A/B at 6t, one binary: the 4 MiB setup isolate reads 17.7 MiB/s
(0.226 s) serial against 39.8 MiB/s (0.101 s) parallel, **2.251x** , 
with the encode rep at 1.007x. On 256 MiB, **rep 0 551.0 -> 730.9
MiB/s (+32.6%)** and the warm reps 0.952x / 1.012x / 0.989x. The 6t
cold ratio against gigatoken moved **0.447x -> 0.598x** in the paired
ladder of this window (ours 724.6 against their 1211.0); warm 0.762x /
0.787x (run 101's window read 0.746/0.752). 1t is untouched (the pool
is not used there): 0.708-0.721x warm, this window's drift.

**Landed: the plong counters are atomic.** The TSan build reported 99
warnings at `bpe.c:1500`, the three plain globals `plong_hits` /
`plong_misses` / `plong_inserts` are bumped from every pool worker's
long-piece probe. Pre-existing (the counters date from the long-span
table), benign, and a real C11 race. Now `_Atomic size_t` with relaxed
adds; the long route is 0.08% of the pieces. TSan is clean end to end
(0 warnings), and the post-change ladder (1t 0.725x/0.754x; 6t 0.584x
cold, 0.911x/0.814x warm) shows no cost.

**Ruled out / reverted:** smaller tables (above; the 1024 MiB budget
stays); narrowing `p->width` on a partial spawn (it leaked the
unspawned workers' workspaces, the NULL guard in the stats is the
right fix); warming the sequential cache to stabilize the test's
insert baseline (the 1M -> 4M upgrade runs in the parallel entry, so the
test measures differentially across pool widths instead). Run 101's
"TSan clean" claim was wrong: I had read only the tail of the log. The
gate now captures the whole log.

Exactness: ctest 47/47; the new `test_pool_cache_init` asserts the seed
per worker (the negative control `TSIMD_NO_WSEED=1` reads 2 instead of
536) and the init-failure path; o200k `c8fdcf23…` and cl100k
`a8fbfce3…` unchanged at 1t and 6t.

Still open: the 1t gap (0.71-0.72x, the emission's store count was not
attempted), and the ten-encoding cold/warm ladder.

## Run 103, three levers priced on the 1t gap and the 6t residue (2026-10-02)

Run 102 left the 1t emission's store count and the 6t serve residue.
Three candidates were priced with paired measurements. **None is a
lever; no code change landed.** The 6t residue is now proven memory
traffic.

**1. The emission's per-piece arm branch + spills, falsified.** The
disassembly shows the production path per piece: the hash-arm flag
reloaded from a stack slot, a taken branch to an out-of-line crc32
path, three spills of k0/k1/h, and a stack reload of the cache pointer
(~13 uops). Forcing the crc32 arm at compile time (removing all of it)
gained +0.1% / +0.2% / +0.1% (two binaries interleaved, 1t, 64 MiB).
The fill is NOT issue-bound, consistent with runs 96-98's four failed
emission attempts.

**2. The serve's table size, smaller is worse at 1t too.** Sweep at 1t
(64 MiB, 3 interleaved rounds, phase probes): 4M slots serve 11.6/11.7/
11.7 cyc/piece, 2M 11.9/12.2/12.0, 1M 13.1/12.6/12.3, and at 512k the
BPE miss rate rises 0.03% -> 0.29%. The sequential table fills to its
3/4 load limit and probe chains lengthen; the 4M default stands at both
widths.

**3. The serve's L1 prefetch, a wash, and the knob costs more than the
prefetch.** Deleting `serve_prefetch_l1(c, batch[j+16].meta)` read
+0.8-2.2% (5/5 pairs) in two windows. A same-window three-way
(prefetch on / off via a runtime flag / line removed outright) showed:
on 444.4/447.5/443.0, flagged-off 440.1/440.6/436.6, hard-removed
452.1/444.7/444.3, the removal is a wash, and the runtime flag is the
slowest in 3/3 rounds (~1.5%): the hook costs more than the change it
guards. Reverted; no knob added.

**4. The 6t residue is memory, proven.** Same binary, same window, 6t:
on a 16 MiB corpus (hot sets L3-resident) the serve is 11.1 cyc/piece
and the fill 16.9, exactly the 1t cost (11.6 / 16.2), against 13.9 /
19.3 on 256 MiB. The residue is DRAM traffic per piece (the 32-byte
entry stores against ~5 input bytes per piece, plus the probe's line
fetches), not a code shape.

Lessons recorded: a serve-loop A/B hook must be priced itself (this one
lost 1.5%); and a 5/5 cross-binary win of ~1% did not reproduce in a
same-binary three-way, so small deltas need the arm-toggle design.

Exactness: ctest 47/47; o200k `c8fdcf23…` unchanged at 1t and 6t
(all temporary experiment edits reverted and verified gone).

Still open: the entry layout (32 -> 24 bytes per piece) as the only
structural lever the data supports; the 1t gap (fill 16.2 against their
11.9, start from the classify or the mask algebra, not the emission);
the ten-encoding ladder.

## Run 104, every open lever priced, the cross-check, the ten-encoding ladder (2026-10-02)

**Six levers priced and closed, no code landed** (all paired, 1t,
256 MiB owt, o200k): the straight-line serve (-1%), the keys-first
4-load probe (a wash), the branchless cmov probe in gigatoken's exact
inline-asm shape (**-5%, 3/3**), the ProbeView hoist (-1%), the fill's
L2 staging prefetch removal (fill -1.4 but serve +5.5, essential,
priced), plus run 103's two. Our fill and serve are insensitive to
instruction-level structure; they are sensitive to the table access.

**Premise validated:** their FastO200k on the same slice yields
53,405,989 pretokens against our 53,406,115 (0.0002%), the per-piece
comparisons are apples-to-apples.

**The decisive cross-measurement** (their instrumented tree, staging
prefetch disabled): their serve 7.3 -> 9.9-10.1, ours 12.7 -> 17.8. The
gap tracks the *touched* table footprint: their Heaps-sized table holds
~580k entries (18.6 MB, L3-resident) against our 1-2M (~25-32 MB). That
is the one structural lever still standing; the slot count alone is
ruled out by our own sweeps.

**The ten-encoding cold/warm ladder** (128 MiB owt, one process at a
time; cold = rep 0, warm = best): o200k 1t 284/462, 6t 595/1564 (3.4x);
cl100k 288/468, 678/1595 (3.4x); gpt2 328/506, 745/1637 (3.2x);
deepseek 240/360, 432/**822 (2.3x)**; nemo 216/301, 378/**605 (2.0x)**;
gem4 7/7, 7/7; spm 8/8, 8/8. **New finding: deepseek and nemo lose
their 6t scaling** (serve 18.5-18.6 cyc/piece against o200k's 13.7, and
0.08-0.12% BPE against 0.03%), the same memory mechanism.

**Cross-engine ladder extended to cl100k:** 1t 0.66-0.67x (worse than
o200k's 0.71-0.75x), 6t 0.71-0.79x, cold 0.53x.

Exactness: ctest 47/47 after every revert; o200k `c8fdcf23…` unchanged
at 1t and 6t. An earlier ladder attempt ran two ladders concurrently on
the same pinned cores and is recorded as invalid (numbers 2x low).

## Run 105, the serve decomposed, the footprint lever corrected (2026-10-02)

**The three-state serve decomposition** (two temporary builds + their
tree patched, 1t, 256 MiB owt, o200k, phase probes):

| state | our serve (cyc/piece) | their serve |
|---|---|---|
| no probe (fake hit, no table read) | 4.70 | 3.3-3.4 |
| fixed line, always hit (perfect L1, no compares) | 6.50 |, |
| the real probe | 11.9-12.7 | 7.3 |

Ours: 4.7 loop+emit machinery, 1.8 probe code + one L1 line, 5.4-6.2
real table access. Theirs: 3.4 machinery, 3.9 probe. The no-probe
ablation also prices the ceiling: 4.7 cyc/piece = 593 MiB/s at 1t.

**Run 104's footprint lever is FALSIFIED (correction).** Our 1t serve
against the working-set size, same binary and window: 12.0 (16 MiB),
12.3 (32), 11.6 (64), 11.9 (256 MiB), flat. The serve is not
footprint- or bandwidth-bound; the memory level is hidden by the loop's
overlap. The staging prefetch's +5.5 when removed is about staying
inside that overlap budget, not L3 residency. Gigatoken's Heaps sizing
is withdrawn as the named lever.

**The register hoist (their ProbeView, done properly) is a wash:**
12.2/12.2 against 11.8/12.0, GCC already hoists the base, because our
probe loop's stores cannot alias the entry type and our table never
reallocates (theirs can). Run 104's view test was invalid (a struct
passed by pointer does not hoist).

Recorded invalid measurements: the first fixed-line patch hit a
different probe function (measured the baseline), the second measured
the miss walk, and one run overlapped gigatoken's bench on the same
pinned core (3x low).

Exactness: ctest 47/47; o200k `c8fdcf23…` unchanged at 1t and 6t. No
code landed. Still open (now a specific question): why our probe's
access costs ~5.5-6 cycles per piece where their whole probe costs 3.9,
when the layout, the pair probe, the staging and the footprint are
equal, the achieved overlap of the probe's chain is the remaining
variable, and the three-state harness makes it measurable.

## Run 106, the serve's chain measured (2026-10-02)

Four states, each one line apart (1t, 256 MiB owt, o200k, phase probes):

| state | serve cyc/piece |
|---|---|
| no probe at all (fake hit) | 4.70 |
| fixed line, always hit (loads hoisted) | 6.50 |
| **real address, two loads, no compares/walk/counter** | **9.80** |
| the real probe | 11.9 |

**The hash-dependent address and its load cost 5.1 of the 7.2 cycles**
(9.8 - 4.7); the compares, the walk and the hit counter cost 2.1
(11.9 - 9.8). Gigatoken's entire probe costs 3.9, so the chain is the
whole remaining difference. Note the second row is the loop with the
loads *hoisted* (a constant address), not a measurement of eight L1
loads.

Ruled out: the memory level (the serve is flat 11.6-12.3 across 16-256
MiB, run 105, the chain's exposure is the same at any level);
`#pragma GCC unroll 2` on the serve loop (11.9-12.0 against 11.8-12.0, a
wash); the compares/walk/counter (2.1 total). Together with the seven
neutral shapes from runs 103-105, the remaining difference is in the
chain's code generation.

Named next lever: hand-schedule the serve's probe+emit in inline asm
(gigatoken escalated to asm for exactly this chain in their own x86 port
plan); their cmov shape alone did not transfer, a full asm block that
also controls the load scheduling is the untried variant, and the
four-build harness makes it a single measurement.

Exactness: ctest 47/47; o200k `c8fdcf23…` unchanged at 1t and 6t. No
code landed.

## Run 107, the hand-scheduled AVX2 serve hot loop (LANDED, 2026-10-02)

Run 106 localized the serve's remaining gap to the code generation of
the probe's chain. This run hand-schedules it in a C++17 AVX2 TU
(`src/serve_x86.cpp`, `-mavx2`, runtime-gated behind
`tsimd_simd_level() >= AVX2` and `TSIMD_SERVE_AVX2`): the pair's keys as
two 16-byte loads, two `pcmpeqq` + movemask, ONE `blendv` for the
val/ext selection, the next piece's address computed while the current
loads are in flight, the same prefetch distance and lane pack as the C
path; the cold routes (walk, long, miss) hand back to C.

**Paired A/B in one binary, 256 MiB owt, o200k:** 1t serve 13.4-13.6 ->
**11.8-12.0 cyc/piece** and BEST +3.5% to +7.8% (**4/4 rounds**: 416/414/
400/406 -> 435/428/431/431); 6t BEST 1800/1722 -> **1847/1769** (2/2,
+2.6%).

**The arms test caught a real bug first.** The first version read
`_mm_movemask_pd`'s bits as "entry matched", but a key is two 64-bit
lanes and the mask has one bit per lane, a single-lane match emitted
another entry's ids (1.68% BPE misses against 0.03%; the bench's
checksum guard saw the id count change between reps). Fix: AND the two
lanes per entry. Recorded as the test-first rule paying for itself.

**A locked add on a shared line cost 4.5% at 6t.** The first 6t
measurement read -4.5%; the cause was the test counter's per-call atomic
add taken by six workers on one cache line. Publishing once per serve
call fixed it (and the regression vanished).

Gates: ctest 47/47; TSan clean (0 warnings); id hashes unchanged at 1t
and 6t for o200k and cl100k; the arms test asserts byte-identical ids
across the arms on the batch and single-buffer parallel paths with
specials, long runs and non-ASCII. Ladder in this window: 1t
0.733-0.738x, 6t warm 0.769-0.850x, cold 0.586x.

Still open: the serve's ~3 cycles per piece above the 4.7 no-probe floor
(extend the hand-scheduled block over the cold walked routes), and the
fill (16.2 against their 11.9) which this line of work has not touched.

## Run 108, the walk inside the hand-scheduled hot loop (2026-10-02)

The run-107 AVX2 loop handed **2.5-3.7%** of pieces back to C (80x the
BPE miss rate): the chained table never displaces, so pieces whose home
pair was occupied live in a later pair, and each took a stop + a C probe
+ a C emit + a re-entry. This run moved the walk inside the loop (retry
`set = (set + 2) & mask`, terminating on an empty pair via a PTEST):
the handoff rate fell to **0.087%** (the long route 0.06-0.08% plus
genuine misses 0.02-0.04%, confirmed by a temporary reason breakdown).

**Measured equal on speed**, five paired rounds at 1t, 256 MiB owt,
o200k: C arm fill 16.5-17.2 / serve 13.0-13.6 / BEST 421-423; AVX2 arm
fill 16.5-16.7 / serve **11.8-11.9** / BEST **437-443** (+3.6% to
+4.8%, 5/5), the same delta as run 107's walk-less version. At 6t at
least level (1743 vs 1742; 1856 vs 1724).

Kept on the handoff-rate reduction with equal measured speed (the
revert-by-hand was the higher-risk edit for zero gain).

Gates: ctest 47/47; TSan 0 warnings; o200k `c8fdcf23…` and cl100k
`a8fbfce3…` unchanged at 1t and 6t.

Still open: the **fill** (16.5-17.2 against their 11.9), now the
largest single gap and untouched by this line; the per-piece put (key
pack, hash, entry store, prefetch) is the schedulable block there.

## Run 109, the fill decomposed: the mask algebra is the largest item (2026-10-02)

**Run 98's leftover is closed (it loses).** The vectorized flatten
(`tsimd_flatten_bits16`) swapped into the two-phase harvest's two per-bit
loops, paired at 1t on 256 MiB owt o200k (the `TSIMD_TWOPHASE_FILL=1`
arm): before fill 18.80/18.90/19.00 (BEST 409/407/406), after
19.40/19.50/19.40 (402/403/402), **3/3 worse**, reverted. Suspected
cause: the flatten's store->load round trip through the u16 array; the
pop loop stays in registers. Consistent with runs 45/46/55 and run 98's
fused-path result (+0.8). The leftover is now closed in both paths.

**The fill's decomposition** (a one-line ablation: keep the piece count
and pointer exact, write the long marker instead of the pack/hash/
prefetch; the serve still produces correct ids):

| state | fill cyc/piece |
|---|---|
| the real put | 16.8 / 17.6 |
| no pack, no hash, no prefetch | **13.4 / 13.7** |

So the **mask machinery (classify + carries/algebra + precompute +
boundary walk) is 13.4-13.7 cycles per piece** (~170 per 12.7-piece
window) and the put is only 3.1-4.2. Against their A 7.6 (including
their classify 6.1, so their algebra+bounds ≈1.5) and B 4.0: our put
matches their B, our classify is 12% cheaper (run 92), and **our algebra
+ walk is ~8 cycles per piece against their ~1.5, the largest single
remaining item in the campaign.**

Next: ablate inside the machinery (classify+precompute / +algebra /
+walk) with the same method, then attack the step carrying the ~100
unexplained cycles per window; the AVX2 method that won the serve fits
the 64-bit mask algebra directly.

Machine note: the session's thermal drift showed the same fill code at
16.8-17.6 early and 21.3 later; only same-window paired numbers are used.
Exactness: ctest 47/47 after the reverts; o200k `c8fdcf23…` unchanged at
1t and 6t. No code landed.

## Run 110, C17 + C++20, and the inline-asm question (2026-10-02)

**Standards moved to the newest stable pair the toolchain fully
supports.** GCC 13.3: C17 complete, C++20 complete; `-std=c23` does not
exist here (only the experimental `-std=c2x`) and C++23 is partial, so
C11 -> **C17**, C++17 -> **C++20** landed, with the rationale in CMake.
Verification (the bump changes codegen in every TU): build clean,
ctest 47/47, id gates unchanged at 1t/6t (o200k + cl100k), TSan 0
warnings, and a paired A/B against the pre-bump binary: BEST
440/429/438 -> 440/436/443, equal or better.

**The inline-asm question, answered by the disassembly.** The landed
hand-scheduling is AVX2 intrinsics (`src/serve_x86.cpp`); the compiled
hot path is the intended schedule and in two places better than typical
asm: `vpcmpeqq 0x20(%rdx),%xmm3,%xmm0` folds e1's key load into the
compare, and `vpextrq $0x1,%xmm0,0x8(%r8)` writes the second output lane
from the extract itself. Literal asm was tried once (the run-104 cmov
probe) and measured 5% slower; the remaining costs are the pair loads'
latency and the emit, not instruction selection.

**One item the disassembly exposed:** the per-piece `addq $0x1,(%r12)`
hit counter (a store-forwarding chain). Dropped (the caller counts
`stop - j`, equivalent by construction), measured a wash (serve
11.5/11.5/11.6 vs 11.5/11.5/11.7), kept as a simplification; the last of
run 108's temporary `dbg_reasons` plumbing was removed in the same edit.

Still open (unchanged from run 109): the fill's mask machinery , 
13.4-13.7 cyc/piece against their A's ~7.6.

## Run 111, the fill's mask pipeline measured (2026-10-02)

A new instrument (`TSIMD_FILL_PROBE=1`, permanent, off by default) times
the fill's window phases with rdtsc. Per 64-byte window (about 12.7
pieces), 256 MiB owt, o200k, 1t:

| phase | cycles |
|---|---|
| classify + the one-ahead precompute | **110.4** |
| the algebra | **120.3** |
| the boundary walk + the put | **15.2** |
| sum (matches the fill + the probe's ~30/window) | 245.9 |

**The fill is bound by the mask pipeline; the emission is shadow.** One
measurement now explains run 106's "13 uops removed, 0.2% gained", runs
96/97/98's four neutral emission restructurings, run 109's put ablation
(the OoO spread the put's work into the pipeline's stalls) and five
flatten losses (work added to a shadow region can only cost). Their
phase A (classify + algebra + bounds) is ~96 cycles/window, **2.4x
faster than our 230**.

**Retried failed idea, with the new toolchain:** their register-resident
`MaskState` precompute vs our struct round trip, the register-local
variant measured **3/3 worse** (fill 16.7/16.9/16.7 vs 16.2/16.2/16.2,
BEST -1.5%), reverted. The discrimination: **the put is uop-insensitive
(shadow), the mask region is uop-sensitive (critical path).**

Next target: `window_usable_bad`, its unconditional 96-byte
`memset(&u, 0, sizeof(u))` per window is unused on pure-ASCII windows
(the algebra's u-terms are zero by construction); then the sub-split of
the classify and the algebra with the same instrument.

Gates: ctest 47/47; o200k `c8fdcf23…` unchanged at 1t and 6t.

## Run 112, the ASCII algebra specialization (LANDED, 2026-10-02)

The run-111 probe pointed at the mask pipeline's algebra (120
cycles/window) and the memset in `window_usable_bad`. **Landed: a
specialized pure-ASCII algebra**, the same body with all 13 unicode
terms as compile-time zeros (GCC folds them), dispatched on `m.hi`, so
pure-ASCII windows also skip the 96-byte memset and the unicode classify.

Paired, alternating arms, 1t, 256 MiB owt, o200k:

| round | before fill/BEST | after fill/BEST |
|---|---|---|
| 1 | 17.00 / 447 | **15.90 / 457** |
| 2 | 16.60 / 444 | **16.10 / 453** |
| 3 | 16.30 / 450 | **16.10 / 452** |
| 4 | 16.60 / 449 | **15.90 / 457** |

**4/4**, fill -0.4 to -1.1 cyc/piece, BEST +0.4% to +2.9%; a wash at 6t
(1838/1844 vs 1858/1831). The first positive fill change in ten runs.

**Hygiene:** the run-111 probe cost 1.8% even when off (four predictable
per-window branches; the mask region charges ~0.6 cycles per uop). It is
now compile-time gated (`-DTSIMD_FILL_PROBE_BUILD=1`) and verified free;
the run's first measurement (a cmov-guarded variant) was confounded by
that cost and is recorded as never cleanly priced.

The discrimination, now paid in both directions: adding uops/branches to
the mask region costs; removing the unicode terms and the memset from
the ASCII path pays. Every earlier fill attempt worked on the put or the
emission, the shadow, and was correctly neutral.

Gates: ctest 47/47; TSan 0 warnings; o200k `c8fdcf23…` and cl100k
`a8fbfce3…` unchanged at 1t and 6t. Still open: the classify+precompute
(110/window), the algebra's remaining ~100, and the same specialization
for the other four mask schemes.

## Run 113, the ASCII algebra for all five schemes (LANDED, 2026-10-02)

Run 112's pure-ASCII algebra specialization replicated to the other four
mask schemes, all had the identical structure, so the transformation is
mechanical (hoist the scheme's `u->` terms, generate `X_algebra_ascii`
with them as compile-time zeros, dispatch on `m.hi` with a guarded
memset/classify). Term counts: cl100k 11, gpt2 8, nemo 13, deepseek 10.

Paired, alternating arms, 1t, 256 MiB owt: **cl100k fill 16.0/15.8/15.8
-> 15.3/15.3/15.3, BEST 446-450 -> 451-454 (+0.7% to +1.3%, 3/3);
deepseek fill 16.5 -> 15.3, BEST 359 -> 372 (+3.6%); gpt2 fill 12.4 ->
12.1, BEST 503 -> 509 (+1.2%); nemo a wash (17.4/301 vs 17.6/302, its
windows are likelier non-ASCII).**

Gates: all five schemes' equivalence tests (masked fill vs the scalar
matcher) pass; ctest 47/47; o200k `c8fdcf23…` and cl100k `a8fbfce3…`
unchanged at 1t and 6t; gpt2 `35ad3c17…` identical at 1t and 6t; TSan 0
warnings.

Still open: the classify + precompute (110 cyc/window on o200k) and the
algebra's remaining ~100, with the compile-time-gated fill probe as the
instrument (`-DTSIMD_FILL_PROBE_BUILD=1`).

## Run 114, the classify sub-split (2026-10-02)

The run-111 instrument extended to five regions and built with
`-DTSIMD_FILL_PROBE_BUILD=1` (1t, 256 MiB owt, o200k, 7.3M windows):

| region | cycles/window |
|---|---|
| classify call (`window_usable_bad`), average | **74.1** |
|, ASCII classify | 21.9 |
|, the carries (`o200k_carries`) | 25.2 |
|, the unicode path (averaged over all windows) | 27.0 |
| the outer classify region (window start → precompute) | **164.5** |
| the algebra | **123.4** |
| the walk + put | reads 0 (a probe bug from run 112's TLS conversion; the run-111 value 15.2 stands) |

**Most suspicious single item: the carries at 25 cycles/window** for at
most nine bytes read, one ~80-byte memset and a short tail scan. The
unicode path averages 27 over all windows (non-ASCII windows pay the
memset + a per-byte classify). The ASCII classify is 22 for 64 bytes
(reasonable). **~90 cycles/window are unattributed** between the window
start and the precompute (the stash handling, the scan arithmetic, and
the OoO attribution of the classify's latency); the next probe build
times the stash pickup and the precompute write separately.

Gates: ctest 47/47; o200k `c8fdcf23…` and cl100k `a8fbfce3…` unchanged at
1t and 6t; the probe is compile-time gated (11 guards) and absent from
the default build.

## Run 115, the carries read: 88 bytes of struct for seven bits (2026-10-02)

The run-114 queue's first item, read line by line. `o200k_carries` fills
an 88-byte `ocar_t` per window: a `memset`, seven single-bit u64 stores
(all derived from the class of `d[w-1]`), a `c2_os_at()` call and four
or five branches, about twenty memory operations, at 25.2 cycles per
window (run 114), to hand the algebra seven booleans, one small enum and
two flags that are a pure function of one byte.

Cheap variants ruled out by reading: field stores without the memset
(`pcl`, `b2b_in`, `p_abs`, `force_bad_lead` would be garbage), a
constant struct assignment (same stores), a 256-entry `ocar_t` table
(22.5 KB, an equally expensive copy).

**Fix design for the next run:** a `uint8_t carry_class[256]` table plus
a `{ uint8_t class; uint8_t c2_os; uint8_t flags; }` carries struct , 
about two stores instead of thirteen memory operations, with the algebra
reading the bits from the class byte. Estimated 10-13 cycles per window
(~0.8-1.0 cycles per piece). Correctness-critical: needs a test-first
run gated by the five schemes' equivalence tests and the id hashes.

No code landed this run. The queue stands: the carries shrink, the
unattributed ~90/window, the algebra's pure-ASCII ~100, and the walk+put
probe slot.

## Run 116, the carries shrink for o200k (LANDED, 2026-10-02)

Run 115's design, implemented for o200k: a bit-packed
`tsimd_o200k_cls[256]` class table (pl/pu/ps/pwt/po/pws/pd in one L1
byte) and a three-byte `ocar_t` (`cls`, `c2_os`, `flags`) replace the
88-byte struct, its memset and its seven single-bit u64 stores per
window. The algebra's prologues derive the bits from `cls`; reading the
setters showed **o200k never sets `pcl` or `b2b_in`**, they were dead
memset-zeroed reads and are now constants.

Paired A/B, alternating arms, 1t, 256 MiB owt, o200k: **4/4**, fill
15.80/15.90/15.90/15.80 -> 15.60/15.50/15.50/15.50, BEST 471/470/468/470
-> 472/475/476/475 (+0.4% to +1.7%).

Gates: `test_o200k_masks` (masked fill vs the scalar matcher), ctest
47/47, o200k `c8fdcf23…` unchanged at 1t and 6t, TSan 0 warnings.

The other four schemes carry the same shape (17/7/17/17 `c->` writes)
and the recipe is in the report; they were not converted in this run.

## Run 117, the carries shrink for cl100k: neutral (LANDED, 2026-10-02)

Run 116's recipe applied to cl100k: `tsimd_cl100k_cls[256]` (same bit
layout), `ocar_t` 96 -> 4 bytes (`cls`, `c2_os`, `osf_in`, `flags`), the
plain carries path becomes one class-table load, both algebra prologues
derive their six class bits from `cls`. Found in passing: `pcl` and
`b2b_in` are never set (dead memset-zeroed reads) and **`pu` is set but
never read**, cl100k's algebra has no case differentiation.

**Measured NEUTRAL:** paired, 1t, 256 MiB owt cl100k: fill
14.90/14.90/14.90 -> 14.80/14.90/14.90, BEST 477/478/477 -> 478/477/474
(3/3 a wash). Kept on the run-108 reasoning (measured-neutral, removes
dead members, unifies the two busiest schemes' carries shape); o200k's
0.3-0.4 cyc/piece gain did not reproduce here, and why is an open
question (cl100k reads 6 class bits against o200k's 7 plus `osf_in`, and
its per-window total is lower).

Gates: `test_cl100k_masks` OK, ctest 47/47, cl100k `a8fbfce3…` unchanged
at 1t and 6t.

Process lesson recorded: the first attempt wrote the table and struct and
then aborted on a count assertion, leaving a broken intermediate state
whose tests ran against a stale binary. Redone scoped to the carries
function with every count verified before the write.

## Run 118, the carries shrink for gpt2: inconclusive window (LANDED, 2026-10-02)

`tsimd_gpt2_cls[256]` + a **2-byte** `ocar_t` (`cls`, `flags`), gpt2
never sets `pu`, `pcl`, `c2_os`, `b2b_in`, `osf_in` or `p_abs` (its
carries' own comment says why), so the old 96-byte struct was a memset
and six single-bit stores of pure class information. Both algebra
prologues derive the six bits from `cls`.

**Measured INCONCLUSIVE:** the before arm alone swung 12.4-14.9 cyc/piece
across three alternating rounds on a 128 MiB slice (session thermal
drift), so no ratio is evidence; the cleanest round read +1.6%. Recorded
as inconclusive, not as a win; a quiet-window re-measurement is queued.

Gates: `test_gpt2_masks` OK, ctest 47/47, gpt2 `35ad3c17…` identical at
1t and 6t.

## Run 119, the whole run-118 queue (LANDED, 2026-10-02)

**The carries shrink for nemo and deepseek landed** (o200k- and
cl100k-shaped carries respectively; their own class tables; 3- and
4-byte `ocar_t`s; `pcl`/`b2b_in` dead in both). Gates green.

**The consolidated measurement, one clean window** (paired, 1t, 128 MiB
owt, each scheme against its pre-shrink binary):

| scheme | fill before -> after | BEST before -> after | delta |
|---|---|---|---|
| o200k | 15.8 -> 15.6 | 470-471 -> 476 | +1.1-1.3% |
| cl100k | 15.0-15.1 -> 14.8-14.9 | 474-476 -> 480-481 | **+1.1-1.3%** |
| gpt2 | 12.4-12.9 -> 12.0 | 507-508 -> 512-513 | **+0.8-1.0%** |
| deepseek | 15.4-15.6 -> 14.9-15.0 | 370-371 -> 376-377 | **+1.6%** |
| nemo | 17.3 -> 17.2 | 302-303 -> 303-304 | ~+0.3% (a wash) |

So run 118's "cl100k a wash / gpt2 inconclusive" were window noise: the
shrink gains ~1% on four of five schemes (nemo immune, its windows are
likelier non-ASCII and its algebra's read set differs).

**The probe corrected:** run 111/114's labels were wrong, the
"walk+put 15.2" slot covered only the emission's tail and the
"algebra ~120" slot covered the algebra *plus* the walk and put. The
corrected split per 64-byte window: scan+fetch 15.2, **precompute 156.0**
(the classify's internals total 71), **algebra 15.6**, **walk+put
115.2**. The "~90 unattributed" is the precompute region's call/latency
(attributed, not a hidden block); the "algebra's ~100" does not exist.
The fill's two real targets are the precompute and the walk+put. The
probe's five rdtsc sites cost ~150 cycles/window, so shares are relative.

Gates: all five scheme tests, ctest 47/47.

## Run 120, the fill's two regions read: both at their floor (2026-10-02)

Reading the run-119 targets against their measured cost. **The
precompute (156 cyc/window)** = the classify, and the classify is
**9 mask fields × 2 `vmovmskps` = 18 movemasks + ~20 SIMD ops + 18 pack
uops ≈ 56 uops → 71 cycles, issue-bound on port 0**. Every field is read
by the ASCII algebra (8 in its prologue, `hi` as the unicode dispatch and
a positional mask), so there is no dead field, and the pack is already 2
uops. **The walk+put (115)** = the pop walk (~25), the put's content
(~45, run 109) and **three stores per entry** (38 stores/window) plus
loop bookkeeping. A 24-byte entry would not help: the cost is the store
count, and the key/pointer/length all still need stores.

Conclusion recorded: the fill's remaining cost is **structural**, not a
tuning problem. The routes from here are (1) an algebra redesign needing
fewer mask fields, worth a design pass against gigatoken's mask set
first, (2) a serve contract without the pointer in the batch entry, or
(3) taking the remaining wins elsewhere (multi-thread scaling, cold
pass, other encodings).

No code landed. The run-119 consolidated shrink results stand: o200k
+1.1-1.3%, cl100k +1.1-1.3%, gpt2 +0.8-1.0%, deepseek +1.6%, nemo ~wash.

## Run 121, the mask-set design pass, nemo diagnosed (2026-10-02)

**Item 1, the design pass (run 120's prerequisite): gigatoken's ASCII
mask set is 7 fields, `{l, d, s, wt, n, hi, ap}`, with `l`
case-folded, and no `up` (uppercase) or `sl` (slash). Their x86
classifier issues 14 `movemask`s per 64-byte window against our 18.**
The case and slash boundaries resolve through their *walker*, not the
mask algebra; ours resolves them in the algebra and pays two fields per
window for it. The reduction therefore needs the same shape on our side:
drop `up`/`sl`, route the within-run case transitions and slash tails
into the existing `bad`/walker path. Estimated 6-8 cycles/window
(~0.5 cyc/piece, ~3% of the fill); it is a correctness-sensitive
redesign with its own test-first run (gate: the five equivalence tests,
plus a measurement of the walker's frequency change).

**Item 2, nemo's immunity, diagnosed:** a one-line ablation (the
carries writing the raw byte instead of the class-table lookup) moves
nemo's fill by 0.3 cyc/piece (17.10 vs 17.40, within noise). The
carries' class work is a minor share of nemo's fill, the highest of the
five (17.4). The shrink optimised something that was never large there;
nemo's cost is in its walker/algebra/emission mix (a nemo probe build is
the next step).

**Item 3, the serve's floor:** unchanged from run 106 (the dependent
chain ~5 of 11.8 cyc/piece; the AVX2 hot loop already hand-schedules
it).

No code landed. Gates: ctest 47/47 after the ablation revert.

## Run 122, the 6t scaling gap is outside the serve (2026-10-02)

128 MiB owt, same binary, same window, 1t and 6t:

| scheme | 1t fill/serve (total) | 1t wall cyc/piece | 6t fill/serve (total) | 6t wall cyc/piece | 6t scaling |
|---|---|---|---|---|---|
| o200k | 15.8 / 11.3 (27.1) | 27.2 | 16.8 / 10.9 (27.7) | **48** | 3.37x |
| deepseek | 15.6 / 12.0 (27.5) | 34.3 | 15.9 / 11.5 (27.4) | **92** | 2.24x |

The per-piece **phase** costs at 6t are equal (27.4 vs 27.7), yet the
wall cost per piece is 48 vs 92, so deepseek's shortfall (and half of
o200k's own) is **unaccounted work outside the fill/serve probes**: ~0.1
cycles/piece for o200k at 1t and ~20 at 6t; ~7 for deepseek at 1t and
**~65 at 6t**. The id volumes are equal (28.70M vs 29.24M), so it is not
output volume. It is the batch path's non-serve machinery, the gather
generation (a second full pass over the ids), the per-job reserves and
the job overhead, growing with the thread count, the signature of a
memory-bound shared section.

**This reframes the item:** the ladder's deepseek/nemo numbers are
limited by the batch path's overhead, not the core; the gather and the
per-job sections are now the largest uninstrumented region in the
pipeline and the next probe target.

The two-field mask reduction (run 121's design) is specified but not
landed: it moves the case/slash logic into the walker and needs its own
test-first protocol, which this run did not have room for. No code
landed; ctest 47/47.

## Run 123, the planning scan: deepseek +15.8% at 6t (LANDED, 2026-10-02)

New probes inside `tsimd_tokenizer_encode_batch` (plan / encode / gather /
stitch, under `TSIMD_PHASE_STATS=1`) found the 6t gap in one section, at
6t on 128 MiB owt (Mcyc per call):

| section | o200k | deepseek |
|---|---|---|
| **plan** | 0.62 | **191.85** |
| encode | 163.43 | 161.23 |
| gather | 25.10 | 24.80 |
| stitch | 23.26 | 27.15 |

**300x in `plan`**: deepseek's tokenizer.json loads added tokens, so the
planner took its added-token scan, which tested **every byte** in a
scalar loop, serially, before the pool (191.85 Mcyc serial against a
27 Mcyc parallel encode). That was the entire deepseek/nemo "2.2x
scaling" finding (runs 104-122), not the serve.

**Landed fix:** the scan now skips with `memchr` (one call per distinct
first byte from the cursor, earliest candidate wins). Measured at 6t,
128 MiB owt: **deepseek 806.7 -> 934.4 MiB/s (+15.8%)**, nemo ~605 ->
619.8 (+2.4%), o200k unchanged (no added tokens). Plan: deepseek
191.85 -> 126.62 Mcyc; nemo now 299.78 Mcyc (its largest 6t item).

Gates: ctest 47/47; o200k `c8fdcf23…`, cl100k `a8fbfce3…`, gpt2
`35ad3c17…` unchanged at 1t and 6t.

Retry specified: a one-pass SIMD set-membership candidate scan (~nfb
times cheaper than the repeated memchrs), nemo's 299.78 Mcyc plan is
the target.

## Run 124, the chained queue: the SIMD scan traded (2026-10-02)

Run 123's retry implemented and measured: the one-pass SIMD candidate
scan (16 bytes per iteration, one `pcmpeqb` per distinct first byte +
movemask + ctz) against the landed memchr form, 6t, 128 MiB owt:

| scheme | memchr (landed) | SIMD | plan Mcyc |
|---|---|---|---|
| deepseek | **934.4** | 884 | 126.62 -> 147.14 |
| nemo | 619.8 | **646** | 299.78 -> 281.40 |

The forms differ by +/-5% and the winner depends on the scheme; summed,
memchr is ahead (1554 vs 1530) and it wins on the targeted scheme.
**Reverted** (re-verified: deepseek 918 MiB/s in this window, ctest
47/47, gpt2 `35ad3c17…` unchanged). Lesson: one candidate-scan shape is
not dominant across schemes, the *candidate density* (how often the
scan stops) decides whether the per-hit match attempt or the scan cost
dominates; an adaptive shape needs the density probe first.

Stage 2 (the two-field mask reduction's feasibility probe: force the
case/slash positions into `bad` with one line and measure the walker's
cost) did not fit in this run and is the next step.

The run-123 win stands: deepseek +15.8% at 6t; the batch-path section
probes stand.

## Run 125, the up/sl feasibility probe: the walker hangs (2026-10-02)

Run 124's pre-specified one-line probe for the two-field mask reduction
(`*bad_out = bad | up | m->sl;`, the simulation of routing the case and
slash boundaries to the walker) **does not terminate**: the 128 MiB run
never finished and an 8 MiB follow-up sat at 99% CPU for twelve minutes
before `pkill -x bench_encode`. The walker/zone loop needs its `bad` set
sparse enough to make bounded progress; uppercase letters every ~30
bytes break that invariant. **The design is closed for this
architecture**, at the cost of one line instead of a rewrite.

Restored and verified: clean build, ctest 47/47, o200k `c8fdcf23…`
unchanged at 1t and 6t, deepseek's 6t at 943 MiB/s (the run-123
planning win intact).

Consequence: their engine resolves case boundaries without an `up` mask
*and* without dense bad zones, so their letter-run/case handling is a
different structure; re-reading it with that question is the
prerequisite for reviving the reduction (the ~3% fill prize stands, the
route does not).

## Run 126, the adaptive candidate scan (LANDED, 2026-10-03)

The density probe (TSIMD_PLAN_DBG) gave the deciding numbers: **deepseek
nfb 1 with 280 candidate stops/MiB; nemo nfb 2 with 610/MiB**. That
explains run 124's trade exactly, at nfb 1 a single glibc AVX2 memchr
pass beats the SIMD compare loop; at nfb 2 the memchr form re-scans once
per distinct byte and loses to the one-pass SIMD loop. The scan is now
**adaptive**: memchr at nfb == 1, the SIMD loop for nfb 2-8, memchr
above 8 (the compare fan-out would cost more than the re-scans).

Measured at 6t, 128 MiB owt: deepseek 913 MiB/s (plan 134.01 Mcyc , 
within window noise of its memchr best 934/126.62) and nemo **638**
MiB/s (plan 280.55, up from memchr's 619.8/299.78, matching its SIMD
best 646/281.40). The adaptive form takes the best of both without
choosing per scheme.

Gates: ctest 47/47; o200k `c8fdcf23…` unchanged at 1t and 6t; gpt2
`35ad3c17…` unchanged. The `TSIMD_PLAN_DBG=1` density line stays as a
diagnostic.

## Run 127, the added-token first-byte index: nemo +19.1% at 6t (LANDED, 2026-10-03)

nemo's plan (280.55 Mcyc, run 126) came from `added_len_at` scanning
**all** added tokens per candidate position: 1000 tokens x 78k candidates
(610 stops/MiB) ~ the whole 200+ Mcyc. The added tokens are now indexed
by first byte (`added_head[256]` + `added_next[]`, rebuilt whenever the
added set changes, freed with it, linear fallback above 65535 tokens),
and the matcher walks only the chain for the position's byte, same
candidates, same longest-match rule.

**Measured at 6t, 128 MiB owt: nemo 638 -> 760 MiB/s (+19.1%), plan
280.55 -> 214.16 Mcyc; deepseek unchanged (922).** Cumulatively nemo at
6t: ~605 -> **760 (+25.6%)** since run 123.

Gates: ctest 47/47 (the added-token/batch tests cover the matcher's
semantics); o200k `c8fdcf23…` unchanged at 1t and 6t.

Next: a **two-byte index**, nemo's nfb is 2, so all 1000 tokens share
two first bytes and the chains are ~500 long; a 65536-entry head would
make them ~1, with the 1-byte head kept for the buffer's last byte.

## Run 128, the two-byte index: nemo +55.8%, deepseek +39.8% at 6t (LANDED, 2026-10-03)

nemo's and deepseek's added tokens now chain under their **two-byte**
key (`added_head2[65536]`, 128 KB pointer-held; length-1 tokens keep the
1-byte chain, which also serves the buffer's last byte), collapsing
run 127's ~500-long chains to ~1-2. Same candidates, same longest-match
rule.

| scheme | run 127 | now | plan (before -> after) |
|---|---|---|---|
| nemo | 760 MiB/s | **1184** | 214.16 -> **26.69 Mcyc** |
| deepseek | 922 | **1289** | 135.07 -> **19.17 Mcyc** |

The plan fell from 35-47% of the 6t wall to 7-9%. **Cumulative over runs
123-128 at 6t: deepseek 806.7 -> 1289 (+59.8%), nemo ~605 -> 1184
(+95.7%).**

**The "deepseek/nemo scale at 2.0-2.3x" finding (runs 104-122) is fully
resolved:** with the serial planning gone, nemo scales ~3.9x and
deepseek ~3.6x from their 1t numbers, at or above o200k's 3.4x. These
schemes never had a parallel problem; they had a serial O(N×tokens)
planning pass in front of the pool.

Gates: ctest 47/47 (length-1 tokens and the last byte are covered by the
added-token tests); o200k `c8fdcf23…` unchanged at 1t and 6t.

Next: the batch path's stitch (43-45 Mcyc, now the second-largest
section) and the gather (26-28), then the letter-run/case re-read, the
nemo 1t fill, and the serve's chain.

## Run 129, the per-job output cache: +43-51% at 6t (LANDED, 2026-10-03)

The batch stitch (run 128's 43-45 Mcyc) was `free(outs[i].ids)` per job , 
an munmap per job per call, with the next call re-mmapping and
re-faulting the same buffers inside the serve. The per-job output
vectors now live in the pool (`pool_outs_get`: grown on demand, counts
reset, capacities kept; freed once in `pool_destroy`), acquired after
`ensure_pool`.

| scheme | run 128 | now | stitch |
|---|---|---|---|
| o200k | 1518.9 MiB/s | **2178 (+43%)** | 29.9 -> 0.00 Mcyc |
| deepseek | 1289 | **1950 (+51%)** | 45.4 -> 0.02 |
| nemo | 1184 | **1716 (+45%)** | 43.2 -> 0.01 |

The encode section fell as well (o200k 163.5 -> 125.1; nemo 188 -> 143.7)
,  the second half of the same cost (the per-job mmap/first-touch).

**Cumulative over runs 123-129 at 6t: o200k 1554.6 -> 2178 (+40.1%),
deepseek 806.7 -> 1950 (+141.7%), nemo ~605 -> 1716 (+183.6%).**

Gates: ctest 47/47; o200k `c8fdcf23…`, cl100k `a8fbfce3…`, gpt2
`35ad3c17…` unchanged at 6t; TSan 0 warnings.

Next: the gather (26-28 Mcyc, now second-largest) and the flat buffer's
per-call first-touch; then the letter-run/case re-read, the nemo 1t
fill, the serve's chain.

## Run 130, the gather at its floor; the case question narrowed (2026-10-03)

Per-rep batch sections at 6t, 128 MiB owt, o200k (Mcyc): rep 0 plan
244.67 / encode 261.88 / gather 26.19; rep 1 0.50 / 142.14 / 28.08;
rep 2 0.56 / 129.03 / 26.17.

**The gather is 26-28 Mcyc every rep**, a per-call copy of ~230 MB
(28.7M ids read + written) at ~8.5 GB/s, i.e. at this machine's
practical read+write copy bandwidth. Its alternatives either cost the
same (worker-direct writes still need a compaction pass; the flat buffer
must be dense) or break the API (caching it: the caller frees it).
Non-temporal destination stores (dropping the RFO read, ~one third of
the traffic) are the one unblocked option.

**The case-handling re-read narrowed the question but did not resolve
it:** their `SMALL_CASES` includes `camelCase`/`PascalCase`/
`HTTPResponse` with regex parity, so their engine resolves case-split
boundaries with no `up` mask and no case field in `UniMasks`, the
mechanism is in their boundary/walk structure. Until it is identified,
the two-field reduction stays blocked behind run 125.

**Cold-pass observation:** rep 0's plan section is 244.67 Mcyc against
0.50 warm (the same code, cold-start faults/reallocs/I-cache), a
concrete measurable component of the batch path's first call.

No code landed; ctest 47/47.

## Run 131, the streaming gather (LANDED) and the cold-plan probe (2026-10-03)

**Landed: the gather's stripe copy uses streaming stores** (aligned
head, 16-byte `_mm_stream_si128`, scalar tail, one sfence per job;
`TSIMD_GATHER_NT=0` is the A/B knob), dropping the destination's
read-for-ownership. Paired A/B, one binary, 6t, 128 MiB owt, o200k:
**2/2**, BEST 2308 vs 2205 and 2139 vs 2123 (+4.7%, +0.8%), gather 25 vs
26-27 Mcyc. Modest because the stripes are socket-bandwidth-bound either
way.

**The cold-plan fault hypothesis is falsified:** rep 0's plan section
(244.67 Mcyc against 0.50 warm) is unchanged under the bench's read path
(244.91), so it is not the input mapping's fault-in. For o200k the
added-token scan does not run at all, so rep 0's plan is the same tiny
per-doc safe-split work that is 0.5 Mcyc warm. Located: nowhere yet.
Next probe: per-document stamps in the planning loop.

Gates: ctest 47/47; o200k `c8fdcf23…`, gpt2 `35ad3c17…` unchanged; TSan
0 warnings.

## Run 132, the cold-plan item dissolves (2026-10-03)

Per-document planning stamps (`TSIMD_PLAN_DBG=1`, 6t, 128 MiB owt,
o200k): the plan section reads 240.94 Mcyc at rep 0 against 0.5-0.63
warm, but the **max document is 0.02 Mcyc in every rep**, and a
32-document call's stamps sum to ~0.5-0.6 Mcyc, exactly the warm
enclosing value. So the planning work is fully accounted for and the cold
interval's 240 Mcyc lies outside it; the window's prologue is three
trivial statements. **The cold plan section is a first-call measurement
artifact** (the window spans the pool threads' creation and the caller's
migration, a stamp pair across that transition is not trustworthy).
Runs 130-131's "the cold pass pays 244 Mcyc in planning" was a
misattribution; the cold pass's real components remain the pool creation
(run 102) and the cache seeding (run 100).

With this, every item the batch-path probes opened is closed: plan ~0.5
Mcyc, gather at its floor (streaming stores landed, run 131), stitch zero
(run 129). The stamps are kept as the diagnostic that falsified the
attribution. ctest 47/47.

## Run 133, the case mechanism identified; the item closes (2026-10-03)

gigatoken resolves case splits with **no `up` mask and no dense bad
zones** because (a) its punctuation mask is *derived*
(`ob = !(l|d|s|wt|n|hi) | uni.o`, o200k_family.rs:1050) and (b) its
**letter runs are walked by a scalar case scanner** (`ascii_letter_state`,
`scan_case_run`, o200k_family.rs:100-190), a deliberate byte-level walk,
not a fallback zone. Their fast masks therefore carry only the
non-letter classes.

Adopting it for us is not a field drop but a different architecture for
letter handling: our algebra derives its case rules from the `l`/`up`
masks (the case-smear seeds, the run splits, the carries' `pcl/pl/pu`
chain), and their replacement is a new scalar letter-run path in five
schemes with its own correctness surface and cost profile (English text
is mostly letters). **Closed as identified, not adoptable as a bounded
change**, the ~3% prize stands unclaimed, with the pointers recorded.

No code landed; ctest 47/47. Remaining queue: the nemo 1t fill, the
serve's chain.

## Run 134, nemo's fill localized (2026-10-03)

1t, 128 MiB owt, warm rep: o200k pieces 26,698,928 (0.199/B), fill 16.5,
serve 12.0, BEST 437; deepseek 27,035,737 (0.201), 15.6, 11.9, 437; nemo
27,492,349 (**0.205**), **18.2**, 12.5, **383**. The densities match, so
the numbers are comparable: **nemo's fill is genuinely +2.6 cycles per
piece** against deepseek and its serve +0.6, the whole of its ~7% BEST
deficit, localized to its mask pipeline. On record already: nemo has the
most complex algebra terms (13 with o200k; cl100k 11, deepseek 10) and
was the only scheme the run-113 ASCII-algebra specialization did not
move, both consistent with an algebra dominated by terms that
specialization does not touch.

Next step (specified, not taken): port the fill probe to nemo_masks.c
(shared header for the helper) and sub-split its classify/carries/
algebra/walk+put. No code landed; ctest 47/47.

## Run 135, the nemo port recipe; the queue state (2026-10-03)

The nemo fill-probe port did not land: two mechanical patches failed on
`nemo_masks.c`'s whitespace drift (the second one; the script aborted
before writing), and rather than risk a botched edit the tree was
verified clean (0 probe markers, 0 build errors, ctest 47/47, o200k
`c8fdcf23…` unchanged). The exact recipe is recorded in the run report:
the helper block, then the four stamp sites **by line range read in that
file**, the run-117 lesson, repeated under worse conditions.

Queue state: landed closures (parallel dispatch, pool/cache cold costs,
batch plan/gather/stitch, added-token indexes, output cache, streaming
gather); measured closures with nothing owed (cold-plan artifact, gather
floor, case-handling mechanism); open and specified (the nemo port, the
serve's chain).

## Run 137, Phase 0: 6t parity reached; a unigram P0; the families (LANDED, 2026-10-04)

The run-136 pipeline's Phase 0, run first as planned. It found a P0
(the unigram memo was not deterministic) and it reached the stopping
criterion: **6t parity with gigatoken**.

**The 6t parity (256 MiB owt, both engines in one window, 4-MiB docs,
2 rounds x 4 reps, the project pinning).** MiB/s warm:

| scheme | width | ours | gigatoken | ratio |
|---|---|---|---|---|
| o200k | 1t | 466-477 | 646-656 | 0.72-0.74x |
| o200k | 6t | 2314-2456 | 1990-2495 | 0.95-1.18x |
| cl100k | 1t | 479-486 | 688-701 | 0.69-0.73x |
| cl100k | 6t | 2259-2505 | 2321-2794 | 0.90-0.98x |
| o200k | 10t | 2543-2633 | 2008-3016 | 0.84-1.29x |
| o200k | 16t | 3089-3097 | 2120-2674 | 1.17-1.46x |

Run 101's baseline was 0.769-0.850x at 6t. Cold at 6t is the remaining
6t gap: o200k 872 vs 1369 (0.64x), cl100k 976 vs 1717 (0.57x).

**The 15-encoding ladder (128 MiB owt, batch 4 MiB, reps 3, pinned, one
process at a time; MiB/s 1t cold/warm, 6t cold/warm):** o200k
265/444, 539/2231 · cl100k 271/472, 626/2206 · gpt2 297/494, 682/2310 ·
deepseek 252/447, 524/1932 · nemo 237/405, 539/1728 · nemo3 237/406,
540/1710 · gem4 5.9/5.9, 5.9/6.0 · llama3 238/433, 510/1541 · qwen2.5
194/329, 351/735 · qwen3 194/327, 406/767 · glm4 233/434, 466/1811 ·
smollm2 107/127, 336/628 · spm-bpe 5.9/6.2, 7.2/7.2 · uni-xlmr 3.2/3.3,
3.4/3.3 · uni-t5 5.3/5.3. Against run 104: o200k 6t warm 1564 -> 2231,
deepseek 822 -> 1932, nemo 605 -> 1728.

**The families (LANDED).** qwen2/qwen2.5/qwen3 rode the scalar walker
(36 MiB/s); their regex is cl100k's with the digit arm `\p{N}`, so
QWEN25 now rides the cl100k mask fill with a `digit_single` flag
(**36 -> 329 MiB/s at 1t warm, 9x**). The NFC normalizer gained an
ASCII-run region loop (**1500 MiB/s normalizer; the NFC premium is
1.33x**, 329 vs 439 for the same model without NFC). Family parity vs
the references on 2 MiB owt: llama3, qwen2.5 (NFC/nonorm), qwen3
(NFC/nonorm), glm4, smollm2, deepseek, nemotron, nemotron3_nano,
gemma4 all **byte-exact vs HF tokenizers 0.23.2**; mistral7b
(spm-BPE) exact vs sentencepiece 0.2.2.

**The unigram P0 (FIXED; the memo is now opt-in).** The memo was not
deterministic: cold and warm passes emitted different ids (t5, 4 MiB
owt: 1001235 vs 1001234). Three causes: the hit path scored unk nodes
with `scores[unk_id]` instead of `min_score - 10` (-23.59 off); the
cached segmentation is offset-dependent (near-ties flip at another
f32 offset); the DP's ±1e5 rebase is not replayable. The guards (a
bit-exact insert-time replay check, a band-wide winning-margin
requirement, a ±90000 hit safety check) restore determinism, but the
memo's unit split still diverges from the whole chain by 2 ids per
16 MiB, so the memo is opt-in and the exact whole-chain path is the
default (5.1 vs 36 MiB/s on 4 MiB owt; the guards cost ~2%).

**Gates:** ctest 49/49; the 16-MiB id gates o200k `c8fdcf23…` and
cl100k `a8fbfce3…` unchanged; the 495.3-MiB document.txt byte-compare
EXACT (4.1 s); diff_fuzz cl100k 0/200, gpt2 0/200, o200k 1/200;
ASan/UBSan and TSan clean on the changed paths.

**New open findings:** (1) o200k does not recognize a contraction after
a combining mark whose starter is a digit or a space, repro
`"0"+U+094D+"'re"` gives ours [15,682,6,264] vs tiktoken [15,682,4118];
(2) `test_cl100k_masks`'s border case fails under Debug/ASan only
(pre-existing, verified against the run-136 file; all Release gates
pass).

## Run 138, the mark/contraction fix; the nemo fill localized (LANDED, 2026-10-04)

**The o200k mark-then-contraction bug fixed.** A combining mark is a
valid o200k letter-run body, so the contraction suffix attaches to it:
`"0"+U+094D+"'re"` now gives the reference's `'0' '्' "'re"`
(tiktoken [15,682,4118]) instead of `'0' '्' "'" 're'`. The mask path
already deferred the shape (the `mk` bad zones), so the fix is
scalar-only (three sites in `src/pretokenizer/o200k.c`). New test
`tests/test_o200k_marks.c` with 27 tiktoken-captured cases; ctest
50/50; o200k `diff_fuzz` **400 samples 0 failures at 1t and 6t** (was
1/200); a 300-case mark battery and a 400-case broad battery vs
tiktoken clean; the 495.3-MiB byte-compare EXACT; the 16-MiB id gates
unchanged.

**The nemo +2.6 cycles/piece localized (not fixed).** The fill probe
(the run-135 recipe) now exists for nemo and deepseek, compile-time
gated. 128 MiB owt, 1t, cyc/window: nemo's classify 54.2 + algebra 26.9
= 81.1 against deepseek's 68.0 + 37.4 = 105.4, **nemo's mask chain is
cheaper**. The difference is the scalar deferral: **nemo 5.8% of pieces
vs deepseek 0.7%** (2.41M vs 0.29M). Control: pure ASCII nemo 10.6 vs
deepseek 10.9 cyc/piece; owt nemo 16.8 vs deepseek 14.9. Cause:
`classify_uni_nemo` defers each hi char's whole scalar piece (char +
punct run + CR/LF/slash tail) plus a 2-byte bad-zone extension. The
narrowing fix needs the reference gates; queued.

**New finding (recorded): the cl100k path is wrong at -O0.** The Debug
build diverges from tiktoken (4/200 fuzz failures) and from its own
oracle; -O1 and above are correct; o200k is correct at -O0. Not an
uninitialized local (`-ftrivial-auto-var-init` both ways), not aliasing
or wrapping, no sanitizer report; no single function's optimization
level reproduces it (only the file-wide -O0 does). Release is clean on
every gate. Queued for MSan/valgrind (neither installed).

## Run 139, the osf_in UB fixed; nemo's resid narrowed (+10.5% at 1t) (LANDED, 2026-10-05)

**An uninitialized self-initialization fixed (cl100k + deepseek).**
`const uint64_t osf_in = osf_in;` (4 sites) read the uninitialized
local instead of the `cr->osf_in` carry that decides the prefix
absorption. The -O0 build produced wrong ids (the run-138 finding);
Release passed only by register luck. Fixed to `cr->osf_in`. The plain
Debug build now passes **ctest 50/50** (it failed 2 before), the Debug
CLI vs tiktoken is 0/200 (was 4/200), and the Release id gates are
**unchanged** (o200k `c8fdcf23…`, cl100k `a8fbfce3…`) with the
495.3-MiB byte-compare EXACT, behavior-neutral in Release, hazard
removed.

**Nemo's resid narrowed (the run-138 localization implemented).**
`classify_uni_nemo` now classifies valid multi-byte characters (the
o200k class dispatch) and keeps the run-77 piece-walk deferral only for
the stray/truncated/invalid hi bytes (`adv <= 1`), with the index-base
bug (`d[i]` → `d[scan + i]`) fixed. Scalar pieces drop **11.4x**
(2,408,233 → 211,978, 5.8% → 0.5%) and the fill goes **16.8 → 15.5
cyc/piece**; the mask chain itself costs more (81.1 → 103.3
cyc/window) because it now does the work.

| nemo | run 138 | run 139 |
|---|---|---|
| 1t cold / warm | 236.6 / 405.3 | 251.3 / **447.9** |
| 6t cold / warm | 538.6 / 1727.8 | 559.8 / **1882.4** |

**+10.5% warm at 1t, +8.9% at 6t.** Gates: `test_nemo_masks` (20000
fuzz strings + the 495.3-MiB span-equal), `test_nemotron`, HF parity
EXACT (2 MiB owt × 2 files, 4 MiB document.txt), the id-level
mask-vs-scalar A/B 0/1000 random + 4-MiB slices, ASan/UBSan clean.

**The 1t state (re-measured, not moved):** 1t warm 0.742-0.749x, 6t
warm 0.94-1.11x, 6t cold 0.578x vs gigatoken in one window (256 MiB
owt, 4-MiB docs). The 1t lever remains the open item.

## Run 140, the 1t gap decomposed; the case route priced at zero (MEASURED, no change, 2026-10-05)

**The 1t split (o200k, 256 MiB owt, batch 4, 1t):** fill 16.1,
serve 11.7, total 27.8 cyc/piece; 53.4M pieces, 99.9% short hits,
448.9 MiB/s warm. Against gigatoken's ~20 cycles/piece the gap is ~7.6,
split ~+4.6 fill / ~+4.4 serve.

**The shape is not the lever.** The doc-size sweep at 1t is flat
(458.9 / 458.7 / 473.8 / 424.4 / 458.9 warm MiB/s for 1/2/4/8/16-MiB
docs), and the batch shape beats the single-buffer path (439.8 vs
379.4). The 1t cost is per-piece work.

**The case-machinery route closes.** Ablated on an all-lowercase slice
(semantically inert there): `m->up = 0` + `ub = clb = p_u = p_cl = 0`
gives fill 15.1-15.5 against the 15.4 baseline, **within noise**. The
run-133 "adopt their case architecture" redesign is not worth building.

**Recorded hazard:** one baseline re-verification read 46.5 cyc/piece
(1.8x) after a benchmark burst, a thermal/frequency dip, not code;
recovered to 26.3-26.5 after a pause.

**Current cross-engine state:** 1t warm 0.719-0.743x, 6t warm
0.93-1.18x, 6t cold 0.637x. No identified 1t lever >= 1 cyc/piece
remains.

## Run 141, the serve's dependent load broken: -1.0 cyc/piece at 1t (LANDED, 2026-10-05)

Read from gigatoken's own probe_pair comment (the "address-cmov feeding
a dependent load" their probe exists to avoid) plus run 106's own
measurement on our side (~5.1 of the 7.2 extra cycles in the
hash-dependent load). The run-107 loop loaded the key as two GPR loads,
compared with 7 vector ops, branched, and only then loaded the payload.
**The rewrite**: the key is one 16-byte load; the compare is
`xor`+`testz` (2 uops); the payload loads issue *with* the key load; the
loop is a flat counted `for (j < n)` with the batch's `+kServeD` slack
covering the look-ahead reads (no guards).

**Paired A/B (256 MiB owt, batch 4, 1t):** serve **10.9/10.8 →
9.8/9.9 cyc/piece (-1.0)**; fill unchanged.

**The 1t cross-engine ratio: 0.719-0.743x → 0.775-0.797x** (ours
480-488 vs 612-622 warm MiB/s); 6t warm 0.958-1.183x; 6t cold 0.652x.

Per-encoding 1t warm (128 MiB owt, batch 4; run-137 window as context):
o200k 503.0 (444.0), cl100k 514.7 (471.6), gpt2 556.9 (494.2), deepseek
491.4 (447.3), nemo 499.3 (405.3), llama3 467.1 (432.6), glm4 474.1
(434.0), qwen2.5 349.0 (328.8).

**Fill candidates closed:** the two-phase shape is slower for us (17.8
vs 15.6 at 1t, our phase A is the cost, three flatten variants pinned
neutral-to-worse); the branchless flatten stays ~0.8 worse than the
per-bit loop; the fused emission's register cursor measured neutral and
was reverted.

Gates: ctest 50/50; the o200k/cl100k 16-MiB id gates unchanged; the
495.3-MiB byte-compare EXACT; o200k fuzz 0/200 at 1t and 6t; family
parity EXACT (deepseek, llama3, qwen2.5, glm4).

## Run 142, the retries, the fill's split, and the vocab seed reversed (LANDED, 2026-10-05)

**Landed: the worker-cache vocab seed is opt-in, +10-12% on the 6t
cold pass.** Re-priced with the current code, 6t on 256 MiB owt: the
seed's init cost is 100 Mcyc (1.2M DRAM-latency inserts across six
private 32-MiB tables) against a 31 Mcyc cold-encode benefit, a net
69 Mcyc = 9% of the cold pass; 1t and warm are washes. Paired, two
rounds: seed off 853.9/875.1 against 778.2/779.5. `TSIMD_ENABLE_WSEED=1`
restores it. The **6t cold cross-engine ratio went 0.652x → 0.719x**
(ours 874.1 against 1215.1). Ids unchanged (the cache is a performance
structure; both id gates and the 495.3-MiB byte-compare hold).

**Two retried levers measured neutral and were reverted:** the fused
emission's register cursor (4 alternating rounds: medians 15.35/15.40
fill cyc/piece) and the per-piece prefetch hoist (3 rounds: 15.3/15.4).
Both were uop-shaving; the fill is not uop-bound.

**New diagnostic, the fill's clean split** (a no-emit ablation at 1t,
256 MiB owt batch 4): **mask chain 9.9 + emission 5.5 = fill 15.4**,
serve 9.7, total 25.1. The emission is already competitive with
gigatoken's whole fill (~11.5); the mask chain is the remaining gap and
resists micro-level work.

**Corrected instrument:** the batch probe's "plan" section is really the
pool-init window (`bp_t1 - bp_t0`); its true sub-parts are 0.05 Mcyc.
The cold 6t excess is ~240 Mcyc of pool init (fixed) plus ~237 Mcyc of
cold caches/TLB in the encode.

| 6t (256 MiB, batch 4) | run 141 | run 142 |
|---|---|---|
| cold MiB/s | 804-874 | **854-875** |
| cold ratio vs gigatoken | 0.652x | **0.719x** |
| 1t warm ratio | 0.775-0.797x | 0.777-0.804x |
| 6t warm ratio | 0.958-1.183x | 1.04-1.35x |

## Run 143, the pool caches sized to the share: +15% cold (LANDED, 2026-10-06)

**The per-worker pretoken cache is now sized to the worker's share of
the input**, `bits = floor(log2(total/width)) − 4`, clamped [12,22]
("the table holds ~2x the share in bytes"), instead of the fixed
1024-MiB budget's 22-bit cap at every width. The pre-fault traffic no
longer grows with the width for nothing.

| width | metric | before | after |
|---|---|---|---|
| 6t | cold | 1074-1108 | **920-947 (+15%)** |
| 6t | warm | 2152-2401 | 2337-2407 (wash) |
| 10t | cold | 993.5 | **1103.6 (+11%)** |
| 16t | cold | 772 | **1152** |
| 16t | warm | 3097 | 3286 |

The 1t path keeps its 2^22 sequential upgrade (measured optimum: 476.4
against 457.4 at 2^20, 145.8 at 2^18).

**Cross-engine ladder (256 MiB, 4 reps):** 1t warm 0.761-0.786x; **6t
cold 0.796x** (run 140: 0.637x → 142: 0.719x); 6t warm 1.01-1.11x.

**The mask chain closes.** The last bounded idea (gigatoken's
`min_epu8`+`cmpeq` compare form) measured neutral (16.1 against 15.9
fill cyc/piece) and was reverted. Nine bounded experiments have now
measured ~zero; the chain is stall-bound, and its remaining avenue is a
dependency-depth redesign of the boundary algebra.

Gates: ctest 50/50; the o200k id gate unchanged; the 495.3-MiB
byte-compare EXACT; o200k fuzz 0/200 at 6t.

## Run 144, the normalize defect fixed (xlmr now reference-exact); the memo stays opt-in (LANDED, 2026-10-06)

**The run-137 memo residual was a `tsimd_spm_normalize` defect, not a
memo bug**, and the whole chain had it too on other shapes. Two parts
fixed: the dummy mark is now added only when the *normalized* content
does not already start with a space (a charsmap replacement that starts
with a space, U+00AF → space + U+0304, U+00A8, U+00B4, must merge with
the dummy into one mark: `"¯2"` gave `3 3 2 357` against the reference's
`3 2 357`), and a space-leading replacement keeps the pending whitespace
run pending so the merged run flushes as one space *before* the
combining char. 12 new reference-id cases per model in
`tests/test_unigram.c` (18 → 4 → 0 failures through the fix).

**Results:** the memo is exact (0 residual at 4/16/32/64 MiB); **xlmr is
now reference-EXACT at 2 and 4 MiB** (was 1–2 divergences per 4 MiB);
t5 keeps its unrelated f32-tie residual. Gates: ctest 50/50, the o200k
id gate unchanged, the 495.3-MiB byte-compare EXACT.

**The memo stays opt-in on performance grounds** (t5/owt/batch 4/1t):
2.3x the whole chain at 4–8 MiB (12.7 vs 5.4 MiB/s), 1.1x at 12, 0.5x
at 16, 0.1x at 32 MiB. The per-doc phase split of a 16-MiB run shows
why, split/lookup/DP in Mcyc: 70/85/750, 70/98/730, **70/2834/843,
70/8509/1062**: the probe walk explodes with the table's load while the
DP stays flat. Growth policies (at saturation, at a quarter load) did
not help; a 2^22 table does (10.1 MiB/s at 32 MiB), so the ceiling is
the shared probe on the memo's private table.

## Run 145, the unigram memo sized for the workload: it pays up to 64 MiB (LANDED, 2026-10-06)

`tsimd_memo_size_for(t, total_len)` now sizes the memo's private table
from the whole workload (2^21 for ≥ 8 MiB, 2^22 for ≥ 16 MiB), called
by all three encode entry points, the batch path passes TOTAL_LEN, the
same signal run 86's BPE upgrade uses. The run-144 explosion came from
the memo sizing itself by the *call's* length (4 MiB per doc) while its
dictionary accumulated across the batch.

| t5, owt, batch 4, 1t (memo enabled) | before | after | whole chain |
|---|---|---|---|
| 8 MiB | 10.5 | 10.5 | 5.4 |
| 16 MiB | 2.7 | **10.3** | 5.4 |
| 32 MiB | 0.5 | **9.1** | 5.4 |
| 64 MiB | 0.2 | **6.9** | 5.4 |
| 128 MiB | 0.2 | 2.8 | 5.4 |

The memo pays 1.3–1.9x from 16 to 64 MiB and loses at 128 MiB (its
dictionary outgrows the tables), so it stays **opt-in** with a measured
envelope. Gates: ctest 50/50, `test_unigram` 38+38 byte-exact, the
o200k id gate unchanged, the 495.3-MiB byte-compare EXACT, the default
unigram path unchanged at 5.1 MiB/s.

## Run 146, the full sentencepiece charsmap and the unk accounting (LANDED, 2026-10-07)

The sentencepiece normalizer now applies the reference's FULL charsmap
(4822 single-codepoint rules, 30 deletions, 12100 canonical sequence
keys = 224696 rules) with the reference's greedy longest-match, and the
UNIGRAM Viterbi books one unk score per codepoint (the reference's
`EncodeOptimized` accounting; the processor merges the ids afterwards).

Correctness, both UNIGRAM models, on this machine:

| corpus | before (t5) | after (t5) | before (xlmr) | after (xlmr) |
|---|---|---|---|---|
| owt 2 MiB | 5 | **0** | 0 | **0** |
| owt 4 MiB | 14 | **0** | 0 | **0** |
| document 2/4 MiB |, | **0** |, | **0** |
| charsmap-heavy fuzz, 256 KiB | 62990 | **0** | 84927 | **0** |
| random bytes, 64 KiB |, | **0** |, | **0** |

Cost, same-window A/B (`/tmp/build-verify145/bench_encode` against the
new build; 128 MiB owt, batch 4 MiB, reps 3, 1t):

| model | before | after |
|---|---|---|
| uni-t5 | 5.0 cold / 5.0 warm | **5.1 / 5.3** |
| uni-xlmr | 3.1 / 3.1 | **3.2 / 3.2** |

Phase split (t5, owt, 32 MiB, 1t): the normalize was 39.7 MiB/s and
12.4% of the chain when the full table landed; the membership bitmap
and the lookahead gate took it to **103.8 MiB/s and 5.4%**. The DP is
82.8% and runs at 5.9 MiB/s, unchanged.

Gates: ctest 50/50; `test_unigram` 74+74 table cases and 7+7 byte-exact
cases; all five id hashes UNCHANGED (o200k c8fdcf23…, cl100k a8fbfce3…,
qwen2.5 95ac549c…, gemma4 924a8796…, mistral7b ff04be8f…); the
495.3-MiB o200k byte-compare EXACT.

## Run 147, the unigram DP's double-array trie (LANDED, 2026-10-08)

The unigram Viterbi walked the vocabulary with a hash probe per piece
length; it now walks a double array built once at load
(`tsimd_vocab_build_spm_trie`). One 8-byte unit load decides each step
(`idx = base + byte`, verified by the parent check), the DP state is one
12-byte node per position instead of three parallel arrays, and the
piece score comes from the trie node.

The reference (sentencepiece 0.2.2, THIS MACHINE, 32-MiB owt slice
pinned to c0): **18.0 MiB/s for t5, 17.3 for xlmr**; its normalize alone
is 76.2 MiB/s, so its Viterbi is ~25 MiB/s. Ours before this run: 5.2
and 3.1 MiB/s.

Paired A/B in one binary (`TSIMD_DISABLE_SPM_TRIE=1` vs unset), t5/xlmr,
32-MiB owt, 1t, `bench_phases`:

| | t5 dp | xlmr dp |
|---|---|---|
| hash probes | 5.4 MiB/s | 3.2 MiB/s |
| double array | **13.6** | **13.8** |

Same-window ladder A/B against the run-146 build (128-MiB owt, batch
4 MiB, reps 3, `bench_encode`):

| model | 1t before | 1t after | 6t before | 6t after |
|---|---|---|---|---|
| uni-t5 | 5.3 | **12.7** | 5.4 | **12.6** |
| uni-xlmr | 3.3 | **12.2** | 3.3 | **12.7** |

Gates: ctest 51/51 (test_spm_trie is new, and it passes under ASan+UBSan
with a 20000-key randomized stress); test_unigram 74+74 + 7+7 cases
byte-exact; t5 and xlmr EXACT against the reference on owt and document
at 2 and 4 MiB, on the 256-KiB charsmap fuzz corpus, on 65536 random
malformed bytes and at 1t/6t/16t; all five id hashes unchanged; the
495.3-MiB o200k byte-compare EXACT. Peak RSS of a 1-MiB encode is
unchanged within noise (t5 60,464 KB against 60,588 KB without the
trie).

## Run 148, the unigram DP's last loads (LANDED, 2026-10-08)

Three changes to the run-147 trie and DP: the piece id and the float
score are interleaved in one 8-byte `tsimd_piece_t` (one cache line per
match); the match update is one compare chain; the per-match frontier
update is gone (the rebase scans `starts_at + max_token_len` instead);
and the walk's bounds check is gone because the builder reserves the
256-unit label span and a padded unit's check is NIL.

Paired A/B against the run-147 build (32-MiB owt, 1t, `bench_phases`):

| model | run 147 dp | run 148 dp |
|---|---|---|
| t5 | 13.1 MiB/s | **15.9 MiB/s** |
| xlmr | 14.6 MiB/s | **16.6 MiB/s** |

Ladder (128-MiB owt, batch 4 MiB, reps 3, `bench_encode`):

| model | 1t run 147 | 1t run 148 | 6t run 147 | 6t run 148 |
|---|---|---|---|---|
| uni-t5 | 11.4 | **13.5** | 11.4 | **13.5** |
| uni-xlmr | 13.0 | **14.0** | 12.6 | **14.0** |

The reference (sentencepiece 0.2.2, this machine, 17.6 MiB/s for t5 in
the same window) is now at 77% of parity from our side; across runs 147
and 148 the unigram models went 5.3 -> 13.5 MiB/s (t5, 2.5x) and
3.3 -> 14.0 (xlmr, 4.2x).

Also closed: `scripts/tiktoken_parity.py` now prefers tiktoken's OWN
official encoding, so the o200k arm no longer panics and both o200k and
cl100k are EXACT against the official references on 2 MiB of owt.

Gates: the visible test suite plus the whole ctest in a Debug
ASan+UBSan build (the three memo stress tests excluded for wall time);
test_unigram 74+74 + 7+7 byte-exact; t5/xlmr EXACT on owt and document
and on the charsmap fuzz corpus; all five id hashes unchanged; the
495.3-MiB o200k byte-compare EXACT.

## Run 149, the profiler closes the unigram gap (LANDED, 2026-10-08)

`perf` was unblocked on this machine, so the chain was profiled instead
of guessed at. Three findings, all fixed:

1. The `TSIMD_DISABLE_SPM_TRIE` A/B hook and the trie pointer were
   inside the DP's per-position loop: `perf` put 28.9% in `getenv`.
   Hoisted once per call.
2. The normalize decoded ASCII through out-of-line calls
   (`tsimd_utf8_next`, `tsimd_class_codepoint`) at ~14%; the ASCII path
   is inline now.
3. The xlmr MODEL LOAD took 18.5 s: a bubble sort over 250000 pieces in
   `tsimd_model_load_spm` (98.9% of the load, `perf annotate` +
   `addr2line`). Replaced by `qsort`; the id gates proved the rank
   order equivalent. The load is 0.2-0.3 s.

Same-window comparison, alternating arms, pinned to c0, the same
32-MiB owt slice (`bench_encode` against sentencepiece 0.2.2):

| model | ours | sentencepiece | ratio |
|---|---|---|---|
| t5 | 24.7 MiB/s | 18.5 MiB/s | **1.33x** |
| xlmr | 25.9 MiB/s | 18.0 MiB/s | **1.44x** |

`bench_phases` (32-MiB owt, 1t): both models 23.6 MiB/s whole chain,
dp 29.9, normalize 129 MiB/s.

Gates: ctest 51/51; test_unigram 74+74 + 7+7; test_spm_trie; t5 and
xlmr EXACT against the reference on owt, document and the charsmap fuzz
corpus; mistral EXACT; all five id hashes unchanged; the 495.3-MiB
o200k byte-compare EXACT.

## Run 150, the load-time gate and the spm-BPE ratio (measurement only, 2026-10-08)

Nothing landed in the engine: the re-profile after run 149 shows the DP
at 64.6% with no single dominant line (IPC 2.09; the remaining misses
are the walk's dependent 8-byte load chain and its data-dependent exit
branches), so the next step needs a structural change and is recorded,
not attempted.

What this run did record:

- **The model load times**, the defect class run 149 found (an 18.5-s
  load): t5 0.0 s, xlmr 0.2 s, mistral7b 0.0 s.
- **The whole sentencepiece family against the reference in one
  window** (128-MiB owt, batch 4 MiB, reps 3; the reference pinned to c0
  on the same 32-MiB slice):

| model | ours 1t | ours 6t | reference | ratio |
|---|---|---|---|---|
| uni-t5 | 25.0 MiB/s | 25.1 | 18.5 | **1.35x** |
| uni-xlmr | 25.9 | 26.1 | 18.0 | **1.44x** |
| spm-bpe (mistral7b) | 7.8 | 7.9 | 2.1 | **3.7x** |

The spm-BPE comparison closes a question open since run 137, whose
sentencepiece rows came from a harness whose units could not be
reproduced.

Gates: `/tmp/build-san150` (ASan+UBSan) `test_spm_trie` and
`test_unigram` pass, the latter 74+74 byte-exact.

## Run 151, the four-byte trie unit and the base reservation (LANDED, 2026-10-08)

The unigram walk paid one dependent 8-byte load per input byte. The unit
is now `label:8 | piece:1 | child:1 | base:22` in one uint32, and the
LABEL carries the parent check, which is sound only if no two nodes
share a base, so the builder reserves bases with a build-only bit array
(the occupancy marker moved to a second build-time bit array, since a
label byte of 0 is a legal key byte). Both arrays are freed after the
build; the steady state is 4 bytes per unit plus the piece array
(xlmr: 4.7 MB of units against 9.4 MB).

The invariant was a test first: `tests/test_spm_trie.c` now asserts that
no two nodes share a base over a 20000-key build, and that test FAILED
on the old design, which is precisely why the old design needed the
8-byte check.

Paired A/B (`/tmp/build-150final`, 32-MiB owt, 1t, `bench_phases`):

| model | run 150 full / dp | run 151 full / dp |
|---|---|---|
| t5 | 24.4 / 31.0 MiB/s | **25.4 / 32.7** |
| xlmr | 25.2 / 32.5 MiB/s | **26.0 / 33.5** |

Against the reference on this machine (t5 18.5, xlmr 18.0, spm-BPE
mistral 2.1): the unigram models are 1.37-1.44x ahead, mistral 3.7x.

Gates: ctest 51/51; test_unigram 74+74 + 7+7 byte-exact; test_spm_trie
all cases plus the invariant; t5, xlmr and mistral reference-EXACT; the
o200k and mistral7b id hashes unchanged; the 495.3-MiB o200k
byte-compare EXACT; ASan+UBSan clean on the two test binaries and a full
t5 chain run.

## Run 154, the DP walk prologue (LANDED, 2026-10-08)

The DP walk paid two unit-array loads per step: the parent's unit for
the base and the child's unit for the label. It now carries the unit
value: `trie->u[0]` is read once per encode call (the walk prologue) and
`tsimd_trie_advance` returns the child's unit next to its index, so each
step is one unit load plus the array-pointer load. The plain walk stays
reachable through `TSIMD_DISABLE_DP_PROLOGUE=1`.

The CHILD check is load-bearing and the run proves it: a leaf stores
base 0, unit 0 is the root, the root's label is 0, so a step from a leaf
with byte 0x00 lands on the root and passes the label test. On the
20000-key stress trie 49071 leaves produce 49071 false targets, all byte
0x00 to unit 0 (`/tmp/unsound_probe`), and the new leaf-wall test fails
on a trie with the check removed. The spm normalize keeps NUL bytes
(`a 00 b c` -> `E2 96 81 61 00 62 63`), so arbitrary input reaches it.
The check stays for every step that can start at a leaf.

Paired A/B in one window (`bench_phases --reps 3 --mib 32`, 1t, c0,
6 rounds, `/tmp/ab3.sh`; best of window):

| model | run 153 baseline full / dp | run 154 prologue full / dp |
|---|---|---|
| t5 | 25.7 / 32.9 MiB/s | **26.3 / 34.3** |
| xlmr | 26.5 / 34.4 MiB/s | **27.3 / 35.8** |

+2.3% and +3.0% whole chain, +4.3% and +4.1% on the DP. The same
binary's plain arm reads 25.5 / 32.7 (t5) and 26.4 / 34.3 (xlmr), so the
second specialization costs about 0.5%. Counters: cycles -4.5%, L1
loads -6.4% (-256.7M), branch misses -2.9%, instructions +0.4%.

Reference in the same window (sentencepiece 0.2.2 Python API, 32-MiB
owt, c0, best of 3): t5 17.3, xlmr 16.4 MiB/s, the same id counts as
ours (8080397 / 8086357). The ratio is 1.52x and 1.66x; run 150's
different reference harness gave 18.5 / 18.0, or 1.42x / 1.52x.

Tried and reverted, with numbers: the literal branchless first step
(remove the CHILD check for step 1, fold the failure into the loop
bound) is sound but loses to the plain carry, 32.0-33.0 MiB/s against
32.6-32.7 on the DP, because it needs a `blen > 1` test on every step;
and the two-level first-byte table (65536 entries x 4 bytes) is a
measured loss, 31.2-31.3 against 32.6 on the DP with +21% L1 loads and
+4.9% instructions, the root's 1-KB first-level region is L1-resident
and the 256-KB table is not.

Gates: ctest 51/51; test_spm_trie all cases plus the leaf wall and the
carried-step equivalence; test_unigram 74+74 byte-exact on both arms;
t5 and xlmr reference-EXACT on both arms.

## Run 155, the spm-BPE heap boundary and the priority tables (LANDED, 2026-10-08)

The two levers of run 153's profile were implemented in order, test
first. The mistral7b spm-BPE path is unchanged at 7.2-7.8 MiB/s over
most windows and reaches 8.0 MiB/s in the pure window; only same-window
pairs count on this machine.

1. The linear-versus-heap threshold. Both paths provably select the same
   merge: lowest priority, ties by leftmost position. The new
   `tests/test_bpe_path_agreement.c` forces each path and compares both
   against each other and a naive oracle (89 checks). The timing ladder
   puts the crossover between about 42 and about 107 symbols, so the
   default 64 is at it, but the spm-BPE worker never reaches the
   linear branch (0 of 676 block calls; `sym_max` 13684), so the limit
   cannot move this benchmark. The knob stays, the default does not
   move.

2. The dense priority grid. It already existed for the spm-BPE vocab
   (`mgr_dense` 16 MiB + `mgr_dense_ids` 16 MiB + `mgr_slots` 1 MiB),
   and it LOSES here: 77.9% of probes carry an id above 2047, so the
   grid serves one probe in five while its footprint evicts the 1-MiB
   slots table that serves half. The spm-BPE loader now builds the
   slots table only; the inline `_fast` accessors landed in the worker.

Interleaved same-window A/B (32-MiB owt, 1t, `bench_phases`, three runs
per arm, best):

| arm | time | rate |
|---|---|---|
| baseline (grid + out-of-line probes) | 4.072 s | 7.86 MiB/s |
| final (slots only + inline probes) | **4.009 s** | **7.99 MiB/s** |
| final with the grid forced on | 4.048 s | 7.91 MiB/s |

The grid alone costs 0.95%; the landing is +1.6% rate. The mistral7b
peak RSS of load plus a trivial encode drops from 73928 KB to 40996 KB.

Scoped deliberately: the tiktoken and tokenizer.json loaders keep the
grid (cl100k 404.5 -> 401.8 MiB/s and o200k 391.4 -> 387.5 without it),
and the gemma4 spm-BPE workload is neutral (5.40 s either way).

Gates: ctest 52/52 (51 + the new test); mistral7b id hash
`ff04be8fda4d115c0b078131e4e96582`; mistral7b reference parity EXACT on
2 MiB of owt; test_spm_integration 11 cases; test_bpe; test_bpe_heap_oracle;
test_merge_id_probe 24/24.

## Run 157, the heap pop revalidation (LANDED, 2026-10-10)

Run 155's open item was the 7764929 pop revalidation probes of the
spm-BPE merge heap (an 8-MiB slice). The popped `(rank, position)` entry
was re-probed against the merge tables at every pop. The revalidation is
gone: the workspace keeps a per-position candidate (`rank << 32 |
merged_id`), written at the three push sites, and the pop validates the
entry against it and merges the stored id. The heap key, the heap order
and the selection rule (lowest priority, ties by leftmost position) are
unchanged. `TSIMD_BPE_POP_PROBE` (default 0, `=1` forces the old
re-probe loop) makes the paired A/B run in one binary.

A tie census first, because two pairs can share a priority and produce
different merged ids: the mistral7b table has 97 such pair entries (95
tied priorities), and the synthetic test vocab proves the same. A
rank-only state with a `rank_to_id` lookup is therefore unsound and was
rejected before implementation. The planted wrong design (refresh the
stored id only when the rank changes) fails the new tests with 227
failures, four named equal-rank shift cases, and real-text divergence
(53 vs 54, 120 vs 124, 244 vs 255, 481 vs 482 ids on the mistral
ladder); the landed design passes 6278 checks.

Paired A/B, two windows, 32-MiB owt, 1t, c0, `bench_phases --reps 3`,
three runs per arm per window, best of run:

| window / arm | time | rate |
|---|---|---|
| window 1 baseline binary | 4.119 s | 7.77 MiB/s |
| window 1 state (default) | **3.959 s** | **8.08 MiB/s** |
| window 1 same binary, `TSIMD_BPE_POP_PROBE=1` | 4.170 s | 7.67 MiB/s |
| window 2 baseline binary | 4.012 s | 7.98 MiB/s |
| window 2 state (default) | **3.862 s** | **8.29 MiB/s** |
| window 2 same binary, `TSIMD_BPE_POP_PROBE=1` | 4.057 s | 7.89 MiB/s |

The state arm beats the baseline binary by 3.9% and 3.7% of the time
(+4.0% and +3.9% rate); it beats the same binary's forced-old arm by
5.1% and 4.8%. The id count is 8382749 in every arm. The scratch census
counts 0 table probes at pop for the state arm against 7764929 for the
old arm, with the same 6216223 accepted merges and 1548706 rejects. The
phase split moves the pop loop from 263.8 to 248.2 cyc/sym and the push
loop from 32.7 to 37.0 (the push sites now also fetch the merged id).

gemma4 (the second spm-BPE workload), same window: 4.699 s / 6.8 MiB/s
baseline against 4.473 s / 7.2 MiB/s state (-4.8%), forced-old arm
4.773 s / 6.7 MiB/s, 7490625 ids in every arm. cl100k and o200k never
enter this worker: hashes unchanged (`a8fbfce3b1e73e97ca75af0511ec680f`,
`c8fdcf233e546eef195d6f886d88ebc5`) and timings identical inside the
noise (429.8/430.0 against 429.6/424.2 and 413.7/399.3 against
413.9/413.5 MiB/s). The 16-MiB CLI gate: 2.06 s against 1.97-1.99 s,
4196756 tokens, hash `ff04be8fda4d115c0b078131e4e96582`.

Gates: ctest 52/52; test_bpe_path_agreement OK (6278 checks);
test_bpe, test_bpe_heap_oracle, test_bpe_short OK; mistral7b and t5
reference parity EXACT on 2 MiB of owt; both tiktoken hashes unchanged.

## Run 158, the tokenizer_json families profiled and the serve-env hoist (LANDED, 2026-10-10)

The ten tokenizer.json families were never profiled; this run ranked
them (32-MiB owt, 1t, pinned c4, best of 3 reps): deepseek 402.4,
nemotron3_nano 395.2, nemotron 391.7, llama3 388.1, glm4 317.6, qwen3
297.0, qwen2.5 296.6, smollm2 111.4, gemma4 8.4 MiB/s. llama3.2-1b
rejects with "invalid JSON", its 29-byte file holds the download
error "Invalid username or password.". Model load lines are 0.06-0.32 s
per family; no quadratic load. Phase split: gemma4 is dp-heap bound
(isolated dp 387.6% of full, 2.2 MiB/s); smollm2 is the gpt2d pretok
byte walk (isolated pretok 71.6% of full, 173.1 MiB/s); qwen2.5/qwen3
carry norm 18.7-19.8% and an isolated nfcloop at 15.4-18.4 MiB/s.

Per-model shares (perf, cycles, --no-children). gemma4: heap_pop_tree
31.76%, tsimd_vocab_merge_prio_id 31.60%, tsimd_bpe_encode_spm 22.37%,
heap_push 3.22%. smollm2: tsimd_match_gpt2d 19.12%, tsimd_utf8_next
18.95%, tsimd_class_ascii 9.80%, tsimd_utf8_decode 8.69% (= 56.6% in
the per-byte walk), bserve_flush 8.52%, fused_span_bserve 8.40%,
tsimd_bpe_encode 5.54%. qwen2.5: tsimd_cl100k_mask_fill_keyed 13.42%,
norm_impl 11.51%, tsimd_serve_hot_avx2 10.11%, tsimd_bpe_encode 8.10%,
window_usable_bad 6.41%, bhash_find 4.61%, encode_buffer 4.60%,
**getenv 2.10%**. deepseek (control): deepseek_mask_fill_keyed 17.87%,
serve_hot_avx2 13.94%, bpe_encode 11.15%, window_usable_bad 6.47%.

The win: three env reads per serve batch or per segment moved to once
per public encode call, `getenv("TSIMD_SERVE_DUMP")` and
`serve_avx2_enabled()`'s `getenv("TSIMD_SERVE_AVX2")` in
`fused_o200k_serve` (37727 + 37727 calls per 32-MiB qwen2.5 encode;
deepseek 33978 each; nemotron 34512 each) and
`qwen25_scalar_forced()`'s `getenv("TSIMD_QWEN25_SCALAR")` (6884 per
qwen2.5 encode). An LD_PRELOAD census counted the reads; the fix cuts
them from 67978-82360 to 26-28 per encode. `tsimd_serve_env_refresh()`
reads the three variables once per public encode entry; the four
entries (encode, encode_parallel, encode_batch,
encode_whole_chain_par) call it, so the tests' runtime-setenv contract
(test_parallel, test_qwen25_serve) still holds, the first
once-per-process design failed test_parallel ("avx2: the arm ran while
disabled") and was replaced the same hour.

Paired A/B in one binary (`TSIMD_NO_SERVE_ENV_CACHE=1` = exact old
behavior, default = cached), position-balanced ABBA, 32-MiB owt, 1t,
c4, best of 3 reps per run, arm means: deepseek 414.9 vs 397.4
(+4.4%), nemotron 420.5 vs 404.1 (+4.1%), nemotron3_nano 421.6 vs
404.3 (+4.3%), llama3 400.8 vs 382.3 (+4.8%), glm4 384.6 vs 369.7
(+4.0%), qwen2.5 307.3 vs 295.6 (+4.0%), qwen3 306.4 vs 295.2 (+3.8%);
gemma4 flat (0.994, spm path), smollm2 flat on the in-window pair (one
B-side outlier). Reversed BAAB windows reproduce the win with the
position bias inverted: deepseek +4.1%, llama3 +4.2%, qwen2.5 +3.8%,
nemotron3_nano +4.5%. Three-way deepseek window: old baseline binary
400.9/398.7, new binary old-arm 402.4, new binary new-arm 418.3 MiB/s.
After the fix `getenv` leaves the qwen2.5 profile top (was 2.10%).

METHOD NOTE: a plain A,B,A,B ladder was rejected, the first arm in
each window always won (window 2's first arm was B and B "won" 416.1
vs 384.3 on deepseek), so run order, not the arm, decided. Only
position-balanced ABBA plus a reversed BAAB control separates a real
arm effect of this size on this machine.

Gates: ctest 52/52; o200k c8fdcf233e546eef195d6f886d88ebc5, cl100k
a8fbfce3b1e73e97ca75af0511ec680f, qwen2.5 95ac549c8688e7c6422518c0e0092642,
gemma4 924a8796ddce779d71cc6e5cf81e32cf all unchanged; tokjson parity
EXACT for gemma4, qwen2.5, deepseek, nemotron, llama3; tiktoken parity
EXACT; build adds no diagnostics over the 7 pre-existing warnings
(verified by compiling the pristine sources with the same flags).
Full detail: docs/158_the_tokenizer_json_families_profiled.md.

## 2026-10-12, Run 160: gemma4's pop scan tree; the recorded priority lever is already banked

The gemma4 census (run 160, /tmp/ws-gemma): 514906 merge pairs, only
31911 (6.20%) dense-eligible at the built DLOG 11 (11.00% at 4096,
0.17% at 256), tables ~80 MiB (slots 32 MiB at load factor 12.3% +
mgr_ids 16 MiB + dense grid 32 MiB). The fresh profile after run 157's
pop-revalidation removal: heap_pop 63.97%, tsimd_bpe_encode_spm
22.74%, tsimd_vocab_merge_prio_id 8.59%, the recorded
prefolded-priorities lever has nothing left on the lookup side, and
`perf annotate` puts 43.76% of heap_pop on the first child load of
each sift level plus a 7-step serial blend chain.

Table-shape arms all LOSE (interleaved window, gemma4 32 MiB):
TSIMD_MGR_LOAD=2 +6.2% slower, load 3 +3.2%, TSIMD_NO_MGR_DENSE +1.3%
to +1.7%, the probe chain beats residency; the built shape is
right-sized.

LANDED: `heap_pop`'s min-scan is now a 3-level pairwise tree over the
8 children (same strict-less rule, same winner; partial tails keep the
old loop), switchable by `TSIMD_BPE_POP_SCAN` (0 = old scan verbatim,
1 = tree, default; 2 = tree + a prefetch-coverage change that LOST
~1.7% and stays non-default). Test first: test_bpe_path_agreement
grows the scan arms (8362 checks, up from 6278); a planted lane-7
omission fails 4 checks and was reverted.

Same-window paired runs (pinned c0, rotated order, 2 reps per run,
best kept): gemma4 32 MiB full, baseline binary 4.484 s mean
(7.14 MiB/s), same binary scan=0 4.459 s, default 4.395 s (7.28 MiB/s):
-2.0% vs baseline, -1.4% vs the same binary's old arm; every default
run fastest in all three windows. mistral7b spm: 3.892 s vs 3.869 s
(-0.6%, 8.22 to 8.27 MiB/s; the same-binary old arm carries +2.9%
layout). Controls flat: qwen2.5 0.107-0.108 s both, o200k 0.066-0.071 s
both.

Gates: ctest 53/53 (47.50 s); gemma4 924a8796ddce779d71cc6e5cf81e32cf
(default and scan=0 arms), cl100k a8fbfce3b1e73e97ca75af0511ec680f,
o200k c8fdcf233e546eef195d6f886d88ebc5, qwen2.5
95ac549c8688e7c6422518c0e0092642, mistral7b ff04be8fda4d115c0b078131
e4e96582 all unchanged; tokjson parity EXACT; bpe.c compiles with 0
warnings. Full detail:
docs/160_gemma4_prefolded_pair_priorities.md.

## Run 161, the gpt2d SIMD spanner (smollm2 125.2 -> 412.8 MiB/s, 3.30x)

The gpt2d scheme (ByteLevel + Digits(individual_digits=true), only
smollm2 in the matrix) moved from the per-byte pretokenizer walk to
the class-table mask-fill serve machine the other served families
use. The gpt2 algebra in gpt2_masks.c gained a digit_single knob with
two extra rules: every digit codepoint start is a boundary (no
space-attachment for digits), and a whitespace run followed by a
digit is one span (the leave-one-behind split is suppressed before
digits, including at the 64-byte window edge). The scalar
tsimd_match_gpt2d walker stays the fallback and the arbiter. Landed
default ON; `TSIMD_GPT2D_SPAN=0` restores the old walk in the same
binary.

TEST FIRST: tests/test_gpt2d_spanner.c checks span-sequence equality
(scalar vs spanner) over 90+ hand cases, 18816 window-edge
constructions, 20000 fuzz strings, and a 4-MiB owt slice. RED proved
on "123" (3 spans vs 1) with the flag set but the algebra unchanged.
Two planted bugs (the edge rule and the interior rule each removed)
were caught immediately (edge: span count 30 vs 31 on a 108-byte
input; border: 339 vs 347 on a 598-byte input); restore green.

Windows (bench_phases full line, best of 3 reps, 32-MiB owt, c4,
position-balanced): ABBA smollm2 A(walk) 124.6 vs B(span) 406.5
(3.26x), deepseek control 0.9988; BAAB smollm2 124.4 vs 419.4
(3.37x), deepseek 1.0076; bench_encode ABBA smollm2 126.6 vs 409.4
(3.23x); landed-semantics ABBA (default vs SPAN=0) smollm2 412.8 vs
125.2 (3.297x), deepseek 1.001. The isolated pretok/scalarref phases
stay at 172.0 MiB/s (the per-byte functions are untouched).

Profile after (bench_encode full, spanner arm): gpt2_mask_fill_keyed
22.09%, serve_hot_avx2 11.81%, bpe_encode 10.56%, encode_buffer
5.21%, bhash_find 4.37%, window_usable_bad 4.08%; the per-byte walk
fell from 56.6% to ~4.6% (utf8_next 1.69% + class_codepoint 1.55% +
utf8_decode 1.40%). The shape now matches the fast families.

Gates: ctest 53/53 (52 baseline + the new test); family id hashes on
the 16-MiB slice at 1t UNCHANGED (o200k c8fdcf23..., cl100k
a8fbfce3..., gemma4 924a8796...); smollm2 3926669 ids, md5
cd349fc2..., byte-identical across baseline binary, default arm,
escape arm, at 1t and 4t; tokjson parity EXACT (479928 ids,
stream-diff 0); clean-build warning diff vs pristine: 6 removed, 0
added. Full detail: docs/161_the_gpt2d_simd_spanner.md.

### The wide-thread ladder refresh (run 274; the same shape, the current tree)

Re-run of the run-252 shape on the run-273 tree (the route architecture
plus the ASCII class table), gigatoken wheel 0.10.0 (latest; still
id-exact: 16-MiB o200k IDS 3,590,319 equals ours), whole owt (499 MB)
and whole document (519 MB), `/tmp/oed2` relinked against the current
library, the same pinning. Headline ratios (ours/giga, warm):

| dataset | model | 1t | 6t | 10t | 16t |
|---|---|---|---|---|---|
| owt | o200k | 0.67 | 1.10 | 0.83 | 0.98 |
| owt | cl100k | 0.67 | 1.24 | 0.97 | 1.09 |
| owt | deepseek | 1.36 | 1.91 | 1.22 | 1.42 |
| owt | qwen2.5 | 0.68 | 1.13 | 0.76 | 0.87 |
| owt | nemotron | 0.94 | 1.37 | 0.94 | 1.09 |
| owt | glm4 | 0.70 | 1.07 | 0.81 | 0.88 |
| owt | gemma4 | 1.33 | 1.07 | 0.76 | 0.87 |
| owt | mistral | 1.16 | 0.93 | 0.69 | 0.77 |
| doc | o200k | 0.50 | 0.94 | 0.68 | 0.89 |
| doc | cl100k | 0.82 | 1.66 | 1.00 | 1.16 |
| doc | deepseek | 1.41 | 1.92 | 1.21 | 1.38 |
| doc | qwen2.5 | 0.84 | 1.25 | 0.91 | 1.06 |
| doc | nemotron | 0.91 | 0.96 | 0.68 | 0.72 |
| doc | glm4 | 0.74 | 1.69 | 0.69 | 0.73 |
| doc | gemma4 | 2.32 | 1.70 | 0.83 | 0.91 |
| doc | mistral | 2.24 | 1.48 | 0.72 | 0.79 |

The delta against run 252 concentrates in o200k and cl100k, the two
schemes whose serve became the fused keyed fill (per-span crc32 key
arming) with the routed walk: ours o200k warm owt 662.7 → 400.5
(-39.6% at 1t), 2,884.4 → 1,769.5 (-38.7% at 6t); cl100k warm owt
675.2 → 473.9 (-29.8% at 1t). Giga's same rows moved -3.6 to -14.1
percent (host drift: this session's host runs ~1.3 background cores).
The warm rounds are 100% pretok-cache-hit (99.17M hits, 0 misses),
so the warm drop is the keyed serve path itself. The json/spm models
moved -1 to -27 percent. t5/xlmr giga rows: no support in gigatoken
(inf, same as run 252). Full 80-row table: docs/274. The recovery
target: the keyed serve and the fill's key-arming cost.

## The post-282 state (runs 279-282; the letter-after-other + uni-other relaxes; 2026-10-29)

The o200k fill's two conservative bad terms were deleted
(`bad |= lb & p_o` and `bad |= uz_o`, the absorb/b2back machinery
already decides both cases; the cl100k algebra never had either).
All numbers this machine, ENCODE_ROUNDS=3, re-pinned host.

Whole-file 1t, ours MB/s (cold / warm):

| corpus | o200k | cl100k |
|---|---|---|
| owt.txt (499 MB) | 390 / **653** | 413 / 688-696 |
| document.txt (519 MB) | 256 / **455-458** | 272 / 449-459 |

Ladder versus gigatoken 0.10.0 (same window, same pinning; ours/giga
WARM):

| corpus | model | 1t | 6t | 10t | 16t |
|---|---|---|---|---|---|
| owt | o200k | 0.92 | 1.58 | 1.39 | 1.52 |
| owt | cl100k | 0.93 | 1.53 |, |, |
| doc | o200k | 0.96 | 1.54 | 1.74 | 1.90 |
| doc | cl100k | 1.00 | 1.50 |, |, |
| owt | deepseek | 1.31 | 2.15 |, |, |
| owt | qwen2.5 | 0.90 | 1.47 |, |, |
| owt | nemotron | 1.02 | 1.60 |, |, |
| owt | glm4 | 0.91 | 1.45 |, |, |
| owt | gemma4 | 1.32 | 1.05 |, |, |

The run-274 route-era ratios for comparison: o200k owt 1t warm 0.67
→ 0.92; document o200k 1t warm 0.50 → 0.96. The lineage (document
o200k 1t warm): 480 (252) → ~250 (274 route) → 246-250 (277/278) →
462-463 (279) → 456 (280) → 455-458 (282). The lineage (owt o200k
1t warm): 662.7 (252) → 400.5 (274) → 493 (277/278) → 568 (279) →
647-648 (280) → 653 (282), 98.5 percent of the pre-campaign line
with byte-exactness everywhere. The spm/unigram families are flat
(mistral 183 → 186; xlmr/t5 unchanged; giga has no spm support).
The 10t/16t rows are stable on the re-pinned host (warm 1.39-1.90).

Walker-window rescues: document 7.6 percent, owt 3.8 percent
(from 88.9 / 41.9 before run 279). The fill's true piece costs
(/tmp/wubbench microbench, no probe overhead): window_usable_bad
27.3 ns/window of ~99, ascii masks 7.8, carries 3.1, algebra+uni
16.4; the walk+put/serve side holds ~70 percent. Full detail:
docs/279, docs/280, docs/281, docs/282.

## Native aarch64 (the NEON arm64 branch, 2026-11)

The NEON arm64 branch runs the same engine natively on ARM,
byte-exact against the x86 build (the cross-arch parity verified on
the full corpora for 13 model families at 1/6/16 threads). The
arm64 coverage: the NEON classify (the 128-bit lane algebra, 1034
checks), the whole-segment serve fill for cl100k/o200k/qwen25/gpt2/
gpt2d, the scalar serve for the other families, and the scalar
encode engines.

Measured on the GitHub Actions arm64 runner (ubuntu-24.04-arm,
4 vCPU Neoverse class), gcc Release, the CI bench harness:

| path | threads | corpus | elapsed |
| --- | --- | --- | --- |
| the scalar serve | 4 | ~21 MB natural | 0.27 s |
| the neon-classified serve | 4 | ~21 MB natural | 0.31 s |

The synthetic 256-byte-token model (the byte-fallback dominated):
~59 MB/s single thread. The adversarial workload = every byte its
own token; the natural-text throughput is higher.

The absolute numbers are not comparable to the x86 table above (the
different ISA, the different runner class). The x86/aarch64 ratio
is not a quality metric — the arm64 branch prioritizes correctness
(byte-exact cross-arch parity) over tuning. The NEON spanner ports
and the arm64-specific optimisations are future work.

The arm64 gate (scripts/run_arm64_gate.sh) runs under qemu on the
x86 development machine: the NEON classify 1034 checks, ctest 78 of
78, and the 13-case cross-arch parity (count + FNV-1a hash per
family at 1/6/16 threads). The CI runs the same gate natively on
the arm runner plus the serve A/B bench.
