# tokensim

The fastest tokenizer engine written in C/C++, 84x faster than tiktoken, 1.39x against [gigatoken](https://github.com/marcelroed/gigatoken). No regex
engine: hand-written scanners and bitmask algebra over 64-byte
blocks, with parallel output equal to sequential output at any
thread count, proven by test.

The fastest family (o200k)
encodes at 585 MB/s single-threaded and 2703 MB/s with a 6-thread
parallel call, against tiktoken's 32.5 MB/s on the same input.
That is **18x per call, 83x parallel**, byte-identical output (the
very first call pays one-time cache warm-up: 259 MB/s).

> **Documentation:** [BENCHMARKS.md](BENCHMARKS.md) - the numbers ·
> [TECHNICAL-REPORT.md](TECHNICAL-REPORT.md) - the deep-dive
> engineering blog: the decisions, the intuition, and the
> correctness story (read it on [GitHub Pages](https://sudoeverything.github.io/tokensim/))
> · How to use it: the Quickstart below, and the per-language
> snippets in Bindings and packaging.

## Performance

Same-window protocol: the same 16 MB natural-text corpus, warm
runs, this machine (6 performance cores + 4 efficiency cores).

| encoding | reference engine, scalar | tokensim, 1 thread | tokensim, 6 threads |
| --- | --- | --- | --- |
| o200k | tiktoken 32.1 MB/s | 571.1 MB/s (**17.8x**) | 2709.0 MB/s |
| cl100k | tiktoken 22.5 MB/s | 576.0 MB/s (**25.6x**) | 2783.4 MB/s |
| deepseek | official HF tokenizer 2.3 MB/s | 509.0 MB/s (**~220x**) | 2355.7 MB/s |

The multi-thread rows scale ~4.7x over our own single thread and
are byte-identical to it. Against gigatoken, the Rust reference
engine under the same protocol: parity cold and 1.34-1.39x warm at
10 threads on both corpora. The deepseek reference's 2.3 MB/s is
its real single-thread behavior; its pre-tokenizer regex is
expensive, and the comparison is apples to apples. See
BENCHMARKS.md for the full matrix.

## Install

From the published packages (the C engine builds from source inside
each one):

```sh
uv pip install tokensimd         # or: pip install tokensimd (Python)
cargo add tokensimd              # Rust
```

For C, use CMake FetchContent or pkg-config (the Bindings section
below has both). From this repository, building from source is the
same three lines:

```sh
git clone https://github.com/SudoEvrything/tokensim && cd tokensim
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build            # 41 tests without local artifacts
```

Then call it. From the command line:

```sh
./build/tsimd-tokenize --threads 6 cl100k_base.tiktoken cl100k file.txt
```

From Python:

```python
import tsimd
tok = tsimd.load("cl100k_base.tiktoken", "tiktoken", "cl100k")
ids = tok.encode(b"hello world, don't stop")     # GIL released
tok.decode(ids)                                  # round-trips exactly
```

## How it works

The pipeline is four stages, each independently callable and
testable:

```
input (read/mmap view)  ->  added-token split  ->  pretokenize
                        ->  encode (memoized)  ->  uint32 token ids
```

First the **added-token split** scans for special strings such as
`<|endoftext|>`. A hit cuts the input: the special token becomes
one id, and the bytes between hits become separate jobs.

Then the **pretokenizer** applies the model family's own split
rules. Those rules decide where a merge may not cross, for example
between two words. This is the stage where other implementations
run a regex engine; tokensim runs compiled scanners.

Then the **encode** stage turns each piece into ids. Most families
use BPE: start from the bytes of the piece, and repeatedly merge
the adjacent pair with the lowest merge rank until nothing merges.
T5 and XLM-R use a different algorithm, unigram, which picks the
best segmentation by score; both are implemented.

The result is an array of `uint32` ids, and the decoder reverses
the whole chain exactly. That last sentence is the contract: what
goes in comes back out, byte for byte.

### Three ideas carry the speed

**Compile the rules, do not interpret them.** The split rules are
fixed at load time, so the engine compiles them into scanners, and
in the fast path into bitmask algebra: one pass classifies 64 bytes
into 13 bitmasks, and the boundary decisions become a few integer
operations on `uint64_t` words. Instead of asking "what kind of
byte is this?" 64 times, the engine asks "which bytes in this block
are letters?" once and answers for all 64 at once.

**Remember what repeats.** Natural text repeats; the word "the"
appears thousands of times and encodes the same way every time.
The engine memoizes short pieces in an open-addressed table, so a
repeat costs one lookup instead of a merge run. Hit rates run
91.7-99.9% on natural text, worth roughly 1.9x end-to-end by
itself. A miss costs nothing, and correctness never depends on a
hit.

**Split only where it is provably safe.** Multi-threading
multiplies ways to be wrong, so the engine splits work only at
seams that satisfy a proven rule: a letter followed by an ASCII
non-letter, or a digit followed by an ASCII non-digit. At such a
seam, no piece's encoding can depend on the bytes in another
piece. Parallel output equals sequential output at any thread
count, and a test keeps that true.

### The ISA story

The engine runs on SSE2 (the x86-64 baseline), AVX2, and AVX-512,
with the ISA flags applied per source file so the portable code
stays portable. CPUs sometimes report features they cannot
execute, so tokensim asks the CPU to prove it: a small AVX-512
kernel runs under a crash handler once at startup, and a fault
means a permanent downgrade to AVX2. The dispatch is verified under
Intel SDE, an emulator that runs the AVX-512 paths on machines
without the hardware.

## Model formats

| format | files | notes |
| --- | --- | --- |
| `tiktoken` | one `.model` vocab file | the OpenAI tiktoken format: one base64 token per line, ranks from line order (the explicit-rank variant is also read) |
| `gpt2` | `encoder.json` + `vocab.bpe` | the classic OpenAI GPT-2 format; the byte-unicode alphabet is decoded at load; merge ranks from the file order |
| `tokenizer_json` | `tokenizer.json` | the Hugging Face format; BPE models with byte-level pre-tokenizers; the pre-tokenizer must match a known scheme exactly (GPT-2 / cl100k / o200k) or the load is rejected; `added_tokens` load with their explicit ids |

Unsupported configurations (normalizers, non-BPE models, any
pre-tokenizer pattern outside the supported set) are rejected with
a clear error rather than mis-tokenized. Llama-3's pattern is the
cl100k regex exactly and loads as cl100k; Llama-4's pattern is
registered and loads through the o200k scheme. Merge order always
follows each format's own rank file, so ids match the reference
implementations exactly. Sentencepiece models (the BPE kind such as
Mistral, and the unigram kind such as T5/XLM-R) and the kimi-k2
tiktoken file round out the fourteen supported families.

## Pre-tokenizer schemes

| scheme | patterns supported | engine |
| --- | --- | --- |
| `gpt2` | r50k/p50k (HF ByteLevel default) | SIMD mask algebra |
| `cl100k` | GPT-3.5/4, Llama-3 (same regex) | SIMD mask algebra |
| `o200k` | GPT-4o/o200k_base, Llama-4, MiniMax-M2 (the same pattern family) | SIMD mask algebra + fused serve path |
| `kimi` | moonshot's tokenization_kimi pattern | window-batch mask algebra |
| `deepseek` / `qwen25` / `nemo` / `gpt2d` | the JSON families | per-scheme mask fill + serve |

Each family's quirks are matched exactly: o200k's case-phase
transition with contractions glued to the run, cl100k's digit
grouping in threes, kimi's class intersections and isolated Han
runs. Every scheme is differential-tested against its reference on
real corpora, because a pattern that is almost right produces ids
that are almost right too, and almost right is wrong.

## Validation

Exactness is layered, because no test suite can enumerate all
inputs.

- **Forced-ISA equivalence**: every pretokenizer at every SIMD
  level must match its scalar oracle on the same inputs. This
  catches code-generation mistakes.
- **Three-reference parity**: ids match tiktoken, HuggingFace
  tokenizers, and sentencepiece across 13 families at multiple
  thread counts, with pins generated by the references, never by
  our own output.
- **Corpus anchors**: whole corpora are hashed per family; the
  final battery re-derives four references over a 476 MB corpus
  and compares, all EXACT.
- **Fuzzing**: property fuzzers feed arbitrary bytes and check the
  invariants, with campaigns in the tens of millions of runs.
- **Sanitizers**: the suite is ASan/UBSan clean and the parallel
  routes are TSan clean.
- **The battery**: `scripts/run_full_battery.sh` chains all of it
  and exits non-zero on any failure. A battery that cannot fail
  proves nothing, so the gates are verified to fire.
- **The Unicode normalizer** is proven by a 5,328-case sweep
  against the Unicode reference, 0 diffs.

The result is 98 tests green and every family byte-exact against
its reference.

## Bindings and packaging

**Python** - install with `uv pip install tokensimd` (or `pip
install tokensimd`); the sdist carries the C sources and builds them
on install:

```python
tok = tsimd.load("/path/cl100k_base.tiktoken", "tiktoken", "cl100k")
ids = tok.encode(b"hello world")                     # GIL released
ids = tok.encode_threads(b"hello world", threads=8)  # == encode()
data = tok.encode_bytes(b"hello world")              # uint32 LE bytes
tok.decode(ids).decode()                             # round-trip
n = tok.encode_bound(len(data))                      # out-buffer bound
```

`encode` takes bytes and returns a list of ids; `encode_threads` is
byte-identical to `encode` for any thread count; `decode` reverses
the encode (`skip_special=True` drops the special ids).

**Rust** - install with `cargo add tokensim`; the crate vendors the
C sources and builds them with cc (no cmake, no network):

```rust
use tokensimd::{Format, Tokenizer};

let mut tok = Tokenizer::open(
    "cl100k_base.tiktoken",
    Format::Tiktoken,
    Some("cl100k"),
)?;
let ids = tok.encode(b"hello world, don't stop")?;   // &[u32]
let ids = tok.encode_threads(b"...", 8)?;            // == encode()
let text = tok.decode(&ids)?;                        // Vec<u8>, round-trips
```

`decode_opt(&ids, true)` drops special ids; `encode_bound(len)` is
the out-buffer bound for the zero-copy `encode_into` form.

**C consumers** - CMake FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
  tokensim
  GIT_REPOSITORY https://github.com/OWNER/tokensim
  GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(tokensim)
target_link_libraries(myapp PRIVATE tokensimd)
```

or pkg-config (after `cmake --install build --prefix ~/.local`):

```sh
cc myapp.c $(pkg-config --cflags --libs tokensimd) -o myapp
```

The library is x86-64 (SSE2 baseline; AVX2/AVX-512 dispatched at
runtime with the fault-tolerant downgrade described above).
`fuzz_tokenizer` checks the three core invariants over arbitrary
bytes, and `scripts/bench_track.py --save|--check` tracks
throughput regressions against a JSON baseline.
