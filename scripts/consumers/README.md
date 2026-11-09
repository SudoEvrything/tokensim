# The real-consumer projects

Three minimal projects that use tokensim the way a user would: no engine
code, only the public API. Each one loads a model, streams a corpus as
documents, tokenizes each document with the DEFAULT call (the automatic
thread count, the bytes form in Python), decodes sample documents for the
round trip, folds an FNV-1a hash over the id counts and the head and tail
ids of every document, then runs an inference-style loop of short prompts.

The three projects print the same shape of output. For the same corpus and
document size the id total and the two hashes must be identical across all
three: that is the cross-wrapper parity check. The RSS samples at 0, 25,
50, 75 and 100 percent of the corpus are the leak check: flat is healthy.

```
usage: <binary> MODEL FORMAT SCHEME CORPUS [DOC_BYTES] [PROMPT_CALLS] [THREADS]
```

FORMAT is tiktoken, json, sentencepiece or gpt2. SCHEME is cl100k, o200k
or gpt2 for the tiktoken families. DOC_BYTES defaults to 2 MiB,
PROMPT_CALLS to 20000, THREADS to 0 = the automatic default.

## The warm and the cold protocol

The loop in these projects is cold by nature: every document passes
once, so the short-pretoken cache misses on new content. The ladder and
the single-shot benchmarks are warm by design (the last of three passes
over the same corpus). The two protocols must never be compared: the
warm single-shot (~4500 MB/s) against the cold sweep (~1600-2300 MB/s)
reads as an engine gap that does not exist. The honest same-protocol
numbers (cl100k, owt.txt, the into variant, 10 workers):

| pass | single-shot | 2 MiB streaming | 16 MiB streaming |
| --- | --- | --- | --- |
| cold (one pass) | 1922-1977 | 1567-1864 | 2236-2279 |
| warm (repeat passes) | ~4500 | ~3000 (the same document) | - |

The cold-to-cold gap is 1.1-1.2x and sits inside the run-to-run band of
the cold passes. Bigger documents amortize the fixed per-call cost
(~0.3 ms); at 16 MiB the streaming matches the single-shot's cold band.
32 and 64 MiB documents add nothing measurable (the amortization
saturates) and cost memory. The per-piece cycles are the honest
instrument: ~22-25 cycles/piece warm in every regime, ~60 cold.

## C (pkg-config)

```
cmake -S . -B build && cmake --build build -j
cmake --install build --prefix /tmp/tokinst
PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig \
  cc -O2 $(PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig pkg-config --cflags tokensim) \
  main.c -o life \
  $(PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig pkg-config --libs tokensim) \
  -lpthread -lm
./life models/cl100k_base.tiktoken tiktoken cl100k testdata/owt.txt
```

`ladder.c` is the older thread-ladder benchmark kept for the into-path
measurements.

## Python (pip)

```
python3 -m build --wheel --outdir dist
python3 -m venv /tmp/venv_local
/tmp/venv_local/bin/pip install dist/tokensim-*.whl
/tmp/venv_local/bin/python app.py models/cl100k_base.tiktoken tiktoken cl100k testdata/owt.txt
```

## Rust (cargo)

```
cargo build --release
./target/release/real_life models/cl100k_base.tiktoken tiktoken cl100k testdata/owt.txt
```

`Cargo.toml` depends on the published crate `tokensim` from crates.io.
Point it at `bindings/rust` with `tokensim = { path = "..." }` to measure
the tree instead; the automatic-thread default and `default_threads` need
a crate version that has them.

The model path and the corpus path are passed as arguments; the examples
assume the repo checkout as the working directory.

