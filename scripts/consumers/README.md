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

