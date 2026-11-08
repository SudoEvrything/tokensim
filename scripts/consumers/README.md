# The real-consumer projects

Three minimal projects that use tokensim the way a user would: no engine
code, only the public API. Each one loads cl100k (tiktoken), reads
`testdata/owt.txt` (the full 476 MB corpus), and reports the speed, the
ids, the parity and the round trip.

## C (pkg-config)

```
cmake -S . -B build && cmake --build build -j
cmake --install build --prefix /tmp/tokinst
PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig \
  cc -O2 $(PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig pkg-config --cflags tokensim) \
  main.c -o rt \
  $(PKG_CONFIG_PATH=/tmp/tokinst/lib/pkgconfig pkg-config --libs tokensim) \
  -lpthread -lm
taskset -c 0-15 ./rt
```

## Python (pip)

```
python3 -m build --wheel --outdir dist
python3 -m venv /tmp/venv_local
/tmp/venv_local/bin/pip install dist/tokensim-*.whl
/tmp/venv_local/bin/python app.py
```

## Rust (cargo)

```
cargo build --release
taskset -c 0-15 ./target/release/real_rs
```

`Cargo.toml` depends on the published crate `tokensim` from crates.io.
Point it at `bindings/rust` with `tokensim = { path = "..." }` to measure
the tree instead.

The model path and the corpus path in the sources are absolute; adjust
them if the checkout moves.
