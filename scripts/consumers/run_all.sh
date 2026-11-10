#!/bin/bash

# Tokenize every testdata corpus with all three wrappers, like a real
# program: the training-style streaming loop and the inference-style
# prompt loop, every document round-tripped and byte-verified.

set -e
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-models/cl100k_base.tiktoken}
FMT=${FMT:-tiktoken}
SCH=${SCH:-cl100k}
THREADS=${THREADS:-0}
DOCS=${DOCS:-$((16 * 1024 * 1024))}
PROMPTS=${PROMPTS:-20000}
CORPORA=${CORPORA:-"testdata/owt.txt testdata/document.txt testdata/owt16m.txt testdata/document.ids.txt testdata/structured_mix.bin"}

cmake --build build --target tsimd_python -j 6 > /dev/null
cp build/tokensim.cpython-*.so bindings/python/ 2>/dev/null || true
cargo build --release --manifest-path scripts/consumers/rs/Cargo.toml > /dev/null

C_BIN=scripts/consumers/c/life
if [ ! -x "$C_BIN" ]; then
  cmake --install build --prefix "$PWD/.consumers-prefix" > /dev/null
  PKG_CONFIG_PATH="$PWD/.consumers-prefix/lib/pkgconfig" \
    cc -O2 $(PKG_CONFIG_PATH="$PWD/.consumers-prefix/lib/pkgconfig" pkg-config --cflags tokensim) \
    scripts/consumers/c/main.c -o "$C_BIN" \
    $(PKG_CONFIG_PATH="$PWD/.consumers-prefix/lib/pkgconfig" pkg-config --libs tokensim) \
    -lpthread -lm
fi

echo "model: $MODEL  docs: $DOCS bytes  threads: $THREADS (0 = auto)"
for corpus in $CORPORA; do
  [ -f "$corpus" ] || { echo "skip: $corpus missing"; continue; }
  echo "== $corpus"
  echo "-- C"
  "$C_BIN" "$MODEL" "$FMT" "$SCH" "$corpus" "$DOCS" "$PROMPTS" "$THREADS" |
    grep -E "^ids:|^hash:|^roundtrip:|^encode:|^decode:|^docs|^prompts:"
  echo "-- Rust"
  scripts/consumers/rs/target/release/real_life "$MODEL" "$FMT" "$SCH" "$corpus" \
      "$DOCS" "$PROMPTS" "$THREADS" |
    grep -E "^ids:|^hash:|^roundtrip:|^encode:|^decode:|^docs|^prompts:"
  echo "-- Python"
  ${PYTHON:-python3} scripts/consumers/py/app.py "$MODEL" "$FMT" "$SCH" \
      "$corpus" "$DOCS" "$PROMPTS" "$THREADS" |
    grep -E "^ids:|^hash:|^roundtrip:|^encode:|^decode:|^docs|^prompts:"
done
