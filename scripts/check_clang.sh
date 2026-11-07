#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
BD="${1:-build-clang}"
echo "== check_clang: configure $BD"
cmake -S . -B "$BD" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    > "$BD.cfg.log" 2>&1 || { tail -20 "$BD.cfg.log"; exit 1; }
echo "== check_clang: build"
cmake --build "$BD" -j6 > "$BD.build.log" 2>&1 || { grep -E "error:" "$BD.build.log" | head; exit 1; }
echo "== check_clang: ctest"
(cd "$BD" && ctest -j1) || exit 1
echo "== check_clang: fuzz harness gate"
clang -fsanitize=fuzzer -O2 -g -I include tests/fuzz_all_models.c \
    "$BD/libtokensim.a" -lm -lpthread -o /tmp/tsimd-fuzz-ci
TSIMD_FUZZ_SPEC="models/reference/t5_spiece.model;spm" \
    /tmp/tsimd-fuzz-ci tests/fixtures/t5_fuzz_mismatch.bin
echo "check_clang: PASS ($BD)"
