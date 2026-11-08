#!/bin/bash

set +e
ulimit -c unlimited
arm_timeout() {
  local secs="$1"; shift
  timeout -s ABRT "$secs" "$@" 2>&1
  local rc=$?
  if [ "$rc" -eq 134 ] || [ "$rc" -eq 131 ]; then
    echo "WEDGE-DUMP: the arm aborted (the core saved)"
  fi
  return 0
}
cd "$(dirname "$0")/.." || exit 1
W=${1:-both}
cmake --build build --target bench_encode -j 6 >/dev/null
cargo build --release --manifest-path bindings/rust/Cargo.toml >/dev/null \
    2>&1 || cargo build --release --manifest-path bindings/rust/Cargo.toml
cmake --build build --target tsimd_python -j 6 >/dev/null
cp build/tokensim.cpython-*.so bindings/python/ 2>/dev/null

run_c() {
  local model="$1" fmt="$2" scheme="$3" corpus="$4"
  local rflag=""
  case "$fmt" in sentencepiece) ;; *) rflag="--reuse" ;; esac
  arm_timeout 600 ./build/bench_encode --threads ${THREADS:-6} --reps 3 $rflag \
      --format $fmt $( [ "$scheme" != auto ] && echo --scheme $scheme ) \
      "$model" "$corpus" 2>&1 | tail -2
}
run_py() {
  local model="$1" fmt="$2" scheme="$3" corpus="$4"
  local kind=tsimd_into
  case "$fmt" in sentencepiece) kind=tsimd_par ;; esac
  arm_timeout 600 python3 scripts/bench_python.py $kind "$model" $fmt $scheme "$corpus" 3 \
      ${THREADS:-6} 2>/dev/null || \
      arm_timeout 600 python3 scripts/bench_python.py tsimd_par "$model" $fmt $scheme \
      "$corpus" 3 ${THREADS:-6}
}
run_rs() {
  local model="$1" fmt="$2" scheme="$3" corpus="$4"
  local rfmt=$fmt
  local into=""
  case "$fmt" in sentencepiece) rfmt=spm ;; *) into="BENCH_INTO=1" ;; esac
  arm_timeout 300 env $into ./bindings/rust/target/release/examples/bench "$model" $rfmt \
      $scheme ${THREADS:-6} "$corpus" 3 2>&1 | tail -2 || \
      arm_timeout 300 ./bindings/rust/target/release/examples/bench "$model" $rfmt \
      $scheme ${THREADS:-6} "$corpus" 3 2>&1 | tail -2
}

MODELS=(
  "models/o200k_base.tiktoken tiktoken o200k"
  "models/cl100k_base.tiktoken tiktoken cl100k"
  "models/reference/deepseek_tokenizer.json tokenizer_json auto"
  "models/reference/t5_spiece.model sentencepiece auto"
  "models/reference/xlmr_spiece.model sentencepiece auto"
  "models/reference/mistral7b_tokenizer.model sentencepiece auto"
  "models/reference/nemotron3_nano_tokenizer.json tokenizer_json auto"
  "models/reference/glm4_tokenizer.json tokenizer_json auto"
  "models/reference/gemma3-4b_tokenizer.json tokenizer_json auto"
  "models/reference/kimi_k2_tiktoken.model tiktoken kimi"
  "models/reference/qwen2.5-0.5b_tokenizer.json tokenizer_json auto"
  "models/reference/gemma4_tokenizer.json tokenizer_json auto"
  "models/reference/smollm2_tokenizer.json tokenizer_json auto"
)
if [ "$W" = "owt" ] || [ "$W" = "both" ]; then
  CORPUS=testdata/owt.txt
  for m in "${MODELS[@]}"; do
    set -- $m
    echo "== C   owt $1 ($2/$3)"
    run_c "$1" "$2" "$3" $CORPUS
    echo "== PY  owt $1 ($2/$3)"
    run_py "$1" "$2" "$3" $CORPUS
    echo "== RUST owt $1 ($2/$3)"
    run_rs "$1" "$2" "$3" $CORPUS
  done
fi
if [ "$W" = "docs" ] || [ "$W" = "both" ]; then
  CORPUS=testdata/document.txt
  m="models/o200k_base.tiktoken tiktoken o200k"
  set -- $m
  echo "== C   docs $1"
  run_c "$1" "$2" "$3" $CORPUS
  echo "== PY  docs $1"
  run_py "$1" "$2" "$3" $CORPUS
  echo "== RUST docs $1"
  run_rs "$1" "$2" "$3" $CORPUS
fi
echo "bench_wrappers: done"
