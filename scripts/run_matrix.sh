#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
cmake --build build --target corpus_hash -j 6 > /dev/null
THS=${@:-"1 6 10 16"}
FAMILIES=(
  "models/o200k_base.tiktoken tiktoken o200k"
  "models/cl100k_base.tiktoken tiktoken cl100k"
  "models/reference/deepseek_tokenizer.json tokenizer_json deepseek"
  "models/reference/qwen2.5-0.5b_tokenizer.json tokenizer_json qwen25"
  "models/reference/nemotron3_nano_tokenizer.json tokenizer_json nemo"
  "models/reference/glm4_tokenizer.json tokenizer_json gem4"
  "models/reference/gemma4_tokenizer.json tokenizer_json gem4"
  "models/reference/gemma3-4b_tokenizer.json tokenizer_json gem4"
  "models/reference/smollm2_tokenizer.json tokenizer_json auto"
  "models/reference/mistral7b_tokenizer.model sentencepiece spm"
  "models/reference/xlmr_spiece.model sentencepiece spm"
  "models/reference/t5_spiece.model sentencepiece spm"
  "models/reference/kimi_k2_tiktoken.model tiktoken kimi"
)
fails=0
for corpus in testdata/owt.txt testdata/document.txt; do
  c=$(basename $corpus)
  for spec in "${FAMILIES[@]}"; do
    set -- $spec
    model=$1; fmt=$2; sch=$3; name=$(basename $model | cut -d. -f1)
    base=""
    for th in $THS; do
      chunk=0
      out=$(./build/corpus_hash "$model" "$fmt" "$sch" "$th" "$corpus" $chunk 2>/dev/null | tail -1)
      [ -z "$out" ] && { echo "MATRIX FAIL $c $name th=$th (no output)"; fails=$((fails+1)); continue; }
      if [ -z "$base" ]; then base="$out"; echo "MATRIX $c $name th=$th $out (base)"; fi
      if [ "$out" != "$base" ]; then
        echo "MATRIX FAIL $c $name th=$th: $out != $base"; fails=$((fails+1))
      fi
    done
    echo "MATRIX PASS $c $name (threads $THS identical)"
  done
done
if [ "$fails" -gt 0 ]; then
  echo "run_matrix: $fails FAILURES"
  exit 1
fi
echo "run_matrix: ALL PASS"
