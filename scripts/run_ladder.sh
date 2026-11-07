#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
gcc -O2 -mavx2 -I include scripts/ladder_encode.c build/libtokensimd.a \
    -lm -lpthread -o /tmp/tsimd-ladder
CORPUS=${LADDER_CORPUS:-testdata/owt.txt}
fams="o200k cl100k deepseek qwen2.5 nemotron glm4 gemma4 gemma3-4b smollm2 mistral xlmr t5 kimi"
[ $# -gt 0 ] && fams="$*"
scheme_of() {
  case "$1" in
    o200k) echo o200k ;;
    cl100k) echo cl100k ;;
    kimi) echo kimi ;;
    gemma3-4b) echo gem4 ;;
    *) echo "" ;;
  esac
}
model_of() {
  case "$1" in
    o200k) echo models/o200k_base.tiktoken ;;
    kimi) echo models/reference/kimi_k2_tiktoken.model ;;
    gemma3-4b) echo models/reference/gemma3-4b_tokenizer.json ;;
    cl100k) echo models/cl100k_base.tiktoken ;;
    deepseek) echo models/reference/deepseek_tokenizer.json ;;
    qwen2.5) echo models/reference/qwen2.5-0.5b_tokenizer.json ;;
    nemotron) echo models/reference/nemotron3_nano_tokenizer.json ;;
    glm4) echo models/reference/glm4_tokenizer.json ;;
    gemma4) echo models/reference/gemma4_tokenizer.json ;;
    smollm2) echo models/reference/smollm2_tokenizer.json ;;
    mistral) echo models/reference/mistral7b_tokenizer.model ;;
    xlmr) echo models/reference/xlmr_spiece.model ;;
    t5) echo models/reference/t5_spiece.model ;;
  esac
}
for f in $fams; do
  m=$(model_of "$f")
  [ -z "$m" ] && continue
  case "$f" in
    xlmr|t5) r=1 ;;
    *) r=3 ;;
  esac
  for th in ${LADDER_THREADS:-1 6}; do
    sc=$(scheme_of "$f")
    if [ -n "$sc" ]; then
      ours=$(ENCODE_ROUNDS=$r ENCODE_SCHEME=$sc taskset -c 0-15 /tmp/tsimd-ladder "$m" "$CORPUS" "$th" 2>/dev/null | tail -1 | sed 's/RESULT //')
    else
      ours=$(ENCODE_ROUNDS=$r taskset -c 0-15 /tmp/tsimd-ladder "$m" "$CORPUS" "$th" 2>/dev/null | tail -1 | sed 's/RESULT //')
    fi
    echo "$f $(basename "$CORPUS") ${th}t ours=$ours MB/s (rounds=$r)"
  done
done
echo "NOTE: absolute rows are only comparable on an idle host; pin the"
echo "session processes to cores 8-15 first and check /proc/loadavg."
