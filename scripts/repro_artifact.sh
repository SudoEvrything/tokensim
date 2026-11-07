#!/bin/bash
set -o pipefail

set -e
cd "$(dirname "$0")/.." || exit 1
case "$1" in
  o200k) m=models/o200k_base.tiktoken; k="tiktoken;o200k" ;;
  cl100k) m=models/cl100k_base.tiktoken; k="tiktoken;cl100k" ;;
  kimi) m=models/reference/kimi_k2_tiktoken.model; k="tiktoken;kimi" ;;
  gemma3-4b) m=models/reference/gemma3-4b_tokenizer.json; k=json ;;
  deepseek) m=models/reference/deepseek_tokenizer.json; k=json ;;
  qwen2.5) m=models/reference/qwen2.5-0.5b_tokenizer.json; k=json ;;
  nemotron) m=models/reference/nemotron3_nano_tokenizer.json; k=json ;;
  smollm2) m=models/reference/smollm2_tokenizer.json; k=json ;;
  glm4) m=models/reference/glm4_tokenizer.json; k=json ;;
  gemma4) m=models/reference/gemma4_tokenizer.json; k=json ;;
  mistral) m=models/reference/mistral7b_tokenizer.model; k=spm ;;
  xlmr) m=models/reference/xlmr_spiece.model; k=spm ;;
  t5) m=models/reference/t5_spiece.model; k=spm ;;
  *) echo "unknown family: $1"; exit 1 ;;
esac
clang -fsanitize=fuzzer -O2 -g -I include tests/fuzz_all_models.c \
    build/libtokensimd.a -lm -lpthread -o /tmp/tsimd-fuzz
TSIMD_FUZZ_SPEC="$m;$k" TSIMD_FUZZ_DBG=1 /tmp/tsimd-fuzz "$2" 2>&1 |
    grep -E "MISMATCH|mine:|ref :|executed" | head -6
