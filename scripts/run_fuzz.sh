#!/bin/bash
set -o pipefail

set -e
cd "$(dirname "$0")/.." || exit 1
SEC="${1:-90}"
PB="${PAR_BUDGET:-20}"
clang -fsanitize=fuzzer -O2 -g -I include tests/fuzz_all_models.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-fuzz
clang -fsanitize=fuzzer -O2 -g -I include tests/fuzz_par_parity.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-fuzz-par
run() {
  local spec="$1" name="$2"
  echo "== $name (${SEC}s)"
  TSIMD_FUZZ_SPEC="$spec" timeout $((SEC + 40)) /tmp/tsimd-fuzz \
      -max_total_time="$SEC" -max_len=200 -rss_limit_mb=4096 \
      -malloc_limit_mb=4096 2>&1 | grep -E "MISMATCH|Done " | tail -1
  echo "== $name par (${PB}s)"
  TSIMD_FUZZ_SPEC="$spec" timeout $((PB + 60)) /tmp/tsimd-fuzz-par \
      -max_total_time="$PB" -max_len=4096 -len_control=0 -timeout=30 \
      -rss_limit_mb=4096 -malloc_limit_mb=4096 2>&1 |
      grep -E "MISMATCH|PAR-|Done |timeout after" | tail -1
}
run "models/o200k_base.tiktoken;tiktoken;o200k" o200k
run "models/cl100k_base.tiktoken;tiktoken;cl100k" cl100k
run "models/reference/gemma3-4b_tokenizer.json;json" gemma3-4b
run "models/reference/deepseek_tokenizer.json;json" deepseek
run "models/reference/qwen2.5-0.5b_tokenizer.json;json" qwen2.5
run "models/reference/nemotron3_nano_tokenizer.json;json" nemotron
run "models/reference/smollm2_tokenizer.json;json" smollm2
run "models/reference/glm4_tokenizer.json;json" glm4
run "models/reference/gemma4_tokenizer.json;json" gemma4
run "models/reference/mistral7b_tokenizer.model;spm" mistral
run "models/reference/xlmr_spiece.model;spm" xlmr
run "models/reference/t5_spiece.model;spm" t5

run "models/reference/kimi_k2_tiktoken.model;tiktoken" kimi
run "models/reference/kimi_k2_tiktoken.model;tiktoken" kimi2

run "models/reference/kimi_k2_tiktoken.model;tiktoken" kimi
