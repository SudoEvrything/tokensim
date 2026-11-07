#!/bin/bash
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
FAILS=0
T=build/tsimd-tokenize
F=testdata/owt16m.txt
[ -f "$F" ] || head -c 16777216 testdata/owt.txt > "$F"
TH=1
case "$2" in 4t) TH=4 ;; 6t) TH=6 ;; 10t) TH=10 ;; 16t) TH=16 ;; esac
one() {
  local name="$1" want="$2"; shift 2
  local out tok
  out=$("$T" --threads "$TH" "$@" 2>/tmp/g.err | md5sum | cut -d' ' -f1)
  tok=$(grep -o '[0-9]* tokens' /tmp/g.err | head -1)
  if [ "$out" = "$want" ]; then
    printf 'OK   %-10s %s (%s) th=%s\n' "$name" "$out" "$tok" "$TH"
  else
    printf 'FAIL %-10s got=%s want=%s (%s) th=%s\n' "$name" "$out" "$want" "$tok" "$TH"
    FAILS=1
  fi
}
one o200k    c8fdcf233e546eef195d6f886d88ebc5 --format tiktoken models/o200k_base.tiktoken o200k "$F"
one cl100k   a8fbfce3b1e73e97ca75af0511ec680f --format tiktoken models/cl100k_base.tiktoken cl100k "$F"
one deepseek 10a2a401bf7a6695fe821836d95752e4 --format tokenizer_json models/reference/deepseek_tokenizer.json auto "$F"
one qwen2.5  95ac549c8688e7c6422518c0e0092642 --format tokenizer_json models/reference/qwen2.5-0.5b_tokenizer.json auto "$F"
one smollm2  cd349fc214b3da708dbf9e6e2704ac9f --format tokenizer_json models/reference/smollm2_tokenizer.json auto "$F"
one gemma4   924a8796ddce779d71cc6e5cf81e32cf --format tokenizer_json models/reference/gemma4_tokenizer.json auto "$F"
one mistral  ff04be8fda4d115c0b078131e4e96582 --format sentencepiece models/reference/mistral7b_tokenizer.model auto "$F"
one xlmr     7db6b4f9abba71816ba9785757ea0dc1 --format sentencepiece models/reference/xlmr_spiece.model auto "$F"
one nemotron 758a438b21e502ab70e0cced5166a0d9 --format tokenizer_json models/reference/nemotron3_nano_tokenizer.json auto "$F"
one glm4     4fb7a92780e4e3d0e31bce0860858bf7 --format tokenizer_json models/reference/glm4_tokenizer.json auto "$F"
one gemma3-4b 167ed38f24b6e8015080cb71300700fd --format tokenizer_json models/reference/gemma3-4b_tokenizer.json auto "$F"
one kimi     e9768d518da489e970d85192331b44a9 --format tiktoken models/reference/kimi_k2_tiktoken.model kimi "$F"
one t5       677b5ab2b792b157f51e4a3c90ae2443 --format sentencepiece models/reference/t5_spiece.model auto "$F"
exit $FAILS
