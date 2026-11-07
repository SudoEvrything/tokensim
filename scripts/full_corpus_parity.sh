#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
W=${1:-both}
cmake --build build --target ids_parity corpus_hash -j 6 >/dev/null

if [ "$W" = "docs" ] || [ "$W" = "both" ]; then
  echo "== docs: document.txt o200k vs document.ids.txt (the reference ids)"
  ./build/ids_parity models/o200k_base.tiktoken tiktoken o200k 6 \
      testdata/document.txt testdata/document.ids.txt
fi

if [ "$W" = "owt" ] || [ "$W" = "both" ]; then
  F=testdata/owt.txt
  echo "== owt: o200k (tiktoken reference, chunked)"
  python3 scripts/derive_corpus_hash.py tiktoken o200k_base $F 1
  ./build/corpus_hash models/o200k_base.tiktoken tiktoken o200k 6 $F 1
  echo "== owt: deepseek (tokenizers reference, chunked)"
  python3 scripts/derive_corpus_hash.py tokenizers \
      models/reference/deepseek_tokenizer.json $F 1
  ./build/corpus_hash models/reference/deepseek_tokenizer.json \
      tokenizer_json auto 6 $F 1
  echo "== owt: t5 (sentencepiece reference, chunked)"
  python3 scripts/derive_corpus_hash.py sentencepiece \
      models/reference/t5_spiece.model $F 1
  ./build/corpus_hash models/reference/t5_spiece.model sentencepiece auto 6 \
      $F 1
  echo "== owt: xlmr (sentencepiece reference, chunked)"
  python3 scripts/derive_corpus_hash.py sentencepiece \
      models/reference/xlmr_spiece.model $F 1
  ./build/corpus_hash models/reference/xlmr_spiece.model sentencepiece auto \
      6 $F 1
fi
echo "full_corpus_parity: done"
