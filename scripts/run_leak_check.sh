#!/bin/bash
set -o pipefail

set -e
cd "$(dirname "$0")/.." || exit 1
R=${1:-8}
cmake --build build --target leak_probe -j 6 >/dev/null
S=testdata/owt16m.txt
for spec in \
  "models/o200k_base.tiktoken tiktoken o200k 6" \
  "models/reference/t5_spiece.model sentencepiece auto 6" \
  "models/reference/deepseek_tokenizer.json tokenizer_json auto 6" \
  "models/reference/t5_spiece.model sentencepiece auto 1"; do
  set -- $spec
  echo "== leak: $1 ($3, th=$4)"
  ./build/leak_probe "$1" "$2" "$3" "$4" $S $R | tail -4
done
if command -v valgrind >/dev/null; then
  echo "== valgrind (t5, 1 round, 1 MiB slice)"
  head -c 1048576 $S > /tmp/tsimd_leak_slice.bin
  valgrind --error-exitcode=9 --leak-check=full \
      ./build/leak_probe models/reference/t5_spiece.model sentencepiece \
      auto 1 /tmp/tsimd_leak_slice.bin 1 2>&1 | tail -8
  rm -f /tmp/tsimd_leak_slice.bin
else
  echo "valgrind: not installed - RSS-stability probe only"
fi
echo "run_leak_check: done"
