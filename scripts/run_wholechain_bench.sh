#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
B=build/wholechain_bench
[ -x "$B" ] || cmake --build build --target wholechain_bench -j 6 || exit 1
[ -f /tmp/owt64m.bin ] || head -c 67108864 testdata/owt.txt > /tmp/owt64m.bin
for src in cjk testdata/owt16m.txt /tmp/owt64m.bin; do
  echo "== $src =="
  "$B" 0 1 "$src" 3
  "$B" 1 6 "$src" 3
  "$B" 2 6 "$src" 3
  "$B" 3 6 "$src" 3
done
rm -f /tmp/owt64m.bin
