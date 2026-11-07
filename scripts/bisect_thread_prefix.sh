#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
M=$1; FMT=$2; SCH=$3; C=$4; T=$5
cmake --build build --target corpus_hash -j 6 > /dev/null
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SZ=$(stat -c%s "$C")
lo=$((32*1024*1024))
hi=$SZ
while [ $((hi - lo)) -gt $((1024*1024)) ]; do
  mid=$(( (lo + hi) / 2 ))
  dd if="$C" of="$TMP/p.bin" bs=1M count=$((mid / 1048576)) status=none
  h1=$(./build/corpus_hash "$M" "$FMT" "$SCH" 1 "$TMP/p.bin" 0 2>/dev/null | tail -1)
  hT=$(./build/corpus_hash "$M" "$FMT" "$SCH" "$T" "$TMP/p.bin" 0 2>/dev/null | tail -1)
  if [ "$h1" != "$hT" ]; then hi=$mid; else lo=$mid; fi
done
echo "first divergent prefix between $((lo/1048576))MiB and $((hi/1048576))MiB"
dd if="$C" of="$TMP/div.bin" bs=1M count=$((hi / 1048576)) status=none
cp "$TMP/div.bin" /tmp/thread_diff_prefix.bin
echo "the divergent prefix saved: /tmp/thread_diff_prefix.bin ($((hi/1048576))MiB)"
echo "the divergence byte range: [$((lo/1048576))MiB, $((hi/1048576))MiB) of $C"
