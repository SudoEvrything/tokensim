#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
M=$1; FMT=$2; SCH=$3; C=$4; T=$5
cmake --build build --target corpus_hash -j 6 > /dev/null
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SZ=$(stat -c%s "$C")
CH=$((32*1024*1024))
div=$CH
while [ "$div" -ge $((1024*1024)) ]; do
  nchunks=$(( (SZ + div - 1) / div ))
  found=""
  for ((i=0; i<nchunks; i++)); do
    dd if="$C" of="$TMP/c.bin" bs=$div skip=$i count=1 status=none
    h1=$(./build/corpus_hash "$M" "$FMT" "$SCH" 1 "$TMP/c.bin" 0 2>/dev/null | tail -1)
    hT=$(./build/corpus_hash "$M" "$FMT" "$SCH" "$T" "$TMP/c.bin" 0 2>/dev/null | tail -1)
    if [ "$h1" != "$hT" ]; then
      found=$i
      echo "DIVERGENT chunk $i of $nchunks (div=$((div/1048576))MiB, file offset $((i*div))..): $h1 vs $hT"
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "no divergent chunk at div=$((div/1048576))MiB (the divergence is cross-chunk or absent)"
    exit 0
  fi
  dd if="$C" of="$TMP/c.bin" bs=$div skip=$found count=1 status=none
  C="$TMP/c.bin"
  SZ=$(stat -c%s "$C")
  div=$((div / 4))
done
echo "final divergent file: $C ($SZ bytes) — span-probe this range at threads 1 vs $T"
