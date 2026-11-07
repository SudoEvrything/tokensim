#!/bin/bash

set -o pipefail
set -e
cd "$(dirname "$0")/.." || exit 1
R6=${1:-300}
R16=${2:-150}
cmake --build build --target corpus_hash -j 6 > /dev/null
M=models/reference/nemotron3_nano_tokenizer.json
F=testdata/owt.txt
WANT="count=111641124 fnv=1b7f481787fae98"
FAILS=0
echo "== 6t x $R6"
for i in $(seq 1 "$R6"); do
  got=$(./build/corpus_hash "$M" tokenizer_json auto 6 "$F" 1 | tail -1)
  if [ "$got" != "$WANT" ]; then
    echo "FAIL 6t run $i: $got"
    FAILS=1
  fi
done
echo "== 16t x $R16"
for i in $(seq 1 "$R16"); do
  got=$(./build/corpus_hash "$M" tokenizer_json auto 16 "$F" 1 | tail -1)
  if [ "$got" != "$WANT" ]; then
    echo "FAIL 16t run $i: $got"
    FAILS=1
  fi
done
if [ $FAILS -eq 0 ]; then
  echo "findc_stress: ALL CLEAN ($R6 x 6t + $R16 x 16t = $WANT)"
fi
exit $FAILS
