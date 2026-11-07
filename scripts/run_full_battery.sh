#!/bin/bash
set -o pipefail

set -e
cd "$(dirname "$0")/.." || exit 1
STAGE=${1:-all}
echo "== build"
if ! cmake --build build -j 6 > /tmp/tsimd-battery-build.log 2>&1; then
  grep -iE "error" /tmp/tsimd-battery-build.log | head -20
  exit 1
fi
grep -ciE "error" /tmp/tsimd-battery-build.log && true

if [ "$STAGE" = "parity" ] || [ "$STAGE" = "all" ]; then
  echo "== parity: gate 1t"
  bash scripts/gate_parity.sh | tail -3
  echo "== parity: gate 4t"
  bash scripts/gate_parity.sh x 4t | tail -3
  echo "== parity: exhaustive"
  bash scripts/run_exhaustive.sh | tail -1
  echo "== parity: full corpora"
  bash scripts/full_corpus_parity.sh both
fi

if [ "$STAGE" = "fuzz" ] || [ "$STAGE" = "all" ]; then
  echo "== fuzz: 60s per family"
  bash scripts/run_fuzz.sh 60
fi

if [ "$STAGE" = "leak" ] || [ "$STAGE" = "all" ]; then
  echo "== leak + memory runway"
  bash scripts/run_leak_check.sh 8
fi

if [ "$STAGE" = "bench" ] || [ "$STAGE" = "all" ]; then
  echo "== wrappers: C / Python / Rust on the full datasets"
  bash scripts/bench_wrappers.sh both
fi

if [ "$STAGE" = "giga" ] || [ "$STAGE" = "all" ]; then
  echo "== gigatoken same-window ladder (owt)"
  python3 scripts/ladder_giga.py --corpus testdata/owt.txt --mib 476 \
      --reps 2 --rounds 1 --threads 1,6,10 --engine both || \
      python3 scripts/ladder_giga.py --corpus testdata/owt.txt --mib 476 \
      --reps 2 --threads 1,6,10 --engine ours
  echo "== gigatoken same-window ladder (docs)"
  python3 scripts/ladder_giga.py --corpus testdata/document.txt --mib 495 \
      --reps 2 --rounds 1 --threads 1,6,10 --engine both || \
      python3 scripts/ladder_giga.py --corpus testdata/document.txt \
      --mib 495 --reps 2 --threads 1,6,10 --engine ours
fi
echo "run_full_battery: $STAGE done"
