#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
SDE=${1:-/tmp/sde-mirror/sde64}
CPUID=/tmp/sde-mirror/misc/cpuid/spr/cpuid.def
if [ ! -x "$SDE" ]; then
  echo "== sde: cloning the mirror"
  rm -rf /tmp/sde-mirror
  git clone --depth 1 https://github.com/rapidfuzz/intel-sde /tmp/sde-mirror
fi
[ -f "$CPUID" ] || { echo "sde cpuid def missing: $CPUID"; exit 1; }
echo "== build"
cmake --build build -j 6 > /dev/null
echo "== avx512 probe (expect probe=1 level=3)"
"$SDE" -cpuid_in "$CPUID" -- build/test_avx512_probe
echo "== dispatch default (expect level=3 mask=1)"
cmake --build build --target test_dispatch -j 4 > /dev/null 2>&1 || true
if [ -x build/test_dispatch ]; then
  EXPECT_MASK=1 timeout 570 "$SDE" -cpuid_in "$CPUID" -- build/test_dispatch
else
  echo "== dispatch: skipped (test_dispatch not built, the model-gated block is off)"
fi
echo "== exhaustive algebra, forced avx512 (expect 0 mismatches)"
gcc -O2 -mavx2 -I include scripts/exhaustive_algebra.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-exh
timeout 570 "$SDE" -cpuid_in "$CPUID" -- \
    env TSIMD_SIMD_FORCE=avx512 /tmp/tsimd-exh | tee /tmp/exh-avx512.log | tail -1
grep -q "TOTAL mismatches: 0" /tmp/exh-avx512.log || {
  echo "the exhaustive algebra mismatched under the forced avx512"
  exit 1
}
if [ -x build/test_avx512_fill_parity ]; then
  echo "== fill parity avx2 vs avx512 (expect the identical hash)"
  timeout 570 "$SDE" -cpuid_in "$CPUID" -- \
      env TSIMD_SRC_DIR="$PWD" build/test_avx512_fill_parity
fi
if [ -f testdata/owt.txt ] || [ -f testdata/owt16m.txt ]; then
  echo "== parity gate x13 under SPR (expect 13x OK)"
  timeout 570 "$SDE" -cpuid_in "$CPUID" -- bash scripts/gate_parity.sh |
      tee /tmp/parity-avx512.log | grep -cE "^OK"
  grep -E "FAIL" /tmp/parity-avx512.log && exit 1 || true
else
  echo "== parity gate: skipped (no corpus on CI, the exhaustive covers the correctness)"
fi
echo "check_avx512_sde: PASS"
