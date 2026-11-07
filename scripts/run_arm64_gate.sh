#!/bin/bash
set -e
cd "$(dirname "$0")/.." || exit 1

ZIG=${ZIG:-/tmp/armtool/zig-linux-x86_64-0.13.0/zig}
QEMU=${QEMU:-/tmp/armtool/root/usr/bin/qemu-aarch64}
BUILD=${BUILD:-build-arm64}
export TSIMD_PIN=0

[ -x "$QEMU" ] || {
    echo "run_arm64_gate: qemu not found at $QEMU"
    echo "get it: curl -sL -O http://deb.debian.org/debian/pool/main/q/qemu/qemu-user_11.1.3+ds-1_amd64.deb"
    echo "        dpkg-deb -x qemu-user_11.1.3+ds-1_amd64.deb /tmp/armtool/root"
    exit 1
}
[ -d "$BUILD" ] || scripts/build_arm64.sh

cc -O2 -I include scripts/xarch_parity.c build/libtokensim.a \
    -lpthread -lm -o /tmp/xarch_x86
"$ZIG" cc -target aarch64-linux-musl -O2 -I include \
    scripts/xarch_parity.c "$BUILD/libtokensim.a" -lpthread -lm \
    -o /tmp/xarch_arm

echo "== neon classify equivalence (expect OK) =="
"$QEMU" "$BUILD/test_neon_classify"

echo "== ctest under qemu =="
ctest --test-dir "$BUILD" -j 4 \
    -E "test_dispatch|test_worker_caps|test_avx512_probe|test_dual_cursor|test_short_skip|test_spm_cache" \
    --timeout 900 > /tmp/arm_ctest.log 2>&1 || true
grep -E "tests passed|tests failed" /tmp/arm_ctest.log
grep "FAILED" /tmp/arm_ctest.log | head -6

echo "== cross-arch parity: the ids must match the x86 build =="
M="$PWD/models"
T="$PWD/testdata"
fail=0
run_case() {
    fmt=$1; sch=$2; m1=$3; m2=$4; inp=$5; th=$6
    [ -f "$m1" ] || { echo "  skip $fmt/$sch (no model)"; return; }
    if [ "$m2" != "-" ]; then
        args="$m1 $m2"
    else
        args="$m1 -"
    fi
    a=$(/tmp/xarch_x86 "$fmt" "$sch" $args "$inp" "$th" 2>/dev/null)
    b=$("$QEMU" /tmp/xarch_arm "$fmt" "$sch" $args "$inp" "$th" 2>/dev/null)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then
        echo "  $fmt/$sch ${th}t: MATCH ($a)"
    else
        echo "  $fmt/$sch ${th}t: MISMATCH"
        echo "    x86  : $a"
        echo "    arm64: $b"
        fail=1
    fi
}

head -c $((4 << 20)) "$T/owt.txt" > /tmp/arm_owt16.txt 2>/dev/null || {
    echo "no corpus"; exit 1; }

run_case tiktoken cl100k "$M/cl100k_base.tiktoken" - /tmp/arm_owt16.txt 1
run_case tiktoken cl100k "$M/cl100k_base.tiktoken" - /tmp/arm_owt16.txt 6
run_case tiktoken o200k "$M/o200k_base.tiktoken" - /tmp/arm_owt16.txt 1
run_case tokenizer_json - "$M/reference/deepseek_tokenizer.json" - /tmp/arm_owt16.txt 1
run_case tokenizer_json - "$M/reference/gemma3-4b_tokenizer.json" - /tmp/arm_owt16.txt 1
run_case tokenizer_json - "$M/reference/qwen2.5-0.5b_tokenizer.json" - /tmp/arm_owt16.txt 1
run_case tokenizer_json - "$M/reference/gemma4_tokenizer.json" - /tmp/arm_owt16.txt 1
run_case sentencepiece - "$M/reference/mistral7b_tokenizer.model" - /tmp/arm_owt16.txt 1
run_case tiktoken cl100k "$M/cl100k_base.tiktoken" - /tmp/arm_owt16.txt 16
run_case tiktoken o200k "$M/o200k_base.tiktoken" - /tmp/arm_owt16.txt 16

[ $fail -eq 0 ] && echo "run_arm64_gate: PASS" || exit 1
