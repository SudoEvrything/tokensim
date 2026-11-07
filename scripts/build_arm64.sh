#!/bin/bash
set -e
cd "$(dirname "$0")/.." || exit 1

ZIG=${ZIG:-/tmp/armtool/zig-linux-x86_64-0.13.0/zig}
QEMU=${QEMU:-/tmp/armtool/root/usr/bin/qemu-aarch64}
BUILD=${BUILD:-build-arm64}

[ -x "$ZIG" ] || {
    echo "build_arm64: zig not found at $ZIG"
    echo "get it: curl -sL -o /tmp/armtool/zig.tar.xz https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz"
    exit 1
}

cmake -B "$BUILD" -S . \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$ZIG;cc;-target;aarch64-linux-musl" \
    -DCMAKE_CXX_COMPILER="$ZIG;c++;-target;aarch64-linux-musl" \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON \
    -DCMAKE_CROSSCOMPILING_EMULATOR="$QEMU" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD" -j 8
echo "build_arm64: OK ($BUILD)"
