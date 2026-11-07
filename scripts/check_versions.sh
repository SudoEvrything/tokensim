#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
V=$(sed -n 's/^#define TSIMD_VERSION_STRING "\([^"]*\)".*/\1/p' \
    include/tokensimd_version.h)
[ -n "$V" ] || { echo "check_versions: cannot read tokensimd_version.h"; exit 1; }
[ $# -ge 1 ] && [ "$1" != "$V" ] && { echo "check_versions: expected $1, header says $V"; exit 1; }
fail=0
cmake_v=$(sed -n 's/^set(TSIMD_VERSION \([^)]*\))/\1/p' CMakeLists.txt | head -1)
[ "$cmake_v" = "$V" ] || { echo "MISMATCH CMakeLists.txt: $cmake_v != $V"; fail=1; }
pp_v=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
[ "$pp_v" = "$V" ] || { echo "MISMATCH pyproject.toml: $pp_v != $V"; fail=1; }
cargo_v=$(sed -n 's/^version = "\(.*\)"/\1/p' bindings/rust/Cargo.toml | head -1)
[ "$cargo_v" = "$V" ] || { echo "MISMATCH bindings/rust/Cargo.toml: $cargo_v != $V"; fail=1; }
lib_v=$(grep -o 'version(), *"[^"]*"' bindings/rust/src/lib.rs 2>/dev/null | sed 's/.*"\(.*\)"/\1/')
[ "$lib_v" = "$V" ] || { echo "MISMATCH lib.rs: $lib_v != $V"; fail=1; }
csrc_v=$(cat bindings/rust/csrc/VERSION 2>/dev/null)
[ "$csrc_v" = "$V" ] || { echo "MISMATCH bindings/rust/csrc/VERSION: $csrc_v != $V"; fail=1; }
[ $fail -eq 0 ] && echo "check_versions: OK ($V)"
exit $fail
