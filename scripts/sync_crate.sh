#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
V=$(sed -n 's/^#define TSIMD_VERSION_STRING "\([^"]*\)".*/\1/p' include/tokensimd_version.h)
[ -n "$V" ] || { echo "sync_crate: cannot read the version header"; exit 1; }
SRC=bindings/rust/csrc
rm -rf "$SRC"
mkdir -p "$SRC"
cp -r src "$SRC/src"
cp -r include "$SRC/include"
cp LICENSE "$SRC/LICENSE"
echo "$V" > "$SRC/VERSION"
echo "sync_crate: vendored src+include+LICENSE at version $V"
find "$SRC" -type f | wc -l
