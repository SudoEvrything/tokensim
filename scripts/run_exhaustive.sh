#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
gcc -O2 -mavx2 -I include scripts/exhaustive_algebra.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-exh
/tmp/tsimd-exh
