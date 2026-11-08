#!/bin/bash

set -e
cd "$(dirname "$0")/.." || exit 1
gcc -O2 -mavx2 -I include scripts/microbench_windows.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-wub-o200k
gcc -O2 -mavx2 -DBENCH_DEEPSEEK -I include scripts/microbench_windows.c \
    build/libtokensim.a -lm -lpthread -o /tmp/tsimd-wub-ds
echo "o200k:"; /tmp/tsimd-wub-o200k testdata/owt.txt
echo "deepseek:"; /tmp/tsimd-wub-ds testdata/owt.txt
