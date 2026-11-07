#!/bin/bash
set -o pipefail

set -e
cd "$(dirname "$0")/.." || exit 1
STAGE=${1:-all}
cd build

if [ "$STAGE" = "asan" ] || [ "$STAGE" = "all" ]; then
  echo "== sanitize: configure ASan+UBSan"
  cmake -S .. -B ../build-san -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g" \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g" \
      -DCMAKE_BUILD_TYPE=Debug > /dev/null
  echo "== sanitize: build ASan"
  cmake --build ../build-san -j 6 > /dev/null
  echo "== sanitize: ASan unit suite"
  (cd ../build-san && ctest -j1 --output-on-failure) | tail -3
  echo "== sanitize: ASan fuzz 30s (json family)"
  clang -fsanitize=address,undefined,fuzzer -O1 -g -I ../include \
      ../tests/fuzz_all_models.c ../build-san/libtokensim.a \
      -lm -lpthread -o /tmp/tsimd-fuzz-asan
  TSIMD_FUZZ_SPEC="models/reference/deepseek_tokenizer.json;json" \
      /tmp/tsimd-fuzz-asan -max_total_time=30 -max_len=200 \
      -rss_limit_mb=4096 2>&1 | tail -1
fi

if [ "$STAGE" = "tsan" ] || [ "$STAGE" = "all" ]; then
  echo "== sanitize: configure TSan"
  cmake -S .. -B ../build-tsan-run -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_C_FLAGS="-fsanitize=thread -O1 -g" > /dev/null
  echo "== sanitize: build TSan (lib target only)"
  cmake --build ../build-tsan-run --target tokensim -j 6 > /dev/null
  echo "== sanitize: TSan parallel routes"
  gcc -fsanitize=thread -O1 -g -I ../include \
      ../tests/test_kimi_par_race.c ../build-tsan-run/libtokensim.a \
      -lpthread -lm -o /tmp/tsan-race
  setarch -R /tmp/tsan-race 2>&1 | tail -1
  gcc -fsanitize=thread -O1 -g -I ../include \
      ../tests/test_unigram_par.c ../build-tsan-run/libtokensim.a \
      ../tests/util.c -lpthread -lm -o /tmp/tsan-uni
  setarch -R /tmp/tsan-uni 2>&1 | tail -1
fi
echo "run_sanitize: DONE"
