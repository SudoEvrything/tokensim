#include "tokenizer.h"
#include "pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void tsimd_o200k_mask_stats(size_t *mask_pieces, size_t *scalar_pieces);

int main(void) {
    if (getenv("TSIMD_MASK_STATS") != NULL) {
        fprintf(stderr,
                "FAIL: this test must run without TSIMD_MASK_STATS (the "
                "ctest entry sets none)\n");
        return 1;
    }

    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load(&t,
                             TSIMD_SRC_DIR "/models/"
                             "o200k_base.tiktoken",
                             TSIMD_PRETOK_O200K, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: cannot load o200k\n");
        return 1;
    }

    static uint8_t buf[1u << 20];
    for (size_t i = 0; i < sizeof(buf); i++) {
        unsigned r = (unsigned)(i * 2654435761u >> 13);
        buf[i] = (uint8_t)("the quick brown fox, don't it's 42 \n\n  "
                           "\xc3\xa9\xef\xbc\xa1"[r % 40]);
    }
    size_t len = sizeof(buf);

    tsimd_token_sequence_t s1;
    if (tsimd_tokenizer_encode(&t, buf, len, &s1) != 0) {
        fprintf(stderr, "FAIL: encode failed\n");
        return 1;
    }

    size_t mask_pieces = 0, scalar_pieces = 0;
    tsimd_o200k_mask_stats(&mask_pieces, &scalar_pieces);
    if (mask_pieces != 0 || scalar_pieces != 0) {
        fprintf(stderr,
                "FAIL: mask stats non-zero without TSIMD_MASK_STATS "
                "(mask=%zu scalar=%zu) — the debug counters sit in the "
                "hot path\n",
                mask_pieces, scalar_pieces);
        return 1;
    }

    tsimd_tokenizer_t t2;
    if (tsimd_tokenizer_load(&t2,
                             TSIMD_SRC_DIR "/models/"
                             "o200k_base.tiktoken",
                             TSIMD_PRETOK_O200K, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: cannot load o200k (2nd)\n");
        return 1;
    }
    tsimd_token_sequence_t s2;
    if (tsimd_tokenizer_encode(&t2, buf, len, &s2) != 0) {
        fprintf(stderr, "FAIL: encode (2nd) failed\n");
        return 1;
    }
    if (s1.count != s2.count ||
        memcmp(s1.ids, s2.ids, (size_t)s1.count * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "FAIL: ids differ between encodes (%zu vs %zu)\n",
                (size_t)s1.count, (size_t)s2.count);
        return 1;
    }

    printf("  mask-stats gate: stats=0 without the env, ids %zu identical\n",
           (size_t)s1.count);
    free(s1.ids);
    free(s2.ids);
    tsimd_tokenizer_free(&t);
    tsimd_tokenizer_free(&t2);
    return 0;
}
