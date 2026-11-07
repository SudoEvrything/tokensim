#include "tokenizer.h"
#include "simd.h"
#include "pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *expect_mask = getenv("EXPECT_MASK");

    setenv("TSIMD_MASK_STATS", "1", 1);
    tsimd_simd_level_t lv = tsimd_simd_level();

    if (getenv("EXPECT_NO_AVX512") != NULL) {
        if (strcmp(getenv("TSIMD_SIMD_FORCE") ? getenv("TSIMD_SIMD_FORCE")
                                              : "", "avx512") == 0 &&
            lv == TSIMD_SIMD_AVX512) {
            fprintf(stderr,
                    "FAIL: forced avx512 honored without cpu support\n");
            return 1;
        }
        printf("  dispatch: forced avx512 fell back to level %d\n",
               (int)lv);
    }

    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load(&t,
                             TSIMD_SRC_DIR "/models/"
                             "o200k_base.tiktoken",
                             TSIMD_PRETOK_O200K, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: cannot load o200k\n");
        return 1;
    }

    FILE *f = fopen(TSIMD_SRC_DIR "/testdata/owt.txt", "rb");
    if (!f) return 1;
    static uint8_t buf[16u << 20];
    size_t got = fread(buf, 1, 16u << 20, f);
    fclose(f);
    size_t doc_len = 4u << 20;
    size_t nd = got / doc_len;
    tsimd_input_span_t docs[4];
    for (size_t d = 0; d < nd; d++) {
        docs[d].data = buf + d * doc_len;
        docs[d].length = doc_len;
    }

    tsimd_token_batch_t b;
    memset(&b, 0, sizeof(b));
    if (tsimd_tokenizer_encode_batch(&t, docs, nd, &b, 4) != 0) {
        fprintf(stderr, "FAIL: encode failed\n");
        return 1;
    }
    uint64_t sum = 0;
    for (size_t i = 0; i < b.id_count; i++) sum += b.ids[i];
    tsimd_token_batch_free(&b);

    size_t mask_pieces = 0, scalar_pieces = 0;
    tsimd_o200k_mask_stats(&mask_pieces, &scalar_pieces);
    int used_mask = mask_pieces > 0;

    if (lv >= TSIMD_SIMD_AVX2) {
        if (expect_mask != NULL && strcmp(expect_mask, "0") == 0) {

        }
        if (!used_mask) {
            fprintf(stderr,
                    "FAIL: AVX2 machine ran no mask fill (pieces=0)\n");
            return 1;
        }
    } else if (used_mask) {
        fprintf(stderr,
                "FAIL: mask fill ran below AVX2 (level %d) — would "
                "SIGILL on old CPUs\n", (int)lv);
        return 1;
    }

    if (getenv("EXPECT_MASK") != NULL) {
        int want_mask = strcmp(getenv("EXPECT_MASK"), "1") == 0;
        if (used_mask != want_mask) {
            fprintf(stderr, "FAIL: mask usage %d, want %d (level %d)\n",
                    used_mask, want_mask, (int)lv);
            return 1;
        }
    }
    printf("  dispatch: level=%d mask=%d ids=%zu sum=%llx\n", (int)lv,
           used_mask, (size_t)nd * doc_len, (unsigned long long)sum);
    tsimd_tokenizer_free(&t);
    return 0;
}
