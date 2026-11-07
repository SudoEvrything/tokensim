#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tokenizer.h"

int main(void) {
    const char *model = TSIMD_SRC_DIR "/models/reference/"
                        "mistral7b_tokenizer.model";
    const char *fixture = TSIMD_SRC_DIR "/tests/fixtures/"
                          "par_unigram_divergence/"
                          "mistral_giant_word_cliff.bin";
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_SPM;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: load\n");
        return 1;
    }
    FILE *fh = fopen(fixture, "rb");
    if (!fh) {
        fprintf(stderr, "FAIL: open %s\n", fixture);
        return 1;
    }
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = 8000;
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    if (!d) return 1;
    for (size_t i = 0; i < total; i += rn) memcpy(d + i, raw, rn);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc = tsimd_tokenizer_encode(&t, d, total, &seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    free(d);
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    if (rc != 0) {
        fprintf(stderr, "FAIL: encode rc=%d\n", rc);
        return 1;
    }
    printf("  giant word %zu bytes in %.3fs (%u ids)\n", total, dt,
           (unsigned)0);
    if (dt > 5.0) {
        fprintf(stderr,
                "FAIL: the giant-word encode took %.3fs (the quadratic "
                "bucket-group scan; the bound is 5.0s)\n",
                dt);
        return 1;
    }
    printf("test_giant_word_cliff: OK\n");
    return 0;
}
