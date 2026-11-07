#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model fmt scheme fixture reps\n", argv[0]);
        return 2;
    }
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (strcmp(argv[3], "spm") == 0) sch = TSIMD_PRETOK_SPM;
    else if (strcmp(argv[3], "gem4") == 0) sch = TSIMD_PRETOK_GEM4;
    else if (strcmp(argv[3], "kimi") == 0) sch = TSIMD_PRETOK_KIMI;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, argv[1],
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    FILE *fh = fopen(argv[4], "rb");
    if (!fh) return 1;
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = (size_t)atoi(argv[5]);
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    for (size_t i = 0; i < total; i += rn) memcpy(d + i, raw, rn);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc = tsimd_tokenizer_encode(&t, d, total, &seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("total=%zu rc=%d count=%zu wall=%.3fs\n", total, rc, seq.count,
           dt);
    tsimd_token_sequence_free(&seq);
    free(d);
    tsimd_tokenizer_free(&t);
    return 0;
}
