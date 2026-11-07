#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

static int check_model(const char *model, tsimd_model_format_t fmt,
                       tsimd_pretok_scheme_t sch, const char *fixture,
                       const char *label) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, fmt, model, NULL, &sch, NULL, 0) !=
        0) {
        fprintf(stderr, "FAIL %s: load\n", label);
        return 1;
    }
    FILE *fh = fopen(fixture, "rb");
    if (!fh) {
        fprintf(stderr, "FAIL %s: open %s\n", label, fixture);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = ((size_t)1 << 21) / rn + 1;
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    if (!d) {
        tsimd_tokenizer_free(&t);
        return 1;
    }
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    tsimd_token_sequence_t s1, s2;
    memset(&s1, 0, sizeof s1);
    memset(&s2, 0, sizeof s2);
    int rc = 0;
    if (tsimd_tokenizer_encode(&t, d, total, &s1) != 0) {
        fprintf(stderr, "FAIL %s: seq encode\n", label);
        rc = 1;
        goto out;
    }
    for (int th = 3; th <= 16 && rc == 0; th += 5) {
        if (tsimd_tokenizer_encode_parallel(&t, d, total, th, &s2) != 0) {
            fprintf(stderr, "FAIL %s: par encode th=%d\n", label, th);
            rc = 1;
            goto out;
        }
        if (s2.count != s1.count ||
            (s1.count && memcmp(s2.ids, s1.ids, s1.count * 4) != 0)) {
            fprintf(stderr,
                    "FAIL %s: th=%d par=%zu seq=%zu (the par path must "
                    "match the sequential on the degenerate content)\n",
                    label, th, s2.count, s1.count);
            rc = 1;
            goto out;
        }
        tsimd_token_sequence_free(&s2);
        memset(&s2, 0, sizeof s2);
    }
    printf("  %s: seq=%zu par matches at 3t/8t/13t\n", label, s1.count);
out:
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s2);
    free(d);
    tsimd_tokenizer_free(&t);
    return rc;
}

int main(void) {
    const char *fx = TSIMD_SRC_DIR "/tests/fixtures/"
                     "par_unigram_divergence";
    char path[512];
    int rc = 0;
    snprintf(path, sizeof path, "%s/mistral_par_trunc.bin", fx);
    if (check_model(
            TSIMD_SRC_DIR "/models/reference/"
            "mistral7b_tokenizer.model",
            TSIMD_MODEL_SENTENCEPIECE, TSIMD_PRETOK_SPM, path,
            "mistral spm-bpe degenerate"))
        rc = 1;
    snprintf(path, sizeof path, "%s/gemma3-4b_par_half.bin", fx);
    if (check_model(
            TSIMD_SRC_DIR "/models/reference/"
            "gemma3-4b_tokenizer.json",
            TSIMD_MODEL_TOKENIZER_JSON, TSIMD_PRETOK_GEM4, path,
            "gemma3-4b gem4 degenerate"))
        rc = 1;
    snprintf(path, sizeof path, "%s/gemma4_par_short.bin", fx);
    if (check_model(
            TSIMD_SRC_DIR "/models/reference/"
            "gemma4_tokenizer.json",
            TSIMD_MODEL_TOKENIZER_JSON, TSIMD_PRETOK_GEM4, path,
            "gemma4 gem4 degenerate"))
        rc = 1;
    if (rc) {
        fprintf(stderr, "test_spm_par_degenerate: FAILED\n");
        return 1;
    }
    printf("test_spm_par_degenerate: OK\n");
    return 0;
}
