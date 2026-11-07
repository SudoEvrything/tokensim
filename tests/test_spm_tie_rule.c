#include "spm_tie_rule.h"
#include "tokenizer.h"

#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tsimd_tokenizer_t tok;

#define MODEL TSIMD_SRC_DIR "/models/reference/t5_spiece.model"

static int load(void) {
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    const char *path = MODEL;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_SENTENCEPIECE, path, NULL,
                                   &scheme, NULL, 0) != 0)
        return 0;
    return 1;
}

static int read_bytes(const char *fixtures, const char *name, uint8_t **buf,
                      size_t *len) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bin", fixtures, name);
    *buf = (uint8_t *)tsimd_test_read_file(path, len);
    return *buf != NULL;
}

static int read_ids(const char *fixtures, const char *name, uint32_t *ids,
                    size_t *n_ids) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.expected", fixtures, name);
    size_t len = 0;
    char *text = tsimd_test_read_file(path, &len);
    if (!text) return 0;
    size_t k = 0;
    for (char *p = text; *p;) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        ids[k++] = (uint32_t)v;
        p = end;
    }
    free(text);
    *n_ids = k;
    return 1;
}

int main(int argc, char **argv) {
    const char *fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    static const char *names[] = {"t5_fuzz_mismatch", "t5_tie_negative"};
    uint8_t *buf[2] = {NULL, NULL};
    size_t len[2] = {0, 0}, wn[2] = {0, 0};
    uint32_t want[2][4096];
    float sum[2];
    int rc = 0;

    if (!load()) {
        fprintf(stderr, "FAIL: model load\n");
        return 2;
    }
    for (int i = 0; i < 2; i++) {
        if (!read_bytes(fixtures, names[i], &buf[i], &len[i])) {
            fprintf(stderr, "FAIL %s: missing .bin\n", names[i]);
            rc = 1;
            continue;
        }
        if (!read_ids(fixtures, names[i], want[i], &wn[i])) {
            fprintf(stderr, "FAIL %s: missing .expected\n", names[i]);
            rc = 1;
            continue;
        }
        sum[i] = tsimd_spm_path_score(&tok.vocab, want[i], wn[i]);
    }
    if (rc == 0) {
        uint32_t mut[4096], mut_unk[4096];
        float sm;
        int a, b, c, d;
        memcpy(mut, want[0], wn[0] * sizeof(uint32_t));
        memcpy(mut_unk, want[0], wn[0] * sizeof(uint32_t));
        for (size_t k = 0; k < wn[0]; k++) {
            if (mut[k] == 4256) {
                mut[k] = 3;
                break;
            }
        }
        for (size_t k = 0; k < wn[0]; k++) {
            if (mut_unk[k] == 2) {
                mut_unk[k] = 3;
                break;
            }
        }
        sm = tsimd_spm_path_score(&tok.vocab, mut, wn[0]);
        a = tsimd_spm_accepts_reference_score(&tok.vocab, want[0], wn[0],
                                              want[0], wn[0], sum[0], sum[0]);
        b = tsimd_spm_accepts_reference_score(&tok.vocab, want[1], wn[1],
                                              want[0], wn[0], sum[0], sum[0]);
        c = tsimd_spm_accepts_reference_score(&tok.vocab, mut, wn[0],
                                              want[0], wn[0], sm, sum[0]);
        d = tsimd_spm_accepts_reference_score(&tok.vocab, mut_unk, wn[0],
                                              want[0], wn[0], sum[0], sum[0]);
        printf("same path => %d (expect 1)\n", a);
        printf("different covering => %d (expect 0)\n", b);
        printf("piece swap, sums apart => %d (expect 0)\n", c);
        printf("unk count shift, sums tied => %d (expect 0)\n", d);
        if (!a || b || c || d) {
            fprintf(stderr, "spm_tie_rule: FAILED\n");
            return 1;
        }
    }
    free(buf[0]);
    free(buf[1]);
    tsimd_tokenizer_free(&tok);
    if (rc) {
        fprintf(stderr, "spm_tie_rule: FAILED\n");
        return 1;
    }
    fprintf(stderr, "spm_tie_rule: OK\n");
    return 0;
}
