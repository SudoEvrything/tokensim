#include "tokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO TSIMD_SRC_DIR

static int run_model(const char *path, tsimd_model_format_t fmt) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, fmt, path, NULL, &sch, NULL, 0) != 0)
        return 1;
    FILE *f = fopen(REPO "/testdata/document.txt", "rb");
    if (!f) return 1;
    static uint8_t buf[1 << 21];
    size_t len = fread(buf, 1, sizeof buf, f);
    fclose(f);

    tsimd_token_sequence_t a, b;
    if (tsimd_tokenizer_encode(&t, buf, len, &a) != 0) return 1;
    if (tsimd_tokenizer_encode_parallel(&t, buf, len, 6, &b) != 0) return 1;
    int rc = 0;
    if (a.count != b.count) {
        fprintf(stderr, "FAIL unigram par %s: %zu vs %zu ids\n", path,
                a.count, b.count);
        rc = 1;
    } else if (memcmp(a.ids, b.ids, a.count * sizeof(uint32_t)) != 0) {
        size_t d = 0, at = 0;
        for (size_t i = 0; i < a.count; i++)
            if (a.ids[i] != b.ids[i]) {
                if (!at) at = i;
                d++;
            }
        fprintf(stderr, "FAIL unigram par %s: %zu ids differ, first at %zu\n",
                path, d, at);
        rc = 1;
    }
    size_t wins = 0;
    tsimd_unigram_par_stats(&wins);
    if (rc == 0 && wins == 0) {
        fprintf(stderr, "FAIL unigram par %s: parallel path never ran\n",
                path);
        rc = 1;
    }
    if (rc == 0)
        printf("  unigram par %s: %zu ids identical at 6 threads "
               "(%zu windows)\n", path, a.count, wins);
    tsimd_token_sequence_free(&a);
    tsimd_token_sequence_free(&b);
    tsimd_tokenizer_free(&t);
    return rc;
}

int main(void) {
    if (run_model(REPO "/models/reference/t5_spiece.model",
                  TSIMD_MODEL_SENTENCEPIECE))
        return 1;
    if (run_model(REPO "/models/reference/xlmr_spiece.model",
                  TSIMD_MODEL_SENTENCEPIECE))
        return 1;
    printf("unigram par: OK\n");
    return 0;
}
