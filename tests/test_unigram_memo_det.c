#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T5 TSIMD_SRC_DIR "/models/reference/t5_spiece.model"
#define XLM TSIMD_SRC_DIR "/models/reference/xlmr_spiece.model"
#define CORPUS TSIMD_SRC_DIR "/testdata/owt.txt"

static int cmp_streams(const char *what, const uint32_t *a, size_t na,
                       const uint32_t *b, size_t nb) {
    if (na != nb) {
        fprintf(stderr, "FAIL %s: counts differ %zu vs %zu\n", what, na, nb);
        return 1;
    }
    for (size_t i = 0; i < na; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "FAIL %s: first diff at id %zu: %u vs %u\n",
                    what, i, a[i], b[i]);
            return 1;
        }
    }
    return 0;
}

static void release(uint32_t *ids, size_t n) {
    tsimd_token_sequence_t seq = {ids, n};
    tsimd_token_sequence_free(&seq);
}

static int encode(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                  uint32_t **ids, size_t *n) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    *ids = seq.ids;
    *n = seq.count;
    return 0;
}

static int run_model(const char *model, const uint8_t *data, size_t len,
                     size_t *memo_div_out) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, model, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", model);
        return 2;
    }
    int fails = 0;
    uint32_t *a = NULL, *b = NULL, *c = NULL, *d = NULL;
    size_t na = 0, nb = 0, nc = 0, nd = 0;

    setenv("TSIMD_ENABLE_UNIGRAM_MEMO", "0", 1);
    if (encode(&t, data, len, &a, &na) != 0 ||
        encode(&t, data, len, &b, &nb) != 0) {
        fprintf(stderr, "FAIL %s: default encode failed\n", model);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    fails += cmp_streams("default pass1 vs pass2", a, na, b, nb);

    setenv("TSIMD_ENABLE_UNIGRAM_MEMO", "1", 1);
    if (encode(&t, data, len, &c, &nc) != 0 ||
        encode(&t, data, len, &d, &nd) != 0) {
        fprintf(stderr, "FAIL %s: memo encode failed\n", model);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    fails += cmp_streams("memo cold vs memo warm", c, nc, d, nd);
    {

        size_t n = na < nc ? na : nc;
        size_t i = 0;
        while (i < n && a[i] == c[i]) i++;
        if (na != nc || i < n) {
            fprintf(stderr, "NOTE %s: the opt-in memo differs from the "
                    "exact path (%zu vs %zu ids, first at %zu)\n",
                    model, nc, na, i);
            *memo_div_out += 1;
        }
    }
    if (!fails)
        printf("%s: default deterministic (%zu ids), memo deterministic\n",
               model, na);
    release(a, na);
    release(b, nb);
    release(c, nc);
    release(d, nd);
    tsimd_tokenizer_free(&t);
    return fails;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 4;
    FILE *f = fopen(CORPUS, "rb");
    if (!f) {
        fprintf(stderr, "corpus absent: %s\n", CORPUS);
        return 0;
    }
    size_t len = mib << 20;
    uint8_t *data = malloc(len);
    size_t got = fread(data, 1, len, f);
    fclose(f);
    if (got < len) len = got;

    int fails = 0;
    size_t memo_div = 0;
    fails += run_model(T5, data, len, &memo_div);
    fails += run_model(XLM, data, len, &memo_div);
    free(data);
    if (fails) {
        fprintf(stderr, "unigram memo determinism FAILED\n");
        return 1;
    }
    printf("test_unigram_memo_det: deterministic (memo residual recorded: "
           "%zu model%s)\n", memo_div, memo_div == 1 ? "" : "s");
    return 0;
}
