#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QWEN_NFC TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"
#define QWEN_NONORM TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_nonorm.json"
#define SMOLLM2 TSIMD_SRC_DIR "/models/reference/smollm2_tokenizer.json"
#define CORPUS TSIMD_SRC_DIR "/testdata/owt.txt"

static const char *CASES[] = {
    "1234",
    "1",
    "12",
    "123",
    "1234567",
    "a1234b",
    " 1234 ",
    "x 12,345.67 y",
    "0",
    "0000 000 00 0",
    "v1.2.3",
    "2024-01-02T03:04:05Z",
    "\xD9\xA1",
    "12\xD9\xA1" "45",
    "\xD9\xA1\xD9\xA2\xD9\xA3\xD9\xA4",
    "abc\xD9\xA1\xD9\xA2\xD9\xA3" "def",
    "\xC3\xA9" "1" "\xC3\xA9",
    "3.14159",
    "1'000'000",
    "price: $1,234.56 (50% off!)",
    "\t12\t\n34\n",
    "a12B34c",

    "\xC3\xA9" "1234",
    "1234" "\xC3\xA9",
    "\xD9\xA1" "1234\xD9\xA2",
    "\xC3\xA9\xC3\xA9" "1234567",
    "a\xD9\xA1" "1234",
    "\xE2\x80\x9C" "1234" "\xE2\x80\x9D",
    "\xC3\xA9" "12" "\xC3\xA9" "34",
    "\xE2\x82\xAC" "12345",
    "12345\xE2\x82\xAC",
    "\xD9\xA1" "1\xD9\xA2" "2\xD9\xA3" "3",
    "abc\xC3\xA9" "1234567890" "\xC3\xA9xyz",
};

static int encode_ids(tsimd_tokenizer_t *t, const uint8_t *d, size_t n,
                      uint32_t **out, size_t *count) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, d, n, &seq) != 0) return -1;
    *out = seq.ids;
    *count = seq.count;
    return 0;
}

static int run_one(const char *model, const uint8_t *corpus, size_t clen) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, model, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", model);
        return 2;
    }
    int fails = 0;

    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        uint32_t *a = NULL, *b = NULL;
        size_t na = 0, nb = 0;
        setenv("TSIMD_QWEN25_SCALAR", "1", 1);
        if (encode_ids(&t, (const uint8_t *)CASES[c], strlen(CASES[c]),
                       &a, &na) != 0) {
            fprintf(stderr, "FAIL %s case %zu: scalar encode failed\n",
                    model, c);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        unsetenv("TSIMD_QWEN25_SCALAR");
        if (encode_ids(&t, (const uint8_t *)CASES[c], strlen(CASES[c]),
                       &b, &nb) != 0) {
            fprintf(stderr, "FAIL %s case %zu: serve encode failed\n",
                    model, c);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        if (na != nb) {
            fprintf(stderr, "FAIL %s case %zu \"%s\": counts %zu vs %zu\n",
                    model, c, CASES[c], na, nb);
            fails++;
        } else {
            for (size_t i = 0; i < na; i++)
                if (a[i] != b[i]) {
                    fprintf(stderr,
                            "FAIL %s case %zu \"%s\": id %zu %u vs %u\n",
                            model, c, CASES[c], i, a[i], b[i]);
                    fails++;
                    break;
                }
        }
        tsimd_token_sequence_t sa = {a, na}, sb = {b, nb};
        tsimd_token_sequence_free(&sa);
        tsimd_token_sequence_free(&sb);
    }

    if (corpus && clen) {
        uint32_t *a = NULL, *b = NULL;
        size_t na = 0, nb = 0;
        setenv("TSIMD_QWEN25_SCALAR", "1", 1);
        int rca = encode_ids(&t, corpus, clen, &a, &na);
        unsetenv("TSIMD_QWEN25_SCALAR");
        int rcb = encode_ids(&t, corpus, clen, &b, &nb);
        if (rca != 0 || rcb != 0) {
            fprintf(stderr, "FAIL %s: corpus encode failed\n", model);
            fails++;
        } else if (na != nb) {
            fprintf(stderr, "FAIL %s corpus: counts %zu vs %zu\n",
                    model, na, nb);
            fails++;
        } else {
            for (size_t i = 0; i < na; i++)
                if (a[i] != b[i]) {
                    fprintf(stderr, "FAIL %s corpus: id %zu %u vs %u\n",
                            model, i, a[i], b[i]);
                    fails++;
                    break;
                }
        }
        if (a) { tsimd_token_sequence_t s = {a, na}; tsimd_token_sequence_free(&s); }
        if (b) { tsimd_token_sequence_t s = {b, nb}; tsimd_token_sequence_free(&s); }
    }
    if (!fails)
        printf("%s: scalar == serve on %zu cases%s\n", model,
               sizeof(CASES) / sizeof(CASES[0]),
               corpus && clen ? " + corpus" : "");
    tsimd_tokenizer_free(&t);
    return fails;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 2;
    FILE *f = fopen(CORPUS, "rb");
    uint8_t *corpus = NULL;
    size_t clen = 0;
    if (f) {
        clen = mib << 20;
        corpus = malloc(clen);
        clen = fread(corpus, 1, clen, f);
        fclose(f);
    }
    int fails = run_one(QWEN_NONORM, corpus, clen);
    fails += run_one(QWEN_NFC, corpus, clen);
    if (access(SMOLLM2, 0) == 0) {
        setenv("TSIMD_GPT2D_SPAN", "0", 1);
        fails += run_one(SMOLLM2, corpus, clen);
        unsetenv("TSIMD_GPT2D_SPAN");
    }
    free(corpus);
    if (fails) {
        fprintf(stderr, "qwen25 serve parity FAILED\n");
        return 1;
    }
    printf("test_qwen25_serve: OK\n");
    return 0;
}
