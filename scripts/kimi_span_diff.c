#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIMI TSIMD_SRC_DIR "/models/reference/kimi_k2_tiktoken.model"

static void span_print(const char *label, const uint8_t *d,
                       const tsimd_span_array_t *sp) {
    printf("%s: %zu spans\n", label, (size_t)sp->count);
}

int main(int argc, char **argv) {
    const char *corpus = argc > 1 ? argv[1]
                                  : "testdata/owt.txt";
    size_t mb = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 4;

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_KIMI;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, KIMI, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 2;
    }
    size_t cap = mb << 20;
    uint8_t *data = malloc(cap);
    if (!data) return 2;
    size_t len = fread(data, 1, cap, f);
    fclose(f);

    tsimd_span_array_t a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_KIMI, data, len, &a) != 0) {
        fprintf(stderr, "scalar ref failed\n");
        return 1;
    }
    if (tsimd_pretokenize_kimi(data, len, &b) != 0) {
        fprintf(stderr, "kimi matcher failed\n");
        return 1;
    }
    int same = a.count == b.count &&
               (a.count == 0 ||
                memcmp(a.spans, b.spans, a.count * sizeof(tsimd_span_t)) == 0);
    if (!same) {
        size_t k = 0;
        while (k < a.count && k < b.count && a.spans[k].start == b.spans[k].start &&
               a.spans[k].end == b.spans[k].end)
            k++;
        fprintf(stderr,
                "DIFF at span %zu of %zu/%zu: scalar [%u,%u) matcher [%u,%u)\n",
                k, (size_t)a.count, (size_t)b.count,
                k < a.count ? a.spans[k].start : 0,
                k < a.count ? a.spans[k].end : 0,
                k < b.count ? b.spans[k].start : 0,
                k < b.count ? b.spans[k].end : 0);
        size_t lo = k > 2 ? k - 2 : 0;
        for (size_t i = lo; i < k + 4 && i < a.count; i++)
            printf("  scalar [%u,%u)\n", a.spans[i].start, a.spans[i].end);
        for (size_t i = lo; i < k + 4 && i < b.count; i++)
            printf("  matchr [%u,%u)\n", b.spans[i].start, b.spans[i].end);
        return 1;
    }
    span_print("kimi spans", data, &a);
    printf("kimi_span_diff: OK (scalar == matcher, %zu bytes)\n", len);
    tsimd_span_array_free(&a);
    tsimd_span_array_free(&b);
    free(data);
    tsimd_tokenizer_free(&t);
    return 0;
}
