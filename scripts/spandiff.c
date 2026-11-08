#include "pretokenizer.h"
#include "internal.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump_spans(const char *label, tsimd_span_array_t *a,
                       size_t limit) {
    printf("%s: %zu spans\n", label, a->count);
    for (size_t i = 0; i < a->count && i < limit; i++)
        printf("  [%3zu] %6u..%6u\n", i, a->spans[i].start,
               a->spans[i].end);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: spandiff <gpt2|cl100k|o200k|qwen25|gpt2d|deepseek|"
                "nemo|kimi|gem4|spm> <model-file> <corpus-file> "
                "[max-bytes] [max-spans]\n");
        return 2;
    }
    tsimd_pretok_scheme_t scheme;
    if (strcmp(argv[1], "gpt2") == 0) scheme = TSIMD_PRETOK_GPT2;
    else if (strcmp(argv[1], "cl100k") == 0) scheme = TSIMD_PRETOK_CL100K;
    else if (strcmp(argv[1], "o200k") == 0) scheme = TSIMD_PRETOK_O200K;
    else if (strcmp(argv[1], "qwen25") == 0) scheme = TSIMD_PRETOK_QWEN25;
    else if (strcmp(argv[1], "gpt2d") == 0) scheme = TSIMD_PRETOK_GPT2D;
    else if (strcmp(argv[1], "deepseek") == 0)
        scheme = TSIMD_PRETOK_DEEPSEEK;
    else if (strcmp(argv[1], "nemo") == 0) scheme = TSIMD_PRETOK_NEMO;
    else if (strcmp(argv[1], "kimi") == 0) scheme = TSIMD_PRETOK_KIMI;
    else if (strcmp(argv[1], "gem4") == 0) scheme = TSIMD_PRETOK_GEM4;
    else if (strcmp(argv[1], "spm") == 0) scheme = TSIMD_PRETOK_SPM;
    else {
        fprintf(stderr, "unknown scheme %s\n", argv[1]);
        return 2;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    size_t cap = (size_t)sz;
    if (argc > 4 && atol(argv[4]) > 0 && (size_t)atol(argv[4]) < cap)
        cap = (size_t)atol(argv[4]);
    uint8_t *buf = malloc(cap);
    if (!buf || fread(buf, 1, cap, f) != cap) {
        fprintf(stderr, "read failed\n");
        return 1;
    }
    fclose(f);
    size_t max_spans = argc > 5 ? (size_t)atol(argv[5]) : 40;

    tsimd_span_array_t ref, got;
    tsimd_span_array_init(&ref);
    tsimd_span_array_init(&got);
    if (tsimd_pretokenize_scalar_ref(scheme, buf, cap, &ref) != 0) {
        fprintf(stderr, "scalar_ref failed\n");
        return 1;
    }
    if (tsimd_pretokenize_simd(scheme, buf, cap, &got) != 0) {
        fprintf(stderr, "driver failed\n");
        return 1;
    }
    printf("scalar_ref: %zu spans | driver: %zu spans\n", ref.count,
           got.count);
    if (ref.count == got.count &&
        memcmp(ref.spans, got.spans,
               ref.count * sizeof(tsimd_span_t)) == 0) {
        printf("spandiff: IDENTICAL\n");
        return 0;
    }

    size_t i = 0;
    while (i < ref.count && i < got.count &&
           ref.spans[i].start == got.spans[i].start &&
           ref.spans[i].end == got.spans[i].end)
        i++;
    printf("first divergence at span %zu:\n", i);
    printf("  ref  [%zu..%zu) = '",
           i < ref.count ? ref.spans[i].start : 0,
           i < ref.count ? ref.spans[i].end : 0);
    size_t s = i < ref.count ? ref.spans[i].start : 0;
    size_t e = i < ref.count ? ref.spans[i].end : 0;
    for (size_t k = s; k < e && k < cap; k++)
        putchar(buf[k] >= 0x20 && buf[k] < 0x7f ? buf[k] : '?');
    printf("'\n  drv  [%zu..%zu) = '", i < got.count ? got.spans[i].start : 0,
           i < got.count ? got.spans[i].end : 0);
    s = i < got.count ? got.spans[i].start : 0;
    e = i < got.count ? got.spans[i].end : 0;
    for (size_t k = s; k < e && k < cap; k++)
        putchar(buf[k] >= 0x20 && buf[k] < 0x7f ? buf[k] : '?');
    printf("'\n  context: '");
    s = i > 0 ? ref.spans[i - 1].start : 0;
    e = i + 1 < ref.count ? ref.spans[i + 1].end : (size_t)cap;
    for (size_t k = s; k < e && k < cap; k++)
        putchar(buf[k] >= 0x20 && buf[k] < 0x7f ? buf[k] : '?');
    printf("'\n");
    dump_spans("ref-tail", &ref, max_spans);
    dump_spans("drv-tail", &got, max_spans);
    return 1;
}
