#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a(const uint32_t *ids, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)ids[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: xarch_parity <format> <scheme|-> <model1> "
                "<model2|-> <file> [threads]\n");
        return 2;
    }
    const char *format = argv[1];
    const char *scheme = argv[2];
    const char *path1 = argv[3];
    const char *path2 = strcmp(argv[4], "-") ? argv[4] : NULL;
    const char *file = argv[5];
    int threads = argc == 7 ? atoi(argv[6]) : 1;
    if (threads < 1) threads = 1;

    tsimd_model_format_t fmt;
    if (strcmp(format, "tiktoken") == 0)
        fmt = TSIMD_MODEL_TIKTOKEN;
    else if (strcmp(format, "tokenizer_json") == 0)
        fmt = TSIMD_MODEL_TOKENIZER_JSON;
    else if (strcmp(format, "sentencepiece") == 0)
        fmt = TSIMD_MODEL_SENTENCEPIECE;
    else if (strcmp(format, "gpt2") == 0)
        fmt = TSIMD_MODEL_GPT2;
    else {
        fprintf(stderr, "unknown format %s\n", format);
        return 2;
    }

    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (strcmp(scheme, "-") != 0) {
        if (strcmp(scheme, "gpt2") == 0) sch = TSIMD_PRETOK_GPT2;
        else if (strcmp(scheme, "cl100k") == 0) sch = TSIMD_PRETOK_CL100K;
        else if (strcmp(scheme, "o200k") == 0) sch = TSIMD_PRETOK_O200K;
        else if (strcmp(scheme, "kimi") == 0) sch = TSIMD_PRETOK_KIMI;
        else if (strcmp(scheme, "deepseek") == 0) sch = TSIMD_PRETOK_DEEPSEEK;
        else if (strcmp(scheme, "nemo") == 0) sch = TSIMD_PRETOK_NEMO;
        else if (strcmp(scheme, "qwen25") == 0) sch = TSIMD_PRETOK_QWEN25;
        else if (strcmp(scheme, "gem4") == 0) sch = TSIMD_PRETOK_GEM4;
        else if (strcmp(scheme, "spm") == 0) sch = TSIMD_PRETOK_SPM;
        else {
            fprintf(stderr, "unknown scheme %s\n", scheme);
            return 2;
        }
    }

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof(t));
    if (tsimd_tokenizer_load_model(&t, fmt, path1, path2, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    FILE *f = fopen(file, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", file);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) {
        fprintf(stderr, "empty file\n");
        fclose(f);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    size_t cap = (size_t)sz;
    if (cap > (4u << 20)) cap = 4u << 20;
    uint8_t *buf = malloc(cap);
    if (!buf) {
        fclose(f);
        tsimd_tokenizer_free(&t);
        return 1;
    }
    size_t len = fread(buf, 1, cap, f);
    fclose(f);

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof(seq));
    int rc = threads == 1
                 ? tsimd_tokenizer_encode(&t, buf, len, &seq)
                 : tsimd_tokenizer_encode_parallel(&t, buf, len, threads,
                                                   &seq);
    if (rc != 0) {
        fprintf(stderr, "encode failed\n");
        free(buf);
        tsimd_tokenizer_free(&t);
        return 1;
    }

    printf("count=%zu hash=%016llx\n", seq.count,
           (unsigned long long)fnv1a(seq.ids, seq.count));
    if (getenv("TSIMD_XARCH_TAIL")) {
        size_t tail = (size_t)atoi(getenv("TSIMD_XARCH_TAIL"));
        if (tail > seq.count) tail = seq.count;
        for (size_t i = seq.count - tail; i < seq.count; i++)
            printf("%u ", seq.ids[i]);
        printf("\n");
    }

    free(seq.ids);
    free(buf);
    tsimd_tokenizer_free(&t);
    return 0;
}
