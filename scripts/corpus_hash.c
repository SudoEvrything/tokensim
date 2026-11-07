#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s model format scheme threads corpus [chunk-mib]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *format = argv[2];
    const char *scheme = argv[3];
    int threads = atoi(argv[4]);
    const char *corpus = argv[5];
    size_t chunk = (size_t)(argc > 6 ? strtoul(argv[6], NULL, 10) : 0)
                   << 20;

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    tsimd_model_format_t kind = TSIMD_MODEL_TIKTOKEN;
    if (strcmp(format, "tiktoken") == 0) kind = TSIMD_MODEL_TIKTOKEN;
    else if (strcmp(format, "sentencepiece") == 0)
        kind = TSIMD_MODEL_SENTENCEPIECE;
    else if (strcmp(format, "tokenizer_json") == 0)
        kind = TSIMD_MODEL_TOKENIZER_JSON;
    if (strcmp(scheme, "o200k") == 0) sch = TSIMD_PRETOK_O200K;
    else if (strcmp(scheme, "cl100k") == 0) sch = TSIMD_PRETOK_CL100K;
    else if (strcmp(scheme, "kimi") == 0) sch = TSIMD_PRETOK_KIMI;
    else if (strcmp(scheme, "gem4") == 0) sch = TSIMD_PRETOK_GEM4;
    else if (strcmp(scheme, "spm") == 0) sch = TSIMD_PRETOK_SPM;
    else if (strcmp(scheme, "deepseek") == 0) sch = TSIMD_PRETOK_DEEPSEEK;
    else if (strcmp(scheme, "nemo") == 0) sch = TSIMD_PRETOK_NEMO;
    else if (strcmp(scheme, "qwen25") == 0) sch = TSIMD_PRETOK_QWEN25;
    if (tsimd_tokenizer_load_model(&tok, kind, model, NULL, &sch, NULL,
                                   0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 1;
    }
    unsigned long long hv = 1469598103934665603ULL;
    size_t total = 0;
    if (chunk == 0) {
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *buf = (uint8_t *)malloc((size_t)len);
        if (!buf) return 1;
        if (fread(buf, 1, (size_t)len, f) != (size_t)len) return 1;
        fclose(f);
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        int rc = tsimd_tokenizer_encode_parallel(&tok, buf, (size_t)len,
                                                 threads, &seq);
        if (rc != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        for (size_t i = 0; i < seq.count; i++) {
            hv ^= seq.ids[i];
            hv *= 1099511628211ULL;
        }
        total = seq.count;
        tsimd_token_sequence_free(&seq);
        free(buf);
    } else {
        uint8_t *buf = (uint8_t *)malloc(chunk + (1u << 20));
        if (!buf) return 1;
        size_t tail = 0;
        size_t got;
        while ((got = fread(buf + tail, 1, chunk, f)) > 0) {
            size_t n = tail + got;
            size_t cut = n;
            if (got == chunk) {
                while (cut > 0 && buf[cut - 1] != ' ') cut--;
                if (cut > 0) cut--;
                if (cut == 0) cut = n;
            }
            tsimd_token_sequence_t seq;
            memset(&seq, 0, sizeof seq);
            int rc = tsimd_tokenizer_encode_parallel(&tok, buf, (uint32_t)cut,
                                                     threads, &seq);
            if (rc != 0) {
                fprintf(stderr, "encode failed\n");
                return 1;
            }
            unsigned long long chv = 1469598103934665603ULL;
            for (size_t i = 0; i < seq.count; i++) {
                hv ^= seq.ids[i];
                hv *= 1099511628211ULL;
                chv ^= seq.ids[i];
                chv *= 1099511628211ULL;
            }
            total += seq.count;
            fprintf(stderr, "chunk n=%zu fnv=%llx cont=%llx\n", seq.count, chv, hv);
            tsimd_token_sequence_free(&seq);
            memmove(buf, buf + cut, n - cut);
            tail = n - cut;
        }
        if (tail > 0) {
            tsimd_token_sequence_t seq;
            memset(&seq, 0, sizeof seq);
            if (tsimd_tokenizer_encode_parallel(&tok, buf, (uint32_t)tail,
                                                threads, &seq) != 0)
                return 1;
            for (size_t i = 0; i < seq.count; i++) {
                hv ^= seq.ids[i];
                hv *= 1099511628211ULL;
            }
            total += seq.count;
            tsimd_token_sequence_free(&seq);
        }
        fclose(f);
        free(buf);
    }
    printf("count=%zu fnv=%llx\n", total, hv);
    tsimd_tokenizer_free(&tok);
    return 0;
}
