#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long rss_now(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    fclose(f);
    return kb;
}

static long hwm_now(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    fclose(f);
    return kb;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s model format scheme threads corpus rounds\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *format = argv[2];
    const char *scheme = argv[3];
    int threads = atoi(argv[4]);
    const char *corpus = argv[5];
    int rounds = atoi(argv[6]);
    size_t cap = 32u << 20;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return 1;
    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 1;
    }
    size_t n = fread(buf, 1, cap, f);
    fclose(f);

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
    if (tsimd_tokenizer_load_model(&tok, kind, model, NULL, &sch, NULL,
                                   0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    long base = rss_now();
    printf("model=%s threads=%d bytes=%zu rss0=%ldKB\n", model, threads, n,
           base);
    for (int r = 0; r < rounds; r++) {
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        int rc;
        if (threads > 1)
            rc = tsimd_tokenizer_encode_parallel(&tok, buf, n, threads, &seq);
        else
            rc = tsimd_tokenizer_encode(&tok, buf, n, &seq);
        if (rc != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        printf("round=%d count=%zu rss=%ldKB\n", r, seq.count, rss_now());
        tsimd_token_sequence_free(&seq);
    }
    printf("hwm=%ldKB\n", hwm_now());
    tsimd_tokenizer_free(&tok);
    free(buf);
    return 0;
}
