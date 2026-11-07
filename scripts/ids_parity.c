#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s model format scheme threads corpus ids_ref\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *format = argv[2];
    const char *scheme = argv[3];
    int threads = atoi(argv[4]);
    const char *corpus = argv[5];
    const char *ref = argv[6];

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

    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)len);
    if (!buf) return 1;
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc = tsimd_tokenizer_encode_parallel(&tok, buf, (size_t)len, threads,
                                             &seq);
    if (rc != 0) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }

    FILE *rf = fopen(ref, "rb");
    if (!rf) {
        fprintf(stderr, "open %s\n", ref);
        return 1;
    }
    static char line[1 << 16];
    size_t li = 0;
    size_t idx = 0;
    size_t total_ref = 0;
    int mismatches = 0;
    int c;
    while (idx < seq.count && mismatches < 8 && (c = getc(rf)) != EOF) {
        if (c == ' ' || c == '\n') {
            line[li] = 0;
            if (li > 0) {
                unsigned long v = strtoul(line, NULL, 10);
                total_ref++;
                if (seq.ids[idx] != (uint32_t)v) {
                    printf("MISMATCH idx=%zu ref=%lu ours=%u\n", idx, v,
                           seq.ids[idx]);
                    mismatches++;
                }
                idx++;
                li = 0;
            }
        } else if (li + 1 < sizeof line) {
            line[li++] = (char)c;
        }
    }
    while (mismatches < 8 && idx < seq.count) {
        printf("MISMATCH idx=%zu ref=EOF ours=%u\n", idx, seq.ids[idx]);
        mismatches++;
        idx++;
    }
    printf("ours=%zu ref=%zu mismatches=%d\n", seq.count, total_ref,
           mismatches);
    int ok = mismatches == 0 && seq.count == total_ref;
    fclose(rf);
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&tok);
    free(buf);
    return ok ? 0 : 1;
}
