







#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tsimd_model_format_t parse_format(const char *s) {
    if (strcmp(s, "tiktoken") == 0) return TSIMD_MODEL_TIKTOKEN;
    if (strcmp(s, "gpt2") == 0) return TSIMD_MODEL_GPT2;
    if (strcmp(s, "tokenizer_json") == 0) return TSIMD_MODEL_TOKENIZER_JSON;
    if (strcmp(s, "sentencepiece") == 0) return TSIMD_MODEL_SENTENCEPIECE;
    fprintf(stderr, "encode_probe: unknown format %s\n", s);
    exit(2);
}

static int encode_one(tsimd_tokenizer_t *t, const uint8_t *data, size_t len) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    for (size_t i = 0; i < seq.count; i++)
        printf("%u\n", seq.ids[i]);
    tsimd_token_sequence_free(&seq);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s FORMAT MODEL [MODEL2|-] [@stdin|string...]\n",
                argv[0]);
        return 2;
    }
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    const char *m2 = strcmp(argv[3], "-") == 0 ? NULL : argv[3];
    if (tsimd_tokenizer_load_model(&t, parse_format(argv[1]), argv[2], m2,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "encode_probe: load failed (%s)\n", argv[2]);
        return 1;
    }
    if (argc == 5 && strcmp(argv[4], "@stdin") == 0) {
        size_t cap = 1 << 20, len = 0;
        uint8_t *buf = malloc(cap);
        if (!buf) return 1;
        for (;;) {
            if (len == cap) {
                cap <<= 1;
                uint8_t *nb = realloc(buf, cap);
                if (!nb) return 1;
                buf = nb;
            }
            size_t got = fread(buf + len, 1, 4096, stdin);
            len += got;
            if (got < 4096) break;
        }
        int rc = encode_one(&t, buf, len);
        free(buf);
        if (rc != 0) return 1;
    } else {
        for (int a = 4; a < argc; a++)
            if (encode_one(&t, (const uint8_t *)argv[a], strlen(argv[a])) != 0)
                return 1;
    }
    tsimd_tokenizer_free(&t);
    return 0;
}
