




#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int encode_one(tsimd_tokenizer_t *t, const uint8_t *data,
                      size_t len) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    for (size_t i = 0; i < seq.count; i++)
        printf("%u\n", seq.ids[i]);
    tsimd_token_sequence_free(&seq);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model string... | model @stdin\n",
                argv[0]);
        return 2;
    }
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, argv[1],
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "spm_probe: load failed\n");
        return 1;
    }
    if (argc == 3 && strcmp(argv[2], "@stdin") == 0) {
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
        if (encode_one(&t, buf, len) != 0) return 1;
        free(buf);
    } else {
        for (int a = 2; a < argc; a++)
            if (encode_one(&t, (const uint8_t *)argv[a],
                           strlen(argv[a])) != 0)
                return 1;
    }
    tsimd_tokenizer_free(&t);
    return 0;
}