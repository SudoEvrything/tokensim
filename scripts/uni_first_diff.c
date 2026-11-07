#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

static int hexdump(const char *tag, const uint8_t *p, size_t n) {
    printf("%s:", tag);
    for (size_t i = 0; i < n && i < 48; i++) printf(" %02x", p[i]);
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s model fmt_json|spm|tiktoken scheme fixture reps th\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    tsimd_model_format_t fmt = strcmp(argv[2], "spm") == 0
                                   ? TSIMD_MODEL_SENTENCEPIECE
                                   : strcmp(argv[3], "tiktoken") == 0
                                         ? TSIMD_MODEL_TIKTOKEN
                                         : TSIMD_MODEL_TOKENIZER_JSON;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (strcmp(argv[3], "spm") == 0) sch = TSIMD_PRETOK_SPM;
    else if (strcmp(argv[3], "auto") == 0) sch = TSIMD_PRETOK_GPT2;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, fmt, model, NULL, &sch, NULL, 0) !=
        0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    FILE *fh = fopen(argv[4], "rb");
    if (!fh) return 1;
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = (size_t)atoi(argv[5]);
    int th = atoi(argv[6]);
    size_t min_reps = ((1u << 18) + rn - 1) / rn;
    size_t max_reps = (8u << 20) / rn;
    if (reps < min_reps) reps = min_reps;
    if (reps > max_reps) reps = max_reps;
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    tsimd_token_sequence_t s1, s2;
    memset(&s1, 0, sizeof s1);
    memset(&s2, 0, sizeof s2);
    if (tsimd_tokenizer_encode(&t, d, total, &s1) != 0) {
        printf("seq failed\n");
        return 1;
    }
    if (tsimd_tokenizer_encode_parallel(&t, d, total, th, &s2) != 0) {
        printf("par failed\n");
        return 1;
    }
    printf("seq=%zu par=%zu\n", s1.count, s2.count);
    printf("seq ids:");
    for (size_t k = 0; k < s1.count && k < 24; k++) printf(" %u", s1.ids[k]);
    printf("\npar ids:");
    for (size_t k = 0; k < s2.count && k < 24; k++) printf(" %u", s2.ids[k]);
    printf("\n");

    tsimd_byte_sequence_t d1, d2;
    memset(&d1, 0, sizeof d1);
    memset(&d2, 0, sizeof d2);
    if (tsimd_tokenizer_decode(&t, s1.ids, s1.count, 0, &d1) != 0) {
        printf("decode seq failed\n");
        return 1;
    }
    if (tsimd_tokenizer_decode(&t, s2.ids, s2.count, 0, &d2) != 0) {
        printf("decode par failed\n");
        return 1;
    }
    size_t n1 = d1.len, n2 = d2.len;
    const uint8_t *b1 = d1.bytes, *b2 = d2.bytes;
    printf("decoded seq=%zu par=%zu (raw %zu)\n", n1, n2, total);
    size_t i = 0;
    while (i < n1 && i < n2 && b1[i] == b2[i]) i++;
    if (i == n1 && i == n2) {
        printf("MATCH\n");
    } else {
        printf("FIRST-DIFF at %zu (seq_n=%zu par_n=%zu)\n", i, n1, n2);
        hexdump("raw ", d + (i > 16 ? i - 16 : 0), 48);
        hexdump("seq ", b1 + (i > 16 ? i - 16 : 0), 48);
        hexdump("par ", b2 + (i > 16 ? i - 16 : 0), 48);
    }
    tsimd_byte_sequence_free(&d1);
    tsimd_byte_sequence_free(&d2);
    free(d);
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s2);
    tsimd_tokenizer_free(&t);
    return 0;
}
