#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define O200K TSIMD_SRC_DIR "/models/o200k_base.tiktoken"

static const struct { const char *in; const char *ids; } CASES[] = {
    {"0\u094d're", "15 682 4118"},
    {"a\u094d're", "64 682 4118"},
    {" \u094d're", "4999 235 4118"},
    {"\u094d're", "682 4118"},
    {"0\u094d's", "15 682 885"},
    {"0\u094d't", "15 682 1507"},
    {"0\u094d'll", "15 682 6090"},
    {"0\u094d've", "15 682 7341"},
    {"0\u094d'm", "15 682 11146"},
    {"0\u094d'd", "15 682 11062"},
    {"0\u094d'RE", "15 682 6 1099"},
    {"0\u094d'S", "15 682 31233"},
    {"0\u094d'x", "15 682 168788"},
    {"0\u094d'", "15 682 6"},
    {"0\u093c're", "15 5992 4118"},
    {"0\u0301're", "15 13430 4118"},
    {"0\u05b8're", "15 5579 4118"},
    {"0\u064b're", "15 4803 4118"},
    {"0\u0c4d're", "15 1177 4118"},
    {"0\u094d\u094d're", "15 682 682 4118"},
    {"\u094d's", "682 885"},
    {"9\u094d're", "24 682 4118"},
    {"00\u094d're", "504 682 4118"},
    {"\u094d're\u094d", "682 4118 682"},
    {"a\u094d's", "64 682 885"},
    {"e\u0301's", "68 13430 885"},
    {"0\u0301's", "15 13430 885"},
};

static int ids_to_str(const tsimd_token_sequence_t *seq, char *out,
                      size_t cap) {
    size_t off = 0;
    for (size_t i = 0; i < seq->count; i++) {
        int n = snprintf(out + off, cap - off, "%s%u", i ? " " : "",
                         seq->ids[i]);
        if (n < 0 || (size_t)n >= cap - off) return -1;
        off += (size_t)n;
    }
    return 0;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_O200K;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, O200K, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", O200K);
        return 2;
    }
    int fails = 0;
    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, (const uint8_t *)CASES[c].in,
                                   strlen(CASES[c].in), &seq) != 0) {
            fprintf(stderr, "FAIL case %zu: encode failed\n", c);
            fails++;
            continue;
        }
        char got[256];
        if (ids_to_str(&seq, got, sizeof got) != 0) {
            fprintf(stderr, "FAIL case %zu: id string too long\n", c);
            fails++;
        } else if (strcmp(got, CASES[c].ids) != 0) {
            fprintf(stderr, "FAIL case %zu: got [%s] want [%s]\n", c, got,
                    CASES[c].ids);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "o200k marks: %d case(s) FAILED\n", fails);
        return 1;
    }
    printf("test_o200k_marks: all %zu cases byte-exact vs tiktoken\n",
           sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
