#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tsimd_model_load_spm(tsimd_tokenizer_t *t, const char *path,
                         const char *fp);
size_t tsimd_metaspace(const uint8_t *in, size_t len, uint8_t *out,
                       size_t cap);

#define MODEL TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"

typedef struct {
    const char *text;
    const char *expect;
} spm_case_t;

static const spm_case_t CASES[] = {

    {"hello world", "6312 28709 1526"},
    {"Port-au-Prince", "4194 28733 581 28733 3393 1261"},
    {" a  b ", "28705 264 28705 287 28705"},
    {"hello", "6312 28709"},

    {"name                  =", "1141 359 28705 327"},
    {"                ", "359 28705"},
    {"                  ", "359 2287"},
    {" ", "259"},
    {"  ", "2287"},
    {"   ", "260"},
    {"     ", "428"},
    {"        ", "2600"},
    {"                                        ", "359 359 2600"},
    {"                                                               ",
     "359 359 359 359"},
    {"                x", "359 1318"},
    {"x                ", "1318 359"},
    {"                                 =                                 ",
     "359 359 28705 327 359 359 28705"},

    {"hello\nworld\n\n", "6312 28709 13 9471 13 13"},
    {"  indented\n  more", "259 1176 12713 13 28705 680"},
    {"caf\u00e9 \u00e9t\u00e9 2024",
     "28345 7166 28705 28750 28734 28750 28781"},
    {"word                                tail",
     "1707 359 1417 28705 8675"},
};

static int encode_case(tsimd_tokenizer_t *t, tsimd_bpe_workspace_t *ws,
                       const char *text, uint32_t *out, int cap) {
    size_t len = strlen(text);
    uint8_t *norm = malloc(len * 3 + 16);
    if (!norm) return -1;
    size_t nn = tsimd_metaspace((const uint8_t *)text, len, norm, len * 3 + 16);
    if (nn == (size_t)-1) return -1;
    ws->out_count = 0;
    if (tsimd_bpe_encode_spm(&t->vocab, ws, norm, (uint32_t)nn) != 0)
        return -1;
    free(norm);
    int n = 0;
    for (size_t i = 0; i < ws->out_count && n < cap; i++)
        out[n++] = ws->out[i];
    return n;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    t.magic_ = TSIMD_TOKENIZER_MAGIC;

    if (tsimd_model_load_spm(&t, MODEL, NULL) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL);
        return 2;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 0) != 0) return 1;

    int fails = 0;
    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        char got[512];
        got[0] = 0;
        uint32_t ids[64];
        int n = encode_case(&t, &ws, CASES[c].text, ids, 64);
        for (int i = 0; i < n; i++) {
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%u", i ? " " : "", ids[i]);
            strncat(got, tmp, sizeof got - strlen(got) - 1);
        }
        if (strcmp(got, CASES[c].expect) != 0) {
            fprintf(stderr, "FAIL case %zu %s\n  got    %s\n  expect %s\n", c,
                    CASES[c].text, got, CASES[c].expect);
            fails++;
        }
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "%d spm tie-break cases FAILED\n", fails);
        return 1;
    }
    printf("test_spm_tiebreak: all %zu cases byte-exact vs the reference\n",
           sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
