#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XLM TSIMD_SRC_DIR "/models/reference/xlmr_spiece.model"
#define T5 TSIMD_SRC_DIR "/models/reference/t5_spiece.model"

typedef struct {
    const char *text;
    const char *expect;
} uni_case_t;

static const uni_case_t CASES_XLM[] = {
    {"hello world", "33599 30 8998"},
    {"the the the the", "69 69 69 69"},
    {" a  b  c  d ", "9 875 500 103"},
    {"hello world hello world hello world",
     "33599 30 8998 33599 30 8998 33599 30 8998"},
    {"  indented\n  more", "18596 3673 1285"},
    {"x y z x y z", "1021 112 96 1021 112 96"},
};

static const uni_case_t CASES_T5[] = {
    {"hello world", "21820 296"},
    {"the the the the", "8 8 8 8"},
    {" a  b  c  d ", "3 9 3 115 3 75 3 26"},
    {"hello world hello world hello world",
     "21820 296 21820 296 21820 296"},
    {"  indented\n  more", "16 537 1054 72"},
    {"x y z x y z", "3 226 3 63 3 172 3 226 3 63 3 172"},
};

static int run_cases(tsimd_tokenizer_t *t, const uni_case_t *cases,
                     size_t n_cases, const char *name) {
    int fails = 0;
    for (size_t c = 0; c < n_cases; c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(t, (const uint8_t *)cases[c].text,
                                   strlen(cases[c].text), &seq) != 0) {
            fprintf(stderr, "FAIL %s case %zu: encode failed\n", name, c);
            fails++;
            continue;
        }
        char got[512] = {0};
        for (size_t i = 0; i < seq.count; i++) {
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%u", i ? " " : "", seq.ids[i]);
            strncat(got, tmp, sizeof got - strlen(got) - 1);
        }
        if (strcmp(got, cases[c].expect) != 0) {
            fprintf(stderr, "FAIL %s case %zu %s\n  got    %s\n  expect %s\n",
                    name, c, cases[c].text, got, cases[c].expect);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    return fails;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "whole") == 0)
        setenv("TSIMD_DISABLE_UNIGRAM_MEMO", "1", 1);
    else
        setenv("TSIMD_ENABLE_UNIGRAM_MEMO", "1", 1);
    int fails = 0;
    tsimd_tokenizer_t tx, tt;
    memset(&tx, 0, sizeof tx);
    memset(&tt, 0, sizeof tt);

    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tx, TSIMD_MODEL_SENTENCEPIECE, XLM, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", XLM);
        return 2;
    }
    fails += run_cases(&tx, CASES_XLM,
                       sizeof(CASES_XLM) / sizeof(CASES_XLM[0]), "xlmr");
    tsimd_tokenizer_free(&tx);

    scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tt, TSIMD_MODEL_SENTENCEPIECE, T5, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", T5);
        return 2;
    }
    fails += run_cases(&tt, CASES_T5,
                       sizeof(CASES_T5) / sizeof(CASES_T5[0]), "t5");
    tsimd_tokenizer_free(&tt);

    if (fails) {
        fprintf(stderr, "%d unigram-memo cases FAILED\n", fails);
        return 1;
    }
    printf("test_unigram_memo: all cases byte-exact vs the reference\n");
    return 0;
}
