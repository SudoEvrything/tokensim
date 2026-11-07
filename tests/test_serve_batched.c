#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"
#define GPT2ENC TSIMD_SRC_DIR "/models/encoder.json"
#define GPT2BPE TSIMD_SRC_DIR "/models/vocab.bpe"
#define NEMO TSIMD_SRC_DIR "/models/reference/nemotron_tokenizer.json"
#define QWEN TSIMD_SRC_DIR "/models/reference/qwen3_tokenizer.json"

typedef struct {
    const char *text;
    const char *expect;
} case_t;

static const case_t CL_CASES[] = {
    {"hello world", "15339 1917"},
    {"Port-au-Prince", "7229 83097 12 69032"},
    {" a  b ", "264 220 293 220"},
    {"1234567", "4513 10961 22"},
    {"don't I've we're", "15357 956 358 3077 584 2351"},
    {"12345678901", "4513 10961 16474 1721"},
    {"x\ty", "87 14862"},
    {"  indented\n  more", "220 1280 16243 198 220 810"},
    {"a\n\n\nb", "64 1432 65"},
    {"word ", "1178 220"},
    {"caf\u00e9 \u00e9t\u00e9 2024", "936 59958 24560 220 2366 19"},
    {"path/to/file.txt", "2398 33529 24849 3996"},
    {"a//b", "64 322 65"},
    {"2024-01-02", "2366 19 12 1721 12 2437"},
    {"\u00e9 \u65e5\u672c\u8a9e", "978 76502 22656 45918 252"},
    {"The quick brown fox  jumps", "791 4062 14198 39935 220 35308"},
};

static const case_t GPT_CASES[] = {
    {"hello world", "31373 995"},
    {"Port-au-Prince", "13924 12 559 12 35784"},
    {" a  b ", "257 220 275 220"},
    {"1234567", "10163 2231 3134"},
    {"don't I've we're", "9099 470 314 1053 356 821"},
    {"12345678901", "10163 2231 3134 4531 486"},
    {"x\ty", "87 197 88"},
    {"  indented\n  more", "220 773 4714 198 220 517"},
    {"a\n\n\nb", "64 628 198 65"},
    {"word ", "4775 220"},
    {"caf\u00e9 \u00e9t\u00e9 2024", "66 1878 2634 220 25125 2634 48609"},
    {"path/to/file.txt", "6978 14 1462 14 7753 13 14116"},
    {"a//b", "64 1003 65"},
    {"2024-01-02", "1238 1731 12 486 12 2999"},
    {"\u00e9 \u65e5\u672c\u8a9e", "2634 10545 245 98 17312 105 45739 252"},
    {"The quick brown fox  jumps", "464 2068 7586 21831 220 18045"},
};

static const case_t NEMO_CASES[] = {
    {"hello world", "29706 4304"},
    {" a  b ", "1261 1032 1289 1032"},
    {"1234567", "1049 1050 1051 1052 1053 1054 1055"},
    {"  indented\n  more", "1032 2495 38498 1010 1032 2081"},
    {"word ", "3494 1032"},
    {"2024-01-02", "1050 1048 1050 1052 1045 1048 1049 1045 1048 1050"},
    {"x \u00b9.", "1120 1032 74363 1046"},
};

static const case_t QWEN_CASES[] = {
    {"hello world", "14990 1879"},
    {"123", "16 17 18"},
    {"1234567", "16 17 18 19 20 21 22"},
    {"\uff11\uff10", "20109 26022"},
    {"a\uff11b", "64 20109 65"},
    {"a123b", "64 16 17 18 65"},
    {" 123", "220 16 17 18"},
    {"caf\u00e9 \u1ebf 2024", "924 58858 220 26068 220 17 15 17 19"},
};

static int run_cases(tsimd_tokenizer_t *t, const case_t *cases, size_t n_cases,
                     const char *name) {
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
    if (argc > 1 && strcmp(argv[1], "perspan") == 0)
        setenv("TSIMD_DISABLE_BATCHED_SERVE", "1", 1);

    tsimd_tokenizer_t tcl, tg, tn, tq;
    memset(&tcl, 0, sizeof tcl);
    memset(&tg, 0, sizeof tg);
    memset(&tn, 0, sizeof tn);
    memset(&tq, 0, sizeof tq);

    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_CL100K;
    int fails = 0;
    if (tsimd_tokenizer_load_model(&tcl, TSIMD_MODEL_TIKTOKEN, CL100K, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", CL100K);
        return 2;
    }
    fails += run_cases(&tcl, CL_CASES,
                       sizeof(CL_CASES) / sizeof(CL_CASES[0]), "cl100k");
    tsimd_tokenizer_free(&tcl);

    scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tg, TSIMD_MODEL_GPT2, GPT2ENC, GPT2BPE,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", GPT2ENC);
        return 2;
    }
    fails += run_cases(&tg, GPT_CASES,
                       sizeof(GPT_CASES) / sizeof(GPT_CASES[0]), "gpt2");
    tsimd_tokenizer_free(&tg);

    scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tn, TSIMD_MODEL_TOKENIZER_JSON, NEMO,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", NEMO);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_NEMO);
    fails += run_cases(&tn, NEMO_CASES,
                       sizeof(NEMO_CASES) / sizeof(NEMO_CASES[0]), "nemo");
    tsimd_tokenizer_free(&tn);

    scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tq, TSIMD_MODEL_TOKENIZER_JSON, QWEN,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", QWEN);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_QWEN25);
    fails += run_cases(&tq, QWEN_CASES,
                       sizeof(QWEN_CASES) / sizeof(QWEN_CASES[0]), "qwen3");
    tsimd_tokenizer_free(&tq);

    if (fails) {
        fprintf(stderr, "%d serve-batched cases FAILED\n", fails);
        return 1;
    }
    printf("test_serve_batched: all cases byte-exact vs the reference\n");
    return 0;
}
