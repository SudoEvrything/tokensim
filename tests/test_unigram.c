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
    {"Port-au-Prince", "13790 8 915 8 112078 328"},
    {" a  b ", "9 875"},
    {"hello", "33599 30"},
    {"caf\u00e9 \u00e9t\u00e9 2024", "26215 5324 386 2356"},
    {"  indented\n  more", "18596 3673 1285"},
    {"name                  =", "9350 2202"},
    {"x", "1021"},
    {" x", "1021"},
    {"x ", "1021"},
    {"x  y", "1021 112"},
    {"x\n", "1021"},
    {"\nx", "1021"},
    {"x\ty", "1021 112"},
    {"x y  z", "1021 112 96"},
    {"  ", ""},
    {"", ""},
    {"Hello world", "35377 8998"},
    {"a  b  c", "9 875 500"},
    {"x\r\ny", "1021 112"},
    {"a\u00a0b", "9 875"},
    {"\u3000x", "1021"},
    {"x\u3000", "1021"},
    {"                ", ""},
    {"                  ", ""},
    {"                                        ", ""},

    {"\u00af2", "5 166756 303"},
    {"\u00af 2", "5 166756 115"},
    {"a \u00af2", "9 5 166756 303"},
    {"\u00af", "5 166756"},
    {"x \u00af", "1021 5 166756"},
    {"\u00a8x", "95352 424"},
    {"\u00b4x", "3308 424"},
    {" \u00af2", "5 166756 303"},
    {"\u00af2\u00af", "5 166756 303 5 166756"},
    {"a\u00a8b", "9 95352 274"},
    {"\u00af\u00b4x", "5 166756 3308 424"},
    {"x \u00af y", "1021 5 166756 112"},

    {"a\u0301", "391"},
    {"a\u0301b", "391 274"},
    {"e\u0301", "392"},
    {"o\u0308", "11951"},
    {"n\u0303", "5 9460"},
    {"a\u0301\u0302b", "391 245913 274"},
    {"e\u0301e\u0301", "392 445"},
    {"A\u030a", "8838"},
    {"i\u0302", "41241"},
    {"u\u0308ber", "1658"},
    {"c\u0327a", "3243"},
    {"e\u0301 \u00e9", "392 392"},

    {"\u00aa", "9"},
    {"\u017f", "90"},
    {"\u00be", "5 162868"},
    {"\u0132", "5 46764"},
    {"\u0149", "20171 18"},
    {"\u01c4", "390 12372"},
    {"\u3392", "129889"},
    {"\u3392x", "129889 424"},
    {"\u00a0", ""},
    {"\x01", ""},
    {"\u2581", ""},
    {"a\x01" "b", "1562"},
    {"a \x01 b", "9 875"},
    {"a\u2028b", "9 875"},
    {"x\u2581y", "1021 112"},
    {"\u2581x", "1021"},
    {"\u00ad", "5 230"},
    {"a\u00adb", "9 230 274"},
    {"\ufdfa", "16508 1422 4471 8755"},
    {"\ufdfb", "39418 70608 175"},
    {"\u2000\u2001a", "9"},
    {"\uff5e", "5 6086"},
    {"\u32ff", "5 0"},
    {"1\u00ba", "105 30"},
};

static const uni_case_t CASES_T5[] = {
    {"hello world", "21820 296"},
    {"Port-au-Prince", "3625 18 402 18 7855 3772"},
    {" a  b ", "3 9 3 115"},
    {"hello", "21820"},
    {"caf\u00e9 \u00e9t\u00e9 2024", "11949 1366 460 2266"},
    {"  indented\n  more", "16 537 1054 72"},
    {"name                  =", "564 3274"},
    {"x", "3 226"},
    {" x", "3 226"},
    {"x ", "3 226"},
    {"x  y", "3 226 3 63"},
    {"x\n", "3 226"},
    {"\nx", "3 226"},
    {"x\ty", "3 226 3 63"},
    {"x y  z", "3 226 3 63 3 172"},
    {"  ", ""},
    {"", ""},
    {"Hello world", "8774 296"},
    {"a  b  c", "3 9 3 115 3 75"},
    {"x\r\ny", "3 226 3 63"},
    {"a\u00a0b", "3 9 3 115"},
    {"\u3000x", "3 226"},
    {"x\u3000", "3 226"},
    {"                ", ""},
    {"                  ", ""},
    {"                                        ", ""},

    {"\u00af2", "3 2 357"},
    {"\u00af 2", "3 2 204"},
    {"a \u00af2", "3 9 3 2 357"},
    {"\u00af", "3 2"},
    {"x \u00af", "3 226 3 2"},
    {"\u00a8x", "3 2 226"},
    {"\u00b4x", "3 2 226"},
    {" \u00af2", "3 2 357"},
    {"\u00af2\u00af", "3 2 357 3 2"},
    {"a\u00a8b", "3 9 3 2 115"},
    {"\u00af\u00b4x", "3 2 3 2 226"},
    {"x \u00af y", "3 226 3 2 3 63"},

    {"a\u0301", "3 2975"},
    {"a\u0301b", "3 2975 115"},
    {"e\u0301", "3 154"},
    {"o\u0308", "3 1872"},
    {"n\u0303", "3 2"},
    {"a\u0301\u0302b", "3 2975 2 115"},
    {"e\u0301e\u0301", "3 154 154"},
    {"A\u030a", "3 2"},
    {"i\u0302", "1889"},
    {"u\u0308ber", "510"},
    {"c\u0327a", "3664"},
    {"e\u0301 \u00e9", "3 154 3 154"},

    {"\u00aa", "3 9"},
    {"\u017f", "3 7"},
    {"\u00be", "220 2 591"},
    {"\u0132", "27 683"},
    {"\u0149", "3 2 29"},
    {"\u01c4", "309 2"},
    {"\u3392", "3 20210"},
    {"\u3392x", "3 20210 226"},
    {"\u00a0", ""},
    {"\x01", ""},
    {"\u2581", ""},
    {"a\x01" "b", "703"},
    {"a \x01 b", "3 9 3 115"},
    {"a\u2028b", "3 9 3 115"},
    {"x\u2581y", "3 226 3 63"},
    {"\u2581x", "3 226"},
    {"\u00ad", "3 575"},
    {"a\u00adb", "3 9 575 115"},
    {"\ufdfa", "3 2 3 2 3 2 3 2"},
    {"\ufdfb", "3 2 3 2"},
    {"\u2000\u2001a", "3 9"},
    {"\uff5e", "3 2"},
    {"\u32ff", "3 2"},
    {"1\u00ba", "209 32"},
};

typedef struct {
    const char *bytes;
    size_t len;
    const char *expect;
} uni_byte_case_t;

static const uni_byte_case_t BYTES_T5[] = {
    {"a\x00" "b", 3, "3 9 2 115"},
    {"a\xc3" "b", 3, "3 9 2 115"},
    {"\xff\xfe", 2, "3 2"},
    {"a\xc3", 2, "3 9 2"},
    {"\xc0" "\xaf", 2, "3 2"},
    {"ab\xe2\x96", 4, "703 2"},
    {"\xed\xa0\x80", 3, "3 2"},
};

static const uni_byte_case_t BYTES_XLM[] = {
    {"a\x00" "b", 3, "9 0 274"},
    {"a\xc3" "b", 3, "9 0 274"},
    {"\xff\xfe", 2, "5 0"},
    {"a\xc3", 2, "9 0"},
    {"\xc0" "\xaf", 2, "5 0"},
    {"ab\xe2\x96", 4, "1562 0"},
    {"\xed\xa0\x80", 3, "5 0"},
};

static int run_byte_block(tsimd_tokenizer_t *t, const uni_byte_case_t *cases,
                          size_t n, const char *tag) {
    int fails = 0;
    for (size_t c = 0; c < n; c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(t, (const uint8_t *)cases[c].bytes,
                                   cases[c].len, &seq) != 0) {
            fprintf(stderr, "%s encode failed: byte case %zu\n", tag, c);
            fails++;
            continue;
        }
        char got[1024] = {0};
        for (size_t i = 0; i < seq.count; i++) {
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%u", i ? " " : "", seq.ids[i]);
            strncat(got, tmp, sizeof got - strlen(got) - 1);
        }
        if (strcmp(got, cases[c].expect) != 0) {
            fprintf(stderr, "%s FAIL byte case %zu\n  got    %s\n  expect %s\n",
                    tag, c, got, cases[c].expect);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    return fails;
}

static int run_block(tsimd_tokenizer_t *t, const uni_case_t *cases,
                      size_t n, const char *tag) {
    int fails = 0;
    for (size_t c = 0; c < n; c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(t, (const uint8_t *)cases[c].text,
                                   strlen(cases[c].text), &seq) != 0) {
            fprintf(stderr, "%s encode failed: case %zu\n", tag, c);
            fails++;
            continue;
        }
        char got[1024] = {0};
        for (size_t i = 0; i < seq.count; i++) {
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%u", i ? " " : "", seq.ids[i]);
            strncat(got, tmp, sizeof got - strlen(got) - 1);
        }
        if (strcmp(got, cases[c].expect) != 0) {
            fprintf(stderr, "%s FAIL case %zu\n  got    %s\n  expect %s\n",
                    tag, c, got, cases[c].expect);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    return fails;
}

int main(void) {
    int fails = 0;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, XLM,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", XLM);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_SPM);
    TEST_ASSERT_EQ((long long)t.vocab.vocab_size, 250000);
    fails += run_block(&t, CASES_XLM, sizeof(CASES_XLM)/sizeof(CASES_XLM[0]),
                       "xlmr");
    fails += run_byte_block(&t, BYTES_XLM, sizeof(BYTES_XLM)/sizeof(BYTES_XLM[0]),
                            "xlmr");
    tsimd_tokenizer_free(&t);

    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, T5,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", T5);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_SPM);
    TEST_ASSERT_EQ((long long)t.vocab.vocab_size, 32000);
    fails += run_block(&t, CASES_T5, sizeof(CASES_T5)/sizeof(CASES_T5[0]),
                       "t5");
    fails += run_byte_block(&t, BYTES_T5, sizeof(BYTES_T5)/sizeof(BYTES_T5[0]),
                            "t5");
    tsimd_tokenizer_free(&t);

    if (fails) {
        fprintf(stderr, "%d unigram cases FAILED\n", fails);
        return 1;
    }
    printf("test_unigram: all %zu+%zu cases byte-exact vs the reference\n",
           sizeof(CASES_XLM)/sizeof(CASES_XLM[0]),
           sizeof(CASES_T5)/sizeof(CASES_T5[0]));
    return 0;
}
