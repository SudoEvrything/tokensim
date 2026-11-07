#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/reference/nemotron_tokenizer.json"

typedef struct {
    const char *text;
    const char *expect;
} nemo_case_t;

static const nemo_case_t CASES[] = {
    {"hello world", "29706 4304"},
    {"Port-au-Prince", "12596 44179 63776 2932"},
    {" a  b ", "1261 1032 1289 1032"},
    {"hello", "29706"},
    {"1234567", "1049 1050 1051 1052 1053 1054 1055"},
    {"ABC_DEF", "37638 113027"},
    {"ABCdef abcDEF", "37638 3149 89031 104570"},
    {"don't I've we're", "21797 2405 1362 6483 1729 6185"},
    {" a  b  c", "1261 1032 1289 1032 1272"},
    {"x  y", "1120 1032 1404"},
    {"  indented\n  more", "1032 2495 38498 1010 1032 2081"},
    {"a\n\n\nb", "1097 4688 1098"},
    {"\nx", "1010 1120"},
    {"x\n", "1120 1010"},
    {"x\ty", "1120 67002"},
    {"caf\u00e9 \u00e9t\u00e9 2024",
     "3173 1102 1337 5320 1032 1050 1048 1050 1052"},
    {"\u00e9 \u65e5\u672c\u8a9e", "1337 30367 15199"},
    {"path/to/file.txt", "7368 35466 55998 13477"},
    {"a//b", "1097 1555 1098"},
    {"word ", "3494 1032"},
    {"\t\t tab", "1458 10271"},
    {"2024-01-02",
     "1050 1048 1050 1052 1045 1048 1049 1045 1048 1050"},
    {"  ", "1256"},
    {"   ", "1293"},

    {" a 1 b", "1261 1032 1049 1289"},
    {" \u00b9", "1032 74363"},
    {" \u00b2b", "1032 16077 1098"},
    {"x \u00b9.", "1120 1032 74363 1046"},
    {"a \u0663 c", "1097 1032 77729 1272"},
    {" \u00b2 \u00b3", "1032 16077 1032 32168"},
    {" \u2070", "1032 23097 1176"},
    {"P \u2160 q", "1080 1032 72861 4367"},
    {"\uff11\uff11", "11181 11181"},
    {"a\uff11b", "1097 11181 1098"},
    {"Brussels \u00b9 a \u00b2 b",
     "20488 2512 2397 1032 74363 1261 1032 16077 1289"},
};

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, MODEL,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_NEMO);

    int fails = 0;
    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, (const uint8_t *)CASES[c].text,
                                   strlen(CASES[c].text), &seq) != 0) {
            fprintf(stderr, "encode failed: case %zu\n", c);
            fails++;
            continue;
        }
        char got[1024] = {0};
        for (size_t i = 0; i < seq.count; i++) {
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%u", i ? " " : "", seq.ids[i]);
            strncat(got, tmp, sizeof got - strlen(got) - 1);
        }
        if (strcmp(got, CASES[c].expect) != 0) {
            fprintf(stderr, "FAIL case %zu %s\n  got    %s\n  expect %s\n",
                    c, CASES[c].text, got, CASES[c].expect);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "%d nemotron cases FAILED\n", fails);
        return 1;
    }
    printf("test_nemotron: all %zu cases byte-exact vs the reference\n",
           sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
