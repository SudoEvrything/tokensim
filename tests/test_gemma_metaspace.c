#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json"

typedef struct {
    const char *text;
    const char *expect;
    size_t len;
} gem_case_t;

#define GEM(t, e) {t, e, sizeof(t) - 1}

static const gem_case_t CASES[] = {
    {"hello world", "23391 1902", sizeof("hello world") - 1},
    {"Port-au-Prince", "8999 236772 826 236772 71049",
     sizeof("Port-au-Prince") - 1},
    {" a  b ", "496 138 236763 236743", sizeof(" a  b ") - 1},
    {"hello", "23391", sizeof("hello") - 1},
    {"caf\u00e9 \u00e9t\u00e9 2024",
     "123125 236859 10267 236743 236778 236771 236778 236812",
     sizeof("caf\u00e9 \u00e9t\u00e9 2024") - 1},
    {"  indented\n  more", "138 724 16764 107 138 5576",
     sizeof("  indented\n  more") - 1},
    {"name                  =", "1201 154 236784",
     sizeof("name                  =") - 1},
    {" ", "236743", sizeof(" ") - 1},
    {"   ", "139", sizeof("   ") - 1},
    {"        ", "144", sizeof("        ") - 1},
    {"                ", "152", sizeof("                ") - 1},
    {"                               ", "167",
     sizeof("                               ") - 1},
    {"                                 ", "167 138",
     sizeof("                                 ") - 1},
    {"                                        ", "167 145",
     sizeof("                                        ") - 1},
    {"                x", "152 236781", sizeof("                x") - 1},
    {"x                ", "236781 152", sizeof("x                ") - 1},
    {"hello\nworld\n\n", "23391 107 12392 108",
     sizeof("hello\nworld\n\n") - 1},
    {"\tindent\tme", "255968 32699 255968 1336",
     sizeof("\tindent\tme") - 1},
    {"x" "\x7f" "y", "236781 251541 236762",
     sizeof("x" "\x7f" "y") - 1},
    {"\x01" "\x02", "249732 241247", sizeof("\x01" "\x02") - 1},
    {"a" "\x00" "b", "236746 238 236763",
     sizeof("a" "\x00" "b") - 1},
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

    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_GEM4);

    int fails = 0;
    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, (const uint8_t *)CASES[c].text,
                                   CASES[c].len, &seq) != 0) {
            fprintf(stderr, "encode failed: %s\n", CASES[c].text);
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
            fprintf(stderr, "FAIL case %zu %s\n  got    %s\n  expect %s\n", c,
                    CASES[c].text, got, CASES[c].expect);
            fails++;
        }
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "%d gemma metaspace cases FAILED\n", fails);
        return 1;
    }
    printf("test_gemma_metaspace: all %zu cases byte-exact vs the reference\n",
           sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
