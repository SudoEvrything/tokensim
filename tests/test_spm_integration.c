#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"

size_t tsimd_metaspace(const uint8_t *in, size_t len, uint8_t *out,
                       size_t cap);

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
    {"                                        ", "359 359 2600"},
    {"hello\nworld\n\n", "6312 28709 13 9471 13 13"},
    {"  indented\n  more", "259 1176 12713 13 28705 680"},
    {"caf\u00e9 \u00e9t\u00e9 2024",
     "28345 7166 28705 28750 28734 28750 28781"},
};

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;

    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, MODEL,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL);
        return 2;
    }
    TEST_ASSERT_EQ((int)scheme, (int)TSIMD_PRETOK_SPM);
    TEST_ASSERT_EQ((int)t.scheme, (int)TSIMD_PRETOK_SPM);
    TEST_ASSERT_EQ((long long)t.vocab.vocab_size, 32000);

    int fails = 0;
    for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, (const uint8_t *)CASES[c].text,
                                   strlen(CASES[c].text), &seq) != 0) {
            fprintf(stderr, "encode failed: %s\n", CASES[c].text);
            fails++;
            continue;
        }
        char got[512] = {0};
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

        tsimd_byte_sequence_t dec;
        if (tsimd_tokenizer_decode(&t, seq.ids, seq.count, 1, &dec) == 0) {
            size_t clen = strlen(CASES[c].text);
            if (clen != dec.len ||
                (clen && memcmp(CASES[c].text, dec.bytes, clen) != 0)) {
                fprintf(stderr, "FAIL decode round-trip case %zu\n", c);
                fails++;
            }
            tsimd_byte_sequence_free(&dec);
        }
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "%d spm integration checks FAILED\n", fails);
        return 1;
    }
    printf("test_spm_integration: all %zu cases byte-exact + decode round-trip\n",
           sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
