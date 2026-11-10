#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model;
    const char *text;
    const char *want;
} dec_case_t;

static const dec_case_t CASES[] = {
    {"t5", "Hello world, this is a test.", "Hello world, this is a test."},
    {"t5", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na \xe2\x81\x87 ve \xe2\x80\x94 ok."},
    {"t5", "  leading and   trailing  ", "leading and trailing"},
    {"xlmr", "Hello world, this is a test.", "Hello world, this is a test."},
    {"xlmr", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"xlmr", "  leading and   trailing  ", "leading and trailing"},
    {"mistral", "Hello world, this is a test.", "Hello world, this is a test."},
    {"mistral", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"mistral", "  leading and   trailing  ", "  leading and   trailing  "},
    {"o200k", "Hello world, this is a test.", "Hello world, this is a test."},
    {"o200k", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"o200k", "  leading and   trailing  ", "  leading and   trailing  "},
    {"cl100k", "Hello world, this is a test.", "Hello world, this is a test."},
    {"cl100k", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"cl100k", "  leading and   trailing  ", "  leading and   trailing  "},
    {"deepseek", "Hello world, this is a test.", "Hello world, this is a test."},
    {"deepseek", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"deepseek", "  leading and   trailing  ", "  leading and   trailing  "},
    {"qwen2.5", "Hello world, this is a test.", "Hello world, this is a test."},
    {"qwen2.5", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"qwen2.5", "  leading and   trailing  ", "  leading and   trailing  "},
    {"gemma4", "Hello world, this is a test.", "Hello world, this is a test."},
    {"gemma4", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"gemma4", "  leading and   trailing  ", "  leading and   trailing  "},
    {"glm4", "Hello world, this is a test.", "Hello world, this is a test."},
    {"glm4", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"glm4", "  leading and   trailing  ", "  leading and   trailing  "},
    {"nemotron3", "Hello world, this is a test.", "Hello world, this is a test."},
    {"nemotron3", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"nemotron3", "  leading and   trailing  ", "  leading and   trailing  "},
    {"smollm2", "Hello world, this is a test.", "Hello world, this is a test."},
    {"smollm2", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok.", "The quick brown fox jumps over the lazy dog 0123456789; caf\xc3\xa9 na\xc3\xafve \xe2\x80\x94 ok."},
    {"smollm2", "  leading and   trailing  ", "  leading and   trailing  "},
};

static int fmt_of(const char *model, const char **path) {
    size_t n = strlen(model);
    if (strcmp(model, "t5") == 0) { *path = TSIMD_SRC_DIR "/models/reference/t5_spiece.model"; return TSIMD_MODEL_SENTENCEPIECE; }
    if (strcmp(model, "xlmr") == 0) { *path = TSIMD_SRC_DIR "/models/reference/xlmr_spiece.model"; return TSIMD_MODEL_SENTENCEPIECE; }
    if (strcmp(model, "mistral") == 0) { *path = TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"; return TSIMD_MODEL_SENTENCEPIECE; }
    if (strcmp(model, "o200k") == 0) { *path = TSIMD_SRC_DIR "/models/o200k_base.tiktoken"; return TSIMD_MODEL_TIKTOKEN; }
    if (strcmp(model, "cl100k") == 0) { *path = TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"; return TSIMD_MODEL_TIKTOKEN; }
    if (strcmp(model, "deepseek") == 0) { *path = TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    if (strcmp(model, "qwen2.5") == 0) { *path = TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    if (strcmp(model, "gemma4") == 0) { *path = TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    if (strcmp(model, "glm4") == 0) { *path = TSIMD_SRC_DIR "/models/reference/glm4_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    if (strcmp(model, "nemotron3") == 0) { *path = TSIMD_SRC_DIR "/models/reference/nemotron3_nano_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    if (strcmp(model, "smollm2") == 0) { *path = TSIMD_SRC_DIR "/models/reference/smollm2_tokenizer.json"; return TSIMD_MODEL_TOKENIZER_JSON; }
    (void)n;
    return -1;
}

static int check_ids(const char *path, int fmt, const char *tag,
                     const uint32_t *ids, size_t n, const char *want,
                     size_t want_len) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (strstr(path, "o200k")) scheme = TSIMD_PRETOK_O200K;
    else if (strstr(path, "cl100k")) scheme = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t, (tsimd_model_format_t)fmt, path, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "%s: load failed\n", tag);
        return 1;
    }
    tsimd_byte_sequence_t dec;
    int fails = 0;
    if (tsimd_tokenizer_decode(&t, ids, n, 0, &dec) != 0) {
        fprintf(stderr, "%s: decode failed\n", tag);
        fails++;
    } else {
        if (dec.len != want_len ||
            (want_len && memcmp(dec.bytes, want, want_len) != 0)) {
            fprintf(stderr, "%s: got %zu bytes", tag, dec.len);
            for (size_t k = 0; k < dec.len; k++)
                fprintf(stderr, " %02x", dec.bytes[k]);
            fprintf(stderr, ", want %zu bytes", want_len);
            for (size_t k = 0; k < want_len; k++)
                fprintf(stderr, " %02x", (unsigned char)want[k]);
            fprintf(stderr, "\n");
            fails++;
        }
        tsimd_byte_sequence_free(&dec);
    }
    tsimd_tokenizer_free(&t);
    return fails;
}

int main(int argc, char **argv) {
    const char *fix = argc > 1 ? argv[1] : "tests/fixtures";
    char patched[512];
    int fails = 0;
    const char *cur = NULL;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    int loaded = 0;
    for (size_t c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        if (!cur || strcmp(cur, CASES[c].model) != 0) {
            if (loaded) tsimd_tokenizer_free(&t);
            memset(&t, 0, sizeof t);
            const char *path = NULL;
            int fmt = fmt_of(CASES[c].model, &path);
            tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
            if (path && strstr(path, "o200k")) scheme = TSIMD_PRETOK_O200K;
            else if (path && strstr(path, "cl100k"))
                scheme = TSIMD_PRETOK_CL100K;
            if (fmt < 0 ||
                tsimd_tokenizer_load_model(&t, (tsimd_model_format_t)fmt, path,
                                           NULL, &scheme, NULL, 0) != 0) {
                fprintf(stderr, "%s: load failed\n", CASES[c].model);
                return 2;
            }
            loaded = 1;
            cur = CASES[c].model;
        }
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, (const uint8_t *)CASES[c].text,
                                   strlen(CASES[c].text), &seq) != 0) {
            fprintf(stderr, "%s: encode failed\n", CASES[c].model);
            fails++;
            continue;
        }
        tsimd_byte_sequence_t dec;
        if (tsimd_tokenizer_decode(&t, seq.ids, seq.count, 0, &dec) != 0) {
            fprintf(stderr, "%s: decode failed\n", CASES[c].model);
            fails++;
            tsimd_token_sequence_free(&seq);
            continue;
        }
        {
            size_t need = 0;
            tsimd_tokenizer_decode_into(&t, seq.ids, seq.count, 0, NULL, 0,
                                        &need);
            uint8_t *serial = malloc(need ? need : 1);
            uint8_t *par = malloc(need ? need : 1);
            size_t got_s = 0, got_p = 0;
            tsimd_tokenizer_decode_into(&t, seq.ids, seq.count, 0, serial,
                                        need, &got_s);
            tsimd_tokenizer_decode_parallel(&t, seq.ids, seq.count, 0, par,
                                            need, &got_p, 0);
            if (got_s != got_p || memcmp(serial, par, got_s) != 0) {
                fprintf(stderr, "FAIL: the parallel decode differs from the "
                                "serial decode\n");
                fails++;
            }
            free(serial);
            free(par);
        }
        size_t wl = strlen(CASES[c].want);
        if (dec.len != wl ||
            (wl && memcmp(dec.bytes, CASES[c].want, wl) != 0)) {
            fprintf(stderr, "%s: case %zu\n  got  %.*s\n  want %s\n",
                    CASES[c].model, c, (int)dec.len,
                    dec.bytes ? (const char *)dec.bytes : "", CASES[c].want);
            fails++;
        }
        tsimd_byte_sequence_free(&dec);
        tsimd_token_sequence_free(&seq);
    }
    if (loaded) tsimd_tokenizer_free(&t);
    {
        const char *t5 = TSIMD_SRC_DIR "/models/reference/"
                         "t5_spiece.model";
        const char *mist = TSIMD_SRC_DIR "/models/reference/"
                           "mistral7b_tokenizer.model";
        static const char SURF[] = " \xE2\x81\x87 ";
        static const uint32_t a0[] = {0};
        static const uint32_t a1[] = {1};
        static const uint32_t a12[] = {1, 2};
        static const uint32_t a2[] = {2};
        static const uint32_t a3[] = {3};
        static const uint32_t a0123[] = {0, 1, 2, 3};
        fails += check_ids(t5, TSIMD_MODEL_SENTENCEPIECE, "t5 [1]", a1, 1, "", 0);
        fails += check_ids(t5, TSIMD_MODEL_SENTENCEPIECE, "t5 [1,2]", a12, 2,
                           SURF, 5);
        fails += check_ids(t5, TSIMD_MODEL_SENTENCEPIECE, "t5 [0,1,2,3]",
                           a0123, 4, " \xE2\x81\x87  ", 6);
        fails += check_ids(t5, TSIMD_MODEL_SENTENCEPIECE, "t5 [3]", a3, 1, "", 0);
        fails += check_ids(t5, TSIMD_MODEL_SENTENCEPIECE, "t5 [2]", a2, 1, SURF,
                           5);
        fails += check_ids(mist, TSIMD_MODEL_SENTENCEPIECE, "mistral [0]", a0, 1,
                           SURF, 5);
        fails += check_ids(mist, TSIMD_MODEL_SENTENCEPIECE, "mistral [1]", a1, 1,
                           "", 0);
        fails += check_ids(mist, TSIMD_MODEL_SENTENCEPIECE, "mistral [2]", a2, 1,
                           "", 0);
        fails += check_ids(mist, TSIMD_MODEL_SENTENCEPIECE, "mistral [3]", a3, 1,
                           "\x00", 1);
        snprintf(patched, sizeof patched, "%s/t5_unk_surface.model", fix);
        fails += check_ids(patched, TSIMD_MODEL_SENTENCEPIECE,
                           "custom unk_surface", a2, 1, "<!UNK!>", 7);
    }
    {
        tsimd_byte_sequence_t dec;
        uint32_t none = 0;
        tsimd_tokenizer_t t0;
        memset(&t0, 0, sizeof t0);
        tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
        const char *t5 = TSIMD_SRC_DIR "/models/reference/"
                         "t5_spiece.model";
        if (tsimd_tokenizer_load_model(&t0, TSIMD_MODEL_SENTENCEPIECE, t5, NULL,
                                       &sch, NULL, 0) != 0) {
            fprintf(stderr, "edge cases: load failed\n");
            return 2;
        }
        if (tsimd_tokenizer_decode(&t0, &none, 0, 0, &dec) != 0 || dec.len != 0) {
            fprintf(stderr, "empty id list must decode to nothing\n");
            fails++;
        }
        tsimd_byte_sequence_free(&dec);
        uint32_t bad[2] = {0u, 4000000000u};
        if (tsimd_tokenizer_decode(&t0, bad, 2, 0, &dec) == 0) {
            fprintf(stderr, "an unknown id must fail\n");
            fails++;
            tsimd_byte_sequence_free(&dec);
        }
        tsimd_tokenizer_free(&t0);
    }
if (fails) {
        fprintf(stderr, "%d decode cases FAILED\n", fails);
        return 1;
    }
    printf("test_decode: %zu reference cases plus the id-level control, "
           "unk-surface and custom-surface checks\n",
           sizeof CASES / sizeof CASES[0]);
    return 0;
}
