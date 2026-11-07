#include "tokenizer.h"
#include "unicode.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *g_fixtures_dir = "tests/fixtures";
static char g_scratch[256];
static char fx[2][512];
static int fx_slot;
static char wfx[4][512];
static int wfx_slot;

static const char *fixture(const char *name) {
    fx_slot = (fx_slot + 1) % 2;
    snprintf(fx[fx_slot], sizeof(fx[0]), "%s/%s", g_fixtures_dir, name);
    return fx[fx_slot];
}

static const char *wfixture(const char *name) {
    if (!g_scratch[0]) {
        snprintf(g_scratch, sizeof(g_scratch), "/tmp/tsimd_nfkc_XXXXXX");
        if (!mkdtemp(g_scratch)) {
            perror("mkdtemp");
            g_scratch[0] = '.';
            g_scratch[1] = '\0';
        }
    }
    wfx_slot = (wfx_slot + 1) % 4;
    snprintf(wfx[wfx_slot], sizeof(wfx[0]), "%s/%s", g_scratch, name);
    return wfx[wfx_slot];
}

#define CHECK(in_, want_)                                                     \
    do {                                                                      \
        const char *in_p = (in_), *w_p = (want_);                             \
        uint8_t out[128];                                                     \
        size_t n = tsimd_nfkc_normalize((const uint8_t *)in_p, strlen(in_p),  \
                                        out, sizeof(out));                    \
        if (n == (size_t)-1) {                                                \
            fprintf(stderr, "FAIL %s:%d: NFKC returned -1 for %s\n",          \
                    __FILE__, __LINE__, in_p);                                \
            return 1;                                                         \
        }                                                                     \
        if (n != strlen(w_p) || memcmp(out, w_p, n) != 0) {                   \
            fprintf(stderr, "FAIL %s:%d: NFKC(", __FILE__, __LINE__);         \
            for (size_t i = 0; i < n; i++) fprintf(stderr, "%02x", out[i]);   \
            fprintf(stderr, ") want(");                                       \
            for (size_t i = 0; i < strlen(w_p); i++)                          \
                fprintf(stderr, "%02x", (unsigned char)w_p[i]);               \
            fprintf(stderr, ")\n");                                           \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static int test_hand_cases(void) {

    CHECK("", "");
    CHECK("hello world", "hello world");
    CHECK("func(int a) { return a*2; }", "func(int a) { return a*2; }");

    CHECK("\xc2\xbc", "1\xe2\x81\x84" "4");
    CHECK("\xe2\x85\x93", "1\xe2\x81\x84" "3");
    CHECK("\xe2\x84\xa2", "TM");
    CHECK("\xc2\xb2", "2");
    CHECK("\xe2\x91\xa0", "1");
    CHECK("\xef\xbc\xa1", "A");
    CHECK("\xc2\xa0", " ");
    CHECK("\xef\xac\x81", "fi");
    CHECK("\xe3\x8e\xa1", "m2");
    CHECK("\xe2\x84\x80", "a/c");

    CHECK("\xe2\x84\xab", "\xc3\x85");

    CHECK("\xef\xb7\xba",
          "\xd8\xb5\xd9\x84\xd9\x89 \xd8\xa7\xd9\x84\xd9\x84\xd9\x87 "
          "\xd8\xb9\xd9\x84\xd9\x8a\xd9\x87 \xd9\x88\xd8\xb3\xd9\x84\xd9\x85");

    CHECK("\xe1\xba\x9b", "\xe1\xb9\xa1");

    CHECK("A\xcc\x8a\xcc\x81", "\xc7\xba");

    CHECK("\xef\xbc\xa1\xcc\x8a\xcc\x81", "\xc7\xba");

    CHECK("\xef\xbd\xb6\xef\xbe\x9e", "\xe3\x82\xac");

    CHECK("\xe3\x84\xb1\xe1\x85\xa1\xe1\x86\xa8", "\xea\xb0\x81");

    CHECK("\xc7\x84", "D\xc5\xbd");

    CHECK("\xc3\x85", "\xc3\x85");
    CHECK("\xc7\xba", "\xc7\xba");
    CHECK("\xc3\xa9", "\xc3\xa9");
    CHECK("\xc3\xa9\xcc\x81", "\xc3\xa9\xcc\x81");
    CHECK("\xea\xb0\x81", "\xea\xb0\x81");

    CHECK("\xe0\xbd\xb7", "\xe0\xbe\xb2\xe0\xbd\xb1\xe0\xbe\x80");
    CHECK("\xe0\xbe\xb2\xe0\xbe\x81", "\xe0\xbe\xb2\xe0\xbd\xb1\xe0\xbe\x80");

    CHECK("caf\xc3\xa9 \xc2\xbc \xef\xac\x81 \xe3\x84\xb1\xe1\x85\xa1\xe1\x86\xa8",
          "caf\xc3\xa9 1\xe2\x81\x84" "4 fi \xea\xb0\x81");
    return 0;
}

static int test_invalid_bytes(void) {

    CHECK("\x80", "\x80");
    CHECK("\xc3", "\xc3");
    CHECK("\xe3\xbd", "\xe3\xbd");
    CHECK("\xf0\x9f\x92", "\xf0\x9f\x92");
    CHECK("a\xff" "b", "a\xff" "b");
    CHECK("\xc0\x80", "\xc0\x80");

    CHECK("a\xcc\x81\xc3", "\xc3\xa1\xc3");
    CHECK("\xc2\xbc\xc3", "1\xe2\x81\x84" "4\xc3");
    return 0;
}

static int test_cap_contract(void) {

    uint8_t small[4];
    size_t n = tsimd_nfkc_normalize((const uint8_t *)"\xc2\xbc", 2, small,
                                    sizeof(small));
    TEST_ASSERT(n == (size_t)-1);

    uint8_t exact[5];
    n = tsimd_nfkc_normalize((const uint8_t *)"\xc2\xbc", 2, exact,
                             sizeof(exact));
    TEST_ASSERT_EQ(n, 5);
    TEST_ASSERT(memcmp(exact, "1\xe2\x81\x84" "4", 5) == 0);

    uint8_t tiny[31];
    n = tsimd_nfkc_normalize((const uint8_t *)"\xef\xb7\xba", 3, tiny,
                             sizeof(tiny));
    TEST_ASSERT(n == (size_t)-1);

    n = tsimd_nfkc_normalize((const uint8_t *)"", 0, NULL, 0);
    TEST_ASSERT_EQ(n, 0);
    return 0;
}

static int test_may_change(void) {

    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"", 0), 0);
    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"plain ascii", 11),
                   0);
    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"a\xcc\x81", 3), 1);
    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"\xc3\xa9", 2), 1);
    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"\xc2\xbc", 2), 1);
    TEST_ASSERT_EQ(tsimd_nfkc_may_change((const uint8_t *)"\x80", 1), 1);
    return 0;
}

static int test_idempotence(void) {

    CHECK("1\xe2\x81\x84" "4", "1\xe2\x81\x84" "4");
    CHECK("\xd8\xb5\xd9\x84\xd9\x89 \xd8\xa7\xd9\x84\xd9\x84\xd9\x87 "
          "\xd8\xb9\xd9\x84\xd9\x8a\xd9\x87 \xd9\x88\xd8\xb3\xd9\x84\xd9\x85",
          "\xd8\xb5\xd9\x84\xd9\x89 \xd8\xa7\xd9\x84\xd9\x84\xd9\x87 "
          "\xd8\xb9\xd9\x84\xd9\x8a\xd9\x87 \xd9\x88\xd8\xb3\xd9\x84\xd9\x85");
    CHECK("\xe3\x82\xac", "\xe3\x82\xac");
    CHECK("\xe1\xb9\xa1", "\xe1\xb9\xa1");
    CHECK("\xe0\xbe\xb2\xe0\xbd\xb1\xe0\xbe\x80",
          "\xe0\xbe\xb2\xe0\xbd\xb1\xe0\xbe\x80");
    return 0;
}

static int write_variant(const char *src, const char *dst,
                         const char *norm_json) {
    size_t len = 0;
    char *doc = tsimd_test_read_file(src, &len);
    if (!doc) return -1;

    static const char *needles[] = {
        "\"normalizer\": null",
        "\"normalizer\": {\"type\": \"NFKC\"}",
        "\"normalizer\": {\"type\": \"NFC\"}",
    };
    char *at = NULL;
    size_t needle_len = 0;
    for (size_t i = 0; i < sizeof(needles) / sizeof(needles[0]) && !at; i++) {
        at = strstr(doc, needles[i]);
        if (at) needle_len = strlen(needles[i]);
    }
    if (!at) {
        free(doc);
        return -1;
    }
    FILE *f = fopen(dst, "wb");
    if (!f) {
        free(doc);
        return -1;
    }
    fwrite(doc, 1, (size_t)(at - doc), f);
    char repl[128];
    snprintf(repl, sizeof(repl), "\"normalizer\": %s", norm_json);
    fwrite(repl, 1, strlen(repl), f);
    size_t tail = len - (size_t)(at - doc) - needle_len;
    fwrite(at + needle_len, 1, tail, f);
    fclose(f);
    free(doc);
    return 0;
}

static int seq_same(const tsimd_token_sequence_t *a,
                    const tsimd_token_sequence_t *b) {
    if (a->count != b->count) return 0;
    if (a->count == 0) return 1;
    return memcmp(a->ids, b->ids, a->count * sizeof(uint32_t)) == 0;
}

static int encode_seq(const char *path, const char *text,
                      tsimd_token_sequence_t *out) {
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, path, NULL,
                                   &scheme, NULL, 0) != 0)
        return -1;
    int rc = tsimd_tokenizer_encode(&t, (const uint8_t *)text, strlen(text),
                                    out);
    tsimd_tokenizer_free(&t);
    return rc;
}

static int test_tokenizer_json_nfkc(void) {
    const char *nfkc = wfixture("mini_nfkc.json");
    const char *nfc = wfixture("mini_nfc.json");
    const char *nul = wfixture("mini_null.json");
    const char *nfd = wfixture("mini_nfd.json");
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfkc,
                              "{\"type\": \"NFKC\"}") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfc,
                              "{\"type\": \"NFC\"}") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nul,
                              "null") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfd,
                              "{\"type\": \"NFD\"}") == 0);
    tsimd_token_sequence_t d, p, q;
    memset(&d, 0, sizeof d);
    memset(&p, 0, sizeof p);
    memset(&q, 0, sizeof q);
    TEST_ASSERT(encode_seq(nfkc, "caf\xc3\xa9 hello", &q) == 0);
    TEST_ASSERT(encode_seq(nul, "caf\xc3\xa9 hello", &p) == 0);
    TEST_ASSERT(seq_same(&q, &p));
    tsimd_token_sequence_free(&q);
    tsimd_token_sequence_free(&p);
    TEST_ASSERT(encode_seq(nul, "\xef\xac\x81", &d) == 0);
    TEST_ASSERT(encode_seq(nul, "fi", &p) == 0);
    TEST_ASSERT(!seq_same(&d, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);
    TEST_ASSERT(encode_seq(nfc, "a\xcc\x81", &d) == 0);
    TEST_ASSERT(encode_seq(nfc, "\xc3\xa1", &p) == 0);
    TEST_ASSERT(seq_same(&d, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);
    {
        tsimd_tokenizer_t t;
        tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
        TEST_ASSERT(tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                               nfd, NULL, &scheme, NULL,
                                               0) != 0);
    }
    if (getenv("NFKC_SKIP_PARALLEL")) return 0;
    TEST_ASSERT(encode_seq(nfkc, "\xef\xac\x81", &d) == 0);
    TEST_ASSERT(encode_seq(nfkc, "fi", &p) == 0);
    TEST_ASSERT(d.count > 0);
    TEST_ASSERT(seq_same(&d, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);
    TEST_ASSERT(encode_seq(nfkc, "\xc2\xbc", &d) == 0);
    TEST_ASSERT(encode_seq(nfkc, "1\xe2\x81\x84" "4", &p) == 0);
    TEST_ASSERT(seq_same(&d, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);
    return 0;
}

static int test_parallel_nfkc(void) {
    const char *nfkc = wfixture("mini_nfkc2.json");
    const char *nfc = wfixture("mini_nfc2.json");
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfkc,
                              "{\"type\": \"NFKC\"}") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfc,
                              "{\"type\": \"NFC\"}") == 0);
    size_t unit_len = strlen("lorem ipsum dolor ");
    size_t reps = ((size_t)1 << 18) / unit_len + 1;
    size_t tail_len = strlen("\xc2\xbc \xef\xac\x81hello<|eot|>");
    size_t len = reps * unit_len + tail_len;
    char *in = (char *)malloc(len);
    TEST_ASSERT(in != NULL);
    for (size_t i = 0; i < reps; i++)
        memcpy(in + i * unit_len, "lorem ipsum dolor ", unit_len);
    memcpy(in + reps * unit_len, "\xc2\xbc \xef\xac\x81hello<|eot|>",
           tail_len);
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    TEST_ASSERT(tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                           nfkc, NULL, &scheme, NULL, 0) == 0);
    tsimd_token_sequence_t seq, par;
    memset(&seq, 0, sizeof seq);
    memset(&par, 0, sizeof par);
    TEST_ASSERT(tsimd_tokenizer_encode(&t, (const uint8_t *)in, len, &seq) ==
                0);
    TEST_ASSERT(seq.count > 0);
    TEST_ASSERT(tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                           nfc, NULL, &scheme, NULL, 0) == 0);
    tsimd_token_sequence_free(&seq);
    memset(&seq, 0, sizeof seq);
    TEST_ASSERT(tsimd_tokenizer_encode(&t, (const uint8_t *)in, len, &seq) ==
                0);
    TEST_ASSERT(tsimd_tokenizer_encode_parallel(&t, (const uint8_t *)in, len,
                                                4, &par) == 0);
    TEST_ASSERT(seq_same(&seq, &par));
    tsimd_token_sequence_free(&par);
    tsimd_input_span_t docs[2];
    docs[0].data = (const uint8_t *)in;
    docs[0].length = len / 2;
    docs[1].data = (const uint8_t *)in + len / 2;
    docs[1].length = len - len / 2;
    tsimd_token_batch_t batch;
    TEST_ASSERT(tsimd_tokenizer_encode_batch(&t, docs, 2, &batch, 4) == 0);
    TEST_ASSERT(batch.id_count == seq.count);
    TEST_ASSERT(memcmp(batch.ids, seq.ids,
                       seq.count * sizeof(uint32_t)) == 0);
    TEST_ASSERT(batch.offsets[0] == 0);
    TEST_ASSERT(batch.offsets[2] == batch.id_count);
    tsimd_token_batch_free(&batch);
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    free(in);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) g_fixtures_dir = argv[1];
    if (test_tokenizer_json_nfkc() != 0) return 1;
    printf("bisect: json group without parallel/batch\n");
    return 0;
}
