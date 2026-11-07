#include "tokenizer.h"
#include "unicode.h"
#include "util.h"

#include "../src/nfc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *g_fixtures_dir = "tests/fixtures";
static char g_scratch[256];
static char fx[4][512];
static char wfx[4][512];
static int fx_slot;
static int wfx_slot;

static const char *fixture(const char *name) {
    fx_slot = (fx_slot + 1) % 4;
    snprintf(fx[fx_slot], sizeof(fx[0]), "%s/%s", g_fixtures_dir, name);
    return fx[fx_slot];
}

static const char *wfixture(const char *name) {
    if (!g_scratch[0]) {
        snprintf(g_scratch, sizeof(g_scratch), "/tmp/tsimd_nfc_XXXXXX");
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

#define CHECK_FORM(fn_, tag_, in_, want_)                                     \
    do {                                                                      \
        const char *in_p = (in_), *w_p = (want_);                             \
        uint8_t out[512];                                                      \
        size_t n = fn_((const uint8_t *)in_p, strlen(in_p), out, sizeof(out));\
        if (n == (size_t)-1) {                                                 \
            fprintf(stderr, "FAIL %s:%d: %s returned -1 for %s\n",             \
                    __FILE__, __LINE__, tag_, in_p);                           \
            return 1;                                                          \
        }                                                                      \
        if (n != strlen(w_p) || memcmp(out, w_p, n) != 0) {                    \
            fprintf(stderr, "FAIL %s:%d: %s(", __FILE__, __LINE__, tag_);      \
            for (size_t i = 0; i < n; i++) fprintf(stderr, "%02x", out[i]);    \
            fprintf(stderr, ") want(");                                        \
            for (size_t i = 0; i < strlen(w_p); i++)                           \
                fprintf(stderr, "%02x", (unsigned char)w_p[i]);                \
            fprintf(stderr, ")\n");                                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)
#define CHECK(in_, want_) CHECK_FORM(tsimd_nfc_normalize, "NFC", in_, want_)
#define CHECK_NFKC(in_, want_)                                                \
    CHECK_FORM(tsimd_nfkc_normalize, "NFKC", in_, want_)

static int test_hand_cases(void) {

    CHECK("", "");
    CHECK("hello world", "hello world");
    CHECK("func(int a) { return a*2; }", "func(int a) { return a*2; }");

    CHECK("a\xcc\x81", "\xc3\xa1");
    CHECK("A\xcc\x8a", "\xc3\x85");

    CHECK("\xe2\x84\xab", "\xc3\x85");

    CHECK("q\xcc\x87\xcc\xa3", "q\xcc\xa3\xcc\x87");

    CHECK("\xe0\xa5\x98", "\xe0\xa4\x95\xe0\xa4\xbc");

    CHECK("\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "\xea\xb0\x81");
    CHECK("\xea\xb0\x80", "\xea\xb0\x80");

    CHECK("\xe0\xa7\x87\xe0\xa6\xbe", "\xe0\xa7\x8b");

    CHECK("\xc3\xa9", "\xc3\xa9");
    CHECK("\xc3\xa9\xcc\x81", "\xc3\xa9\xcc\x81");
    CHECK("\xe0\xbd\xb7", "\xe0\xbd\xb7");
    CHECK("\xef\xbe\x9e", "\xef\xbe\x9e");

    CHECK("e\xcc\x81\xcc\xa3", "\xe1\xba\xb9\xcc\x81");
    CHECK("e\xcc\xa3\xcc\x81", "\xe1\xba\xb9\xcc\x81");

    CHECK("\xe1\xb8\x8a\xcc\xa3", "\xe1\xb8\x8c\xcc\x87");

    CHECK("\xe1\xb8\x8a\xcc\xa3\xcc\x87",
          "\xe1\xb8\x8c\xcc\x87\xcc\x87");

    CHECK("e\xcc\x82\xcc\x81", "\xe1\xba\xbf");
    CHECK("\xe1\xba\xbf", "\xe1\xba\xbf");
    CHECK("e\xcc\x81\xcc\x82", "\xc3\xa9\xcc\x82");

    CHECK("caf\xc3\xa9 q\xcc\x87\xcc\xa3 \xe0\xa5\x98 \xea\xb0\x80",
          "caf\xc3\xa9 q\xcc\xa3\xcc\x87 \xe0\xa4\x95\xe0\xa4\xbc \xea\xb0\x80");
    return 0;
}

static int test_ascii_run_boundaries(void) {

    CHECK("hello e\xcc\x81", "hello \xc3\xa9");
    CHECK("cafe\xcc\x81", "caf\xc3\xa9");
    CHECK("ay\xcc\x8a" "z", "a\xe1\xba\x99z");
    CHECK("A\xcc\x8a" "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          "e\xcc\x81",
          "\xc3\x85xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          "\xc3\xa9");

    CHECK("x\xcc\x81" "y", "x\xcc\x81y");

    CHECK("e\xcc\x82\xcc\x81!", "\xe1\xba\xbf!");

    CHECK("ab\xc3\xa9" "cd", "ab\xc3\xa9" "cd");

    CHECK("abc \xe2\x84\xab def", "abc \xc3\x85 def");
    CHECK("e\xcc\x81\xcc\xa3" " tail", "\xe1\xba\xb9\xcc\x81 tail");

    CHECK("\xcc\x81" "abc", "\xcc\x81" "abc");

    CHECK("a\xe1\x85\xa1", "a\xe1\x85\xa1");

    for (size_t i = 0; i < tsimd_nfc_comp_n; i++) {
        if (tsimd_nfc_comp[i].b < 0x80) {
            fprintf(stderr, "FAIL: composition pair %u+%u has an ASCII "
                    "second element\n", tsimd_nfc_comp[i].a,
                    tsimd_nfc_comp[i].b);
            return 1;
        }
    }
    return 0;
}

static int test_invalid_bytes(void) {

    CHECK("\x80", "\x80");
    CHECK("\xc3", "\xc3");
    CHECK("\xe2\x96", "\xe2\x96");
    CHECK("\xf0\x9f\x92", "\xf0\x9f\x92");
    CHECK("a\xff" "b", "a\xff" "b");
    CHECK("\xc0\x80", "\xc0\x80");

    CHECK("a\xcc\x81\xc3", "\xc3\xa1\xc3");
    return 0;
}

static int test_cap_contract(void) {

    uint8_t small[1];
    size_t n = tsimd_nfc_normalize((const uint8_t *)"a\xcc\x81", 3, small,
                                   sizeof(small));
    TEST_ASSERT(n == (size_t)-1);

    uint8_t exact[2];
    n = tsimd_nfc_normalize((const uint8_t *)"a\xcc\x81", 3, exact,
                            sizeof(exact));
    TEST_ASSERT_EQ(n, 2);
    TEST_ASSERT(memcmp(exact, "\xc3\xa1", 2) == 0);

    n = tsimd_nfc_normalize((const uint8_t *)"", 0, NULL, 0);
    TEST_ASSERT_EQ(n, 0);
    return 0;
}

static int test_may_change(void) {

    TEST_ASSERT_EQ(tsimd_nfc_may_change((const uint8_t *)"", 0), 0);
    TEST_ASSERT_EQ(tsimd_nfc_may_change((const uint8_t *)"plain ascii", 11),
                   0);
    TEST_ASSERT_EQ(tsimd_nfc_may_change((const uint8_t *)"a\xcc\x81", 3), 1);
    TEST_ASSERT_EQ(tsimd_nfc_may_change((const uint8_t *)"\xc3\xa9", 2), 1);
    TEST_ASSERT_EQ(tsimd_nfc_may_change((const uint8_t *)"\x80", 1), 1);
    return 0;
}

static uint8_t ccc_oracle(uint32_t cp) {
    size_t lo = 0, hi = tsimd_nfc_ccc_n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (tsimd_nfc_ccc[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < tsimd_nfc_ccc_n && tsimd_nfc_ccc[lo].cp == cp)
        return tsimd_nfc_ccc[lo].ccc;
    return 0;
}

static int test_ccc_direct(void) {

    for (uint32_t cp = 0; cp <= 0x110004; cp++) {
        uint8_t a = tsimd_nfc_ccc_lookup(cp);
        uint8_t b = ccc_oracle(cp);
        if (a != b) {
            fprintf(stderr, "ccc mismatch at 0x%X: direct %u oracle %u\n",
                    cp, a, b);
            return 1;
        }
    }

    TEST_ASSERT_EQ(tsimd_nfc_ccc_lookup(0x0300), 230);
    TEST_ASSERT_EQ(tsimd_nfc_ccc_lookup(0x1D165), 216);
    TEST_ASSERT_EQ(tsimd_nfc_ccc_lookup(0x41), 0);
    printf("  ccc direct lookup: OK\n");
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

static int test_tokenizer_json_nfc(void) {
    const char *nfc = wfixture("mini_nfc.json");
    const char *nul = wfixture("mini_null.json");
    const char *nfkc = wfixture("mini_nfkc.json");
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfc,
                              "{\"type\": \"NFC\"}") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nul,
                              "null") == 0);
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfkc,
                              "{\"type\": \"NFKC\"}") == 0);

    tsimd_token_sequence_t d, p, q;
    TEST_ASSERT(encode_seq(nfc, "a\xcc\x81", &d) == 0);
    TEST_ASSERT(encode_seq(nfc, "\xc3\xa1", &p) == 0);
    TEST_ASSERT(d.count > 0);
    TEST_ASSERT(seq_same(&d, &p));

    TEST_ASSERT(encode_seq(nul, "\xc3\xa1", &q) == 0);
    TEST_ASSERT(seq_same(&q, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);
    tsimd_token_sequence_free(&q);

    TEST_ASSERT(encode_seq(nul, "a\xcc\x81", &d) == 0);
    TEST_ASSERT(encode_seq(nul, "\xc3\xa1", &p) == 0);
    TEST_ASSERT(!seq_same(&d, &p));
    tsimd_token_sequence_free(&d);
    tsimd_token_sequence_free(&p);

    {
        tsimd_tokenizer_t t;
        tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
        TEST_ASSERT(tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                               nfkc, NULL, &scheme, NULL,
                                               0) == 0);
        TEST_ASSERT(t.norm == TSIMD_NORM_NFKC);
        tsimd_tokenizer_free(&t);
    }
    return 0;
}

static int test_normalize_all_nonorm(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                   TSIMD_SRC_DIR "/models/"
                                   "reference/nemotron3_nano_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0)
        return 1;
    static const uint8_t in[] = {0x5c, 0xef, 0xef, 0xef, 0xa4, 0xa4, 0x7e,
                                 0xc3, 0xa9, 0x61, 0xcc, 0x81, 0xef, 0xa4};
    size_t n = 0;
    uint8_t *out = tsimd_nfc_normalize_all(&t, in, sizeof in, 1, &n);
    int fail = 0;
    if (!out || n != sizeof in || memcmp(out, in, sizeof in) != 0)
        fail = 1;
    free(out);
    tsimd_tokenizer_free(&t);
    if (fail) {
        fprintf(stderr,
                "FAIL normalize_all on a no-norm model changed bytes\n");
        return 1;
    }
    return 0;
}

static int test_nfc_parallel_matches_seq(void) {
    const char *nfc = wfixture("mini_nfc_par.json");
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfc,
                              "{\"type\": \"NFC\"}") == 0);

    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    TEST_ASSERT(tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, nfc,
                                           NULL, &scheme, NULL, 0) == 0);
    TEST_ASSERT(t.norm == TSIMD_NORM_NFC);

    enum { N = 40000 };
    size_t piece = strlen("a\xcc\x81 word ");
    size_t len = (size_t)N * piece;
    uint8_t *buf = (uint8_t *)malloc(len);
    TEST_ASSERT(buf != NULL);
    for (int i = 0; i < N; i++)
        memcpy(buf + (size_t)i * piece, "a\xcc\x81 word ", piece);

    tsimd_token_sequence_t seq, par;
    TEST_ASSERT(tsimd_tokenizer_encode(&t, buf, len, &seq) == 0);
    TEST_ASSERT(tsimd_tokenizer_encode_parallel(&t, buf, len, 4, &par) == 0);
    TEST_ASSERT(seq.count > 0);
    TEST_ASSERT(seq_same(&seq, &par));

    const char *pre_piece = "\xc3\xa1 word ";
    size_t pre_len_piece = strlen(pre_piece);
    size_t len2 = (size_t)N * pre_len_piece;
    uint8_t *buf2 = (uint8_t *)malloc(len2);
    TEST_ASSERT(buf2 != NULL);
    for (int i = 0; i < N; i++)
        memcpy(buf2 + (size_t)i * pre_len_piece, pre_piece, pre_len_piece);
    tsimd_token_sequence_t pre;
    TEST_ASSERT(tsimd_tokenizer_encode_parallel(&t, buf2, len2, 4, &pre) == 0);
    TEST_ASSERT(seq_same(&seq, &pre));

    free(buf);
    free(buf2);
    tsimd_token_sequence_free(&seq);
    tsimd_token_sequence_free(&par);
    tsimd_token_sequence_free(&pre);
    tsimd_tokenizer_free(&t);
    return 0;
}

static int test_nfkc_hand_cases(void) {

    CHECK_NFKC("hello", "hello");
    CHECK_NFKC("a\xc2\xa0" "b", "a b");
    CHECK_NFKC("\xef\xac\x81", "fi");
    CHECK_NFKC("\xef\xbc\xa1", "A");
    CHECK_NFKC("\xe2\x91\xa0", "1");
    CHECK_NFKC("a\xcc\x81", "\xc3\xa1");
    CHECK_NFKC("\xc3\xa1", "\xc3\xa1");

    {
        uint8_t in[] = {0x61, 0xFF, 0x62};
        uint8_t out[8];
        size_t n = tsimd_nfkc_normalize(in, 3, out, sizeof(out));
        TEST_ASSERT(n == 3);
        TEST_ASSERT(out[0] == 0x61 && out[1] == 0xFF && out[2] == 0x62);
    }
    return 0;
}

static int test_tokenizer_json_nfkc_encode(void) {
    const char *nfkc = wfixture("mini_nfkc_enc.json");
    TEST_ASSERT(write_variant(fixture("mini_tokenizer.json"), nfkc,
                              "{\"type\": \"NFKC\"}") == 0);
    tsimd_token_sequence_t a, b;
    TEST_ASSERT(encode_seq(nfkc, "\xef\xac\x81", &a) == 0);
    TEST_ASSERT(encode_seq(nfkc, "fi", &b) == 0);
    TEST_ASSERT(a.count > 0);
    TEST_ASSERT(seq_same(&a, &b));
    tsimd_token_sequence_free(&a);
    tsimd_token_sequence_free(&b);
    return 0;
}

static int test_stable_property(void) {
    static const struct { const char *s; } cases[] = {
        {"hello world"},
        {"caf\xc3\xa9" " na\xc3\xafve"},
        {"em\xe2\x80\x93" "dash \xe2\x80\x94" "quote"},
        {"\xf0\x9f\x98\x80" "emoji"},
        {"\xd0\xb1\xd0\xb2\xd0\xb3" " cyrillic"},
        {"\xe4\xb8\xad\xe6\x96\x87" " cjk"},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const uint8_t *p = (const uint8_t *)cases[i].s;
        size_t n = strlen(cases[i].s);
        if (tsimd_nfc_may_change(p, n)) continue;
        uint8_t buf[256];
        size_t nn = tsimd_nfc_normalize(p, (uint32_t)n, buf, sizeof buf);
        if (nn != n || memcmp(buf, p, n) != 0) {
            fprintf(stderr, "FAIL ascii identity case %zu\n", i);
            return 1;
        }
    }
    printf("  ascii identity: %zu cases\n",
           sizeof cases / sizeof cases[0]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) g_fixtures_dir = argv[1];
    if (test_hand_cases() != 0) return 1;
    if (test_ascii_run_boundaries() != 0) return 1;
    if (test_invalid_bytes() != 0) return 1;
    if (test_cap_contract() != 0) return 1;
    if (test_may_change() != 0) return 1;
    if (test_ccc_direct() != 0) return 1;
    if (test_nfkc_hand_cases() != 0) return 1;
    if (test_tokenizer_json_nfc() != 0) return 1;
    if (test_stable_property() != 0) return 1;
    if (test_tokenizer_json_nfkc_encode() != 0) return 1;
    if (test_nfc_parallel_matches_seq() != 0) return 1;
    if (test_normalize_all_nonorm() != 0) return 1;
    printf("test_nfc: all groups passed\n");
    tsimd_nfc_tls_release();
    return 0;
}
