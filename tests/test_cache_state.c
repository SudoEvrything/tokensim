#include "tokenizer.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static int checks;

static int encode_ids(tsimd_tokenizer_t *t, const uint8_t *s, size_t n,
                      uint32_t *out, size_t cap, size_t *nout) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, s, n, &seq) != 0) return 1;
    if (seq.count > cap) {
        tsimd_token_sequence_free(&seq);
        return 2;
    }
    memcpy(out, seq.ids, seq.count * sizeof(uint32_t));
    *nout = seq.count;
    tsimd_token_sequence_free(&seq);
    return 0;
}

static uint32_t seen[4][1 << 17];
static size_t seen_n[4];

static void check_case(tsimd_tokenizer_t *t, const uint8_t *s, size_t n,
                       const char *what) {
    checks++;
    for (int pass = 0; pass < 4; pass++) {
        int rc = encode_ids(t, s, n, seen[pass], 1 << 17, &seen_n[pass]);
        if (rc != 0) {
            fprintf(stderr, "FAIL %s: pass %d rc %d\n", what, pass, rc);
            g_fail = 1;
            return;
        }
    }
    unsetenv("TSIMD_CACHE_STATE_PROBE");
    for (int pass = 1; pass < 4; pass++) {
        if (seen_n[pass] != seen_n[0] ||
            memcmp(seen[pass], seen[0],
                   seen_n[0] * sizeof(uint32_t)) != 0) {
            size_t i = 0;
            while (i < seen_n[0] && i < seen_n[pass] &&
                   seen[0][i] == seen[pass][i])
                i++;
            fprintf(stderr,
                    "FAIL %s: pass %d %zu vs %zu ids, first diff %zu "
                    "(%u vs %u)\n",
                    what, pass, seen_n[pass], seen_n[0], i,
                    i < seen_n[pass] ? seen[pass][i] : 0u,
                    i < seen_n[0] ? seen[0][i] : 0u);
            g_fail = 1;
            return;
        }
    }
}

static const char *CASES[] = {
    "",
    "a",
    "hello world",
    "don't can't won't it's",
    "rock 'n' roll o'clock",
    "'s 't 're 'll 've 'm",
    "x'a'a'a'a",
    "1234567890",
    "1 12 123 1234 12345",
    "a1b2c3d4",
    "\331\241\062\063\064\065\066\067\070\071",
    "the quick brown fox. It's 12:30 AM.\r\nJumped over the lazy dog.",
    "a\r\nb\r\n\rc",
    "\n\n\n   \n\t\t  x",
    "   ",
    "a  b",
    "a   ",
    "a.\nb",
    "a///b//c",
    "x!!abc?!?!?",
    "\344\270\255\346\226\207\343\201\202\343\202\242",
    "ab\344\270\255cd",
    "\360\237\230\200\360\237\230\200",
    "\303\251\303\251\303\251",
    "cam\303\251ra update",
    "https://example.com/a/b?c=d&e=f",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "000000000000000000000000000000000",
    "a z a z a z a z a z a z a z a z a z a z a z a z a z",
};

int main(void) {
    setenv("TSIMD_SERVE_ENV_PER_CALL", "1", 1);
    char tmpl[] = "/tmp/tsimd_direct_serve_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return 1;
    close(fd);
    static const char *EXTRAS[] = {"he", "hel", "hell", "hello", "wo",
                                   "wor", "worl", "world"};
    size_t vlen = 0;
    char *vbuf = tsimd_test_make_vocab(EXTRAS, 6, &vlen);
    if (!vbuf) return 1;
    FILE *f = fopen(tmpl, "wb");
    if (!f || fwrite(vbuf, 1, vlen, f) != vlen) return 1;
    fclose(f);
    free(vbuf);

    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load(&t, tmpl, TSIMD_PRETOK_CL100K, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: load\n");
        remove(tmpl);
        return 1;
    }

    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++)
        check_case(&t, (const uint8_t *)CASES[i], strlen(CASES[i]),
                   CASES[i][0] ? CASES[i] : "empty");

    {
        size_t n = 0;
        static uint8_t big[1 << 17];
        for (int rep = 0; rep < 400 && n < sizeof(big) - 300; rep++) {
            for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
                size_t l = strlen(CASES[i]);
                if (n + l + 1 >= sizeof(big)) break;
                memcpy(big + n, CASES[i], l);
                n += l;
                big[n++] = ' ';
            }
        }
        check_case(&t, big, n, "big mixed document");
        for (size_t i = 0; i < n; i++)
            if (big[i] >= 'a' && big[i] <= 'z') big[i] = 'q';
        check_case(&t, big, n, "big uniform document");
        memset(big, '7', n);
        check_case(&t, big, n, "big digit run");
    }

    {
        uint64_t rs = 0x9e3779b97f4a7c15ull;
        static const char pool[] = {
            'a',  'z',  'A',  'Z',  '0',  '9',  ' ',  ' ', ' ',
            '\'', '.',  '/',  '\n', '\r', '\t', '?',  '!',
            '\001', 0x80, 0x81, '\303', '\251', '\344', '\270', '\255'};
        uint8_t sb[512];
        for (int iter = 0; iter < 6000 && !g_fail; iter++) {
            rs ^= rs << 13;
            rs ^= rs >> 7;
            rs ^= rs << 17;
            size_t n = (size_t)(rs % 400);
            for (size_t i = 0; i < n; i++) {
                rs ^= rs << 13;
                rs ^= rs >> 7;
                sb[i] = (uint8_t)pool[rs % (sizeof(pool) / sizeof(pool[0]))];
            }
            check_case(&t, sb, n, "direct fuzz");
        }
    }

    tsimd_tokenizer_free(&t);
    remove(tmpl);
    if (g_fail) {
        fprintf(stderr, "test_cache_state: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_cache_state: OK (%d checks)\n", checks);
    return 0;
}
