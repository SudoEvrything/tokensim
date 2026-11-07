#include "tokenizer.h"
#include "input.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *EXTRAS[] = {"ab", "abc", "abcd", " x"};

static const char *SPEC_TEXT[] = {"<|eot|>", "<|",    "<", "a",
                                  " x",       "<|eot|>extra", "\x00Z"};
#define NSPEC 7
#define SPEC_ID_BASE 900u

static int g_fail;
static int checks;

static int write_vocab_file(const char *path) {
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 4, &len);
    if (!buf) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) {
        free(buf);
        return -1;
    }
    size_t wrote = fwrite(buf, 1, len, f);
    fclose(f);
    free(buf);
    return wrote == len ? 0 : -1;
}

static int oracle(tsimd_tokenizer_t *t, tsimd_tokenizer_t *plain,
                  const uint8_t *s, size_t n, uint32_t *out, size_t cap,
                  size_t *nout) {
    size_t i = 0, run = 0;
    *nout = 0;
    while (i < n) {
        int best = -1;
        size_t best_len = 0;
        for (size_t k = 0; k < t->added_count; k++) {
            size_t l = t->added[k].length;
            if (l == 0 || l > n - i || l <= best_len) continue;
            if (memcmp(s + i, t->added[k].text, l) == 0) {
                best_len = l;
                best = (int)k;
            }
        }
        if (best >= 0) {
            if (i > run) {
                tsimd_token_sequence_t seq;
                if (tsimd_tokenizer_encode(plain, s + run, i - run, &seq) != 0)
                    return 1;
                if (*nout + seq.count > cap) {
                    tsimd_token_sequence_free(&seq);
                    return 1;
                }
                for (size_t j = 0; j < seq.count; j++)
                    out[(*nout)++] = seq.ids[j];
                tsimd_token_sequence_free(&seq);
            }
            if (*nout >= cap) return 1;
            out[(*nout)++] = t->added[best].id;
            i += best_len;
            run = i;
        } else {
            i++;
        }
    }
    if (n > run) {
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(plain, s + run, n - run, &seq) != 0)
            return 1;
        if (*nout + seq.count > cap) {
            tsimd_token_sequence_free(&seq);
            return 1;
        }
        for (size_t j = 0; j < seq.count; j++) out[(*nout)++] = seq.ids[j];
        tsimd_token_sequence_free(&seq);
    }
    return 0;
}

static int seam_oracle(tsimd_tokenizer_t *t, const uint8_t *s, size_t n,
                       tsimd_added_seam_t *want, size_t cap, size_t *nwant) {
    size_t pos = 0;
    *nwant = 0;
    while (pos < n) {
        size_t best_len = 0;
        size_t best_i = 0;
        for (size_t k = 0; k < t->added_count; k++) {
            size_t l = t->added[k].length;
            if (l == 0 || l > n - pos || l <= best_len) continue;
            if (memcmp(s + pos, t->added[k].text, l) == 0) {
                best_len = l;
                best_i = k;
            }
        }
        if (best_len) {
            if (*nwant >= cap) return 1;
            want[*nwant].off = pos;
            want[*nwant].len = (uint32_t)best_len;
            want[*nwant].idx = (uint32_t)best_i;
            (*nwant)++;
            pos += best_len;
        } else {
            pos++;
        }
    }
    return 0;
}

static void check_seams(tsimd_tokenizer_t *t, const uint8_t *s, size_t n,
                        const char *what) {
    tsimd_added_seam_t want[512];
    size_t nwant = 0;
    checks++;
    if (seam_oracle(t, s, n, want, 512, &nwant) != 0) {
        fprintf(stderr, "FAIL %s: seam oracle overflow\n", what);
        g_fail = 1;
        return;
    }
    tsimd_added_seam_list_t sl;
    if (tsimd_tokenizer_added_seams(t, s, n, &sl) != 0) {
        fprintf(stderr, "FAIL %s: seams rc\n", what);
        g_fail = 1;
        return;
    }
    size_t ngot = sl.count;
    int bad = ngot != nwant;
    for (size_t i = 0; !bad && i < nwant; i++)
        if (sl.items[i].off != want[i].off || sl.items[i].len != want[i].len ||
            sl.items[i].idx != want[i].idx)
            bad = 1;
    tsimd_added_seam_list_free(&sl);
    if (bad) {
        fprintf(stderr, "FAIL %s: %zu seams, want %zu\n", what, ngot, nwant);
        g_fail = 1;
    }
}

static void check_case(tsimd_tokenizer_t *t, tsimd_tokenizer_t *plain,
                       const uint8_t *s, size_t n, const char *what) {
    check_seams(t, s, n, what);
    uint32_t want[8192], got[8192];
    size_t nwant = 0;
    checks++;
    if (oracle(t, plain, s, n, want, 8192, &nwant) != 0) {
        fprintf(stderr, "FAIL %s: oracle overflow\n", what);
        g_fail = 1;
        return;
    }
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, s, n, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode rc\n", what);
        g_fail = 1;
        return;
    }
    size_t ngot = seq.count < 8192 ? seq.count : 8192;
    memcpy(got, seq.ids, ngot * sizeof(uint32_t));
    tsimd_token_sequence_free(&seq);
    if (ngot != nwant || memcmp(got, want, nwant * sizeof(uint32_t)) != 0) {
        size_t first = 0;
        while (first < ngot && first < nwant && got[first] == want[first])
            first++;
        fprintf(stderr,
                "FAIL %s: got %zu ids, want %zu, first diff at %zu "
                "(got %u want %u)\n",
                what, ngot, nwant, first,
                first < ngot ? got[first] : 0u, first < nwant ? want[first] : 0u);
        g_fail = 1;
    }
}

static void run_cases(tsimd_tokenizer_t *t, tsimd_tokenizer_t *plain) {
    uint8_t buf[8192];
    for (int k = 0; k <= 130; k++) {
        size_t n = 0;
        for (int j = 0; j < k; j++) buf[n++] = 'z';
        memcpy(buf + n, "<|eot|>", 7);
        n += 7;
        for (int j = 0; j < 40; j++) buf[n++] = 'y';
        char what[64];
        snprintf(what, sizeof what, "align %d", k);
        check_case(t, plain, buf, n, what);
    }

    {
        static const char *CASES[] = {
            "<|eot|>",
            "<|eot|><|eot|>",
            "zzz<|eot|>",
            "<|eot|>zzz",
            "z<|eot|>",
            "<|eot|>z",
            "zz<|eo",
            "zz<|e",
            "zz<|",
            "zz<",
            "zzz<",
            "zz<|eot|>extra",
            "z<|eot|>extra",
            "<|eot|>extra",
            "<a<a<a<a<a<a<a<a<a<a<a<a<a<a<a<a",
            "aaaa",
            "a",
            "z xz xz xz x",
            " x",
            "z x",
            "1 2 3",
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            "z<|eot|>z<|eot|>z",
            "<<<<<<<<",
            "||||||||",
        };
        for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
            check_case(t, plain, (const uint8_t *)CASES[i],
                       strlen(CASES[i]), CASES[i]);
        }
    }

    {
        uint8_t nb[64];
        size_t n = 0;
        nb[n++] = 'z';
        nb[n++] = 0x00;
        nb[n++] = 'Z';
        nb[n++] = 'z';
        check_case(t, plain, nb, n, "nul special");
        nb[0] = 0x00;
        nb[1] = 'Z';
        check_case(t, plain, nb, 2, "nul only");
        nb[0] = 0x00;
        check_case(t, plain, nb, 1, "lone nul");
    }

    {
        size_t n = 0;
        for (int j = 0; j < 4096; j++) buf[n++] = (uint8_t)('a' + (j % 25));
        memcpy(buf + 2000, "<|eot|>", 7);
        check_case(t, plain, buf, n, "long with special");
        for (size_t j = 0; j < n; j++) buf[j] = (uint8_t)('A' + (j % 25));
        check_case(t, plain, buf, n, "long no special");
        tsimd_token_sequence_t a, b;
        if (tsimd_tokenizer_encode(t, buf, n, &a) != 0 ||
            tsimd_tokenizer_encode(plain, buf, n, &b) != 0) {
            fprintf(stderr, "FAIL: long no-special encode rc\n");
            g_fail = 1;
        } else {
            checks++;
            if (a.count != b.count ||
                memcmp(a.ids, b.ids, a.count * sizeof(uint32_t)) != 0) {
                fprintf(stderr, "FAIL: no-special paths differ (%zu vs %zu)\n",
                        a.count, b.count);
                g_fail = 1;
            }
            tsimd_token_sequence_free(&a);
            tsimd_token_sequence_free(&b);
        }
    }

    {
        uint64_t rs = 0x243f6a8885a308d3ull;
        static const char alpha[] = {'<', '|', 'e', 'o', 't',
                                     'x', ' ', 'a', 'Z', 0x01, 0x00};
        uint8_t sb[96];
        for (int iter = 0; iter < 20000 && !g_fail; iter++) {
            rs ^= rs << 13;
            rs ^= rs >> 7;
            rs ^= rs << 17;
            size_t n = (size_t)(rs % 61);
            for (size_t i = 0; i < n; i++) {
                rs ^= rs << 13;
                rs ^= rs >> 7;
                sb[i] = (uint8_t)alpha[rs % (sizeof(alpha) / sizeof(alpha[0]))];
            }
            check_seams(t, sb, n, "seam fuzz");
        }
    }
}

static const char *SINGLE_TEXT[] = {"<|eot|>"};
static const char *MANY_TEXT[] = {"A1", "B2", "C3", "D4", "E5",
                                  "F6", "G7", "H8", "I9", "J0", "<|eot|>"};

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    char vocab_path_tmpl[] = "/tmp/tsimd_added_scan_XXXXXX";
    int vocab_fd = mkstemp(vocab_path_tmpl);
    if (vocab_fd < 0) return 1;
    close(vocab_fd);
    const char *vocab_path = vocab_path_tmpl;
    if (write_vocab_file(vocab_path) != 0) {
        fprintf(stderr, "FAIL: vocab write\n");
        return 1;
    }

    tsimd_added_token_t specs[NSPEC];
    for (int k = 0; k < NSPEC; k++) {
        specs[k].text = SPEC_TEXT[k];
        specs[k].id = SPEC_ID_BASE + (uint32_t)k;
        specs[k].length = (uint32_t)strlen(SPEC_TEXT[k]);
        specs[k].is_special = 1;
    }

    tsimd_tokenizer_t t, plain;
    if (tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2, specs,
                             NSPEC) != 0) {
        fprintf(stderr, "FAIL: tokenizer load\n");
        remove(vocab_path);
        return 1;
    }
    if (tsimd_tokenizer_load(&plain, vocab_path, TSIMD_PRETOK_GPT2, NULL,
                             0) != 0) {
        fprintf(stderr, "FAIL: plain load\n");
        tsimd_tokenizer_free(&t);
        remove(vocab_path);
        return 1;
    }

    run_cases(&t, &plain);

    tsimd_tokenizer_t t1;
    tsimd_added_token_t sp1[1];
    sp1[0].text = SINGLE_TEXT[0];
    sp1[0].id = SPEC_ID_BASE + 40;
    sp1[0].length = (uint32_t)strlen(SINGLE_TEXT[0]);
    sp1[0].is_special = 1;
    if (tsimd_tokenizer_load(&t1, vocab_path, TSIMD_PRETOK_GPT2, sp1, 1) !=
        0) {
        fprintf(stderr, "FAIL: single-spec load\n");
        g_fail = 1;
    } else {
        run_cases(&t1, &plain);
        tsimd_tokenizer_free(&t1);
    }

    tsimd_tokenizer_t t9;
    tsimd_added_token_t sp9[sizeof(MANY_TEXT) / sizeof(MANY_TEXT[0])];
    for (size_t k = 0; k < sizeof(MANY_TEXT) / sizeof(MANY_TEXT[0]); k++) {
        sp9[k].text = MANY_TEXT[k];
        sp9[k].id = SPEC_ID_BASE + 50 + (uint32_t)k;
        sp9[k].length = (uint32_t)strlen(MANY_TEXT[k]);
        sp9[k].is_special = 1;
    }
    if (tsimd_tokenizer_load(&t9, vocab_path, TSIMD_PRETOK_GPT2, sp9,
                             sizeof(MANY_TEXT) / sizeof(MANY_TEXT[0])) != 0) {
        fprintf(stderr, "FAIL: many-spec load\n");
        g_fail = 1;
    } else {
        run_cases(&t9, &plain);
        tsimd_tokenizer_free(&t9);
    }

    {
        tsimd_added_token_t nsp[NSPEC];
        for (int k = 0; k < NSPEC; k++) {
            nsp[k].text = SPEC_TEXT[k];
            nsp[k].id = SPEC_ID_BASE + 100u + (uint32_t)k;
            nsp[k].length = (uint32_t)strlen(SPEC_TEXT[k]);
            nsp[k].is_special = 0;
        }
        tsimd_tokenizer_t t2;
        if (tsimd_tokenizer_load(&t2, vocab_path, TSIMD_PRETOK_GPT2, nsp,
                                 NSPEC) != 0) {
            fprintf(stderr, "FAIL: nonspecial tokenizer load\n");
            g_fail = 1;
        } else {
            static const char *cases[] = {
                "<|eot|>", "x<|eot|>y", "<|", "a", "<|eot|><|eot|>",
                "aa<|eot|><|>b", "hello <|eot|> world",
            };
            for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
                uint32_t exp[256];
                size_t ne = 0;
                if (oracle(&t2, &plain, (const uint8_t *)cases[ci],
                           strlen(cases[ci]), exp, 256, &ne) != 0) {
                    fprintf(stderr, "FAIL: nonspecial oracle rc\n");
                    g_fail = 1;
                    continue;
                }
                tsimd_token_sequence_t seq;
                if (tsimd_tokenizer_encode(&t2, (const uint8_t *)cases[ci],
                                           strlen(cases[ci]), &seq) != 0) {
                    fprintf(stderr, "FAIL: nonspecial encode rc\n");
                    g_fail = 1;
                    continue;
                }
                checks++;
                if (seq.count != ne ||
                    memcmp(seq.ids, exp, ne * sizeof(uint32_t)) != 0) {
                    fprintf(stderr,
                            "FAIL: nonspecial seam ids differ (%zu vs %zu) "
                            "for '%s'\n",
                            seq.count, ne, cases[ci]);
                    g_fail = 1;
                }
                tsimd_token_sequence_free(&seq);
            }
            tsimd_tokenizer_free(&t2);
        }
    }

    tsimd_tokenizer_free(&t);
    tsimd_tokenizer_free(&plain);
    remove(vocab_path);
    if (g_fail) {
        fprintf(stderr, "test_added_scan: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_added_scan: OK (%d checks)\n", checks);
    return 0;
}
