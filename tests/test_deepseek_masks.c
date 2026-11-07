#include "pretokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int spans_equal(const char *label, const uint8_t *data, size_t len) {
    tsimd_span_array_t a, b;
    tsimd_span_array_init(&a);
    tsimd_span_array_init(&b);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, data, len, &a) !=
        0) {
        rc = 1;
        goto out;
    }
    if (tsimd_pretokenize_deepseek_masks(data, len, &b) != 0) {
        rc = 1;
        goto out;
    }
    if (a.count != b.count) {
        fprintf(stderr, "FAIL %s: span count %zu vs %zu (len %zu)\n", label,
                a.count, b.count, len);
        size_t nm = a.count < b.count ? a.count : b.count;
        for (size_t i = 0; i < nm; i++)
            if (a.spans[i].start != b.spans[i].start ||
                a.spans[i].end != b.spans[i].end) {
                fprintf(stderr,
                        "  first diff span %zu: scalar [%u,%u) masks [%u,%u)\n",
                        i, (unsigned)a.spans[i].start, (unsigned)a.spans[i].end,
                        (unsigned)b.spans[i].start, (unsigned)b.spans[i].end);
                size_t lo =
                    a.spans[i].start >= 24 ? a.spans[i].start - 24 : 0;
                for (size_t k = lo; k < a.spans[i].end + 24 && k < len; k++)
                    fprintf(stderr, "%02x ", data[k]);
                fprintf(stderr, "\n  as text: ");
                for (size_t k = lo; k < a.spans[i].end + 24 && k < len; k++)
                    fprintf(stderr, "%c",
                            data[k] >= 32 && data[k] < 127 ? data[k] : '.');
                fprintf(stderr, "\n");
                break;
            }
        rc = 1;
        goto out;
    }
    for (size_t i = 0; i < a.count; i++) {
        if (a.spans[i].start != b.spans[i].start ||
            a.spans[i].end != b.spans[i].end) {
            fprintf(stderr,
                    "FAIL %s: span %zu [%u,%u) vs [%u,%u) (len %zu)\n", label,
                    i, (unsigned)a.spans[i].start, (unsigned)a.spans[i].end,
                    (unsigned)b.spans[i].start, (unsigned)b.spans[i].end, len);
            rc = 1;
            goto out;
        }
    }
out:
    tsimd_span_array_free(&a);
    tsimd_span_array_free(&b);
    return rc;
}

static int run_str(const char *label, const char *s) {
    return spans_equal(label, (const uint8_t *)s, strlen(s));
}

static int test_hand_controls(void) {
    uint8_t buf[256];
    static const struct {
        const char *tail;
        size_t tlen;
    } tails[] = {
        {"/B", 2},   {"/Bc", 3},  {"/!ab", 4}, {"a", 1},   {"ab", 2},
        {"!!a", 3},  {"/", 1},    {"/!a", 3},  {"!ab", 3}, {"x/y", 3},
        {"\344\270\255a", 4},
    };
    for (size_t t = 0; t < sizeof(tails) / sizeof(tails[0]); t++) {
        for (size_t pad = 60; pad <= 70; pad++) {
            memset(buf, 0, pad);
            memcpy(buf + pad, tails[t].tail, tails[t].tlen);
            size_t n = pad + tails[t].tlen;
            if (spans_equal("ctrl-pad", buf, n)) return 1;
            memset(buf, 0, pad);
            memcpy(buf, "ab", 2);
            memset(buf + 2, 0, pad - 2);
            memcpy(buf + pad, tails[t].tail, tails[t].tlen);
            if (spans_equal("ctrl-pad2", buf, n)) return 1;
        }
        for (size_t pad = 66; pad <= 90; pad++) {
            memset(buf, 0, sizeof buf);
            memcpy(buf, "ab", 2);
            memset(buf + 2, 0, pad);
            memcpy(buf + 2 + pad, tails[t].tail, tails[t].tlen);
            size_t n = 2 + pad + tails[t].tlen;
            memset(buf + n, 0, 100);
            n += 100;
            if (spans_equal("ctrl-long", buf, n)) return 1;
        }
    }
    printf("  hand control-pad cases: OK\n");
    return 0;
}

static int test_hand_cases(void) {
    const char *cases[] = {
        "", "a", "A", "1", " ", "\n", "\t", "'", ".", "/",

        "1234567890", "1", "12", "123", "1234", "a12b3", "12a", "a1",
        " 12", "a 123",

        "\344\270\255", "\344\270\255\346\226\207", "\343\201\202",
        "\343\202\242", "ab\344\270\255cd", "a \344\270\255",
        "\344\270\255" "\343\201\202" "\343\202\242",
        "x" "\343\201\202" "y",

        "aBcDe", "ABCd", "aaAAbb", "hello", " hello", "  hello",
        "a.b", "a.b.c", "x!abc", "x!!abc", "a,b c", "a/ b",

        "a.\nb", "a.\r\nb", "a///b", "a.\n\nb", "a/\nb",

        "it's", "don't", "won't", "'s", "a'b", "a''b", "o'clock",
        "rock 'n' roll", "they're", "x'll", "I'm", "can't",

        "a\nb", "a\n\nb", "a \n b", "a\r\nb", "\n\n", "a\n", "\na",
        " \n", "\n ", "a\n  b", " \n\n b", "\t\n\nx",

        "The quick brown fox. It's 12:30 AM.\n\nJumped over.",
        "a ",

        "\001!abc", "\002!abc", "\177!abc", "\001/ab", "\001.Bc",
        "\000/B", "\000\000a", "\000\000\000ab", "\000a", "\001a",
        "a\001!b", "a\000/BC", " \001!b", "1\002!b", "\n\003!b",
        "\001\001/xyz", "q\001/xyz", "/\001!ab", "!!\001!ab",
        "\001!!ab", "\002//ab", "\177.a", "x\177/y",
        "\001\342\200\224!b", "\001\344\270\255!b", "\001!a\002!b",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (run_str("hand", cases[i])) return 1;

    char buf[4096];
    for (int rep = 0; rep < 96; rep++) {
        size_t n = 0;
        for (const char **cp = cases;
             cp < cases + sizeof(cases) / sizeof(cases[0]) && n < 3500;
             cp++) {
            size_t l = strlen(*cp);
            if (n + l > 3900) break;
            memcpy(buf + n, *cp, l);
            n += l;
        }
        if (spans_equal("border", (const uint8_t *)buf, n)) return 1;
    }
    printf("  hand cases: OK\n");
    return 0;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int test_fuzz(void) {
    static const char pool[] =
        "abAB01 .,'/\n\t\rxyx"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        " \n\r\t.,/?'-"

        "\344\270\255\346\226\207"
        "\343\201\202\343\202\242"
        "\302\240\343\200\200"
        "\314\201\314\250"
        "\303\251\303\211"

        "\342\200\230\342\200\231"
        "\342\200\234\342\200\235"
        "\342\200\224\342\200\223"
        "\342\200\246\342\200\242"
        "\342\202\254\302\251"
        "\342\200\213\342\200\215"
        "\303\015\344\270\315";
    uint8_t buf[512];
    for (int iter = 0; iter < 20000; iter++) {
        size_t n = (size_t)(rnd() % 400);
        for (size_t i = 0; i < n; i++)
            buf[i] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];
        if (spans_equal("fuzz", buf, n)) {
            FILE *df = fopen("/tmp/ds286fz.bin", "wb");
            fwrite(buf, 1, n, df);
            fclose(df);
            return 1;
        }
    }
    printf("  20000 fuzz strings: OK\n");
    return 0;
}

static int test_fullbyte_fuzz(void) {
    static const uint8_t alpha[] = {
        'a', 'A', 'z', '0', '7', ' ', '\t', '\n', '\r', '/', '!', '\'',
        '.', ',', 0x00, 0x01, 0x08, 0x0b, 0x1a, 0x7f, 0x20, 0x2f,
        0xc2, 0xa0, 0xa1, 0xad, 0xe2, 0x80, 0x94, 0xa2, 0xe4, 0xb8,
        0xad, 0xf0, 0x9f, 0x98, 0x80, 0x8b, 0xff, 0xfe, 0xc0,
    };
    uint8_t buf[160];
    for (int iter = 0; iter < 120000; iter++) {
        size_t n = 1 + (size_t)(rnd() % 130);
        for (size_t i = 0; i < n; i++)
            buf[i] = alpha[rnd() % (sizeof(alpha) - 1)];
        if (spans_equal("fullbyte", buf, n)) {
            FILE *df = fopen("/tmp/ds286fb.bin", "wb");
            fwrite(buf, 1, n, df);
            fclose(df);
            return 1;
        }
    }
    static const uint8_t asc[] = {'a', 'A', '0', ' ', '\t', '\n', '\r',
                                  '/', '!', '\'', '.'};
    uint8_t ex[8];
    int idx[8];
    for (int len = 1; len <= 5; len++) {
        long total = 1;
        for (int i = 0; i < len; i++) total *= 10;
        for (long v = 0; v < total; v++) {
            long x = v;
            for (int i = 0; i < len; i++) {
                ex[i] = asc[x % 10];
                x /= 10;
            }
            if (spans_equal("exh5", ex, (size_t)len)) return 1;
        }
    }
    printf("  fullbyte fuzz + exhaustive-5: OK\n");
    return 0;
}

static int file_ok(const char *name, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    static uint8_t *buf = NULL;
    buf = realloc(buf, (size_t)n);
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) return -1;
    return spans_equal(name, buf, (size_t)n);
}

static int test_chunk(void) {
    const char *paths[] = {
        TSIMD_SRC_DIR "/testdata/document.txt",
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        int rc = file_ok("chunk", paths[i]);
        if (rc < 0) continue;
        if (rc > 0) return 1;
    }
    printf("  real-text chunk: OK\n");
    return 0;
}

static int keyed_spans_equal(const char *label, const uint8_t *data,
                             size_t len) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, data, len, &a) != 0) {
        fprintf(stderr, "FAIL %s: scalar ref rc\n", label);
        return 1;
    }
    static tsimd_pcache_span_t ents[16384];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    int n = tsimd_deepseek_mask_fill_keyed(data, len, &st, ents, 16384, NULL,
                                         0);
    if (n < 0) {
        fprintf(stderr, "FAIL %s: keyed fill rc\n", label);
        tsimd_span_array_free(&a);
        return 1;
    }
    if ((size_t)n != a.count) {
        fprintf(stderr, "FAIL %s: keyed %d spans vs scalar %zu (len %zu)\n",
                label, n, a.count, len);
        rc = 1;
        goto out;
    }
    for (int i = 0; i < n; i++) {
        size_t off = (size_t)(ents[i].ptr - data);
        size_t sl = tsimd_pcache_span_len(&ents[i]);
        if (off != a.spans[i].start ||
            sl != a.spans[i].end - a.spans[i].start) {
            fprintf(stderr,
                    "FAIL %s: span %d keyed [%zu,%zu) scalar [%u,%u) "
                    "(len %zu)\n",
                    label, i, off, off + sl, (unsigned)a.spans[i].start,
                    (unsigned)a.spans[i].end, len);
            rc = 1;
            goto out;
        }
    }
out:
    tsimd_span_array_free(&a);
    return rc;
}

static int test_keyed_end_cases(void) {
    static const char *tails[] = {
        "",   " ",  "  ",     " x",          "x ",    "\n",   "a\n",
        "12", "123", "1234",  "12345",       "'s",    "'",     "''",
        ".",  "..",  ". ",    "\344\270\255", "\344", "\344\270",
        "\360\237", "\360\237\230\200", "/",     "/ ",   "\r\n",
        "\t", "a",  "' x",    "z 12",
        "/!ab", "/!a", "!!ab", "\001!ab", "\001!a", "\001\001ab",
        "\000/B", "\000/Bc", "\002/xy", "\177/A", "\001!!ab",
        "\000\001!a", "/\001!ab", "\001/!ab", "x\001!ab",
    };
    char buf[512];
    for (size_t t = 0; t < sizeof(tails) / sizeof(tails[0]); t++) {
        size_t tl = strlen(tails[t]);
        for (int pre = 48; pre <= 80; pre++) {
            size_t n = 0;
            while (n + 2 < (size_t)pre && n + 2 + tl < sizeof(buf) - 1) {
                buf[n++] = 'a';
                buf[n++] = 'b';
            }
            while (n < (size_t)pre && n + tl < sizeof(buf) - 1)
                buf[n++] = 'q';
            memcpy(buf + n, tails[t], tl);
            n += tl;
            char lbl[64];
            snprintf(lbl, sizeof lbl, "end sweep t=%zu pre=%d", t, pre);
            if (keyed_spans_equal(lbl, (const uint8_t *)buf, n))
                return 1;
        }
    }
    printf("  keyed end sweep: OK\n");
    return 0;
}

static int spans_match_exact(const char *label, const uint8_t *data,
                             size_t len, const tsimd_span_array_t *sp,
                             const uint32_t *want, size_t want_pairs) {
    if (sp->count * 2 != want_pairs) {
        fprintf(stderr, "FAIL %s: span count %zu want %zu (len %zu)\n",
                label, sp->count, want_pairs / 2, len);
        return 1;
    }
    for (size_t i = 0; i < want_pairs / 2; i++) {
        if (sp->spans[i].start != want[i * 2] ||
            sp->spans[i].end != want[i * 2 + 1]) {
            fprintf(stderr,
                    "FAIL %s: span %zu [%u,%u) want [%u,%u) (len %zu)\n",
                    label, i, (unsigned)sp->spans[i].start,
                    (unsigned)sp->spans[i].end, want[i * 2],
                    want[i * 2 + 1], len);
            return 1;
        }
    }
    return 0;
}

static int seam_one(const char *label, const char *s, const uint32_t *want,
                    size_t want_pairs) {
    uint8_t buf[512];
    size_t n = strlen(s);
    memcpy(buf, s, n);
    tsimd_span_array_t sp;
    tsimd_span_array_init(&sp);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, buf, n, &sp) !=
        0) {
        fprintf(stderr, "FAIL %s: scalar pretokenize error\n", label);
        rc = 1;
        goto out;
    }
    if (spans_match_exact(label, buf, n, &sp, want, want_pairs)) {
        rc = 1;
        goto out;
    }
    tsimd_span_array_free(&sp);
    tsimd_span_array_init(&sp);
    if (tsimd_pretokenize_deepseek_masks(buf, n, &sp) != 0) {
        fprintf(stderr, "FAIL %s: masks pretokenize error\n", label);
        rc = 1;
        goto out;
    }
    if (spans_match_exact(label, buf, n, &sp, want, want_pairs)) rc = 1;
out:
    tsimd_span_array_free(&sp);
    return rc;
}

static int seam_padded(const char *label, const char *s, const uint32_t *want,
                       size_t want_pairs) {
    uint8_t buf[512];
    tsimd_span_array_t sp;
    for (size_t pad = 0; pad <= 96; pad++) {
        size_t n = pad + strlen(s);
        if (n + 8 > sizeof buf) break;
        memset(buf, '\n', pad);
        memcpy(buf + pad, s, strlen(s));
        uint32_t wbuf[64];
        size_t wp = 0;
        if (pad > 0) {
            wbuf[wp++] = 0;
            wbuf[wp++] = (uint32_t)pad;
        }
        for (size_t i = 0; i < want_pairs / 2; i++) {
            wbuf[wp++] = want[i * 2] + (uint32_t)pad;
            wbuf[wp++] = want[i * 2 + 1] + (uint32_t)pad;
        }
        tsimd_span_array_init(&sp);
        int rc = 0;
        if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, buf, n,
                                         &sp) != 0 ||
            spans_match_exact(label, buf, n, &sp, wbuf, wp))
            rc = 1;
        tsimd_span_array_free(&sp);
        if (rc) return 1;
        tsimd_span_array_init(&sp);
        if (tsimd_pretokenize_deepseek_masks(buf, n, &sp) != 0 ||
            spans_match_exact(label, buf, n, &sp, wbuf, wp)) {
            tsimd_span_array_free(&sp);
            return 1;
        }
        tsimd_span_array_free(&sp);
    }
    return 0;
}

static int test_seam_cases(void) {
    static const struct {
        const char *s;
        uint32_t want[10];
        size_t pairs;
    } cases[] = {
        {")\343\203\273\"S", {0, 1, 1, 4, 4, 6}, 3},
        {"\343\203\273\"S", {0, 3, 3, 5}, 2},
        {"\"Sam", {0, 4}, 1},
        {"\343\203\273\"Sam", {0, 3, 3, 7}, 2},
        {"(\"Sam", {0, 2, 2, 5}, 2},
        {"a\"S", {0, 1, 1, 3}, 2},
        {"x\")\343\203\273\"S", {0, 1, 1, 3, 3, 6, 6, 8}, 4},
        {"\"\343\203\273S", {0, 1, 1, 4, 4, 5}, 3},
        {"\343\203\273\" Sam", {0, 3, 3, 4, 4, 8}, 3},
        {" \343\203\273\"Sam", {0, 1, 1, 4, 4, 8}, 3},
        {"&\343\203\273", {0, 1, 1, 4}, 2},
        {" &\343\203\273", {0, 2, 2, 5}, 2},
        {"&\343\200\202", {0, 4}, 1},
        {"a\343\203\273b", {0, 1, 1, 4, 4, 5}, 3},
        {"\343\203\273\na", {0, 3, 3, 4, 4, 5}, 3},
        {"123\343\203\273\"S", {0, 3, 3, 6, 6, 8}, 3},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (seam_one("seam", cases[i].s, cases[i].want, cases[i].pairs * 2))
            return 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (seam_padded("seam-pad", cases[i].s, cases[i].want,
                        cases[i].pairs * 2))
            return 1;
    {
        uint8_t buf[512];
        size_t n = 0;
        memset(buf, 'x', 300);
        n = 300;
        memcpy(buf + n, ")\343\203\273\"S", 6);
        n += 6;
        memset(buf + n, '\n', 50);
        n += 50;
        uint32_t want[] = {0,   300, 300, 301, 301, 304,
                           304, 306, 306, 356};
        tsimd_span_array_t sp;
        tsimd_span_array_init(&sp);
        int rc = 0;
        if (tsimd_pretokenize_deepseek_masks(buf, n, &sp) != 0 ||
            spans_match_exact("seam-long", buf, n, &sp, want, 10))
            rc = 1;
        tsimd_span_array_free(&sp);
        if (rc) return 1;
    }
    printf("  seam cases: OK\n");
    return 0;
}

static int keyed_spans_equal_from(const char *label, const uint8_t *data,
                                  size_t len, size_t start_pos, size_t cap,
                                  int twophase) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, data, len, &a) !=
        0) {
        fprintf(stderr, "FAIL %s: scalar ref rc\n", label);
        return 1;
    }
    size_t i0 = a.count;
    for (size_t i = 0; i < a.count; i++) {
        if (a.spans[i].start == start_pos) {
            i0 = i;
            break;
        }
    }
    if (i0 == a.count) {
        fprintf(stderr, "FAIL %s: pos %zu not a span boundary\n", label,
                start_pos);
        tsimd_span_array_free(&a);
        return 1;
    }
    static tsimd_pcache_span_t ents[512];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    st.pos = start_pos;
    st.scan = 64;
    size_t total = 0;
    for (;;) {
        int n = tsimd_deepseek_mask_fill_keyed(data, len, &st, ents, cap,
                                               NULL, twophase);
        if (n < 0) {
            fprintf(stderr, "FAIL %s: keyed fill rc\n", label);
            rc = 1;
            goto out;
        }
        if (n == 0) break;
        for (int i = 0; i < n; i++) {
            size_t off = (size_t)(ents[i].ptr - data);
            size_t sl = tsimd_pcache_span_len(&ents[i]);
            if (i0 + total >= a.count ||
                off != a.spans[i0 + total].start ||
                sl != (size_t)(a.spans[i0 + total].end -
                               a.spans[i0 + total].start)) {
                fprintf(stderr,
                        "FAIL %s: span %zu pos=%zu cap=%zu tp=%d keyed "
                        "[%zu,%zu) scalar [%u,%u) (len %zu)\n",
                        label, total, start_pos, cap, twophase, off, off + sl,
                        (unsigned)a.spans[i0 + total].start,
                        (unsigned)a.spans[i0 + total].end, len);
                rc = 1;
                goto out;
            }
            total++;
        }
    }
    if (i0 + total != a.count) {
        fprintf(stderr, "FAIL %s: total %zu+%zu vs scalar %zu\n", label, i0,
                total, a.count);
        rc = 1;
        goto out;
    }
out:
    tsimd_span_array_free(&a);
    return rc;
}

static int test_walker_cap(void) {
    uint8_t buf[256];
    size_t len = 0;
    while (len < 127) {
        buf[len++] = 'a';
        if (len < 127) buf[len++] = '1';
    }
    buf[126] = '1';
    buf[len++] = 0xe4;
    buf[len++] = 0xb8;
    buf[len++] = 0xad;
    while (len < 220) {
        buf[len++] = 'z';
        buf[len++] = '.';
    }
    int rc = 0;
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, buf, len, &a) !=
        0) {
        fprintf(stderr, "FAIL walker cap: scalar ref rc\n");
        return 1;
    }
    for (size_t i = 0; i < a.count; i++) {
        size_t s = a.spans[i].start;
        size_t e = a.spans[i].end;
        if (s == 0 || s >= 64 || e < 64) continue;
        char lbl[80];
        snprintf(lbl, sizeof lbl, "walker-cap pos=%zu", s);
        if (keyed_spans_equal_from(lbl, buf, len, s, 64, 0)) rc = 1;
        if (keyed_spans_equal_from(lbl, buf, len, s, 64, 1)) rc = 1;
    }
    tsimd_span_array_free(&a);
    if (!rc) printf("  walker cap: OK\n");
    return rc;
}

int main(void) {
    if (test_walker_cap()) return 1;
    if (test_keyed_end_cases()) return 1;
    if (test_hand_controls()) return 1;
    if (test_hand_cases()) return 1;
    if (test_seam_cases()) return 1;
    if (test_fuzz()) return 1;
    if (test_fullbyte_fuzz()) return 1;
    if (test_chunk()) return 1;
    printf("deepseek masks: OK\n");
    return 0;
}
