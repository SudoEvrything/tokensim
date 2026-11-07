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
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2, data, len, &a) != 0) {
        rc = 1;
        goto out;
    }
    if (tsimd_pretokenize_gpt2_masks(data, len, &b) != 0) {
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

static int test_hand_cases(void) {
    const char *cases[] = {
        "", "a", "A", "1", " ", "\n", "\t", "'", ".", "/",

        "it's", "don't", "won't", "'s", "'t", "'m", "'d", "x'll",
        "they're", "we've", "I'm", "o'clock", "rock 'n' roll",

        "DON'T", "X'S", "X'T", "X'LL", "X'VE", "X'RE", "I'M",
        "a'b", "a''b", "it'S", "it'T",

        " hello", "  hello", "hello", " 12", "  12", " 12ab", "a 123",
        "a 1", "a, b", ",abc", ".abc", "!abc", "x!abc", "x!!abc",
        "a ,b", "a. b",

        "1234567", "12345678901234567890", "12", "123", "1234",
        "a12b3", "12a", "a1", "a1b2c3", "  42",

        "a.b", "a.b.c", "a.\nb", "a.\r\nb", "a///b", "a/\nb", "a/ b",
        "a,\nb", "a.\n\nb",

        "a\nb", "a\n\nb", "a \n b", "a\r\nb", "\n\n", "a\n", "\na",
        " \n", "\n ", "a\n  b", " \n\n b", "\t\n\nx", "a\n\n\n b",

        "The quick brown fox. It's 12:30 AM.\n\nJumped over.",
        "a ",
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

        "\xc3\xa9\xc3\x89"
        "\xcc\x81\xcc\xa8"
        "\xc2\xa0\xe3\x80\x80"
        "\xe2\x80\x8b\xe2\x80\x8d"
        "\xc3\x0d\xe4\xb8\xad";
    uint8_t buf[512];
    for (int iter = 0; iter < 20000; iter++) {
        size_t n = (size_t)(rnd() % 400);
        for (size_t i = 0; i < n; i++)
            buf[i] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];
        if (spans_equal("fuzz", buf, n)) return 1;
    }
    printf("  20000 fuzz strings: OK\n");
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
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2, data, len, &a) != 0) {
        fprintf(stderr, "FAIL %s: scalar ref rc\n", label);
        return 1;
    }
    static tsimd_pcache_span_t ents[16384];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    int n = tsimd_gpt2_mask_fill_keyed(data, len, &st, ents, 16384, NULL,
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
        ".",  "..",  ". ",    "\xe4\xb8\xad", "\xe4", "\xe4\xb8",
        "\xf0\x9f", "\xf0\x9f\x98\x80", "/",     "/ ",   "\r\n",
        "\t", "a",  "' x",    "z 12",
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

static int keyed_spans_equal_ds(const char *label, const uint8_t *data,
                             size_t len) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2D, data, len, &a) != 0) {
        fprintf(stderr, "FAIL %s: scalar ref rc\n", label);
        return 1;
    }
    static tsimd_pcache_span_t ents[16384];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    st.digit_single = 1;
    int n = tsimd_gpt2_mask_fill_keyed(data, len, &st, ents, 16384, NULL,
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

static int test_keyed_end_cases_ds(void) {
    static const char *tails[] = {
        "",   " ",  "  ",     " x",          "x ",    "\n",   "a\n",
        "12", "123", "1234",  "12345",       "'s",    "'",     "''",
        ".",  "..",  ". ",    "\xe4\xb8\xad", "\xe4", "\xe4\xb8",
        "\xf0\x9f", "\xf0\x9f\x98\x80", "/",     "/ ",   "\r\n",
        "\t", "a",  "' x",    "z 12",
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
            if (keyed_spans_equal_ds(lbl, (const uint8_t *)buf, n))
                return 1;
        }
    }
    printf("  keyed end sweep: OK\n");
    return 0;
}

static int keyed_spans_equal_from(const char *label, const uint8_t *data,
                                  size_t len, size_t start_pos, size_t cap,
                                  int twophase) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2, data, len, &a) != 0) {
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
        int n = tsimd_gpt2_mask_fill_keyed(data, len, &st, ents, cap, NULL,
                                           twophase);
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
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2, buf, len, &a) != 0) {
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
    if (test_keyed_end_cases_ds()) return 1;
    if (test_hand_cases()) return 1;
    if (test_fuzz()) return 1;
    if (test_chunk()) return 1;
    printf("gpt2 masks: OK\n");
    return 0;
}
