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
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2D, data, len, &a) != 0) {
        rc = 1;
        goto out;
    }
    if (tsimd_pretokenize_gpt2d_masks(data, len, &b) != 0) {
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
    static const char *cases[] = {
        "", "a", "A", "1", " ", "\n", "\t", "'", ".", "/",

        "123", " 12", "  12", "   1", " 12ab", "a 123", "a 1", "a1",
        "a1b2c3", "  42", "1234567", "12345678901234567890",
        "12 34 56", "1 2 3", "1,234", "1.5", "3.14 2.71",

        "it's", "don't", "won't", "'s", "'t", "'m", "'d", "x'll",
        "they're", "we've", "I'm", "o'clock", "rock 'n' roll",
        "1's", "2'll", "3're", "42've", "7'd", "7's",
        "x 1's y", "1 's", "1'2", "1''2", "it's 12 o'clock",

        "DON'T", "X'S", "X'T", "X'LL", "X'VE", "X'RE", "I'M",
        "a'b", "a''b", "it'S", "it'T",

        " hello", "  hello", "hello", " 12", "  12", " 12ab",
        "a, b", ",abc", ".abc", "!abc", "x!abc", "x!!abc",
        "a ,b", "a. b", "a .1", "a . 1",

        "a\nb", "a\n\nb", "a \n b", "a\r\nb", "\n\n", "a\n", "\na",
        " \n", "\n ", "a\n  b", " \n\n b", "\t\n\nx", "a\n\n\n b",
        "a \n1", "a\n\n1", "a \n 1", "\n\n1", "  \n 1", "a\n1\n",
        "word  ", "a ", "  ", "  \n  ", "\t\t\t", " \t\n\r \r\n",

        "The quick brown fox. It's 12:30 AM.\n\nJumped over 42 times.",
        "In 2023, they've built 1,000,000 homes; that's 50% more.",
        "e\xCC\x81", "h\xC3\xA9llo", "\xE4\xB8\xAD\xE6\x96\x87",
        "\xE4\xB8\xAD" "1\xE6\x96\x87", "1" "\xE4\xB8\xAD",
        "\xD9\xA1\xD9\xA2\xD9\xA3", " \xD9\xA1", "a \xD9\xA1\xD9\xA2",
        "\xD9\xA1" "a", "12\xD9\xA1" "\xD9\xA2" "34",
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

static int test_window_edges(void) {
    static const char *frag[] = {
        "word ", "123 ", " 's ", " \n ", "45", "x", "  ", "\t",
        "678", " re ", "hi ", "90\n", "don't ", "12 34 ",
    };
    for (int off = 1; off < 96; off++) {
        for (size_t f = 0; f < sizeof(frag) / sizeof(frag[0]); f++) {
            for (size_t g = 0; g < sizeof(frag) / sizeof(frag[0]); g++) {
                char buf[512];
                size_t n = 0;
                for (int k = 0; k < off && n < sizeof(buf) - 1; k++)
                    buf[n++] = (char)('a' + (k % 26));
                size_t l1 = strlen(frag[f]);
                if (n + l1 > sizeof(buf)) continue;
                memcpy(buf + n, frag[f], l1);
                n += l1;
                size_t l2 = strlen(frag[g]);
                if (n + l2 > sizeof(buf)) continue;
                memcpy(buf + n, frag[g], l2);
                n += l2;
                for (int k = 0; k < 40 && n < sizeof(buf); k++)
                    buf[n++] = "word 12 's \n"[k % 12];
                if (spans_equal("edge", (const uint8_t *)buf, n)) return 1;
            }
        }
    }
    printf("  window edges: OK\n");
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
        "1234567890 12345 6789"
        "\xc3\xa9\xc3\x89"
        "\xcc\x81\xcc\xa8"
        "\xc2\xa0\xe3\x80\x80"
        "\xd9\xa1\xd9\xa2\xe4\xb8\x80"
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

static int test_corpus(void) {
    const char *path = "testdata/owt.txt";
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("  (corpus not present, skipping)\n");
        return 0;
    }
    static uint8_t buf[4u << 20];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (spans_equal("corpus", buf, n)) return 1;
    printf("  real-text slice (%zu bytes): OK\n", n);
    return 0;
}

static int test_invalid_seam(void) {
    static const struct {
        const char *s;
        size_t len;
        const uint32_t want[20];
        size_t pairs;
    } cases[] = {
        {"\x0a\xd7\x20\xc2\xa0\xa0\xc7\xa0\xa0\xd7\x0a\x0a", 12,
         {0, 1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 8, 8, 10, 10, 12}, 8},
        {"\x20\xc2\xa0\x61", 4, {0, 1, 1, 3, 3, 4}, 3},
        {"\x20\xc2\xa0", 3, {0, 3}, 1},
        {"\x61\xc2\xa0\x62", 4, {0, 1, 1, 3, 3, 4}, 3},
        {"\x61\x20\xc2\xa0\x62", 5, {0, 1, 1, 2, 2, 4, 4, 5}, 4},
        {"\xa0", 1, {0, 1}, 1},
        {"\x61\xa0\x62", 3, {0, 1, 1, 2, 2, 3}, 3},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        tsimd_span_array_t sp;
        tsimd_span_array_init(&sp);
        int rc = 0;
        if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_GPT2D,
                                         (const uint8_t *)cases[i].s,
                                         cases[i].len, &sp) != 0) {
            fprintf(stderr, "FAIL invalid-seam %zu: scalar error\n", i);
            rc = 1;
            goto out;
        }
        if (sp.count != cases[i].pairs) {
            fprintf(stderr, "FAIL invalid-seam %zu: scalar count %zu want %zu\n",
                    i, sp.count, (size_t)cases[i].pairs);
            rc = 1;
            goto out;
        }
        for (size_t k = 0; k < cases[i].pairs; k++) {
            if (sp.spans[k].start != cases[i].want[k * 2] ||
                sp.spans[k].end != cases[i].want[k * 2 + 1]) {
                fprintf(stderr,
                        "FAIL invalid-seam %zu: scalar span %zu [%u,%u) want [%u,%u)\n",
                        i, k, (unsigned)sp.spans[k].start,
                        (unsigned)sp.spans[k].end, cases[i].want[k * 2],
                        cases[i].want[k * 2 + 1]);
                rc = 1;
                goto out;
            }
        }
        tsimd_span_array_free(&sp);
        tsimd_span_array_init(&sp);
        if (tsimd_pretokenize_gpt2d_masks((const uint8_t *)cases[i].s,
                                          cases[i].len, &sp) != 0) {
            fprintf(stderr, "FAIL invalid-seam %zu: masks error\n", i);
            rc = 1;
            goto out;
        }
        if (sp.count != cases[i].pairs) {
            fprintf(stderr, "FAIL invalid-seam %zu: masks count %zu want %zu\n",
                    i, sp.count, (size_t)cases[i].pairs);
            rc = 1;
            goto out;
        }
        for (size_t k = 0; k < cases[i].pairs; k++) {
            if (sp.spans[k].start != cases[i].want[k * 2] ||
                sp.spans[k].end != cases[i].want[k * 2 + 1]) {
                fprintf(stderr,
                        "FAIL invalid-seam %zu: masks span %zu [%u,%u) want [%u,%u)\n",
                        i, k, (unsigned)sp.spans[k].start,
                        (unsigned)sp.spans[k].end, cases[i].want[k * 2],
                        cases[i].want[k * 2 + 1]);
                rc = 1;
                goto out;
            }
        }
out:
        tsimd_span_array_free(&sp);
        if (rc) return 1;
    }
    printf("  invalid seam: OK\n");
    return 0;
}

int main(void) {
    if (test_hand_cases()) return 1;
    if (test_window_edges()) return 1;
    if (test_invalid_seam()) return 1;
    if (test_fuzz()) return 1;
    if (test_corpus()) return 1;
    printf("gpt2d spanner: OK\n");
    return 0;
}
