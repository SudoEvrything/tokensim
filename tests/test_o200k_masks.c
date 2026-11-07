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
    if (tsimd_pretokenize_o200k(data, len, &a) != 0) { rc = 1; goto out; }
    if (tsimd_pretokenize_o200k_masks(data, len, &b) != 0) { rc = 1; goto out; }
    if (a.count != b.count) {
        fprintf(stderr, "FAIL %s: span count %zu vs %zu (len %zu)\n",
                label, a.count, b.count, len);
        fprintf(stderr, "  input:");
        for (size_t k = 0; k < len; k++) fprintf(stderr, " %02x", data[k]);
        fprintf(stderr, "\n");
        size_t nm = a.count < b.count ? a.count : b.count;
        for (size_t i = 0; i < nm; i++)
            if (a.spans[i].start != b.spans[i].start ||
                a.spans[i].end != b.spans[i].end) {
                fprintf(stderr,
                        "  first diff span %zu: masks [%u,%u) scalar [%u,%u)\n",
                        i, (unsigned)a.spans[i].start, (unsigned)a.spans[i].end,
                        (unsigned)b.spans[i].start, (unsigned)b.spans[i].end);
                size_t lo = a.spans[i].start >= 24 ? a.spans[i].start - 24 : 0;
                for (size_t k = lo; k < a.spans[i].end + 24 && k < len; k++)
                    fprintf(stderr, "%02x ", data[k]);
                fprintf(stderr, "\n");
                for (size_t j = (i > 3 ? i - 3 : 0); j < i + 5 && j < a.count; j++)
                    fprintf(stderr, "  m[%u,%u) ", (unsigned)a.spans[j].start, (unsigned)a.spans[j].end);
                fprintf(stderr, "\n  s");
                for (size_t j = (i > 3 ? i - 3 : 0); j < i + 5 && j < b.count; j++)
                    fprintf(stderr, " [%u,%u)", (unsigned)b.spans[j].start, (unsigned)b.spans[j].end);
                fprintf(stderr, "\n");
                fprintf(stderr, "  as text: ");
                for (size_t k = lo; k < a.spans[i].end + 24 && k < len; k++)
                    fprintf(stderr, "%c", data[k] >= 32 && data[k] < 127 ? data[k] : '.');
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
                    "FAIL %s: span %zu [%u,%u) vs [%u,%u) (len %zu)\n",
                    label, i, (unsigned)a.spans[i].start,
                    (unsigned)a.spans[i].end, (unsigned)b.spans[i].start,
                    (unsigned)b.spans[i].end, len);
            fprintf(stderr, "  input:");
            for (size_t k = 0; k < len; k++)
                fprintf(stderr, " %02x", data[k]);
            fprintf(stderr, "\n");
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

        "aBcDe", "ABCd", "aBC", "ABCD", "abCD", "aBcdef", "ſaB",
        "aaAAbb", "A1b", "a1B", "1aB",

        "1234567890", "1", "12", "123", "1234", "a12b3", "12a", "a1",
        "123456789012345678901234567890", "ｱ123",

        " hello", "  hello", "hello ", "h e l l o", " a", "a ",
        "\t\n hi", "a\tb", "a b ", "  ", "\t\t",

        "a\nb", "a\n\nb", "a \n b", "a\r\nb", "a\r\n\r\nb", "\n\n",
        "a\n", "\na", " \n", "\n ", "a\n  b", "a \n  b",

        "a.b", "a.b.c", "a...b", "a,b c", "a/ b", "a/\nb", "a.\nb",
        "a.\r\nb", "a/\r\nb", "a///b", "a.\n\nb", "..a", ".a", "a.",
        "a", "x.y", "a.b/c", "a..\n\nb",

        "it's", "don't", "won't", "they're", "x'll", "I'm", "can't",
        "it' s", "'s", "a'b", "a''b", "it'x", "it'ſ", "we've", "she'll",
        "the dog's", "o'clock", "rock 'n' roll",

        "The quick brown fox. It's 12:30 AM.\n\nJumped over.",
        "a ",
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (run_str("hand", cases[i])) return 1;

    {
        static const struct {
            const char *label;
            const uint8_t *data;
            size_t len;
        } byte_cases[] = {
            { "tail-fresh", (const uint8_t *)
              "\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x8b\x0d\x0d"
              "\x2f\x2f\x2f\x3c\x48\x0d\x0a\x0a\x41\x42", 23 },
            { "url-slash", (const uint8_t *)"://x", 4 },
            { "cr-fresh", (const uint8_t *)"\r/<H", 4 },
            { "run-slash", (const uint8_t *)"a/<H", 4 },
            { "cr-own-span", (const uint8_t *)"x\r\r//<H", 7 },
            { "tail-mid", (const uint8_t *)"x..\r\r/<H", 8 },
            { "tail-letters", (const uint8_t *)"x..\r\r//<Ab", 10 },
        };
        char pad[4096];
        memset(pad, 'q', 72);
        for (size_t i = 0; i < sizeof(byte_cases) / sizeof(byte_cases[0]);
             i++) {
            memcpy(pad + 72, byte_cases[i].data, byte_cases[i].len);
            if (spans_equal(byte_cases[i].label, (const uint8_t *)pad,
                            72 + byte_cases[i].len))
                return 1;
        }
    }

    char buf[4096];
    for (int rep = 0; rep < 96; rep++) {
        size_t n = 0;
        for (const char **cp = cases;
             cp < cases + sizeof(cases) / sizeof(cases[0]) && n < 3500; cp++) {
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
        " \n\r\t.,/?'-";
    uint8_t buf[512];
    for (int iter = 0; iter < 20000; iter++) {
        size_t n = (size_t)(rnd() % 400);
        for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];
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
    int r = file_ok("owt-full", TSIMD_SRC_DIR "/testdata/owt.txt");
    if (r < 0) r = file_ok("owt-full", "testdata/owt.txt");
    if (r < 0) return 1;
    r = file_ok("doc-full", TSIMD_SRC_DIR "/testdata/document.txt");
    if (r < 0) r = file_ok("doc-full", "testdata/document.txt");
    if (r < 0) return 1;
    printf("  full owt.txt + full document.txt: OK\n");
    return 0;
}

static int spans_vs_keyed(const uint8_t *buf, size_t n,
                          const char *label, int s2_shape) {
    tsimd_o200k_mstate_t s1 = {0, 0}, s2 = {0, 0};
    size_t *all_s = malloc(2000000 * sizeof(size_t));
    size_t *all_e = malloc(2000000 * sizeof(size_t));
    if (!all_s || !all_e) return -1;
    size_t starts[320], ends[320];
    size_t mi = 0;
    tsimd_pcache_span_t ents[320];
    size_t ei = 0;
    int rc = 0;

    for (;;) {
        int c = tsimd_o200k_mask_fill(buf, n, &s1, starts, ends, 256);
        if (c < 0) { rc = -1; break; }
        if (c == 0) break;
        for (int j = 0; j < c; j++) {
            if (mi >= 2000000) { rc = -1; break; }
            all_s[mi] = starts[j];
            all_e[mi] = ends[j];
            mi++;
        }
        if (rc != 0) break;
    }

    for (;;) {
        int c = tsimd_o200k_mask_fill_keyed(buf, n, &s2, ents, 256, NULL,
                                            s2_shape);
        if (c < 0) { rc = -1; break; }
        if (c == 0) break;
        for (int j = 0; j < c; j++) {
            uint64_t k0, k1, h;
            if (ei >= mi) { rc = -1; break; }
            tsimd_pcache_key_hash_bounded(buf + all_s[ei],
                                          (uint32_t)(all_e[ei] - all_s[ei]),
                                          buf + n, &k0, &k1, &h);
            uint32_t el = (uint32_t)(all_e[ei] - all_s[ei]);
            if (ents[j].ptr != buf + all_s[ei] ||
                tsimd_pcache_span_len(&ents[j]) != el) {
                fprintf(stderr, "%s: span mismatch at piece %zu\n",
                        label, ei);
                rc = -1;
                break;
            }
            if (el <= TSIMD_PCACHE_MAX_SPAN) {
                if (ents[j].k0 != k0 || ents[j].k1 != k1) {
                    fprintf(stderr, "%s: key mismatch at piece %zu\n",
                            label, ei);
                    rc = -1;
                    break;
                }
            } else if ((ents[j].k1 >> 56) != 0) {

                fprintf(stderr, "%s: long route carries a tag at %zu\n",
                        label, ei);
                rc = -1;
                break;
            }
            ei++;
        }
        if (rc != 0) break;
    }
    if (rc == 0 && ei != mi) {
        fprintf(stderr, "%s: piece count %zu vs %zu\n", label, ei, mi);
        rc = -1;
    }
    printf("  keyed-fill %s: pieces=%zu %s\n", label, ei,
           rc == 0 ? "OK" : "FAIL");
    free(all_s);
    free(all_e);
    return rc;
}

static int test_keyed_fill(void) {

    static const char *cases[] = {
        "", "a", "A", "1", " ", "\n", "\t", "'", ".", "/",
        "aBcDe", "ABCd", "1234567890", "\xEF\xBD\xB1""123",
        " hello", "  hello", "hello ", "h e l l o",
        "a\nb", "a\n\nb", "a\r\nb", "a\n  b", "a\t b ",
        "it's", "don't", "won't", "they're", "x'll",
        "the dog's", "o'clock", "rock 'n' roll",
        "The quick brown fox. It's 12:30 AM.\n\nJumped over.",
        "a", "x.y", "a.b/c", "a..\n\nb",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    };
    char buf[8192];
    for (int rep = 0; rep < 8; rep++) {
        size_t n = 0;
        for (int i = 0; i < 2000 && n < 7900; i++) {
            const char *s = cases[(size_t)(i * 2654435761u) %
                                  (sizeof(cases) / sizeof(cases[0]))];
            size_t l = strlen(s);
            if (n + l > 7950) break;
            memcpy(buf + n, s, l);
            n += l;
        }
        if (spans_vs_keyed(buf, n, "fuzzpool", 0) != 0) return 1;
    }
    return 0;
}

static int test_keyed_fill_twophase(void) {

    static const char *cases[] = {
        "", "a", "A", "1", " ", "\n", "\t", "'", ".", "/",
        "aBcDe", "ABCd", "1234567890", "\xEF\xBD\xB1""123",
        " hello", "  hello", "hello ", "h e l l o",
        "a\nb", "a\n\nb", "a\r\nb", "a\n  b", "a\t b ",
        "it's", "don't", "won't", "they're", "x'll",
        "the dog's", "o'clock", "rock 'n' roll",
        "The quick brown fox. It's 12:30 AM.\n\nJumped over.",
        "a", "x.y", "a.b/c", "a..\n\nb",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    };
    char buf[8192];
    for (int rep = 0; rep < 8; rep++) {
        size_t n = 0;
        for (int i = 0; i < 2000 && n < 7900; i++) {
            const char *s = cases[(size_t)(i * 2654435761u) %
                                  (sizeof(cases) / sizeof(cases[0]))];
            size_t l = strlen(s);
            if (n + l > 7950) break;
            memcpy(buf + n, s, l);
            n += l;
        }
        if (spans_vs_keyed(buf, n, "fuzzpool", 1) != 0) return 1;
    }
    return 0;
}

static int test_flatten_bounds(void) {

    static const uint64_t edge_masks[] = {
        0, 1, 1ull << 63, 1ull << 8, 1ull << 39, ~0ull,
        0x5555555555555555ull, 0xAAAAAAAAAAAAAAAAull,
        0x00000000000000FFull, 0xFF00000000000000ull,
        0x8000000000000080ull, 0x8000000000000001ull,
        0xFFFFFFFFFFFFFFFFull << 32, 0xFFFFFFFFull,
        0x80000000000000FFull, 0xFFull << 56 | 0xFF,
        0x8000000000000000ull | 0xFF,
    };
    uint32_t got[64 + 8], got2[64 + 8], ref[64 + 8];
    for (size_t m = 0; m < sizeof(edge_masks) / sizeof(edge_masks[0]); m++) {
        uint64_t bits = edge_masks[m];
        unsigned total = 0;
        for (uint64_t b = bits; b; b &= b - 1) total++;
        for (size_t cap = 0; cap <= total + 1; cap++) {

            static const uint32_t sbs[] = {
                0, 1, 2, 0xFFFFFFFEu, 0xFFFFFFC1u, 0xFFFFFFFFu,
            };
            for (size_t sv = 0; sv < sizeof(sbs) / sizeof(sbs[0]); sv++) {
                uint32_t sb = sbs[sv];
                size_t ng = tsimd_o200k_flatten_bounds(bits, sb, got, cap);
                size_t nb2 = tsimd_o200k_flatten_bounds_bmi2(bits, sb, got2,
                                                             cap);
                if (nb2 != ng ||
                    memcmp(got2, got, ng * sizeof(uint32_t)) != 0) {
                    fprintf(stderr,
                            "FAIL bmi2 flatten mask %zu cap %zu sb %u\n",
                            m, cap, sb);
                    return 1;
                }
                size_t nr = 0;
                for (uint64_t b = bits; b && nr < cap; b &= b - 1)
                    ref[nr++] = sb + (uint32_t)__builtin_ctzll(b);
                if (ng != nr) {
                    fprintf(stderr, "FAIL flatten mask %zu cap %zu: %zu vs %zu\n",
                            m, cap, ng, nr);
                    return 1;
                }
                for (size_t i = 0; i < nr; i++)
                    if (got[i] != ref[i]) {
                        fprintf(stderr,
                                "FAIL flatten mask %zu cap %zu at %zu: %u vs %u\n",
                                m, cap, i, got[i], ref[i]);
                        return 1;
                    }
            }
        }
    }

    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int it = 0; it < 200000; it++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint64_t bits = (it & 1) ? s & s >> 17
                                 : s | s >> 23;
        if ((it & 7) == 0) bits &= (1ull << (s & 63)) - 1;
        unsigned total = 0;
        for (uint64_t b = bits; b; b &= b - 1) total++;
        size_t cap = (s >> 33) % (total + 3);
        uint32_t sb = (uint32_t)(s >> 24);
        size_t ng = tsimd_o200k_flatten_bounds(bits, sb, got, cap);
        size_t nb2 = tsimd_o200k_flatten_bounds_bmi2(bits, sb, got2, cap);
        if (nb2 != ng || memcmp(got2, got, ng * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "FAIL bmi2 flatten rand it %d\n", it);
            return 1;
        }
        size_t nr = 0;
        for (uint64_t b = bits; b && nr < cap; b &= b - 1)
            ref[nr++] = sb + (uint32_t)__builtin_ctzll(b);
        if (ng != nr) {
            fprintf(stderr, "FAIL flatten rand it %d: %zu vs %zu\n",
                    it, ng, nr);
            return 1;
        }
        for (size_t i = 0; i < nr; i++)
            if (got[i] != ref[i]) {
                fprintf(stderr, "FAIL flatten rand it %d at %zu\n", it, i);
                return 1;
            }
    }
    return 0;
}

static int test_invalid_leads(void) {
    static const uint8_t filler[] =
        "the quick brown fox jumps over the lazy dog 1234 ";
    const size_t fl = sizeof(filler) - 1;
    static const uint8_t pat_bytes[][32] = {
        {0xff},
        {0xff,0x36,0x62},
        {0xff,0x39,0x68,0x62},
        {0xe6,0x20},
        {0xe6,0x20,0xa5,0x00},
        {0xf0,0x9f},
        {0xf0},
        {0xc3},
        {0xc3,0x20},
        {0xed,0xa0,0x80},
        {0xd9,0xa1,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39},
        {0xe6,0x97},
        {0xff,0xff,0xff,0xff},
        {0xf4,0x90,0x80,0x80},
        {0xc0,0x80},
    };
    static const size_t pat_len[] = {
        1,
        3,
        4,
        2,
        4,
        2,
        1,
        1,
        2,
        3,
        10,
        2,
        4,
        4,
        2,
    };

    static const uint8_t repro[] = {
        0x37, 0x65, 0x0d, 0x2e, 0x28, 0x3a, 0x30, 0x2c, 0x37, 0xa9, 0x20,
        0x2c, 0x35, 0x69, 0x09, 0xa9, 0x36, 0x00, 0x3b, 0x62, 0x64, 0x66,
        0xff, 0x3a, 0x0d, 0x39, 0x68, 0x62, 0x39, 0x2e, 0x68, 0x32, 0x33,
        0x64, 0x29, 0x00, 0x3a, 0x63, 0x5f, 0x20, 0xe6, 0x20, 0xa5, 0x00,
        0x65, 0x68, 0x67, 0x65, 0xff, 0x36, 0x62, 0x09, 0x31, 0xa5, 0x66,
        0x6a, 0x3a, 0x66, 0xa5, 0x20, 0x6b, 0x69, 0x20, 0x20, 0x66, 0x62,
        0x31, 0x37, 0x2d, 0x29, 0x66, 0x09, 0x62,
    };
    if (spans_equal("invalid repro", repro, sizeof(repro))) return 1;
    for (size_t p = 0; p < sizeof(pat_len) / sizeof(pat_len[0]); p++) {
        for (size_t shift = 0; shift <= 80; shift++) {
            uint8_t buf[256];
            size_t n = 0;
            while (n < shift) {
                buf[n] = filler[n % fl];
                n++;
            }
            size_t pl = pat_len[p];
            memcpy(buf + n, pat_bytes[p], pl);
            n += pl;

            const char *tail = "9hb 9.h23d) 1f 2g3h";
            memcpy(buf + n, tail, strlen(tail));
            n += strlen(tail);
            char label[96];
            snprintf(label, sizeof(label), "invalid pat%zu shift%zu", p,
                     shift);
            if (spans_equal(label, buf, n)) return 1;
        }
    }
    printf("  invalid-lead sweep: %zu patterns x 81 offsets OK\n",
           sizeof(pat_len) / sizeof(pat_len[0]));
    return 0;
}

static int test_flatten_bits16(void) {
    uint16_t o16[128];
    uint32_t o32[64];
    for (int rep = 0; rep < 3; rep++) {
        uint64_t s = 0x9E3779B97F4A7C15ull * (uint64_t)(rep + 1);
        for (int it = 0; it < 20000; it++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint64_t m = (it == 0) ? 0 : (it == 1) ? ~0ull
                       : (it < 10) ? (1ull << (it - 2)) : s;
            for (int k = 0; k < 128; k++) o16[k] = 0xDEAD;
            size_t n16 = tsimd_flatten_bits16(m, 0, o16);
            size_t n32 = tsimd_o200k_flatten_bounds(m, 0, o32, 64);
            if (n16 != n32) {
                fprintf(stderr, "FAIL flatten16 it %d: %zu vs %zu\n", it,
                        n16, n32);
                return 1;
            }
            for (size_t i = 0; i < n32; i++)
                if ((uint32_t)o16[i] != o32[i]) {
                    fprintf(stderr, "FAIL flatten16 it %d at %zu: %u vs %u\n",
                            it, i, o16[i], o32[i]);
                    return 1;
                }
        }
    }

    {
        uint16_t a[128];
        size_t n = tsimd_flatten_bits16(0x0F0Full, 7, a);

        if (n != 8 || a[0] != 7 || a[3] != 10 || a[4] != 15 || a[7] != 18) {
            fprintf(stderr, "FAIL flatten16 rel: n=%zu a0=%u a7=%u\n", n,
                    a[0], a[7]);
            return 1;
        }
    }
    printf("  flatten16: 60000 masks bit-identical to the reference\n");
    return 0;
}

static int keyed_spans_equal(const char *label, const uint8_t *data,
                             size_t len) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_O200K, data, len, &a) != 0) {
        fprintf(stderr, "FAIL %s: scalar ref rc\n", label);
        return 1;
    }
    static tsimd_pcache_span_t ents[16384];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    int n = tsimd_o200k_mask_fill_keyed(data, len, &st, ents, 16384, NULL,
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
        "\xe2\x80\x99", "a\xe2\x80\x99s",
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

static int test_tail_veto_artifact(void) {
    static const uint8_t art[] = {
        0x4f, 0xeb, 0x14, 0xb3, 0x6b, 0x34, 0x5d, 0xf0, 0x77, 0x72,
        0x9f, 0x3d, 0xaf, 0x4d, 0x65, 0x72, 0xde, 0xeb, 0xf0, 0x14,
        0x14, 0xf4, 0x4b, 0x5f, 0xd0, 0xf0, 0x5b, 0x25, 0x99, 0x30,
        0xe3, 0xd6, 0x83, 0x38, 0xf2, 0x21, 0x3b, 0xf4, 0x0a, 0x2f,
        0x04, 0x70, 0x04, 0xd8, 0x1e, 0x67, 0x66, 0x32, 0x46, 0xf0,
        0x32, 0x2a, 0x3c, 0x83, 0x7a, 0x22, 0x0c, 0x2c, 0xbd, 0xa3,
        0xa3, 0xe7, 0x72, 0xe3, 0xf0, 0x02, 0x47, 0xf0, 0x32, 0x37,
        0xed, 0x56, 0xda, 0x37, 0x1c, 0x2d, 0x32, 0x70, 0x18, 0x21,
        0x5a
    };
    for (size_t pad = 0; pad <= 24; pad++) {
        uint8_t buf[256];
        memset(buf, 'q', pad);
        memcpy(buf + pad, art, sizeof(art));
        if (spans_equal("tail-veto-artifact", buf,
                        pad + sizeof(art)))
            return 1;
    }
    printf("  tail-veto artifact: OK\n");
    return 0;
}

static int keyed_spans_equal_from(const char *label, const uint8_t *data,
                                  size_t len, size_t start_pos, size_t cap,
                                  int twophase) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = 0;
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_O200K, data, len, &a) !=
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
        int n = tsimd_o200k_mask_fill_keyed(data, len, &st, ents, cap, NULL,
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
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_O200K, buf, len, &a) != 0) {
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
    if (test_tail_veto_artifact()) return 1;
    if (test_hand_cases()) return 1;
    if (test_fuzz()) return 1;
    if (test_chunk()) return 1;
    if (test_keyed_fill()) return 1;
    if (test_keyed_fill_twophase()) return 1;
    if (test_flatten_bounds()) return 1;
    if (test_invalid_leads()) return 1;
    if (test_flatten_bits16()) return 1;
    printf("test_o200k_masks: all equivalence gates passed\n");
    return 0;
}
