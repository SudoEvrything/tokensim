#include "pretokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SCH_NAME[] = {"gpt2",   "cl100k", "gpt2d",
                                 "deepseek", "o200k",  "nemo"};
static const tsimd_pretok_scheme_t SCH_ID[] = {
    TSIMD_PRETOK_GPT2, TSIMD_PRETOK_CL100K, TSIMD_PRETOK_GPT2D,
    TSIMD_PRETOK_DEEPSEEK, TSIMD_PRETOK_O200K, TSIMD_PRETOK_NEMO};

typedef int (*masks_fn)(const uint8_t *, size_t, tsimd_span_array_t *);
static const masks_fn MASKS_FN[] = {
    tsimd_pretokenize_gpt2_masks, tsimd_pretokenize_cl100k_masks,
    tsimd_pretokenize_gpt2d_masks, tsimd_pretokenize_deepseek_masks,
    tsimd_pretokenize_o200k_masks, tsimd_pretokenize_nemo_masks};

static int cmp_spans(tsimd_span_array_t *a, tsimd_span_array_t *b) {
    if (a->count != b->count) return 0;
    for (size_t i = 0; i < a->count; i++)
        if (a->spans[i].start != b->spans[i].start ||
            a->spans[i].end != b->spans[i].end)
            return 0;
    return 1;
}

static void show(const uint8_t *d, size_t n, tsimd_span_array_t *a,
                 tsimd_span_array_t *b) {
    printf("  in(%zu):", n);
    for (size_t i = 0; i < n; i++) printf(" %02x", d[i]);
    printf("\n  law:");
    for (size_t i = 0; i < a->count; i++)
        printf(" [%u,%u)", a->spans[i].start, a->spans[i].end);
    printf("\n  masks:");
    for (size_t i = 0; i < b->count; i++)
        printf(" [%u,%u)", b->spans[i].start, b->spans[i].end);
    printf("\n");
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int one_case(int scheme, const uint8_t *d, size_t n, int show_max,
                    int *shown) {
    tsimd_span_array_t a, b;
    tsimd_span_array_init(&a);
    tsimd_span_array_init(&b);
    int rc = tsimd_pretokenize_scalar_ref(SCH_ID[scheme], d, n, &a);
    if (rc == 0) rc = MASKS_FN[scheme](d, n, &b);
    int ok = rc == 0 && cmp_spans(&a, &b);
    if (!ok && *shown < show_max) {
        printf("MISMATCH %s:\n", SCH_NAME[scheme]);
        show(d, n, &a, &b);
        (*shown)++;
    }
    tsimd_span_array_free(&a);
    tsimd_span_array_free(&b);
    return ok;
}

static int exhaustive(int scheme, const uint8_t *alpha, int na, int maxlen,
                      long cap, const char *tag, int *shown) {
    uint8_t buf[16];
    int idx[16];
    long run = 0;
    int bad = 0;
    for (int len = 1; len <= maxlen; len++) {
        memset(idx, 0, sizeof idx);
        long total = 1;
        for (int i = 0; i < len; i++) total *= na;
        if (cap > 0 && total > cap) break;
        for (long v = 0; v < total; v++) {
            long x = v;
            for (int i = 0; i < len; i++) {
                buf[i] = (uint8_t)alpha[x % na];
                x /= na;
            }
            if (!one_case(scheme, buf, (size_t)len, 3, shown)) bad++;
            run++;
        }
    }
    printf("%s %s: %ld cases, %d mismatches\n", SCH_NAME[scheme], tag, run,
           bad);
    return bad;
}

static int boundary_tier(int scheme, const uint8_t *seeds, const int *slen,
                         int ns, const uint8_t *fill, int nf, int *shown) {
    uint8_t buf[256];
    long run = 0;
    int bad = 0;
    for (int pre = 0; pre < ns; pre++)
        for (int p = 0; p < 100; p++)
            for (int post = 0; post < ns; post++) {
                size_t n = 0;
                memcpy(buf, seeds + pre * 8, (size_t)slen[pre]);
                n += (size_t)slen[pre];
                for (int k = 0; k < p; k++) buf[n++] = fill[k % nf];
                memcpy(buf + n, seeds + post * 8, (size_t)slen[post]);
                n += (size_t)slen[post];
                if (n < 48 || n > 130) continue;
                if (!one_case(scheme, buf, n, 3, shown)) bad++;
                run++;
            }
    printf("%s boundary: %ld cases, %d mismatches\n", SCH_NAME[scheme], run,
           bad);
    return bad;
}

static int random_tier(int scheme, long iters, int maxlen, int *shown) {
    uint8_t buf[256];
    long run = 0;
    int bad = 0;
    for (long it = 0; it < iters; it++) {
        size_t n = 1 + (size_t)(rnd() % (size_t)maxlen);
        for (size_t i = 0; i < n; i++) {
            uint8_t c;
            uint64_t r = rnd();
            switch (r % 10) {
                case 0: c = (uint8_t)(r % 0x20); break;
                case 1: c = (uint8_t)(0x80 + (r % 0x40)); break;
                case 2: c = (uint8_t)(0xc0 + (r % 0x20)); break;
                case 3: c = (uint8_t)(0xe0 + (r % 0x10)); break;
                case 4: c = (uint8_t)(0xf0 + (r % 0x08)); break;
                default: c = (uint8_t)(0x20 + (r % 0x60)); break;
            }
            buf[i] = c;
        }
        if (!one_case(scheme, buf, n, 3, shown)) bad++;
        run++;
    }
    printf("%s random: %ld cases, %d mismatches\n", SCH_NAME[scheme], run,
           bad);
    return bad;
}

static int hunt_tier(int scheme, long iters, int maxlen, int *shown) {
    uint8_t buf[256];
    long run = 0;
    int bad = 0;
    for (long it = 0; it < iters; it++) {
        size_t n = 1 + (size_t)(rnd() % (size_t)maxlen);
        for (size_t i = 0; i < n; i++) {
            uint8_t c;
            uint64_t r = rnd();
            switch (r % 8) {
                case 0: c = '\n'; break;
                case 1: c = '\r'; break;
                case 2: c = '/'; break;
                case 3: c = (uint8_t)(r % 0x20); break;
                case 4: c = (uint8_t)(0x80 + (r % 0x40)); break;
                case 5: c = (uint8_t)('a' + (r % 26)); break;
                case 6: c = (uint8_t)('A' + (r % 26)); break;
                default: c = (uint8_t)(0x20 + (r % 0x60)); break;
            }
            buf[i] = c;
        }
        if (!one_case(scheme, buf, n, 3, shown)) bad++;
        run++;
    }
    printf("%s hunt: %ld cases, %d mismatches\n", SCH_NAME[scheme], run,
           bad);
    return bad;
}

int main(int argc, char **argv) {
    int only = argc > 1 ? atoi(argv[1]) : -1;
    long hunt_iters = argc > 2 ? strtol(argv[2], NULL, 10) : 0;
    if (hunt_iters > 0) {
        int total = 0;
        for (int s = 0; s < 6; s++) {
            if (only >= 0 && s != only) continue;
            int shown = 0;
            total += hunt_tier(s, hunt_iters, 130, &shown);
        }
        printf("HUNT mismatches: %d\n", total);
        return total != 0;
    }
    static const uint8_t ascii_a[] = {
        'a', 'A', '0', '7', ' ', '\t', '\n', '\r', '/', '!', '\'', '.'};
    static const uint8_t ascii_b[] = {'a', '0', ' ', '\n', '/', '!', '\'', '\t'};
    static const uint8_t uni_a[] = {'a', ' ', '\n', '/', '!', 0xC2, 0xA0,
                                    0xA1, 0xE2, 0x80, 0x94, 0xAD};
    static const uint8_t fill[] = {'q', ' ', '/', '\n', '!', '\'',
                                   0xC2, 0xA0, '\t', 0x00};
    static const uint8_t seeds[][8] = {
        {'a', 'b'}, {'a'}, {'1', '2', '3'}, {' '}, {' ', ' '},
        {'/', 'B'}, {'/'}, {'!'}, {'\''}, {'\'', 's'},
        {'.'}, {'\n'}, {'\r', '\n'}, {'\t'},
        {'q', 0xC2, 0xA0}, {'q', 0xE2, 0x80, 0x94}, {'q', 0xE4, 0xB8, 0xAD},
        {0x00, 0x00}, {0x80}, {0x7f},
        {'a', '/', 'B'}, {'a', ' ', 'b'}, {'a', '!', 'B'},
        {'0', 0x00}, {'\n', 'a'}, {' ', 'a'},
    };
    static const int slen[] = {2, 1, 3, 1, 2, 2, 1, 1, 1, 2,
                               1, 1, 2, 1, 3, 4, 4, 2, 1, 1,
                               3, 3, 3, 2, 2, 2};
    int total = 0;
    int ns = (int)(sizeof(slen) / sizeof(slen[0]));
    for (int s = 0; s < 6; s++) {
        if (only >= 0 && s != only) continue;
        int shown = 0;
        total += exhaustive(s, ascii_a, 12, 6, 0, "ascii-6", &shown);
        total += exhaustive(s, ascii_b, 8, 8, 0, "ascii-8b", &shown);
        total += exhaustive(s, uni_a, 12, 5, 0, "uni-5", &shown);
        total += exhaustive(s, uni_a, 12, 6, 4000000, "uni-6cap", &shown);
        total += boundary_tier(s, (const uint8_t *)seeds, slen, ns, fill, 10,
                               &shown);
        total += random_tier(s, 150000, 120, &shown);
    }
    printf("TOTAL mismatches: %d\n", total);
    return total != 0;
}
