#include "tokenizer.h"
#include "../src/model_internal.h"
#include "metaspace.h"
#include "unicode.h"

#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>

static int ms_old_has_invalid(const uint8_t *in, size_t len) {
    size_t i = 0;
    while (i < len) {
        uint8_t b = in[i];
        if (b < 0x80) {
            i++;
            continue;
        }
        size_t need = 0;
        if (b >= 0xC2 && b <= 0xDF) need = 1;
        else if (b >= 0xE0 && b <= 0xEF) need = 2;
        else if (b >= 0xF0 && b <= 0xF4) need = 3;
        else return 1;
        if (i + need >= len) return 1;
        for (size_t k = 1; k <= need; k++)
            if (in[i + k] < 0x80 || in[i + k] > 0xBF) return 1;
        if (b == 0xE0 && in[i + 1] < 0xA0) return 1;
        if (b == 0xED && in[i + 1] > 0x9F) return 1;
        if (b == 0xF0 && in[i + 1] < 0x90) return 1;
        if (b == 0xF4 && in[i + 1] > 0x8F) return 1;
        i += need + 1;
    }
    return 0;
}

static size_t ms_old(const uint8_t *in, size_t len, uint8_t *out,
                          size_t cap, int dummy_prefix) {
    size_t n = 0;
    size_t i = 0;
    if (dummy_prefix && len > 0) {

        if (n + 3 > cap) return (size_t)-1;
        out[n++] = 0xE2; out[n++] = 0x96; out[n++] = 0x81;
    }
    int clean = !ms_old_has_invalid(in, len);
#if defined(__x86_64__) || defined(__i386__)
    if (clean) {
        const __m128i sp = _mm_set1_epi8(' ');
        while (i + 16 <= len && n + 16 <= cap) {
            __m128i v =
                _mm_loadu_si128((const __m128i *)(const void *)(in + i));
            unsigned m =
                (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(v, sp));
            if (m == 0) {
                _mm_storeu_si128((__m128i *)(void *)(out + n), v);
                n += 16;
                i += 16;
                continue;
            }
            unsigned k = (unsigned)__builtin_ctz(m);
            if (k) {
                _mm_storeu_si128((__m128i *)(void *)(out + n), v);
                n += k;
                i += k;
            }
            if (n + 3 > cap) return (size_t)-1;
            out[n++] = 0xE2;
            out[n++] = 0x96;
            out[n++] = 0x81;
            i++;
        }
    }
#endif
    while (i < len) {
        if (in[i] == ' ') {
            if (n + 3 > cap) return (size_t)-1;
            out[n++] = 0xE2; out[n++] = 0x96; out[n++] = 0x81;
            i++;
            continue;
        }
        if (clean || in[i] < 0x80) {
            if (n + 1 > cap) return (size_t)-1;
            out[n++] = in[i++];
            continue;
        }
        {
            uint8_t b = in[i];
            size_t need = 0;
            if (b >= 0xC2 && b <= 0xDF) need = 1;
            else if (b >= 0xE0 && b <= 0xEF) need = 2;
            else if (b >= 0xF0 && b <= 0xF4) need = 3;
            size_t adv = 1;
            int valid = 0;
            if (need > 0) {
                size_t k = 0;
                while (k < need && i + 1 + k < len && in[i + 1 + k] >= 0x80 &&
                       in[i + 1 + k] <= 0xBF)
                    k++;
                adv = 1 + k;
                if (k == need) {
                    valid = 1;
                    if (b == 0xE0 && in[i + 1] < 0xA0) valid = 0;
                    if (b == 0xED && in[i + 1] > 0x9F) valid = 0;
                    if (b == 0xF0 && in[i + 1] < 0x90) valid = 0;
                    if (b == 0xF4 && in[i + 1] > 0x8F) valid = 0;
                }
            }
            if (valid) {
                if (n + adv > cap) return (size_t)-1;
                memcpy(out + n, in + i, adv);
                n += adv;
                i += adv;
            } else {
                if (n + 3 > cap) return (size_t)-1;
                out[n++] = 0xEF;
                out[n++] = 0xBF;
                out[n++] = 0xBD;
                i += adv;
            }
        }
    }
    return n;
}
static uint64_t rs = 0x12345678abcdefULL;

static uint64_t rnd(void) {
    rs ^= rs >> 12;
    rs ^= rs << 25;
    rs ^= rs >> 27;
    return rs * 0x2545F4914F6CDD1DULL;
}

int main(void) {
    size_t cap = 4096;
    uint8_t *in = malloc(cap);
    uint8_t *o1 = malloc(cap * 3 + 64);
    uint8_t *o2 = malloc(cap * 3 + 64);
    int fails = 0;
    for (int t = 0; t < 40000; t++) {
        size_t n = rnd() % 300;
        int kind = (int)(rnd() % 4);
        for (size_t i = 0; i < n; i++) {
            uint64_t r = rnd();
            if (kind == 0)
                in[i] = (uint8_t)(0x20 + (r % 95));
            else if (kind == 1)
                in[i] = (uint8_t)(r % 256);
            else if (kind == 2)
                in[i] = (uint8_t)(r % 2 ? 0x20 : 0x41 + (r >> 8) % 26);
            else
                in[i] = (uint8_t)(r % 8 ? 0x20 + (r >> 8) % 95 : r % 256);
        }
        for (int dp = 0; dp < 2; dp++) {
            size_t c1 = ms_old(in, n, o1, cap * 3 + 32, dp);
            size_t c2 = tsimd_metaspace_ex(in, n, o2, cap * 3 + 32, dp);
            if (c1 != c2 || (c1 != (size_t)-1 &&
                             memcmp(o1, o2, c1) != 0)) {
                if (fails < 5)
                    fprintf(stderr,
                            "DIFF t=%d kind=%d dp=%d n=%zu old=%zu "
                            "new=%zu\n",
                            t, kind, dp, n, c1, c2);
                fails++;
            }
        }
    }
    {
        uint8_t *pg = mmap(NULL, 8192, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pg == MAP_FAILED) {
            fprintf(stderr, "mmap failed\n");
            return 2;
        }
        if (mprotect(pg + 4096, 4096, PROT_NONE) != 0) return 2;
        for (size_t n = 1; n <= 4096; n += 7) {
            for (int kind = 0; kind < 2; kind++) {
                for (size_t i = 0; i < n; i++)
                    pg[i] = kind ? (uint8_t)(rnd() % 256)
                                 : (uint8_t)(rnd() % 2 ? ' ' : 0x41 + i % 26);
                size_t c1 = ms_old(pg, n, o1, 3 * n + 64, 0);
                size_t c2 = tsimd_metaspace_ex(pg, n, o2, 3 * n + 64, 0);
                if (c1 != c2 || (c1 != (size_t)-1 &&
                                 memcmp(o1, o2, c1) != 0)) {
                    fprintf(stderr, "PAGE DIFF n=%zu kind=%d\n", n, kind);
                    fails++;
                }
            }
        }
        munmap(pg, 8192);
    }
    printf("ms_diff: %d fails over 80000 pairs + page cases\n", fails);
    return fails != 0;
}
