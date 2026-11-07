#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unit8(uint8_t *b, uint32_t i) {
    memset(b, 0, 8);
    b[0] = (uint8_t)i;
    b[1] = (uint8_t)(i >> 8);
    b[2] = 0xAA;
    b[3] = 0xBB;
    b[4] = 1;
    b[5] = 2;
}

static int test_primary_growth(void) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 64) != 0) return 1;
    uint8_t b[8];
    uint64_t k0, k1, h;
    for (uint32_t i = 0; i < 900; i++) {
        unit8(b, i);
        tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
        uint32_t ids[1] = {0x1000 + i};
        tsimd_pcache_insert_keyed(&c, k0, k1, h, 8, ids, 1);
    }
    uint32_t out[4];
    uint32_t n = 0;
    for (uint32_t i = 0; i < 900; i++) {
        unit8(b, i);
        tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
        uint32_t *hit = tsimd_pcache_lookup_keyed(&c, k0, k1, h, out, &n);
        if (!hit || n != 1 || out[0] != 0x1000 + i) {
            fprintf(stderr, "FAIL primary growth: unit %u lost "
                            "(hit=%p n=%u id=%u table_bits=%zu count=%zu)\n",
                    i, (void *)hit, n, hit ? out[0] : 0,
                    c.mask + 1 ? 63 - __builtin_clzll(c.mask + 1) : 0,
                    c.count);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
    }
    if (c.mask + 1 < 2048) {
        fprintf(stderr, "FAIL primary growth: table did not grow (%zu)\n",
                c.mask + 1);
        tsimd_pretok_cache_free(&c);
        return 1;
    }
    printf("  primary growth: 900/900 hits at 2^%zu entries\n",
           (size_t)(63 - __builtin_clzll(c.mask + 1)));
    tsimd_pretok_cache_free(&c);
    return 0;
}

static void unit6(uint8_t *b, uint32_t i) {
    b[0] = (uint8_t)i;
    b[1] = (uint8_t)(i >> 8);
    b[2] = 0x5A;
    b[3] = (uint8_t)(i * 7);
    b[4] = 0xC3;
    b[5] = 0x11;
}

static int test_plong_growth(void) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 1024) != 0) return 1;
    c.lmin = 5;
    uint8_t b[6];
    const uint32_t N = 500000;
    for (uint32_t i = 0; i < N; i++) {
        unit6(b, i);
        uint32_t ids[4] = {i & 0xFFFF, 0x77, 0x12, 0x34};
        tsimd_plong_insert(&c, b, 6, ids, 4);
    }
    uint32_t n = 0;
    for (uint32_t i = 0; i < N; i += 97) {
        unit6(b, i);
        const uint32_t *ids = tsimd_plong_probe_ex(&c, b, 6, &n);
        if (!ids || n != 4 || ids[0] != (i & 0xFFFF)) {
            fprintf(stderr, "FAIL plong growth: unit %u lost (lbits=%u "
                            "lcount=%zu)\n",
                    i, c.lbits, c.lcount);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
    }
    if (c.lbits < 20) {
        fprintf(stderr, "FAIL plong growth: stuck at lbits=%u\n", c.lbits);
        tsimd_pretok_cache_free(&c);
        return 1;
    }
    printf("  plong growth: %u units intact at lbits=%u\n", N, c.lbits);
    tsimd_pretok_cache_free(&c);
    return 0;
}

int main(void) {
    if (test_primary_growth()) return 1;
    if (test_plong_growth()) return 1;
    printf("pcache grow: OK\n");
    return 0;
}
