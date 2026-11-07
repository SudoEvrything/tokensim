#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unit8(uint8_t *b, uint32_t i) {
    memset(b, 0, 8);
    b[0] = (uint8_t)i;
    b[1] = (uint8_t)(i >> 8);
    b[2] = 0xA5;
    b[3] = 0x5A;
    b[4] = 7;
    b[5] = 9;
}

static int probe_hit(uint32_t i, uint32_t *ids) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 256) != 0) return 1;
    uint8_t b[8];
    unit8(b, i);
    uint64_t k0, k1, h;
    tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
    tsimd_pcache_insert_keyed(&c, k0, k1, h, 8, ids, 4);
    uint64_t val = 0, ext = 0;
    size_t slot = (size_t)-1;
    int got = tsimd_pcache_probe_serve(&c, k0, k1, h, &val, &ext, &slot);
    int rc = 0;
    if (!got) {
        fprintf(stderr, "FAIL serve hit: unit %u missed\n", i);
        rc = 1;
        goto out;
    }
    uint32_t n = (uint32_t)(val & 7);
    uint32_t v[4];
    v[0] = (uint32_t)((val >> 3) & 0x1FFFFFFF);
    v[1] = (uint32_t)(val >> 32);
    v[2] = (uint32_t)ext;
    v[3] = (uint32_t)(ext >> 32);
    if (n != 4 || v[0] != ids[0] || v[1] != ids[1] || v[2] != ids[2] ||
        v[3] != ids[3]) {
        fprintf(stderr,
                "FAIL serve hit: unit %u ids %u/%u/%u/%u want "
                "%u/%u/%u/%u\n",
                i, v[0], v[1], v[2], v[3], ids[0], ids[1], ids[2], ids[3]);
        rc = 1;
        goto out;
    }
out:
    tsimd_pretok_cache_free(&c);
    return rc;
}

static int test_hits(void) {
    uint32_t ids[4] = {11, 22, 33, 44};
    if (probe_hit(0, ids)) return 1;
    if (probe_hit(1, ids)) return 1;
    if (probe_hit(0x1234, ids)) return 1;
    printf("  serve probe hits: OK\n");
    return 0;
}

static int test_miss_slot(void) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 256) != 0) return 1;
    uint8_t b[8];
    unit8(b, 777);
    uint64_t k0, k1, h;
    tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
    uint64_t val = 0, ext = 0;
    size_t slot = (size_t)-1;
    int got = tsimd_pcache_probe_serve(&c, k0, k1, h, &val, &ext, &slot);
    if (got || slot == (size_t)-1) {
        fprintf(stderr, "FAIL serve miss: got=%d slot=%zu\n", got, slot);
        tsimd_pretok_cache_free(&c);
        return 1;
    }
    uint32_t ids[2] = {5, 6};
    tsimd_pcache_insert_keyed_at(&c, slot, k0, k1, h, 8, ids, 2);
    uint32_t n = 0;
    uint32_t out[4];
    if (!tsimd_pcache_lookup_keyed(&c, k0, k1, h, out, &n) || n != 2 ||
        out[0] != 5 || out[1] != 6) {
        fprintf(stderr, "FAIL serve miss->insert->lookup\n");
        tsimd_pretok_cache_free(&c);
        return 1;
    }
    printf("  serve probe miss+slot: OK\n");
    tsimd_pretok_cache_free(&c);
    return 0;
}

static int test_chain(void) {
    tsimd_pretok_cache_t c;
    if (tsimd_pretok_cache_init(&c, 256) != 0) return 1;
    uint8_t b[8];
    uint64_t k0, k1, h, want_slot = (size_t)-1;
    uint32_t filler[4] = {1, 2, 3, 4};
    uint32_t last = 0;
    for (uint32_t i = 0; i < 64; i++) {
        unit8(b, 0xC000 + i);
        tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
        uint64_t val = 0, ext = 0;
        size_t slot = (size_t)-1;
        int got = tsimd_pcache_probe_serve(&c, k0, k1, h, &val, &ext,
                                           &slot);
        if (got) {
            fprintf(stderr, "FAIL chain: unit %u hit before insert\n", i);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
        if (slot == (size_t)-1) {
            fprintf(stderr, "FAIL chain: no slot at %u\n", i);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
        tsimd_pcache_insert_keyed_at(&c, slot, k0, k1, h, 8, filler, 4);
        last = i;
        (void)want_slot;
    }
    for (uint32_t i = 0; i <= last; i++) {
        unit8(b, 0xC000 + i);
        tsimd_pcache_key_hash(b, 8, &k0, &k1, &h);
        uint64_t val = 0, ext = 0;
        size_t slot = (size_t)-1;
        if (!tsimd_pcache_probe_serve(&c, k0, k1, h, &val, &ext, &slot)) {
            fprintf(stderr, "FAIL chain: unit %u lost\n", i);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
        if ((uint32_t)((val >> 3) & 0x1FFFFFFF) != 1 ||
            (uint32_t)(val >> 32) != 2) {
            fprintf(stderr, "FAIL chain: unit %u ids wrong\n", i);
            tsimd_pretok_cache_free(&c);
            return 1;
        }
    }
    printf("  serve probe chain: OK\n");
    tsimd_pretok_cache_free(&c);
    return 0;
}

int main(void) {
    if (test_hits()) return 1;
    if (test_miss_slot()) return 1;
    if (test_chain()) return 1;
    printf("pcache serve probe: OK\n");
    return 0;
}
