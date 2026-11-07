#include "bpe.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    tsimd_pretok_cache_t ref, at;
    if (tsimd_pretok_cache_init(&ref, 1u << 15) != 0) return 1;
    if (tsimd_pretok_cache_init(&at, 1u << 15) != 0) return 1;

    static const char pool[] = "abAB01 .,'/\n\txyx-?";

    {
        uint8_t nul[16] = {0, 0, 0, 0, 0, 0, 0, 0, 'a', 'b', 'c', 'd', 0, 0,
                           0, 0};
        uint64_t nk0, nk1, nh;
        tsimd_pcache_key_hash(nul, 12, &nk0, &nk1, &nh);
        if (nk0 != 0 || nk1 == 0) {
            fprintf(stderr, "FAIL sentinel: NUL-led key k0=%llu k1=%llu\n",
                    (unsigned long long)nk0, (unsigned long long)nk1);
            return 1;
        }
        tsimd_pretok_cache_t sc;
        if (tsimd_pretok_cache_init(&sc, 1u << 10) != 0) return 1;
        uint32_t sids[2] = { 7u, 9u };
        tsimd_pcache_insert_keyed(&sc, nk0, nk1, nh, 12, sids, 2);
        uint64_t sval = 0, sext = 0;
        size_t sslot = 0;
        if (tsimd_pcache_probe_merged(&sc, nk0, nk1, nh, &sval, &sext,
                                      &sslot) != 1) {
            fprintf(stderr, "FAIL sentinel: the NUL-led entry missed\n");
            return 1;
        }
        if ((uint32_t)(sval & 7) != 2 ||
            (uint32_t)((sval >> 3) & 0x1FFFFFFF) != 7 ||
            (uint32_t)(sval >> 32) != 9) {
            fprintf(stderr, "FAIL sentinel: served val=%llx\n",
                    (unsigned long long)sval);
            return 1;
        }
        tsimd_pretok_cache_free(&sc);
        printf("  sentinel: the NUL-led span inserts and probes back\n");
    }
    uint8_t span[16];
    size_t mismatches = 0;
    for (int iter = 0; iter < 20000; iter++) {
        uint32_t n = (uint32_t)(rnd() % 15) + 1;
        for (uint32_t i = 0; i < n; i++)
            span[i] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];

        if (iter % 7 == 0) {
            for (uint32_t i = 0; i < 8 && i < n; i++) span[i] = 0;
            if (n < 9) n = 9;
            span[8] = (uint8_t)pool[rnd() % (sizeof(pool) - 1)];
        }
        uint64_t k0, k1, hash;
        tsimd_pcache_key_hash(span, n, &k0, &k1, &hash);
        if ((k0 | k1) == 0) continue;

        uint32_t ids[4];
        uint32_t n_ids = (uint32_t)(rnd() % 4) + 1;
        for (uint32_t i = 0; i < n_ids; i++)
            ids[i] = (uint32_t)(rnd() % 100000);

        uint64_t rval = 0, rext = 0;
        size_t rslot = 0;
        if (tsimd_pcache_probe_merged(&ref, k0, k1, hash, &rval, &rext,
                                      &rslot) != 0)
            continue;
        tsimd_pcache_insert_keyed(&ref, k0, k1, hash, n, ids, n_ids);

        uint64_t val = 0, ext = 0;
        size_t slot = 0;
        if (tsimd_pcache_probe_merged(&at, k0, k1, hash, &val, &ext,
                                      &slot) == 0)
            tsimd_pcache_insert_keyed_at(&at, slot, k0, k1, hash, n, ids,
                                         n_ids);
    }

    size_t cap = ref.mask + 1;
    if (memcmp(ref.entries, at.entries, cap * sizeof(tsimd_pcache_entry_t))
        != 0) {
        for (size_t i = 0; i < cap; i++) {
            tsimd_pcache_entry_t *r = &ref.entries[i];
            tsimd_pcache_entry_t *a = &at.entries[i];
            if (r->key0 != a->key0 || r->key1 != a->key1 ||
                r->val != a->val || r->ext != a->ext) {
                fprintf(stderr,
                        "FAIL: entry %zu differs: ref(%016llx,%016llx,"
                        "%016llx,%016llx) at(%016llx,%016llx,%016llx,"
                        "%016llx)\n",
                        i, (unsigned long long)r->key0,
                        (unsigned long long)r->key1,
                        (unsigned long long)r->val,
                        (unsigned long long)r->ext,
                        (unsigned long long)a->key0,
                        (unsigned long long)a->key1,
                        (unsigned long long)a->val,
                        (unsigned long long)a->ext);
                mismatches++;
                if (mismatches > 4) break;
            }
        }
        fprintf(stderr, "FAIL: %zu table mismatches (count %zu vs %zu)\n",
                mismatches, ref.count, at.count);
        return 1;
    }
    if (ref.count != at.count) {
        fprintf(stderr, "FAIL: count %zu vs %zu\n", ref.count, at.count);
        return 1;
    }
    printf("  insert-at-slot == walking insert: %zu entries identical\n",
           at.count);
    tsimd_pretok_cache_free(&ref);
    tsimd_pretok_cache_free(&at);
    return 0;
}
