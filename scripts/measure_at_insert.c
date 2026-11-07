#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (512 * 1024)

static tsimd_pretok_cache_t cache;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void key_of(uint32_t k, uint8_t *b) {
    memset(b, 0, 6);
    b[0] = (uint8_t)k;
    b[1] = (uint8_t)(k >> 8);
    b[2] = (uint8_t)(k >> 16);
    b[3] = (uint8_t)(k >> 24);
    b[4] = 0x5A;
}

static void old_body(tsimd_pretok_cache_t *c, size_t slot, uint64_t k0,
                     uint64_t k1, const uint32_t *ids) {
    if (c->count >= c->max_count) return;
    uint64_t val = 2ull | ((uint64_t)ids[0] << 3) | ((uint64_t)ids[1] << 32);
    uint64_t ext = 0;
    tsimd_pcache_entry_t *e = &c->entries[slot];
    e->val = val;
    e->ext = ext;
    e->key1 = k1;
    e->key0 = k0;
    __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
}

static void clear_entries(void) {
    for (size_t i = 0; i <= cache.mask; i++) {
        cache.entries[i].key0 = 0;
        cache.entries[i].key1 = 0;
        cache.entries[i].val = 0;
        cache.entries[i].ext = 0;
    }
    cache.count = 0;
}

int main(void) {
    static uint32_t ids[N][2];
    static size_t slot[N];
    static uint64_t k0s[N], k1s[N];
    if (tsimd_pretok_cache_init(&cache, 1 << 20) != 0) return 2;
    for (uint32_t k = 0; k < N; k++) {
        uint8_t b[6];
        uint64_t hash;
        key_of(k, b);
        tsimd_pcache_key_hash(b, 6, &k0s[k], &k1s[k], &hash);
        slot[k] = ((size_t)hash << 1) & cache.pair_mask;
        ids[k][0] = 0x100 + k;
        ids[k][1] = 0x200 + k;
    }
    double sum_old = 0, sum_new = 0;
    for (int rep = 0; rep < 6; rep++) {
        uint64_t t0, t1;
        clear_entries();
        t0 = now_ns();
        for (uint32_t k = 0; k < N; k++)
            old_body(&cache, slot[k], k0s[k], k1s[k], ids[k]);
        t1 = now_ns();
        sum_old += (double)(t1 - t0);
        clear_entries();
        t0 = now_ns();
        for (uint32_t k = 0; k < N; k++)
            tsimd_pcache_insert_at_keyed(&cache, slot[k], k0s[k], k1s[k], 0,
                                         ids[k], 2);
        t1 = now_ns();
        sum_new += (double)(t1 - t0);
    }
    printf("at-insert ns/op: old-plain=%.2f claim=%.2f (n=%d, 6 reps)\n",
           sum_old / 6.0 / N, sum_new / 6.0 / N, N);
    tsimd_pretok_cache_free(&cache);
    return 0;
}
