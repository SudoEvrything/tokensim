#include "bpe.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NKEYS 40
#define ROUNDS 300
#define PASSES 3
#define NPAIR_LIMIT 12

static tsimd_pretok_cache_t cache;
static atomic_int writers_done;
static atomic_int bad_hits;
static pthread_barrier_t bar;

static uint8_t sel_bytes[NKEYS][4];
static uint64_t sel_k0[NKEYS], sel_k1[NKEYS], sel_hash[NKEYS];
static size_t sel_home[NKEYS];
static int sel_n;

static void key_bytes(int k, uint8_t *b) {
    memset(b, 0, 4);
    b[0] = 0x41;
    b[1] = 0x61;
    b[2] = 0x42;
    b[3] = (uint8_t)k;
}

static void payload_of(int k, uint32_t *ids) {
    ids[0] = (uint32_t)(0x100 + k);
    ids[1] = (uint32_t)(0x200 + k);
}

static void select_keys(void) {
    for (int k = 0; k < 256 && sel_n < NKEYS; k++) {
        uint8_t b[4];
        uint64_t k0, k1, hash;
        key_bytes(k, b);
        tsimd_pcache_key_hash(b, 4, &k0, &k1, &hash);
        size_t pair = (((size_t)hash << 1) & cache.pair_mask) >> 1;
        if (pair >= NPAIR_LIMIT) continue;
        memcpy(sel_bytes[sel_n], b, 4);
        sel_k0[sel_n] = k0;
        sel_k1[sel_n] = k1;
        sel_hash[sel_n] = hash;
        sel_home[sel_n] = ((size_t)hash << 1) & cache.pair_mask;
        sel_n++;
    }
    if (sel_n < 2) {
        fprintf(stderr, "key selection degenerate: %d\n", sel_n);
        exit(2);
    }
}

static int check_hit(int k, const uint32_t *out, uint32_t n) {
    uint32_t want[2];
    payload_of(k, want);
    return n == 2 && out[0] == want[0] && out[1] == want[1];
}

static void note_bad(int k, uint32_t n, const uint32_t *out, const char *who) {
    int total = atomic_fetch_add(&bad_hits, 1);
    if (total < 5)
        fprintf(stderr, "BAD HIT %s key=%d n=%u ids=%u,%u\n", who, k, n,
                n > 0 ? out[0] : 0, n > 1 ? out[1] : 0);
}

static void insert_at_variant(int which, int k, const uint32_t *ids) {
    if (which == 0)
        tsimd_pcache_insert_at_keyed(&cache, sel_home[k], sel_k0[k],
                                     sel_k1[k], sel_hash[k], ids, 2);
    else
        tsimd_pcache_insert_keyed_at(&cache, sel_home[k] + 1, sel_k0[k],
                                     sel_k1[k], sel_hash[k], 4, ids, 2);
}

static void *writer(void *arg) {
    int which = (int)(intptr_t)arg;
    uint32_t ids[2];
    for (int r = 0; r < ROUNDS; r++) {
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
        for (int p = 0; p < PASSES; p++) {
            for (int k = (which * 7) % sel_n; k < sel_n; k++) {
                payload_of(k, ids);
                insert_at_variant(which, k, ids);
            }
        }
        atomic_fetch_add(&writers_done, 1);
    }
    return NULL;
}

static void *reader(void *arg) {
    (void)arg;
    uint32_t out[8];
    for (int r = 0; r < ROUNDS; r++) {
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
        int done_last = 0;
        while (!done_last) {
            for (int k = 0; k < sel_n; k++) {
                uint32_t n = 0;
                tsimd_pcache_lookup_into(&cache, sel_bytes[k], 4, out, &n);
                if (n != 0 && !check_hit(k, out, n))
                    note_bad(k, n, out, "lookup");
            }
            done_last = atomic_load_explicit(&writers_done,
                                             memory_order_acquire) >=
                        (r + 1) * 2;
        }
    }
    return NULL;
}

int main(void) {
    pthread_t th[3];
    uint32_t ids[2], out[8];
    if (tsimd_pretok_cache_init(&cache, 1024) != 0) {
        fprintf(stderr, "cache init failed\n");
        return 2;
    }
    cache.mask = 127;
    cache.pair_mask = 126;
    cache.max_count = 96;
    select_keys();
    printf("selected %d keys in %d pairs\n", sel_n, NPAIR_LIMIT);

    cache.shared = 0;
    for (int k = 0; k < sel_n; k++) {
        payload_of(k, ids);
        insert_at_variant(0, k, ids);
        insert_at_variant(1, k, ids);
    }
    for (int k = 0; k < sel_n; k++) {
        uint32_t n = 0;
        tsimd_pcache_lookup_into(&cache, sel_bytes[k], 4, out, &n);
        if (!check_hit(k, out, n)) {
            fprintf(stderr, "FAIL single-thread key=%d n=%u\n", k, n);
            return 1;
        }
    }
    payload_of(0, ids);
    insert_at_variant(0, 0, ids);
    insert_at_variant(1, 0, ids);
    for (size_t i = 0; i <= cache.mask; i++) {
        cache.entries[i].key0 = 0;
        cache.entries[i].key1 = 0;
        cache.entries[i].val = 0;
        cache.entries[i].ext = 0;
    }
    cache.count = 0;

    cache.shared = 1;
    if (pthread_barrier_init(&bar, NULL, 4) != 0) {
        fprintf(stderr, "barrier init failed\n");
        return 2;
    }
    atomic_store(&writers_done, 0);
    atomic_store(&bad_hits, 0);
    if (pthread_create(&th[0], NULL, writer, (void *)(intptr_t)0) != 0 ||
        pthread_create(&th[1], NULL, writer, (void *)(intptr_t)1) != 0 ||
        pthread_create(&th[2], NULL, reader, NULL) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        return 2;
    }
    for (int r = 0; r < ROUNDS; r++) {
        pthread_barrier_wait(&bar);
        for (size_t i = 0; i <= cache.mask; i++) {
            cache.entries[i].key0 = 0;
            cache.entries[i].key1 = 0;
            cache.entries[i].val = 0;
            cache.entries[i].ext = 0;
        }
        cache.count = 0;
        pthread_barrier_wait(&bar);
    }
    for (int i = 0; i < 3; i++)
        pthread_join(th[i], NULL);
    int lost = 0;
    for (int k = 0; k < sel_n; k++) {
        uint32_t n = 0;
        tsimd_pcache_lookup_into(&cache, sel_bytes[k], 4, out, &n);
        if (n == 0 || !check_hit(k, out, n)) {
            if (lost < 5)
                fprintf(stderr, "LOST key=%d n=%u\n", k, n);
            lost++;
        }
    }
    tsimd_pretok_cache_free(&cache);
    int bad = atomic_load(&bad_hits);
    printf("pcache_race: bad_hits=%d lost=%d\n", bad, lost);
    if (bad || lost) {
        fprintf(stderr, "pcache_race: FAILED\n");
        return 1;
    }
    fprintf(stderr, "pcache_race: OK\n");
    return 0;
}
