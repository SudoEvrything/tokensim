#ifndef TSIMD_BPE_H
#define TSIMD_BPE_H

#include <string.h>
#if defined(__x86_64__)
#include <emmintrin.h>
#endif

#if !defined(__cplusplus) && \
    (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include "vocabulary.h"

typedef struct {
    uint32_t *ids;
    uint32_t *next;
    uint32_t *prev;
    uint32_t capacity;
    uint32_t *out;
    size_t out_count;
    size_t out_capacity;

    uint64_t *heap;
    size_t heap_count, heap_cap;
    uint32_t *lin_ranks;
    uint32_t *lin_ids;
    uint32_t lin_cap;
    uint64_t *cand_state;
    uint32_t cand_cap;

    uint32_t *bq_head;
    uint32_t bq_nbuckets;
    uint64_t *bq_l0;
    uint64_t *bq_l1;
    uint64_t *bq_l2;
    uint32_t bq_l0_words;
    uint32_t bq_l1_words;
    uint32_t bq_l2_words;
    uint64_t *bq_entries;
    uint32_t *bq_rank;
    uint16_t *bq_gmask;
    uint32_t bq_count, bq_cap;
    uint32_t *bq_touched;
    uint32_t bq_touched_count, bq_touched_cap;
    int bq_gp;
    int bq_goverflow;
    uint32_t *bq_tail;
    void *uni_st;
    size_t uni_st_cap;
    void *uni_mcache[2][16];
    size_t uni_mcap[2][16];
    uint16_t *uni_ccache[2][16];
    size_t uni_cccap[2][16];
    uint32_t *uni_seg;
    size_t uni_seg_cap;
    int out_foreign;
    uint8_t *nfc_buf;
    size_t nfc_cap;
} tsimd_bpe_workspace_t;

int  tsimd_bpe_workspace_init(tsimd_bpe_workspace_t *ws, size_t max_token_len);

void tsimd_bpe_workspace_free(tsimd_bpe_workspace_t *ws);
int  tsimd_ws_out_ensure(tsimd_bpe_workspace_t *ws, size_t need);

void tsimd_bpe_set_heap_linear(uint32_t limit);
uint32_t tsimd_bpe_heap_linear(void);

void tsimd_bpe_set_pop_probe(uint32_t on);
uint32_t tsimd_bpe_pop_probe(void);

void tsimd_bpe_set_pop_scan(uint32_t mode);
uint32_t tsimd_bpe_pop_scan(void);

void tsimd_bpe_set_queue(uint32_t mode);
uint32_t tsimd_bpe_queue(void);

void tsimd_bpe_set_bqgp(uint32_t mode);
uint32_t tsimd_bpe_bqgp(void);
uint32_t tsimd_bpe_bucket_runs(void);

int tsimd_bpe_encode(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                    const uint8_t *bytes, uint32_t len);

int tsimd_bpe_encode_spm(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                         const uint8_t *bytes, uint32_t len);

int tsimd_vocab_spm_seam_ok(const tsimd_vocab_t *v);
int tsimd_vocab_spm_cross_ready(const tsimd_vocab_t *v);
size_t tsimd_spm_word_target(const tsimd_vocab_t *v, size_t block_target);

int tsimd_bpe_encode_spm_blocked(const tsimd_vocab_t *v,
                                 tsimd_bpe_workspace_t *ws,
                                 const uint8_t *bytes, uint32_t len,
                                 size_t target);

int tsimd_bpe_encode_spm_blocked_par(const tsimd_vocab_t *v,
                                     tsimd_bpe_workspace_t *ws,
                                     const uint8_t *bytes, uint32_t len,
                                     size_t target, int threads);

size_t tsimd_spm_seam_find(const tsimd_vocab_t *v, const uint8_t *bytes,
                           size_t len, size_t want, size_t min_pos);

int tsimd_unigram_encode(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                         const uint8_t *bytes, uint32_t len);

int tsimd_unigram_encode_ex(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                            const uint8_t *bytes, uint32_t len,
                            float offset, float *out_score,
                            float *min_margin);

#define TSIMD_PCACHE_MAX_IDS 4
#define TSIMD_PCACHE_MAX_SPAN 15
#define TSIMD_PCACHE_ID0_BITS 29

typedef struct {
    uint64_t key0;
    uint64_t key1;
    uint64_t val;
    uint64_t ext;
} tsimd_pcache_entry_t;

typedef struct {
    uint64_t key0;
    uint64_t key1;
    uint64_t val;
    uint64_t ext;
    uint64_t nid;
} tsimd_pcache_entry_old_t;

enum { TSIMD_PLONG_MAX_IDS = 32767 };
typedef struct {
    uint32_t seq;
    uint64_t hash;
    uint32_t lensz;
    uint32_t off;
    uint32_t ids[4];

} tsimd_plong_entry_t;

typedef struct {
    tsimd_pcache_entry_t *entries;
    size_t mask;
    size_t pair_mask;

    size_t count;

    tsimd_plong_entry_t *ltab;
    tsimd_plong_entry_t *ltab_old[8];
    uint32_t ltab_nold;
    uint8_t *larena;
    uint8_t *larena_old[8];
    uint32_t larena_nold;
    uint32_t *lids_old[8];
    uint32_t lids_nold;
    size_t lused, lcap;
    uint32_t *lids;
    size_t lids_used, lids_cap;
    uint32_t *lids_out;
    size_t lmask;
    size_t lcount;
    uint32_t lmin;
    uint32_t lbits;
    _Atomic int shared;
    int shared_plong;
    int lensure_busy;

    size_t max_count;
    size_t inserts;
    size_t hits;
    size_t misses;
    size_t map_bytes;
    uint32_t scratch[TSIMD_PCACHE_MAX_IDS];
} tsimd_pretok_cache_t;

int tsimd_bpe_encode_spm_cached(const tsimd_vocab_t *v,
                                tsimd_pretok_cache_t *c,
                               tsimd_pretok_cache_t *shared,
                                tsimd_bpe_workspace_t *ws,
                                const uint8_t *bytes, uint32_t len);
int tsimd_bpe_encode_spm_blocked_cached(const tsimd_vocab_t *v,
                                        tsimd_pretok_cache_t *c,
                                     tsimd_pretok_cache_t *shared,
                                        tsimd_bpe_workspace_t *ws,
                                        const uint8_t *bytes, uint32_t len,
                                        size_t target);

void tsimd_unigram_par_stats(size_t *windows);

int tsimd_unigram_encode_par(const tsimd_vocab_t *v,
                             tsimd_bpe_workspace_t *ws,
                             const uint8_t *bytes, uint32_t len,
                             int threads);
int tsimd_bpe_encode_spm_blocked_par_cached(const tsimd_vocab_t *v,
                                          tsimd_pretok_cache_t *cache,
                                          tsimd_pretok_cache_t *const *wcaches,
                                          tsimd_bpe_workspace_t *ws,
                                          const uint8_t *bytes, uint32_t len,
                                          size_t target, int threads);

int tsimd_plong_probe(const tsimd_pretok_cache_t *c,
                      const uint8_t *bytes, size_t len,
                      uint32_t *ids, uint32_t *n);

const uint32_t *tsimd_plong_probe_ex(const tsimd_pretok_cache_t *c,
                                     const uint8_t *bytes, size_t len,
                                     uint32_t *n);
void tsimd_plong_insert(tsimd_pretok_cache_t *c,
                        const uint8_t *bytes, size_t len,
                        const uint32_t *ids, uint32_t n);
void tsimd_plong_stats(size_t *h, size_t *m, size_t *i);
void tsimd_plong_reserve(tsimd_pretok_cache_t *c, size_t entries);
unsigned tsimd_pcache_lbits_of(const tsimd_pretok_cache_t *c);
size_t tsimd_pcache_lcount_of(const tsimd_pretok_cache_t *c);

static const uint64_t tsimd_pack_masks[16][2] = {
    {0x0000000000000000ull, 0x0000000000000000ull},
    {0x00000000000000FFull, 0x0000000000000000ull},
    {0x000000000000FFFFull, 0x0000000000000000ull},
    {0x0000000000FFFFFFull, 0x0000000000000000ull},
    {0x00000000FFFFFFFFull, 0x0000000000000000ull},
    {0x000000FFFFFFFFFFull, 0x0000000000000000ull},
    {0x0000FFFFFFFFFFFFull, 0x0000000000000000ull},
    {0x00FFFFFFFFFFFFFFull, 0x0000000000000000ull},
    {0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull},
    {0xFFFFFFFFFFFFFFFFull, 0x00000000000000FFull},
    {0xFFFFFFFFFFFFFFFFull, 0x000000000000FFFFull},
    {0xFFFFFFFFFFFFFFFFull, 0x0000000000FFFFFFull},
    {0xFFFFFFFFFFFFFFFFull, 0x00000000FFFFFFFFull},
    {0xFFFFFFFFFFFFFFFFull, 0x000000FFFFFFFFFFull},
    {0xFFFFFFFFFFFFFFFFull, 0x0000FFFFFFFFFFFFull},
    {0xFFFFFFFFFFFFFFFFull, 0x00FFFFFFFFFFFFFFull},
};

void tsimd_pcache_key_hash(const uint8_t *bytes, uint32_t len,
                           uint64_t *k0, uint64_t *k1, uint64_t *hash);

static inline int tsimd_pcache_stats_on(void) {
    static atomic_int state;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_PCACHE_STATS");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static inline int tsimd_hash_use_crc32(void) {
#if defined(__x86_64__) || defined(__i386__)
    static atomic_int state;
    int v = atomic_load(&state);
    if (v == 0) {

        const char *e = getenv("TSIMD_HASH_SCALAR");
        int on = !(e && e[0] == '1');
        if (on) {
            uint32_t a = 0, b = 0, c = 0, d = 0;
            if (!__get_cpuid_count(1, 0, &a, &b, &c, &d) || !(c & (1u << 20)))
                on = 0;
        }
        v = on ? 1 : 2;
        atomic_store(&state, v);
    }
    return v == 1;
#else
    return 0;
#endif
}

uint64_t tsimd_hash_words_crc32(uint64_t a, uint64_t b);

static inline uint64_t tsimd_pcache_hash_words_arm(uint64_t a, uint64_t b,
                                                   int crc) {
    if (crc) {
#if defined(__SSE4_2__)
        uint64_t h = (uint64_t)__builtin_ia32_crc32di(0, a);
        h = (uint64_t)__builtin_ia32_crc32di((uint32_t)h, b);
        h ^= h >> 32;
        return h ? h : 1;
#else
        return tsimd_hash_words_crc32(a, b);
#endif
    }
    uint64_t h = a * 0x9E3779B97F4A7C15ull ^ b;
    h ^= h >> 32;
    return h ? h : 1;
}

static inline uint64_t tsimd_pcache_hash_words_64(uint64_t a, uint64_t b) {
    return tsimd_pcache_hash_words_arm(a, b, tsimd_hash_use_crc32());
}

static inline void tsimd_pcache_key_inline_arm(const uint8_t *bytes,
                                               size_t len,
                                               const uint8_t *buf_end,
                                               uint64_t *k0, uint64_t *k1,
                                               uint64_t *hash, int crc) {
    if (bytes + 16 > buf_end) {
        tsimd_pcache_key_hash(bytes, (uint32_t)len, k0, k1, hash);
        return;
    }

    const uint64_t (*pm)[2] = tsimd_pack_masks;
    __asm__("" : "+r"(pm));
    unsigned m = len < 15 ? (unsigned)len : 15;
    uint64_t a, b;
    memcpy(&a, bytes, 8);
    memcpy(&b, bytes + 8, 8);
    a &= pm[m][0];
    b = (b & pm[m][1]) | ((uint64_t)m << 56);
    *k0 = a;
    *k1 = b;
    *hash = tsimd_pcache_hash_words_arm(a, b, crc);
}

static inline void tsimd_pcache_key_inline(const uint8_t *bytes, size_t len,
                                           const uint8_t *buf_end,
                                           uint64_t *k0, uint64_t *k1,
                                           uint64_t *hash) {
    tsimd_pcache_key_inline_arm(bytes, len, buf_end, k0, k1, hash,
                                tsimd_hash_use_crc32());
}

typedef struct {

    uint64_t k0, k1;
    const uint8_t *ptr;
    uint64_t meta;
} __attribute__((aligned(16))) tsimd_pcache_span_t;

static inline uint32_t tsimd_pcache_span_len(const tsimd_pcache_span_t *e) {
    uint32_t t = (uint32_t)(e->k1 >> 56);
    return t ? t : (uint32_t)e->meta;
}

static inline void tsimd_pcache_prefetch_l2(const tsimd_pretok_cache_t *c,
                                            uint64_t hash) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
#if defined(__x86_64__) || defined(__i386__)
    _mm_prefetch((const char *)&c->entries[set], _MM_HINT_T1);
#else
    __builtin_prefetch(&c->entries[set], 0, 2);
#endif
}

static inline void tsimd_pcache_prefetch_l1(const tsimd_pretok_cache_t *c,
                                            uint64_t hash) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
#if defined(__x86_64__) || defined(__i386__)
    _mm_prefetch((const char *)&c->entries[set], _MM_HINT_T0);
#else
    __builtin_prefetch(&c->entries[set], 0, 3);
#endif
}

int tsimd_o200k_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                                tsimd_pcache_span_t *ents, size_t cap,
                                tsimd_pretok_cache_t *pc, int twophase);

int tsimd_cl100k_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                                 tsimd_pcache_span_t *ents, size_t cap,
                                 tsimd_pretok_cache_t *pc, int twophase);

int tsimd_deepseek_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                                   tsimd_pcache_span_t *ents, size_t cap,
                                   tsimd_pretok_cache_t *pc, int twophase);

int tsimd_gpt2_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                               tsimd_pcache_span_t *ents, size_t cap,
                               tsimd_pretok_cache_t *pc, int twophase);

int tsimd_nemo_mask_fill_keyed(const uint8_t *d, size_t len, void *state,
                               tsimd_pcache_span_t *ents, size_t cap,
                               tsimd_pretok_cache_t *pc, int twophase);

static inline uint32_t *tsimd_pcache_probe_inline(
    tsimd_pretok_cache_t *c, uint64_t k0, uint64_t k1, uint64_t hash,
    uint32_t *out, uint32_t *n_ids) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
    for (;;) {
        tsimd_pcache_entry_t *e0 = &c->entries[set];
        tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
        uint64_t a0 = __atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE);
        uint64_t a1 = __atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE);
        uint64_t av = __atomic_load_n(&e0->val, __ATOMIC_ACQUIRE);
        uint64_t ax = __atomic_load_n(&e0->ext, __ATOMIC_ACQUIRE);
        uint64_t b0 = __atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE);
        uint64_t b1 = __atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE);
        uint64_t bv = __atomic_load_n(&e1->val, __ATOMIC_ACQUIRE);
        uint64_t bx = __atomic_load_n(&e1->ext, __ATOMIC_ACQUIRE);
        uint64_t m0 = (a0 == k0 && a1 == k1) ? ~0ull : 0;
        uint64_t m1 = (b0 == k0 && b1 == k1) ? ~0ull : 0;
        if (m0 | m1) {
            uint64_t val = (av & m0) | (bv & m1);
            uint64_t ext = (ax & m0) | (bx & m1);
            out[0] = (uint32_t)((val >> 3) & 0x1FFFFFFF);
            out[1] = (uint32_t)(val >> 32);
            out[2] = (uint32_t)(ext & 0xFFFFFFFFu);
            out[3] = (uint32_t)(ext >> 32);
            *n_ids = (uint32_t)(val & 7);
            if (tsimd_pcache_stats_on()) c->hits++;
            return out;
        }
        if (((a0 | a1) == 0) || ((b0 | b1) == 0)) {
            if (tsimd_pcache_stats_on()) c->misses++;
            return NULL;
        }
        set = (set + 2) & c->mask;
    }
}

static inline int tsimd_pcache_probe_home(tsimd_pretok_cache_t *c,
                                          uint64_t k0, uint64_t k1,
                                          uint64_t hash,
                                          uint64_t *val, uint64_t *ext) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
    tsimd_pcache_entry_t *e0 = &c->entries[set];
    tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
#ifdef __SSE2__
    __m128i kk = _mm_set_epi64x((long long)k1, (long long)k0);
    __m128i a = _mm_load_si128((const __m128i *)(const void *)&e0->key0);
    __m128i b = _mm_load_si128((const __m128i *)(const void *)&e1->key0);
    uint64_t m0 = (_mm_movemask_epi8(_mm_cmpeq_epi8(a, kk)) == 0xFFFF)
                      ? ~0ull
                      : 0;
    uint64_t m1 = (_mm_movemask_epi8(_mm_cmpeq_epi8(b, kk)) == 0xFFFF)
                      ? ~0ull
                      : 0;
#else
    uint64_t a0 = __atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE);
    uint64_t a1 = __atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE);
    uint64_t b0 = __atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE);
    uint64_t b1 = __atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE);
    uint64_t m0 = (a0 == k0 && a1 == k1) ? ~0ull : 0;
    uint64_t m1 = (b0 == k0 && b1 == k1) ? ~0ull : 0;
#endif
    if (m0 | m1) {
        uint64_t av = __atomic_load_n(&e0->val, __ATOMIC_ACQUIRE);
        uint64_t ax = __atomic_load_n(&e0->ext, __ATOMIC_ACQUIRE);
        uint64_t bv = __atomic_load_n(&e1->val, __ATOMIC_ACQUIRE);
        uint64_t bx = __atomic_load_n(&e1->ext, __ATOMIC_ACQUIRE);
        *val = (av & m0) | (bv & m1);
        *ext = (ax & m0) | (bx & m1);
        c->hits++;
        return 1;
    }
    return 0;
}

int  tsimd_pretok_cache_init(tsimd_pretok_cache_t *c, size_t capacity_entries);

int  tsimd_pcache_bits_env(void);

int  tsimd_pool_cache_bits(int width, size_t total_len);
void tsimd_pretok_cache_free(tsimd_pretok_cache_t *c);

void tsimd_pcache_merge(tsimd_pretok_cache_t *dst, tsimd_pretok_cache_t *src);

void tsimd_pcache_key_hash(const uint8_t *bytes, uint32_t len,
                           uint64_t *k0, uint64_t *k1, uint64_t *hash);

void tsimd_pcache_key_hash_bounded(const uint8_t *bytes, size_t len,
                                   const uint8_t *buf_end,
                                   uint64_t *k0, uint64_t *k1,
                                   uint64_t *hash);
uint32_t *tsimd_pcache_lookup(tsimd_pretok_cache_t *c,
                             const uint8_t *bytes, uint32_t len,
                             uint32_t *n_ids);

uint32_t *tsimd_pcache_lookup_into(tsimd_pretok_cache_t *c,
                                   const uint8_t *bytes, uint32_t len,
                                   uint32_t *out, uint32_t *n_ids);

uint32_t *tsimd_pcache_lookup_into_bounded(tsimd_pretok_cache_t *c,
                                           const uint8_t *bytes,
                                           uint32_t len,
                                           const uint8_t *buf_end,
                                           uint32_t *out,
                                           uint32_t *n_ids);

uint32_t *tsimd_pcache_lookup_keyed(tsimd_pretok_cache_t *c, uint64_t k0,
                                    uint64_t k1, uint64_t hash,
                                    uint32_t *out, uint32_t *n_ids);

void tsimd_pcache_insert_keyed_at(tsimd_pretok_cache_t *c, size_t slot,
                                  uint64_t k0, uint64_t k1, uint64_t hash,
                                  uint32_t len, const uint32_t *ids,
                                  uint32_t n_ids);
void tsimd_pcache_insert_at_keyed(tsimd_pretok_cache_t *c, size_t slot,
                                  uint64_t k0, uint64_t k1, uint64_t hash,
                                  const uint32_t *ids, uint32_t n_ids);
void tsimd_pcache_insert_keyed(tsimd_pretok_cache_t *c, uint64_t k0,
                               uint64_t k1, uint64_t hash, uint32_t len,
                               const uint32_t *ids, uint32_t n_ids);
void tsimd_pcache_insert(tsimd_pretok_cache_t *c, const uint8_t *bytes,
                        uint32_t len, const uint32_t *ids, uint32_t n_ids);

static inline int tsimd_pcache_probe_merged_ns(tsimd_pretok_cache_t *c,
                                               uint64_t k0, uint64_t k1,
                                               uint64_t hash,
                                               uint64_t *val, uint64_t *ext,
                                               size_t *slot, uint64_t *hits) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
    for (;;) {
        tsimd_pcache_entry_t *e0 = &c->entries[set];
        tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
        uint64_t a0 = __atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE);
        uint64_t a1 = __atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE);
        uint64_t av = __atomic_load_n(&e0->val, __ATOMIC_ACQUIRE);
        uint64_t ax = __atomic_load_n(&e0->ext, __ATOMIC_ACQUIRE);
        uint64_t b0 = __atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE);
        uint64_t b1 = __atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE);
        uint64_t bv = __atomic_load_n(&e1->val, __ATOMIC_ACQUIRE);
        uint64_t bx = __atomic_load_n(&e1->ext, __ATOMIC_ACQUIRE);

        if (a0 == k0 && a1 == k1) {
            *val = av;
            *ext = ax;
            *hits += 1;
            return 1;
        }
        if (b0 == k0 && b1 == k1) {
            *val = bv;
            *ext = bx;
            *hits += 1;
            return 1;
        }
        if ((a0 | a1) == 0) { *slot = set; return 0; }
        if ((b0 | b1) == 0) { *slot = set + 1; return 0; }
        set = (set + 2) & c->mask;
    }
}

static inline int tsimd_pcache_probe_merged(tsimd_pretok_cache_t *c,
                                            uint64_t k0, uint64_t k1,
                                            uint64_t hash,
                                            uint64_t *val, uint64_t *ext,
                                            size_t *slot) {
    return tsimd_pcache_probe_merged_ns(c, k0, k1, hash, val, ext, slot,
                                        &c->hits);
}

static inline int tsimd_pcache_probe_serve(tsimd_pretok_cache_t *c,
                                           uint64_t k0, uint64_t k1,
                                           uint64_t hash, uint64_t *val,
                                           uint64_t *ext, size_t *slot) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
#ifdef __SSE2__
    __m128i kk = _mm_set_epi64x((long long)k1, (long long)k0);
    __m128i zero = _mm_setzero_si128();
    for (;;) {
        tsimd_pcache_entry_t *e0 = &c->entries[set];
        tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
        __m128i a = _mm_load_si128((const __m128i *)(const void *)&e0->key0);
        __m128i b = _mm_load_si128((const __m128i *)(const void *)&e1->key0);
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, kk)) == 0xFFFF) {
            *val = __atomic_load_n(&e0->val, __ATOMIC_ACQUIRE);
            *ext = __atomic_load_n(&e0->ext, __ATOMIC_ACQUIRE);
            return 1;
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(b, kk)) == 0xFFFF) {
            *val = __atomic_load_n(&e1->val, __ATOMIC_ACQUIRE);
            *ext = __atomic_load_n(&e1->ext, __ATOMIC_ACQUIRE);
            return 1;
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, zero)) == 0xFFFF) {
            *slot = set;
            return 0;
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(b, zero)) == 0xFFFF) {
            *slot = set + 1;
            return 0;
        }
        set = (set + 2) & c->mask;
    }
#else
    return tsimd_pcache_probe_merged_ns(c, k0, k1, hash, val, ext, slot,
                                        &c->hits);
#endif
}

static inline int tsimd_pcache_probe_walk(tsimd_pretok_cache_t *c,
                                          uint64_t k0, uint64_t k1,
                                          uint64_t hash,
                                          uint64_t *val, uint64_t *ext,
                                          size_t *slot) {
    size_t set = ((size_t)hash << 1) & c->pair_mask;
    for (;;) {
        tsimd_pcache_entry_t *e0 = &c->entries[set];
        tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
        uint64_t a0 = __atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE);
        uint64_t a1 = __atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE);
        uint64_t b0 = __atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE);
        uint64_t b1 = __atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE);
        if (a0 == k0 && a1 == k1) {
            *val = __atomic_load_n(&e0->val, __ATOMIC_ACQUIRE);
            *ext = __atomic_load_n(&e0->ext, __ATOMIC_ACQUIRE);
            return 1;
        }
        if (b0 == k0 && b1 == k1) {
            *val = __atomic_load_n(&e1->val, __ATOMIC_ACQUIRE);
            *ext = __atomic_load_n(&e1->ext, __ATOMIC_ACQUIRE);
            return 1;
        }
        if ((a0 | a1) == 0) { *slot = set; return 0; }
        if ((b0 | b1) == 0) { *slot = set + 1; return 0; }
        set = (set + 2) & c->mask;
    }
}

typedef struct {
    uint64_t k0, k1;
    uint64_t val;
    uint64_t ext;
} tsimd_pcache_frozen_entry_t;

typedef struct {
    tsimd_pcache_frozen_entry_t *entries;
    size_t mask;
} tsimd_pcache_frozen_t;

int  tsimd_pcache_frozen_build(tsimd_pcache_frozen_t *f,
                               const tsimd_vocab_t *v);
void tsimd_pcache_frozen_free(tsimd_pcache_frozen_t *f);

int  tsimd_pcache_frozen_probe(const tsimd_pcache_frozen_t *f,
                               uint64_t k0, uint64_t k1, uint64_t hash,
                               uint64_t *val_out);

int tsimd_pcache_seed_vocab(tsimd_pretok_cache_t *c,
                            const tsimd_vocab_t *v);
#endif
