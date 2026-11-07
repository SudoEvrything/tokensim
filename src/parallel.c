#define _GNU_SOURCE
#include "tokenizer.h"
#include "batch.h"
#include "internal.h"
#include "simd.h"
#include "allocator.h"

#include <emmintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

extern size_t tsimd_serve_hot_avx2(const void *ents, size_t pair_mask,
                                   size_t mask, const void *batch, size_t n,
                                   size_t j0, uint32_t *out, size_t *cur);
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void tsimd_pcache_upgrade_for(tsimd_tokenizer_t *t, size_t len);
void tsimd_memo_size_for(tsimd_tokenizer_t *t, size_t total_len);

int tsimd_tokenizer_encode_whole_chain_par(tsimd_tokenizer_t *t,
                                           const uint8_t *data, size_t len,
                                           int threads,
                                           tsimd_token_sequence_t *out);

int tsimd_tokenizer_encode_whole_chain_par_into(tsimd_tokenizer_t *t,
                                                const uint8_t *data,
                                                size_t len, int threads,
                                                uint32_t *buf, size_t cap,
                                                size_t *count);

#define TSIMD_PAR_MIN_LEN (1u << 18)

#define TSIMD_POOL_CACHE_MIB 1024
#define TSIMD_PAR_CACHE_BITS_MIN 12

#define TSIMD_PAR_CACHE_BITS_MAX 22
#define TSIMD_PAR_MIN_PIECE (1u << 16)
#define TSIMD_PAR_OVERSUBSCRIBE 16

#define TSIMD_POOL_MAX 64

typedef struct {
    uint32_t *ids;
    size_t count, cap;
    int foreign;
} idvec_t;

enum { JOB_TEXT = 0, JOB_SPECIAL, JOB_BPE, JOB_GATHER, JOB_NFC };

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t doc;
    int kind;
    uint32_t special_id;

    uint32_t *gdst;
    const void *gsrc;
    size_t gj0, gj1;
} job_t;

typedef struct {
    tsimd_pool_t *pool;
    tsimd_tokenizer_t *t;
    tsimd_bpe_workspace_t ws;
    tsimd_pretok_cache_t *cache;

    idvec_t dual_a, dual_b;

    tsimd_o200k_mstate_t mask_st;

    uint64_t ph_fill, ph_serve, ph_pieces;

    uint64_t ph_hits, ph_big, ph_long, ph_longhit, ph_bpe;
    int idx;
    int seeded;
    int shared_cache;
    int init_rc;

} worker_t;

struct tsimd_pool {
    tsimd_tokenizer_t *t;
    worker_t *workers;
    int width;
    int n_spawned;
    pthread_t *threads;

    job_t *jobs;
    idvec_t *outs;
    size_t n_jobs;
    atomic_size_t next;
    atomic_int err;
    unsigned generation;
    atomic_uint done;
    atomic_int ready;
    int active;

    size_t *weights;
    size_t *plo, *phi;
    size_t range_cap;
    atomic_uchar *claims;
    size_t claims_cap;
    int cache_bits;

    atomic_size_t steal_next;
    int par_mode;

    idvec_t *outs_cache;
    size_t outs_cache_n;
    int n_range;
    int shutdown;
    pthread_mutex_t mu;
    pthread_cond_t wake_cv;
    pthread_cond_t done_cv;
    pthread_cond_t ready_cv;
};

static int idvec_reserve(idvec_t *v, size_t want) {
    if (want <= v->cap) return 0;
    size_t cap = v->cap ? v->cap : 4096;
    while (cap < want) cap *= 2;
    uint32_t *grown;
    if (v->foreign) {
        grown = tsimd_xmalloc(cap * sizeof(uint32_t));
        if (!grown) return -1;
        if (v->count) memcpy(grown, v->ids, v->count * sizeof(uint32_t));
        v->foreign = 0;
    } else {
        grown = tsimd_xrealloc(v->ids, cap * sizeof(uint32_t));
        if (!grown) return -1;
    }
    v->ids = grown;
    v->cap = cap;
    if (getenv("TSIMD_IDBG") && cap * 4 >= 20480)
        fprintf(stderr, "IDBG reserve v=%p ids=%p bytes=%zu\n", (void*)v,
                (void*)grown, cap * 4);
    return 0;
}

static int idvec_push(idvec_t *v, uint32_t id) {
    if (v->count == v->cap) {
        size_t cap = v->cap ? v->cap * 2 : 4096;
        uint32_t *grown;
        if (v->foreign) {
            grown = tsimd_xmalloc(cap * sizeof(uint32_t));
            if (!grown) return -1;
            if (v->count)
                memcpy(grown, v->ids, v->count * sizeof(uint32_t));
            v->foreign = 0;
        } else {
            grown = tsimd_xrealloc(v->ids, cap * sizeof(uint32_t));
            if (!grown) return -1;
        }
        v->ids = grown;
        v->cap = cap;
    }
    v->ids[v->count++] = id;
    return 0;
}

static int idvec_push_n(idvec_t *v, const uint32_t *ids, size_t n) {
    if (v->count + n > v->cap) {
        size_t cap = v->cap ? v->cap : 4096;
        while (cap < v->count + n) cap *= 2;
        uint32_t *grown;
        if (v->foreign) {
            grown = tsimd_xmalloc(cap * sizeof(uint32_t));
            if (!grown) return -1;
            if (v->count)
                memcpy(grown, v->ids, v->count * sizeof(uint32_t));
            v->foreign = 0;
        } else {
            grown = tsimd_xrealloc(v->ids, cap * sizeof(uint32_t));
            if (!grown) return -1;
        }
        v->ids = grown;
        v->cap = cap;
    }
    if (n <= 4) {
        for (size_t k = 0; k < n; k++) v->ids[v->count + k] = ids[k];
    } else {
        memcpy(v->ids + v->count, ids, n * sizeof(uint32_t));
    }
    v->count += n;
    return 0;
}

typedef struct {
    worker_t *w;
    idvec_t *out;
    const uint8_t *data;
} pfuse_t;

static int fused_span_p(void *ud, size_t s, size_t e) {
    pfuse_t *f = ud;
    worker_t *w = f->w;
    const uint8_t *p = f->data + s;
    size_t n = e - s;
    if (n > UINT32_MAX) return -1;
    uint32_t n_ids = 0;
    uint32_t *cached = NULL;
    if (tsimd_ws_out_ensure(&w->ws, w->ws.out_count + TSIMD_PCACHE_MAX_IDS) ==
        0) {
        cached = tsimd_pcache_lookup_into(w->cache, p, (uint32_t)n,
                                          w->ws.out + w->ws.out_count,
                                          &n_ids);
    }
    if (cached) {
        if (n_ids > 2)
            return idvec_push_n(f->out, cached, n_ids);
        for (uint32_t k = 0; k < n_ids; k++)
            if (idvec_push(f->out, cached[k]) != 0) return -1;
        return 0;
    }
    size_t base = w->ws.out_count;
    if (tsimd_bpe_encode(&w->t->vocab, &w->ws, p, (uint32_t)n) != 0)
        return -1;
    if (w->ws.out_count - base > 2)
        idvec_push_n(f->out, w->ws.out + base, w->ws.out_count - base);
    else
        for (size_t k = base; k < w->ws.out_count; k++)
            if (idvec_push(f->out, w->ws.out[k]) != 0) return -1;
    tsimd_pcache_insert(w->cache, p, (uint32_t)n, w->ws.out + base,
                        (uint32_t)(w->ws.out_count - base));
    w->ws.out_count = base;
    return 0;
}

#define TSIMD_SERVE_BATCH 512
#define TSIMD_SERVE_D 12

#define TSIMD_FUSE_BATCH 16
#define TSIMD_FUSE_DUAL_MIN (1u << 18)

static int dual_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_DISABLE_DUAL");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static int fused_o200k_flush(worker_t *w, const uint8_t *data,
                             const uint8_t *end,
                             const size_t *starts, const size_t *ends,
                             uint32_t (*ids)[TSIMD_PCACHE_MAX_IDS],
                             uint32_t *nids, int *hit, int k,
                             idvec_t *out) {
    for (int j = 0; j < k; j++) {
        size_t n = ends[j] - starts[j];
        nids[j] = 0;
        hit[j] = tsimd_pcache_lookup_into_bounded(w->cache, data + starts[j],
                                                  (uint32_t)n, end, ids[j],
                                                  &nids[j]) != NULL;
        (void)0;
    }
    for (int j = 0; j < k; j++) {
        if (hit[j]) {
            if (nids[j] > 2) {
                if (idvec_push_n(out, ids[j], nids[j]) != 0) return -1;
            } else {
                for (uint32_t q = 0; q < nids[j]; q++)
                    if (idvec_push(out, ids[j][q]) != 0) return -1;
            }
            continue;
        }
        const uint8_t *p = data + starts[j];
        size_t n = ends[j] - starts[j];
        size_t base = w->ws.out_count;
        if (tsimd_bpe_encode(&w->t->vocab, &w->ws, p, (uint32_t)n) != 0)
            return -1;
        if (idvec_push_n(out, w->ws.out + base,
                         w->ws.out_count - base) != 0)
            return -1;
        tsimd_pcache_insert(w->cache, p, (uint32_t)n, w->ws.out + base,
                            (uint32_t)(w->ws.out_count - base));
        w->ws.out_count = base;
    }
    return 0;
}

static inline void serve_prefetch_l2(const tsimd_pretok_cache_t *c,
                                     uint64_t hash) {
    size_t set = ((size_t)hash & (c->mask >> 1)) << 1;
    _mm_prefetch((const char *)&c->entries[set], _MM_HINT_T1);
}

static inline void serve_prefetch_l1(const tsimd_pretok_cache_t *c,
                                     uint64_t hash) {
    size_t set = ((size_t)hash & (c->mask >> 1)) << 1;
    _mm_prefetch((const char *)&c->entries[set], _MM_HINT_T0);
}

static int no_long_cache(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_NO_LONG");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static int verify_long(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_VERIFY_LONG");
        atomic_store(&state, (e && e[0] == '1') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v == 1;
}

static atomic_int g_serve_env_mode;
static atomic_int g_serve_avx2_on;
static atomic_int g_serve_dump_flag;
static atomic_int g_qwen25_scalar_flag;

static int serve_env_per_call(void) {
    int m = atomic_load(&g_serve_env_mode);
    if (m == 0) {
        atomic_store(&g_serve_env_mode,
                     getenv("TSIMD_NO_SERVE_ENV_CACHE") ? 1 : 2);
        m = atomic_load(&g_serve_env_mode);
    }
    return m == 1;
}

void tsimd_serve_env_refresh(void) {
    if (serve_env_per_call()) return;
    const char *e = getenv("TSIMD_SERVE_AVX2");
    int avx2 = !(e && e[0] == '0') &&
               tsimd_simd_level() >= TSIMD_SIMD_AVX2;
    atomic_store(&g_serve_avx2_on, avx2 ? 1 : 2);
    atomic_store(&g_serve_dump_flag,
                 getenv("TSIMD_SERVE_DUMP") != NULL ? 1 : 2);
    const char *q = getenv("TSIMD_QWEN25_SCALAR");
    atomic_store(&g_qwen25_scalar_flag,
                 (q && q[0] == '1') ? 1 : 2);
}

int tsimd_serve_env_per_call(void) { return serve_env_per_call(); }

int tsimd_serve_qwen25_scalar_cached(void) {
    int v = atomic_load(&g_qwen25_scalar_flag);
    if (v == 0) {
        tsimd_serve_env_refresh();
        v = atomic_load(&g_qwen25_scalar_flag);
    }
    return v == 1;
}

static int serve_avx2_enabled(void) {
    if (serve_env_per_call()) {
        const char *e = getenv("TSIMD_SERVE_AVX2");
        if (e && e[0] == '0') return 0;
        return tsimd_simd_level() >= TSIMD_SIMD_AVX2;
    }
    int v = atomic_load(&g_serve_avx2_on);
    if (v == 0) {
        tsimd_serve_env_refresh();
        v = atomic_load(&g_serve_avx2_on);
    }
    return v == 1;
}

static atomic_size_t g_serve_avx2_pieces;
size_t tsimd_serve_avx2_pieces(void) {
    return atomic_load_explicit(&g_serve_avx2_pieces, memory_order_relaxed);
}

int tsimd_gpt2d_span_on(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_GPT2D_SPAN");
        atomic_store(&state, (e && e[0] == '0') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 1;
}

static int scalar_fill_enabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_SCALAR_FILL");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static int dual_cur_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_NO_DUALCUR");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static size_t serve_find_split(const uint8_t *d, size_t len, size_t from) {
    size_t lim = from + 256 < len ? from + 256 : len;
    for (size_t k = from; k + 1 < lim; k++) {
        if (d[k] != '\n') continue;
        uint8_t b = d[k + 1];
        if (b < 0x80 && b != ' ' && b != '\t' && b != '\r' && b != '\n' &&
            b != '\v' && b != '\f')
            return k + 1;
    }
    return 0;
}

static inline int serve_scalar_step(worker_t *w, const uint8_t *data,
                                    size_t len, tsimd_pcache_span_t *ents,
                                    tsimd_pretok_cache_t *c, size_t cap,
                                    size_t *n, size_t *pos) {
    size_t e = w->t->scheme == TSIMD_PRETOK_CL100K
                   ? tsimd_match_cl100k(data, len, *pos)
                   : w->t->scheme == TSIMD_PRETOK_DEEPSEEK
                         ? tsimd_match_deepseek(data, len, *pos)
                         : w->t->scheme == TSIMD_PRETOK_NEMO
                               ? tsimd_match_nemo(data, len, *pos)
                               : w->t->scheme == TSIMD_PRETOK_GPT2
                                     ? tsimd_match_gpt2(data, len, *pos)
                                     : tsimd_match_o200k(data, len, *pos);
    if (e <= *pos || e > len) return -1;
    if (*n < cap) {
        tsimd_pcache_span_t *en = &ents[*n];
        en->ptr = data + *pos;
        uint32_t el = (uint32_t)(e - *pos);
        if (el <= TSIMD_PCACHE_MAX_SPAN) {
            tsimd_pcache_key_inline(en->ptr, el, data + len,
                                    &en->k0, &en->k1, &en->meta);

            serve_prefetch_l2(c, en->meta);
        } else {
            en->k0 = 0;
            en->k1 = 1;

            en->meta = el;
        }
    }
    (*n)++;
    *pos = e;
    return 0;
}

static int serve_dump_on(void) {
    if (serve_env_per_call()) return getenv("TSIMD_SERVE_DUMP") != NULL;
    int v = atomic_load(&g_serve_dump_flag);
    if (v == 0) {
        tsimd_serve_env_refresh();
        v = atomic_load(&g_serve_dump_flag);
    }
    return v == 1;
}

static void dump_serve(size_t n, const tsimd_pcache_span_t *batch,
                       const uint8_t *data) {
    fprintf(stderr, "SERVE batch n=%zu:\n", n);
    for (size_t j = 0; j < n; j++) {
        fprintf(stderr, "  [%zu] ptr_off=%zu len=%u k0=%016llx\n", j,
                (size_t)(batch[j].ptr - data),
                tsimd_pcache_span_len(&batch[j]),
                (unsigned long long)batch[j].k0);
    }
}

static int phase_stats_on(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_PHASE_STATS");
        atomic_store(&state, (e && e[0] == '1') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v == 1;
}

static inline uint64_t ph_rdtsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc();
#else
    return 0;
#endif
}

static void phase_report(const char *where, uint64_t fill, uint64_t serve,
                         uint64_t pieces) {
    if (!phase_stats_on() || pieces == 0) return;
    double per = (double)(fill + serve) / (double)pieces;
    fprintf(stderr,
            "tsimd phases %s: %llu pieces, %.1f cyc/piece — fill %.1f "
            "(%.0f%%), serve %.1f (%.0f%%)\n",
            where, (unsigned long long)pieces, per,
            (double)fill / (double)pieces,
            100.0 * (double)fill / (double)(fill + serve),
            (double)serve / (double)pieces,
            100.0 * (double)serve / (double)(fill + serve));
}

static void phase_report_mix(const char *where, uint64_t pieces,
                             uint64_t hits, uint64_t big, uint64_t longp,
                             uint64_t longhit, uint64_t bpe) {
    if (!phase_stats_on() || pieces == 0) return;
    fprintf(stderr,
            "tsimd mix %s: %llu pieces — short-hit %llu (%.1f%%), "
            "big-hit %llu (%.2f%%), long %llu (%.2f%%, hit %llu), "
            "bpe %llu (%.2f%%)\n",
            where, (unsigned long long)pieces, (unsigned long long)hits,
            100.0 * (double)hits / (double)pieces, (unsigned long long)big,
            100.0 * (double)big / (double)pieces, (unsigned long long)longp,
            100.0 * (double)longp / (double)pieces,
            (unsigned long long)longhit, (unsigned long long)bpe,
            100.0 * (double)bpe / (double)pieces);
}

static void phase_report_pool(const char *where, tsimd_pool_t *p) {
    if (!phase_stats_on() || !p || !p->workers) return;
    uint64_t fill = 0, serve = 0, pieces = 0;
    uint64_t hits = 0, big = 0, longp = 0, longhit = 0, bpe = 0;

    uint64_t pw_pieces[TSIMD_POOL_MAX], pw_hit[TSIMD_POOL_MAX];
    uint64_t pw_long[TSIMD_POOL_MAX], pw_longhit[TSIMD_POOL_MAX];
    uint64_t pw_bpe[TSIMD_POOL_MAX];
    int nw = p->width < TSIMD_POOL_MAX ? p->width : TSIMD_POOL_MAX;
    for (int i = 0; i < nw; i++) {
        pw_pieces[i] = p->workers[i].ph_pieces;
        pw_hit[i] = p->workers[i].ph_hits;
        pw_long[i] = p->workers[i].ph_long;
        pw_longhit[i] = p->workers[i].ph_longhit;
        pw_bpe[i] = p->workers[i].ph_bpe;
    }
    for (int i = 0; i < p->width; i++) {
        fill += p->workers[i].ph_fill;
        serve += p->workers[i].ph_serve;
        pieces += p->workers[i].ph_pieces;
        hits += p->workers[i].ph_hits;
        big += p->workers[i].ph_big;
        longp += p->workers[i].ph_long;
        longhit += p->workers[i].ph_longhit;
        bpe += p->workers[i].ph_bpe;
        p->workers[i].ph_fill = 0;
        p->workers[i].ph_serve = 0;
        p->workers[i].ph_pieces = 0;
        p->workers[i].ph_hits = 0;
        p->workers[i].ph_big = 0;
        p->workers[i].ph_long = 0;
        p->workers[i].ph_longhit = 0;
        p->workers[i].ph_bpe = 0;
    }
    phase_report(where, fill, serve, pieces);
    phase_report_mix(where, pieces, hits, big, longp, longhit, bpe);
    {
        const char *e = getenv("TSIMD_PERWORKER");
        if (e && e[0] == '1')
            for (int i = 0; i < nw; i++) {
                if (pw_pieces[i] == 0) continue;
                fprintf(stderr,
                        "tsimd worker %d %s: %llu pieces, %llu short-hit "
                        "(%.2f%%), %llu long (hit %llu), %llu bpe "
                        "(%.3f%%)\n",
                        i, where, (unsigned long long)pw_pieces[i],
                        (unsigned long long)pw_hit[i],
                        100.0 * (double)pw_hit[i] / (double)pw_pieces[i],
                        (unsigned long long)pw_long[i],
                        (unsigned long long)pw_longhit[i],
                        (unsigned long long)pw_bpe[i],
                        100.0 * (double)pw_bpe[i] / (double)pw_pieces[i]);
            }
    }
}

typedef struct {
    uint64_t hits, big, longp, longhit, bpe;
} serve_mix_t;

static int fused_o200k_serve(worker_t *w, const uint8_t *data, size_t len,
                             idvec_t *out) {
    tsimd_pcache_span_t batch[TSIMD_SERVE_BATCH + TSIMD_SERVE_D];
    const size_t cap = TSIMD_SERVE_BATCH;

    tsimd_pretok_cache_t *c = w->cache;
    size_t i = 0;
    int rc = 0;
    memset(&w->mask_st, 0, sizeof(w->mask_st));

    int use_scalar;
    if (tsimd_simd_level() < TSIMD_SIMD_AVX2)
        use_scalar = 1;
    else
        use_scalar = scalar_fill_enabled();

    static int twophase_force = -1;
    if (twophase_force < 0) {
        const char *e = getenv("TSIMD_TWOPHASE_FILL");
        twophase_force = (e && e[0] == '1') ? 1 : 0;
    }
    const int twophase = twophase_force;

    if (!out->foreign && out->cap < out->count + len + 64)
        if (idvec_reserve(out, out->count + len / 4 + (1u << 20)) != 0)
            return -1;
    const int ph = phase_stats_on();

    size_t x86_pieces = 0;
    while (rc == 0) {

        size_t n = 0;
        uint64_t ph_t0 = ph ? ph_rdtsc() : 0;
        if (!use_scalar) {

            int c2;
            if (w->t->scheme == TSIMD_PRETOK_GPT2)
                c2 = tsimd_gpt2_mask_fill_keyed(data, len,
                                                &w->mask_st, batch,
                                                cap, c,
                                                twophase);
            else if (w->t->scheme == TSIMD_PRETOK_CL100K)
                c2 = tsimd_cl100k_mask_fill_keyed(data, len,
                                                  &w->mask_st, batch,
                                                  cap, c,
                                                  twophase);
            else if (w->t->scheme == TSIMD_PRETOK_QWEN25) {

                w->mask_st.digit_single = 1;
                c2 = tsimd_cl100k_mask_fill_keyed(data, len,
                                                  &w->mask_st, batch,
                                                  cap, c,
                                                  twophase);
            }
            else if (w->t->scheme == TSIMD_PRETOK_GPT2D) {
                w->mask_st.digit_single = 1;
                c2 = tsimd_gpt2_mask_fill_keyed(data, len,
                                                &w->mask_st, batch,
                                                cap, c,
                                                twophase);
            }
            else if (w->t->scheme == TSIMD_PRETOK_DEEPSEEK)
                c2 = tsimd_deepseek_mask_fill_keyed(data, len,
                                                    &w->mask_st, batch,
                                                    cap, c,
                                                    twophase);
            else if (w->t->scheme == TSIMD_PRETOK_NEMO)
                c2 = tsimd_nemo_mask_fill_keyed(data, len,
                                                &w->mask_st, batch,
                                                cap, c,
                                                twophase);
            else
                c2 = tsimd_o200k_mask_fill_keyed(data, len,
                                                 &w->mask_st, batch,
                                                 cap, c,
                                                 twophase);
            if (c2 < 0) { rc = -1; break; }
            n = (size_t)c2;
        } else {

            if (!dual_cur_disabled() && i + 128 < len) {
                size_t s = serve_find_split(data, len,
                                            i + (len - i) / 2);
                if (s > i && s < len) {
                    size_t j = s, bn = 0;
                    tsimd_pcache_span_t bents[TSIMD_SERVE_BATCH / 2];
                    size_t half = cap / 2;
                    while (rc == 0 && n < half && bn < half &&
                           i < s && j < len) {
                        rc = serve_scalar_step(w, data, len, batch, c,
                                               cap, &n, &i);
                        if (rc == 0)
                            rc = serve_scalar_step(w, data, len, bents,
                                                   c, half, &bn, &j);
                    }

                    while (rc == 0 && n < cap && i < s)
                        rc = serve_scalar_step(w, data, len, batch, c,
                                               cap, &n, &i);
                    if (rc == 0 && i >= s) {

                        while (rc == 0 && bn < half && j < len)
                            rc = serve_scalar_step(w, data, len, bents,
                                                   c, half, &bn, &j);
                        size_t k = 0;
                        while (rc == 0 && k < bn && n < cap)
                            batch[n++] = bents[k++];
                        if (k == bn)
                            i = j;
                        else if (k > 0)
                            i = (size_t)(bents[k - 1].ptr - data) +
                                tsimd_pcache_span_len(&bents[k - 1]);

                    }
                    if (rc != 0) break;
                } else {
                    while (n < cap && i < len)
                        if (serve_scalar_step(w, data, len, batch, c,
                                              cap, &n, &i) != 0) {
                            rc = -1;
                            break;
                        }
                }
            } else {
                while (n < cap && i < len)
                    if (serve_scalar_step(w, data, len, batch, c, cap,
                                          &n, &i) != 0) {
                        rc = -1;
                        break;
                    }
            }
        }
        if (n == 0) break;
        uint64_t ph_t1 = ph ? ph_rdtsc() : 0;

        if (serve_dump_on())
            dump_serve(n, batch, data);
        if (idvec_reserve(out, out->count + 4 * n + 16) != 0) return -1;
        size_t base = out->count;
        size_t cur = 0;

        uint64_t bh = 0;
        uint64_t m_hits = 0, m_big = 0, m_long = 0, m_longhit = 0,
                 m_bpe = 0;
        const int use_x86 = serve_avx2_enabled();
        for (size_t j = 0; j < n; ) {

            if (use_x86) {
                size_t stop = tsimd_serve_hot_avx2(
                    c->entries, c->pair_mask, c->mask, batch, n, j,
                    out->ids + base, &cur);
                if (stop > j) {

                    bh += stop - j;
                    if (ph) m_hits += stop - j;
                    x86_pieces += stop - j;
                }
                j = stop;
                if (j >= n) break;
            }
            if (j + TSIMD_SERVE_D < n)
                serve_prefetch_l1(c, batch[j + TSIMD_SERVE_D].meta);
            tsimd_pcache_span_t *en = &batch[j];
            uint32_t el = tsimd_pcache_span_len(en);
            uint64_t pval = 0, pext = 0;
            size_t wslot = 0;
            int got_hit = 0;
            int big_hit = 0;
            const uint32_t *big_ids = NULL;
            uint32_t big_n = 0;
            uint64_t ab = 0, ex = 0;
            size_t adv = 0;

            if ((en->k1 >> 56) != 0 &&
                tsimd_pcache_probe_serve(c, en->k0, en->k1, en->meta,
                                         &pval, &pext, &wslot)) {

                ab = ((pval >> 3) & 0x1FFFFFFFull) |
                     (pval & 0xFFFFFFFF00000000ull);
                ex = pext;
                adv = (pval & 7);
                got_hit = 1;
                m_hits++;
                bh++;
            } else if (el > TSIMD_PCACHE_MAX_SPAN && !no_long_cache()) {
                m_long++;

                uint32_t ln = 0;
                const uint32_t *lid = tsimd_plong_probe_ex(c, en->ptr,
                                                           el, &ln);
                if (lid) {
                    got_hit = 1;
                    m_longhit++;
                    if (ln <= 4) {
                        ab = (uint64_t)lid[0] | ((uint64_t)lid[1] << 32);
                        ex = (uint64_t)lid[2] | ((uint64_t)lid[3] << 32);
                        adv = ln;
                    } else {
                        big_hit = 1;
                        big_ids = lid;
                        big_n = ln;
                    }
                }
            }
            if (big_hit) {
                m_big++;
                if (verify_long()) {
                    size_t vb = w->ws.out_count;
                    if (tsimd_bpe_encode(&w->t->vocab, &w->ws, en->ptr,
                                         el) != 0)
                        return -1;
                    if (w->ws.out_count - vb != big_n ||
                        memcmp(w->ws.out + vb, big_ids,
                               (size_t)big_n * sizeof(uint32_t)) != 0) {
                        fprintf(stderr,
                                "tsimd: VERIFY_LONG mismatch: served %u "
                                "ids, bpe gives %zu\n", big_n,
                                w->ws.out_count - vb);
                        return -1;
                    }
                    w->ws.out_count = vb;
                }

                out->count = base + cur;
                base = out->count;
                cur = 0;
                if (idvec_reserve(out, out->count + big_n +
                                      4 * (n - j) + 16) != 0)
                    return -1;
                if (idvec_push_n(out, big_ids, big_n) != 0)
                    return -1;
                base = out->count;
            } else if (got_hit) {

                uint32_t *p = out->ids + base + cur;
                memcpy(p, &ab, 8);
                memcpy(p + 2, &ex, 8);
                cur += adv;
            } else {

                size_t wbase = w->ws.out_count;
                uint32_t el2 = tsimd_pcache_span_len(en);
                m_bpe++;
                if (tsimd_bpe_encode(&w->t->vocab, &w->ws, en->ptr,
                                     el2) != 0) {
                    rc = -1;
                    break;
                }
                size_t nbpe = w->ws.out_count - wbase;
                if (nbpe <= 4) {
                    uint32_t *dst = out->ids + base + cur;
                    for (size_t k = 0; k < nbpe; k++)
                        dst[k] = w->ws.out[wbase + k];
                    cur += nbpe;
                } else {

                    out->count = base + cur;
                    base = out->count;
                    cur = 0;
                    if (idvec_reserve(out, out->count + nbpe +
                                          4 * (n - j) + 16) != 0) {
                        rc = -1;
                        break;
                    }
                    if (idvec_push_n(out, w->ws.out + wbase, nbpe) != 0) {
                        rc = -1;
                        break;
                    }
                    base = out->count;
                }
                if ((en->k1 >> 56) != 0)
                    tsimd_pcache_insert_keyed_at(c, wslot, en->k0,
                                                 en->k1, en->meta, el,
                                                 w->ws.out + wbase,
                                                 (uint32_t)nbpe);
                else if (el > TSIMD_PCACHE_MAX_SPAN &&
                         !no_long_cache())
                    tsimd_plong_insert(c, en->ptr, el,
                                       w->ws.out + wbase, (uint32_t)nbpe);
                w->ws.out_count = wbase;
            }
            j++;
        }
        c->hits += bh;
        if (ph) {
            uint64_t now = ph_rdtsc();
            w->ph_fill += ph_t1 - ph_t0;
            w->ph_serve += now - ph_t1;
            w->ph_pieces += n;
            w->ph_hits += m_hits;
            w->ph_big += m_big;
            w->ph_long += m_long;
            w->ph_longhit += m_longhit;
            w->ph_bpe += m_bpe;
        }
        if (rc != 0) break;
        out->count = base + cur;

        if (scalar_fill_enabled() && n < cap) break;
    }
    if (x86_pieces)
        atomic_fetch_add_explicit(&g_serve_avx2_pieces, x86_pieces,
                                  memory_order_relaxed);
    return rc;
}

typedef struct {
    worker_t w;
    int ready;
} seg_cache_t;

void tsimd_serve_state_free(tsimd_tokenizer_t *t) {
    seg_cache_t *sc = (seg_cache_t *)t->seg_state;
    if (!sc) return;
    if (sc->ready) {
        tsimd_bpe_workspace_free(&sc->w.ws);
        free(sc->w.dual_a.ids);
        free(sc->w.dual_b.ids);
        sc->w.dual_a.ids = NULL;
        sc->w.dual_b.ids = NULL;
        if (sc->w.cache && sc->w.cache != &t->cache) {
            tsimd_pretok_cache_free(sc->w.cache);
            free(sc->w.cache);
        }
        sc->w.cache = NULL;
    }
    tsimd_xfree(sc);
    t->seg_state = NULL;
    t->seg_busy = 0;
}

static worker_t *serve_stack_fallback(worker_t *stack_w,
                                      tsimd_tokenizer_t *t) {
    memset(stack_w, 0, sizeof(*stack_w));
    if (tsimd_bpe_workspace_init(&stack_w->ws, t->vocab.max_token_len) != 0)
        return NULL;
    return stack_w;
}

static worker_t *serve_worker_claim(tsimd_tokenizer_t *t,
                                    worker_t *stack_w) {
    int expect = 0;
    if (!atomic_compare_exchange_strong(&t->seg_busy, &expect, 1))
        return serve_stack_fallback(stack_w, t);
    seg_cache_t *sc = (seg_cache_t *)atomic_load_explicit(
        &t->seg_state, memory_order_acquire);
    if (!sc) {
        sc = tsimd_xmalloc(sizeof(*sc));
        seg_cache_t *raced = NULL;
        if (sc) memset(sc, 0, sizeof(*sc));
        if (!sc ||
            !atomic_compare_exchange_strong(&t->seg_state, &raced, sc)) {
            if (sc) tsimd_xfree(sc);
            if (!raced) {
                atomic_store_explicit(&t->seg_busy, 0, memory_order_release);
                return serve_stack_fallback(stack_w, t);
            }
            sc = raced;
        }
    }
    if (!sc->ready) {
        if (tsimd_bpe_workspace_init(&sc->w.ws, t->vocab.max_token_len) != 0) {
            atomic_store_explicit(&t->seg_busy, 0, memory_order_release);
            return NULL;
        }
        sc->ready = 1;
    }
    return &sc->w;
}

static void serve_worker_release(tsimd_tokenizer_t *t, worker_t *w,
                                 worker_t *stack_w) {
    if (w == stack_w) {
        tsimd_bpe_workspace_free(&stack_w->ws);
        return;
    }
    atomic_store_explicit(&t->seg_busy, 0, memory_order_release);
}

int tsimd_serve_segment(tsimd_tokenizer_t *t, tsimd_bpe_workspace_t *out_ws,
                        const uint8_t *data, size_t len) {
    worker_t stack_w;
    worker_t *w = serve_worker_claim(t, &stack_w);
    if (!w) return -1;
    w->t = t;
    t->cache.shared = 1;
    w->cache = &t->cache;
    w->ws.out_count = 0;
    memset(&w->mask_st, 0, sizeof(w->mask_st));
    idvec_t ov;
    ov.ids = out_ws->out;
    ov.cap = out_ws->out_capacity;
    ov.count = out_ws->out_count;
    ov.foreign = out_ws->out_foreign;
    int rc = fused_o200k_serve(w, data, len, &ov);
    phase_report("serve_segment", w->ph_fill, w->ph_serve, w->ph_pieces);
    phase_report_mix("serve_segment", w->ph_pieces, w->ph_hits, w->ph_big,
                     w->ph_long, w->ph_longhit, w->ph_bpe);
    w->ph_fill = 0;
    w->ph_serve = 0;
    w->ph_pieces = 0;
    w->ph_hits = 0;
    w->ph_big = 0;
    w->ph_long = 0;
    w->ph_longhit = 0;
    w->ph_bpe = 0;
    serve_worker_release(t, w, &stack_w);
    out_ws->out = ov.ids;
    out_ws->out_capacity = ov.cap;
    out_ws->out_count = ov.count;
    out_ws->out_foreign = ov.foreign;
    return rc;
}

static int batched_serve_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_DISABLE_BATCHED_SERVE");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

int tsimd_batched_serve_disabled(void) {
    return batched_serve_disabled();
}

typedef struct {
    worker_t *w;
    idvec_t *out;
    const uint8_t *data;
    const uint8_t *end;
    tsimd_pcache_span_t batch[TSIMD_SERVE_BATCH];
    size_t n;
} bserve_ctx_t;

static int bserve_flush(bserve_ctx_t *f) {
    worker_t *w = f->w;
    idvec_t *out = f->out;
    tsimd_pcache_span_t *batch = f->batch;
    size_t n = f->n;
    f->n = 0;

    tsimd_pretok_cache_t *c = w->cache;
    if (n == 0) return 0;
    if (idvec_reserve(out, out->count + 4 * n + 16) != 0) return -1;
    size_t base = out->count;
    size_t cur = 0;
    int rc = 0;

    uint64_t bh = 0;
    for (size_t j = 0; j < n; j++) {
        if (j + TSIMD_SERVE_D < n)
            serve_prefetch_l1(c, batch[j + TSIMD_SERVE_D].meta);
        tsimd_pcache_span_t *en = &batch[j];
        uint32_t el = tsimd_pcache_span_len(en);
        uint64_t pval = 0, pext = 0;
        size_t wslot = 0;
        int got_hit = 0;
        int big_hit = 0;
        const uint32_t *big_ids = NULL;
        uint32_t big_n = 0;
        uint64_t ab = 0, ex = 0;
        size_t adv = 0;
        if ((en->k1 >> 56) != 0 &&
            tsimd_pcache_probe_serve(c, en->k0, en->k1, en->meta,
                                     &pval, &pext, &wslot)) {
            ab = ((pval >> 3) & 0x1FFFFFFFull) |
                 (pval & 0xFFFFFFFF00000000ull);
            ex = pext;
            adv = (pval & 7);
            got_hit = 1;
            bh++;
        } else if (el > TSIMD_PCACHE_MAX_SPAN && !no_long_cache()) {
            uint32_t ln = 0;
            const uint32_t *lid = tsimd_plong_probe_ex(c, en->ptr,
                                                       el, &ln);
            if (lid) {
                got_hit = 1;
                if (ln <= 4) {
                    ab = (uint64_t)lid[0] | ((uint64_t)lid[1] << 32);
                    ex = (uint64_t)lid[2] | ((uint64_t)lid[3] << 32);
                    adv = ln;
                } else {
                    big_hit = 1;
                    big_ids = lid;
                    big_n = ln;
                }
            }
        }
        if (big_hit) {
            if (verify_long()) {
                size_t vb = w->ws.out_count;
                if (tsimd_bpe_encode(&w->t->vocab, &w->ws, en->ptr,
                                     el) != 0)
                    return -1;
                if (w->ws.out_count - vb != big_n ||
                    memcmp(w->ws.out + vb, big_ids,
                           (size_t)big_n * sizeof(uint32_t)) != 0) {
                    fprintf(stderr,
                            "tsimd: VERIFY_LONG mismatch: served %u "
                            "ids, bpe gives %zu\n", big_n,
                            w->ws.out_count - vb);
                    return -1;
                }
                w->ws.out_count = vb;
            }
            out->count = base + cur;
            base = out->count;
            cur = 0;
            if (idvec_reserve(out, out->count + big_n +
                                  4 * (n - j) + 16) != 0)
                return -1;
            if (idvec_push_n(out, big_ids, big_n) != 0)
                return -1;
            base = out->count;
        } else if (got_hit) {
            uint32_t *p = out->ids + base + cur;

            _mm_storeu_si128((__m128i *)(void *)p,
                             _mm_set_epi64x((long long)ex, (long long)ab));
            cur += adv;
        } else {
            size_t wbase = w->ws.out_count;
            uint32_t el2 = tsimd_pcache_span_len(en);
            if (tsimd_bpe_encode(&w->t->vocab, &w->ws, en->ptr,
                                 el2) != 0) {
                rc = -1;
                break;
            }
            size_t nbpe = w->ws.out_count - wbase;
            if (nbpe <= 4) {
                uint32_t *dst = out->ids + base + cur;
                for (size_t k = 0; k < nbpe; k++)
                    dst[k] = w->ws.out[wbase + k];
                cur += nbpe;
            } else {
                out->count = base + cur;
                base = out->count;
                cur = 0;
                if (idvec_reserve(out, out->count + nbpe +
                                      4 * (n - j) + 16) != 0) {
                    rc = -1;
                    break;
                }
                if (idvec_push_n(out, w->ws.out + wbase, nbpe) != 0) {
                    rc = -1;
                    break;
                }
                base = out->count;
            }
            if ((en->k1 >> 56) != 0)
                tsimd_pcache_insert_keyed_at(c, wslot, en->k0, en->k1,
                                             en->meta, el,
                                             w->ws.out + wbase,
                                             (uint32_t)nbpe);
            else if (el > TSIMD_PCACHE_MAX_SPAN &&
                     !no_long_cache())
                tsimd_plong_insert(c, en->ptr, el,
                                   w->ws.out + wbase, (uint32_t)nbpe);
            w->ws.out_count = wbase;
        }
    }
    c->hits += bh;
    if (rc != 0) return rc;
    out->count = base + cur;
    return 0;
}

static int fused_span_bserve(void *ud, size_t s, size_t e) {
    bserve_ctx_t *f = ud;
    tsimd_pcache_span_t *en = &f->batch[f->n];
    en->ptr = f->data + s;
    uint32_t el = (uint32_t)(e - s);
    if (el <= TSIMD_PCACHE_MAX_SPAN) {
        tsimd_pcache_key_inline(en->ptr, el, f->end, &en->k0, &en->k1,
                                &en->meta);

        serve_prefetch_l2(f->w->cache, en->meta);
    } else {
        en->k0 = 0;
        en->k1 = 0;
        en->meta = el;
    }
    f->n++;
    if (f->n == TSIMD_SERVE_BATCH) return bserve_flush(f);
    return 0;
}

static int fused_batched_serve(worker_t *w, const uint8_t *data, size_t len,
                               idvec_t *out) {
    bserve_ctx_t f;
    f.w = w;
    f.out = out;
    f.data = data;
    f.end = data + len;
    f.n = 0;

    if (!out->foreign && out->cap < out->count + len + 64)
        if (idvec_reserve(out, out->count + len / 4 + (1u << 20)) != 0)
            return -1;
    int rc = tsimd_pretokenize_stream(w->t->scheme, data, len,
                                      fused_span_bserve, &f, NULL);
    if (rc == 0) rc = bserve_flush(&f);
    return rc;
}

int tsimd_serve_batched_segment(tsimd_tokenizer_t *t,
                                tsimd_bpe_workspace_t *out_ws,
                                const uint8_t *data, size_t len) {
    worker_t w;
    memset(&w, 0, sizeof(w));
    w.t = t;
    t->cache.shared = 1;
    w.cache = &t->cache;
    if (tsimd_bpe_workspace_init(&w.ws, t->vocab.max_token_len) != 0)
        return -1;
    idvec_t ov;
    ov.ids = out_ws->out;
    ov.cap = out_ws->out_capacity;
    ov.count = out_ws->out_count;
    ov.foreign = out_ws->out_foreign;
    int rc = fused_batched_serve(&w, data, len, &ov);
    tsimd_bpe_workspace_free(&w.ws);
    out_ws->out = ov.ids;
    out_ws->out_capacity = ov.cap;
    out_ws->out_count = ov.count;
    if (getenv("TSIMD_IDBG") && ov.cap == 6144)
        fprintf(stderr, "IDBG sync out_ws=%p ids=%p\n", (void*)out_ws,
                (void*)ov.ids);
    return rc;
}

static int idvec_append_out(worker_t *w, idvec_t *out);
static int kimi_masks_off(void);

static int run_text_job(worker_t *w, const job_t *job, idvec_t *out) {

    if (w->t->scheme == TSIMD_PRETOK_GEM4 ||
        w->t->scheme == TSIMD_PRETOK_SPM) {
        tsimd_pretok_cache_t *pc =
            w->idx < 64 ? tsimd_spm_worker_cache(w->t, w->idx, job->len)
                        : &w->t->cache;
        if (tsimd_encode_buffer_pool(w->t, job->data, job->len, &w->ws, pc,
                                     &w->t->cache) != 0)
            return -1;
        return idvec_append_out(w, out);
    }
    if (w->t->scheme == TSIMD_PRETOK_O200K ||
        w->t->scheme == TSIMD_PRETOK_CL100K ||
        w->t->scheme == TSIMD_PRETOK_DEEPSEEK ||
        w->t->scheme == TSIMD_PRETOK_NEMO ||
        w->t->scheme == TSIMD_PRETOK_QWEN25 ||

        w->t->scheme == TSIMD_PRETOK_GPT2 ||
        (w->t->scheme == TSIMD_PRETOK_GPT2D && tsimd_gpt2d_span_on()))

        return fused_o200k_serve(w, job->data, job->len, out);
    if (w->t->scheme == TSIMD_PRETOK_KIMI) {
        pfuse_t f = {w, out, job->data};
        if (!kimi_masks_off())
            return tsimd_pretokenize_kimi_masks_ex(job->data, job->len,
                                                   fused_span_p, &f);
        return tsimd_pretokenize_stream(w->t->scheme, job->data, job->len,
                                        fused_span_p, &f, NULL);
    }
    if (!batched_serve_disabled())
        return fused_batched_serve(w, job->data, job->len, out);
    pfuse_t f = {w, out, job->data};
    return tsimd_pretokenize_stream(w->t->scheme, job->data, job->len,
                                   fused_span_p, &f, NULL);
}

static int idvec_append_out(worker_t *w, idvec_t *out) {
    for (size_t k = 0; k < w->ws.out_count; k++)
        if (idvec_push(out, w->ws.out[k]) != 0) return -1;
    w->ws.out_count = 0;
    return 0;
}

static int kimi_masks_off(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_KIMI_MASKS");
        atomic_store(&state, (e && e[0] == '1') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v != 1;
}

static int gather_nt_enabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_GATHER_NT");
        atomic_store(&state, (e && e[0] == '0') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 1;
}

static void copy_stream_u32(uint32_t *dst, const uint32_t *src, size_t n) {
    size_t i = 0;
    while (i < n && ((uintptr_t)(dst + i) & 15)) {
        dst[i] = src[i];
        i++;
    }
    for (; i + 4 <= n; i += 4)
        _mm_stream_si128((__m128i *)(void *)(dst + i),
                         _mm_loadu_si128(
                             (const __m128i *)(const void *)(src + i)));
    for (; i < n; i++) dst[i] = src[i];
}

static int run_one_job(worker_t *w, const job_t *job, idvec_t *out) {

    if (job->kind == JOB_NFC) {
        size_t base = w->ws.out_count;
        if (tsimd_encode_chunk_nfc_fused(w->t, &w->ws, job->data, job->len,
                                         w->t->norm == TSIMD_NORM_NFKC) != 0)
            return -1;
        (void)base;
        return idvec_append_out(w, out);
    }
    if (job->kind == JOB_GATHER) {

        const idvec_t *src = job->gsrc;
        const int nt = gather_nt_enabled();
        size_t off = 0;
        for (size_t i = 0; i < job->gj0; i++) off += src[i].count;
        for (size_t j = job->gj0; j < job->gj1; j++) {
            if (src[j].count) {
                if (nt)
                    copy_stream_u32(job->gdst + off, src[j].ids,
                                    src[j].count);
                else
                    memcpy(job->gdst + off, src[j].ids,
                           src[j].count * sizeof(uint32_t));
            }
            off += src[j].count;
        }
        if (nt) _mm_sfence();
        return 0;
    }
    if (job->kind == JOB_SPECIAL) return idvec_push(out, job->special_id);
    if (job->kind == JOB_BPE) {

        return job->len == 0
                   ? 0
                   : tsimd_bpe_encode(&w->t->vocab, &w->ws, job->data,
                                      (uint32_t)job->len) != 0
                         ? -1
                         : idvec_append_out(w, out);
    }
    if (job->len == 0) return 0;

    if (idvec_reserve(out, job->len / 2 + 16) != 0) return -1;
    return run_text_job(w, job, out);
}

int tsimd_par_split(const size_t *weights, size_t n, int width,
                    size_t *lo, size_t *hi) {
    if (width < 1) return 0;
    for (int k = 0; k < width; k++) {
        lo[k] = 0;
        hi[k] = 0;
    }
    if (n == 0) return 1;
    unsigned long long total = 0;
    for (size_t i = 0; i < n; i++)
        total += weights[i] ? weights[i] : 1;

    size_t j = 0;
    unsigned long long cw = 0;
    for (int k = 1; k < width; k++) {
        unsigned long long ideal =
            (total / (unsigned long long)width) * (unsigned long long)k +
            ((total % (unsigned long long)width) * (unsigned long long)k) /
                (unsigned long long)width;
        while (j < n && cw < ideal) {
            cw += weights[j] ? weights[j] : 1;
            j++;
        }
        hi[k - 1] = j;
        lo[k] = j;
    }
    hi[width - 1] = n;

    unsigned long long mx = 0;
    for (int k = 0; k < width; k++) {
        unsigned long long r = 0;
        for (size_t i = lo[k]; i < hi[k]; i++)
            r += weights[i] ? weights[i] : 1;
        if (r > mx) mx = r;
    }
    return mx * (unsigned long long)width <= 2ull * total;
}

#define TSIMD_PIN_MAX 64
static int pin_order[TSIMD_PIN_MAX];
static _Atomic int pin_count;
static _Atomic int pin_ready;

static int weighted_split_mode(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_WEIGHTED");
        int m = 1;
        if (e && e[0] == '0')
            m = 2;
        else if (e && e[0] == '1')
            m = 3;
        atomic_store(&state, m);
        v = m;
    }
    return v;
}

int tsimd_cpu_core_kind(int cpu);
static void wait_backoff(void);

static double worker_cap_of_cpu(int cpu) {
    if (cpu < 0) return 1.0;
    if (weighted_split_mode() == 3) {
        if (cpu >= 12 && cpu <= 15) return 0.6;
        if (cpu < 6) return 1.0;
        return 0.5;
    }
    switch (tsimd_cpu_core_kind(cpu)) {
        case 1: return 0.6;
        case 3: return 0.6;
        default: return 1.0;
    }
}

static int weighted_split_on(void) {
    return weighted_split_mode() == 1 || weighted_split_mode() == 3;
}

int tsimd_cpu_core_kind(int cpu) {
    static int cached[TSIMD_PIN_MAX];
    static _Atomic int tried[TSIMD_PIN_MAX];
    if (cpu < 0 || cpu >= TSIMD_PIN_MAX) return 0;
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&tried[cpu], &expected, 1,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) {
        while (atomic_load_explicit(&tried[cpu], memory_order_acquire) == 1)
            wait_backoff();
        return cached[cpu];
    }
    int kind = 0;
    char path[128];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
             cpu);
    int smt_index = 0;
    FILE *f = fopen(path, "rb");
    if (f) {
        char buf[256];
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[n] = 0;
        const char *p = buf;
        int idx = 0;
        while (*p) {
            char *e = NULL;
            long a = strtol(p, &e, 10);
            if (e == p) break;
            p = e;
            long b = a;
            if (*p == '-') {
                b = strtol(p + 1, &e, 10);
                p = e;
            }
            if (cpu >= a && cpu <= b) {
                smt_index = idx + (cpu - (int)a);
                break;
            }
            idx += (int)(b - a) + 1;
            if (*p == ',')
                p++;
            else
                break;
        }
    }
    cpu_set_t save;
    if (sched_getaffinity(0, sizeof save, &save) == 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set) == 0) {
#if defined(__x86_64__) || defined(__i386__)
            uint32_t va = 0, vb = 0, vc = 0, vd = 0;
            if (__get_cpuid_count(0x1A, 0, &va, &vb, &vc, &vd)) {
                unsigned t = va >> 24;
                if (t == 0x20)
                    kind = 1;
                else if (t == 0x40)
                    kind = smt_index > 0 ? 3 : 2;
            }
#endif
            sched_setaffinity(0, sizeof save, &save);
        }
    }
    cached[cpu] = kind;
    atomic_store_explicit(&tried[cpu], 2, memory_order_release);
    return kind;
}

static int par_split_caps(const size_t *weights, size_t n, int width,
                          size_t *lo, size_t *hi) {
    if (width < 2) return 0;
    double caps[TSIMD_POOL_MAX + 1];
    double total_cap = 0;
    caps[0] = 1.0;
    for (int k = 1; k < width; k++) {
        caps[k] = k - 1 < pin_count ? worker_cap_of_cpu(pin_order[k - 1])
                                    : 1.0;
    }
    for (int k = 0; k < width; k++) total_cap += caps[k];
    if (total_cap <= 0) return 0;

    unsigned long long total = 0;
    for (size_t i = 0; i < n; i++)
        total += weights[i] ? weights[i] : 1;
    for (int k = 0; k < width; k++) {
        lo[k] = 0;
        hi[k] = 0;
    }
    size_t j = 0;
    unsigned long long cw = 0;
    double cum = 0;
    for (int k = 1; k < width; k++) {
        cum += caps[k - 1];
        unsigned long long ideal =
            (unsigned long long)((double)total * cum / total_cap);
        while (j < n && cw < ideal) {
            cw += weights[j] ? weights[j] : 1;
            j++;
        }
        hi[k - 1] = j;
        lo[k] = j;
    }
    hi[width - 1] = n;
    return 1;
}

static int par_static_mode(void) {
    const char *e = getenv("TSIMD_PAR_STATIC");
    if (!e || !e[0]) return 2;
    return e[0] - '0';
}

static size_t job_weight(const job_t *j) {
    switch (j->kind) {
    case JOB_SPECIAL:
        return 1;
    case JOB_BPE:
        return (j->len ? j->len : 1) * 4 + 16;
    case JOB_GATHER: {
        const idvec_t *src = j->gsrc;
        size_t n = 0;
        for (size_t i = j->gj0; i < j->gj1; i++) n += src[i].count;
        return n * 4 + 16;
    }
    default:
        return j->len ? j->len : 1;
    }
}

typedef void (*claim_visit_fn)(void *ctx, size_t i);

static void claim_drain(size_t lo, size_t hi, size_t n,
                        atomic_uchar *claims, atomic_size_t *steal_next,
                        claim_visit_fn visit, void *ctx) {
    for (size_t i = lo; i < hi; i++)
        if (atomic_exchange_explicit(&claims[i], 1, memory_order_relaxed) == 0)
            visit(ctx, i);
    if (!steal_next || hi - lo >= n) return;
    size_t cur = atomic_load_explicit(steal_next, memory_order_relaxed);
    while (cur > 0) {
        size_t idx = cur - 1;
        if (!atomic_compare_exchange_weak_explicit(
                steal_next, &cur, idx, memory_order_relaxed,
                memory_order_relaxed))
            continue;
        if (atomic_exchange_explicit(&claims[idx], 1, memory_order_relaxed) != 0)
            continue;
        visit(ctx, idx);
        while (idx > 0 &&
               atomic_exchange_explicit(&claims[idx - 1], 1,
                                        memory_order_relaxed) == 0) {
            idx--;
            visit(ctx, idx);
        }
    }
}

static void visit_run_job(void *ctx, size_t i) {
    worker_t *w = ctx;
    tsimd_pool_t *p = w->pool;
    if (run_one_job(w, &p->jobs[i], &p->outs[i]) != 0)
        atomic_store_explicit(&p->err, 1, memory_order_relaxed);
}

typedef struct {
    atomic_uchar *claims;
    atomic_size_t *steal_next;
    size_t lo, hi, n;
    const uint32_t *spin;
    unsigned *runs;
} probe_thread_t;

static void probe_visit(void *vctx, size_t i) {
    probe_thread_t *c = vctx;
    uint32_t s = c->spin ? c->spin[i] : 1;
    volatile uint32_t acc = 0;
    for (uint32_t t = 0; t < s; t++) acc += t * 2654435761u;
    (void)acc;

    __atomic_fetch_add(&c->runs[i], 1, __ATOMIC_RELAXED);
}

static void *probe_thread(void *arg) {
    probe_thread_t *c = arg;
    claim_drain(c->lo, c->hi, c->n, c->claims, c->steal_next, probe_visit, c);
    return NULL;
}

int tsimd_par_drain_probe(size_t n, int width, const uint32_t *spin,
                          unsigned *runs) {
    if (width < 1 || !runs) return -1;
    if (n == 0) return 0;
    atomic_uchar *claims = calloc(n, 1);
    size_t *weights = malloc(n * sizeof(size_t));
    size_t *lo = malloc((size_t)width * sizeof(size_t));
    size_t *hi = malloc((size_t)width * sizeof(size_t));
    pthread_t *threads = calloc((size_t)width, sizeof(pthread_t));
    probe_thread_t *ctx = calloc((size_t)width, sizeof(probe_thread_t));
    int rc = -1;
    if (!claims || !weights || !lo || !hi || !threads || !ctx) goto done;
    for (size_t i = 0; i < n; i++)
        weights[i] = spin ? (spin[i] ? spin[i] : 1) : 1;

    tsimd_par_split(weights, n, width, lo, hi);
    atomic_size_t steal_next;
    atomic_store(&steal_next, n);
    int spawned = 0;
    for (int k = 0; k < width; k++) {
        ctx[k].claims = claims;
        ctx[k].steal_next = &steal_next;
        ctx[k].lo = lo[k];
        ctx[k].hi = hi[k];
        ctx[k].n = n;
        ctx[k].spin = spin;
        ctx[k].runs = runs;
    }
    for (int k = 1; k < width; k++)
        if (pthread_create(&threads[k], NULL, probe_thread, &ctx[k]) == 0)
            spawned++;
    probe_thread(&ctx[0]);
    for (int k = 1; k <= spawned; k++)
        pthread_join(threads[k], NULL);
    rc = spawned == width - 1 ? 0 : -1;
done:
    free(claims);
    free(weights);
    free(lo);
    free(hi);
    free(threads);
    free(ctx);
    return rc;
}

static void drain_jobs(worker_t *w) {
    tsimd_pool_t *p = w->pool;
    if (p->par_mode != 0 && w->idx < p->n_range) {
        claim_drain(p->plo[w->idx], p->phi[w->idx], p->n_jobs, p->claims,
                    p->par_mode == 2 ? &p->steal_next : NULL,
                    visit_run_job, w);
        return;
    }
    if (p->par_mode != 0) return;
    for (;;) {
        size_t i = atomic_fetch_add(&p->next, 1);
        if (i >= p->n_jobs) break;
        if (run_one_job(w, &p->jobs[i], &p->outs[i]) != 0)
            atomic_store(&p->err, 1);
    }
}

static atomic_int pin_build_busy;

static int pin_fault_on(void) {
    static atomic_int state = 0;
    int v = atomic_load_explicit(&state, memory_order_relaxed);
    if (v == 0) {
        const char *e = getenv("TSIMD_PIN_FAULT");
        atomic_store_explicit(&state, (e && e[0] == '1') ? 2 : 1,
                              memory_order_relaxed);
        v = atomic_load_explicit(&state, memory_order_relaxed);
    }
    return v == 2;
}

static void wait_backoff(void) {
    struct timespec ts = {0, 200000};
    sched_yield();
    nanosleep(&ts, NULL);
}

static void pin_order_build(void) {
    if (atomic_load_explicit(&pin_ready, memory_order_acquire)) return;
    if (atomic_exchange(&pin_build_busy, 1)) {
        while (!atomic_load_explicit(&pin_ready, memory_order_acquire))
            wait_backoff();
        return;
    }
    FILE *f = pin_fault_on() ? NULL : fopen("/proc/cpuinfo", "r");
    if (!f) {
        atomic_store_explicit(&pin_count, 0, memory_order_release);
        atomic_store_explicit(&pin_ready, 1, memory_order_release);
        return;
    }
    char line[256];
    int cur_proc = -1;
    int first_of_core[512];
    int seen[512];
    int sib_of[512];
    int sib[TSIMD_PIN_MAX];
    int nsib = 0, ncores = 0;
    memset(seen, 0, sizeof(seen));
    for (int c = 0; c < 512; c++) sib_of[c] = -1;
    while (fgets(line, sizeof(line), f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) {
            cur_proc = v;
            continue;
        }
        if (cur_proc < 0) continue;
        if (sscanf(line, "core id : %d", &v) == 1) {
            if (v >= 0 && v < 512) {
                if (!seen[v]) {
                    seen[v] = 1;
                    first_of_core[v] = cur_proc;
                    ncores++;
                } else if (nsib < TSIMD_PIN_MAX) {
                    sib[nsib++] = cur_proc;
                    sib_of[cur_proc] = first_of_core[v];
                    sib_of[first_of_core[v]] = cur_proc;
                }
            }
        }
    }
    fclose(f);

    int elastic = 0;
    const char *pe = getenv("TSIMD_PIN_ELASTIC");
    if (pe && pe[0] == '1') elastic = 1;
    int n = 0;
    if (elastic) {
        for (int c = 0; c < 512 && n < TSIMD_PIN_MAX; c++)
            if (seen[c] && sib_of[first_of_core[c]] >= 0)
                pin_order[n++] = first_of_core[c];
        for (int i = 0; i < nsib && n < TSIMD_PIN_MAX; i++)
            pin_order[n++] = sib[i];
        for (int c = 0; c < 512 && n < TSIMD_PIN_MAX; c++)
            if (seen[c] && sib_of[first_of_core[c]] < 0)
                pin_order[n++] = first_of_core[c];
    } else {
        for (int c = 0; c < 512 && n < TSIMD_PIN_MAX; c++)
            if (seen[c] && n < TSIMD_PIN_MAX) pin_order[n++] = first_of_core[c];
        for (int i = 0; i < nsib && n < TSIMD_PIN_MAX; i++)
            pin_order[n++] = sib[i];
    }
    if (getenv("TSIMD_PIN_DBG")) {
        fprintf(stderr, "tsimd pin order:");
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", pin_order[i]);
        fprintf(stderr, "\n");
    }
    atomic_store_explicit(&pin_count, n, memory_order_release);
    atomic_store_explicit(&pin_ready, 1, memory_order_release);
}

static cpu_set_t g_saved_mask;
static atomic_int g_saved_mask_state;

static void mask_save_once(void) {
    int v = atomic_load_explicit(&g_saved_mask_state, memory_order_acquire);
    if (v == 2) return;
    if (atomic_exchange(&g_saved_mask_state, 1)) {
        while (atomic_load_explicit(&g_saved_mask_state, memory_order_acquire) == 1)
            wait_backoff();
        return;
    }
    if (pthread_getaffinity_np(pthread_self(), sizeof(g_saved_mask),
                               &g_saved_mask) == 0)
        atomic_store_explicit(&g_saved_mask_state, 2, memory_order_release);
    else
        atomic_store_explicit(&g_saved_mask_state, 0, memory_order_release);
}

static int pin_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_PIN");
        atomic_store(&state, e && e[0] == '0' ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

void tsimd_pin_thread_nth(int idx) {
    if (pin_disabled()) return;
    mask_save_once();
    if (atomic_load_explicit(&g_saved_mask_state, __ATOMIC_ACQUIRE) != 2)
        return;
    int count = CPU_COUNT(&g_saved_mask);
    if (count <= 0) return;
    int want = idx % count;
    if (want < 0) want += count;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &g_saved_mask)) continue;
        if (want == 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(c, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            return;
        }
        want--;
    }
}

void tsimd_restore_process_affinity(void) {
    mask_save_once();
    if (atomic_load_explicit(&g_saved_mask_state, __ATOMIC_ACQUIRE) == 2)
        pthread_setaffinity_np(pthread_self(), sizeof(g_saved_mask),
                               &g_saved_mask);
}

static void worker_pin(int idx) {
    if (pin_disabled()) return;
    mask_save_once();
    pin_order_build();
    if (idx <= 0 || idx >= pin_count) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pin_order[idx], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void caller_pin(void) {
    if (pin_disabled()) return;
    mask_save_once();
    pin_order_build();
    if (pin_count < 1) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pin_order[0], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static int worker_cache_build(worker_t *w, int bits);

static void *pool_worker(void *arg) {
    worker_t *w = arg;
    tsimd_pool_t *p = w->pool;

    worker_pin(w->idx);

    if (w->cache == NULL && worker_cache_build(w, p->cache_bits) != 0)
        w->init_rc = -1;

    pthread_mutex_lock(&p->mu);

    atomic_fetch_add(&p->ready, 1);
    pthread_cond_broadcast(&p->ready_cv);
    unsigned seen = p->generation;
    while (!p->shutdown) {
        while (!p->shutdown && p->generation == seen)
            pthread_cond_wait(&p->wake_cv, &p->mu);
        if (p->shutdown) break;
        seen = p->generation;
        int participate = w->idx <= p->active && w->init_rc == 0;
        pthread_mutex_unlock(&p->mu);

        if (participate) drain_jobs(w);

        pthread_mutex_lock(&p->mu);
        atomic_fetch_add(&p->done, 1);
        pthread_cond_broadcast(&p->done_cv);
    }
    pthread_mutex_unlock(&p->mu);
    tsimd_nfc_tls_release();
    return NULL;
}

static void worker_state_free(worker_t *w) {
    tsimd_bpe_workspace_free(&w->ws);
    free(w->dual_a.ids);
    free(w->dual_b.ids);
    if (w->cache) {
        if (!w->shared_cache) {
            tsimd_pretok_cache_free(w->cache);
            free(w->cache);
        }
        w->cache = NULL;
    }
}

static void pool_destroy(tsimd_pool_t *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->wake_cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_spawned; i++)
        pthread_join(p->threads[i], NULL);
    for (int i = 0; i < p->width; i++)
        worker_state_free(&p->workers[i]);
    free(p->workers);
    free(p->threads);
    for (size_t i = 0; p->outs_cache && i < p->outs_cache_n; i++)
        free(p->outs_cache[i].ids);
    free(p->outs_cache);
    free(p->weights);
    free(p->plo);
    free(p->phi);
    free(p->claims);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->wake_cv);
    pthread_cond_destroy(&p->done_cv);
    pthread_cond_destroy(&p->ready_cv);
    free(p);
}

int tsimd_pool_cache_bits(int width, size_t total_len) {
    int bits = TSIMD_PAR_CACHE_BITS_MAX;
    if (total_len > 0 && width > 0) {
        size_t share = total_len / (size_t)width;
        if (share) {
            int lg = 63 - __builtin_clzll((unsigned long long)share);
            bits = lg - 5;
        }
    }
    if (bits < TSIMD_PAR_CACHE_BITS_MIN) bits = TSIMD_PAR_CACHE_BITS_MIN;
    if (bits > TSIMD_PAR_CACHE_BITS_MAX) bits = TSIMD_PAR_CACHE_BITS_MAX;
    int env = tsimd_pcache_bits_env();
    if (env > 0 && bits > env) bits = env;
    return bits;
}

static int worker_cache_build(worker_t *w, int bits) {
    {
        const char *e = getenv("TSIMD_POOL_INIT_FAIL");
        if (e && atoi(e) == w->idx) return -1;
    }
    {
        const char *sh = getenv("TSIMD_POOL_SHARED");
        if (w->t->scheme == TSIMD_PRETOK_KIMI && w->idx < 64 &&
            !(sh && sh[0] == '1')) {
            tsimd_pretok_cache_t *pc =
                tsimd_spm_worker_cache(w->t, w->idx, (size_t)1 << bits);
            if (pc) {
                w->cache = pc;
                w->shared_cache = 1;
                return 0;
            }
        }
        if (!(sh && sh[0] == '0')) {
            w->t->cache.shared = 1;
            w->cache = &w->t->cache;
            w->shared_cache = 1;
            return 0;
        }
    }
    w->cache = malloc(sizeof(tsimd_pretok_cache_t));
    if (!w->cache)
        return -1;
    if (tsimd_pretok_cache_init(w->cache, (size_t)1 << bits) != 0)
        return -1;

    const char *on = getenv("TSIMD_ENABLE_WSEED");
    const char *off = getenv("TSIMD_NO_WSEED");
    if (on && on[0] == '1' && !(off && off[0] == '1')) {
        if (tsimd_pcache_seed_vocab(w->cache, &w->t->vocab) != 0)
            return -1;
        w->seeded = 1;
    }
    return 0;
}

static int serial_pool_init(void) {
    const char *e = getenv("TSIMD_SERIAL_POOL_INIT");
    return e && e[0] == '1';
}

static tsimd_pool_t *pool_create(tsimd_tokenizer_t *t, int width,
                                size_t total_len) {
    if (width < 1) width = 1;
    tsimd_pool_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->t = t;
    p->width = width;
    p->n_spawned = 0;

    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->wake_cv, NULL);
    pthread_cond_init(&p->done_cv, NULL);
    pthread_cond_init(&p->ready_cv, NULL);

    p->workers = calloc((size_t)width, sizeof(worker_t));
    p->threads = calloc((size_t)(width > 1 ? width - 1 : 1),
                        sizeof(pthread_t));
    p->plo = calloc((size_t)width + 1, sizeof(size_t));
    p->phi = calloc((size_t)width + 1, sizeof(size_t));
    if (!p->workers || !p->threads || !p->plo || !p->phi) goto fail;

    const int serial = serial_pool_init();
    p->cache_bits = tsimd_pool_cache_bits(width, total_len);
    t->cache.shared_plong = 1;
    tsimd_plong_reserve(&t->cache, total_len >> 6);
    if (getenv("TSIMD_POOL_DBG"))
        fprintf(stderr, "tsimd pool_dbg: create width=%d total_len=%zu bits=%d\n",
                width, total_len, p->cache_bits);
    for (int i = 0; i < width; i++) {
        worker_t *w = &p->workers[i];
        w->pool = p;
        w->t = t;
        w->idx = i;
        if (tsimd_bpe_workspace_init(&w->ws, t->vocab.max_token_len) != 0)
            goto fail;

        if ((i == 0 || serial) && worker_cache_build(w, p->cache_bits) != 0) {
            w->init_rc = -1;
            goto fail;
        }
    }
    int spawned = 0;
    for (int i = 1; i < width; i++) {
        if (pthread_create(&p->threads[i - 1], NULL, pool_worker,
                           &p->workers[i]) != 0)
            break;
        spawned++;
    }
    p->n_spawned = spawned;

    pthread_mutex_lock(&p->mu);
    while (atomic_load(&p->ready) < spawned)
        pthread_cond_wait(&p->ready_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);

    for (int i = 0; i <= spawned; i++)
        if (p->workers[i].init_rc != 0) goto fail;
    return p;

fail:
    pool_destroy(p);
    return NULL;
}

void tsimd_tokenizer_release_pool(tsimd_tokenizer_t *t) {
    if (!t || !t->pool) return;
    pool_destroy(t->pool);
    t->pool = NULL;
}

void tsimd_tokenizer_cache_stats(const tsimd_tokenizer_t *t,
                                 size_t *hits, size_t *misses,
                                 size_t *inserts) {
    size_t h = 0, m = 0, ins = 0;
    if (t) {
        h = t->cache.hits;
        m = t->cache.misses;
        ins = t->cache.inserts;
        if (t->pool)
            for (int i = 0; i < t->pool->width; i++) {
                if (!t->pool->workers[i].cache) continue;
                h += t->pool->workers[i].cache->hits;
                m += t->pool->workers[i].cache->misses;
                ins += t->pool->workers[i].cache->inserts;
            }
    }
    if (hits) *hits = h;
    if (misses) *misses = m;
    if (inserts) *inserts = ins;
}

static long physical_cores(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return sysconf(_SC_NPROCESSORS_ONLN);
    char line[256];

    static uint64_t seen[64];
    long n = 0;
    long phys = -1, core = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "physical id : %ld", &phys) == 1) continue;
        if (sscanf(line, "core id : %ld", &core) != 1) continue;
        uint64_t h = (uint64_t)(phys * 64 + core) % 64;
        if (!(seen[h >> 6] & (1ULL << (h & 63)))) {
            seen[h >> 6] |= 1ULL << (h & 63);
            n++;
        }
    }
    fclose(f);
    if (n < 1) return sysconf(_SC_NPROCESSORS_ONLN);
    return n;
}

static int ensure_pool(tsimd_tokenizer_t *t, int max_threads,
                       size_t total_len) {
    long nproc = physical_cores();
    if (nproc < 1) nproc = 1;
    if (nproc > TSIMD_POOL_MAX) nproc = TSIMD_POOL_MAX;
    int want = max_threads < (int)nproc ? max_threads : (int)nproc;

    if (!t->pool) {
        for (int cpu = 0; cpu < TSIMD_PIN_MAX; cpu++)
            (void)tsimd_cpu_core_kind(cpu);
        t->cache.shared = 1;
        t->pool = pool_create(t, want, total_len);
        return t->pool ? 0 : -1;
    }
    if (want > t->pool->width) {
        pool_destroy(t->pool);
        t->cache.shared = 1;
        t->pool = pool_create(t, want, total_len);
        return t->pool ? 0 : -1;
    }
    return 0;
}

static idvec_t *pool_outs_get(tsimd_pool_t *p, size_t n) {
    if (n > p->outs_cache_n) {
        idvec_t *grown = tsimd_xrealloc(p->outs_cache,
                                        n * sizeof(idvec_t));
        if (!grown) return NULL;
        memset(grown + p->outs_cache_n, 0,
               (n - p->outs_cache_n) * sizeof(idvec_t));
        p->outs_cache = grown;
        p->outs_cache_n = n;
    }
    for (size_t i = 0; i < n; i++) p->outs_cache[i].count = 0;
    return p->outs_cache;
}

static int par_scratch_reserve(tsimd_pool_t *p, size_t n) {
    if (n > p->claims_cap) {
        size_t cap = n < 256 ? 256 : n;
        atomic_uchar *c = realloc(p->claims, cap);
        if (!c) return -1;
        p->claims = c;
        p->claims_cap = cap;
    }
    if (n > p->range_cap) {
        size_t cap = n < 256 ? 256 : n;
        size_t *w = tsimd_xrealloc(p->weights, cap * sizeof(size_t));
        if (!w) return -1;
        p->weights = w;
        p->range_cap = cap;
    }
    return 0;
}

static int pool_run(tsimd_pool_t *p, job_t *jobs, idvec_t *outs,
                    size_t n_jobs, int active) {
    p->jobs = jobs;
    p->outs = outs;
    p->n_jobs = n_jobs;
    atomic_store(&p->next, 0);
    atomic_store(&p->err, 0);
    atomic_store(&p->done, 0);

    pthread_mutex_lock(&p->mu);
    if (active < 0) active = 0;
    if (active > p->n_spawned) active = p->n_spawned;
    p->active = active;

    int participants = active + 1;
    p->par_mode = 0;
    p->n_range = participants;
    int mode = par_static_mode();
    if (mode > 0 && participants > 1 && n_jobs > 0 &&
        par_scratch_reserve(p, n_jobs) == 0) {
        for (size_t i = 0; i < n_jobs; i++)
            p->weights[i] = job_weight(&jobs[i]);
        int split_ok =
            (weighted_split_on()
                 ? par_split_caps(p->weights, n_jobs, participants, p->plo,
                                  p->phi)
                 : tsimd_par_split(p->weights, n_jobs, participants, p->plo,
                                   p->phi));
        if (split_ok) {
            memset(p->claims, 0, n_jobs);
            atomic_store(&p->steal_next, n_jobs);
            p->par_mode = mode == 2 ? 2 : 1;
        }
    }
    p->generation++;
    pthread_cond_broadcast(&p->wake_cv);
    pthread_mutex_unlock(&p->mu);

    if (p->width > 1) caller_pin();

    drain_jobs(&p->workers[0]);

    pthread_mutex_lock(&p->mu);
    while (atomic_load(&p->done) < (unsigned)p->n_spawned)
        pthread_cond_wait(&p->done_cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
    return atomic_load(&p->err) ? -1 : 0;
}

typedef struct {
    job_t *jobs;
    size_t count, cap;
} joblist_t;

static int joblist_push(joblist_t *jl, const job_t *job) {
    if (jl->count == jl->cap) {
        size_t cap = jl->cap ? jl->cap * 2 : 64;
        job_t *grown = tsimd_xrealloc(jl->jobs, cap * sizeof(job_t));
        if (!grown) return -1;
        jl->jobs = grown;
        jl->cap = cap;
    }
    jl->jobs[jl->count++] = *job;
    return 0;
}

static int added_hits_sparse(const tsimd_tokenizer_t *t, const uint8_t *data,
                             size_t len, size_t cap) {
    uint8_t first[256] = {0};
    for (size_t s = 0; s < t->added_count; s++)
        if (t->added[s].length > 0)
            first[(uint8_t)t->added[s].text[0]] = 1;
    size_t hits = 0;
    size_t pos = 0;
    while (pos < len) {
        if (!first[data[pos]]) {
            pos++;
            continue;
        }
        size_t idx = 0;
        size_t l = tsimd_tokenizer_added_len_at(t, data, len, pos, &idx);
        if (l) {
            hits++;
            if (hits > cap) return 0;
            pos += l;
        } else {
            pos++;
        }
    }
    return 1;
}

static int plan_added_jobs(joblist_t *jl, tsimd_tokenizer_t *t,
                           const uint8_t *data, size_t len, size_t target) {
    size_t maxlen = 0;
    for (size_t s = 0; s < t->added_count; s++)
        if (t->added[s].length > maxlen) maxlen = t->added[s].length;
    size_t pos = 0;
    while (len - pos > 2 * target) {
        size_t window = 2 * target + maxlen;
        if (window > len - pos) window = len - pos;
        tsimd_added_seam_list_t seams;
        if (tsimd_tokenizer_added_seams(t, data + pos, window, &seams) != 0)
            return -1;
        size_t boundary = 0;
        for (size_t i = 0; i < seams.count; i++)
            if (seams.items[i].off >= target) {
                boundary = seams.items[i].off;
                break;
            }
        tsimd_added_seam_list_free(&seams);
        if (boundary == 0 || pos + boundary >= len) break;
        job_t job = {data + pos, boundary, 0, JOB_TEXT, 0};
        if (joblist_push(jl, &job) != 0) return -1;
        pos += boundary;
    }
    if (len > pos) {
        job_t job = {data + pos, len - pos, 0, JOB_TEXT, 0};
        if (joblist_push(jl, &job) != 0) return -1;
    }
    return 0;
}

static int plan_segment(joblist_t *jl, tsimd_tokenizer_t *t,
                        const uint8_t *data, size_t start, size_t end,
                        size_t doc, size_t target) {
    size_t pos = start;
    while (end - pos > 2 * target) {
        size_t hint = pos + target - start;
        long p = tsimd_pretok_safe_split(t->scheme, data + start,
                                        end - start, hint, pos - start);
        if (p < 0) break;
        size_t abs = start + (size_t)p;
        job_t job = {data + pos, abs - pos, doc, JOB_TEXT, 0};
        if (joblist_push(jl, &job) != 0) return -1;
        pos = abs;
    }
    if (end > pos) {
        job_t job = {data + pos, end - pos, doc, JOB_TEXT, 0};
        if (joblist_push(jl, &job) != 0) return -1;
    }
    return 0;
}

static int plan_buffer(joblist_t *jl, tsimd_tokenizer_t *t,
                       const uint8_t *data, size_t len, size_t doc,
                       size_t target) {
    uint8_t first[256] = {0};
    for (size_t s = 0; s < t->added_count; s++)
        if (t->added[s].length > 0)
            first[(uint8_t)t->added[s].text[0]] = 1;

    if (t->added_count == 0) {

        return plan_segment(jl, t, data, 0, len, doc, target);
    }

    uint8_t fbytes[256];
    int nfb = 0;
    for (int b = 0; b < 256; b++)
        if (first[b]) fbytes[nfb++] = (uint8_t)b;

    size_t dbg_stops = 0;

    const int use_simd_scan = nfb >= 2 && nfb <= 8;
    size_t seg = 0, pos = 0;
    while (pos < len) {
        size_t cand = len;
        if (use_simd_scan) {
            __m128i vb[8];
            for (int k = 0; k < nfb; k++)
                vb[k] = _mm_set1_epi8((char)fbytes[k]);
            size_t p = pos;
            while (p + 16 <= len) {
                __m128i v = _mm_loadu_si128(
                    (const __m128i *)(const void *)(data + p));
                __m128i m = _mm_cmpeq_epi8(v, vb[0]);
                for (int k = 1; k < nfb; k++)
                    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, vb[k]));
                int mask = _mm_movemask_epi8(m);
                if (mask) {
                    cand = p + (size_t)__builtin_ctz((unsigned)mask);
                    break;
                }
                p += 16;
            }
            for (; p < len && cand == len; p++)
                if (first[data[p]]) cand = p;
        } else {
            const uint8_t *hit = NULL;
            for (int k = 0; k < nfb; k++) {
                const uint8_t *h = memchr(data + pos, fbytes[k], len - pos);
                if (h && (!hit || h < hit)) hit = h;
            }
            if (hit) cand = (size_t)(hit - data);
        }
        if (cand >= len) break;
        dbg_stops++;
        pos = cand;
        size_t added_index = 0;
        size_t hit_len = tsimd_tokenizer_added_len_at(t, data, len, pos,
                                                     &added_index);
        if (hit_len == 0) {
            pos++;
            continue;
        }
        if (pos > seg &&
            plan_segment(jl, t, data, seg, pos, doc, target) != 0)
            return -1;
        const tsimd_added_token_t *at = &t->added[added_index];
        job_t job;
        job.kind = JOB_SPECIAL;
        job.data = NULL;
        job.len = 0;
        job.special_id = at->id;
        job.doc = doc;
        if (joblist_push(jl, &job) != 0) return -1;
        pos += hit_len;
        seg = pos;
    }
    if (len > seg && plan_segment(jl, t, data, seg, len, doc, target) != 0)
        return -1;
    if (getenv("TSIMD_PLAN_DBG"))
        fprintf(stderr, "tsimd plan: %zu B, nfb %d, stops %zu (%.0f/MiB), "
                        "added %zu\n", len, nfb, dbg_stops,
                (double)dbg_stops * 1048576.0 / (double)(len ? len : 1),
                t->added_count);
    return 0;
}

static size_t pick_target(size_t len, int n_threads) {
    static long over = -2;
    if (over == -2) {
        const char *e = getenv("TSIMD_PAR_OVERSUBSCRIBE");
        over = TSIMD_PAR_OVERSUBSCRIBE;
        if (e && e[0]) {
            long p = atol(e);
            if (p >= 4 && p <= 256) over = p;
        }
    }
    size_t target = len / (size_t)(n_threads * (long)over);
    if (target < TSIMD_PAR_MIN_PIECE) target = TSIMD_PAR_MIN_PIECE;
    if (target > (size_t)8 << 20) target = (size_t)8 << 20;
    return target;
}

static int drain_sequential(tsimd_tokenizer_t *t, const joblist_t *jl,
                            idvec_t *flat, size_t *job_counts) {

    worker_t w;
    memset(&w, 0, sizeof(w));
    w.t = t;
    w.cache = &t->cache;

    if (tsimd_bpe_workspace_init(&w.ws, t->vocab.max_token_len) != 0)
        return -1;
    int rc = 0;
    for (size_t i = 0; i < jl->count && rc == 0; i++) {
        size_t before = flat->count;
        rc = run_one_job(&w, &jl->jobs[i], flat);
        job_counts[i] = flat->count - before;
    }
    phase_report("drain_sequential", w.ph_fill, w.ph_serve, w.ph_pieces);
    phase_report_mix("drain_sequential", w.ph_pieces, w.ph_hits, w.ph_big,
                     w.ph_long, w.ph_longhit, w.ph_bpe);
    tsimd_bpe_workspace_free(&w.ws);
    return rc;
}

static int parallel_pool_path_ex(tsimd_tokenizer_t *t, const uint8_t *data,
                                 size_t len, int max_threads,
                                 tsimd_token_sequence_t *out, uint32_t *buf,
                                 size_t cap, size_t *count, int nfc_runs);

int tsimd_tokenizer_encode_parallel(tsimd_tokenizer_t *t,
                                   const uint8_t *data, size_t len,
                                   int max_threads,
                                   tsimd_token_sequence_t *out) {
    tsimd_serve_env_refresh();
    tsimd_pcache_upgrade_for(t, len);
    tsimd_memo_size_for(t, len);
    if (max_threads > TSIMD_POOL_MAX) max_threads = TSIMD_POOL_MAX;
    if (max_threads <= 1 || len < TSIMD_PAR_MIN_LEN)
        return tsimd_tokenizer_encode(t, data, len, out);

    if (t->norm == TSIMD_NORM_NFC) {
        size_t nlen = 0;
        if (!getenv("TSIMD_NFC_MAT")) {
            int rc = parallel_pool_path_ex(t, data, len, max_threads, out,
                                           NULL, 0, NULL, 1);
            if (rc == 0) return 0;
        }
        uint8_t *nbuf = tsimd_nfc_normalize_all_tl(t, data, len,
                                                   max_threads, &nlen);
        if (!nbuf)
            return tsimd_tokenizer_encode(t, data, len, out);
        return parallel_pool_path_ex(t, nbuf, nlen, max_threads, out, NULL, 0,
                                     NULL, 0);
    }
    if (t->norm != TSIMD_NORM_NONE)
        return tsimd_tokenizer_encode(t, data, len, out);
    return parallel_pool_path_ex(t, data, len, max_threads, out, NULL, 0, NULL,
                                 0);
}

int tsimd_tokenizer_encode_parallel_into(tsimd_tokenizer_t *t,
                                        const uint8_t *data, size_t len,
                                        int max_threads, uint32_t *buf,
                                        size_t cap, size_t *count) {
    tsimd_serve_env_refresh();
    tsimd_pcache_upgrade_for(t, len);
    tsimd_memo_size_for(t, len);
    if (!buf || !count) return -1;
    if (cap < tsimd_tokenizer_encode_bound(t, len)) return -1;
    *count = 0;
    if (max_threads > TSIMD_POOL_MAX) max_threads = TSIMD_POOL_MAX;
    if (max_threads <= 1 || len < TSIMD_PAR_MIN_LEN)
        return tsimd_tokenizer_encode_into_raw(t, data, len, buf, cap, count);
    if (t->norm == TSIMD_NORM_NFC) {
        size_t nlen = 0;
        if (!getenv("TSIMD_NFC_MAT")) {
            int rc = parallel_pool_path_ex(t, data, len, max_threads, NULL, buf,
                                           cap, count, 1);
            if (rc == 0) return 0;
        }
        uint8_t *nbuf = tsimd_nfc_normalize_all_tl(t, data, len, max_threads,
                                                   &nlen);
        if (!nbuf)
            return tsimd_tokenizer_encode_into_raw(t, data, len, buf, cap,
                                                   count);
        return parallel_pool_path_ex(t, nbuf, nlen, max_threads, NULL, buf, cap,
                                     count, 0);
    }
    if (t->norm != TSIMD_NORM_NONE)
        return tsimd_tokenizer_encode_into_raw(t, data, len, buf, cap, count);
    return parallel_pool_path_ex(t, data, len, max_threads, NULL, buf, cap,
                                 count, 0);
}

static int parallel_pool_path_ex(tsimd_tokenizer_t *t, const uint8_t *data,
                                 size_t len, int max_threads,
                                 tsimd_token_sequence_t *out, uint32_t *buf,
                                 size_t cap, size_t *count, int nfc_runs) {

    if (t->scheme == TSIMD_PRETOK_SPM || t->scheme == TSIMD_PRETOK_GEM4) {
        t->cache.shared_plong = 1;
        tsimd_plong_reserve(&t->cache, len >> 6);
        if (t->added_count == 0)
            return out ? tsimd_tokenizer_encode_whole_chain_par(t, data, len,
                                                                max_threads,
                                                                out)
                       : tsimd_tokenizer_encode_whole_chain_par_into(
                             t, data, len, max_threads, buf, cap, count);
        if (added_hits_sparse(t, data, len, 4096))
            return out ? tsimd_tokenizer_encode_whole_chain_par(t, data, len,
                                                                max_threads,
                                                                out)
                       : tsimd_tokenizer_encode_whole_chain_par_into(
                             t, data, len, max_threads, buf, cap, count);
    }
    if (ensure_pool(t, max_threads, len) != 0) return -1;
    joblist_t jl = {NULL, 0, 0};
    size_t target = pick_target(len, max_threads);
    uint64_t dbg_t0 = phase_stats_on() ? ph_rdtsc() : 0;
    int plan_rc;
    if (nfc_runs) {
        plan_rc = 0;
        size_t pos = 0;
        const char *nt = getenv("TSIMD_NFC_TARGET");
        if (nt && nt[0]) {
            long v = strtol(nt, NULL, 10);
            if (v > 0) target = (size_t)v;
            else if (v < 0) target = len;
        }
        while (len - pos > 2 * target) {
            size_t seam = tsimd_nfc_seam_at(t, data, len, pos + target);
            if (seam == 0 || seam <= pos) break;
            job_t nj = {data + pos, seam - pos, 0, JOB_NFC, 0};
            if (joblist_push(&jl, &nj) != 0) { plan_rc = -1; break; }
            pos = seam;
        }
        if (plan_rc == 0 && len > pos) {
            job_t nj = {data + pos, len - pos, 0, JOB_NFC, 0};
            if (joblist_push(&jl, &nj) != 0) plan_rc = -1;
        }
    } else if (t->scheme == TSIMD_PRETOK_SPM ||
               t->scheme == TSIMD_PRETOK_GEM4) {
        plan_rc = plan_added_jobs(&jl, t, data, len, target);
    } else {
        plan_rc = plan_buffer(&jl, t, data, len, 0, target);
    }
    if (plan_rc != 0) {
        free(jl.jobs);
        return -1;
    }
    uint64_t dbg_t1 = phase_stats_on() ? ph_rdtsc() : 0;

    idvec_t *outs = jl.count ? pool_outs_get(t->pool, jl.count) : NULL;
    if (jl.count && !outs) {
        free(jl.jobs);
        return -1;
    }
    int rc = jl.count == 0
                 ? 0
                 : pool_run(t->pool, jl.jobs, outs, jl.count,
                            max_threads - 1);
    uint64_t dbg_t2 = phase_stats_on() ? ph_rdtsc() : 0;
    phase_report_pool("encode_parallel", t->pool);

    size_t total = 0;
    for (size_t i = 0; i < jl.count; i++) total += outs[i].count;
    uint32_t *ids = buf;
    if (rc == 0 && total > 0) {
        if (!ids) {
            size_t bytes = total * sizeof(uint32_t);
            if (bytes >= (2u << 20)) {
                size_t rounded =
                    (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
                ids = (uint32_t *)aligned_alloc(2u << 20, rounded);
#if defined(MADV_HUGEPAGE)
                if (ids) madvise(ids, bytes, MADV_HUGEPAGE);
#endif
            }
            if (!ids) ids = tsimd_xmalloc(bytes);
        } else if (total > cap) {
            rc = -1;
        }
        if (rc == 0 && max_threads > 1 && jl.count > 0) {
            int ng = max_threads - 1;
            if (ng < 1) ng = 1;
            if ((size_t)ng > jl.count) ng = (int)jl.count;
            job_t *gjobs = calloc((size_t)ng, sizeof(job_t));
            if (gjobs) {
                for (int g = 0; g < ng; g++) {
                    gjobs[g].kind = JOB_GATHER;
                    gjobs[g].gdst = ids;
                    gjobs[g].gsrc = outs;
                    gjobs[g].gj0 = (size_t)g * jl.count / (size_t)ng;
                    gjobs[g].gj1 = (size_t)(g + 1) * jl.count / (size_t)ng;
                    if (g + 1 == ng) gjobs[g].gj1 = jl.count;
                }
                rc = pool_run(t->pool, gjobs, outs, (size_t)ng,
                              max_threads - 1);
                free(gjobs);
            }
        } else {
            size_t off = 0;
            for (size_t i = 0; rc == 0 && i < jl.count; i++) {
                copy_stream_u32(ids + off, outs[i].ids, outs[i].count);
                off += outs[i].count;
            }
        }
    }
    free(jl.jobs);

    if (rc != 0) {
        if (!buf) free(ids);
        return -1;
    }
    if (out) {
        out->ids = ids;
        out->count = total;
        out->cap = total;
    } else if (count) {
        *count = total;
    }
    if (dbg_t0 && getenv("TSIMD_POOL_DBG"))
        fprintf(stderr,
                "tsimd pool_dbg: jobs %zu plan %.2f Mcyc run %.2f Mcyc "
                "copy %.2f Mcyc\n",
                jl.count, (double)(dbg_t1 - dbg_t0) / 1e6,
                (double)(dbg_t2 - dbg_t1) / 1e6,
                (double)(ph_rdtsc() - dbg_t2) / 1e6);
    return 0;
}

int tsimd_tokenizer_encode_batch(tsimd_tokenizer_t *t,
                                const tsimd_input_span_t *docs,
                                size_t n_docs,
                                tsimd_token_batch_t *out,
                                int max_threads) {
    tsimd_serve_env_refresh();
    memset(out, 0, sizeof(*out));
    out->offsets = tsimd_xmalloc((n_docs + 1) * sizeof(size_t));
    if (!out->offsets) return -1;
    if (getenv("TSIMD_BATCH_DBG"))
        fprintf(stderr, "tsimd batch dbg: enter n_docs=%zu norm=%d scheme=%d\n",
                n_docs, t->norm, (int)t->scheme);

    size_t total_len = 0;
    for (size_t d = 0; d < n_docs; d++) total_len += docs[d].length;
    tsimd_pcache_upgrade_for(t, total_len);
    tsimd_memo_size_for(t, total_len);

    uint32_t *flat = NULL;

    uint8_t **ncopies = NULL;
    size_t *nlens = NULL;
    if (t->norm == TSIMD_NORM_NFC) {
        ncopies = tsimd_xmalloc(n_docs * sizeof(uint8_t *));
        nlens = tsimd_xmalloc(n_docs * sizeof(size_t));
        if (!ncopies || !nlens) {
            tsimd_xfree(ncopies); tsimd_xfree(nlens);
            free(out->offsets);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        int fellback = 0;
        size_t nset = 0;
        for (size_t d = 0; d < n_docs; d++) {
            ncopies[d] = tsimd_nfc_normalize_all(t, docs[d].data,
                                                 docs[d].length,
                                                 max_threads, &nlens[d]);
            if (!ncopies[d]) {
                fellback = 1;
                break;
            }
            nset++;
        }
        if (fellback) {
            for (size_t d = 0; d < nset; d++) tsimd_xfree(ncopies[d]);
            tsimd_xfree(ncopies);
            tsimd_xfree(nlens);
            ncopies = NULL;
            nlens = NULL;
        }
    }
    if (t->scheme == TSIMD_PRETOK_SPM || t->scheme == TSIMD_PRETOK_GEM4 ||
        (t->norm != TSIMD_NORM_NONE && ncopies == NULL)) {
        flat = tsimd_xmalloc((total_len + 64) * sizeof(uint32_t));
        if (!flat) {
            free(out->offsets);
            if (ncopies) {
                for (size_t d = 0; d < n_docs; d++)
                    tsimd_xfree(ncopies[d]);
                tsimd_xfree(ncopies);
                tsimd_xfree(nlens);
            }
            return -1;
        }
        size_t off = 0;
        out->offsets[0] = 0;
        for (size_t d = 0; d < n_docs; d++) {
            tsimd_token_sequence_t seq;
            if (tsimd_tokenizer_encode(t, docs[d].data, docs[d].length,
                                       &seq) != 0) {
                free(flat);
                free(out->offsets);
                if (ncopies) {
                    for (size_t e = 0; e < n_docs; e++)
                        tsimd_xfree(ncopies[e]);
                    tsimd_xfree(ncopies);
                    tsimd_xfree(nlens);
                }
                return -1;
            }
            memcpy(flat + off, seq.ids, seq.count * sizeof(uint32_t));
            off += seq.count;
            out->offsets[d + 1] = off;
            tsimd_token_sequence_free(&seq);
        }
        out->ids = flat;
        out->id_count = off;
        out->n_docs = n_docs;
        if (getenv("TSIMD_BATCH_DBG"))
            fprintf(stderr, "tsimd batch dbg: per-doc path, count=%zu off1=%zu\n",
                    off, n_docs ? out->offsets[1] : 0);
        if (ncopies) {
            for (size_t d = 0; d < n_docs; d++) tsimd_xfree(ncopies[d]);
            tsimd_xfree(ncopies);
            tsimd_xfree(nlens);
        }
        return 0;
    }

    const int bp = phase_stats_on();
    uint64_t bp_t0 = bp ? ph_rdtsc() : 0;
    uint64_t bp_enc = 0, bp_gath = 0, bp_end = 0, bp_t1 = 0;
    joblist_t jl = {NULL, 0, 0};
    if (max_threads > TSIMD_POOL_MAX) max_threads = TSIMD_POOL_MAX;
    size_t target = pick_target(total_len, max_threads > 1 ? max_threads : 1);
    int rc = 0;

    uint64_t plan_dbg_max = 0;
    size_t plan_dbg_max_doc = 0, plan_dbg_max_off = 0;
    for (size_t d = 0; d < n_docs && rc == 0; d++) {
        uint64_t d0 = bp ? ph_rdtsc() : 0;
        rc = plan_buffer(&jl, t,
                         ncopies ? ncopies[d] : docs[d].data,
                         ncopies ? nlens[d] : docs[d].length, d, target);
        if (bp) {
            uint64_t dc = ph_rdtsc() - d0;
            if (dc > plan_dbg_max) {
                plan_dbg_max = dc;
                plan_dbg_max_doc = d;
                plan_dbg_max_off = jl.count;
            }
        }
    }
    if (getenv("TSIMD_PLAN_DBG") && bp && plan_dbg_max)
        fprintf(stderr, "tsimd plan: %zu docs, max doc %zu = %.2f Mcyc "
                        "(jobs %zu)\n", n_docs, plan_dbg_max_doc,
                (double)plan_dbg_max / 1e6, plan_dbg_max_off);

    if (rc == 0) {
        flat = tsimd_xmalloc((total_len + 64) * sizeof(uint32_t));
        if (!flat) rc = -1;
#ifdef MADV_HUGEPAGE
        else if (total_len >= (size_t)2 << 20)
            madvise(flat, (total_len + 64) * sizeof(uint32_t),
                    MADV_HUGEPAGE);
#endif
    }
    idvec_t flatv = {flat, 0, total_len + 64};
    idvec_t *outs = NULL;
    size_t *job_counts = NULL;
    size_t *doc_tot = NULL;
    size_t *doc_off = NULL;
    if (rc == 0) {
        job_counts = calloc(jl.count ? jl.count : 1, sizeof(size_t));
        if (!job_counts) rc = -1;
    }
    size_t encode_count = jl.count;
    if (rc == 0) {
        if (max_threads > 1 && total_len >= TSIMD_PAR_MIN_LEN) {
            if (ensure_pool(t, max_threads, total_len) != 0)
                rc = -1;
            else if (jl.count > 0) {

                outs = pool_outs_get(t->pool, jl.count);
                bp_t1 = bp ? ph_rdtsc() : 0;
                rc = outs ? pool_run(t->pool, jl.jobs, outs, encode_count,
                                     max_threads - 1)
                          : -1;
                phase_report_pool("encode_batch", t->pool);
                bp_enc = bp ? ph_rdtsc() : 0;
                for (size_t i = 0; i < encode_count; i++)
                    job_counts[i] = outs[i].count;
            }
        } else if (jl.count > 0) {
            rc = drain_sequential(t, &jl, &flatv, job_counts);
        }
    }
    if (rc == 0 && outs != NULL && encode_count > 0) {

        int ng = max_threads - 1;
        if (ng < 1) ng = 1;
        if ((size_t)ng > encode_count) ng = (int)encode_count;
        job_t *gjobs = calloc((size_t)ng, sizeof(job_t));
        if (!gjobs) rc = -1;
        else {
            for (int g = 0; g < ng; g++) {
                gjobs[g].kind = JOB_GATHER;
                gjobs[g].gdst = flat;
                gjobs[g].gsrc = outs;
                gjobs[g].gj0 = (size_t)g * encode_count / (size_t)ng;
                gjobs[g].gj1 = (size_t)(g + 1) * encode_count / (size_t)ng;
                if (g + 1 == ng) gjobs[g].gj1 = encode_count;
            }
            rc = pool_run(t->pool, gjobs, outs, (size_t)ng,
                          max_threads - 1);
            free(gjobs);
        }
    }

    bp_gath = bp ? ph_rdtsc() : 0;
    if (rc == 0) {
        doc_tot = calloc(n_docs ? n_docs : 1, sizeof(size_t));
        doc_off = calloc(n_docs ? n_docs : 1, sizeof(size_t));
        if (!doc_tot || !doc_off) rc = -1;
    }
    if (rc == 0) {
        for (size_t i = 0; i < encode_count; i++)
            doc_tot[jl.jobs[i].doc] += job_counts[i];
        out->offsets[0] = 0;
        for (size_t d = 0; d < n_docs; d++)
            out->offsets[d + 1] = out->offsets[d] + doc_tot[d];
        out->id_count = out->offsets[n_docs];
        out->n_docs = n_docs;
        out->ids = flat;
        if (getenv("TSIMD_BATCH_DBG"))
            fprintf(stderr,
                    "tsimd batch dbg: parallel path, jobs=%zu count=%zu "
                    "doc_tot[0]=%zu off1=%zu\n",
                    encode_count, out->id_count, n_docs ? doc_tot[0] : 0,
                    n_docs ? out->offsets[1] : 0);

        (void)doc_off;
    }

    free(doc_tot);
    free(doc_off);
    free(job_counts);

    free(jl.jobs);
    if (ncopies) {
        for (size_t d = 0; d < n_docs; d++) tsimd_xfree(ncopies[d]);
        tsimd_xfree(ncopies);
        tsimd_xfree(nlens);
    }
    if (rc != 0) {
        free(flat);
        free(out->offsets);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    bp_end = bp ? ph_rdtsc() : 0;
    if (bp && bp_t1 && bp_enc && bp_end > bp_t0)
        fprintf(stderr,
                "tsimd batch: %zu jobs, %zu docs — plan %.2f, encode "
                "%.2f, gather %.2f, stitch %.2f Mcyc (total %.2f)\n",
                jl.count, n_docs, (double)(bp_t1 - bp_t0) / 1e6,
                (double)(bp_enc - bp_t1) / 1e6,
                (double)(bp_gath - bp_enc) / 1e6,
                (double)(bp_end - bp_gath) / 1e6,
                (double)(bp_end - bp_t0) / 1e6);
    return 0;

}

void tsimd_token_batch_free(tsimd_token_batch_t *batch) {
    free(batch->ids);
    free(batch->offsets);
    memset(batch, 0, sizeof(*batch));
}
