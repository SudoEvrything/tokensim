#include "bpe.h"
#include "trie.h"
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include "unicode.h"
#include "allocator.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int pcache_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_DISABLE_PCACHE");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static int unit_dbg(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_UNIT_DBG");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

#define TSIMD_NIL UINT32_MAX

int tsimd_bpe_workspace_init(tsimd_bpe_workspace_t *ws, size_t max_token_len) {
    memset(ws, 0, sizeof(*ws));
    size_t cap = 64;
    while (cap < max_token_len) cap <<= 1;
    ws->ids = tsimd_xmalloc(cap * sizeof(uint32_t));
    ws->next = tsimd_xmalloc(cap * sizeof(uint32_t));
    ws->prev = tsimd_xmalloc(cap * sizeof(uint32_t));
    ws->capacity = (uint32_t)cap;
    ws->out = tsimd_xmalloc(1024 * sizeof(uint32_t));
    ws->out_capacity = 1024;
    ws->out_foreign = 0;
    ws->heap = NULL;
    ws->heap_count = 0;
    ws->heap_cap = 0;
    ws->lin_ranks = tsimd_xmalloc(cap * sizeof(uint32_t));
    ws->lin_ids = tsimd_xmalloc(cap * sizeof(uint32_t));
    ws->lin_cap = (uint32_t)cap;
    ws->cand_state = tsimd_xmalloc(cap * sizeof(uint64_t));
    ws->cand_cap = (uint32_t)cap;
    return 0;
}

void tsimd_bpe_workspace_free(tsimd_bpe_workspace_t *ws) {
    free(ws->nfc_buf);
    free(ws->ids);
    free(ws->next);
    free(ws->prev);
    free(ws->out);
    free(ws->heap);
    free(ws->lin_ranks);
    free(ws->lin_ids);
    free(ws->cand_state);
    free(ws->bq_head);
    free(ws->bq_tail);
    free(ws->bq_l0);
    free(ws->bq_l1);
    free(ws->bq_l2);
    free(ws->bq_entries);
    for (int b = 0; b < 2; b++)
        for (int t = 0; t < 16; t++) {
            free(ws->uni_mcache[b][t]);
            free(ws->uni_ccache[b][t]);
        }
    free(ws->uni_seg);
    free(ws->bq_rank);
    free(ws->bq_gmask);
    free(ws->bq_touched);
    free(ws->uni_st);
    memset(ws, 0, sizeof(*ws));
}

static int ensure_capacity(tsimd_bpe_workspace_t *ws, uint32_t need) {
    if (need <= ws->capacity) return 0;
    uint32_t cap = ws->capacity;
    while (cap < need) cap <<= 1;
    ws->ids = tsimd_xrealloc(ws->ids, (size_t)cap * sizeof(uint32_t));
    ws->next = tsimd_xrealloc(ws->next, (size_t)cap * sizeof(uint32_t));
    ws->prev = tsimd_xrealloc(ws->prev, (size_t)cap * sizeof(uint32_t));
    ws->capacity = cap;
    return 0;
}

static int ensure_linear_scratch(tsimd_bpe_workspace_t *ws, uint32_t need) {
    if (need <= ws->lin_cap) return 0;
    uint32_t cap = ws->lin_cap ? ws->lin_cap : 64;
    while (cap < need) cap <<= 1;
    ws->lin_ranks =
        tsimd_xrealloc(ws->lin_ranks, (size_t)cap * sizeof(uint32_t));
    ws->lin_ids = tsimd_xrealloc(ws->lin_ids, (size_t)cap * sizeof(uint32_t));
    ws->lin_cap = cap;
    return 0;
}

static int ensure_cand_state(tsimd_bpe_workspace_t *ws, uint32_t need) {
    if (need <= ws->cand_cap) return 0;
    uint32_t cap = ws->cand_cap ? ws->cand_cap : 64;
    while (cap < need) cap <<= 1;
    ws->cand_state =
        tsimd_xrealloc(ws->cand_state, (size_t)cap * sizeof(uint64_t));
    ws->cand_cap = cap;
    return 0;
}

static inline void cand_store(tsimd_bpe_workspace_t *ws, uint32_t p,
                              uint32_t rank, uint32_t id) {
    ws->cand_state[p] = ((uint64_t)rank << 32) | id;
}

static inline int cand_match(const tsimd_bpe_workspace_t *ws, uint32_t p,
                             uint32_t rank, uint32_t *id_out) {
    uint64_t s = ws->cand_state[p];
    if ((uint32_t)(s >> 32) != rank) return 0;
    *id_out = (uint32_t)s;
    return 1;
}

int tsimd_ws_out_ensure(tsimd_bpe_workspace_t *ws, size_t need) {
    if (need <= ws->out_capacity) return 0;
    size_t cap = ws->out_capacity ? ws->out_capacity : 1024;
    while (cap < need) cap <<= 1;
    size_t bytes = cap * sizeof(uint32_t);
    if (bytes >= (2u << 20)) {
        size_t rounded = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
        uint32_t *p = (uint32_t *)aligned_alloc(2u << 20, rounded);
        if (p) {
            if (ws->out_count)
                memcpy(p, ws->out, ws->out_count * sizeof(uint32_t));
            if (!ws->out_foreign) tsimd_xfree(ws->out);
            ws->out = p;
            ws->out_capacity = cap;
            ws->out_foreign = 0;
#if defined(MADV_HUGEPAGE)
            madvise(p, bytes, MADV_HUGEPAGE);
#endif
            return 0;
        }
    }
    if (ws->out_foreign) {
        uint32_t *p = tsimd_xmalloc(bytes);
        if (!p) return -1;
        if (ws->out_count) memcpy(p, ws->out, ws->out_count * sizeof(uint32_t));
        ws->out = p;
    } else {
        ws->out = ws->out ? tsimd_xrealloc(ws->out, bytes) : tsimd_xmalloc(bytes);
    }
    ws->out_capacity = cap;
    ws->out_foreign = 0;
    return 0;
}

static int out_push(tsimd_bpe_workspace_t *ws, uint32_t id) {
    if (ws->out_count == ws->out_capacity) {
        if (tsimd_ws_out_ensure(ws, ws->out_count + 1) != 0) return -1;
    }
    ws->out[ws->out_count++] = id;
    return 0;
}

#define TSIMD_BPE_HEAP_LINEAR 64

static atomic_uint heap_linear_state = 0;

uint32_t tsimd_bpe_heap_linear(void) {
    uint32_t s = atomic_load_explicit(&heap_linear_state,
                                      memory_order_relaxed);
    if (s == 0) {
        long k = TSIMD_BPE_HEAP_LINEAR;
        const char *e = getenv("TSIMD_BPE_HEAP_LINEAR");
        if (e && e[0]) {
            char *end = NULL;
            long v = strtol(e, &end, 10);
            if (end && end != e && v >= 0 && v <= 100000000L) k = v;
        }
        s = (uint32_t)k + 1;
        atomic_store_explicit(&heap_linear_state, s, memory_order_relaxed);
    }
    return s - 1;
}

void tsimd_bpe_set_heap_linear(uint32_t limit) {
    atomic_store_explicit(&heap_linear_state, limit + 1,
                          memory_order_relaxed);
}

static atomic_uint pop_probe_state = 0;

uint32_t tsimd_bpe_pop_probe(void) {
    uint32_t s = atomic_load_explicit(&pop_probe_state, memory_order_relaxed);
    if (s == 0) {
        long k = 0;
        const char *e = getenv("TSIMD_BPE_POP_PROBE");
        if (e && e[0]) {
            char *end = NULL;
            long v = strtol(e, &end, 10);
            if (end && end != e && v >= 0 && v <= 1) k = v;
        }
        s = (uint32_t)k + 1;
        atomic_store_explicit(&pop_probe_state, s, memory_order_relaxed);
    }
    return s - 1;
}

void tsimd_bpe_set_pop_probe(uint32_t on) {
    atomic_store_explicit(&pop_probe_state, on ? 2u : 1u,
                          memory_order_relaxed);
}

static atomic_uint pop_scan_state = 0;

uint32_t tsimd_bpe_pop_scan(void) {
    uint32_t s = atomic_load_explicit(&pop_scan_state, memory_order_relaxed);
    if (s == 0) {
        long k = 1;
        const char *e = getenv("TSIMD_BPE_POP_SCAN");
        if (e && e[0]) {
            char *end = NULL;
            long v = strtol(e, &end, 10);
            if (end && end != e && v >= 0 && v <= 2) k = v;
        }
        s = (uint32_t)k + 1;
        atomic_store_explicit(&pop_scan_state, s, memory_order_relaxed);
    }
    return s - 1;
}

void tsimd_bpe_set_pop_scan(uint32_t mode) {
    atomic_store_explicit(&pop_scan_state, (uint32_t)mode + 1u,
                          memory_order_relaxed);
}

static atomic_uint queue_state = 0;

uint32_t tsimd_bpe_queue(void) {
    uint32_t s = atomic_load_explicit(&queue_state, memory_order_relaxed);
    if (s == 0) {

        long k = 2;
        const char *e = getenv("TSIMD_BPE_QUEUE");
        if (e && e[0]) {
            char *end = NULL;
            long v = strtol(e, &end, 10);
            if (end && end != e && v >= 0 && v <= 2) k = v;
        }
        s = (uint32_t)k + 1;
        atomic_store_explicit(&queue_state, s, memory_order_relaxed);
    }
    return s - 1;
}

void tsimd_bpe_set_queue(uint32_t mode) {
    atomic_store_explicit(&queue_state,
                          mode > 2 ? 3u : mode + 1u, memory_order_relaxed);
}

static atomic_uint bqgp_state = 0;

uint32_t tsimd_bpe_bqgp(void) {
    uint32_t s = atomic_load_explicit(&bqgp_state, memory_order_relaxed);
    if (s == 0) {
        long k = 0;
        const char *e = getenv("TSIMD_BPE_BQGP");
        if (e && e[0]) {
            char *end = NULL;
            long v = strtol(e, &end, 10);
            if (end && end != e && v >= 0 && v <= 1) k = v;
        }
        s = (uint32_t)k + 1;
        atomic_store_explicit(&bqgp_state, s, memory_order_relaxed);
    }
    return s - 1;
}

void tsimd_bpe_set_bqgp(uint32_t mode) {
    atomic_store_explicit(&bqgp_state, mode ? 2u : 1u, memory_order_relaxed);
}

static atomic_uint bucket_runs = 0;

uint32_t tsimd_bpe_bucket_runs(void) {
    return atomic_load_explicit(&bucket_runs, memory_order_relaxed);
}

static inline void heap_min2(uint64_t va, size_t ia, uint64_t vb, size_t ib,
                             uint64_t *vout, size_t *iout) {
    uint64_t better = vb < va ? ~(uint64_t)0 : 0;
    *vout = (va & ~better) | (vb & better);
    *iout = (ia & ~(size_t)better) | (ib & (size_t)better);
}

static int heap_less(uint64_t a, uint64_t b) { return a < b; }

static int heap_push(tsimd_bpe_workspace_t *ws, uint32_t rank, uint32_t pos) {
    if (ws->heap_count == ws->heap_cap) {
        size_t cap = ws->heap_cap ? ws->heap_cap * 2 : 256;
        uint64_t *grown = tsimd_xrealloc(ws->heap, cap * sizeof(uint64_t));
        if (!grown) return -1;
        ws->heap = grown;
        ws->heap_cap = cap;
    }
    size_t i = ws->heap_count++;
    ws->heap[i] = ((uint64_t)rank << 32) | pos;

    while (i > 0) {
        size_t parent = (i - 1) / 8;
        if (!heap_less(ws->heap[i], ws->heap[parent])) break;
        uint64_t tmp = ws->heap[i];
        ws->heap[i] = ws->heap[parent];
        ws->heap[parent] = tmp;
        i = parent;
    }
    return 0;
}

static uint64_t heap_pop(tsimd_bpe_workspace_t *ws) {
    uint64_t top = ws->heap[0];
    ws->heap[0] = ws->heap[--ws->heap_count];
    size_t i = 0;
    uint32_t scan = tsimd_bpe_pop_scan();
    for (;;) {
        size_t c0 = 8 * i + 1;
        if (c0 >= ws->heap_count) break;
        size_t cend = c0 + 8;
        if (cend > ws->heap_count) cend = ws->heap_count;

        size_t best = c0;
        uint64_t bestv = ws->heap[c0];
        if (scan && cend - c0 == 8) {
            uint64_t v1 = ws->heap[c0 + 1];
            uint64_t v2 = ws->heap[c0 + 2];
            uint64_t v3 = ws->heap[c0 + 3];
            uint64_t v4 = ws->heap[c0 + 4];
            uint64_t v5 = ws->heap[c0 + 5];
            uint64_t v6 = ws->heap[c0 + 6];
            uint64_t v7 = ws->heap[c0 + 7];
            uint64_t va, vb, vc, vd, ve, vf;
            size_t ia, ib, ic, id, ie, ig;
            heap_min2(bestv, c0, v1, c0 + 1, &va, &ia);
            heap_min2(v2, c0 + 2, v3, c0 + 3, &vb, &ib);
            heap_min2(v4, c0 + 4, v5, c0 + 5, &vc, &ic);
            heap_min2(v6, c0 + 6, v7, c0 + 7, &vd, &id);
            heap_min2(va, ia, vb, ib, &ve, &ie);
            heap_min2(vc, ic, vd, id, &vf, &ig);
            heap_min2(ve, ie, vf, ig, &bestv, &best);
        } else {
            for (size_t k = c0 + 1; k < cend; k++) {
                uint64_t v = ws->heap[k];
                uint64_t better = (v < bestv) ? ~0ull : 0;
                best = (best & ~(size_t)better) | ((size_t)better & k);
                bestv = (bestv & ~better) | (v & better);
            }
        }
        if (!heap_less(ws->heap[best], ws->heap[i])) break;

        size_t g0 = 8 * best + 1;
        if (g0 < ws->heap_count) {
            if (scan == 2) {
                __builtin_prefetch(&ws->heap[g0], 0, 1);
                __builtin_prefetch(&ws->heap[g0 + 7], 0, 1);
            } else {
                __builtin_prefetch(&ws->heap[g0], 0, 1);
                if (g0 + 1 < ws->heap_count)
                    __builtin_prefetch(&ws->heap[g0 + 1], 0, 1);
            }
        }
        uint64_t tmp = ws->heap[i];
        ws->heap[i] = ws->heap[best];
        ws->heap[best] = tmp;
        i = best;
    }
    return top;
}

static int whole_piece_ok(const tsimd_vocab_t *v, uint32_t id) {
    if (v->ignore_merges) return 1;
    if (v->merge_sim) return v->merge_sim[id] != 0;
    if (v->merge_reach) return v->merge_reach[id] != 0;
    return 1;
}

int tsimd_bpe_encode(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                    const uint8_t *bytes, uint32_t len) {
    if (len == 0) return 0;

    {
        uint32_t whole = tsimd_vocab_find(v, bytes, len);
        if (whole != UINT32_MAX && whole_piece_ok(v, whole))
            return out_push(ws, whole);
    }
    if (ensure_capacity(ws, len) != 0) return -1;

    for (uint32_t i = 0; i < len; i++) {
        ws->ids[i] = v->byte_ids[bytes[i]];
        ws->prev[i] = (i == 0) ? TSIMD_NIL : i - 1;
        ws->next[i] = (i + 1 == len) ? TSIMD_NIL : i + 1;
    }

    uint32_t head = 0;
    uint32_t count = len;

    if (len <= TSIMD_BPE_HEAP_LINEAR) {

        uint32_t ranks[TSIMD_BPE_HEAP_LINEAR];

        uint32_t pids[TSIMD_BPE_HEAP_LINEAR];
        for (uint32_t i = 0; i + 1 < len; i++) {
            if (i + 2 < len)
                tsimd_vocab_merge_prefetch(v, ws->ids[i + 1], ws->ids[i + 2]);
            ranks[i] = tsimd_vocab_merge_prio_id_fast(
                v, ws->ids[i], ws->ids[i + 1], &pids[i]);
        }
        while (count > 1) {

            uint32_t best = UINT32_MAX, best_i = 0;
            for (uint32_t i = 0; i + 1 < len; i++) {
                uint32_t r = ranks[i];
                uint32_t better = (r < best) ? ~0u : 0;
                best_i = (better & i) | (best_i & ~better);
                best = (better & r) | (best & ~better);
            }
            if (best == UINT32_MAX) break;
            uint32_t i = best_i;
            uint32_t dead = ws->next[i];
            uint32_t new_right = ws->next[dead];
            uint32_t left = ws->prev[i];
            if (new_right < len)
                tsimd_vocab_merge_prefetch(v, pids[i], ws->ids[new_right]);
            if (left < len)
                tsimd_vocab_merge_prefetch(v, ws->ids[left], pids[i]);
            ws->ids[i] = pids[i];
            ws->next[i] = new_right;
            ranks[dead] = UINT32_MAX;
            count--;
            if (new_right < len) {
                ws->prev[new_right] = i;
                ranks[i] = tsimd_vocab_merge_prio_id_fast(
                    v, ws->ids[i], ws->ids[new_right], &pids[i]);
            } else {
                ranks[i] = UINT32_MAX;
            }
            if (left < len)
                ranks[left] = tsimd_vocab_merge_prio_id_fast(
                    v, ws->ids[left], ws->ids[i], &pids[left]);
        }
    } else {

        ws->heap_count = 0;
        for (uint32_t p = head; p != TSIMD_NIL; p = ws->next[p]) {
            uint32_t q = ws->next[p];
            if (q == TSIMD_NIL) break;
            if (v->mgr_prefetch_ok) {
                if (v->mgr_prefetch_ok) {
                    uint32_t qn = ws->next[q];
                    if (qn != TSIMD_NIL)
                        tsimd_vocab_merge_prefetch(v, ws->ids[q], ws->ids[qn]);
                }
            }
            uint32_t r = tsimd_vocab_merge_priority(v, ws->ids[p],
                                                    ws->ids[q]);
            if (r == UINT32_MAX) continue;
            if (heap_push(ws, r, p) != 0) return -1;
        }
        while (ws->heap_count > 0 && count > 1) {
            uint64_t top = heap_pop(ws);
            uint32_t rank = (uint32_t)(top >> 32);
            uint32_t p = (uint32_t)(top & 0xFFFFFFFFu);
            uint32_t q = ws->next[p];
            if (q == TSIMD_NIL) continue;

            uint32_t idv = 0;
            if (tsimd_vocab_merge_prio_id(v, ws->ids[p], ws->ids[q], &idv) !=
                rank)
                continue;

            uint32_t after = ws->next[q];
            uint32_t u = ws->prev[p];
            if (u != TSIMD_NIL)
                tsimd_vocab_merge_prefetch(v, ws->ids[u], idv);
            if (after != TSIMD_NIL)
                tsimd_vocab_merge_prefetch(v, idv, ws->ids[after]);
            ws->ids[p] = idv;
            ws->next[p] = after;
            if (after != TSIMD_NIL) ws->prev[after] = p;
            ws->next[q] = TSIMD_NIL;

            count--;

            if (u != TSIMD_NIL) {
                uint32_t r2 = tsimd_vocab_merge_priority(v, ws->ids[u],
                                                         ws->ids[p]);
                if (r2 != UINT32_MAX && heap_push(ws, r2, u) != 0)
                    return -1;
            }
            if (after != TSIMD_NIL) {
                uint32_t r3 = tsimd_vocab_merge_priority(v, ws->ids[p],
                                                         ws->ids[after]);
                if (r3 != UINT32_MAX && heap_push(ws, r3, p) != 0)
                    return -1;
            }
        }
    }

    for (uint32_t p = head; p != TSIMD_NIL; p = ws->next[p])
        if (out_push(ws, ws->ids[p]) != 0) return -1;
    return 0;
}

static int spm_stats_on(void) {
    static atomic_int state = 0;
    int v = atomic_load_explicit(&state, memory_order_relaxed);
    if (v == 0) {
        const char *e = getenv("TSIMD_SPM_STATS");
        v = (e && e[0] == '1') ? 1 : 2;
        atomic_store_explicit(&state, v, memory_order_relaxed);
    }
    return v == 1;
}
#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t spm_rdtsc(void) { return __builtin_ia32_rdtsc(); }
#else
static inline uint64_t spm_rdtsc(void) { return 0; }
#endif
static _Thread_local uint64_t tl_spm_init, tl_spm_heap, tl_spm_merge;
static _Thread_local uint64_t tl_spm_syms, tl_spm_merges;

void tsimd_spm_stats_report(void) {
    if (!spm_stats_on() || tl_spm_syms == 0) return;
    double tot = (double)(tl_spm_init + tl_spm_heap + tl_spm_merge);
    fprintf(stderr,
            "tsimd spm: %llu syms, %llu merges — %.1f cyc/sym: init %.1f "
            "(%.0f%%), heap %.1f (%.0f%%), merge %.1f (%.0f%%)\n",
            (unsigned long long)tl_spm_syms,
            (unsigned long long)tl_spm_merges,
            tot / (double)tl_spm_syms,
            (double)tl_spm_init / (double)tl_spm_syms,
            100.0 * (double)tl_spm_init / tot,
            (double)tl_spm_heap / (double)tl_spm_syms,
            100.0 * (double)tl_spm_heap / tot,
            (double)tl_spm_merge / (double)tl_spm_syms,
            100.0 * (double)tl_spm_merge / tot);
    tl_spm_init = tl_spm_heap = tl_spm_merge = 0;
    tl_spm_syms = tl_spm_merges = 0;
}

#define TSIMD_BQ_MAX_BUCKETS (1u << 20)
#define TSIMD_BQ_AUTO_MAX (1u << 20)
#define TSIMD_BQ_SCAN_CAP 1024u
#define TSIMD_BQ_GROUP 16
#define TSIMD_BQ_GSHIFT 4
#define TSIMD_BQ_GMASK 15

static void bq_reserve(tsimd_bpe_workspace_t *ws, uint32_t nbuckets) {
    if (nbuckets <= ws->bq_nbuckets) return;
    uint32_t ngroups = (nbuckets + TSIMD_BQ_GMASK) >> TSIMD_BQ_GSHIFT;
    uint32_t l0w = (ngroups + 63u) >> 6;
    uint32_t l1w = (l0w + 63u) >> 6;
    uint32_t l2w = (l1w + 63u) >> 6;
    uint32_t *head =
        tsimd_xmalloc_aligned((size_t)nbuckets * sizeof(uint32_t), 64);
    uint32_t *tail =
        tsimd_xmalloc_aligned((size_t)nbuckets * sizeof(uint32_t), 64);
    uint16_t *gmask = tsimd_xmalloc((size_t)ngroups * sizeof(uint16_t));
    uint64_t *l0 = tsimd_xmalloc((size_t)l0w * sizeof(uint64_t));
    uint64_t *l1 = tsimd_xmalloc((size_t)l1w * sizeof(uint64_t));
    uint64_t *l2 = tsimd_xmalloc((size_t)l2w * sizeof(uint64_t));
    memset(head, 0, (size_t)nbuckets * sizeof(uint32_t));
    memset(tail, 0, (size_t)nbuckets * sizeof(uint32_t));
    memset(gmask, 0, (size_t)ngroups * sizeof(uint16_t));
    memset(l0, 0, (size_t)l0w * sizeof(uint64_t));
    memset(l1, 0, (size_t)l1w * sizeof(uint64_t));
    memset(l2, 0, (size_t)l2w * sizeof(uint64_t));
    tsimd_xfree(ws->bq_head);
    tsimd_xfree(ws->bq_tail);
    free(ws->bq_gmask);
    ws->bq_gmask = gmask;
    free(ws->bq_l0);
    free(ws->bq_l1);
    free(ws->bq_l2);
    ws->bq_head = head;
    ws->bq_tail = tail;
    ws->bq_l0 = l0;
    ws->bq_l1 = l1;
    ws->bq_l2 = l2;
    ws->bq_nbuckets = nbuckets;
    ws->bq_l0_words = l0w;
    ws->bq_l1_words = l1w;
    ws->bq_l2_words = l2w;
    ws->bq_touched_count = 0;
}

static inline void bq_mark(tsimd_bpe_workspace_t *ws, uint32_t group) {
    uint32_t w0 = group >> 6;
    uint32_t w1 = w0 >> 6;
    ws->bq_l0[w0] |= 1ull << (group & 63);
    ws->bq_l1[w1] |= 1ull << (w0 & 63);
    ws->bq_l2[w1 >> 6] |= 1ull << (w1 & 63);
}

static inline int bq_marked(const tsimd_bpe_workspace_t *ws, uint32_t group) {
    return (ws->bq_l0[group >> 6] >> (group & 63)) & 1u;
}

static inline void bq_unmark(tsimd_bpe_workspace_t *ws, uint32_t group) {
    uint32_t w0 = group >> 6;
    if ((ws->bq_l0[w0] &= ~(1ull << (group & 63))) != 0) return;
    uint32_t w1 = w0 >> 6;
    if ((ws->bq_l1[w1] &= ~(1ull << (w0 & 63))) != 0) return;
    ws->bq_l2[w1 >> 6] &= ~(1ull << (w1 & 63));
}

static void bq_reset(tsimd_bpe_workspace_t *ws) {
    int gp = tsimd_bpe_bqgp() == 1;
    ws->bq_gp = gp;
    for (uint32_t i = 0; i < ws->bq_touched_count; i++) {
        uint32_t group = ws->bq_touched[i];
        memset(ws->bq_head + ((size_t)group << (gp ? 0 : TSIMD_BQ_GSHIFT)), 0,
               (size_t)(gp ? 1 : TSIMD_BQ_GROUP) * sizeof(uint32_t));
        memset(ws->bq_tail + ((size_t)group << (gp ? 0 : TSIMD_BQ_GSHIFT)), 0,
               (size_t)(gp ? 1 : TSIMD_BQ_GROUP) * sizeof(uint32_t));
        ws->bq_gmask[group] = 0;
        bq_unmark(ws, group);
    }
    ws->bq_touched_count = 0;
    ws->bq_count = 0;
    ws->bq_goverflow = 0;
}

static void bq_push(tsimd_bpe_workspace_t *ws, uint32_t rank, uint32_t pos) {
    if (ws->bq_count == ws->bq_cap) {
        uint32_t cap = ws->bq_cap ? ws->bq_cap * 2 : 512;
        ws->bq_entries =
            tsimd_xrealloc(ws->bq_entries, (size_t)cap * sizeof(uint64_t));
        ws->bq_rank =
            tsimd_xrealloc(ws->bq_rank, (size_t)cap * sizeof(uint32_t));
        ws->bq_cap = cap;
    }
    uint32_t group = rank >> TSIMD_BQ_GSHIFT;
    uint32_t slot = rank & TSIMD_BQ_GMASK;
    uint32_t *g = ws->bq_head + ((size_t)group << (ws->bq_gp ? 0 : TSIMD_BQ_GSHIFT));
    uint32_t h = ws->bq_gp ? *g : g[slot];
    if (!bq_marked(ws, group)) {
        if (ws->bq_touched_count == ws->bq_touched_cap) {
            uint32_t cap = ws->bq_touched_cap ? ws->bq_touched_cap * 2 : 256;
            ws->bq_touched = tsimd_xrealloc(ws->bq_touched,
                                            (size_t)cap * sizeof(uint32_t));
            ws->bq_touched_cap = cap;
        }
        ws->bq_touched[ws->bq_touched_count++] = group;
        bq_mark(ws, group);
    }

    uint32_t *t_ptr = ws->bq_gp
                          ? ws->bq_tail + group
                          : ws->bq_tail + ((size_t)(group << TSIMD_BQ_GSHIFT) + slot);
    if (ws->bq_gp) {
        uint32_t idx = ws->bq_count + 1;
        ws->bq_rank[ws->bq_count] = rank;
        ws->bq_entries[ws->bq_count] = pos;
        ws->bq_count++;
        if (*g == 0)
            *g = idx;
        else
            ws->bq_entries[(*t_ptr) - 1] |= ((uint64_t)idx << 32);
        *t_ptr = idx;
        return;
    }
    if (h == 0) ws->bq_gmask[group] |= (uint16_t)(1u << slot);
    uint32_t idx = ws->bq_count + 1;
    ws->bq_entries[ws->bq_count] = pos;
    ws->bq_count++;
    if (h == 0)
        g[slot] = idx;
    else
        ws->bq_entries[(*t_ptr) - 1] |= ((uint64_t)idx << 32);
    *t_ptr = idx;
}

static void bq_list_unlink(tsimd_bpe_workspace_t *ws, uint32_t *headp,
                           uint32_t *tailp, uint32_t node, uint32_t prev) {
    uint32_t nxt = (uint32_t)(ws->bq_entries[node - 1] >> 32);
    if (prev == 0)
        *headp = nxt;
    else
        ws->bq_entries[prev - 1] =
            (ws->bq_entries[prev - 1] & 0xFFFFFFFFull) | ((uint64_t)nxt << 32);
    if (*tailp == node) *tailp = prev;
}

static uint32_t bq_pop(tsimd_bpe_workspace_t *ws, uint32_t *id_out) {
    for (;;) {
        uint32_t w2 = 0;
        while (w2 < ws->bq_l2_words && ws->bq_l2[w2] == 0) w2++;
        if (w2 == ws->bq_l2_words) return UINT32_MAX;
        uint32_t w1 = (w2 << 6) | (uint32_t)__builtin_ctzll(ws->bq_l2[w2]);
        uint32_t w0 = (w1 << 6) | (uint32_t)__builtin_ctzll(ws->bq_l1[w1]);
        uint32_t group = (w0 << 6) | (uint32_t)__builtin_ctzll(ws->bq_l0[w0]);
        if (ws->bq_gp) {
            uint32_t *g = ws->bq_head + group;
            uint32_t *t_ptr = ws->bq_tail + group;
            uint32_t cur = *g;
            if (cur == 0) {
                bq_unmark(ws, group);
                continue;
            }
            uint32_t best = 0, best_prev = 0, best_rank = 0, best_pos = 0;
            uint32_t best_id = 0;
            uint32_t prev = 0;
            uint32_t scanned = 0;
            for (uint32_t it = cur; it != 0;) {
                if (++scanned > TSIMD_BQ_SCAN_CAP) {
                    ws->bq_goverflow = 1;
                    return UINT32_MAX;
                }
                uint64_t e = ws->bq_entries[it - 1];
                uint32_t nxt = (uint32_t)(e >> 32);
                uint32_t pos = (uint32_t)e;
                uint32_t r = ws->bq_rank[it - 1];
                uint32_t idv = 0;
                if (ws->next[pos] != TSIMD_NIL &&
                    cand_match(ws, pos, r, &idv)) {
                    if (best == 0 || r < best_rank ||
                        (r == best_rank && pos < best_pos)) {
                        best = it;
                        best_prev = prev;
                        best_rank = r;
                        best_pos = pos;
                        best_id = idv;
                    }
                    prev = it;
                } else {
                    bq_list_unlink(ws, g, t_ptr, it, prev);
                }
                it = nxt;
            }
            if (best != 0) {
                bq_list_unlink(ws, g, t_ptr, best, best_prev);
                *id_out = best_id;
                return best_pos;
            }
            bq_unmark(ws, group);
            continue;
        }
        uint32_t gmask = ws->bq_gmask[group];
        if (gmask == 0) {
            bq_unmark(ws, group);
            continue;
        }
        uint32_t slot = (uint32_t)__builtin_ctz(gmask);
        uint32_t rank = (group << TSIMD_BQ_GSHIFT) | slot;
        uint32_t *g = ws->bq_head + ((size_t)group << TSIMD_BQ_GSHIFT);
        uint32_t *t_ptr =
            ws->bq_tail + ((size_t)(group << TSIMD_BQ_GSHIFT) + slot);
        uint32_t cur = g[slot];
        if (cur == 0) {
            ws->bq_gmask[group] &= (uint16_t)~(1u << slot);
            if (ws->bq_gmask[group] == 0) bq_unmark(ws, group);
            continue;
        }
        uint32_t best = 0, best_prev = 0, best_pos = 0, best_id = 0;
        uint32_t prev = 0;
        uint32_t scanned = 0;
        for (uint32_t it = cur; it != 0;) {
            if (++scanned > TSIMD_BQ_SCAN_CAP) {
                ws->bq_goverflow = 1;
                return UINT32_MAX;
            }
            uint64_t e = ws->bq_entries[it - 1];
            uint32_t nxt = (uint32_t)(e >> 32);
            uint32_t pos = (uint32_t)e;
            uint32_t idv = 0;
            if (ws->next[pos] != TSIMD_NIL &&
                cand_match(ws, pos, rank, &idv)) {
                if (best == 0 || pos < best_pos) {
                    best = it;
                    best_prev = prev;
                    best_pos = pos;
                    best_id = idv;
                }
                prev = it;
            } else {
                bq_list_unlink(ws, g + slot, t_ptr, it, prev);
                if (best_prev == it) best_prev = prev;
            }
            it = nxt;
        }
        if (best != 0) {
            bq_list_unlink(ws, g + slot, t_ptr, best, best_prev);
            *id_out = best_id;
            return best_pos;
        }
        ws->bq_gmask[group] &= (uint16_t)~(1u << slot);
        if (ws->bq_gmask[group] == 0) bq_unmark(ws, group);
        continue;
    }
}

static int spm_bucket_merge(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                            uint32_t *count_io, uint64_t *t_fill_out) {
    bq_reset(ws);
    for (uint32_t p = 0; p != TSIMD_NIL; p = ws->next[p]) {
        uint32_t q = ws->next[p];
        if (q == TSIMD_NIL) break;
        if (v->mgr_prefetch_ok) {
            if (v->mgr_prefetch_ok) {
                uint32_t qn = ws->next[q];
                if (qn != TSIMD_NIL)
                    tsimd_vocab_merge_prefetch(v, ws->ids[q], ws->ids[qn]);
            }
        }
        uint32_t idv = 0;
        uint32_t r = tsimd_vocab_merge_prio_id_fast(v, ws->ids[p], ws->ids[q],
                                                    &idv);
        cand_store(ws, p, r, idv);
        if (r == UINT32_MAX) continue;
        if (r >= ws->bq_nbuckets) {
            bq_reset(ws);
            return -2;
        }
        bq_push(ws, r, p);
    }
    atomic_fetch_add_explicit(&bucket_runs, 1, memory_order_relaxed);
    if (t_fill_out) *t_fill_out = spm_rdtsc();
    while (*count_io > 1) {
        uint32_t idv = 0;
        uint32_t p = bq_pop(ws, &idv);
        if (ws->bq_goverflow) {
            bq_reset(ws);
            return -2;
        }
        if (p == UINT32_MAX) break;
        uint32_t q = ws->next[p];
        uint32_t after = ws->next[q];
        uint32_t u = ws->prev[p];
        if (u != TSIMD_NIL)
            tsimd_vocab_merge_prefetch(v, ws->ids[u], idv);
        if (after != TSIMD_NIL)
            tsimd_vocab_merge_prefetch(v, idv, ws->ids[after]);
        ws->ids[p] = idv;
        ws->next[p] = after;
        if (after != TSIMD_NIL) ws->prev[after] = p;
        ws->next[q] = TSIMD_NIL;
        (*count_io)--;
        if (u != TSIMD_NIL) {
            uint32_t id2 = 0;
            uint32_t r2 = tsimd_vocab_merge_prio_id_fast(v, ws->ids[u],
                                                         ws->ids[p], &id2);
            cand_store(ws, u, r2, id2);
            if (r2 != UINT32_MAX) {
                if (r2 >= ws->bq_nbuckets) {
                    bq_reset(ws);
                    return -2;
                }
                bq_push(ws, r2, u);
            }
        }
        if (after != TSIMD_NIL) {
            uint32_t id3 = 0;
            uint32_t r3 = tsimd_vocab_merge_prio_id_fast(v, ws->ids[p],
                                                         ws->ids[after], &id3);
            cand_store(ws, p, r3, id3);
            if (r3 != UINT32_MAX) {
                if (r3 >= ws->bq_nbuckets) {
                    bq_reset(ws);
                    return -2;
                }
                bq_push(ws, r3, p);
            }
        }
    }
    bq_reset(ws);
    return 0;
}

#if defined(__x86_64__) || defined(__i386__)
static uint64_t bpe_rdtsc(void) {
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#else
static uint64_t bpe_rdtsc(void) { return 0; }
#endif

static uint64_t t_seam, t_hash, t_probe, t_merge, t_push;
static uint64_t n_units, n_miss, n_seamcalls;
static int merge_dbg_on(void) {
    static atomic_int state = 0;
    int v = atomic_load_explicit(&state, memory_order_relaxed);
    if (v == 0) {
        atomic_store_explicit(&state, getenv("TSIMD_MERGE_DBG") ? 2 : 1,
                              memory_order_relaxed);
        v = atomic_load_explicit(&state, memory_order_relaxed);
    }
    return v == 2;
}
static int bpe_split_on(void) {
    static atomic_int state = 0;
    int v = atomic_load_explicit(&state, memory_order_relaxed);
    if (v == 0) {
        const char *e = getenv("TSIMD_BPE_SPLIT");
        atomic_store_explicit(&state, (e && e[0] == '1') ? 2 : 1,
                              memory_order_relaxed);
        v = atomic_load_explicit(&state, memory_order_relaxed);
    }
    return v == 2;
}

static int spm_merge_body(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                          const uint8_t *bytes, uint32_t len, int skip_whole) {
    if (len == 0) return 0;
    if (unit_dbg()) {
        fprintf(stderr, "unit len=%u off16=", len);
        for (uint32_t z = 0; z < len && z < 16; z++)
            fprintf(stderr, "%02x", bytes[z]);
        fprintf(stderr, "\n");
    }
    if (!skip_whole) {
        uint32_t whole = tsimd_vocab_find(v, bytes, len);
        if (whole != UINT32_MAX && whole_piece_ok(v, whole))
            return out_push(ws, whole);
    }
    if (ensure_capacity(ws, len) != 0) return -1;
    const int spm_st = spm_stats_on();
    const uint64_t spm_t0 = spm_st ? spm_rdtsc() : 0;
    uint32_t sym = 0;
    for (uint32_t i = 0; i < len;) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t a = tsimd_utf8_next(bytes + i, len - i, &cp, &cls);
        if (a == 0) a = 1;
        {

            uint32_t p = a == 1 ? v->byte_ids[bytes[i]]
                                : tsimd_vocab_find(v, bytes + i,
                                                   (uint32_t)a);
            if (p != UINT32_MAX) {
                ws->ids[sym] = p;
                ws->prev[sym] = sym == 0 ? TSIMD_NIL : sym - 1;
                ws->next[sym] = sym + 1;
                sym++;
                i += (uint32_t)a;
                continue;
            }
        }
        for (uint32_t b = 0; b < (uint32_t)a; b++) {
            ws->ids[sym] = v->byte_ids[bytes[i + b]];
            ws->prev[sym] = sym == 0 ? TSIMD_NIL : sym - 1;
            ws->next[sym] = sym + 1;
            sym++;
        }
        i += (uint32_t)a;
    }
    if (sym == 0) {
        ws->prev[0] = TSIMD_NIL;
        ws->next[0] = TSIMD_NIL;
        sym = 1;
    }
    ws->next[sym - 1] = TSIMD_NIL;
    uint32_t head = 0;
    uint32_t count = sym;
    const uint64_t spm_t1 = spm_st ? spm_rdtsc() : 0;
    uint64_t spm_t2 = 0;
    const uint32_t linear_limit = tsimd_bpe_heap_linear();
    int used_bucket = 0;
    if (sym > linear_limit && !tsimd_bpe_pop_probe()) {
        if (ensure_cand_state(ws, sym) != 0) return -1;
        uint32_t qv = tsimd_bpe_queue();
        int bq_on = qv == 1 || (qv == 2 && v->vocab_size <= TSIMD_BQ_AUTO_MAX);
        if (bq_on && v->vocab_size <= TSIMD_BQ_MAX_BUCKETS) {
            bq_reserve(ws, v->vocab_size);
            int rc = spm_bucket_merge(v, ws, &count, spm_st ? &spm_t2 : NULL);
            if (rc == 0) {
                used_bucket = 1;
            } else if (rc != -2) {
                return -1;
            }
        }
        if (!used_bucket) {
            ws->heap_count = 0;
            for (uint32_t p = head; p != TSIMD_NIL; p = ws->next[p]) {
                uint32_t q = ws->next[p];
                if (q == TSIMD_NIL) break;
                uint32_t qn = ws->next[q];
                if (qn != TSIMD_NIL)
                    tsimd_vocab_merge_prefetch(v, ws->ids[q], ws->ids[qn]);
                uint32_t idv = 0;
                uint32_t r = tsimd_vocab_merge_prio_id_fast(v, ws->ids[p],
                                                            ws->ids[q], &idv);
                cand_store(ws, p, r, idv);
                if (r == UINT32_MAX) continue;
                if (heap_push(ws, r, p) != 0) return -1;
            }
            spm_t2 = spm_st ? spm_rdtsc() : 0;
            uint64_t dbg_pops = 0, dbg_stale = 0, dbg_merged = 0;
            while (ws->heap_count > 0 && count > 1) {
                dbg_pops++;
                uint64_t top = heap_pop(ws);
                uint32_t rank = (uint32_t)(top >> 32);
                uint32_t p = (uint32_t)(top & 0xFFFFFFFFu);
                uint32_t q = ws->next[p];
                if (q == TSIMD_NIL) continue;
                uint32_t idv = 0;
                if (!cand_match(ws, p, rank, &idv)) {
                    dbg_stale++;
                    continue;
                }
                uint32_t after = ws->next[q];
                uint32_t u = ws->prev[p];
                if (u != TSIMD_NIL)
                    tsimd_vocab_merge_prefetch(v, ws->ids[u], idv);
                if (after != TSIMD_NIL)
                    tsimd_vocab_merge_prefetch(v, idv, ws->ids[after]);
                ws->ids[p] = idv;
                ws->next[p] = after;
                if (after != TSIMD_NIL) ws->prev[after] = p;
                ws->next[q] = TSIMD_NIL;
                count--;
                dbg_merged++;
                if (u != TSIMD_NIL) {
                    uint32_t id2 = 0;
                    uint32_t r2 = tsimd_vocab_merge_prio_id_fast(
                        v, ws->ids[u], ws->ids[p], &id2);
                    cand_store(ws, u, r2, id2);
                    if (r2 != UINT32_MAX && heap_push(ws, r2, u) != 0) return -1;
                }
                if (after != TSIMD_NIL) {
                    uint32_t id3 = 0;
                    uint32_t r3 = tsimd_vocab_merge_prio_id_fast(
                        v, ws->ids[p], ws->ids[after], &id3);
                    if (merge_dbg_on() && dbg_merged == 1)
                        fprintf(stderr, "MERGEDBG first merge done\n");
                    cand_store(ws, p, r3, id3);
                    if (r3 != UINT32_MAX && heap_push(ws, r3, p) != 0) return -1;
                }
            }
            if (merge_dbg_on())
                fprintf(stderr,
                        "MERGEDBG sym=%u pops=%llu stale=%llu merged=%llu\n",
                        sym, (unsigned long long)dbg_pops,
                        (unsigned long long)dbg_stale,
                        (unsigned long long)dbg_merged);
        }
    } else if (sym <= linear_limit) {
        if (ensure_linear_scratch(ws, sym) != 0) return -1;
        uint32_t *ranks = ws->lin_ranks;
        int lsplit = bpe_split_on();
        uint64_t lscan = 0, lapp = 0;

        uint32_t *pids = ws->lin_ids;
        for (uint32_t i = 0; i + 1 < sym; i++) {
            if (i + 2 < sym)
                tsimd_vocab_merge_prefetch(v, ws->ids[i + 1], ws->ids[i + 2]);
            ranks[i] = tsimd_vocab_merge_prio_id_fast(v, ws->ids[i],
                                                 ws->ids[i + 1], &pids[i]);
        }
        while (count > 1) {
            uint64_t sc0 = lsplit ? bpe_rdtsc() : 0;
            uint32_t best = UINT32_MAX, best_i = 0;
            for (uint32_t i = 0; i + 1 < sym; i++)
                if (ranks[i] < best) {
                    best = ranks[i];
                    best_i = i;
                }
            if (lsplit) lscan += bpe_rdtsc() - sc0;
            if (best == UINT32_MAX) break;
            uint64_t ap0 = lsplit ? bpe_rdtsc() : 0;
            uint32_t i = best_i;
            uint32_t dead = ws->next[i];
            uint32_t new_right = ws->next[dead];
            uint32_t left = ws->prev[i];
            if (new_right < sym)
                tsimd_vocab_merge_prefetch(v, pids[i], ws->ids[new_right]);
            if (left < sym)
                tsimd_vocab_merge_prefetch(v, ws->ids[left], pids[i]);
            ws->ids[i] = pids[i];
            ws->next[i] = new_right;
            ranks[dead] = UINT32_MAX;
            count--;
            if (new_right < sym) {
                ws->prev[new_right] = i;
                ranks[i] = tsimd_vocab_merge_prio_id_fast(v, ws->ids[i],
                                                     ws->ids[new_right],
                                                     &pids[i]);
            } else {
                ranks[i] = UINT32_MAX;
            }
            if (left < sym)
                ranks[left] = tsimd_vocab_merge_prio_id_fast(
                    v, ws->ids[left], ws->ids[i], &pids[left]);
            if (lsplit) lapp += bpe_rdtsc() - ap0;
        }
        if (lsplit) {
            t_seam += lscan;
            t_hash += lapp;
        }
    } else {
        ws->heap_count = 0;
        for (uint32_t p = head; p != TSIMD_NIL; p = ws->next[p]) {
            uint32_t q = ws->next[p];
            if (q == TSIMD_NIL) break;
            uint32_t qn = ws->next[q];
            if (qn != TSIMD_NIL)
                tsimd_vocab_merge_prefetch(v, ws->ids[q], ws->ids[qn]);
            uint32_t r = tsimd_vocab_merge_priority_fast(v, ws->ids[p],
                                                    ws->ids[q]);
            if (r == UINT32_MAX) continue;
            if (heap_push(ws, r, p) != 0) return -1;
        }
        spm_t2 = spm_st ? spm_rdtsc() : 0;
        while (ws->heap_count > 0 && count > 1) {
            uint64_t top = heap_pop(ws);
            uint32_t rank = (uint32_t)(top >> 32);
            uint32_t p = (uint32_t)(top & 0xFFFFFFFFu);
            uint32_t q = ws->next[p];
            if (q == TSIMD_NIL) continue;

            uint32_t idv = 0;
            if (tsimd_vocab_merge_prio_id_fast(v, ws->ids[p], ws->ids[q],
                                               &idv) != rank)
                continue;
            uint32_t after = ws->next[q];
            uint32_t u = ws->prev[p];
            if (u != TSIMD_NIL)
                tsimd_vocab_merge_prefetch(v, ws->ids[u], idv);
            if (after != TSIMD_NIL)
                tsimd_vocab_merge_prefetch(v, idv, ws->ids[after]);
            ws->ids[p] = idv;
            ws->next[p] = after;
            if (after != TSIMD_NIL) ws->prev[after] = p;
            ws->next[q] = TSIMD_NIL;
            count--;
            if (u != TSIMD_NIL) {
                uint32_t r2 = tsimd_vocab_merge_priority_fast(v, ws->ids[u],
                                                         ws->ids[p]);
                if (r2 != UINT32_MAX && heap_push(ws, r2, u) != 0)
                    return -1;
            }
            if (after != TSIMD_NIL) {
                uint32_t r3 = tsimd_vocab_merge_priority_fast(v, ws->ids[p],
                                                         ws->ids[after]);
                if (r3 != UINT32_MAX && heap_push(ws, r3, p) != 0)
                    return -1;
            }
        }
    }
    for (uint32_t p = head; p != TSIMD_NIL; p = ws->next[p])
        if (out_push(ws, ws->ids[p]) != 0) return -1;
    if (spm_st) {
        uint64_t t3 = spm_rdtsc();
        if (spm_t2 == 0) spm_t2 = spm_t1;
        tl_spm_init += spm_t1 - spm_t0;
        tl_spm_heap += spm_t2 - spm_t1;
        tl_spm_merge += t3 - spm_t2;
        tl_spm_syms += sym;
        tl_spm_merges += sym - count;
    }
    return 0;
}

int tsimd_bpe_encode_spm(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                         const uint8_t *bytes, uint32_t len) {
    return spm_merge_body(v, ws, bytes, len, 0);
}

static void spm_long_ready(tsimd_pretok_cache_t *c) {
    if (__atomic_load_n(&c->lmin, __ATOMIC_RELAXED) == 5) return;
    __atomic_store_n(&c->lmin, 5, __ATOMIC_RELAXED);
}

static int spm_push_ids(tsimd_bpe_workspace_t *ws, const uint32_t *ids,
                        uint32_t n) {
    if (tsimd_ws_out_ensure(ws, ws->out_count + n) != 0) return -1;
    memcpy(ws->out + ws->out_count, ids, (size_t)n * sizeof(uint32_t));
    ws->out_count += n;
    return 0;
}

static int publish_enabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_PUBLISH");
        atomic_store(&state, (!e || e[0] != '0') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v == 1;
}

static void spm_publish_shared(tsimd_pretok_cache_t *sh,
                              const uint8_t *bytes, uint32_t len,
                              const uint32_t *ids, uint32_t n) {
    if (n == 0 || n > TSIMD_PCACHE_MAX_IDS) return;
    if (len == 0 || len > TSIMD_PCACHE_MAX_SPAN) return;
    if (ids[0] >= (1u << TSIMD_PCACHE_ID0_BITS)) return;
    if (__atomic_load_n(&sh->count, __ATOMIC_RELAXED) >=
        sh->max_count)
        return;
    uint64_t k0, k1, hash;
    tsimd_pcache_key_hash(bytes, len, &k0, &k1, &hash);
    if ((k0 | k1) == 0) return;
    uint64_t val = (uint64_t)n | ((uint64_t)ids[0] << 3) |
                   (n > 1 ? (uint64_t)ids[1] << 32 : 0);
    uint64_t ext = (n > 2 ? (uint64_t)ids[2] : 0) |
                   (n > 3 ? (uint64_t)ids[3] << 32 : 0);
    size_t set = ((size_t)hash << 1) & sh->pair_mask;
    unsigned steps = 0;
    for (;;) {
        tsimd_pcache_entry_t *e0 = &sh->entries[set];
        tsimd_pcache_entry_t *e1 = &sh->entries[set + 1];
        if (__atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE) == 0) {
            if (__sync_bool_compare_and_swap(&e0->key0, 0, k0)) {
                __atomic_store_n(&e0->val, val, __ATOMIC_RELAXED);
                __atomic_store_n(&e0->ext, ext, __ATOMIC_RELAXED);
                __atomic_store_n(&e0->key1, k1, __ATOMIC_RELEASE);
                __atomic_fetch_add(&sh->count, 1, __ATOMIC_RELAXED);
                if (tsimd_pcache_stats_on()) sh->inserts++;
                return;
            }
            if (__atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE) == k1 &&
                __atomic_load_n(&e0->key0, __ATOMIC_RELAXED) == k0)
                return;
        }
        if (__atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE) == 0) {
            if (__sync_bool_compare_and_swap(&e1->key0, 0, k0)) {
                __atomic_store_n(&e1->val, val, __ATOMIC_RELAXED);
                __atomic_store_n(&e1->ext, ext, __ATOMIC_RELAXED);
                __atomic_store_n(&e1->key1, k1, __ATOMIC_RELEASE);
                __atomic_fetch_add(&sh->count, 1, __ATOMIC_RELAXED);
                if (tsimd_pcache_stats_on()) sh->inserts++;
                return;
            }
            if (__atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE) == k1 &&
                __atomic_load_n(&e1->key0, __ATOMIC_RELAXED) == k0)
                return;
        }
        if (++steps >= 64) return;
        set = (set + 2) & sh->mask;
    }
}

static int spm_unit_cached_key(const tsimd_vocab_t *v,
                               tsimd_pretok_cache_t *c,
                               tsimd_pretok_cache_t *shared,
                               tsimd_bpe_workspace_t *ws,
                               const uint8_t *bytes, uint32_t len,
                               uint64_t k0, uint64_t k1, uint64_t hash,
                               int have_key);

static int spm_unit_cached_key(const tsimd_vocab_t *v,
                               tsimd_pretok_cache_t *c,
                               tsimd_pretok_cache_t *shared,
                               tsimd_bpe_workspace_t *ws,
                               const uint8_t *bytes, uint32_t len,
                               uint64_t k0, uint64_t k1, uint64_t hash,
                               int have_key) {
    if (len == 0) return 0;
    int pub = 0;
    int split = bpe_split_on();
    uint64_t s0 = split ? bpe_rdtsc() : 0;
    if (c) {
        uint32_t n_ids = 0;
        if (tsimd_ws_out_ensure(ws, ws->out_count + TSIMD_PCACHE_MAX_IDS) != 0)
            return -1;
        uint32_t *slot = ws->out + ws->out_count;
        uint32_t *hit =
            have_key
                ? tsimd_pcache_lookup_keyed(c, k0, k1, hash, slot, &n_ids)
                : tsimd_pcache_lookup_into_bounded(c, bytes, len,
                                                   bytes + len, slot, &n_ids);
        if (hit) {
            ws->out_count += n_ids;
            if (split) {
                n_units++;
                t_probe += bpe_rdtsc() - s0;
            }
            return 0;
        }
        if (split) n_miss++;
        if (len >= 5) {
            spm_long_ready(c);
            const uint32_t *lids = tsimd_plong_probe_ex(c, bytes, len, &n_ids);
            if (lids) return spm_push_ids(ws, lids, n_ids);
        }
        if (shared && shared != c) {
            uint32_t sn = 0;
            uint32_t *hit;
            if (have_key)
                hit = tsimd_pcache_lookup_keyed(shared, k0, k1, hash, slot,
                                                &sn);
            else
                hit = tsimd_pcache_lookup_into_bounded(shared, bytes, len,
                                                       bytes + len, slot,
                                                       &sn);
            if (hit) {
                if (sn <= TSIMD_PCACHE_MAX_IDS &&
                    len <= TSIMD_PCACHE_MAX_SPAN)
                    tsimd_pcache_insert(c, bytes, len,
                                        ws->out + ws->out_count, sn);
                ws->out_count += sn;
                return 0;
            }
            if (len >= 5) {
                const uint32_t *lids =
                    tsimd_plong_probe_ex(shared, bytes, len, &sn);
                if (lids) {
                    if (sn <= TSIMD_PLONG_MAX_IDS)
                        tsimd_plong_insert(c, bytes, len, lids, sn);
                    return spm_push_ids(ws, lids, sn);
                }
            }
            pub = 1;
        }
    }
    size_t base = ws->out_count;
    uint64_t m1 = split ? bpe_rdtsc() : 0;
    if (split) t_probe += m1 - s0;
    if (spm_merge_body(v, ws, bytes, len, 0) != 0) return -1;
    uint64_t m2 = split ? bpe_rdtsc() : 0;
    if (split) t_merge += m2 - m1;
    size_t got = ws->out_count - base;
    if (c && got >= 1) {
        if (got <= TSIMD_PCACHE_MAX_IDS && len <= TSIMD_PCACHE_MAX_SPAN)
            tsimd_pcache_insert(c, bytes, len, ws->out + base, (uint32_t)got);
        else if (len >= 5)
            tsimd_plong_insert(c, bytes, len, ws->out + base, (uint32_t)got);
    }
    if (pub && shared && got >= 1 && publish_enabled())
        spm_publish_shared(shared, bytes, len, ws->out + base,
                           (uint32_t)got);
    if (split) t_push += bpe_rdtsc() - m2;
    return 0;
}

int tsimd_bpe_encode_spm_cached(const tsimd_vocab_t *v,
                                tsimd_pretok_cache_t *c,
                                tsimd_pretok_cache_t *shared,
                                tsimd_bpe_workspace_t *ws,
                                const uint8_t *bytes, uint32_t len) {
    if (len == 0) return 0;
    int split = bpe_split_on();
    uint64_t s0 = split ? bpe_rdtsc() : 0;
    if (c) {
        uint32_t n_ids = 0;
        if (tsimd_ws_out_ensure(ws, ws->out_count + TSIMD_PCACHE_MAX_IDS) != 0)
            return -1;
        uint32_t *slot = ws->out + ws->out_count;
        uint32_t *hit = tsimd_pcache_lookup_into_bounded(c, bytes, len,
                                                         bytes + len, slot,
                                                         &n_ids);
        if (hit) {
            ws->out_count += n_ids;
            if (split) {
                n_units++;
                t_probe += bpe_rdtsc() - s0;
            }
            return 0;
        }
        if (split) n_miss++;
        if (len >= 5) {
            spm_long_ready(c);
            const uint32_t *lids = tsimd_plong_probe_ex(c, bytes, len, &n_ids);
            if (lids) {
                if (split)
                    t_probe += bpe_rdtsc() - s0;
                return spm_push_ids(ws, lids, n_ids);
            }
        }
    }
    size_t base = ws->out_count;
    uint64_t m1 = split ? bpe_rdtsc() : 0;
    if (split) t_probe += m1 - s0;
    if (spm_merge_body(v, ws, bytes, len, 0) != 0) return -1;
    uint64_t m2 = split ? bpe_rdtsc() : 0;
    if (split) t_merge += m2 - m1;
    size_t got = ws->out_count - base;
    if (c && got >= 1) {
        if (got <= TSIMD_PCACHE_MAX_IDS && len <= TSIMD_PCACHE_MAX_SPAN)
            tsimd_pcache_insert(c, bytes, len, ws->out + base, (uint32_t)got);
        else if (len >= 5)
            tsimd_plong_insert(c, bytes, len, ws->out + base, (uint32_t)got);
    }
    if (split) t_push += bpe_rdtsc() - m2;
    return 0;
}

int tsimd_bpe_encode_spm_blocked_cached(const tsimd_vocab_t *v,
                                        tsimd_pretok_cache_t *c,
                                        tsimd_pretok_cache_t *shared,
                                        tsimd_bpe_workspace_t *ws,
                                        const uint8_t *bytes, uint32_t len,
                                        size_t target) {
    if (len == 0) return 0;
    if (target == 0 || len < 2 * target)
        return tsimd_bpe_encode_spm_cached(v, c, shared, ws, bytes, len);
    size_t pos = 0;
    size_t end = len;
    int split = bpe_split_on();
    if ((size_t)len > target) {
        uint64_t c0 = split ? bpe_rdtsc() : 0;
        size_t seam = tsimd_spm_seam_find(v, bytes, len, target, target / 2);
        if (split) {
            t_seam += bpe_rdtsc() - c0;
            n_seamcalls++;
        }
        if (seam != 0) end = seam;
    }
    uint64_t k0 = 0, k1 = 0, hash = 0;
    int have_key = 0;
    if (c != NULL && end < len) {
        size_t nl = end;
        if (nl >= 1 && nl <= TSIMD_PCACHE_MAX_SPAN) {
            uint64_t c1 = split ? bpe_rdtsc() : 0;
            tsimd_pcache_key_hash_bounded(bytes, (uint32_t)nl, bytes + len,
                                          &k0, &k1, &hash);
            if (split) t_hash += bpe_rdtsc() - c1;
            have_key = 1;
            tsimd_pcache_prefetch_l2(c, hash);
        }
    }
    size_t nend = len;
    uint64_t nk0 = 0, nk1 = 0, nhash = 0;
    int nkey = 0;
    if (end < len) {
        if ((size_t)len - end > target) {
            size_t seam = tsimd_spm_seam_find(v, bytes, len, end + target,
                                              end + target / 2);
            if (seam != 0) nend = seam;
        }
        if (c != NULL && nend < len) {
            size_t nl = nend - end;
            if (nl >= 1 && nl <= TSIMD_PCACHE_MAX_SPAN) {
                tsimd_pcache_key_hash_bounded(bytes + end, (uint32_t)nl,
                                              bytes + len, &nk0, &nk1,
                                              &nhash);
                nkey = 1;
                tsimd_pcache_prefetch_l2(c, nhash);
            }
        }
    }
    if (have_key) tsimd_pcache_prefetch_l1(c, hash);
    for (;;) {
        size_t ulen = end - pos;
        if (unit_dbg())
            fprintf(stderr, "blk unit pos=%zu len=%zu\n", pos, ulen);
        if (spm_unit_cached_key(v, c, shared, ws, bytes + pos,
                                (uint32_t)ulen, k0, k1,
                                hash, have_key) != 0)
            return -1;
        pos = end;
        if (pos >= len) break;
        end = nend;
        k0 = nk0;
        k1 = nk1;
        hash = nhash;
        have_key = nkey;
        if (have_key) tsimd_pcache_prefetch_l1(c, hash);
        nend = len;
        nkey = 0;
        {
            size_t npos = end;
            if ((size_t)len - npos > target) {
                uint64_t c0 = split ? bpe_rdtsc() : 0;
                size_t seam = tsimd_spm_seam_find(v, bytes, len,
                                                  npos + target,
                                                  npos + target / 2);
                if (split) {
                    t_seam += bpe_rdtsc() - c0;
                    n_seamcalls++;
                }
                if (seam != 0) nend = seam;
            }
            if (c != NULL && nend < len) {
                size_t nl = nend - npos;
                if (nl >= 1 && nl <= TSIMD_PCACHE_MAX_SPAN) {
                    uint64_t c1 = split ? bpe_rdtsc() : 0;
                    tsimd_pcache_key_hash_bounded(bytes + npos, (uint32_t)nl,
                                                  bytes + len, &nk0, &nk1,
                                                  &nhash);
                    if (split) t_hash += bpe_rdtsc() - c1;
                    nkey = 1;
                    tsimd_pcache_prefetch_l2(c, nhash);
                }
            }
        }
    }
    tsimd_spm_stats_report();
    return 0;
}

void tsimd_bpe_split_dump(void) {
    if (!bpe_split_on()) return;
    fprintf(stderr,
            "bpe split: probes(hits)=%llu misses=%llu seam_ticks=%.1f "
            "hash_ticks=%.1f probe_ticks=%.1f merge_ticks=%.1f "
            "push_ticks=%.1f seam_calls=%llu (the ticks are TSC units; "
            "read the shares)\n",
            (unsigned long long)n_units, (unsigned long long)n_miss,
            (double)t_seam / 2.5e6, (double)t_hash / 2.5e6,
            (double)t_probe / 2.5e6, (double)t_merge / 2.5e6,
            (double)t_push / 2.5e6, (unsigned long long)n_seamcalls);
}

int tsimd_vocab_spm_seam_ok(const tsimd_vocab_t *v) {
    tsimd_vocab_t *m = (tsimd_vocab_t *)v;
    if (m->spm_seam_ok) return m->spm_seam_ok == 1;
    int ok = 1;
    for (uint32_t id = 0; id < v->vocab_size && ok; id++) {
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t n = v->offsets[id + 1] - v->offsets[id];
        uint32_t i = 0;
        int prev_meta = 0;
        while (i < n) {
            if (i + 3 <= n && b[i] == 0xE2 && b[i + 1] == 0x96 &&
                b[i + 2] == 0x81) {
                if (i > 0 && !prev_meta) { ok = 0; break; }
                prev_meta = 1;
                i += 3;
                continue;
            }
            prev_meta = 0;
            i++;

        }
    }
    m->spm_seam_ok = ok ? 1 : 2;
    return ok;
}

static int seam_pair_free(const tsimd_vocab_t *v, const uint8_t *b, size_t i,
                          size_t len) {
    uint32_t a_ids[256];
    size_t na = 0;
    uint32_t maxlen = v->max_token_len;

    if (maxlen > 255) return 0;
    for (uint32_t L = 1; L <= maxlen && L <= i; L++) {
        uint32_t id = tsimd_vocab_find(v, b + i - L, L);
        if (id != UINT32_MAX) a_ids[na++] = id;
    }
    for (uint32_t R = 1; R <= maxlen && i + R <= len; R++) {
        uint32_t id = tsimd_vocab_find(v, b + i, R);
        if (id == UINT32_MAX) continue;
        for (size_t k = 0; k < na; k++)
            if (tsimd_vocab_merge_priority(v, a_ids[k], id) != UINT32_MAX)
                return 0;
    }
    return 1;
}

typedef struct {
    uint32_t n;
    uint32_t pre_off[4096], pre_len[4096], post_off[4096], post_len[4096];
    uint8_t bytes[];
} spm_cross_t;

#define TSIMD_SPM_MAX_CROSS 4096

static void spm_cross_build(tsimd_vocab_t *v) {
    size_t cap = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++)
        cap += (size_t)(v->offsets[id + 1] - v->offsets[id]);
    spm_cross_t *c = tsimd_xmalloc(sizeof(spm_cross_t) + cap + 8);
    c->n = 0;
    size_t used = 0;
    int ok = 1;
    for (uint32_t id = 0; id < v->vocab_size && ok; id++) {
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t n = v->offsets[id + 1] - v->offsets[id];
        uint32_t i = 0;
        int prev_mark = 1;
        while (i < n) {
            if (i + 3 <= n && b[i] == 0xE2 && b[i + 1] == 0x96 && b[i + 2] == 0x81) {
                if (!prev_mark) {
                    size_t plen = i, qlen = n - i - 3;
                    if (c->n == TSIMD_SPM_MAX_CROSS || plen == 0) {
                        ok = 0;
                        break;
                    }
                    const uint8_t *post = b + i + 3;
                    for (size_t k = 0; k < plen && ok; k++)
                        if (b[k] == ' ' || (k + 3 <= plen && b[k] == 0xE2 &&
                                            b[k + 1] == 0x96 && b[k + 2] == 0x81))
                            ok = 0;
                    for (size_t k = 0; k < qlen && ok; k++)
                        if (post[k] == ' ' || (k + 3 <= qlen && post[k] == 0xE2 &&
                                               post[k + 1] == 0x96 &&
                                               post[k + 2] == 0x81))
                            ok = 0;
                    if (!ok) break;
                    c->pre_off[c->n] = (uint32_t)used;
                    c->pre_len[c->n] = (uint32_t)plen;
                    memcpy(c->bytes + used, b, plen);
                    used += plen;
                    c->post_off[c->n] = (uint32_t)used;
                    c->post_len[c->n] = (uint32_t)qlen;
                    memcpy(c->bytes + used, post, qlen);
                    used += qlen;
                    v->spm_cross_prev[b[plen - 1] >> 6] |=
                        1ull << (b[plen - 1] & 63);
                    if (qlen)
                        v->spm_cross_post[post[0] >> 6] |=
                            1ull << (post[0] & 63);
                    c->n++;
                }
                prev_mark = 1;
                i += 3;
                continue;
            }
            prev_mark = 0;
            i++;
        }
    }
    if (!ok) {
        tsimd_xfree(c);
        v->spm_cross = NULL;
        v->spm_cross_state = 2;
        return;
    }
    v->spm_cross = c;
    v->spm_cross_state = 1;
}

int tsimd_vocab_spm_cross_ready(const tsimd_vocab_t *v) {
    tsimd_vocab_t *m = (tsimd_vocab_t *)v;
    if (m->spm_cross_state == 0) spm_cross_build(m);
    return m->spm_cross_state == 1;
}

static inline __attribute__((always_inline)) int
spm_boundary_safe(const tsimd_vocab_t *v, const uint8_t *bytes,
                  size_t len, size_t i) {
    if (!tsimd_vocab_spm_cross_ready(v)) return -1;
    const spm_cross_t *c = (const spm_cross_t *)v->spm_cross;
    if (c->n == 0) return 1;
    uint8_t prev = bytes[i - 1];
    if (!(v->spm_cross_prev[prev >> 6] & (1ull << (prev & 63)))) return 1;
    if (i + 3 < len) {
        uint8_t nxt = bytes[i + 3];
        if (!(v->spm_cross_post[nxt >> 6] & (1ull << (nxt & 63)))) return 1;
    }
    for (uint32_t k = 0; k < c->n; k++) {
        size_t plen = c->pre_len[k], qlen = c->post_len[k];
        if (i < plen) continue;
        if (memcmp(bytes + i - plen, c->bytes + c->pre_off[k], plen) != 0)
            continue;
        if (i + 3 + qlen > len) continue;
        if (memcmp(bytes + i + 3, c->bytes + c->post_off[k], qlen) != 0)
            continue;
        return 0;
    }
    return 1;
}

size_t tsimd_spm_seam_find(const tsimd_vocab_t *v, const uint8_t *bytes,
                           size_t len, size_t want, size_t min_pos) {
    if (want < min_pos) want = min_pos;
    int coarse = tsimd_vocab_spm_seam_ok(v);
    size_t i = want;
    while (i + 3 <= len) {
        const uint8_t *hit = (const uint8_t *)memchr(bytes + i, 0xE2, len - i);
        if (!hit) break;
        i = (size_t)(hit - bytes);
        if (i + 3 > len) break;
        if (bytes[i + 1] == 0x96 && bytes[i + 2] == 0x81 && i != 0 &&
            !(i >= 3 && bytes[i - 3] == 0xE2 && bytes[i - 2] == 0x96 &&
              bytes[i - 1] == 0x81)) {
            int safe = coarse ? 1 : spm_boundary_safe(v, bytes, len, i);
            if (safe != 0 &&
                !(safe < 0 && !seam_pair_free(v, bytes, i, len)))
                return i;
        }
        i++;
    }
    return 0;
}

size_t tsimd_spm_word_target(const tsimd_vocab_t *v, size_t block_target) {
    static atomic_int wt_state;
    int wv = atomic_load(&wt_state);
    if (wv == 0) {
        size_t r = 4;
        const char *e = getenv("TSIMD_SPM_WORD_TARGET");
        if (e && e[0]) {
            long k = strtol(e, NULL, 10);
            if (k >= 0 && k <= (1 << 20)) r = (size_t)k;
        }
        atomic_store(&wt_state, (int)(r + 1));
        wv = atomic_load(&wt_state);
    }
    size_t t = (size_t)(wv - 1);
    if (t == 0) return block_target;
    if (!tsimd_vocab_spm_cross_ready(v)) return block_target;
    if (t >= block_target) return block_target;
    return t;
}

int tsimd_bpe_encode_spm_blocked(const tsimd_vocab_t *v,
                                 tsimd_bpe_workspace_t *ws,
                                 const uint8_t *bytes, uint32_t len,
                                 size_t target) {
    if (len == 0) return 0;
    if (target == 0 || len < 2 * target)
        return tsimd_bpe_encode_spm(v, ws, bytes, len);
    size_t pos = 0;
    while (pos < len) {
        size_t end = len;
        if ((size_t)len - pos > target) {
            size_t seam = tsimd_spm_seam_find(v, bytes, len, pos + target,
                                              pos + target / 2);
            if (seam != 0) end = seam;
        }
        if (tsimd_bpe_encode_spm(v, ws, bytes + pos,
                                 (uint32_t)(end - pos)) != 0)
            return -1;
        pos = end;
    }
    tsimd_spm_stats_report();
    return 0;
}

typedef struct {
    const tsimd_vocab_t *v;
    const uint8_t *bytes;
    size_t lo, hi;
    const size_t *seams;
    size_t nseams;
    tsimd_bpe_workspace_t ws;
    tsimd_pretok_cache_t *cache;
    tsimd_pretok_cache_t *shared;
    uint64_t cyc;
    size_t nbytes;
    int rc;
} seam_stripe_t;

static void *seam_stripe_run(void *p) {
    seam_stripe_t *s = (seam_stripe_t *)p;
    if (unit_dbg())
        fprintf(stderr, "stripe lo=%zu hi=%zu nseams=%zu\n", s->lo, s->hi,
                s->nseams);
    uint64_t t0 = getenv("TSIMD_POOL_DBG") ? spm_rdtsc() : 0;
    size_t pos = s->lo;
    size_t wt = tsimd_spm_word_target(s->v, (size_t)1 << 14);
    for (size_t k = 0; k < s->nseams && s->rc == 0; k++) {
        if (tsimd_bpe_encode_spm_blocked_cached(s->v, s->cache, s->shared,
                                                &s->ws,
                                                s->bytes + pos,
                                                (uint32_t)(s->seams[k] - pos),
                                                wt) != 0)
            s->rc = -1;
        pos = s->seams[k];
    }
    if (s->rc == 0 && pos < s->hi)
        if (tsimd_bpe_encode_spm_blocked_cached(s->v, s->cache, s->shared,
                                                &s->ws,
                                                s->bytes + pos,
                                                (uint32_t)(s->hi - pos),
                                                wt) != 0)
            s->rc = -1;
    if (t0) {
        s->cyc = spm_rdtsc() - t0;
        s->nbytes = s->hi - s->lo;
        s->cyc = s->cyc;
    } else {
        s->cyc = 0;
        s->nbytes = 0;
    }
    return NULL;
}

int tsimd_bpe_encode_spm_blocked_par_cached(const tsimd_vocab_t *v,
                                          tsimd_pretok_cache_t *cache,
                                          tsimd_pretok_cache_t *const *wcaches,
                                     tsimd_bpe_workspace_t *ws,
                                     const uint8_t *bytes, uint32_t len,
                                     size_t target, int threads) {
    if (threads <= 1 || target == 0 || len < 2 * target)
        return cache ? tsimd_bpe_encode_spm_blocked_cached(v, cache, cache,
                                                           ws, bytes, len,
                                                           target)
                     : tsimd_bpe_encode_spm_blocked(v, ws, bytes, len,
                                                    target);
    const int dbg = getenv("TSIMD_POOL_DBG") != NULL;
    uint64_t d_seam = 0, d_setup = 0, d_run = 0, d_join = 0, d_asm = 0, d_t0 = 0;
    if (dbg) d_t0 = spm_rdtsc();
    size_t cap = (size_t)len / target + 2;
    size_t *seams = malloc(cap * sizeof(size_t));
    if (!seams)
        return cache ? tsimd_bpe_encode_spm_blocked_cached(v, cache, cache,
                                                           ws, bytes, len,
                                                           target)
                     : tsimd_bpe_encode_spm_blocked(v, ws, bytes, len,
                                                    target);
    size_t ns = 0, pos = 0;
    while ((size_t)len - pos > target && ns + 1 < cap) {
        size_t seam = tsimd_spm_seam_find(v, bytes, len, pos + target,
                                          pos + target / 2);
        if (seam == 0) break;
        seams[ns++] = seam;
        pos = seam;
    }
    if (dbg) { uint64_t t = spm_rdtsc(); d_seam = t - d_t0; d_t0 = t; }
    size_t nblocks = ns + 1;
    size_t nthreads = (size_t)threads < nblocks ? (size_t)threads : nblocks;
    if (nthreads <= 1) {
        free(seams);
        return cache ? tsimd_bpe_encode_spm_blocked_cached(v, cache, cache,
                                                           ws, bytes, len,
                                                           target)
                     : tsimd_bpe_encode_spm_blocked(v, ws, bytes, len,
                                                    target);
    }
    seam_stripe_t *st = calloc(nthreads, sizeof *st);
    pthread_t *tids = calloc(nthreads, sizeof *tids);
    int rc = 0;
    if (!st || !tids) rc = -1;
    if (rc == 0) {

        for (size_t t = 0; t < nthreads; t++) {
            size_t tb = nblocks * t / nthreads;
            size_t te = nblocks * (t + 1) / nthreads;
            st[t].v = v;
            st[t].cache = wcaches ? wcaches[t] : cache;
            st[t].shared = cache;
            st[t].bytes = bytes;
            st[t].lo = tb == 0 ? 0 : seams[tb - 1];
            st[t].hi = te >= nblocks ? len : seams[te - 1];
            st[t].seams = seams + tb;
            st[t].nseams = (te > tb) ? te - tb - 1 : 0;
            if (te <= tb) st[t].hi = st[t].lo;
            st[t].rc = 0;
            st[t].cyc = 0;
            st[t].nbytes = 0;
            if (tsimd_bpe_workspace_init(&st[t].ws, v->max_token_len) != 0) {
                rc = -1;
                break;
            }
        }
    }
    if (dbg) { uint64_t t = spm_rdtsc(); d_setup = t - d_t0; d_t0 = t; }
    if (rc == 0) {
        size_t launched = 0;
        for (size_t t = 1; t < nthreads; t++) {
            if (st[t].lo == st[t].hi) continue;
            if (pthread_create(&tids[t], NULL, seam_stripe_run, &st[t]) != 0) {

                seam_stripe_run(&st[t]);
                continue;
            }
            launched++;
        }
        seam_stripe_run(&st[0]);
        if (dbg) { uint64_t t = spm_rdtsc(); d_run = t - d_t0; d_t0 = t; }
        for (size_t t = 1; t < nthreads; t++)
            if (tids[t]) pthread_join(tids[t], NULL);
        (void)launched;
    }
    if (dbg) { uint64_t t = spm_rdtsc(); d_join = t - d_t0; d_t0 = t; }
    if (getenv("TSIMD_POOL_DBG")) {
        uint64_t cyc = 0;
        size_t by = 0;
        for (size_t t = 0; t < nthreads; t++) {
            cyc += st[t].cyc;
            by += st[t].nbytes;
        }
        if (by) {
            fprintf(stderr,
                    "tsimd spm stripe: %zu bytes over %zu workers, %.1f "
                    "TSC/byte worker-side\n",
                    by, nthreads, (double)cyc / (double)by);
            for (size_t t = 0; t < nthreads; t++)
                fprintf(stderr, "  w%zu: %zu bytes, %.2f Mcyc\n", t,
                        st[t].nbytes, (double)st[t].cyc / 1e6);
        }
    }
    if (dbg && rc == 0) {
        fprintf(stderr,
                "tsimd spm phase: seam %.1f setup %.1f run %.1f join %.1f "
                "Mcyc\n", (double)d_seam / 1e6, (double)d_setup / 1e6,
                (double)d_run / 1e6, (double)d_join / 1e6);
    }
    if (rc == 0) {
        size_t total = 0;
        for (size_t t = 0; t < nthreads; t++) {
            if (st[t].rc != 0) rc = -1;
            total += st[t].ws.out_count;
        }
        if (rc == 0) {
            while (ws->out_capacity < ws->out_count + total)
                ws->out_capacity *= 2;
            ws->out = tsimd_xrealloc(ws->out,
                                     ws->out_capacity * sizeof(uint32_t));
            size_t off = ws->out_count;
            for (size_t t = 0; t < nthreads; t++) {
                memcpy(ws->out + off, st[t].ws.out,
                       st[t].ws.out_count * sizeof(uint32_t));
                off += st[t].ws.out_count;
            }
            ws->out_count = off;
        }
    }
    if (st)
        for (size_t t = 0; t < nthreads; t++)
            tsimd_bpe_workspace_free(&st[t].ws);
    free(st);
    free(tids);
    free(seams);
    return rc;
}

typedef struct {
    float score;
    uint32_t prev;
    uint32_t id;
} tsimd_uni_node_t;

static inline __attribute__((always_inline)) void
tsimd_unigram_scan(const tsimd_vocab_t *v, tsimd_uni_node_t *st, float *sec,
                   const uint8_t *bytes, uint32_t len,
                   const tsimd_trie_t *trie, uint32_t root_cu, uint32_t carry,
                   float unk_score, uint32_t max_len, float kThreshold,
                   int want_sec) {
    uint32_t starts_at = 0;
    {
        const char *nr = getenv("TSIMD_UNI_NO_REBASE");
        if (nr && nr[0] == '1') kThreshold = 1e30f;
    }
    while (starts_at < len) {
        float bsh = st[starts_at].score;
        if (bsh < -kThreshold || bsh > kThreshold) {
            const float rb_off = bsh;
            uint32_t rb_end = starts_at + max_len;
            if (rb_end > len) rb_end = len;
            for (uint32_t i = starts_at; i <= rb_end; i++)
                if (i == starts_at || st[i].prev != UINT32_MAX)
                    st[i].score -= rb_off;
            bsh = 0.0f;
        }

        uint32_t mblen = 1;
        if (bytes[starts_at] >= 0x80) {
            uint32_t cp;
            size_t a = tsimd_utf8_decode(bytes + starts_at, len - starts_at,
                                         &cp);
            if (a > 0) mblen = (uint32_t)a;
            if (mblen > len - starts_at) mblen = len - starts_at;
        }
        int have_one = 0;
        uint32_t remain = len - starts_at;
        uint32_t lmax = max_len < remain ? max_len : remain;
        uint32_t bcap = v->maxlen1[bytes[starts_at]];
        if (bcap > lmax) bcap = lmax;

        uint32_t trie_cur = trie ? tsimd_trie_root(trie) : TSIMD_TRIE_NIL;
        uint32_t trie_cu = root_cu;
        const uint32_t *restrict tu = trie ? trie->u : NULL;
        const tsimd_piece_t *restrict tp = trie ? trie->piece : NULL;
        const uint8_t *bs = bytes + starts_at;
        const uint8_t *bp = bs;
        tsimd_uni_node_t *dst = st + starts_at + 1;
        float *sp = sec ? sec + starts_at + 1 : NULL;
        for (uint32_t blen = 1; blen <= bcap; blen++, bp++, dst++, sp++) {
            uint32_t id;
            float pscore;
            if (trie) {
                if (carry) {
                    if (!(trie_cu & TSIMD_TRIE_CHILD)) break;
                    uint32_t ii = (trie_cu >> TSIMD_TRIE_BASE_SHIFT) + *bp;
                    trie_cu = tu[ii];
                    if ((trie_cu & TSIMD_TRIE_LABEL) != *bp) break;
                    id = tp[ii].id;
                    if (id == TSIMD_TRIE_NIL) continue;
                    pscore = tp[ii].score;
                } else {
                    if (!(tu[trie_cur] & TSIMD_TRIE_CHILD)) break;
                    uint32_t ii = (tu[trie_cur] >> TSIMD_TRIE_BASE_SHIFT) + *bp;
                    uint32_t nu = tu[ii];
                    if ((nu & TSIMD_TRIE_LABEL) != *bp) break;
                    trie_cur = nu;
                    id = tp[ii].id;
                    if (id == TSIMD_TRIE_NIL) continue;
                    pscore = tp[ii].score;
                }
            } else {
                id = tsimd_vocab_find(v, bs, blen);
                if (id == UINT32_MAX) continue;
                pscore = v->scores[id];
            }
            float cand = pscore + bsh;
            if (dst->prev == UINT32_MAX || cand > dst->score) {
                if (want_sec && dst->prev != UINT32_MAX) *sp = dst->score;
                dst->score = cand;
                dst->prev = starts_at;
                dst->id = id;
            } else if (want_sec && cand > *sp) {
                *sp = cand;
            }
            if (blen == mblen) have_one = 1;
        }
        if (!have_one) {
            uint32_t e = starts_at + mblen;
            float cand = unk_score + bsh;
            uint32_t back = starts_at;
            if (st[starts_at].id == v->spm_unk_id &&
                st[starts_at].prev != UINT32_MAX)
                back = st[starts_at].prev;
            for (;;) {
                tsimd_uni_node_t *ud = &st[e];
                if (ud->prev != UINT32_MAX && ud->id == v->spm_unk_id &&
                    cand <= ud->score) {
                    if (want_sec && cand > sec[e]) sec[e] = cand;
                    break;
                }
                if (ud->prev == UINT32_MAX) {
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                } else if (cand > ud->score) {
                    if (sec) sec[e] = ud->score;
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                } else if (want_sec && cand > sec[e]) {
                    sec[e] = cand;
                }
                if (e >= len) break;
                uint32_t cp;
                size_t a = tsimd_utf8_decode(bytes + e, len - e, &cp);
                if (a == 0) a = 1;
                uint32_t nxt = (uint32_t)a;
                if (tsimd_vocab_find(v, bytes + e, nxt) != UINT32_MAX) break;
                e += nxt;
                cand += unk_score;
            }
            starts_at += mblen;
            continue;
        }
        starts_at += mblen;
    }
}

int tsimd_unigram_encode_ex(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                            const uint8_t *bytes, uint32_t len,
                            float offset, float *out_score,
                            float *min_margin) {

    if (len == 0) {

        if (out_score) *out_score = offset;
        return 0;
    }
    if (!v->scores) return -1;
    tsimd_uni_node_t *st;
    size_t st_need = ((size_t)len + 1) * sizeof(tsimd_uni_node_t);
    if (ws->uni_st_cap < st_need) {
        tsimd_xfree(ws->uni_st);
        ws->uni_st = tsimd_xmalloc(st_need);
        ws->uni_st_cap = ws->uni_st ? st_need : 0;
    }
    st = (tsimd_uni_node_t *)ws->uni_st;
    float *sec = NULL;
    if (min_margin) {
        sec = tsimd_xmalloc(((size_t)len + 1) * sizeof(float));
        if (sec) {
            for (uint32_t e = 0; e <= len; e++) sec[e] = -INFINITY;
            *min_margin = INFINITY;
        }
    }
    if (!st || (min_margin && !sec)) {
        tsimd_xfree(sec);
        return -1;
    }
    for (uint32_t e = 0; e <= len; e++) {
        st[e].score = 0.0f;
        st[e].prev = UINT32_MAX;
        st[e].id = 0;
    }
    st[0].score = offset;
    uint32_t max_len = v->max_token_len;
    if (max_len > len) max_len = len;
    const float unk_score = v->spm_min_score - 10.0f;
    const float kThreshold = 100000.0f;
    const tsimd_trie_t *trie = (const tsimd_trie_t *)v->spm_trie;
    {
        const char *e = getenv("TSIMD_DISABLE_SPM_TRIE");
        if (trie && e && e[0] == '1') trie = NULL;
    }
    uint32_t carry = 1;
    {
        const char *e = getenv("TSIMD_DISABLE_DP_PROLOGUE");
        if (e && e[0] == '1') carry = 0;
    }
    uint32_t root_cu = trie ? trie->u[0] : 0;
    if (sec) {
        if (carry) {
            tsimd_unigram_scan(v, st, sec, bytes, len, trie, root_cu, 1,
                               unk_score, max_len, kThreshold, 1);
        } else {
            tsimd_unigram_scan(v, st, sec, bytes, len, trie, root_cu, 0,
                               unk_score, max_len, kThreshold, 1);
        }
    } else if (carry) {
        tsimd_unigram_scan(v, st, sec, bytes, len, trie, root_cu, 1, unk_score,
                           max_len, kThreshold, 0);
    } else {
        tsimd_unigram_scan(v, st, sec, bytes, len, trie, root_cu, 0, unk_score,
                           max_len, kThreshold, 0);
    }

    if (min_margin) {
        float m = INFINITY;
        for (uint32_t e = 0; e <= len; e++)
            if (sec[e] != -INFINITY && st[e].score - sec[e] < m)
                m = st[e].score - sec[e];
        *min_margin = m;
    }

    {
        size_t base = ws->out_count;
        uint32_t e = len;
        while (e > 0 && st[e].prev != UINT32_MAX) {
            if (out_push(ws, st[e].id) != 0) {
                tsimd_xfree(sec);
                return -1;
            }
            e = st[e].prev;
        }
        size_t lo = base, hi = ws->out_count;
        while (lo + 1 < hi) {
            uint32_t tmp = ws->out[lo];
            ws->out[lo] = ws->out[hi - 1];
            ws->out[hi - 1] = tmp;
            lo++;
            hi--;
        }
    }
    if (out_score) *out_score = st[len].score;
    tsimd_xfree(sec);
    return 0;
}

int tsimd_unigram_encode(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                         const uint8_t *bytes, uint32_t len) {
    return tsimd_unigram_encode_ex(v, ws, bytes, len, 0.0f, NULL, NULL);
}

typedef struct {
    uint16_t blen;
    uint16_t pad;
    uint32_t id;
    float pscore;
} tsimd_uni_match_t;

typedef struct {
    const tsimd_vocab_t *v;
    const uint8_t *bytes;
    uint32_t len;
    uint32_t lo, hi;
    const tsimd_trie_t *trie;
    uint32_t root_cu;
    uint32_t max_len;
    uint32_t carry;
    tsimd_uni_match_t *m;
    size_t n, cap;
    uint16_t *cnt;
    uint32_t hi_un;
    int rc;
} uni_enum_job_t;

static void uni_enum_range(uni_enum_job_t *j) {
    const tsimd_vocab_t *v = j->v;
    const uint8_t *bytes = j->bytes;
    const uint32_t len = j->len;
    const tsimd_trie_t *trie = j->trie;
    const uint32_t *restrict tu = trie ? trie->u : NULL;
    const tsimd_piece_t *restrict tp = trie ? trie->piece : NULL;
    uint32_t starts_at = j->lo;
    if (starts_at < j->hi)
        while (starts_at < j->hi && (bytes[starts_at] & 0xC0) == 0x80)
            starts_at++;
    for (; starts_at < j->hi; ) {
        uint32_t remain = len - starts_at;
        uint32_t lmax = j->max_len < remain ? j->max_len : remain;
        uint32_t bcap = v->maxlen1[bytes[starts_at]];
        if (bcap > lmax) bcap = lmax;
        uint32_t trie_cur = trie ? tsimd_trie_root(trie) : TSIMD_TRIE_NIL;
        uint32_t trie_cu = j->root_cu;
        const uint8_t *bp = bytes + starts_at;
        uint16_t k = 0;
        uint32_t mblen = 1;
        if (bytes[starts_at] >= 0x80) {
            uint32_t cp;
            size_t a = tsimd_utf8_decode(bytes + starts_at, len - starts_at,
                                         &cp);
            if (a > 0) mblen = (uint32_t)a;
            if (mblen > len - starts_at) mblen = len - starts_at;
        }
        uint32_t idx = starts_at - j->lo;
        for (uint32_t blen = 1; blen <= bcap; blen++, bp++) {
            uint32_t id;
            float pscore;
            if (trie) {
                if (j->carry) {
                    if (!(trie_cu & TSIMD_TRIE_CHILD)) break;
                    uint32_t ii = (trie_cu >> TSIMD_TRIE_BASE_SHIFT) + *bp;
                    trie_cu = tu[ii];
                    if ((trie_cu & TSIMD_TRIE_LABEL) != *bp) break;
                    id = tp[ii].id;
                    if (id == TSIMD_TRIE_NIL) continue;
                    pscore = tp[ii].score;
                } else {
                    if (!(tu[trie_cur] & TSIMD_TRIE_CHILD)) break;
                    uint32_t ii = (tu[trie_cur] >> TSIMD_TRIE_BASE_SHIFT) + *bp;
                    uint32_t nu = tu[ii];
                    if ((nu & TSIMD_TRIE_LABEL) != *bp) break;
                    trie_cur = nu;
                    id = tp[ii].id;
                    if (id == TSIMD_TRIE_NIL) continue;
                    pscore = tp[ii].score;
                }
            } else {
                id = tsimd_vocab_find(v, bytes + starts_at, blen);
                if (id == UINT32_MAX) continue;
                pscore = v->scores[id];
            }
            if (j->n == j->cap) {
                size_t nc = j->cap ? j->cap * 2 : 256;
                tsimd_uni_match_t *nm =
                    (tsimd_uni_match_t *)realloc(j->m, nc * sizeof(*nm));
                if (!nm) { j->rc = -1; return; }
                j->m = nm;
                j->cap = nc;
            }
            j->m[j->n].blen = (uint16_t)blen;
            j->m[j->n].pad = 0;
            j->m[j->n].id = id;
            j->m[j->n].pscore = pscore;
            j->n++;
            k++;
        }
        j->cnt[idx] = (uint16_t)(k | (mblen << 12));
        starts_at += mblen;
    }
}

static void *uni_enum_thread(void *arg) {
    uni_enum_range((uni_enum_job_t *)arg);
    return NULL;
}

static int uni_cut_safe(const tsimd_vocab_t *v, const uint8_t *b,
                        uint32_t len, uint32_t cut, int *anchored_out) {
    const tsimd_trie_t *trie = (const tsimd_trie_t *)v->spm_trie;
    if (!trie) return 1;
    if (cut == 0 || cut >= len) return 1;
    uint32_t max_len = v->max_token_len;
    if (max_len > cut) max_len = cut;
    int anchored = 0;
    uint32_t s0 = cut >= max_len ? cut - max_len : 0;
    for (uint32_t stp = s0; stp <= cut && stp < len; stp++) {
        if (stp < cut && (b[stp] & 0xC0) == 0x80) continue;
        uint32_t cu = trie->u[0];
        uint32_t cl = 1;
        if (stp == cut) {
            uint32_t cp;
            size_t a = tsimd_utf8_decode(b + stp, len - stp, &cp);
            if (a) cl = (uint32_t)a;
        }
        for (uint32_t k = 0; stp + k < len && k < max_len; k++) {
            uint32_t idx = 0, unit = 0;
            uint32_t nx = tsimd_trie_advance(trie, cu, b[stp + k], &idx,
                                             &unit);
            if (nx == TSIMD_TRIE_NIL) break;
            cu = unit;
            uint32_t pid = tsimd_trie_piece_id(trie, nx, idx);
            if (pid == TSIMD_TRIE_NIL) continue;
            uint32_t end = stp + k + 1;
            if (stp < cut && end > cut) return 0;
            if (stp == cut && end == stp + cl) anchored = 1;
        }
    }
    *anchored_out = anchored;
    return 1;
}

static void uni_replay(const tsimd_vocab_t *v, tsimd_uni_node_t *st,
                       const uint8_t *bytes, uint32_t len,
                       uint32_t w, uint32_t we,
                       const tsimd_uni_match_t **seg, const uint16_t **cnts,
                       const uint32_t *seg_lo, const uint32_t *seg_hi,
                       int nseg, float unk_score, uint32_t max_len,
                       float kThreshold) {
    int ti = 0;
    size_t cur = 0;
    for (uint32_t starts_at = w; starts_at < we; ) {
        while (ti + 1 < nseg && starts_at >= seg_hi[ti]) {
            ti++;
            cur = 0;
        }
        float bsh = st[starts_at].score;
        if (bsh < -kThreshold || bsh > kThreshold) {
            const float rb_off = bsh;
            uint32_t rb_end = starts_at + max_len;
            if (rb_end > len) rb_end = len;
            for (uint32_t i = starts_at; i <= rb_end; i++)
                if (i == starts_at || st[i].prev != UINT32_MAX)
                    st[i].score -= rb_off;
            bsh = 0.0f;
        }
        uint16_t cc = cnts[ti][starts_at - seg_lo[ti]];
        uint16_t k = cc & 0xFFFu;
        uint32_t mblen = (uint32_t)(cc >> 12);
        if (mblen == 0) mblen = 1;
        int have_one = 0;
        for (uint16_t i = 0; i < k; i++) {
            const tsimd_uni_match_t *mm = &seg[ti][cur];
            cur++;
            float cand = mm->pscore + bsh;
            tsimd_uni_node_t *dst = st + starts_at + mm->blen;
            if (dst->prev == UINT32_MAX || cand > dst->score) {
                dst->score = cand;
                dst->prev = starts_at;
                dst->id = mm->id;
            }
            if (mm->blen == mblen) have_one = 1;
        }
        if (!have_one) {
            uint32_t e = starts_at + mblen;
            float cand = unk_score + bsh;
            uint32_t back = starts_at;
            if (st[starts_at].id == v->spm_unk_id &&
                st[starts_at].prev != UINT32_MAX)
                back = st[starts_at].prev;
            for (;;) {
                tsimd_uni_node_t *ud = &st[e];
                if (ud->prev != UINT32_MAX && ud->id == v->spm_unk_id &&
                    cand <= ud->score)
                    break;
                if (ud->prev == UINT32_MAX) {
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                } else if (cand > ud->score) {
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                }
                if (e >= len) break;
                uint32_t cp;
                size_t a = tsimd_utf8_decode(bytes + e, len - e, &cp);
                if (a == 0) a = 1;
                uint32_t nxt = (uint32_t)a;
                if (tsimd_vocab_find(v, bytes + e, nxt) != UINT32_MAX) break;
                e += nxt;
                cand += unk_score;
            }
        }
        starts_at += mblen;
    }
}

static void uni_replay_windowed(const tsimd_vocab_t *v,
                                tsimd_uni_node_t *st, uint32_t base,
                                const uint8_t *bytes, uint32_t len,
                       uint32_t w, uint32_t we,
                       const tsimd_uni_match_t **seg, const uint16_t **cnts,
                       const uint32_t *seg_lo, const uint32_t *seg_hi,
                       int nseg, float unk_score, uint32_t max_len,
                       float kThreshold) {
    int ti = 0;
    size_t cur = 0;
    for (uint32_t starts_at = w; starts_at < we; ) {
        while (ti + 1 < nseg && starts_at >= seg_hi[ti]) {
            ti++;
            cur = 0;
        }
        float bsh = st[starts_at - base].score;
        if (bsh < -kThreshold || bsh > kThreshold) {
            const float rb_off = bsh;
            uint32_t rb_end = starts_at + max_len;
            if (rb_end > we) rb_end = we;
            for (uint32_t i = starts_at; i <= rb_end; i++)
                if (i == starts_at || st[i - base].prev != UINT32_MAX)
                    st[i - base].score -= rb_off;
            bsh = 0.0f;
        }
        uint16_t cc = cnts[ti][starts_at - seg_lo[ti]];
        uint16_t k = cc & 0xFFFu;
        uint32_t mblen = (uint32_t)(cc >> 12);
        if (mblen == 0) mblen = 1;
        int have_one = 0;
        for (uint16_t i = 0; i < k; i++) {
            const tsimd_uni_match_t *mm = &seg[ti][cur];
            cur++;
            float cand = mm->pscore + bsh;
            tsimd_uni_node_t *dst = st + starts_at + mm->blen - base;
            if (dst->prev == UINT32_MAX || cand > dst->score) {
                dst->score = cand;
                dst->prev = starts_at;
                dst->id = mm->id;
            }
            if (mm->blen == mblen) have_one = 1;
        }
        if (!have_one) {
            uint32_t e = starts_at + mblen;
            float cand = unk_score + bsh;
            uint32_t back = starts_at;
            if (st[starts_at - base].id == v->spm_unk_id &&
                st[starts_at - base].prev != UINT32_MAX) {
                uint32_t pb2 = st[starts_at - base].prev;
                if (pb2 >= w) back = pb2;
            }
            for (;;) {
                tsimd_uni_node_t *ud = &st[e - base];
                if (ud->prev != UINT32_MAX && ud->id == v->spm_unk_id &&
                    cand <= ud->score)
                    break;
                if (ud->prev == UINT32_MAX) {
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                } else if (cand > ud->score) {
                    ud->score = cand;
                    ud->prev = back;
                    ud->id = v->spm_unk_id;
                }
                if (e >= we) break;
                uint32_t cp;
                size_t a = tsimd_utf8_decode(bytes + e, len - e, &cp);
                if (a == 0) a = 1;
                uint32_t nxt = (uint32_t)a;
                if (tsimd_vocab_find(v, bytes + e, nxt) != UINT32_MAX) break;
                e += nxt;
                cand += unk_score;
            }
        }
        starts_at += mblen;
    }
}

static size_t uni_par_windows = 0;

void tsimd_unigram_par_stats(size_t *windows) {
    *windows = uni_par_windows;
}

static int tsimd_unigram_encode_par_windowed(
    const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
    const uint8_t *bytes, uint32_t len, int threads) {
    if (threads <= 1 || len < (2u << 20))
        return tsimd_unigram_encode(v, ws, bytes, len);
    if (!v->scores) return -1;
    uint32_t max_len = v->max_token_len;
    if (max_len > len) max_len = len;
    const uint32_t W = 2u << 20;
    size_t wcap = (size_t)W + (64u << 10) + 2 * (size_t)max_len + 4;
    size_t st_need = wcap * 2 * sizeof(tsimd_uni_node_t);
    if (ws->uni_st_cap < st_need) {
        tsimd_xfree(ws->uni_st);
        ws->uni_st = tsimd_xmalloc(st_need);
        ws->uni_st_cap = ws->uni_st ? st_need : 0;
    }
    tsimd_uni_node_t *stbase = (tsimd_uni_node_t *)ws->uni_st;
    if (!stbase) return -1;
    tsimd_uni_node_t *stbuf[2] = { stbase, stbase + wcap };
    int par = 0;
    const float unk_score = v->spm_min_score - 10.0f;
    const float kThreshold = 100000.0f;
    const tsimd_trie_t *trie = (const tsimd_trie_t *)v->spm_trie;
    {
        const char *e = getenv("TSIMD_DISABLE_SPM_TRIE");
        if (trie && e && e[0] == '1') trie = NULL;
    }
    uint32_t root_cu = trie ? trie->u[0] : 0;
    uint32_t carry = 1;
    {
        const char *e = getenv("TSIMD_DISABLE_DP_PROLOGUE");
        if (e && e[0] == '1') carry = 0;
    }
    if (threads > 16) threads = 16;
    uni_enum_job_t jobs[2][16];
    pthread_t tids[2][16];
    int pend = 0;
    int pnt = 0;
    int pbi = 0;
    int ppar = 0;
    int ppar_next = 0;
    int p_split = 0;
    int split_next = 0;
    uint32_t pw = 0, pwe = 0;
    uint32_t w = 0;
    float carry_score = 0.0f;
    int nt = threads;
    {
        const char *es = getenv("TSIMD_UNI_SEG");
        if (es && es[0]) {
            int k = atoi(es);
            if (k >= 1 && k <= 16) nt = k;
        }
    }
    while (w < len || pend) {
        int spawned = 0;
        int bi = 0;
        int ntc = 0;
        int start_split = split_next;
        uint32_t sw = 0, swe = 0;
        if (w < len) {
            uint32_t we = w + W;
            if (we > len) we = len;
            else while (we < len && (bytes[we] & 0xC0) == 0x80) we++;
            int anchored = 0;
            split_next = 0;
            while (we < len && we > w &&
                   !uni_cut_safe(v, bytes, len, we, &anchored)) {
                uint32_t cp;
                size_t a = tsimd_utf8_decode(bytes + we, len - we, &cp);
                if (a == 0) a = 1;
                we += (uint32_t)a;
                if (we > len) we = len;
                if (we - w >= W + (64u << 10)) {
                    split_next = 1;
                    break;
                }
            }
            if (!anchored) split_next = 1;
            ntc = nt;
            if ((size_t)ntc > (size_t)(we - w)) ntc = (int)(we - w);
            bi = pend ? (pbi ^ 1) : 0;
            uni_enum_job_t *jb = jobs[bi];
            int ok = 1;
            for (int t = 0; t < ntc; t++) {
                jb[t].v = v;
                jb[t].bytes = bytes;
                jb[t].len = len;
                jb[t].lo =
                    w + (size_t)(we - w) * (uint32_t)t / (uint32_t)ntc;
                jb[t].hi =
                    w + (size_t)(we - w) * (uint32_t)(t + 1) / (uint32_t)ntc;
                jb[t].trie = trie;
                jb[t].root_cu = root_cu;
                jb[t].max_len = max_len;
                jb[t].carry = carry;
                jb[t].m = ws->uni_mcache[bi][t];
                jb[t].cap = ws->uni_mcap[bi][t];
                jb[t].n = 0;
                size_t cneed = jb[t].hi - jb[t].lo;
                if (ws->uni_cccap[bi][t] < cneed) {
                    size_t nc = cneed * 2;
                    uint16_t *ncc = (uint16_t *)realloc(
                        ws->uni_ccache[bi][t], nc * sizeof(uint16_t));
                    if (!ncc) {
                        jb[t].rc = -1;
                        ok = 0;
                        continue;
                    }
                    ws->uni_ccache[bi][t] = ncc;
                    ws->uni_cccap[bi][t] = nc;
                }
                memset(ws->uni_ccache[bi][t], 0, cneed * sizeof(uint16_t));
                jb[t].cnt = ws->uni_ccache[bi][t];
                jb[t].rc = 0;
            }
            if (ok) {
                tsimd_uni_node_t *sb = stbuf[par];
                for (uint32_t e = w; e <= we; e++) {
                    sb[e - w].score = 0.0f;
                    sb[e - w].prev = UINT32_MAX;
                    sb[e - w].id = 0;
                }
                for (int t = 1; t < ntc; t++)
                    pthread_create(&tids[bi][t], NULL, uni_enum_thread,
                                   &jb[t]);
                uni_enum_range(&jb[0]);
            }
            spawned = ok;
            sw = w;
            swe = we;
            w = we;
            if (spawned) {
                ppar_next = par;
                par ^= 1;
            }
        }
        if (pend) {
            uni_enum_job_t *pb = jobs[pbi];
            int bad = 0;
            for (int t = 1; t < pnt; t++)
                pthread_join(tids[pbi][t], NULL);
            for (int t = 0; t < pnt; t++)
                if (pb[t].rc != 0) bad = 1;
            if (!bad) {
                const tsimd_uni_match_t *seg[16];
                const uint16_t *cnts[16];
                uint32_t seg_lo[16], seg_hi[16];
                for (int t = 0; t < pnt; t++) {
                    seg[t] = pb[t].m;
                    cnts[t] = pb[t].cnt;
                    seg_lo[t] = pb[t].lo;
                    seg_hi[t] = pb[t].hi;
                }
                stbuf[ppar][0].score = carry_score;
                uni_replay_windowed(v, stbuf[ppar], pw, bytes, len, pw,
                                    pwe, seg, cnts, seg_lo, seg_hi, pnt,
                                    unk_score, max_len, kThreshold);
                tsimd_uni_node_t *sbw = stbuf[ppar];
                uint32_t e = pwe;
                size_t sn = 0;
                int walked_out = 0;
                for (;;) {
                    if (e <= pw) break;
                    tsimd_uni_node_t *nd = &sbw[e - pw];
                    if (nd->prev == UINT32_MAX) {
                        walked_out = 1;
                        break;
                    }
                    if (ws->uni_seg_cap < sn + 1) {
                        size_t nc = ws->uni_seg_cap
                                        ? ws->uni_seg_cap * 2
                                        : (size_t)1 << 16;
                        uint32_t *nb =
                            realloc(ws->uni_seg, nc * sizeof(uint32_t));
                        if (!nb) return -1;
                        ws->uni_seg = nb;
                        ws->uni_seg_cap = nc;
                    }
                    ws->uni_seg[sn++] = nd->id;
                    e = nd->prev;
                }
                size_t skip = 0;
                if (p_split && sn > 0 &&
                    ws->uni_seg[sn - 1] == v->spm_unk_id &&
                    ws->out_count > 0 &&
                    ws->out[ws->out_count - 1] == v->spm_unk_id)
                    skip = 1;
                for (size_t r = sn - skip; r > 0; r--)
                    if (out_push(ws, ws->uni_seg[r - 1]) != 0) return -1;
                carry_score = sbw[pwe - pw].score;
                if (getenv("TSIMD_UNI_WALK_DBG"))
                    fprintf(stderr,
                            "W pw=%u pwe=%u sn=%zu out=%s stop=%s stop_e=%u\n",
                            pw, pwe, sn, skip ? "skip1" : "full",
                            walked_out ? "PREVMAX" : "base", e);
                uni_par_windows++;
            }
            for (int t = 0; t < pnt; t++) {
                ws->uni_mcache[pbi][t] = pb[t].m;
                ws->uni_mcap[pbi][t] = pb[t].cap;
                ws->uni_ccache[pbi][t] = pb[t].cnt;
            }
            pend = 0;
            if (bad) return -1;
        }
        ppar = ppar_next;
        if (spawned) {
            pend = 1;
            pbi = bi;
            pnt = ntc;
            pw = sw;
            pwe = swe;
            p_split = start_split;
        }
    }
    return 0;
}

static int tsimd_unigram_encode_par_whole(
    const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
    const uint8_t *bytes, uint32_t len, int threads);

int tsimd_unigram_encode_par(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                             const uint8_t *bytes, uint32_t len,
                             int threads) {
    const char *ev = getenv("TSIMD_UNI_WINDOW");
    if (ev && ev[0] == '0')
        return tsimd_unigram_encode_par_whole(v, ws, bytes, len, threads);
    return tsimd_unigram_encode_par_windowed(v, ws, bytes, len, threads);
}

static int tsimd_unigram_encode_par_whole(
    const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
    const uint8_t *bytes, uint32_t len, int threads) {
    if (threads <= 1 || len < (2u << 20))
        return tsimd_unigram_encode(v, ws, bytes, len);
    if (!v->scores) return -1;
    tsimd_uni_node_t *st;
    size_t st_need = ((size_t)len + 1) * sizeof(tsimd_uni_node_t);
    if (ws->uni_st_cap < st_need) {
        tsimd_xfree(ws->uni_st);
        ws->uni_st = tsimd_xmalloc(st_need);
        ws->uni_st_cap = ws->uni_st ? st_need : 0;
    }
    st = (tsimd_uni_node_t *)ws->uni_st;
    if (!st) return -1;
    for (uint32_t e = 0; e <= len; e++) {
        st[e].score = 0.0f;
        st[e].prev = UINT32_MAX;
        st[e].id = 0;
    }
    uint32_t max_len = v->max_token_len;
    if (max_len > len) max_len = len;
    const float unk_score = v->spm_min_score - 10.0f;
    const float kThreshold = 100000.0f;
    const tsimd_trie_t *trie = (const tsimd_trie_t *)v->spm_trie;
    {
        const char *e = getenv("TSIMD_DISABLE_SPM_TRIE");
        if (trie && e && e[0] == '1') trie = NULL;
    }
    uint32_t root_cu = trie ? trie->u[0] : 0;
    uint32_t carry = 1;
    {
        const char *e = getenv("TSIMD_DISABLE_DP_PROLOGUE");
        if (e && e[0] == '1') carry = 0;
    }
    if (threads > 16) threads = 16;
    const uint32_t W = 2u << 20;
    uni_enum_job_t jobs[2][16];
    pthread_t tids[2][16];
    int pend = 0;
    int pnt = 0;
    int pbi = 0;
    uint32_t pw = 0, pwe = 0;
    uint32_t w = 0;
    int nt = threads;
    {
        const char *es = getenv("TSIMD_UNI_SEG");
        if (es && es[0]) {
            int k = atoi(es);
            if (k >= 1 && k <= 16) nt = k;
        }
    }
    while (w < len || pend) {
        int spawned = 0;
        int bi = 0;
        int ntc = 0;
        uint32_t sw = 0, swe = 0;
        if (w < len) {
            uint32_t we = w + W;
            if (we > len) we = len;
            else while (we < len && (bytes[we] & 0xC0) == 0x80) we++;
            ntc = nt;
            if ((size_t)ntc > (size_t)(we - w)) ntc = (int)(we - w);
            bi = pend ? (pbi ^ 1) : 0;
            uni_enum_job_t *jb = jobs[bi];
            int ok = 1;
            for (int t = 0; t < ntc; t++) {
                jb[t].v = v;
                jb[t].bytes = bytes;
                jb[t].len = len;
                jb[t].lo =
                    w + (size_t)(we - w) * (uint32_t)t / (uint32_t)ntc;
                jb[t].hi =
                    w + (size_t)(we - w) * (uint32_t)(t + 1) / (uint32_t)ntc;
                jb[t].trie = trie;
                jb[t].root_cu = root_cu;
                jb[t].max_len = max_len;
                jb[t].carry = carry;
                jb[t].m = NULL;
                jb[t].n = 0;
                jb[t].cap = 0;
                jb[t].cnt = (uint16_t *)calloc(jb[t].hi - jb[t].lo,
                                               sizeof(uint16_t));
                jb[t].rc = jb[t].cnt ? 0 : -1;
                if (jb[t].rc != 0) ok = 0;
            }
            if (ok) {
                for (int t = 1; t < ntc; t++)
                    pthread_create(&tids[bi][t], NULL, uni_enum_thread,
                                   &jb[t]);
                uni_enum_range(&jb[0]);
            }
            spawned = ok;
            sw = w;
            swe = we;
            w = we;
        }
        if (pend) {
            uni_enum_job_t *pb = jobs[pbi];
            int bad = 0;
            for (int t = 1; t < pnt; t++)
                pthread_join(tids[pbi][t], NULL);
            for (int t = 0; t < pnt; t++)
                if (pb[t].rc != 0) bad = 1;
            if (!bad) {
                const tsimd_uni_match_t *seg[16];
                const uint16_t *cnts[16];
                uint32_t seg_lo[16], seg_hi[16];
                for (int t = 0; t < pnt; t++) {
                    seg[t] = pb[t].m;
                    cnts[t] = pb[t].cnt;
                    seg_lo[t] = pb[t].lo;
                    seg_hi[t] = pb[t].hi;
                }
                uni_replay(v, st, bytes, len, pw, pwe, seg, cnts,
                           seg_lo, seg_hi, pnt, unk_score, max_len,
                           kThreshold);
                uni_par_windows++;
            }
            for (int t = 0; t < pnt; t++) {
                free(pb[t].m);
                free(pb[t].cnt);
            }
            pend = 0;
            if (bad) return -1;
        }
        if (spawned) {
            pend = 1;
            pbi = bi;
            pnt = ntc;
            pw = sw;
            pwe = swe;
        }
    }
    {
        size_t base = ws->out_count;
        uint32_t e = len;
        while (e > 0 && st[e].prev != UINT32_MAX) {
            if (out_push(ws, st[e].id) != 0) return -1;
            e = st[e].prev;
        }
        size_t lo = base, hi = ws->out_count;
        while (lo + 1 < hi) {
            uint32_t tmp = ws->out[lo];
            ws->out[lo] = ws->out[hi - 1];
            ws->out[hi - 1] = tmp;
            lo++;
            hi--;
        }
    }
    return 0;
}

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static inline uint64_t span_hash_words(uint64_t a, uint64_t b) {
    return tsimd_pcache_hash_words_64(a, b);
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("sse4.2"), noinline))
uint64_t tsimd_hash_words_crc32(uint64_t a, uint64_t b) {
    return (uint64_t)__builtin_ia32_crc32di(
        (uint32_t)__builtin_ia32_crc32di(0, a), b);
}
#else
uint64_t tsimd_hash_words_crc32(uint64_t a, uint64_t b) {
    (void)a;
    (void)b;
    return 1;
}
#endif

void tsimd_pcache_key_hash_bounded(const uint8_t *bytes, size_t len,
                                   const uint8_t *buf_end,
                                   uint64_t *k0, uint64_t *k1,
                                   uint64_t *hash) {
    if (len == 0 || len > TSIMD_PCACHE_MAX_SPAN) {
        *k0 = 0;
        *k1 = 0;
        *hash = 1;
        return;
    }
    if (bytes + 16 > buf_end) {
        tsimd_pcache_key_hash(bytes, (uint32_t)len, k0, k1, hash);
        return;
    }
    uint64_t a, b;
    memcpy(&a, bytes, 8);
    memcpy(&b, bytes + 8, 8);
    if (len < 8) {
        a &= (1ull << (8 * len)) - 1;
        b = 0;
    } else if (len < 15) {
        b &= (1ull << (8 * (len - 8))) - 1;
    } else {
        b &= 0x00FFFFFFFFFFFFFFull;

    }
    b |= (uint64_t)len << 56;
    *k0 = a;
    *k1 = b;
    *hash = span_hash_words(a, b);
}

static inline uint64_t span_hash(const uint8_t *k) {
    uint64_t a, b;
    memcpy(&a, k, 8);
    memcpy(&b, k + 8, 8);
    return span_hash_words(a, b);
}

#include <sys/mman.h>

int tsimd_pcache_bits_env(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_PCACHE_BITS");
        int bits = -1;
        if (e && e[0]) {
            int p = atoi(e);
            if (p >= 10 && p <= 24) bits = p;
        }
        atomic_store(&state, bits);
        v = atomic_load(&state);
    }
    return v;
}

int tsimd_pretok_cache_init(tsimd_pretok_cache_t *c, size_t capacity_entries) {
    memset(c, 0, sizeof(*c));
    size_t cap = 1024;
    while (cap < capacity_entries) cap <<= 1;

    size_t bytes = cap * sizeof(tsimd_pcache_entry_t);
    size_t rounded = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    if (rounded >= (2u << 20)) {
        void *map = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map != MAP_FAILED) {
            c->entries = (tsimd_pcache_entry_t *)map;
            c->map_bytes = rounded;
#if defined(MADV_HUGEPAGE)
            madvise(map, bytes, MADV_HUGEPAGE);
#endif
        }
    }
    if (!c->entries) {
        c->entries = aligned_alloc(2u << 20, rounded ? rounded : 32);
        if (!c->entries) c->entries = calloc(cap, sizeof(tsimd_pcache_entry_t));
        else {
#if defined(MADV_HUGEPAGE)
            madvise(c->entries, bytes, MADV_HUGEPAGE);
#endif
        }
        for (size_t i = 0; i < cap; i++) {
            c->entries[i].key0 = 0;
            c->entries[i].key1 = 0;
            c->entries[i].val = 0;
            c->entries[i].ext = 0;
        }
    }
    if (!c->entries) return -1;
    c->mask = cap - 1;
    c->pair_mask = c->mask & ~(size_t)1;

    c->max_count = cap - cap / 4;

    {
        const char *e = getenv("TSIMD_SHARED_CACHE");
        if (e && e[0] == '1') c->shared = 1;
    }
    return 0;
}

void tsimd_pretok_cache_free(tsimd_pretok_cache_t *c) {
    for (uint32_t i = 0; i < c->ltab_nold; i++) free(c->ltab_old[i]);
    for (uint32_t i = 0; i < c->larena_nold; i++) free(c->larena_old[i]);
    for (uint32_t i = 0; i < c->lids_nold; i++) free(c->lids_old[i]);
    if (c->map_bytes) munmap(c->entries, c->map_bytes);
    else free(c->entries);

    if (getenv("TSIMD_PDBG")) fprintf(stderr, "PDBG free c=%p larena=%p nold=%u\n", (void*)c, (void*)c->larena, c->larena_nold);
    free(c->ltab);
    free(c->larena);
    free(c->lids);
    free(c->lids_out);
    memset(c, 0, sizeof(*c));
}

static inline void pcache_key(uint8_t *k, const uint8_t *bytes, uint32_t len) {
    memset(k, 0, 16);
    memcpy(k, bytes, len);
    k[15] = (uint8_t)len;
}

void tsimd_pcache_key_hash(const uint8_t *bytes, uint32_t len,
                           uint64_t *k0, uint64_t *k1, uint64_t *hash) {
    uint8_t k[16];
    pcache_key(k, bytes, len);
    memcpy(k0, k, 8);
    memcpy(k1, k + 8, 8);
    *hash = span_hash(k);
}

uint32_t *tsimd_pcache_lookup(tsimd_pretok_cache_t *c,
                             const uint8_t *bytes, uint32_t len,
                             uint32_t *n_ids) {
    return tsimd_pcache_lookup_into(c, bytes, len, c->scratch, n_ids);
}

void tsimd_pcache_merge(tsimd_pretok_cache_t *dst, tsimd_pretok_cache_t *src) {
    for (size_t slot = 0; slot <= src->mask; slot++) {
        tsimd_pcache_entry_t *e = &src->entries[slot];
        if (e->key1 == 0) continue;
        uint32_t len = (uint32_t)(e->key1 >> 56);
        if (len == 0 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        uint8_t bytes[16];
        memcpy(bytes, &e->key0, 8);
        memcpy(bytes + 8, &e->key1, 7);
        uint32_t n = (uint32_t)(e->val & 7);
        if (n == 0 || n > TSIMD_PCACHE_MAX_IDS) continue;
        uint32_t ids[4];
        ids[0] = (uint32_t)((e->val >> 3) & 0x1FFFFFFF);
        ids[1] = (uint32_t)(e->val >> 32);
        ids[2] = (uint32_t)(e->ext & 0xFFFFFFFFu);
        ids[3] = (uint32_t)(e->ext >> 32);
        tsimd_pcache_insert(dst, bytes, len, ids, n);
    }
}

uint32_t *tsimd_pcache_lookup_keyed(tsimd_pretok_cache_t *c,
                                    uint64_t k0, uint64_t k1,
                                    uint64_t hash,
                                    uint32_t *out, uint32_t *n_ids) {
    return tsimd_pcache_probe_inline(c, k0, k1, hash, out, n_ids);
}

void tsimd_pcache_insert(tsimd_pretok_cache_t *c, const uint8_t *bytes,
                        uint32_t len, const uint32_t *ids, uint32_t n_ids) {
    if (pcache_disabled()) return;
    if (len == 0 || len > TSIMD_PCACHE_MAX_SPAN) return;
    if (n_ids == 0 || n_ids > TSIMD_PCACHE_MAX_IDS) return;
    uint8_t k[16];
    pcache_key(k, bytes, len);
    uint64_t k0, k1;
    memcpy(&k0, k, 8);
    memcpy(&k1, k + 8, 8);
    if ((k0 | k1) == 0)
        return;
    uint64_t hash = span_hash(k);
    tsimd_pcache_insert_keyed(c, k0, k1, hash, len, ids, n_ids);
}

uint32_t *tsimd_pcache_lookup_into(tsimd_pretok_cache_t *c,
                                   const uint8_t *bytes, uint32_t len,
                                   uint32_t *out, uint32_t *n_ids) {
    return tsimd_pcache_lookup_into_bounded(c, bytes, len, bytes + len,
                                            out, n_ids);
}

uint32_t *tsimd_pcache_lookup_into_bounded(tsimd_pretok_cache_t *c,
                                           const uint8_t *bytes,
                                           uint32_t len,
                                           const uint8_t *buf_end,
                                           uint32_t *out,
                                           uint32_t *n_ids) {
    if (pcache_disabled()) {
        if (tsimd_pcache_stats_on()) c->misses++;
        return NULL;
    }
    if (len == 0 || len > TSIMD_PCACHE_MAX_SPAN) {
        if (tsimd_pcache_stats_on()) c->misses++;
        return NULL;
    }
    uint64_t k0, k1, hash;
    tsimd_pcache_key_hash_bounded(bytes, len, buf_end, &k0, &k1, &hash);
    return tsimd_pcache_lookup_keyed(c, k0, k1, hash, out, n_ids);
}

static void pcache_rehash(tsimd_pretok_cache_t *c, size_t newcap) {
    tsimd_pcache_entry_t *old = c->entries;
    size_t old_cap = c->mask + 1;
    size_t bytes = newcap * sizeof(tsimd_pcache_entry_t);
    size_t rounded = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    tsimd_pcache_entry_t *ne = NULL;
    size_t nmap = 0;
    if (rounded >= (2u << 20)) {
        void *map = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map != MAP_FAILED) {
            ne = map;
            nmap = rounded;
#if defined(MADV_HUGEPAGE)
            madvise(map, bytes, MADV_HUGEPAGE);
#endif
            memset(map, 0, bytes);
        }
    }
    if (!ne) {
        ne = calloc(newcap, sizeof(tsimd_pcache_entry_t));
        if (!ne) return;
    }
    size_t npair = (newcap - 1) & ~(size_t)1;
    size_t moved = 0;
    for (size_t i = 0; i < old_cap; i++) {
        tsimd_pcache_entry_t *e = &old[i];
        if ((e->key0 | e->key1) == 0) continue;
        uint8_t k[16];
        memcpy(k, &e->key0, 8);
        memcpy(k + 8, &e->key1, 8);
        uint64_t h = span_hash(k);
        size_t set = ((size_t)h << 1) & npair;
        for (;;) {
            tsimd_pcache_entry_t *d0 = &ne[set];
            tsimd_pcache_entry_t *d1 = &ne[set + 1];
            if ((d0->key0 | d0->key1) == 0) { *d0 = *e; break; }
            if ((d1->key0 | d1->key1) == 0) { *d1 = *e; break; }
            set = (set + 2) & npair;
        }
        moved++;
    }
    if (c->map_bytes) munmap(c->entries, c->map_bytes);
    else free(c->entries);
    c->entries = ne;
    c->map_bytes = nmap;
    c->mask = newcap - 1;
    c->pair_mask = npair;
    c->max_count = newcap - newcap / 4;
    c->count = moved;
}

void tsimd_pcache_insert_keyed(tsimd_pretok_cache_t *c, uint64_t k0,
                               uint64_t k1, uint64_t hash, uint32_t len,
                               const uint32_t *ids, uint32_t n_ids) {
    if (pcache_disabled()) return;
    if (n_ids == 0 || n_ids > TSIMD_PCACHE_MAX_IDS) return;
    if (__atomic_load_n(&c->count, __ATOMIC_RELAXED) >= c->max_count) {
        if (c->shared || (c->mask + 1) >= ((size_t)1 << 24)) return;
        pcache_rehash(c, (c->mask + 1) << 1);
    }
    if ((k0 | k1) == 0)
        return;

    if (c->shared && k0 == 0)
        return;

    (void)len;
    if (ids[0] >= (1u << TSIMD_PCACHE_ID0_BITS))
        return;

    uint64_t val = (uint64_t)n_ids | ((uint64_t)ids[0] << 3) |
                   (n_ids > 1 ? (uint64_t)ids[1] << 32 : 0);
    uint64_t ext = (n_ids > 2 ? (uint64_t)ids[2] : 0) |
                   (n_ids > 3 ? (uint64_t)ids[3] << 32 : 0);
    size_t set = ((size_t)hash << 1) & c->pair_mask;
    for (;;) {
        tsimd_pcache_entry_t *e0 = &c->entries[set];
        tsimd_pcache_entry_t *e1 = &c->entries[set + 1];
        if (c->shared) {

            if (__atomic_load_n(&e0->key0, __ATOMIC_ACQUIRE) == 0) {
                if (__sync_bool_compare_and_swap(&e0->key0, 0, k0)) {
                    __atomic_store_n(&e0->val, val, __ATOMIC_RELAXED);
                    __atomic_store_n(&e0->ext, ext, __ATOMIC_RELAXED);
                    __atomic_store_n(&e0->key1, k1, __ATOMIC_RELEASE);
                    __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
                    if (tsimd_pcache_stats_on()) c->inserts++;
                    return;
                }

                if (__atomic_load_n(&e0->key1, __ATOMIC_ACQUIRE) == k1 &&
                    __atomic_load_n(&e0->key0, __ATOMIC_RELAXED) == k0)
                    return;
            }
            if (__atomic_load_n(&e1->key0, __ATOMIC_ACQUIRE) == 0) {
                if (__sync_bool_compare_and_swap(&e1->key0, 0, k0)) {
                    __atomic_store_n(&e1->val, val, __ATOMIC_RELAXED);
                    __atomic_store_n(&e1->ext, ext, __ATOMIC_RELAXED);
                    __atomic_store_n(&e1->key1, k1, __ATOMIC_RELEASE);
                    __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
                    if (tsimd_pcache_stats_on()) c->inserts++;
                    return;
                }

                if (__atomic_load_n(&e1->key1, __ATOMIC_ACQUIRE) == k1 &&
                    __atomic_load_n(&e1->key0, __ATOMIC_RELAXED) == k0)
                    return;
            }
        } else {
            if (e0->key0 == k0 && e0->key1 == k1) return;
            if (e1->key0 == k0 && e1->key1 == k1) return;

            if ((e0->key0 | e0->key1) == 0) {
                e0->key0 = k0;
                __atomic_store_n(&e0->val, val, __ATOMIC_RELAXED);
                __atomic_store_n(&e0->ext, ext, __ATOMIC_RELAXED);
                e0->key1 = k1;
                __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
                if (tsimd_pcache_stats_on()) c->inserts++;
                return;
            }
            if ((e1->key0 | e1->key1) == 0) {
                e1->key0 = k0;
                __atomic_store_n(&e1->val, val, __ATOMIC_RELAXED);
                __atomic_store_n(&e1->ext, ext, __ATOMIC_RELAXED);
                e1->key1 = k1;
                __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
                if (tsimd_pcache_stats_on()) c->inserts++;
                return;
            }
        }
        set = (set + 2) & c->pair_mask;
    }
}

void tsimd_pcache_insert_keyed_at(tsimd_pretok_cache_t *c, size_t slot,
                                  uint64_t k0, uint64_t k1, uint64_t hash,
                                  uint32_t len, const uint32_t *ids,
                                  uint32_t n_ids) {
    if (pcache_disabled()) return;
    {
        static atomic_int state = 0;
        int v = atomic_load(&state);
        if (v == 0) {
            const char *e = getenv("TSIMD_NO_INSERT_AT");
            atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
            v = atomic_load(&state);
        }
        if (v == 2) {
            tsimd_pcache_insert_keyed(c, k0, k1, hash, len, ids, n_ids);
            return;
        }
    }
    if (n_ids == 0 || n_ids > TSIMD_PCACHE_MAX_IDS) return;
    if (__atomic_load_n(&c->count, __ATOMIC_RELAXED) >= c->max_count) return;
    if ((k0 | k1) == 0) return;
    (void)len;
    if (ids[0] >= (1u << TSIMD_PCACHE_ID0_BITS)) return;
    if (c->shared) {
        tsimd_pcache_insert_keyed(c, k0, k1, hash, len, ids, n_ids);
        return;
    }
    uint64_t val = (uint64_t)n_ids | ((uint64_t)ids[0] << 3) |
                   (n_ids > 1 ? (uint64_t)ids[1] << 32 : 0);
    uint64_t ext = (n_ids > 2 ? (uint64_t)ids[2] : 0) |
                   (n_ids > 3 ? (uint64_t)ids[3] << 32 : 0);
    (void)hash;
    tsimd_pcache_entry_t *e = &c->entries[slot];
    uint64_t prev = 0;
    if (__atomic_compare_exchange_n(&e->key0, &prev, k0, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        __atomic_store_n(&e->val, val, __ATOMIC_RELAXED);
        __atomic_store_n(&e->ext, ext, __ATOMIC_RELAXED);
        __atomic_store_n(&e->key1, k1, __ATOMIC_RELEASE);
        __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
        if (tsimd_pcache_stats_on()) c->inserts++;
        return;
    }
    if (prev == k0 &&
        __atomic_load_n(&e->key1, __ATOMIC_ACQUIRE) == k1)
        return;
    tsimd_pcache_insert_keyed(c, k0, k1, hash, 0, ids, n_ids);
}

void tsimd_pcache_insert_at_keyed(tsimd_pretok_cache_t *c, size_t slot,
                                  uint64_t k0, uint64_t k1, uint64_t hash,
                                  const uint32_t *ids, uint32_t n_ids) {
    (void)hash;
    if (pcache_disabled()) return;
    if (n_ids == 0 || n_ids > TSIMD_PCACHE_MAX_IDS) return;
    if (ids[0] >= (1u << TSIMD_PCACHE_ID0_BITS)) return;

    if (__atomic_load_n(&c->count, __ATOMIC_RELAXED) >= c->max_count) return;
    uint64_t val = (uint64_t)n_ids | ((uint64_t)ids[0] << 3) |
                   ((uint64_t)ids[1] << 32);
    uint64_t ext = (n_ids > 2 ? (uint64_t)ids[2] : 0) |
                   (n_ids > 3 ? (uint64_t)ids[3] << 32 : 0);
    tsimd_pcache_entry_t *e = &c->entries[slot];
    if (c->shared) {

        (void)e;
        tsimd_pcache_insert_keyed(c, k0, k1, hash, 0, ids, n_ids);
        return;
    }

    uint64_t prev = 0;
    if (__atomic_compare_exchange_n(&e->key0, &prev, k0, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        __atomic_store_n(&e->val, val, __ATOMIC_RELAXED);
        __atomic_store_n(&e->ext, ext, __ATOMIC_RELAXED);
        __atomic_store_n(&e->key1, k1, __ATOMIC_RELEASE);
        __atomic_fetch_add(&c->count, 1, __ATOMIC_RELAXED);
        if (tsimd_pcache_stats_on()) c->inserts++;
        return;
    }
    if (prev == k0 &&
        __atomic_load_n(&e->key1, __ATOMIC_ACQUIRE) == k1)
        return;
    tsimd_pcache_insert_keyed(c, k0, k1, hash, 0, ids, n_ids);
}

int tsimd_pcache_frozen_build(tsimd_pcache_frozen_t *f,
                              const tsimd_vocab_t *v) {
    memset(f, 0, sizeof(*f));
    if (!v->bytes || !v->offsets) return -1;
    size_t n = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 1 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t z = 0;
        while (z < len && z < 8 && b[z] == 0) z++;
        if (z >= (len < 8 ? len : 8)) continue;
        n++;
    }
    size_t n_slots = 64;
    while (n_slots < n * 2) n_slots <<= 1;
    tsimd_pcache_frozen_entry_t *entries =
        malloc(n_slots * sizeof(tsimd_pcache_frozen_entry_t));
    if (!entries) return -1;
    for (size_t i = 0; i < n_slots; i++) {
        entries[i].k0 = 0;
        entries[i].k1 = 0;
        entries[i].val = 0;
        entries[i].ext = 0;
    }
    size_t mask = n_slots - 1;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 1 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t z = 0;
        while (z < len && z < 8 && b[z] == 0) z++;
        if (z >= (len < 8 ? len : 8)) continue;
        uint8_t k[16];
        memset(k, 0, 16);
        memcpy(k, b, len);
        k[15] = (uint8_t)len;
        uint64_t k0, k1, hash;
        memcpy(&k0, k, 8);
        memcpy(&k1, k + 8, 8);
        {
            uint64_t x0, x1;
            tsimd_pcache_key_hash_bounded(k, len, k + 16, &x0, &x1, &hash);
        }
        uint64_t val = (uint64_t)1 | ((uint64_t)id << 3);
        size_t idx = (size_t)hash & mask;
        while (entries[idx].k0 != 0 || entries[idx].k1 != 0)
            idx = (idx + 1) & mask;
        entries[idx].k0 = k0;
        entries[idx].k1 = k1;
        entries[idx].val = val;
        entries[idx].ext = 0;
    }
    f->entries = entries;
    f->mask = mask;
    return 0;
}

void tsimd_pcache_frozen_free(tsimd_pcache_frozen_t *f) {
    free(f->entries);
    memset(f, 0, sizeof(*f));
}

int tsimd_pcache_frozen_probe(const tsimd_pcache_frozen_t *f,
                              uint64_t k0, uint64_t k1, uint64_t hash,
                              uint64_t *val_out) {
    size_t idx = (size_t)hash & f->mask;
    const tsimd_pcache_frozen_entry_t *e = &f->entries[idx];
    for (;;) {
        if (e->k0 == 0 && e->k1 == 0) return 0;
        if (e->k0 == k0 && e->k1 == k1) {
            *val_out = e->val;
            return 1;
        }
        idx = (idx + 1) & f->mask;
        e = &f->entries[idx];
    }
}

int tsimd_pcache_seed_vocab(tsimd_pretok_cache_t *c,
                            const tsimd_vocab_t *v) {
    if (!v->bytes || !v->offsets) return -1;
    for (int pass = 0; pass < 2; pass++)
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        if (v->byte_tok) {
            int isb = (v->byte_tok[id >> 3] >> (id & 7)) & 1;
            if ((pass == 0) == isb) continue;
        } else if (pass == 1) {
            continue;
        }
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 1 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        const uint8_t *b = v->bytes + v->offsets[id];
        if (len == 1 && v->byte_ids[b[0]] != id &&
            v->byte_ids[b[0]] < v->vocab_size)
            continue;
        if (v->pair_table && !whole_piece_ok(v, id)) continue;

        uint32_t z = 0;
        while (z < len && z < 8 && b[z] == 0) z++;
        if (z == 8) continue;
        uint32_t ids[1] = {id};
        uint8_t k[16];
        memset(k, 0, 16);
        memcpy(k, b, len);
        k[15] = (uint8_t)len;
        uint64_t kk0, kk1;
        memcpy(&kk0, k, 8);
        memcpy(&kk1, k + 8, 8);
        if ((kk0 | kk1) == 0) continue;
        uint64_t hash = span_hash(k);
        tsimd_pcache_insert_keyed(c, kk0, kk1, hash, len, ids, 1);
    }
    return 0;
}

static void plong_ensure(tsimd_pretok_cache_t *c) {
    if (c->ltab) return;
    if (__sync_lock_test_and_set(&c->lensure_busy, 1)) {
        while (__atomic_load_n(&c->lensure_busy, __ATOMIC_ACQUIRE))
            sched_yield();
        return;
    }
    if (c->lmin == 0) c->lmin = 16;
    size_t bits = c->lbits ? c->lbits : 17;
    c->lmask = ((size_t)1 << bits) - 1;
    c->ltab = tsimd_alloc_huge_zero(((size_t)c->lmask + 1) *
                                    sizeof(tsimd_plong_entry_t));
    size_t shift = bits > 17 ? bits - 17 : 0;
    if (shift > 3) shift = 3;
    size_t scale = (size_t)1 << shift;
    c->lcap = (4u << 20) * scale;
    c->larena = tsimd_alloc_huge(c->lcap);
    if (getenv("TSIMD_PDBG")) fprintf(stderr, "PDBG ensure c=%p larena=%p\n", (void*)c, (void*)c->larena);
    c->lids_cap = (1u << 19) * scale;

    c->lids = tsimd_alloc_huge(c->lids_cap * sizeof(uint32_t));
    if (!c->lids_out)
        c->lids_out = malloc(((size_t)TSIMD_PLONG_MAX_IDS + 1) *
                             sizeof(uint32_t));
    if (!c->ltab || !c->larena || !c->lids || !c->lids_out) {
        free(c->ltab);
        free(c->larena);
        free(c->lids);
        free(c->lids_out);
        c->ltab = NULL;
        c->larena = NULL;
        c->lids = NULL;
        c->lids_out = NULL;
        c->lmask = 0;
        __sync_lock_release(&c->lensure_busy);
        return;
    }
    __atomic_store_n(&c->lused, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&c->lcount, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&c->lids_used, 0, __ATOMIC_RELAXED);
    __sync_lock_release(&c->lensure_busy);
}

#define TSIMD_PLONG_SPILL_BIT 0x80000000u
#define TSIMD_PLONG_PROBE_MAX 12u

static int plong_payload_grow(tsimd_pretok_cache_t *c) {
    if (c->lcap >= (2048u << 20)) return -1;
    size_t ncap = c->lcap * 2;
    uint8_t *na = tsimd_alloc_huge(ncap);
    if (!na) return -1;
    memcpy(na, c->larena, c->lcap);
    {
        int dup = 0;
        for (uint32_t q = 0; q < c->larena_nold; q++)
            if (c->larena_old[q] == c->larena) dup = 1;
        if (!dup) {
            if (c->larena_nold < 8)
                c->larena_old[c->larena_nold++] = c->larena;
            else if (!c->shared_plong)
                free(c->larena);
        }
    }
    c->larena = na;
    c->lcap = ncap;
    if (c->lids_used + (1u << 20) > c->lids_cap && c->lids_cap < (256u << 20)) {
        size_t nlc = c->lids_cap * 2;
        uint32_t *nl = tsimd_alloc_huge(nlc * sizeof(uint32_t));
        if (nl) {
            memcpy(nl, c->lids, c->lids_cap * sizeof(uint32_t));
            {
                int dup = 0;
                for (uint32_t q = 0; q < c->lids_nold; q++)
                    if (c->lids_old[q] == c->lids) dup = 1;
                if (!dup) {
                    if (c->lids_nold < 8)
                        c->lids_old[c->lids_nold++] = c->lids;
                    else if (!c->shared_plong)
                        free(c->lids);
                }
            }
            c->lids = nl;
            c->lids_cap = nlc;
        }
    }
    return 0;
}

static void plong_grow(tsimd_pretok_cache_t *c) {
    size_t obits = c->lbits;
    tsimd_plong_entry_t *old = c->ltab;
    size_t on = c->lmask + 1;
    c->lbits = (uint32_t)(obits + 1);
    c->lmask = ((size_t)1 << c->lbits) - 1;
    c->ltab = tsimd_alloc_huge_zero((c->lmask + 1) *
                                    sizeof(tsimd_plong_entry_t));
    if (!c->ltab) {
        c->ltab = old;
        c->lbits = (uint32_t)obits;
        c->lmask = on - 1;
        return;
    }
    c->lcount = 0;
    for (size_t i = 0; i < on; i++) {
        if (!old[i].hash) continue;
        size_t b = (size_t)old[i].hash & c->lmask;
        for (unsigned step = 0; step < (unsigned)TSIMD_PLONG_PROBE_MAX;
             step++) {
            if (!c->ltab[b].hash) break;
            b = (b + 1) & c->lmask;
        }
        if (c->ltab[b].hash == 0) c->lcount++;
        c->ltab[b] = old[i];
    }
    {
        int dup = 0;
        for (uint32_t q = 0; q < c->ltab_nold; q++)
            if (c->ltab_old[q] == old) dup = 1;
        if (!dup && c->ltab_nold < 8) c->ltab_old[c->ltab_nold++] = old;
        else if (!c->shared_plong) free(old);
    }
}

void tsimd_plong_reserve(tsimd_pretok_cache_t *c, size_t entries) {
    size_t bits = 17;
    while (bits < 24 && ((size_t)1 << bits) < entries) bits++;
    if (c->ltab) {
        if (bits <= c->lbits) return;
        if (c->ltab_nold < 8) c->ltab_old[c->ltab_nold++] = c->ltab;
        else free(c->ltab);
        c->ltab = NULL;
        c->lcount = 0;
        c->lused = 0;
        c->lids_used = 0;
        if (c->larena) {
            if (c->larena_nold < 8)
                c->larena_old[c->larena_nold++] = c->larena;
            else free(c->larena);
            c->larena = NULL;
        }
        if (c->lids) {
            if (c->lids_nold < 8) c->lids_old[c->lids_nold++] = c->lids;
            else free(c->lids);
            c->lids = NULL;
        }
    }
    c->lbits = (uint32_t)bits;
}

static _Atomic size_t plong_hits = 0, plong_misses = 0,
                      plong_inserts = 0;
unsigned tsimd_pcache_lbits_of(const tsimd_pretok_cache_t *c) {
    return c->lbits;
}

size_t tsimd_pcache_lcount_of(const tsimd_pretok_cache_t *c) {
    return c->lcount;
}

void tsimd_plong_stats(size_t *h, size_t *m, size_t *i) {
    *h = atomic_load_explicit(&plong_hits, memory_order_relaxed);
    *m = atomic_load_explicit(&plong_misses, memory_order_relaxed);
    *i = atomic_load_explicit(&plong_inserts, memory_order_relaxed);
}

static uint64_t plong_hash(const uint8_t *b, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int plong_disabled(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_NO_PLONG");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

const uint32_t *tsimd_plong_probe_ex(const tsimd_pretok_cache_t *c,
                                     const uint8_t *bytes, size_t len,
                                     uint32_t *n_out) {
    if (pcache_disabled() || plong_disabled()) return NULL;
    if (!c->ltab || len < c->lmin) return NULL;
    uint64_t h = plong_hash(bytes, len);
    size_t lm = c->lmask;
    const tsimd_plong_entry_t *lt = c->ltab;
    size_t b = (size_t)h & lm;
    for (unsigned step = 0; step <= (unsigned)TSIMD_PLONG_PROBE_MAX; step++) {
        uint32_t s0 = __atomic_load_n(&lt[b].seq, __ATOMIC_ACQUIRE);
        if (s0 == 0) break;
        if ((s0 & 1u) == 0) {
            uint64_t eh = __atomic_load_n(&lt[b].hash, __ATOMIC_RELAXED);
            uint32_t el = __atomic_load_n(&lt[b].lensz, __ATOMIC_RELAXED);
            if (eh == h && (size_t)(el & 0x1FFFFFFFu) == len) {
                uint32_t eoff = __atomic_load_n(&lt[b].off, __ATOMIC_RELAXED);
                uint32_t e0 = __atomic_load_n(&lt[b].ids[0], __ATOMIC_RELAXED);
                uint32_t e1 = __atomic_load_n(&lt[b].ids[1], __ATOMIC_RELAXED);
                uint32_t e2 = __atomic_load_n(&lt[b].ids[2], __ATOMIC_RELAXED);
                uint32_t e3 = __atomic_load_n(&lt[b].ids[3], __ATOMIC_RELAXED);
                if (__atomic_load_n(&lt[b].seq, __ATOMIC_ACQUIRE) != s0)
                    continue;
                if (memcmp(c->larena + eoff, bytes, len) == 0) {
                    static _Thread_local uint32_t tl_out[TSIMD_PLONG_MAX_IDS];
                    if (el & TSIMD_PLONG_SPILL_BIT) {
                        memcpy(tl_out, c->lids + e0,
                               (size_t)e1 * sizeof(uint32_t));
                        *n_out = e1;
                        atomic_fetch_add_explicit(&plong_hits, 1,
                                              memory_order_relaxed);
                        return tl_out;
                    }
                    uint32_t *dst = tl_out;
                    dst[0] = e0;
                    dst[1] = e1;
                    dst[2] = e2;
                    dst[3] = e3;
                    *n_out = ((el >> 29) & 3) + 1;
                    atomic_fetch_add_explicit(&plong_hits, 1,
                                              memory_order_relaxed);
                    return dst;
                }
            }
        }
        b = (b + 1) & lm;
    }
    atomic_fetch_add_explicit(&plong_misses, 1, memory_order_relaxed);
    return NULL;
}

int tsimd_plong_probe(const tsimd_pretok_cache_t *c,
                      const uint8_t *bytes, size_t len,
                      uint32_t *ids, uint32_t *n) {

    uint32_t cnt = 0;
    const uint32_t *r = tsimd_plong_probe_ex(c, bytes, len, &cnt);
    if (!r || cnt > 4) return 0;
    for (uint32_t i = 0; i < cnt; i++) ids[i] = r[i];
    *n = cnt;
    return 1;
}

void tsimd_plong_insert(tsimd_pretok_cache_t *c,
                        const uint8_t *bytes, size_t len,
                        const uint32_t *ids, uint32_t n) {
    if (pcache_disabled() || plong_disabled()) return;
    if (n == 0 || n > TSIMD_PLONG_MAX_IDS) return;
    int spill = n > 4;
    if (spill && c->shared_plong) return;
    if (c->shared_plong &&
        __atomic_load_n(&c->lcount, __ATOMIC_RELAXED) > (c->lmask >> 1))
        return;
    if (len < __atomic_load_n(&c->lmin, __ATOMIC_RELAXED) ||
        len >= (1u << 29)) return;
    if (!__atomic_load_n(&c->ltab, __ATOMIC_RELAXED) ||
        !__atomic_load_n(&c->lids, __ATOMIC_RELAXED) ||
        !__atomic_load_n(&c->lids_out, __ATOMIC_RELAXED))
        plong_ensure(c);
    if (!__atomic_load_n(&c->ltab, __ATOMIC_RELAXED) ||
        !__atomic_load_n(&c->lids, __ATOMIC_RELAXED) ||
        !__atomic_load_n(&c->lids_out, __ATOMIC_RELAXED))
        return;
    if (__atomic_load_n(&c->lused, __ATOMIC_RELAXED) + len > c->lcap ||
        (spill && __atomic_load_n(&c->lids_used, __ATOMIC_RELAXED) + n >
                       c->lids_cap)) {
        if (plong_payload_grow(c) != 0 && !c->shared_plong) {
            memset(c->ltab, 0,
                   ((size_t)c->lmask + 1) * sizeof(tsimd_plong_entry_t));
            c->lcount = 0;
            __atomic_store_n(&c->lused, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&c->lids_used, 0, __ATOMIC_RELAXED);
        }
    }
    uint64_t h = plong_hash(bytes, len);
    size_t lm = c->lmask;
    tsimd_plong_entry_t *lt = c->ltab;
    size_t b = (size_t)h & lm;
    int claimed = 0;
    for (unsigned step = 0; step < (unsigned)TSIMD_PLONG_PROBE_MAX; step++) {
        if (__atomic_load_n(&lt[b].seq, __ATOMIC_ACQUIRE) == 0) {
            if (__sync_bool_compare_and_swap(&lt[b].seq, 0, 1)) {
                claimed = 1;
                break;
            }
            continue;
        }
        b = (b + 1) & lm;
    }
    if (!claimed) {
        b = (b + (size_t)TSIMD_PLONG_PROBE_MAX) & lm;
        for (unsigned step = 0; step < (unsigned)TSIMD_PLONG_PROBE_MAX;
             step++) {
            uint32_t sv = __atomic_load_n(&lt[b].seq, __ATOMIC_ACQUIRE);
            if ((sv & 1u) == 0 &&
                __sync_bool_compare_and_swap(&lt[b].seq, sv, sv + 1)) {
                claimed = 1;
                break;
            }
            b = (b + 1) & lm;
        }
        if (!claimed) return;
        if (__atomic_load_n(&lt[b].hash, __ATOMIC_RELAXED) == 0)
            __atomic_fetch_add(&c->lcount, 1, __ATOMIC_RELAXED);
    } else if (__atomic_load_n(&lt[b].hash, __ATOMIC_RELAXED) == 0) {
        __atomic_fetch_add(&c->lcount, 1, __ATOMIC_RELAXED);
    }
    size_t loff = 0;
    if (spill) {
        loff = __atomic_fetch_add(&c->lids_used, (size_t)n,
                                  __ATOMIC_RELAXED);
        if (loff + n > c->lids_cap) return;
    }
    size_t aoff = __atomic_fetch_add(&c->lused, (size_t)len,
                                     __ATOMIC_RELAXED);
    if (aoff + len > c->lcap) return;
    if (spill) {
        __atomic_store_n(&lt[b].lensz,
                         (uint32_t)(len & 0x1FFFFFFFu) | (3u << 29) |
                             TSIMD_PLONG_SPILL_BIT,
                         __ATOMIC_RELAXED);
        __atomic_store_n(&lt[b].ids[0], (uint32_t)loff, __ATOMIC_RELAXED);
        __atomic_store_n(&lt[b].ids[1], n, __ATOMIC_RELAXED);
        memcpy(c->lids + loff, ids, (size_t)n * sizeof(uint32_t));
    } else {
        __atomic_store_n(&lt[b].lensz,
                         (uint32_t)(len & 0x1FFFFFFFu) | ((n - 1) << 29),
                         __ATOMIC_RELAXED);
        for (uint32_t i = 0; i < n; i++)
            __atomic_store_n(&lt[b].ids[i], ids[i], __ATOMIC_RELAXED);
    }
    __atomic_store_n(&lt[b].off, (uint32_t)aoff, __ATOMIC_RELAXED);
    memcpy(c->larena + aoff, bytes, len);
    __atomic_store_n(&lt[b].hash, h, __ATOMIC_RELAXED);
    __atomic_store_n(&lt[b].seq, __atomic_load_n(&lt[b].seq,
                                                 __ATOMIC_RELAXED) + 1,
                     __ATOMIC_RELEASE);
    if (__atomic_load_n(&c->lcount, __ATOMIC_RELAXED) > (c->lmask >> 1) &&
        c->lbits < 24 && !c->shared_plong) {
        plong_grow(c);
        return;
    }
    atomic_fetch_add_explicit(&plong_inserts, 1,
                                          memory_order_relaxed);
}
