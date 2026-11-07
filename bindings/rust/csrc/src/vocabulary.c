#include "vocabulary.h"
#include "unicode.h"
#include "trie.h"
#include "allocator.h"

#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static int8_t b64_value[256];
static int b64_table_ready;

static void b64_init(void) {
    if (b64_table_ready) return;
    memset(b64_value, -1, sizeof(b64_value));
    const char *alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++)
        b64_value[(uint8_t)alpha[i]] = (int8_t)i;
    b64_table_ready = 1;
}

static long b64_decode(const char *s, size_t n, uint8_t *out) {
    b64_init();
    size_t o = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t c = (uint8_t)s[i];
        if (c == '=') break;
        int8_t v = b64_value[c];
        if (v < 0) return -1;
        acc = (acc << 6) | (uint32_t)(uint8_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = (uint8_t)((acc >> bits) & 0xFFu);
        }
    }
    return (long)o;
}

typedef struct {
    uint64_t hash;
    uint32_t id;
    uint32_t pad;
} bhash_entry_t;

typedef struct {
    bhash_entry_t *slots;
    size_t mask;
} bhash_t;

static uint64_t fnv1a(const uint8_t *p, size_t n) {

    uint64_t h = 1469598103934665603ULL;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t chunk;
        memcpy(&chunk, p + i, 8);
        h = (h ^ chunk) * 1099511628211ULL;
    }
    for (; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h ? h : 1;
}

static int bhash_build(bhash_t *t, const tsimd_vocab_t *v, int heavy) {
    size_t cap = 16;
    while (cap < (size_t)v->vocab_size * (heavy ? 4 : 2))
        cap <<= 1;
    t->slots = tsimd_alloc_huge_zero(cap * sizeof(bhash_entry_t));
    if (!t->slots) return -1;
    t->mask = cap - 1;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len == 0) continue;
        uint64_t h = fnv1a(b, len);
        size_t i = h & t->mask;
        while (t->slots[i].hash != 0) i = (i + 1) & t->mask;
        t->slots[i].hash = h;
        t->slots[i].id = id;
    }
    return 0;
}

static uint32_t bhash_find(const bhash_t *t, const tsimd_vocab_t *v,
                           const uint8_t *bytes, uint32_t len) {
    uint64_t h = fnv1a(bytes, len);

    if (v->byte_fp) {
        size_t fi = (size_t)((h >> 32) & v->fp_mask);
        uint8_t want = (uint8_t)(h >> 56);
        if (!want) want = 1;
        int cand = 0;
        while (v->byte_fp[fi] != 0) {
            if (v->byte_fp[fi] == want) {
                cand = 1;
                break;
            }
            fi = (fi + 1) & v->fp_mask;
        }
        if (!cand) return UINT32_MAX;
    }
    size_t i = h & t->mask;
    uint32_t fb = UINT32_MAX;
    while (t->slots[i].hash != 0) {
        if (t->slots[i].hash == h) {
            uint32_t id = t->slots[i].id;
            if (v->offsets[id + 1] - v->offsets[id] == len &&
                memcmp(v->bytes + v->offsets[id], bytes, len) == 0) {
                if (v->byte_tok &&
                    (v->byte_tok[id >> 3] >> (id & 7)) & 1) {
                    if (fb == UINT32_MAX) fb = id;
                } else {
                    return id;
                }
            }
        }
        i = (i + 1) & t->mask;
    }
    return fb;
}

#define PAIR_OCCUPIED (1ULL << 63)
#define PAIR_FIELDS   21
#define PAIR_MAX_ID   ((1u << PAIR_FIELDS) - 1u)

static uint64_t pair_pack(uint32_t left, uint32_t right, uint32_t merged) {
    return PAIR_OCCUPIED | ((uint64_t)left << (2 * PAIR_FIELDS)) |
           ((uint64_t)right << PAIR_FIELDS) | merged;
}

static size_t pair_index(const tsimd_vocab_t *v, uint64_t key) {
    unsigned bits =
        64 - (unsigned)__builtin_clzll((unsigned long long)(v->pair_mask + 1));
    return (size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - bits)) & v->pair_mask;
}

static uint64_t pair_key(uint32_t left, uint32_t right) {
    return ((uint64_t)left << (2 * PAIR_FIELDS)) |
           ((uint64_t)right << PAIR_FIELDS);
}

uint32_t tsimd_vocab_merge_rank(const tsimd_vocab_t *v, uint32_t left,
                               uint32_t right) {
    return tsimd_vocab_merge_rank2(v, left, right);
}

uint32_t tsimd_vocab_merge_priority(const tsimd_vocab_t *v, uint32_t left,
                                    uint32_t right) {
    if (v->mgr_dense && ((left | right) >> v->mgr_dense_log2) == 0)
        return v->mgr_dense[((size_t)left << v->mgr_dense_log2) | right];
    if (v->mgr_slots) {
        uint64_t key = ((uint64_t)left << PAIR_FIELDS) | right;
        size_t idx =
            (size_t)((key * 0x9E3779B97F4A7C15ULL) >> v->mgr_shift);
        for (;;) {
            uint64_t s = v->mgr_slots[idx];
            if ((s >> PAIR_FIELDS) == key)
                return (uint32_t)(s & PAIR_MAX_ID);
            if (s == UINT64_MAX) return UINT32_MAX;
            idx = (idx + 1) & v->mgr_mask;
        }
    }

    {
        uint64_t key = pair_key(left, right);
        size_t i = pair_index(v, key);
        while (v->pair_table[i] != 0) {
            if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
                (PAIR_OCCUPIED | key))
                return tsimd_vocab_priority(
                    v, (uint32_t)(v->pair_table[i] & PAIR_MAX_ID));
            i = (i + 1) & v->pair_mask;
        }
    }
    return UINT32_MAX;
}

uint32_t tsimd_vocab_merge_id(const tsimd_vocab_t *v, uint32_t left,
                              uint32_t right) {
    if (!v->pair_table) return UINT32_MAX;
    uint64_t key = pair_key(left, right);
    size_t i = pair_index(v, key);
    while (v->pair_table[i] != 0) {
        if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
            (PAIR_OCCUPIED | key))
            return (uint32_t)(v->pair_table[i] & PAIR_MAX_ID);
        i = (i + 1) & v->pair_mask;
    }
    return UINT32_MAX;
}

uint32_t tsimd_vocab_id_of_priority(const tsimd_vocab_t *v, uint32_t prio) {
    return v->rank_to_id ? v->rank_to_id[prio] : prio;
}

uint32_t tsimd_vocab_merge_prio_id(const tsimd_vocab_t *v, uint32_t left,
                                   uint32_t right, uint32_t *id_out) {
    if (v->mgr_dense && ((left | right) >> v->mgr_dense_log2) == 0) {
        size_t i = ((size_t)left << v->mgr_dense_log2) | right;
        uint32_t p = v->mgr_dense[i];
        if (p == UINT32_MAX) return UINT32_MAX;
        *id_out = v->mgr_dense_ids ? v->mgr_dense_ids[i] : p;
        return p;
    }
    if (v->mgr_bgrid && left < v->vocab_size && right < v->vocab_size) {
        uint8_t li = v->mgr_id_to_b[left];
        uint8_t ri = v->mgr_id_to_b[right];
        if (li != 0xFF && ri != 0xFF) {
            size_t gi = ((size_t)li << 8) | ri;
            uint32_t p = v->mgr_bgrid[gi];
            if (p == UINT32_MAX) return UINT32_MAX;
            *id_out = v->mgr_bgrid_ids[gi];
            return p;
        }
    }
    if (v->mgr_slots) {
        uint64_t key = ((uint64_t)left << PAIR_FIELDS) | right;
        size_t idx =
            (size_t)((key * 0x9E3779B97F4A7C15ULL) >> v->mgr_shift);
        for (;;) {
            uint64_t s = v->mgr_slots[idx];
            if ((s >> PAIR_FIELDS) == key) {
                uint32_t p = (uint32_t)(s & PAIR_MAX_ID);
                *id_out = v->mgr_ids ? v->mgr_ids[idx] : p;
                return p;
            }
            if (s == UINT64_MAX) return UINT32_MAX;
            idx = (idx + 1) & v->mgr_mask;
        }
    }

    {
        uint64_t key = pair_key(left, right);
        size_t i = pair_index(v, key);
        while (v->pair_table[i] != 0) {
            if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
                (PAIR_OCCUPIED | key)) {
                uint32_t m = (uint32_t)(v->pair_table[i] & PAIR_MAX_ID);
                *id_out = m;
                return tsimd_vocab_priority(v, m);
            }
            i = (i + 1) & v->pair_mask;
        }
    }
    return UINT32_MAX;
}

uint32_t tsimd_vocab_merge_rank2(const tsimd_vocab_t *v, uint32_t left,
                                 uint32_t right) {
    if (!v->mgr_dense && !v->mgr_slots) {

        uint64_t key = pair_key(left, right);
        size_t i = pair_index(v, key);
        while (v->pair_table[i] != 0) {
            if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
                (PAIR_OCCUPIED | key))
                return (uint32_t)(v->pair_table[i] & PAIR_MAX_ID);
            i = (i + 1) & v->pair_mask;
        }
        return UINT32_MAX;
    }
    uint32_t prio = tsimd_vocab_merge_priority(v, left, right);
    return prio == UINT32_MAX ? UINT32_MAX
                              : tsimd_vocab_id_of_priority(v, prio);
}

int tsimd_vocab_pair_rank_build(tsimd_vocab_t *v) {
    return tsimd_vocab_pair_rank_build_ex(v, 1);
}

int tsimd_vocab_pair_rank_build_slots(tsimd_vocab_t *v) {
    return tsimd_vocab_pair_rank_build_ex(v, 0);
}

int tsimd_vocab_pair_rank_build_ex(tsimd_vocab_t *v, int want_dense) {
    if (v->mgr_dense || v->mgr_slots) return 0;
    if (!v->pair_table) return -1;
    uint32_t DLOG = 11;
    const char *dl = getenv("TSIMD_MGR_DLOG");
    if (dl && dl[0]) {
        long k = strtol(dl, NULL, 10);
        if (k >= 6 && k <= 12) DLOG = (uint32_t)k;
    }
    const char *nd = getenv("TSIMD_NO_MGR_DENSE");
    const char *fd = getenv("TSIMD_MGR_DENSE");
    if (nd && nd[0] == '1') want_dense = 0;
    if (fd && fd[0] == '1') want_dense = 1;
    size_t grid = (size_t)1 << (2 * DLOG);
    uint32_t *dense = NULL;
    if (want_dense) {
        dense = tsimd_alloc_huge(grid * sizeof(uint32_t));
        if (!dense) return -1;
        for (size_t i = 0; i < grid; i++) dense[i] = UINT32_MAX;
    }
    size_t n_merges = 0;
    for (size_t i = 0; i <= v->pair_mask; i++)
        if (v->pair_table[i] & PAIR_OCCUPIED) n_merges++;
    size_t n_slots = 64;

    while (n_slots < n_merges * 5 / 4) n_slots <<= 1;
    uint64_t *slots = tsimd_alloc_huge(n_slots * sizeof(uint64_t));
    if (!slots) {
        free(dense);
        return -1;
    }
    for (size_t i = 0; i < n_slots; i++) slots[i] = UINT64_MAX;

    uint32_t *rank_to_id = NULL;

    uint32_t *mgr_ids = NULL, *mgr_dense_ids = NULL;
    if (v->ranks) {

        rank_to_id = tsimd_alloc_huge(n_merges * sizeof(uint32_t));
        mgr_ids = tsimd_alloc_huge(n_slots * sizeof(uint32_t));
        if (want_dense)
            mgr_dense_ids = tsimd_alloc_huge(grid * sizeof(uint32_t));
        if (!rank_to_id || !mgr_ids || (want_dense && !mgr_dense_ids)) {
            free(dense);
            free(slots);
            free(rank_to_id);
            free(mgr_ids);
            free(mgr_dense_ids);
            return -1;
        }
        for (size_t i = 0; i < n_merges; i++)
            rank_to_id[i] = UINT32_MAX;
        for (size_t i = 0; i < grid; i++)
            if (mgr_dense_ids) mgr_dense_ids[i] = UINT32_MAX;
    }
    unsigned shift =
        64 - (unsigned)__builtin_ctzll((unsigned long long)n_slots);
    size_t mask = n_slots - 1;
    for (size_t i = 0; i <= v->pair_mask; i++) {
        uint64_t e = v->pair_table[i];
        if (!(e & PAIR_OCCUPIED)) continue;
        uint32_t merged = (uint32_t)(e & PAIR_MAX_ID);
        uint32_t left = (uint32_t)((e >> (2 * PAIR_FIELDS)) & PAIR_MAX_ID);
        uint32_t right = (uint32_t)((e >> PAIR_FIELDS) & PAIR_MAX_ID);
        uint32_t prio = tsimd_vocab_priority(v, merged);
        if (rank_to_id && prio < n_merges)
            rank_to_id[prio] = merged;
        if (dense && ((left | right) >> DLOG) == 0) {
            size_t di = ((size_t)left << DLOG) | right;
            dense[di] = prio;
            if (mgr_dense_ids) mgr_dense_ids[di] = merged;
        }
        uint64_t key = ((uint64_t)left << PAIR_FIELDS) | right;
        size_t idx = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> shift);
        while (slots[idx] != UINT64_MAX) idx = (idx + 1) & mask;
        slots[idx] = (key << PAIR_FIELDS) | prio;
        if (mgr_ids) mgr_ids[idx] = merged;
    }
    uint8_t *btok = tsimd_alloc_huge_zero((size_t)v->vocab_size / 8 + 1);
    if (btok) {
        for (int b = 0; b < 256; b++) {
            uint32_t bid = v->byte_ids_hex[b];
            if (bid != 0xFFFFFFFFu && bid < v->vocab_size)
                btok[bid >> 3] |= (uint8_t)(1u << (bid & 7));
        }
        v->byte_tok = btok;
    }
    uint8_t *reach = tsimd_alloc_huge_zero(v->vocab_size);
    if (reach) {
        for (int b = 0; b < 256; b++) {
            uint32_t bid = v->byte_ids[b];
            if (bid < v->vocab_size) reach[bid] = 1;
        }
        for (uint32_t id = 0; id < v->vocab_size; id++) {
            uint32_t plen =
                v->offsets[id + 1] - v->offsets[id];
            if (plen == 0 || plen > 4) continue;
            const uint8_t *pb = v->bytes + v->offsets[id];
            uint32_t pcp;
            tsimd_char_class_t pcls;
            size_t pa =
                tsimd_utf8_next(pb, plen, &pcp, &pcls);
            if (pa == (size_t)plen) reach[id] = 1;
        }
        int changed = 1;
        while (changed && v->pair_table) {
            changed = 0;
            for (size_t i = 0; i <= v->pair_mask; i++) {
                uint64_t e = v->pair_table[i];
                if (!(e & PAIR_OCCUPIED)) continue;
                uint32_t merged = (uint32_t)(e & PAIR_MAX_ID);
                uint32_t left = (uint32_t)((e >> (2 * PAIR_FIELDS)) &
                                           PAIR_MAX_ID);
                uint32_t right = (uint32_t)((e >> PAIR_FIELDS) &
                                            PAIR_MAX_ID);
                if (!reach[merged] && reach[left] && reach[right]) {
                    reach[merged] = 1;
                    changed = 1;
                }
            }
        }
        v->merge_reach = reach;
    }
    uint8_t *id_to_b = malloc(v->vocab_size);
    uint32_t *bgrid = NULL, *bgrid_ids = NULL;
    if (id_to_b) {
        memset(id_to_b, 0xFF, v->vocab_size);
        int nbyte = 0;
        for (int b = 0; b < 256; b++)
            if (v->byte_ids[b] != UINT32_MAX &&
                v->byte_ids[b] < v->vocab_size) {
                id_to_b[v->byte_ids[b]] = (uint8_t)b;
                nbyte++;
            }
        if (nbyte == 256) {
            bgrid = malloc(65536 * sizeof(uint32_t));
            bgrid_ids = malloc(65536 * sizeof(uint32_t));
            if (bgrid && bgrid_ids) {
                for (size_t i = 0; i < 65536; i++) {
                    bgrid[i] = UINT32_MAX;
                    bgrid_ids[i] = UINT32_MAX;
                }
                for (size_t i = 0; i <= v->pair_mask; i++) {
                    uint64_t e = v->pair_table[i];
                    if (!(e & PAIR_OCCUPIED)) continue;
                    uint32_t merged = (uint32_t)(e & PAIR_MAX_ID);
                    uint32_t left =
                        (uint32_t)((e >> (2 * PAIR_FIELDS)) & PAIR_MAX_ID);
                    uint32_t right =
                        (uint32_t)((e >> PAIR_FIELDS) & PAIR_MAX_ID);
                    uint8_t li = id_to_b[left];
                    uint8_t ri = id_to_b[right];
                    if (li == 0xFF || ri == 0xFF) continue;
                    size_t gi = ((size_t)li << 8) | ri;
                    bgrid[gi] = tsimd_vocab_priority(v, merged);
                    bgrid_ids[gi] = merged;
                }
            } else {
                free(bgrid);
                free(bgrid_ids);
                bgrid = NULL;
                bgrid_ids = NULL;
            }
        }
    }
    v->mgr_dense = dense;
    v->mgr_dense_log2 = DLOG;
    v->mgr_slots = slots;
    v->mgr_mask = mask;
    v->mgr_shift = shift;
    v->rank_to_id = rank_to_id;
    v->mgr_ids = mgr_ids;
    v->mgr_dense_ids = mgr_dense_ids;
    v->mgr_bgrid = bgrid;
    v->mgr_bgrid_ids = bgrid_ids;
    v->mgr_id_to_b = id_to_b;
    size_t dense_bytes =
        dense ? grid * (sizeof(uint32_t) * (mgr_dense_ids ? 2 : 1)) : 0;
    v->mgr_prefetch_ok =
        (dense_bytes >= ((size_t)2 << 20) ||
         n_slots * sizeof(uint64_t) >= ((size_t)2 << 20))
            ? 1
            : 0;
    return 0;
}

int tsimd_vocab_sim_build(tsimd_vocab_t *v) {
    if (!v->pair_table || v->merge_sim) return 0;
    uint32_t max_len = 1;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t l = v->offsets[id + 1] - v->offsets[id];
        if (l > max_len) max_len = l;
    }
    uint8_t *sim = tsimd_alloc_huge_zero(v->vocab_size);
    uint32_t *parts = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *mids = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *ranks = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *nxt = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *prv = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    if (!sim || !parts || !mids || !ranks || !nxt || !prv) {
        free(sim);
        free(parts);
        free(mids);
        free(ranks);
        free(nxt);
        free(prv);
        return -1;
    }
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len <= 1) {
            sim[id] = 1;
            continue;
        }
        const uint8_t *b = v->bytes + v->offsets[id];
        uint32_t np = 0;
        int dead = 0;
        for (uint32_t k = 0; k < len;) {
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t a = tsimd_utf8_next(b + k, len - k, &cp, &cls);
            if (a == 0) a = 1;
            if (a == 1 || !v->raw_piece_units) {
                uint32_t pid = v->byte_ids[b[k]];
                if (pid == UINT32_MAX || pid >= v->vocab_size) {
                    dead = 1;
                    break;
                }
                parts[np++] = pid;
                k += 1;
                continue;
            }
            uint32_t pid = bhash_find((const bhash_t *)v->byte_hash, v,
                                      b + k, (uint32_t)a);
            if (pid != UINT32_MAX && pid < v->vocab_size) {
                parts[np++] = pid;
                k += (uint32_t)a;
                continue;
            }
            for (uint32_t j = 0; j < (uint32_t)a; j++) {
                uint32_t bid = v->byte_ids[b[k + j]];
                if (bid == UINT32_MAX || bid >= v->vocab_size) {
                    dead = 1;
                    break;
                }
                parts[np++] = bid;
            }
            if (dead) break;
            k += (uint32_t)a;
        }
        if (dead || np < 2) {
            if (!dead && np == 1 && parts[0] == id) sim[id] = 1;
            continue;
        }
        for (uint32_t k = 0; k + 1 < np; k++) {
            uint32_t m = 0;
            ranks[k] = tsimd_vocab_merge_prio_id(v, parts[k], parts[k + 1],
                                                 &m);
            mids[k] = m;
            nxt[k] = k + 1;
            prv[k + 1] = k;
        }
        nxt[np - 1] = UINT32_MAX;
        prv[0] = UINT32_MAX;
        uint32_t count = np;
        while (count > 1) {
            uint32_t best_r = UINT32_MAX;
            uint32_t best_pos = UINT32_MAX;
            for (uint32_t k = 0; k + 1 < np; k++) {
                if (ranks[k] < best_r) {
                    best_r = ranks[k];
                    best_pos = k;
                }
            }
            if (best_r == UINT32_MAX) break;
            uint32_t i = best_pos;
            uint32_t d = nxt[i];
            parts[i] = mids[i];
            uint32_t nr = nxt[d];
            nxt[i] = nr;
            if (nr != UINT32_MAX) prv[nr] = i;
            ranks[d] = UINT32_MAX;
            if (nr == UINT32_MAX) ranks[i] = UINT32_MAX;
            count--;
            uint32_t lf = prv[i];
            if (lf != UINT32_MAX) {
                uint32_t m = 0;
                ranks[lf] = tsimd_vocab_merge_prio_id(v, parts[lf], parts[i],
                                                      &m);
                mids[lf] = m;
            }
            if (nr != UINT32_MAX) {
                uint32_t m = 0;
                ranks[i] = tsimd_vocab_merge_prio_id(v, parts[i], parts[nr],
                                                     &m);
                mids[i] = m;
            }
        }
        uint32_t cur = 0;
        while (nxt[cur] != UINT32_MAX) cur = nxt[cur];
        if (count == 1 && parts[cur] == id) sim[id] = 1;
    }
    free(parts);
    free(mids);
    free(ranks);
    free(nxt);
    free(prv);
    v->merge_sim = sim;
    return 0;
}

uint32_t tsimd_vocab_find(const tsimd_vocab_t *v, const uint8_t *bytes,
                         uint32_t len) {
    if (!v->byte_hash) return UINT32_MAX;
    if (len == 1) {
        uint32_t bid = v->byte_ids[bytes[0]];
        if (bid != 0xFFFFFFFFu && bid < v->vocab_size &&
            v->offsets[bid + 1] - v->offsets[bid] == 1 &&
            v->bytes[v->offsets[bid]] == bytes[0])
            return bid;
    }
    return bhash_find((const bhash_t *)v->byte_hash, v, bytes, len);
}

static int reconstruct_merges(tsimd_vocab_t *v, const bhash_t *bh) {

    uint32_t max_len = v->max_token_len > 0 ? v->max_token_len : 1;
    uint32_t *parts = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *ranks = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *nxt = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint32_t *prv = tsimd_xmalloc((size_t)max_len * sizeof(uint32_t));
    uint8_t *concat = tsimd_xmalloc((size_t)max_len * 2);

    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 2) continue;
        const uint8_t *b = v->bytes + v->offsets[id];

        uint32_t np = 0;
        for (uint32_t k = 0; k < len;) {
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t a = tsimd_utf8_next(b + k, len - k, &cp, &cls);
            if (a == 0) a = 1;
            uint32_t pid = UINT32_MAX;
            if (a == 1 || !v->raw_piece_units) {
                pid = v->byte_ids[b[k]];
                parts[np++] = pid;
                k += 1;
            } else {
                for (uint32_t l = a; l >= 2; l--)
                    if (k + l <= len) {
                        pid = bhash_find(bh, v, b + k, l);
                        if (pid != UINT32_MAX) break;
                    }
                if (pid == UINT32_MAX) break;
                parts[np++] = pid;
                k += (uint32_t)a;
            }
        }
        if (np < 2) continue;
        for (uint32_t k = 0; k + 1 < np; k++) {
            uint32_t llen =
                v->offsets[parts[k] + 1] - v->offsets[parts[k]];
            uint32_t rlen = v->offsets[parts[k + 1] + 1] -
                            v->offsets[parts[k + 1]];
            if ((uint64_t)llen + rlen > (uint64_t)max_len * 2) {
                ranks[k] = UINT32_MAX;
                continue;
            }
            memcpy(concat, v->bytes + v->offsets[parts[k]], llen);
            memcpy(concat + llen, v->bytes + v->offsets[parts[k + 1]],
                   rlen);
            ranks[k] = bhash_find(bh, v, concat, llen + rlen);
            nxt[k] = k + 1;
            prv[k + 1] = k;
        }
        nxt[np - 1] = UINT32_MAX;
        prv[0] = UINT32_MAX;
        uint32_t count = np;

        while (count > 1) {
            uint32_t best_rank = UINT32_MAX;
            uint32_t best_pos = UINT32_MAX;
            for (uint32_t k = 0; k + 1 < np; k++) {
                if (ranks[k] < best_rank) {
                    best_rank = ranks[k];
                    best_pos = k;
                }
            }
            if (best_rank == UINT32_MAX) break;

            uint32_t i = best_pos;
            uint32_t dead = nxt[i];
            tsimd_vocab_record_merge(v, parts[i], parts[dead], best_rank);
            parts[i] = best_rank;
            uint32_t new_right = nxt[dead];
            nxt[i] = new_right;
            if (new_right != UINT32_MAX) prv[new_right] = i;
            ranks[dead] = UINT32_MAX;
            if (new_right == UINT32_MAX) ranks[i] = UINT32_MAX;
            count--;
            uint32_t left = prv[i];
            if (left != UINT32_MAX) {
                uint32_t llen =
                    v->offsets[parts[left] + 1] - v->offsets[parts[left]];
                uint32_t rlen = v->offsets[parts[i] + 1] -
                                v->offsets[parts[i]];
                if ((uint64_t)llen + rlen > (uint64_t)max_len * 2) {
                    ranks[left] = UINT32_MAX;
                } else {
                    memcpy(concat, v->bytes + v->offsets[parts[left]],
                           llen);
                    memcpy(concat + llen,
                           v->bytes + v->offsets[parts[i]], rlen);
                    ranks[left] = bhash_find(bh, v, concat, llen + rlen);
                }
            }
            if (new_right != UINT32_MAX) {
                uint32_t llen =
                    v->offsets[parts[i] + 1] - v->offsets[parts[i]];
                uint32_t rlen = v->offsets[parts[new_right] + 1] -
                                v->offsets[parts[new_right]];
                if ((uint64_t)llen + rlen > (uint64_t)max_len * 2) {
                    ranks[i] = UINT32_MAX;
                } else {
                    memcpy(concat, v->bytes + v->offsets[parts[i]], llen);
                    memcpy(concat + llen,
                           v->bytes + v->offsets[parts[new_right]], rlen);
                    ranks[i] = bhash_find(bh, v, concat, llen + rlen);
                }
            }
        }

        uint32_t cur = 0;
        while (nxt[cur] != UINT32_MAX) cur = nxt[cur];
        if (count != 1 || parts[cur] != id) {

            free(parts);
            free(ranks);
            free(nxt);
            free(prv);
            free(concat);
            return -1;
        }
    }
    free(parts);
    free(ranks);
    free(nxt);
    free(prv);
    free(concat);
    return 0;
}

typedef struct {
    uint8_t *bytes;
    size_t size, cap;
    uint32_t *start;
    uint32_t *len;
    uint32_t cap_ids;
    uint32_t max_id_plus_1;
    int heavy_bhash;
} vbuild_t;

void tsimd_vocab_mark_bhash_heavy(tsimd_vocab_t *v) {
    if (v->builder)
        ((vbuild_t *)v->builder)->heavy_bhash = 1;
}

int tsimd_vocab_build_start(tsimd_vocab_t *v) {
    memset(v, 0, sizeof(*v));
    v->builder = calloc(1, sizeof(vbuild_t));
    return v->builder ? 0 : -1;
}

uint32_t tsimd_vocab_add_token(tsimd_vocab_t *v, const uint8_t *bytes,
                              uint32_t len, uint32_t id) {
    vbuild_t *b = v->builder;
    if (!b || id + 1 >= PAIR_MAX_ID) return UINT32_MAX;
    if (id + 1 > b->cap_ids) {
        uint32_t cap = b->cap_ids ? b->cap_ids : 256;
        while (cap < id + 1) cap <<= 1;
        b->start = tsimd_xrealloc(b->start, (size_t)cap * sizeof(uint32_t));
        b->len = tsimd_xrealloc(b->len, (size_t)cap * sizeof(uint32_t));
        for (uint32_t i = b->cap_ids; i < cap; i++) b->len[i] = UINT32_MAX;
        b->cap_ids = cap;
    }
    if (b->len[id] != UINT32_MAX) return UINT32_MAX;
    if (b->size + len > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->size + len) cap <<= 1;
        b->bytes = tsimd_xrealloc(b->bytes, cap);
        b->cap = cap;
    }
    memcpy(b->bytes + b->size, bytes, len);
    b->start[id] = (uint32_t)b->size;
    b->len[id] = len;
    b->size += len;
    if (len > v->max_token_len) v->max_token_len = len;
    if (id + 1 > b->max_id_plus_1) b->max_id_plus_1 = id + 1;
    return id;
}

int tsimd_vocab_seal(tsimd_vocab_t *v, size_t merge_count) {
    return tsimd_vocab_seal_ex(v, merge_count, 1);
}

int tsimd_vocab_seal_ex(tsimd_vocab_t *v, size_t merge_count,
                        int require_bytes) {
    vbuild_t *b = v->builder;
    if (!b || b->max_id_plus_1 == 0) return -1;
    v->bhash_heavy = b->heavy_bhash;
    v->vocab_size = b->max_id_plus_1;
    if (v->vocab_size > PAIR_MAX_ID) return -1;

    v->bytes = tsimd_alloc_huge(b->size ? b->size : 1);
    v->bytes_size = b->size;
    v->offsets = tsimd_alloc_huge(
        ((size_t)v->vocab_size + 1) * sizeof(uint32_t));
    size_t cur = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        v->offsets[id] = (uint32_t)cur;
        if (b->len[id] != UINT32_MAX) {
            memcpy(v->bytes + cur, b->bytes + b->start[id], b->len[id]);
            cur += b->len[id];
        }
    }
    v->offsets[v->vocab_size] = (uint32_t)cur;

    memset(v->maxlen1, 0, sizeof(v->maxlen1));
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t plen = v->offsets[id + 1] - v->offsets[id];
        if (plen == 0) continue;
        uint8_t b0 = v->bytes[v->offsets[id]];
        if (plen > v->maxlen1[b0]) v->maxlen1[b0] = plen;
    }
    free(b->bytes);
    free(b->start);
    free(b->len);
    free(b);
    v->builder = NULL;

    int have[256] = {0};
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        if (v->offsets[id + 1] - v->offsets[id] == 1) {
            uint8_t byte = v->bytes[v->offsets[id]];
            if (!have[byte]) {
                v->byte_ids[byte] = id;
                have[byte] = 1;
            }
        }
    }
    for (int i = 0; i < 256; i++)
        if (!have[i]) {
            if (require_bytes) return -1;
            v->byte_ids[i] = 0;
        }

    {
        size_t lower = merge_count + (size_t)v->vocab_size;
        size_t bound = lower > v->bytes_size / 2 ? lower : v->bytes_size / 2;
        size_t cap = 16;
        while (cap < bound * 2) cap <<= 1;
        v->pair_table = tsimd_alloc_huge_zero(cap *
                                           sizeof(uint64_t));
        if (!v->pair_table) return -1;
        v->pair_mask = cap - 1;
    }

    bhash_t *bh = calloc(1, sizeof(bhash_t));
    if (!bh) return -1;
    if (bhash_build(bh, v, v->bhash_heavy) != 0) {
        free(bh);
        return -1;
    }
    v->byte_hash = bh;

    {
        const char *e = getenv("TSIMD_DISABLE_VOCAB_FP");
        if (!(e && e[0] == '1')) {
            size_t fcap = 16;
            while (fcap < (size_t)v->vocab_size * 4) fcap <<= 1;
            if (fcap > (1u << 22)) fcap = 1u << 22;
            uint8_t *fp = tsimd_alloc_huge_zero(fcap);
            if (fp) {
                v->byte_fp = fp;
                v->fp_mask = fcap - 1;
                for (uint32_t id = 0; id < v->vocab_size; id++) {
                    uint32_t len = v->offsets[id + 1] - v->offsets[id];
                    if (len == 0) continue;
                    uint64_t h =
                        fnv1a(v->bytes + v->offsets[id], len);

                    uint8_t fp8 = (uint8_t)(h >> 56);
                    if (!fp8) fp8 = 1;
                    size_t fi = (size_t)((h >> 32) & v->fp_mask);
                    while (v->byte_fp[fi] != 0) fi = (fi + 1) & v->fp_mask;
                    v->byte_fp[fi] = fp8;
                }
            }
        }
    }

    return 0;
}

void tsimd_vocab_build_spm_trie(tsimd_vocab_t *v) {
    if (v->scores == NULL || v->vocab_size == 0) return;
    tsimd_trie_free(v->spm_trie);
    v->spm_trie = NULL;
    const uint8_t **keys =
        tsimd_xmalloc((size_t)v->vocab_size * sizeof(*keys));
    uint32_t *lens = tsimd_xmalloc((size_t)v->vocab_size * sizeof(*lens));
    uint32_t *ids = tsimd_xmalloc((size_t)v->vocab_size * sizeof(*ids));
    uint32_t kn = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len == 0) continue;
        keys[kn] = v->bytes + v->offsets[id];
        lens[kn] = len;
        ids[kn] = id;
        kn++;
    }
    v->spm_trie = tsimd_trie_build(keys, lens, ids, v->scores, kn);
    tsimd_xfree(keys);
    tsimd_xfree(lens);
    tsimd_xfree(ids);
}

static int pair_table_grow(tsimd_vocab_t *v) {
    size_t old_cap = v->pair_mask + 1;
    uint64_t *old = v->pair_table;
    size_t cap = old_cap * 2;
    uint64_t *grown = calloc(cap, sizeof(uint64_t));
    if (!grown) return -1;
    v->pair_table = grown;
    v->pair_mask = cap - 1;
    for (size_t j = 0; j < old_cap; j++) {
        uint64_t e = old[j];
        if (!e) continue;
        uint64_t key = e & ~(uint64_t)PAIR_OCCUPIED & ~(uint64_t)PAIR_MAX_ID;
        size_t i = pair_index(v, key);
        while (v->pair_table[i] != 0) i = (i + 1) & v->pair_mask;
        v->pair_table[i] = e;
    }
    free(old);
    return 0;
}

int tsimd_vocab_record_merge(tsimd_vocab_t *v, uint32_t left, uint32_t right,
                            uint32_t merged) {
    if (!v->pair_table || left >= PAIR_MAX_ID || right >= PAIR_MAX_ID ||
        merged >= PAIR_MAX_ID)
        return -1;
    uint64_t key = pair_key(left, right);
    size_t i = pair_index(v, key);
    while (v->pair_table[i] != 0) {
        if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
            (PAIR_OCCUPIED | key))
            return 0;
        i = (i + 1) & v->pair_mask;
    }

    if (v->pair_count * 4 >= (v->pair_mask + 1) * 3) {
        if (pair_table_grow(v) != 0) return -1;
        i = pair_index(v, key);
        while (v->pair_table[i] != 0) {
            if ((v->pair_table[i] & ~(uint64_t)PAIR_MAX_ID) ==
                (PAIR_OCCUPIED | key))
                return 0;
            i = (i + 1) & v->pair_mask;
        }
    }
    v->pair_table[i] = pair_pack(left, right, merged);
    v->pair_count++;
    return 0;
}

void tsimd_vocab_set_rank(tsimd_vocab_t *v, uint32_t id, uint32_t rank) {
    if (id >= v->vocab_size) return;
    if (!v->ranks) {
        v->ranks = tsimd_alloc_huge_zero(v->vocab_size *
                                        sizeof(uint32_t));
        if (!v->ranks) return;
    }
    v->ranks[id] = rank;
}

int tsimd_vocab_load_buffer(const uint8_t *data, size_t len, tsimd_vocab_t *v) {
    if (len == 0) return -1;
    if (data[len - 1] != '\n') return -1;
    if (tsimd_vocab_build_start(v) != 0) return -1;

    const uint8_t *p = data, *end = data + len;
    uint32_t id = 0;
    uint8_t decoded[8192];
    while (p < end) {
        const uint8_t *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) break;
        size_t line = (size_t)(nl - p);
        if (line > 0 && nl[-1] == '\r') line--;
        const uint8_t *sp = memchr(p, ' ', line);
        size_t blen = sp ? (size_t)(sp - p) : line;
        if (sp) {

            size_t rank = 0;
            for (const uint8_t *q = sp + 1; q < p + line; q++) {
                if (*q < '0' || *q > '9') goto fail;
                rank = rank * 10 + (size_t)(*q - '0');
            }
            if (rank != id) goto fail;
        }
        if (blen > 0) {
            if (blen > (sizeof(decoded) + 3) / 4 * 3) goto fail;
            long dlen = b64_decode((const char *)p, blen, decoded);
            if (dlen < 0) goto fail;
            if (tsimd_vocab_add_token(v, decoded, (uint32_t)dlen, id) ==
                UINT32_MAX)
                goto fail;
            id++;
        }
        p = nl + 1;
    }
    if (id == 0 || id > PAIR_MAX_ID) goto fail;

    tsimd_vocab_mark_bhash_heavy(v);
    if (tsimd_vocab_seal(v, 0) != 0) goto fail;
    if (reconstruct_merges(v, (const bhash_t *)v->byte_hash) != 0) goto fail;
    if (tsimd_vocab_pair_rank_build(v) != 0) goto fail;
    if (tsimd_vocab_sim_build(v) != 0) goto fail;
    return 0;

fail:
    tsimd_vocab_free(v);
    return -1;
}

int tsimd_vocab_load_file(const char *path, tsimd_vocab_t *v) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    rewind(f);
    uint8_t *buf = tsimd_xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    int rc = tsimd_vocab_load_buffer(buf, (size_t)sz, v);
    free(buf);
    return rc;
}

void tsimd_vocab_free(tsimd_vocab_t *v) {
    if (v->builder) {
        vbuild_t *b = v->builder;
        free(b->bytes);
        free(b->start);
        free(b->len);
        free(b);
    }
    free(v->bytes);
    free(v->offsets);
    free(v->pair_table);
    free(v->ranks);
    free(v->scores);
    free(v->mgr_dense);

    free(v->mgr_slots);
    free(v->mgr_ids);
    free(v->mgr_bgrid);
    free(v->mgr_bgrid_ids);
    free(v->mgr_id_to_b);
    free(v->rank_to_id);
    free(v->mgr_dense_ids);
    if (v->byte_hash) {
        free(((bhash_t *)v->byte_hash)->slots);
        free(v->byte_hash);
    }
    free(v->merge_reach);
    free(v->merge_sim);
    free(v->byte_tok);
    free(v->byte_fp);
    tsimd_trie_free(v->spm_trie);
    tsimd_xfree(v->spm_cross);
    memset(v, 0, sizeof(*v));
}
