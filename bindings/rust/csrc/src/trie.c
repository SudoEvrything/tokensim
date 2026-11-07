#include "trie.h"

#include "allocator.h"

#include <stdio.h>
#include <stdlib.h>

#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *p;
    uint32_t len;
    uint32_t id;
} tkey_t;

static int key_cmp(const void *a, const void *b) {
    const tkey_t *x = (const tkey_t *)a;
    const tkey_t *y = (const tkey_t *)b;
    uint32_t m = x->len < y->len ? x->len : y->len;
    int c = m ? memcmp(x->p, y->p, m) : 0;
    if (c != 0) return c;
    if (x->len != y->len) return x->len < y->len ? -1 : 1;
    if (x->id != y->id) return x->id < y->id ? -1 : 1;
    return 0;
}

void tsimd_trie_free(tsimd_trie_t *t) {
    if (!t) return;
    tsimd_xfree(t->u);
    tsimd_xfree(t->piece);
    tsimd_xfree(t);
}

static tsimd_trie_t *empty_trie(void) {
    tsimd_trie_t *t = tsimd_xmalloc(sizeof(*t));
    t->n = 1;
    t->u = tsimd_xmalloc(sizeof(uint32_t));
    t->piece = tsimd_xmalloc(sizeof(tsimd_piece_t));
    t->u[0] = 0;
    t->piece[0].id = TSIMD_TRIE_NIL;
    t->piece[0].score = 0.0f;
    return t;
}

static void pool_grow(tsimd_trie_t *t, uint32_t **next_free,
                      uint32_t **prev_free, uint32_t *cap,
                      uint32_t *free_head) {
    uint32_t old = *cap;
    uint32_t ncap = old + old / 2 + 768;
    if (ncap > TSIMD_TRIE_MAX_UNITS) ncap = TSIMD_TRIE_MAX_UNITS;
    if (ncap <= old) return;
    t->u = tsimd_xrealloc(t->u, (size_t)ncap * sizeof(uint32_t));
    t->piece = tsimd_xrealloc(t->piece, (size_t)ncap * sizeof(*t->piece));
    for (uint32_t i = old; i < ncap; i++) {
        t->u[i] = 0;
        t->piece[i].id = TSIMD_TRIE_NIL;
        t->piece[i].score = 0.0f;
    }
    *next_free = tsimd_xrealloc(*next_free, (size_t)ncap * sizeof(uint32_t));
    *prev_free = tsimd_xrealloc(*prev_free, (size_t)ncap * sizeof(uint32_t));
    for (uint32_t i = old; i < ncap; i++) {
        (*next_free)[i] = (i + 1 < ncap) ? i + 1 : TSIMD_TRIE_NIL;
        (*prev_free)[i] = i - 1;
    }
    uint32_t z = *free_head;
    if (z == TSIMD_TRIE_NIL) {
        *free_head = old;
        (*prev_free)[old] = TSIMD_TRIE_NIL;
    } else {
        while ((*next_free)[z] != TSIMD_TRIE_NIL) z = (*next_free)[z];
        (*next_free)[z] = old;
        (*prev_free)[old] = z;
    }
    *cap = ncap;
    t->n = ncap;
}

typedef struct {
    uint8_t *used;
    uint8_t *bases;
    uint32_t cap;
} trie_bits_t;

static int used_bits_of(const trie_bits_t *b, uint32_t i) {
    return (b->used[i >> 3] >> (i & 7)) & 1u;
}

static void used_bits_set(trie_bits_t *b, uint32_t i) {
    b->used[i >> 3] |= (uint8_t)(1u << (i & 7));
}

static int base_bits_of(const trie_bits_t *b, uint32_t i) {
    return (b->bases[i >> 3] >> (i & 7)) & 1u;
}

static void base_bits_set(trie_bits_t *b, uint32_t i) {
    b->bases[i >> 3] |= (uint8_t)(1u << (i & 7));
}

static void bits_grow(trie_bits_t *b, uint32_t cap) {
    size_t nb = ((size_t)cap + 7) / 8;
    b->used = tsimd_xrealloc(b->used, nb);
    b->bases = tsimd_xrealloc(b->bases, nb);
    memset(b->used + (b->cap + 7) / 8, 0, nb - (b->cap + 7) / 8);
    memset(b->bases + (b->cap + 7) / 8, 0, nb - (b->cap + 7) / 8);
    b->cap = cap;
}

tsimd_trie_t *tsimd_trie_build(const uint8_t *const *keys,
                               const uint32_t *lens, const uint32_t *ids,
                               const float *scores, uint32_t n) {
    size_t total = 0;
    uint32_t kn = 0, maxlen = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (keys[i] == NULL || lens[i] == 0) continue;
        total += lens[i];
        if (lens[i] > maxlen) maxlen = lens[i];
        kn++;
    }
    if (kn == 0) return empty_trie();

    tkey_t *ks = tsimd_xmalloc((size_t)kn * sizeof(*ks));
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (keys[i] == NULL || lens[i] == 0) continue;
        ks[m].p = keys[i];
        ks[m].len = lens[i];
        ks[m].id = ids[i];
        m++;
    }
    qsort(ks, m, sizeof(*ks), key_cmp);

    uint32_t nodes_cap = (uint32_t)total + 1;
    uint32_t *nparent = tsimd_xmalloc((size_t)nodes_cap * sizeof(uint32_t));
    uint32_t *nid = tsimd_xmalloc((size_t)nodes_cap * sizeof(uint32_t));
    uint8_t *nlabel = tsimd_xmalloc(nodes_cap);
    uint32_t nodes = 1;
    nparent[0] = 0;
    nlabel[0] = 0;
    nid[0] = TSIMD_TRIE_NIL;
    uint32_t *path = tsimd_xmalloc(((size_t)maxlen + 1) * sizeof(uint32_t));
    path[0] = 0;
    const uint8_t *prev = NULL;
    uint32_t prev_len = 0;
    for (uint32_t i = 0; i < m; i++) {
        uint32_t len = ks[i].len;
        uint32_t common = 0;
        uint32_t mm = len < prev_len ? len : prev_len;
        while (common < mm && prev[common] == ks[i].p[common]) common++;
        uint32_t cur = path[common];
        for (uint32_t d = common; d < len; d++) {
            uint32_t node = nodes++;
            nparent[node] = cur;
            nlabel[node] = ks[i].p[d];
            nid[node] = TSIMD_TRIE_NIL;
            cur = node;
            path[d + 1] = node;
        }
        if (len > 0 && nid[cur] == TSIMD_TRIE_NIL) nid[cur] = ks[i].id;
        prev = ks[i].p;
        prev_len = len;
    }
    tsimd_xfree(path);
    tsimd_xfree(ks);

    uint32_t *child_off = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    uint32_t *child_n = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    uint32_t *child = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    uint32_t *fill = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    for (uint32_t i = 0; i < nodes; i++) child_n[i] = 0;
    for (uint32_t i = 1; i < nodes; i++) child_n[nparent[i]]++;
    uint32_t acc = 0;
    for (uint32_t i = 0; i < nodes; i++) {
        child_off[i] = acc;
        fill[i] = acc;
        acc += child_n[i];
    }
    for (uint32_t i = 1; i < nodes; i++) child[fill[nparent[i]]++] = i;

    tsimd_trie_t *t = tsimd_xmalloc(sizeof(*t));
    uint32_t cap = nodes + nodes / 2 + 768;
    if (cap > TSIMD_TRIE_MAX_UNITS) cap = TSIMD_TRIE_MAX_UNITS;
    t->n = cap;
    t->u = tsimd_xmalloc((size_t)cap * sizeof(uint32_t));
    t->piece = tsimd_xmalloc((size_t)cap * sizeof(*t->piece));
    for (uint32_t i = 0; i < cap; i++) {
        t->u[i] = 0;
        t->piece[i].id = TSIMD_TRIE_NIL;
        t->piece[i].score = 0.0f;
    }
    trie_bits_t bits;
    {
        size_t nb = ((size_t)cap + 7) / 8;
        bits.used = tsimd_xmalloc(nb);
        bits.bases = tsimd_xmalloc(nb);
        memset(bits.used, 0, nb);
        memset(bits.bases, 0, nb);
        bits.cap = cap;
    }
    uint32_t *next_free = tsimd_xmalloc((size_t)cap * sizeof(uint32_t));
    uint32_t *prev_free = tsimd_xmalloc((size_t)cap * sizeof(uint32_t));
    for (uint32_t i = 0; i < cap; i++) {
        next_free[i] = (i + 1 < cap) ? i + 1 : TSIMD_TRIE_NIL;
        prev_free[i] = (i == 0) ? TSIMD_TRIE_NIL : i - 1;
    }
    next_free[0] = TSIMD_TRIE_NIL;
    prev_free[1] = TSIMD_TRIE_NIL;
    uint32_t free_head = 1;
    uint32_t *unit_of = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    uint32_t *queue = tsimd_xmalloc((size_t)nodes * sizeof(uint32_t));
    uint32_t qh = 0, qt = 0;
    queue[qt++] = 0;
    unit_of[0] = 0;
    int ok = 1;
    while (qh < qt && ok) {
        uint32_t node = queue[qh++];
        uint32_t nc = child_n[node];
        if (nc == 0) continue;
        uint8_t cmin = 255, cmax = 0;
        for (uint32_t k = 0; k < nc; k++) {
            uint8_t b = nlabel[child[child_off[node] + k]];
            if (b < cmin) cmin = b;
            if (b > cmax) cmax = b;
        }
        uint32_t base = 0;
        uint32_t sf = free_head;
        uint32_t tries = 0;
        for (;;) {
            for (; sf != TSIMD_TRIE_NIL; sf = next_free[sf]) {
                tries++;
                if (sf < cmin) continue;
                uint32_t b = sf - cmin;
                if (b == 0 || b + 255u >= cap) continue;
                if (base_bits_of(&bits, b)) continue;
                int fit = 1;
                for (uint32_t k = 0; k < nc; k++) {
                    uint32_t t2 = b + nlabel[child[child_off[node] + k]];
                    if (used_bits_of(&bits, t2)) {
                        fit = 0;
                        break;
                    }
                }
                if (fit) {
                    base = b;
                    break;
                }
            }
            if (base != 0) break;
            uint32_t before = cap;
            uint32_t fresh = cap;
            pool_grow(t, &next_free, &prev_free, &cap, &free_head);
            if (cap == before) {
                ok = 0;
                break;
            }
            bits_grow(&bits, cap);
            sf = fresh;
            tries = 0;
        }
        if (!ok) break;
        base_bits_set(&bits, base);
        t->u[unit_of[node]] |=
            (base << TSIMD_TRIE_BASE_SHIFT) | TSIMD_TRIE_CHILD;
        for (uint32_t k = 0; k < nc; k++) {
            uint32_t ch = child[child_off[node] + k];
            uint32_t s = base + nlabel[ch];
            uint32_t nx = next_free[s], pr = prev_free[s];
            if (pr != TSIMD_TRIE_NIL) next_free[pr] = nx;
            else free_head = nx;
            if (nx != TSIMD_TRIE_NIL) prev_free[nx] = pr;
            used_bits_set(&bits, s);
            unit_of[ch] = s;
            t->u[s] = nlabel[ch] |
                     (nid[ch] != TSIMD_TRIE_NIL ? TSIMD_TRIE_PIECE : 0);
            t->piece[s].id = nid[ch];
            t->piece[s].score = (scores && nid[ch] != TSIMD_TRIE_NIL)
                                    ? scores[nid[ch]]
                                    : 0.0f;
            queue[qt++] = ch;
        }
    }
    tsimd_xfree(queue);
    tsimd_xfree(unit_of);
    tsimd_xfree(next_free);
    tsimd_xfree(prev_free);
    tsimd_xfree(bits.used);
    tsimd_xfree(bits.bases);
    tsimd_xfree(child);
    tsimd_xfree(child_off);
    tsimd_xfree(child_n);
    tsimd_xfree(fill);
    tsimd_xfree(nparent);
    tsimd_xfree(nlabel);
    tsimd_xfree(nid);
    if (!ok) {
        tsimd_trie_free(t);
        return NULL;
    }
    return t;
}
