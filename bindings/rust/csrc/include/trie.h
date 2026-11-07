#ifndef TSIMD_TRIE_H
#define TSIMD_TRIE_H

#include <stdint.h>

#define TSIMD_TRIE_NIL 0xFFFFFFFFu
#define TSIMD_TRIE_LABEL 0x000000FFu
#define TSIMD_TRIE_PIECE 0x00000100u
#define TSIMD_TRIE_CHILD 0x00000200u
#define TSIMD_TRIE_BASE_SHIFT 10
#define TSIMD_TRIE_MAX_UNITS (1u << 22)

typedef struct {
    float score;
    uint32_t id;
} tsimd_piece_t;

typedef struct tsimd_trie_t {
    uint32_t *u;
    tsimd_piece_t *piece;
    uint32_t n;
} tsimd_trie_t;

tsimd_trie_t *tsimd_trie_build(const uint8_t *const *keys,
                               const uint32_t *lens, const uint32_t *ids,
                               const float *scores, uint32_t n);
void tsimd_trie_free(tsimd_trie_t *t);

static inline uint32_t tsimd_trie_root(const tsimd_trie_t *t) {
    (void)t;
    return 0;
}

static inline uint32_t tsimd_trie_step(const tsimd_trie_t *t, uint32_t cur,
                                       uint8_t byte, uint32_t *idx_out) {
    uint32_t cu = t->u[cur];
    if (!(cu & TSIMD_TRIE_CHILD)) return TSIMD_TRIE_NIL;
    uint32_t idx = (cu >> TSIMD_TRIE_BASE_SHIFT) + byte;
    uint32_t nu = t->u[idx];
    if ((nu & TSIMD_TRIE_LABEL) != byte) return TSIMD_TRIE_NIL;
    *idx_out = idx;
    return idx;
}

static inline uint32_t tsimd_trie_piece_id(const tsimd_trie_t *t,
                                           uint32_t cur, uint32_t idx) {
    (void)cur;
    return (t->u[idx] & TSIMD_TRIE_PIECE) ? t->piece[idx].id
                                          : TSIMD_TRIE_NIL;
}

static inline uint32_t tsimd_trie_advance(const tsimd_trie_t *t, uint32_t cu,
                                          uint8_t byte, uint32_t *idx_out,
                                          uint32_t *unit_out) {
    if (!(cu & TSIMD_TRIE_CHILD)) {
        *idx_out = TSIMD_TRIE_NIL;
        *unit_out = 0;
        return TSIMD_TRIE_NIL;
    }
    uint32_t idx = (cu >> TSIMD_TRIE_BASE_SHIFT) + byte;
    uint32_t nu = t->u[idx];
    *idx_out = idx;
    *unit_out = nu;
    if ((nu & TSIMD_TRIE_LABEL) != byte) return TSIMD_TRIE_NIL;
    return idx;
}

static inline const tsimd_piece_t *tsimd_trie_piece(const tsimd_trie_t *t,
                                                    uint32_t idx) {
    return &t->piece[idx];
}

static inline uint32_t tsimd_trie_base_of(const tsimd_trie_t *t,
                                          uint32_t unit) {
    return t->u[unit] >> TSIMD_TRIE_BASE_SHIFT;
}

#endif
