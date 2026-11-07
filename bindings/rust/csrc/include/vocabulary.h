#ifndef TSIMD_VOCABULARY_H
#define TSIMD_VOCABULARY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {

    uint8_t *bytes;
    size_t bytes_size;
    uint32_t *offsets;
    uint32_t vocab_size;
    uint32_t max_token_len;

    uint32_t byte_ids[256];
    uint32_t byte_ids_hex[256];
    uint8_t raw_piece_units;
    uint8_t ignore_merges;
    uint8_t byte_fallback;

    uint64_t *pair_table;
    size_t pair_mask;
    uint32_t *mgr_dense;

    uint32_t mgr_dense_log2;
    uint64_t *mgr_slots;
    uint32_t *mgr_ids;
    size_t mgr_mask;
    unsigned mgr_shift;
    size_t pair_count;
    uint32_t *rank_to_id;
    int mgr_prefetch_ok;

    uint32_t *mgr_dense_ids;
    uint32_t *mgr_bgrid;
    uint32_t *mgr_bgrid_ids;
    uint8_t *mgr_id_to_b;

    void *byte_hash;

    uint8_t *merge_reach;
    uint8_t *merge_sim;
    uint8_t *byte_tok;
    uint8_t *byte_fp;
    size_t fp_mask;

    uint32_t *ranks;

    int spm_seam_ok;

    void *spm_cross;
    uint64_t spm_cross_prev[4];
    uint64_t spm_cross_post[4];
    uint8_t spm_cross_state;

    float *scores;
    float spm_min_score;
    uint32_t spm_unk_id;
    uint8_t spm_unk_surface[16];
    uint8_t spm_unk_surface_len;
    uint32_t spm_control_ids[8];
    uint32_t spm_control_n;

    uint32_t maxlen1[256];
    int bhash_heavy;

    struct tsimd_trie_t *spm_trie;

    void *builder;
} tsimd_vocab_t;

int tsimd_vocab_load_file(const char *path, tsimd_vocab_t *v);
int tsimd_vocab_load_buffer(const uint8_t *data, size_t len, tsimd_vocab_t *v);
void tsimd_vocab_free(tsimd_vocab_t *v);

int tsimd_vocab_build_start(tsimd_vocab_t *v);
void tsimd_vocab_mark_bhash_heavy(tsimd_vocab_t *v);

uint32_t tsimd_vocab_add_token(tsimd_vocab_t *v, const uint8_t *bytes,
                              uint32_t len, uint32_t id);

int tsimd_vocab_seal(tsimd_vocab_t *v, size_t merge_count);
void tsimd_vocab_build_spm_trie(tsimd_vocab_t *v);
int tsimd_vocab_seal_ex(tsimd_vocab_t *v, size_t merge_count,
                       int require_bytes);
int tsimd_vocab_record_merge(tsimd_vocab_t *v, uint32_t left, uint32_t right,
                            uint32_t merged);

void tsimd_vocab_set_rank(tsimd_vocab_t *v, uint32_t id, uint32_t rank);

static inline const uint8_t *tsimd_vocab_bytes(const tsimd_vocab_t *v,
                                              uint32_t id, uint32_t *len) {
    *len = v->offsets[id + 1] - v->offsets[id];
    return v->bytes + v->offsets[id];
}

uint32_t tsimd_vocab_merge_rank(const tsimd_vocab_t *v, uint32_t left,
                               uint32_t right);

uint32_t tsimd_vocab_merge_rank2(const tsimd_vocab_t *v, uint32_t left,
                                 uint32_t right);

uint32_t tsimd_vocab_merge_priority(const tsimd_vocab_t *v, uint32_t left,
                                    uint32_t right);

static inline uint32_t tsimd_vocab_merge_priority_fast(
    const tsimd_vocab_t *v, uint32_t left, uint32_t right) {
    if (v->mgr_dense && ((left | right) >> v->mgr_dense_log2) == 0)
        return v->mgr_dense[((size_t)left << v->mgr_dense_log2) | right];
    return tsimd_vocab_merge_priority(v, left, right);
}

static inline void tsimd_vocab_merge_prefetch(const tsimd_vocab_t *v,
                                             uint32_t left, uint32_t right) {
    if (!v->mgr_prefetch_ok) return;
    if (v->mgr_dense && ((left | right) >> v->mgr_dense_log2) == 0) {
        __builtin_prefetch(&v->mgr_dense[((size_t)left
                                          << v->mgr_dense_log2) |
                                         right], 0, 1);
        return;
    }
    if (v->mgr_bgrid && left < v->vocab_size && right < v->vocab_size) {
        uint8_t li = v->mgr_id_to_b[left];
        uint8_t ri = v->mgr_id_to_b[right];
        if (li != 0xFF && ri != 0xFF) {
            __builtin_prefetch(
                &v->mgr_bgrid[((size_t)li << 8) | ri], 0, 1);
            return;
        }
    }
    if (v->mgr_slots) {
        uint64_t key = ((uint64_t)left << 21) | right;
        size_t idx = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> v->mgr_shift);
        __builtin_prefetch(&v->mgr_slots[idx], 0, 1);
    }
}
uint32_t tsimd_vocab_id_of_priority(const tsimd_vocab_t *v, uint32_t prio);

uint32_t tsimd_vocab_merge_prio_id(const tsimd_vocab_t *v, uint32_t left,
                                   uint32_t right, uint32_t *id_out);

static inline uint32_t tsimd_vocab_merge_prio_id_fast(
    const tsimd_vocab_t *v, uint32_t left, uint32_t right,
    uint32_t *id_out) {
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
    return tsimd_vocab_merge_prio_id(v, left, right, id_out);
}

uint32_t tsimd_vocab_merge_id(const tsimd_vocab_t *v, uint32_t left,
                              uint32_t right);
int tsimd_vocab_pair_rank_build(tsimd_vocab_t *v);
int tsimd_vocab_pair_rank_build_slots(tsimd_vocab_t *v);
int tsimd_vocab_pair_rank_build_ex(tsimd_vocab_t *v, int want_dense);
int tsimd_vocab_sim_build(tsimd_vocab_t *v);

static inline uint32_t tsimd_vocab_priority(const tsimd_vocab_t *v,
                                           uint32_t merged_id) {
    return v->ranks ? v->ranks[merged_id] : merged_id;
}

uint32_t tsimd_vocab_find(const tsimd_vocab_t *v, const uint8_t *bytes,
                         uint32_t len);

#endif
