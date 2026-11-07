#include "simd_internal.h"

#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

#define RANGE256(vec, a, n)                                                   \
    _mm256_cmpgt_epi8(_mm256_set1_epi8((int8_t)(uint8_t)((n) + 0x80)),        \
                      _mm256_sub_epi8((vec),                                   \
                          _mm256_set1_epi8((int8_t)(uint8_t)((a) + 0x80))))

#define EQ256(vec, val) _mm256_cmpeq_epi8((vec), _mm256_set1_epi8((int8_t)(val)))

static inline uint32_t m256(__m256i v) {
    return (uint32_t)_mm256_movemask_epi8(v);
}

static inline void classify_vec256(__m256i x, int case_insensitive, int shift,
                                   tsimd_cls_block_t *blk) {
    __m256i lo = _mm256_or_si256(x, _mm256_set1_epi8(0x20));
    __m256i suf_src = case_insensitive ? lo : x;

    __m256i is_l = RANGE256(lo, 'a', 26);
    __m256i is_d = RANGE256(x, '0', 10);
    __m256i is_sp = EQ256(x, 0x20);
    __m256i t9 = _mm256_sub_epi8(x, _mm256_set1_epi8(9));
    __m256i is_w = _mm256_or_si256(_mm256_or_si256(EQ256(t9, 0), EQ256(t9, 2)),
                                   EQ256(t9, 3));
    __m256i is_c = _mm256_or_si256(EQ256(x, 10), EQ256(x, 13));
    __m256i is_cls =
        _mm256_or_si256(_mm256_or_si256(is_l, is_d),
                        _mm256_or_si256(_mm256_or_si256(is_sp, is_w), is_c));
    __m256i is_hi = _mm256_cmpgt_epi8(_mm256_set1_epi8(0), x);
    __m256i is_o =
        _mm256_andnot_si256(_mm256_or_si256(is_cls, is_hi),
                            _mm256_set1_epi8(-1));

    __m256i is_lu = RANGE256(x, 'A', 26);
    __m256i is_lc = _mm256_andnot_si256(is_lu, is_l);
    __m256i is_sl = EQ256(x, '/');
    __m256i is_ap = EQ256(x, '\'');
    __m256i is_suf =
        _mm256_or_si256(_mm256_or_si256(EQ256(suf_src, 's'), EQ256(suf_src, 't')),
                        _mm256_or_si256(EQ256(suf_src, 'm'), EQ256(suf_src, 'd')));
    __m256i is_r = EQ256(suf_src, 'r');
    __m256i is_v = EQ256(suf_src, 'v');
    __m256i is_l1 = EQ256(suf_src, 'l');
    __m256i is_e = EQ256(suf_src, 'e');

    uint64_t s = (uint64_t)shift;
    blk->l |= (uint64_t)m256(is_l) << s;
    blk->d |= (uint64_t)m256(is_d) << s;
    blk->sp |= (uint64_t)m256(is_sp) << s;
    blk->w |= (uint64_t)m256(is_w) << s;
    blk->c |= (uint64_t)m256(is_c) << s;
    blk->o |= (uint64_t)m256(is_o) << s;
    blk->hi |= (uint64_t)m256(is_hi) << s;
    blk->ap |= (uint64_t)m256(is_ap) << s;
    blk->suf |= (uint64_t)m256(is_suf) << s;
    blk->r |= (uint64_t)m256(is_r) << s;
    blk->v |= (uint64_t)m256(is_v) << s;
    blk->l1 |= (uint64_t)m256(is_l1) << s;
    blk->e |= (uint64_t)m256(is_e) << s;
    blk->lu |= (uint64_t)m256(is_lu) << s;
    blk->lc |= (uint64_t)m256(is_lc) << s;
    blk->sl |= (uint64_t)m256(is_sl) << s;
}

void tsimd_classify_block_avx2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    memset(blk, 0, sizeof(*blk));
    classify_vec256(
        _mm256_loadu_si256((const __m256i *)(const void *)data),
        case_insensitive, 0, blk);
    classify_vec256(
        _mm256_loadu_si256((const __m256i *)(const void *)(data + 32)),
        case_insensitive, 32, blk);
}

#else

void tsimd_classify_block_avx2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    tsimd_classify_block_scalar(data, case_insensitive, blk);
}

#endif
