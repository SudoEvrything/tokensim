#include "simd_internal.h"

#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>

#define RANGE8(vec, a, n)                                                     \
    _mm_cmplt_epi8(_mm_sub_epi8((vec),                                        \
                                _mm_set1_epi8((int8_t)(uint8_t)((a) + 0x80))),\
                   _mm_set1_epi8((int8_t)(uint8_t)((n) + 0x80)))

#define EQ8(vec, val) _mm_cmpeq_epi8((vec), _mm_set1_epi8((int8_t)(val)))

static inline uint32_t m8(__m128i v) {
    return (uint32_t)(uint16_t)_mm_movemask_epi8(v);
}

static inline void classify_vec128(__m128i x, int case_insensitive, int shift,
                                   tsimd_cls_block_t *blk) {
    __m128i lo = _mm_or_si128(x, _mm_set1_epi8(0x20));
    __m128i suf_src = case_insensitive ? lo : x;

    __m128i is_l = RANGE8(lo, 'a', 26);
    __m128i is_d = RANGE8(x, '0', 10);
    __m128i is_sp = EQ8(x, 0x20);
    __m128i t9 = _mm_sub_epi8(x, _mm_set1_epi8(9));
    __m128i is_w = _mm_or_si128(_mm_or_si128(EQ8(t9, 0), EQ8(t9, 2)),
                                EQ8(t9, 3));
    __m128i is_c = _mm_or_si128(EQ8(x, 10), EQ8(x, 13));
    __m128i is_cls = _mm_or_si128(_mm_or_si128(is_l, is_d),
                                  _mm_or_si128(_mm_or_si128(is_sp, is_w),
                                               is_c));
    __m128i is_hi = _mm_cmplt_epi8(x, _mm_set1_epi8(0));
    __m128i is_o = _mm_andnot_si128(_mm_or_si128(is_cls, is_hi),
                                    _mm_set1_epi8(-1));

    __m128i is_lu = RANGE8(x, 'A', 26);
    __m128i is_lc = _mm_andnot_si128(is_lu, is_l);
    __m128i is_sl = EQ8(x, '/');
    __m128i is_ap = EQ8(x, '\'');
    __m128i is_suf = _mm_or_si128(_mm_or_si128(EQ8(suf_src, 's'),
                                               EQ8(suf_src, 't')),
                                  _mm_or_si128(EQ8(suf_src, 'm'),
                                               EQ8(suf_src, 'd')));
    __m128i is_r = EQ8(suf_src, 'r');
    __m128i is_v = EQ8(suf_src, 'v');
    __m128i is_l1 = EQ8(suf_src, 'l');
    __m128i is_e = EQ8(suf_src, 'e');

    uint64_t s = (uint64_t)shift;
    blk->l |= (uint64_t)m8(is_l) << s;
    blk->d |= (uint64_t)m8(is_d) << s;
    blk->sp |= (uint64_t)m8(is_sp) << s;
    blk->w |= (uint64_t)m8(is_w) << s;
    blk->c |= (uint64_t)m8(is_c) << s;
    blk->o |= (uint64_t)m8(is_o) << s;
    blk->hi |= (uint64_t)m8(is_hi) << s;
    blk->ap |= (uint64_t)m8(is_ap) << s;
    blk->suf |= (uint64_t)m8(is_suf) << s;
    blk->r |= (uint64_t)m8(is_r) << s;
    blk->v |= (uint64_t)m8(is_v) << s;
    blk->l1 |= (uint64_t)m8(is_l1) << s;
    blk->e |= (uint64_t)m8(is_e) << s;
    blk->lu |= (uint64_t)m8(is_lu) << s;
    blk->lc |= (uint64_t)m8(is_lc) << s;
    blk->sl |= (uint64_t)m8(is_sl) << s;
}

void tsimd_classify_block_sse2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    memset(blk, 0, sizeof(*blk));
    for (int q = 0; q < 4; q++)
        classify_vec128(
            _mm_loadu_si128((const __m128i *)(const void *)(data + 16 * q)),
            case_insensitive, 16 * q, blk);
}

#else

void tsimd_classify_block_sse2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    tsimd_classify_block_scalar(data, case_insensitive, blk);
}

#endif
