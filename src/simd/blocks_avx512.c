#include "simd_internal.h"

#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

#define RANGE512(vec, a, n)                                                   \
    _mm512_cmplt_epu8_mask(_mm512_sub_epi8((vec),                              \
        _mm512_set1_epi8((int8_t)(uint8_t)(a))), _mm512_set1_epi8((uint8_t)(n)))

#define EQ512(vec, val)                                                       \
    _mm512_cmpeq_epi8_mask((vec), _mm512_set1_epi8((int8_t)(val)))

void tsimd_classify_block_avx512(const uint8_t *data, int case_insensitive,
                                tsimd_cls_block_t *blk) {
    memset(blk, 0, sizeof(*blk));
    __m512i x = _mm512_loadu_si512((const void *)data);
    __m512i lo = _mm512_or_si512(x, _mm512_set1_epi8(0x20));
    __m512i suf_src = case_insensitive ? lo : x;

    __mmask64 is_l = RANGE512(lo, 'a', 26);
    __mmask64 is_d = RANGE512(x, '0', 10);
    __mmask64 is_sp = EQ512(x, 0x20);
    __m512i t9 = _mm512_sub_epi8(x, _mm512_set1_epi8(9));
    __mmask64 is_w = EQ512(t9, 0) | EQ512(t9, 2) | EQ512(t9, 3);
    __mmask64 is_c = EQ512(x, 10) | EQ512(x, 13);
    __mmask64 is_cls = is_l | is_d | is_sp | is_w | is_c;
    __mmask64 is_hi = _mm512_cmpgt_epu8_mask(x, _mm512_set1_epi8(0x7F));
    __mmask64 is_o = ~is_cls & ~is_hi;

    __mmask64 is_lu = RANGE512(x, 'A', 26);
    __mmask64 is_lc = is_l & ~is_lu;
    __mmask64 is_sl = EQ512(x, '/');
    __mmask64 is_ap = EQ512(x, '\'');
    __mmask64 is_suf = EQ512(suf_src, 's') | EQ512(suf_src, 't') |
                       EQ512(suf_src, 'm') | EQ512(suf_src, 'd');

    blk->l = is_l;
    blk->d = is_d;
    blk->sp = is_sp;
    blk->w = is_w;
    blk->c = is_c;
    blk->o = is_o;
    blk->hi = is_hi;
    blk->ap = is_ap;
    blk->suf = is_suf;
    blk->r = EQ512(suf_src, 'r');
    blk->v = EQ512(suf_src, 'v');
    blk->l1 = EQ512(suf_src, 'l');
    blk->e = EQ512(suf_src, 'e');
    blk->lu = is_lu;
    blk->lc = is_lc;
    blk->sl = is_sl;
}

int tsimd_avx512_selftest(void) {
    __m512i v = _mm512_set1_epi32(0x12345678);
    __mmask16 k = _mm512_cmpeq_epi32_mask(v, v);
    __m512i w = _mm512_maskz_mov_epi32(
        k, _mm512_add_epi32(v, _mm512_set1_epi32(1)));
    uint32_t tmp[16];
    _mm512_storeu_si512((void *)tmp, w);
    return ((int)k == 0xFFFF && tmp[0] == 0x12345679u &&
            tmp[15] == 0x12345679u)
               ? 1
               : 0;
}

#else

void tsimd_classify_block_avx512(const uint8_t *data, int case_insensitive,
                                tsimd_cls_block_t *blk) {
    tsimd_classify_block_scalar(data, case_insensitive, blk);
}

int tsimd_avx512_selftest(void) {
    return 0;
}

#endif
