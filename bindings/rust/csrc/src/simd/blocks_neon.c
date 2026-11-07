#include "simd_internal.h"

#include <string.h>

#if defined(__aarch64__)

#define RANGE8(vec, a, n)                                                     \
    vcltq_s8(vsubq_s8(vreinterpretq_s8_u8(vec),                                \
                      vdupq_n_s8((int8_t)(uint8_t)((a) + 0x80))),              \
             vdupq_n_s8((int8_t)(uint8_t)((n) + 0x80)))

#define EQ8(vec, val) vceqq_u8(vec, vdupq_n_u8(val))

static inline uint32_t m16(uint8x16_t v) {
    return tsimd_neon_mask16(v);
}

static inline void classify_vec128(uint8x16_t x, int case_insensitive,
                                   int shift, tsimd_cls_block_t *blk) {
    uint8x16_t lo = vorrq_u8(x, vdupq_n_u8(0x20));
    uint8x16_t suf_src = case_insensitive ? lo : x;

    uint8x16_t is_l = RANGE8(lo, 'a', 26);
    uint8x16_t is_d = RANGE8(x, '0', 10);
    uint8x16_t is_sp = EQ8(x, 0x20);
    uint8x16_t t9 = vsubq_u8(x, vdupq_n_u8(9));
    uint8x16_t is_w = vorrq_u8(vorrq_u8(EQ8(t9, 0), EQ8(t9, 2)),
                               EQ8(t9, 3));
    uint8x16_t is_c = vorrq_u8(EQ8(x, 10), EQ8(x, 13));
    uint8x16_t is_cls = vorrq_u8(vorrq_u8(is_l, is_d),
                                 vorrq_u8(vorrq_u8(is_sp, is_w),
                                          is_c));
    uint8x16_t is_hi = vcltq_s8(vreinterpretq_s8_u8(x), vdupq_n_s8(0));
    uint8x16_t is_o = vmvnq_u8(vorrq_u8(is_cls, is_hi));

    uint8x16_t is_lu = RANGE8(x, 'A', 26);
    uint8x16_t is_lc = vbicq_u8(is_l, is_lu);
    uint8x16_t is_sl = EQ8(x, '/');
    uint8x16_t is_ap = EQ8(x, '\'');
    uint8x16_t is_suf = vorrq_u8(vorrq_u8(EQ8(suf_src, 's'),
                                          EQ8(suf_src, 't')),
                                 vorrq_u8(EQ8(suf_src, 'm'),
                                          EQ8(suf_src, 'd')));
    uint8x16_t is_r = EQ8(suf_src, 'r');
    uint8x16_t is_v = EQ8(suf_src, 'v');
    uint8x16_t is_l1 = EQ8(suf_src, 'l');
    uint8x16_t is_e = EQ8(suf_src, 'e');

    uint64_t s = (uint64_t)shift;
    blk->l |= (uint64_t)m16(is_l) << s;
    blk->d |= (uint64_t)m16(is_d) << s;
    blk->sp |= (uint64_t)m16(is_sp) << s;
    blk->w |= (uint64_t)m16(is_w) << s;
    blk->c |= (uint64_t)m16(is_c) << s;
    blk->o |= (uint64_t)m16(is_o) << s;
    blk->hi |= (uint64_t)m16(is_hi) << s;
    blk->ap |= (uint64_t)m16(is_ap) << s;
    blk->suf |= (uint64_t)m16(is_suf) << s;
    blk->r |= (uint64_t)m16(is_r) << s;
    blk->v |= (uint64_t)m16(is_v) << s;
    blk->l1 |= (uint64_t)m16(is_l1) << s;
    blk->e |= (uint64_t)m16(is_e) << s;
    blk->lu |= (uint64_t)m16(is_lu) << s;
    blk->lc |= (uint64_t)m16(is_lc) << s;
    blk->sl |= (uint64_t)m16(is_sl) << s;
}

void tsimd_classify_block_neon(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    memset(blk, 0, sizeof(*blk));
    for (int q = 0; q < 4; q++)
        classify_vec128(vld1q_u8(data + 16 * q), case_insensitive, 16 * q,
                        blk);
}

#else

void tsimd_classify_block_neon(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk) {
    tsimd_classify_block_scalar(data, case_insensitive, blk);
}

#endif
