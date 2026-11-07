#ifndef TSIMD_SIMD_INTERNAL_H
#define TSIMD_SIMD_INTERNAL_H

#include "simd.h"

enum {
    CLS_L = 0,
    CLS_D,
    CLS_SP,
    CLS_W,
    CLS_C,
    CLS_O,
    CLS_HI
};

typedef struct {
    uint8_t byte;
    uint8_t cls;
} cls_hist_t;

static inline uint8_t tsimd_class_byte(uint8_t b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return CLS_L;
    if (b >= '0' && b <= '9') return CLS_D;
    if (b == 0x20) return CLS_SP;
    if (b == 9 || b == 11 || b == 12) return CLS_W;
    if (b == 10 || b == 13) return CLS_C;
    return CLS_O;
}

#if !defined(__cplusplus) && \
    (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>

static inline int tsimd_cpu_bmi2(void) {
    uint32_t a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 1;
    return (b >> 8) & 1u;
}
#endif

#if defined(__aarch64__)
#include <arm_neon.h>

static inline uint32_t tsimd_neon_mask16(uint8x16_t v) {
    uint64x2_t q = vreinterpretq_u64_u8(vshrq_n_u8(v, 7));
    uint64_t lo = vgetq_lane_u64(q, 0);
    uint64_t hi = vgetq_lane_u64(q, 1);
    uint32_t r = 0;
    for (int i = 0; i < 8; i++) {
        r |= (uint32_t)(lo & 1) << i;
        lo >>= 8;
    }
    for (int i = 0; i < 8; i++) {
        r |= (uint32_t)(hi & 1) << (8 + i);
        hi >>= 8;
    }
    return r;
}
#endif

void tsimd_classify_block_scalar(const uint8_t *data, int case_insensitive,
                                tsimd_cls_block_t *blk);
void tsimd_classify_block_neon(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk);
void tsimd_classify_block_sse2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk);
void tsimd_classify_block_avx2(const uint8_t *data, int case_insensitive,
                              tsimd_cls_block_t *blk);
void tsimd_classify_block_avx512(const uint8_t *data, int case_insensitive,
                                tsimd_cls_block_t *blk);
int tsimd_avx512_selftest(void);

#endif
