#ifndef TSIMD_SIMD_H
#define TSIMD_SIMD_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TSIMD_SIMD_SCALAR = 0,
    TSIMD_SIMD_SSE2,
    TSIMD_SIMD_AVX2,
    TSIMD_SIMD_AVX512
} tsimd_simd_level_t;

int tsimd_simd_avx512_probe(void);

typedef struct {
    uint64_t l, d, sp, w, c, o, hi;
    uint64_t ap, suf, r, v, l1, e;
    uint64_t lu, lc;

    uint64_t sl;

} tsimd_cls_block_t;

#define TSIMD_CLS_BLOCK_BYTES 64

typedef void (*tsimd_classify_fn_t)(const uint8_t *data, int case_insensitive,
                                   tsimd_cls_block_t *blk);

tsimd_simd_level_t tsimd_simd_level(void);
tsimd_classify_fn_t tsimd_simd_classify(void);

int tsimd_simd_level_supported(tsimd_simd_level_t level);

void tsimd_simd_set_forced(int level);

#endif
