#include "pretokenizer.h"
#include "internal.h"
#include "cl100k_match.h"

#include <stddef.h>
#include <stdint.h>

size_t tsimd_match_cl100k(const uint8_t *d, size_t n, size_t i) {
    return tsimd_cl_match_inline(d, n, i, 1);
}

size_t tsimd_match_qwen25(const uint8_t *d, size_t n, size_t i) {
    return tsimd_cl_match_inline(d, n, i, 0);
}

int tsimd_pretokenize_cl100k(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_cl100k(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}

int tsimd_pretokenize_qwen25(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_qwen25(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}
