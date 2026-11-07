#ifndef TSIMD_PRETOK_INTERNAL_H
#define TSIMD_PRETOK_INTERNAL_H

#include "pretokenizer.h"
#include "unicode.h"
#include "../simd/simd_internal.h"

enum { TSIMD_CLS_HI = 4 };

static const uint8_t tsimd_ascii_cls[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
};

static inline size_t tsimd_scan_run(const uint8_t *d, size_t n, size_t pos,
                                   tsimd_char_class_t cls) {
    size_t p = pos;
    while (p < n) {
        uint8_t b = d[p];
        if (b < 0x80) {
            if (tsimd_ascii_cls[b] != (uint8_t)cls) break;
            p++;
            continue;
        }
        uint32_t cp;
        tsimd_char_class_t c;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &c);
        if (adv == 0 || c != cls) break;
        p += adv;
    }
    return p;
}

static inline size_t tsimd_scan_run_ex(const uint8_t *d, size_t n, size_t pos,
                                      tsimd_char_class_t cls,
                                      size_t *last_cp_start) {
    size_t p = pos;
    *last_cp_start = pos;
    while (p < n) {
        uint8_t b = d[p];
        if (b < 0x80) {
            if (tsimd_ascii_cls[b] != (uint8_t)cls) break;
            *last_cp_start = p;
            p++;
            continue;
        }
        uint32_t cp;
        tsimd_char_class_t c;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &c);
        if (adv == 0 || c != cls) break;
        *last_cp_start = p;
        p += adv;
    }
    return p;
}

static inline size_t tsimd_ws_run_match(const uint8_t *d, size_t n, size_t i) {
    size_t last;
    size_t end = tsimd_scan_run_ex(d, n, i, TSIMD_CLASS_SPACE, &last);
    if (end == n) return end;
    if (last != i) return last;
    return end;
}

size_t tsimd_match_gpt2(const uint8_t *d, size_t n, size_t i);
size_t tsimd_match_cl100k(const uint8_t *d, size_t n, size_t i);
int tsimd_pretokenize_cl100k(const uint8_t *data, size_t len,
                             tsimd_span_array_t *out);
int tsimd_pretokenize_qwen25(const uint8_t *data, size_t len,
                             tsimd_span_array_t *out);
size_t tsimd_match_qwen25(const uint8_t *d, size_t n, size_t i);
size_t tsimd_match_gpt2d(const uint8_t *d, size_t n, size_t i);
size_t tsimd_match_deepseek(const uint8_t *d, size_t n, size_t i);
int tsimd_pretokenize_gem4(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out);
int tsimd_pretokenize_deepseek(const uint8_t *data, size_t len,
                               tsimd_span_array_t *out);
size_t tsimd_match_o200k(const uint8_t *d, size_t n, size_t i);
size_t tsimd_match_nemo(const uint8_t *d, size_t n, size_t i);
size_t tsimd_match_kimi(const uint8_t *d, size_t n, size_t i);

int tsimd_pretokenize_o200k_masks(const uint8_t *data, size_t len,
                                  tsimd_span_array_t *out);
int tsimd_pretokenize_o200k(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out);

int tsimd_pretokenize_scalar_ref(tsimd_pretok_scheme_t scheme,
                                const uint8_t *data, size_t len,
                                tsimd_span_array_t *out);

int tsimd_pretokenize_simd(tsimd_pretok_scheme_t scheme,
                          const uint8_t *data, size_t len,
                          tsimd_span_array_t *out);

int tsimd_pretokenize_kimi_masks(const uint8_t *data, size_t len,
                                 tsimd_span_array_t *out);
int tsimd_pretokenize_kimi_masks_ex(const uint8_t *data, size_t len,
                                    tsimd_span_fn fn, void *ud);
int tsimd_pretokenize_simd_ex(tsimd_pretok_scheme_t scheme,
                             const uint8_t *data, size_t len,
                             tsimd_span_fn fn, void *ud,
                             tsimd_span_array_t *out);

void tsimd_debug_o200k_mask(const uint8_t *data, size_t base,
                            uint64_t *cc_out, uint64_t *supp_out,
                            uint64_t *suppPre_out, uint64_t *suppC_out,
                            uint64_t *caseB_out, uint64_t *m_out);

#endif
