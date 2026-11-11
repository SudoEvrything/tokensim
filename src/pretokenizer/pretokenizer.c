#include "pretokenizer.h"
#include "internal.h"

#include <stdlib.h>

void tsimd_span_array_init(tsimd_span_array_t *arr) {
    arr->spans = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

void tsimd_span_array_free(tsimd_span_array_t *arr) {
    free(arr->spans);
    arr->spans = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

int tsimd_span_array_push(tsimd_span_array_t *arr, uint32_t start, uint32_t end) {
    if (arr->count == arr->capacity) {
        size_t cap = arr->capacity ? arr->capacity * 2 : 256;
        tsimd_span_t *spans = realloc(arr->spans, cap * sizeof(*spans));
        if (!spans) return -1;
        arr->spans = spans;
        arr->capacity = cap;
    }
    arr->spans[arr->count].start = start;
    arr->spans[arr->count].end = end;
    arr->count++;
    return 0;
}

int tsimd_pretokenize_gpt2(const uint8_t *data, size_t len,
                          tsimd_span_array_t *out);
int tsimd_pretokenize_cl100k(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out);
int tsimd_pretokenize_nemo(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out);

int tsimd_pretokenize_scalar_ref(tsimd_pretok_scheme_t scheme,
                                const uint8_t *data, size_t len,
                                tsimd_span_array_t *out) {
    switch (scheme) {
        case TSIMD_PRETOK_GPT2:
            return tsimd_pretokenize_gpt2(data, len, out);
        case TSIMD_PRETOK_CL100K:
            return tsimd_pretokenize_cl100k(data, len, out);
        case TSIMD_PRETOK_QWEN25:
            return tsimd_pretokenize_qwen25(data, len, out);
        case TSIMD_PRETOK_GPT2D:
            return tsimd_pretokenize_gpt2d(data, len, out);
        case TSIMD_PRETOK_DEEPSEEK:
            return tsimd_pretokenize_deepseek(data, len, out);
        case TSIMD_PRETOK_O200K:
            return tsimd_pretokenize_o200k(data, len, out);
        case TSIMD_PRETOK_SPM:
        case TSIMD_PRETOK_GEM4:

            return tsimd_pretokenize_gem4(data, len, out);
        case TSIMD_PRETOK_NEMO:
            return tsimd_pretokenize_nemo(data, len, out);
        case TSIMD_PRETOK_KIMI:
            return tsimd_pretokenize_kimi(data, len, out);
    }
    return -1;
}

int tsimd_pretokenize_gem4(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t s = i;
        i++;
        while (i < len && data[i] != ' ') i++;
        if (tsimd_span_array_push(out, (uint32_t)s, (uint32_t)i) != 0)
            return -1;
    }
    return 0;
}

int tsimd_pretokenize(tsimd_pretok_scheme_t scheme,
                     const uint8_t *data, size_t len,
                     tsimd_span_array_t *out) {

    if (scheme == TSIMD_PRETOK_GEM4)
        return tsimd_pretokenize_gem4(data, len, out);
    if (scheme == TSIMD_PRETOK_O200K ||
        scheme == TSIMD_PRETOK_CL100K ||
        scheme == TSIMD_PRETOK_QWEN25 ||
        scheme == TSIMD_PRETOK_GPT2D ||
        scheme == TSIMD_PRETOK_DEEPSEEK ||
        scheme == TSIMD_PRETOK_NEMO ||
        scheme == TSIMD_PRETOK_KIMI) {
        if ((data == NULL && len > 0) || out == NULL) return -1;
        if (scheme == TSIMD_PRETOK_O200K)
            return tsimd_pretokenize_o200k(data, len, out);
        if (scheme == TSIMD_PRETOK_CL100K)
            return tsimd_pretokenize_cl100k(data, len, out);
        if (scheme == TSIMD_PRETOK_NEMO)
            return tsimd_pretokenize_nemo(data, len, out);
        if (scheme == TSIMD_PRETOK_QWEN25)
            return tsimd_pretokenize_qwen25(data, len, out);
        if (scheme == TSIMD_PRETOK_GPT2D)
            return tsimd_pretokenize_gpt2d(data, len, out);
        if (scheme == TSIMD_PRETOK_KIMI)
            return tsimd_pretokenize_kimi(data, len, out);
        return tsimd_pretokenize_deepseek(data, len, out);
    }
    return tsimd_pretokenize_simd_ex(scheme, data, len, NULL, NULL, out);
}

int tsimd_pretokenize_stream(tsimd_pretok_scheme_t scheme,
                            const uint8_t *data, size_t len,
                            tsimd_span_fn fn, void *ud,
                            tsimd_span_array_t *out) {
    if ((fn != NULL) == (out != NULL)) return -1;
    if (scheme == TSIMD_PRETOK_O200K ||
        scheme == TSIMD_PRETOK_QWEN25 ||
        scheme == TSIMD_PRETOK_GPT2D ||
        scheme == TSIMD_PRETOK_DEEPSEEK ||
        scheme == TSIMD_PRETOK_NEMO ||
        scheme == TSIMD_PRETOK_KIMI) {
        if (fn) {
            size_t i = 0;
            while (i < len) {
                size_t e = scheme == TSIMD_PRETOK_O200K
                               ? tsimd_match_o200k(data, len, i)
                               : scheme == TSIMD_PRETOK_NEMO
                                     ? tsimd_match_nemo(data, len, i)
                                     : scheme == TSIMD_PRETOK_KIMI
                                           ? tsimd_match_kimi(data, len, i)
                                           : scheme == TSIMD_PRETOK_QWEN25
                                           ? tsimd_match_qwen25(data, len, i)
                                           : scheme == TSIMD_PRETOK_GPT2D
                                                 ? tsimd_match_gpt2d(data, len, i)
                                                 : tsimd_match_deepseek(data, len,
                                                                        i);
                if (e <= i || e > len) return -1;
                if (fn(ud, i, e) != 0) return -1;
                i = e;
            }
            return 0;
        }
        if ((data == NULL && len > 0) || out == NULL) return -1;
        if (scheme == TSIMD_PRETOK_O200K)
            return tsimd_pretokenize_o200k(data, len, out);
        if (scheme == TSIMD_PRETOK_NEMO)
            return tsimd_pretokenize_nemo(data, len, out);
        if (scheme == TSIMD_PRETOK_QWEN25)
            return tsimd_pretokenize_qwen25(data, len, out);
        if (scheme == TSIMD_PRETOK_GPT2D)
            return tsimd_pretokenize_gpt2d(data, len, out);
        if (scheme == TSIMD_PRETOK_KIMI)
            return tsimd_pretokenize_kimi(data, len, out);
        return tsimd_pretokenize_deepseek(data, len, out);
    }
    return tsimd_pretokenize_simd_ex(scheme, data, len, fn, ud, out);
}
