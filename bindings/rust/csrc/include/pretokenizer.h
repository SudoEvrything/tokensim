#ifndef TSIMD_PRETOKENIZER_H
#define TSIMD_PRETOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TSIMD_PRETOK_GPT2 = 0,
    TSIMD_PRETOK_CL100K,
    TSIMD_PRETOK_QWEN25,

    TSIMD_PRETOK_GPT2D,

    TSIMD_PRETOK_DEEPSEEK,

    TSIMD_PRETOK_O200K,

    TSIMD_PRETOK_SPM,

    TSIMD_PRETOK_GEM4,

    TSIMD_PRETOK_NEMO,
    TSIMD_PRETOK_KIMI

} tsimd_pretok_scheme_t;

typedef struct {
    uint32_t start;
    uint32_t end;

} tsimd_span_t;

typedef struct {
    tsimd_span_t *spans;
    size_t count;
    size_t capacity;
} tsimd_span_array_t;

void tsimd_span_array_init(tsimd_span_array_t *arr);
void tsimd_span_array_free(tsimd_span_array_t *arr);
int  tsimd_span_array_push(tsimd_span_array_t *arr, uint32_t start, uint32_t end);

int tsimd_pretokenize(tsimd_pretok_scheme_t scheme,
                     const uint8_t *data, size_t len,
                     tsimd_span_array_t *out);

typedef int (*tsimd_span_fn)(void *ud, size_t start, size_t end);

int tsimd_pretokenize_stream(tsimd_pretok_scheme_t scheme,
                            const uint8_t *data, size_t len,
                            tsimd_span_fn fn, void *ud,
                            tsimd_span_array_t *out);

int tsimd_pretokenize_scalar_ref(tsimd_pretok_scheme_t scheme,
                                const uint8_t *data, size_t len,
                                tsimd_span_array_t *out);

long tsimd_pretok_safe_split(tsimd_pretok_scheme_t scheme,
                            const uint8_t *data, size_t len, size_t hint,
                            size_t lo);

int  tsimd_pretokenize_qwen25(const uint8_t *data, size_t len,
                              tsimd_span_array_t *out);
int  tsimd_pretokenize_o200k(const uint8_t *data, size_t len,
                             tsimd_span_array_t *out);
int  tsimd_pretokenize_cl100k_masks(const uint8_t *data, size_t len,
                                    tsimd_span_array_t *out);
int  tsimd_pretokenize_nemo(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out);
int  tsimd_pretokenize_kimi(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out);
int  tsimd_pretokenize_gpt2d(const uint8_t *data, size_t len,
                             tsimd_span_array_t *out);
int  tsimd_pretokenize_gpt2_masks(const uint8_t *data, size_t len,
                                  tsimd_span_array_t *out);
int  tsimd_pretokenize_gpt2d_masks(const uint8_t *data, size_t len,
                                   tsimd_span_array_t *out);
int  tsimd_pretokenize_o200k_masks(const uint8_t *data, size_t len,
                                   tsimd_span_array_t *out);
int  tsimd_pretokenize_deepseek_masks(const uint8_t *data, size_t len,
                                      tsimd_span_array_t *out);
int  tsimd_pretokenize_nemo_masks(const uint8_t *data, size_t len,
                                  tsimd_span_array_t *out);

typedef struct {
    size_t pos;
    size_t scan;

    int digit_single;
    int pre_valid;
    size_t pre_scan;
    uint64_t pre_usable;
    uint64_t pre_bad;
} tsimd_o200k_mstate_t;

void tsimd_o200k_mask_stats(size_t *, size_t *);
void tsimd_cl100k_mask_stats(size_t *, size_t *);
void tsimd_deepseek_mask_stats(size_t *, size_t *);
int  tsimd_o200k_mask_fill(const uint8_t *data, size_t len, void *state,
                           size_t *starts, size_t *ends, size_t cap);

size_t tsimd_o200k_flatten_bounds(uint64_t bits, uint32_t scan_base,
                                  uint32_t *out, size_t cap);

size_t tsimd_o200k_flatten_bounds_bmi2(uint64_t bits, uint32_t scan_base,
                                       uint32_t *out, size_t cap);

size_t tsimd_flatten_bits16(uint64_t m, uint16_t rel, uint16_t *out);

#endif
