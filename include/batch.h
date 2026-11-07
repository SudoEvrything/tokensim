#ifndef TSIMD_BATCH_H
#define TSIMD_BATCH_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *data;
    size_t length;
} tsimd_input_span_t;

typedef struct {
    uint32_t *ids;
    size_t id_count;
    size_t *offsets;
    size_t n_docs;
} tsimd_token_batch_t;

#endif
