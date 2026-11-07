#ifndef TSIMD_INPUT_H
#define TSIMD_INPUT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *data;
    size_t length;
    void *owned;
    int is_mmap;
} tsimd_input_t;

int tsimd_input_read_file(const char *path, tsimd_input_t *out);

int tsimd_input_mmap_file(const char *path, tsimd_input_t *out);

int tsimd_input_from_buffer(const uint8_t *data, size_t length, tsimd_input_t *out);
void tsimd_input_release(tsimd_input_t *in);

#endif
