#ifndef TSIMD_BYTE_UNICODE_H
#define TSIMD_BYTE_UNICODE_H

#include <stdint.h>

void tsimd_byteunicode_init(void);

uint32_t tsimd_byte_to_cp(uint8_t b);

int tsimd_cp_to_byte(uint32_t cp, uint8_t *out);

#endif
