#ifndef TSIMD_UNICODE_H
#define TSIMD_UNICODE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TSIMD_CLASS_OTHER = 0,
    TSIMD_CLASS_LETTER,
    TSIMD_CLASS_NUMBER,
    TSIMD_CLASS_SPACE
} tsimd_char_class_t;

tsimd_char_class_t tsimd_class_ascii(uint8_t b);

tsimd_char_class_t tsimd_class_codepoint(uint32_t cp);

size_t tsimd_utf8_decode(const uint8_t *data, size_t len, uint32_t *cp);

size_t tsimd_utf8_next(const uint8_t *data, size_t len,
                      uint32_t *cp, tsimd_char_class_t *cls);

uint8_t tsimd_class_case(uint32_t cp);

#define TSIMD_CAT_PS  (1u << 3)
#define TSIMD_CAT_M   (1u << 4)
#define TSIMD_CAT_CF  (1u << 5)
uint32_t tsimd_class_codepoint_flags(uint32_t cp);

extern const uint16_t tsimd_bmp_props[0xFF80];
int tsimd_bmp_props_selftest(uint32_t *n_bad);
#define TSIMD_CAT_BITS (TSIMD_CAT_PS | TSIMD_CAT_M | TSIMD_CAT_CF)

size_t tsimd_nfc_normalize(const uint8_t *in, size_t len,
                           uint8_t *out, size_t cap);

int tsimd_nfc_may_change(const uint8_t *in, size_t len);

uint8_t tsimd_nfc_ccc_lookup(uint32_t cp);

size_t tsimd_nfkc_normalize(const uint8_t *in, size_t len,
                            uint8_t *out, size_t cap);

int tsimd_nfkc_may_change(const uint8_t *in, size_t len);

extern const size_t tsimd_nfkc_worst;
extern const size_t tsimd_nfkc_out_worst;

#endif
