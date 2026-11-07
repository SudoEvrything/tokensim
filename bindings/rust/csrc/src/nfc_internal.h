#ifndef TSIMD_NFC_INTERNAL_H
#define TSIMD_NFC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t in_off, in_len, out_off, out_len;
} tsimd_nfc_run_t;

typedef struct {
    uint32_t cp;
    uint32_t off;
    uint32_t len;

} tsimd_nfc_decomp_t;

typedef struct {
    uint32_t cp;
    uint8_t ccc;
} tsimd_nfc_ccc_t;

typedef struct {
    uint32_t a, b, c;
} tsimd_nfc_pair_t;

extern const uint32_t tsimd_nfc_decomp_cps[];
extern const tsimd_nfc_decomp_t tsimd_nfc_decomp[];
extern const size_t tsimd_nfc_decomp_n;

extern const uint32_t tsimd_nfkc_decomp_cps[];
extern const tsimd_nfc_decomp_t tsimd_nfkc_decomp[];
extern const size_t tsimd_nfkc_decomp_n;

extern const tsimd_nfc_ccc_t tsimd_nfc_ccc[];
extern const size_t tsimd_nfc_ccc_n;
extern const tsimd_nfc_pair_t tsimd_nfc_comp[];
extern const size_t tsimd_nfc_comp_n;

extern const uint16_t tsimd_nfc_ccc_idx[];
extern const uint8_t tsimd_nfc_ccc_blk[];

#endif
