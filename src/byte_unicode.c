#include "byte_unicode.h"

#include <string.h>

static uint32_t k_byte_to_cp[256];
static int16_t k_cp_to_byte[0x150];
static int k_ready;

void tsimd_byteunicode_init(void) {
    if (k_ready) return;
    memset(k_cp_to_byte, -1, sizeof(k_cp_to_byte));
    unsigned n = 0;
    for (unsigned b = 0; b < 256; b++) {
        int self = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                   (b >= 0xAE && b <= 0xFF);
        if (self) k_byte_to_cp[b] = b;
        else k_byte_to_cp[b] = 0x100u + n++;
    }
    for (unsigned b = 0; b < 256; b++) k_cp_to_byte[k_byte_to_cp[b]] = (int16_t)b;
    k_ready = 1;
}

uint32_t tsimd_byte_to_cp(uint8_t b) { return k_byte_to_cp[b]; }

int tsimd_cp_to_byte(uint32_t cp, uint8_t *out) {
    if (cp >= 0x150) return 0;
    if (k_cp_to_byte[cp] < 0) return 0;
    *out = (uint8_t)k_cp_to_byte[cp];
    return 1;
}
