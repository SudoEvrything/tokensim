#include "unicode.h"

tsimd_char_class_t tsimd_class_ascii(uint8_t b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return TSIMD_CLASS_LETTER;
    if (b >= '0' && b <= '9') return TSIMD_CLASS_NUMBER;
    switch (b) {
        case '\t': case '\n': case '\v': case '\f': case '\r': case ' ':
            return TSIMD_CLASS_SPACE;
        default:
            return TSIMD_CLASS_OTHER;
    }
}

size_t tsimd_utf8_decode(const uint8_t *data, size_t len, uint32_t *cp) {
    if (len == 0) {
        *cp = 0;
        return 0;
    }
    uint8_t b = data[0];
    if (b < 0x80) {
        *cp = b;
        return 1;
    }
    uint32_t cp_val;
    size_t need;
    if ((b & 0xE0) == 0xC0) { cp_val = b & 0x1Fu; need = 1; }
    else if ((b & 0xF0) == 0xE0) { cp_val = b & 0x0Fu; need = 2; }
    else if ((b & 0xF8) == 0xF0) { cp_val = b & 0x07u; need = 3; }
    else { *cp = b; return 1; }
    if (len < need + 1) { *cp = b; return 1; }
    for (size_t i = 1; i <= need; i++) {
        if ((data[i] & 0xC0) != 0x80) { *cp = b; return 1; }
        cp_val = (cp_val << 6) | (uint32_t)(data[i] & 0x3Fu);
    }

    if ((need == 1 && cp_val < 0x80) ||
        (need == 2 && cp_val < 0x800) ||
        (need == 3 && cp_val < 0x10000) ||
        cp_val > 0x10FFFF ||
        (cp_val >= 0xD800 && cp_val <= 0xDFFF)) {
        *cp = b;
        return 1;
    }
    *cp = cp_val;
    return need + 1;
}

size_t tsimd_utf8_next(const uint8_t *data, size_t len,
                      uint32_t *cp, tsimd_char_class_t *cls) {
    size_t n = tsimd_utf8_decode(data, len, cp);
    if (n == 1 && data[0] >= 0x80) {
        *cls = TSIMD_CLASS_OTHER;
        return n;
    }
    *cls = (*cp < 0x80) ? tsimd_class_ascii((uint8_t)*cp)
                        : tsimd_class_codepoint(*cp);
    return n;
}
