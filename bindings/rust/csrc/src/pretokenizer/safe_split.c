#include "pretokenizer.h"

static int is_ascii_letter(uint8_t c) {
    return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
}

static int is_ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static int safe_pair(tsimd_pretok_scheme_t scheme, uint8_t prev,
                     uint8_t cur) {
    if (cur >= 0x80) return 0;
    if ((scheme == TSIMD_PRETOK_O200K ||
         scheme == TSIMD_PRETOK_KIMI) &&
        cur == '\'' && is_ascii_letter(prev))
        return 0;
    if (is_ascii_letter(prev)) return !is_ascii_letter(cur);
    if (is_ascii_digit(prev)) return !is_ascii_digit(cur);
    return 0;
}

#define TSIMD_SPLIT_WINDOW 65536

long tsimd_pretok_safe_split(tsimd_pretok_scheme_t scheme,
                            const uint8_t *data, size_t len, size_t hint,
                            size_t lo) {
    if (len < 2 || lo + 1 >= len) return -1;
    size_t last = len - 1;
    if (hint <= lo) hint = lo + 1;
    if (hint > last) hint = last;

    size_t floor = hint - lo > TSIMD_SPLIT_WINDOW ? hint - TSIMD_SPLIT_WINDOW
                                                 : lo + 1;
    for (size_t p = hint + 1; p-- > floor; )
        if (safe_pair(scheme, data[p - 1], data[p])) return (long)p;

    size_t ceil = last - hint > TSIMD_SPLIT_WINDOW ? hint + TSIMD_SPLIT_WINDOW
                                                  : last;
    for (size_t p = hint + 1; p <= ceil; p++)
        if (safe_pair(scheme, data[p - 1], data[p])) return (long)p;
    return -1;
}
