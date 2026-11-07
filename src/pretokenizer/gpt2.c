#include "internal.h"

#include <stddef.h>
#include <stdint.h>

size_t tsimd_match_gpt2(const uint8_t *d, size_t n, size_t i) {

    if (d[i] == '\'' && i + 1 < n) {
        uint8_t c = d[i + 1];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < n) {
            uint8_t c2 = d[i + 2];
            if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') ||
                (c == 'l' && c2 == 'l'))
                return i + 3;
        }
    }

    size_t j = (d[i] == ' ' && i + 1 < n) ? i + 1 : i;
    uint32_t cp;
    tsimd_char_class_t cls;
    tsimd_utf8_next(d + j, n - j, &cp, &cls);

    if (cls == TSIMD_CLASS_LETTER)
        return tsimd_scan_run(d, n, j, TSIMD_CLASS_LETTER);
    if (cls == TSIMD_CLASS_NUMBER)
        return tsimd_scan_run(d, n, j, TSIMD_CLASS_NUMBER);
    if (cls == TSIMD_CLASS_OTHER)
        return tsimd_scan_run(d, n, j, TSIMD_CLASS_OTHER);

    return tsimd_ws_run_match(d, n, i);
}

size_t tsimd_match_gpt2d(const uint8_t *d, size_t n, size_t i) {

    if (d[i] == '\'' && i + 1 < n) {
        uint8_t c = d[i + 1];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < n) {
            uint8_t c2 = d[i + 2];
            if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') ||
                (c == 'l' && c2 == 'l'))
                return i + 3;
        }
    }
    size_t j = (d[i] == ' ' && i + 1 < n) ? i + 1 : i;
    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv = tsimd_utf8_next(d + j, n - j, &cp, &cls);
    if (cls == TSIMD_CLASS_NUMBER) {
        if (j == i) return i + adv;

    } else {
        if (cls == TSIMD_CLASS_LETTER)
            return tsimd_scan_run(d, n, j, TSIMD_CLASS_LETTER);
        if (cls == TSIMD_CLASS_OTHER)
            return tsimd_scan_run(d, n, j, TSIMD_CLASS_OTHER);
    }

    {
        uint32_t cp2;
        tsimd_char_class_t c2;
        size_t r = i;
        while (r < n) {
            size_t a = tsimd_utf8_next(d + r, n - r, &cp2, &c2);
            if (a == 0 || c2 != TSIMD_CLASS_SPACE) break;
            r += a;
        }
        if (r < n) {
            tsimd_utf8_next(d + r, n - r, &cp2, &c2);
            if (c2 == TSIMD_CLASS_NUMBER) return r;
        }
    }
    return tsimd_ws_run_match(d, n, i);
}

int tsimd_pretokenize_gpt2d(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_gpt2d(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}

int tsimd_pretokenize_gpt2(const uint8_t *data, size_t len,
                          tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_gpt2(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}
