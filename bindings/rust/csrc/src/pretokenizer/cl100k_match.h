#ifndef TSIMD_CL100K_MATCH_H
#define TSIMD_CL100K_MATCH_H

#include "internal.h"

static uint8_t lower_ascii(uint8_t c) {
    return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + ('a' - 'A')) : c;
}

static size_t cl_match(const uint8_t *d, size_t n, size_t i,
                       int digit_cap3) {

    if (d[i] == '\'' && i + 1 < n) {
        uint8_t c = lower_ascii(d[i + 1]);
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < n) {
            uint8_t c2 = lower_ascii(d[i + 2]);
            if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') ||
                (c == 'l' && c2 == 'l'))
                return i + 3;
        }
    }

    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv;
    if (d[i] < 0x80) {
        cp = d[i];
        cls = (tsimd_char_class_t)tsimd_ascii_cls[d[i]];
        adv = 1;
    } else {
        adv = tsimd_utf8_next(d + i, n - i, &cp, &cls);
    }

    if (cls == TSIMD_CLASS_LETTER)
        return tsimd_scan_run(d, n, i, TSIMD_CLASS_LETTER);
    if (cp != '\r' && cp != '\n' && cls != TSIMD_CLASS_NUMBER && adv > 0) {
        size_t k = i + adv;
        if (k < n) {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            if (d[k] < 0x80) {
                cls2 = (tsimd_char_class_t)tsimd_ascii_cls[d[k]];
            } else {
                tsimd_utf8_next(d + k, n - k, &cp2, &cls2);
            }
            if (cls2 == TSIMD_CLASS_LETTER)
                return tsimd_scan_run(d, n, k, TSIMD_CLASS_LETTER);
        }
    }

    if (cls == TSIMD_CLASS_NUMBER) {
        if (!digit_cap3) return i + adv;
        size_t p = i;
        int count = 0;
        while (count < 3 && p < n) {
            if (d[p] < 0x80) {
                if (tsimd_ascii_cls[d[p]] != (uint8_t)TSIMD_CLASS_NUMBER)
                    break;
                p++;
            } else {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                if (a == 0 || cls2 != TSIMD_CLASS_NUMBER) break;
                p += a;
            }
            count++;
        }
        return p;
    }

    {
        size_t j = (d[i] == ' ') ? i + 1 : i;
        if (j < n) {
            tsimd_char_class_t cls3;
            if (d[j] < 0x80) {
                cls3 = (tsimd_char_class_t)tsimd_ascii_cls[d[j]];
            } else {
                uint32_t cp3;
                tsimd_utf8_next(d + j, n - j, &cp3, &cls3);
            }
            if (cls3 == TSIMD_CLASS_OTHER) {
                size_t e = tsimd_scan_run(d, n, j, TSIMD_CLASS_OTHER);
                while (e < n && (d[e] == '\r' || d[e] == '\n')) e++;
                return e;
            }
        }
    }

    {
        size_t p = i;
        size_t best = (size_t)-1;
        while (p < n) {
            if (d[p] < 0x80) {
                if (tsimd_ascii_cls[d[p]] != (uint8_t)TSIMD_CLASS_SPACE) break;
                if (d[p] == '\r' || d[p] == '\n') best = p;
                p++;
                continue;
            }
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
            if (a == 0 || cls2 != TSIMD_CLASS_SPACE) break;
            if (cp2 == '\r' || cp2 == '\n') best = p;
            p += a;
        }
        if (best != (size_t)-1) {
            size_t e = best;
            while (e < n && (d[e] == '\r' || d[e] == '\n')) e++;
            return e;
        }
    }

    return tsimd_ws_run_match(d, n, i);
}

static inline size_t tsimd_cl_match_inline(const uint8_t *d, size_t n,
                                           size_t i, int digit_cap3) {

    if (d[i] == '\'' && i + 1 < n) {
        uint8_t c = lower_ascii(d[i + 1]);
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return i + 2;
        if (i + 2 < n) {
            uint8_t c2 = lower_ascii(d[i + 2]);
            if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') ||
                (c == 'l' && c2 == 'l'))
                return i + 3;
        }
    }

    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv;
    if (d[i] < 0x80) {
        cp = d[i];
        cls = (tsimd_char_class_t)tsimd_ascii_cls[d[i]];
        adv = 1;
    } else {
        adv = tsimd_utf8_next(d + i, n - i, &cp, &cls);
    }

    if (cls == TSIMD_CLASS_LETTER)
        return tsimd_scan_run(d, n, i, TSIMD_CLASS_LETTER);
    if (cp != '\r' && cp != '\n' && cls != TSIMD_CLASS_NUMBER && adv > 0) {
        size_t k = i + adv;
        if (k < n) {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            if (d[k] < 0x80) {
                cls2 = (tsimd_char_class_t)tsimd_ascii_cls[d[k]];
            } else {
                tsimd_utf8_next(d + k, n - k, &cp2, &cls2);
            }
            if (cls2 == TSIMD_CLASS_LETTER)
                return tsimd_scan_run(d, n, k, TSIMD_CLASS_LETTER);
        }
    }

    if (cls == TSIMD_CLASS_NUMBER) {
        if (!digit_cap3) return i + adv;
        size_t p = i;
        int count = 0;
        while (count < 3 && p < n) {
            if (d[p] < 0x80) {
                if (tsimd_ascii_cls[d[p]] != (uint8_t)TSIMD_CLASS_NUMBER)
                    break;
                p++;
            } else {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                if (a == 0 || cls2 != TSIMD_CLASS_NUMBER) break;
                p += a;
            }
            count++;
        }
        return p;
    }

    {
        size_t j = (d[i] == ' ') ? i + 1 : i;
        if (j < n) {
            tsimd_char_class_t cls3;
            if (d[j] < 0x80) {
                cls3 = (tsimd_char_class_t)tsimd_ascii_cls[d[j]];
            } else {
                uint32_t cp3;
                tsimd_utf8_next(d + j, n - j, &cp3, &cls3);
            }
            if (cls3 == TSIMD_CLASS_OTHER) {
                size_t e = tsimd_scan_run(d, n, j, TSIMD_CLASS_OTHER);
                while (e < n && (d[e] == '\r' || d[e] == '\n')) e++;
                return e;
            }
        }
    }

    {
        size_t p = i;
        size_t best = (size_t)-1;
        while (p < n) {
            if (d[p] < 0x80) {
                if (tsimd_ascii_cls[d[p]] != (uint8_t)TSIMD_CLASS_SPACE) break;
                if (d[p] == '\r' || d[p] == '\n') best = p;
                p++;
                continue;
            }
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
            if (a == 0 || cls2 != TSIMD_CLASS_SPACE) break;
            if (cp2 == '\r' || cp2 == '\n') best = p;
            p += a;
        }
        if (best != (size_t)-1) {
            size_t e = best;
            while (e < n && (d[e] == '\r' || d[e] == '\n')) e++;
            return e;
        }
    }

    return tsimd_ws_run_match(d, n, i);
}

#endif
