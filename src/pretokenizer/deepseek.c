#include "internal.h"

#include <stddef.h>
#include <stdint.h>

static int cp_cjk_kana(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

static int cp_ascii_punct(uint32_t cp) {
    if (cp < 0x21 || cp > 0x7E) return 0;
    if (cp >= '0' && cp <= '9') return 0;
    if (cp >= 'A' && cp <= 'Z') return 0;
    if (cp >= 'a' && cp <= 'z') return 0;
    return 1;
}

static size_t ascii_letter_run(const uint8_t *d, size_t n, size_t p) {
    while (p < n) {
        uint8_t c = d[p];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) break;
        p++;
    }
    return p;
}

size_t tsimd_match_deepseek(const uint8_t *d, size_t n, size_t i) {
    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv = tsimd_utf8_next(d + i, n - i, &cp, &cls);

    if (cls == TSIMD_CLASS_NUMBER) {
        size_t p = i;
        int count = 0;
        while (count < 3) {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
            if (a == 0 || cls2 != TSIMD_CLASS_NUMBER) break;
            p += a;
            count++;
        }
        return p;
    }

    if (cp_cjk_kana(cp)) {
        size_t p = i + adv;
        while (p < n) {
            uint32_t cp2;
            size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls);
            if (a == 0 || !cp_cjk_kana(cp2)) break;
            p += a;
        }
        return p;
    }

    if (cp_ascii_punct(cp)) {
        size_t p = i + adv;
        if (p < n) {
            uint8_t c = d[p];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                return ascii_letter_run(d, n, p);
        }
    } else {

        if (cls == TSIMD_CLASS_SPACE && cp != '\r' && cp != '\n') {
            size_t p = i + adv;
            if (p < n) {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                if (a > 0 &&
                    (cls2 == TSIMD_CLASS_LETTER ||
                     (tsimd_class_codepoint_flags(cp2) & TSIMD_CAT_M)) &&
                    !cp_cjk_kana(cp2)) {
                    p += a;
                    while (p < n) {
                        a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                        uint32_t f2 = tsimd_class_codepoint_flags(cp2);
                        if (a == 0 ||
                            (cls2 != TSIMD_CLASS_LETTER && !(f2 & TSIMD_CAT_M)) ||
                            cp_cjk_kana(cp2))
                            break;
                        p += a;
                    }
                    return p;
                }
            }

            size_t q = i + adv;
            if (q < n && d[i] == ' ') {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + q, n - q, &cp2, &cls2);
                if (a > 0 && (tsimd_class_codepoint_flags(cp2) &
                              TSIMD_CAT_PS) &&
                    !cp_cjk_kana(cp2)) {
                    q += a;
                    while (q < n) {
                        a = tsimd_utf8_next(d + q, n - q, &cp2, &cls2);
                        if (a == 0 ||
                            !(tsimd_class_codepoint_flags(cp2) &
                              TSIMD_CAT_PS) ||
                            cp_cjk_kana(cp2))
                            break;
                        q += a;
                    }
                    while (q < n && (d[q] == '\r' || d[q] == '\n')) q++;
                    return q;
                }
            }
        } else if (cls == TSIMD_CLASS_LETTER && !cp_cjk_kana(cp)) {
            size_t p = i + adv;
            while (p < n) {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                uint32_t f2 = tsimd_class_codepoint_flags(cp2);
                if (a == 0 ||
                    (cls2 != TSIMD_CLASS_LETTER && !(f2 & TSIMD_CAT_M)) ||
                    cp_cjk_kana(cp2))
                    break;
                p += a;
            }
            return p;
        } else {

            uint32_t f0 = tsimd_class_codepoint_flags(cp);
            if (cp != '\r' && cp != '\n' && !(f0 & TSIMD_CAT_PS) &&
                cls != TSIMD_CLASS_LETTER) {
                size_t p = i + adv;
                if (p < n) {
                    uint32_t cp2;
                    tsimd_char_class_t cls2;
                    size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                    if (a > 0 &&
                        (cls2 == TSIMD_CLASS_LETTER ||
                         (tsimd_class_codepoint_flags(cp2) & TSIMD_CAT_M)) &&
                        !cp_cjk_kana(cp2)) {
                        p += a;
                        while (p < n) {
                            a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                            uint32_t f2 = tsimd_class_codepoint_flags(cp2);
                            if (a == 0 ||
                                (cls2 != TSIMD_CLASS_LETTER &&
                                 !(f2 & TSIMD_CAT_M)) ||
                                cp_cjk_kana(cp2))
                                break;
                            p += a;
                        }
                        return p;
                    }
                }
            }
        }
    }

    {
        uint32_t f = tsimd_class_codepoint_flags(cp);
        if (f & TSIMD_CAT_PS) {
            size_t p = i + adv;
            while (p < n) {
                size_t a = tsimd_utf8_next(d + p, n - p, &cp, &cls);
                if (a == 0 ||
                    !(tsimd_class_codepoint_flags(cp) & TSIMD_CAT_PS) ||
                    cp_cjk_kana(cp))
                    break;
                p += a;
            }
            while (p < n && (d[p] == '\r' || d[p] == '\n')) p++;
            return p;
        }
    }

    {
        size_t p = i;
        size_t best = (size_t)-1;
        uint32_t cp2;
        tsimd_char_class_t cls2;
        while (p < n) {
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
        if (cls == TSIMD_CLASS_SPACE) {

            {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
                if (a > 0 &&
                    (cls2 == TSIMD_CLASS_NUMBER || cp_cjk_kana(cp2)))
                    return p;
            }
            return tsimd_ws_run_match(d, n, i);
        }
    }

    if (adv > 0) {
        if (tsimd_class_codepoint_flags(cp) & TSIMD_CAT_M) return i + adv;
        size_t p = i + adv;
        while (p < n) {
            uint32_t cp3;
            tsimd_char_class_t cls3;
            size_t a3 = tsimd_utf8_next(d + p, n - p, &cp3, &cls3);
            if (a3 == 0) {
                p++;
                continue;
            }
            uint32_t f3 = tsimd_class_codepoint_flags(cp3);
            if (cls3 == TSIMD_CLASS_NUMBER || cls3 == TSIMD_CLASS_SPACE ||
                cls3 == TSIMD_CLASS_LETTER || cp_cjk_kana(cp3) ||
                cp_ascii_punct(cp3) || (f3 & TSIMD_CAT_PS) ||
                (f3 & TSIMD_CAT_M))
                break;
            if (p + a3 < n) {
                uint32_t cp4;
                tsimd_char_class_t cls4;
                size_t a4 =
                    tsimd_utf8_next(d + p + a3, n - p - a3, &cp4, &cls4);
                if (a4 > 0 &&
                    (cls4 == TSIMD_CLASS_LETTER ||
                     (tsimd_class_codepoint_flags(cp4) & TSIMD_CAT_M)) &&
                    !cp_cjk_kana(cp4))
                    break;
            }
            p += a3;
        }
        return p;
    }
    return i + 1;
}
int tsimd_pretokenize_deepseek(const uint8_t *data, size_t len,
                               tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_deepseek(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}
