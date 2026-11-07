#ifndef TSIMD_O200K_MATCH_H
#define TSIMD_O200K_MATCH_H

#include "internal.h"

#if defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define TSIMD_O_SSE2 1
#endif

enum {
    OK_OTHER = 0,
    OK_LOWER,
    OK_UPPER,
    OK_DIGIT,
    OK_SPACE,
    OK_TAB,
    OK_CR,
    OK_LF,
    OK_APOS,
    OK_MULTI
};

static const uint8_t k_okind[256] = {
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,    OK_TAB,     OK_LF,    OK_TAB,    OK_TAB,     OK_CR,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_SPACE,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,   OK_APOS,
     OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,
     OK_DIGIT,  OK_DIGIT,  OK_DIGIT,  OK_DIGIT,  OK_DIGIT,  OK_DIGIT,  OK_DIGIT,  OK_DIGIT,
     OK_DIGIT,  OK_DIGIT,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,
     OK_OTHER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,
     OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,
     OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_UPPER,
     OK_UPPER,  OK_UPPER,  OK_UPPER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,
     OK_OTHER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,
     OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,
     OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_LOWER,
     OK_LOWER,  OK_LOWER,  OK_LOWER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,  OK_OTHER,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,
     OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI,  OK_MULTI
};

#define OF_UC(cp) (tsimd_class_case(cp) & 1)
#define OF_LC(cp) (tsimd_class_case(cp) & 2)

static size_t o_scan_case_run(const uint8_t *d, size_t n, size_t p,
                              int phase_l) {
    size_t last_cl = 0;
#ifdef TSIMD_O_SSE2
    while (p + 16 <= n) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(d + p));
        __m128i m;
        if (phase_l) {

            m = _mm_and_si128(_mm_cmpgt_epi8(v, _mm_set1_epi8(0x60)),
                              _mm_cmplt_epi8(v, _mm_set1_epi8(0x7B)));
        } else {

            m = _mm_and_si128(_mm_cmpgt_epi8(v, _mm_set1_epi8(0x40)),
                              _mm_cmplt_epi8(v, _mm_set1_epi8(0x5B)));
        }
        if (_mm_movemask_epi8(m) != (int)0xFFFF) break;
        p += 16;
    }
#endif
    while (p < n) {
        uint8_t kk = k_okind[d[p]];
        if (kk == OK_LOWER) {
            phase_l = 1;
            last_cl = 0;
            p++;
        } else if (kk == OK_UPPER) {
            if (phase_l) return p;
            p++;
        } else if (kk == OK_MULTI) {
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
            if (adv == 0) break;
            uint8_t cf = tsimd_class_case(cp);
            if (cf == 2) {
                phase_l = 1;
                last_cl = 0;
                p += adv;
            } else if (cf == 3 || cf == 4) {

                if (!phase_l) last_cl = p + adv;
                p += adv;
            } else if (cf == 1) {
                if (phase_l) return p;
                p += adv;
            } else break;
        } else break;
    }
    if (!phase_l && last_cl) return last_cl;
    return p;
}

static size_t o_try_suffix(const uint8_t *d, size_t n, size_t p) {
    if (p >= n || d[p] != '\'') return p;
    if (p + 1 >= n) return p;
    uint8_t c1 = (uint8_t)(d[p + 1] | 0x20);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return p + 2;
    if (p + 2 >= n) return p;
    uint8_t c2 = (uint8_t)(d[p + 2] | 0x20);
    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
        (c1 == 'l' && c2 == 'l'))
        return p + 3;
    return p;
}

static size_t o_scan_punct(const uint8_t *d, size_t n, size_t p) {
#ifdef TSIMD_O_SSE2
    while (p + 16 <= n) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(d + p));

        __m128i m = _mm_cmpgt_epi8(v, _mm_set1_epi8(0x20));
        m = _mm_and_si128(m, _mm_or_si128(
                _mm_cmplt_epi8(v, _mm_set1_epi8(0x30)),
                _mm_cmpgt_epi8(v, _mm_set1_epi8(0x39))));
        m = _mm_and_si128(m, _mm_or_si128(
                _mm_cmplt_epi8(v, _mm_set1_epi8(0x41)),
                _mm_cmpgt_epi8(v, _mm_set1_epi8(0x5A))));
        m = _mm_and_si128(m, _mm_or_si128(
                _mm_cmplt_epi8(v, _mm_set1_epi8(0x61)),
                _mm_cmpgt_epi8(v, _mm_set1_epi8(0x7A))));
        if (_mm_movemask_epi8(m) != (int)0xFFFF) break;
        p += 16;
    }
#endif
    while (p < n) {
        uint8_t k = k_okind[d[p]];
        if (k == OK_OTHER || k == OK_APOS || k == OK_MULTI) {
            if (k == OK_MULTI) {
                uint32_t cp;
                tsimd_char_class_t cls;
                size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
                if (adv == 0 || cls != TSIMD_CLASS_OTHER) return p;
                p += adv;
                continue;
            }
            p++;
            continue;
        }
        return p;
    }
    return p;
}

static size_t o_scan_tail(const uint8_t *d, size_t n, size_t p) {
    while (p < n) {
        uint8_t b = d[p];
        if (b == '\r' || b == '\n' || b == '/') p++;
        else break;
    }
    return p;
}

static size_t o_digits3(const uint8_t *d, size_t n, size_t p) {
    int count = 1;
    while (count < 3 && p < n) {
        uint8_t k = k_okind[d[p]];
        if (k == OK_DIGIT) {
            p++;
            count++;
            continue;
        }
        if (k == OK_MULTI) {
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
            if (adv == 0 || cls != TSIMD_CLASS_NUMBER) break;
            p += adv;
            count++;
            continue;
        }
        break;
    }
    return p;
}

static size_t o_ws_token_end(const uint8_t *d, size_t n, size_t start) {
    size_t p = start;
    size_t last_nl_end = 0;
    size_t last_char_start = start;
#ifdef TSIMD_O_SSE2

    while (p + 16 <= n) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(d + p));
        __m128i m = _mm_or_si128(
            _mm_or_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x20)),
                         _mm_cmpeq_epi8(v, _mm_set1_epi8(0x09))),
            _mm_or_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x0B)),
                         _mm_cmpeq_epi8(v, _mm_set1_epi8(0x0C))));
        if (_mm_movemask_epi8(m) != (int)0xFFFF) break;
        last_char_start = p + 15;
        p += 16;
    }
#endif
    while (p < n) {
        uint8_t k = k_okind[d[p]];
        size_t adv = 1;
        if (k == OK_CR || k == OK_LF) {
            last_char_start = p;
            p++;
            last_nl_end = p;
        } else if (k == OK_SPACE || k == OK_TAB) {
            last_char_start = p;
            p++;
        } else if (k == OK_MULTI) {

            uint32_t cp;
            tsimd_char_class_t cls;
            adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
            if (adv == 0 || cls != TSIMD_CLASS_SPACE) break;
            last_char_start = p;
            p += adv;
        } else break;
    }
    if (last_nl_end) return last_nl_end;
    if (p >= n) return p;
    if (last_char_start > start)
        return last_char_start;
    return p;
}

static size_t o_letter_run_from(const uint8_t *d, size_t n, size_t p,
                                int *started) {
    *started = 0;
    if (p >= n) return 0;
    uint8_t k = k_okind[d[p]];
    if (k != OK_LOWER && k != OK_UPPER && k != OK_MULTI) return 0;
    if (k != OK_MULTI) {
        *started = 1;
        int phase_l = (k == OK_LOWER);
        return o_scan_case_run(d, n, p + 1, phase_l);
    }

    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
    if (adv == 0) return 0;
    uint8_t cf = tsimd_class_case(cp);
    if (cf == 0) return 0;
    *started = 1;

    int phase_l = (cf == 2 || cf == 4) ? 1 : 0;
    size_t last_cl = (cf == 3) ? p + adv : 0;
    p += adv;
    while (p < n) {
        uint8_t kk = k_okind[d[p]];
        if (kk == OK_LOWER) { phase_l = 1; last_cl = 0; p++; }
        else if (kk == OK_UPPER) {
            if (phase_l) return p;
            p++;
        } else if (kk == OK_MULTI) {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t adv2 = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
            if (adv2 == 0) break;
            uint8_t cf2 = tsimd_class_case(cp2);
            if (cf2 == 2) {
                phase_l = 1;
                last_cl = 0;
                p += adv2;
            } else if (cf2 == 3 || cf2 == 4) {
                if (!phase_l) last_cl = p + adv2;
                p += adv2;
            } else if (cf2 == 1) {
                if (phase_l) return p;
                p += adv2;
            } else break;
        } else break;
    }
    if (!phase_l && last_cl) return last_cl;
    return p;
}

static inline size_t tsimd_o_match_inline(const uint8_t *d, size_t n, size_t i) {
    uint8_t k0 = k_okind[d[i]];

    if (k0 == OK_LOWER || k0 == OK_UPPER) {
        int phase_l = (k0 == OK_LOWER);
        size_t p = o_scan_case_run(d, n, i + 1, phase_l);
        return o_try_suffix(d, n, p);
    }

    if (k0 == OK_SPACE) {
        if (i + 1 >= n) return i + 1;
        uint8_t k1 = k_okind[d[i + 1]];
        if (k1 == OK_LOWER || k1 == OK_UPPER) {
            int phase_l = (k1 == OK_LOWER);
            size_t p = o_scan_case_run(d, n, i + 2, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (k1 == OK_DIGIT) return i + 1;
        if (k1 == OK_SPACE || k1 == OK_TAB || k1 == OK_CR || k1 == OK_LF)
            return o_ws_token_end(d, n, i);
        if (k1 != OK_MULTI)
            return o_scan_tail(d, n, o_scan_punct(d, n, i + 2));

        uint32_t cp1;
        tsimd_char_class_t cls1;
        size_t adv1 = tsimd_utf8_next(d + i + 1, n - i - 1, &cp1, &cls1);
        if (adv1 == 0) return i + 1;
        uint8_t cf1 = tsimd_class_case(cp1);
        if (cf1 & 2) {
            int phase_l = (cf1 == 2);
            size_t p = o_scan_case_run(d, n, i + 1 + adv1, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (cf1 & 1) {
            int phase_l = 0;
            size_t p = o_scan_case_run(d, n, i + 1 + adv1, phase_l);
            return o_try_suffix(d, n, p);
        }

        if (cf1 == 4) {
            int phase_l = 1;
            size_t p = o_scan_case_run(d, n, i + 1 + adv1, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (cls1 == TSIMD_CLASS_NUMBER) return i + 1;
        if (cls1 == TSIMD_CLASS_SPACE) return o_ws_token_end(d, n, i);
        return o_scan_tail(d, n, o_scan_punct(d, n, i + 1 + adv1));
    }

    if (k0 == OK_MULTI) {
        uint32_t cp0;
        tsimd_char_class_t cls0;
        size_t adv0 = tsimd_utf8_next(d + i, n - i, &cp0, &cls0);
        if (adv0 == 0) return i + 1;
        uint8_t cf0 = tsimd_class_case(cp0);
        if (cf0 & 2) {
            int phase_l = (cf0 == 2);
            size_t p = o_scan_case_run(d, n, i + adv0, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (cf0 & 1) {
            int phase_l = 0;
            size_t p = o_scan_case_run(d, n, i + adv0, phase_l);
            return o_try_suffix(d, n, p);
        }

        if (cf0 == 4) {
            int phase_l = 1;
            size_t p = o_scan_case_run(d, n, i + adv0, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (cls0 == TSIMD_CLASS_NUMBER)
            return o_digits3(d, n, i + adv0);

        int started = 0;
        size_t p = o_letter_run_from(d, n, i + adv0, &started);
        if (started) return o_try_suffix(d, n, p);
        if (cls0 == TSIMD_CLASS_SPACE) return o_ws_token_end(d, n, i);
        return o_scan_tail(d, n, o_scan_punct(d, n, i + adv0));
    }

    if (k0 == OK_DIGIT) return o_digits3(d, n, i + 1);

    if (k0 == OK_CR || k0 == OK_LF) return o_ws_token_end(d, n, i);

    if (k0 == OK_TAB) {
        if (i + 1 < n) {
            uint8_t k1 = k_okind[d[i + 1]];
            if (k1 == OK_LOWER || k1 == OK_UPPER) {
                int phase_l = (k1 == OK_LOWER);
                size_t p = o_scan_case_run(d, n, i + 2, phase_l);
                return o_try_suffix(d, n, p);
            }
            if (k1 == OK_MULTI) {
                int started = 0;
                size_t p = o_letter_run_from(d, n, i + 1, &started);
                if (started) return o_try_suffix(d, n, p);
            }
        }
        return o_ws_token_end(d, n, i);
    }

    if (i + 1 < n) {
        uint8_t k1 = k_okind[d[i + 1]];
        if (k1 == OK_LOWER || k1 == OK_UPPER) {
            int phase_l = (k1 == OK_LOWER);
            size_t p = o_scan_case_run(d, n, i + 2, phase_l);
            return o_try_suffix(d, n, p);
        }
        if (k1 == OK_MULTI) {
            int started = 0;
            size_t p = o_letter_run_from(d, n, i + 1, &started);
            if (started) return o_try_suffix(d, n, p);
        }
    }
    return o_scan_tail(d, n, o_scan_punct(d, n, i + 1));
}

#endif
