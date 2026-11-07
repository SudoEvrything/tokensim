#include "internal.h"
#include "unicode.h"

#include <stddef.h>
#include <stdint.h>

static int is_ascii_ws(uint8_t b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == 0x0b ||
           b == 0x0c;
}

static int is_letter_ascii(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
}

static int cp_han(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x20000 && cp <= 0x2FA1F) ||
           cp == 0x3005 || cp == 0x3007 ||
           (cp >= 0x3021 && cp <= 0x3029);
}

static int kimi_letterish(uint32_t cp, uint8_t cf) {
    if (cp_han(cp)) return 0;
    if (cf == 1 || cf == 2 || cf == 3 || cf == 4) return 1;
    return (tsimd_class_codepoint_flags(cp) & TSIMD_CAT_M) != 0;
}

static size_t han_run(const uint8_t *d, size_t n, size_t i) {
    size_t p = i;
    while (p < n) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
        if (adv == 0 || !cp_han(cp)) break;
        p += adv;
    }
    return p > i ? p : i + 1;
}

static size_t contraction_end(const uint8_t *d, size_t n, size_t p) {
    if (p >= n || d[p] != '\'') return p;
    if (p + 1 < n) {
        uint8_t c = d[p + 1];
        if (c == 's' || c == 'S' || c == 't' || c == 'T' || c == 'm' ||
            c == 'M' || c == 'd' || c == 'D')
            return p + 2;
        if (p + 2 < n) {
            uint8_t c2 = d[p + 2];
            if ((c == 'r' || c == 'R') && (c2 == 'e' || c2 == 'E'))
                return p + 3;
            if ((c == 'v' || c == 'V') && (c2 == 'e' || c2 == 'E'))
                return p + 3;
            if ((c == 'l' || c == 'L') && (c2 == 'l' || c2 == 'L'))
                return p + 3;
        }
    }
    return p;
}

static size_t kimi_word(const uint8_t *d, size_t n, size_t i, size_t start,
                        int phase_lower) {
    int any = 0;
    size_t p = start;
    while (p < n) {
        uint8_t b = d[p];
        if (b < 0x80) {
            if (!is_letter_ascii(b)) break;
            if (b >= 'A' && b <= 'Z') {
                if (phase_lower) break;
            } else {
                phase_lower = 1;
            }
            any = 1;
            p++;
            continue;
        }
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
        if (adv == 0) break;
        uint8_t cf = tsimd_class_case(cp);
        if (!kimi_letterish(cp, cf)) break;
        if (cf == 1) {
            if (phase_lower) break;
        } else if (cf == 2) {
            phase_lower = 1;
        }
        any = 1;
        p += adv;
    }
    return contraction_end(d, n, p);
}

static int punct_byte(uint8_t b) {
    return !is_ascii_ws(b) && !is_letter_ascii(b) && (b < '0' || b > '9');
}

static size_t punct_run(const uint8_t *d, size_t n, size_t p) {
    while (p < n) {
        uint8_t b = d[p];
        if (b < 0x80) {
            if (!punct_byte(b)) break;
            p++;
            continue;
        }
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &cls);
        if (adv == 0) break;
        if (is_ascii_ws(b) || cp_han(cp) ||
            cls == TSIMD_CLASS_LETTER || cls == TSIMD_CLASS_NUMBER)
            break;
        p += adv;
    }
    return p;
}

static size_t crlf_tail(const uint8_t *d, size_t n, size_t p) {
    while (p < n && (d[p] == '\r' || d[p] == '\n')) p++;
    return p;
}

size_t tsimd_match_kimi(const uint8_t *d, size_t n, size_t i) {
    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv = tsimd_utf8_next(d + i, n - i, &cp, &cls);
    if (adv == 0) return i + 1;

    if (cp_han(cp)) return han_run(d, n, i);

    uint8_t cf = tsimd_class_case(cp);
    int bl = is_letter_ascii(d[i]);

    if (bl || (kimi_letterish(cp, cf) && cf != 0)) {
                int seed = bl ? (d[i] >= 'a' && d[i] <= 'z') : (cf == 2);
        size_t e = kimi_word(d, n, i, i + adv, seed);
        return e > i + adv ? e : i + adv;
    }

    if (d[i] == '\'') {
        size_t j = i + 1;
        if (j < n) {
            uint8_t b2 = d[j];
            int l2 = is_letter_ascii(b2);
            int seed = is_letter_ascii(b2) ? (b2 >= 'a' && b2 <= 'z') : 0;
            if (!l2 && b2 >= 0x80) {
                uint32_t cp2;
                tsimd_char_class_t cls2;
                size_t adv2 = tsimd_utf8_next(d + j, n - j, &cp2, &cls2);
                if (adv2 > 0) {
                    uint8_t cf2 = tsimd_class_case(cp2);
                    l2 = kimi_letterish(cp2, cf2);
                    seed = (cf2 == 2);
                }
            }
            if (l2) {
                size_t e = kimi_word(d, n, i, j, seed);
                return e > j ? e : j;
            }
        }
        size_t p = punct_run(d, n, i + adv);
        p = crlf_tail(d, n, p);
        return p > i ? p : i + adv;
    }

    if (cls == TSIMD_CLASS_NUMBER) {
        size_t p = i;
        int cnt = 0;
        while (p < n && cnt < 3) {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t a = tsimd_utf8_next(d + p, n - p, &cp2, &cls2);
            if (a == 0 || cls2 != TSIMD_CLASS_NUMBER) break;
            p += a;
            cnt++;
        }
        return p;
    }

    int prefix_ok = cp != '\r' && cp != '\n' && cls != TSIMD_CLASS_NUMBER;
    if (prefix_ok && i + adv < n) {
        uint32_t cp2;
        tsimd_char_class_t cls2;
        size_t adv2 = tsimd_utf8_next(d + i + adv, n - i - adv, &cp2, &cls2);
        if (adv2 > 0) {
            uint8_t cf2 = tsimd_class_case(cp2);
            int l2 = is_letter_ascii(d[i + adv]) ||
                     (kimi_letterish(cp2, cf2) && !cp_han(cp2));
            if (l2) {
                int seed = is_letter_ascii(d[i + adv])
                               ? (d[i + adv] >= 'a' && d[i + adv] <= 'z')
                               : (cf2 == 2);
                size_t e = kimi_word(d, n, i, i + adv + adv2, seed);
                return e > i + adv + adv2 ? e : i + adv + adv2;
            }
        }
        size_t p = punct_run(d, n, i + adv);
        p = crlf_tail(d, n, p);
        if (!(p == i + adv && is_ascii_ws(d[i])))
            return p > i ? p : i + adv;
    }

    if (is_ascii_ws(d[i])) {
        size_t wsend = i;
        while (wsend < n && is_ascii_ws(d[wsend])) wsend++;
        size_t last_crlf = 0;
        int found = 0;
        for (size_t k = i; k < wsend; k++)
            if (d[k] == '\r' || d[k] == '\n') {
                last_crlf = k;
                found = 1;
            }
        if (found) return last_crlf + 1;
        if (wsend == n) return wsend;
        if (wsend == i + 1 && d[i] == ' ') {
            uint32_t cp2;
            tsimd_char_class_t cls2;
            size_t adv2 = tsimd_utf8_next(d + wsend, n - wsend, &cp2, &cls2);
            if (adv2 > 0) {
                int p2 = !is_ascii_ws(d[wsend]) && cp2 != '\r' &&
                         cp2 != '\n' && !cp_han(cp2) &&
                         cls2 != TSIMD_CLASS_LETTER &&
                         cls2 != TSIMD_CLASS_NUMBER;
                if (p2 && !is_letter_ascii(d[wsend])) {
                    size_t pe = punct_run(d, n, wsend);
                    return crlf_tail(d, n, pe);
                }
                if (cls2 == TSIMD_CLASS_NUMBER) return wsend;
            }
        }
        if (wsend > i + 1) return wsend - 1;
        return wsend;
    }

    return i + adv;
}

int tsimd_pretokenize_kimi(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_kimi(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}
