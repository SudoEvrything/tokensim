#include "pretokenizer.h"
#include "internal.h"
#include "o200k_match.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

size_t tsimd_match_o200k(const uint8_t *d, size_t n, size_t i) {
    return tsimd_o_match_inline(d, n, i);
}

int tsimd_pretokenize_o200k(const uint8_t *data, size_t len,
                            tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_o200k(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}

size_t tsimd_match_nemo(const uint8_t *d, size_t n, size_t i) {
    uint8_t k0 = k_okind[d[i]];

    if (k0 == OK_LOWER || k0 == OK_UPPER) {
        int phase_l = (k0 == OK_LOWER);
        return o_scan_case_run(d, n, i + 1, phase_l);
    }

    if (k0 == OK_SPACE) {
        if (i + 1 >= n) return i + 1;
        uint8_t k1 = k_okind[d[i + 1]];
        if (k1 == OK_LOWER || k1 == OK_UPPER) {
            int phase_l = (k1 == OK_LOWER);
            return o_scan_case_run(d, n, i + 2, phase_l);
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
            return o_scan_case_run(d, n, i + 1 + adv1, phase_l);
        }
        if (cf1 & 1) {
            size_t p = o_scan_case_run(d, n, i + 1 + adv1, 0);
            return p;
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
            return o_scan_case_run(d, n, i + adv0, phase_l);
        }
        if (cf0 & 1) {
            return o_scan_case_run(d, n, i + adv0, 0);
        }
        if (cls0 == TSIMD_CLASS_NUMBER)
            return i + adv0;

        int started = 0;
        size_t p = o_letter_run_from(d, n, i + adv0, &started);
        if (started) return p;
        if (cls0 == TSIMD_CLASS_SPACE) return o_ws_token_end(d, n, i);
        return o_scan_tail(d, n, o_scan_punct(d, n, i + adv0));
    }

    if (k0 == OK_DIGIT) return i + 1;

    if (k0 == OK_CR || k0 == OK_LF) return o_ws_token_end(d, n, i);

    if (k0 == OK_TAB) {
        if (i + 1 < n) {
            uint8_t k1 = k_okind[d[i + 1]];
            if (k1 == OK_LOWER || k1 == OK_UPPER) {
                int phase_l = (k1 == OK_LOWER);
                return o_scan_case_run(d, n, i + 2, phase_l);
            }
            if (k1 == OK_MULTI) {
                int started = 0;
                size_t p = o_letter_run_from(d, n, i + 1, &started);
                if (started) return p;
            }
        }
        return o_ws_token_end(d, n, i);
    }

    if (i + 1 < n) {
        uint8_t k1 = k_okind[d[i + 1]];
        if (k1 == OK_LOWER || k1 == OK_UPPER) {
            int phase_l = (k1 == OK_LOWER);
            return o_scan_case_run(d, n, i + 2, phase_l);
        }
        if (k1 == OK_MULTI) {
            int started = 0;
            size_t p = o_letter_run_from(d, n, i + 1, &started);
            if (started) return p;
        }
    }
    return o_scan_tail(d, n, o_scan_punct(d, n, i + 1));
}

int tsimd_pretokenize_nemo(const uint8_t *data, size_t len,
                           tsimd_span_array_t *out) {
    size_t i = 0;
    while (i < len) {
        size_t e = tsimd_match_nemo(data, len, i);
        if (e <= i || e > len) return -1;
        if (tsimd_span_array_push(out, (uint32_t)i, (uint32_t)e) != 0)
            return -1;
        i = e;
    }
    return 0;
}
