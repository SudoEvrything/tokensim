#include "unicode.h"

#include "../src/pretokenizer/internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void chk(int ok, const char *what, long a, long b, size_t idx) {
    if (ok) return;
    fails++;
    fprintf(stderr, "FAIL %s: got %ld want %ld at case %zu\n", what, a, b,
            idx);
}

static size_t ref_scan_run(const uint8_t *d, size_t n, size_t pos,
                           tsimd_char_class_t cls) {
    size_t p = pos;
    while (p < n) {
        uint32_t cp;
        tsimd_char_class_t c;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &c);
        if (adv == 0 || c != cls) break;
        p += adv;
    }
    return p;
}

static size_t ref_scan_run_ex(const uint8_t *d, size_t n, size_t pos,
                              tsimd_char_class_t cls, size_t *last) {
    size_t p = pos;
    *last = pos;
    while (p < n) {
        uint32_t cp;
        tsimd_char_class_t c;
        size_t adv = tsimd_utf8_next(d + p, n - p, &cp, &c);
        if (adv == 0 || c != cls) break;
        *last = p;
        p += adv;
    }
    return p;
}

static size_t ref_ws_run_match(const uint8_t *d, size_t n, size_t i) {
    size_t last;
    size_t end = ref_scan_run_ex(d, n, i, TSIMD_CLASS_SPACE, &last);
    if (end == n) return end;
    if (last != i) return last;
    return end;
}

static void check_buffer(const uint8_t *d, size_t n, size_t case_idx) {
    static const tsimd_char_class_t all[] = {
        TSIMD_CLASS_OTHER, TSIMD_CLASS_LETTER, TSIMD_CLASS_NUMBER,
        TSIMD_CLASS_SPACE
    };
    size_t starts[3];
    starts[0] = 0;
    starts[1] = n / 2;
    starts[2] = n ? n - 1 : 0;
    for (size_t si = 0; si < 3; si++) {
        for (size_t ci = 0; ci < 4; ci++) {
            tsimd_char_class_t cls = all[ci];
            size_t last_a = 0, last_b = 0;
            size_t a = tsimd_scan_run(d, n, starts[si], cls);
            size_t b = ref_scan_run(d, n, starts[si], cls);
            chk(a == b, "scan_run", (long)a, (long)b, case_idx * 1000 + si * 4 + ci);
            a = tsimd_scan_run_ex(d, n, starts[si], cls, &last_a);
            b = ref_scan_run_ex(d, n, starts[si], cls, &last_b);
            chk(a == b, "scan_run_ex", (long)a, (long)b,
                case_idx * 1000 + si * 4 + ci);
            chk(last_a == last_b, "scan_run_ex last", (long)last_a,
                (long)last_b, case_idx * 1000 + si * 4 + ci);
            if (cls == TSIMD_CLASS_SPACE) {
                a = tsimd_ws_run_match(d, n, starts[si]);
                b = ref_ws_run_match(d, n, starts[si]);
                chk(a == b, "ws_run_match", (long)a, (long)b,
                    case_idx * 1000 + si * 4 + ci);
            }
        }
    }
}

int main(void) {
    for (unsigned b = 0; b < 256; b++) {
        int hi = b >= 0x80;
        tsimd_char_class_t want =
            hi ? (tsimd_char_class_t)TSIMD_CLS_HI : tsimd_class_ascii((uint8_t)b);
        chk((int)tsimd_ascii_cls[b] == (int)want, "table",
            (long)tsimd_ascii_cls[b], (long)want, b);
    }

    static const uint8_t cases[][64] = {
        { 'a', 'b', 'c', 'Z', '9', ' ' },
        { 'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!' },
        { ' ', ' ', '\t', '\n', '\v', '\f', '\r', ' ', 'x' },
        { '0', '1', '2', 'a', '3', '4' },
        { 0xC3, 0xA9, 'a', 'b', 0xC3, 0xA9, 'c', 'd' },
        { 'a', 0xC2, 0xA0, 'b', ' ', 'c' },
        { '1', 0xD9, 0xA1, '2', '3' },
        { 0xE2, 0x82, 0xAC, 'x', 0xC2, 0xBC, 'y' },
        { 0xFF, 0x80, 'a', 'b' },
        { 'a', 'b', 0xC3 },
        { 'x', 0x41, 0x5A, 0x5B, 0x61, 0x7A, 0x7B, 0x30, 0x39, 0x3A },
        { 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x21 },
        { ' ', ' ', ' ', ' ', 'a', ' ', ' ', ' ' },
        { 'a' },
        { 0xC3, 0xA9, 0xC3, 0xA9, 0xC3, 0xA9 },
        { ' ', 0xC3, 0xA9, ' ', 'b' },
        { '/', 'a', '/', 'b' },
        { '\'', 's', ' ', 'A' },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        size_t n = 0;
        while (n < sizeof cases[0] && cases[i][n]) n++;
        if (n == 0) n = sizeof cases[0];
        check_buffer(cases[i], n, i);
    }

    uint64_t rng = 0x12345678;
    uint8_t buf[160];
    static const uint8_t alpha[] = {
        'a', 'z', 'A', 'Z', '0', '9', ' ', '\t', '\n', '\r', '.', '/', '\'',
        0xC3, 0xA9, 0xC2, 0xA0, 0xD9, 0xA1, 0xE2, 0x82, 0xAC, 0xFF, 0x80,
        0x41, 0x5B, 0x7B, 0x3A, 0x21, 0x7E
    };
    for (int iter = 0; iter < 4096; iter++) {
        size_t n = 1 + (size_t)(rng % 160);
        for (size_t j = 0; j < n; j++)
            buf[j] = alpha[rng % (sizeof alpha)];
        check_buffer(buf, n, 10000 + (size_t)iter);
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
    }

    if (fails == 0) printf("test_cls_table: OK\n");
    return fails ? 1 : 0;
}
