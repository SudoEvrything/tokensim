#include "pretokenizer.h"
#include "bpe.h"

#include "../src/pretokenizer/internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void spans_of(int o200k, const uint8_t *d, size_t n,
                     tsimd_span_array_t *out) {
    tsimd_span_array_init(out);
    int rc = o200k ? tsimd_pretokenize_o200k(d, n, out)
                   : tsimd_pretokenize_cl100k(d, n, out);
    if (rc != 0) {
        fails++;
        fprintf(stderr, "FAIL matcher rc\n");
    }
}

static void check_input(int o200k, const char *tag, const uint8_t *d,
                        size_t n) {
    tsimd_span_array_t a, b;
    spans_of(o200k, d, n, &a);

    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    tsimd_pcache_span_t ents[4096];
    size_t nb = 0;
    int bad = 0;
    for (;;) {
        int c = o200k
                    ? tsimd_o200k_mask_fill_keyed(d, n, &st, ents, 4096, NULL,
                                                  0)
                    : tsimd_cl100k_mask_fill_keyed(d, n, &st, ents, 4096, NULL,
                                                   0);
        if (c < 0) {
            fails++;
            fprintf(stderr, "FAIL fill rc %s\n", tag);
            break;
        }
        if (c == 0) break;
        for (int j = 0; j < c && nb < 4096; j++, nb++) {
            size_t s0 = (size_t)(ents[j].ptr - d);
            size_t e0 = s0 + tsimd_pcache_span_len(&ents[j]);
            if (nb < a.count && a.spans[nb].start == s0 &&
                a.spans[nb].end == e0)
                continue;
            bad = 1;
            fprintf(stderr, "FAIL %s %s span %zu: fill [%zu,%zu) matcher",
                    tag, o200k ? "o200k" : "cl100k", nb, s0, e0);
            if (nb < a.count)
                fprintf(stderr, " [%u,%u)\n", a.spans[nb].start,
                        a.spans[nb].end);
            else
                fprintf(stderr, " (end)\n");
        }
    }
    if (nb != a.count) {
        bad = 1;
        fprintf(stderr, "FAIL %s %s span count fill=%zu matcher=%zu\n", tag,
                o200k ? "o200k" : "cl100k", nb, a.count);
    }
    if (bad) fails++;
    tsimd_span_array_free(&a);
}

int main(void) {
    setenv("TSIMD_HI_ALGEBRA", "1", 1);

    static const uint8_t acc[] = {
        0xc2, 0xa0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x01, 0xc3,
        0x99, 0xcb, 0xa3, 0x99, 0xcb, 0x00, 0x00, 0x32, 0x99, 0x63, 0x6c,
        0xb8, 0x6c, 0x20, 0xc2, 0xa0, 0xa0, 0xa1, 0x99, 0xae, 0xc3, 0x00,
        0x7b, 0x2c, 0x75, 0x27, 0x76, 0x61, 0x6c, 0xb8, 0x6c, 0xa3, 0x20,
        0x34, 0xc3, 0xa1, 0xc3, 0x99, 0x00, 0x80, 0x20, 0xc2, 0xa0, 0xa1,
        0xc3, 0x99, 0xae, 0xa0, 0x43, 0x43, 0x00, 0x05, 0x20, 0x22, 0x6b,
        0x2c, 0x01, 0x00, 0x00, 0x00, 0x00, 0x73
    };
    for (size_t pad = 0; pad < 72; pad += 8) {
        uint8_t buf[160];
        memset(buf, 'q', sizeof buf);
        memset(buf, 'q', pad);
        memcpy(buf + pad, acc, sizeof acc);
        size_t n = pad + sizeof acc;
        if (n > 130) n = 130;
        check_input(1, "acc-pad", buf, n);
        check_input(0, "acc-pad", buf, n);
    }

    static const uint8_t mixed[][40] = {
        { ' ', 0xC2, 0xA0, 0xA0, 0xA1, 'x' },
        { ' ', 0xC2, 0xA0, ' ', 0xC2, 0xA0, 'x' },
        { 0xC2, 0xA0, ' ', 0xC2, 0xA0, 'x' },
        { ' ', ' ', 0xC2, 0xA0, 0x80, 0x81, 'y' },
        { 'a', ' ', 0xC2, 0xA0, 0xA0, 'b' },
        { 0xC2, 0xA0, 0xA0, 0x20, 0xC2, 0xA0, 0x99, 'z' },
        { ' ', 0xE1, 0x9A, 0x80, 0xA0, 'w' },
        { 0x20, 0xC2, 0xA0, 0x20, 0xC2, 0xA0, 0xA0, 0x0A, 'q' },
    };
    for (size_t i = 0; i < sizeof mixed / sizeof mixed[0]; i++) {
        size_t n = 0;
        while (n < sizeof mixed[0] && mixed[i][n]) n++;
        if (n == 0) n = sizeof mixed[0];
        check_input(1, "mixed", mixed[i], n);
        check_input(0, "mixed", mixed[i], n);
    }

    static const uint8_t seam[][24] = {
        { '\t', '\t', '\t', 0x85, 0x85, ';', 'x' },
        { '\t', 0x85, 'x' },
        { 'a', '\t', 0x85, 0x85, 'b' },
        { 0x0A, '\t', '\t', 0x85, 0x85, ';', 'q' },
        { ' ', '\v', 0x85, '\f', '.', 0x85, 'z' },
        { '\t', 0x85, 0x85, '\t', '\t', '\t', 0x85, 'w' },
        { '>', 'w', 'o', 'r', 'd' },
        { 'a', ' ', '>', 'b' },
        { 'a', '>', '>', 'b' },
        { '/', 'x' },
        { 'a', '/', 'b' },
        { '\'', 's' },
        { 'a', '\'', 's', ' ', 'b' },
        { '"', 'q', '=' },
        { '=', 'x', '-', 'y' },
        { '<', 't', '>' },
        { 0xE6, 0x97, 0xA5, 'x' },
        { '>', 0xE6, 0x97, 0xA5 },
        { '"', '>', 'x' },
        { 'a', '\'', '>', 'b' },
        { 'a', '.', 0xE6, 0x97, 0xA5, '.', 'b' },
        { 'a', '\'', 'l', 'l' },
        { 'a', '\'', 'l', 'l', 'x' },
        { 'a', '\'', 't' },
        { 'a', '\'', 'S' },
        { 'x', '/', '/', 'y' },
        { '>', '>', '/', 'x' },
        { 'h', 't', 't', 'p', ':', '/', '/', 'a', 'b', 'c' },
        { '<', '/', 's', 'i', 't', 'e', 'n', 'a', 'm', 'e', '>' },
        { 0xE2, 0x80, 0x94, 'w', 'o', 'r', 'd' },
        { 'a', 0xE2, 0x80, 0x94, 'b' },
        { 0xE2, 0x80, 0xA6, 0xE2, 0x80, 0x94, 'x' },
        { 0xC2, 0xAB, 'q', 0xC2, 0xBB },
        { 0xE2, 0x80, 0x94, '1', '2', '3', '4', '5', '6', '7' },
        { 'x', 0xE2, 0x80, 0x94, 0xE2, 0x80, 0x94, 'y' },
    };
    for (size_t i = 0; i < sizeof seam / sizeof seam[0]; i++) {
        size_t n = 0;
        while (n < sizeof seam[0] && seam[i][n]) n++;
        check_input(1, "seam", seam[i], n);
        check_input(0, "seam", seam[i], n);
    }

    uint8_t seam_pad[80];
    memset(seam_pad, 'q', sizeof seam_pad);
    memcpy(seam_pad + 64, "\t\t\t\x85\x85;x", 7);
    check_input(1, "seam-pad", seam_pad, sizeof seam_pad);
    check_input(0, "seam-pad", seam_pad, sizeof seam_pad);

    static const uint8_t art1[] = {
        0x85, 0x0a, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
        0x09, 0x09, 0x09, 0x09, 0x85, 0x85, 0x85, 0x3b, 0x09, 0x09, 0x09,
        0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
        0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
        0x09, 0x09, 0x0b, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x19, 0x09,
        0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
        0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x85, 0x85, 0x85,
        0x3b
    };
    check_input(1, "art-tabs85", art1, sizeof art1);
    check_input(0, "art-tabs85", art1, sizeof art1);

    static const uint8_t art2[] = {
        0x32, 0x01, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x34, 0xe5, 0xa0, 0x01, 0x31,
        0x08, 0x08, 0xc0, 0xab, 0x31, 0xa0, 0x00, 0xa0, 0x5d, 0x01, 0x03,
        0x01, 0x08, 0x08, 0xc0, 0xab, 0x31, 0x00, 0xa0, 0x5d, 0x01, 0x03,
        0x01, 0x08, 0x08, 0xc0, 0xab, 0x31, 0x5b
    };
    check_input(1, "art-a0pair", art2, sizeof art2);
    check_input(0, "art-a0pair", art2, sizeof art2);

    static const uint8_t art3[] = { 0x0a, 0xc2, 0xa0, 0xa0 };
    check_input(1, "art-eos", art3, sizeof art3);
    check_input(0, "art-eos", art3, sizeof art3);

    uint64_t rng = 0x9e3779b9;
    uint8_t buf[200];
    static const uint8_t alpha[] = {
        ' ', '\t', '\n', '\v', '\f', '\r', 'a', 'z', 'A', '0', '.', '/',
        0xC2, 0xA0, 0xA1, 0x85, 0x80, 0x99, 0xC3, 0x28, 0xE1, 0x9A, 0x80,
        0x00, 0x05, '>', '<', '=', '"', '\'', '-', 0xE6, 0x97, 0xA5, ';',
        'q', 'W', 0xE2, 0x80, 0x94, 0xA6, 0xC2, 0xAB
    };
    for (int iter = 0; iter < 3000; iter++) {
        size_t n = 4 + (size_t)(rng % 190);
        for (size_t j = 0; j < n; j++) buf[j] = alpha[rng % (sizeof alpha)];
        check_input(1, "fuzz", buf, n);
        if (iter % 3 == 0) check_input(0, "fuzz", buf, n);
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
    }

    if (fails == 0) printf("test_hi_algebra: OK\n");
    return fails ? 1 : 0;
}
