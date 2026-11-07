#include "tokenizer.h"
#include "unicode.h"
#include "util.h"

#include "../src/nfc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t in_off, in_len, out_off, out_len;
} nfc_run_t;

size_t tsimd_nfc_normalize_runs(const uint8_t *in, size_t len, uint8_t *out,
                                size_t cap, nfc_run_t *runs, size_t max_runs);
size_t tsimd_nfkc_normalize_runs(const uint8_t *in, size_t len, uint8_t *out,
                                 size_t cap, nfc_run_t *runs, size_t max_runs);

#define NRUNS 2048

#define P_mark "a\xcc\x81"
#define P_cjk "\xe4\xb8\xad\xe6\x96\x87"
#define P_hangul "\xe1\x84\x92" "\xe1\x85\xa1" "\xe1\x86\xab"
#define P_compat "\xef\xac\x81"
#define P_ascii "hello world 12345"
#define P_dash "\xe2\x80\x94"
#define P_multi "a\xcc\x81\xcc\x80\xcc\x82"
#define P_utf8bad "\xc3\x28"

static int g_fail;
static int checks;

static void check_case(const uint8_t *in, size_t n, int compat,
                       const char *what) {
    static uint8_t whole[65536];
    static uint8_t seg[65536];
    static uint8_t rebuilt[65536];
    nfc_run_t runs[NRUNS];
    checks++;
    size_t nw = compat ? tsimd_nfkc_normalize(in, n, whole, sizeof whole)
                       : tsimd_nfc_normalize(in, n, whole, sizeof whole);
    if (nw == (size_t)-1) {
        fprintf(stderr, "FAIL %s: whole normalize failed\n", what);
        g_fail = 1;
        return;
    }
    size_t nr = compat ? tsimd_nfkc_normalize_runs(in, n, seg, sizeof seg, runs,
                                                   NRUNS)
                       : tsimd_nfc_normalize_runs(in, n, seg, sizeof seg, runs,
                                                  NRUNS);
    if (nr == (size_t)-1) {
        fprintf(stderr, "FAIL %s: runs normalize failed\n", what);
        g_fail = 1;
        return;
    }
    size_t prev = 0, ro = 0;
    for (size_t k = 0; k < nr; k++) {
        if (runs[k].in_off < prev || runs[k].in_off + runs[k].in_len > n) {
            fprintf(stderr, "FAIL %s: run %zu out of order or out of range\n",
                    what, k);
            g_fail = 1;
            return;
        }
        size_t gap = runs[k].in_off - prev;
        memcpy(rebuilt + ro, in + prev, gap);
        ro += gap;
        memcpy(rebuilt + ro, seg + runs[k].out_off, runs[k].out_len);
        ro += runs[k].out_len;
        prev = runs[k].in_off + runs[k].in_len;
    }
    memcpy(rebuilt + ro, in + prev, n - prev);
    ro += n - prev;
    if (ro != nw || memcmp(rebuilt, whole, nw) != 0) {
        fprintf(stderr,
                "FAIL %s: reconstruction %zu bytes != whole %zu bytes\n", what,
                ro, nw);
        g_fail = 1;
        return;
    }
    for (size_t k = 0; k < nr; k++) {
        if (runs[k].out_off + runs[k].out_len > nw) {
            fprintf(stderr, "FAIL %s: run %zu output out of range\n", what, k);
            g_fail = 1;
            return;
        }
    }
}

static size_t mk(uint8_t *dst, const char *parts[], size_t nparts) {
    size_t o = 0;
    for (size_t k = 0; k < nparts; k++) {
        size_t l = strlen(parts[k]);
        memcpy(dst + o, parts[k], l);
        o += l;
    }
    return o;
}

int main(void) {
    static uint8_t in[65536];

    {
        static const char *parts[] = {P_mark};
        check_case(in, mk(in, parts, 1), 0, "single mark");
    }
    {
        static const char *parts[] = {P_ascii, P_mark, P_ascii};
        check_case(in, mk(in, parts, 3), 0, "ascii mark ascii");
    }
    {
        static const char *parts[] = {P_ascii, P_cjk, P_ascii, P_dash};
        check_case(in, mk(in, parts, 4), 0, "ascii cjk");
    }
    {
        static const char *parts[] = {P_hangul};
        check_case(in, mk(in, parts, 1), 0, "hangul jamo");
    }
    {
        static const char *parts[] = {P_compat};
        check_case(in, mk(in, parts, 1), 0, "compat form nfc");
        check_case(in, mk(in, parts, 1), 1, "compat form nfkc");
    }
    {
        static const char *parts[] = {P_multi};
        check_case(in, mk(in, parts, 1), 0, "multiple marks");
    }
    {
        static const char *parts[] = {P_ascii, P_utf8bad, P_ascii};
        check_case(in, mk(in, parts, 3), 0, "invalid utf8");
    }
    {
        static const char *parts[] = {P_mark, P_ascii, P_cjk, P_mark, P_dash,
                                      P_compat};
        check_case(in, mk(in, parts, 6), 0, "mixed nfc");
        check_case(in, mk(in, parts, 6), 1, "mixed nfkc");
    }
    {
        size_t n = 0;
        for (int k = 0; k < 700; k++) {
            memcpy(in + n, P_ascii, 17);
            n += 17;
            memcpy(in + n, P_cjk, 6);
            n += 6;
        }
        check_case(in, n, 0, "long alternating");
    }
    {
        size_t n = 0;
        for (int k = 0; k < 64; k++) {
            memcpy(in + n, P_mark, 3);
            n += 3;
            in[n++] = ' ';
        }
        check_case(in, n, 0, "mark runs");
    }
    {
        static const char *parts[] = {P_ascii};
        check_case(in, mk(in, parts, 1), 0, "ascii only");
    }
    {
        check_case(in, 0, 0, "empty");
    }
    {
        static const char *parts[] = {P_cjk, P_cjk, P_ascii, P_mark};
        size_t n = mk(in, parts, 4);
        nfc_run_t runs[NRUNS];
        static uint8_t seg[65536];
        size_t nr = tsimd_nfc_normalize_runs(in, n, seg, sizeof seg, runs, NRUNS);
        checks++;
        if (nr == (size_t)-1) {
            fprintf(stderr, "FAIL: overflow case failed outright\n");
            g_fail = 1;
        } else {
            size_t nr1 = tsimd_nfc_normalize_runs(in, n, seg, sizeof seg, runs, 1);
            checks++;
            if (nr1 != (size_t)-1) {
                fprintf(stderr, "FAIL: max_runs=1 did not report overflow\n");
                g_fail = 1;
            }
        }
    }

    if (g_fail) {
        fprintf(stderr, "test_nfc_seg: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_nfc_seg: OK (%d checks)\n", checks);
    return 0;
}
