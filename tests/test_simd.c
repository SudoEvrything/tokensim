#include "pretokenizer.h"
#include "simd.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int spans_equal(const tsimd_span_array_t *a,
                       const tsimd_span_array_t *b) {
    if (a->count != b->count) return 0;
    return a->count == 0 ||
           memcmp(a->spans, b->spans, a->count * sizeof(tsimd_span_t)) == 0;
}

static int check_input(tsimd_pretok_scheme_t scheme, const uint8_t *data,
                       size_t len, const char *label) {
    tsimd_span_array_t ref, got;
    tsimd_span_array_init(&ref);
    tsimd_span_array_init(&got);
    if (tsimd_pretokenize_scalar_ref(scheme, data, len, &ref) != 0) {
        fprintf(stderr, "reference failed: %s\n", label);
        {
            char path[128];
            snprintf(path, sizeof(path), "/tmp/tsimd_reffail_%s.bin",
                     scheme == TSIMD_PRETOK_GPT2 ? "gpt2"
                     : scheme == TSIMD_PRETOK_CL100K ? "cl100k" : "o200k");
            FILE *f = fopen(path, "wb");
            if (f && len) fwrite(data, 1, len, f);
            if (f) fclose(f);
            fprintf(stderr, "  input dumped to %s\n", path);
        }
        return 1;
    }
    static const tsimd_simd_level_t levels[] = {
        TSIMD_SIMD_SCALAR, TSIMD_SIMD_SSE2, TSIMD_SIMD_AVX2, TSIMD_SIMD_AVX512,
    };
    static const char *names[] = {"scalar-block", "sse2", "avx2", "avx512"};
    int fail = 0;
    for (size_t li = 0; li < 4; li++) {
        if (!tsimd_simd_level_supported(levels[li])) continue;
        tsimd_simd_set_forced((int)levels[li]);
        tsimd_span_array_free(&got);
        tsimd_span_array_init(&got);
        if (tsimd_pretokenize(scheme, data, len, &got) != 0 ||
            !spans_equal(&ref, &got)) {
            fprintf(stderr,
                    "%s MISMATCH scheme=%s level=%s len=%zu: %zu spans (ref %zu)\n",
                    label,
                    scheme == TSIMD_PRETOK_GPT2 ? "gpt2"
                    : scheme == TSIMD_PRETOK_CL100K ? "cl100k" : "o200k",
                    names[li], len, got.count, ref.count);
            {

                char path[128];
                snprintf(path, sizeof(path), "/tmp/tsimd_fail_%s_%s_%s.bin",
                         scheme == TSIMD_PRETOK_GPT2 ? "gpt2"
                         : scheme == TSIMD_PRETOK_CL100K ? "cl100k" : "o200k",
                         names[li], label);
                FILE *f = fopen(path, "wb");
                if (f && len) fwrite(data, 1, len, f);
                if (f) fclose(f);
                fprintf(stderr, "  input dumped to %s\n", path);
            }
            size_t n = got.count < ref.count ? got.count : ref.count;
            for (size_t i = 0; i < n; i++) {
                if (got.spans[i].start != ref.spans[i].start ||
                    got.spans[i].end != ref.spans[i].end) {
                    fprintf(stderr, "  first diff at span %zu: "
                                    "got [%u,%u) ref [%u,%u)\n", i,
                            got.spans[i].start, got.spans[i].end,
                            ref.spans[i].start, ref.spans[i].end);
                    break;
                }
            }
            fail = 1;
        }
    }
    tsimd_simd_set_forced(-1);
    tsimd_span_array_free(&ref);
    tsimd_span_array_free(&got);
    return fail;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void gen(uint8_t *buf, size_t n, int mode) {
    for (size_t i = 0; i < n; i++) {
        uint64_t r = rnd();
        uint8_t b;
        switch (mode) {
            case 0: b = (uint8_t)(r >> 32); break;
            case 1:
                b = (uint8_t)" \n \t.,;:'\"abcdefgxyzABC0129?!-()"[(r >> 33) % 30];
                break;
            case 2: b = (uint8_t)" \t\n\r \n \t"[(r >> 33) % 9]; break;
            case 3:
                b = (uint8_t)"a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\xFF\x80\xC0\xED"[(r >> 33) % 13];
                break;
            case 4: b = (uint8_t)"'stlveS'TL dmRE"[(r >> 33) % 15]; break;
            case 5:
                b = (uint8_t)"0123456789abc x"[(r >> 33) % 15];
                break;
            case 6:
                b = (uint8_t)" \r\n \t\nx"[(r >> 33) % 8];
                break;
            default: b = (uint8_t)r; break;
        }
        buf[i] = b;
    }
}

static int failing_sink(void *ud, size_t s, size_t e) {
    (void)s; (void)e;
    (*(int *)ud)++;
    return 1;
}

static int expect_spans(tsimd_pretok_scheme_t scheme, const uint8_t *data,
                        size_t len, const uint32_t *want, size_t wcnt) {
    tsimd_span_array_t ref, got;
    tsimd_span_array_init(&ref);
    tsimd_span_array_init(&got);
    if (tsimd_pretokenize_scalar_ref(scheme, data, len, &ref) != 0 ||
        ref.count != wcnt) {
        fprintf(stderr, "expect_spans: ref count %zu want %zu\n", ref.count,
                wcnt);
        tsimd_span_array_free(&ref);
        tsimd_span_array_free(&got);
        return 1;
    }
    for (size_t i = 0; i < wcnt; i++) {
        if (ref.spans[i].start != want[2 * i] ||
            ref.spans[i].end != want[2 * i + 1]) {
            fprintf(stderr, "expect_spans: ref span %zu [%u,%u) want [%u,%u)\n",
                    i, ref.spans[i].start, ref.spans[i].end, want[2 * i],
                    want[2 * i + 1]);
            tsimd_span_array_free(&ref);
            tsimd_span_array_free(&got);
            return 1;
        }
    }
    int fail = 0;
    for (size_t li = 0; li < 4; li++) {
        if (!tsimd_simd_level_supported(li == 0 ? TSIMD_SIMD_SCALAR
                                       : li == 1 ? TSIMD_SIMD_SSE2
                                       : li == 2 ? TSIMD_SIMD_AVX2
                                                 : TSIMD_SIMD_AVX512))
            continue;
        tsimd_span_array_free(&got);
        tsimd_span_array_init(&got);
        if (tsimd_pretokenize(scheme, data, len, &got) != 0 ||
            got.count != wcnt) {
            fprintf(stderr, "expect_spans: simd count %zu want %zu\n",
                    got.count, wcnt);
            fail = 1;
            break;
        }
        for (size_t i = 0; i < wcnt; i++) {
            if (got.spans[i].start != want[2 * i] ||
                got.spans[i].end != want[2 * i + 1]) {
                fprintf(stderr, "expect_spans: simd span %zu [%u,%u) "
                                "want [%u,%u)\n", i, got.spans[i].start,
                        got.spans[i].end, want[2 * i], want[2 * i + 1]);
                fail = 1;
                break;
            }
        }
        if (fail) break;
    }
    tsimd_span_array_free(&ref);
    tsimd_span_array_free(&got);
    return fail;
}

int main(void) {

    tsimd_simd_set_forced(-1);
    if (tsimd_simd_level() == TSIMD_SIMD_AVX512) {
        fprintf(stderr,
                "FAIL: the auto path must cap at AVX2 (the AVX-512 blocks are "
                "untested; see docs/207)\n");
        return 1;
    }

    {
        static const char txt[] =
            "\xC3\xA9 non-ascii start forces the scalar fallback path";
        int calls = 0;
        TEST_ASSERT(tsimd_pretokenize_stream(TSIMD_PRETOK_GPT2, (const uint8_t *)txt,
                                             sizeof(txt) - 1, failing_sink,
                                             &calls, NULL) == -1);
        TEST_ASSERT(calls == 1);
    }

    static const char *cases[] = {
        "",
        "a",
        "hello world",
        "1234",
        "1'234'567'890",
        "1234567890123456789012345678901234567890123456789012345678901234567890",
        "abc 1234 def 5678 90",
        "1'2 3'4'5 6''78'9 0'1'2'3x don'ts DON'TS 1's 2'tx",
        "x/y x//y x" "\n" "/y x./" "\n" "y /y /" "\n" "y a." "\n" "/b /x //x" "\n" "/  path" "\n" "/to/file",
        "iPhone HELLOworld camelCaseSnake_case don't I'm we'll 'll 'LL",
        "foo.bar foo..bar ((s x((s 'x ''x '''x  'q x'q x''q",
        "aBcDe ABcDEf ABcDON'Ts x's 'ss ''s x''s 1's 1'tx don'ts DON'TS",
        "tab" "\t" "sep " "\t" "tab" "\t" "!! " "\t" "'t " "\t" "tt",
        "!!!???...,,,;;;:::",
        "e" "\xcc" "\x81" " combining",
        "\xc3" "\xa9" " w" "\xc3" "\xb6" "rld " "\xe4" "\xbd" "\xa0" "\xe5" "\xa5" "\xbd",
        "\xef" "\xbc" "\x91" "\xef" "\xbc" "\x90" "\xe6" "\x9c" "\x88" " " "\xef" "\xbc" "\x91" "\xef" "\xbc" "\x90" " 12" "\xef" "\xbc" "\x93" "34",
        "1'234'567 " "\xd9" "\xa1" "\xd9" "\xa2" "\xd9" "\xa3" " 4" "\xd9" "\xa5" " 6",
        "\xef" "\xbc" "\x90" "\xef" "\xbc" "\x91" "23456xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "\xd9" "\xa1" "23456789xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "\xd9" "\xa1" "\xd9" "\xa2" "\xd9" "\xa3" "4567890123456789xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "A" "\xcc" "\x80" "grave",
        "\xe2" "\x84" "\xa6" "ohm",
        "\xe1" "\x86" "\xa8" "jamo",
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint8_t *d = (const uint8_t *)cases[ci];
        size_t n = strlen(cases[ci]);
        TEST_ASSERT(check_input(TSIMD_PRETOK_GPT2, d, n, "hand") == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, d, n, "hand") == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, d, n, "hand") == 0);
    }

    {
        uint8_t buf[384];
        uint32_t want[16];
        size_t wn;
        for (size_t pad = 56; pad < 64; pad++) {
            wn = 0;
            memset(buf, 'q', pad);
            buf[pad] = '/';
            buf[pad + 1] = (uint8_t)0xa0;
            buf[pad + 2] = '&';
            buf[pad + 3] = 'x';
            buf[pad + 4] = 'x';
            buf[pad + 5] = ' ';
            memset(buf + pad + 6, 'z', 70);
            size_t n = pad + 76;
            if (pad == 62) {
                want[0] = 0; want[1] = 62;
                want[2] = 62; want[3] = 65;
                want[4] = 65; want[5] = 67;
                want[6] = 67; want[7] = 138;
                TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, n, want,
                                         4) == 0);
            }
            TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, n,
                                    "inv-seam") == 0);
            TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, n,
                                    "inv-seam") == 0);
        }
        memset(buf, 'q', 62);
        buf[62] = '/';
        buf[63] = '/';
        buf[64] = '&';
        buf[65] = 'x';
        buf[66] = 'x';
        buf[67] = ' ';
        memset(buf + 68, 'z', 70);
        want[0] = 0; want[1] = 62;
        want[2] = 62; want[3] = 65;
        want[4] = 65; want[5] = 67;
        want[6] = 67; want[7] = 138;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 138, want, 4) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 138, "inv-seam") == 0);
        memset(buf, 'q', 61);
        buf[61] = (uint8_t)0xe2;
        buf[62] = (uint8_t)0x82;
        buf[63] = (uint8_t)0xac;
        buf[64] = '&';
        buf[65] = 'x';
        buf[66] = 'x';
        buf[67] = ' ';
        memset(buf + 68, 'z', 70);
        want[0] = 0; want[1] = 61;
        want[2] = 61; want[3] = 65;
        want[4] = 65; want[5] = 67;
        want[6] = 67; want[7] = 138;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 138, want, 4) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 138, "inv-seam") == 0);
        memset(buf, 'q', 60);
        buf[60] = (uint8_t)0xe2;
        buf[61] = (uint8_t)0x82;
        buf[62] = (uint8_t)0xac;
        buf[63] = (uint8_t)0xa0;
        buf[64] = '&';
        buf[65] = 'x';
        buf[66] = 'x';
        buf[67] = ' ';
        memset(buf + 68, 'z', 70);
        want[0] = 0; want[1] = 60;
        want[2] = 60; want[3] = 65;
        want[4] = 65; want[5] = 67;
        want[6] = 67; want[7] = 138;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 138, want, 4) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 138, "inv-seam") == 0);
        memset(buf, 'q', 63);
        buf[63] = (uint8_t)0xa0;
        buf[64] = 'x';
        buf[65] = 'x';
        buf[66] = ' ';
        memset(buf + 67, 'z', 70);
        want[0] = 0; want[1] = 63;
        want[2] = 63; want[3] = 66;
        want[4] = 66; want[5] = 137;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 137, want, 3) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 137, "inv-seam") == 0);
        memset(buf, 'q', 61);
        buf[61] = '1';
        buf[62] = '2';
        buf[63] = '3';
        buf[64] = '&';
        buf[65] = 'x';
        buf[66] = 'x';
        buf[67] = ' ';
        memset(buf + 68, 'z', 70);
        want[0] = 0; want[1] = 61;
        want[2] = 61; want[3] = 64;
        want[4] = 64; want[5] = 67;
        want[6] = 67; want[7] = 138;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 138, want, 4) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 138, "inv-seam") == 0);
        memset(buf, 'q', 60);
        buf[60] = (uint8_t)0xe2;
        buf[61] = (uint8_t)0x82;
        buf[62] = (uint8_t)0xac;
        buf[63] = '\n';
        buf[64] = '\n';
        buf[65] = 'x';
        buf[66] = 'x';
        buf[67] = ' ';
        memset(buf + 68, 'z', 70);
        TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, 138, "inv-seam") == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 138, "inv-seam") == 0);
        memset(buf, '.', 88);
        buf[88] = 'G';
        buf[89] = (uint8_t)0xa2;
        buf[90] = '\'';
        buf[91] = 's';
        buf[92] = 'G';
        buf[93] = '1';
        memset(buf + 94, 'G', 10);
        buf[104] = ' ';
        memset(buf + 105, 'z', 70);
        want[0] = 0; want[1] = 88;
        want[2] = 88; want[3] = 89;
        want[4] = 89; want[5] = 91;
        want[6] = 91; want[7] = 93;
        want[8] = 93; want[9] = 94;
        want[10] = 94; want[11] = 104;
        want[12] = 104; want[13] = 175;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_GPT2, buf, 175, want, 7) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, 175, "inv-seam") == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 175, "inv-seam") == 0);
        memset(buf, 'q', 62);
        buf[62] = ' ';
        buf[63] = ' ';
        buf[64] = (uint8_t)0xa0;
        memset(buf + 65, ' ', 11);
        buf[76] = 0;
        buf[77] = 0;
        buf[78] = 0;
        buf[79] = 0;
        buf[80] = 0;
        memset(buf + 81, 'z', 20);
        want[0] = 0; want[1] = 62;
        want[2] = 62; want[3] = 63;
        want[4] = 63; want[5] = 65;
        want[6] = 65; want[7] = 75;
        want[8] = 75; want[9] = 81;
        want[10] = 81; want[11] = 101;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_GPT2, buf, 101, want, 6) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, 101, "inv-seam") == 0);
        memset(buf, 'q', 62);
        buf[62] = ' ';
        buf[63] = ' ';
        buf[64] = (uint8_t)0xc2;
        buf[65] = (uint8_t)0xa0;
        memset(buf + 66, ' ', 10);
        buf[76] = 0;
        buf[77] = 0;
        buf[78] = 0;
        buf[79] = 0;
        buf[80] = 0;
        memset(buf + 81, 'z', 20);
        want[0] = 0; want[1] = 62;
        want[2] = 62; want[3] = 75;
        want[4] = 75; want[5] = 81;
        want[6] = 81; want[7] = 101;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_GPT2, buf, 101, want, 4) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, 101, "inv-seam") == 0);
        memset(buf, 'q', 10);
        memset(buf + 10, '=', 130);
        memset(buf + 140, '\r', 125);
        memset(buf + 265, 0x0c, 20);
        buf[285] = '5';
        memset(buf + 286, 'z', 20);
        want[0] = 0; want[1] = 10;
        want[2] = 10; want[3] = 265;
        want[4] = 265; want[5] = 284;
        want[6] = 284; want[7] = 285;
        want[8] = 285; want[9] = 286;
        want[10] = 286; want[11] = 306;
        TEST_ASSERT(expect_spans(TSIMD_PRETOK_CL100K, buf, 306, want, 6) == 0);
        TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, 306, "inv-seam") == 0);
    }

    {
        uint8_t buf[256];
        for (size_t n = 0; n <= 200; n++) {
            gen(buf, n, 1);
            TEST_ASSERT(check_input(TSIMD_PRETOK_GPT2, buf, n, "len-sweep") == 0);
            TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, n, "len-sweep") == 0);
        }
    }

    {
        uint8_t buf[512];
        for (int mode = 0; mode <= 6; mode++) {
            for (int iter = 0; iter < 60; iter++) {
                size_t n = 1 + (size_t)(rnd() % 500);
                gen(buf, n, mode);
                TEST_ASSERT(check_input(TSIMD_PRETOK_GPT2, buf, n, "fuzz") == 0);
                TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, buf, n, "fuzz") == 0);
                TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, buf, n, "fuzz") == 0);
            }
        }

        for (int iter = 0; iter < 8; iter++) {
            size_t n = 5000 + (size_t)(rnd() % 40000);
            uint8_t *big = malloc(n);
            TEST_ASSERT(big != NULL);
            gen(big, n, iter % 7);
            TEST_ASSERT(check_input(TSIMD_PRETOK_GPT2, big, n, "big") == 0);
            TEST_ASSERT(check_input(TSIMD_PRETOK_CL100K, big, n, "big") == 0);
            TEST_ASSERT(check_input(TSIMD_PRETOK_O200K, big, n, "big") == 0);
            free(big);
        }
    }

    printf("test_simd: OK\n");
    return 0;
}
