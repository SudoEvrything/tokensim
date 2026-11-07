#include "pretokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_case_bytes(tsimd_pretok_scheme_t scheme, const uint8_t *data,
                          size_t len, const uint8_t **expected,
                          const size_t *elen, size_t n) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    int rc = tsimd_pretokenize(scheme, data, len, &spans);
    if (rc != 0) {
        fprintf(stderr, "pretokenize failed\n");
        return 1;
    }

    size_t pos = 0;
    for (size_t i = 0; i < spans.count; i++) {
        if (spans.spans[i].start != pos || spans.spans[i].end <= spans.spans[i].start) {
            fprintf(stderr, "coverage gap at span %zu\n", i);
            return 1;
        }
        pos = spans.spans[i].end;
    }
    if (pos != len) {
        fprintf(stderr, "spans end at %zu, input len %zu\n", pos, len);
        return 1;
    }

    int fail = 0;
    if (spans.count != n) {
        fprintf(stderr, "got %zu pretokens, want %zu\n", spans.count, n);
        fail = 1;
    } else {
        for (size_t i = 0; i < n && !fail; i++) {
            size_t slen = spans.spans[i].end - spans.spans[i].start;
            if (elen[i] != slen ||
                memcmp(data + spans.spans[i].start, expected[i], slen) != 0) {
                fprintf(stderr, "pretoken %zu mismatch (got %zu bytes, want %zu)\n",
                        i, slen, elen[i]);
                fail = 1;
            }
        }
    }
    tsimd_span_array_free(&spans);
    return fail;
}

static int run_case(tsimd_pretok_scheme_t scheme, const char *input,
                    const char **expected, size_t n) {
    const uint8_t *data = (const uint8_t *)input;
    size_t len = strlen(input);
    const uint8_t *exp[512];
    size_t elen[512];
    for (size_t i = 0; i < n && i < 512; i++) {
        exp[i] = (const uint8_t *)expected[i];
        elen[i] = strlen(expected[i]);
    }
    return run_case_bytes(scheme, data, len, exp, elen, n);
}

static int run_raw_coverage(tsimd_pretok_scheme_t scheme, const uint8_t *data,
                            size_t len) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    if (tsimd_pretokenize(scheme, data, len, &spans) != 0) return 1;
    size_t pos = 0;
    for (size_t i = 0; i < spans.count; i++) {
        if (spans.spans[i].start != pos) {
            tsimd_span_array_free(&spans);
            return 1;
        }
        pos = spans.spans[i].end;
    }
    int ok = (pos == len);
    tsimd_span_array_free(&spans);
    return ok ? 0 : 1;
}

static int run_fixtures(tsimd_pretok_scheme_t scheme, const char *dir,
                        const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    size_t flen;
    char *file = tsimd_test_read_file(path, &flen);
    if (!file) {
        printf("(fixture %s not present, skipping)\n", name);
        return 0;
    }
    int failures = 0, cases = 0;
    char *line = file;
    while (line < file + flen) {
        char *nl = strchr(line, '\n');
        if (!nl) break;
        *nl = 0;
        char *tab = strchr(line, '\t');
        if (!tab) { line = nl + 1; continue; }
        *tab = 0;
        uint8_t input[4096];
        size_t ilen;
        if (tsimd_test_unescape(line, strlen(line), input, sizeof(input),
                               &ilen) != 0) {
            failures++;
            line = nl + 1;
            continue;
        }

        static uint8_t arena[512][1024];
        const uint8_t *expected[512];
        size_t elen[512];
        size_t n = 0;
        char *p = tab + 1;
        int bad = 0;
        while (*p && n < 512) {
            char *t = strchr(p, '\t');
            if (t) *t = 0;
            if (*p) {
                if (tsimd_test_unescape(p, strlen(p), arena[n], 1024,
                                       &elen[n]) != 0) {
                    bad = 1;
                    break;
                }
                expected[n] = arena[n];
                n++;
            }
            if (!t) break;
            p = t + 1;
        }
        cases++;
        if (bad || run_case_bytes(scheme, input, ilen, expected, elen, n) != 0)
            failures++;
        line = nl + 1;
    }
    free(file);
    printf("fixtures %s: %d cases, %d failures\n", name, cases, failures);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *fixtures = argc > 1 ? argv[1] : "tests/fixtures";

    {
        static const char *c1[] = {"1", "2", "3", "4", "5", "6", "7",
                                   "8", "9", "0"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_QWEN25, "1234567890", c1, 10) == 0);
        static const char *c2[] = {"abc", "1", "2", "3", "def"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_QWEN25, "abc123def", c2, 5) == 0);

        static const char *c3[] = {"1", "ab", " c"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_QWEN25, "1ab c", c3, 3) == 0);
        static const char *c4[] = {"I", "'m", " sure"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_QWEN25, "I'm sure", c4, 3) == 0);
        static const char *d1[] = {"123", "456", "789", "0"};

        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "1234567890", d1, 4) == 0);
        TEST_ASSERT(run_case(TSIMD_PRETOK_QWEN25, "1234567890", c1, 10) == 0);
    }

    {
        static const char *c1[] = {"1", "2", "3"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2D, "123", c1, 3) == 0);
        static const char *c2[] = {" ", "1", "2"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2D, " 12", c2, 3) == 0);
        static const char *c3[] = {"hello", " ", "1", "world"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2D, "hello 1world", c3, 4) == 0);
        static const char *c4[] = {"I", "'m", " sure"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2D, "I'm sure", c4, 3) == 0);
    }

    {
        static const char *c1[] = {"hello", " world"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "hello world", c1, 2) == 0);
        static const char *c2[] = {"I", "'m", " sure", " they", "'ll", "'ve",
                                   " said"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2,
                             "I'm sure they'll've said", c2, 7) == 0);

        static const char *c3[] = {" ", " double", " ", " spaces"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "  double  spaces", c3, 4) == 0);
        static const char *c4[] = {"1234", " 56"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "1234 56", c4, 2) == 0);
        static const char *c5[] = {"Hello", ",", " WORLD", "!!"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "Hello, WORLD!!", c5, 4) == 0);
        static const char *c6[] = {"line", "1", "\n", "line", "2"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "line1\nline2", c6, 5) == 0);

        static const char *c7[] = {"  \n ", " x"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "  \n  x", c7, 2) == 0);
        static const char *c8[] = {"héllo", " wörld"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "héllo wörld", c8, 2) == 0);
        static const char *c9[] = {"你好世界"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "你好世界", c9, 1) == 0);

        static const char *c10[] = {"DON", "'", "T"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "DON'T", c10, 3) == 0);

        static const char *c11[] = {"word", "  "};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "word  ", c11, 2) == 0);

        static const char *c12[] = {"'", "x"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "'x", c12, 2) == 0);

        static const char *c13[] = {"e", "\xCC\x81"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "e\xCC\x81", c13, 2) == 0);

        static const char *c14[] = {"\t", " x"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_GPT2, "\t x", c14, 2) == 0);
    }

    {
        static const char *c1[] = {"hello", " world"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "hello world", c1, 2) == 0);
        static const char *c2[] = {"I", "'m", " sure", " they", "'ll", "'ve",
                                   " said"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K,
                             "I'm sure they'll've said", c2, 7) == 0);

        static const char *c3[] = {"DON", "'T"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "DON'T", c3, 2) == 0);

        static const char *c4[] = {"123", "4", " ", "56"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "1234 56", c4, 4) == 0);
        static const char *c5[] = {"Hello", ",", " WORLD", "!!"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "Hello, WORLD!!", c5, 4) == 0);

        static const char *c6[] = {"!\n"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "!\n", c6, 1) == 0);

        static const char *c7[] = {"  \n", " ", " x"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "  \n  x", c7, 3) == 0);

        static const char *c8[] = {"(hello"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "(hello", c8, 1) == 0);
        static const char *c9[] = {"\tabc"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "\tabc", c9, 1) == 0);

        static const char *c10[] = {"\xC2\xA0" "abc"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "\xC2\xA0" "abc", c10, 1) == 0);
        static const char *c11[] = {" ", " double", " ", " spaces"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "  double  spaces", c11, 4) == 0);
        static const char *c12[] = {"   "};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "   ", c12, 1) == 0);

        static const char *c13[] = {"e", "\xCC\x81"};
        TEST_ASSERT(run_case(TSIMD_PRETOK_CL100K, "e\xCC\x81", c13, 2) == 0);
    }

    {
        tsimd_span_array_t spans;
        tsimd_span_array_init(&spans);
        const char *s = "x  \n\n  y";
        TEST_ASSERT(tsimd_pretokenize(TSIMD_PRETOK_CL100K, (const uint8_t *)s,
                                     strlen(s), &spans) == 0);
        const char *want[] = {"x", "  \n\n", " ", " y"};
        TEST_ASSERT_EQ(spans.count, 4);
        for (int i = 0; i < 4; i++) {
            size_t wl = strlen(want[i]);
            TEST_ASSERT(spans.spans[i].end - spans.spans[i].start == wl);
            TEST_ASSERT(memcmp(s + spans.spans[i].start, want[i], wl) == 0);
        }
        tsimd_span_array_free(&spans);
    }

    {
        const uint8_t junk[] = {0x00, 0x01, 0xC3, 0x28, 0xFF, 0xFE, 0x80,
                                0x41, 0xF0, 0x9F, 0x9A, 0x20, 0x7F};
        TEST_ASSERT(run_raw_coverage(TSIMD_PRETOK_GPT2, junk, sizeof(junk)) == 0);
        TEST_ASSERT(run_raw_coverage(TSIMD_PRETOK_CL100K, junk, sizeof(junk)) == 0);
        TEST_ASSERT(run_raw_coverage(TSIMD_PRETOK_GPT2, NULL, 0) == 0);
        TEST_ASSERT(run_raw_coverage(TSIMD_PRETOK_CL100K, NULL, 0) == 0);
    }

    TEST_ASSERT(run_fixtures(TSIMD_PRETOK_GPT2, fixtures,
                             "golden_pretokens_gpt2.tsv") == 0);
    TEST_ASSERT(run_fixtures(TSIMD_PRETOK_CL100K, fixtures,
                             "golden_pretokens_cl100k.tsv") == 0);

    {
        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "iPhone",
                             (const char *[]){"i", "Phone"}, 2) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "don'ts",
                             (const char *[]){"don't", "s"}, 2) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "x's",
                             (const char *[]){"x's"}, 1) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "HELLOworld",
                             (const char *[]){"HELLOworld"}, 1) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "abc1234",
                             (const char *[]){"abc", "123", "4"}, 3) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "ABcDEf",
                             (const char *[]){"ABc", "DEf"}, 2) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "''s",
                             (const char *[]){"''", "s"}, 2) == 0);

        TEST_ASSERT(run_case(TSIMD_PRETOK_O200K, "ABcDON'Ts",
                             (const char *[]){"ABc", "DON'T", "s"}, 3) == 0);

        {
            const uint8_t in[] = " \xE6\xB8\xA1\xE3\x82\x8B aB";
            size_t in_len = sizeof(in) - 1;
            const uint8_t *e9[] = {(const uint8_t *)" \xE6\xB8\xA1\xE3\x82\x8B",
                                   (const uint8_t *)" a",
                                   (const uint8_t *)"B"};
            const size_t l9[] = {7, 2, 1};
            TEST_ASSERT(run_case_bytes(TSIMD_PRETOK_O200K, in, in_len,
                                       e9, l9, 3) == 0);
        }
    }

    printf("test_pretokenizer: OK\n");
    return 0;
}
