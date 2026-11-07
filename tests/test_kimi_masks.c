#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"
#include "../src/pretokenizer/internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIMI TSIMD_SRC_DIR "/models/reference/kimi_k2_tiktoken.model"

typedef struct {
    const char *text;
    size_t len;
} str_case_t;

#define CS(t) {t, sizeof(t) - 1}

static const str_case_t CASES[] = {
    CS("Hello, world! The quick brown fox 123."),
    CS("\xe6\xbc\xa2\xe5\xad\x97\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88"),
    {"a\xe6\xbc\xa2" "b", 5},
    CS("it's don't we'll I'm you're they've he'd"),
    CS("  leading spaces\nnew\ttab"),
    CS("UPPERlower MiXeD 42 1234567 x'y'z"),
    CS("don't caf\xc3\xa9 na\xc3\xafve"),
    CS("\t\tindent x"),
    CS("\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88\xe3\x80\x81\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xae\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88\xe3\x80\x82"),
    CS("\n  x"),
    CS("  \n\n  x"),
    CS("a \n  b"),
    CS("\n\n\n  "),
    CS("x\n   y"),
    CS("a\n\t\nb"),
    CS("--x --a -b -.a -1 - + x"),
    CS("'s 't 're plain'ix"),
    CS("AB ABc ABcd ABcdE aB Ab abcdefgh ABcdEfghij"),
    CS("1 12 123 1234 12345 123456 1234567"),
    CS(" \n\t\r\n mixed \r\n ws \n"),
    CS("a'b it'sfine it'sX it' sx"),
    CS("...!!!???;;"),
    CS("word."),
    CS("trailing spaces   "),
    CS("double  spaces  between  words"),
    CS("MiXeD CaSe WoRdS In A Row"),
    CS("supercalifragilisticexpialidociousandthenalonglowerwordfollowedbyanUpperAtByteEighty and more"),
    CS("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABbbbb"),
    CS("aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa UpperTail"),
    CS("                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          "),
    CS("word ---punct-run-that-is-longer-than-one-window-continuing-and-continuing-and-continuing------  x"),
    CS("the quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."),
    CS("\x11?s"),
    CS("\x11\x11?s ??s a?sb ?s"),
};

static int spans_equal(const tsimd_span_array_t *a,
                       const tsimd_span_array_t *b) {
    return a->count == b->count &&
           (a->count == 0 ||
            memcmp(a->spans, b->spans, a->count * sizeof(tsimd_span_t)) == 0);
}

static void dump_diff(const uint8_t *d, size_t n,
                      const tsimd_span_array_t *a,
                      const tsimd_span_array_t *b) {
    size_t k = 0;
    while (k < a->count && k < b->count && a->spans[k].start == b->spans[k].start &&
           a->spans[k].end == b->spans[k].end)
        k++;
    fprintf(stderr, "  first diff at span %zu of %zu/%zu\n", k,
            (size_t)a->count, (size_t)b->count);
    size_t lo = k > 2 ? k - 2 : 0;
    for (size_t i = lo; i < k + 4 && i < a->count; i++) {
        fprintf(stderr, "  scalar [%u,%u) '", a->spans[i].start,
                a->spans[i].end);
        for (uint32_t q = a->spans[i].start;
             q < a->spans[i].end && q < n; q++)
            fputc(d[q] >= 0x20 && d[q] < 0x7f ? d[q] : '?', stderr);
        fprintf(stderr, "'\n");
    }
    for (size_t i = lo; i < k + 4 && i < b->count; i++) {
        fprintf(stderr, "  masks  [%u,%u) '", b->spans[i].start,
                b->spans[i].end);
        for (uint32_t q = b->spans[i].start;
             q < b->spans[i].end && q < n; q++)
            fputc(d[q] >= 0x20 && d[q] < 0x7f ? d[q] : '?', stderr);
        fprintf(stderr, "'\n");
    }
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_KIMI;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, KIMI, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    tsimd_tokenizer_free(&t);

    int rc = 0;
    size_t ncases = sizeof(CASES) / sizeof(CASES[0]);
    for (size_t c = 0; c < ncases * 2; c++) {
        const uint8_t *d;
        size_t n;
        uint8_t *big = NULL;
        if (c < ncases) {
            d = (const uint8_t *)CASES[c].text;
            n = CASES[c].len;
        } else {
            const str_case_t *cs = &CASES[c - ncases];
            size_t total = 0;
            big = malloc(cs->len * 4 + 4);
            if (!big) return 2;
            for (int r = 0; r < 4; r++) {
                memcpy(big + total, cs->text, cs->len);
                total += cs->len;
                big[total++] = (r == 3) ? '\n' : ' ';
            }
            d = big;
            n = total;
        }
        tsimd_span_array_t a, b;
        memset(&a, 0, sizeof a);
        memset(&b, 0, sizeof b);
        if (tsimd_pretokenize_kimi(d, n, &a) != 0) {
            fprintf(stderr, "case %zu: scalar err\n", c);
            rc = 1;
            continue;
        }
        int mr = tsimd_pretokenize_kimi_masks(d, n, &b);
        if (mr == -2) {
            fprintf(stderr, "case %zu: masks unavailable (stub)\n", c);
            rc = 1;
            tsimd_span_array_free(&a);
            continue;
        }
        if (mr != 0) {
            const char *lbl = c < ncases
                                  ? CASES[c].text
                                  : CASES[c - ncases].text;
            fprintf(stderr, "case %zu (%.24s): masks err\n", c, lbl);
            rc = 1;
            tsimd_span_array_free(&a);
            continue;
        }
        if (!spans_equal(&a, &b)) {
            const char *lbl = c < ncases
                                  ? CASES[c].text
                                  : CASES[c - ncases].text;
            fprintf(stderr, "case %zu (%.24s): SPAN DIFF\n", c, lbl);
            dump_diff(d, n, &a, &b);
            rc = 1;
        }
        tsimd_span_array_t c2;
        memset(&c2, 0, sizeof c2);
        if (tsimd_pretokenize(TSIMD_PRETOK_KIMI, d, n, &c2) != 0) {
            fprintf(stderr, "case %zu: dispatch err\n", c);
            rc = 1;
        } else if (!spans_equal(&a, &c2)) {
            fprintf(stderr, "case %zu (%.24s): DISPATCH SPAN DIFF\n", c,
                    c < ncases ? CASES[c].text : CASES[c - ncases].text);
            dump_diff(d, n, &a, &c2);
            rc = 1;
        }
        tsimd_span_array_free(&c2);
        tsimd_span_array_free(&a);
        tsimd_span_array_free(&b);
        free(big);
    }
    if (rc) return 1;
    printf("test_kimi_masks: OK (%zu cases)\n", ncases * 2);
    return 0;
}
