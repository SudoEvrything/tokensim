#include "pretokenizer.h"
#include "bpe.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int tsimd_pretokenize_deepseek_masks(const uint8_t *data, size_t len,
                                     tsimd_span_array_t *out);

typedef struct {
    const char *bytes;
    size_t len;
    uint32_t spans[8][2];
    size_t n_spans;
} attach_case_t;

static const attach_case_t cases[] = {
    {"a \xcc\x8a" "C", 5, {{0, 1}, {1, 5}}, 2},
    {" \xcc\x8a" "C", 4, {{0, 4}}, 1},
    {"\xcc\x8a" "C", 3, {{0, 3}}, 1},
    {"\xcc\x8a", 2, {{0, 2}}, 1},
    {"\xcc\x8a\xcc\x81" "a", 5, {{0, 5}}, 1},
    {"a \xcc\x8a", 4, {{0, 1}, {1, 4}}, 2},
    {" \xcc\x8a\xcc\x81" "C", 6, {{0, 6}}, 1},
    {"\xcc\x8a a", 4, {{0, 2}, {2, 4}}, 2},
    {"a \xcc\x8a b", 6, {{0, 1}, {1, 4}, {4, 6}}, 3},
    {"a  \xcc\x8a", 5, {{0, 1}, {1, 2}, {2, 5}}, 3},
    {"\t\xcc\x8a" "A", 4, {{0, 4}}, 1},
    {"\x0b\xcc\x8a" "A", 4, {{0, 4}}, 1},
    {"\xc2\xa0\xcc\x8a" "A", 5, {{0, 5}}, 1},
    {"1 \xcc\x8a", 4, {{0, 1}, {1, 4}}, 2},
    {"123 \xcc\x8a" "A", 7, {{0, 3}, {3, 7}}, 2},
    {"\xcc\x8a\xcc\x81", 4, {{0, 4}}, 1},
    {"\xcc\x8a" "1", 3, {{0, 2}, {2, 3}}, 2},
    {" \xcc\x8a" "1", 4, {{0, 3}, {3, 4}}, 2},
    {"a\xcc\x8a b", 5, {{0, 3}, {3, 5}}, 2},
    {"\xe2\x80\x8b" "A", 4, {{0, 4}}, 1},
    {" \xe2\x80\x8b" "A", 5, {{0, 1}, {1, 5}}, 2},
    {"\xe2\x80\x8b\xcc\x8a" "A", 6, {{0, 6}}, 1},
    {"\xcc\x81\xcc\x88\xcc\x8a abc", 10, {{0, 6}, {6, 10}}, 2},
    {"\xc3\xa9 \xcc\x8a" "A", 6, {{0, 2}, {2, 6}}, 2},
    {"\xe2\x80\x94 \xcc\x8a" "A", 7, {{0, 3}, {3, 7}}, 2},
    {" \xc3\xa9", 3, {{0, 3}}, 1},
    {"\xcc\x8a\xc3\xa9" "A", 5, {{0, 5}}, 1},
    {" \xcc\x8a\xc3\xa9" "A", 6, {{0, 6}}, 1},
    {"  \xe4\xb8\xad", 5, {{0, 2}, {2, 5}}, 2},
    {" \xe4\xb8\xad", 4, {{0, 1}, {1, 4}}, 2},
    {"x \xcc\x8a\xcc\x81" "y", 7, {{0, 1}, {1, 7}}, 2},
    {"x \xcc\x8a\xcc\x81 y", 8, {{0, 1}, {1, 6}, {6, 8}}, 3},
    {"  x", 3, {{0, 1}, {1, 3}}, 2},
    {"  1", 3, {{0, 2}, {2, 3}}, 2},
    {"a \xe2\x80\x94", 5, {{0, 1}, {1, 5}}, 2},
    {" \xe2\x80\x94", 4, {{0, 4}}, 1},
    {"\xe2\x80\x8b\xe2\x80\x8b", 6, {{0, 6}}, 1},
    {"\xe2\x80\x8b\xe2\x80\x8b\xe2\x80\x8b", 9, {{0, 9}}, 1},
    {"\xe2\x80\x8b\xe2\x80\x8b" "1", 7, {{0, 6}, {6, 7}}, 2},
    {"a\xe2\x80\x8b\xe2\x80\x8b b", 9, {{0, 1}, {1, 7}, {7, 9}}, 3},
    {"\xe2\x80\x8b" "a\xe2\x80\x8b\xe2\x80\x8b", 10, {{0, 4}, {4, 10}}, 2},
    {"\x00\x00", 2, {{0, 2}}, 1},
    {"\x00\x00" "a", 3, {{0, 1}, {1, 3}}, 2},
    {"a\x00\x00", 3, {{0, 1}, {1, 3}}, 2},
    {"\xe2\x80\x8b\x00\xe2\x80\x8b", 7, {{0, 7}}, 1},
};

static int check_one(const attach_case_t *c, size_t idx, int masks) {
    tsimd_span_array_t a;
    tsimd_span_array_init(&a);
    int rc = masks ? tsimd_pretokenize_deepseek_masks((const uint8_t *)c->bytes,
                                                      c->len, &a)
                   : tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK,
                                                  (const uint8_t *)c->bytes,
                                                  c->len, &a);
    if (rc != 0) {
        fprintf(stderr, "FAIL case %zu (%s): pretokenize error\n", idx,
                masks ? "masks" : "walker");
        tsimd_span_array_free(&a);
        return 1;
    }
    if (a.count != c->n_spans) {
        fprintf(stderr, "FAIL case %zu (%s): span count %zu vs %zu\n", idx,
                masks ? "masks" : "walker", a.count, c->n_spans);
        for (size_t i = 0; i < a.count; i++)
            fprintf(stderr, "  got [%u,%u)\n", (unsigned)a.spans[i].start,
                    (unsigned)a.spans[i].end);
        tsimd_span_array_free(&a);
        return 1;
    }
    for (size_t i = 0; i < c->n_spans; i++) {
        if (a.spans[i].start != c->spans[i][0] ||
            a.spans[i].end != c->spans[i][1]) {
            fprintf(stderr,
                    "FAIL case %zu (%s): span %zu [%u,%u) expected [%u,%u)\n",
                    idx, masks ? "masks" : "walker", i,
                    (unsigned)a.spans[i].start, (unsigned)a.spans[i].end,
                    (unsigned)c->spans[i][0], (unsigned)c->spans[i][1]);
            tsimd_span_array_free(&a);
            return 1;
        }
    }
    tsimd_span_array_free(&a);
    return 0;
}

int main(void) {
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (check_one(&cases[i], i, 0)) return 1;
        if (check_one(&cases[i], i, 1)) return 1;
    }
    printf("deepseek attach: OK (%zu cases)\n",
           sizeof(cases) / sizeof(cases[0]));
    return 0;
}
