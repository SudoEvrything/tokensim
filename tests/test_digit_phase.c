#include "pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSPANS 4096

typedef struct {
    size_t starts[MAXSPANS];
    size_t lens[MAXSPANS];
    size_t n;
} spans_t;

static int collect(void *ud, size_t off, size_t end) {
    spans_t *s = ud;
    if (s->n >= MAXSPANS) return -1;
    s->starts[s->n] = off;
    s->lens[s->n] = end - off;
    s->n++;
    return 0;
}

int main(void) {
    unsigned char text[1024];
    size_t n = 0;
    text[n++] = 0xDB; text[n++] = 0xA3;
    text[n++] = '4';
    text[n++] = 0xDB; text[n++] = 0xA5;
    for (int i = 0; i < 300; i++) text[n++] = (unsigned char)('0' + i % 10);
    text[n++] = 'x';

    spans_t walker, scalar;
    memset(&walker, 0, sizeof walker);
    memset(&scalar, 0, sizeof scalar);
    if (tsimd_pretokenize_stream(TSIMD_PRETOK_CL100K, text, n, collect,
                                 &walker, NULL) != 0) {
        fprintf(stderr, "walker failed\n");
        return 1;
    }
    tsimd_span_array_t arr;
    memset(&arr, 0, sizeof arr);
    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_CL100K, text, n,
                                     &arr) != 0) {
        fprintf(stderr, "scalar ref failed\n");
        return 1;
    }
    if (arr.count > MAXSPANS) {
        fprintf(stderr, "too many spans for the test\n");
        tsimd_span_array_free(&arr);
        return 1;
    }
    int bad = 0;
    if (walker.n != arr.count) {
        fprintf(stderr, "span count differs: walker %zu vs scalar %zu\n",
                walker.n, arr.count);
        bad = 1;
    } else {
        for (size_t i = 0; i < walker.n; i++) {
            if (walker.starts[i] != arr.spans[i].start ||
                walker.lens[i] !=
                    (size_t)(arr.spans[i].end - arr.spans[i].start)) {
                if (!bad)
                    fprintf(stderr,
                            "span %zu differs: walker [%zu,%zu) vs scalar "
                            "[%u,%u)\n",
                            i, walker.starts[i],
                            walker.starts[i] + walker.lens[i],
                            arr.spans[i].start, arr.spans[i].end);
                bad = 1;
            }
        }
    }
    tsimd_span_array_free(&arr);
    if (bad) {
        fprintf(stderr, "digit-phase walker/scalar divergence\n");
        return 1;
    }
    printf("test_digit_phase: the mixed-width digit run spans match the "
           "scalar reference\n");
    return 0;
}
