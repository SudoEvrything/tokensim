#include "bpe.h"

#include <stdio.h>
#include <stddef.h>

static int cases[] = {
    22, 22, 22, 21, 22, 18, 19, 19, 12, 22,
};

struct in {
    int width;
    size_t len;
};

static struct in inputs[] = {
    {0, 500ull << 20},
    {6, 0},
    {1, 500ull << 20},
    {6, 500ull << 20},
    {2, 500ull << 20},
    {6, 64ull << 20},
    {4, 64ull << 20},
    {16, 500ull << 20},
    {4, 1000},
    {4, (size_t)2 << 30},
};

int main(void) {
    size_t n = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < n; i++) {
        int got = tsimd_pool_cache_bits(inputs[i].width, inputs[i].len);
        if (got != cases[i]) {
            fprintf(stderr,
                    "FAIL case %zu: width=%d len=%zu got=%d want=%d\n", i,
                    inputs[i].width, inputs[i].len, got, cases[i]);
            return 1;
        }
    }
    printf("pool cache bits: OK\n");
    return 0;
}
