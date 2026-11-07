#include "model_internal.h"
#include <stddef.h>
#include <stdio.h>
int main(void) {
    uint8_t in[2] = {0xff, 0xff};
    uint8_t out[64];
    size_t r = tsimd_metaspace_ex(in, 2, out, 64, 0);
    printf("r=%zu bytes:", r);
    for (size_t i = 0; i < r; i++) printf(" %02x", out[i]);
    printf("\n");
    return 0;
}
