#include "model_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static size_t orig_copy(const uint8_t *in, size_t len, uint8_t *out,
                        size_t cap) {
    size_t n = 0, i = 0;
    while (i < len) {
        if (in[i] == ' ') {
            if (n + 3 > cap) return (size_t)-1;
            out[n++] = 0xE2;
            out[n++] = 0x96;
            out[n++] = 0x81;
            i++;
        } else {
            if (n + 1 > cap) return (size_t)-1;
            out[n++] = in[i++];
        }
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    size_t cap = 1u << 20;
    uint8_t *buf = malloc(cap);
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    uint8_t *out = malloc(cap * 2);
    double best_new = 1e30, best_old = 1e30;
    for (int rep = 0; rep < 40; rep++) {
        double t0 = now_s();
        size_t r = tsimd_metaspace_ex(buf, n, out, cap * 2, 0);
        double t1 = now_s();
        if (r == (size_t)-1) return 2;
        if (t1 - t0 < best_new) best_new = t1 - t0;
        t0 = now_s();
        r = orig_copy(buf, n, out, cap * 2);
        t1 = now_s();
        if (r == (size_t)-1) return 2;
        if (t1 - t0 < best_old) best_old = t1 - t0;
    }
    printf("chunk=%zu KB new=%.3f ms old=%.3f ms ratio=%.2fx\n",
           (unsigned long)(n / 1024), best_new * 1e3, best_old * 1e3,
           best_new / best_old);
    free(buf);
    free(out);
    return 0;
}
