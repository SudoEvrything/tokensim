#include "tokenizer.h"
#include "pretokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef BENCH_DEEPSEEK
#include "../src/pretokenizer/deepseek_masks.c"
#else
#include "../src/pretokenizer/o200k_masks.c"
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    static uint8_t d[64 * 1024 * 1024];
    size_t n = fread(d, 1, sizeof d, f);
    fclose(f);
    size_t wins = 0;
    uint64_t acc = 0;
    for (size_t scan = 0; scan + 72 < 16 * 1024 * 1024; scan += 64) wins++;
    double t0 = now();
    for (int rep = 0; rep < 8; rep++)
        for (size_t scan = 0; scan + 72 < 16 * 1024 * 1024; scan += 64) {
            uint64_t u, b;
            window_usable_bad(d, 16 * 1024 * 1024, scan, &u, &b);
            acc += u;
        }
    double t1 = now();
    double full = (t1 - t0) / (8.0 * wins);
    double t2 = now();
    for (int rep = 0; rep < 8; rep++)
        for (size_t scan = 0; scan + 72 < 16 * 1024 * 1024; scan += 64) {
            amask_t m;
            ascii_masks_64(d + scan, &m);
            acc += m.l;
        }
    double t3 = now();
    double masks = (t3 - t2) / (8.0 * wins);
    double t4 = now();
    for (int rep = 0; rep < 8; rep++)
        for (size_t scan = 0; scan + 72 < 16 * 1024 * 1024; scan += 64) {
            ocar_t cr;
            o200k_carries(d, scan, &cr);
            acc += cr.cls;
        }
    double t5 = now();
    double carr = (t5 - t4) / (8.0 * wins);
    printf("window_usable_bad %.1f ns | ascii_masks %.1f ns | carries %.1f ns | remainder(algebra+uni) %.1f ns | acc %llu\n",
           full * 1e9, masks * 1e9, carr * 1e9, (full - masks - carr) * 1e9, (unsigned long long)acc);
    return 0;
}
