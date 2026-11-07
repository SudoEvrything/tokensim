#include "tokenizer.h"
#include "util.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_contract(const size_t *w, size_t n, int width,
                          const char *what) {
    size_t *lo = malloc((size_t)width * sizeof(size_t));
    size_t *hi = malloc((size_t)width * sizeof(size_t));
    size_t *lo2 = malloc((size_t)width * sizeof(size_t));
    size_t *hi2 = malloc((size_t)width * sizeof(size_t));
    if (!lo || !hi || !lo2 || !hi2) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    int bal = tsimd_par_split(w, n, width, lo, hi);
    int bal2 = tsimd_par_split(w, n, width, lo2, hi2);
    int fail = 0;
    if (bal != bal2 || memcmp(lo, lo2, (size_t)width * sizeof(size_t)) ||
        memcmp(hi, hi2, (size_t)width * sizeof(size_t))) {
        fprintf(stderr, "%s: not deterministic\n", what);
        fail = 1;
    }
    if (width > 0) {
        if (lo[0] != 0 || hi[width - 1] != n) {
            fprintf(stderr, "%s: ends wrong: lo[0]=%zu hi[%d]=%zu n=%zu\n",
                    what, lo[0], width - 1, hi[width - 1], n);
            fail = 1;
        }
        for (int k = 0; k < width; k++) {
            if (lo[k] > hi[k] || hi[k] > n) {
                fprintf(stderr, "%s: range %d bad: [%zu,%zu) n=%zu\n",
                        what, k, lo[k], hi[k], n);
                fail = 1;
                break;
            }
            if (k > 0 && lo[k] != hi[k - 1]) {
                fprintf(stderr, "%s: range %d not contiguous: lo=%zu "
                        "prev hi=%zu\n", what, k, lo[k], hi[k - 1]);
                fail = 1;
                break;
            }
        }
    }

    if (!fail && n > 0) {
        unsigned char *seen = calloc(n, 1);
        if (!seen) fail = 1;
        else {
            for (int k = 0; k < width && !fail; k++)
                for (size_t i = lo[k]; i < hi[k]; i++) {
                    if (seen[i]) {
                        fprintf(stderr, "%s: job %zu in two ranges\n",
                                what, i);
                        fail = 1;
                        break;
                    }
                    seen[i] = 1;
                }
            for (size_t i = 0; i < n && !fail; i++)
                if (!seen[i]) {
                    fprintf(stderr, "%s: job %zu in no range\n", what, i);
                    fail = 1;
                }
            free(seen);
        }
    }

    if (!fail && bal && n > 0 && width > 0) {
        unsigned long long total = 0, mx = 0;
        for (size_t i = 0; i < n; i++) total += (w[i] ? w[i] : 1);
        for (int k = 0; k < width; k++) {
            unsigned long long r = 0;
            for (size_t i = lo[k]; i < hi[k]; i++) r += (w[i] ? w[i] : 1);
            if (r > mx) mx = r;
        }
        if (mx * (unsigned long long)width > 2ull * total) {
            fprintf(stderr, "%s: balanced but range holds %llu of %llu "
                    "(width %d)\n", what, mx, total, width);
            fail = 1;
        }
    }
    free(lo);
    free(hi);
    free(lo2);
    free(hi2);
    return fail;
}

static int test_split_shapes(void) {
    enum { MAXW = 64 };
    size_t w[MAXW];
    static const size_t NS[] = {0, 1, 2, 3, 7, 8, 64, 1000};
    static const int WS[] = {1, 2, 3, 4, 6, 10, 16, 64};

    size_t lo[MAXW], hi[MAXW];
    for (size_t i = 0; i < 64; i++) w[i] = 7;
    if (!tsimd_par_split(w, 64, 8, lo, hi)) {
        fprintf(stderr, "uniform 64/8: reported unbalanced\n");
        return 1;
    }
    for (int k = 0; k < 8; k++)
        if (lo[k] != (size_t)k * 8 || hi[k] != (size_t)(k + 1) * 8) {
            fprintf(stderr, "uniform 64/8: range %d = [%zu,%zu)\n",
                    k, lo[k], hi[k]);
            return 1;
        }

    for (size_t i = 0; i < 3; i++) w[i] = 1;
    if (check_contract(w, 3, 8, "n<width")) return 1;

    {
        size_t g[4] = {1000, 1, 1, 1};
        if (check_contract(g, 4, 4, "one giant")) return 1;
        if (tsimd_par_split(g, 4, 4, lo, hi)) {
            fprintf(stderr, "one giant: reported balanced\n");
            return 1;
        }
    }

    for (size_t i = 0; i < 9; i++) w[i] = 0;
    if (check_contract(w, 9, 4, "zeros")) return 1;

    for (size_t i = 0; i < 5; i++) w[i] = 3;
    if (!tsimd_par_split(w, 5, 1, lo, hi) || lo[0] != 0 || hi[0] != 5) {
        fprintf(stderr, "width 1: [%zu,%zu)\n", lo[0], hi[0]);
        return 1;
    }

    uint64_t rng = 0x5DEECE66Dull;
    for (size_t ni = 0; ni < sizeof(NS) / sizeof(NS[0]); ni++) {
        size_t n = NS[ni] > MAXW ? MAXW : NS[ni];
        for (size_t wi = 0; wi < sizeof(WS) / sizeof(WS[0]); wi++) {
            for (int shape = 0; shape < 5; shape++) {
                for (size_t i = 0; i < n; i++) {
                    switch (shape) {
                    case 0: w[i] = 1; break;
                    case 1: w[i] = i < n / 4 ? 64 : 1; break;
                    case 2: w[i] = i >= n - n / 4 ? 64 : 1; break;
                    case 3: w[i] = (i & 1) ? 64 : 1; break;
                    default: w[i] = 1 + tsimd_test_rand(&rng) % 100; break;
                    }
                }
                char what[64];
                snprintf(what, sizeof(what), "sweep n=%zu w=%d shape=%d",
                         n, WS[wi], shape);
                if (check_contract(w, n, WS[wi], what)) return 1;
            }
        }
    }
    printf("par split contract: OK\n");
    return 0;
}

static int probe_shape(size_t n, int width, const uint32_t *spin,
                       unsigned reps, const char *what) {
    for (unsigned r = 0; r < reps; r++) {
        unsigned *runs = calloc(n ? n : 1, sizeof(unsigned));
        if (!runs) return 1;
        if (tsimd_par_drain_probe(n, width, spin, runs) != 0) {
            fprintf(stderr, "%s: probe failed\n", what);
            free(runs);
            return 1;
        }
        unsigned long long total = 0;
        for (size_t i = 0; i < n; i++) {
            if (runs[i] != 1) {
                fprintf(stderr, "%s: slot %zu ran %u times (rep %u)\n",
                        what, i, runs[i], r);
                free(runs);
                return 1;
            }
            total += runs[i];
        }
        if (total != n) {
            fprintf(stderr, "%s: %llu runs for %zu slots\n", what, total, n);
            free(runs);
            return 1;
        }
        free(runs);
    }
    return 0;
}

static int test_drain_probe(void) {
    enum { N = 64 };
    static uint32_t spin[N];

    for (size_t i = 0; i < N; i++) spin[i] = 100;
    if (probe_shape(N, 4, spin, 3, "uniform")) return 1;

    for (size_t i = 0; i < N; i++) spin[i] = i < N / 2 ? 200000 : 1;
    if (probe_shape(N, 4, spin, 3, "front heavy")) return 1;

    for (size_t i = 0; i < N; i++) spin[i] = i >= N / 2 ? 200000 : 1;
    if (probe_shape(N, 4, spin, 3, "back heavy")) return 1;

    for (size_t i = 0; i < N; i++) spin[i] = (i & 1) ? 300000 : 1;
    if (probe_shape(N, 6, spin, 3, "alternating")) return 1;
    for (size_t i = 0; i < N; i++) spin[i] = 1;
    spin[17] = 400000;
    if (probe_shape(N, 4, spin, 3, "one giant")) return 1;

    for (size_t i = 0; i < N; i++) spin[i] = 1 + (uint32_t)(i * 37 % 500);
    if (probe_shape(3, 8, spin, 3, "n<width")) return 1;
    if (probe_shape(1, 16, spin, 3, "n=1")) return 1;
    if (probe_shape(0, 4, NULL, 1, "n=0")) return 1;
    if (probe_shape(5, 1, spin, 3, "width=1")) return 1;

    static uint32_t big[1024];
    for (size_t i = 0; i < 1024; i++) big[i] = 1 + (uint32_t)(i % 7) * 3;
    if (probe_shape(1024, 16, big, 2, "1024/16")) return 1;

    for (size_t i = 0; i < N; i++) spin[i] = 1;
    spin[0] = 0;
    spin[N - 1] = 0;
    if (probe_shape(N, 4, spin, 3, "zero spins")) return 1;

    printf("par drain probe (exactly once): OK\n");
    return 0;
}

int main(void) {
    if (test_split_shapes()) return 1;
    if (test_drain_probe()) return 1;
    return 0;
}
