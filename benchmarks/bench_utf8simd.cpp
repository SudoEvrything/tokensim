




#include "simdutf.h"
#include <immintrin.h>

extern "C" {
#include "unicode.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}



static int ascii_check_simd(const uint8_t *p, size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(p + i));
        if (_mm256_movemask_epi8(v) != 0) return 1;
    }
    for (; i < n; i++)
        if (p[i] >= 0x80) return 1;
    return 0;
#else
    for (size_t i = 0; i < n; i++)
        if (p[i] >= 0x80) return 1;
    return 0;
#endif
}

static void bench(const char *name, double sec, double bytes) {
    printf("%-34s %8.4f s  %8.1f MiB/s\n", name, sec,
           bytes / sec / 1048576.0);
}

int main(int argc, char **argv) {
    const char *corpus = argc > 1 ? argv[1] : "testdata/owt.txt";
    FILE *f = fopen(corpus, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t cap = 20u << 20;
    if ((long)cap > sz) cap = (size_t)sz;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return 1;
    if (fread(buf, 1, cap, f) != cap) return 1;
    fclose(f);



    for (int r = 0; r < 3; r++) {
        volatile int acc = 0;
        double t0 = now();
        for (size_t off = 0; off < cap; off += (1u << 20)) {
            size_t n = off + (1u << 20) <= cap ? (1u << 20) : cap - off;
            acc += tsimd_nfc_may_change(buf + off, n);
        }
        bench("ours nfc_may_change (1MiB chunks)", now() - t0,
              (double)cap);
    }
    for (int r = 0; r < 3; r++) {
        volatile int acc = 0;
        double t0 = now();
        for (size_t off = 0; off < cap; off += (1u << 20)) {
            size_t n = off + (1u << 20) <= cap ? (1u << 20) : cap - off;
            acc += ascii_check_simd(buf + off, n);
        }
        bench("ours SIMD ascii check (1MiB chunks)", now() - t0,
              (double)cap);
    }




    {
        size_t cps_cap = cap + 16;
        char32_t *cps = (char32_t *)malloc(cps_cap * sizeof(char32_t));
        if (!cps) return 1;
        for (int r = 0; r < 3; r++) {
            volatile size_t m = 0;
            double t0 = now();
            size_t i = 0;
            tsimd_char_class_t cls_dummy = TSIMD_CLASS_OTHER;
            while (i < cap) {
                uint32_t cp;
                size_t adv = tsimd_utf8_next(buf + i, cap - i, &cp, &cls_dummy);
                if (adv == 0) break;
                cps[m++] = (char32_t)cp;
                i += adv;
            }
            bench("ours utf8_next loop (one char)", now() - t0,
                  (double)cap);
        }
        for (int r = 0; r < 3; r++) {
            double t0 = now();
            size_t m = simdutf::convert_utf8_to_utf32(
                (const char *)buf, cap, cps);
            bench("simdutf convert_utf8_to_utf32", now() - t0,
                  (double)cap);
            (void)m;
        }
        free(cps);
    }



    for (int r = 0; r < 3; r++) {
        volatile size_t m = 0;
        double t0 = now();
        size_t i = 0;
        tsimd_char_class_t cls_dummy = TSIMD_CLASS_OTHER;
        while (i < cap) {
            uint32_t cp;
            size_t adv = tsimd_utf8_next(buf + i, cap - i, &cp, &cls_dummy);
            if (adv == 0) break;
            m++;
            i += adv;
        }
        bench("ours utf8_next count", now() - t0, (double)cap);
    }
    for (int r = 0; r < 3; r++) {
        double t0 = now();
        size_t m = simdutf::count_utf8((const char *)buf, cap);
        bench("simdutf count_utf8", now() - t0, (double)cap);
        (void)m;
    }



    for (int r = 0; r < 3; r++) {
        volatile int acc = 0;
        double t0 = now();
        for (size_t off = 0; off < cap; off += (1u << 20)) {
            size_t n = off + (1u << 20) <= cap ? (1u << 20) : cap - off;
            acc += simdutf::validate_utf8((const char *)buf + off, n);
        }
        bench("simdutf validate_utf8 (1MiB chunks)", now() - t0,
              (double)cap);
    }

    free(buf);
    return 0;
}
