






#include "pretokenizer.h"
#include "simd.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t rng = 0x123456789ABCDEF0ULL;
static uint32_t rnd(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}


static void build_corpus(uint8_t *buf, size_t n) {
    static const char *words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "Hello", "world", "testing", "tokenization", "performance",
        "isn't", "don't", "they'll", "we've", "I'm", "it's",
        "123", "4567", "89", "3.14", "1,000", "x=1;", "foo_bar",
        "(", ")", "{", "}", "<|endoftext|>", ":", "--", "...",
    };
    size_t nw = sizeof(words) / sizeof(words[0]);
    size_t pos = 0;
    while (pos < n) {
        const char *w = words[rnd() % nw];
        size_t wl = strlen(w);
        if (pos + wl + 2 > n) break;
        memcpy(buf + pos, w, wl);
        pos += wl;
        uint32_t r = rnd();
        buf[pos++] = (uint8_t)" \n\t "[r % 4];
        if ((r >> 8) % 11 == 0 && pos < n) buf[pos++] = ' ';
    }
    while (pos < n) buf[pos++] = ' ';
}

static double run(const char *name, tsimd_pretok_scheme_t scheme,
                  const uint8_t *data, size_t len, int use_simd,
                  size_t *spans_out) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);

    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        spans.count = 0;
        double t0 = now_sec();
        if (use_simd) {
            tsimd_simd_set_forced(-1);
            if (tsimd_pretokenize(scheme, data, len, &spans) != 0) {
                fprintf(stderr, "simd pretokenize failed\n");
                exit(1);
            }
        } else {
            if (tsimd_pretokenize_scalar_ref(scheme, data, len, &spans) != 0) {
                fprintf(stderr, "scalar pretokenize failed\n");
                exit(1);
            }
        }
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    *spans_out = spans.count;
    double mibs = (double)len / (1024.0 * 1024.0);
    printf("  %-14s %8.1f MiB/s  (%zu spans, %.3f s)\n", name,
           mibs / best, spans.count, best);
    tsimd_span_array_free(&spans);
    return best;
}

int main(int argc, char **argv) {
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    size_t mib = 64;
    if (argc > 1 && strcmp(argv[1], "cl100k") == 0) scheme = TSIMD_PRETOK_CL100K;
    if (argc > 2) mib = (size_t)atol(argv[2]);

    size_t len = mib * 1024 * 1024;
    uint8_t *buf = malloc(len);
    if (!buf) return 1;
    build_corpus(buf, len);

    printf("bench_pretokenizer: %s, %zu MiB, corpus=ascii-text\n",
           scheme == TSIMD_PRETOK_GPT2 ? "gpt2" : "cl100k", mib);
    printf("  detected SIMD level: %d (0=scalar 1=sse2 2=avx2)\n",
           (int)tsimd_simd_level());

    size_t s1, s2;
    double t_ref = run("scalar-ref", scheme, buf, len, 0, &s1);
    double t_simd = run("simd", scheme, buf, len, 1, &s2);
    if (s1 != s2) {
        fprintf(stderr, "SPAN COUNT MISMATCH: %zu vs %zu\n", s1, s2);
        return 1;
    }
    printf("  speedup: %.2fx\n", t_ref / t_simd);
    free(buf);
    return 0;
}
