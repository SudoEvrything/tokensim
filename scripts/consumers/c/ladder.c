#include <tokensim/tokenizer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
static long rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}
int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN,
                                   "/home/raising-reuel/tokenizer/models/cl100k_base.tiktoken",
                                   NULL, &sch, NULL, 0) != 0)
        return 1;
    FILE *f = fopen("/home/raising-reuel/tokenizer/testdata/owt.txt", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    uint8_t *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) return 1;
    fclose(f);
    printf("input: %ld MB\n", sz / (1024 * 1024));
    printf("rss after load+read: %ld KB\n", rss_kb());
    size_t bound = tsimd_tokenizer_encode_bound(&t, sz);
    uint32_t *obuf = malloc(bound * 4);
    size_t n1 = 0;
    double t0 = now();
    tsimd_tokenizer_encode_into_raw(&t, buf, sz, obuf, bound, &n1);
    double t1 = now();
    printf("1t into cold: %.1f MB/s (%zu ids)\n", sz / 1e6 / (t1 - t0), n1);
    t0 = now();
    tsimd_tokenizer_encode_into_raw(&t, buf, sz, obuf, bound, &n1);
    t1 = now();
    printf("1t into warm: %.1f MB/s (%zu ids)\n", sz / 1e6 / (t1 - t0), n1);
    size_t nref = n1;
    int ladder[] = {2, 4, 6, 8, 10, 12, 16};
    for (int i = 0; i < 7; i++) {
        size_t n = 0;
        tsimd_tokenizer_encode_parallel_into(&t, buf, sz, ladder[i], obuf,
                                             bound, &n);
        t0 = now();
        tsimd_tokenizer_encode_parallel_into(&t, buf, sz, ladder[i], obuf,
                                             bound, &n);
        t1 = now();
        printf("%dt into: %.1f MB/s (%zu ids) parity %s\n", ladder[i],
               sz / 1e6 / (t1 - t0), n,
               n == nref ? "yes" : "NO");
    }
    printf("rss after encode: %ld KB\n", rss_kb());
    free(obuf);
    free(buf);
    tsimd_tokenizer_free(&t);
    return 0;
}
