#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void run_mode(const char *mode, char *out, size_t cap) {
    char exe[4096];
    ssize_t r = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (r <= 0) {
        fprintf(stderr, "cannot resolve self\n");
        exit(2);
    }
    exe[r] = 0;
    char cmd[8192];
    snprintf(cmd, sizeof cmd, "TSIMD_SIMD_FORCE=%s '%s' child", mode, exe);
    FILE *p = popen(cmd, "r");
    if (!p) exit(2);
    size_t n = fread(out, 1, cap - 1, p);
    out[n] = 0;
    pclose(p);
}

static uint64_t span_hash(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(
            &t, TSIMD_MODEL_TIKTOKEN,
            TSIMD_SRC_DIR "/models/cl100k_base.tiktoken", NULL, &sch, NULL,
            0) != 0) {
        fprintf(stderr, "no cl100k model\n");
        exit(2);
    }
    FILE *f = fopen(TSIMD_SRC_DIR "/testdata/owt16m.txt", "rb");
    if (!f) {
        fprintf(stderr, "no corpus\n");
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    uint8_t *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) exit(2);
    fclose(f);
    uint64_t h = 1469598103934665603ULL;
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    if (tsimd_pretokenize(sch, buf, (size_t)sz, &spans) != 0) exit(2);
    for (size_t i = 0; i < spans.count; i++) {
        uint32_t v[2] = {spans.spans[i].start, spans.spans[i].end};
        const uint8_t *pb = (const uint8_t *)v;
        for (int k = 0; k < 8; k++) {
            h ^= pb[k];
            h *= 1099511628211ULL;
        }
    }
    tsimd_span_array_free(&spans);
    free(buf);
    tsimd_tokenizer_free(&t);
    return h;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "child") == 0) {
        printf("%016llx\n", (unsigned long long)span_hash());
        return 0;
    }
    char a[128], b[128];
    run_mode("avx2", a, sizeof a);
    run_mode("avx512", b, sizeof b);
    if (strcmp(a, b) != 0) {
        fprintf(stderr, "FAIL: the avx2 span hash %s != the avx512 %s\n", a,
                b);
        return 1;
    }
    printf("test_avx512_fill_parity: the avx2 and the avx512 fills agree "
           "(%s)\n",
           a);
    return 0;
}
