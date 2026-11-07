#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/o200k_base.tiktoken"

static void fill_buffer(uint8_t *buf, size_t n) {
    static const char *seeds[] = {
        "the quick brown fox, don't it's 42 \n\n  ",
        "x!abc 12:30 AM.\r\n\nJumped over. ",
        "a\nb c\n\n  d e\n",
    };
    size_t k = 0;
    uint32_t s = 12345;
    while (k < n) {
        const char *p = seeds[s % 3];
        size_t l = strlen(p);
        if (k + l > n) l = n - k;
        memcpy(buf + k, p, l);
        k += l;
        s = s * 1103515245u + 12345u;
        if ((s & 0xff) == 0 && k + 2 < n) {
            buf[k++] = 0xc3;
            buf[k++] = 0xa9;
        }
    }
}

static int run_encode(uint64_t *count_out, uint64_t *sum_out) {
    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load(&t, MODEL, TSIMD_PRETOK_O200K, NULL, 0) != 0)
        return -1;
    static uint8_t buf[2u << 20];
    fill_buffer(buf, sizeof(buf));
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&t, buf, sizeof(buf), &seq) != 0) {
        tsimd_tokenizer_free(&t);
        return -1;
    }
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < seq.count; i++) {
        h ^= seq.ids[i];
        h *= 0x100000001b3ull;
    }
    *count_out = seq.count;
    *sum_out = h;
    free(seq.ids);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {

        uint64_t c = 0, s = 0;
        if (run_encode(&c, &s) != 0) return 1;
        printf("%llu %llu\n", (unsigned long long)c, (unsigned long long)s);
        return 0;
    }

    setenv("TSIMD_SCALAR_FILL", "1", 1);
    uint64_t c1 = 0, s1 = 0;
    if (run_encode(&c1, &s1) != 0) {
        fprintf(stderr, "FAIL: parent encode\n");
        return 1;
    }

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "TSIMD_SCALAR_FILL=1 TSIMD_NO_DUALCUR=1 '%s' --child", argv[0]);
    FILE *p = popen(cmd, "r");
    if (!p) {
        fprintf(stderr, "FAIL: popen\n");
        return 1;
    }
    unsigned long long c2 = 0, s2 = 0;
    int got = fscanf(p, "%llu %llu", &c2, &s2);
    pclose(p);
    if (got != 2) {
        fprintf(stderr, "FAIL: the child produced no result\n");
        return 1;
    }
    if (c1 != c2 || s1 != s2) {
        fprintf(stderr,
                "FAIL: dual-cursor %llu ids (fnv %llx) vs single-cursor "
                "%llu ids (fnv %llx)\n",
                (unsigned long long)c1, (unsigned long long)s1, c2, s2);
        return 1;
    }
    printf("  dual-cursor == single-cursor: %llu ids, fnv %llx\n",
           (unsigned long long)c1, (unsigned long long)s1);
    return 0;
}
