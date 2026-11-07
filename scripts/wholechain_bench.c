#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static size_t fill_cjk(uint8_t *buf, size_t n) {
    const char *unit = "数字123と漢字のテスト、日本語のテキスト。 héllo wörld ";
    size_t ul = strlen(unit);
    size_t off = 0;
    while (off < n) {
        size_t take = n - off < ul ? n - off : ul;
        memcpy(buf + off, unit, take);
        off += take;
    }
    return n;
}

static unsigned long long id_hash(const uint32_t *ids, size_t n) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s mode threads file rounds\n", argv[0]);
        return 2;
    }
    int mode = atoi(argv[1]);
    int threads = atoi(argv[2]);
    const char *src = argv[3];
    int rounds = atoi(argv[4]);
    if (mode == 2) setenv("TSIMD_UNI_WINDOW", "0", 1);
    if (mode == 3) setenv("TSIMD_UNI_WINDOW", "1", 1);

    size_t n;
    uint8_t *buf;
    if (strcmp(src, "cjk") == 0) {
        n = 4698000u;
        buf = (uint8_t *)malloc(n);
        if (!buf) return 1;
        fill_cjk(buf, n);
    } else {
        FILE *f = fopen(src, "rb");
        if (!f) {
            fprintf(stderr, "open failed: %s\n", src);
            return 1;
        }
        buf = (uint8_t *)malloc(256u << 20);
        if (!buf) return 1;
        n = fread(buf, 1, 256u << 20, f);
        fclose(f);
    }

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &tok, TSIMD_MODEL_SENTENCEPIECE,
            TSIMD_SRC_DIR "/models/reference/t5_spiece.model",
            NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    unsigned long long hash = 0, prev_hash = 0;
    size_t count = 0;
    double last_t = 0;
    for (int r = 0; r < rounds; r++) {
        seq.ids = NULL;
        seq.count = 0;
        double t0 = now_s();
        int rc;
        if (mode == 0 || threads <= 1)
            rc = tsimd_tokenizer_encode(&tok, buf, n, &seq);
        else if (mode == 1)
            rc = tsimd_tokenizer_encode_parallel(&tok, buf, n, threads,
                                                 &seq);
        else
            rc = tsimd_tokenizer_encode_whole_chain_par(&tok, buf, n,
                                                        threads, &seq);
        double t1 = now_s();
        if (rc != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        last_t = t1 - t0;
        count = seq.count;
        hash = id_hash(seq.ids, seq.count);
        if (r > 0 && hash != prev_hash) {
            fprintf(stderr, "UNSTABLE HASH across rounds\n");
            return 1;
        }
        prev_hash = hash;
        tsimd_token_sequence_free(&seq);
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("mode=%d th=%d bytes=%zu rounds=%d last=%.4fs %.1fMB/s "
           "count=%zu hash=%llx peak=%ldMB\n",
           mode, threads, n, rounds, last_t,
           (double)n / 1048576.0 / last_t, count, hash,
           ru.ru_maxrss / 1024);
    tsimd_tokenizer_free(&tok);
    free(buf);
    return 0;
}
