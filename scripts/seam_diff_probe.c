#include "tokenizer.h"
#include "simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static tsimd_tokenizer_t tok;

static uint64_t rs = 0x9e3779b97f4a7c15ULL;

static uint64_t rnd(void) {
    rs ^= rs >> 12;
    rs ^= rs << 25;
    rs ^= rs >> 27;
    return rs * 0x2545F4914F6CDD1DULL;
}

static int capture(const uint8_t *data, size_t len, int level,
                   tsimd_added_seam_list_t *out) {
    tsimd_simd_set_forced(level);
    int rc = tsimd_tokenizer_added_seams(&tok, data, len, out);
    tsimd_simd_set_forced(-1);
    return rc;
}

static int diff_buf(const uint8_t *data, size_t len) {
    tsimd_added_seam_list_t a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    if (capture(data, len, TSIMD_SIMD_SSE2, &a) != 0) return 2;
    if (capture(data, len, TSIMD_SIMD_AVX2, &b) != 0) {
        tsimd_added_seam_list_free(&a);
        return 2;
    }
    int bad = 0;
    if (a.count != b.count)
        bad = 1;
    else
        for (size_t i = 0; i < a.count; i++)
            if (a.items[i].off != b.items[i].off ||
                a.items[i].len != b.items[i].len ||
                a.items[i].idx != b.items[i].idx) {
                bad = 1;
                break;
            }
    if (bad)
        fprintf(stderr, "SEAM DIFF len=%zu a=%zu b=%zu\\n", len, a.count,
                b.count);
    tsimd_added_seam_list_free(&a);
    tsimd_added_seam_list_free(&b);
    return bad;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
                                 : "models/reference/gemma3-4b_tokenizer.json";
    const char *corpus = argc > 2 ? argv[2] : "testdata/owt16m.txt";
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GEM4;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\\n");
        return 2;
    }
    int fails = 0;
    FILE *f = fopen(corpus, "rb");
    if (!f) return 2;
    uint8_t *buf = malloc(1u << 20);
    size_t chunks = 0;
    size_t got;
    while (chunks < 6 && (got = fread(buf, 1, 1u << 20, f)) > 0) {
        fails += diff_buf(buf, got) != 0 ? 1 : 0;
        chunks++;
    }
    fclose(f);
    for (int t = 0; t < 20000; t++) {
        size_t n = rnd() % 2000;
        int kind = (int)(rnd() % 4);
        for (size_t i = 0; i < n; i++) {
            uint64_t r = rnd();
            if (kind == 0)
                buf[i] = (uint8_t)(0x20 + (r % 95));
            else if (kind == 1)
                buf[i] = (uint8_t)(r % 256);
            else if (kind == 2)
                buf[i] = (uint8_t)(r % 3 ? '<' : 0x20 + (r >> 8) % 95);
            else
                buf[i] = (uint8_t)(r % 6 ? '<' : 0x41 + (r >> 8) % 26);
        }
        fails += diff_buf(buf, n) != 0 ? 1 : 0;
    }
    {
        uint8_t *pg = mmap(NULL, 8192, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pg == MAP_FAILED) return 2;
        if (mprotect(pg + 4096, 4096, PROT_NONE) != 0) return 2;
        for (size_t n = 1; n <= 4096; n += 13) {
            for (int kind = 0; kind < 2; kind++) {
                for (size_t i = 0; i < n; i++)
                    pg[i] = kind ? (uint8_t)(rnd() % 256)
                                 : (uint8_t)(rnd() % 2 ? '<' : 0x61 + i % 26);
                fails += diff_buf(pg, n) != 0 ? 1 : 0;
            }
        }
        munmap(pg, 8192);
    }
    printf("seam_diff: %d fails (6 corpus chunks + 20000 random + page "
           "cases)\n",
           fails);
    return fails != 0;
}
