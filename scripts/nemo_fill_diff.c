#include "tokenizer.h"
#include "pretokenizer.h"
#include "bpe.h"
#include "../src/model_internal.h"
#include "../src/pretokenizer/internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                                : "tests/fixtures/nemo_serve_drop.bin";
    long caplen = argc > 2 ? atol(argv[2]) : -1;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_NEMO;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
        "models/reference/nemotron3_nano_tokenizer.json", NULL, &sch,
        NULL, 0) != 0) { fprintf(stderr, "load failed\n"); return 2; }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    uint8_t *data = malloc(sz);
    if (fread(data, 1, sz, f) != (size_t)sz) return 2;
    fclose(f);
    if (caplen > 0 && sz > caplen) sz = caplen;

    tsimd_span_array_t ref;
    memset(&ref, 0, sizeof ref);
    if (tsimd_pretokenize(sch, data, sz, &ref) != 0) return 2;
    printf("pretokenizer spans: %zu\n", ref.count);

    static tsimd_o200k_mstate_t st_store;
    void *st = &st_store;
    tsimd_pcache_span_t batch[512];
    size_t fill_total = 0, fill_pos = 0;
    int rc = 0;
    static size_t fill_off[700000];
    static uint32_t fill_len[700000];
    size_t fill_cap = 700000;
    while (rc == 0) {
        int n = tsimd_nemo_mask_fill_keyed(data, sz, st, batch, 512, NULL, 0);
        if (n < 0) { fprintf(stderr, "fill failed\n"); return 2; }
        for (int k = 0; k < n && fill_total < fill_cap; k++) {
            const uint8_t *p = batch[k].ptr;
            fill_off[fill_total] = (size_t)(p - data);
            fill_len[fill_total] =
                (batch[k].k0 == 0 && batch[k].k1 == 1)
                    ? (uint32_t)batch[k].meta
                    : (uint32_t)(batch[k].k1 >> 56);
            fill_total++;
        }
        if (n == 0) break;
    }
    printf("fill spans: %zu\n", fill_total);
    size_t ndiff = 0;
    size_t ri = 0, fi = 0;
    while (ri < ref.count && fi < fill_total) {
        uint32_t rs = ref.spans[ri].start, re = ref.spans[ri].end;
        size_t fs = fill_off[fi], fl = fill_len[fi];
        if (rs == fs && (size_t)(re - rs) == fl) { ri++; fi++; continue; }
        ndiff++;
        if (ndiff <= 10) {
            printf("DIFF at fill_off=%zu pre_start=%u: fill len=%zu '",
                   fs, rs, fl);
            for (size_t q = fs; q < fs + fl && q < (size_t)sz; q++)
                fputc(data[q] == '\n' ? 'N' : data[q], stdout);
            printf("' | pre len=%u '", re - rs);
            for (uint32_t q = rs; q < re && q < (uint32_t)sz; q++)
                fputc(data[q] == '\n' ? 'N' : data[q], stdout);
            printf("'\n");
        }
        if (fs <= rs) fi++; else ri++;
    }
    printf("fill last 3:");
    for (size_t k = fill_total - 3; k < fill_total; k++)
        printf(" [%zu,%zu)", fill_off[k], fill_off[k] + fill_len[k]);
    printf("\n");
    printf("pre last 3:");
    for (size_t k = ref.count - 3; k < ref.count; k++)
        printf(" [%u,%u)", ref.spans[k].start, ref.spans[k].end);
    printf("\n");
    printf("diff pairs: %zu\n", ndiff);
    if (fi < fill_total) {
        printf("fill tail:");
        for (size_t k = fill_total - 2; k < fill_total; k++)
            printf(" [%zu,%zu)", fill_off[k], fill_off[k] + fill_len[k]);
        printf("\n");
    }
    if (ri < ref.count) {
        printf("pre tail:");
        for (size_t k = ref.count - 2; k < ref.count; k++)
            printf(" [%u,%u)", ref.spans[k].start, ref.spans[k].end);
        printf("\n");
        for (size_t q = ref.spans[ri].start;
             q < ref.spans[ri].end && q < (size_t)sz; q++)
            fputc(data[q] == '\n' ? 'N' : data[q], stdout);
        printf("\n");
    }
    return ndiff != 0;
}
