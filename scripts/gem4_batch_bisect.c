#include "tokenizer.h"
#include "pretokenizer.h"
#include "unicode.h"

#include "../src/model_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tsimd_tokenizer_t tok;

static tsimd_bpe_workspace_t ws;

static int encode_pool(const uint8_t *data, size_t len, uint32_t **ids,
                       size_t *n) {
    ws.out_count = 0;
    if (tsimd_encode_buffer_pool(&tok, data, len, &ws, &tok.cache,
                                 &tok.cache) != 0)
        return -1;
    *ids = ws.out;
    *n = ws.out_count;
    return 0;
}

static int encode_both(const uint8_t *data, size_t len, uint32_t **aid,
                       size_t *an, uint32_t **bid, size_t *bn) {
    unsetenv("TSIMD_SPM_NORM_BATCH");
    if (encode_pool(data, len, aid, an) != 0) return -1;
    setenv("TSIMD_SPM_NORM_BATCH", "1", 1);
    if (encode_pool(data, len, bid, bn) != 0) return -1;
    unsetenv("TSIMD_SPM_NORM_BATCH");
    return 0;
}

static size_t first_diff(const uint32_t *a, size_t an, const uint32_t *b,
                         size_t bn) {
    size_t n = an < bn ? an : bn;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) return i;
    return an == bn ? (size_t)-1 : n;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
                                 : "models/reference/gemma3-4b_tokenizer.json";
    const char *corpus = argc > 2 ? argv[2] : "testdata/owt16m.txt";
    size_t chunk_sz = (size_t)(argc > 3 ? atoi(argv[3]) : 2) << 20;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GEM4;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    if (tsimd_bpe_workspace_init(&ws, tok.vocab.max_token_len) != 0) {
        fprintf(stderr, "ws init failed\n");
        return 2;
    }
    FILE *f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", corpus);
        return 2;
    }
    uint8_t *buf = malloc(chunk_sz + (1u << 20));
    size_t chunk_idx = 0, got;
    while ((got = fread(buf, 1, chunk_sz, f)) > 0) {
        uint32_t *aid, *bid;
        size_t an, bn;
        if (encode_both(buf, got, &aid, &an, &bid, &bn) != 0) {
            fprintf(stderr, "encode failed chunk %zu\n", chunk_idx);
            return 2;
        }
        size_t d = first_diff(aid, an, bid, bn);
        if (d != (size_t)-1) {
            printf("DIVERGENT chunk %zu (file bytes %zu..) at id %zu "
                   "(a_n=%zu b_n=%zu a_id=%u b_id=%u)\n",
                   chunk_idx, chunk_idx * chunk_sz, d, an, bn, aid[d],
                   bid[d]);
            size_t lo = d > 4 ? d - 4 : 0;
            printf("  a:");
            for (size_t i = lo; i < d + 4 && i < an; i++)
                printf(" %u", aid[i]);
            printf("\n  b:");
            for (size_t i = lo; i < d + 4 && i < bn; i++)
                printf(" %u", bid[i]);
            printf("\n");
            return 0;
        }
        chunk_idx++;
    }
    fclose(f);
    printf("no divergence across %zu chunks\n", chunk_idx);
    return 0;
}
