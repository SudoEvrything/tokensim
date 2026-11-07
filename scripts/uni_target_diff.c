#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe.h"
#include "tokenizer.h"

typedef int (*enc_fn)(const tsimd_vocab_t *, tsimd_pretok_cache_t *,
                      tsimd_pretok_cache_t *, tsimd_bpe_workspace_t *,
                      const uint8_t *, uint32_t, size_t);

static size_t run_enc(enc_fn f, const tsimd_vocab_t *v,
                      tsimd_pretok_cache_t *c, tsimd_pretok_cache_t *shared,
                      const uint8_t *norm, uint32_t nn, size_t target,
                      const char *tag) {
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, v->max_token_len) != 0) return 0;
    if (f(v, c, shared, &ws, norm, nn, target) != 0) {
        printf("%s FAILED\n", tag);
        tsimd_bpe_workspace_free(&ws);
        return (size_t)-1;
    }
    printf("%s count=%zu\n", tag, ws.out_count);
    size_t n = ws.out_count;
    tsimd_bpe_workspace_free(&ws);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model fixture reps\n", argv[0]);
        return 2;
    }
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GEM4;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, argv[1],
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    FILE *fh = fopen(argv[2], "rb");
    if (!fh) return 1;
    static uint8_t raw[1 << 16];
    size_t rn = fread(raw, 1, sizeof raw, fh);
    fclose(fh);
    size_t reps = (size_t)atoi(argv[3]);
    size_t total = rn * reps;
    uint8_t *d = malloc(total);
    for (size_t i = 0; i < reps; i++) memcpy(d + i * rn, raw, rn);

    size_t cap = total * 3 + 64;
    uint8_t *norm = malloc(cap);
    if (!norm) return 1;
    size_t nn = tsimd_metaspace_ex(d, total, norm, cap, 0);
    if (nn == (size_t)-1) return 1;
    printf("norm=%zu (raw %zu) word_target=%zu seam_target=%zu\n", nn, total,
           tsimd_spm_word_target(&t.vocab, 1u << 14), (size_t)1 << 14);

    run_enc(tsimd_bpe_encode_spm_blocked_cached, &t.vocab, &t.cache, NULL,
            norm, (uint32_t)nn, tsimd_spm_word_target(&t.vocab, 1u << 14),
            "A seq-like (shared=NULL word_target)");
    run_enc(tsimd_bpe_encode_spm_blocked_cached, &t.vocab, &t.cache, &t.cache,
            norm, (uint32_t)nn, (size_t)1 << 14,
            "B par-like (shared=c, 16K)");
    run_enc(tsimd_bpe_encode_spm_blocked_cached, &t.vocab, &t.cache, NULL,
            norm, (uint32_t)nn, (size_t)1 << 14,
            "C (shared=NULL, 16K)");
    run_enc(tsimd_bpe_encode_spm_blocked_cached, &t.vocab, &t.cache, &t.cache,
            norm, (uint32_t)nn, tsimd_spm_word_target(&t.vocab, 1u << 14),
            "D (shared=c, word_target)");
    free(d);
    tsimd_tokenizer_free(&t);
    return 0;
}
