#include "tokenizer.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO TSIMD_SRC_DIR

int main(void) {
    setenv("TSIMD_PCACHE_STATS", "1", 1);
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                   REPO "/models/reference/gemma4_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL shared fallback: load\n");
        return 1;
    }
    static uint8_t buf[1 << 16];
    size_t len = 0;
    static const char *words[] = {"hello", "world", "the", "token",
                                  "vector", "cache", "stream", "merge"};
    for (int i = 0; i < 6000; i++) {
        const char *w = words[i % 8];
        buf[len++] = 0xE2;
        buf[len++] = 0x96;
        buf[len++] = 0x81;
        int n = (int)strlen(w);
        for (int k = 0; k < n; k++) buf[len++] = (uint8_t)w[k];
        buf[len++] = (uint8_t)('a' + (i % 6));
    }

    size_t wt = tsimd_spm_word_target(&t.vocab, 1u << 14);
    tsimd_bpe_workspace_t ws1;
    if (tsimd_bpe_workspace_init(&ws1, t.vocab.max_token_len) != 0) return 1;
    if (tsimd_bpe_encode_spm_blocked_cached(&t.vocab, &t.cache, NULL, &ws1,
                                            buf, (uint32_t)len, wt) != 0) {
        fprintf(stderr, "FAIL shared fallback: pass1 rc\n");
        return 1;
    }
    uint32_t *ref = malloc(ws1.out_count * sizeof(uint32_t));
    memcpy(ref, ws1.out, ws1.out_count * sizeof(uint32_t));
    size_t ref_n = ws1.out_count;

    size_t h0 = 0, m0 = 0, i0 = 0;
    tsimd_tokenizer_cache_stats(&t, &h0, &m0, &i0);

    tsimd_pretok_cache_t priv;
    if (tsimd_pretok_cache_init(&priv, 1u << 12) != 0) return 1;
    tsimd_bpe_workspace_t ws2;
    if (tsimd_bpe_workspace_init(&ws2, t.vocab.max_token_len) != 0) return 1;
    if (tsimd_bpe_encode_spm_blocked_cached(&t.vocab, &priv, &t.cache, &ws2,
                                            buf, (uint32_t)len, wt) != 0) {
        fprintf(stderr, "FAIL shared fallback: pass2 rc\n");
        return 1;
    }
    if (ws2.out_count != ref_n ||
        memcmp(ws2.out, ref, ref_n * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "FAIL shared fallback: ids differ (%zu vs %zu)\n",
                ws2.out_count, ref_n);
        return 1;
    }
    size_t h1 = 0, m1 = 0, i1 = 0;
    tsimd_tokenizer_cache_stats(&t, &h1, &m1, &i1);
    if (h1 - h0 < 20) {
        fprintf(stderr, "FAIL shared fallback: shared served only %zu units\n",
                h1 - h0);
        return 1;
    }

    tsimd_pretok_cache_free(&priv);
    tsimd_bpe_workspace_free(&ws1);
    tsimd_bpe_workspace_free(&ws2);
    tsimd_tokenizer_free(&t);
    free(ref);
    printf("shared fallback: OK\n");
    return 0;
}
